"""TiRex-2 access for explainability: differentiable forecasts, hidden states, mLSTM gates.

TiRex-2 internals relevant here (see tirex2/model/tirex2.py):
* PostProcessor: rule-based trend detection (OLS z-slope * len >= 1.5 and R^2 >= 0.87 over the
  whole context); detected series are first-differenced before entering the network (tta_diff).
* Scaler: instance standardisation + arcsinh over the last ``context_len`` steps.
* Tokenizer: non-overlapping patches of 32 steps; context_len=2048 -> 64 context patches,
  future_len=320 -> 10 future patches (NaN-filled, the model writes its forecast there).
* 12 blocks alternating mLSTM / sLSTM time mixers; the variate-mixer attention is trivial for
  univariate inputs (a single token per group).
"""

from __future__ import annotations

import types
from contextlib import contextmanager, nullcontext
from functools import cache

import numpy as np
import torch
from tirex2 import TimeseriesType, load_model
from xlstm.xlstm_large.model import soft_cap

QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
MEDIAN = 4
PATCH = 32
CONTEXT_LEN = 2048
N_CTX_PATCHES = CONTEXT_LEN // PATCH


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _safe_scale(self, x: torch.Tensor):
    """Scaler.scale with identical forward values but NaN-free gradients.

    The original computes nanmean((x - loc) ** 2); at NaN-padded positions the backward pass
    yields NaN * 0 = NaN for every input. Masking before the arithmetic avoids that.
    """
    valid = ~torch.isnan(x)
    n = valid.sum(-1, keepdim=True).clamp_min(1)
    xz = torch.where(valid, x, torch.zeros_like(x))
    loc = torch.where(valid.any(-1, keepdim=True), xz.sum(-1, keepdim=True) / n, torch.zeros_like(n, dtype=x.dtype))
    dev = torch.where(valid, x - loc, torch.zeros_like(x))
    var = (dev * dev).sum(-1, keepdim=True) / n
    scale = torch.where(valid.any(-1, keepdim=True), var.clamp_min(1e-30).sqrt(), torch.ones_like(var))
    scale = torch.clamp_min(scale, min=self.eps)
    is_binary = None
    if self.binaryaware:
        is_binary = ((x == 0.0) | (x == 1.0) | ~valid).all(dim=-1, keepdim=True)
        loc = torch.where(is_binary, torch.zeros_like(loc), loc)
        scale = torch.where(is_binary, torch.ones_like(scale), scale)
    scaled_x = (xz - loc) / scale
    if self.use_arcsinh:
        scaled_x = torch.where(is_binary, scaled_x, torch.arcsinh(scaled_x)) if is_binary is not None else torch.arcsinh(scaled_x)
    scaled_x = torch.where(valid, scaled_x, torch.full_like(scaled_x, float("nan")))
    return scaled_x, (loc, scale, is_binary)


@cache
def get_model(device: str | None = None):
    # Built with the pure-PyTorch ("cpu") kernels, then moved: the Triton mLSTM kernels are
    # incompatible with the installed Triton, and native kernels match the CPU Phase I run.
    model = load_model("NX-AI/TiRex-2", device="cpu")
    model.eval()
    model.scaler.scale = types.MethodType(_safe_scale, model.scaler)
    return model.to(device or default_device())


def _to_ts(c, device) -> TimeseriesType:
    t = c if isinstance(c, torch.Tensor) else torch.as_tensor(np.asarray(c, dtype=np.float32))
    return TimeseriesType(target=t.reshape(1, -1).to(device), past_covariates=None, future_covariates=None)


def forecast(contexts, prediction_length: int, tta_diff: bool = True, batch_size: int = 64) -> np.ndarray:
    """Quantile forecasts, shape [B, 9, prediction_length] (no gradients)."""
    model = get_model()
    dev = next(model.parameters()).device
    out = []
    for s in range(0, len(contexts), batch_size):
        batch = [_to_ts(c, dev) for c in contexts[s : s + batch_size]]
        with torch.no_grad():
            preds = model._predict(batch, prediction_length, tta_diff=tta_diff)
        out.extend(p[0].float().cpu().numpy() for p in preds)
    return np.stack(out)


def forecast_torch(contexts: list[torch.Tensor], prediction_length: int, tta_diff: bool = True) -> torch.Tensor:
    """Differentiable forecast; ``contexts`` are 1-D tensors (may require grad). Returns [B, 9, H]."""
    model = get_model()
    dev = next(model.parameters()).device
    preds = model._predict([_to_ts(c, dev) for c in contexts], prediction_length, tta_diff=tta_diff, preserve_grad=True)
    return torch.stack([p[0] for p in preds])


def trend_flag(context) -> bool:
    """Whether the PostProcessor differences this series before the network (tta_diff path)."""
    t = torch.as_tensor(np.asarray(context, dtype=np.float32)).reshape(1, -1)
    return bool(get_model().postprocessor._detect_trends(t)[0])


@contextmanager
def _capture_gates():
    """Capture soft-capped mLSTM input/forget-gate pre-activations of every mLSTM forward cell."""
    model = get_model()
    records: dict[int, dict[str, torch.Tensor]] = {}
    handles = []
    for idx, block in enumerate(model.stack):
        cell = block.time_mixer.fwd_cell
        if not hasattr(cell, "fgate_preact"):
            continue
        records[idx] = {}

        def make(name, i=idx, cap=cell.config.gate_soft_cap):
            def hook(_m, _inp, out):
                records[i][name] = soft_cap(out, cap_value=cap).detach().float().cpu()

            return hook

        handles.append(cell.igate_preact.register_forward_hook(make("i")))
        handles.append(cell.fgate_preact.register_forward_hook(make("f")))
    try:
        yield records
    finally:
        for h in handles:
            h.remove()


def internals(contexts, prediction_length: int, tta_diff: bool = True, gates: bool = False) -> dict:
    """Run the network and return per-block token states (and optionally mLSTM gates).

    Returns dict with
      states: [B, n_blocks + 1, L, D] float16 (index 0 = input embedding, i = after block i)
      last_ctx_token: index of the last token that holds observed context
      gates: {block_idx: {"i": [B, L, H], "f": [B, L, H]}} if requested
    """
    model = get_model()
    dev = next(model.parameters()).device
    batch_ts = [_to_ts(c, dev) for c in contexts]
    with torch.no_grad():
        _, batch, _, _ = model._prepare_input(batch_ts, prediction_length, tta_diff)
        with _capture_gates() if gates else nullcontext() as gate_rec:
            x, gv, tm, kcm, _, _ = model._prepare_forward(batch)
            states = [x.detach().half().cpu()]
            for x in model._forward_layerwise(x, gv, tm, kcm):
                states.append(x.detach().half().cpu())
    L = states[0].shape[1]
    # token layout: [64 context patches | 10 future patches], the future part is NaN-padded
    out = {"states": torch.stack(states, 1).numpy(), "last_ctx_token": L - model.future_len // PATCH - 1}
    if gates:
        out["gates"] = {k: {g: v.numpy() for g, v in d.items()} for k, d in gate_rec.items()}
    return out


def mlstm_memory_weights(i_gate: np.ndarray, f_gate: np.ndarray, t: int) -> np.ndarray:
    """Normalised weight of each token s<=t in the mLSTM memory read at token t.

    The mLSTM memory is C_t = sum_s exp(i_s + sum_{s<k<=t} logsigmoid(f_k)) v_s k_s^T, so this row
    of the decay matrix says how strongly each past patch can still contribute at token t.
    i_gate, f_gate: [L, H]  ->  [t+1, H], columns sum to 1.
    """
    logf = -np.logaddexp(0.0, -f_gate[: t + 1])
    suffix = np.cumsum(logf[::-1], axis=0)[::-1]
    decay = np.concatenate([suffix[1:], np.zeros((1, suffix.shape[1]))], axis=0)
    logw = i_gate[: t + 1] + decay
    logw -= logw.max(axis=0, keepdims=True)
    w = np.exp(logw)
    return w / w.sum(axis=0, keepdims=True)
