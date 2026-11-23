"""Attribution methods over the network input (the last 2048 + H context steps).

Scalar explanation targets per window (all divided by the window's MASE scale):
  level  - mean median forecast minus last observed value  (level gap / mean reversion)
  slope  - OLS slope of the median forecast times H          (trend continuation)
  width  - mean 80 % interval width                          (uncertainty)
  error  - MAE of the median forecast vs. the truth          (what drives the error)

Methods
  * Integrated Gradients (captum) with a flat baseline at the context mean.
  * LIME (lime.lime_base) and KernelSHAP (shap.KernelExplainer) over 64 lag segments of 32 steps
    counted back from the forecast origin; a segment is switched "off" by setting it to NaN,
    which TiRex-2 natively treats as missing (in-distribution perturbation).
  * Exact Shapley values over STL components {trend, seasonal, residual} (level always kept).
"""

from __future__ import annotations

from itertools import combinations
from math import factorial

import numpy as np
import torch
from captum.attr import IntegratedGradients
from lime.lime_base import LimeBase
from sklearn.linear_model import Ridge
from statsmodels.tsa.seasonal import STL

from . import model_io as M
from . import windows as W

TARGETS = ["level", "slope", "width", "error"]
SEG = M.PATCH
N_SEG = M.CONTEXT_LEN // SEG


def network_input(w: W.Window) -> np.ndarray:
    """Tail of the context handed to TiRex-2 (2048 + H steps).

    The PostProcessor appends H future placeholders, so the network itself only sees the last
    2048 observed steps; the extra H older steps keep the trend detector's view unchanged
    and receive zero attribution.
    """
    return w.context[-(M.CONTEXT_LEN + len(w.truth)) :].astype(np.float32)


def _targets_torch(q: torch.Tensor, last: torch.Tensor, truth: torch.Tensor, scale: float) -> torch.Tensor:
    med = q[:, M.MEDIAN]
    H = med.shape[-1]
    t = torch.arange(H, dtype=med.dtype, device=med.device)
    tc = t - t.mean()
    slope = ((med - med.mean(-1, keepdim=True)) * tc).sum(-1) / (tc * tc).sum().clamp_min(1e-9) * H
    valid = ~torch.isnan(truth)
    err = torch.where(valid, (med - torch.nan_to_num(truth)).abs(), torch.zeros_like(med)).sum(-1) / valid.sum().clamp_min(1)
    level = med.mean(-1) - last
    width = (q[:, -1] - q[:, 0]).mean(-1)
    return torch.stack([level, slope, width, err], -1) / scale


def _last_obs(x: np.ndarray) -> float:
    v = x[~np.isnan(x)]
    return float(v[-1]) if len(v) else 0.0


def evaluate(w: W.Window, inputs: list[np.ndarray] | np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Targets for many variants of the network input, [N, 4] (no gradients)."""
    x0 = network_input(w)
    H = len(w.truth)
    out = []
    for s in range(0, len(inputs), batch_size):
        q = torch.as_tensor(M.forecast(list(inputs[s : s + batch_size]), H, batch_size=batch_size))
        out.append(_targets_torch(q, torch.tensor(_last_obs(x0)), torch.as_tensor(w.truth), w.mase_scale).numpy())
    return np.concatenate(out)


# ---------------------------------------------------------------- gradients


def integrated_gradients(w: W.Window, n_steps: int = 8) -> np.ndarray:
    """IG per input step and target, [4, len(input)].

    TiRex-2 is affine-equivariant (instance standardisation), so along the straight path from a
    flat mean baseline the gradient is constant and IG equals gradient x (input - mean); n_steps
    is kept small and completeness is checked in the notebook.
    """
    x0 = network_input(w)
    H = len(w.truth)
    dev = next(M.get_model().parameters()).device
    last, truth = torch.tensor(_last_obs(x0)), torch.as_tensor(w.truth)
    fill = np.nanmean(x0)
    nan_mask = torch.as_tensor(np.isnan(x0))
    x = torch.as_tensor(np.nan_to_num(x0, nan=fill))[None]

    def f(batch: torch.Tensor) -> torch.Tensor:
        ctx = [torch.where(nan_mask.to(b.device), torch.tensor(float("nan"), device=b.device), b) for b in batch]
        q = M.forecast_torch(ctx, H)
        return _targets_torch(q, last.to(q.device), truth.to(q.device), w.mase_scale)

    ig = IntegratedGradients(f)
    base = torch.full_like(x, float(fill))
    attrs = [
        ig.attribute(x.to(dev), baselines=base.to(dev), target=k, n_steps=n_steps, internal_batch_size=n_steps)[0]
        .detach().cpu().numpy()
        for k in range(len(TARGETS))
    ]
    out = np.stack(attrs)
    out[:, nan_mask.numpy()] = 0.0
    return out


def to_segments(step_attr: np.ndarray) -> np.ndarray:
    """Sum step attributions into N_SEG lag segments; index 0 = most recent 32 steps."""
    a = step_attr[..., -M.CONTEXT_LEN :]
    if a.shape[-1] < M.CONTEXT_LEN:
        pad = [(0, 0)] * (a.ndim - 1) + [(M.CONTEXT_LEN - a.shape[-1], 0)]
        a = np.pad(a, pad)
    a = a.reshape(*a.shape[:-1], N_SEG, SEG).sum(-1)
    return a[..., ::-1].copy()


# ---------------------------------------------------------------- perturbation


def masked_inputs(x0: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Apply segment masks [N, N_SEG] (1 = keep) to the network input -> [N, len(x0)]."""
    n_old = max(len(x0) - M.CONTEXT_LEN, 0)
    step_keep = np.repeat(masks[:, ::-1], SEG, axis=1).astype(bool)
    step_keep = np.concatenate([np.ones((len(masks), n_old), bool), step_keep], axis=1)[:, -len(x0) :]
    return np.where(step_keep, x0[None], np.nan).astype(np.float32)


def lime_attribution(w: W.Window, n_samples: int = 600, seed: int = 0, kernel_width: float = 0.25) -> np.ndarray:
    """LIME weights per lag segment and target, [4, N_SEG]."""
    rng = np.random.default_rng(seed)
    x0 = network_input(w)
    n_off = rng.integers(1, N_SEG, size=n_samples)
    masks = np.ones((n_samples, N_SEG))
    for m, k in zip(masks, n_off):
        m[rng.choice(N_SEG, k, replace=False)] = 0
    masks[0] = 1
    y = evaluate(w, masked_inputs(x0, masks))
    distances = 1 - masks.sum(1) / N_SEG  # cosine distance to the all-ones vector ~ share masked
    lime = LimeBase(kernel_fn=lambda d: np.sqrt(np.exp(-(d**2) / kernel_width**2)), random_state=seed)
    out = np.zeros((len(TARGETS), N_SEG))
    for k in range(len(TARGETS)):
        _, coefs, _, _ = lime.explain_instance_with_data(
            masks, y[:, k : k + 1], distances, 0, N_SEG, feature_selection="none", model_regressor=Ridge(alpha=1.0)
        )
        for idx, c in coefs:
            out[k, idx] = c
    return out


def kernel_shap(w: W.Window, n_samples: int = 1024, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """KernelSHAP values per lag segment and target ([4, N_SEG]) and the base value ([4])."""
    import shap

    x0 = network_input(w)
    f = lambda z: evaluate(w, masked_inputs(x0, z))  # noqa: E731
    explainer = shap.KernelExplainer(f, np.zeros((1, N_SEG)), silent=True)
    np.random.seed(seed)
    sv = explainer.shap_values(np.ones((1, N_SEG)), nsamples=n_samples, l1_reg=False, silent=True)
    sv = np.asarray(sv)
    sv = sv[0].T if sv.ndim == 3 and sv.shape[0] == 1 else np.stack([s[0] for s in sv])
    return sv, np.asarray(explainer.expected_value)


# ---------------------------------------------------------------- STL Shapley


def stl_components(w: W.Window) -> dict[str, np.ndarray]:
    x0 = network_input(w)
    x = np.nan_to_num(x0, nan=np.nanmean(x0)).astype(float)
    level = np.full_like(x, x.mean())
    p = w.seasonality
    if 2 <= p <= len(x) // 3:
        res = STL(x, period=p, robust=True).fit()
        trend, seasonal, resid = res.trend, res.seasonal, res.resid
    else:
        from statsmodels.nonparametric.smoothers_lowess import lowess

        trend = lowess(x, np.arange(len(x)), frac=0.1, return_sorted=False)
        seasonal, resid = np.zeros_like(x), x - trend
    return {"level": level, "trend": trend - level, "seasonal": seasonal, "residual": resid}


def stl_shapley(w: W.Window) -> tuple[np.ndarray, dict]:
    """Exact Shapley values of {trend, seasonal, residual} for each target -> [3, 4]."""
    comps = stl_components(w)
    players = ["trend", "seasonal", "residual"]
    coalitions = [c for r in range(4) for c in combinations(range(3), r)]
    nan_mask = np.isnan(network_input(w))
    inputs = []
    for c in coalitions:
        x = comps["level"] + sum((comps[players[i]] for i in c), np.zeros_like(comps["level"]))
        if len(c) == 0:  # a perfectly flat input is degenerate for the scaler; add tiny jitter
            x = x + 1e-6 * np.abs(comps["level"][0] + 1) * np.sin(np.arange(len(x)))
        inputs.append(np.where(nan_mask, np.nan, x).astype(np.float32))
    v = dict(zip(coalitions, evaluate(w, inputs)))
    n = 3
    phi = np.zeros((n, len(TARGETS)))
    for i in range(n):
        for c in coalitions:
            if i in c:
                continue
            weight = factorial(len(c)) * factorial(n - len(c) - 1) / factorial(n)
            phi[i] += weight * (v[tuple(sorted(c + (i,)))] - v[c])
    energy = {k: float(np.var(comps[k])) for k in players}
    return phi, energy


# ---------------------------------------------------------------- summaries


def lag_summary(seg_attr: np.ndarray, seasonality: int) -> dict:
    """Scale-free descriptors of a lag-segment attribution vector (index 0 = most recent)."""
    a = np.abs(seg_attr)
    tot = a.sum()
    if not np.isfinite(tot) or tot <= 0:
        return {"recency1": np.nan, "recency4": np.nan, "lookback80": np.nan, "lookback80_season": np.nan}
    share = a / tot
    cum = np.cumsum(share)
    lb = (int(np.searchsorted(cum, 0.8)) + 1) * SEG
    return {
        "recency1": float(share[0]),
        "recency4": float(share[:4].sum()),
        "lookback80": lb,
        "lookback80_season": lb / max(seasonality, 1),
    }
