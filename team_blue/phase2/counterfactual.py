"""Counterfactual / synthetic experiments that test the Phase I behaviours causally.

E1 trend pass-through (F4a, F5): add a linear ramp of slope s (in MASE-scale units per horizon)
   to real windows, either over the whole context ("global") or only the recent 4H steps
   ("recent" = trend onset). Ideal pass-through: forecast slope changes by s (beta = 1);
   mean reversion shows up as beta < 0, flat forecasts as beta ~ 0.
E2 level shift (F4b, F8): insert a step of delta context-std at lag L before the forecast origin.
   adopt = how much of the new level the forecast follows (1 = fully, 0.5 = "flat in the middle"),
   width_ratio = interval width relative to the unperturbed forecast (uncertainty reaction).
E3 residual autocorrelation (F1): synthetic seasonal series + AR(1) residual with phi in [-0.9, 0.9]
   and its true continuation; measures error, amplitude and coverage as a function of phi.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import CACHE_DIR
from . import attribution as A
from . import model_io as M
from . import pipeline as P

SLOPES = np.array([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3], dtype=float)
LAGS_H = np.array([0.25, 0.5, 1, 2, 4, 8, 16])


def _fc_slope(q: np.ndarray) -> float:
    med = q[M.MEDIAN]
    return float(np.polyfit(np.arange(len(med)), med, 1)[0] * len(med)) if len(med) > 1 else float(med[0])


def _pick(sample: pd.DataFrame, groups: list[str], n: int, seed: int = 0) -> pd.DataFrame:
    s = sample[sample["group"].isin(groups)].drop_duplicates(["uid", "group"])
    return s.groupby("group", group_keys=False).apply(lambda g: g.sample(min(n, len(g)), random_state=seed))


def trend_passthrough(sample: pd.DataFrame, n: int = 50, force: bool = False) -> pd.DataFrame:
    path = CACHE_DIR / "cf_trend.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    rows = []
    for r in _pick(sample, ["trend_fail", "trend_ok", "season_ok", "switch_fail"], n).itertuples():
        w = P.window_of(r)
        x0 = A.network_input(w)
        H, sc = len(w.truth), w.mase_scale
        t = np.arange(len(x0)) - (len(x0) - 1)  # 0 at the forecast origin
        variants, meta = [], []
        for mode in ["global", "recent"]:
            # both ramps are 0 at the forecast origin; "recent" is flat before the onset at -4H
            ramp_t = t if mode == "global" else np.maximum(t, -4 * H)
            for s in SLOPES:
                variants.append((x0 + s * sc * ramp_t / H).astype(np.float32))
                meta.append((mode, s))
        for diff in [True, False]:
            q = M.forecast(variants, H, tta_diff=diff)
            flags = [M.trend_flag(v) for v in variants] if diff else [False] * len(variants)
            for (mode, s), qi, fl in zip(meta, q, flags):
                rows.append({
                    "uid": r.uid, "group": r.group, "dataset": r.dataset, "mode": mode, "slope_in": s,
                    "tta_diff": diff, "trend_flag": fl, "slope_out": _fc_slope(qi) / sc,
                    "width": float(np.mean(qi[-1] - qi[0]) / sc),
                })
    out = pd.DataFrame(rows)
    base = out[out["slope_in"] == 0].set_index(["uid", "group", "mode", "tta_diff"])["slope_out"]
    out["delta_slope"] = out["slope_out"] - base.loc[list(zip(out["uid"], out["group"], out["mode"], out["tta_diff"]))].to_numpy()
    out.to_parquet(path)
    return out


def passthrough_beta(cf: pd.DataFrame) -> pd.DataFrame:
    """Per window/mode/diff setting: regression slope of delta forecast slope on injected slope."""
    def beta(g):
        return pd.Series({"beta": np.polyfit(g["slope_in"], g["delta_slope"], 1)[0], "flag_share": g["trend_flag"].mean()})

    return cf.groupby(["uid", "group", "dataset", "mode", "tta_diff"]).apply(beta, include_groups=False).reset_index()


def level_shift(sample: pd.DataFrame, n: int = 50, delta: float = 3.0, force: bool = False) -> pd.DataFrame:
    path = CACHE_DIR / "cf_level.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    rows = []
    for r in _pick(sample, ["switch_fail", "season_ok", "overconf", "calibrated"], n).itertuples():
        w = P.window_of(r)
        x0 = A.network_input(w)
        H = len(w.truth)
        sd = float(np.nanstd(x0[-M.CONTEXT_LEN :])) or 1.0
        variants, lags = [x0], [0]
        for lh in LAGS_H:
            L = max(1, int(round(lh * H)))
            if L >= M.CONTEXT_LEN:
                continue
            v = x0.copy()
            v[-L:] += delta * sd
            variants.append(v)
            lags.append(L)
        q = M.forecast(variants, H)
        base_level, base_width = q[0, M.MEDIAN].mean(), np.mean(q[0, -1] - q[0, 0])
        for L, qi in zip(lags[1:], q[1:]):
            rows.append({
                "uid": r.uid, "group": r.group, "dataset": r.dataset, "lag": L, "lag_H": L / H,
                "lag_season": L / max(w.seasonality, 1),
                "adopt": float((qi[M.MEDIAN].mean() - base_level) / (delta * sd)),
                "width_ratio": float(np.mean(qi[-1] - qi[0]) / max(base_width, 1e-9)),
            })
    out = pd.DataFrame(rows)
    out.to_parquet(path)
    return out


def ar_residual_experiment(
    phis=np.round(np.linspace(-0.9, 0.9, 13), 2), n_seeds: int = 30, period: int = 24, H: int = 24,
    noise: float = 0.5, force: bool = False,
) -> pd.DataFrame:
    path = CACHE_DIR / "cf_ar.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    rows = []
    n = M.CONTEXT_LEN + H
    for phi in phis:
        series, truths = [], []
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            e = rng.normal(size=n) * noise * np.sqrt(1 - phi**2)  # stationary variance noise^2
            r = np.zeros(n)
            r[0] = rng.normal() * noise
            for i in range(1, n):
                r[i] = phi * r[i - 1] + e[i]
            y = np.sin(2 * np.pi * np.arange(n) / period + rng.uniform(0, 2 * np.pi)) + r
            series.append(y[:-H].astype(np.float32))
            truths.append(y[-H:])
        q = M.forecast(series, H)
        for seed, (x, y, qi) in enumerate(zip(series, truths, q)):
            scale = np.mean(np.abs(x[period:] - x[:-period]))
            rows.append({
                "phi": phi, "seed": seed,
                "MASE": float(np.mean(np.abs(qi[M.MEDIAN] - y)) / scale),
                "amplitude_ratio": float(np.std(qi[M.MEDIAN]) / np.std(np.sin(2 * np.pi * np.arange(H) / period))),
                "coverage80": float(np.mean((y >= qi[0]) & (y <= qi[-1]))),
                "width": float(np.mean(qi[-1] - qi[0]) / scale),
            })
    out = pd.DataFrame(rows)
    out.to_parquet(path)
    return out
