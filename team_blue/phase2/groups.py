"""Behaviour table for every forecast window and matched behaviour-group sampling."""

from __future__ import annotations

import model_io as M
import numpy as np
import pandas as pd
import windows as W
from phase2 import CACHE_DIR

FEATURES = [
    "feat_trend_strength",
    "feat_seasonal_strength",
    "feat_e_acf1",
    "feat_x_entropy",
    "feat_e_entropy",
    "feat_trend_stability",
    "feat_spike",
    "feat_linearity",
]


def _slope(y: np.ndarray) -> float:
    y = y[~np.isnan(y)]
    if len(y) < 3:
        return np.nan
    x = np.arange(len(y), dtype=float)
    return float(np.polyfit(x, y, 1)[0])


def window_behaviour(w: W.Window) -> dict:
    """Forecast behaviour descriptors of one window (all scale-free via the MASE scale)."""
    H = len(w.truth)
    s = w.mase_scale if np.isfinite(w.mase_scale) and w.mase_scale > 0 else np.nan
    med, lo, hi = w.stored[M.MEDIAN], w.stored[0], w.stored[-1]
    recent = w.context[-max(4 * H, 2 * w.seasonality, 32) :]
    valid_truth = ~np.isnan(w.truth)
    return {
        "coverage80": float(
            np.mean((w.truth >= lo) & (w.truth <= hi) & valid_truth)
            / max(valid_truth.mean(), 1e-9)
        ),
        "width_rel": float(np.mean(hi - lo) / s),
        # flatness: variability of the median forecast relative to the recent history
        "flat_ratio": float(np.std(med) / (np.nanstd(recent[-H:]) + 1e-9))
        if H > 1
        else np.nan,
        "hist_slope": _slope(recent) * H / s,
        "fc_slope": _slope(med) * H / s,
        "truth_slope": _slope(w.truth) * H / s,
        "level_gap": float((med[0] - w.context[~np.isnan(w.context)][-1]) / s),
        "ctx_len": len(w.context),
        "trend_flag": M.trend_flag(w.context),
    }


def build_behaviour_table(term: str = "short", force: bool = False) -> pd.DataFrame:
    path = CACHE_DIR / f"behaviour_{term}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    wc = W.load_window_cube(term)
    wc = wc[np.isfinite(wc["MASE"])]
    rows = [window_behaviour(W.get_window(r)) for r in wc.itertuples()]
    keep = [
        *W.KEYS,
        "MASE",
        "sMAPE",
        "CRPS",
        "prediction_length",
        "seasonality",
        *FEATURES,
    ]
    table = pd.concat([wc[keep].reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    table.to_parquet(path)
    return table


def define_groups(t: pd.DataFrame) -> pd.DataFrame:
    """Boolean membership columns for the Phase I behaviour groups (windows may be in several)."""
    ts, ss, acf = (
        t["feat_trend_strength"],
        t["feat_seasonal_strength"],
        t["feat_e_acf1"],
    )
    ts_hi, ts_lo = ts.quantile(0.9), ts.quantile(0.1)
    fail, ok = t["MASE"] > 1.1, t["MASE"] < 0.7
    g = pd.DataFrame(index=t.index)
    # F4a / F7: strong, non-seasonal trend
    g["trend_fail"] = (ts >= ts_hi) & (ss < 0.5) & fail
    g["trend_ok"] = (ts >= ts_hi) & (ss < 0.5) & ok
    # F4b / F6: weak trend, level switching, model flat-lines
    g["switch_fail"] = (ts <= ts_lo) & fail & (t["flat_ratio"] < 0.3)
    g["season_ok"] = (ss >= 0.8) & ok
    # F1: negative vs clearly positive lag-1 residual autocorrelation
    g["negacf"] = acf < 0
    g["posacf"] = acf > 0.5
    # F8: over-confident (80 % band misses most of the truth) vs well calibrated
    g["overconf"] = (t["coverage80"] <= 0.2) & (t["MASE"] > 1.2)
    g["calibrated"] = (t["coverage80"] >= 0.8) & ok
    return g


CONTRASTS = {
    "trend": ("trend_fail", "trend_ok"),
    "switch": ("switch_fail", "season_ok"),
    "acf": ("negacf", "posacf"),
    "calib": ("overconf", "calibrated"),
}


def sample_groups(
    t: pd.DataFrame, n_per_group: int = 100, max_per_dataset: int = 15, seed: int = 0
) -> pd.DataFrame:
    """Stratified sample per group; for each contrast both sides prefer shared datasets
    (within-dataset matching, Phase I showed pooled effects are dataset confounded)."""
    rng = np.random.default_rng(seed)
    g = define_groups(t)
    out = []
    for contrast, (a, b) in CONTRASTS.items():
        shared = set(t.loc[g[a], "dataset"]) & set(t.loc[g[b], "dataset"])
        for name in (a, b):
            pool = t[g[name]].copy()
            pool["shared"] = pool["dataset"].isin(shared)
            # one window per series first (diversity), shared datasets first, then cap per dataset
            pool = pool.sample(frac=1, random_state=int(rng.integers(1 << 31)))
            pool = pool.drop_duplicates(
                ["dataset", "freq", "series_idx", "variate_idx"]
            )
            pool = pool.sort_values("shared", ascending=False, kind="stable")
            pool = pool.groupby("dataset", group_keys=False).head(max_per_dataset)
            pick = pool.head(n_per_group).copy()
            pick["group"], pick["contrast"] = name, contrast
            pick["side"] = "fail" if name == a else "ok"
            out.append(pick)
    s = pd.concat(out, ignore_index=True)
    s["uid"] = s[W.KEYS].astype(str).agg("|".join, axis=1)
    return s.drop(columns="shared")
