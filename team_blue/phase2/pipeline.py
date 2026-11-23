"""Shared sample of analysis windows and cached experiment runners."""

from __future__ import annotations

import groups as G
import model_io as M
import numpy as np
import pandas as pd
import windows as W
from phase2 import CACHE_DIR


def load_sample(n_per_group: int = 100, force: bool = False) -> pd.DataFrame:
    path = CACHE_DIR / f"sample_{n_per_group}.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    s = G.sample_groups(G.build_behaviour_table("short"), n_per_group=n_per_group)
    s.to_parquet(path)
    return s


def unique_windows(sample: pd.DataFrame) -> pd.DataFrame:
    return sample.drop_duplicates("uid").reset_index(drop=True)


def window_of(row) -> W.Window:
    return W.get_window(row)


def _mase(fc_median: np.ndarray, w: W.Window) -> float:
    return float(np.nanmean(np.abs(fc_median - w.truth)) / w.mase_scale)


def flagged_population_check(
    max_windows: int = 3000, force: bool = False
) -> pd.DataFrame:
    """tta_diff on/off for (up to ``max_windows``) windows where the trend detector fires."""
    path = CACHE_DIR / "tta_diff_flagged.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    t = G.build_behaviour_table("short")
    f = t[t["trend_flag"]].sample(frac=1, random_state=0).head(max_windows).copy()
    f["uid"] = f[W.KEYS].astype(str).agg("|".join, axis=1)
    out = tta_diff_check(f, force=True, path=path)
    return out.merge(f.drop(columns="trend_flag"), on="uid")


def tta_diff_check(
    sample: pd.DataFrame, force: bool = False, path=None
) -> pd.DataFrame:
    """Forecast every window with the differencing path forced on (official) and off."""
    path = path or CACHE_DIR / "tta_diff_check.parquet"
    if path.exists() and not force:
        return pd.read_parquet(path)
    uw = unique_windows(sample)
    rows = []
    for (ds, fr, H), part in uw.groupby(["dataset", "freq", "prediction_length"]):
        wins = [window_of(r) for r in part.itertuples()]
        ctx = [w.context for w in wins]
        on, off = (
            M.forecast(ctx, int(H), tta_diff=True),
            M.forecast(ctx, int(H), tta_diff=False),
        )
        for r, w, a, b in zip(part.itertuples(), wins, on, off):
            rows.append(
                {
                    "uid": r.uid,
                    "trend_flag": r.trend_flag,
                    "MASE_on": _mase(a[M.MEDIAN], w),
                    "MASE_off": _mase(b[M.MEDIAN], w),
                    "width_on": float(np.mean(a[-1] - a[0]) / w.mase_scale),
                    "width_off": float(np.mean(b[-1] - b[0]) / w.mase_scale),
                    "max_abs_diff": float(np.max(np.abs(a - b)) / w.mase_scale),
                }
            )
    out = pd.DataFrame(rows)
    out.to_parquet(path)
    return out


ATTR_DIR = CACHE_DIR / "attr"


def _attr_path(uid: str):
    import hashlib

    return ATTR_DIR / (hashlib.md5(uid.encode()).hexdigest() + ".npz")


def run_attributions(sample: pd.DataFrame, limit: int | None = None) -> None:
    """IG, LIME, KernelSHAP and STL-Shapley per unique window, cached one file per window."""
    from tqdm.auto import tqdm

    from . import attribution as A

    ATTR_DIR.mkdir(exist_ok=True)
    uw = unique_windows(sample)
    todo = [r for r in uw.itertuples() if not _attr_path(r.uid).exists()][:limit]
    for r in tqdm(todo, desc="attributions"):
        w = window_of(r)
        ig = A.integrated_gradients(w)
        sv, base = A.kernel_shap(w)
        phi, energy = A.stl_shapley(w)
        np.savez_compressed(
            _attr_path(r.uid),
            uid=r.uid,
            f_x=A.evaluate(w, [A.network_input(w)])[0],
            ig_step=np.pad(
                ig[:, -M.CONTEXT_LEN :],
                ((0, 0), (max(M.CONTEXT_LEN - ig.shape[1], 0), 0)),
            ).astype(np.float32),
            ig=A.to_segments(ig),
            lime=A.lime_attribution(w),
            shap=sv,
            shap_base=base,
            stl=phi,
            stl_energy=np.array([energy[k] for k in ["trend", "seasonal", "residual"]]),
        )


def load_attributions(sample: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for uid in sample["uid"].unique():
        p = _attr_path(uid)
        if p.exists():
            with np.load(p) as z:
                out[uid] = {k: z[k] for k in z.files if k != "uid"}
    return out
