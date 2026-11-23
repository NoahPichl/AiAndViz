"""Aggregations and statistics on top of the cached attributions and representations."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from . import attribution as A
from . import groups as G

METHODS = ["ig", "lime", "shap"]


def attribution_summary(sample: pd.DataFrame, attrs: dict) -> pd.DataFrame:
    """Long table: one row per (window, group, method, target) with lag descriptors."""
    rows = []
    for r in sample.itertuples():
        a = attrs.get(r.uid)
        if a is None:
            continue
        for m in METHODS:
            for k, target in enumerate(A.TARGETS):
                d = A.lag_summary(a[m][k], int(r.seasonality))
                rows.append({"uid": r.uid, "group": r.group, "contrast": r.contrast, "side": r.side,
                             "dataset": r.dataset, "method": m, "target": target, **d})
    return pd.DataFrame(rows)


def lag_profiles(sample: pd.DataFrame, attrs: dict, method: str, target: str) -> pd.DataFrame:
    """Normalised |attribution| share per lag segment, one row per (window, group)."""
    k = A.TARGETS.index(target)
    rows = []
    for r in sample.itertuples():
        a = attrs.get(r.uid)
        if a is None:
            continue
        v = np.abs(a[method][k])
        if v.sum() > 0:
            rows.append({"uid": r.uid, "group": r.group, "contrast": r.contrast, "side": r.side,
                         "dataset": r.dataset, **{i: x for i, x in enumerate(v / v.sum())}})
    return pd.DataFrame(rows)


def within_dataset_test(summary: pd.DataFrame, metric: str = "recency4") -> pd.DataFrame:
    """Fail vs ok per contrast, paired by dataset (median per dataset and side), Wilcoxon test.

    Pooling across datasets would mostly compare datasets (Phase I lesson), so each dataset that
    has windows on both sides contributes one paired difference.
    """
    rows = []
    for (contrast, method, target), g in summary.groupby(["contrast", "method", "target"]):
        per = g.groupby(["dataset", "side"])[metric].median().unstack("side").dropna()
        if len(per) < 3:
            continue
        diff = per["fail"] - per["ok"]
        p = wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
        rows.append({"contrast": contrast, "method": method, "target": target, "n_datasets": len(per),
                     "median_fail": per["fail"].median(), "median_ok": per["ok"].median(),
                     "median_diff": diff.median(), "share_fail_higher": (diff > 0).mean(), "p_wilcoxon": p})
    return pd.DataFrame(rows)


def method_agreement(attrs: dict) -> pd.DataFrame:
    """Mean Spearman correlation between the segment attributions of two methods, per target."""
    rows = []
    for a in attrs.values():
        for k, target in enumerate(A.TARGETS):
            for m1, m2 in [("ig", "lime"), ("ig", "shap"), ("lime", "shap")]:
                x, y = np.abs(a[m1][k]), np.abs(a[m2][k])
                if x.std() > 0 and y.std() > 0:
                    rows.append({"target": target, "pair": f"{m1}-{m2}", "rho": spearmanr(x, y).statistic})
    return pd.DataFrame(rows).groupby(["target", "pair"])["rho"].agg(["mean", "median", "count"]).reset_index()


def recency_agreement(summary: pd.DataFrame, metric: str = "recency4") -> pd.DataFrame:
    """Do the methods agree on *which windows* rely on the recent past? Spearman across windows."""
    pv = summary.drop_duplicates(["uid", "method", "target"]).pivot_table(index=["uid", "target"], columns="method", values=metric)
    rows = []
    for target, g in pv.groupby("target"):
        c = g.corr(method="spearman")
        for m1, m2 in [("ig", "lime"), ("ig", "shap"), ("lime", "shap")]:
            rows.append({"target": target, "pair": f"{m1}-{m2}", "mean": c.loc[m1, m2], "count": len(g)})
    return pd.DataFrame(rows)


def direction_table(sample: pd.DataFrame) -> pd.DataFrame:
    """Per group: does the forecast revert against the recent history, and how far does the truth move?"""
    s = sample.drop_duplicates(["uid", "group"]).copy()
    s["forecast_reverts"] = np.sign(s["fc_slope"]) == -np.sign(s["hist_slope"])
    s["truth_continues"] = np.sign(s["truth_slope"]) == np.sign(s["hist_slope"])
    agg = s.groupby("group").agg(
        forecast_reverts=("forecast_reverts", "mean"), truth_continues=("truth_continues", "mean"),
        abs_hist_slope=("hist_slope", lambda x: x.abs().median()), abs_fc_slope=("fc_slope", lambda x: x.abs().median()),
        abs_truth_slope=("truth_slope", lambda x: x.abs().median()), abs_level_gap=("level_gap", lambda x: x.abs().median()),
    )
    return agg


def stl_table(sample: pd.DataFrame, attrs: dict) -> pd.DataFrame:
    """Share of |Shapley| per STL component and target, one row per (window, group, target)."""
    rows = []
    comps = ["trend", "seasonal", "residual"]
    for r in sample.itertuples():
        a = attrs.get(r.uid)
        if a is None:
            continue
        phi = np.abs(a["stl"])
        energy = a["stl_energy"] / max(a["stl_energy"].sum(), 1e-12)
        for k, target in enumerate(A.TARGETS):
            tot = phi[:, k].sum()
            if tot <= 0:
                continue
            rows.append({"uid": r.uid, "group": r.group, "contrast": r.contrast, "side": r.side, "dataset": r.dataset,
                         "target": target, **{c: phi[i, k] / tot for i, c in enumerate(comps)},
                         **{f"energy_{c}": energy[i] for i, c in enumerate(comps)}})
    return pd.DataFrame(rows)


def memory_table(table: pd.DataFrame, data: dict, sample: pd.DataFrame) -> pd.DataFrame:
    """mLSTM memory share in the most recent 4 patches per layer, per (window, group)."""
    rec = data["memory"][:, :, :4].sum(2).mean(2)  # [n, layers]
    idx = table.reset_index().set_index("uid")["index"]
    s = sample.drop_duplicates(["uid", "group"])
    out = pd.DataFrame(rec[idx.loc[s["uid"]].to_numpy()], columns=[f"mlstm_{b}" for b in (0, 2, 4, 6, 8, 10)])
    for c in ["uid", "group", "contrast", "side", "dataset"]:
        out[c] = s[c].to_numpy()
    return out


def group_labels() -> dict[str, str]:
    return {
        "trend_fail": "trend: fail", "trend_ok": "trend: ok",
        "switch_fail": "switch: fail", "season_ok": "seasonal: ok",
        "negacf": "ACF1 < 0", "posacf": "ACF1 > 0.5",
        "overconf": "over-confident", "calibrated": "calibrated",
    }


CONTRASTS = G.CONTRASTS
