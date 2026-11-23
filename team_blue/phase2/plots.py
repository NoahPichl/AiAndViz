"""Phase II figures: static matplotlib for the narrative, plotly/ipywidgets for exploration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import CACHE_DIR
from . import analysis as AN
from . import attribution as A
from . import model_io as M
from . import pipeline as P

FIG_DIR = CACHE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
LABELS = AN.group_labels()
SIDE_COLORS = {"fail": "#d62728", "ok": "#1f77b4"}
METHOD_NAMES = {"ig": "Integrated Gradients", "lime": "LIME", "shap": "KernelSHAP"}


def _save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=130, bbox_inches="tight")
    return fig


def _boot_ci(x: np.ndarray, n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = np.array([x[rng.integers(0, len(x), len(x))].mean(0) for _ in range(n)])
    return np.percentile(means, [5, 95], axis=0)


# ---------------------------------------------------------------- attribution


def lag_profile_grid(sample, attrs, target: str = "level", max_lag: int = 32):
    """Rows: contrasts, columns: methods. Mean |attribution| share per lag segment, fail vs ok."""
    contrasts = list(AN.CONTRASTS)
    fig, axes = plt.subplots(len(contrasts), 3, figsize=(14, 2.6 * len(contrasts)), sharex=True)
    for j, m in enumerate(AN.METHODS):
        prof = AN.lag_profiles(sample, attrs, m, target)
        for i, c in enumerate(contrasts):
            ax = axes[i, j]
            for side, g in prof[prof["contrast"] == c].groupby("side"):
                v = g[list(range(max_lag))].to_numpy()
                lo, hi = _boot_ci(v)
                lag = (np.arange(max_lag) + 0.5) * A.SEG
                ax.plot(lag, v.mean(0), color=SIDE_COLORS[side], label=f"{LABELS[g['group'].iloc[0]]} (n={len(g)})")
                ax.fill_between(lag, lo, hi, color=SIDE_COLORS[side], alpha=0.2)
            ax.set_xscale("log")
            ax.grid(alpha=0.3)
            if i == 0:
                ax.set_title(METHOD_NAMES[m])
            if j == 0:
                ax.set_ylabel(f"{c}\n|attr| share")
            if j == 2:
                ax.legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("lag before forecast origin [steps]")
    fig.suptitle(f"Where in the context does TiRex-2 look? target = {target}", y=1.0)
    fig.tight_layout()
    return _save(fig, f"lag_profiles_{target}")


def recency_by_dataset(summary: pd.DataFrame, method: str = "shap", target: str = "level", metric: str = "recency4"):
    """Within-dataset pairs: per dataset median of the metric for fail (red) and ok (blue)."""
    s = summary[(summary["method"] == method) & (summary["target"] == target)]
    contrasts = list(AN.CONTRASTS)
    fig, axes = plt.subplots(1, len(contrasts), figsize=(4 * len(contrasts), 4), sharey=True)
    for ax, c in zip(axes, contrasts):
        per = s[s["contrast"] == c].groupby(["dataset", "side"])[metric].median().unstack("side")
        if not {"fail", "ok"} <= set(per.columns):
            ax.set_title(f"{c} (no data yet)")
            continue
        per = per.dropna().sort_values("ok")
        y = np.arange(len(per))
        ax.hlines(y, per["ok"], per["fail"], color="grey", lw=1)
        ax.scatter(per["ok"], y, color=SIDE_COLORS["ok"], label="ok", zorder=3)
        ax.scatter(per["fail"], y, color=SIDE_COLORS["fail"], label="fail", zorder=3)
        ax.set_yticks(y, per.index, fontsize=7)
        ax.set_title(f"{c} ({len(per)} shared datasets)")
        ax.set_xlabel(metric)
        ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"{METHOD_NAMES[method]} - {metric} of the '{target}' attribution, paired within dataset")
    fig.tight_layout()
    return _save(fig, f"recency_pairs_{method}_{target}")


def method_agreement_plot(agree: pd.DataFrame):
    pv = agree.pivot(index="pair", columns="target", values="mean")
    fig, ax = plt.subplots(figsize=(6, 2.6))
    im = ax.imshow(pv.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(pv.shape[1]), pv.columns)
    ax.set_yticks(range(pv.shape[0]), pv.index)
    for (i, j), v in np.ndenumerate(pv.to_numpy()):
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="mean Spearman of |attr| over segments")
    ax.set_title("Agreement between attribution methods")
    return _save(fig, "method_agreement")


def window_strip(row, attrs=None, ax_count: int = 3, show_steps: int = 512):
    """One window: context tail + forecast + truth, IG heat strip, SHAP/LIME segment bars."""
    w = P.window_of(row)
    a = (attrs or {}).get(row.uid) or P.load_attributions(pd.DataFrame({"uid": [row.uid]})).get(row.uid)
    x = A.network_input(w)[-show_steps:]
    H = len(w.truth)
    t = np.arange(-len(x), 0)
    fig, axes = plt.subplots(ax_count, 1, figsize=(12, 6), sharex=True, gridspec_kw={"height_ratios": [3, 0.6, 1.4]})
    ax = axes[0]
    ax.plot(t, x, color="k", lw=0.8, label="context")
    ax.plot(np.arange(H), w.truth, color="k", ls="--", lw=0.8, label="truth")
    ax.plot(np.arange(H), w.stored[M.MEDIAN], color="C3", label="median forecast")
    ax.fill_between(np.arange(H), w.stored[0], w.stored[-1], color="C3", alpha=0.25, label="80 % interval")
    ax.set_title(f"{row.dataset}/{row.freq} series {row.series_idx} var {row.variate_idx} win {row.window_idx} | "
                 f"group={row.group} MASE={row.MASE:.2f}")
    ax.legend(fontsize=7, loc="upper left")
    if a is not None:
        ig = a["ig_step"][A.TARGETS.index("level")][-show_steps:]
        axes[1].imshow(ig[None], aspect="auto", cmap="RdBu_r", extent=[-len(ig), 0, 0, 1],
                       vmin=-np.abs(ig).max(), vmax=np.abs(ig).max())
        axes[1].set_yticks([])
        axes[1].set_ylabel("IG", rotation=0, ha="right")
        n_seg = show_steps // A.SEG
        centers = -(np.arange(n_seg) + 0.5) * A.SEG
        k = A.TARGETS.index("level")
        axes[2].bar(centers - 6, a["shap"][k][:n_seg], width=12, label="KernelSHAP", color="C0")
        axes[2].bar(centers + 6, a["lime"][k][:n_seg], width=12, label="LIME", color="C1")
        axes[2].axhline(0, color="k", lw=0.5)
        axes[2].legend(fontsize=7)
        axes[2].set_ylabel("segment\nattribution")
    axes[-1].set_xlabel("steps relative to forecast origin")
    fig.tight_layout()
    return fig


def stl_bars(stl: pd.DataFrame):
    comps = ["trend", "seasonal", "residual"]
    groups = [g for g in LABELS if g in stl["group"].unique()]
    targets = A.TARGETS
    fig, axes = plt.subplots(1, len(targets), figsize=(4 * len(targets), 4), sharey=True)
    colors = {"trend": "#8c564b", "seasonal": "#2ca02c", "residual": "#7f7f7f"}
    for ax, target in zip(axes, targets):
        m = stl[stl["target"] == target].groupby("group")[comps].mean().reindex(groups)
        bottom = np.zeros(len(m))
        for c in comps:
            ax.barh(range(len(m)), m[c], left=bottom, color=colors[c], label=c)
            bottom += m[c].to_numpy()
        ax.set_yticks(range(len(m)), [LABELS[g] for g in m.index])
        ax.set_title(f"target = {target}")
        ax.set_xlim(0, 1)
    axes[0].legend(fontsize=8, loc="lower left")
    fig.suptitle("STL-component Shapley: which signal component drives the forecast?")
    fig.tight_layout()
    return _save(fig, "stl_shapley")


# ---------------------------------------------------------------- representations


def probe_curves(probes: pd.DataFrame, auc: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for target, g in probes.groupby("target"):
        axes[0].plot(g["layer"], g["r2"].clip(lower=-0.2), marker="o", ms=3, label=target)
    axes[0].set(xlabel="layer (0 = input embedding)", ylabel="dataset-grouped CV $R^2$",
                title="What is linearly decodable from the last context token?")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)
    for c, g in auc.groupby("contrast"):
        axes[1].plot(g["layer"], g["auc"], marker="o", ms=3, label=c)
    axes[1].axhline(0.5, color="k", lw=0.5)
    axes[1].set(xlabel="layer", ylabel="dataset-grouped CV AUC", title="Can a linear probe separate fail vs ok?")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    return _save(fig, "probes")


def knn_plot(knn: pd.DataFrame):
    cols = ["fail_purity", "same_dataset", "rho_log_MASE", "rho_log_flat_ratio", "rho_fc_slope"]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3.2), sharey=True)
    for ax, c in zip(axes, cols):
        ax.barh(knn["space"], knn[c], color=["#999"] + ["C0"] * (len(knn) - 1))
        if c == "fail_purity":
            ax.axvline(knn["fail_base_rate"].iloc[0], color="k", ls=":", label="base rate")
            ax.legend(fontsize=7)
        ax.set_title(c, fontsize=9)
        ax.grid(alpha=0.3)
    fig.suptitle("kNN: TIME feature space (Phase I) vs. TiRex-2 representation space (rho: neighbours from other datasets)")
    fig.tight_layout()
    return _save(fig, "knn")


def memory_plot(table, data, sample):
    s = sample.drop_duplicates(["uid", "group"])
    idx = table.reset_index().set_index("uid")["index"]
    groups = [g for g in LABELS if g in s["group"].unique()]
    layers = [0, 2, 4, 6, 8, 10]
    fig, axes = plt.subplots(1, len(layers), figsize=(3 * len(layers), 3.2), sharey=True)
    lag = (np.arange(M.N_CTX_PATCHES) + 0.5) * M.PATCH
    for j, ax in enumerate(axes):
        for gname in groups:
            ii = idx.loc[s.loc[s["group"] == gname, "uid"]].to_numpy()
            mem = data["memory"][ii, j].mean(-1)  # average heads
            side = s.loc[s["group"] == gname, "side"].iloc[0]
            ax.plot(lag, np.cumsum(mem.mean(0)), label=LABELS[gname], color=SIDE_COLORS[side],
                    ls="-" if gname in ("trend_fail", "trend_ok") else ":", alpha=0.85)
        ax.set_xscale("log")
        ax.set_title(f"mLSTM block {layers[j]}")
        ax.grid(alpha=0.3)
        ax.set_xlabel("lag [steps]")
    axes[0].set_ylabel("cumulative memory weight")
    axes[-1].legend(fontsize=6)
    fig.suptitle("mLSTM gate memory of the last context token (exact decay weights)")
    fig.tight_layout()
    return _save(fig, "gate_memory")


def representation_explorer(table, data, maps: dict, sample):
    """Interactive layer-wise UMAP of hidden states. Slider = layer, dropdown = colour,
    click a point to show its forecast and attribution strip below."""
    import ipywidgets as widgets
    import plotly.graph_objects as go

    t = table.copy()
    t["log_MASE"] = np.log10(t["MASE"].clip(1e-3, 1e3))
    t["log_flat_ratio"] = np.log10(t["flat_ratio"].clip(1e-3, 1e3))
    grp = sample.groupby("uid")["group"].agg(lambda g: ", ".join(sorted(set(g))))
    t["groups"] = t["uid"].map(grp).fillna("background")
    color_opts = ["log_MASE", "log_flat_ratio", "fc_slope", "hist_slope", "coverage80", "feat_trend_strength",
                  "feat_seasonal_strength", "feat_e_acf1", "dataset", "groups"]
    hover = (t["dataset"] + "/" + t["freq"] + "<br>" + t["groups"] + "<br>MASE=" + t["MASE"].round(2).astype(str))

    def colour(name):
        v = t[name]
        if v.dtype == object:
            codes, uniq = pd.factorize(v)
            return dict(color=codes, colorscale="Turbo", showscale=False)
        lo, hi = np.nanquantile(v, [0.02, 0.98])
        return dict(color=v.clip(lo, hi), colorscale="Viridis", showscale=True, colorbar=dict(title=name))

    fig = go.FigureWidget()
    fig.add_scattergl(x=maps[12][:, 0], y=maps[12][:, 1], mode="markers", text=hover, hoverinfo="text",
                      marker=dict(size=np.where(t["in_sample"], 6, 3), opacity=0.8, **colour("log_MASE")))
    fig.update_layout(width=850, height=650, title="Last-context-token state, layer 12 (UMAP)",
                      xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False))
    layer = widgets.IntSlider(value=12, min=0, max=12, description="layer")
    col = widgets.Dropdown(options=color_opts, value="log_MASE", description="colour")
    out = widgets.Output()

    def update(_=None):
        with fig.batch_update():
            m = maps[layer.value]
            fig.data[0].x, fig.data[0].y = m[:, 0], m[:, 1]
            fig.data[0].marker.update(**colour(col.value))
            fig.layout.title = f"Last-context-token state, layer {layer.value} (UMAP) - colour: {col.value}"

    def on_click(trace, points, _state):
        if not points.point_inds:
            return
        r = t.iloc[points.point_inds[0]].copy()
        r["group"] = r["groups"]
        with out:
            out.clear_output(wait=True)
            plt.show(window_strip(pd.Series(r).to_frame().T.itertuples().__next__()))

    layer.observe(update, "value")
    col.observe(update, "value")
    fig.data[0].on_click(on_click)
    return widgets.VBox([widgets.HBox([layer, col]), fig, out])


# ---------------------------------------------------------------- counterfactuals / diff


def passthrough_plot(cf: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    panels = [("global", False), ("global", True), ("recent", True)]
    for ax, (mode, diff) in zip(axes, panels):
        d = cf[(cf["mode"] == mode) & (cf["tta_diff"] == diff)]
        for gname, g in d.groupby("group"):
            m = g.groupby("slope_in")["delta_slope"].median()
            ax.plot(m.index, m.values, marker="o", label=LABELS[gname])
        ax.plot(cf["slope_in"].unique(), cf["slope_in"].unique(), color="k", ls=":", label="full pass-through")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set(title=f"{mode} ramp, tta_diff={'on' if diff else 'off'}", xlabel="injected slope [MASE scale / horizon]")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("change in forecast slope")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Counterfactual trend injection: does TiRex-2 extrapolate a trend?")
    fig.tight_layout()
    return _save(fig, "cf_trend")


def level_shift_plot(lv: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for gname, g in lv.groupby("group"):
        m = g.groupby("lag_H")[["adopt", "width_ratio"]].median()
        axes[0].plot(m.index, m["adopt"], marker="o", label=LABELS[gname])
        axes[1].plot(m.index, m["width_ratio"], marker="o", label=LABELS[gname])
    axes[0].axhline(1, color="k", ls=":")
    axes[0].axhline(0.5, color="grey", ls=":")
    axes[0].set(xscale="log", xlabel="steps since the level shift [x horizon]", ylabel="share of new level adopted",
                title="How fast is a level shift adopted?")
    axes[1].axhline(1, color="k", ls=":")
    axes[1].set(xscale="log", xlabel="steps since the level shift [x horizon]", ylabel="interval width / unperturbed",
                title="Does uncertainty react to the break?")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "cf_level")


def ar_plot(ar: pd.DataFrame):
    m = ar.groupby("phi")[["MASE", "amplitude_ratio", "coverage80", "width"]].median()
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))
    for ax, c in zip(axes, m.columns):
        ax.plot(m.index, m[c], marker="o")
        ax.set(title=c, xlabel="AR(1) coefficient of residual")
        ax.grid(alpha=0.3)
    fig.suptitle("Synthetic seasonal series: effect of residual autocorrelation (F1)")
    fig.tight_layout()
    return _save(fig, "cf_ar")


def tta_diff_plot(flagged: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lim = np.nanquantile(np.r_[flagged["MASE_on"], flagged["MASE_off"]], 0.99)
    axes[0].scatter(flagged["MASE_off"], flagged["MASE_on"], s=6, alpha=0.4)
    axes[0].plot([0, lim], [0, lim], "k:")
    axes[0].set(xlim=(0, lim), ylim=(0, lim), xlabel="MASE, differencing off", ylabel="MASE, differencing on (official)",
                title=f"{len(flagged)} windows where the trend rule fires")
    r = (flagged["width_on"] / flagged["width_off"]).clip(0, 3)
    axes[1].hist(r, bins=40)
    axes[1].axvline(1, color="k", ls=":")
    axes[1].set(xlabel="interval width on / off", title="Differencing narrows the intervals")
    fig.tight_layout()
    return _save(fig, "tta_diff")
