"""Hidden representations of TiRex-2: extraction, per-layer maps, linear probes and kNN analysis.

The token that emits the first forecast patch is the last context token (index 63 of 74); for
H <= 32 it is the only token that matters for the forecast. Its state after every block (plus
the input embedding, 13 layers in total) is the representation analysed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import CACHE_DIR
from . import attribution as A
from . import groups as G
from . import model_io as M
from . import pipeline as P
from . import windows as W

REPR_PATH = CACHE_DIR / "representations.npz"
MLSTM_BLOCKS = [0, 2, 4, 6, 8, 10]


def background_windows(n: int = 4000, max_per_dataset: int = 150, seed: int = 1) -> pd.DataFrame:
    """Random short-term windows (one per series where possible) as context for the maps."""
    t = G.build_behaviour_table("short")
    t = t.sample(frac=1, random_state=seed).drop_duplicates(["dataset", "freq", "series_idx", "variate_idx"])
    t = t.groupby("dataset", group_keys=False).head(max_per_dataset).head(n).copy()
    t["uid"] = t[W.KEYS].astype(str).agg("|".join, axis=1)
    t["group"], t["contrast"], t["side"] = "background", "none", "none"
    return t


def analysis_table(sample: pd.DataFrame) -> pd.DataFrame:
    """Unique windows: analysis sample first, then background windows not in the sample."""
    uw = P.unique_windows(sample)
    bg = background_windows()
    bg = bg[~bg["uid"].isin(uw["uid"])]
    return pd.concat([uw.assign(in_sample=True), bg.assign(in_sample=False)], ignore_index=True)


def extract(sample: pd.DataFrame, force: bool = False, batch_size: int = 64) -> tuple[pd.DataFrame, dict]:
    if REPR_PATH.exists() and not force:
        with np.load(REPR_PATH, allow_pickle=True) as z:
            data = {k: z[k] for k in z.files}
        return pd.read_parquet(CACHE_DIR / "representations_index.parquet"), data
    table = analysis_table(sample)
    n = len(table)
    states = np.zeros((n, 13, 512), np.float16)
    memory = np.zeros((n, len(MLSTM_BLOCKS), M.N_CTX_PATCHES, 4), np.float32)
    fgate = np.zeros((n, len(MLSTM_BLOCKS), M.N_CTX_PATCHES, 4), np.float32)
    for H, part in table.groupby("prediction_length"):
        idx = part.index.to_numpy()
        for s in range(0, len(idx), batch_size):
            ii = idx[s : s + batch_size]
            ctx = [A.network_input(W.get_window(table.loc[i])) for i in ii]
            out = M.internals(ctx, int(H), gates=True)
            t = out["last_ctx_token"]
            states[ii] = out["states"][:, :, t]
            for j, b in enumerate(MLSTM_BLOCKS):
                g = out["gates"][b]
                fgate[ii, j] = 1 / (1 + np.exp(-g["f"][:, t - M.N_CTX_PATCHES + 1 : t + 1][:, ::-1]))
                for k, i in enumerate(ii):
                    memory[i, j] = M.mlstm_memory_weights(g["i"][k], g["f"][k], t)[-M.N_CTX_PATCHES :][::-1]
    data = {"states": states, "memory": memory, "fgate": fgate}
    np.savez_compressed(REPR_PATH, **data)
    table.to_parquet(CACHE_DIR / "representations_index.parquet")
    return table, data


# ---------------------------------------------------------------- maps


def layer_maps(states: np.ndarray, layers=None, method: str = "umap", seed: int = 0) -> dict[int, np.ndarray]:
    """2-D embedding of the last-context-token state per layer (cached)."""
    path = CACHE_DIR / f"layer_maps_{method}.npz"
    if path.exists():
        with np.load(path) as z:
            cached = {int(k): z[k] for k in z.files}
        if len(cached) == states.shape[1] and next(iter(cached.values())).shape[0] == len(states):
            return cached
    layers = range(states.shape[1]) if layers is None else layers
    out = {}
    for layer in layers:
        x = PCA(50, random_state=seed).fit_transform(StandardScaler().fit_transform(states[:, layer].astype(np.float32)))
        if method == "umap":
            import umap

            out[layer] = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=seed).fit_transform(x)
        else:
            out[layer] = x[:, :2]
    np.savez_compressed(path, **{str(k): v for k, v in out.items()})
    return out


# ---------------------------------------------------------------- probes


PROBE_TARGETS = {
    "hist_slope": "regression",
    "feat_trend_strength": "regression",
    "feat_seasonal_strength": "regression",
    "feat_e_acf1": "regression",
    "feat_x_entropy": "regression",
    "fc_slope": "regression",
    "log_flat_ratio": "regression",
    "log_MASE": "regression",
}


def _clip(y: np.ndarray) -> np.ndarray:
    lo, hi = np.nanquantile(y, [0.01, 0.99])
    return np.clip(y, lo, hi)


def probe_frame(table: pd.DataFrame) -> pd.DataFrame:
    t = table.copy()
    t["log_flat_ratio"] = np.log10(t["flat_ratio"].clip(1e-3, 1e3))
    t["log_MASE"] = np.log10(t["MASE"].clip(1e-3, 1e3))
    return t


def linear_probes(states: np.ndarray, table: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    """Cross-validated R^2 of ridge probes per layer; folds are grouped by dataset so a probe
    cannot succeed by merely recognising the dataset."""
    t = probe_frame(table)
    groups = t["dataset"].to_numpy()
    rows = []
    for target in PROBE_TARGETS:
        y = t[target].to_numpy(float)
        ok = np.isfinite(y)
        y = _clip(y[ok])
        for layer in range(states.shape[1]):
            X = states[ok, layer].astype(np.float32)
            pred = np.zeros_like(y)
            for tr, te in GroupKFold(n_splits).split(X, y, groups[ok]):
                model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 9)))
                pred[te] = model.fit(X[tr], y[tr]).predict(X[te])
            rows.append({"target": target, "layer": layer, "r2": r2_score(y, pred)})
    return pd.DataFrame(rows)


def probe_predictions(states: np.ndarray, table: pd.DataFrame, target: str, layer: int, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold probe predictions (NaN where the target is missing)."""
    t = probe_frame(table)
    y = t[target].to_numpy(float)
    ok = np.isfinite(y)
    yc = _clip(y[ok])
    X = states[ok, layer].astype(np.float32)
    pred = np.zeros_like(yc)
    for tr, te in GroupKFold(n_splits).split(X, yc, t["dataset"].to_numpy()[ok]):
        model = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 4, 9)))
        pred[te] = model.fit(X[tr], yc[tr]).predict(X[te])
    out = np.full(len(t), np.nan)
    out[ok] = pred
    return out


def contrast_probe_auc(states: np.ndarray, table: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    """Per layer: can a linear probe separate fail vs ok of each contrast (dataset-grouped CV)?"""
    rows = []
    for contrast, (a, b) in G.CONTRASTS.items():
        sub = sample[sample["contrast"] == contrast].drop_duplicates("uid")
        idx = table.reset_index().set_index("uid").loc[sub["uid"], "index"].to_numpy()
        y = (sub["side"] == "fail").to_numpy(int)
        g = sub["dataset"].to_numpy()
        n_splits = min(5, len(np.unique(g)))
        for layer in range(states.shape[1]):
            X = states[idx, layer].astype(np.float32)
            score = np.zeros(len(y))
            for tr, te in GroupKFold(n_splits).split(X, y, g):
                if len(np.unique(y[tr])) < 2:
                    score[te] = 0.5
                    continue
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=2000))
                score[te] = clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            rows.append({"contrast": contrast, "layer": layer, "auc": roc_auc_score(y, score)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- kNN


def feature_space(table: pd.DataFrame) -> np.ndarray:
    X = table[G.FEATURES].to_numpy(float)
    X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
    return StandardScaler().fit_transform(X)


def knn_neighbours(X: np.ndarray, k: int = 10) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X)
    return nn.kneighbors(X, return_distance=False)[:, 1:]


def knn_report(states: np.ndarray, table: pd.DataFrame, layers=(0, 4, 8, 12), k: int = 10) -> pd.DataFrame:
    """Compare representation spaces with the TIME feature space (Phase I) by kNN.

    * failure purity: share of neighbours of a failing window that also fail (MASE > 1.1),
      against the base rate
    * same-dataset share: how much the neighbourhood is just "same dataset"
    * kNN regression (Spearman) of log MASE, flat ratio and forecast slope from neighbours,
      restricted to neighbours from *other* datasets so dataset identity cannot leak
    """
    from scipy.stats import spearmanr

    t = probe_frame(table)
    fail = (t["MASE"] > 1.1).to_numpy()
    ds = t["dataset"].to_numpy()
    spaces = {"features": feature_space(t)}
    for layer in layers:
        spaces[f"layer {layer}"] = StandardScaler().fit_transform(states[:, layer].astype(np.float32))
    rows = []
    for name, X in spaces.items():
        nb = knn_neighbours(X, k=50)
        purity = fail[nb[:, :k]][fail].mean()
        same_ds = (ds[nb[:, :k]] == ds[:, None]).mean()
        row = {"space": name, "fail_purity": purity, "fail_base_rate": fail.mean(), "same_dataset": same_ds}
        for target in ["log_MASE", "log_flat_ratio", "fc_slope"]:
            y = _clip(t[target].to_numpy(float))
            pred = np.full(len(y), np.nan)
            for i in range(len(y)):
                other = nb[i][ds[nb[i]] != ds[i]][:k]
                if len(other):
                    pred[i] = np.nanmean(y[other])
            ok = np.isfinite(pred) & np.isfinite(y)
            row[f"rho_{target}"] = spearmanr(pred[ok], y[ok]).statistic
        rows.append(row)
    return pd.DataFrame(rows)
