"""Rebuild forecast windows exactly like the TIME runner (TIME/experiments/tirex2_model.py).

Window w of a series uses context = series[: len - test_length + w * H] and truth = next H steps,
which matches gluonts ``split(offset=-test_length).generate_instances(distance=H)``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache, lru_cache

import datasets
import numpy as np
import pandas as pd

from team_blue.data_cube import CACHE_DIR as CUBE_DIR
from team_blue.data_cube import DATA_DIR, RESULTS_DIR, load_dataset_config

KEYS = ["dataset", "freq", "term", "series_idx", "variate_idx", "window_idx"]


@dataclass
class Window:
    context: np.ndarray
    truth: np.ndarray
    stored: (
        np.ndarray
    )  # Phase I forecast [9, H]; npz layout is (series, window, quantile, variate, H)
    seasonality: int
    mase_scale: float  # MAE / MASE of this window (seasonal-naive in-sample MAE)


def load_window_cube(term: str = "short") -> pd.DataFrame:
    wc = pd.read_parquet(CUBE_DIR / "window_cube.parquet")
    wc = wc[wc["term"] == term].copy()
    for k in ["dataset", "freq", "term"]:
        wc[k] = wc[k].astype(str)
    return wc.reset_index(drop=True)


@cache
def _hf(name: str):
    return datasets.load_from_disk(str(DATA_DIR / name)).with_format("numpy")


@cache
def _run(dataset: str, freq: str, term: str):
    run = RESULTS_DIR / "TiRex2" / dataset / freq / term
    config = json.loads((run / "config.json").read_text())
    preds = np.load(run / "predictions.npz")["predictions_quantiles"].astype(np.float32)
    with np.load(run / "metrics.npz") as m:
        scale = m["MAE"] / m["MASE"]
    return config, preds * config.get("prediction_scale_factor", 1.0), scale


@lru_cache(maxsize=4096)
def _series(name: str, series_idx: int, variate_idx: int) -> np.ndarray:
    target = np.asarray(_hf(name)[int(series_idx)]["target"], dtype=np.float32)
    return np.atleast_2d(target)[variate_idx]


def get_window(row) -> Window:
    name = f"{row.dataset}/{row.freq}"
    config, preds, scale = _run(row.dataset, row.freq, row.term)
    H = config["prediction_length"]
    series = _series(name, int(row.series_idx), int(row.variate_idx))
    start = (
        len(series)
        - load_dataset_config()[name]["test_length"]
        + int(row.window_idx) * H
    )
    return Window(
        context=series[:start],
        truth=series[start : start + H],
        stored=preds[
            int(row.series_idx), int(row.window_idx), :, int(row.variate_idx), :
        ],
        seasonality=int(config["seasonality"]),
        mase_scale=float(
            scale[int(row.series_idx), int(row.window_idx), int(row.variate_idx)]
        ),
    )


def prediction_length(dataset: str, freq: str, term: str) -> int:
    return _run(dataset, freq, term)[0]["prediction_length"]
