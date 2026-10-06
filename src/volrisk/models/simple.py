"""Naive baselines: random walk on the target and RiskMetrics EWMA (SPEC §7, models 1–2).

- ``RW``:   ``F = n_t · mean(tv over the last n_max sessions incl. t)`` (1d: ``tv_t``).
- ``EWMA``: ``σ²_{t+1} = λσ²_t + (1−λ) r_cc,t²`` with ``λ = 0.94``; ``F = n_t · σ²_{t+1}``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volrisk import config as C
from volrisk.models.base import align_targets, predict_mask_for, to_forecast_frame

EWMA_INIT = 250


def trailing_mean(x: np.ndarray, n: int) -> np.ndarray:
    """Mean of the last ``n`` values including the current one; NaN for the first ``n − 1`` rows."""
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1 :] = np.lib.stride_tricks.sliding_window_view(x, n).mean(axis=1)
    return out


def ewma_variance(r_cc: np.ndarray, lam: float = 0.94, n_init: int = EWMA_INIT) -> np.ndarray:
    """One-step-ahead EWMA variance ``σ²_{t+1|t}`` per row (NaN where not yet available).

    Initialisation: ``σ²`` for the session after the first ``n_init`` finite returns is their sample variance
    (ddof=1); the recursion then starts at the next row, so with no missing returns the first forecast is at
    row ``n_init``: ``σ²_{n_init+1} = λ·var(r_0..r_{n_init−1}) + (1−λ)·r_{n_init}²``. A missing return leaves
    ``σ²`` unchanged. Only data up to the origin enters each value.
    """
    r = np.asarray(r_cc, dtype=float)
    out = np.full(len(r), np.nan)
    finite = np.flatnonzero(np.isfinite(r))
    if len(finite) <= n_init:
        return out
    init_rows = finite[:n_init]
    s2 = float(np.var(r[init_rows], ddof=1))
    for i in range(init_rows[-1] + 1, len(r)):
        if np.isfinite(r[i]):
            s2 = lam * s2 + (1.0 - lam) * r[i] * r[i]
        out[i] = s2
    return out


class RW:
    """Random walk on the per-session target level."""

    name = "RW"

    def forecast(self, daily: pd.DataFrame, targets: pd.DataFrame, asset: str, horizon: str) -> pd.DataFrame:
        a = align_targets(daily, targets)
        level = trailing_mean(daily["tv"].to_numpy(dtype=float), C.n_max(horizon, asset))
        n_t = a["n_t"].to_numpy(dtype=float)
        F = np.where(predict_mask_for(a), n_t * level, np.nan)
        return to_forecast_frame(asset, horizon, self.name, daily["session_date"], a["n_t"], F)


class EWMA:
    """RiskMetrics EWMA on close-to-close returns, scaled flat over the window."""

    name = "EWMA"

    def __init__(self, lam: float | None = None, n_init: int = EWMA_INIT):
        self.lam = float(C.load()["risk"]["ewma_lambda"]) if lam is None else float(lam)
        self.n_init = int(n_init)

    def forecast(self, daily: pd.DataFrame, targets: pd.DataFrame, asset: str, horizon: str) -> pd.DataFrame:
        a = align_targets(daily, targets)
        s2 = ewma_variance(daily["r_cc"].to_numpy(dtype=float), self.lam, self.n_init)
        n_t = a["n_t"].to_numpy(dtype=float)
        F = np.where(predict_mask_for(a), n_t * s2, np.nan)
        return to_forecast_frame(asset, horizon, self.name, daily["session_date"], a["n_t"], F)
