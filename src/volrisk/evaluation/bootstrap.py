"""Stationary-bootstrap percentile CI for the QLIKE ratio vs the reference model (SPEC §8, holdout headline).

The two loss series are resampled jointly (same block indices) with ``arch.bootstrap.StationaryBootstrap``,
expected block length ``max(n_max, ⌈T^{1/3}⌉)``; the statistic is ``mean(L_model) / mean(L_ref)``.
"""

from __future__ import annotations

import numpy as np
from arch.bootstrap import StationaryBootstrap
from numpy.typing import ArrayLike

from volrisk import config as C
from volrisk.evaluation.mcs import block_size


def _ratio(lm: np.ndarray, lr: np.ndarray) -> np.ndarray:
    return np.array([lm.mean() / lr.mean()])


def qlike_ratio_ci(
    loss_model: ArrayLike,
    loss_ref: ArrayLike,
    n_max: int,
    reps: int = 5000,
    seed: int | None = None,
    level: float = 0.90,
) -> tuple[float, float, float]:
    """Return ``(ratio, lo, hi)``: the point ratio and its ``level`` percentile interval.

    Losses must be paired on common dates. ``seed`` defaults to the project seed.
    """
    lm = np.asarray(loss_model, dtype=float)
    lr = np.asarray(loss_ref, dtype=float)
    if lm.ndim != 1 or lm.shape != lr.shape:
        raise ValueError(f"loss series must be 1-d and of equal length: {lm.shape} vs {lr.shape}")
    if not (np.all(np.isfinite(lm)) and np.all(np.isfinite(lr))):
        raise ValueError("loss series contain non-finite values; align on common dates first")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    if lr.mean() <= 0:
        raise ValueError("reference mean loss must be positive")
    T = lm.size
    bs = StationaryBootstrap(block_size(T, n_max), lm, lr, seed=C.seed() if seed is None else seed)
    draws = bs.apply(_ratio, reps).ravel()
    lo, hi = np.quantile(draws, [(1 - level) / 2, (1 + level) / 2])
    return float(_ratio(lm, lr)[0]), float(lo), float(hi)
