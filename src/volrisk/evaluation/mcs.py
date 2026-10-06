"""Model Confidence Set (Hansen, Lunde & Nason 2011) via ``arch.bootstrap.MCS`` (SPEC §8).

Range statistic (``method='R'``), stationary bootstrap with expected block length
``max(n_max, ⌈T^{1/3}⌉)``. MCS p-values do not depend on ``size`` (it only sets arch's own inclusion flag),
so one run gives both sets: ``in_90 = p ≥ 0.10`` and ``in_75 = p ≥ 0.25``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from arch.bootstrap import MCS

from volrisk import config as C


def block_size(T: int, n_max: int) -> int:
    """SPEC §8 bootstrap block rule ``max(n_max, ⌈T^{1/3}⌉)`` (shared with the holdout ratio CI)."""
    c = math.ceil(T ** (1 / 3))
    # exact integer ceiling of the cube root (float rounding can be off by one at perfect cubes)
    while c > 1 and (c - 1) ** 3 >= T:
        c -= 1
    while c**3 < T:
        c += 1
    return max(int(n_max), c)


def run_mcs(
    losses: pd.DataFrame,
    n_max: int,
    size: float = 0.10,
    reps: int = 5000,
    seed: int | None = None,
) -> pd.DataFrame:
    """MCS over the columns (models) of a common-dates loss frame (rows = dates, no missing values).

    Returns ``model, pvalue, in_90, in_75`` in the column order of ``losses``. ``seed`` defaults to the
    project seed.
    """
    if losses.shape[1] == 0:
        raise ValueError("no models")
    if not np.all(np.isfinite(losses.to_numpy(dtype=float))):
        raise ValueError("losses contain missing/non-finite values; restrict to common dates first")
    models = list(losses.columns)
    if len(models) == 1:  # a single model is trivially the MCS
        p = pd.Series([1.0], index=models)
    else:
        T = len(losses)
        mcs = MCS(
            losses.astype(float),
            size,
            reps=reps,
            block_size=block_size(T, n_max),
            method="R",
            bootstrap="stationary",
            seed=C.seed() if seed is None else seed,
        )
        mcs.compute()
        p = mcs.pvalues["Pvalue"]
    pv = p.reindex(models).astype(float).to_numpy()
    return pd.DataFrame({"model": models, "pvalue": pv, "in_90": pv >= 0.10, "in_75": pv >= 0.25})
