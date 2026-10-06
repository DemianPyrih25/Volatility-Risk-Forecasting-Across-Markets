"""Diebold–Mariano test with the Harvey–Leybourne–Newbold small-sample correction (SPEC §8).

Long-run variance of ``d_t = L_A,t − L_B,t``:

- ``n_max = 1``: Newey–West (Bartlett) HAC with ``maxlags = ⌊4(T/100)^{2/9}⌋``.
- ``n_max > 1`` (overlapping windows): the original DM/HLN estimator — uniform (truncated) kernel with
  ``n_max − 1`` lags, the estimator the HLN factor was derived for. Bartlett with
  ``max(n_max − 1, ⌊4(T/100)^{2/9}⌋)`` lags is oversized on overlapping 1w/1m losses (≈10–12% at a nominal
  5%, vs ≈6% for the uniform kernel; see docs/DEVIATIONS.md). If the uniform estimate is not positive the
  Bartlett estimate with ``max(n_max − 1, ⌊4(T/100)^{2/9}⌋)`` lags is used (``kernel='bartlett'``).
- An explicit ``maxlags`` (robustness: ``2·n_max``) always uses Bartlett.

The t-ratio is scaled by ``√((T + 1 − 2n_max + n_max(n_max − 1)/T)/T)`` and compared with ``t_{T−1}``
(two-sided). A negative ``mean_diff`` means model A has the lower loss.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import ArrayLike
from scipy import stats


def default_maxlags(T: int, n_max: int) -> int:
    """Bartlett lag count of the default (no explicit ``maxlags``) path.

    SPEC §8: ``n_max = 1`` uses Bartlett with ``⌊4(T/100)^{2/9}⌋`` lags (the ``max`` below is then that value);
    ``n_max > 1`` uses the uniform kernel with ``n_max − 1`` lags, and this function only sets the lag count of
    the Bartlett fallback when that estimate is not positive: ``max(n_max − 1, ⌊4(T/100)^{2/9}⌋)``, so the
    fallback never spans fewer lags than the window overlap (docs/DEVIATIONS.md, DM-HLN kernel).
    """
    return max(n_max - 1, math.floor(4 * (T / 100) ** (2 / 9)))


def hln_factor(T: int, n_max: int) -> float:
    """HLN (1997) correction ``√((T + 1 − 2h + h(h − 1)/T)/T)`` with ``h = n_max``."""
    return math.sqrt((T + 1 - 2 * n_max + n_max * (n_max - 1) / T) / T)


def dm_test(loss_a: ArrayLike, loss_b: ArrayLike, n_max: int, maxlags: int | None = None) -> dict:
    """DM-HLN test of equal predictive accuracy on paired losses (same dates, same order).

    Returns ``dict(mean_diff, dm, dm_hln, pvalue, T, maxlags, kernel)``.
    """
    a = np.asarray(loss_a, dtype=float)
    b = np.asarray(loss_b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape:
        raise ValueError(f"loss series must be 1-d and of equal length: {a.shape} vs {b.shape}")
    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("loss series contain non-finite values; align on common dates first")
    d = a - b
    T = d.size
    if T < 3:
        raise ValueError(f"DM test needs at least 3 observations, got {T}")
    mean_diff = float(d.mean())
    se = math.nan
    if maxlags is None and n_max > 1:
        L, kernel = n_max - 1, "uniform"
        se = _uniform_se(d, L)
    if not (np.isfinite(se) and se > 0):
        L = default_maxlags(T, n_max) if maxlags is None else int(maxlags)
        kernel = "bartlett"
        res = sm.OLS(d, np.ones((T, 1))).fit(cov_type="HAC", cov_kwds={"maxlags": L, "kernel": "bartlett"})
        se = float(res.bse[0])
    if np.isfinite(se) and se > 0:
        dm = mean_diff / se
    else:  # constant differential (e.g. identical forecasts)
        dm = 0.0 if mean_diff == 0 else math.copysign(math.inf, mean_diff)
    dm_hln = dm * hln_factor(T, n_max)
    pvalue = float(2 * stats.t.sf(abs(dm_hln), df=T - 1))
    return {
        "mean_diff": mean_diff, "dm": dm, "dm_hln": dm_hln, "pvalue": pvalue, "T": T, "maxlags": L, "kernel": kernel,
    }


def _uniform_se(d: np.ndarray, L: int) -> float:
    """Standard error of mean(d) from the truncated (uniform-kernel) long-run variance with L lags."""
    T = d.size
    u = d - d.mean()
    lrv = u @ u / T + 2 * sum(u[k:] @ u[:-k] / T for k in range(1, L + 1))
    return math.sqrt(lrv / T) if lrv > 0 else math.nan


def dm_vs_ref(wide: pd.DataFrame, ref: str, n_max: int, maxlags: int | None = None) -> pd.DataFrame:
    """DM-HLN of every column of a common-dates loss frame against column ``ref`` (``d = L_model − L_ref``).

    Returns one row per model: ``model, ref, mean_diff, dm, dm_hln, pvalue, T, maxlags, kernel``.
    """
    if ref not in wide.columns:
        raise KeyError(f"reference model {ref!r} not in {list(wide.columns)}")
    rows = [
        {"model": m, "ref": ref, **dm_test(wide[m], wide[ref], n_max, maxlags)}
        for m in wide.columns
        if m != ref
    ]
    cols = ["model", "ref", "mean_diff", "dm", "dm_hln", "pvalue", "T", "maxlags", "kernel"]
    return pd.DataFrame(rows, columns=cols)
