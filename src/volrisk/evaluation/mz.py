"""Mincer–Zarnowitz and forecast-encompassing regressions (SPEC §8).

All regressions are OLS with a Newey–West (Bartlett) HAC covariance. Inference is asymptotic: Wald
statistics vs χ², t-ratios vs N(0, 1). The lag comes from ``hac_lag``:

- full sample (overlapping 1m windows): ``hac_lag(n_max) = 2·n_max`` (SPEC §8);
- ``non_overlapping`` subsample: ``hac_lag(n_max, overlapping=False) = 0`` (White). Origins ``n_max`` apart
  leave no overlap, so forecast errors are serially uncorrelated under H0. A lag of ``2·n_max`` on the
  ~60–160 subsample rows pushes the Bartlett variance towards zero; the levels MZ then rejected a true H0 in
  74% of runs (T = 57, lag 60). ``lag >= T`` raises and ``lag > T/4`` warns.

Size under H0 (slow tests: overlapping 30-day sums of heavy-tailed daily variance, ``F = E[y | x_t]``,
T = 1700, nominal 5%): the encompassing t-test is close to nominal (≈ 6%), ``mz_log`` is mildly oversized
(≈ 11%), and the **levels** MZ Wald rejects ≈ 35% (≈ 23% on the non-overlapping subsample with lag 0).
With a skewed, persistent ``F`` the HAC variance of the levels regression is unreliable at this effective
sample size, and neither an F reference nor a small-sample correction fixes it. Report the levels
``(a, b)`` as descriptive and read MZ inference from ``mz_log``.

- ``mincer_zarnowitz``: ``y = a + b F`` (levels), Wald H0 ``a = 0, b = 1``.
- ``mz_log``: ``log y = a + b log F``, Wald H0 ``b = 1``.
- ``encompassing``: ``log y = a + b log IVvar + c log F + e``, H0 ``c = 0`` (F adds information beyond IV).
- ``non_overlapping``: every ``step``-th origin (30 calendar days for crypto, 22 sessions otherwise,
  i.e. ``config.n_max('1m', asset)``) for the non-overlapping robustness subsample.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from numpy.typing import ArrayLike
from scipy import stats


def _arrays(*xs: ArrayLike, positive: bool = False) -> list[np.ndarray]:
    out = [np.asarray(x, dtype=float) for x in xs]
    n = out[0].shape
    for x in out:
        if x.ndim != 1 or x.shape != n:
            raise ValueError("inputs must be 1-d and of equal length")
        if not np.all(np.isfinite(x)):
            raise ValueError("inputs contain non-finite values")
        if positive and not np.all(x > 0):
            raise ValueError("log regressions need strictly positive inputs")
    return out


def hac_lag(n_max: int, overlapping: bool = True) -> int:
    """HAC lag: ``2·n_max`` on the full overlapping sample, ``0`` on the ``non_overlapping`` subsample."""
    if n_max < 1:
        raise ValueError("n_max must be >= 1")
    return 2 * int(n_max) if overlapping else 0


def _hac_ols(yv: np.ndarray, regressors: list[np.ndarray], lag: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.column_stack([np.ones_like(yv), *regressors])
    T = len(yv)
    if T <= X.shape[1]:
        raise ValueError(f"not enough observations ({T}) for {X.shape[1]} coefficients")
    lag = int(lag)
    if not 0 <= lag < T:
        raise ValueError(
            f"HAC lag must be in [0, T) = [0, {T}), got {lag}; on the non-overlapping subsample use "
            "hac_lag(n_max, overlapping=False)"
        )
    if lag > T / 4:
        warnings.warn(
            f"HAC lag {lag} > T/4 = {T / 4:g}: the Bartlett variance is biased towards zero and the test "
            "over-rejects",
            RuntimeWarning,
            stacklevel=3,
        )
    res = sm.OLS(yv, X).fit(cov_type="HAC", cov_kwds={"maxlags": lag, "kernel": "bartlett"})
    return np.asarray(res.params), np.asarray(res.cov_params())


def _wald(params: np.ndarray, cov: np.ndarray, idx: list[int], h0: list[float]) -> tuple[float, float]:
    r = params[idx] - np.asarray(h0)
    V = cov[np.ix_(idx, idx)]
    w = float(r @ np.linalg.solve(V, r))
    return w, float(stats.chi2.sf(w, df=len(idx)))


def mincer_zarnowitz(y: ArrayLike, F: ArrayLike, lag: int) -> dict:
    """``y = a + b F``; returns ``dict(a, b, se_a, se_b, wald, p_wald)`` with Wald H0 ``a = 0, b = 1``.

    ``p_wald`` is badly oversized on 1m variance targets (module docstring), so report it as descriptive.
    """
    yv, Fv = _arrays(y, F)
    params, cov = _hac_ols(yv, [Fv], lag)
    wald, p = _wald(params, cov, [0, 1], [0.0, 1.0])
    se = np.sqrt(np.diag(cov))
    return {"a": float(params[0]), "b": float(params[1]), "se_a": float(se[0]), "se_b": float(se[1]),
            "wald": wald, "p_wald": p}


def mz_log(y: ArrayLike, F: ArrayLike, lag: int) -> dict:
    """``log y = a + b log F``; returns ``dict(a, b, se_a, se_b, wald, p_wald)`` with Wald H0 ``b = 1``."""
    yv, Fv = _arrays(y, F, positive=True)
    params, cov = _hac_ols(np.log(yv), [np.log(Fv)], lag)
    wald, p = _wald(params, cov, [1], [1.0])
    se = np.sqrt(np.diag(cov))
    return {"a": float(params[0]), "b": float(params[1]), "se_a": float(se[0]), "se_b": float(se[1]),
            "wald": wald, "p_wald": p}


def encompassing(y: ArrayLike, iv_var: ArrayLike, F: ArrayLike, lag: int) -> dict:
    """``log y = a + b log IVvar + c log F``; returns ``dict(b, c, se_c, t_c, p_c)`` (H0 ``c = 0``, two-sided)."""
    yv, iv, Fv = _arrays(y, iv_var, F, positive=True)
    params, cov = _hac_ols(np.log(yv), [np.log(iv), np.log(Fv)], lag)
    se_c = float(np.sqrt(cov[2, 2]))
    t_c = float(params[2] / se_c)
    p_c = float(2 * stats.norm.sf(abs(t_c)))
    return {"b": float(params[1]), "c": float(params[2]), "se_c": se_c, "t_c": t_c, "p_c": p_c}


def non_overlapping(df: pd.DataFrame, step: int, col: str = "origin") -> pd.DataFrame:
    """Every ``step``-th row by ``col`` (starting with the first), for one (asset, horizon, model) frame.

    Run the regressions on the result with ``lag = hac_lag(n_max, overlapping=False)`` (= 0), not ``2·n_max``.
    """
    if step < 1:
        raise ValueError("step must be >= 1")
    return df.sort_values(col, kind="stable").iloc[::step].reset_index(drop=True)
