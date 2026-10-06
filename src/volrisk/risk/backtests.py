"""VaR backtests: Kupiec, Christoffersen, Engle–Manganelli DQ, UC power, Basel traffic light (SPEC §9).

Hits ``I_t = 1{L_t > VaR_t}`` with loss ``L = -r_cc`` and VaR a positive number. Statistics that are undefined
(Christoffersen with zero breaches) are returned as NaN, the pandas NA for floats.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import xlogy
from scipy.stats import binom, binomtest, chi2

from volrisk import config as C
from volrisk.risk.var_es import ALPHA_99, ALPHA_975

GREEN_CDF = 0.95
RED_CDF = 0.9999
BASEL_PLUS = {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}  # N=250 @1%; <=4 -> 0, >=10 -> 1.0
LEVELS = (("99", ALPHA_99, "var99"), ("97.5", ALPHA_975, "var975"))
_TIE = 1e-9  # MC p-values: simulated statistics within this tolerance of the observed one count as ties
_CHUNK = 2000  # Monte Carlo paths per block (memory cap)


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(C.seed() if seed is None else seed)


def _as_hits(h) -> np.ndarray:
    a = np.asarray(h)
    if a.ndim != 1 or a.size == 0:
        raise ValueError("hits must be a non-empty 1-d sequence")
    if not np.isin(a, (0, 1)).all():
        raise ValueError("hits must be 0/1")
    return a.astype(bool)


def hits(loss, var) -> np.ndarray:
    """``I_t = 1{L_t > VaR_t}`` as an int array; inputs must be finite and of equal length."""
    loss_a, var_a = np.asarray(loss, dtype=float), np.asarray(var, dtype=float)
    if loss_a.shape != var_a.shape:
        raise ValueError("loss and var must have the same shape")
    if not (np.isfinite(loss_a).all() and np.isfinite(var_a).all()):
        raise ValueError("loss and var must be finite")
    return (loss_a > var_a).astype(np.int64)


# --------------------------------------------------------------------------------------------- Kupiec
def _ll_bernoulli(n_hit, n, prob):
    """Bernoulli log-likelihood with 0·ln0 = 0 (vectorised)."""
    return xlogy(n - n_hit, 1.0 - prob) + xlogy(n_hit, prob)


def kupiec(hits, p: float) -> dict:
    """Kupiec unconditional coverage: LR_uc ~ chi2(1) and the exact two-sided binomial p-value."""
    h = _as_hits(hits)
    T, x = h.size, int(h.sum())
    lr = 2.0 * (_ll_bernoulli(x, T, x / T) - _ll_bernoulli(x, T, p))
    lr = max(float(lr), 0.0)
    return {
        "x": x,
        "T": T,
        "rate": x / T,
        "lr_uc": lr,
        "p_chi2": float(chi2.sf(lr, 1)),
        "p_binom": float(binomtest(x, T, p, alternative="two-sided").pvalue),
    }


# --------------------------------------------------------------------------------------------- Christoffersen
def _transitions(H: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Counts n00, n01, n10, n11 of the T-1 transitions along the last axis of a bool array."""
    a, b = H[..., :-1], H[..., 1:]
    n11 = (a & b).sum(-1)
    n10 = (a & ~b).sum(-1)
    n01 = (~a & b).sum(-1)
    n00 = a.shape[-1] - n11 - n10 - n01
    return n00, n01, n10, n11


def _lr_ind_cc(n00, n01, n10, n11, p: float) -> tuple[np.ndarray, np.ndarray]:
    """LR_ind and LR_cc = LR_uc + LR_ind, both on the T-1 transition sample (0·ln0 = 0)."""
    n00, n01, n10, n11 = (np.asarray(v, dtype=float) for v in (n00, n01, n10, n11))
    n0, n1 = n00 + n01, n10 + n11  # transitions out of state 0 / state 1
    pi01 = np.divide(n01, n0, out=np.zeros_like(n0), where=n0 > 0)
    pi11 = np.divide(n11, n1, out=np.zeros_like(n1), where=n1 > 0)
    n_hit, n = n01 + n11, n0 + n1
    ll_markov = _ll_bernoulli(n01, n0, pi01) + _ll_bernoulli(n11, n1, pi11)
    ll_iid = _ll_bernoulli(n_hit, n, n_hit / n)
    lr_ind = np.maximum(2.0 * (ll_markov - ll_iid), 0.0)
    lr_uc = np.maximum(2.0 * (ll_iid - _ll_bernoulli(n_hit, n, p)), 0.0)
    return lr_ind, lr_uc + lr_ind


def christoffersen(hits, p: float, mc_reps: int = 10000, seed: int | None = None) -> dict:
    """Christoffersen independence / conditional coverage with Monte Carlo p-values.

    The MC null is ``mc_reps`` iid Bernoulli(p) paths of length T (seeded; default ``config.seed``);
    ``p = share(LR* >= LR_obs)``. With zero breaches the statistics and p-values are NaN (NA).
    """
    h = _as_hits(hits)
    if h.size < 2:
        raise ValueError("Christoffersen needs at least two observations")
    n00, n01, n10, n11 = (int(v) for v in _transitions(h))
    out = {"n00": n00, "n01": n01, "n10": n10, "n11": n11}
    if not h.any():
        return out | {"lr_ind": np.nan, "lr_cc": np.nan, "p_ind_mc": np.nan, "p_cc_mc": np.nan}
    lr_ind, lr_cc = (float(v) for v in _lr_ind_cc(n00, n01, n10, n11, p))
    rng = _rng(seed)
    sim_ind, sim_cc = [], []
    for start in range(0, mc_reps, _CHUNK):
        H = rng.random((min(_CHUNK, mc_reps - start), h.size)) < p
        si, sc = _lr_ind_cc(*_transitions(H), p)
        sim_ind.append(si)
        sim_cc.append(sc)
    si, sc = np.concatenate(sim_ind), np.concatenate(sim_cc)
    return out | {
        "lr_ind": lr_ind,
        "lr_cc": lr_cc,
        "p_ind_mc": float(np.mean(si >= lr_ind - _TIE)),
        "p_cc_mc": float(np.mean(sc >= lr_cc - _TIE)),
    }


# --------------------------------------------------------------------------------------------- DQ
def dq_test(hits, p: float, var, lags: int = 4) -> dict:
    """Engle–Manganelli dynamic quantile test.

    ``Hit_t = I_t - p`` regressed on ``X = [1, Hit_{t-1..t-lags}, VaR_t]`` for t > lags;
    ``DQ = Hit'X(X'X)^-1 X'Hit / (p(1-p)) ~ chi2(lags+2)`` (chi2(6) for the SPEC's 4 lags).

    X loses rank whenever a lag column is constant: zero breaches, or all breaches in the last ``lags``
    observations (a lag column only sees ``I_1..I_{T-k}``), and also a constant VaR. The projection then
    uses least squares, which is the same as a generalized inverse. The degrees of freedom stay at ``lags+2``
    in every case. Using rank(X) would make the p-value jump for a nearly unchanged statistic, depending
    only on where a single breach falls.
    """
    h = _as_hits(hits).astype(float)
    v = np.asarray(var, dtype=float)
    if v.shape != h.shape:
        raise ValueError("hits and var must have the same length")
    T = h.size
    if T <= 2 * (lags + 2):
        raise ValueError("too few observations for the DQ test")
    hit = h - p
    y = hit[lags:]
    X = np.column_stack([np.ones(T - lags)] + [hit[lags - k : T - k] for k in range(1, lags + 1)] + [v[lags:]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    dq = float(y @ (X @ beta)) / (p * (1.0 - p))
    return {"dq": dq, "p": float(chi2.sf(dq, lags + 2))}


# --------------------------------------------------------------------------------------------- power
def uc_power(
    T: int, p: float, reps: int = 10000, seed: int | None = None, *, rate: float | None = None, level: float = 0.05
) -> float:
    """Simulated rejection rate of the exact two-sided binomial UC test at ``level`` for a sample of T
    observations when the true breach rate is ``rate`` (default 2p — power; ``rate=p`` gives the size)."""
    rate = 2.0 * p if rate is None else rate
    x = _rng(seed).binomial(T, rate, size=reps)
    vals, inv = np.unique(x, return_inverse=True)
    reject = np.array([binomtest(int(v), T, p).pvalue < level for v in vals])
    return float(reject[inv].mean())


# --------------------------------------------------------------------------------------------- traffic light
def traffic_light_zones(N: int, p: float) -> tuple[int, int]:
    """(green_max, yellow_max): green if P(X <= x; N, p) < 0.95, red if >= 0.9999, yellow otherwise."""
    cdf = binom.cdf(np.arange(N + 1), N, p)
    return int(np.sum(cdf < GREEN_CDF)) - 1, int(np.sum(cdf < RED_CDF)) - 1


def zone(x: int, N: int, p: float) -> str:
    green_max, yellow_max = traffic_light_zones(N, p)
    return "green" if x <= green_max else "yellow" if x <= yellow_max else "red"


def plus_factor(x: int) -> float:
    """Basel plus factor for x exceptions in 250 observations at 99%."""
    if x <= 4:
        return 0.0
    return BASEL_PLUS.get(int(x), 1.0)


def rolling_traffic_light(hits, window: int = 250, p: float = 0.01) -> pd.DataFrame:
    """Exceptions, zone and plus factor over rolling ``window``-observation windows (ending at each date).

    ``hits`` is a 0/1 series indexed by date (or an array). Only complete windows are returned. The plus
    factor is defined only for the Basel case (window 250, p = 1%) and is NaN otherwise.
    """
    s = hits if isinstance(hits, pd.Series) else pd.Series(np.asarray(hits))
    _as_hits(s.to_numpy())
    exc = s.astype(float).rolling(window, min_periods=window).sum().dropna().astype(np.int64)
    green_max, yellow_max = traffic_light_zones(window, p)
    zones = np.select([exc <= green_max, exc <= yellow_max], ["green", "yellow"], "red")
    basel = window == 250 and np.isclose(p, 0.01)
    pf = exc.map(plus_factor).to_numpy(dtype=float) if basel else np.full(len(exc), np.nan)
    return pd.DataFrame({"exceptions": exc.to_numpy(), "zone": zones, "plus_factor": pf}, index=exc.index)


# --------------------------------------------------------------------------------------------- summary
def var_backtests(
    r, var, p: float, mc_reps: int | None = None, seed: int | None = None, window: int | None = None
) -> dict:
    """All VaR backtests of one series (returns ``r``, positive ``var``) at level ``p``."""
    cfg = C.load()["risk"]
    mc_reps = int(cfg["mc_reps"]) if mc_reps is None else mc_reps
    window = int(cfg["traffic_window"]) if window is None else window
    r_a = np.asarray(r, dtype=float)
    h = hits(-r_a, var)
    k = kupiec(h, p)
    c = christoffersen(h, p, mc_reps=mc_reps, seed=seed)
    dq = dq_test(h, p, var)
    out = {
        **k,
        "lr_ind": c["lr_ind"],
        "p_ind_mc": c["p_ind_mc"],
        "lr_cc": c["lr_cc"],
        "p_cc_mc": c["p_cc_mc"],
        "dq": dq["dq"],
        "p_dq": dq["p"],
        "power_uc": uc_power(k["T"], p, reps=mc_reps, seed=seed),
        "zone_full": zone(k["x"], k["T"], p),
    }
    if h.size >= window:
        roll = rolling_traffic_light(h, window, p)
        out |= {"green_share": float(np.mean(roll["zone"] == "green")), "zone_last": roll["zone"].iloc[-1]}
    else:
        out |= {"green_share": np.nan, "zone_last": None}
    return out


def backtest_table(
    risk: pd.DataFrame, mc_reps: int | None = None, seed: int | None = None, window: int | None = None
) -> pd.DataFrame:
    """VaR backtests at 99% and 97.5% for every (asset, model) of a risk frame (RISK_COLUMNS).

    The caller selects the backtest window (e.g. ``split == 'dev'`` and dates >= 2021-01-01) beforehand.
    """
    rows = []
    for (asset, model), g in risk.groupby(["asset", "model"], sort=False):
        g = g.sort_values("date")
        for level, p, col in LEVELS:
            res = var_backtests(g["r_cc"], g[col].to_numpy(), p, mc_reps=mc_reps, seed=seed, window=window)
            rows.append(
                {
                    "asset": asset,
                    "model": model,
                    "level": level,
                    "start": g["date"].iloc[0],
                    "end": g["date"].iloc[-1],
                    **res,
                }
            )
    return pd.DataFrame(rows)
