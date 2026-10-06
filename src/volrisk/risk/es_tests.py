"""Acerbi–Székely Z2 expected-shortfall backtest with Monte Carlo p-values (SPEC §9).

``Z2 = Σ r_t I_t / (T·α·ES_t) + 1`` with ``I_t = 1{r_t < -VaR97.5_t}``; ``Z2 < 0`` means ES is understated.
The null distribution is simulated from each model's own predictive distribution: ``r*_t = σ̂_t z*`` with
``z*`` drawn from that day's FHS pool, HS draws from that day's window, Normal models draw ``N(0, σ̂_t²)``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from volrisk import config as C
from volrisk.risk import var_es

Sampler = Callable[[np.random.Generator, int], np.ndarray]  # (rng, reps) -> (reps, T) simulated returns
_CHUNK = 1000  # simulated paths per block (memory cap)


def _inputs(r, var975, es975) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r_a, v, e = (np.asarray(x, dtype=float) for x in (r, var975, es975))
    if not (r_a.ndim == 1 and r_a.shape == v.shape == e.shape and r_a.size):
        raise ValueError("r, var975 and es975 must be non-empty 1-d arrays of equal length")
    if not (np.isfinite(r_a).all() and np.isfinite(v).all() and np.isfinite(e).all()):
        raise ValueError("r, var975 and es975 must be finite")
    if (e <= 0).any():
        raise ValueError("ES must be positive")
    return r_a, v, e


def _z2(R: np.ndarray, v: np.ndarray, e: np.ndarray, alpha: float) -> np.ndarray:
    """Z2 along the last axis of R (works for one path or a (reps, T) block)."""
    return np.sum(np.where(R < -v, R / e, 0.0), axis=-1) / (R.shape[-1] * alpha) + 1.0


def acerbi_szekely_z2(r, var975, es975, alpha: float = 0.025) -> float:
    """Acerbi–Székely Z2 statistic (expected value 0 under a correct model, negative if ES is too small)."""
    r_a, v, e = _inputs(r, var975, es975)
    return float(_z2(r_a, v, e, alpha))


def z2_pvalue(
    r, var975, es975, sampler: Sampler, reps: int = 10000, seed: int | None = None, alpha: float = 0.025
) -> float:
    """One-sided Monte Carlo p-value ``share(Z2* <= Z2_obs)`` with Z2* from ``sampler`` paths (seeded)."""
    r_a, v, e = _inputs(r, var975, es975)
    z_obs = float(_z2(r_a, v, e, alpha))
    rng = np.random.default_rng(C.seed() if seed is None else seed)
    sims = []
    for start in range(0, reps, _CHUNK):
        n = min(_CHUNK, reps - start)
        R = np.asarray(sampler(rng, n), dtype=float)
        if R.shape != (n, r_a.size):
            raise ValueError(f"sampler returned shape {R.shape}, expected {(n, r_a.size)}")
        sims.append(_z2(R, v, e, alpha))
    return float(np.mean(np.concatenate(sims) <= z_obs))


# --------------------------------------------------------------------------------------------- samplers
def normal_sampler(sigma) -> Sampler:
    """r*_t ~ N(0, σ_t²) independently over t."""
    s = np.asarray(sigma, dtype=float)

    def draw(rng: np.random.Generator, reps: int) -> np.ndarray:
        return rng.standard_normal((reps, s.size)) * s

    return draw


def hs_sampler(windows: Sequence[np.ndarray]) -> Sampler:
    """r*_t drawn uniformly (with replacement) from that day's window ``windows[t]``."""
    sizes = np.array([len(w) for w in windows], dtype=np.int64)
    if sizes.size == 0 or (sizes == 0).any():
        raise ValueError("every day needs a non-empty window")
    flat = np.concatenate([np.asarray(w, dtype=float) for w in windows])
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])

    def draw(rng: np.random.Generator, reps: int) -> np.ndarray:
        return flat[offsets + rng.integers(0, sizes, size=(reps, sizes.size))]

    return draw


def fhs_sampler(sigma, pools: Sequence[np.ndarray]) -> Sampler:
    """r*_t = σ_t z*, with z* drawn uniformly from that day's FHS pool ``pools[t]``."""
    s = np.asarray(sigma, dtype=float)
    if len(pools) != s.size:
        raise ValueError("sigma and pools must have the same length")
    base = hs_sampler(pools)

    def draw(rng: np.random.Generator, reps: int) -> np.ndarray:
        return base(rng, reps) * s

    return draw


def sampler_for(
    model: str,
    rows: pd.DataFrame,
    daily: pd.DataFrame,
    forecasts_1d: pd.DataFrame | None = None,
    *,
    pool: int | None = None,
    window: int | None = None,
) -> Sampler:
    """Null sampler for the rows of ONE (asset, model) of a risk frame (RISK_COLUMNS, sorted by date).

    Rebuilds that day's FHS pool / HS window from ``daily`` (+ ``forecasts_1d`` for FHS models) with the
    same rules as ``var_es.risk_frame``; Normal models only need the rows' ``sigma``.
    """
    cfg = C.load()["risk"]
    pool = int(cfg["fhs_pool"]) if pool is None else pool
    window = int(cfg["hs_window"]) if window is None else window
    if model in var_es.NORMAL_MODELS:
        return normal_sampler(rows["sigma"].to_numpy())
    asset = rows["asset"].iloc[0] if "asset" in rows.columns else None
    r = var_es.session_returns(daily, asset)
    if model == "HS-250":
        return hs_sampler(var_es.hs_windows(r, rows["date"], window))
    if forecasts_1d is None:
        raise ValueError(f"{model} needs the 1d forecasts to rebuild its FHS pools")
    member = rows["har_member"].iloc[0] if "har_member" in rows.columns else None
    base = var_es.fhs_base_model(model, member)
    s2 = var_es.sigma2_by_target(daily, forecasts_1d, base, asset)
    return fhs_sampler(rows["sigma"].to_numpy(), var_es.fhs_pools(r, s2, rows["date"], pool))
