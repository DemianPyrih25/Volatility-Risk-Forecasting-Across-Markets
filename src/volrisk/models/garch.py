"""GARCH(1,1)-t and GJR-GARCH(1,1)-t walk-forward forecasters (SPEC §6, §7 models 3–4).

Model: ``arch_model(r_cc, mean='Zero', vol='GARCH', p=1, o=0|1, q=1, dist='t')`` on percent close-to-close
returns (rows of the gold table with a finite ``r_cc``). With ``W = config.window()`` and
``R = config.refit_every('garch', asset)``:

* refit origins are the return positions ``j = W-1, W-1+R, W-1+2R, …`` (sessions; the same for every horizon);
  each fit uses the ``W`` most recent returns ending at ``j`` inclusive (arch's ``last_obs`` is exclusive);
* origins ``s ∈ [j, j+R)`` keep the parameters of refit ``j``; the variance is filtered daily from the start
  of the estimation window, so ``σ²_{s+k|s}`` (``method='analytic'``) uses returns up to ``s`` only;
* ``F_s = Σ_{k=1..n_s} σ²_{s+k|s}`` — the per-step forecasts are summed, never ``n_s · σ²_{s+1|s}``.

Each block of origins ``[j, e)`` gets its own arch model on ``r[j-W+1 : e]``: arch's analytic multi-step
forecast loops in Python over every row from ``start`` to the end of its series, so a model on the full
sample would cost O(T) per refit. Returns after ``e`` never enter; inside the block later returns only reach
arch's loose variance bounds (``var/1e8``, ``1e7·(1+max r²)``), which never bind for a fitted model, so the
forecasts are bitwise those of a strictly real-time run (asserted in ``tests/test_garch.py``).

One walk-forward per (model, asset, data) computes the per-step path up to ``H = max n_max`` once and every
horizon is a cumulative sum of it (module-level cache), so the 1d/1w/1m forecasts share their fits.

Convergence policy (not specified by the SPEC): a refit uses arch's estimate unchanged whenever the optimizer
converged inside arch's own parameter space — persistence ``≤ 1`` up to ``FEASIBILITY_TOL``, so IGARCH-boundary
MLEs are accepted. Only non-converged or out-of-space fits trigger retries and then a fallback to the previous
refit's parameters (see ``fit_with_retries``); statuses are recorded per refit.
"""

from __future__ import annotations

import hashlib
import logging
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model

from volrisk import config as C
from volrisk.models.base import align_targets, predict_mask_for, to_forecast_frame

log = logging.getLogger(__name__)

MODELS = {"GARCH": 0, "GJR": 1}  # model name -> asymmetry order o
STATUSES = ("ok", "retry", "fallback", "nonconverged", "failed")
_RETRY_OPTIONS = {"maxiter": 1000}
_CACHE_SIZE = 16
# SLSQP meets arch's linear constraints only up to its tolerance: converged fits (flag 0) stop on the IGARCH
# boundary α+γ/2+β = 1 or up to ~4e-6 past it (largest seen in simulations at W = 1000).
FEASIBILITY_TOL = 1e-5
_START_MAX_PERSISTENCE = 0.999


def param_names(o: int) -> list[str]:
    return ["omega", "alpha[1]"] + (["gamma[1]"] if o else []) + ["beta[1]", "nu"]


def make_model(y: np.ndarray, o: int):
    """Zero-mean GARCH(1,o,1) with standardised Student-t errors on percent returns (no rescaling)."""
    return arch_model(
        np.asarray(y, dtype=float), mean="Zero", vol="GARCH", p=1, o=o, q=1, dist="t", rescale=False
    )


def persistence(params: np.ndarray, o: int) -> float:
    """``α + γ/2 + β`` (``γ = 0`` for GARCH)."""
    alpha, beta = params[1], params[2 + o]
    gamma = params[2] if o else 0.0
    return float(alpha + 0.5 * gamma + beta)


def valid_params(params: np.ndarray | None, o: int) -> bool:
    """Inside arch's GARCH parameter space: finite, ``ω > 0``, ``α, β ≥ 0``, ``ν > 2`` (box bounds, which
    SLSQP enforces exactly) and the linear constraints ``α + γ ≥ 0``, ``α + γ/2 + β ≤ 1`` up to
    ``FEASIBILITY_TOL``. The IGARCH boundary is admissible: the analytic multi-step forecast is a plain
    recursion, finite and positive at persistence 1, so converged boundary MLEs are used unchanged.
    """
    if params is None or len(params) != len(param_names(o)) or not np.all(np.isfinite(params)):
        return False
    omega, alpha, beta, nu = params[0], params[1], params[2 + o], params[-1]
    gamma = params[2] if o else 0.0
    checks = (
        omega > 0,
        alpha >= 0,
        beta >= 0,
        nu > 2,
        alpha + gamma >= -FEASIBILITY_TOL,
        persistence(params, o) <= 1 + FEASIBILITY_TOL,
    )
    return bool(all(checks))


@dataclass(frozen=True)
class FitOutcome:
    params: np.ndarray | None  # None only when status == "failed"
    status: str  # one of STATUSES
    attempts: int
    loglik: float  # NaN unless the parameters were estimated at this refit


@dataclass
class GarchPaths:
    """Per-step variance forecasts of one walk-forward, rows aligned with the input series."""

    model: str
    variances: np.ndarray  # (n, H): row s = σ²_{s+1|s}, …, σ²_{s+H|s}; NaN where no forecast was made
    refit_rows: np.ndarray  # row positions of the refit origins
    params: pd.DataFrame  # one row per refit: row, status, attempts, loglik, parameter columns
    seconds: float

    @property
    def counts(self) -> dict[str, int]:
        vc = self.params["status"].value_counts() if len(self.params) else pd.Series(dtype=int)
        return {s: int(vc.get(s, 0)) for s in STATUSES}


# (model, asset, window, refit_every, data fingerprint) -> paths; shared by all horizons and instances
_CACHE: OrderedDict[tuple, GarchPaths] = OrderedDict()


def _fit_once(am, last_obs: int, starting_values: np.ndarray | None, options: dict | None):
    """One arch estimation on observations ``[0, last_obs)`` of the model's series (warnings silenced)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return am.fit(
            first_obs=0,
            last_obs=last_obs,
            disp="off",
            show_warning=False,
            starting_values=starting_values,
            options=options,
        )


def _generic_start(y: np.ndarray, o: int) -> np.ndarray:
    """Plain starting values with persistence 0.95 and the sample variance as long-run level."""
    var = float(np.mean(y**2)) or 1.0
    arch_terms = [0.03, 0.08] if o else [0.07]
    return np.array([0.05 * var, *arch_terms, 0.88, 8.0])


def _start_from(params: np.ndarray, o: int) -> np.ndarray:
    """``params`` as starting values strictly inside arch's constraints.

    arch silently replaces starting values that violate its constraints (exactly, without tolerance) by its
    own, so a previous fit on or just past the boundary gets ``α + γ ≥ 0`` restored and its
    ``α, γ, β`` scaled down to persistence ``_START_MAX_PERSISTENCE``.
    """
    sv = np.asarray(params, dtype=float).copy()
    if o:
        sv[2] = max(sv[2], -sv[1])
    pers = persistence(sv, o)
    if pers > _START_MAX_PERSISTENCE:
        sv[1 : 3 + o] *= _START_MAX_PERSISTENCE / pers
    return sv


def fit_with_retries(am, last_obs: int, o: int, prev_params: np.ndarray | None = None) -> FitOutcome:
    """Estimate on ``[0, last_obs)``; retry from other starting values, else fall back.

    Attempts: arch's own starting values, then the previous refit's parameters (if any), then generic values
    (retries with a larger iteration budget). The first converged attempt whose parameters lie in arch's
    parameter space (``valid_params``: IGARCH boundary included, up to optimizer tolerance) wins, unchanged
    (``ok``/``retry``). Otherwise the previous parameters are kept (``fallback``); at the first refit the
    best valid non-converged attempt is used (``nonconverged``); with nothing valid the block is ``failed``.
    """
    y = np.asarray(am.y, dtype=float)[:last_obs]
    starts: list[np.ndarray | None] = [None]
    if prev_params is not None:
        starts.append(_start_from(prev_params, o))
    starts.append(_generic_start(y, o))
    best: tuple[np.ndarray, float] | None = None
    for k, sv in enumerate(starts, start=1):
        try:
            res = _fit_once(am, last_obs, sv, None if k == 1 else dict(_RETRY_OPTIONS))
        except Exception as exc:  # arch/scipy raise assorted errors on degenerate windows
            log.debug("fit attempt %d raised %r", k, exc)
            continue
        p = np.asarray(res.params, dtype=float)
        if not valid_params(p, o):
            continue
        ll = float(res.loglikelihood)
        if res.convergence_flag == 0:
            return FitOutcome(p, "ok" if k == 1 else "retry", k, ll)
        if best is None or ll > best[1]:
            best = (p, ll)
    if prev_params is not None:
        return FitOutcome(np.asarray(prev_params, dtype=float).copy(), "fallback", len(starts), np.nan)
    if best is not None:
        return FitOutcome(best[0], "nonconverged", len(starts), best[1])
    return FitOutcome(None, "failed", len(starts), np.nan)


def walk_forward_variances(
    r: np.ndarray, o: int, window: int, refit_every: int, steps: int, model: str | None = None
) -> GarchPaths:
    """Rolling-window walk-forward on a finite return series (positions = sessions).

    Row ``s >= window-1`` of the result holds ``σ²_{s+k|s}``, ``k = 1..steps``, from the parameters of the
    latest refit origin ``j <= s`` (``j = window-1 + m·refit_every``) estimated on ``r[j-window+1 : j+1]``.
    """
    r = np.asarray(r, dtype=float)
    if not np.all(np.isfinite(r)):
        raise ValueError("returns must be finite")
    if window < 2 or refit_every < 1 or steps < 1:
        raise ValueError(f"bad walk-forward settings window={window} refit_every={refit_every} steps={steps}")
    t0 = time.perf_counter()
    n = len(r)
    names = param_names(o)
    var = np.full((n, steps), np.nan)
    records: list[dict] = []
    prev: np.ndarray | None = None
    for j in range(window - 1, n, refit_every):
        a, e = j - window + 1, min(j + refit_every, n)
        am = make_model(r[a:e], o)
        out = fit_with_retries(am, window, o, prev)
        rec = {"row": j, "status": out.status, "attempts": out.attempts, "loglik": out.loglik}
        rec.update(dict(zip(names, out.params if out.params is not None else [np.nan] * len(names))))
        records.append(rec)
        if out.params is None:
            continue
        prev = out.params
        fc = am.forecast(params=out.params, horizon=steps, start=window - 1, method="analytic", reindex=False)
        v = fc.variance.to_numpy()
        if v.shape != (e - j, steps):  # guards the start/reindex semantics relied on above
            raise RuntimeError(f"unexpected arch forecast shape {v.shape}, expected {(e - j, steps)}")
        var[j:e] = v
    params = pd.DataFrame.from_records(records, columns=["row", "status", "attempts", "loglik", *names])
    return GarchPaths(
        model=model or ("GJR" if o else "GARCH"),
        variances=var,
        refit_rows=params["row"].to_numpy(dtype=np.int64),
        params=params,
        seconds=time.perf_counter() - t0,
    )


def default_steps(asset: str) -> int:
    """Forecast depth shared by all horizons: the largest ``n_max`` of the asset's clock."""
    return max(C.n_max(h, asset) for h in C.HORIZONS)


def _fingerprint(daily: pd.DataFrame) -> str:
    days = pd.to_datetime(daily["session_date"]).to_numpy().astype("datetime64[D]").astype(np.int64)
    r = daily["r_cc"].to_numpy(dtype=float)
    return hashlib.sha256(days.tobytes() + r.tobytes()).hexdigest()


def variance_paths(
    daily: pd.DataFrame,
    asset: str,
    model: str,
    *,
    window: int | None = None,
    refit_every: int | None = None,
    steps: int | None = None,
) -> GarchPaths:
    """Walk-forward per-step variance paths aligned with the rows of ``daily`` (cached per data/settings).

    Rows whose ``r_cc`` is missing are skipped by the filter and get no forecast.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}; expected one of {sorted(MODELS)}")
    window = C.window() if window is None else int(window)
    refit_every = C.refit_every("garch", asset) if refit_every is None else int(refit_every)
    steps = default_steps(asset) if steps is None else int(steps)
    key = (model, asset, window, refit_every, _fingerprint(daily))
    hit = _CACHE.get(key)
    if hit is not None and hit.variances.shape[1] >= steps:
        _CACHE.move_to_end(key)
        return hit

    r_all = daily["r_cc"].to_numpy(dtype=float)
    rows = np.flatnonzero(np.isfinite(r_all))
    wf = walk_forward_variances(r_all[rows], MODELS[model], window, refit_every, steps, model=model)
    var = np.full((len(daily), steps), np.nan)
    var[rows] = wf.variances
    params = wf.params.copy()
    params["row"] = rows[params["row"].to_numpy(dtype=np.int64)]
    params.insert(1, "origin", daily["session_date"].to_numpy()[params["row"].to_numpy()])
    paths = GarchPaths(model, var, params["row"].to_numpy(dtype=np.int64), params, wf.seconds)

    c = paths.counts
    level = logging.WARNING if c["fallback"] + c["nonconverged"] + c["failed"] else logging.INFO
    log.log(
        level,
        "%s %s: %d refits (W=%d, every %d) in %.1fs; %s",
        model,
        asset,
        len(params),
        window,
        refit_every,
        wf.seconds,
        ", ".join(f"{k} {v}" for k, v in c.items()),
    )

    _CACHE[key] = paths
    while len(_CACHE) > _CACHE_SIZE:
        _CACHE.popitem(last=False)
    return paths


def clear_cache() -> None:
    _CACHE.clear()


def forecast_all_horizons(
    daily: pd.DataFrame,
    targets_by_h: dict[str, pd.DataFrame],
    asset: str,
    model: str = "GJR",
    *,
    window: int | None = None,
    refit_every: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Forecast frames (``FORECAST_COLUMNS``) for several horizons from ONE walk-forward.

    ``F`` at origin ``t`` is the sum of the first ``n_t`` per-step variances (``n_t`` from that horizon's
    targets). Only origins in the dev/holdout split with at least ``W`` returns are forecast.
    """
    n_ts = [tg["n_t"].to_numpy(dtype=float) for tg in targets_by_h.values()]
    need = [int(v[np.isfinite(v)].max()) for v in n_ts if np.isfinite(v).any()]
    steps = max([default_steps(asset), *need])
    paths = variance_paths(daily, asset, model, window=window, refit_every=refit_every, steps=steps)
    csum = np.cumsum(paths.variances, axis=1)
    out: dict[str, pd.DataFrame] = {}
    for h, tg in targets_by_h.items():
        al = align_targets(daily, tg)
        n_t = al["n_t"].to_numpy(dtype=float)
        idx = np.flatnonzero(predict_mask_for(al) & np.isfinite(n_t) & (n_t >= 1))
        F = np.full(len(al), np.nan)
        F[idx] = csum[idx, n_t[idx].astype(np.int64) - 1]
        out[h] = to_forecast_frame(asset, h, model, al["origin"], al["n_t"], F)
    return out


class _ArchForecaster:
    """Forecaster protocol adapter; ``window``/``refit_every`` default to the config (SPEC §6)."""

    name = ""

    def __init__(self, window: int | None = None, refit_every: int | None = None) -> None:
        self.window = window
        self.refit_every = refit_every

    def forecast(self, daily: pd.DataFrame, targets: pd.DataFrame, asset: str, horizon: str) -> pd.DataFrame:
        frames = forecast_all_horizons(
            daily, {horizon: targets}, asset, self.name, window=self.window, refit_every=self.refit_every
        )
        return frames[horizon]

    def paths(self, daily: pd.DataFrame, asset: str) -> GarchPaths:
        """The cached walk-forward behind ``forecast`` (refit parameters and convergence statuses)."""
        return variance_paths(daily, asset, self.name, window=self.window, refit_every=self.refit_every)


class GARCH(_ArchForecaster):
    """Model 3: GARCH(1,1) with Student-t errors."""

    name = "GARCH"


class GJR(_ArchForecaster):
    """Model 4: GJR-GARCH(1,1) with Student-t errors."""

    name = "GJR"
