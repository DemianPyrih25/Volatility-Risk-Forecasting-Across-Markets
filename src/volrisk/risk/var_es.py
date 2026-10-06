"""VaR/ES risk models: FHS, historical simulation, RiskMetrics, Normal (SPEC §9).

Conventions: percent log returns ``r_cc``; loss ``L = -r``; VaR and ES are **positive** numbers; ``mu = 0``.
Every risk series is indexed by its **target** session date ``d`` and uses information up to the close of the
previous session only. A 1d variance forecast made at origin ``t`` (``origin`` in the forecasts table) is the
forecast for the next session of the same asset in the gold daily table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import norm

from volrisk import config as C
from volrisk.models.simple import EWMA_INIT

ALPHA_99 = 0.01
ALPHA_975 = 0.025
RISK_MODELS = ("HS-250", "RiskMetrics", "GJR+FHS", "HAR*+FHS", "COMBO+FHS", "COMBO+Normal")
NORMAL_MODELS = ("RiskMetrics", "COMBO+Normal")
FHS_BASE = {"GJR+FHS": "GJR", "COMBO+FHS": "COMBO"}  # 'HAR*+FHS' -> the frozen HAR* member
MEASURE_COLUMNS = ["sigma", "var99", "var975", "es975", "pool_n"]
RISK_COLUMNS = ["asset", "model", "date", "r_cc", "var99", "var975", "es975", "sigma", "har_member", "split"]

# Standard-normal multipliers (positive): VaR99 ≈ 2.326, VaR97.5 ≈ 1.960, ES97.5 = φ(z)/α ≈ 2.338.
Z99 = float(-norm.ppf(ALPHA_99))
Z975 = float(-norm.ppf(ALPHA_975))
ES975_NORMAL = float(norm.pdf(norm.ppf(ALPHA_975)) / ALPHA_975)


# --------------------------------------------------------------------------------------------- helpers
def _dates(x) -> pd.Index:
    """Index of dates; date-like values become a ``DatetimeIndex[ns]`` so different units compare."""
    idx = pd.Index(x)
    if isinstance(idx, pd.DatetimeIndex) or pd.api.types.infer_dtype(idx, skipna=True) in (
        "date",
        "datetime",
        "datetime64",
    ):
        idx = pd.DatetimeIndex(pd.to_datetime(idx)).as_unit("ns")
    return idx


def _series(x: pd.Series, name: str) -> pd.Series:
    """Float series on a normalised, unique, sorted date index."""
    if not isinstance(x, pd.Series):
        raise TypeError(f"{name} must be a pandas Series indexed by session date")
    s = pd.Series(np.asarray(x, dtype=float), index=_dates(x.index), name=name)
    if not s.index.is_unique:
        raise ValueError(f"{name}: duplicate dates in the index")
    return s.sort_index()


def _measures(dates, sigma, var99, var975, es975, pool_n) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "sigma": np.asarray(sigma, dtype=float),
            "var99": np.asarray(var99, dtype=float),
            "var975": np.asarray(var975, dtype=float),
            "es975": np.asarray(es975, dtype=float),
            "pool_n": np.asarray(pool_n, dtype=np.int64),
        },
        index=pd.Index(dates, name="date"),
    )
    return out


def _empty_measures() -> pd.DataFrame:
    return _measures(pd.DatetimeIndex([], dtype="datetime64[ns]"), [], [], [], [], [])


def _tail(pool: np.ndarray) -> tuple[float, float, float]:
    """(Q_0.01, Q_0.025, mean of the pool at or below Q_0.025) with np.quantile(method='linear')."""
    q99, q975 = np.quantile(pool, [ALPHA_99, ALPHA_975], method="linear")
    return float(q99), float(q975), float(pool[pool <= q975].mean())


# --------------------------------------------------------------------------------------------- FHS
def _z_scores(r: pd.Series, sigma2: pd.Series) -> pd.Series:
    """z_s = r_s / sqrt(sigma2_s) on dates where both are observed (sigma2_s was made at the previous session)."""
    z = r / np.sqrt(sigma2.reindex(r.index))
    return z[np.isfinite(z)]


def _fhs_inputs(r: pd.Series, sigma2: pd.Series) -> tuple[pd.Series, pd.Series]:
    r = _series(r, "r")
    s2 = _series(sigma2, "sigma2")
    s2 = s2[np.isfinite(s2)]
    if (s2 <= 0).any():
        raise ValueError("sigma2 must be strictly positive")
    return r, s2


def _pool_bounds(z_index: pd.Index, dates: pd.Index, pool: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice [lo, hi) of the z series forming the pool for each date: the last ``pool`` z strictly before it."""
    hi = z_index.searchsorted(dates, side="left")
    return np.maximum(hi - pool, 0), hi


def fhs(r: pd.Series, sigma2: pd.Series, pool: int = 1000, min_pool: int = 500) -> pd.DataFrame:
    """Filtered historical simulation (SPEC §9).

    r: returns indexed by session date; sigma2: 1d variance forecasts indexed by **target** session date.
    For each target date d with at least ``min_pool`` past z: ``VaR = -sigma_d·Q_alpha(pool)``,
    ``ES975 = -sigma_d·mean(z | z <= Q_0.025(pool))`` where the pool is the last ``pool`` z strictly before d.
    Returns MEASURE_COLUMNS indexed by date (only dates where the pool is large enough).
    """
    r, s2 = _fhs_inputs(r, sigma2)
    z = _z_scores(r, s2)
    zv = z.to_numpy()
    lo, hi = _pool_bounds(z.index, s2.index, pool)
    n = hi - lo
    keep = n >= min_pool
    if not keep.any():
        return _empty_measures()
    sigma = np.sqrt(s2.to_numpy()[keep])
    tails = np.array([_tail(zv[a:b]) for a, b in zip(lo[keep], hi[keep], strict=True)])
    return _measures(s2.index[keep], sigma, -sigma * tails[:, 0], -sigma * tails[:, 1], -sigma * tails[:, 2], n[keep])


def fhs_pools(r: pd.Series, sigma2: pd.Series, dates, pool: int = 1000) -> list[np.ndarray]:
    """The FHS pool of standardised returns behind each date (same pool as ``fhs``); used by the ES backtest."""
    r, s2 = _fhs_inputs(r, sigma2)
    z = _z_scores(r, s2)
    zv = z.to_numpy()
    lo, hi = _pool_bounds(z.index, _dates(dates), pool)
    if np.any(hi - lo == 0):
        raise ValueError("empty FHS pool for some dates")
    return [zv[a:b] for a, b in zip(lo, hi, strict=True)]


# --------------------------------------------------------------------------------------------- HS / parametric
def historical_sim(r: pd.Series, window: int = 250) -> pd.DataFrame:
    """Historical simulation: empirical quantiles / tail mean of the previous ``window`` returns (strictly
    before d). ``sigma`` is the window's sample std (reference only); ``pool_n = window``."""
    r = _series(r, "r").dropna()
    x = r.to_numpy()
    if x.size <= window:
        return _empty_measures()
    w = sliding_window_view(x, window)[:-1]  # row i = x[i : i+window] -> date i+window
    q = np.quantile(w, [ALPHA_99, ALPHA_975], axis=1, method="linear")
    tail = np.where(w <= q[1][:, None], w, np.nan)
    es = -np.nanmean(tail, axis=1)
    return _measures(r.index[window:], w.std(axis=1, ddof=1), -q[0], -q[1], es, np.full(len(w), window))


def hs_windows(r: pd.Series, dates, window: int = 250) -> list[np.ndarray]:
    """The HS window (previous ``window`` returns strictly before each date); used by the ES backtest."""
    r = _series(r, "r").dropna()
    x = r.to_numpy()
    hi = r.index.searchsorted(_dates(dates), side="left")
    if np.any(hi < window):
        raise ValueError(f"fewer than {window} returns before some dates")
    return [x[h - window : h] for h in hi]


def normal_from_sigma2(sigma2: pd.Series) -> pd.DataFrame:
    """Normal VaR/ES with mu = 0 from a variance series indexed by target date (``pool_n = 0``)."""
    s2 = _series(sigma2, "sigma2").dropna()
    if (s2 <= 0).any():
        raise ValueError("sigma2 must be strictly positive")
    sigma = np.sqrt(s2.to_numpy())
    return _measures(s2.index, sigma, Z99 * sigma, Z975 * sigma, ES975_NORMAL * sigma, np.zeros(len(s2)))


def riskmetrics(r: pd.Series, lam: float = 0.94, init: int = EWMA_INIT) -> pd.DataFrame:
    """RiskMetrics: EWMA variance + Normal quantiles.

    The variance for the first date after the burn-in, ``sigma2_init``, is the sample variance of the first
    ``init`` returns; then ``sigma2_{t+1} = lam·sigma2_t + (1-lam)·r_t²``. With the defaults this is the
    ``EWMA`` forecaster's 1d variance (SPEC §7 #2) re-indexed to the target date.
    """
    r = _series(r, "r").dropna()
    x = r.to_numpy()
    if x.size <= init:
        return _empty_measures()
    s2 = np.empty(x.size - init)
    s = float(np.var(x[:init], ddof=1))
    for k, t in enumerate(range(init, x.size)):
        s2[k] = s
        s = lam * s + (1.0 - lam) * x[t] ** 2
    return normal_from_sigma2(pd.Series(s2, index=r.index[init:]))


# --------------------------------------------------------------------------------------------- risk frame
def _risk_cfg() -> dict:
    return C.load()["risk"]


def _one_asset(daily: pd.DataFrame, asset: str | None) -> pd.DataFrame:
    if asset is not None and "asset" in daily.columns:
        daily = daily[daily["asset"] == asset]
    return daily.sort_values("session_date")


def session_returns(daily: pd.DataFrame, asset: str | None = None) -> pd.Series:
    """``r_cc`` of one asset indexed by session date (DatetimeIndex[ns])."""
    d = _one_asset(daily, asset)
    return pd.Series(d["r_cc"].to_numpy(dtype=float), index=_dates(d["session_date"]), name="r_cc")


def _combo_from_members(fc: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight mean of the COMBO members' F on origins where all members exist (SPEC §7, #11)."""
    members = list(C.load()["models"]["combo_members"])
    wide = fc[fc["model"].isin(members)].pivot(index="origin", columns="model", values="F")
    if set(members) - set(wide.columns):
        return fc.iloc[0:0]
    wide = wide[members].dropna()
    return pd.DataFrame({"model": "COMBO", "origin": wide.index, "F": wide.mean(axis=1).to_numpy()})


def sigma2_by_target(
    daily: pd.DataFrame, forecasts_1d: pd.DataFrame, model: str, asset: str | None = None
) -> pd.Series:
    """1d variance forecasts of ``model`` re-indexed from origin t to the next session in ``daily``.

    The forecast made at the last session (no next session in ``daily``) is dropped. If the table has no
    ``COMBO`` rows, COMBO is rebuilt as the equal-weight mean of the configured members.
    """
    sessions = _dates(_one_asset(daily, asset)["session_date"])
    fc = forecasts_1d
    if "horizon" in fc.columns:
        fc = fc[fc["horizon"] == "1d"]
    if asset is not None and "asset" in fc.columns:
        fc = fc[fc["asset"] == asset]
    sel = fc[fc["model"] == model]
    if sel.empty and model == "COMBO":
        sel = _combo_from_members(fc)
    if sel.empty:
        raise KeyError(f"no 1d forecasts for model {model!r}")
    origins = _dates(sel["origin"])
    if not origins.is_unique:
        raise ValueError(f"{model}: duplicate 1d forecast origins")
    pos = sessions.get_indexer(origins)
    if np.any(pos < 0):
        raise ValueError(f"{model}: forecast origins that are not sessions of the daily table")
    nxt = pos + 1
    ok = nxt < len(sessions)
    s2 = pd.Series(sel["F"].to_numpy(dtype=float)[ok], index=sessions[nxt[ok]], name=model)
    return s2.sort_index()


def risk_frame(
    asset: str,
    daily: pd.DataFrame,
    forecasts_1d: pd.DataFrame,
    har_star: str,
    start_date=None,
    *,
    common: bool = True,
) -> pd.DataFrame:
    """Long VaR/ES frame of the six SPEC §9 risk models for one asset (RISK_COLUMNS).

    daily: gold rows (pandas) with ``session_date`` and ``r_cc``; forecasts_1d: FORECAST_COLUMNS rows with
    horizon '1d' (other horizons/assets are filtered out); har_star: the frozen HAR-family member whose
    forecasts feed 'HAR*+FHS' (recorded in ``har_member``). With ``common=True`` only dates on which all six
    models and ``r_cc`` are available are kept, so the window starts when the last FHS pool reaches its
    minimum size. ``start_date`` (e.g. 2021-01-01) trims the output only — pools still use all earlier z.
    """
    cfg = _risk_cfg()
    pool, min_pool = int(cfg["fhs_pool"]), int(cfg["fhs_min"])
    window, lam = int(cfg["hs_window"]), float(cfg["ewma_lambda"])
    d = _one_asset(daily, asset)
    r = session_returns(d)
    s2 = {m: sigma2_by_target(d, forecasts_1d, m, asset) for m in dict.fromkeys(("GJR", har_star, "COMBO"))}
    measures = {
        "HS-250": historical_sim(r, window),
        "RiskMetrics": riskmetrics(r, lam, init=EWMA_INIT),  # same burn-in as the EWMA model, not hs_window
        "GJR+FHS": fhs(r, s2["GJR"], pool, min_pool),
        "HAR*+FHS": fhs(r, s2[har_star], pool, min_pool),
        "COMBO+FHS": fhs(r, s2["COMBO"], pool, min_pool),
        "COMBO+Normal": normal_from_sigma2(s2["COMBO"]),
    }
    parts = []
    for name, m in measures.items():
        m = m.dropna(subset=["var99", "var975", "es975"])
        parts.append(
            pd.DataFrame(
                {
                    "asset": asset,
                    "model": name,
                    "date": m.index,
                    "r_cc": r.reindex(m.index).to_numpy(),
                    "var99": m["var99"].to_numpy(),
                    "var975": m["var975"].to_numpy(),
                    "es975": m["es975"].to_numpy(),
                    "sigma": m["sigma"].to_numpy(),
                }
            )
        )
    out = pd.concat(parts, ignore_index=True)
    out = out[np.isfinite(out["r_cc"])]
    if start_date is not None:
        out = out[out["date"] >= pd.Timestamp(start_date)]
    if common:
        n_models = out.groupby("date")["model"].transform("nunique")
        out = out[n_models == len(RISK_MODELS)]
    out = out.copy()
    out["har_member"] = np.where(out["model"] == "HAR*+FHS", har_star, None)
    out["split"] = np.where(out["date"] < pd.Timestamp(C.holdout_start()), "dev", "holdout")
    sd = d["session_date"]
    if pd.api.types.is_datetime64_dtype(sd.dtype):
        out["date"] = out["date"].astype(sd.dtype)  # keep the gold table's date unit for joins
    order = {m: i for i, m in enumerate(RISK_MODELS)}
    out = out.sort_values(["model", "date"], key=lambda c: c.map(order) if c.name == "model" else c)
    return out[RISK_COLUMNS].reset_index(drop=True)


def fhs_base_model(model: str, har_member: str | None = None) -> str:
    """Variance model behind an FHS risk model ('HAR*+FHS' needs the recorded ``har_member``)."""
    if model == "HAR*+FHS":
        if not har_member:
            raise ValueError("HAR*+FHS needs its har_member")
        return har_member
    return FHS_BASE[model]
