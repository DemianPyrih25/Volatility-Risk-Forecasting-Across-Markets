"""Forecast horizons, cumulative-variance targets and the dev/holdout split (SPEC §6).

For an origin ``t`` (a session of one asset) the target window is

- ``1d``: the next session (the next row of the gold table);
- ``1w`` / ``1m``: every session dated in ``(t, t+7]`` / ``(t, t+30]`` calendar days.

``n_t`` = sessions in the window (at most ``n_max``), ``y = Σ tv``, ``ybar = y / n_t`` and ``window_end`` = date
of the last session in the window. A window is *complete* only when it is fully observable by ``last_date``
(``t + days <= last_date``; for 1d the next row exists and is dated ``<= last_date``) and non-empty;
otherwise ``window_end``/``y``/``ybar`` are null and the row is ``dropped``. An empty window (``n_t = 0``) can
also occur mid-sample after a long run of invalid sessions, so ``window_end`` may be NaT before the end of the
sample. Complete windows that cross the dev/holdout boundary keep their ``y`` (they are legitimate training rows
for later holdout origins) but are ``dropped`` from evaluation.

OOS start (§6): ``build_all_targets`` also labels every origin before the asset's out-of-sample start
(``oos_start``: first origin with ``W`` eligible training rows at every horizon) as ``dropped``. Those rows
keep ``y``/``ybar``/``window_end`` and still train later origins. Every forecaster and the evaluation select
origins by ``split``, so all models and horizons of an asset share one start date.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd

from volrisk import config as C

log = logging.getLogger(__name__)

TARGET_COLUMNS = ["asset", "horizon", "origin", "window_end", "n_t", "y", "ybar", "split"]
SPLITS = ("dev", "holdout", "dropped")
_STALE_DAYS = 4  # longest gap between the last session and last_date on any calendar (weekend + holiday)


def _day(x) -> np.datetime64:
    return np.datetime64(pd.Timestamp(x).date(), "D")


def _session_dates(daily: pd.DataFrame) -> pd.Series:
    """``session_date`` as datetime64 (``io.load_daily`` already returns datetime64[ms])."""
    sd = daily["session_date"]
    if not pd.api.types.is_datetime64_any_dtype(sd):
        sd = pd.to_datetime(sd).astype("datetime64[ms]")
    return sd


def window_bounds(dates: np.ndarray, horizon: str, last_date: date) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row bounds ``[lo, hi)`` of every origin's target window and whether it is observable by ``last_date``.

    ``dates`` are the strictly increasing session dates of one asset. Returns ``(lo, hi, observable)``.
    """
    d = np.asarray(dates).astype("datetime64[D]")
    T = len(d)
    last = _day(last_date)
    idx = np.arange(T)
    if horizon == "1d":
        lo = np.minimum(idx + 1, T)
        hi = np.minimum(idx + 2, T)
        observable = np.zeros(T, dtype=bool)
        has_next = lo < T
        observable[has_next] = d[lo[has_next]] <= last
    else:
        end = d + np.timedelta64(C.horizon_days(horizon), "D")
        lo = np.searchsorted(d, d, side="right")
        hi = np.searchsorted(d, end, side="right")
        observable = end <= last
    return lo, hi, observable


def assign_split(origin: np.ndarray, window_end: np.ndarray) -> np.ndarray:
    """SPEC §6 ``split``: ``dev`` if ``window_end <= dev_end``; ``holdout`` if ``origin >= dev_end`` (2025-09-30,
    the last dev session) and ``window_end <= data_end``; otherwise ``dropped`` (also for a NaT window_end).

    The thresholds are the literal calendar dates, never the last session present in the data, so the sealed
    holdout sample does not depend on whether 2025-09-30 is a valid session of the asset.
    """
    o = np.asarray(origin).astype("datetime64[D]")
    we = np.asarray(window_end).astype("datetime64[D]")
    ok = ~np.isnat(we)
    dev_end = _day(C.dev_end())
    dev = ok & (we <= dev_end)
    hold = ok & ~dev & (o >= dev_end) & (we <= _day(C.data_end()))
    return np.where(dev, "dev", np.where(hold, "holdout", "dropped")).astype(object)


def build_targets(daily: pd.DataFrame, asset: str, horizon: str, last_date: date) -> pd.DataFrame:
    """Targets of one (asset, horizon); one row per session of ``daily`` (SPEC §6).

    daily: gold rows of ONE asset sorted by ``session_date`` (rows = sessions).
    last_date: ``config.dev_end()`` in dev mode, ``config.data_end()`` in the holdout run.
    """
    if horizon not in C.HORIZONS:
        raise ValueError(f"unknown horizon {horizon!r}")
    if "asset" in daily.columns and len(daily) and not (daily["asset"] == asset).all():
        raise ValueError(f"build_targets({asset}): daily contains rows of other assets")
    sd = _session_dates(daily)
    d = sd.to_numpy().astype("datetime64[D]")
    if len(d) > 1 and not (np.diff(d.astype("int64")) > 0).all():
        raise ValueError(f"build_targets({asset}): session_date must be strictly increasing")

    T = len(d)
    if T == 0:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    if (_day(last_date) - d[-1]).astype(int) > _STALE_DAYS:
        log.warning("%s: data ends %s, last_date %s - trailing windows look truncated", asset, d[-1], last_date)
    lo, hi, observable = window_bounds(d, horizon, last_date)
    n_t = (hi - lo).astype("int64")
    cap = C.n_max(horizon, asset)
    if n_t.max() > cap:
        bad = sd.to_numpy()[n_t > cap][:5]
        raise ValueError(f"{asset} {horizon}: n_t exceeds n_max={cap} at origins {bad} (calendar/data bug)")

    # Exact row-order sums over at most n_max rows (no cumulative-sum cancellation).
    tv = daily["tv"].to_numpy(dtype=float)
    off = np.arange(max(int(n_t.max()), 1))
    take = np.minimum(lo[:, None] + off, T - 1)
    y = np.where(off[None, :] < n_t[:, None], tv[take], 0.0).sum(axis=1)

    complete = observable & (n_t > 0) & np.isfinite(y)
    n_bad = int((observable & (n_t > 0) & ~np.isfinite(y)).sum())
    if n_bad:
        log.warning("%s %s: %d windows with non-finite tv dropped", asset, horizon, n_bad)

    origin = sd.to_numpy()
    window_end = np.where(complete, origin[np.maximum(hi - 1, 0)], np.array("NaT", dtype=origin.dtype))
    y = np.where(complete, y, np.nan)
    ybar = np.where(complete, y / np.maximum(n_t, 1), np.nan)
    split = assign_split(origin, window_end)

    out = pd.DataFrame(
        {
            "asset": asset,
            "horizon": horizon,
            "origin": origin,
            "window_end": window_end,
            "n_t": n_t,
            "y": y,
            "ybar": ybar,
            "split": split,
        },
        columns=TARGET_COLUMNS,
    )
    counts = out["split"].value_counts().to_dict()
    log.info("targets %s %s: %s", asset, horizon, {k: counts.get(k, 0) for k in SPLITS})
    return out


def default_warmup(asset: str) -> int:
    """Leading sessions that cannot train a HAR/ML model: the monthly regressor (mean over the last ``m``
    sessions, ``features.har_frame``) first exists at row ``m − 1``."""
    return C.har_lags(asset)[2] - 1


def _days(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s).to_numpy().astype("datetime64[D]")


def eligible_counts(targets: pd.DataFrame, warmup: int = 0) -> np.ndarray:
    """Eligible training rows per origin for ONE (asset, horizon) (§6 purge).

    ``targets`` holds one row per session in date order (``build_targets`` output). The result for origin
    ``t`` counts rows ``s >= warmup`` whose target is observed (finite ``ybar``) and whose window has ended
    (``window_end_s <= t``).
    """
    o = _days(targets["origin"])
    if len(o) > 1 and not (np.diff(o.astype("int64")) > 0).all():
        raise ValueError("eligible_counts: origins must be strictly increasing (one horizon of one asset)")
    we = _days(targets["window_end"])
    ok = np.isfinite(targets["ybar"].to_numpy(dtype=float)) & ~np.isnat(we)
    ok[: max(int(warmup), 0)] = False
    return np.searchsorted(np.sort(we[ok]), o, side="right")


def oos_start(targets: pd.DataFrame, window: int | None = None, warmup: int | None = None) -> pd.Timestamp | None:
    """SPEC §6 OOS start of ONE asset: the first origin with ``window`` eligible rows at every horizon.

    targets: ``build_targets`` rows of ONE asset for one or more horizons. window: ``W`` (default
    ``config.window()``). warmup: leading rows without regressors (default ``default_warmup(asset)``), so at
    the start every model, the HAR reference included, has a full window. Returns None if the start is
    never reached.
    """
    if targets.empty:
        return None
    assets = pd.unique(targets["asset"])
    if len(assets) != 1:
        raise ValueError(f"oos_start expects one asset, got {list(assets)}")
    W = C.window() if window is None else int(window)
    wu = default_warmup(str(assets[0])) if warmup is None else int(warmup)
    firsts = []
    for _, g in targets.groupby("horizon", sort=False):
        hit = np.flatnonzero(eligible_counts(g, wu) >= W)
        if not hit.size:
            return None
        firsts.append(pd.Timestamp(g["origin"].iloc[hit[0]]))
    return max(firsts)  # counts are non-decreasing in the origin, so this is the first origin valid for all


def apply_oos_start(targets: pd.DataFrame, start: pd.Timestamp | date | None) -> pd.DataFrame:
    """Copy of ``targets`` with every origin before ``start`` labelled ``dropped`` (every origin if ``start``
    is None). ``window_end``/``y``/``ybar`` are kept, so those rows still train later origins."""
    out = targets.copy()
    if start is None:
        pre = np.ones(len(out), dtype=bool)
    else:
        pre = (pd.to_datetime(out["origin"]) < pd.Timestamp(start)).to_numpy()
    out.loc[pre, "split"] = "dropped"
    return out


def build_all_targets(daily_all: pd.DataFrame, last_date: date, window: int | None = None) -> pd.DataFrame:
    """Targets for every asset in ``daily_all`` (gold rows, any number of assets) × every horizon.

    Origins before each asset's ``oos_start`` (``W = window``, default ``config.window()``) are ``dropped``.
    """
    present = list(pd.unique(daily_all["asset"]))
    order = [a for a in C.ASSETS if a in present] + sorted(a for a in present if a not in C.ASSETS)
    frames = []
    for a in order:
        da = daily_all[daily_all["asset"] == a].sort_values("session_date", kind="stable").reset_index(drop=True)
        ta = pd.concat([build_targets(da, a, h, last_date) for h in C.HORIZONS], ignore_index=True)
        start = oos_start(ta, window)
        if start is None:
            log.warning("%s: never reaches W eligible rows at every horizon - no out-of-sample origins", a)
        else:
            log.info("%s: OOS start %s", a, start.date())
        frames.append(apply_oos_start(ta, start))
    if not frames:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    return pd.concat(frames, ignore_index=True)
