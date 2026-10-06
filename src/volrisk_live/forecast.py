"""Live forecasts of the next session, week and month with the frozen models (docs/LIVE_SPEC.md §3).

Nothing here re-implements a model. The live run is the frozen holdout walk-forward
(``volrisk.pipeline.compute_forecasts``) on the live rebuild of the gold table, with one runtime difference: the
data end date is moved to the live end (``context.live_end``), and the **live origins** get an ex-ante target
window so that the frozen ``predict_mask_for`` forecasts them:

- live origins = the last session of each asset plus every origin after the asset's last holdout origin whose
  window is still incomplete. They get ``split = 'holdout'``, ``y = ybar = NaN``, ``window_end = NaT`` (so they
  never train) and ``n_t`` = the number of *scheduled* sessions in ``(t, t+days]`` (1d: the next scheduled session),
  capped at ``n_max`` (``volrisk.sessions.session_schedule``).
- Refit points cannot move. GARCH refits sit at fixed return positions, the ML refits at fixed session rows from
  the OOS start, and HAR refits at every origin. A row whose window is not complete never trains. Extra origins at
  the end of the sample therefore leave every earlier forecast unchanged. ``check_reproduces_holdout`` proves this
  on every run: each forecast at an origin of the dev run (``data/results/forecasts.parquet``, hashed in
  SEALED.json) and of the one-time holdout run (``data/results/holdout/forecasts.parquet``) must equal the stored
  value (rtol 1e-9, atol 0), otherwise the run is aborted.
- Next-session VaR/ES uses the frozen ``risk_frame`` on the gold rows plus one placeholder row for the next
  scheduled session. All six risk models read only returns strictly before the target date, so the placeholder's
  value never enters a number. The rows on realised dates are checked against the holdout run's ``risk.parquet``.
- The holdout-run result files were written at the opening, after the seal, so neither SEALED.json nor
  holdout_log.jsonl hashes them. Every payload therefore records the SHA-256 of each file it was compared with
  (``checks.reproduction.files``, ``checks.risk_reproduction.files``), and the first ledger entry pins them.
- Timing: ``run_utc`` / ``run_id`` is the start of the run (``now_utc``). The per-asset flags
  ``recorded_before_open`` / ``recorded_before_close`` are evaluated at the **recording time**: the machine clock
  when the payload is built, immediately before ``ledger.append``, plus ``RECORD_MARGIN`` for the ledger write. An
  asset whose next session has already closed by then gets no forecast or VaR/ES row (``checks.closed_assets``),
  because its outcome is already known. On the project ledger a ``now_utc`` more than ``FUTURE_SLACK`` before the
  machine clock is refused. Test sandboxes may replay a past time, and the payload is then marked
  ``checks.replayed_now``.

``data_end`` is overridden only in this process. Spawned forecast workers (Windows ``spawn``) see the frozen value,
so every split/target computation happens here and the workers only run the frozen models on ready-made targets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

from volrisk import config as C
from volrisk import holdout, io
from volrisk.pipeline import FORECAST_MODELS, REPRO_RTOL, _forecast_task, results_dir
from volrisk.sessions import session_schedule
from volrisk.targets import TARGET_COLUMNS, build_all_targets
from volrisk_live import context, paths, schema

log = logging.getLogger("volrisk_live.forecast")

ANN = {"crypto": 365, "xnys": 252, "fx": 260}  # sessions per year for the annualised display (§5)
LIVE_COLUMNS = ["live", "window_first", "window_last"]
FC_KEYS = ["asset", "horizon", "model", "origin"]
RISK_KEYS = ["asset", "model", "date"]
RISK_VALUES = ["r_cc", "var99", "var975", "es975", "sigma"]
MODEL_ORDER = ("COMBO", *FORECAST_MODELS, "IV", "IV-cal")
HORIZON_TEXT = {"1d": "next session", "1w": "next week", "1m": "next month"}
_SCHEDULE_PAD = 14  # calendar days searched for the next scheduled session (longest closure is far shorter)
# update summary fields copied into payload.checks.data_consistency (data/live/state.json, update.py)
FUTURE_SLACK = timedelta(minutes=5)  # tolerated clock skew for an explicit ``now_utc`` (as update.py)
RECORD_MARGIN = timedelta(minutes=1)  # payload built -> ledger line written (lock, chain scan, writes take seconds)
IV_MAX_AGE_DAYS = 7  # payload.implied: latest IV only if this recent (EVZ, used for EURUSD, ended in 2023)
STATE_KEYS = ("now_utc", "finished_utc", "end", "last_sessions", "expected_last_sessions", "stale_assets", "files")


class ReproductionError(RuntimeError):
    """The live run did not reproduce a stored dev/holdout-run forecast or VaR/ES value: nothing is recorded."""


# --------------------------------------------------------------------------------------------- helpers
def _days(x) -> np.ndarray:
    return pd.to_datetime(pd.Series(x)).to_numpy().astype("datetime64[D]")


def _iso(x) -> str | None:
    return None if x is None or pd.isna(x) else pd.Timestamp(x).date().isoformat()


def _utc(now: datetime | None) -> datetime:
    now = datetime.now(timezone.utc) if now is None else now
    now = now.replace(tzinfo=timezone.utc) if now.tzinfo is None else now.astimezone(timezone.utc)
    return now.replace(microsecond=0)


def _wall_clock() -> datetime:
    """The machine's UTC clock (recorded next to ``run_utc``; patched in tests)."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def normalise_daily(daily_all: pd.DataFrame) -> pd.DataFrame:
    """Gold rows sorted by (asset, session_date) with datetime64 session dates, as ``io.load_daily`` returns them."""
    d = daily_all
    if not pd.api.types.is_datetime64_any_dtype(d["session_date"]):
        d = d.assign(session_date=pd.to_datetime(d["session_date"]).astype("datetime64[ms]"))
    return d.sort_values(["asset", "session_date"], kind="stable").reset_index(drop=True)


def _assets(daily_all: pd.DataFrame) -> list[str]:
    present = set(daily_all["asset"].unique())
    return [a for a in C.ASSETS if a in present]


def ann_factor(asset: str) -> int:
    return ANN[C.clock(asset)]


def vol_ann(F: float, n_t: int, asset: str) -> float:
    """Annualised volatility in % from a cumulative variance forecast ``F`` (%²) over ``n_t`` sessions."""
    return math.sqrt(F / n_t * ann_factor(asset))


def rows_sha256(rows: pd.DataFrame) -> str:
    """SHA-256 of one asset's gold rows as CSV text, so anyone can recompute it from the live parquet.

    Rows are sorted by ``session_date`` (written ``YYYY-MM-DD``) and the columns are kept in table order. Floats use
    ``%.17g`` (round-trip exact), NaN becomes an empty field, lines end in ``\\n`` and there is no index.
    """
    d = rows.sort_values("session_date", kind="stable")
    d = d.assign(session_date=pd.to_datetime(d["session_date"]).dt.strftime("%Y-%m-%d"))
    text = d.to_csv(index=False, lineterminator="\n", float_format="%.17g", na_rep="")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------------------------- ex-ante windows
def schedule_window(asset: str, horizon: str, origins) -> pd.DataFrame:
    """Ex-ante target window of each origin from the session calendar (no data needed).

    ``n_t`` = scheduled sessions in ``(t, t+days]`` capped at ``n_max`` (1d: the next scheduled session, ``n_t = 1``);
    ``window_first`` / ``window_last`` = first / last of those sessions (the ``n_max``-th one when capped).
    Returns columns ``origin, n_t, window_first, window_last`` aligned with ``origins``.
    """
    o = _days(origins)
    out = pd.DataFrame({"origin": pd.to_datetime(pd.Series(origins)).to_numpy()})
    if len(o) == 0:
        return out.assign(n_t=np.array([], dtype="int64"), window_first=pd.NaT, window_last=pd.NaT)
    days = C.horizon_days(horizon)
    cap = C.n_max(horizon, asset)
    lo_day = pd.Timestamp(o.min()).date() + timedelta(days=1)
    hi_day = pd.Timestamp(o.max()).date() + timedelta(days=days + _SCHEDULE_PAD)
    sched = session_schedule(asset, lo_day, hi_day)["session_date"].to_numpy().astype("datetime64[D]")
    lo = np.searchsorted(sched, o, side="right")  # first scheduled session after t
    if np.any(lo >= len(sched)):
        raise ValueError(f"{asset}: no scheduled session within {_SCHEDULE_PAD} days after some origins")
    hi = lo + 1 if horizon == "1d" else np.searchsorted(sched, o + np.timedelta64(days, "D"), side="right")
    n = np.minimum(hi - lo, cap).astype("int64")
    if np.any(n < 1):
        raise ValueError(f"{asset} {horizon}: empty scheduled window at origins {o[n < 1][:5]}")
    out["n_t"] = n
    out["window_first"] = sched[lo].astype("datetime64[ms]")
    out["window_last"] = sched[lo + n - 1].astype("datetime64[ms]")
    return out


def next_session(asset: str, after) -> date:
    """First scheduled session of ``asset`` strictly after the date ``after``."""
    w = schedule_window(asset, "1d", [pd.Timestamp(after)])
    return pd.Timestamp(w["window_first"].iloc[0]).date()


def session_bounds(asset: str, day: date) -> tuple[datetime, datetime]:
    """(open_utc, close_utc) of a scheduled session."""
    s = session_schedule(asset, day, day)
    if s.height != 1:
        raise ValueError(f"{asset}: {day} is not a scheduled session")
    return s["open_utc"][0], s["close_utc"][0]


# --------------------------------------------------------------------------------------------- targets
def live_targets(daily_all: pd.DataFrame, end: date) -> pd.DataFrame:
    """Frozen ``build_all_targets(daily_all, last_date=end)`` under ``live_end(end)``, plus the live origins (§3).

    Returns TARGET_COLUMNS + ``live`` (bool), ``window_first``/``window_last`` (scheduled window of live origins,
    NaT elsewhere). Rows that are not live are exactly the frozen targets at that end date.
    """
    daily_all = normalise_daily(daily_all)
    with context.live_end(end):
        tg = build_all_targets(daily_all, last_date=end)
    tg = tg.reset_index(drop=True)
    live = np.zeros(len(tg), dtype=bool)
    n_t = tg["n_t"].to_numpy(dtype="int64").copy()
    first = np.full(len(tg), np.datetime64("NaT"), dtype="datetime64[ms]")
    last = first.copy()
    for (asset, horizon), idx in tg.groupby(["asset", "horizon"], sort=False).groups.items():
        g = tg.loc[idx]
        origin = pd.to_datetime(g["origin"])
        hold = (g["split"] == "holdout").to_numpy()
        after = (origin > origin[hold].max()) if hold.any() else (origin >= pd.Timestamp(C.dev_end()))
        is_live = (after & g["window_end"].isna()).to_numpy() | (origin == origin.max()).to_numpy()
        rows = np.asarray(idx)[is_live]
        win = schedule_window(asset, horizon, tg.loc[rows, "origin"])
        live[rows] = True
        n_t[rows] = win["n_t"].to_numpy()
        first[rows] = win["window_first"].to_numpy()
        last[rows] = win["window_last"].to_numpy()
    tg.loc[live, "split"] = "holdout"
    tg.loc[live, ["y", "ybar"]] = np.nan
    tg.loc[live, "window_end"] = pd.NaT
    tg["n_t"] = n_t
    tg["live"] = live
    tg["window_first"] = first
    tg["window_last"] = last
    return tg


# --------------------------------------------------------------------------------------------- forecasts
def run_forecasts(
    daily_all: pd.DataFrame,
    targets: pd.DataFrame,
    workers: int = 10,
    *,
    implied: pd.DataFrame | None = None,
    models=FORECAST_MODELS,
) -> pd.DataFrame:
    """The holdout walk-forward of ``pipeline.compute_forecasts`` on live data and live targets.

    Frozen models (``pipeline._forecast_task``), frozen ``combine`` and frozen ``iv_benchmarks`` (1m, on
    ``implied``, default ``update.live_implied()``). Returns FORECAST_COLUMNS + ``split``. ``workers <= 1`` runs
    in-process (tests).
    """
    from volrisk.evaluation.iv import iv_benchmarks
    from volrisk.models.combo import combine

    daily_all = normalise_daily(daily_all)
    tg_all = targets[TARGET_COLUMNS]
    assets = _assets(daily_all)
    tasks = []
    for a in assets:
        d = daily_all[daily_all["asset"] == a].reset_index(drop=True)
        tg = tg_all[tg_all["asset"] == a].reset_index(drop=True)
        tasks += [(a, m, d, tg) for m in models]
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
            parts = list(ex.map(_forecast_task, tasks))
    else:
        parts = [_forecast_task(t) for t in tasks]
    fc = pd.concat(parts, ignore_index=True)
    fc = pd.concat([fc, combine(fc)], ignore_index=True)

    if implied is None:
        from volrisk_live import update

        implied = update.live_implied()
    ev = C.load()["evaluation"]
    iv_parts = []
    for a in assets:
        tg = tg_all[(tg_all["asset"] == a) & (tg_all["horizon"] == "1m")].reset_index(drop=True)
        iv = implied[implied["asset"] == a]
        if len(iv):
            iv_parts.append(iv_benchmarks(tg, iv, cal_min=int(ev["iv_cal_min"]), cal_max=int(ev["iv_cal_max"])))
    if iv_parts:
        fc = pd.concat([fc, *iv_parts], ignore_index=True)
    return fc.merge(tg_all[["asset", "horizon", "origin", "split"]], on=["asset", "horizon", "origin"], how="left")


def _keyed(df: pd.DataFrame, keys: list[str], cols: list[str], name: str) -> pd.DataFrame:
    d = df[[*keys, *cols]].copy()
    d[keys[-1]] = pd.to_datetime(d[keys[-1]]).astype("datetime64[ns]")
    if d.duplicated(keys).any():
        raise ReproductionError(f"{name} contains duplicate {keys} rows")
    return d


def _compare(ref: pd.DataFrame, live: pd.DataFrame, keys: list[str], values: list[str], name: str,
             exact: tuple[str, ...] = ()) -> dict:
    """Every reference row must exist in ``live`` with equal values: rtol 1e-9, atol 0, or ``==`` for the ``exact``
    columns. NaN equals NaN only when the live row exists. Raises ``ReproductionError``; returns statistics."""
    m = _keyed(ref, keys, values, f"reference {name}").merge(
        _keyed(live, keys, values, f"live {name}"), on=keys, how="left", suffixes=("_ref", "_live"),
        indicator=True)
    if m.empty:
        raise ReproductionError(f"no reference {name} rows to compare")
    present = (m["_merge"] == "both").to_numpy()
    ok = present.copy()
    max_abs, max_rel = 0.0, 0.0
    for v in values:
        a, b = m[f"{v}_ref"].to_numpy(dtype=float), m[f"{v}_live"].to_numpy(dtype=float)
        same = (a == b) if v in exact else np.isclose(a, b, rtol=REPRO_RTOL, atol=0.0)
        ok &= same | (np.isnan(a) & np.isnan(b))
        both = np.isfinite(a) & np.isfinite(b)
        if both.any():
            diff = np.abs(a[both] - b[both])
            max_abs = max(max_abs, float(diff.max()))
            max_rel = max(max_rel, float((diff / np.maximum(np.abs(a[both]), np.finfo(float).tiny)).max()))
    if not ok.all():
        bad = m.loc[~ok].drop(columns="_merge")
        by = bad.groupby([k for k in keys if k in ("asset", "model")]).size().to_dict()
        raise ReproductionError(
            f"the live run did not reproduce the {name}: {len(bad)} of {len(m)} rows differ "
            f"({int((~present).sum())} missing in the live run; rtol {REPRO_RTOL:g}, atol 0); by {by}; first: "
            f"{bad.head(3).to_dict('records')}")
    return {"rows": int(len(m)), "max_abs_diff": max_abs, "max_rel_diff": max_rel, "rtol": REPRO_RTOL,
            "atol": 0.0, "passed": True}


def _sealed_names(sha: str) -> list[str]:
    """SEALED.json entries whose hash is ``sha`` (empty for a file written after the seal)."""
    try:
        hashes = json.loads(holdout.SEALED.read_text(encoding="utf-8")).get("hashes") or {}
    except (OSError, ValueError):
        return []
    return sorted(k for k, v in hashes.items() if v == sha)


def read_reference(path: Path) -> tuple[str, pd.DataFrame, dict]:
    """``(relative path, table, {sha256, sealed_as})`` of a stored result file. The bytes are read once, so the
    SHA-256 is of exactly the bytes that were parsed and compared. ``sealed_as`` names the SEALED.json entry with
    that hash, or None (the holdout-run results were written at the opening, after the seal)."""
    path = Path(path)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    names = _sealed_names(sha)
    name = path.relative_to(C.ROOT).as_posix() if path.is_relative_to(C.ROOT) else path.as_posix()
    return name, pd.read_parquet(BytesIO(data)), {"sha256": sha, "sealed_as": names[0] if names else None}


def reference_forecasts() -> list[tuple[str, pd.DataFrame, dict]]:
    """The stored forecast tables with their SHA-256: the one-time holdout run (dev + holdout rows; written at the
    opening, not in SEALED.json) and the dev run (``dev_forecasts`` in SEALED.json)."""
    context.require_opened_holdout()
    return [read_reference(p) for p in (results_dir(True) / "forecasts.parquet", io.FORECASTS)]


def _file_entry(rows: int, meta: dict) -> dict:
    return {"rows": int(rows), "sha256": meta.get("sha256"), "sealed_as": meta.get("sealed_as")}


def reproduction_check(fc: pd.DataFrame, reference: list[tuple] | None = None) -> dict:
    """§3 reproduction proof with statistics: every stored forecast (``asset, horizon, model, origin``) of the dev
    and holdout runs must be in ``fc`` with the same ``n_t`` and ``F`` (rtol 1e-9, atol 0). Raises
    ``ReproductionError`` otherwise.

    ``reference``: ``(name, table)`` or ``(name, table, {sha256, sealed_as})`` items; default
    ``reference_forecasts()``. Returns ``rows`` (distinct stored forecasts checked), ``max_abs_diff`` /
    ``max_rel_diff`` of ``F``, ``files`` = ``{name: {rows, sha256, sealed_as}}`` (the payload pins exactly which
    files were compared) and ``new_rows`` (live forecasts at origins the earlier runs could not make).
    """
    reference = reference_forecasts() if reference is None else reference
    per_file, keys_seen = {}, None
    max_abs = max_rel = 0.0
    for name, df, *meta in reference:
        r = _compare(df, fc, FC_KEYS, ["F", "n_t"], f"forecasts ({name})", exact=("n_t",))
        per_file[name] = _file_entry(r["rows"], meta[0] if meta else {})
        max_abs, max_rel = max(max_abs, r["max_abs_diff"]), max(max_rel, r["max_rel_diff"])
        k = _keyed(df, FC_KEYS, [], name)
        keys_seen = k if keys_seen is None else pd.concat([keys_seen, k]).drop_duplicates()
    n = int(len(keys_seen))
    return {"rows": n, "max_abs_diff": max_abs, "max_rel_diff": max_rel, "rtol": REPRO_RTOL, "atol": 0.0,
            "files": per_file, "new_rows": int(len(fc) - n), "passed": True}


def check_reproduces_holdout(fc: pd.DataFrame) -> int:
    """Every forecast at an origin of the stored dev/holdout run is identical (rtol 1e-9, atol 0); returns the
    number of stored forecasts checked and raises ``ReproductionError`` on any mismatch or missing row."""
    return reproduction_check(fc)["rows"]


# --------------------------------------------------------------------------------------------- risk
def _har_star() -> dict[str, str]:
    return dict(holdout.frozen()["har_star"])


def live_risk_frame(daily_all: pd.DataFrame, fc: pd.DataFrame, har_star: dict[str, str] | None = None,
                    placeholder: float = 0.0) -> pd.DataFrame:
    """Frozen ``risk_frame`` per asset (as ``risk.suite.build_risk``) on the gold rows + one placeholder row for
    the next scheduled session. The placeholder lets the frozen "strictly before ``d``" logic produce the
    next date. Its ``r_cc`` is ignored by every model and is NaN in the output. Adds ``next`` (bool)."""
    from volrisk.risk.var_es import RISK_COLUMNS, risk_frame

    daily_all = normalise_daily(daily_all)
    star = _har_star() if har_star is None else har_star
    fc1d = fc[fc["horizon"] == "1d"]
    frames = []
    for asset in _assets(daily_all):
        if asset not in star:
            continue
        d = daily_all.loc[daily_all["asset"] == asset, ["asset", "session_date", "r_cc"]].reset_index(drop=True)
        nxt = pd.Timestamp(next_session(asset, d["session_date"].iloc[-1]))
        ph = pd.DataFrame({"asset": [asset], "session_date": [nxt], "r_cc": [float(placeholder)]})
        ph["session_date"] = ph["session_date"].astype(d["session_date"].dtype)
        ext = pd.concat([d, ph], ignore_index=True)
        rf = risk_frame(asset, ext, fc1d[fc1d["asset"] == asset], star[asset], start_date=C.dev_eval_start())
        is_next = (pd.to_datetime(rf["date"]) == nxt).to_numpy()
        rf.loc[is_next, "r_cc"] = np.nan
        rf["next"] = is_next
        frames.append(rf)
    if not frames:
        return pd.DataFrame(columns=[*RISK_COLUMNS, "next"])
    return pd.concat(frames, ignore_index=True)


def risk_reproduction_check(risk: pd.DataFrame, reference: pd.DataFrame | tuple | None = None) -> dict:
    """Every VaR/ES row of the holdout run's ``risk.parquet`` (realised dates; written at the opening, not in
    SEALED.json) is reproduced by the live risk frame (``r_cc`` exactly, ``var99/var975/es975/sigma`` to rtol 1e-9,
    atol 0); raises ``ReproductionError``. ``reference``: a table, a ``(name, table[, {sha256, sealed_as}])`` item or
    None (read the file). The result's ``files`` entry pins the compared file by SHA-256."""
    if reference is None:
        context.require_opened_holdout()
        reference = read_reference(results_dir(True) / "risk.parquet")
    name, df, *meta = reference if isinstance(reference, tuple) else ("(given table)", reference)
    out = _compare(df, risk, RISK_KEYS, RISK_VALUES, f"VaR/ES ({name})", exact=("r_cc",))
    out["files"] = {name: _file_entry(out["rows"], meta[0] if meta else {})}
    return out


def next_session_risk(daily_all: pd.DataFrame, fc: pd.DataFrame, har_star: dict[str, str] | None = None, *,
                      check: bool = True, reference: pd.DataFrame | tuple | None = None) -> pd.DataFrame:
    """VaR99, VaR97.5, ES97.5 (positive, % log return) and sigma of the six frozen risk models for the next
    scheduled session of each asset. With ``check`` the realised dates must reproduce the holdout run's risk
    table (raises ``ReproductionError``); the statistics are in ``result.attrs['reproduction']``."""
    risk = live_risk_frame(daily_all, fc, har_star)
    stats = risk_reproduction_check(risk, reference) if check else None
    out = risk[risk["next"]].drop(columns=["next", "r_cc", "split"]).reset_index(drop=True)
    out.attrs["reproduction"] = stats
    return out


# --------------------------------------------------------------------------------------------- one live run
@dataclass
class LiveRun:
    """Everything a live run computes (nothing written)."""

    end: date
    daily_all: pd.DataFrame
    implied: pd.DataFrame
    targets: pd.DataFrame
    fc: pd.DataFrame
    risk: pd.DataFrame  # full live risk frame (realised dates + next session, column ``next``)
    reproduction: dict
    risk_reproduction: dict
    seconds: dict = field(default_factory=dict)


def compute(end: date, daily_all: pd.DataFrame, implied: pd.DataFrame, workers: int = 10, *,
            models=FORECAST_MODELS, har_star: dict[str, str] | None = None,
            reference_fc: list[tuple] | None = None,
            reference_risk: pd.DataFrame | tuple | None = None) -> LiveRun:
    """Targets -> walk-forward -> reproduction proof -> risk -> risk reproduction proof. Writes nothing."""
    context.require_opened_holdout()
    t0 = time.perf_counter()
    daily_all = normalise_daily(daily_all)
    targets = live_targets(daily_all, end)
    t1 = time.perf_counter()
    fc = run_forecasts(daily_all, targets, workers, implied=implied, models=models)
    t2 = time.perf_counter()
    repro = reproduction_check(fc, reference_fc)
    log.info("reproduction: %d stored dev/holdout-run forecasts identical (max |dF| %.3g)", repro["rows"],
             repro["max_abs_diff"])
    risk = live_risk_frame(daily_all, fc, har_star)
    risk_repro = risk_reproduction_check(risk, reference_risk)
    log.info("risk reproduction: %d holdout-run VaR/ES rows identical", risk_repro["rows"])
    t3 = time.perf_counter()
    secs = {"targets": t1 - t0, "walk_forward": t2 - t1, "checks_and_risk": t3 - t2, "total": t3 - t0}
    return LiveRun(end, daily_all, implied, targets, fc, risk, repro, risk_repro, secs)


# --------------------------------------------------------------------------------------------- payload
def latest_forecasts(fc: pd.DataFrame, targets: pd.DataFrame, origins: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Forecast rows at each asset's latest origin, with the scheduled window and ``vol_ann`` (§5)."""
    t = targets[targets["live"]][["asset", "horizon", "origin", "window_first", "window_last"]]
    rows = []
    for asset, origin in origins.items():
        f = fc[(fc["asset"] == asset) & (pd.to_datetime(fc["origin"]) == pd.Timestamp(origin))]
        rows.append(f.merge(t, on=["asset", "horizon", "origin"], how="left"))
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=schema.FORECAST_KEYS)
    if out[["window_first", "window_last"]].isna().any().any():
        raise RuntimeError("a latest-origin forecast has no scheduled window (it is not a live origin)")
    out["vol_ann"] = [vol_ann(F, n, a) for F, n, a in zip(out["F"], out["n_t"], out["asset"], strict=True)]
    order = {m: i for i, m in enumerate(MODEL_ORDER)}
    out = out.assign(_a=out["asset"].map(C.ASSETS.index), _h=out["horizon"].map(C.HORIZONS.index),
                     _m=out["model"].map(lambda m: order.get(m, len(order))))
    return out.sort_values(["_a", "_h", "_m"]).drop(columns=["_a", "_h", "_m"]).reset_index(drop=True)


def _frozen_block() -> dict:
    seal = json.loads(holdout.SEALED.read_text(encoding="utf-8"))
    return {"code_sha": holdout.code_sha(), "seal_ok": holdout.verify_seal() == [],
            "sealed_utc": seal["created_utc"], "holdout_opened_utc": context.require_opened_holdout()["utc"]}


def live_code_sha() -> str:
    """Code hash of ``src/volrisk_live/`` (same rule as the frozen seal's code hash)."""
    return holdout.code_sha(C.ROOT / "src" / "volrisk_live")


def asset_status(daily_all: pd.DataFrame, end: date, recorded_utc: datetime) -> dict[str, dict]:
    """Per asset: last session, next scheduled session (+ open/close UTC), whether the data is current and when the
    payload is recorded relative to that session.

    ``recorded_utc`` is the recording time (``build_payload``: machine clock + ``RECORD_MARGIN``), not the run's
    start. An asset is ``stale`` when its next scheduled session after the last gold session is already complete by
    the live end (missing or invalid data). ``recorded_before_close`` is False when that session has already closed
    at ``recorded_utc``, so its outcome is known (a run late in the UTC day, after the SPX / EURUSD close). In
    both cases the asset gets no forecast (``build_payload``).
    """
    out = {}
    for asset in _assets(daily_all):
        d = daily_all[daily_all["asset"] == asset]
        last = pd.Timestamp(d["session_date"].max()).date()
        nxt = next_session(asset, last)
        o, c = session_bounds(asset, nxt)
        out[asset] = {"last_session": last, "n_sessions": int(len(d)), "rows_sha256": rows_sha256(d),
                      "next_session": nxt, "next_open_utc": o, "next_close_utc": c,
                      "stale": nxt <= end, "recorded_before_open": recorded_utc < o,
                      "recorded_before_close": recorded_utc < c}
    return out


def recording_time(run_utc: datetime) -> tuple[datetime, datetime]:
    """``(computed_utc, timing_utc)``: the machine clock now and the time the timing flags are evaluated at, i.e.
    the later of ``run_utc`` and the clock, plus ``RECORD_MARGIN`` (the ledger line is written right after)."""
    computed = _wall_clock()
    return computed, max(_utc(run_utc), computed) + RECORD_MARGIN


def build_payload(run: LiveRun, run_utc: datetime, consistency: dict | None = None, *,
                  replayed: bool = False) -> dict:
    """Payload ``volrisk-live/1`` (schema.py) for one live run; ``validate_payload`` passes on the result.

    Call it immediately before ``ledger.append``: the per-asset timing flags use the machine clock at this call
    (``recording_time``). Assets that are stale, or whose next session has already closed, get no forecast or
    VaR/ES rows. ``replayed`` marks a payload whose ``run_utc`` was a past time supplied by the caller (test
    sandboxes only; ``forecast`` refuses it for the project ledger).
    """
    run_utc = _utc(run_utc)
    computed, timing = recording_time(run_utc)
    status = asset_status(run.daily_all, run.end, timing)
    current = {a: s for a, s in status.items() if not s["stale"] and s["recorded_before_close"]}
    closed = sorted(a for a, s in status.items() if not s["stale"] and not s["recorded_before_close"])
    origins = {a: pd.Timestamp(s["last_session"]) for a, s in current.items()}
    lf = latest_forecasts(run.fc, run.targets, origins)
    forecasts = [
        {"asset": r.asset, "horizon": r.horizon, "model": r.model, "origin": _iso(r.origin),
         "window_first": _iso(r.window_first), "window_last": _iso(r.window_last), "n_t": int(r.n_t),
         "F": float(r.F), "vol_ann": float(r.vol_ann)}
        for r in lf.itertuples(index=False)
    ]
    nxt = run.risk[run.risk["next"] & run.risk["asset"].isin(list(current))]
    risk = [
        {"asset": r.asset, "model": r.model, "date": _iso(r.date), "sigma": float(r.sigma), "var99": float(r.var99),
         "var975": float(r.var975), "es975": float(r.es975)}
        for r in nxt.itertuples(index=False)
    ]
    implied = []
    iv = run.implied.assign(_o=pd.to_datetime(run.implied["origin"]))
    lo, hi = pd.Timestamp(run.end) - pd.Timedelta(days=IV_MAX_AGE_DAYS), pd.Timestamp(run.end)
    for asset in status:
        g = iv[(iv["asset"] == asset) & (iv["_o"] > lo) & (iv["_o"] <= hi)].sort_values("_o")
        if len(g):
            r = g.iloc[-1]
            implied.append({"asset": asset, "origin": _iso(r["origin"]), "iv": float(r["iv"]),
                            "iv_var_30d": float(r["iv_var_30d"]), "source": str(r["source"])})
    data = {a: {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in s.items()}
            for a, s in status.items()}
    payload = {
        "schema": schema.SCHEMA,
        "run_id": run_utc.strftime("%Y%m%dT%H%M%SZ"),
        "run_utc": run_utc.isoformat(timespec="seconds"),
        "frozen": _frozen_block(),
        "live_code_sha": live_code_sha(),
        "data": data,
        "forecasts": forecasts,
        "risk": risk,
        "implied": implied,
        "checks": {
            "live_end": run.end.isoformat(),
            "computed_utc": computed.isoformat(timespec="seconds"),
            "timing_utc": timing.isoformat(timespec="seconds"),
            "record_margin_s": int(RECORD_MARGIN.total_seconds()),
            "replayed_now": bool(replayed),
            "reproduction": run.reproduction,
            "risk_reproduction": run.risk_reproduction,
            "data_consistency": consistency,
            "stale_assets": sorted(a for a, s in status.items() if s["stale"]),
            "closed_assets": closed,
        },
    }
    schema.validate_payload(payload)
    return payload


# --------------------------------------------------------------------------------------------- tomorrow.md
def _fmt(x, nd: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and not math.isfinite(x)) else f"{x:.{nd}f}"


def _consistency_text(dc: dict) -> str:
    """One line from ``update.check_against_sealed``'s report (row counts compared cell by cell)."""
    rep = dc.get("check_against_sealed") or {}
    parts = [f"{name} {rep[k]['rows_compared']:,} rows" for k, name in
             (("dev", "dev gold"), ("holdout", "holdout gold"), ("implied", "implied vol")) if
             isinstance(rep.get(k), dict) and "rows_compared" in rep[k]]
    if not parts:
        return f"`{json.dumps(rep or dc, default=str)[:300]}`"
    verdict = "identical, cell by cell" if dc.get("ok", rep.get("ok")) else "NOT identical"
    return (f"the live rebuild from the raw files reproduces the sealed tables up to "
            f"{rep.get('frozen_data_end', context.frozen_data_end())}: {', '.join(parts)} {verdict} "
            "(`update.check_against_sealed`).")


def _stamp_text(entry: dict) -> str:
    st = entry.get("stamp")
    if not st:
        return "not stamped in this run (`--no-stamp`); `python -m volrisk_live stamp` stamps it later"
    cals = st.get("calendars_ok") or []
    return f"{st.get('status')}" + (f", submitted to {len(cals)} calendar(s)" if cals else "")


def _window_text(rows: list[dict]) -> str:
    r = rows[0]
    if r["window_first"] == r["window_last"]:
        return r["window_first"]
    return f"{r['window_first']} → {r['window_last']}, {r['n_t']} sessions"


def _files_text(files: dict | None) -> str:
    """The compared result files with their SHA-256 and whether SEALED.json covers them."""
    parts = []
    for name, f in (files or {}).items():
        f = f if isinstance(f, dict) else {"rows": f}
        sha = f.get("sha256")
        pin = f"sha256 `{sha}`" if sha else "in-memory table, no file hash"
        how = (f"sealed in SEALED.json as `{f['sealed_as']}`" if f.get("sealed_as") else
               "not in SEALED.json, pinned by the hash recorded in this payload")
        parts.append(f"`{name}` ({int(f.get('rows') or 0):,} rows; {pin}; {how})")
    return "; ".join(parts) if parts else "—"


def _no_forecast_text(d: dict, timing: str) -> str:
    if d.get("stale"):
        return (f"No forecast: last session {d['last_session']}; the next scheduled session {d['next_session']} is "
                "already past the data cutoff (its data are missing or invalid), so a forecast would target a past "
                "session.")
    if d.get("recorded_before_close") is False:
        close = datetime.fromisoformat(d["next_close_utc"])
        return (f"No forecast: the next session **{d['next_session']}** ({d['next_open_utc']} → "
                f"{d['next_close_utc']}) had already closed when this payload was recorded ({timing}), so its "
                f"outcome was known. Its data enter the live tables after the UTC day ends. A run recorded before "
                f"{close:%H:%M} UTC on a session day records this asset.")
    return "No forecast in this run."


def render_tomorrow(payload: dict, entry: dict | None = None) -> str:
    """Human-readable view of one payload (reports/live/tomorrow.md)."""
    ck = payload["checks"]
    rep, rrep = ck["reproduction"], ck["risk_reproduction"]
    fz = payload["frozen"]
    run_id = payload["run_id"]
    timing_iso = ck.get("timing_utc") or payload["run_utc"]
    timing = datetime.fromisoformat(timing_iso)
    margin = ck.get("record_margin_s", int(RECORD_MARGIN.total_seconds()))
    L = [
        "# Volatility & risk forecasts — next session, week and month",
        "",
        f"Run `{run_id}` (started {payload['run_utc']}, payload built {ck.get('computed_utc', '—')}) · data cutoff "
        f"**{ck['live_end']}** (last complete UTC day) · frozen code `{fz['code_sha'][:16]}…` (seal "
        f"{'OK' if fz['seal_ok'] else 'BROKEN'}, sealed {fz['sealed_utc']}, holdout opened once "
        f"{fz['holdout_opened_utc']}) · live code `{payload['live_code_sha'][:16]}…`",
        "",
    ]
    if ck.get("replayed_now"):
        L += ["**Replayed run:** the run time was supplied by the caller and lies before the machine clock (a test "
              "sandbox). This payload is not a forward-test record.", ""]
    L += [
        "These forecasts come from the **frozen** models. Their specification, features and hyperparameters were "
        "sealed before the holdout. They use data up to the cutoff only. Every forecast and VaR/ES row below is "
        "written to the append-only, timestamped ledger before the first session of its target window closes. "
        f"Timing is checked at {timing_iso}: the machine clock when the payload was built, plus {margin} s for the "
        "ledger write. An asset whose next session had already closed by then gets no row, because its outcome was "
        "known. Each asset section says whether that session had already opened. If it had, part of it had traded, "
        "but none of its data is in the models' input. Scores appear in `reports/live/forward_test.md` once each "
        "window has passed.",
        "",
        "## Checks behind this run",
        "",
        f"- **Reproduction:** {rep['rows']:,} forecasts at origins of the earlier dev and holdout runs were "
        f"recomputed by this live run and are identical (max |ΔF| = {rep['max_abs_diff']:.3g}, rtol "
        f"{rep['rtol']:g}, atol 0). The run would have aborted otherwise. Compared with: "
        f"{_files_text(rep.get('files'))}.",
        f"- **Risk reproduction:** {rrep['rows']:,} VaR/ES values of the holdout run recomputed, identical "
        f"(max |Δ| = {rrep['max_abs_diff']:.3g}). Compared with: {_files_text(rrep.get('files'))}.",
    ]
    dc = ck.get("data_consistency")
    if dc:
        L.append(f"- **Live data = sealed data:** {_consistency_text(dc)}")
    if ck.get("stale_assets"):
        L.append(f"- **No forecast** for {', '.join(ck['stale_assets'])}: the data does not yet cover its next "
                 "scheduled session after the last one, so a forecast would target a past session.")
    if ck.get("closed_assets"):
        L.append(f"- **No forecast** for {', '.join(ck['closed_assets'])}: the next session had already closed "
                 f"when the payload was recorded ({timing_iso}), so its outcome was known. No forecast or VaR/ES "
                 "row is recorded for it.")
    L.append("")

    fcs = payload["forecasts"]
    risks = payload["risk"]
    ivs = {r["asset"]: r for r in payload["implied"]}
    for asset, d in payload["data"].items():
        L += [f"## {asset}", ""]
        rows = [r for r in fcs if r["asset"] == asset]
        if not rows:
            L += [_no_forecast_text(d, timing_iso), ""]
            continue
        lead = (datetime.fromisoformat(d["next_open_utc"]) - timing).total_seconds() / 3600
        left = (datetime.fromisoformat(d["next_close_utc"]) - timing).total_seconds() / 3600
        when = (f"opens {lead:.1f} h after this payload was recorded" if lead >= 0 else
                f"opened {-lead:.1f} h before this payload was recorded and closes {left:.1f} h after it; the "
                f"models use data up to {d['last_session']} only")
        L += [f"Last session {d['last_session']} ({d['n_sessions']:,} sessions of history). Next session "
              f"**{d['next_session']}** ({d['next_open_utc']} → {d['next_close_utc']}; {when}).", ""]
        by_h = {h: [r for r in rows if r["horizon"] == h] for h in C.HORIZONS}
        head = [f"{HORIZON_TEXT[h]} ({_window_text(by_h[h])})" for h in C.HORIZONS if by_h[h]]
        hs = [h for h in C.HORIZONS if by_h[h]]
        L += [f"Annualised volatility forecast, % (√(F / n_t · {ann_factor(asset)})):", "",
              "| model | " + " | ".join(head) + " |", "|---|" + "---:|" * len(hs)]
        models = [m for m in MODEL_ORDER if any(r["model"] == m for r in rows)]
        for m in models:
            cells = []
            for h in hs:
                v = next((r["vol_ann"] for r in by_h[h] if r["model"] == m), None)
                cells.append(_fmt(v))
            if m == "COMBO":
                members = " + ".join(C.load()["models"]["combo_members"])
                L.append(f"| **COMBO** ({members}, primary) | " + " | ".join(f"**{c}**" for c in cells) + " |")
            else:
                L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
        iv = ivs.get(asset)
        if iv:
            L += [f"Latest implied vol: **{iv['source']} {iv['iv']:.2f}** on {iv['origin']} (30-day variance "
                  f"{iv['iv_var_30d']:.3f} %²). `IV` / `IV-cal` above are the frozen implied-vol benchmarks "
                  "(1 month only).", ""]
        else:
            why = " (EVZ is used only up to 2023, SPEC §2.4)" if C.asset(asset).iv == "EVZ" else ""
            L += [f"No {C.asset(asset).iv} value in the last {IV_MAX_AGE_DAYS} days{why}, so there is no "
                  "`IV` benchmark here.", ""]
        rk = [r for r in risks if r["asset"] == asset]
        if rk:
            L += [f"Next-session 1-day VaR / ES on {rk[0]['date']} (loss of a long position, % log return):", "",
                  "| risk model | σ | VaR 99% | VaR 97.5% | ES 97.5% |", "|---|---:|---:|---:|---:|"]
            for r in rk:
                cells = [_fmt(r["sigma"], 2), _fmt(r["var99"], 2), _fmt(r["var975"], 2), _fmt(r["es975"], 2)]
                if r["model"] == "COMBO+FHS":
                    L.append("| **COMBO+FHS** (primary) | " + " | ".join(f"**{c}**" for c in cells) + " |")
                else:
                    L.append(f"| {r['model']} | " + " | ".join(cells) + " |")
            L.append("")

    run_json = (paths.RUNS / f"{run_id}.json").relative_to(C.ROOT).as_posix()
    L += ["## How to verify", ""]
    if entry:
        L.append(f"- Ledger entry #{entry.get('seq')} in `forecasts/ledger.jsonl`: payload sha256 "
                 f"`{entry.get('payload_sha256')}`, entry sha256 `{entry.get('entry_sha256')}`, previous entry "
                 f"`{entry.get('prev_entry_sha256')}`. OpenTimestamps: {_stamp_text(entry)}.")
    L += [
        f"- The payload is `{run_json}`. Its SHA-256 must equal the ledger's `payload_sha256`. Every entry hashes "
        "the previous one, so an edited, removed or reordered forecast breaks the chain (`verify_chain()`).",
        f"- `{run_json}.ots` is the OpenTimestamps proof. Once a Bitcoin block confirms it (a few hours), "
        "`uv run python -m volrisk_live stamp` upgrades it. The proof then shows that the file existed no later "
        "than that block's time, independently of this machine's clock; `verify` compares that time with the "
        "outcomes.",
        "- The holdout-run result files under `data/results/holdout/` were written at the one-time opening, after "
        "the seal, so SEALED.json does not cover them. Their SHA-256 is recorded above and in "
        "`checks.reproduction.files` / `checks.risk_reproduction.files` of every payload, so a later change to "
        "them shows against the ledger.",
        "- `uv run python -m volrisk_live verify` re-runs every check: frozen code unchanged, holdout opened once, "
        "raw data vs the sources' own checksums and fresh downloads, independent price sources, reproduction, "
        "negative controls (cheating models are caught) and the ledger chain. See `reports/live/verification.md`.",
        "- `uv run python -m volrisk_live score` scores each forecast once its window has passed. See "
        "`reports/live/forward_test.md`.",
        "",
    ]
    return "\n".join(L)


def write_tomorrow(payload: dict, entry: dict | None = None, path: Path | None = None) -> Path:
    path = paths.TOMORROW_MD if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(render_tomorrow(payload, entry), encoding="utf-8")
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------------------------- entry point
def _state_end(state: dict) -> str | None:
    for k in ("end", "live_end", "end_date"):
        if state.get(k):
            return str(state[k])[:10]
    return None


def require_update(end: date) -> dict:
    """``update`` must have run for ``end``: ``data/live/state.json`` exists and records that live end."""
    if not paths.LIVE_STATE.exists():
        raise RuntimeError(f"{paths.LIVE_STATE} is missing — run `python -m volrisk_live update` first")
    state = json.loads(paths.LIVE_STATE.read_text(encoding="utf-8"))
    recorded = _state_end(state)
    if recorded != end.isoformat():
        raise RuntimeError(f"the live data was built for {recorded}, this run needs {end} — run update first")
    return state


def write_scratch(run: LiveRun) -> None:
    """Latest live walk-forward outputs under data/live/forecasts/ (scratch, overwritten by every run)."""
    out = paths.LIVE_RESULTS
    io.write_parquet(run.fc, out / "forecasts.parquet")
    io.write_parquet(run.targets, out / "targets.parquet")
    io.write_parquet(run.risk, out / "risk.parquet")


def _project_ledger(ledger) -> bool:
    """True when the ledger writes the project's forecasts/ledger.jsonl (assumed when the module cannot tell)."""
    check = getattr(ledger, "is_project_ledger", None)
    return True if check is None else bool(check())


def forecast(now_utc: datetime | None = None, workers: int = 10, stamp: bool = True) -> str:
    """One live forecast run (§3): checks -> compute -> payload -> ledger (+ OTS stamp) -> tomorrow.md.

    Refuses to run on a broken seal, before the holdout opening, without a current update, or on live data that
    differs from the sealed data. ``now_utc`` (default: the machine clock) sets ``run_utc`` / ``run_id`` and the
    data cutoff. It may not lead the machine clock by more than ``FUTURE_SLACK``. On the project ledger it may not
    lag it by more than that either (no backdated run). A test sandbox may replay a past time, and the payload is
    then marked ``checks.replayed_now``. The payload is built right before ``ledger.append``, so its timing flags
    use the recording time, not the start. Returns the ``run_id``.
    """
    from volrisk_live import ledger, update

    started = _wall_clock()
    run_utc = started if now_utc is None else _utc(now_utc)
    if run_utc > started + FUTURE_SLACK:
        raise ValueError(f"now_utc {run_utc.isoformat()} is in the future — a payload is never post-dated")
    replayed = run_utc < started - FUTURE_SLACK
    if replayed and _project_ledger(ledger):
        raise ValueError(f"now_utc {run_utc.isoformat()} is more than {FUTURE_SLACK} before the machine clock "
                         f"{started.isoformat()}: the project ledger records a run only when it is made (a past "
                         "now_utc is for test sandboxes)")
    bad = holdout.verify_seal()
    if bad:
        raise holdout.SealError(f"frozen artefacts changed since the seal: {bad} — no live forecast")
    context.require_opened_holdout()
    end = update.live_end_date(run_utc)
    state = require_update(end)
    check = update.check_against_sealed()  # raises LiveDataMismatch on drifted data
    consistency = {"ok": check.get("ok") is True, "check_against_sealed": check,
                   "state": {k: state.get(k) for k in STATE_KEYS if k in state}}
    run = compute(end, update.live_daily(), update.live_implied(), workers)
    log.info("live walk-forward for %s in %.0fs", end, run.seconds["total"])
    write_scratch(run)  # before the payload: nothing slow between the timing flags and the ledger write
    payload = build_payload(run, run_utc, consistency, replayed=replayed)
    entry = ledger.append(payload, stamp=stamp)
    write_tomorrow(payload, entry)
    ck = payload["checks"]
    log.info("forecast run %s recorded (ledger seq %s, timing %s); no forecast for stale %s / closed %s",
             payload["run_id"], entry.get("seq"), ck["timing_utc"], ck["stale_assets"] or "none",
             ck["closed_assets"] or "none")
    return payload["run_id"]
