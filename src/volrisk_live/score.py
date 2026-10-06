"""Forward-test scoring of the ledger forecasts and the forward-test report (docs/LIVE_SPEC.md §6).

Every payload in the forecast ledger (``volrisk_live.ledger``, §4) is scored once its outcome exists in the live
gold table (``update.live_daily()``, §2):

- **Forecast rows** ``(asset, horizon, model, origin, n_t, F)``. 1d: the window is the first session after the
  origin in the gold table (the frozen ``targets.window_bounds`` rule); it is complete once that session exists.
  1w / 1m: the calendar window ``(origin, origin + days]`` is complete only when the live data end date
  ``>= origin + days`` **and** the asset's own data reach the ex-ante last scheduled session of the window (so a
  failed download of one source never scores a window with a missing session). ``y = Σ tv`` over the realised
  sessions; when the realised count differs from the ex-ante ``n_t`` (an invalid session dropped), the
  per-session means ``ybar = y / n_real`` and ``Fbar = F / n_t`` are compared. QLIKE is the frozen
  ``volrisk.evaluation.losses.qlike`` (SPEC §8).
- **Risk rows** ``(asset, model, date, var99, var975, es975)``: the realised ``r_cc`` of the first gold session
  after the run's last session of the asset (= ``date`` unless that session was invalid; the frozen ``var_es``
  convention for 1d forecasts) and the breach flags ``1{-r_cc > VaR}`` of the frozen ``backtests.hits``.

Both outputs are append-only CSVs: ``forecasts/scores.csv`` keyed by ``(run_id, asset, horizon, model, origin)``
and ``forecasts/risk_scores.csv`` keyed by ``(run_id, asset, model, date)``. A key is written once; later runs
never rewrite it, whatever the data then say. Forecasts whose window cannot be scored (no valid session, a
non-positive target or forecast) get one row with a non-``ok`` status, so every passed window is accounted for.

Integrity rules (a score is evidence only if nobody could have chosen it after seeing the outcome):

- **Verified ledger prefix only** (:func:`ledger_payloads`). Entries at or after the first broken link reported by
  ``ledger.verify_chain()`` are never scored, nor is a payload that does not match its entry's SHA-256.
- **Checked live data only** (:func:`verified_live_daily`). By default the outcomes come from
  ``update.live_daily()`` only when ``data/live/state.json`` records a passing ``check_against_sealed`` and the
  live gold files still have the SHA-256 recorded with it; otherwise :class:`LiveDataUnverified` and nothing is
  scored. Before a payload's rows are scored, the asset's gold rows up to the run's last session must still hash
  to the payload's ``data[asset].rows_sha256`` (the data the forecast saw); if not, the rows get the status
  ``history_changed`` and are left out of every table.
- **Recorded before the outcome.** ``recorded_utc`` = the latest of ``run_utc``, the payload's own wall clock
  ``checks.computed_utc`` and ``checks.timing_utc`` (:func:`recorded_utc`; ``run_utc`` alone can be set into the
  past with ``--now``). ``before_close``: recorded
  before the close of the window's first scheduled session (frozen ``sessions.session_schedule``); only such rows
  enter the tables, the others are counted separately. ``before_open``: recorded before that session opened.
  The independent bound on the recording time is the payload's Bitcoin timestamp, checked by ``verify`` (§7.8).
- **Every stored score is re-derived on every run** (:func:`audit_scores`) from its ledger payload and the live
  data. A row that does not re-derive exactly (an edited score file, a forged or repaired payload, data revised
  after scoring) is left out of every table and listed in the report, and ``score_all`` then raises
  :class:`ScoreIntegrityError` (non-zero exit) after writing the scores and the report.

``report_forward`` writes ``reports/live/forward_test.md``: running QLIKE ratio vs HAR per asset/horizon/model on
the forecasts scored so far (frozen ``leaderboard``, common origins; only rows that re-derive, recorded ex ante
and before their first target session closed, per (asset, horizon, model, origin) the earliest run), scored
counts, VaR breaches vs expected, ledger-chain, live-data and timestamp status.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import logging
import math
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from volrisk import config as C
from volrisk import holdout
from volrisk import palette as P
from volrisk.evaluation.iv import IV_MODELS
from volrisk.evaluation.leaderboard import leaderboard
from volrisk.evaluation.losses import mse, qlike
from volrisk.hypotheses import fmt_num, fmt_p
from volrisk.report import bold, esc, md_table
from volrisk.risk.backtests import hits, kupiec
from volrisk.risk.var_es import ALPHA_99, ALPHA_975
from volrisk.sessions import session_schedule
from volrisk_live import paths, schema

log = logging.getLogger(__name__)

REF = "HAR"
PRIMARY_MODEL = "COMBO"
PRIMARY_RISK = "COMBO+FHS"
OK = "ok"
HISTORY_CHANGED = "history_changed"

SCORE_KEY = ["run_id", "asset", "horizon", "model", "origin"]
RISK_KEY = ["run_id", "asset", "model", "date"]
SCORE_COLUMNS = [
    *SCORE_KEY, "run_utc", "recorded_utc", "ex_ante", "before_open", "before_close", "first_close_utc",
    "window_first", "window_last", "n_t", "F", "real_first", "real_last", "n_real", "y", "ybar", "Fbar",
    "per_session", "qlike", "status", "data_end",
]
RISK_SCORE_COLUMNS = [
    *RISK_KEY, "run_utc", "recorded_utc", "before_close", "first_close_utc", "session_date", "sigma", "var99",
    "var975", "es975", "r_cc", "loss", "breach99", "breach975", "status", "data_end",
]
_NOT_REDERIVED = {"data_end"}  # when a row was scored, not what it says
_PENDING_COLUMNS = ["kind", "run_id", "asset", "horizon", "model", "due"]
GOLD_FILES = (("live_dev_gold", "LIVE_DAILY_DEV"), ("live_holdout_gold", "LIVE_DAILY_HOLDOUT"))  # state.json names


class LiveDataUnverified(RuntimeError):
    """The live gold on disk is not the data the last passing consistency check (``update``, §2) was run on."""


class ScoreIntegrityError(RuntimeError):
    """Stored scores that do not re-derive, or ledger entries that could not be scored; ``summary`` is attached."""

    def __init__(self, message: str, summary: dict):
        super().__init__(message)
        self.summary = summary


# ============================================================================================= helpers
class _Series(NamedTuple):
    """Gold sessions of one asset up to the live end: strictly increasing days with ``tv`` and ``r_cc``."""

    dates: np.ndarray  # datetime64[D]
    tv: np.ndarray
    r_cc: np.ndarray


def _day(x) -> np.datetime64 | None:
    """Calendar day of a date-like value (ISO string, date, Timestamp); None when missing or unparsable."""
    if isinstance(x, str):
        return _day_of_str(x)
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return None
    try:
        ts = pd.Timestamp(x)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(ts) else np.datetime64(ts.date(), "D")


@functools.lru_cache(maxsize=1 << 16)
def _day_of_str(x: str) -> np.datetime64 | None:  # the audit parses the same few hundred dates many times
    try:
        ts = pd.Timestamp(x)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(ts) else np.datetime64(ts.date(), "D")


def _iso(d) -> str:
    """``YYYY-MM-DD`` of a day (np.datetime64 or anything ``_day`` parses); '' when missing."""
    if d is not None and not isinstance(d, np.datetime64):
        d = _day(d)
    return "" if d is None or np.isnat(d) else _iso_of_day(d.astype("datetime64[D]"))


@functools.lru_cache(maxsize=1 << 16)
def _iso_of_day(d: np.datetime64) -> str:
    return str(d)


def _float(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def _int(x) -> int | None:
    v = _float(x)
    return int(v) if math.isfinite(v) and v == int(v) else None


def _utc(x) -> datetime | None:
    """Aware UTC time of an ISO string ('Z' accepted, naive = UTC as in the ledger); None if missing/unparsable."""
    return _utc_of_str(x) if isinstance(x, str) else None


@functools.lru_cache(maxsize=1 << 14)
def _utc_of_str(x: str) -> datetime | None:
    if not x.strip():
        return None
    try:
        t = datetime.fromisoformat(x.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


@functools.lru_cache(maxsize=1 << 14)
def _utc_iso(t: datetime | None) -> str:
    return "" if t is None else t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _max_day(daily: pd.DataFrame | None) -> np.datetime64 | None:
    if daily is None or not len(daily):
        return None
    m = pd.to_datetime(daily["session_date"]).max()
    return None if pd.isna(m) else _day(m)


def _by_asset(daily: pd.DataFrame, end: np.datetime64) -> dict[str, _Series]:
    days = pd.to_datetime(daily["session_date"]).to_numpy().astype("datetime64[D]")
    assets = daily["asset"].astype(str).to_numpy()
    tv = daily["tv"].to_numpy(dtype=float)
    r_cc = daily["r_cc"].to_numpy(dtype=float)
    keep = days <= end
    out: dict[str, _Series] = {}
    for a in dict.fromkeys(assets[keep]):
        m = np.flatnonzero(keep & (assets == a))
        m = m[np.argsort(days[m], kind="stable")]
        d = days[m]
        if len(d) > 1 and not (np.diff(d.astype("int64")) > 0).all():
            raise ValueError(f"live daily has duplicate sessions for {a}")
        out[str(a)] = _Series(d, tv[m], r_cc[m])
    return out


def _last_session(payload: dict, asset: str) -> np.datetime64 | None:
    """The run's last session of ``asset`` (the newest data the run used); None if the payload lacks it."""
    d = (payload.get("data") or {}).get(asset) or {}
    return _day(d.get("last_session"))


def recorded_utc(payload: dict) -> datetime | None:
    """When the payload was written by its own clocks: the latest of ``run_utc``, ``checks.computed_utc`` (the wall
    clock ``forecast.build_payload`` records) and ``checks.timing_utc`` if present (its timing-flag time, with the
    ledger-write margin) — as ``cli.recorded_utc``. None when ``run_utc`` or ``computed_utc`` is missing, so such a
    payload is never 'before' anything."""
    checks = payload.get("checks") or {}
    run, computed = _utc(payload.get("run_utc")), _utc(checks.get("computed_utc"))
    if run is None or computed is None:
        return None
    timing = _utc(checks.get("timing_utc"))
    return max(run, computed) if timing is None else max(run, computed, timing)


def _module(name: str):
    """Sibling live module, imported lazily (``ledger``, ``ots``, ``update`` are built alongside this one)."""
    return importlib.import_module(f"volrisk_live.{name}")


class _Calendar:
    """(open_utc, close_utc) of scheduled sessions from the frozen ``sessions.session_schedule``, cached."""

    def __init__(self):
        self._cache: dict[tuple[str, np.datetime64], tuple[datetime, datetime] | None] = {}

    def bounds(self, asset: str, day: np.datetime64 | None) -> tuple[datetime, datetime] | None:
        if day is None:
            return None
        k = (asset, day)
        if k not in self._cache:
            d = day.astype(object)  # datetime.date
            try:
                s = session_schedule(asset, d, d)
            except (KeyError, ValueError) as e:  # not an asset of the frozen config
                log.warning("no session calendar for %s %s: %s", asset, d, e)
                s = None
            self._cache[k] = (s["open_utc"][0], s["close_utc"][0]) if s is not None and s.height == 1 else None
        return self._cache[k]


def _timing(rec_utc: datetime | None, bounds: tuple[datetime, datetime] | None) -> dict:
    """``before_open`` / ``before_close`` of a recording time vs the target session; unknown -> not before."""
    if bounds is None:
        return {"before_open": None, "before_close": False, "first_close_utc": ""}
    o, c = bounds
    return {"before_open": None if rec_utc is None else bool(rec_utc < o),
            "before_close": rec_utc is not None and bool(rec_utc < c),
            "first_close_utc": _utc_iso(c)}


class _History:
    """SHA-256 of an asset's gold rows up to a day, exactly as ``forecast.rows_sha256`` (the payload's
    ``data[asset].rows_sha256``): CSV of the rows sorted by ``session_date`` (``YYYY-MM-DD``), floats ``%.17g``,
    NaN empty, LF, no index. Cells are formatted one by one, so the text of the rows up to a day is a prefix of
    the asset's full text: one CSV text per asset, one hash per cut."""

    def __init__(self, daily: pd.DataFrame):
        self._daily = daily
        self._text: dict[str, tuple[bytes, np.ndarray, np.ndarray]] = {}
        self._sha: dict[tuple[str, int], str] = {}

    def sha(self, asset: str, last: np.datetime64) -> str:
        if asset not in self._text:
            d = self._daily[self._daily["asset"].astype(str) == asset].sort_values("session_date", kind="stable")
            days = pd.to_datetime(d["session_date"]).to_numpy().astype("datetime64[D]")
            d = d.assign(session_date=pd.to_datetime(d["session_date"]).dt.strftime("%Y-%m-%d"))
            text = d.to_csv(index=False, lineterminator="\n", float_format="%.17g", na_rep="").encode("utf-8")
            ends = np.flatnonzero(np.frombuffer(text, dtype=np.uint8) == ord("\n"))  # ends[0]: the header
            if len(ends) != len(d) + 1:
                raise ValueError(f"gold rows of {asset} contain line breaks; the history hash is undefined")
            self._text[asset] = (text, days, ends)
        text, days, ends = self._text[asset]
        k = int(np.searchsorted(days, last, side="right"))  # rows on or before ``last``
        if (asset, k) not in self._sha:
            self._sha[(asset, k)] = hashlib.sha256(text[: ends[k] + 1]).hexdigest()
        return self._sha[(asset, k)]


class _Context:
    """What every score is derived from: the gold sessions up to ``end``, the history hashes and the calendar."""

    def __init__(self, daily: pd.DataFrame, end: np.datetime64 | None, check_history: bool = True,
                 history: _History | None = None, calendar: _Calendar | None = None):
        self.daily, self.end = daily, end
        self.series = _by_asset(daily, end) if end is not None else {}
        self.history = (history or _History(daily)) if check_history else None
        self.calendar = calendar or _Calendar()
        self._history_ok: dict[tuple[str, str], bool] = {}

    def at(self, end: np.datetime64 | None) -> _Context:
        """The same data with another end date (history hashes and calendar shared)."""
        if end is not None and self.end is not None and end == self.end:
            return self
        return _Context(self.daily, end, self.history is not None, self.history, self.calendar)

    def history_ok(self, payload: dict, asset: str) -> bool:
        """The gold rows of ``asset`` up to the run's last session still hash to the payload's ``rows_sha256``."""
        if self.history is None:
            return True
        k = (str(payload["run_id"]), asset)
        if k not in self._history_ok:
            d = (payload.get("data") or {}).get(asset) or {}
            last, want = _day(d.get("last_session")), d.get("rows_sha256")
            self._history_ok[k] = last is not None and isinstance(want, str) and self.history.sha(asset, last) == want
        return self._history_ok[k]


# ============================================================================================= inputs
def _chain_ok(chain) -> bool:
    if chain is None or chain is True:
        return True
    if isinstance(chain, dict):
        return chain.get("ok") is True or (chain.get("ok") is None and chain.get("first_bad") is None)
    return False


def _chain_cut(chain, n: int) -> tuple[int | None, str | None]:
    """Index of the first ledger entry outside the verified chain (None: all verified) and the reason."""
    if _chain_ok(chain):
        return None, None
    if isinstance(chain, dict):
        fb, why = chain.get("first_bad"), str(chain.get("reason") or "chain does not verify")
        if fb is None:
            return 0, why
        return (int(fb), why) if int(fb) < n else (None, None)  # first_bad == n: a payload file without entry
    return 0, "chain does not verify"


def _ledger_state() -> tuple[list[dict], list[dict], list[str], dict | None]:
    """``(entries, payloads, problems, chain)``: the verified ledger prefix (see :func:`ledger_payloads`)."""
    ledger = _module("ledger")
    chain = ledger.verify_chain()
    try:
        entries = [dict(e) for e in ledger.entries()]
    except (RuntimeError, OSError, ValueError) as err:  # LedgerError: an unparsable line
        log.error("ledger not readable, nothing is scored: %s", err)
        return [], [], [f"ledger not readable, nothing scored: {err}"], chain
    cut, why = _chain_cut(chain, len(entries))
    kept, payloads, problems = [], [], []
    for i, e in enumerate(entries):
        rid = e.get("run_id", "?")
        if cut is not None and i >= cut:
            problems.append(f"{rid}: ledger entry {i} is at or after the first broken link ({why})")
            continue
        kept.append(e)
        try:
            payloads.append(ledger.load_payload(e))
        except (RuntimeError, OSError, ValueError, KeyError) as err:
            problems.append(f"{rid}: {err}")
    for p in problems:
        log.error("ledger entry not scored: %s", p)
    return kept, payloads, problems, chain


def ledger_payloads() -> tuple[list[dict], list[dict], list[str]]:
    """``(entries, payloads, problems)`` of the verified part of the forecast ledger, in ledger order.

    Only entries before the first broken link of ``ledger.verify_chain()`` are used: an entry at or after it (an
    edited, removed, inserted or reordered entry, or a payload rewritten together with its recorded hash) is never
    scored, nor is a payload that is missing or does not match its entry's SHA-256 (``ledger.load_payload``).
    Every entry left out is listed in ``problems``.
    """
    entries, payloads, problems, _ = _ledger_state()
    return entries, payloads, problems


def _valid_payloads(payloads: Iterable[dict]) -> tuple[list[dict], int]:
    """Payloads that follow ``volrisk-live/1``, sorted by run_id (UTC, so chronological); invalid ones skipped."""
    ok, bad = [], 0
    for p in payloads:
        try:
            schema.validate_payload(p)
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            bad += 1
            log.warning("payload %s skipped: %s", p.get("run_id", "?") if isinstance(p, dict) else "?", e)
            continue
        ok.append(p)
    return sorted(ok, key=lambda p: str(p["run_id"])), bad


def verified_live_daily() -> tuple[pd.DataFrame, dict]:
    """``update.live_daily()``, only if it is the data the last passing consistency check saw (LIVE_SPEC §2).

    Requires ``data/live/state.json`` (``update.read_state``) with ``check.ok`` true and, for each live gold file,
    its current SHA-256 equal to the one recorded with that check (before and after loading); the rows are
    restricted to the checked assets. Raises :class:`LiveDataUnverified` otherwise: a failed or skipped check, a
    rebuild after it, an edited file. Returns ``(daily, evidence)``.
    """
    upd = _module("update")
    state = upd.read_state()
    if not state:
        raise LiveDataUnverified(f"{paths.LIVE_STATE} is missing — run `python -m volrisk_live update` first")
    check = state.get("check")
    if not isinstance(check, dict) or check.get("ok") is not True:
        why = ("was skipped" if check is None else
               f"failed: {check.get('error') or check.get('problems')}" if isinstance(check, dict) else "is unreadable")
        raise LiveDataUnverified(f"the consistency check of the last update {why} — nothing is scored on "
                                 "unverified live data")
    files = state.get("files") or {}

    def hashes() -> dict[str, str]:
        out = {}
        for name, attr in GOLD_FILES:
            path = getattr(paths, attr)
            want, have = (files.get(name) or {}).get("sha256"), holdout.file_sha(path) if path.exists() else None
            if not want or have != want:
                raise LiveDataUnverified(f"{path} {'is missing' if have is None else 'changed'} since the checked "
                                         f"update (sha256 {have} != recorded {want}) — run update again")
            out[name] = have
        return out

    before = hashes()
    daily = upd.live_daily()
    if hashes() != before:  # pragma: no cover - a rebuild while loading
        raise LiveDataUnverified("the live gold changed while it was being read")
    assets = [str(a) for a in (check.get("assets") or state.get("assets") or [])]
    if assets:
        daily = daily[daily["asset"].astype(str).isin(assets)].reset_index(drop=True)
    return daily, {"end": state.get("end"), "checked_utc": state.get("finished_utc"), "assets": assets,
                   "sha256": before}


# ============================================================================================= scoring
def _window(s: _Series | None, origin: np.datetime64, horizon: str, end: np.datetime64,
            window_last: np.datetime64 | None) -> tuple[int, int] | None:
    """Row bounds ``[lo, hi)`` of the realised window of one forecast, or None while it is incomplete."""
    if s is None:
        return None
    d = s.dates
    lo = int(np.searchsorted(d, origin, side="right"))
    if horizon == "1d":  # the first session after the origin (frozen targets rule)
        return (lo, lo + 1) if lo < len(d) else None
    stop = origin + np.timedelta64(C.horizon_days(horizon), "D")
    if end < stop:
        return None
    if window_last is not None and (len(d) == 0 or d[-1] < window_last):
        return None  # this asset's data do not reach the window's last scheduled session yet
    return lo, int(np.searchsorted(d, stop, side="right"))


def score_forecast_row(row: dict, s: _Series | None, end: np.datetime64,
                       last_session: np.datetime64 | None = None) -> dict | None:
    """Outcome part of one payload forecast row's score (no run, timing or history fields), None while open."""
    horizon = str(row.get("horizon"))
    origin = _day(row.get("origin"))
    if origin is None or horizon not in C.HORIZONS:
        log.warning("forecast row with origin %r / horizon %r cannot be scored", row.get("origin"), horizon)
        return None
    bounds = _window(s, origin, horizon, end, _day(row.get("window_last")))
    if bounds is None:
        return None
    lo, hi = bounds
    n_t, F = _int(row.get("n_t")), _float(row.get("F"))
    n_real = hi - lo
    tv = s.tv[lo:hi]
    y = float(tv.sum()) if n_real else math.nan
    rec = {
        "asset": str(row["asset"]),
        "horizon": horizon,
        "model": str(row["model"]),
        "origin": _iso(origin),
        "ex_ante": bool(last_session is None or origin >= last_session),
        "window_first": _iso(_day(row.get("window_first"))),
        "window_last": _iso(_day(row.get("window_last"))),
        "n_t": n_t,
        "F": F,
        "real_first": _iso(s.dates[lo]) if n_real else "",
        "real_last": _iso(s.dates[hi - 1]) if n_real else "",
        "n_real": n_real,
        "y": y,
        "ybar": y / n_real if n_real else math.nan,
        "Fbar": F / n_t if n_t else math.nan,
        "per_session": False,
        "qlike": math.nan,
        "data_end": _iso(end),
    }
    if n_real == 0:
        rec["status"] = "no_sessions"
    elif not (np.isfinite(tv).all() and y > 0):
        rec["status"] = "bad_target"
    elif not (math.isfinite(F) and F > 0 and n_t is not None and n_t > 0):
        rec["status"] = "bad_forecast"
    else:
        per_session = n_real != n_t
        yc, Fc = (rec["ybar"], rec["Fbar"]) if per_session else (y, F)
        rec.update(per_session=per_session, qlike=float(qlike([yc], [Fc])[0]), status=OK)
    return rec


def score_risk_row(row: dict, s: _Series | None, end: np.datetime64,
                   last_session: np.datetime64 | None = None) -> dict | None:
    """Outcome part of one payload risk row's score (breach flags), or None while unrealised."""
    target = _day(row.get("date"))
    if s is None or (target is None and last_session is None):
        return None
    d = s.dates
    i = int(np.searchsorted(d, last_session, side="right") if last_session is not None
            else np.searchsorted(d, target, side="left"))
    if i >= len(d):
        return None
    r = float(s.r_cc[i])
    rec = {
        "asset": str(row["asset"]),
        "model": str(row["model"]),
        "date": _iso(target),
        "session_date": _iso(d[i]),
        **{k: _float(row.get(k)) for k in ("sigma", "var99", "var975", "es975")},
        "r_cc": r,
        "loss": -r,
        "breach99": None,
        "breach975": None,
        "data_end": _iso(end),
    }
    if not math.isfinite(r):
        rec["status"] = "bad_return"
    elif not (math.isfinite(rec["var99"]) and math.isfinite(rec["var975"])):
        rec["status"] = "bad_var"
    else:
        rec["breach99"] = int(hits([-r], [rec["var99"]])[0])
        rec["breach975"] = int(hits([-r], [rec["var975"]])[0])
        rec["status"] = OK
    return rec


def forecast_record(payload: dict, row: dict, ctx: _Context) -> dict | None:
    """Complete score row (SCORE_COLUMNS) of one payload forecast row, or None while its window is open."""
    a = str(row.get("asset"))
    rec = score_forecast_row(row, ctx.series.get(a), ctx.end, _last_session(payload, a))
    if rec is None:
        return None
    t = recorded_utc(payload)
    first = _day(row.get("window_first"))
    first = _day(rec["real_first"]) if first is None else first
    out = {"run_id": str(payload["run_id"]), "run_utc": str(payload["run_utc"]), "recorded_utc": _utc_iso(t),
           **_timing(t, ctx.calendar.bounds(a, first)), **rec}
    if not ctx.history_ok(payload, a):
        out["status"] = HISTORY_CHANGED
    return {c: out[c] for c in SCORE_COLUMNS}


def risk_record(payload: dict, row: dict, ctx: _Context) -> dict | None:
    """Complete score row (RISK_SCORE_COLUMNS) of one payload risk row, or None while unrealised."""
    a = str(row.get("asset"))
    rec = score_risk_row(row, ctx.series.get(a), ctx.end, _last_session(payload, a))
    if rec is None:
        return None
    t = recorded_utc(payload)
    target = _day(row.get("date"))
    target = _day(rec["session_date"]) if target is None else target
    timing = _timing(t, ctx.calendar.bounds(a, target))
    out = {"run_id": str(payload["run_id"]), "run_utc": str(payload["run_utc"]), "recorded_utc": _utc_iso(t),
           "before_close": timing["before_close"], "first_close_utc": timing["first_close_utc"], **rec}
    if not ctx.history_ok(payload, a):
        out["status"] = HISTORY_CHANGED
    return {c: out[c] for c in RISK_SCORE_COLUMNS}


def _fkey(f: dict) -> tuple[str, ...]:
    return str(f.get("asset")), str(f.get("horizon")), str(f.get("model")), _iso(_day(f.get("origin")))


def _rkey(r: dict) -> tuple[str, ...]:
    return str(r.get("asset")), str(r.get("model")), _iso(_day(r.get("date")))


# ============================================================================================= CSV store
def _read(path: Path, columns: Sequence[str], key: Sequence[str]) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=list(columns))
    df = pd.read_csv(path, dtype={k: str for k in key}, keep_default_na=False, na_values=[""],
                     float_precision="round_trip")  # floats exactly as written (the audit compares bit for bit)
    if list(df.columns) != list(columns):
        raise ValueError(f"{path}: columns {list(df.columns)} differ from the expected {list(columns)}")
    for k in key:
        df[k] = df[k].fillna("").astype(str)
    return df


def load_scores(path: Path | None = None) -> pd.DataFrame:
    """``forecasts/scores.csv`` (empty frame with SCORE_COLUMNS when absent)."""
    return _read(Path(path or paths.SCORES), SCORE_COLUMNS, SCORE_KEY)


def load_risk_scores(path: Path | None = None) -> pd.DataFrame:
    """``forecasts/risk_scores.csv`` (empty frame with RISK_SCORE_COLUMNS when absent)."""
    return _read(Path(path or paths.RISK_SCORES), RISK_SCORE_COLUMNS, RISK_KEY)


def _key_list(df: pd.DataFrame, key: Sequence[str]) -> list[tuple[str, ...]]:
    return list(map(tuple, df[list(key)].astype(str).to_numpy())) if len(df) else []


def _keys(df: pd.DataFrame, key: Sequence[str]) -> set[tuple[str, ...]]:
    return set(_key_list(df, key))


def _append(path: Path, rows: list[dict], columns: Sequence[str], key: Sequence[str], existing: pd.DataFrame) -> int:
    """Append rows whose key is not yet in ``path`` (append-only, idempotent); returns the number written."""
    have = _keys(existing, key)
    new = []
    for r in rows:
        k = tuple(str(r[c]) for c in key)
        if k in have:
            continue
        have.add(k)
        new.append(r)
    if not new:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.exists() or path.stat().st_size == 0
    text = pd.DataFrame(new, columns=list(columns)).to_csv(index=False, header=fresh, lineterminator="\n")
    with open(path, "a", encoding="utf-8", newline="") as f:
        f.write(text)
    return len(new)


# ============================================================================================= audit
def _norm(v):
    """Comparable form of a cell, stored (CSV) or re-derived: None for empty/NaN, bool, float or str."""
    if v is None:
        return None
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip()
        if s in ("", "True", "False"):
            return None if s == "" else s == "True"
        try:
            v = float(s)
        except ValueError:
            return s
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        return None if math.isnan(x) else x
    return str(v)


def _same(stored, derived) -> bool:
    """A stored cell equals its re-derived value (fast path for identical values, else :func:`_norm`)."""
    if type(stored) is type(derived) and not isinstance(stored, float) and stored == derived:
        return True
    return _norm(stored) == _norm(derived)


def _audit_part(df: pd.DataFrame, key: Sequence[str], columns: Sequence[str], by_run: dict[str, list[dict]],
                part: str, keyfn: Callable[[dict], tuple[str, ...]],
                derive: Callable[[dict, dict, _Context], dict | None], ctx: _Context) -> dict[tuple[str, ...], str]:
    bad: dict[tuple[str, ...], str] = {}
    keys = _key_list(df, key)
    counts = Counter(keys)
    rows_of: dict[str, dict[tuple[str, ...], tuple[dict, dict]]] = {}
    for k, row in zip(keys, df.to_dict("records"), strict=True):
        if counts[k] > 1:
            bad[k] = "the key appears more than once in the file"
            continue
        if k[0] not in by_run:
            bad[k] = "its run is not in the verified ledger"
            continue
        if k[0] not in rows_of:  # first payload, first row of a key wins, as when scoring
            rows_of[k[0]] = {}
            for p in by_run[k[0]]:
                for r in p[part]:
                    rows_of[k[0]].setdefault(keyfn(r), (p, r))
        hit = rows_of[k[0]].get(k[1:])
        if hit is None:
            bad[k] = "the row is not in its run's payload"
            continue
        p, src = hit
        rec = derive(p, src, ctx)
        if rec is None:
            bad[k] = "its target is not in the current live data"
            continue
        diff = [c for c in columns if c not in key and c not in _NOT_REDERIVED and not _same(row[c], rec[c])]
        if diff:
            bad[k] = "differs from the re-derived score in " + ", ".join(
                f"{c} (stored {row[c]!r}, re-derived {rec[c]!r})" for c in diff[:4])
    return bad


def audit_scores(scores: pd.DataFrame, risk_scores: pd.DataFrame, payloads: Iterable[dict],
                 daily: pd.DataFrame | None = None, *, check_history: bool = True,
                 ctx: _Context | None = None) -> dict:
    """Re-derive every stored score from its payload and the live data (all sessions in ``daily``).

    A row fails when its run is not among ``payloads`` (the verified ledger), the payload has no such row, its
    target is not in the data, or any stored column except ``data_end`` differs from the re-derived value (floats
    exactly: the CSV is read back round-trip). Returns ``{"forecast": {key: reason}, "risk": {key: reason},
    "checked": {"forecast": n, "risk": n}}``.
    """
    if ctx is None:
        ctx = _Context(daily if daily is not None else pd.DataFrame(columns=["asset", "session_date", "tv", "r_cc"]),
                       _max_day(daily), check_history)
    by_run: dict[str, list[dict]] = {}
    for p in payloads:
        by_run.setdefault(str(p["run_id"]), []).append(p)
    return {
        "forecast": _audit_part(scores, SCORE_KEY, SCORE_COLUMNS, by_run, "forecasts", _fkey, forecast_record, ctx),
        "risk": _audit_part(risk_scores, RISK_KEY, RISK_SCORE_COLUMNS, by_run, "risk", _rkey, risk_record, ctx),
        "checked": {"forecast": len(scores), "risk": len(risk_scores)},
    }


def _unaudited(scores: pd.DataFrame, risk_scores: pd.DataFrame, reason: str) -> dict:
    """Audit result when no stored score can be re-derived (live data not verified): every row fails."""
    return {"forecast": dict.fromkeys(_key_list(scores, SCORE_KEY), reason),
            "risk": dict.fromkeys(_key_list(risk_scores, RISK_KEY), reason),
            "checked": {"forecast": len(scores), "risk": len(risk_scores)}}


def _drop(df: pd.DataFrame, key: Sequence[str], bad: dict) -> pd.DataFrame:
    if not bad or df.empty:
        return df
    keep = np.array([k not in bad for k in _key_list(df, key)], dtype=bool)
    return df[keep].reset_index(drop=True)


def drop_failed(scores: pd.DataFrame, risk_scores: pd.DataFrame, audit: dict | None) -> tuple[pd.DataFrame,
                                                                                               pd.DataFrame]:
    """The score frames without the rows that failed :func:`audit_scores`."""
    if not audit:
        return scores, risk_scores
    return _drop(scores, SCORE_KEY, audit["forecast"]), _drop(risk_scores, RISK_KEY, audit["risk"])


def verified_scores(*, payloads: list[dict] | None = None, daily: pd.DataFrame | None = None
                    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Stored scores minus every row that does not re-derive, and the audit — what the report's tables use.

    Defaults: the verified ledger prefix and :func:`verified_live_daily` (a :class:`LiveDataUnverified` makes every
    row fail). For a display (the dashboard) that should show exactly the numbers of ``forward_test.md``.
    """
    scores, risk = load_scores(), load_risk_scores()
    if payloads is None:
        payloads = ledger_payloads()[1]
    payloads, _ = _valid_payloads(payloads)
    audit, _ = _audit_with_data(scores, risk, payloads, daily)
    s, r = drop_failed(scores, risk, audit)
    return s, r, audit


def _audit_with_data(scores, risk, payloads, daily) -> tuple[dict, str | None]:
    """Audit with the given data or the verified live data; ``(audit, note on the live data)``."""
    if scores.empty and risk.empty:
        return {"forecast": {}, "risk": {}, "checked": {"forecast": 0, "risk": 0}}, None
    note = "supplied by the caller (not checked against state.json)"
    if daily is None:
        try:
            daily, info = verified_live_daily()
            note = _data_note(info)
        except (LiveDataUnverified, ImportError, AttributeError, OSError, ValueError) as e:
            why = f"live data not verified ({type(e).__name__}: {e})"
            log.error("stored scores cannot be re-derived: %s", why)
            return _unaudited(scores, risk, why), f"**NOT VERIFIED** — {esc(str(e))}"
    return audit_scores(scores, risk, payloads, daily), note


def _data_note(info: dict | None) -> str | None:
    if not info:
        return None
    sha = info.get("sha256") or {}
    files = ", ".join(f"{n.replace('live_', '').replace('_gold', '')} `{h[:12]}…`" for n, h in sha.items())
    return (f"the consistency check of the last update (data through {info.get('end')}, finished "
            f"{info.get('checked_utc')}) passed, and the live gold files still have the SHA-256 recorded with it "
            f"({files})")


# ============================================================================================= pending
def _due(row: dict, horizon: str) -> np.datetime64 | None:
    """Last calendar day of a forecast's target window (scored by the first daily run after that UTC day)."""
    origin = _day(row.get("origin"))
    if horizon == "1d":
        first = _day(row.get("window_first"))
        return first if first is not None else (None if origin is None else origin + np.timedelta64(1, "D"))
    if origin is None or horizon not in C.HORIZONS:
        return None
    return origin + np.timedelta64(C.horizon_days(horizon), "D")


def pending_rows(payloads: Iterable[dict], scores: pd.DataFrame, risk_scores: pd.DataFrame) -> pd.DataFrame:
    """Payload rows without a score yet, with the day their window closes (``due``)."""
    fkeys, rkeys = _keys(scores, SCORE_KEY), _keys(risk_scores, RISK_KEY)
    rows = []
    for p in payloads:
        rid = str(p["run_id"])
        for f in p["forecasts"]:
            h = str(f.get("horizon"))
            k = (rid, *_fkey(f))
            if k not in fkeys:
                rows.append(("forecast", rid, k[1], h, k[3], _iso(_due(f, h)) or None))
        for r in p["risk"]:
            k = (rid, *_rkey(r))
            if k not in rkeys:
                rows.append(("risk", rid, k[1], "1d", k[2], k[3] or None))
    df = pd.DataFrame(rows, columns=_PENDING_COLUMNS)
    df["due"] = pd.to_datetime(df["due"])
    return df


# ============================================================================================= score_all
def score_all(daily: pd.DataFrame | None = None, *, payloads: Iterable[dict] | None = None, end=None,
              entries: list[dict] | None = None, report: bool = True, now: datetime | None = None,
              check_history: bool = True) -> dict:
    """Score every ledger forecast and risk row whose outcome is now observable (LIVE_SPEC §6).

    daily: the live gold table of every asset (default :func:`verified_live_daily`; raises
    :class:`LiveDataUnverified` and scores nothing when the last update's consistency check did not pass or the
    files changed since). payloads: forecast payloads (default: the verified prefix of the ledger). end: the live
    data end date (default: the last session date in ``daily``). check_history: compare each payload's
    ``rows_sha256`` with the data (off only for callers whose tables are not live gold).

    Appends new scores to ``forecasts/scores.csv`` / ``forecasts/risk_scores.csv``, re-derives every stored score
    (:func:`audit_scores`) and, with ``report``, rewrites ``reports/live/forward_test.md``. Returns a summary dict;
    raises :class:`ScoreIntegrityError` (after writing) when stored scores do not re-derive, ledger entries were
    left out (broken chain, unreadable payload) or new rows were recorded as ``history_changed``.
    """
    problems: list[str] = []
    chain = None
    from_ledger = payloads is None
    if from_ledger:
        entries, payloads, problems, chain = _ledger_state()
    payloads, n_invalid = _valid_payloads(payloads)
    data_info, data_note = None, "supplied by the caller (not checked against state.json)"
    if daily is None:
        daily, data_info = verified_live_daily()  # raises LiveDataUnverified before anything is written
        data_note = _data_note(data_info)
    max_day = _max_day(daily)
    end_day = max_day if end is None else _day(end)

    ctx = _Context(daily, end_day, check_history)
    scores, risk = load_scores(), load_risk_scores()
    fdone, rdone = _keys(scores, SCORE_KEY), _keys(risk, RISK_KEY)
    new_f: list[dict] = []
    new_r: list[dict] = []
    for p in payloads:
        if end_day is None:
            break
        rid = str(p["run_id"])
        for f in p["forecasts"]:
            if (rid, *_fkey(f)) in fdone:
                continue  # scored once, never rescored
            rec = forecast_record(p, f, ctx)
            if rec is not None:
                new_f.append(rec)
        for r in p["risk"]:
            if (rid, *_rkey(r)) in rdone:
                continue
            rec = risk_record(p, r, ctx)
            if rec is not None:
                new_r.append(rec)

    n_f = _append(paths.SCORES, new_f, SCORE_COLUMNS, SCORE_KEY, scores)
    n_r = _append(paths.RISK_SCORES, new_r, RISK_SCORE_COLUMNS, RISK_KEY, risk)
    scores, risk = load_scores(), load_risk_scores()
    audit = audit_scores(scores, risk, payloads, ctx=ctx.at(max_day))  # every row, the new ones included
    pend = pending_rows(payloads, scores, risk)
    next_due = pend["due"].min() if len(pend) else None
    hist_new = [f"{r['run_id']} {r['asset']}" for r in (*new_f, *new_r) if r["status"] == HISTORY_CHANGED]
    hist_new = list(dict.fromkeys(hist_new))
    summary = {
        "data_end": _iso(end_day) or None,
        "payloads": len(payloads),
        "invalid_payloads": n_invalid,
        "unreadable_payloads": problems,
        "forecasts": {"new": n_f, "total": len(scores), "ok": int((scores["status"] == OK).sum()),
                      "pending": int((pend["kind"] == "forecast").sum())},
        "risk": {"new": n_r, "total": len(risk), "ok": int((risk["status"] == OK).sum()),
                 "pending": int((pend["kind"] == "risk").sum())},
        "next_due": None if next_due is None or pd.isna(next_due) else next_due.date().isoformat(),
        "integrity": {
            "chain_ok": None if chain is None else _chain_ok(chain),
            "live_data": data_info,
            "rederived": {"forecast": audit["checked"]["forecast"] - len(audit["forecast"]),
                          "risk": audit["checked"]["risk"] - len(audit["risk"])},
            "failed": {"forecast": len(audit["forecast"]), "risk": len(audit["risk"])},
            "examples": [f"{'/'.join(k)}: {why}" for k, why in
                         [*audit["forecast"].items(), *audit["risk"].items()][:5]],
            "history_changed_new": hist_new,
        },
        "report": None,
    }
    log.info("scored %d forecasts and %d risk rows (data end %s); %d / %d pending; %d / %d stored scores "
             "re-derived", n_f, n_r, summary["data_end"], summary["forecasts"]["pending"],
             summary["risk"]["pending"], sum(summary["integrity"]["rederived"].values()),
             sum(audit["checked"].values()))
    if report:
        summary["report"] = str(report_forward(scores, risk, payloads=payloads, entries=entries,
                                               data_end=summary["data_end"], now=now, problems=problems,
                                               chain=chain if from_ledger else _UNSET, audit=audit,
                                               data_note=data_note))
    failures = []
    if problems:
        failures.append(f"{len(problems)} ledger entr{'y' if len(problems) == 1 else 'ies'} not scored "
                        f"({problems[0]})")
    if chain is not None and not _chain_ok(chain):
        failures.append(f"ledger chain: {describe_chain(chain)}")
    if audit["forecast"] or audit["risk"]:
        failures.append(f"{len(audit['forecast'])} forecast and {len(audit['risk'])} risk score(s) do not "
                        f"re-derive from the ledger and the live data (e.g. {summary['integrity']['examples'][0]})")
    if hist_new:
        failures.append(f"the live data up to the run's last session changed since run(s) {', '.join(hist_new)} "
                        "(rows_sha256) — recorded as history_changed")
    for f in failures:
        log.error("integrity: %s", f)
    if failures:
        raise ScoreIntegrityError("forward-test integrity problem: " + "; ".join(failures), summary)
    return summary


# ============================================================================================= tables
def headline_scores(scores: pd.DataFrame) -> pd.DataFrame:
    """Scores behind the ratio table: status ok, recorded ex ante (no part of the window was in the run's data)
    and before the window's first session closed, one row per (asset, horizon, model, origin) — the earliest
    recorded run (run_id is UTC time). Pass audited scores (:func:`drop_failed`) for the headline."""
    s = scores[(scores["status"] == OK) & _truthy(scores["ex_ante"]) & _truthy(scores["before_close"])]
    s = s.sort_values("run_id", kind="stable").drop_duplicates(["asset", "horizon", "model", "origin"], keep="first")
    return s.reset_index(drop=True)


def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))


def ratio_table(scores: pd.DataFrame, with_iv: bool = False) -> pd.DataFrame:
    """Frozen leaderboard (``volrisk.evaluation.leaderboard``): mean QLIKE ratio vs HAR per (asset, horizon,
    model) on common origins of the headline scores. IV benchmarks only with ``with_iv`` (IV subsample, 1m)."""
    h = headline_scores(scores)
    if h.empty:
        return pd.DataFrame(columns=["asset", "horizon", "split", "model", "qlike", "mse", "qlike_ratio", "n"])
    ybar, Fbar = h["ybar"].astype(float), h["Fbar"].astype(float)
    losses = pd.DataFrame({
        "asset": h["asset"], "horizon": h["horizon"], "model": h["model"],
        "origin": pd.to_datetime(h["origin"]), "split": "forward",
        "qlike": h["qlike"].astype(float), "mse": mse(ybar, Fbar),  # MSE on per-session means (not shown)
    })
    if not with_iv:
        return leaderboard(losses, ref=REF)
    l1m = losses[losses["horizon"] == "1m"]
    parts = []
    for asset, g in l1m.groupby("asset", sort=False):
        present = list(dict.fromkeys(g["model"]))
        if any(m in IV_MODELS for m in present):
            parts.append(leaderboard(g, ref=REF, models=present))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def risk_table(risk_scores: pd.DataFrame) -> pd.DataFrame:
    """Breaches vs expected per (model, asset) and per model over all assets, one row per (asset, model, date):
    status ok, recorded before the target session closed, the earliest run."""
    r = risk_scores[(risk_scores["status"] == OK) & _truthy(risk_scores["before_close"])]
    r = r.sort_values("run_id", kind="stable").drop_duplicates(["asset", "model", "date"], keep="first")
    rows = [_risk_row(model, asset, g) for (model, asset), g in r.groupby(["model", "asset"], sort=False)]
    rows += [_risk_row(model, "ALL", g) for model, g in r.groupby("model", sort=False)]
    return pd.DataFrame(rows, columns=["model", "asset", "N", "b99", "exp99", "p99", "b975", "exp975", "p975"])


def _risk_row(model: str, asset: str, g: pd.DataFrame) -> dict:
    N = len(g)
    b99, b975 = g["breach99"].astype(int).to_numpy(), g["breach975"].astype(int).to_numpy()
    return {
        "model": model, "asset": asset, "N": N,
        "b99": int(b99.sum()), "exp99": N * ALPHA_99, "p99": kupiec(b99, ALPHA_99)["p_binom"] if N else math.nan,
        "b975": int(b975.sum()), "exp975": N * ALPHA_975,
        "p975": kupiec(b975, ALPHA_975)["p_binom"] if N else math.nan,
    }


# ============================================================================================= integrity status
_UNSET = object()


def _first(d: dict, keys: Sequence[str]):
    return next((d[k] for k in keys if k in d), None)


def describe_chain(res) -> str:
    """Plain text for ``ledger.verify_chain()`` (``{ok, n, first_bad, reason, head, orphans}``; a bool, None = no
    broken link, a list of problems or a string are accepted too)."""
    if res is None or res is True:
        return "intact"
    if res is False:
        return "**BROKEN**"
    if isinstance(res, str):
        return res
    if isinstance(res, dict):
        reason = _first(res, ("reason", "first_broken", "broken", "error", "problem", "detail", "message"))
        ok = _first(res, ("ok", "intact", "valid", "passed", "pass"))
        ok = reason in (None, "", [], {}) if ok is None else bool(ok)
        n = _first(res, ("n", "n_entries", "entries", "checked", "count"))
        n_txt = "" if n is None else f"{len(n) if isinstance(n, (list, tuple)) else n} entries"
        if ok:
            head = res.get("head")
            parts = [p for p in (n_txt, f"head `{head[:16]}…`" if isinstance(head, str) and head else "") if p]
            return "intact" + (f" ({', '.join(parts)})" if parts else "")
        where = f" at line {res['first_bad']}" if res.get("first_bad") is not None else ""
        return f"**BROKEN**{where}" + (f" of {n_txt}" if n_txt else "") + (f": {reason}" if reason else "")
    if isinstance(res, (list, tuple)):
        return "intact" if not res else f"**BROKEN**: {res[0]}"
    return str(res)


def _status_of(v) -> str:
    if isinstance(v, dict):
        s = _first(v, ("status", "state"))
        return "unknown" if s is None else str(s)
    return str(v)


_OTS_MEANING = {"complete": "Bitcoin-attested", "pending": "submitted, awaiting Bitcoin attestation",
                "partial": "only one calendar answered", "unstamped": "no proof yet", "invalid": "proof does not match"}


def describe_timestamps(res) -> str:
    """Plain text for ``ots.status_all()`` (``{n, counts, files}``; per-file statuses or plain counts accepted)."""
    if res is None:
        return "no stamps"
    if isinstance(res, str):
        return res
    if isinstance(res, dict) and isinstance(res.get("counts"), dict):
        counts = res["counts"]
        n = res.get("n", sum(counts.values()))
        if not n:
            return "no payload stamped yet"
        parts = [f"{v} {k}" + (f" ({_OTS_MEANING[k]})" if k in _OTS_MEANING else "") for k, v in counts.items()]
        return f"{n} payload file(s): " + ", ".join(parts)
    if isinstance(res, dict):
        vals = list(res.values())
        if vals and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            return ", ".join(f"{k}: {v}" for k, v in res.items())
        statuses = [_status_of(v) for v in vals]
    elif isinstance(res, (list, tuple)):
        statuses = [_status_of(v) for v in res]
    else:
        return str(res)
    if not statuses:
        return "no stamps yet"
    return f"{len(statuses)} payload(s): " + ", ".join(f"{k} {v}" for k, v in Counter(statuses).most_common())


def _call_status(module: str, fn: str, describe) -> str:
    try:
        res = getattr(_module(module), fn)()
    except (ImportError, AttributeError) as e:
        return f"not available ({type(e).__name__}: {e})"
    except Exception as e:  # noqa: BLE001 - the report shows a failed check instead of aborting
        return f"**check failed** ({type(e).__name__}: {e})"
    return describe(res)


# ============================================================================================= report
def _runs(entries: list[dict] | None, payloads: list[dict]) -> pd.DataFrame:
    src = entries if entries else payloads
    rows = [(str(e.get("run_id", "")), str(e.get("run_utc", ""))) for e in src]
    return pd.DataFrame(rows, columns=["run_id", "run_utc"]).sort_values("run_id", kind="stable")


def report_forward(scores: pd.DataFrame | None = None, risk_scores: pd.DataFrame | None = None, *,
                   payloads: list[dict] | None = None, entries: list[dict] | None = None, data_end=None,
                   out: Path | None = None, now: datetime | None = None, chain=_UNSET, timestamps=_UNSET,
                   problems: list[str] | None = None, daily: pd.DataFrame | None = None, audit: dict | None = None,
                   data_note: str | None = None) -> Path:
    """Write ``reports/live/forward_test.md`` from the score files and the ledger (LIVE_SPEC §6).

    Every argument is optional: scores default to the CSVs, payloads/entries to the verified ledger prefix,
    ``chain`` to ``ledger.verify_chain()`` and ``timestamps`` to ``ots.status_all()`` (shown as 'not available' if
    missing). ``audit``: the :func:`audit_scores` result (default: re-derived here from ``daily``, or from
    :func:`verified_live_daily` — when the live data cannot be verified, no stored score enters the tables).
    ``problems``: ledger entries that were not scored (listed in the report). Never raises on integrity problems:
    it reports them.
    """
    scores = load_scores() if scores is None else scores
    risk_scores = load_risk_scores() if risk_scores is None else risk_scores
    problems = list(problems or [])
    if payloads is None:
        try:
            entries, payloads, problems, led_chain = _ledger_state()
            chain = led_chain if chain is _UNSET else chain
        except (ImportError, AttributeError, RuntimeError, OSError, ValueError) as e:
            log.warning("ledger not readable for the forward report: %s", e)
            entries, payloads, problems = [], [], [f"ledger not readable: {type(e).__name__}: {e}"]
    payloads, _ = _valid_payloads(payloads)
    if audit is None:
        audit, note = _audit_with_data(scores, risk_scores, payloads, daily)
        data_note = data_note or note
    if data_end is None:
        ends = [str(x) for x in (*scores["data_end"].dropna(), *risk_scores["data_end"].dropna()) if str(x)]
        data_end = max(ends) if ends else None
    text = forward_markdown(
        scores, risk_scores, _runs(entries, payloads), pending_rows(payloads, scores, risk_scores),
        data_end=None if data_end is None else _iso(_day(data_end)),
        now=now or datetime.now(timezone.utc),
        chain=_call_status("ledger", "verify_chain", describe_chain) if chain is _UNSET else describe_chain(chain),
        timestamps=(_call_status("ots", "status_all", describe_timestamps) if timestamps is _UNSET
                    else describe_timestamps(timestamps)),
        problems=problems, audit=audit, data_note=data_note,
    )
    out = Path(out or paths.FORWARD_MD)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    log.info("wrote %s", out)
    return out


def _date_of(run_utc: str) -> str:
    d = _day(str(run_utc).replace("Z", ""))
    return _iso(d) if d is not None else str(run_utc)


def _min_due(p: pd.DataFrame) -> str:
    due = p["due"].min() if len(p) else None
    return "—" if due is None or pd.isna(due) else due.date().isoformat()


def _status_table(scores: pd.DataFrame, risk_scores: pd.DataFrame, pending: pd.DataFrame, audit: dict) -> str:
    """Counts per horizon; ``scores`` / ``risk_scores`` are the full files, ``audit`` says which rows failed."""
    clean, clean_r = drop_failed(scores, risk_scores, audit)
    head = headline_scores(clean)
    failed = Counter(k[2] for k in audit["forecast"])  # SCORE_KEY[2] = horizon
    rows = []
    for h in C.HORIZONS:
        s = clean[clean["horizon"] == h]
        ok = s[s["status"] == OK]
        late = ok[_truthy(ok["ex_ante"]) & ~_truthy(ok["before_close"])]
        p = pending[(pending["kind"] == "forecast") & (pending["horizon"] == h)]
        rows.append([h, f"{len(ok):,}", f"{len(ok[['asset', 'origin']].drop_duplicates()):,}",
                     f"{int((head['horizon'] == h).sum()):,}", f"{len(late):,}", f"{len(s) - len(ok):,}",
                     f"{failed.get(h, 0):,}", f"{len(p):,}", _min_due(p)])
    ok = clean_r[clean_r["status"] == OK]
    late = ok[~_truthy(ok["before_close"])]
    p = pending[pending["kind"] == "risk"]
    used = ok[_truthy(ok["before_close"])].drop_duplicates(["asset", "model", "date"])
    rows.append(["VaR (next session)", f"{len(ok):,}", f"{len(ok[['asset', 'date']].drop_duplicates()):,}",
                 f"{len(used):,}", f"{len(late):,}", f"{len(clean_r) - len(ok):,}", f"{len(audit['risk']):,}",
                 f"{len(p):,}", _min_due(p)])
    return md_table(["horizon", "scored rows", "target windows", "used below", "recorded after first close",
                     "not scorable", "failed re-check", "pending rows", "next window closes"], rows)


def _ratio_md(lb: pd.DataFrame, horizon: str, title: str) -> list[str]:
    g = lb[lb["horizon"] == horizon]
    if g.empty:
        return []
    assets = P.order_assets(g["asset"].unique())
    n = {a: int(g.loc[g["asset"] == a, "n"].max()) for a in assets}
    rows = [["n (common origins)", *[f"{n[a]:,}" for a in assets]]]
    for m in P.order_models(g["model"].unique()):
        cells = []
        for a in assets:
            r = g[(g["asset"] == a) & (g["model"] == m)]
            cells.append(fmt_num(r["qlike_ratio"].iloc[0], 3) if len(r) and n[a] else "—")
        rows.append([bold(esc(m)), *[bold(c) for c in cells]] if m == PRIMARY_MODEL else [esc(m), *cells])
    return [f"### {title}", "", md_table(["model", *assets], rows), ""]


def _risk_md(rt: pd.DataFrame) -> str:
    assets = P.order_assets(rt.loc[rt["asset"] != "ALL", "asset"].unique())
    rows = []
    for m in P.order_risk_models(rt["model"].unique()):
        tot = rt[(rt["model"] == m) & (rt["asset"] == "ALL")].iloc[0]
        cells = [f"{int(tot.N):,}", f"{int(tot.b99)}", fmt_num(tot.exp99, 2), fmt_p(tot.p99),
                 f"{int(tot.b975)}", fmt_num(tot.exp975, 2), fmt_p(tot.p975)]
        for a in assets:
            r = rt[(rt["model"] == m) & (rt["asset"] == a)]
            cells.append(f"{int(r.b99.iloc[0])} / {int(r.b975.iloc[0])} of {int(r.N.iloc[0])}" if len(r) else "—")
        rows.append([bold(esc(m)), *[bold(c) for c in cells]] if m == PRIMARY_RISK else [esc(m), *cells])
    header = ["risk model", "days", "99% breaches", "expected", "p (binom.)", "97.5% breaches", "expected",
              "p (binom.)", *[f"{a} 99 / 97.5" for a in assets]]
    return md_table(header, rows)


_EXPLAIN = [
    "## Why this is a forward test that cannot be tuned",
    "",
    "1. **The forecast is written down before the outcome exists.** Each daily run uses data up to the last complete "
    "UTC day and forecasts the next session, the next 7 days and the next 30 days. All numbers go into "
    "`forecasts/runs/<run_id>.json` before any of those target windows has closed. Each score carries "
    "`recorded_utc`, the latest of the run time and the payload's own clock readings (`checks.computed_utc`, "
    "`checks.timing_utc`); a forecast "
    "is used in the tables only if that time is before its first target session closed (`before_close`), so a "
    "payload made later with a past `--now` is counted separately and never mixed in.",
    "2. **It cannot be changed afterwards.** The SHA-256 of every payload is chained in `forecasts/ledger.jsonl`: each "
    "entry carries the hash of the previous one, so editing, deleting or reordering any run breaks every later link, "
    "and nothing at or after a broken link is scored. The payload's hash is also sent to public OpenTimestamps "
    "calendars and anchored in a Bitcoin block, which proves when the file existed without trusting this computer's "
    "clock (only the hash leaves the machine).",
    "3. **There is nothing left to tune.** Models, features and hyperparameters are the ones hash-sealed before the "
    "holdout was opened; the live code only moves the data end date forward. Changing a model would change *future* "
    "forecasts, never the ones already in the ledger.",
    "4. **Scoring is mechanical and write-once.** A forecast is scored only when its whole window is in live data "
    "that passed the consistency check against the sealed data, and only if the data up to its origin still hash to "
    "what the forecast saw (`rows_sha256`). Realised variance `y` = sum of the daily target `tv` over the window, "
    "loss = QLIKE with the same frozen function as the development and holdout evaluation. Each score is appended "
    "once; bad results stay in the file.",
    "5. **Every number is re-derived on every run.** `python -m volrisk_live score` recomputes each stored score from "
    "its ledger payload and the live data; a score that does not come out identical (an edited file, a forged "
    "payload, data revised after scoring) is listed under *Integrity*, left out of every table, and makes the "
    "command fail. `python -m volrisk_live verify` re-checks the chain, the timestamps and the frozen code.",
    "",
    "*Timing.* A run needs the previous UTC day's data files, so it starts after 00:00 UTC. The next BTC/ETH session "
    "is that UTC day (already under way, but the run uses no data from it); the EUR/USD session opened at 17:00 New "
    "York the evening before; S&P 500 forecasts are recorded before the US open. `recorded_utc` and "
    "`first_close_utc` of every score are in `scores.csv`; the recording time is bounded independently by the "
    "payload's Bitcoin timestamp (`verify`, check 8).",
    "",
]


def _failure_lines(audit: dict, limit: int = 20) -> list[str]:
    items = [*(("forecast", k, w) for k, w in audit["forecast"].items()),
             *(("risk", k, w) for k, w in audit["risk"].items())]
    out = [f"  - {kind} `{' / '.join(k)}`: {esc(w)}" for kind, k, w in items[:limit]]
    if len(items) > limit:
        out.append(f"  - … and {len(items) - limit:,} more")
    return out


def forward_markdown(scores: pd.DataFrame, risk_scores: pd.DataFrame, runs: pd.DataFrame, pending: pd.DataFrame,
                     *, data_end: str | None, now: datetime, chain: str, timestamps: str,
                     problems: Sequence[str] = (), audit: dict | None = None, data_note: str | None = None) -> str:
    """Markdown of the forward-test report (pure; ``report_forward`` gathers the inputs). ``scores`` /
    ``risk_scores`` are the full files; rows that failed ``audit`` are counted but left out of every table."""
    audit = audit or {"forecast": {}, "risk": {}, "checked": {"forecast": len(scores), "risk": len(risk_scores)}}
    n_failed = len(audit["forecast"]) + len(audit["risk"])
    clean, clean_r = drop_failed(scores, risk_scores, audit)
    first = runs["run_utc"].iloc[0] if len(runs) else None
    ok = clean[clean["status"] == OK]
    L = ["# Forward test — live forecasts scored against outcomes that did not exist when they were recorded", ""]
    if first is None:
        L += ["**Forward test, forecasts recorded before outcomes — no forecast run has been recorded yet.** "
              "Run `python -m volrisk_live daily` to record the first one.", ""]
    else:
        L += [f"**Forward test, forecasts recorded before outcomes, since {_date_of(first)}.**", ""]
    if n_failed or problems:
        L += [f"**Integrity problem: {n_failed:,} stored score(s) do not re-derive and {len(problems):,} ledger "
              "entr(ies) were not scored — they are left out of every table below; see *Integrity*.**", ""]
    origins = [str(x) for x in ok["origin"] if str(x)]
    facts = [
        ["first forecast run (UTC)", f"{first} (`{runs['run_id'].iloc[0]}`)" if first is not None else "—"],
        ["forecast runs in the ledger (verified chain)", f"{len(runs):,}"],
        ["first scored forecast origin", min(origins) if origins else "—"],
        ["realised data through", data_end or "—"],
        ["report generated", now.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")],
    ]
    L += [md_table(["", ""], facts), ""]

    # ------------------------------------------------------------------ counts
    L += ["## Scored so far", ""]
    if ok.empty and n_failed:
        L += ["**No stored score could be re-derived on this run** — see *Integrity*.", ""]
    elif ok.empty:
        due = _min_due(pending)
        if due != "—":
            nxt = pending.sort_values("due", kind="stable").iloc[0]
            L += [f"**No forecast has been scored yet — first scores after {due}.** The earliest target window "
                  f"({nxt['asset']} {nxt['horizon']}) closes on that day; it is scored by the first daily run after "
                  "that UTC day is complete and in the data.", ""]
        else:
            L += ["**No forecast has been scored yet** and none is pending.", ""]
    L += [_status_table(scores, risk_scores, pending, audit), "",
          "*scored rows*: forecasts (all models, all runs) whose window has fully passed; *target windows*: distinct "
          "(asset, origin); *used below*: one row per (asset, horizon, model, origin) — the earliest run — recorded "
          "ex ante and before the first target session closed; *recorded after first close*: the payload was "
          "written (by its own clock) after that close, so it is not a forecast in the sense of this report; "
          "*not scorable*: window passed without a valid session, with a non-positive target/forecast or with "
          "changed history (kept in the file with its status); *failed re-check*: stored row that does not "
          "re-derive (see *Integrity*); *pending*: window not closed yet.", ""]
    n_late = int((~_truthy(ok["ex_ante"])).sum())
    if n_late:
        L += [f"{n_late:,} scored row(s) had origins before the run's last session (part of their window was already "
              "in the data when they were recorded); they stay in `scores.csv` but are left out of the ratios.", ""]
    n_after = int((_truthy(ok["ex_ante"]) & ~_truthy(ok["before_close"])).sum())
    n_after_r = int((~_truthy(clean_r.loc[clean_r["status"] == OK, "before_close"])).sum())
    if n_after or n_after_r:
        L += [f"{n_after:,} forecast and {n_after_r:,} VaR row(s) were recorded after their first target session "
              "had closed (`recorded_utc` ≥ `first_close_utc`, e.g. a run made later with a past `--now`); they stay "
              "in the score files but are left out of every table.", ""]
    hist = sorted({f"{r} {a}" for df in (clean, clean_r)
                   for r, a in df.loc[df["status"] == HISTORY_CHANGED, ["run_id", "asset"]].to_numpy()})
    if hist:
        L += [f"**Data revised after a forecast:** for {len(hist)} run/asset pair(s) ({esc(', '.join(hist[:10]))}"
              f"{', …' if len(hist) > 10 else ''}) the live data up to the run's last session no longer hash to the "
              "`rows_sha256` in the payload; their rows have status `history_changed` and are left out.", ""]

    # ------------------------------------------------------------------ QLIKE ratios
    L += ["## QLIKE ratio vs HAR on the forecasts scored so far", ""]
    lb = ratio_table(clean)
    if lb.empty:
        L += ["No scored forecast yet — the tables appear once the first target windows have closed.", ""]
    else:
        L += ["Mean QLIKE(model) / mean QLIKE(HAR) per asset and horizon on the origins where every model has a score "
              "(SPEC §8, the frozen `leaderboard` function). Below 1 = lower loss than HAR. "
              f"**{PRIMARY_MODEL}** (equal-weight GJR + HARQ + LGBM) is the pre-registered combination. When fewer "
              "sessions were realised than scheduled (an invalid session), per-session means are compared. Few scored "
              "forecasts make these ratios noisy, and 1w / 1m windows overlap (n origins ≠ n independent outcomes): "
              "this is a running record, not a test.", ""]
        names = {"1d": "1d — next session", "1w": "1w — next 7 calendar days", "1m": "1m — next 30 calendar days"}
        for h in C.HORIZONS:
            L += _ratio_md(lb, h, names[h])
        iv = ratio_table(clean, with_iv=True)
        if not iv.empty:
            L += _ratio_md(iv, "1m", "1m including implied-volatility benchmarks (IV subsample)")

    # ------------------------------------------------------------------ VaR
    L += ["## VaR breaches vs expected (next-session VaR, 1 unit long)", ""]
    rt = risk_table(clean_r)
    if rt.empty:
        L += ["No next-session VaR has been scored yet.", ""]
    else:
        L += ["A breach is a loss `−r_cc` above the VaR recorded before the session closed (frozen `backtests.hits`); "
              "expected = days × 1% (99%) or × 2.5% (97.5%); p = exact two-sided binomial (Kupiec). One row per "
              f"(asset, model, date) — the earliest run. **{esc(PRIMARY_RISK)}** is the pre-registered primary model.",
              "", _risk_md(rt), ""]

    # ------------------------------------------------------------------ integrity
    checked = audit["checked"]["forecast"] + audit["checked"]["risk"]
    if not checked:
        rederive = "nothing stored yet"
    elif not n_failed:
        rederive = (f"all {audit['checked']['forecast']:,} forecast and {audit['checked']['risk']:,} risk score(s) "
                    "re-derived exactly from the verified ledger payloads and the live data")
    else:
        rederive = (f"**{len(audit['forecast']):,} forecast and {len(audit['risk']):,} risk score(s) of {checked:,} do "
                    "NOT re-derive** — left out of every table (remove such a line from the score file and the next "
                    "run scores it again from its payload):")
    L += ["## Integrity", "",
          f"- **Ledger hash chain** (`forecasts/ledger.jsonl`, `ledger.verify_chain()`): {chain}",
          f"- **OpenTimestamps proofs** (`forecasts/runs/<run_id>.json.ots`, `ots.status_all()`): {timestamps}",
          *[f"- **Ledger entry not scored**: {esc(p)}" for p in problems],
          *([f"- **Live data**: {data_note}"] if data_note else []),
          f"- **Stored scores re-derived on this run**: {rederive}",
          *(_failure_lines(audit) if n_failed else []),
          f"- **Score files** (append-only; a score, once written, is never rewritten): `forecasts/scores.csv` "
          f"({len(scores):,} rows), `forecasts/risk_scores.csv` ({len(risk_scores):,} rows)",
          "- **Frozen models**: the sealed package `src/volrisk/` is checked by `volrisk.holdout.verify_seal()`; the "
          "full check list is in `reports/live/verification.md`.", ""]
    return "\n".join(L + _EXPLAIN + _timing_md(clean))


def _timing_md(scores: pd.DataFrame) -> list[str]:
    """Per asset: scored target windows recorded before their first session opened (from the session calendar)."""
    w = headline_scores(scores).drop_duplicates(["asset", "origin"])
    w = w[w["before_open"].astype(str).str.strip().str.lower().isin(("true", "false"))]
    if w.empty:
        return []
    parts = [f"{a} {int(_truthy(g['before_open']).sum()):,} of {len(g):,}"
             for a in P.order_assets(w["asset"].unique()) for g in [w[w["asset"] == a]]]
    return ["Scored target windows (asset × origin) recorded before their first session opened: "
            + ", ".join(parts) + ". The others were recorded during that first session, without any data from it.",
            ""]
