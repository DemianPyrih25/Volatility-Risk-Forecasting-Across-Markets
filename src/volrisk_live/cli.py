"""Command line of the live layer (docs/LIVE_SPEC.md §8): ``uv run python -m volrisk_live <command> [options]``.

Commands: ``update`` (§2), ``forecast`` (§3–4), ``score`` (§6), ``verify`` (§7), ``stamp`` (§4: stamp payloads that
have no OpenTimestamps proof yet, then ask the calendars for Bitcoin attestations), ``dashboard`` (§8) and
``daily`` = update → score → stamp/upgrade earlier payloads → forecast (+ stamp) → forward-test report → a short
human summary. The work is done by the modules ``update``, ``forecast``, ``score``, ``verify``, ``ledger`` and
``ots``; this file only wires them (they are imported lazily, so ``--help`` and the tests need none of them).

Recording time (LIVE_SPEC §0: forecasts are recorded before their outcome exists). Every "recorded before ..."
statement here and in the dashboard uses :func:`recorded_utc`, the latest of the payload's ``run_utc`` (the run's
start), ``checks.computed_utc`` (the machine clock when the payload was built) and ``checks.timing_utc``, against
the frozen session calendar. ``--now`` is refused by ``forecast`` and ``daily`` on the project ledger (a recorded run
always carries the real clock). ``forecast`` / ``daily`` warn, before the walk-forward and from the written payload,
when the next session of an asset has closed (or closes within ``PREFLIGHT_MARGIN``): its forecast could not precede
its outcome. ``scripts/register_daily_task.ps1`` schedules ``daily`` in the early UTC morning for that reason.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
import logging
import sys
from collections.abc import Callable, Iterable
from datetime import UTC, date, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from types import ModuleType
from typing import Any

from volrisk_live import paths

log = logging.getLogger("volrisk_live")
_PROJECT_LEDGER = paths.LEDGER  # the real forecasts/ledger.jsonl, captured at import (tests redirect paths.LEDGER)

COMMANDS = ("update", "forecast", "score", "verify", "stamp", "dashboard", "daily")
HELP = """Live forecasting, forward test and verification on top of the frozen volrisk models (docs/LIVE_SPEC.md).

commands:
  update     download new raw data, rebuild the live gold tables, prove they equal the sealed data
  forecast   next-session / week / month forecasts and VaR/ES -> hash-chained ledger (+ OpenTimestamps)
  score      score recorded forecasts whose outcome now exists; rewrite reports/live/forward_test.md
  verify     re-runnable verification report (reports/live/verification.md + .json); exit 1 on any FAIL
  stamp      stamp payloads without a proof and upgrade pending proofs to Bitcoin attestations
  dashboard  frozen dashboard + Tomorrow / Forward test / Verification tabs
  daily      update -> score -> stamp/upgrade -> forecast (+ stamp) -> forward-test report -> summary
"""
PRIMARY_FORECAST = "COMBO"  # pre-registered primary forecast model (SPEC §0)
PRIMARY_RISK = "COMBO+FHS"  # pre-registered primary risk model (SPEC §0)
HORIZONS = ("1d", "1w", "1m")
# forecast / daily warn when a next session closes within this time of the start (walk-forward + ledger + stamp)
PREFLIGHT_MARGIN = timedelta(hours=1)
CLOCK_SKEW = timedelta(minutes=5)  # computed_utc may precede run_utc by this much (clock jitter), no more
TIMING_TEXT = {"before_open": "before open", "in_session": "in session", "after_close": "AFTER CLOSE",
               "unknown": "n/a"}
_SCHEDULE_DAYS = 14  # calendar days searched for the next scheduled session (longest closure is far shorter)


# --------------------------------------------------------------------------------------------- helpers
def parse_now(text: str) -> datetime:
    """``--now``: an ISO time in UTC ('2026-10-02T07:00:00Z'; a naive time or a date is read as UTC)."""
    try:
        t = datetime.fromisoformat(text.strip())
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"not an ISO date/time: {text!r}") from e
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def _mod(name: str) -> ModuleType:
    """A live module, imported only when a command needs it."""
    return importlib.import_module(f"volrisk_live.{name}")


def _call(fn: Callable, must: Iterable[str] = (), **kwargs: Any) -> Any:
    """Call ``fn`` with the keyword arguments its signature accepts.

    The spec fixes the arguments of ``update.update`` and ``forecast.forecast``; the scoring and verification
    entry points only by name. Arguments named in ``must`` (an explicit user option such as ``--full``) are always
    passed, so an entry point that cannot honour them fails loudly instead of silently ignoring them.
    """
    must = set(must)
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return fn(**kwargs)
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return fn(**kwargs)
    names = {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    return fn(**{k: v for k, v in kwargs.items() if k in names or k in must})


def _rel(p: Path) -> str:
    try:
        return p.resolve().relative_to(paths.ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def _brief(x: Any, limit: int = 600) -> str:
    """Short one-line rendering of a module's summary value for the log."""
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        x = dataclasses.asdict(x)
    try:
        s = json.dumps(x, default=str, sort_keys=True)
    except (TypeError, ValueError):
        s = repr(x)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def ots_path(path: Path) -> Path:
    """The OpenTimestamps proof of ``path`` (``<file>.ots``, LIVE_SPEC §4)."""
    return Path(str(path) + ".ots")


# --------------------------------------------------------------------------------------------- recording time
def utc_time(x: Any) -> datetime | None:
    """An ISO time or datetime as an aware UTC datetime (naive = UTC); None when missing or unparsable."""
    if isinstance(x, datetime):
        t = x
    else:
        if x is None or not str(x).strip():
            return None
        try:
            t = datetime.fromisoformat(str(x).strip())
        except ValueError:
            return None
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def recorded_utc(payload: dict) -> datetime | None:
    """When a payload was written, taken conservatively: the latest of ``run_utc`` (the run's start, which ``--now``
    sets in a test sandbox), ``checks.computed_utc`` (the machine clock when the payload was built, after the
    walk-forward) and ``checks.timing_utc`` (the time ``forecast.py`` evaluated its timing flags at). A back-dated
    ``run_utc`` therefore never makes a forecast look recorded earlier than it was."""
    checks = payload.get("checks") or {}
    times = [t for t in (utc_time(payload.get("run_utc")), utc_time(checks.get("computed_utc")),
                         utc_time(checks.get("timing_utc"))) if t is not None]
    return max(times) if times else None


@lru_cache(maxsize=4096)
def session_bounds(asset: str, day: str) -> tuple[datetime, datetime] | None:
    """(open_utc, close_utc) of ``asset``'s scheduled session on ``day`` (ISO date) from the frozen calendar
    (``volrisk.sessions.session_schedule``, as ``verify.first_outcome_utc``); None if it is no scheduled session."""
    from volrisk import sessions

    try:
        d = date.fromisoformat(str(day)[:10])
        s = sessions.session_schedule(asset, d, d)
    except (KeyError, ValueError, TypeError):  # unknown asset or not a date
        return None
    if s.height != 1:
        return None
    o, c = utc_time(s["open_utc"][0]), utc_time(s["close_utc"][0])
    return (o, c) if o is not None and c is not None else None


def timing_state(recorded: datetime | None, bounds: tuple[datetime, datetime] | None) -> str:
    """``before_open`` / ``in_session`` / ``after_close`` of a session at the time ``recorded``, else ``unknown``."""
    if recorded is None or bounds is None:
        return "unknown"
    o, c = bounds
    return "before_open" if recorded < o else ("in_session" if recorded < c else "after_close")


def payload_timing(payload: dict) -> dict[str, dict]:
    """Per asset of ``payload`` that is not stale: its next session (the first session of every forecast window),
    the session's open/close (frozen calendar; the payload's own ``next_open_utc`` / ``next_close_utc`` only as a
    fallback), its ``state`` at :func:`recorded_utc` (``timing_state``) and ``rows``, the number of forecast and
    VaR/ES rows recorded for it (0 for an asset left out because its session had closed: ``checks.closed_assets``)."""
    rec = recorded_utc(payload)
    data = payload.get("data") or {}
    first: dict[str, str] = {}
    rows: dict[str, int] = {}
    for r in payload.get("forecasts") or []:
        if r.get("asset"):
            rows[str(r["asset"])] = rows.get(str(r["asset"]), 0) + 1
            if r.get("window_first") and r.get("horizon") == "1d":
                first.setdefault(str(r["asset"]), str(r["window_first"])[:10])
    for r in payload.get("risk") or []:
        if r.get("asset"):
            rows[str(r["asset"])] = rows.get(str(r["asset"]), 0) + 1
            if r.get("date"):
                first.setdefault(str(r["asset"]), str(r["date"])[:10])
    out: dict[str, dict] = {}
    for asset in dict.fromkeys([*rows, *(a for a, d in data.items() if isinstance(d, dict) and not d.get("stale"))]):
        d = data.get(asset) or {}
        day = d.get("next_session") or first.get(asset)
        bounds = session_bounds(asset, str(day)[:10]) if day else None
        if bounds is None:
            o, c = utc_time(d.get("next_open_utc")), utc_time(d.get("next_close_utc"))
            bounds = (o, c) if o is not None and c is not None else None
        out[asset] = {"session": str(day)[:10] if day else None, "open_utc": bounds[0] if bounds else None,
                      "close_utc": bounds[1] if bounds else None, "recorded_utc": rec,
                      "state": timing_state(rec, bounds), "rows": rows.get(asset, 0)}
    return out


def _live_end(now: datetime) -> date:
    """Live data end of a run at ``now``: ``update.live_end_date`` (the last complete UTC day)."""
    fn = getattr(_mod("update"), "live_end_date", None)
    return fn(now) if callable(fn) else now.date() - timedelta(days=1)


def next_sessions(end: date, assets: Iterable[str] | None = None) -> dict[str, tuple[date, datetime, datetime]]:
    """Per asset: (first scheduled session after ``end``, its open, its close) from the frozen calendar."""
    from volrisk import config as C
    from volrisk import sessions

    out = {}
    for asset in assets or C.ASSETS:
        s = sessions.session_schedule(asset, end + timedelta(days=1), end + timedelta(days=_SCHEDULE_DAYS))
        if s.height:
            out[asset] = (s["session_date"][0], utc_time(s["open_utc"][0]), utc_time(s["close_utc"][0]))
    return out


def closed_targets(now: datetime, margin: timedelta = timedelta(0)) -> dict[str, tuple[date, datetime]]:
    """Assets whose next session after the live data end of a run at ``now`` closes before ``now + margin``:
    {asset: (session, close_utc)}. A forecast recorded then would not precede its outcome."""
    now = utc_time(now) or datetime.now(UTC)
    return {a: (day, c) for a, (day, _o, c) in next_sessions(_live_end(now)).items() if c <= now + margin}


def _closed_text(closed: dict[str, tuple[date, datetime]]) -> str:
    return ", ".join(f"{a} {day} (closed {c:%H:%M} UTC)" for a, (day, c) in closed.items())


def warn_timing(payload: dict | None) -> list[str]:
    """Log (and return) the assets of a recorded payload whose next session had already closed when it was written:
    rows recorded for them are not forward-test evidence; assets without rows got no forecast for that reason."""
    if not payload:
        return []
    late = {a: t for a, t in payload_timing(payload).items() if t["state"] == "after_close"}
    rec = recorded_utc(payload)
    when = f"{rec:%Y-%m-%d %H:%M} UTC" if rec else "an unknown time"

    def names(items: dict) -> str:
        return ", ".join(f"{a} {t['session']} (closed {t['close_utc']:%H:%M} UTC)" for a, t in items.items())

    with_rows = {a: t for a, t in late.items() if t["rows"]}
    without = {a: t for a, t in late.items() if not t["rows"]}
    if with_rows:
        log.warning("run %s was written at %s, after the next session of %s had closed: their outcome already existed, "
                    "so these forecasts are not forward-test evidence", payload.get("run_id"), when, names(with_rows))
    if without:
        log.warning("run %s (written at %s) has no forecast for %s: the next session had already closed (run in the "
                    "early UTC morning, see scripts/register_daily_task.ps1)", payload.get("run_id"), when,
                    names(without))
    return list(late)


def is_project_ledger() -> bool:
    """True when the ledger in use is the project's ``forecasts/ledger.jsonl`` (not a test sandbox): by this module's
    import-time path or by ``ledger.is_project_ledger``; either one is enough (fail-safe)."""
    if Path(paths.LEDGER).resolve() == Path(_PROJECT_LEDGER).resolve():
        return True
    try:
        return bool(_mod("ledger").is_project_ledger())
    except (ImportError, AttributeError):
        return False


# --------------------------------------------------------------------------------------------- verification
_PASS = {"PASS", "PASSED", "OK", "TRUE"}
_FAIL = {"FAIL", "FAILED", "ERROR", "FALSE"}


def _status(raw: Any) -> tuple[str, str | None]:
    """Normalised status (PASS / FAIL / SKIPPED / other upper-case word) and a reason given inside the status."""
    if isinstance(raw, bool):
        return ("PASS" if raw else "FAIL"), None
    s = str(raw if raw is not None else "").strip()
    head, _, rest = s.partition("(")
    word = head.strip().upper()
    reason = rest.rstrip(")").strip() or None
    if word in _PASS:
        return "PASS", reason
    if word in _FAIL:
        return "FAIL", reason
    if word.startswith("SKIP"):
        return "SKIPPED", reason
    return (word or "UNKNOWN"), reason


def _as_dict(x: Any) -> Any:
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return dataclasses.asdict(x)
    if hasattr(x, "to_dict") and callable(x.to_dict):
        return x.to_dict()
    return x


def normalise_checks(obj: Any) -> list[dict]:
    """Checks of a verification result or ``verification.json`` as ``[{id, title, status, reason, method,
    evidence}]`` in report order. Accepts ``{"checks": [...]}``, ``{"checks": {id: check}}``, a list of checks or
    an ``{id: check}`` mapping; a check is a dict (or dataclass) whose status is PASS / FAIL / SKIPPED (reason)."""
    obj = _as_dict(obj)
    if obj is None:
        return []
    items: Any = obj
    if isinstance(obj, dict):
        key = next((k for k in ("checks", "results") if k in obj), None)
        if key:
            items = obj[key]
        elif "status" in obj:
            items = [obj]
    if isinstance(items, dict):  # {id: check}; non-dict values are metadata, not checks
        items = [{"id": k, **v} for k, v in ((k, _as_dict(v)) for k, v in items.items()) if isinstance(v, dict)]
    if not isinstance(items, list):
        return []
    out = []
    for i, c in enumerate(items, start=1):
        c = _as_dict(c)
        if not isinstance(c, dict):
            continue
        status, why = _status(c.get("status", c.get("result", c.get("ok"))))
        cid = c.get("id", c.get("key", c.get("number", i)))
        out.append({
            "id": str(cid),
            "title": str(c.get("title") or c.get("name") or c.get("check") or cid),
            "status": status,
            "reason": c.get("reason") or why,
            "method": c.get("method") or c.get("description") or c.get("how"),
            "evidence": c.get("evidence", c.get("details", c.get("data"))),
        })
    return out


def failed_checks(result: Any) -> list[str]:
    """Titles of the FAIL checks in a verification result (the ``verify`` exit code)."""
    return [c["title"] for c in normalise_checks(result) if c["status"] == "FAIL"]


# --------------------------------------------------------------------------------------------- commands
def cmd_update(now: datetime | None) -> Any:
    res = _mod("update").update(now_utc=now)
    if isinstance(res, dict):
        check = res.get("check")
        log.info("update done: data end %s, last sessions %s, stale %s, consistency check %s",
                 res.get("end"), _brief(res.get("last_sessions")), res.get("stale_assets") or "none",
                 "skipped" if check is None else ("passed" if isinstance(check, dict) and check.get("ok")
                                                  else "FAILED"))
    else:
        log.info("update done: %s", _brief(res))
    return res


def cmd_forecast(now: datetime | None, workers: int, stamp: bool) -> str:
    """``forecast.forecast``; warns before the walk-forward (frozen calendar) and after it (the written payload) when
    the next session of an asset has closed, or closes within ``PREFLIGHT_MARGIN``: a forecast for it cannot precede
    its outcome (``forecast`` leaves such an asset out; any row recorded for it anyway is flagged)."""
    closed = closed_targets(now or datetime.now(UTC), PREFLIGHT_MARGIN)
    if closed:
        log.warning("the next session of %s closes before this run can record its forecast: no forward-test forecast "
                    "for it from this run (run in the early UTC morning, see scripts/register_daily_task.ps1)",
                    _closed_text(closed))
    run_id = _mod("forecast").forecast(now_utc=now, workers=workers, stamp=stamp)
    log.info("forecast run %s recorded: %s", run_id, _rel(paths.RUNS / f"{run_id}.json"))
    try:
        warn_timing(load_payload(run_id))
    except (OSError, ValueError):
        log.exception("could not read the payload of run %s", run_id)
    return run_id


def cmd_score(now: datetime | None, report: bool = True) -> Any:
    """``score.score_all``; with ``report`` it also rewrites ``reports/live/forward_test.md``."""
    res = _call(_mod("score").score_all, report=report, now=now)
    log.info("scoring done: %s", _brief(res))
    return res


def cmd_report(now: datetime | None) -> Any:
    res = _call(_mod("score").report_forward, now=now)
    log.info("forward-test report written: %s", _rel(paths.FORWARD_MD))
    return res


def cmd_verify(full: bool, workers: int, now: datetime | None) -> int:
    res = _call(_mod("verify").run_all, must=("full",) if full else (), full=full, workers=workers, now_utc=now)
    if isinstance(res, int) and not isinstance(res, bool):
        return res
    checks = normalise_checks(res)
    if not checks and paths.VERIFY_JSON.exists():  # the module returned nothing usable: read what it wrote
        checks = normalise_checks(json.loads(paths.VERIFY_JSON.read_text(encoding="utf-8")))
    counts = {s: sum(c["status"] == s for c in checks) for s in ("PASS", "FAIL", "SKIPPED")}
    for c in checks:  # verify logs every check itself; repeat the failures next to the verdict
        if c["status"] == "FAIL":
            log.error("FAIL %s. %s%s", c["id"], c["title"], f" ({c['reason']})" if c["reason"] else "")
    log.info("verification: %d checks - %d PASS, %d FAIL, %d SKIPPED - %s", len(checks), counts["PASS"],
             counts["FAIL"], counts["SKIPPED"], _rel(paths.VERIFY_MD))
    return 1 if counts["FAIL"] else 0


def cmd_stamp() -> Any:
    """``ots.upgrade_all``: stamp payloads whose stamp is missing or partial (a network failure earlier), then ask
    the calendars for the Bitcoin attestation of the pending ones. Network failures never abort (LIVE_SPEC §4)."""
    try:
        res = _mod("ots").upgrade_all()
    except Exception as e:  # noqa: BLE001 - upgrade_all should not raise; a stamp is retried by the next run
        log.warning("timestamp upgrade failed (%s: %s); retried by the next run", type(e).__name__, e)
        return None
    counts = res.get("counts") if isinstance(res, dict) else None
    log.info("timestamps of %s payload(s): %s", res.get("n") if isinstance(res, dict) else "?",
             _brief(counts if counts is not None else res))
    return res


def cmd_dashboard(port: int, no_browser: bool, open_browser: bool) -> int:
    argv = ["--port", str(port)] + (["--no-browser"] if no_browser else [])
    _mod("dashboard").main(argv, open_browser=open_browser)
    return 0


# --------------------------------------------------------------------------------------------- daily
def load_payload(run_id: str) -> dict | None:
    p = paths.RUNS / f"{run_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _fmt(x: Any, digits: int = 2) -> str:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "-"
    return f"{v:.{digits}f}" if v == v else "-"


def summary_lines(payload: dict | None, run_id: str | None, steps: dict[str, str]) -> list[str]:
    """The human summary printed by ``daily``: where the files are, then per asset the next session, the COMBO
    volatility forecasts (annualised, display only) and the next-session COMBO+FHS VaR/ES."""
    lines = [f"volrisk-live daily - run {run_id or '(no new forecast)'}"]
    lines.append("  steps     " + ", ".join(f"{k}: {v}" for k, v in steps.items()))
    if run_id:
        p = paths.RUNS / f"{run_id}.json"
        proof = "stamped" if ots_path(p).exists() else "no .ots proof yet"
        lines.append(f"  payload   {_rel(p)} ({proof})")
    lines.append(f"  ledger    {_rel(paths.LEDGER)}")
    lines.append(f"  reports   {_rel(paths.TOMORROW_MD)}, {_rel(paths.FORWARD_MD)}")
    lines.append(f"  scores    {_rel(paths.SCORES)}, {_rel(paths.RISK_SCORES)}")
    lines.append("  dashboard uv run python -m volrisk_live dashboard")
    if not payload:
        return lines
    data = payload.get("data") or {}
    fc = [r for r in payload.get("forecasts") or [] if r.get("model") == PRIMARY_FORECAST]
    rk = [r for r in payload.get("risk") or [] if r.get("model") == PRIMARY_RISK]
    timing = payload_timing(payload)
    assets = list(dict.fromkeys([*data, *(r.get("asset") for r in fc), *(r.get("asset") for r in rk)]))
    lines.append("")
    lines.append(f"  {'asset':<7} {'data to':<10} {'next':<10} "
                 f"{'COMBO vol % p.a. 1d / 1w / 1m':>30}   {'COMBO+FHS VaR99 / VaR97.5 / ES97.5 %':>36}")
    for a in assets:
        d = data.get(a) or {}
        t = timing.get(a) or {}
        vol = {r.get("horizon"): r.get("vol_ann") for r in fc if r.get("asset") == a}
        one = next((r for r in fc if r.get("asset") == a and r.get("horizon") == "1d"), {})
        risk = next((r for r in rk if r.get("asset") == a), {})
        nxt = d.get("next_session") or one.get("window_first") or risk.get("date") or "-"
        last = d.get("last_session") or "-"
        note = ("stale data: no forecast for this asset" if d.get("stale") else
                "next session closed before the run: no forecast" if t.get("state") == "after_close"
                and not t.get("rows") else None)
        if note:
            lines.append(f"  {a:<7} {str(last)[:10]:<10} {str(nxt)[:10]:<10} {note:>69}")
            continue
        vols = " / ".join(_fmt(vol.get(h), 1) for h in HORIZONS)
        var = " / ".join(_fmt(risk.get(k)) for k in ("var99", "var975", "es975"))
        lines.append(f"  {a:<7} {str(last)[:10]:<10} {str(nxt)[:10]:<10} {vols:>30}   {var:>36}")
    lines.append("  (volatility annualised for display only; VaR/ES = loss in % of a 1-unit long position, "
                 "next session)")
    if timing:
        rec = recorded_utc(payload)
        head = f"{rec:%Y-%m-%d %H:%M} UTC (payload written)" if rec else "time n/a"
        lines.append(f"  recorded  {head}; next session at that time: "
                     + ", ".join(f"{a} {TIMING_TEXT[t['state']]}" for a, t in timing.items()))
        late = [a for a, t in timing.items() if t["state"] == "after_close" and t["rows"]]
        if late:
            lines.append(f"  WARNING   {', '.join(late)}: recorded after the next session had closed (its outcome "
                         "existed) - not forward-test evidence")
    return lines


def cmd_daily(now: datetime | None, workers: int, stamp: bool) -> int:
    """update → score → stamp/upgrade earlier payloads → forecast (+ stamp) → forward-test report → summary.

    A failed update aborts (forecasts are never produced on stale or drifted data); a failed scoring, forecast or
    report step is logged and the run continues, but the exit code is then 1. Unreachable timestamp calendars are
    not an error (the proofs are retried by the next run). A run so late in the UTC day that the next session of an
    asset has closed (``forecast`` records no forecast for that asset) is logged as a warning and in the summary.
    """
    steps: dict[str, str] = {}
    run_id = None
    try:
        cmd_update(now)
        steps["update"] = "ok"
    except Exception:
        log.exception("update failed - no forecast is produced on stale or unverified data")
        steps["update"] = "FAILED"
        print("\n".join(summary_lines(None, None, steps)), flush=True)
        return 1
    try:
        cmd_score(now, report=False)
        steps["score"] = "ok"
    except Exception:
        log.exception("scoring failed")
        steps["score"] = "FAILED"
    if stamp:  # network trouble is expected now and then: logged, retried tomorrow, not an error
        steps["timestamps"] = "ok" if cmd_stamp() is not None else "unreachable, retried next run"
    else:
        steps["timestamps"] = "skipped (--no-stamp)"
    try:
        run_id = cmd_forecast(now, workers, stamp)
        steps["forecast"] = "ok"
    except Exception:
        log.exception("forecast failed")
        steps["forecast"] = "FAILED"
    try:
        cmd_report(now)
        steps["report"] = "ok"
    except Exception:
        log.exception("forward-test report failed")
        steps["report"] = "FAILED"
    payload = None
    if run_id:
        try:
            payload = load_payload(run_id)
        except (OSError, ValueError):
            log.exception("could not read the payload of run %s", run_id)
    print("\n" + "\n".join(summary_lines(payload, run_id, steps)) + "\n", flush=True)
    return 1 if any(v == "FAILED" for v in steps.values()) else 0


# --------------------------------------------------------------------------------------------- main
def _safe_streams() -> None:
    """A redirected stdout/stderr on Windows uses the ANSI code page (cp1251 here): replace characters it cannot
    encode instead of crashing a scheduled run on a '≥' in some log message."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):  # not a TextIOWrapper (pytest capture) or already detached
            pass


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m volrisk_live", description=HELP,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=COMMANDS)
    p.add_argument("--now", type=parse_now, default=None,
                   help="pretend the current time is this UTC ISO time (test sandboxes only: refused by forecast and "
                        "daily on the project ledger); default: the real clock")
    p.add_argument("--workers", type=int, default=10, help="processes for the walk-forward (forecast, verify --full)")
    p.add_argument("--no-stamp", action="store_true",
                   help="do not submit OpenTimestamps proofs or upgrade pending ones (offline)")
    p.add_argument("--full", action="store_true",
                   help="verify: also re-run the complete dev walk-forward in memory (slow)")
    p.add_argument("--port", type=int, default=8050, help="dashboard: port on 127.0.0.1 (default 8050)")
    p.add_argument("--no-browser", action="store_true", help="dashboard: do not open the web browser")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None, open_browser: bool = False) -> int:
    """Run one command; the dashboard opens the browser only from the command line entry point."""
    _safe_streams()
    p = build_parser()
    a = p.parse_args(argv)
    if a.workers < 1:
        p.error("--workers must be at least 1")
    if a.now is not None and a.command in ("forecast", "daily") and is_project_ledger():
        # run_utc / run_id and the before-open flags come from this time: the project ledger only takes the real clock
        p.error(f"--now is refused for {a.command!r} on the project ledger (forecasts/ledger.jsonl): a recorded run "
                "always carries the real clock time. --now is for test sandboxes only.")
    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for flag, used_by in (("full", ("verify",)), ("no_stamp", ("forecast", "daily")),
                          ("no_browser", ("dashboard",))):
        if getattr(a, flag) and a.command not in used_by:
            log.info("--%s has no effect on command %r", flag.replace("_", "-"), a.command)
    c, now, stamp = a.command, a.now, not a.no_stamp
    if c == "update":
        cmd_update(now)
    elif c == "forecast":
        cmd_forecast(now, a.workers, stamp)
    elif c == "score":
        cmd_score(now)
    elif c == "verify":
        return cmd_verify(a.full, a.workers, now)
    elif c == "stamp":
        cmd_stamp()
    elif c == "dashboard":
        return cmd_dashboard(a.port, a.no_browser, open_browser)
    elif c == "daily":
        return cmd_daily(now, a.workers, stamp)
    return 0
