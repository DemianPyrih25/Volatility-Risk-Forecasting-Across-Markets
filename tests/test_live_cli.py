"""Live CLI wiring (``python -m volrisk_live``, docs/LIVE_SPEC.md §8). Every live module is a stub: these tests never
touch ``data/``, ``forecasts/`` or the network, and never start a server or a browser."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import re
import runpy
import shutil
import subprocess
import sys
import types
from datetime import UTC, date, datetime, time, timedelta, timezone

import pytest

# ledger is imported before any test redirects paths.LEDGER (it captures the project ledger's path)
from volrisk_live import cli, context, ledger, paths

NOW = "2026-10-02T07:00:00Z"  # the scheduled time (scripts/register_daily_task.ps1)
NOW_DT = datetime(2026, 10, 2, 7, 0, tzinfo=UTC)
RUN_ID = "20261002T070000Z"
LATE = "2026-10-02T22:30:00Z"  # 23:30 local in London (BST): the old default schedule
LATE_DT = datetime(2026, 10, 2, 22, 30, tzinfo=UTC)
SCRIPT = paths.ROOT / "scripts" / "register_daily_task.ps1"


def _payload(now: datetime = NOW_DT) -> dict:
    """The parts of a ``volrisk-live/1`` payload the daily summary reads; written 20 minutes after ``now``."""
    fc, risk = [], []
    for asset, base in (("BTC", 40.0), ("SPX", 15.0)):
        for h, k in (("1d", 1.0), ("1w", 1.1), ("1m", 1.2)):
            for model in ("HAR", "COMBO"):
                fc.append({"asset": asset, "horizon": h, "model": model, "origin": "2026-10-01",
                           "window_first": "2026-10-02", "window_last": "2026-10-02", "n_t": 1, "F": 1.0,
                           "vol_ann": base * k + (0.5 if model == "HAR" else 0.0)})
        for model, scale in (("HS-250", 1.1), ("COMBO+FHS", 1.0)):
            risk.append({"asset": asset, "model": model, "date": "2026-10-02", "sigma": 1.0,
                         "var99": 2.33 * scale, "var975": 1.96 * scale, "es975": 2.34 * scale})
    data = {"BTC": {"last_session": "2026-10-01", "next_session": "2026-10-02", "stale": False},
            "SPX": {"last_session": "2026-10-01", "next_session": "2026-10-02", "stale": False},
            "EURUSD": {"last_session": "2026-09-29", "next_session": "2026-09-30", "stale": True}}
    return {"run_id": RUN_ID, "run_utc": now.isoformat(), "data": data, "forecasts": fc, "risk": risk,
            "checks": {"computed_utc": (now + timedelta(minutes=20)).isoformat()}}


@pytest.fixture
def live(tmp_path, monkeypatch):
    """Stub modules for update / forecast / score / verify / ots / dashboard, live paths under tmp_path; returns
    the ordered call log and a dict of knobs (``fail``: step names that raise, ``verify``: run_all result)."""
    for name in ("FORECASTS", "RUNS", "LEDGER", "SCORES", "RISK_SCORES", "LIVE_REPORTS", "TOMORROW_MD", "FORWARD_MD",
                 "VERIFY_MD", "VERIFY_JSON"):
        rel = getattr(paths, name).relative_to(paths.ROOT)
        monkeypatch.setattr(paths, name, tmp_path / rel)
    calls: list[tuple[str, dict]] = []
    knobs: dict = {"fail": set(), "verify": {"checks": [{"id": 1, "title": "Frozen code", "status": "PASS"}]}}

    def rec(name, ret=None):
        def fn(**kw):
            calls.append((name, kw))
            if name in knobs["fail"]:
                raise RuntimeError(f"{name} failed (stub)")
            return ret() if callable(ret) else ret
        return fn

    def forecast(now_utc=None, workers=10, stamp=True):
        calls.append(("forecast", {"now_utc": now_utc, "workers": workers, "stamp": stamp}))
        if "forecast" in knobs["fail"]:
            raise RuntimeError("forecast failed (stub)")
        paths.RUNS.mkdir(parents=True, exist_ok=True)
        (paths.RUNS / f"{RUN_ID}.json").write_text(json.dumps(_payload(now_utc or NOW_DT)), encoding="utf-8")
        return RUN_ID

    def score_all(daily=None, *, payloads=None, end=None, entries=None, report=True, now=None):
        return rec("score_all", {"forecasts": {"new": 3}})(report=report, now=now)

    def report_forward(scores=None, risk_scores=None, *, payloads=None, entries=None, data_end=None, out=None,
                       now=None, chain=None, timestamps=None):
        return rec("report_forward", paths.FORWARD_MD)(now=now)

    def run_all(full=False, workers=10):
        calls.append(("run_all", {"full": full, "workers": workers}))
        return knobs["verify"]

    def dash_main(argv=None, open_browser=False):
        calls.append(("dashboard", {"argv": argv, "open_browser": open_browser}))

    mods = {
        "update": {"update": lambda now_utc=None: rec("update", {"end": "2026-10-01", "check": {"ok": True}})(
            now_utc=now_utc)},
        "forecast": {"forecast": forecast},
        "score": {"score_all": score_all, "report_forward": report_forward},
        "verify": {"run_all": run_all},
        "ots": {"upgrade_all": lambda: rec("upgrade_all", {"n": 1, "counts": {"pending": 1}})()},
        "dashboard": {"main": dash_main},
    }
    for name, attrs in mods.items():
        m = types.ModuleType(f"volrisk_live.{name}")
        for k, v in attrs.items():
            setattr(m, k, v)
        monkeypatch.setitem(sys.modules, f"volrisk_live.{name}", m)
    return calls, knobs


def _names(calls):
    return [c[0] for c in calls]


# --------------------------------------------------------------------------------------------- arguments
def test_now_is_parsed_as_utc():
    assert cli.parse_now(NOW) == NOW_DT
    assert cli.parse_now("2026-10-02T07:00") == NOW_DT  # naive = UTC
    assert cli.parse_now("2026-10-02T10:00:00+03:00") == NOW_DT  # converted to UTC
    assert cli.parse_now("2026-10-02") == datetime(2026, 10, 2, tzinfo=UTC)
    assert cli.parse_now(NOW).utcoffset() == timedelta(0)


@pytest.mark.parametrize("argv", [["update", "--now", "yesterday"], ["bogus"], ["forecast", "--workers", "0"]])
def test_bad_arguments_exit_with_usage_error(live, argv):
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2
    assert live[0] == []


def test_help_text_is_ascii():
    # a scheduled run writes to a cp1251 log/console on Windows: the help and summary must encode anywhere
    cli.build_parser().format_help().encode("ascii")


# --------------------------------------------------------------------------------------------- single commands
def test_update_passes_now(live):
    calls, _ = live
    assert cli.main(["update", "--now", NOW]) == 0
    assert calls == [("update", {"now_utc": NOW_DT})]


def test_update_without_now_uses_the_real_clock(live):
    calls, _ = live
    assert cli.main(["update"]) == 0
    assert calls == [("update", {"now_utc": None})]


def test_forecast_passes_workers_and_no_stamp(live):
    calls, _ = live
    assert cli.main(["forecast", "--workers", "3", "--no-stamp", "--now", NOW]) == 0
    assert calls == [("forecast", {"now_utc": NOW_DT, "workers": 3, "stamp": False})]
    assert cli.main(["forecast"]) == 0
    assert calls[-1] == ("forecast", {"now_utc": None, "workers": 10, "stamp": True})


def test_score_scores_and_writes_the_report(live):
    calls, _ = live
    assert cli.main(["score", "--now", NOW]) == 0
    assert calls == [("score_all", {"report": True, "now": NOW_DT})]


def test_stamp_upgrades_and_restamps_through_ots(live):
    calls, _ = live
    assert cli.main(["stamp"]) == 0
    assert _names(calls) == ["upgrade_all"]


def test_stamp_never_fails_on_network_trouble(live):
    calls, knobs = live
    knobs["fail"].add("upgrade_all")
    assert cli.main(["stamp"]) == 0


def test_dashboard_forwards_port_and_never_opens_a_browser_from_main(live):
    calls, _ = live
    assert cli.main(["dashboard", "--port", "9123", "--no-browser"]) == 0
    assert calls == [("dashboard", {"argv": ["--port", "9123", "--no-browser"], "open_browser": False})]


def test_module_entry_point_opens_the_browser_only_for_the_command_line(live, monkeypatch):
    calls, _ = live
    monkeypatch.setattr(sys, "argv", ["volrisk_live", "dashboard", "--port", "9124"])
    with pytest.raises(SystemExit) as e:
        runpy.run_module("volrisk_live", run_name="__main__")
    assert e.value.code == 0
    assert calls == [("dashboard", {"argv": ["--port", "9124"], "open_browser": True})]


def test_flags_without_effect_are_logged(live, caplog):
    with caplog.at_level("INFO", logger="volrisk_live"):
        assert cli.main(["update", "--full", "--no-stamp"]) == 0
    assert "--full has no effect on command 'update'" in caplog.text
    assert "--no-stamp has no effect on command 'update'" in caplog.text


# --------------------------------------------------------------------------------------------- verify
def test_verify_exit_code_follows_the_checks(live):
    calls, knobs = live
    knobs["verify"] = {"checks": [{"id": 1, "title": "Frozen code", "status": "PASS"},
                                  {"id": 4, "title": "Independent sources", "status": "SKIPPED (offline)"}]}
    assert cli.main(["verify"]) == 0
    knobs["verify"]["checks"].append({"id": 6, "title": "Negative controls", "status": "FAIL"})
    assert cli.main(["verify", "--full", "--workers", "4"]) == 1
    assert calls[-1] == ("run_all", {"full": True, "workers": 4})


def test_verify_reads_the_json_when_run_all_returns_nothing(live):
    _, knobs = live
    knobs["verify"] = None
    paths.VERIFY_JSON.parent.mkdir(parents=True, exist_ok=True)
    paths.VERIFY_JSON.write_text(json.dumps({"checks": [{"id": 2, "title": "Holdout once", "status": "FAIL"}]}),
                                 encoding="utf-8")
    assert cli.main(["verify"]) == 1


def test_verify_full_fails_loudly_when_run_all_cannot_take_it(live):
    sys.modules["volrisk_live.verify"].run_all = lambda: {"checks": []}
    with pytest.raises(TypeError):
        cli.main(["verify", "--full"])


def test_normalise_checks_accepts_the_usual_shapes():
    as_list = [{"id": 1, "title": "Frozen code unchanged", "status": "PASS", "method": "hashes",
                "evidence": {"bad": []}},
               {"id": 3, "name": "Raw data", "status": "SKIPPED (no network)"},
               {"id": 6, "title": "Negative controls", "status": False}]
    out = cli.normalise_checks({"generated_utc": "x", "checks": as_list})
    assert [c["status"] for c in out] == ["PASS", "SKIPPED", "FAIL"]
    assert out[1]["title"] == "Raw data" and out[1]["reason"] == "no network"
    assert out[0]["method"] == "hashes" and out[0]["evidence"] == {"bad": []}
    by_id = cli.normalise_checks({"frozen": {"status": "ok"}, "ledger": {"status": "failed", "reason": "edited"},
                                  "generated_utc": "2026-10-02"})
    assert [(c["id"], c["status"], c["reason"]) for c in by_id] == [("frozen", "PASS", None),
                                                                   ("ledger", "FAIL", "edited")]
    assert cli.failed_checks(as_list) == ["Negative controls"]
    assert cli.normalise_checks(None) == [] and cli.normalise_checks("garbage") == []


# --------------------------------------------------------------------------------------------- daily
def test_daily_runs_the_steps_in_order_and_prints_a_summary(live, capsys):
    calls, _ = live
    assert cli.main(["daily", "--now", NOW, "--workers", "2"]) == 0
    assert _names(calls) == ["update", "score_all", "upgrade_all", "forecast", "report_forward"]
    assert calls[1] == ("score_all", {"report": False, "now": NOW_DT})  # the report follows the new forecast
    assert calls[3] == ("forecast", {"now_utc": NOW_DT, "workers": 2, "stamp": True})
    out = capsys.readouterr().out
    out.encode("ascii")
    assert RUN_ID in out and "forecasts/runs/" + RUN_ID + ".json" in out
    assert "reports/live/tomorrow.md" in out and "forecasts/ledger.jsonl" in out
    btc = next(line for line in out.splitlines() if line.strip().startswith("BTC"))
    assert "2026-10-01" in btc and "2026-10-02" in btc
    assert "40.0 / 44.0 / 48.0" in btc  # COMBO vol 1d / 1w / 1m from the payload, not HAR's
    assert "2.33 / 1.96 / 2.34" in btc  # COMBO+FHS VaR99 / VaR97.5 / ES97.5
    eur = next(line for line in out.splitlines() if line.strip().startswith("EURUSD"))
    assert "stale data: no forecast" in eur
    # written 07:20 UTC: BTC's UTC-day session is running, the S&P 500 opens at 13:30 UTC
    assert ("recorded  2026-10-02 07:20 UTC (payload written); next session at that time: BTC in session, "
            "SPX before open") in out
    assert "WARNING" not in out


def test_daily_without_stamp_stays_offline(live):
    calls, _ = live
    assert cli.main(["daily", "--no-stamp"]) == 0
    assert "upgrade_all" not in _names(calls)
    assert ("forecast", {"now_utc": None, "workers": 10, "stamp": False}) in calls


def test_daily_aborts_when_the_update_fails(live, capsys):
    calls, knobs = live
    knobs["fail"].add("update")
    assert cli.main(["daily", "--now", NOW]) == 1
    assert _names(calls) == ["update"]  # never a forecast on stale or unverified data
    assert "update: FAILED" in capsys.readouterr().out


def test_daily_still_forecasts_when_scoring_fails_but_exits_1(live):
    calls, knobs = live
    knobs["fail"].add("score_all")
    assert cli.main(["daily", "--now", NOW]) == 1
    assert _names(calls) == ["update", "score_all", "upgrade_all", "forecast", "report_forward"]


def test_daily_unreachable_calendars_are_not_an_error(live, capsys):
    calls, knobs = live
    knobs["fail"].add("upgrade_all")
    assert cli.main(["daily", "--now", NOW]) == 0
    assert "forecast" in _names(calls)
    assert "unreachable, retried next run" in capsys.readouterr().out


def test_daily_reports_a_failed_forecast(live, capsys):
    calls, knobs = live
    knobs["fail"].add("forecast")
    assert cli.main(["daily", "--now", NOW]) == 1
    assert _names(calls)[-1] == "report_forward"
    out = capsys.readouterr().out
    assert "forecast: FAILED" in out and "(no new forecast)" in out


@pytest.mark.parametrize("mod, fn, kwargs", [
    ("update", "update", {"now_utc": NOW_DT}),
    ("forecast", "forecast", {"now_utc": NOW_DT, "workers": 2, "stamp": False}),
    ("score", "score_all", {"report": False, "now": NOW_DT}),
    ("score", "report_forward", {"now": NOW_DT}),
    ("ots", "upgrade_all", {}),
    ("verify", "run_all", {"full": True, "workers": 2, "now_utc": NOW_DT}),
    ("dashboard", "main", {"argv": ["--port", "1"], "open_browser": False}),
    ("update", "live_end_date", {"now_utc": NOW_DT}),
    ("ledger", "is_project_ledger", {}),
])
def test_real_modules_accept_what_the_cli_passes(mod, fn, kwargs):
    # interface contract with the real modules (signatures only; nothing is run)
    f = getattr(importlib.import_module(f"volrisk_live.{mod}"), fn)
    accepted = set(inspect.signature(f).parameters)
    assert set(kwargs) <= accepted, f"{mod}.{fn} no longer accepts {set(kwargs) - accepted}"
    inspect.signature(f).bind(**kwargs)


def test_summary_lines_without_payload_lists_the_files():
    lines = cli.summary_lines(None, None, {"update": "ok"})
    assert lines[0].endswith("(no new forecast)")
    assert any("forecasts/ledger.jsonl" in s for s in lines)


def test_parse_now_rejects_garbage():
    with pytest.raises(argparse.ArgumentTypeError):
        cli.parse_now("not a time")
    assert cli.parse_now("2026-10-02T23:30:00-02:00") == datetime(2026, 10, 3, 1, 30, tzinfo=timezone.utc)


# --------------------------------------------------------------------------------------------- --now (back-dating)
@pytest.mark.parametrize("argv", [["forecast", "--now", NOW], ["daily", "--now", NOW, "--no-stamp"]])
def test_now_is_refused_for_recording_commands_on_the_project_ledger(live, monkeypatch, capsys, argv):
    # run_utc / run_id and the before-open flags come from --now: a back-dated run would look recorded earlier
    calls, _ = live
    monkeypatch.setattr(paths, "LEDGER", ledger.PROJECT_LEDGER)
    assert cli.is_project_ledger()
    with pytest.raises(SystemExit) as e:
        cli.main(argv)
    assert e.value.code == 2 and calls == []  # refused before any step runs
    assert "--now is refused" in capsys.readouterr().err


def test_now_still_works_in_sandboxes_and_for_update(live, monkeypatch):
    calls, _ = live
    assert not cli.is_project_ledger()  # the fixture's ledger lives under tmp_path
    assert cli.main(["forecast", "--now", NOW, "--no-stamp"]) == 0
    monkeypatch.setattr(paths, "LEDGER", ledger.PROJECT_LEDGER)
    assert cli.main(["update", "--now", NOW]) == 0  # only picks the live data end; records nothing
    assert _names(calls) == ["forecast", "update"]


def test_project_ledger_check_is_fail_safe(live, monkeypatch):
    # either the CLI's own import-time path or ledger.is_project_ledger() is enough to refuse --now
    calls, _ = live
    monkeypatch.setattr(ledger, "is_project_ledger", lambda: True)
    with pytest.raises(SystemExit):
        cli.main(["forecast", "--now", NOW])
    assert calls == []


# --------------------------------------------------------------------------------------------- recording time
def _late_payload(rows_for_closed: bool) -> dict:
    """A run written at 22:50 UTC: BTC's session is still running, EUR/USD and SPX closed at 21:00 / 20:00 UTC."""
    p = _payload(LATE_DT)
    p["data"]["EURUSD"] = {"last_session": "2026-10-01", "next_session": "2026-10-02", "stale": False}
    if not rows_for_closed:  # forecast.py leaves out an asset whose session has closed (checks.closed_assets)
        p["forecasts"] = [r for r in p["forecasts"] if r["asset"] == "BTC"]
        p["risk"] = [r for r in p["risk"] if r["asset"] == "BTC"]
    return p


def test_recorded_time_is_the_later_of_start_and_write_time():
    p = {"run_utc": "2026-10-02T13:00:00+00:00", "checks": {"computed_utc": "2026-10-02T13:45:00+00:00"}}
    assert cli.recorded_utc(p) == datetime(2026, 10, 2, 13, 45, tzinfo=UTC)
    p["checks"]["timing_utc"] = "2026-10-02T13:46:00+00:00"
    assert cli.recorded_utc(p) == datetime(2026, 10, 2, 13, 46, tzinfo=UTC)
    assert cli.recorded_utc({"run_utc": "2026-10-02T13:00:00Z"}) == datetime(2026, 10, 2, 13, 0, tzinfo=UTC)
    assert cli.recorded_utc({}) is None
    # a start time before the S&P 500 open does not count when the payload was written after it
    spx = cli.payload_timing({**p, "data": {"SPX": {"next_session": "2026-10-02", "stale": False}}})["SPX"]
    assert spx["state"] == "in_session" and spx["open_utc"] == datetime(2026, 10, 2, 13, 30, tzinfo=UTC)


def test_frozen_calendar_session_bounds():
    assert cli.session_bounds("SPX", "2026-10-02") == (datetime(2026, 10, 2, 13, 30, tzinfo=UTC),
                                                        datetime(2026, 10, 2, 20, 0, tzinfo=UTC))
    assert cli.session_bounds("SPX", "2026-11-26") is None  # Thanksgiving
    assert cli.session_bounds("SPX", "2026-11-27")[1] == datetime(2026, 11, 27, 18, 0, tzinfo=UTC)  # early close
    assert cli.session_bounds("EURUSD", "2026-11-10") == (datetime(2026, 11, 9, 22, 0, tzinfo=UTC),
                                                           datetime(2026, 11, 10, 22, 0, tzinfo=UTC))
    assert cli.session_bounds("NOPE", "2026-10-02") is None and cli.session_bounds("BTC", "garbage") is None


def test_late_run_rows_are_flagged_in_the_summary_and_the_log(caplog):
    p = _late_payload(rows_for_closed=True)
    timing = cli.payload_timing(p)
    assert {a: t["state"] for a, t in timing.items()} == {"BTC": "in_session", "SPX": "after_close",
                                                         "EURUSD": "after_close"}
    out = "\n".join(cli.summary_lines(p, RUN_ID, {"forecast": "ok"}))
    out.encode("ascii")
    assert "recorded  2026-10-02 22:50 UTC (payload written)" in out
    assert "BTC in session, SPX AFTER CLOSE, EURUSD AFTER CLOSE" in out
    assert "WARNING   SPX: recorded after the next session had closed" in out  # EURUSD has no rows here
    with caplog.at_level("WARNING", logger="volrisk_live"):
        assert sorted(cli.warn_timing(p)) == ["EURUSD", "SPX"]
    assert "SPX 2026-10-02 (closed 20:00 UTC)" in caplog.text and "not forward-test evidence" in caplog.text
    assert "no forecast for EURUSD 2026-10-02 (closed 21:00 UTC)" in caplog.text


def test_assets_left_out_after_their_close_are_named_not_flagged():
    out = "\n".join(cli.summary_lines(_late_payload(rows_for_closed=False), RUN_ID, {"forecast": "ok"}))
    spx = next(line for line in out.splitlines() if line.strip().startswith("SPX"))
    assert "next session closed before the run: no forecast" in spx
    assert "WARNING" not in out


def test_closed_targets_uses_the_frozen_calendar_and_the_margin(live):
    # live data end = the previous UTC day, so the targets are the sessions of the run's UTC day
    assert cli.closed_targets(NOW_DT, cli.PREFLIGHT_MARGIN) == {}
    late = cli.closed_targets(LATE_DT, cli.PREFLIGHT_MARGIN)
    assert set(late) == {"EURUSD", "SPX"}  # BTC / ETH close at 24:00 UTC
    assert late["SPX"] == (date(2026, 10, 2), datetime(2026, 10, 2, 20, 0, tzinfo=UTC))
    assert set(cli.closed_targets(datetime(2026, 10, 2, 19, 30, tzinfo=UTC), cli.PREFLIGHT_MARGIN)) == {"SPX"}
    assert cli.closed_targets(datetime(2026, 10, 2, 19, 30, tzinfo=UTC)) == {}  # without the margin: still open


def test_forecast_warns_before_and_after_a_late_run(live, caplog):
    calls, _ = live
    with caplog.at_level("WARNING", logger="volrisk_live"):
        assert cli.main(["forecast", "--now", LATE, "--no-stamp"]) == 0
    assert _names(calls) == ["forecast"]  # forecast.py itself leaves the closed assets out
    assert "closes before this run can record its forecast" in caplog.text  # before the walk-forward
    assert "not forward-test evidence" in caplog.text  # the stub's payload still has SPX rows


def test_daily_at_a_late_hour_warns_in_the_log(live, caplog, capsys):
    calls, _ = live
    with caplog.at_level("WARNING", logger="volrisk_live"):
        assert cli.main(["daily", "--now", LATE]) == 0
    assert _names(calls) == ["update", "score_all", "upgrade_all", "forecast", "report_forward"]
    assert "EURUSD 2026-10-02 (closed 21:00 UTC), SPX 2026-10-02 (closed 20:00 UTC)" in caplog.text
    assert "SPX AFTER CLOSE" in capsys.readouterr().out


# --------------------------------------------------------------------------------------------- scheduled task
def _script_times() -> tuple[str, str, str, str]:
    text = SCRIPT.read_text(encoding="utf-8")
    default = re.search(r'\[string\]\$AtUtc\s*=\s*"(\d\d:\d\d)"', text)
    lo = re.search(r'\$EarliestUtc\s*=\s*\[timespan\]"(\d\d:\d\d)"', text)
    hi = re.search(r'\$LatestUtc\s*=\s*\[timespan\]"(\d\d:\d\d)"', text)
    assert default and lo and hi, "the task script must define -AtUtc and its accepted UTC window"
    return text, default.group(1), lo.group(1), hi.group(1)


def test_scheduled_run_records_every_next_session_before_it_closes():
    """The default and both ends of the accepted window: on every day from 2026-10-01 to 2027-04-30 (US and EU
    summer-time switches, Thanksgiving, Christmas, New Year), each asset's next session after the live data end
    closes more than PREFLIGHT_MARGIN after the run, and the S&P 500 opens after it."""
    text, default, lo, hi = _script_times()
    assert default == "07:00"
    assert "$Trigger.StartBoundary = " in text and '+ "Z"' in text  # a UTC trigger: no drift with summer time
    update = importlib.import_module("volrisk_live.update")  # the real live-end rule (last complete UTC day)
    for hhmm in (default, lo, hi):
        t = time.fromisoformat(hhmm)
        day = date(2026, 10, 1)
        while day <= date(2027, 4, 30):
            run = datetime.combine(day, t, tzinfo=UTC)
            nxt = cli.next_sessions(update.live_end_date(run))
            assert set(nxt) == {"BTC", "ETH", "EURUSD", "SPX"}
            for asset, (session, _o, c) in nxt.items():
                assert run + cli.PREFLIGHT_MARGIN < c, f"{hhmm} UTC on {day}: {asset} {session} closes {c}"
            assert run + cli.PREFLIGHT_MARGIN <= nxt["SPX"][1], f"{hhmm} UTC on {day}: SPX opens {nxt['SPX'][1]}"
            day += timedelta(days=1)
    # the old default, 23:30 local time on this GMT Standard Time machine (22:30 UTC in summer, 23:30 in winter)
    for run in (LATE_DT, datetime(2026, 11, 10, 23, 30, tzinfo=UTC)):
        assert set(cli.closed_targets(run)) == {"EURUSD", "SPX"}


def test_next_sessions_agree_with_the_forecast_module():
    fc = importlib.import_module("volrisk_live.forecast")
    for end in (date(2026, 10, 1), date(2026, 10, 2), date(2026, 11, 25), date(2026, 12, 24)):
        with context.live_end(end):
            want = {a: (fc.next_session(a, end), *fc.session_bounds(a, fc.next_session(a, end)))
                    for a in ("BTC", "ETH", "EURUSD", "SPX")}
        got = cli.next_sessions(end)
        assert {a: (d, cli.utc_time(o), cli.utc_time(c)) for a, (d, o, c) in want.items()} == got


@pytest.mark.skipif(shutil.which("powershell") is None, reason="Windows PowerShell not available")
def test_task_script_parses_without_registering_anything():
    cmd = ("$e = $null; $t = $null; [void][System.Management.Automation.Language.Parser]::ParseFile("
           f"'{SCRIPT}', [ref]$t, [ref]$e); $e.Count")
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", cmd], capture_output=True,
                       text=True, timeout=120)
    assert r.stdout.strip() == "0", r.stdout + r.stderr
