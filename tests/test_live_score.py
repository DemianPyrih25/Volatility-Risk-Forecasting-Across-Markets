"""Forward-test scoring (docs/LIVE_SPEC.md §6) on synthetic payloads and a synthetic live gold table.

Checked here:

- window completeness: 1d = the first session after the origin (weekends included); 1w/1m only once the live data
  end date reaches ``origin + days`` and the asset's own data reach the window's last scheduled session;
- ``y = Σ tv`` over the realised sessions, per-session means when an invalid session was dropped, QLIKE equal to
  the SPEC §8 formula;
- risk rows: realised ``r_cc`` of the first session after the run's last session, breach flags;
- idempotent, append-only CSVs (a key is scored once; later data never rewrite a score);
- the report: 'first scores after <date>' with nothing scored, the ratio table vs HAR with COMBO highlighted, VaR
  breaches vs expected, chain and timestamp status;
- integrity: every stored score is re-derived (an edited score file is caught, left out of the tables and fails
  the run); only the verified ledger prefix is scored (a forged entry is not); a payload written after its first
  target session closed (``checks.computed_utc``) is left out of the tables; scoring refuses live data whose
  consistency check did not pass or whose files changed since; data revised after a forecast (``rows_sha256``)
  give ``history_changed``;
- the ledger interface (``entries`` / ``load_payload`` / ``verify_chain``) through a stub and the real module.

Every output path is redirected to ``tmp_path``; the ledger, OTS and update modules are stubs (the real ledger
writes to ``tmp_path`` only), nothing touches the network, the real ``forecasts/`` or the real live data.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import types
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from volrisk_live import forecast as FC
from volrisk_live import paths, schema
from volrisk_live import score as S

try:  # imported before any stub is installed; used by the round-trip tests only
    from volrisk_live import ledger as REAL_LEDGER
except ImportError:  # pragma: no cover - the ledger module is built alongside this one
    REAL_LEDGER = None

NOW = datetime(2026, 11, 20, 7, 0, tzinfo=timezone.utc)
REAL_OUTPUTS = (paths.SCORES, paths.RISK_SCORES, paths.FORWARD_MD)  # captured before any monkeypatch


# ============================================================================================= synthetic data
def _make_daily(end: str = "2026-11-15") -> pd.DataFrame:
    rows = []
    for i, d in enumerate(pd.date_range("2026-08-01", end, freq="D")):
        rows.append(("BTC", d, 1.0 + 0.25 * (i % 7), 0.8 * math.sin(i)))
    for i, d in enumerate(pd.bdate_range("2026-08-03", end)):
        if d == pd.Timestamp("2026-10-07"):
            continue  # an invalid SPX session, dropped from the gold table
        rows.append(("SPX", d, 0.5 + 0.1 * (i % 5), 0.6 * math.cos(i)))
    df = pd.DataFrame(rows, columns=["asset", "session_date", "tv", "r_cc"])
    df.loc[(df["asset"] == "BTC") & (df["session_date"] == "2026-10-02"), "r_cc"] = -2.0
    df["session_date"] = df["session_date"].astype("datetime64[ms]")
    return df.sort_values(["asset", "session_date"]).reset_index(drop=True)


DAILY = _make_daily()


def upto(end: str, spx_end: str | None = None) -> pd.DataFrame:
    d = DAILY[DAILY["session_date"] <= pd.Timestamp(end)]
    if spx_end is not None:
        d = d[(d["asset"] != "SPX") | (d["session_date"] <= pd.Timestamp(spx_end))]
    return d.reset_index(drop=True)


def tv(asset: str, *days: str) -> float:
    d = DAILY[(DAILY["asset"] == asset) & DAILY["session_date"].isin(pd.to_datetime(list(days)))]
    assert len(d) == len(days), "a requested session is not in the synthetic table"
    return float(d["tv"].sum())


def ql(y: float, F: float) -> float:
    return y / F - math.log(y / F) - 1.0


def history_sha(daily: pd.DataFrame, asset: str, last: str) -> str:
    """``data[asset].rows_sha256`` as ``forecast.build_payload`` writes it (the independent implementation)."""
    return FC.rows_sha256(daily[(daily["asset"] == asset) & (daily["session_date"] <= pd.Timestamp(last))])


def frow(asset, horizon, model, origin, n_t, F, first=None, last=None) -> dict:
    return {"asset": asset, "horizon": horizon, "model": model, "origin": origin, "window_first": first,
            "window_last": last, "n_t": n_t, "F": F, "vol_ann": 1.0}


def rrow(asset, model, date, var99, var975, sigma=1.0, es975=None) -> dict:
    return {"asset": asset, "model": model, "date": date, "sigma": sigma, "var99": var99, "var975": var975,
            "es975": es975 if es975 is not None else var975 * 1.2}


def payload(run_id: str, last: dict[str, str], forecasts=(), risk=(), *, computed: str | None = "auto",
            history: pd.DataFrame | None = None) -> dict:
    """A payload as ``forecast.build_payload`` writes it: ``computed_utc`` 3 minutes after ``run_utc`` and
    ``timing_utc`` one minute later (``computed``: an explicit ``computed_utc`` without ``timing_utc``; None: neither),
    ``rows_sha256`` of ``history`` (default: the synthetic gold) up to each last session."""
    run_dt = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    hist = DAILY if history is None else history
    checks = ({"computed_utc": (run_dt + timedelta(minutes=3)).isoformat(),
               "timing_utc": (run_dt + timedelta(minutes=4)).isoformat()} if computed == "auto" else
              {} if computed is None else {"computed_utc": computed})
    p = {
        "schema": schema.SCHEMA,
        "run_id": run_id,
        "run_utc": run_dt.isoformat(timespec="seconds"),
        "frozen": {k: "x" for k in schema.FROZEN_KEYS},
        "live_code_sha": "0" * 64,
        "data": {a: {"last_session": d, "n_sessions": 100, "rows_sha256": history_sha(hist, a, d),
                     "recorded_before_open": False} for a, d in last.items()},
        "forecasts": list(forecasts),
        "risk": list(risk),
        "implied": [],
        "checks": checks,
    }
    return json.loads(schema.canonical_json(p))  # as stored in forecasts/runs/<run_id>.json


def install_ledger(monkeypatch, payloads: list[dict], chain=None, stamps=None) -> None:
    """Stub ``ledger`` / ``ots`` modules with the real call signatures and return shapes."""
    store = {p["run_id"]: p for p in payloads}
    led = types.ModuleType("volrisk_live.ledger")
    led.entries = lambda: [{"seq": i, "run_id": p["run_id"], "run_utc": p["run_utc"]} for i, p in enumerate(payloads)]
    led.load_payload = lambda entry: store[entry["run_id"]]
    led.verify_chain = lambda: ({"ok": True, "n": len(payloads), "first_bad": None, "reason": None, "head": "ab" * 32,
                                 "orphans": []} if chain is None else chain)
    ots = types.ModuleType("volrisk_live.ots")
    ots.status_all = lambda: ({"n": len(payloads), "counts": {"pending": len(payloads)} if payloads else {},
                               "files": {f"{p['run_id']}.json": {"status": "pending"} for p in payloads}}
                              if stamps is None else stamps)
    monkeypatch.setitem(sys.modules, "volrisk_live.ledger", led)
    monkeypatch.setitem(sys.modules, "volrisk_live.ots", ots)


def install_update(monkeypatch, tmp_path, daily: pd.DataFrame | None, *, ok: bool | None = True,
                   assets=("BTC", "SPX")) -> dict:
    """Stub ``update`` (``read_state`` / ``live_daily``) over two tmp live gold files hashed into the state."""
    dev, ho = tmp_path / "live" / "gold" / "daily.parquet", tmp_path / "live" / "holdout" / "daily.parquet"
    for p, content in ((dev, b"dev gold"), (ho, b"holdout gold")):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    monkeypatch.setattr(paths, "LIVE_DAILY_DEV", dev)
    monkeypatch.setattr(paths, "LIVE_DAILY_HOLDOUT", ho)
    monkeypatch.setattr(paths, "LIVE_STATE", tmp_path / "live" / "state.json")
    sha = {n: hashlib.sha256(p.read_bytes()).hexdigest() for n, p in (("live_dev_gold", dev),
                                                                         ("live_holdout_gold", ho))}
    state = {} if daily is None else {
        "end": "2026-10-05", "finished_utc": "2026-10-06T00:30:00+00:00", "assets": list(assets),
        "check": None if ok is None else {"ok": ok, "assets": list(assets), "error": None if ok else "drift"},
        "files": {n: {"path": "x", "sha256": h} for n, h in sha.items()}}
    upd = types.ModuleType("volrisk_live.update")
    upd.read_state = lambda: state

    def live_daily():
        if daily is None:
            raise AssertionError("live_daily must not be read without a verified state")
        return daily

    upd.live_daily = live_daily
    monkeypatch.setitem(sys.modules, "volrisk_live.update", upd)
    return state


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "SCORES", tmp_path / "forecasts" / "scores.csv")
    monkeypatch.setattr(paths, "RISK_SCORES", tmp_path / "forecasts" / "risk_scores.csv")
    monkeypatch.setattr(paths, "FORWARD_MD", tmp_path / "reports" / "live" / "forward_test.md")
    install_ledger(monkeypatch, [])
    install_update(monkeypatch, tmp_path, None)  # no state: the real live data are never read
    return tmp_path


def run(daily, payloads, **kw) -> dict:
    kw.setdefault("report", False)
    return S.score_all(daily, payloads=payloads, now=NOW, **kw)


def scored(key: dict) -> pd.Series:
    df = S.load_scores()
    m = np.ones(len(df), dtype=bool)
    for k, v in key.items():
        m &= (df[k].astype(str) == str(v)).to_numpy()
    assert m.sum() == 1, f"expected one score row for {key}, got {m.sum()}"
    return df[m].iloc[0]


def five_btc_runs(models=(("HAR", 1.0), ("COMBO", 1.3))) -> list[dict]:
    """Five daily BTC runs (origins 10-01..10-05), 1d forecasts of each model and one VaR row."""
    ps = []
    for k, origin in enumerate(pd.date_range("2026-10-01", "2026-10-05")):
        o, nxt = origin.strftime("%Y-%m-%d"), (origin + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        rid = (origin + pd.Timedelta(days=1)).strftime("%Y%m%dT061500Z")
        ps.append(payload(rid, {"BTC": o}, [frow("BTC", "1d", m, o, 1, F * (1 + 0.1 * k)) for m, F in models],
                          [rrow("BTC", "COMBO+FHS", nxt, 1.0, 0.5)]))
    return ps


# ============================================================================================= forecasts
def test_1d_waits_for_the_next_session_then_scores_qlike():
    p = payload("20261002T061500Z", {"BTC": "2026-10-01"},
                [frow("BTC", "1d", "HAR", "2026-10-01", 1, 1.2, "2026-10-02", "2026-10-02"),
                 frow("BTC", "1d", "COMBO", "2026-10-01T00:00:00", 1, 0.9, "2026-10-02", "2026-10-02")])
    s = run(upto("2026-10-01"), [p])
    assert s["forecasts"] == {"new": 0, "total": 0, "ok": 0, "pending": 2}
    assert s["next_due"] == "2026-10-02"

    s = run(upto("2026-10-02"), [p])
    assert s["forecasts"]["new"] == 2 and s["forecasts"]["pending"] == 0
    assert s["integrity"]["rederived"] == {"forecast": 2, "risk": 0} and s["integrity"]["failed"] == {
        "forecast": 0, "risk": 0}
    y = tv("BTC", "2026-10-02")
    for model, F in (("HAR", 1.2), ("COMBO", 0.9)):
        r = scored({"model": model})
        assert r["origin"] == "2026-10-01" and r["real_first"] == r["real_last"] == "2026-10-02"
        assert r["n_real"] == 1 and r["status"] == "ok" and not r["per_session"] and r["ex_ante"]
        assert r["y"] == pytest.approx(y, rel=1e-15)
        assert r["qlike"] == pytest.approx(ql(y, F), rel=1e-12)
        assert r["data_end"] == "2026-10-02" and r["run_utc"] == "2026-10-02T06:15:00+00:00"
        assert r["recorded_utc"] == "2026-10-02T06:19:00Z" and r["first_close_utc"] == "2026-10-03T00:00:00Z"
        assert r["before_close"] and not r["before_open"]  # crypto: the run happens during the target UTC day


def test_1d_spx_friday_origin_is_scored_against_monday():
    p = payload("20261003T060000Z", {"SPX": "2026-10-02"},
                [frow("SPX", "1d", "HAR", "2026-10-02", 1, 0.4, "2026-10-05", "2026-10-05")])
    assert run(upto("2026-10-04"), [p])["forecasts"]["pending"] == 1  # weekend: no SPX session yet
    run(upto("2026-10-05"), [p])
    r = scored({"asset": "SPX"})
    assert r["real_first"] == "2026-10-05"
    assert r["qlike"] == pytest.approx(ql(tv("SPX", "2026-10-05"), 0.4), rel=1e-12)
    assert r["before_open"] and r["before_close"] and r["first_close_utc"] == "2026-10-05T20:00:00Z"


def test_1w_needs_the_full_calendar_window():
    p = payload("20261002T060000Z", {"BTC": "2026-10-01"},
                [frow("BTC", "1w", "HAR", "2026-10-01", 7, 9.0, "2026-10-02", "2026-10-08")])
    assert run(upto("2026-10-07"), [p])["forecasts"]["new"] == 0
    run(upto("2026-10-08"), [p])
    r = scored({"horizon": "1w"})
    days = [f"2026-10-{d:02d}" for d in range(2, 9)]
    assert (r["n_real"], r["real_first"], r["real_last"]) == (7, "2026-10-02", "2026-10-08")
    assert r["y"] == pytest.approx(tv("BTC", *days), rel=1e-14)
    assert not r["per_session"] and r["qlike"] == pytest.approx(ql(tv("BTC", *days), 9.0), rel=1e-12)


def test_dropped_session_compares_per_session_means():
    # ex-ante n_t = 5 scheduled sessions in (Fri 10-02, Fri 10-09]; 10-07 was invalid, so 4 are realised
    p = payload("20261003T060000Z", {"SPX": "2026-10-02"},
                [frow("SPX", "1w", "HAR", "2026-10-02", 5, 3.0, "2026-10-05", "2026-10-09")])
    run(upto("2026-10-09"), [p])
    r = scored({"asset": "SPX"})
    y = tv("SPX", "2026-10-05", "2026-10-06", "2026-10-08", "2026-10-09")
    assert r["n_real"] == 4 and r["per_session"]
    assert r["ybar"] == pytest.approx(y / 4) and r["Fbar"] == pytest.approx(3.0 / 5)
    assert r["qlike"] == pytest.approx(ql(y / 4, 3.0 / 5), rel=1e-12)
    assert r["qlike"] != pytest.approx(ql(y, 3.0))  # not the cumulative comparison


def test_window_waits_while_the_asset_data_lag_behind_the_live_end():
    # crypto data reach 10-12 (live end), but the SPX source failed after 10-08: the window must not be scored
    p = payload("20261003T060000Z", {"SPX": "2026-10-02"},
                [frow("SPX", "1w", "HAR", "2026-10-02", 5, 3.0, "2026-10-05", "2026-10-09")])
    s = run(upto("2026-10-12", spx_end="2026-10-08"), [p])
    assert s["data_end"] == "2026-10-12" and s["forecasts"]["new"] == 0
    assert run(upto("2026-10-12"), [p])["forecasts"]["new"] == 1


def test_1m_and_explicit_end_override():
    p = payload("20261002T060000Z", {"BTC": "2026-10-01"},
                [frow("BTC", "1m", "HAR", "2026-10-01", 30, 40.0, "2026-10-02", "2026-10-31")])
    assert run(DAILY, [p], end="2026-10-30")["forecasts"]["new"] == 0  # rows after the end are ignored
    run(DAILY, [p], end="2026-10-31")
    r = scored({"horizon": "1m"})
    days = pd.date_range("2026-10-02", "2026-10-31").strftime("%Y-%m-%d")
    assert r["n_real"] == 30 and r["y"] == pytest.approx(tv("BTC", *days), rel=1e-13)


def test_unscorable_rows_are_recorded_once_with_their_status():
    p = payload("20261002T060000Z", {"BTC": "2026-10-01"},
                [frow("BTC", "1d", "BAD0", "2026-10-01", 1, 0.0), frow("BTC", "1d", "BADNAN", "2026-10-01", 1, None),
                 frow("BTC", "1d", "HAR", "2026-10-01", 1, 1.0)])
    zero = upto("2026-10-02").copy()
    zero.loc[(zero["asset"] == "BTC") & (zero["session_date"] == "2026-10-02"), "tv"] = 0.0
    s = run(zero, [p])
    assert s["forecasts"] == {"new": 3, "total": 3, "ok": 0, "pending": 0}
    assert set(S.load_scores()["status"]) == {"bad_target"}  # y = 0 cannot be scored (QLIKE needs y > 0)
    first = paths.SCORES.read_bytes()
    with pytest.raises(S.ScoreIntegrityError, match="3 forecast and 0 risk score"):
        run(upto("2026-10-02"), [p])  # the table changed after scoring: flagged, never rescored
    assert paths.SCORES.read_bytes() == first


def test_bad_forecast_status():
    p = payload("20261002T060000Z", {"BTC": "2026-10-01"},
                [frow("BTC", "1d", "BAD0", "2026-10-01", 1, 0.0), frow("BTC", "1d", "BADNAN", "2026-10-01", 1, None)])
    run(upto("2026-10-02"), [p])
    df = S.load_scores()
    assert list(df["status"]) == ["bad_forecast", "bad_forecast"] and df["qlike"].isna().all()


def test_scoring_is_idempotent_and_append_only():
    p1 = payload("20261002T060000Z", {"BTC": "2026-10-01", "SPX": "2026-10-01"},
                 [frow("BTC", "1d", "HAR", "2026-10-01", 1, 1.1), frow("SPX", "1d", "HAR", "2026-10-01", 1, 0.5)],
                 [rrow("BTC", "HS-250", "2026-10-02", 3.0, 2.0)])
    run(upto("2026-10-05"), [p1])
    first = paths.SCORES.read_bytes(), paths.RISK_SCORES.read_bytes()
    s = run(upto("2026-10-05"), [p1])
    assert s["forecasts"]["new"] == 0 and s["risk"]["new"] == 0
    assert (paths.SCORES.read_bytes(), paths.RISK_SCORES.read_bytes()) == first

    # a new run only appends
    p2 = payload("20261003T060000Z", {"BTC": "2026-10-02"}, [frow("BTC", "1d", "HAR", "2026-10-02", 1, 1.1)])
    s = run(upto("2026-10-06"), [p1, p2])
    assert s["forecasts"]["new"] == 1 and s["risk"]["new"] == 0
    assert paths.SCORES.read_bytes().startswith(first[0])
    assert paths.RISK_SCORES.read_bytes() == first[1]
    assert len(S.load_scores()) == 3
    assert b"\r\n" not in paths.SCORES.read_bytes()

    # later data with other numbers never rewrite a score: the stored rows no longer re-derive -> flagged
    changed = upto("2026-10-06").copy()
    changed["tv"] *= 10.0
    changed["r_cc"] *= -1.0
    before = paths.SCORES.read_bytes(), paths.RISK_SCORES.read_bytes()
    with pytest.raises(S.ScoreIntegrityError) as e:
        run(changed, [p1, p2])
    assert (paths.SCORES.read_bytes(), paths.RISK_SCORES.read_bytes()) == before
    assert e.value.summary["integrity"]["failed"] == {"forecast": 3, "risk": 1}


def test_csv_header_mismatch_is_refused():
    paths.SCORES.parent.mkdir(parents=True)
    paths.SCORES.write_text("run_id,asset\nx,BTC\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        run(upto("2026-10-02"), [payload("20261002T060000Z", {"BTC": "2026-10-01"},
                                         [frow("BTC", "1d", "HAR", "2026-10-01", 1, 1.0)])])


def test_invalid_payload_is_skipped():
    good = payload("20261002T060000Z", {"BTC": "2026-10-01"}, [frow("BTC", "1d", "HAR", "2026-10-01", 1, 1.0)])
    bad = {k: v for k, v in good.items() if k != "frozen"} | {"run_id": "20261002T070000Z"}
    s = run(upto("2026-10-02"), [good, bad])
    assert s["invalid_payloads"] == 1 and s["payloads"] == 1 and s["forecasts"]["new"] == 1


# ============================================================================================= risk
def test_risk_breach_flags_on_the_next_session():
    # BTC r_cc on 10-02 is -2.0 -> loss 2.0
    p = payload("20261002T060000Z", {"BTC": "2026-10-01", "SPX": "2026-10-06"},
                [], [rrow("BTC", "HS-250", "2026-10-02", 1.5, 1.0), rrow("BTC", "COMBO+FHS", "2026-10-02", 3.0, 1.9),
                     rrow("BTC", "RiskMetrics", "2026-10-02", 2.5, 2.1),
                     rrow("SPX", "COMBO+FHS", "2026-10-07", 9.0, 8.0)])
    s = run(upto("2026-10-01"), [p])
    assert s["risk"]["new"] == 0 and s["risk"]["pending"] == 4
    s = run(upto("2026-10-07"), [p])
    assert s["risk"]["new"] == 3  # SPX 10-07 was invalid: wait for the next valid session
    r = S.load_risk_scores().set_index("model")
    assert (r.loc["HS-250", ["breach99", "breach975"]] == [1, 1]).all()
    assert (r.loc["COMBO+FHS", ["breach99", "breach975"]] == [0, 1]).all()
    assert (r.loc["RiskMetrics", ["breach99", "breach975"]] == [0, 0]).all()
    assert (r["loss"] == 2.0).all() and (r["session_date"] == "2026-10-02").all()
    assert r["before_close"].all() and (r["first_close_utc"] == "2026-10-03T00:00:00Z").all()

    run(upto("2026-10-08"), [p])
    spx = S.load_risk_scores().query("asset == 'SPX'").iloc[0]
    assert spx["date"] == "2026-10-07" and spx["session_date"] == "2026-10-08"  # frozen next-row convention
    r_cc = float(DAILY.loc[(DAILY["asset"] == "SPX") & (DAILY["session_date"] == pd.Timestamp("2026-10-08")),
                           "r_cc"].iloc[0])
    assert spx["r_cc"] == pytest.approx(r_cc) and spx["breach99"] == 0


# ============================================================================================= report
def test_report_with_nothing_scored_says_when_first_scores_come(monkeypatch):
    p = payload("20261002T061500Z", {"BTC": "2026-10-01", "SPX": "2026-10-01"},
                [frow("BTC", "1d", "HAR", "2026-10-01", 1, 1.0, "2026-10-02", "2026-10-02"),
                 frow("SPX", "1m", "HAR", "2026-10-01", 21, 9.0, "2026-10-02", "2026-10-30")],
                [rrow("BTC", "COMBO+FHS", "2026-10-02", 3.0, 2.0)])
    install_ledger(monkeypatch, [p])
    s = S.score_all(upto("2026-10-01"), now=NOW)  # payloads from the (stub) ledger
    text = paths.FORWARD_MD.read_text(encoding="utf-8")
    assert s["report"] == str(paths.FORWARD_MD) and s["payloads"] == 1 and s["unreadable_payloads"] == []
    assert "Forward test, forecasts recorded before outcomes, since 2026-10-02" in text
    assert "No forecast has been scored yet — first scores after 2026-10-02" in text
    assert "| forecast runs in the ledger (verified chain) | 1 |" in text
    assert "intact (1 entries, head `abababababababab…`)" in text
    assert "1 payload file(s): 1 pending (submitted, awaiting Bitcoin attestation)" in text
    assert "Why this is a forward test that cannot be tuned" in text
    assert "Stored scores re-derived on this run**: nothing stored yet" in text
    assert "Integrity problem" not in text
    assert not paths.SCORES.exists()  # nothing to append yet


def test_report_ratio_table_highlights_combo_and_counts_breaches():
    models = {"HAR": 1.0, "GJR": 1.3, "COMBO": 0.95}
    payloads, rows = [], {m: [] for m in models}
    for k, origin in enumerate(pd.date_range("2026-10-01", "2026-10-10")):
        o = origin.strftime("%Y-%m-%d")
        nxt = (origin + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        rid = (origin + pd.Timedelta(days=1)).strftime("%Y%m%dT060000Z")
        fc = [frow("BTC", "1d", m, o, 1, F * (1 + 0.05 * k)) for m, F in models.items()]
        rk = [rrow("BTC", "COMBO+FHS", nxt, 1.0, 0.5), rrow("BTC", "HS-250", nxt, 5.0, 4.0)]
        payloads.append(payload(rid, {"BTC": o}, fc, rk))
        for m, F in models.items():
            rows[m].append(ql(tv("BTC", nxt), F * (1 + 0.05 * k)))
    # a duplicate run at the same origin and a late (not ex-ante) row must not enter the ratios
    payloads.append(payload("20261002T090000Z", {"BTC": "2026-10-01"}, [frow("BTC", "1d", "HAR", "2026-10-01", 1, 50.0)]))
    payloads.append(payload("20261005T070000Z", {"BTC": "2026-10-04"}, [frow("BTC", "1d", "COMBO", "2026-10-01", 1, 50.0)]))

    s = S.score_all(upto("2026-10-11"), payloads=payloads, now=NOW, report=False)
    assert s["forecasts"]["ok"] == 32
    out = S.report_forward(payloads=payloads, now=NOW, chain=[], timestamps={"complete": 3, "pending": 9},
                           daily=upto("2026-10-11"))
    text = out.read_text(encoding="utf-8")

    lb = S.ratio_table(S.load_scores()).set_index("model")
    for m in models:
        assert lb.loc[m, "qlike_ratio"] == pytest.approx(np.mean(rows[m]) / np.mean(rows["HAR"]), rel=1e-12)
        assert lb.loc[m, "n"] == 10
    assert f"| **COMBO** | **{lb.loc['COMBO', 'qlike_ratio']:.3f}** |" in text
    assert "| HAR | 1.000 |" in text and "| n (common origins) | 10 |" in text
    assert "1 scored row(s) had origins before the run's last session" in text
    # VaR: loss > 1.0 on BTC days with r_cc < -1.0; HS-250 at 5.0 never breaches
    sd = DAILY["session_date"]
    days = (DAILY["asset"] == "BTC") & (sd >= pd.Timestamp("2026-10-02")) & (sd <= pd.Timestamp("2026-10-11"))
    losses = -DAILY.loc[days, "r_cc"].to_numpy()
    b99 = int((losses > 1.0).sum())
    assert f"| **COMBO+FHS** | **10** | **{b99}** | **0.10** |" in text
    assert "| HS-250 | 10 | 0 | 0.10 |" in text
    assert "## Integrity" in text and "intact" in text and "complete: 3, pending: 9" in text
    assert "all 32 forecast and 20 risk score(s) re-derived exactly" in text
    assert "first scores after" not in text
    assert "recorded before their first session opened: BTC 0 of 10." in text


def test_iv_benchmarks_get_their_own_1m_table():
    fc = []
    for m, F in (("HAR", 30.0), ("COMBO", 28.0), ("IV", 35.0), ("IV-cal", 31.0)):
        fc.append(frow("BTC", "1m", m, "2026-09-01", 30, F))
    p = payload("20260902T060000Z", {"BTC": "2026-09-01"}, fc)
    run(DAILY, [p], end="2026-10-01")
    assert set(S.ratio_table(S.load_scores())["model"]) == {"HAR", "COMBO"}
    iv = S.ratio_table(S.load_scores(), with_iv=True)
    assert set(iv["model"]) == {"HAR", "COMBO", "IV", "IV-cal"}
    S.report_forward(payloads=[p], now=NOW, chain=True, timestamps=None, daily=DAILY)
    assert "1m including implied-volatility benchmarks" in paths.FORWARD_MD.read_text(encoding="utf-8")


def test_report_survives_missing_ledger_and_ots(monkeypatch):
    for name in ("volrisk_live.ledger", "volrisk_live.ots"):
        monkeypatch.setitem(sys.modules, name, None)  # import of a None entry raises ImportError
    out = S.report_forward(now=NOW)
    text = out.read_text(encoding="utf-8")
    assert "no forecast run has been recorded yet" in text and "not available" in text


def test_describe_status_shapes():
    assert S.describe_chain(None) == "intact" and S.describe_chain(True) == "intact"
    assert "BROKEN" in S.describe_chain(False)
    broken = {"ok": False, "n": 4, "first_bad": 2, "reason": "entry 2: payload hash mismatch", "head": "x"}
    assert S.describe_chain(broken) == "**BROKEN** at line 2 of 4 entries: entry 2: payload hash mismatch"
    assert S.describe_chain({"ok": True, "n": 0, "first_bad": None, "reason": None, "head": ""}) == "intact (0 entries)"
    assert S.describe_chain({"first_broken": None, "entries": [1, 2]}) == "intact (2 entries)"
    assert S.describe_chain(["entry 3: payload hash mismatch"]).startswith("**BROKEN**")
    real = {"n": 3, "counts": {"complete": 1, "pending": 2}, "files": {}}
    assert S.describe_timestamps(real) == ("3 payload file(s): 1 complete (Bitcoin-attested), "
                                           "2 pending (submitted, awaiting Bitcoin attestation)")
    assert S.describe_timestamps({"n": 0, "counts": {}, "files": {}}) == "no payload stamped yet"
    assert S.describe_timestamps({"a": {"status": "complete"}, "b": {"status": "pending"}, "c": "pending"}) == \
        "3 payload(s): pending 2, complete 1"
    assert S.describe_timestamps([]) == "no stamps yet"


# ============================================================================================= integrity: re-derivation
@pytest.mark.parametrize("edit", ["qlike", "y_and_qlike", "F_and_qlike", "before_close", "status"])
def test_an_edited_score_file_is_caught_left_out_and_fails_the_run(edit):
    """Finding: stored scores were never re-checked, so an edited scores.csv changed the headline ratios."""
    ps = five_btc_runs()
    run(upto("2026-10-06"), ps)
    honest = S.ratio_table(S.load_scores()).set_index("model").loc["COMBO", "qlike_ratio"]
    df = S.load_scores()
    m = (df["model"] == "COMBO").to_numpy()
    if edit == "qlike":  # the reviewer's probe: COMBO "wins" by a factor of 700
        df.loc[m, "qlike"] = 0.0001
    elif edit == "y_and_qlike":  # internally consistent, only the live data can tell
        df.loc[m, "y"] = df.loc[m, "Fbar"] * 1.01
        df.loc[m, "ybar"] = df.loc[m, "y"]
        df.loc[m, "qlike"] = [ql(y, F) for y, F in zip(df.loc[m, "y"], df.loc[m, "F"], strict=True)]
    elif edit == "F_and_qlike":  # consistent with the outcome, only the payload can tell
        df.loc[m, "F"] = df.loc[m, "y"] * 1.01
        df.loc[m, "Fbar"] = df.loc[m, "F"]
        df.loc[m, "qlike"] = [ql(y, F) for y, F in zip(df.loc[m, "y"], df.loc[m, "F"], strict=True)]
    elif edit == "before_close":  # pretend a late HAR row was early
        df.loc[~m, "before_close"] = True
        df.loc[~m, "recorded_utc"] = "2026-10-01T00:00:00Z"
    else:
        df.loc[m, "status"] = "bad_target"  # hide a forecast
    df.to_csv(paths.SCORES, index=False, lineterminator="\n")
    tampered = df.copy()

    with pytest.raises(S.ScoreIntegrityError, match="do not re-derive") as e:
        run(upto("2026-10-06"), ps, report=True)
    n_bad = 5
    assert e.value.summary["integrity"]["failed"] == {"forecast": n_bad, "risk": 0}
    assert e.value.summary["forecasts"]["new"] == 0  # nothing rewritten
    pd.testing.assert_frame_equal(S.load_scores(), S._read(paths.SCORES, S.SCORE_COLUMNS, S.SCORE_KEY))
    assert len(S.load_scores()) == len(tampered)

    audit = S.audit_scores(S.load_scores(), S.load_risk_scores(), ps, upto("2026-10-06"))
    clean, _ = S.drop_failed(S.load_scores(), S.load_risk_scores(), audit)
    assert len(clean) == 10 - n_bad and len(audit["forecast"]) == n_bad
    text = paths.FORWARD_MD.read_text(encoding="utf-8")
    assert "Integrity problem: 5 stored score(s) do not re-derive" in text
    assert "do NOT re-derive" in text and "differs from the re-derived score in" in text
    if edit != "before_close":  # the COMBO rows are out: the tampered ratio never reaches the report
        assert f"{honest:.3f}" not in text and "| **COMBO**" not in text


def test_a_score_without_its_payload_or_with_a_duplicate_key_fails():
    ps = five_btc_runs()
    run(upto("2026-10-06"), ps)
    df = S.load_scores()
    extra = df.iloc[[0, 1]].copy()
    extra.loc[extra.index[0], "run_id"] = "20261003T000000Z"  # a run that is not in the ledger
    pd.concat([df, extra]).to_csv(paths.SCORES, index=False, lineterminator="\n")
    audit = S.audit_scores(S.load_scores(), S.load_risk_scores(), ps, upto("2026-10-06"))
    reasons = sorted(set(audit["forecast"].values()))
    assert reasons == ["its run is not in the verified ledger", "the key appears more than once in the file"]
    assert len(audit["forecast"]) == 2  # the forged run and the duplicated key (both copies)


def test_scores_written_by_this_module_re_derive_bit_for_bit():
    ps = five_btc_runs()
    s = run(upto("2026-10-06"), ps)
    assert s["integrity"]["rederived"] == {"forecast": 10, "risk": 5}
    raw = pd.read_csv(paths.SCORES, float_precision="round_trip")
    again = [S.forecast_record(p, f, S._Context(upto("2026-10-06"), np.datetime64("2026-10-06"))) for p in ps
             for f in p["forecasts"]]
    assert raw["qlike"].tolist() == [r["qlike"] for r in again]  # exact, not approx


# ============================================================================================= integrity: ledger prefix
@pytest.mark.skipif(REAL_LEDGER is None, reason="volrisk_live.ledger not importable")
def test_real_ledger_roundtrip_and_tampered_payload(monkeypatch, tmp_path):
    """The real ledger (tmp paths, no stamping): entries/load_payload feed the scorer; after an edited payload
    nothing at or after the broken link is scored, and the report shows the broken chain."""
    monkeypatch.setattr(paths, "LEDGER", tmp_path / "forecasts" / "ledger.jsonl")
    monkeypatch.setattr(paths, "RUNS", tmp_path / "forecasts" / "runs")
    monkeypatch.setitem(sys.modules, "volrisk_live.ledger", REAL_LEDGER)
    p1 = payload("20261002T061500Z", {"BTC": "2026-10-01"}, [frow("BTC", "1d", "HAR", "2026-10-01", 1, 1.0)])
    p2 = payload("20261003T061500Z", {"BTC": "2026-10-02"}, [frow("BTC", "1d", "HAR", "2026-10-02", 1, 1.1)])
    for p in (p1, p2):
        REAL_LEDGER.append(p, stamp=False)
    s = S.score_all(upto("2026-10-03"), now=NOW, report=False)
    assert s["payloads"] == 2 and s["forecasts"]["new"] == 2 and s["unreadable_payloads"] == []
    assert s["integrity"]["chain_ok"] is True

    # edit the first payload after it was recorded (a "better" forecast): the chain breaks at entry 0
    paths.SCORES.unlink()
    f1 = paths.RUNS / "20261002T061500Z.json"
    f1.write_bytes(f1.read_bytes().replace(b'"F":1.0', b'"F":1.5'))
    with pytest.raises(S.ScoreIntegrityError, match="ledger"):
        S.score_all(upto("2026-10-03"), now=NOW)
    assert not paths.SCORES.exists()
    text = paths.FORWARD_MD.read_text(encoding="utf-8")
    assert "**BROKEN** at line 0" in text and "Ledger entry not scored" in text


@pytest.mark.skipif(REAL_LEDGER is None, reason="volrisk_live.ledger not importable")
def test_forged_entry_with_rewritten_payload_hash_is_never_scored(monkeypatch, tmp_path):
    """Finding: a payload edited together with its entry's payload_sha256 still matched and was scored, and the
    forged score then stayed in scores.csv even after the ledger was repaired."""
    monkeypatch.setattr(paths, "LEDGER", tmp_path / "forecasts" / "ledger.jsonl")
    monkeypatch.setattr(paths, "RUNS", tmp_path / "forecasts" / "runs")
    monkeypatch.setitem(sys.modules, "volrisk_live.ledger", REAL_LEDGER)
    p1 = payload("20261002T061500Z", {"BTC": "2026-10-01"}, [frow("BTC", "1d", "HAR", "2026-10-01", 1, 1.0)])
    p2 = payload("20261003T061500Z", {"BTC": "2026-10-02"}, [frow("BTC", "1d", "HAR", "2026-10-02", 1, 1.1)])
    for p in (p1, p2):
        REAL_LEDGER.append(p, stamp=False)
    original_ledger = paths.LEDGER.read_bytes()
    f1 = paths.RUNS / "20261002T061500Z.json"
    original_payload = f1.read_bytes()
    f1.write_bytes(original_payload.replace(b'"F":1.0', b'"F":2.5'))
    lines = paths.LEDGER.read_text(encoding="utf-8").splitlines()
    e0 = json.loads(lines[0])
    e0["payload_sha256"] = hashlib.sha256(f1.read_bytes()).hexdigest()
    lines[0] = schema.canonical_json(e0)
    paths.LEDGER.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert REAL_LEDGER.load_payload(json.loads(lines[0]))["forecasts"][0]["F"] == 2.5  # the payload "matches"
    assert REAL_LEDGER.verify_chain()["first_bad"] == 0

    with pytest.raises(S.ScoreIntegrityError) as e:
        S.score_all(upto("2026-10-03"), now=NOW, report=False)
    skipped = e.value.summary["unreadable_payloads"]
    assert [x.split(":")[0] for x in skipped] == ["20261002T061500Z", "20261003T061500Z"]
    assert "first broken link" in skipped[0] and e.value.summary["forecasts"]["total"] == 0
    assert not paths.SCORES.exists()

    # a forged score written earlier (old code) does not survive the repair of the ledger
    rec = S.forecast_record(json.loads(f1.read_bytes()), json.loads(f1.read_bytes())["forecasts"][0],
                            S._Context(upto("2026-10-03"), np.datetime64("2026-10-03")))
    S._append(paths.SCORES, [rec], S.SCORE_COLUMNS, S.SCORE_KEY, S.load_scores())
    paths.LEDGER.write_bytes(original_ledger)
    f1.write_bytes(original_payload)
    assert REAL_LEDGER.verify_chain()["ok"]
    with pytest.raises(S.ScoreIntegrityError, match="1 forecast and 0 risk"):
        S.score_all(upto("2026-10-03"), now=NOW, report=False)
    audit = S.audit_scores(S.load_scores(), S.load_risk_scores(), [p1, p2], upto("2026-10-03"))
    assert list(audit["forecast"]) == [("20261002T061500Z", "BTC", "1d", "HAR", "2026-10-01")]
    assert "F (stored 2.5, re-derived 1.0)" in audit["forecast"][("20261002T061500Z", "BTC", "1d", "HAR",
                                                                   "2026-10-01")]


def test_only_entries_before_the_first_broken_link_are_scored(monkeypatch):
    p1 = payload("20261002T061500Z", {"BTC": "2026-10-01"}, [frow("BTC", "1d", "HAR", "2026-10-01", 1, 1.0)])
    p2 = payload("20261003T061500Z", {"BTC": "2026-10-02"}, [frow("BTC", "1d", "HAR", "2026-10-02", 1, 1.1)])
    install_ledger(monkeypatch, [p1, p2], chain={"ok": False, "n": 2, "first_bad": 1, "reason": "entry 1: edited",
                                                 "head": "x", "orphans": []})
    with pytest.raises(S.ScoreIntegrityError):
        S.score_all(upto("2026-10-03"), now=NOW, report=False)
    assert set(S.load_scores()["run_id"]) == {"20261002T061500Z"}
    entries, payloads, problems = S.ledger_payloads()
    assert [e["run_id"] for e in entries] == [p["run_id"] for p in payloads] == ["20261002T061500Z"]
    assert problems == ["20261003T061500Z: ledger entry 1 is at or after the first broken link (entry 1: edited)"]


# ============================================================================================= integrity: recorded time
def test_payload_written_after_its_first_session_closed_is_left_out():
    """Finding: a payload created weeks after its outcome (past ``--now``) counted as recorded before outcomes."""
    ps = five_btc_runs()
    late = payload("20261002T061600Z", {"BTC": "2026-10-01"},
                   [frow("BTC", "1d", "HAR", "2026-10-01", 1, 9.0), frow("BTC", "1d", "COMBO", "2026-10-01", 1, 9.0)],
                   [rrow("BTC", "COMBO+FHS", "2026-10-02", 0.1, 0.05)],
                   computed="2026-11-15T09:00:00+00:00")
    no_clock = payload("20261002T061700Z", {"BTC": "2026-10-01"}, [frow("BTC", "1d", "GJR", "2026-10-01", 1, 9.0)],
                       computed=None)
    s = run(upto("2026-10-06"), [*ps, late, no_clock], report=True)
    assert s["integrity"]["failed"] == {"forecast": 0, "risk": 0}  # stored honestly, re-derives
    r = scored({"run_id": late["run_id"], "model": "COMBO"})
    assert r["status"] == "ok" and r["ex_ante"] and not r["before_close"] and not r["before_open"]
    assert r["recorded_utc"] == "2026-11-15T09:00:00Z" and r["first_close_utc"] == "2026-10-03T00:00:00Z"
    g = scored({"run_id": no_clock["run_id"]})
    assert g["recorded_utc"] != g["recorded_utc"] and not g["before_close"]  # NaN: no wall clock, never "before"

    head = S.headline_scores(S.load_scores())
    assert late["run_id"] not in set(head["run_id"]) and no_clock["run_id"] not in set(head["run_id"])
    lb = S.ratio_table(S.load_scores())
    assert set(lb["model"]) == {"HAR", "COMBO"} and (lb["n"] == 5).all()
    rt = S.risk_table(S.load_risk_scores()).set_index(["model", "asset"])
    assert rt.loc[("COMBO+FHS", "ALL"), "N"] == 5  # the late VaR row (0.1, would breach) is not counted
    text = paths.FORWARD_MD.read_text(encoding="utf-8")
    assert "3 forecast and 1 VaR row(s) were recorded after their first target session had closed" in text


def test_recorded_utc_is_the_latest_payload_clock_and_needs_computed_utc():
    p = payload("20261002T061500Z", {"BTC": "2026-10-01"})
    assert S.recorded_utc(p) == datetime(2026, 10, 2, 6, 19, tzinfo=timezone.utc)  # timing_utc
    back = {**p, "checks": {"computed_utc": "2026-10-02T23:59:30+00:00", "timing_utc": "2026-10-03T00:00:30+00:00"}}
    assert S.recorded_utc(back) == datetime(2026, 10, 3, 0, 0, 30, tzinfo=timezone.utc)  # after the BTC close
    assert S.recorded_utc({**p, "checks": {"timing_utc": "2026-10-02T06:19:00+00:00"}}) is None
    try:
        from volrisk_live import cli
    except ImportError:  # pragma: no cover - the CLI is built alongside this module
        return
    if hasattr(cli, "recorded_utc"):  # the report, the CLI summary and the dashboard use one recording time
        assert cli.recorded_utc(p) == S.recorded_utc(p) and cli.recorded_utc(back) == S.recorded_utc(back)


# ============================================================================================= integrity: live data
def test_score_refuses_live_data_whose_consistency_check_did_not_pass(monkeypatch, tmp_path):
    """Finding: a standalone ``score`` after a failed update scored permanently against drifted data."""
    ps = five_btc_runs()
    install_ledger(monkeypatch, ps)
    for ok in (False, None):
        install_update(monkeypatch, tmp_path, upto("2026-10-06"), ok=ok)
        with pytest.raises(S.LiveDataUnverified, match="consistency check of the last update"):
            S.score_all(now=NOW)
        assert not paths.SCORES.exists() and not paths.FORWARD_MD.exists()

    install_update(monkeypatch, tmp_path, None)  # no state.json at all
    with pytest.raises(S.LiveDataUnverified, match="missing"):
        S.score_all(now=NOW)


def test_score_refuses_live_files_changed_after_the_check(monkeypatch, tmp_path):
    install_ledger(monkeypatch, five_btc_runs())
    install_update(monkeypatch, tmp_path, upto("2026-10-06"))
    paths.LIVE_DAILY_HOLDOUT.write_bytes(b"rebuilt after the check")
    with pytest.raises(S.LiveDataUnverified, match="changed since the checked update"):
        S.score_all(now=NOW)
    assert not paths.SCORES.exists()


def test_score_uses_verified_live_data_of_the_checked_assets_only(monkeypatch, tmp_path):
    ps = five_btc_runs()
    spx = payload("20261003T060000Z", {"SPX": "2026-10-02"}, [frow("SPX", "1d", "HAR", "2026-10-02", 1, 0.4)])
    install_ledger(monkeypatch, [*ps, spx])
    install_update(monkeypatch, tmp_path, upto("2026-10-06"), assets=("BTC",))
    s = S.score_all(now=NOW)
    assert s["forecasts"]["new"] == 10 and s["forecasts"]["pending"] == 1  # SPX was not in the checked assets
    assert s["integrity"]["live_data"]["assets"] == ["BTC"]
    text = paths.FORWARD_MD.read_text(encoding="utf-8")
    assert "the consistency check of the last update" in text and "dev `" in text

    # the report re-derives with the same verified data; once they change, no stored score is shown
    S.report_forward(now=NOW)
    assert "all 10 forecast and 5 risk score(s) re-derived exactly" in paths.FORWARD_MD.read_text(encoding="utf-8")
    paths.LIVE_DAILY_DEV.write_bytes(b"edited")
    S.report_forward(now=NOW)
    text = paths.FORWARD_MD.read_text(encoding="utf-8")
    assert "**NOT VERIFIED**" in text and "No stored score could be re-derived on this run" in text
    assert "| **COMBO** |" not in text


def test_data_revised_after_a_forecast_gives_history_changed_once():
    """The gold rows up to the run's last session must still hash to the payload's rows_sha256."""
    old = upto("2026-10-06").copy()
    old.loc[(old["asset"] == "BTC") & (old["session_date"] == "2026-09-15"), "tv"] = 7.0  # what the run saw
    p = payload("20261002T061500Z", {"BTC": "2026-10-01"}, [frow("BTC", "1d", "HAR", "2026-10-01", 1, 1.0)],
                [rrow("BTC", "HS-250", "2026-10-02", 3.0, 2.0)], history=old)
    with pytest.raises(S.ScoreIntegrityError, match="history_changed") as e:
        run(upto("2026-10-06"), [p], report=True)
    assert e.value.summary["integrity"]["history_changed_new"] == ["20261002T061500Z BTC"]
    assert S.load_scores()["status"].tolist() == ["history_changed"]
    assert S.load_risk_scores()["status"].tolist() == ["history_changed"]
    assert S.ratio_table(S.load_scores()).empty and S.risk_table(S.load_risk_scores()).empty
    assert "Data revised after a forecast" in paths.FORWARD_MD.read_text(encoding="utf-8")
    s = run(upto("2026-10-06"), [p])  # recorded once; the stored status re-derives on later runs
    assert s["forecasts"]["new"] == 0 and s["integrity"]["failed"] == {"forecast": 0, "risk": 0}


def test_history_hash_matches_forecast_rows_sha256_on_gold_like_columns():
    rng = np.random.default_rng(3)
    n = 40
    d = pd.DataFrame({
        "asset": ["SPX"] * n,
        "session_date": pd.bdate_range("2026-08-03", periods=n).astype("datetime64[ms]"),
        "n_sched": np.full(n, 78, dtype="int32"),
        "valid": rng.random(n) > 0.1,
        "coverage": np.where(rng.random(n) > 0.8, np.nan, rng.random(n)),
        "tv": rng.random(n) * 1e-3 + 1e-17,
        "r_cc": rng.standard_normal(n),
    })
    shuffled = pd.concat([d.iloc[::-1], d.assign(asset="BTC")]).reset_index(drop=True)
    h = S._History(shuffled)
    for cut in (0, 1, 17, n - 1):
        last = d["session_date"].iloc[cut]
        assert h.sha("SPX", np.datetime64(last.date(), "D")) == FC.rows_sha256(d[d["session_date"] <= last])
    assert h.sha("SPX", np.datetime64("2026-01-01")) == FC.rows_sha256(d.iloc[0:0])


def test_outputs_stay_out_of_sealed_locations():
    for p in REAL_OUTPUTS:
        assert not any(p.is_relative_to(t) for t in paths.SEALED_TARGETS)
        assert p.is_relative_to(paths.FORECASTS) or p.is_relative_to(paths.LIVE_REPORTS)
