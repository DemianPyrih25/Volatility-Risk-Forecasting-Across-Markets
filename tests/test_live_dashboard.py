"""Live dashboard (docs/LIVE_SPEC.md §8): the frozen app plus the Tomorrow / Forward test / Verification tabs.

Everything is synthetic and lives in tmp_path: a schema-valid payload recorded through the real ledger
(``ledger.append(stamp=False)`` with the live paths redirected), score files with ``volrisk_live.score``'s columns,
a verification.json and a tiny live gold table. No server (callbacks run through Flask's in-process test client),
no browser, no network; the frozen results directories are empty.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from dash import dcc, html
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
from opentimestamps.core.op import OpAppend, OpSHA256
from opentimestamps.core.timestamp import DetachedTimestampFile

from volrisk_live import cli as CLI
from volrisk_live import dashboard as D
from volrisk_live import ledger, ots, paths, schema
from volrisk_live import score as SC

RUN_UTC = datetime(2026, 10, 2, 6, 0, tzinfo=UTC)
RUN_ID = "20261002T060000Z"
WRITTEN = RUN_UTC + timedelta(minutes=20)  # checks.computed_utc: the machine clock after the walk-forward
MODELS = {"HAR": 1.00, "GARCH": 1.25, "LGBM": 1.10, "COMBO": 1.05}  # per-session variance multipliers
RISK_MODELS = ["HS-250", "RiskMetrics", "GJR+FHS", "HAR*+FHS", "COMBO+FHS", "COMBO+Normal"]
BASE = {"BTC": (4.0, 365), "SPX": (0.64, 252)}  # per-session variance (%²), annualisation
WINDOWS = {  # (window_first, window_last, n_t) per asset and horizon, from origin 2026-10-01
    "BTC": {"1d": ("2026-10-02", "2026-10-02", 1), "1w": ("2026-10-02", "2026-10-08", 7),
            "1m": ("2026-10-02", "2026-10-31", 30)},
    "SPX": {"1d": ("2026-10-02", "2026-10-02", 1), "1w": ("2026-10-02", "2026-10-08", 5),
            "1m": ("2026-10-02", "2026-10-30", 21)},
}


# --------------------------------------------------------------------------------------------- synthetic files
def make_payload() -> dict:
    fc, risk, implied, data = [], [], [], {}
    for asset, (v, ann) in BASE.items():
        for h, (first, last, n) in WINDOWS[asset].items():
            models = dict(MODELS, **({"IV": 1.3, "IV-cal": 1.2} if h == "1m" else {}))
            for m, k in models.items():
                F = v * k * n
                fc.append({"asset": asset, "horizon": h, "model": m, "origin": "2026-10-01", "window_first": first,
                           "window_last": last, "n_t": n, "F": F, "vol_ann": math.sqrt(F / n * ann)})
        for i, m in enumerate(RISK_MODELS):
            s = math.sqrt(v) * (1 + 0.05 * i)
            risk.append({"asset": asset, "model": m, "date": "2026-10-02", "sigma": s, "var99": 2.33 * s,
                         "var975": 1.96 * s, "es975": 2.34 * s})
        implied.append({"asset": asset, "origin": "2026-10-01", "iv": 45.0 if asset == "BTC" else 16.3,
                        "iv_var_30d": 1.0, "source": "DVOL" if asset == "BTC" else "VIX"})
        data[asset] = {"last_session": "2026-10-01", "n_sessions": 1000, "rows_sha256": "ab" * 32,
                       "next_session": "2026-10-02", "next_open_utc": "2026-10-02T00:00:00+00:00"
                       if asset == "BTC" else "2026-10-02T13:30:00+00:00",
                       "next_close_utc": "2026-10-03T00:00:00+00:00", "stale": False,
                       "recorded_before_open": asset == "SPX"}
    return {
        "schema": schema.SCHEMA, "run_id": RUN_ID, "run_utc": RUN_UTC.isoformat(timespec="seconds"),
        "frozen": {"code_sha": "c0de" * 16, "seal_ok": True, "sealed_utc": "2026-10-01T10:00:00+00:00",
                   "holdout_opened_utc": "2026-10-02T09:00:00+00:00"},
        "live_code_sha": "11ve" * 16, "data": data, "forecasts": fc, "risk": risk, "implied": implied,
        "checks": {"live_end": "2026-10-01", "computed_utc": WRITTEN.isoformat(timespec="seconds"),
                   "reproduction": {"rows": 123456, "max_abs_diff": 0.0, "max_rel_diff": 0.0, "passed": True},
                   "data_consistency": {"check_against_sealed": {"ok": True, "dev": {"rows_compared": 9000},
                                                                 "holdout": {"rows_compared": 1200}}},
                   "stale_assets": []},
    }


def _write_csv(path, columns, rows) -> None:
    pd.DataFrame([{c: r.get(c, "") for c in columns} for r in rows], columns=columns).to_csv(path, index=False)


def score_rows(runs=(("2026-09-29", "20260930T060000Z", 3.0), ("2026-09-30", "20261001T060000Z", 6.0))) -> list:
    """Scored BTC 1d rows per (origin, run_id, realised y) with score.py's timing columns: recorded at the run id's
    time, target window = the session after the origin (BTC: the next UTC day), frozen calendar."""
    rows = []
    for origin, run_id, y in runs:
        first = (date.fromisoformat(origin) + timedelta(days=1)).isoformat()
        rec = datetime.strptime(run_id, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        o, c = CLI.session_bounds("BTC", first)
        for m, k in MODELS.items():
            F = 4.0 * k
            q = y / F - math.log(y / F) - 1
            rows.append({"run_id": run_id, "asset": "BTC", "horizon": "1d", "model": m, "origin": origin,
                         "run_utc": rec.isoformat(), "recorded_utc": rec.isoformat(), "ex_ante": True,
                         "before_open": rec < o, "before_close": rec < c, "first_close_utc": c.isoformat(),
                         "window_first": first,
                         "window_last": first, "n_t": 1, "F": F, "n_real": 1, "y": y, "ybar": y, "Fbar": F,
                         "per_session": False, "qlike": q, "status": "ok", "data_end": "2026-10-01"})
    return rows


def write_scores(forecasts_dir) -> None:
    """Two scored BTC origins (1d) for every model and one scored risk day, with score.py's columns."""
    _write_csv(forecasts_dir / "scores.csv", SC.SCORE_COLUMNS, score_rows())
    risk = [{"run_id": "20261001T060000Z", "asset": "BTC", "model": m, "date": "2026-10-01", "run_utc": "x",
             "recorded_utc": "2026-10-01T06:00:00+00:00", "before_close": True,
             "first_close_utc": "2026-10-02T00:00:00+00:00",
             "session_date": "2026-10-01", "sigma": 2.0, "var99": 4.66 + 0.5 * i, "var975": 3.92 + 0.5 * i,
             "es975": 4.7, "r_cc": -5.0, "loss": 5.0, "breach99": 5.0 > 4.66 + 0.5 * i,
             "breach975": 5.0 > 3.92 + 0.5 * i, "status": "ok", "data_end": "2026-10-01"}
            for i, m in enumerate(RISK_MODELS)]
    _write_csv(forecasts_dir / "risk_scores.csv", SC.RISK_SCORE_COLUMNS, risk)


VERIFICATION = {
    "generated_utc": "2026-10-02T07:00:00+00:00",
    "checks": [
        {"id": 1, "title": "Frozen code unchanged", "status": "PASS", "method": "verify_seal() and the code hash "
         "logged at the holdout opening", "evidence": {"verify_seal": [], "code_sha": "c0de"}},
        {"id": 4, "title": "Independent sources", "status": "SKIPPED (network unavailable)"},
        {"id": 6, "title": "Negative controls", "status": "FAIL", "method": "oracle and peek models must be flagged",
         "evidence": [{"model": "oracle", "flagged": True}, {"model": "peek", "flagged": False}]},
    ],
}


@pytest.fixture
def live(tmp_path, monkeypatch) -> D.LiveData:
    """Live paths redirected to tmp_path; one payload recorded through the real ledger; a live gold table."""
    fdir = tmp_path / "forecasts"
    for name, p in (("FORECASTS", fdir), ("RUNS", fdir / "runs"), ("LEDGER", fdir / "ledger.jsonl"),
                    ("SCORES", fdir / "scores.csv"), ("RISK_SCORES", fdir / "risk_scores.csv"),
                    ("LIVE_REPORTS", tmp_path / "reports_live"), ("FORWARD_MD", tmp_path / "reports_live" / "f.md")):
        monkeypatch.setattr(paths, name, p)
    entry = ledger.append(make_payload(), stamp=False)
    assert entry["seq"] == 0
    days = pd.date_range("2026-08-01", "2026-10-01", freq="D")
    btc = pd.DataFrame({"asset": "BTC", "session_date": days.date, "tv": [4.0] * (len(days) - 1) + [9.0]})
    spx_days = pd.bdate_range("2026-08-03", "2026-10-01")
    spx = pd.DataFrame({"asset": "SPX", "session_date": spx_days.date, "tv": 0.64})
    (tmp_path / "live" / "holdout").mkdir(parents=True)
    pd.concat([btc, spx]).to_parquet(tmp_path / "live" / "holdout" / "daily.parquet", index=False)
    (tmp_path / "reports_live").mkdir()
    return D.LiveData(fdir, tmp_path / "reports_live", tmp_path / "live")


@pytest.fixture
def app(live, tmp_path):
    return D.create_app(results_dir=tmp_path / "no_results", reports_dir=tmp_path / "no_reports",
                        forecasts_dir=live.forecasts_dir, live_reports_dir=live.reports_dir, live_dir=live.live_dir)


# --------------------------------------------------------------------------------------------- helpers
def _text(c) -> str:
    if c is None:
        return ""
    if isinstance(c, (str, int, float)):
        return str(c)
    if isinstance(c, (list, tuple)):
        return " ".join(_text(x) for x in c)
    return _text(getattr(c, "children", None))


def _walk(c):
    if isinstance(c, (list, tuple)):
        for x in c:
            yield from _walk(x)
    elif hasattr(c, "children"):
        yield c
        yield from _walk(c.children)


def _rows(table_div) -> list[list[str]]:
    table = next(x for x in _walk(table_div) if isinstance(x, html.Table))
    return [[_text(td).strip() for td in tr.children] for tr in _walk(table.children[1]) if isinstance(tr, html.Tr)]


def _row_classes(table_div) -> dict[str, str]:
    table = next(x for x in _walk(table_div) if isinstance(x, html.Table))
    return {_text(tr.children[0]).strip(): tr.className or "" for tr in _walk(table.children[1])
            if isinstance(tr, html.Tr)}


def _post(app, outputs: list[tuple[str, str]], inputs: list[tuple[str, str, object]]) -> dict:
    """Run one callback through Dash's in-process HTTP endpoint (no server) and return its response."""
    if len(outputs) == 1:
        out = f"{outputs[0][0]}.{outputs[0][1]}"
        outs = {"id": outputs[0][0], "property": outputs[0][1]}
    else:
        out = ".." + "...".join(f"{i}.{p}" for i, p in outputs) + ".."
        outs = [{"id": i, "property": p} for i, p in outputs]
    body = {"output": out, "outputs": outs, "state": [],
            "inputs": [{"id": i, "property": p, "value": v} for i, p, v in inputs],
            "changedPropIds": [f"{inputs[0][0]}.{inputs[0][1]}"]}
    r = app.server.test_client().post("/_dash-update-component", json=body)
    assert r.status_code == 200, r.get_data(as_text=True)[:500]
    return r.get_json()["response"]


# --------------------------------------------------------------------------------------------- layout
def test_frozen_app_gets_the_three_live_tabs(app):
    tabs = next(c for c in app.layout._traverse() if isinstance(c, dcc.Tabs))
    assert [t.label for t in tabs.children] == ["Tomorrow", "Forecasts vs realized", "Leaderboard", "VaR breaches",
                                               "Forward test", "Verification"]
    assert tabs.value == "tomorrow"
    ids = {c.id for c in app.layout._traverse() if getattr(c, "id", None)}
    assert {"tomorrow-vol", "tomorrow-vol-table", "tomorrow-risk", "tomorrow-risk-table", "tomorrow-meta",
            "fwd-status", "fwd-graph", "fwd-heatmap", "fwd-table", "fwd-var", "verify-panel", "live-meta",
            "live-refresh"} <= ids
    assert {"asset", "horizon", "split", "models", "risk-models", "forecast-graph", "var-graph"} <= ids  # frozen
    keys = " ".join(app.callback_map)
    for out in ("tomorrow-vol.figure", "fwd-heatmap.figure", "verify-panel.children", "live-meta.children",
                "forecast-graph.figure", "var-graph.figure"):
        assert out in keys
    assert RUN_ID in _text(app.layout)  # header line names the latest ledger run
    assert "live ledger" in _text(app.layout).lower()
    assert ".badge" in app.index_string and "{%app_entry%}" in app.index_string


def test_layout_without_frozen_tabs_is_refused(live):
    with pytest.raises(RuntimeError, match="no dcc.Tabs"):
        D.add_live_tabs(html.Div([html.P("x")]), live)


def test_no_browser_and_no_server_outside_the_command_line(monkeypatch, tmp_path):
    seen = {}

    class FakeApp:
        def run(self, **kw):
            seen["run"] = kw

    class FakeTimer:
        def __init__(self, delay, fn, args=()):
            seen["timer"] = (fn, args)

        def start(self):
            seen["started"] = True

    monkeypatch.setattr(D, "create_app", lambda **kw: FakeApp())
    monkeypatch.setattr(D.threading, "Timer", FakeTimer)
    D.main(["--port", "9131"])
    assert seen == {"run": {"host": "127.0.0.1", "port": 9131, "debug": False}}
    D.main(["--port", "9132", "--no-browser"], open_browser=True)
    assert "timer" not in seen
    D.main(["--port", "9133"], open_browser=True)
    assert seen["timer"] == (D.webbrowser.open, ("http://127.0.0.1:9133/",))


# --------------------------------------------------------------------------------------------- tomorrow
def test_tomorrow_view_shows_the_latest_payload(live):
    vol, vol_table, risk, risk_table, meta = D.tomorrow_view(live, "BTC")
    assert isinstance(vol, go.Figure) and isinstance(risk, go.Figure)
    panels = [t for t in vol.data if t.mode == "markers+text"]
    assert len(panels) == 3  # 1d / 1w / 1m on one shared scale
    one_day = panels[0]
    ys = list(one_day.y)
    combo = ys.index("COMBO (primary)")
    assert one_day.x[combo] == pytest.approx(math.sqrt(4.0 * 1.05 * 365))
    assert one_day.marker.size[combo] == 14 and one_day.text[combo].endswith("%")
    assert all(s == 10 for i, s in enumerate(one_day.marker.size) if i != combo)
    assert "IV" in list(panels[2].y) and "IV" not in ys  # IV benchmarks only at 1m
    assert len(vol.layout.shapes) == 6  # realised + IV reference line in each panel
    xs = sorted(s.x0 for s in vol.layout.shapes)
    assert any(x == pytest.approx(math.sqrt(9.0 * 365)) for x in xs)  # last session's realised (tv = 9)
    assert any(x == pytest.approx(45.0) for x in xs)  # DVOL
    assert "next session 2026-10-02" in vol.layout.annotations[0].text
    rows = _rows(vol_table)
    assert [r[0] for r in rows][:4] == ["HAR", "GARCH", "LGBM", "COMBO"]
    assert rows[-2][0] == "Realised (trailing window)" and rows[-1][0].startswith("Implied vol (DVOL")
    assert _row_classes(vol_table)["COMBO"] == "ref"
    risk_rows = _rows(risk_table)
    assert [r[0] for r in risk_rows] == RISK_MODELS
    assert _row_classes(risk_table)["COMBO+FHS"] == "ref"
    assert risk_rows[4][3] == f"{2.33 * 2.0 * 1.2:.2f}"  # COMBO+FHS VaR99 from the payload
    assert {t.name for t in risk.data if t.showlegend is not False} == {"VaR 99%", "VaR 97.5%", "ES 97.5%"}
    text = _text(meta)
    assert RUN_ID in text and "2026-10-01" in text and "seal intact" in text
    assert "intact · 1 entries" in text  # ledger.verify_chain() on the redirected ledger
    assert "file matches the ledger entry" in text
    assert "not stamped yet" in text  # stamp=False: no .ots proof (offline)
    # BTC session 2026-10-02 opened at 00:00 UTC: recorded during it, from data up to the previous session
    assert "recorded during the session (opened 2026-10-02 00:00, closes 2026-10-03 00:00 UTC)" in text
    assert "started 2026-10-02 06:00 UTC" in text
    assert "written 2026-10-02 06:20 UTC (machine clock, 20 min after the start)" in text
    assert "123,456 sealed rows reproduced" in text and "live data == sealed data (dev 9,000 rows" in text


def test_tomorrow_view_spx_recorded_before_open_and_fallback_asset(live):
    *_, meta = D.tomorrow_view(live, "SPX")
    assert "recorded before it opened (2026-10-02 13:30 UTC)" in _text(meta)
    vol, *_ = D.tomorrow_view(live, "EURUSD")  # not in the payload: the first payload asset is shown
    assert "Bitcoin" in vol.layout.title.text


def test_tampered_payload_is_flagged(live):
    f = live.runs_dir / f"{RUN_ID}.json"
    p = json.loads(f.read_text(encoding="utf-8"))
    p["forecasts"][0]["F"] *= 0.5
    f.write_bytes(schema.canonical_json(p).encode("utf-8"))
    *_, meta = D.tomorrow_view(live, "BTC")
    text = _text(meta)
    assert "FILE DIFFERS from the ledger entry" in text
    assert "BROKEN" in text and "payload edited" in text


def test_stale_implied_vol_is_never_shown_as_current():
    # EVZ ended in 2023 (SPEC §1): its last quote must not appear as today's EUR/USD implied vol
    p = {"implied": [{"asset": "EURUSD", "origin": "2023-12-29", "iv": 7.3, "iv_var_30d": 0.2, "source": "EVZ"}]}
    iv = D.implied_now(p, "EURUSD", "2026-10-01")
    assert math.isnan(iv[0]) and iv[1:] == ("EVZ", "2023-12-29")
    assert D.implied_now(p, "EURUSD", "2024-01-02")[0] == 7.3
    assert D.implied_now(p, "SPX", "2026-10-01") == (pytest.approx(float("nan"), nan_ok=True), "", "")
    fc = pd.DataFrame([{"asset": "EURUSD", "horizon": h, "model": m, "origin": "2026-10-01", "window_first": "x",
                        "window_last": "y", "n_t": n, "F": 0.01 * n, "vol_ann": 5.0}
                       for h, n in (("1d", 1), ("1w", 5), ("1m", 22)) for m in ("HAR", "COMBO")])
    real = {"1d": (5.5, 1), "1w": (5.1, 5), "1m": (4.9, 22)}
    fig = D.tomorrow_vol_figure(fc, "EURUSD", real, iv)
    assert len(fig.layout.shapes) == 3 and not any("Implied" in (t.name or "") for t in fig.data)
    assert _rows(D.tomorrow_vol_table(fc, real, iv))[-1] == ["Implied vol: no current quote (last EVZ 2023-12-29)",
                                                             "—", "—", "—"]


def test_tomorrow_without_any_run_says_so(tmp_path):
    live = D.LiveData(tmp_path / "f", tmp_path / "r", tmp_path / "l")
    vol, _, risk, _, meta = D.tomorrow_view(live, "BTC")
    assert vol.layout.annotations[0].text == D.NO_RUN
    assert D.NO_RUN in _text(meta)


# --------------------------------------------------------------------------------------------- forward test
def test_forward_view_before_any_score_names_the_first_scoring_date(live):
    status, fig, heat, table, var = D.forward_view(live, "BTC", "1d", ["HAR", "COMBO"])
    text = _text(status)
    assert "No scored forecasts yet" in text and "first scores after 2026-10-02" in text
    assert "since 2026-10-02T06:00:00+00:00" in text and "1 run(s) in the ledger" in text
    assert fig.layout.annotations[0].text == "No scored forecasts yet — first scores after 2026-10-02"
    assert len(heat.data) == 0 and "No scored VaR/ES yet" in _text(var)


def test_forward_view_with_scores(live):
    write_scores(live.forecasts_dir)
    status, fig, heat, table, var = D.forward_view(live, "BTC", "1d", ["HAR", "COMBO"])
    text = _text(status)
    assert "1 run(s) in the ledger" in text and "8 forecasts scored" in text and "(8 for BTC)" in text
    assert "Forward test — forecasts recorded before their outcomes" in text
    assert ("written before their first target session opened: 0, during it (from data up to the previous "
            "session): 8") in text
    assert "left out of every table" not in text
    assert [t.type for t in fig.data] == ["bar", "scatter", "scatter"]
    assert [t.name for t in fig.data[1:]] == ["HAR", "COMBO"]
    assert list(fig.data[0].y) == pytest.approx([math.sqrt(3.0 * 365), math.sqrt(6.0 * 365)])
    assert list(fig.data[2].y) == pytest.approx([math.sqrt(4.2 * 365)] * 2)
    assert heat.data[0].type == "heatmap" and "forward-test accuracy" in heat.layout.title.text
    rows = {r[1]: r for r in _rows(table)}
    assert rows["HAR"][2] == "1.000" and rows["HAR"][4] == "2"
    q = {m: np.mean([y / (4.0 * k) - math.log(y / (4.0 * k)) - 1 for y in (3.0, 6.0)]) for m, k in MODELS.items()}
    assert rows["GARCH"][2] == f"{q['GARCH'] / q['HAR']:.3f}"  # = score.ratio_table, = forward_test.md
    vrows = {r[0]: r for r in _rows(var)}
    assert vrows["HS-250"][1] == "1" and vrows["HS-250"][2] == "1" and vrows["HS-250"][5] == "1"
    assert vrows["COMBO+Normal"][2] == "0"  # VaR99 6.66 > loss 5
    other = D.forward_view(live, "SPX", "1d", None)
    assert "No scored forecasts yet" in _text(other[0])


# --------------------------------------------------------------------------------------------- verification
def test_verification_view_renders_badges_method_and_evidence(live):
    assert "No verification report yet" in _text(D.verification_view(live))
    live.verification_path.write_text(json.dumps(VERIFICATION), encoding="utf-8")
    out = D.verification_view(live)
    head, rows = out[0], out[1:]
    assert "3 checks" in _text(head) and "2026-10-02T07:00:00+00:00" in _text(head)
    badges = [next(x for x in _walk(r) if isinstance(x, html.Span) and "badge" in (x.className or ""))
              for r in rows]
    assert [_text(b).strip() for b in badges] == ["PASS", "SKIPPED", "FAIL"]
    assert [b.style["borderColor"] for b in badges] == [D.GREEN, D.A.NA_COLOR, D.RED]
    assert "verify_seal() and the code hash" in _text(rows[0]) and "code_sha" in _text(rows[0])
    assert "Reason: network unavailable" in _text(rows[1])
    assert [r[0] for r in _rows(rows[2])] == ["oracle", "peek"]


# --------------------------------------------------------------------------------------------- callbacks
def test_callbacks_answer_through_the_dash_endpoint(app, live):
    live.verification_path.write_text(json.dumps(VERIFICATION), encoding="utf-8")
    write_scores(live.forecasts_dir)
    r = _post(app, [("verify-panel", "children")], [("live-refresh", "n_intervals", 0)])
    assert "Negative controls" in json.dumps(r)
    r = _post(app, [("tomorrow-vol", "figure"), ("tomorrow-vol-table", "children"), ("tomorrow-risk", "figure"),
                    ("tomorrow-risk-table", "children"), ("tomorrow-meta", "children")],
              [("asset", "value", "SPX"), ("live-refresh", "n_intervals", 0)])
    assert len(r["tomorrow-vol"]["figure"]["data"]) >= 3 and RUN_ID in json.dumps(r["tomorrow-meta"])
    r = _post(app, [("fwd-status", "children"), ("fwd-graph", "figure"), ("fwd-heatmap", "figure"),
                    ("fwd-table", "children"), ("fwd-var", "children")],
              [("asset", "value", "BTC"), ("horizon", "value", "1d"), ("models", "value", ["COMBO"]),
               ("live-refresh", "n_intervals", 0)])
    assert r["fwd-heatmap"]["figure"]["data"][0]["type"] == "heatmap"
    r = _post(app, [("live-meta", "children")], [("live-refresh", "n_intervals", 1)])
    assert RUN_ID in r["live-meta"]["children"]


def test_trailing_realised_uses_calendar_windows():
    daily = pd.DataFrame({"session_date": pd.to_datetime(["2026-09-24", "2026-09-28", "2026-10-01"]),
                          "tv": [1.0, 2.0, 3.0]})
    assert D.trailing_realised(daily, "SPX", date(2026, 10, 1), "1d") == (pytest.approx(math.sqrt(3 * 252)), 1)
    assert D.trailing_realised(daily, "SPX", date(2026, 10, 1), "1w") == (pytest.approx(math.sqrt(2.5 * 252)), 2)
    assert D.trailing_realised(daily, "SPX", date(2026, 10, 1) + timedelta(days=60), "1m")[1] == 0


# --------------------------------------------------------------------------------------------- recording time
def record(run_utc: datetime, written: datetime | None, drop: tuple[str, ...] = ()) -> dict:
    """Append another schema-valid payload to the sandbox ledger (started ``run_utc``, written ``written``),
    without the forecast / VaR rows of the assets in ``drop`` (left out because their session had closed)."""
    p = make_payload()
    p["run_utc"], p["run_id"] = run_utc.isoformat(timespec="seconds"), run_utc.strftime("%Y%m%dT%H%M%SZ")
    if written is None:
        del p["checks"]["computed_utc"]
    else:
        p["checks"]["computed_utc"] = written.isoformat(timespec="seconds")
    p["forecasts"] = [r for r in p["forecasts"] if r["asset"] not in drop]
    p["risk"] = [r for r in p["risk"] if r["asset"] not in drop]
    ledger.append(p, stamp=False)
    return p


def test_write_time_decides_not_the_start_time(live):
    # started 13:00 (before the 13:30 S&P 500 open), written 13:45: the payload's own flag says "before open"
    p = record(datetime(2026, 10, 2, 13, 0, tzinfo=UTC), datetime(2026, 10, 2, 13, 45, tzinfo=UTC))
    assert p["data"]["SPX"]["recorded_before_open"] is True
    text = _text(D.tomorrow_view(live, "SPX")[-1])
    assert "recorded before it opened" not in text
    assert "recorded during the session (opened 2026-10-02 13:30, closes 2026-10-02 20:00 UTC)" in text
    assert "written 2026-10-02 13:45 UTC (machine clock, 45 min after the start)" in text


def test_post_dated_start_and_missing_write_time_are_flagged(live):
    record(datetime(2026, 10, 2, 13, 0, tzinfo=UTC), datetime(2026, 10, 2, 12, 0, tzinfo=UTC))
    assert "start time LATER than the machine clock at writing: post-dated" in _text(D.tomorrow_view(live, "SPX")[-1])
    record(datetime(2026, 10, 2, 14, 0, tzinfo=UTC), None)
    assert "write time not in the payload: timing judged by the start time" in \
        _text(D.tomorrow_view(live, "SPX")[-1])


def test_run_after_the_close_is_never_shown_as_recorded_before_its_outcome(live):
    # 23:30 in London (BST): the S&P 500 session of 2026-10-02 closed at 20:00 UTC
    record(datetime(2026, 10, 2, 22, 30, tzinfo=UTC), datetime(2026, 10, 2, 22, 50, tzinfo=UTC))
    vol, *_, meta = D.tomorrow_view(live, "SPX")
    chip = next(c for c in _walk(meta) if isinstance(c, html.Span) and "AFTER" in _text(c))
    assert "recorded AFTER the session closed (2026-10-02 20:00 UTC)" in _text(chip)
    assert "not forward-test evidence" in _text(chip)
    assert next(x for x in _walk(chip) if getattr(x, "className", "") == "dot").style["background"] == D.RED
    assert "recorded during the session" in _text(D.tomorrow_view(live, "BTC")[-1])  # closes at 24:00 UTC


def test_asset_left_out_after_its_close_is_named(live):
    record(datetime(2026, 10, 2, 22, 30, tzinfo=UTC), datetime(2026, 10, 2, 22, 50, tzinfo=UTC), drop=("SPX",))
    vol, *_, meta = D.tomorrow_view(live, "SPX")
    assert "No forecasts for SPX" in vol.layout.annotations[0].text
    assert "no forecast: the session had closed (2026-10-02 20:00 UTC) when the run was written" in _text(meta)


def test_forward_status_counts_scores_written_after_their_session_closed(live):
    # origin 09-28, BTC session 09-29 closed at 09-30 00:00 UTC; this run was written at 01:00 UTC on 09-30
    rows = score_rows() + score_rows((("2026-09-28", "20260930T010000Z", 5.0),))
    assert sum(not r["before_close"] for r in rows) == 4
    _write_csv(live.scores_path, SC.SCORE_COLUMNS, rows)
    status, fig, heat, table, var = D.forward_view(live, "BTC", "1d", ["HAR"])
    text = _text(status)
    assert "8 forecasts scored" in text  # the late origin is not in the headline scores (score.headline_scores)
    assert "4 forecast and 0 VaR row(s) were written after their first target session had closed" in text
    assert pd.Timestamp("2026-09-28") not in set(fig.data[0].x)  # nor in the forecast-vs-realised chart
    assert {r[4] for r in _rows(table)} == {"2"}  # ratios on the two origins recorded in time


def test_scores_with_unknown_write_time_are_left_out(live):
    rows = score_rows()
    for r in rows:
        r["recorded_utc"], r["before_open"], r["before_close"] = "", "", False  # score.py: unknown -> not before
    _write_csv(live.scores_path, SC.SCORE_COLUMNS, rows)
    text = _text(D.forward_view(live, "BTC", "1d", ["HAR"])[0])
    assert "No scored forecasts yet" in text and "8 forecast and 0 VaR row(s)" in text
    assert "or at an unknown time" in text


# --------------------------------------------------------------------------------------------- timestamps
def write_proof(path, calendars=(), heights=()) -> None:
    """A real OpenTimestamps proof of ``path`` (opentimestamps core) with the given attestations."""
    with open(path, "rb") as f:
        dtf = DetachedTimestampFile.from_fd(OpSHA256(), f)
    leaf = dtf.timestamp.ops.add(OpAppend(b"\x01" * 16)).ops.add(OpSHA256())
    for uri in calendars:
        leaf.attestations.add(PendingAttestation(uri))
    for h in heights:
        leaf.attestations.add(BitcoinBlockHeaderAttestation(h))
    ots.write_proof(dtf, ots.ots_path(path))


def write_8b(live, **record) -> None:
    row = {"payload": f"{RUN_ID}.json", "ots": True, "digest_ok": True, "n_calendars": 1,
           "bitcoin_heights": [900000], "first_outcome_utc": "2026-10-02T20:00:00Z", **record}
    live.verification_path.write_text(json.dumps({
        "generated_utc": "2026-10-02T12:00:00+00:00",
        "checks": [{"id": "8b", "title": "Forecast payloads timestamped", "status": "PASS",
                    "evidence": {"payloads": [row]}}]}), encoding="utf-8")


CALS = ("https://a.pool.opentimestamps.org", "https://b.pool.opentimestamps.org")


def test_stamp_chip_has_a_wording_for_every_ots_status(live, monkeypatch):
    # the chip vocabulary is ots.STATUS_TEXT: a new or renamed status fails here instead of drifting silently
    assert set(D.STAMP_LEVEL) == set(ots.STATUS_TEXT)
    f = live.runs_dir / f"{RUN_ID}.json"
    for s, words in ots.STATUS_TEXT.items():
        claims = s in ("upgraded", "verified", "unverified")
        monkeypatch.setattr(ots, "status", lambda path, s=s, claims=claims: {
            "status": s, "reason": None, "pending_calendars": list(CALS), "block_heights": [900000] if claims else []})
        ok, text = live.stamp_status(f)
        assert ok is not True, s  # never green from the proof file alone
        assert words in text and "unusable" not in text and "unknown" not in text, (s, text)
    monkeypatch.setattr(ots, "status", lambda path: {"status": "brand-new", "reason": None})
    assert live.stamp_status(f) == (None, "brand-new: unknown proof status")


def test_stamp_chip_on_real_proof_files(live):
    f = live.runs_dir / f"{RUN_ID}.json"
    assert live.stamp_status(f) == (None, "not stamped yet: no proof yet (retried by the next daily run)")
    write_proof(f, CALS[:1])
    assert ots.status(f)["status"] == "partial" and live.stamp_status(f)[0] is None
    write_proof(f, CALS)
    ok, text = live.stamp_status(f)
    assert ots.status(f)["status"] == "pending" and ok is None and "2 calendar(s)" in text
    # a hand-written Bitcoin attestation is only a claim: grey, never "anchored", never red "unusable"
    write_proof(f, CALS[:1], heights=(900000,))
    assert ots.status(f)["status"] == "upgraded"
    ok, text = live.stamp_status(f)
    assert ok is None and text.startswith("claims Bitcoin block 900,000") and "verification 8b" in text
    assert "claims Bitcoin block 900,000" in _text(D.tomorrow_view(live, "BTC")[-1])


def test_stamp_chip_is_green_only_after_verification_8b(live):
    f = live.runs_dir / f"{RUN_ID}.json"
    write_proof(f, CALS[:1], heights=(900000,))
    write_8b(live, bitcoin="verified", attested_utc="2026-10-02T08:00:00Z", hours_before_first_outcome=12.0)
    ok, text = live.stamp_status(f)
    assert ok is True
    assert text == ("Bitcoin block 900,000 mined 2026-10-02 08:00 UTC, before the first outcome (2026-10-02 20:00 "
                    "UTC) - verified by verification 8b (2026-10-02T12:00:00+00:00)")
    write_8b(live, bitcoin="verified", attested_utc="2026-10-02T21:00:00Z")
    ok, text = live.stamp_status(f)
    assert ok is False and "NOT before the first outcome" in text
    write_8b(live, bitcoin="failed", reason="merkle root differs")
    assert live.stamp_status(f) == (False, "claims Bitcoin block 900,000: Bitcoin check FAILED in verification 8b")
    write_8b(live, bitcoin="not checked (network disabled)")
    assert live.stamp_status(f)[0] is None
    write_8b(live, bitcoin="verified", attested_utc="2026-10-02T08:00:00Z", bitcoin_heights=[800000])
    ok, text = live.stamp_status(f)
    assert ok is None and "changed after verification 8b" in text


def test_stamp_chip_red_when_the_payload_changed_after_stamping(live):
    f = live.runs_dir / f"{RUN_ID}.json"
    write_proof(f, CALS)
    f.write_bytes(f.read_bytes().replace(b'"seal_ok":true', b'"seal_ok":false'))
    ok, text = live.stamp_status(f)
    assert ots.status(f)["status"] == "invalid" and ok is False and "file digest differs" in text


# --------------------------------------------------------------------------------------------- timeline
def _write_recent(live, origins, F=4.0, models=("COMBO", "HAR", "GJR")) -> None:
    d = live.live_dir / "forecasts"
    d.mkdir(parents=True, exist_ok=True)
    rows = [{"asset": "BTC", "horizon": "1d", "model": m, "origin": pd.Timestamp(o), "n_t": 1, "F": F,
             "split": "holdout"} for o in origins for m in models]
    rows.append({"asset": "SPX", "horizon": "1d", "model": "COMBO", "origin": pd.Timestamp("2026-09-30"), "n_t": 1,
                 "F": 99.0, "split": "holdout"})  # another asset is filtered out
    pd.DataFrame(rows).to_parquet(d / "forecasts.parquet", index=False)


def test_timeline_runs_from_recent_days_to_the_next_session(live):
    _write_recent(live, pd.date_range("2026-09-20", "2026-10-01", freq="D"))
    fig = D.timeline_view(live, "BTC")
    by = {t.name: t for t in fig.data}
    real = by["Realised (what really happened)"]
    assert len(real.x) == D.TIMELINE_SESSIONS and pd.Timestamp(real.x[-1]) == pd.Timestamp("2026-10-01")
    assert real.y[-1] == pytest.approx(math.sqrt(9.0 * 365))  # last complete session
    combo = by["COMBO: forecast made the day before"]
    # the forecast made at origin t is drawn on the NEXT session: origins 09-20..10-01 -> targets 09-21..10-01
    assert pd.Timestamp(combo.x[0]) == pd.Timestamp("2026-09-21") and pd.Timestamp(combo.x[-1]) == pd.Timestamp("2026-10-01")
    assert combo.y[-1] == pytest.approx(math.sqrt(4.0 * 365))
    star = by["COMBO: next session"]
    assert pd.Timestamp(star.x[0]) == pd.Timestamp("2026-10-02") and star.marker.symbol == "star"
    assert star.y[0] == pytest.approx(math.sqrt(4.0 * 1.05 * 365)) and "next session" in star.text[0]
    assert any("next session" in (a.text or "") for a in fig.layout.annotations)
    assert any(n.startswith("COMBO: next week") for n in by) and any(n.startswith("COMBO: next month") for n in by)
    assert any(s.type == "rect" for s in fig.layout.shapes)  # the future is shaded


def test_timeline_without_live_walk_forward_still_shows_realised_and_next_session(live):
    fig = D.timeline_view(live, "BTC")
    names = [t.name for t in fig.data]
    assert "Realised (what really happened)" in names and "COMBO: next session" in names
    assert not any("forecast made the day before" in n for n in names)


def test_timeline_empty_ledger(tmp_path):
    empty = D.LiveData(tmp_path / "f", tmp_path / "r", tmp_path / "l")
    fig = D.timeline_view(empty, "BTC")
    assert D.NO_RUN in json.dumps(fig.to_plotly_json(), ensure_ascii=False) and not any(t.name == "COMBO: next session" for t in fig.data)


def test_layout_has_the_timeline_graph(app):
    assert "tomorrow-timeline" in json.dumps(app.layout.to_plotly_json(), default=str)


# --------------------------------------------------------------------------------------------- demo bundle
def test_resolve_dirs_uses_the_demo_bundle_on_a_fresh_clone(tmp_path, monkeypatch):
    from volrisk import config as C

    demo = tmp_path / "demo"
    (demo / "results").mkdir(parents=True)
    (demo / "results" / "forecasts.parquet").write_bytes(b"x")
    monkeypatch.setattr(paths, "DEMO_RESULTS", demo / "results")
    monkeypatch.setattr(paths, "DEMO_LIVE", demo / "live")
    monkeypatch.setattr(C, "RESULTS", tmp_path / "no_data" / "results")  # fresh clone: no pipeline outputs
    r, l, note = D.resolve_dirs(None, None, demo=False)
    assert (r, l) == (demo / "results", demo / "live") and "demo bundle" in note
    explicit = tmp_path / "mine"
    assert D.resolve_dirs(explicit, None, demo=False)[0] == explicit  # explicit directories win


def test_resolve_dirs_keeps_real_results_when_present(tmp_path, monkeypatch):
    from volrisk import config as C

    real = tmp_path / "data" / "results"
    real.mkdir(parents=True)
    (real / "forecasts.parquet").write_bytes(b"x")
    demo = tmp_path / "demo" / "results"
    demo.mkdir(parents=True)
    (demo / "forecasts.parquet").write_bytes(b"x")
    monkeypatch.setattr(C, "RESULTS", real)
    monkeypatch.setattr(paths, "DEMO_RESULTS", demo)
    assert D.resolve_dirs(None, None, demo=False) == (None, None, "")
    assert D.resolve_dirs(None, None, demo=True)[0] == demo  # --demo forces the bundle


def test_layout_shows_no_machine_specific_paths():
    import json as _json

    from volrisk_live import paths as _paths

    app = D.create_app(results_dir=_paths.DEMO_RESULTS, live_dir=_paths.DEMO_LIVE)
    text = _json.dumps(app.layout.to_plotly_json(), default=str)
    assert _paths.ROOT.as_posix() not in text and str(_paths.ROOT) not in text
    assert D.project_relative(_paths.ROOT / "forecasts") == "forecasts"
