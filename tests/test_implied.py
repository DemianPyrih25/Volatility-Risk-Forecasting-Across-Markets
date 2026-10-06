"""Tests for implied-vol & reference-series ingestion (SPEC §2.4). Offline: fixtures and stubbed fetchers."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest
import requests

from volrisk.data import implied as I

FIX = Path(__file__).parent / "fixtures"
DVOL_FIX = FIX / "deribit_dvol_btc_sample.json"
VIX_FIX = FIX / "cboe_vix_head.csv"
SP500_FIX = FIX / "fred_sp500_sample.csv"
DAY_MS = 86_400_000


def _utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def _ms(d: date) -> int:
    return int(_utc(d.year, d.month, d.day).timestamp() * 1000)


def _write_snapshot(path: Path, rows: list[list], downloaded_at: str | None = None) -> Path:
    payload: dict = {"currency": "BTC", "data": rows}
    if downloaded_at is not None:
        payload["downloaded_at"] = downloaded_at
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ------------------------------------------------------------------------------------------------ DVOL parse


def test_parse_dvol_fixture():
    df = I.parse_dvol(DVOL_FIX)  # raw Deribit response; download time from usIn (2026)
    assert df.schema == {"origin": pl.Date, "iv": pl.Float64}
    assert df.height == 8
    assert df["origin"][0] == date(2024, 10, 1)
    assert df["iv"][0] == pytest.approx(56.64)  # close of the candle stamped 2024-10-01, not its open
    assert df["origin"][-1] == date(2024, 10, 8)
    assert df["origin"].is_sorted() and df["origin"].n_unique() == df.height


def test_parse_dvol_drops_live_candle():
    # snapshot taken intraday on 2024-10-08: that day's candle is still live
    df = I.parse_dvol(DVOL_FIX, asof=_utc(2024, 10, 8, 12, 30))
    assert df.height == 7 and df["origin"][-1] == date(2024, 10, 7)
    # exactly at the next midnight the 2024-10-08 candle is complete (start + 1 day == download time)
    assert I.parse_dvol(DVOL_FIX, asof=_utc(2024, 10, 9))["origin"][-1] == date(2024, 10, 8)
    assert I.parse_dvol(DVOL_FIX, asof=_utc(2024, 10, 8, 23, 59, 59)).height == 7


def test_parse_dvol_snapshot_time_from_name_and_json(tmp_path):
    rows = json.loads(DVOL_FIX.read_text())["result"]["data"]
    rows = rows + [rows[2]]  # duplicate candle (overlapping pages)
    # no downloaded_at -> file-name date at 00:00 UTC (conservative)
    by_name = I.parse_dvol(_write_snapshot(tmp_path / "dvol_BTC_2024-10-08.json", rows))
    assert by_name.height == 7 and by_name["origin"][-1] == date(2024, 10, 7)
    assert by_name["origin"].n_unique() == by_name.height
    # explicit downloaded_at in the JSON takes precedence over the name
    stored = I.parse_dvol(
        _write_snapshot(tmp_path / "dvol_BTC_2024-10-08b.json", rows, downloaded_at="2024-10-06T08:00:00+00:00")
    )
    assert stored["origin"][-1] == date(2024, 10, 5)
    with pytest.raises(ValueError, match="download time"):
        I.parse_dvol(_write_snapshot(tmp_path / "nodate.json", rows))


def test_parse_dvol_rejects_misaligned_candles(tmp_path):
    rows = [[_ms(date(2024, 10, 1)) + 3_600_000, 50.0, 51.0, 49.0, 50.5]]
    with pytest.raises(ValueError, match="00:00 UTC"):
        I.parse_dvol(_write_snapshot(tmp_path / "dvol_BTC_2024-10-08.json", rows))


# ------------------------------------------------------------------------------------------------ DVOL download


def _deribit_stub(first: date, last: date, page: int):
    """Fake Deribit: candles first..last (close = day index); returns at most ``page`` newest rows per call
    and a continuation (next end_timestamp) like the real API."""
    all_ts = [_ms(first) + k * DAY_MS for k in range((last - first).days + 1)]
    calls: list[dict] = []

    def fetch(url: str, params: dict) -> dict:
        assert url == I.DVOL_URL and params["resolution"] == "1D"
        calls.append(dict(params))
        sel = [t for t in all_ts if params["start_timestamp"] <= t <= params["end_timestamp"]]
        sel = sel[-page:]
        more = bool(sel) and any(params["start_timestamp"] <= t < sel[0] for t in all_ts)
        data = [[t, 1.0, 2.0, 0.5, float((t - all_ts[0]) // DAY_MS)] for t in sel]
        return {"result": {"data": data, "continuation": sel[0] - DAY_MS if more else None}}

    return fetch, calls


def test_download_dvol_paginates_until_continuation_null(tmp_path):
    now = _utc(2024, 3, 10, 15, 0)
    # the stub also serves the live candle of 'today'; download_dvol must never ask for it
    fetch, calls = _deribit_stub(date(2024, 1, 1), date(2024, 3, 10), page=4)
    path = I.download_dvol("btc", tmp_path, start=date(2024, 1, 1), chunk_days=20, now=now, fetch=fetch)

    assert path == tmp_path / "dvol_BTC_2024-03-10.json"
    snap = json.loads(path.read_text())
    assert snap["currency"] == "BTC" and snap["downloaded_at"].startswith("2024-03-10T15:00")
    today_ms = _ms(date(2024, 3, 10))
    assert all(c["end_timestamp"] < today_ms for c in calls)
    # 69 complete days in 4 chunks (20, 20, 20, 9 days) -> 5 + 5 + 5 + 3 pages of <= 4 rows
    assert len(calls) == 18 and snap["n_requests"] == 18
    followed = [c for c in calls if (c["end_timestamp"] + 1) % DAY_MS != 0]
    assert followed and all(c["end_timestamp"] % DAY_MS == 0 for c in followed)  # continuation tokens used

    df = I.parse_dvol(path)
    assert df.height == 69  # every day exactly once, 2024-01-01 .. 2024-03-09
    assert df["origin"][0] == date(2024, 1, 1) and df["origin"][-1] == date(2024, 3, 9)
    assert df["iv"].to_list() == [float(k) for k in range(69)]


def test_download_dvol_reuses_same_day_snapshot_and_validates(tmp_path):
    now = _utc(2024, 3, 10, 15, 0)
    fetch, calls = _deribit_stub(date(2024, 3, 1), date(2024, 3, 10), page=100)
    p1 = I.download_dvol("ETH", tmp_path, start=date(2024, 3, 1), now=now, fetch=fetch)
    n = len(calls)
    assert I.download_dvol("ETH", tmp_path, start=date(2024, 3, 1), now=now, fetch=fetch) == p1
    assert len(calls) == n  # immutable raw snapshot: no new request on the same day
    with pytest.raises(ValueError):
        I.download_dvol("SOL", tmp_path, now=now, fetch=fetch)
    with pytest.raises(RuntimeError, match="error"):
        I.download_dvol("BTC", tmp_path, now=now, fetch=lambda u, p: {"error": {"code": 10}})


# ------------------------------------------------------------------------------------------------ VIX / FRED


def test_parse_vix_fixture():
    df = I.parse_vix(VIX_FIX)
    assert df.schema == {"origin": pl.Date, "iv": pl.Float64}
    assert df.height == 39
    assert df["origin"][0] == date(1990, 1, 2) and df["iv"][0] == pytest.approx(17.24)
    # '%m/%d/%Y': 01/12/1990 is 12 January (not 1 December)
    assert df.filter(pl.col("origin") == date(1990, 1, 12))["iv"].item() == pytest.approx(24.64)
    assert df["origin"][-1] == date(1990, 2, 26)


def test_parse_fred_sp500_fixture():
    df = I.parse_fred(SP500_FIX, "SP500")
    lines = SP500_FIX.read_text().splitlines()[1:]
    n_missing = sum(1 for ln in lines if ln.split(",")[1].strip() in ("", "."))
    assert n_missing > 0
    assert df.schema == {"date": pl.Date, "value": pl.Float64}
    assert df.height == len(lines) - n_missing
    assert df["value"].null_count() == 0
    assert date(2016, 11, 24) not in df["date"].to_list()  # Thanksgiving: empty in the file
    assert df["date"][0] == date(2016, 10, 3) and df["value"][0] == pytest.approx(2161.20)


def test_parse_fred_evz_drops_missing_stale_and_late(tmp_path):
    csv = "\n".join(
        [
            "observation_date,EVZCLS",
            "2023-12-19,7.10",
            "2023-12-20,7.10",  # equal to the previous day's value -> stale
            "2023-12-21,7.10",  # still a repeat of the previous day -> stale
            "2023-12-22,.",
            "2023-12-26,7.10",  # previous day missing: not a repeat of the previous day's value -> kept
            "2023-12-27,7.25",
            "2023-12-27,7.25",  # duplicated row: deduplicated, not 'stale'
            "2023-12-28,",
            "2023-12-29,7.25",  # previous day empty -> kept
            "2023-12-31,7.40",
            "2024-01-02,7.80",  # after evz_last
            "2025-03-10,7.80",
        ]
    )
    p = tmp_path / "fred_EVZCLS_2026-10-02.csv"
    p.write_text(csv + "\n")
    df = I.parse_fred(p, "EVZCLS")
    want = [date(2023, 12, 19), date(2023, 12, 26), date(2023, 12, 27), date(2023, 12, 29), date(2023, 12, 31)]
    assert df["date"].to_list() == want
    assert df["value"].to_list() == [7.10, 7.10, 7.25, 7.25, 7.40]
    assert df.schema == {"date": pl.Date, "value": pl.Float64}
    assert I.evz_last_date() == date(2023, 12, 31)
    # SP500 keeps repeats (no stale rule) and the older 'DATE' header is accepted
    q = tmp_path / "fred_SP500_2026-10-02.csv"
    q.write_text("DATE,SP500\n2024-01-02,4700.0\n2024-01-03,4700.0\n2024-01-04,.\n")
    assert I.parse_fred(q, "SP500")["value"].to_list() == [4700.0, 4700.0]


def test_latest_snapshot(tmp_path):
    for name in [
        "vix_2026-09-30.csv",
        "vix_2026-10-02.csv",
        "vix_2026-10-01.csv",
        "vix_2026-10-05.csv.part",  # incomplete download
        "fred_SP500_2026-10-03.csv",
        "fred_SP500X_2026-12-01.csv",  # other series sharing the prefix text
    ]:
        (tmp_path / name).write_text("x")
    (tmp_path / "vix_2026-10-04.csv").write_text("")  # empty file is ignored
    assert I.latest_snapshot("vix", tmp_path).name == "vix_2026-10-02.csv"
    assert I.latest_snapshot("fred_SP500", tmp_path).name == "fred_SP500_2026-10-03.csv"
    assert I.latest_snapshot("dvol_BTC", tmp_path) is None
    assert I.latest_snapshot("vix", tmp_path / "missing") is None


class _Resp:
    def __init__(self, status: int, body: bytes = b""):
        self.status_code, self.content = status, body


def test_download_vix_and_fred_with_stubbed_http(tmp_path, monkeypatch):
    now = _utc(2026, 10, 2, 9)
    seen: list[tuple[str, dict | None, float, int]] = []

    def fake_get(url, params=None, timeout=60.0, retries=5):
        seen.append((url, params, timeout, retries))
        if url == I.VIX_URL:
            return _Resp(200, VIX_FIX.read_bytes())
        return _Resp(200, b"observation_date,EVZCLS\n2023-12-29,7.25\n")

    monkeypatch.setattr(I, "_http_get", fake_get)
    v = I.download_vix(tmp_path, now=now)
    assert v == tmp_path / "vix_2026-10-02.csv" and I.parse_vix(v).height == 39
    f = I.download_fred("EVZCLS", tmp_path, now=now)
    assert f == tmp_path / "fred_EVZCLS_2026-10-02.csv"
    assert seen[-1][1] == {"id": "EVZCLS"}
    assert seen[-1][2] >= 90 and seen[-1][3] >= 5  # FRED: long timeout, many retries

    # an HTML page served with HTTP 200 is not a snapshot, and never overwrites a good one
    monkeypatch.setattr(I, "_http_get", lambda *a, **k: _Resp(200, b"<html>busy</html>"))
    assert I.download_fred("SP500", tmp_path, now=now) is None
    assert not (tmp_path / "fred_SP500_2026-10-02.csv").exists()
    with pytest.raises(RuntimeError):
        I.download_vix(tmp_path, now=now, force=True)
    assert v.read_bytes() == VIX_FIX.read_bytes()
    monkeypatch.setattr(I, "_http_get", lambda *a, **k: _Resp(503, b"x"))
    assert I.download_vix(tmp_path, now=now) == v  # same-day snapshot reused, no request needed


def test_download_fred_unreachable_returns_none(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(I, "_http_get", lambda *a, **k: None)
    with caplog.at_level(logging.WARNING, logger=I.__name__):
        assert I.download_fred("EVZCLS", tmp_path, now=_utc(2026, 10, 2)) is None
    assert "EVZCLS" in caplog.text
    assert list(tmp_path.iterdir()) == []


def test_http_get_retries_then_gives_up(monkeypatch):
    attempts: list[float] = []

    def boom(url, params=None, timeout=None, headers=None, allow_redirects=True):
        attempts.append(timeout)
        raise requests.Timeout("slow")

    monkeypatch.setattr(I.requests, "get", boom)
    monkeypatch.setattr(I.time, "sleep", lambda s: None)
    assert I._http_get("https://example.invalid", timeout=95.0, retries=5) is None
    assert attempts == [95.0] * 6

    statuses = iter([503, 429, 200])
    monkeypatch.setattr(I.requests, "get", lambda *a, **k: _Resp(next(statuses), b"ok"))
    assert I._http_get("https://example.invalid", retries=5).status_code == 200


# ------------------------------------------------------------------------------------------------ gold


def _raw_dir(tmp_path: Path, with_evz: bool = True) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    rows = json.loads(DVOL_FIX.read_text())["result"]["data"]
    _write_snapshot(raw / "dvol_BTC_2026-10-02.json", rows)
    eth = [[r[0], 0.0, 0.0, 0.0, 20.0] for r in rows]  # constant 20 vol points
    _write_snapshot(raw / "dvol_ETH_2026-10-02.json", eth)
    _write_snapshot(raw / "dvol_ETH_2026-09-01.json", [[r[0], 0, 0, 0, 99.0] for r in rows])  # older
    shutil.copy(VIX_FIX, raw / "vix_2026-10-02.csv")
    if with_evz:
        (raw / "fred_EVZCLS_2026-10-02.csv").write_text(
            "observation_date,EVZCLS\n2023-12-28,7.1\n2023-12-29,7.1\n2024-01-02,7.3\n"
        )
    return raw


def test_build_implied_schema_mapping_and_units(tmp_path):
    out_path = tmp_path / "gold" / "implied.parquet"
    df = I.build_implied(_raw_dir(tmp_path), out_path)
    assert out_path.exists()
    assert pl.read_parquet(out_path).equals(df)
    assert df.schema == I.IMPLIED_SCHEMA
    assert list(df.columns) == ["asset", "origin", "iv", "iv_var_30d", "source"]
    src = dict(df.group_by("asset").agg(pl.col("source").unique()).iter_rows())
    assert src == {"BTC": ["DVOL"], "ETH": ["DVOL"], "SPX": ["VIX"], "EURUSD": ["EVZ"]}
    counts = dict(df.group_by("asset").len().iter_rows())
    assert counts == {"BTC": 8, "ETH": 8, "SPX": 39, "EURUSD": 1}

    # units: iv in vol points, iv_var_30d in %^2 over 30 calendar days
    eth = df.filter(pl.col("asset") == "ETH")
    assert eth["iv"].to_list() == [20.0] * 8  # newest ETH snapshot used
    assert eth["iv_var_30d"][0] == pytest.approx(400 * 30 / 365)
    assert (df["iv_var_30d"] - df["iv"] ** 2 * 30 / 365).abs().max() < 1e-12
    assert df.filter(pl.col("asset") == "EURUSD")["origin"].to_list() == [date(2023, 12, 28)]
    assert df.select(pl.struct("asset", "origin").is_unique().all()).item()


def test_build_implied_without_evz_continues(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger=I.__name__):
        df = I.build_implied(_raw_dir(tmp_path, with_evz=False), tmp_path / "implied.parquet")
    assert "EURUSD" not in df["asset"].to_list()
    assert set(df["asset"].unique()) == {"BTC", "ETH", "SPX"}
    assert "EURUSD IV benchmark unavailable" in caplog.text


def test_build_implied_requires_vix(tmp_path):
    raw = _raw_dir(tmp_path)
    (raw / "vix_2026-10-02.csv").unlink()
    with pytest.raises(FileNotFoundError, match="VIX"):
        I.build_implied(raw, tmp_path / "implied.parquet")


def test_build_implied_keeps_only_session_origins(tmp_path, caplog):
    raw = _raw_dir(tmp_path)
    # CBOE prints VIX on XNYS holidays since 2022-05-30 (GTH); 2004-06-11 predates the default
    # exchange_calendars window (2006-10-02) and must be dropped too
    (raw / "vix_2026-10-02.csv").write_text(
        "DATE,OPEN,HIGH,LOW,CLOSE\n"
        "06/10/2004,15.0,15.5,14.5,15.10\n"
        "06/11/2004,15.0,15.5,14.5,15.04\n"  # Reagan day of mourning
        "06/14/2004,15.0,17.0,14.5,16.83\n"
        "07/01/2022,28.0,29.0,26.0,26.70\n"
        "07/04/2022,27.0,28.0,26.5,27.53\n"  # Independence Day
        "07/05/2022,28.0,30.0,27.0,27.54\n"
        "01/08/2025,17.0,19.0,16.5,17.70\n"
        "01/09/2025,18.0,19.0,17.5,18.07\n"  # Carter day of mourning
        "01/10/2025,18.0,20.0,17.5,19.54\n"
    )
    (raw / "fred_EVZCLS_2026-10-02.csv").write_text(
        "observation_date,EVZCLS\n2023-12-22,7.0\n2023-12-25,7.2\n2023-12-26,7.3\n"  # Dec 25: no FX session
    )
    with caplog.at_level(logging.INFO, logger=I.__name__):
        df = I.build_implied(raw, tmp_path / "implied.parquet")

    spx = df.filter(pl.col("asset") == "SPX")
    assert spx["origin"].to_list() == [
        date(2004, 6, 10),
        date(2004, 6, 14),
        date(2022, 7, 1),
        date(2022, 7, 5),
        date(2025, 1, 8),
        date(2025, 1, 10),
    ]
    # lagging by one row gives the previous session's close, never a holiday print
    lagged = dict(zip(spx["origin"], spx["iv"].shift(1)))
    assert lagged[date(2025, 1, 10)] == pytest.approx(17.70)
    assert lagged[date(2022, 7, 5)] == pytest.approx(26.70)
    assert "SPX: dropping 3 IV rows on non-session dates" in caplog.text
    eur = df.filter(pl.col("asset") == "EURUSD")
    assert eur["origin"].to_list() == [date(2023, 12, 22), date(2023, 12, 26)]
    # crypto trades every day: nothing dropped
    assert df.filter(pl.col("asset") == "BTC").height == 8


def test_build_implied_caps_at_data_end(tmp_path):
    raw = _raw_dir(tmp_path)
    for p in raw.glob("dvol_*"):
        p.unlink()
    start = date(2026, 9, 28)
    rows = [[_ms(start + timedelta(days=k)), 0, 0, 0, 40.0] for k in range(4)]
    for cur in ("BTC", "ETH"):
        _write_snapshot(raw / f"dvol_{cur}_2026-10-05.json", rows)
    df = I.build_implied(raw, tmp_path / "implied.parquet")
    assert df.filter(pl.col("asset") == "BTC")["origin"].max() == date(2026, 9, 30)
