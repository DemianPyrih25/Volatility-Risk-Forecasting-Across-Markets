"""Tests for the Dukascopy ingestion (SPEC §2.2, §3): URLs, jobs, .bi5 decoding, BID/ASK merge, flags."""

from __future__ import annotations

import lzma
import struct
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from volrisk import config as C
from volrisk.data import dukascopy as D
from volrisk.data import http

FIX = Path(__file__).parent / "fixtures"
DAY = date(2024, 1, 2)


def _fixture(instr: str, side: str) -> bytes:
    return (FIX / f"dukascopy_{instr}_{DAY.isoformat()}_{side}.bi5").read_bytes()


def _bi5(t, o, c, l, h, v) -> bytes:  # noqa: E741
    """Synthetic .bi5 body packed per SPEC §2.2: t, open, close, low, high (int32 BE), volume (float32 BE)."""
    body = b"".join(struct.pack(">5if", *map(int, r[:5]), float(r[5])) for r in zip(t, o, c, l, h, v))
    return lzma.compress(body, format=lzma.FORMAT_ALONE)


def _put(raw: Path, instr: str, day: date, side: str, body: bytes) -> None:
    p = D.raw_path(instr, day, side, raw)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)


def _flat_day(raw: Path, day: date, mids: list[float], vols: list[float], instr="EURUSD", scale=1e5, half=1):
    """Write BID/ASK files whose minute i has flat candles at mid ± half point and volume vols[i] per side."""
    t = np.arange(len(mids)) * 60
    px = np.round(np.asarray(mids) * scale).astype(np.int64)
    for side, off in (("BID", -half), ("ASK", half)):
        p = px + off
        _put(raw, instr, day, side, _bi5(t, p, p, p, p, vols))


# ---------------------------------------------------------------------------------------------- URLs & jobs


@pytest.mark.parametrize("month, mm0", [(1, "00"), (9, "08"), (12, "11")])
def test_url_uses_zero_based_month_and_http(month, mm0):
    u = D.url("EURUSD", date(2024, month, 5), "BID")
    assert u == f"http://datafeed.dukascopy.com/datafeed/EURUSD/2024/{mm0}/05/BID_candles_min_1.bi5"
    assert D.url("USA500IDXUSD", date(2013, 3, 1), "ask").endswith("/2013/02/01/ASK_candles_min_1.bi5")


def test_url_rejects_unknown_side():
    with pytest.raises(ValueError):
        D.url("EURUSD", DAY, "MID")


def test_raw_path_layout(tmp_path):
    assert D.raw_path("EURUSD", DAY, "ASK", tmp_path) == tmp_path / "EURUSD" / "2024" / "2024-01-02_ASK.bi5"
    assert D.raw_path("EURUSD", DAY, "BID") == C.RAW / "dukascopy" / "EURUSD" / "2024" / "2024-01-02_BID.bi5"


def test_jobs_skip_saturdays_and_cover_both_sides(tmp_path):
    js = D.jobs("SPX", date(2024, 1, 1), date(2024, 1, 7), raw_dir=tmp_path)  # Mon..Sun
    days = sorted({date.fromisoformat(j.local_path.name[:10]) for j in js})
    assert date(2024, 1, 6) not in days and len(days) == 6
    assert len(js) == 12 and all(j.source == "dukascopy" for j in js)
    assert {j.url.rsplit("/", 1)[1] for j in js} == {"BID_candles_min_1.bi5", "ASK_candles_min_1.bi5"}
    assert all("/USA500IDXUSD/2024/00/" in j.url for j in js)
    assert all(j.local_path.is_relative_to(tmp_path / "USA500IDXUSD" / "2024") for j in js)


def test_jobs_never_today_or_after_data_end(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "_utc_today", lambda: date(2024, 1, 4))
    js = D.jobs("EURUSD", date(2024, 1, 1), date(2024, 1, 31), raw_dir=tmp_path)
    assert max(date.fromisoformat(j.local_path.name[:10]) for j in js) == date(2024, 1, 3)
    monkeypatch.setattr(D, "_utc_today", lambda: date(2030, 1, 1))
    js = D.jobs("EURUSD", C.data_end() - timedelta(days=3), date(2029, 1, 1), raw_dir=tmp_path)
    assert max(date.fromisoformat(j.local_path.name[:10]) for j in js) == C.data_end()


def test_jobs_reject_non_dukascopy_asset():
    with pytest.raises(ValueError):
        D.jobs("BTC", DAY, DAY)


def test_download_counts_and_keeps_manifest_out_of_data(monkeypatch, tmp_path):
    seen = {}

    def fake_fetch_many(jobs, workers=16, **kw):
        seen["manifest"] = http.MANIFEST
        status = ["ok", "empty", "missing"]
        return [
            http.ManifestRow(
                "dukascopy", j.url, str(j.local_path), 200, 1, "", status[i % 3], datetime.now(timezone.utc)
            )
            for i, j in enumerate(jobs[:-1])  # last job pretend-skipped (already in manifest)
        ]

    default_manifest = http.MANIFEST
    monkeypatch.setattr(http, "fetch_many", fake_fetch_many)
    monkeypatch.setattr(D, "_utc_today", lambda: date(2025, 1, 1))
    s = D.download("EURUSD", date(2024, 1, 1), date(2024, 1, 3), raw_dir=tmp_path)  # Mon..Wed -> 6 jobs
    assert s == {"asset": "EURUSD", "jobs": 6, "skipped": 1, "ok": 2, "empty": 2, "missing": 1, "error": 0}
    assert seen["manifest"] == tmp_path / "manifest.parquet"
    assert http.MANIFEST == default_manifest


# ---------------------------------------------------------------------------------------------- decoding


@pytest.mark.parametrize("instr, scale, lo, hi", [("EURUSD", 1e5, 1.08, 1.12), ("USA500IDXUSD", 1e3, 4600, 4900)])
@pytest.mark.parametrize("side", ["BID", "ASK"])
def test_decode_fixture(instr, scale, lo, hi, side):
    df = D.decode_bi5(_fixture(instr, side), DAY, scale)
    assert df.schema == pl.Schema(D.CANDLE_SCHEMA)
    assert df.height == 1440
    assert df["ts"][0] == datetime(2024, 1, 2, tzinfo=timezone.utc)
    assert df["ts"][-1] == datetime(2024, 1, 2, 23, 59, tzinfo=timezone.utc)
    assert (df["ts"].diff().drop_nulls() == timedelta(seconds=60)).all()
    assert lo < df["close"].median() < hi
    # with the open, close, low, high record order, high/low must bracket open and close on every minute
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))  # noqa: E741
    assert (h >= np.maximum(o, c)).all() and (l <= np.minimum(o, c)).all()
    assert (h > np.maximum(o, c)).any()  # not all-flat, so the bracket check has teeth
    assert (df["volume"] >= 0).all() and (df["volume"] > 0).any()


def test_decode_field_order_synthetic():
    raw = _bi5([0, 60], [100, 200], [110, 210], [90, 190], [120, 220], [1.5, 0.0])
    df = D.decode_bi5(raw, date(2020, 2, 29), 100.0)
    assert df.row(0) == (datetime(2020, 2, 29, tzinfo=timezone.utc), 1.0, 1.2, 0.9, 1.1, 1.5)
    assert df.row(1) == (datetime(2020, 2, 29, 0, 1, tzinfo=timezone.utc), 2.0, 2.2, 1.9, 2.1, 0.0)


def test_decode_raw_bytes_follow_spec_layout():
    """SPEC §2.2 bytes, hand-packed: t=0, open=100, close=110, low=90, high=120 (int32 BE), volume=1.5 (f32 BE)."""
    raw = lzma.compress(struct.pack(">5if", 0, 100, 110, 90, 120, 1.5), format=lzma.FORMAT_ALONE)
    r = D.decode_bi5(raw, DAY, 100.0).row(0, named=True)
    assert (r["open"], r["close"], r["low"], r["high"], r["volume"]) == (1.0, 1.1, 0.9, 1.2, 1.5)


@pytest.mark.parametrize("instr, scale", [("EURUSD", 1e5), ("USA500IDXUSD", 1e3)])
@pytest.mark.parametrize("side", ["BID", "ASK"])
def test_decode_fixture_open_continues_previous_close(instr, scale, side):
    """A minute opens near the previous minute's close: catches an open/close byte swap the bracket check misses."""
    df = D.decode_bi5(_fixture(instr, side), DAY, scale)
    o, c, v = (df[k].to_numpy() for k in ("open", "close", "volume"))
    both = (v[1:] > 0) & (v[:-1] > 0)
    assert both.sum() > 1000
    assert np.abs(o[1:] - c[:-1])[both].mean() < 0.5 * np.abs(c[1:] - o[:-1])[both].mean()


def test_decode_empty_body():
    df = D.decode_bi5(b"", DAY, 1e5)
    assert df.height == 0 and df.schema == pl.Schema(D.CANDLE_SCHEMA)


def test_decode_rejects_corrupt_payload():
    with pytest.raises(ValueError):
        D.decode_bi5(lzma.compress(b"x" * 23, format=lzma.FORMAT_ALONE), DAY, 1e5)
    with pytest.raises(ValueError):  # off-grid minute offset
        D.decode_bi5(_bi5([30], [1], [1], [1], [1], [1.0]), DAY, 1e5)


# ---------------------------------------------------------------------------------------------- bronze


@pytest.mark.parametrize("asset, instr, scale", [("EURUSD", "EURUSD", 1e5), ("SPX", "USA500IDXUSD", 1e3)])
def test_build_bronze_from_fixtures(tmp_path, asset, instr, scale):
    raw, out, flags = tmp_path / "raw", tmp_path / "bronze", tmp_path / "flags"
    for side in D.SIDES:
        _put(raw, instr, DAY, side, _fixture(instr, side))
    s = D.build_bronze(asset, raw_dir=raw, out_dir=out, flags_dir=flags)
    assert s["rows"] == 1440 and s["days_merged"] == 1 and s["continuity_flags"] == 0

    df = pl.read_parquet(out / f"asset={asset}" / "year=2024" / "part.parquet")
    assert df.schema == pl.Schema(D.BRONZE_SCHEMA)
    assert df["ts"].is_unique().all() and df["ts"].is_sorted()
    assert df["n_trades"].null_count() == df.height
    bid = D.decode_bi5(_fixture(instr, "BID"), DAY, scale)
    ask = D.decode_bi5(_fixture(instr, "ASK"), DAY, scale)
    for k in ("open", "high", "low", "close"):
        np.testing.assert_allclose(df[k].to_numpy(), (bid[k].to_numpy() + ask[k].to_numpy()) / 2, rtol=1e-12)
    np.testing.assert_allclose(df["bid_close"].to_numpy(), bid["close"].to_numpy())
    np.testing.assert_allclose(df["spread"].to_numpy(), ask["close"].to_numpy() - bid["close"].to_numpy(), atol=1e-9)
    assert (df["spread"] >= 0).all()
    np.testing.assert_allclose(df["volume"].to_numpy(), bid["volume"].to_numpy() + ask["volume"].to_numpy())
    real = (bid["volume"] > 0) | (ask["volume"] > 0)
    assert df["is_real"].to_list() == real.to_list()
    assert s["real_rows"] == int(real.sum())
    assert pl.read_parquet(D.flags_path(asset, flags)).height == 0


def test_merge_mid_volume_and_is_real(tmp_path):
    raw = tmp_path / "raw"
    t = [0, 60, 120]
    _put(raw, "EURUSD", DAY, "BID", _bi5(t, [110000] * 3, [110010] * 3, [109990] * 3, [110020] * 3, [1.0, 0.0, 0.0]))
    _put(raw, "EURUSD", DAY, "ASK", _bi5(t, [110004] * 3, [110014] * 3, [109994] * 3, [110024] * 3, [2.0, 0.5, 0.0]))
    D.build_bronze("EURUSD", raw_dir=raw, out_dir=tmp_path / "b", flags_dir=tmp_path / "f")
    df = pl.read_parquet(D.bronze_path("EURUSD", 2024, tmp_path / "b"))
    r = df.row(0, named=True)
    assert r["open"] == pytest.approx(1.10002) and r["close"] == pytest.approx(1.10012)
    assert r["high"] == pytest.approx(1.10022) and r["low"] == pytest.approx(1.09992)
    assert r["bid_close"] == pytest.approx(1.1001) and r["ask_close"] == pytest.approx(1.10014)
    assert r["spread"] == pytest.approx(0.00004)
    assert df["volume"].to_list() == [3.0, 0.5, 0.0]
    assert df["is_real"].to_list() == [True, True, False]  # one side with volume is enough


def test_continuity_flag_on_synthetic_jump(tmp_path):
    raw, flags = tmp_path / "raw", tmp_path / "flags"
    _flat_day(raw, date(2024, 1, 2), [1.10, 1.10, 1.101], [1, 1, 1])
    # stale non-real candle first, then the first real mid 10% higher -> flagged
    _flat_day(raw, date(2024, 1, 3), [1.101, 1.211, 1.212], [0, 1, 1])
    _flat_day(raw, date(2024, 1, 4), [1.215, 1.214], [1, 1])  # +0.25% -> fine
    s = D.build_bronze("EURUSD", raw_dir=raw, out_dir=tmp_path / "b", flags_dir=flags)
    assert s["continuity_flags"] == 1
    f = pl.read_parquet(D.flags_path("EURUSD", flags))
    assert f.schema == pl.Schema(D.FLAG_SCHEMA)
    r = f.row(0, named=True)
    assert (r["flag"], r["date"], r["prev_date"]) == ("continuity", date(2024, 1, 3), date(2024, 1, 2))
    assert r["prev_mid"] == pytest.approx(1.101) and r["first_mid"] == pytest.approx(1.211)
    assert r["rel_change"] == pytest.approx(1.211 / 1.101 - 1)


def test_continuity_threshold_is_two_percent_both_ways(tmp_path):
    """SPEC §2.2: a first real mid within 2% of the previous last real mid passes; beyond 2% (either sign) flags."""
    assert D.CONTINUITY_TOL == 0.02
    raw, flags = tmp_path / "raw", tmp_path / "flags"
    _flat_day(raw, date(2024, 1, 2), [1.0], [1])
    _flat_day(raw, date(2024, 1, 3), [1.019, 1.0], [1, 1])  # +1.9% -> fine
    _flat_day(raw, date(2024, 1, 4), [1.021, 1.0], [1, 1])  # +2.1% -> flagged
    _flat_day(raw, date(2024, 1, 5), [0.979], [1])  # -2.1% -> flagged
    s = D.build_bronze("EURUSD", raw_dir=raw, out_dir=tmp_path / "b", flags_dir=flags)
    f = pl.read_parquet(D.flags_path("EURUSD", flags))
    assert s["continuity_flags"] == 2
    assert f["date"].to_list() == [date(2024, 1, 4), date(2024, 1, 5)]
    np.testing.assert_allclose(f["rel_change"].to_numpy(), [0.021, -0.021], rtol=1e-9)


def test_continuity_spans_years_and_skips_days_without_real_minutes(tmp_path):
    raw, flags = tmp_path / "raw", tmp_path / "flags"
    _flat_day(raw, date(2023, 12, 29), [1.10, 1.105], [1, 1])
    _flat_day(raw, date(2023, 12, 31), [1.50, 1.50], [0, 0])  # no real minute: not part of the chain
    _flat_day(raw, date(2024, 1, 2), [1.104, 1.11], [1, 1])
    s = D.build_bronze("EURUSD", raw_dir=raw, out_dir=tmp_path / "b", flags_dir=flags, workers=2)
    assert s["years"] == [2023, 2024] and s["rows"] == 6 and s["real_rows"] == 4
    assert s["continuity_flags"] == 0
    assert pl.read_parquet(D.bronze_path("EURUSD", 2023, tmp_path / "b")).height == 4

    _flat_day(raw, date(2024, 1, 3), [1.0, 1.0], [1, 1])  # -9.9% vs 1.11 -> flagged
    s = D.build_bronze("EURUSD", raw_dir=raw, out_dir=tmp_path / "b", flags_dir=flags, workers=2)
    assert s["continuity_flags"] == 1
    assert pl.read_parquet(D.flags_path("EURUSD", flags))["date"].to_list() == [date(2024, 1, 3)]


def test_missing_side_empty_and_corrupt_days(tmp_path):
    raw, out, flags = tmp_path / "raw", tmp_path / "b", tmp_path / "f"
    _flat_day(raw, date(2024, 1, 2), [1.10, 1.10], [1, 1])
    p = np.array([110000, 110000])
    _put(raw, "EURUSD", date(2024, 1, 3), "BID", _bi5([0, 60], p, p, p, p, [1.0, 1.0]))  # ASK file absent
    _put(raw, "EURUSD", date(2024, 1, 4), "BID", _bi5([0, 60], p, p, p, p, [1.0, 1.0]))
    _put(raw, "EURUSD", date(2024, 1, 4), "ASK", b"")  # 0-byte body on one side
    _put(raw, "EURUSD", date(2024, 1, 5), "BID", b"")  # both 0-byte -> just empty
    _put(raw, "EURUSD", date(2024, 1, 5), "ASK", b"")
    _put(raw, "EURUSD", date(2024, 1, 7), "BID", b"not lzma")
    _put(raw, "EURUSD", date(2024, 1, 7), "ASK", b"not lzma")
    _put(raw, "EURUSD", date(2024, 1, 8), "ASK", b"")  # BID absent (404), ASK 0-byte: a side is missing, not empty
    s = D.build_bronze("EURUSD", raw_dir=raw, out_dir=out, flags_dir=flags)
    assert (s["days"], s["days_merged"], s["missing_side"], s["days_empty"], s["decode_errors"]) == (6, 1, 3, 1, 1)
    df = pl.read_parquet(D.bronze_path("EURUSD", 2024, out))
    assert df["ts"].dt.date().unique().to_list() == [date(2024, 1, 2)]
    f = pl.read_parquet(D.flags_path("EURUSD", flags))
    assert sorted(zip(f["date"].to_list(), f["flag"].to_list())) == [
        (date(2024, 1, 3), "missing_side"),
        (date(2024, 1, 4), "missing_side"),
        (date(2024, 1, 7), "decode_error"),
        (date(2024, 1, 8), "missing_side"),
    ]
    assert f.filter(pl.col("date") == date(2024, 1, 8))["detail"].item() == "BID: no file, ASK: 0 rows"


def test_build_bronze_without_raw_files_writes_empty_flags(tmp_path):
    s = D.build_bronze("SPX", raw_dir=tmp_path / "raw", out_dir=tmp_path / "b", flags_dir=tmp_path / "f")
    assert s["rows"] == 0 and s["years"] == []
    assert pl.read_parquet(D.flags_path("SPX", tmp_path / "f")).height == 0
    assert not (tmp_path / "b").exists()


def test_build_bronze_merges_misaligned_minutes_on_common_ts(tmp_path):
    raw = tmp_path / "raw"
    p = np.full(3, 110000)
    _put(raw, "EURUSD", DAY, "BID", _bi5([0, 60, 120], p, p, p, p, [1.0] * 3))
    _put(raw, "EURUSD", DAY, "ASK", _bi5([0, 120], p[:2], p[:2], p[:2], p[:2], [1.0] * 2))
    s = D.build_bronze("EURUSD", raw_dir=raw, out_dir=tmp_path / "b", flags_dir=tmp_path / "f")
    assert (s["misaligned_days"], s["days_merged"], s["missing_side"], s["rows"]) == (1, 1, 0, 2)
    df = pl.read_parquet(D.bronze_path("EURUSD", 2024, tmp_path / "b"))
    assert df["ts"].dt.minute().to_list() == [0, 2]
    f = pl.read_parquet(D.flags_path("EURUSD", tmp_path / "f"))  # listed in the DQ report (SPEC §10)
    assert f.select("date", "flag", "detail").rows() == [(DAY, "misaligned", "BID 3 / ASK 2 / common 2")]


def test_build_bronze_day_without_common_minute_is_missing_side(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "b"
    one = np.array([110000])
    _put(raw, "EURUSD", DAY, "BID", _bi5([0], one, one, one, one, [1.0]))
    _put(raw, "EURUSD", DAY, "ASK", _bi5([60], one, one, one, one, [1.0]))
    s = D.build_bronze("EURUSD", raw_dir=raw, out_dir=out, flags_dir=tmp_path / "f")
    assert (s["days_merged"], s["misaligned_days"], s["missing_side"], s["rows"]) == (0, 0, 1, 0)
    assert not D.bronze_path("EURUSD", 2024, out).exists()  # no 0-row partition
    f = pl.read_parquet(D.flags_path("EURUSD", tmp_path / "f"))
    assert f.select("date", "flag", "detail").rows() == [(DAY, "missing_side", "BID 1 / ASK 1 / common 0")]


def test_rebuild_removes_partitions_that_no_longer_yield_rows(tmp_path):
    """Bronze mirrors the current raw files: stale year partitions must not survive a rebuild."""
    raw, out, flags = tmp_path / "raw", tmp_path / "b", tmp_path / "f"
    _flat_day(raw, date(2022, 12, 30), [1.10, 1.10], [1, 1])
    _flat_day(raw, date(2023, 12, 29), [1.10, 1.10], [1, 1])
    _flat_day(raw, DAY, [1.10, 1.10], [1, 1])
    other = D.bronze_path("SPX", 2024, out)  # another asset's partition is never touched
    other.parent.mkdir(parents=True)
    other.write_bytes(b"x")
    s = D.build_bronze("EURUSD", raw_dir=raw, out_dir=out, flags_dir=flags)
    assert s["rows"] == 6 and all(D.bronze_path("EURUSD", y, out).exists() for y in (2022, 2023, 2024))

    D.raw_path("EURUSD", DAY, "ASK", raw).unlink()  # 2024 now has a one-sided day only
    for side in D.SIDES:  # 2022 has no raw file at all any more
        D.raw_path("EURUSD", date(2022, 12, 30), side, raw).unlink()
    s = D.build_bronze("EURUSD", raw_dir=raw, out_dir=out, flags_dir=flags)
    assert s["years"] == [2023, 2024] and s["rows"] == 2 and s["missing_side"] == 1
    assert sorted(p.parent.name for p in (out / "asset=EURUSD").glob("year=*/part.parquet")) == ["year=2023"]
    assert sorted(p.name for p in (out / "asset=EURUSD").iterdir()) == ["year=2023"]  # empty dirs removed too
    assert other.read_bytes() == b"x"


def test_fixture_copy_through_raw_layout_is_untouched(tmp_path):
    """build_bronze only reads raw files (raw data is immutable, SPEC §2)."""
    raw = tmp_path / "raw"
    for side in D.SIDES:
        _put(raw, "EURUSD", DAY, side, _fixture("EURUSD", side))
    before = {p: p.read_bytes() for p in raw.rglob("*.bi5")}
    D.build_bronze("EURUSD", raw_dir=raw, out_dir=tmp_path / "b", flags_dir=tmp_path / "f")
    assert {p: p.read_bytes() for p in raw.rglob("*.bi5")} == before
