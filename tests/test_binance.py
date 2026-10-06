"""Binance 1m-kline ingestion (SPEC §2.1, §2.3, §3): urls, parsing, checksums, download planning, bronze.

Offline: downloads go through a stub ``fetch`` that serves an in-memory url -> body map (absent url = 404).
"""

from __future__ import annotations

import hashlib
import io
import shutil
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from volrisk.data import binance as B
from volrisk.data import http
from volrisk.data.http import ManifestRow

FIX = Path(__file__).parent / "fixtures"
SYM = "BTCUSDT"
UTC = timezone.utc


# --------------------------------------------------------------------------------------------- helpers


def kline_line(ts: datetime, unit: str = "ms", price: float = 100.0, volume: float = 1.5, n: int = 7) -> str:
    us = int(ts.timestamp()) * 1_000_000 + ts.microsecond
    ot = us if unit == "us" else us // 1000
    ct = ot + (59_999_999 if unit == "us" else 59_999)
    return f"{ot},{price:.8f},{price + 1:.8f},{price - 1:.8f},{price + 0.5:.8f},{volume:.8f},{ct},1.0,{n},0.5,0.5,0"


def make_zip(name: str, lines: list[str], header: bool = False) -> bytes:
    text = "\n".join(lines) + "\n"
    if header:
        text = ("open_time,open,high,low,close,volume,close_time,quote_volume,count,"
                "taker_buy_volume,taker_buy_quote_volume,ignore\n") + text
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(name.replace(".zip", ".csv"), text)
    return buf.getvalue()


def day_lines(day: date, minutes: list[int] | range, unit: str = "ms", price: float = 100.0) -> list[str]:
    t0 = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return [kline_line(t0 + timedelta(minutes=m), unit, price + m) for m in minutes]


def checksum_body(body: bytes, name: str) -> bytes:
    return f"{hashlib.sha256(body).hexdigest()}  {name}".encode()


def put_raw(raw_dir: Path, kind: str, name: str, body: bytes, checksum: bytes | None = None) -> Path:
    """Write a zip (+ CHECKSUM) into the Binance raw layout under ``raw_dir``."""
    p = raw_dir / SYM / kind / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    B.checksum_path(p).write_bytes(checksum if checksum is not None else checksum_body(body, name))
    return p


class FakeServer:
    """Stand-in for ``http.fetch_many``: serves ``files[url]`` (a body, or a list of bodies served in turn).

    Urls in ``errors`` fail like an exhausted retry loop (status ``error``, nothing written).
    """

    def __init__(self, files: dict[str, bytes | list[bytes]], errors: set[str] | None = None):
        self.files = files
        self.errors = errors if errors is not None else set()
        self.calls: list[list[str]] = []
        self.manifests: list[Path] = []

    def add_zip(self, url: str, body: bytes, checksum: bytes | None = None) -> None:
        self.files[url] = body
        self.files[B.checksum_url(url)] = checksum if checksum is not None else checksum_body(
            body if isinstance(body, bytes) else body[-1], url.rsplit("/", 1)[-1])

    def __call__(self, jobs, workers=16, retries=5, timeout=60.0, skip_done=True, flush_every=200):
        self.calls.append([j.url for j in jobs])
        self.manifests.append(http.MANIFEST)
        now = datetime.now(UTC)
        rows = []
        for j in jobs:
            if j.url in self.errors:
                rows.append(ManifestRow("binance", j.url, str(j.local_path), -1, 0, "", "error", now))
                continue
            body = self.files.get(j.url)
            if isinstance(body, list):
                body = body.pop(0) if len(body) > 1 else body[0]
            if body is None:
                rows.append(ManifestRow("binance", j.url, str(j.local_path), 404, 0, "", "missing", now))
                continue
            j.local_path.parent.mkdir(parents=True, exist_ok=True)
            j.local_path.write_bytes(body)
            rows.append(ManifestRow("binance", j.url, str(j.local_path), 200, len(body),
                                    hashlib.sha256(body).hexdigest(), "ok" if body else "empty", now))
        return rows

    @property
    def requested(self) -> list[str]:
        return [u for c in self.calls for u in c]


# --------------------------------------------------------------------------------------------- urls


def test_urls_and_raw_layout(tmp_path):
    m = B.monthly_url(SYM, 2024, 2)
    d = B.daily_url(SYM, date(2025, 1, 1))
    assert m == "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-02.zip"
    assert d == "https://data.binance.vision/data/spot/daily/klines/BTCUSDT/1m/BTCUSDT-1m-2025-01-01.zip"
    assert B.checksum_url(m) == m + ".CHECKSUM"
    assert B.raw_path(m, tmp_path) == tmp_path / SYM / "monthly" / "BTCUSDT-1m-2024-02.zip"
    assert B.raw_path(B.checksum_url(d), tmp_path) == tmp_path / SYM / "daily" / "BTCUSDT-1m-2025-01-01.zip.CHECKSUM"
    assert B.RAW_DIR.parts[-2:] == ("raw", "binance")
    assert B.bronze_path("ETH", 2019, tmp_path) == tmp_path / "asset=ETH" / "year=2019" / "part.parquet"
    # Same directory and naming pattern as dukascopy_{ASSET}.parquet, which quality.read_flags globs.
    assert B.flags_path("ETH", tmp_path) == tmp_path / "binance_ETH.parquet"
    assert B.FLAGS_DIR == http.C.BRONZE / "flags" and B.BRONZE_DIR.parent / "flags" == B.FLAGS_DIR


def test_known_incidents():
    assert B.KNOWN_INCIDENTS == (date(2018, 2, 8), date(2018, 2, 9), date(2019, 5, 15), date(2023, 3, 24))


def test_non_binance_asset_rejected():
    with pytest.raises(ValueError):
        B.download("EURUSD", fetch=FakeServer({}))


# --------------------------------------------------------------------------------------------- parsing


@pytest.mark.parametrize(
    ("name", "day", "first_open"),
    [("BTCUSDT-1m-2024-12-31.zip", date(2024, 12, 31), 92792.05),  # millisecond timestamps
     ("BTCUSDT-1m-2025-01-01.zip", date(2025, 1, 1), 93576.0)],  # microsecond timestamps
)
def test_parse_fixture(name, day, first_open):
    df = B.parse_zip(FIX / name)
    assert df.schema == pl.Schema(B.CANDLE_SCHEMA)
    assert df.height == 1440
    t0 = datetime(day.year, day.month, day.day, tzinfo=UTC)
    assert df["ts"][0] == t0
    assert df["ts"][-1] == t0 + timedelta(minutes=1439)
    assert df["ts"].diff().drop_nulls().unique().to_list() == [timedelta(minutes=1)]
    assert df["open"][0] == pytest.approx(first_open)
    assert df["is_real"].to_list() == (df["volume"] > 0).to_list()
    assert df["n_trades"][0] > 0


def test_fixture_days_are_contiguous_across_unit_switch():
    a, b = B.parse_zip(FIX / "BTCUSDT-1m-2024-12-31.zip"), B.parse_zip(FIX / "BTCUSDT-1m-2025-01-01.zip")
    assert b["ts"][0] - a["ts"][-1] == timedelta(minutes=1)


def test_header_detection(tmp_path):
    day = date(2024, 3, 5)
    name = f"{SYM}-1m-{day}.zip"
    lines = day_lines(day, range(5))
    (tmp_path / "h").mkdir()
    (tmp_path / "n").mkdir()
    with_hdr = tmp_path / "h" / name
    no_hdr = tmp_path / "n" / name
    with_hdr.write_bytes(make_zip(name, lines, header=True))
    no_hdr.write_bytes(make_zip(name, lines, header=False))
    a, b = B.parse_zip(with_hdr), B.parse_zip(no_hdr)
    assert a.height == 5  # header line skipped, first data row kept
    assert a.equals(b)
    assert a["ts"][0] == datetime(2024, 3, 5, tzinfo=UTC)


def test_parse_rejects_mixed_units_and_out_of_period(tmp_path):
    day = date(2024, 3, 5)
    name = f"{SYM}-1m-{day}.zip"
    mixed = tmp_path / "mixed" / name
    mixed.parent.mkdir()
    mixed.write_bytes(make_zip(name, day_lines(day, [0], "ms") + day_lines(day, [1], "us")))
    with pytest.raises(ValueError, match="mixed"):
        B.parse_zip(mixed)
    wrong_day = tmp_path / "wrong" / name
    wrong_day.parent.mkdir()
    wrong_day.write_bytes(make_zip(name, day_lines(date(2024, 3, 6), [0, 1])))
    with pytest.raises(ValueError, match="outside the file period"):
        B.parse_zip(wrong_day)


def test_parse_floors_misaligned_open_time(tmp_path):
    day = date(2024, 3, 5)
    name = f"{SYM}-1m-{day}.zip"
    t0 = datetime(2024, 3, 5, tzinfo=UTC)
    p = tmp_path / name
    p.write_bytes(make_zip(name, [kline_line(t0, "ms"), kline_line(t0 + timedelta(seconds=90, milliseconds=250))]))
    df = B.parse_zip(p)
    assert df["ts"].to_list() == [t0, t0 + timedelta(minutes=1)]


# --------------------------------------------------------------------------------------------- checksum


def test_checksum_verification_on_fixture(tmp_path):
    z = FIX / "BTCUSDT-1m-2024-12-31.zip"
    assert B.read_checksum(B.checksum_path(z)) == hashlib.sha256(z.read_bytes()).hexdigest()
    assert B.verify_checksum(z)
    assert B.verify_checksum(FIX / "BTCUSDT-1m-2025-01-01.zip")
    # A corrupted copy fails; a zip without a CHECKSUM file is never "verified".
    bad = tmp_path / z.name
    body = bytearray(z.read_bytes())
    body[100] ^= 0xFF
    bad.write_bytes(bytes(body))
    assert not B.verify_checksum(bad)
    shutil.copy(B.checksum_path(z), B.checksum_path(bad))
    assert not B.verify_checksum(bad)
    assert B.verify_checksum(bad, B.checksum_path(z)) is False
    lonely = tmp_path / "lonely" / z.name
    lonely.parent.mkdir()
    shutil.copy(z, lonely)
    assert not B.verify_checksum(lonely)


# --------------------------------------------------------------------------------------------- download


def _server_for(days_ok: dict[date, bytes], months_ok: dict[tuple[int, int], bytes]) -> FakeServer:
    srv = FakeServer({})
    for (y, m), body in months_ok.items():
        srv.add_zip(B.monthly_url(SYM, y, m), body)
    for d, body in days_ok.items():
        srv.add_zip(B.daily_url(SYM, d), body)
    return srv


def _all_days(first: date, last: date) -> list[date]:
    return [first + timedelta(days=k) for k in range((last - first).days + 1)]


def _synthetic_server(first: date, last: date, monthly: set[tuple[int, int]]) -> FakeServer:
    days = {d: make_zip(f"{SYM}-1m-{d}.zip", day_lines(d, range(3))) for d in _all_days(first, last)}
    months = {}
    for y, m in monthly:
        ds = [d for d in days if (d.year, d.month) == (y, m)]
        months[(y, m)] = make_zip(f"{SYM}-1m-{y:04d}-{m:02d}.zip", [ln for d in ds for ln in day_lines(d, range(3))])
    return _server_for(days, months)


def test_download_month_planning(tmp_path):
    # Jan: monthly ok. Feb: monthly 404 -> all 29 dailies. Mar: incomplete (end 03-15) -> dailies, no monthly try.
    srv = _synthetic_server(date(2024, 1, 1), date(2024, 3, 31), monthly={(2024, 1)})
    s = B.download("BTC", date(2024, 1, 1), date(2024, 3, 15), workers=2, raw_dir=tmp_path,
                   today=date(2024, 3, 20), fetch=srv)
    req = srv.requested
    assert B.monthly_url(SYM, 2024, 1) in req and B.monthly_url(SYM, 2024, 2) in req
    assert B.monthly_url(SYM, 2024, 3) not in req
    assert not any("2024-01-" in u.rsplit("/", 1)[-1] for u in req if "/daily/" in u)  # never mix
    daily_req = sorted(u for u in req if "/daily/" in u and u.endswith(".zip"))
    assert daily_req == sorted(B.daily_url(SYM, d) for d in _all_days(date(2024, 2, 1), date(2024, 3, 15)))
    assert all(B.checksum_url(u) in req for u in daily_req)
    assert s["monthly_ok"] == 1
    assert s["daily_months"] == ["2024-02", "2024-03"]
    assert s["monthly_404_fallback"] == ["2024-02"]
    assert s["daily_ok"] == 29 + 15
    assert s["daily_missing"] == [] and s["errors"] == []
    assert (tmp_path / SYM / "monthly" / "BTCUSDT-1m-2024-01.zip").exists()
    assert not (tmp_path / SYM / "monthly" / "BTCUSDT-1m-2024-02.zip").exists()
    assert len(list((tmp_path / SYM / "daily").glob("*.zip"))) == 44
    # The manifest of a non-default raw_dir is redirected away from data/raw and restored afterwards.
    assert all(m == tmp_path / "manifest.parquet" for m in srv.manifests)
    assert http.MANIFEST == http.C.RAW / "manifest.parquet"

    # Re-run: files on disk are the cache; only the 404 monthly zip (+ CHECKSUM) is re-checked.
    srv.calls.clear()
    s2 = B.download("BTC", date(2024, 1, 1), date(2024, 3, 15), raw_dir=tmp_path, today=date(2024, 3, 20), fetch=srv)
    assert sorted(srv.requested) == sorted([B.monthly_url(SYM, 2024, 2), B.checksum_url(B.monthly_url(SYM, 2024, 2))])
    assert s2["daily_ok"] == 44

    # Bronze uses the monthly zip for Jan and dailies for Feb/Mar.
    out = B.build_bronze("BTC", tmp_path, tmp_path / "bronze", start=date(2024, 1, 1), end=date(2024, 3, 15))
    assert out["daily_months"] == ["2024-02", "2024-03"]
    assert out["rows"] == 3 * (31 + 29 + 15)
    assert out["rejected"] == {}


def test_download_never_requests_today_or_later(tmp_path):
    srv = _synthetic_server(date(2024, 3, 1), date(2024, 3, 31), monthly={(2024, 3)})
    s = B.download("BTC", date(2024, 3, 1), date(2024, 3, 31), raw_dir=tmp_path, today=date(2024, 3, 10), fetch=srv)
    assert B.monthly_url(SYM, 2024, 3) not in srv.requested  # month not complete before today
    days = sorted(u.rsplit("/", 1)[-1] for u in srv.requested if u.endswith(".zip"))
    assert days == [f"{SYM}-1m-2024-03-{d:02d}.zip" for d in range(1, 10)]
    assert s["daily_ok"] == 9


def test_download_missing_daily_and_defaults(tmp_path):
    srv = _synthetic_server(date(2024, 2, 1), date(2024, 2, 29), monthly=set())
    del srv.files[B.daily_url(SYM, date(2024, 2, 10))]
    s = B.download("BTC", date(2024, 2, 1), date(2024, 2, 29), raw_dir=tmp_path, today=date(2024, 6, 1), fetch=srv)
    assert s["monthly_404_fallback"] == ["2024-02"]
    assert s["daily_missing"] == ["2024-02-10"]
    assert s["daily_ok"] == 28


def test_download_checksum_mismatch_redownloads_once(tmp_path):
    good = make_zip(f"{SYM}-1m-2024-01.zip", day_lines(date(2024, 1, 1), range(3)))
    corrupt = good[:-5] + b"xxxxx"
    url = B.monthly_url(SYM, 2024, 1)
    srv = FakeServer({})
    srv.add_zip(url, [corrupt, good])  # first body is corrupt, the re-download is fine
    s = B.download("BTC", date(2024, 1, 1), date(2024, 1, 31), raw_dir=tmp_path, today=date(2024, 3, 1), fetch=srv)
    assert srv.requested.count(url) == 2
    assert s["monthly_ok"] == 1 and s["errors"] == []
    assert B.verify_checksum(B.raw_path(url, tmp_path))

    # Persistently corrupt -> error after exactly one re-download, the bad zip is removed, no daily fallback.
    url2 = B.monthly_url(SYM, 2024, 2)
    srv.add_zip(url2, [corrupt], checksum=checksum_body(good, "x"))
    s = B.download("BTC", date(2024, 2, 1), date(2024, 2, 29), raw_dir=tmp_path, today=date(2024, 6, 1), fetch=srv)
    assert srv.requested.count(url2) == 2
    assert s["errors"] == ["BTCUSDT-1m-2024-02.zip"]
    assert s["daily_months"] == [] and s["monthly_ok"] == 0
    assert not B.raw_path(url2, tmp_path).exists()


def test_download_empty_range(tmp_path):
    srv = FakeServer({})
    s = B.download("ETH", date(2024, 1, 5), date(2024, 1, 31), raw_dir=tmp_path, today=date(2024, 1, 5), fetch=srv)
    assert s["months"] == 0 and srv.calls == []


# --------------------------------------------------------------------------------------------- bronze


def test_build_bronze_fixtures_across_years(tmp_path):
    raw = tmp_path / "raw"
    for f in ("BTCUSDT-1m-2024-12-31.zip", "BTCUSDT-1m-2025-01-01.zip"):
        put_raw(raw, "daily", f, (FIX / f).read_bytes(), (FIX / (f + ".CHECKSUM")).read_bytes())
    out = tmp_path / "bronze"
    s = B.build_bronze("BTC", raw, out)
    assert s["rows"] == 2880 and s["rows_raw"] == 2880
    assert s["years"] == [2024, 2025]
    assert s["n_gaps"] == 0 and s["moved"] == 0 and s["duplicates"] == 0
    assert s["first_ts"] == "2024-12-31T00:00:00+00:00" and s["last_ts"] == "2025-01-01T23:59:00+00:00"
    df = pl.read_parquet(B.bronze_path("BTC", 2025, out))
    assert df.schema == pl.Schema(B.BRONZE_SCHEMA)
    assert df.height == 1440 and df["ts"].is_sorted() and df["ts"].n_unique() == 1440
    assert df["bid_close"].null_count() == df["ask_close"].null_count() == df["spread"].null_count() == 1440
    assert df["n_trades"].null_count() == 0
    assert pl.read_parquet(B.bronze_path("BTC", 2024, out))["ts"].max() == datetime(2024, 12, 31, 23, 59, tzinfo=UTC)
    # Rebuilding a narrower range removes the stale 2025 partition.
    s = B.build_bronze("BTC", raw, out, end=date(2024, 12, 31))
    assert s["years"] == [2024] and not B.bronze_path("BTC", 2025, out).exists()
    # An empty raw dir (nothing downloaded / wrong path) never wipes existing bronze or its flags.
    flags = B.flags_path("BTC", tmp_path / "flags")
    before = flags.read_bytes()
    s = B.build_bronze("BTC", tmp_path / "nothing", out)
    assert s["rows"] == 0 and s["years"] == [] and B.bronze_path("BTC", 2024, out).exists()
    assert s["flags_path"] is None and flags.read_bytes() == before


def test_build_bronze_dedupe_floor_and_gaps(tmp_path):
    raw = tmp_path / "raw"
    day = date(2024, 1, 1)
    t0 = datetime(2024, 1, 1, tzinfo=UTC)
    lines = day_lines(day, [0, 1, 2, 3, 7, 8])  # gap of 3 minutes (04-06)
    # Misaligned row 00:02:30.5 -> floored onto 00:02 and later in the file -> it is the one kept.
    lines.insert(4, kline_line(t0 + timedelta(minutes=2, seconds=30, milliseconds=500), price=555.0))
    name = f"{SYM}-1m-{day}.zip"
    put_raw(raw, "daily", name, make_zip(name, lines))
    s = B.build_bronze("BTC", raw, tmp_path / "bronze", start=day, end=day)
    assert s["rows_raw"] == 7 and s["rows"] == 6
    assert s["moved"] == 1 and s["moved_by_month"] == {"2024-01": 1}
    assert s["duplicates"] == 1 and s["duplicates_by_month"] == {"2024-01": 1}
    # File order 00:03 -> 00:02:30.5 (backwards) and 00:02:30.5 -> 00:07 break 60-second spacing.
    assert s["spacing_violations"] == 2
    assert (s["n_gaps"], s["gap_minutes"], s["max_gap_minutes"]) == (1, 3, 3)
    df = pl.read_parquet(B.bronze_path("BTC", 2024, tmp_path / "bronze"))
    assert df["ts"].to_list() == [t0 + timedelta(minutes=m) for m in (0, 1, 2, 3, 7, 8)]
    assert df.filter(pl.col("ts") == t0 + timedelta(minutes=2))["open"].item() == 555.0
    assert df["is_real"].all()
    # The pre-dedupe counts survive in the flags file (bronze itself is aligned and unique by then).
    assert s["flags_path"] == str(tmp_path / "flags" / "binance_BTC.parquet")  # default: sibling of out_dir
    fl = pl.read_parquet(s["flags_path"])
    assert fl.schema == pl.Schema(B.FLAG_SCHEMA)
    assert fl.select("date", "flag", "n_rows").rows() == [(day, "duplicate", 1), (day, "moved", 1),
                                                          (day, "spacing", 2)]
    detail = dict(zip(fl["flag"], fl["detail"]))
    assert "2024-01-01T00:02:30.500 -> 2024-01-01T00:02:00.000" in detail["moved"]
    assert f"2024-01-01T00:03:00.000 -> 2024-01-01T00:02:30.500 in {name}" in detail["spacing"]


def test_spacing_check_is_per_file_and_allows_gaps(tmp_path):
    raw = tmp_path / "raw"
    d1, d2 = date(2024, 1, 1), date(2024, 1, 2)
    n1, n2 = f"{SYM}-1m-{d1}.zip", f"{SYM}-1m-{d2}.zip"
    # File 1: a 1-minute and a 10-minute gap (fine), then a row out of order and an exact in-file duplicate.
    put_raw(raw, "daily", n1, make_zip(n1, day_lines(d1, [0, 2, 13, 12, 14, 14])))
    # File 2 starts right after file 1 ends; a step across files is never checked.
    put_raw(raw, "daily", n2, make_zip(n2, day_lines(d2, [0, 1, 2])))
    s = B.build_bronze("BTC", raw, tmp_path / "bronze", tmp_path / "flags", start=d1, end=d2)
    assert s["spacing_violations"] == 2  # 13 -> 12 (negative), 14 -> 14 (zero)
    assert s["duplicates"] == 1 and s["moved"] == 0 and s["rows"] == 5 + 3
    fl = pl.read_parquet(B.flags_path("BTC", tmp_path / "flags"))
    assert fl.filter(pl.col("flag") == "spacing").select("date", "n_rows").rows() == [(d1, 2)]
    # Clean consecutive files: no spacing violation, no flag rows (no incident date in range).
    shutil.rmtree(raw)
    put_raw(raw, "daily", n1, make_zip(n1, day_lines(d1, range(5))))
    s = B.build_bronze("BTC", raw, tmp_path / "bronze", tmp_path / "flags", start=d1, end=d2)
    assert s["spacing_violations"] == s["moved"] == s["duplicates"] == 0
    assert s["flag_rows"] == 0 and pl.read_parquet(B.flags_path("BTC", tmp_path / "flags")).height == 0


def test_flags_for_offset_incident_reach_quality_report(tmp_path):
    # Shape of the real 2018-02-09 incident: a stretch of open times at hh:mm:14.789 between aligned rows.
    raw = tmp_path / "raw"
    day = date(2018, 2, 9)
    t0 = datetime(2018, 2, 9, tzinfo=UTC)
    off = timedelta(seconds=14, milliseconds=789)
    lines = (day_lines(day, range(5))
             + [kline_line(t0 + timedelta(minutes=m) + off, price=200.0 + m) for m in (4, 5, 6)]
             + day_lines(day, [7, 8]))
    name = f"{SYM}-1m-2018-02.zip"
    put_raw(raw, "monthly", name, make_zip(name, lines, header=True))
    flags_dir = tmp_path / "flags"
    s = B.build_bronze("BTC", raw, tmp_path / "bronze", flags_dir, start=date(2018, 2, 1), end=date(2018, 2, 28))
    assert s["moved"] == 3 and s["duplicates"] == 1 and s["spacing_violations"] == 2  # into and out of the stretch
    assert s["rows"] == 9 and s["incidents"] == {"2018-02-08": 0, "2018-02-09": 9}
    df = pl.read_parquet(B.bronze_path("BTC", 2018, tmp_path / "bronze"))
    assert df["ts"].to_list() == [t0 + timedelta(minutes=m) for m in range(9)]
    assert df.filter(pl.col("ts") == t0 + timedelta(minutes=4))["open"].item() == 204.0  # last row kept
    fl = pl.read_parquet(B.flags_path("BTC", flags_dir))
    assert fl.select("date", "flag", "n_rows").rows() == [
        (date(2018, 2, 8), "incident", 0),
        (day, "duplicate", 1),
        (day, "incident", 9),
        (day, "moved", 3),
        (day, "spacing", 2),
    ]
    # quality.read_flags stacks it with a Dukascopy-shaped flag file and keeps every Binance row.
    from volrisk import quality
    from volrisk.data import dukascopy

    duka = pl.DataFrame([{"asset": "EURUSD", "date": date(2018, 2, 9), "flag": "continuity", "detail": "x"}],
                        schema={k: dukascopy.FLAG_SCHEMA[k] for k in ("asset", "date", "flag", "detail")})
    duka.write_parquet(flags_dir / "dukascopy_EURUSD.parquet")
    both = quality.read_flags(flags_dir, until=date(2025, 9, 30))
    got = both.filter(pl.col("source_file") == "binance_BTC")
    assert got.height == fl.height and set(got["flag"]) == {"incident", "duplicate", "moved", "spacing"}
    assert both.filter(pl.col("source_file") == "dukascopy_EURUSD").height == 1


def test_build_bronze_zero_volume_is_not_real(tmp_path):
    raw = tmp_path / "raw"
    day = date(2024, 1, 2)
    t0 = datetime(2024, 1, 2, tzinfo=UTC)
    name = f"{SYM}-1m-{day}.zip"
    put_raw(raw, "daily", name, make_zip(name, [kline_line(t0), kline_line(t0 + timedelta(minutes=1), volume=0.0)]))
    B.build_bronze("BTC", raw, tmp_path / "bronze")
    df = pl.read_parquet(B.bronze_path("BTC", 2024, tmp_path / "bronze"))
    assert df["is_real"].to_list() == [True, False]


def test_build_bronze_never_mixes_and_rejects_unverified(tmp_path):
    raw = tmp_path / "raw"
    jan = [d for d in _all_days(date(2024, 1, 1), date(2024, 1, 31))]
    mname = f"{SYM}-1m-2024-01.zip"
    put_raw(raw, "monthly", mname, make_zip(mname, [ln for d in jan for ln in day_lines(d, range(2))]))
    # A stray daily zip for January with different prices must be ignored (monthly present).
    dname = f"{SYM}-1m-2024-01-05.zip"
    put_raw(raw, "daily", dname, make_zip(dname, day_lines(date(2024, 1, 5), range(2), price=999.0)))
    # A February daily whose CHECKSUM does not match is rejected, not parsed.
    fname = f"{SYM}-1m-2024-02-01.zip"
    put_raw(raw, "daily", fname, make_zip(fname, day_lines(date(2024, 2, 1), range(2))), checksum=b"0" * 64)
    sel = B.select_files("BTC", raw, date(2024, 1, 1), date(2024, 2, 29))
    assert [p.name for p in sel["2024-01"]] == [mname]
    bronze, flags = tmp_path / "bronze", tmp_path / "flags"
    with pytest.raises(B.RejectedFilesError) as ei:  # a rejected file fails the build: nothing is written
        B.build_bronze("BTC", raw, bronze, flags)
    assert ei.value.rejected == {fname: "checksum missing or mismatched"}
    assert not bronze.exists() and not flags.exists()
    # Explicit opt-in: built without the file, which is listed in the flags.
    s = B.build_bronze("BTC", raw, bronze, flags, allow_rejected=True)
    assert s["rows"] == 62
    assert list(s["rejected"]) == [fname]
    df = pl.read_parquet(B.bronze_path("BTC", 2024, bronze))
    assert df["open"].max() < 999.0
    assert s["incidents"] == {d.isoformat(): 0 for d in B.KNOWN_INCIDENTS}
    fl = pl.read_parquet(B.flags_path("BTC", flags))
    rej = fl.filter(pl.col("flag") == "rejected_file")
    assert rej.select("date", "n_rows").rows() == [(date(2024, 2, 1), None)]
    assert rej["detail"][0].startswith(f"{fname}: checksum missing or mismatched; 2024-02-01..2024-02-01 absent")
    assert sorted(fl.filter(pl.col("flag") == "incident")["date"].to_list()) == list(B.KNOWN_INCIDENTS)
    # Without verification the February file is parsed too.
    s = B.build_bronze("BTC", raw, bronze, flags, verify=False)
    assert s["rows"] == 64 and s["rejected"] == {}
    assert "rejected_file" not in pl.read_parquet(B.flags_path("BTC", flags))["flag"].to_list()


def test_rejected_file_never_drops_existing_bronze(tmp_path):
    raw, bronze, flags = tmp_path / "raw", tmp_path / "bronze", tmp_path / "flags"
    names = {}
    for d in (date(2019, 6, 1), date(2020, 6, 1)):
        names[d.year] = f"{SYM}-1m-{d}.zip"
        put_raw(raw, "daily", names[d.year], make_zip(names[d.year], day_lines(d, range(3))))
    assert B.build_bronze("BTC", raw, bronze, flags)["years"] == [2019, 2020]
    outputs = [B.bronze_path("BTC", 2019, bronze), B.bronze_path("BTC", 2020, bronze), B.flags_path("BTC", flags)]
    before = {p: p.read_bytes() for p in outputs}

    # The only 2019 file loses its CHECKSUM match: the 2019 partition must not be deleted as stale.
    z2019 = raw / SYM / "daily" / names[2019]
    B.checksum_path(z2019).write_bytes(b"0" * 64)
    with pytest.raises(B.RejectedFilesError, match="checksum missing or mismatched"):
        B.build_bronze("BTC", raw, bronze, flags)
    assert {p: p.read_bytes() for p in outputs} == before

    # A verified file that fails parsing (rows outside its day) is rejected the same way.
    put_raw(raw, "daily", names[2019], (FIX / "BTCUSDT-1m-2024-12-31.zip").read_bytes())
    with pytest.raises(B.RejectedFilesError, match="parse error"):
        B.build_bronze("BTC", raw, bronze, flags)
    assert {p: p.read_bytes() for p in outputs} == before

    # Only with allow_rejected does the 2019 partition go, and the flags say why.
    s = B.build_bronze("BTC", raw, bronze, flags, allow_rejected=True)
    assert s["years"] == [2020] and not B.bronze_path("BTC", 2019, bronze).exists()
    rej = pl.read_parquet(B.flags_path("BTC", flags)).filter(pl.col("flag") == "rejected_file")
    assert rej["date"].to_list() == [date(2019, 6, 1)] and "parse error" in rej["detail"][0]


def test_checksum_fetch_error_blocks_bronze_until_redownloaded(tmp_path):
    raw, bronze = tmp_path / "raw", tmp_path / "bronze"
    jan = _all_days(date(2024, 1, 1), date(2024, 1, 31))
    url = B.monthly_url(SYM, 2024, 1)
    srv = FakeServer({}, errors={B.checksum_url(url)})  # CHECKSUM request fails transiently, the zip arrives
    srv.add_zip(url, make_zip(f"{SYM}-1m-2024-01.zip", [ln for d in jan for ln in day_lines(d, range(2))]))
    kw = dict(raw_dir=raw, today=date(2024, 3, 1), fetch=srv)
    s = B.download("BTC", date(2024, 1, 1), date(2024, 1, 31), **kw)
    assert s["errors"] == [f"{SYM}-1m-2024-01.zip"] and s["daily_months"] == []
    assert B.raw_path(url, raw).exists() and not B.raw_path(B.checksum_url(url), raw).exists()
    with pytest.raises(B.RejectedFilesError, match="checksum missing or mismatched"):
        B.build_bronze("BTC", raw, bronze, start=date(2024, 1, 1), end=date(2024, 1, 31))
    assert not bronze.exists() and not (tmp_path / "flags").exists()

    # Once the error clears, a re-run fetches just the CHECKSUM and bronze builds.
    srv.errors.clear()
    srv.calls.clear()
    s = B.download("BTC", date(2024, 1, 1), date(2024, 1, 31), **kw)
    assert srv.requested == [B.checksum_url(url)]
    assert s["errors"] == [] and s["monthly_ok"] == 1
    s = B.build_bronze("BTC", raw, bronze, start=date(2024, 1, 1), end=date(2024, 1, 31))
    assert s["rows"] == 62 and s["rejected"] == {}


def test_build_bronze_counts_incident_days(tmp_path):
    raw = tmp_path / "raw"
    day = date(2023, 3, 24)
    name = f"{SYM}-1m-{day}.zip"
    put_raw(raw, "daily", name, make_zip(name, day_lines(day, range(10))))
    s = B.build_bronze("BTC", raw, tmp_path / "bronze")
    assert s["incidents"]["2023-03-24"] == 10
