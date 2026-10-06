"""Live update (docs/LIVE_SPEC.md §2) on a synthetic sandbox: end date, wiring of the frozen builders, consistency proof.

Every sealed path (dev/holdout gold, implied, SEALED.json), every live path and the shared raw store + download
manifest are redirected to tmp_path; the frozen downloaders and builders are replaced by recorders (or, for the
Dukascopy re-request tests, the frozen downloader runs with ``http._fetch_one`` stubbed), so nothing touches the
network or the real data.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import polars as pl
import pytest

from volrisk import bars, holdout, io, measures
from volrisk import config as C
from volrisk.data import binance, dukascopy, http, implied
from volrisk_live import context
from volrisk_live import paths as P
from volrisk_live import update as U

FROZEN_END = date(2026, 9, 30)
LIVE_DAY = date(2026, 10, 1)  # a Thursday
NOW = "2026-10-02T16:00:00Z"
DEV_DAYS = [date(2025, 9, 26) + timedelta(days=k) for k in range(5)]  # ... 2025-09-30
HO_DAYS = [date(2025, 10, 1), date(2025, 10, 2), date(2026, 9, 29), date(2026, 9, 30)]
SANDBOX_ASSETS = ("BTC", "SPX")
SPX_SYM = C.asset("SPX").symbol
FROZEN_DUKA_DOWNLOAD = dukascopy.download  # the real frozen function (the recorder replaces it)


def _gold(days: list[date], seed: int) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for a in SANDBOX_ASSETS:
        for d in days:
            row = {"asset": a, "session_date": d}
            for k, dt in measures.GOLD_SCHEMA.items():
                if k in row:
                    continue
                if dt == pl.Int32:
                    row[k] = int(rng.integers(1, 289))
                elif dt == pl.Boolean:
                    row[k] = bool(rng.integers(0, 2))
                else:
                    row[k] = float(rng.normal())
            rows.append(row)
    df = pl.DataFrame(rows, schema=measures.GOLD_SCHEMA)
    # a NaN cell on both sides must compare equal
    return df.with_columns(pl.when(pl.int_range(pl.len()) == 1).then(float("nan")).otherwise(pl.col("z_jump"))
                           .alias("z_jump"))


def _implied(days: list[date]) -> pl.DataFrame:
    rows = [{"asset": a, "origin": d, "iv": 20.0 + i, "iv_var_30d": (20.0 + i) ** 2 * 30 / 365, "source": s}
            for a, s in (("BTC", "DVOL"), ("SPX", "VIX")) for i, d in enumerate(days)]
    return pl.DataFrame(rows, schema=implied.IMPLIED_SCHEMA)


@pytest.fixture
def opened(monkeypatch):
    monkeypatch.setattr(holdout, "openings", lambda: [{"utc": "2026-10-02T15:20:42+00:00"}])


@pytest.fixture
def sandbox(tmp_path, monkeypatch, opened):
    """Sealed gold/implied + SEALED.json + an identical live rebuild (plus one live session) under tmp_path."""
    live = tmp_path / "data" / "live"
    for name, p in {
        "LIVE": live, "LIVE_BRONZE": live / "bronze" / "minute", "LIVE_FLAGS": live / "bronze" / "flags",
        "LIVE_SILVER": live / "silver", "LIVE_GOLD": live / "gold", "LIVE_HOLDOUT": live / "holdout",
        "LIVE_DAILY_DEV": live / "gold" / "daily.parquet", "LIVE_DAILY_HOLDOUT": live / "holdout" / "daily.parquet",
        "LIVE_IMPLIED": live / "implied.parquet", "LIVE_STATE": live / "state.json",
    }.items():
        monkeypatch.setattr(P, name, p)
    sealed = tmp_path / "sealed"
    monkeypatch.setattr(io, "DAILY_DEV", sealed / "gold" / "daily.parquet")
    monkeypatch.setattr(io, "DAILY_HOLDOUT", sealed / "holdout" / "daily.parquet")
    monkeypatch.setattr(io, "IMPLIED", sealed / "gold" / "implied.parquet")
    monkeypatch.setattr(holdout, "SEALED", sealed / "SEALED.json")
    raw = tmp_path / "raw"  # shared raw store + manifest (read by refetch_live_window / completeness)
    monkeypatch.setattr(http, "MANIFEST", raw / "manifest.parquet")
    monkeypatch.setattr(dukascopy, "RAW_DIR", raw / "dukascopy")
    monkeypatch.setattr(binance, "RAW_DIR", raw / "binance")
    monkeypatch.setattr(http, "_fetch_one", _no_network)

    dev, ho = _gold(DEV_DAYS, 1), _gold(HO_DAYS, 2)
    iv = _implied([*DEV_DAYS, *HO_DAYS])
    io.write_parquet(dev, io.DAILY_DEV)
    io.write_parquet(ho, io.DAILY_HOLDOUT)
    io.write_parquet(iv, io.IMPLIED)
    hashes = {"dev_data": holdout.file_sha(io.DAILY_DEV), "holdout_data": holdout.file_sha(io.DAILY_HOLDOUT),
              "implied": holdout.file_sha(io.IMPLIED)}
    holdout.SEALED.write_text(json.dumps({"hashes": hashes}), encoding="utf-8")

    new_ho = _gold([LIVE_DAY], 3)
    new_iv = _implied([LIVE_DAY])
    write_live(dev, pl.concat([ho, new_ho]), pl.concat([iv, new_iv]).sort("asset", "origin"))
    return {"dev": dev, "holdout": ho, "implied": iv, "new_holdout": new_ho, "new_implied": new_iv}


def write_live(dev: pl.DataFrame, ho: pl.DataFrame, iv: pl.DataFrame | None = None) -> None:
    io.write_parquet(dev, P.LIVE_DAILY_DEV)
    io.write_parquet(ho, P.LIVE_DAILY_HOLDOUT)
    if iv is not None:
        io.write_parquet(iv, P.LIVE_IMPLIED)


def _no_network(job, *args, **kwargs):
    raise AssertionError(f"unexpected download of {job.url}")


def _utc(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _duka_row(sym: str, day: date, side: str, status: str, fetched: str, write: bool | None = None):
    """Manifest row for one Dukascopy day file (the file is written for ``ok`` unless ``write=False``)."""
    p = dukascopy.raw_path(sym, day, side)
    body = b"bi5" if status == "ok" else b""
    if (status in ("ok", "empty")) if write is None else write:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(body)
    code = {"ok": 200, "empty": 200, "missing": 404, "error": 503}[status]
    return http.ManifestRow(dukascopy.SOURCE, dukascopy.url(sym, day, side), str(p), code, len(body), "", status,
                            _utc(fetched))


def _binance_zip(asset: str, day: date) -> None:
    z = binance.raw_path(binance.daily_url(C.asset(asset).symbol, day))
    z.parent.mkdir(parents=True, exist_ok=True)
    z.write_bytes(b"zip")
    binance.checksum_path(z).write_text("0" * 64, encoding="ascii")


def _manifest_status() -> dict[str, str]:
    return dict(http.read_manifest().select("url", "status").iter_rows())


# ------------------------------------------------------------------------------------------------ end date


@pytest.mark.parametrize("now, end", [
    (datetime(2026, 10, 2, 16, 0, tzinfo=timezone.utc), date(2026, 10, 1)),
    (datetime(2026, 10, 2, 0, 0, tzinfo=timezone.utc), date(2026, 10, 1)),
    (datetime(2026, 10, 2, 23, 59, 59, tzinfo=timezone.utc), date(2026, 10, 1)),
    (datetime(2026, 10, 2, 1, 0, tzinfo=timezone(timedelta(hours=2))), date(2026, 9, 30)),  # 2026-10-01 23:00Z
    ("2026-10-02T16:00Z", date(2026, 10, 1)),
    ("2026-01-01T00:30:00+00:00", date(2025, 12, 31)),
])
def test_live_end_date_is_the_last_complete_utc_day(now, end):
    assert U.live_end_date(now) == end


def test_live_end_date_rejects_naive_time():
    with pytest.raises(ValueError, match="timezone-aware"):
        U.live_end_date(datetime(2026, 10, 2, 16, 0))


# ------------------------------------------------------------------------------------------------ consistency


def test_check_passes_on_an_identical_rebuild(sandbox):
    rep = U.check_against_sealed()
    assert rep["ok"] and rep["problems"] == []
    assert all(b["ok"] for b in rep["sealed_base"].values())
    n_dev, n_ho = len(DEV_DAYS) * 2, len(HO_DAYS) * 2
    assert rep["dev"]["rows_compared"] == n_dev and rep["dev"]["cells_compared"] == n_dev * 22
    assert rep["holdout"]["rows_compared"] == n_ho  # the live session after the frozen end is not compared ...
    assert rep["live_after_frozen_end"] == {a: {"rows": 1, "first": LIVE_DAY, "last": LIVE_DAY, "scheduled": 1,
                                                "missing": []} for a in SANDBOX_ASSETS}  # ... but reported
    assert rep["implied"]["rows_compared"] == n_dev + n_ho and rep["implied"]["live_rows_after_frozen_end"] == 2
    for part in ("dev", "holdout", "implied"):
        assert rep[part]["equal"] and rep[part]["sha256_sealed"] == rep[part]["sha256_live"]
    assert rep["live_per_asset"]["BTC"] == {"dev_rows": 5, "holdout_rows": 5, "first": DEV_DAYS[0], "last": LIVE_DAY}


@pytest.mark.parametrize("col, sealed_value, live_value", [
    ("rv", 0.1, float(np.nextafter(0.1, 1.0))),  # one ulp
    ("rv", 0.0, -0.0),  # bit for bit
    ("z_jump", float("nan"), None),  # NaN is not null
    ("valid", True, False),
    ("M", 77, 78),
])
def test_check_detects_a_single_changed_cell(sandbox, col, sealed_value, live_value):
    dt = measures.GOLD_SCHEMA[col]
    hit = (pl.col("asset") == "SPX") & (pl.col("session_date") == HO_DAYS[2])
    sealed_ho = sandbox["holdout"].with_columns(pl.when(hit).then(pl.lit(sealed_value, dt)).otherwise(col).alias(col))
    io.write_parquet(sealed_ho, io.DAILY_HOLDOUT)
    seal = json.loads(holdout.SEALED.read_text(encoding="utf-8"))
    seal["hashes"]["holdout_data"] = holdout.file_sha(io.DAILY_HOLDOUT)
    holdout.SEALED.write_text(json.dumps(seal), encoding="utf-8")
    live_ho = sealed_ho.with_columns(pl.when(hit).then(pl.lit(live_value, dt)).otherwise(col).alias(col))
    write_live(sandbox["dev"], pl.concat([live_ho, sandbox["new_holdout"]]))

    with pytest.raises(U.LiveDataMismatch, match=f"holdout: column '{col}' differs in 1 row") as exc:
        U.check_against_sealed()
    rep = exc.value.report
    assert not rep["ok"] and rep["dev"]["equal"] and not rep["holdout"]["equal"]
    assert list(rep["holdout"]["mismatches"]) == [col]
    ex = rep["holdout"]["mismatches"][col]["examples"][0]
    assert (ex["asset"], ex["session_date"]) == ("SPX", HO_DAYS[2])


def test_check_detects_missing_extra_and_duplicate_rows(sandbox):
    dev, ho = sandbox["dev"], sandbox["holdout"]
    extra_dev = dev.head(1).with_columns(session_date=pl.lit(date(2025, 9, 25)))
    write_live(pl.concat([dev, extra_dev, dev.tail(1)]), ho.filter(pl.col("session_date") != HO_DAYS[0]))
    with pytest.raises(U.LiveDataMismatch) as exc:
        U.check_against_sealed()
    rep = exc.value.report
    assert rep["dev"]["only_live"]["n"] == 1 and rep["dev"]["only_live"]["examples"] == [("BTC", date(2025, 9, 25))]
    assert any("duplicate keys" in p for p in rep["dev"]["problems"])
    assert rep["holdout"]["only_sealed"]["n"] == 2  # both assets lost 2025-10-01


def test_check_detects_a_dtype_change(sandbox):
    write_live(sandbox["dev"].with_columns(pl.col("M").cast(pl.Int64)), pl.concat([sandbox["holdout"],
                                                                                    sandbox["new_holdout"]]))
    with pytest.raises(U.LiveDataMismatch, match=r"live dtypes differ from the schema: \{'M': 'Int64'\}"):
        U.check_against_sealed()


def test_check_compares_implied_vol_up_to_the_frozen_end(sandbox):
    iv = sandbox["implied"]
    changed_new = sandbox["new_implied"].with_columns(iv=pl.col("iv") * 2)  # after the frozen end: not compared
    io.write_parquet(pl.concat([iv, changed_new]), P.LIVE_IMPLIED)
    assert U.check_against_sealed()["ok"]
    io.write_parquet(pl.concat([iv.with_columns(iv=pl.col("iv") + 1e-12), changed_new]), P.LIVE_IMPLIED)
    with pytest.raises(U.LiveDataMismatch, match="implied: column 'iv' differs"):
        U.check_against_sealed()
    assert U.check_against_sealed(implied_vol=False)["ok"]


def test_check_refuses_a_sealed_base_that_does_not_match_its_seal(sandbox):
    io.write_parquet(sandbox["dev"].head(3), io.DAILY_DEV)  # sealed file replaced after the seal
    write_live(sandbox["dev"].head(3), pl.concat([sandbox["holdout"], sandbox["new_holdout"]]))
    with pytest.raises(U.LiveDataMismatch, match="sealed dev_data .* does not match its SEALED.json hash"):
        U.check_against_sealed()


def test_check_restricted_to_assets(sandbox):
    dev = sandbox["dev"]
    bad = dev.with_columns(pl.when(pl.col("asset") == "SPX").then(pl.col("tv") * 2).otherwise("tv").alias("tv"))
    write_live(bad, pl.concat([sandbox["holdout"], sandbox["new_holdout"]]))
    assert U.check_against_sealed(assets=["BTC"])["ok"]
    with pytest.raises(U.LiveDataMismatch, match="dev: column 'tv' differs in 5 row"):
        U.check_against_sealed()


def test_live_reads_require_the_logged_opening(sandbox, monkeypatch):
    monkeypatch.setattr(holdout, "openings", lambda: [])
    for fn in (U.check_against_sealed, U.live_daily, U.live_implied, U.last_sessions):
        with pytest.raises(context.HoldoutNotOpenedError):
            fn()


# ------------------------------------------------------------------------------------------------ loaders


def test_live_loaders(sandbox):
    df = U.live_daily()
    assert list(df.columns) == measures.GOLD_COLUMNS
    assert len(df) == 2 * (len(DEV_DAYS) + len(HO_DAYS) + 1)
    assert df[["asset", "session_date"]].equals(df[["asset", "session_date"]].sort_values(
        ["asset", "session_date"], ignore_index=True))
    assert str(df["session_date"].dtype) == "datetime64[ms]" and str(df["M"].dtype) == "int32"
    assert U.live_daily("SPX")["asset"].unique().tolist() == ["SPX"]
    iv = U.live_implied("BTC")
    assert iv["origin"].is_monotonic_increasing and iv["origin"].iloc[-1] == np.datetime64("2026-10-01")
    assert U.last_sessions() == {"BTC": LIVE_DAY, "SPX": LIVE_DAY}


def test_live_daily_refuses_holdout_rows_in_the_dev_file(sandbox):
    write_live(pl.concat([sandbox["dev"], sandbox["holdout"]]), sandbox["holdout"])
    with pytest.raises(RuntimeError, match="contains holdout sessions"):
        U.live_daily()


# ------------------------------------------------------------------------------------------------ update wiring


@pytest.fixture
def recorder(sandbox, monkeypatch):
    """Replace the frozen downloaders/builders by recorders; build_gold / build_implied write the live tables."""
    calls: list[tuple[str, tuple, dict, date]] = []
    gold_frames = {"dev": sandbox["dev"], "holdout": pl.concat([sandbox["holdout"], sandbox["new_holdout"]])}

    def rec(name, result=None, side_effect=None):
        def fn(*args, **kwargs):
            calls.append((name, args, kwargs, C.data_end()))
            if side_effect is not None:
                side_effect(*args, **kwargs)
            return result() if callable(result) else result
        return fn

    def gold(assets, silver_dir, gold_dir, holdout_dir):
        io.write_parquet(gold_frames["dev"], gold_dir / "daily.parquet")
        io.write_parquet(gold_frames["holdout"], holdout_dir / "daily.parquet")

    iv_all = pl.concat([sandbox["implied"], sandbox["new_implied"]]).sort("asset", "origin")
    real_refetch, record_refetch = U.refetch_live_window, rec("refetch_live_window")

    def refetch(*args, **kwargs):  # recorded, then the real function (it reads the sandbox manifest)
        record_refetch(*args, **kwargs)
        return real_refetch(*args, **kwargs)

    monkeypatch.setattr(U, "refetch_live_window", refetch)
    monkeypatch.setattr(binance, "download", rec("binance.download", {"errors": []}))
    monkeypatch.setattr(dukascopy, "download", rec("dukascopy.download", {"ok": 2}))
    monkeypatch.setattr(implied, "download_all", rec("implied.download_all", {"VIX": P.LIVE / "vix_x.csv"}))
    monkeypatch.setattr(binance, "build_bronze", rec("binance.build_bronze", {"rows": 1, "moved_by_month": {}}))
    monkeypatch.setattr(dukascopy, "build_bronze", rec("dukascopy.build_bronze", {"rows": 1, "real_share": np.nan}))
    monkeypatch.setattr(bars, "build_bars", rec("bars.build_bars", {"first_session": date(2018, 1, 1)}))
    monkeypatch.setattr(measures, "build_gold", rec("measures.build_gold", {"BTC": {"start": date(2018, 1, 1)}},
                                                    side_effect=gold))
    monkeypatch.setattr(implied, "build_implied", rec("implied.build_implied", lambda: iv_all,
                                                      side_effect=lambda out_path: io.write_parquet(iv_all, out_path)))
    return {"calls": calls, "gold": gold_frames}


def _no_nan(s):
    raise ValueError(f"non-JSON constant {s}")


def test_update_wires_the_frozen_builders_to_the_live_dirs(recorder, tmp_path):
    state = U.update(NOW, assets=SANDBOX_ASSETS)
    assert C.data_end() == FROZEN_END  # override restored
    calls = recorder["calls"]
    assert [c[0] for c in calls] == [
        "binance.download", "refetch_live_window", "dukascopy.download", "implied.download_all",
        "binance.build_bronze", "bars.build_bars", "dukascopy.build_bronze", "bars.build_bars",
        "measures.build_gold", "implied.build_implied",
    ]
    assert all(c[3] == LIVE_DAY for c in calls)  # every frozen call ran under live_end(2026-10-01)
    by = {}
    for name, args, kw, _ in calls:
        by.setdefault(name, []).append((args, kw))
    assert by["binance.build_bronze"] == [(("BTC",), {"out_dir": P.LIVE_BRONZE, "flags_dir": P.LIVE_FLAGS,
                                                       "end": LIVE_DAY})]
    assert by["dukascopy.build_bronze"] == [(("SPX",), {"out_dir": P.LIVE_BRONZE, "flags_dir": P.LIVE_FLAGS})]
    assert [kw for _, kw in by["bars.build_bars"]] == [
        {"bronze_dir": P.LIVE / "bronze", "out_dir": P.LIVE_SILVER, "end": LIVE_DAY}] * 2
    assert by["measures.build_gold"] == [((SANDBOX_ASSETS,), {"silver_dir": P.LIVE_SILVER, "gold_dir": P.LIVE_GOLD,
                                                             "holdout_dir": P.LIVE_HOLDOUT})]
    assert by["implied.build_implied"] == [((), {"out_path": P.LIVE_IMPLIED})]
    for _, args, kw, _ in calls:  # nothing aimed at a sealed location
        for v in kw.values():
            if hasattr(v, "resolve"):
                assert v.resolve().is_relative_to(tmp_path.resolve())
                assert not any(v.resolve().is_relative_to(t.resolve()) for t in P.SEALED_TARGETS)

    assert state["end"] == LIVE_DAY and state["frozen_data_end"] == FROZEN_END
    assert state["last_sessions"] == {"BTC": LIVE_DAY, "SPX": LIVE_DAY}
    assert state["expected_last_sessions"] == {"BTC": LIVE_DAY, "SPX": LIVE_DAY} and state["stale_assets"] == []
    assert state["check"]["ok"] and state["download"]["implied"] == {"VIX": "vix_x.csv"}
    assert state["download"]["refetch"]["SPX"]["requested"] == 0 and "BTC" not in state["download"]["refetch"]
    # every session is in the gold, but the sandbox raw store holds no file: reported, not hidden
    assert state["missing_sessions"] == {} and state["incomplete_assets"] == ["BTC", "SPX"]
    assert state["completeness"]["SPX"]["raw_not_ok"][0]["status"] == "not_requested"
    assert "moved_by_month" not in state["bronze"]["BTC"]
    on_disk = json.loads(P.LIVE_STATE.read_text(encoding="utf-8"), parse_constant=_no_nan)
    assert on_disk["end"] == "2026-10-01" and on_disk["last_sessions"] == {"BTC": "2026-10-01", "SPX": "2026-10-01"}
    assert on_disk["bronze"]["SPX"]["real_share"] is None  # NaN -> null
    assert on_disk["files"]["live_dev_gold"]["sha256"] == holdout.file_sha(P.LIVE_DAILY_DEV)
    assert U.read_state() == on_disk


def test_update_without_download_and_with_a_stale_asset(recorder):
    recorder["gold"]["holdout"] = recorder["gold"]["holdout"].filter(
        ~((pl.col("asset") == "SPX") & (pl.col("session_date") == LIVE_DAY)))
    state = U.update(NOW, assets=SANDBOX_ASSETS, download=False)
    assert not any(c[0].endswith("download") or c[0].endswith("download_all") for c in recorder["calls"])
    assert state["download"] is None
    assert state["last_sessions"] == {"BTC": LIVE_DAY, "SPX": FROZEN_END} and state["stale_assets"] == ["SPX"]
    assert state["missing_sessions"] == {"SPX": [LIVE_DAY]} and "SPX" in state["incomplete_assets"]


def test_update_records_a_failed_check_and_raises(recorder):
    recorder["gold"]["dev"] = recorder["gold"]["dev"].with_columns(pl.col("rv") * (1 + 1e-15))
    with pytest.raises(U.LiveDataMismatch, match="dev: column 'rv' differs"):
        U.update(NOW, assets=SANDBOX_ASSETS)
    assert C.data_end() == FROZEN_END
    st = U.read_state()
    assert st["check"]["ok"] is False and "column 'rv' differs" in st["check"]["error"]
    assert st["check"]["dev"]["mismatches"]["rv"]["n"] > 0


def test_update_continues_when_the_implied_download_fails(recorder, monkeypatch):
    def boom():
        raise RuntimeError("Deribit unreachable")

    monkeypatch.setattr(implied, "download_all", boom)
    state = U.update(NOW, assets=SANDBOX_ASSETS)
    assert "Deribit unreachable" in state["download"]["implied"]["error"] and state["check"]["ok"]


def test_update_rejects_future_now_and_unknown_assets(recorder):
    future = datetime.now(timezone.utc) + timedelta(days=2)
    with pytest.raises(ValueError, match="in the future"):
        U.update(future)
    with pytest.raises(ValueError, match="subset"):
        U.update(NOW, assets=["DOGE"])
    assert recorder["calls"] == []


def test_update_requires_the_logged_opening(recorder, monkeypatch):
    monkeypatch.setattr(holdout, "openings", lambda: [])
    with pytest.raises(context.HoldoutNotOpenedError):
        U.update(NOW, assets=SANDBOX_ASSETS)
    assert recorder["calls"] == []


# ------------------------------------------------------------------------------------------------ live window gaps


def _fake_fetch(published: set[date], calls: list[str], fetched: str):
    """``http._fetch_one`` stand-in: 200 with a body for a published day, else 404 (as Dukascopy answers)."""

    def fetch_one(job, session, retries, timeout):
        calls.append(job.url)
        day = date.fromisoformat(job.local_path.name[:10])
        if day not in published:
            return http.ManifestRow(job.source, job.url, str(job.local_path), 404, 0, "", "missing", _utc(fetched))
        job.local_path.parent.mkdir(parents=True, exist_ok=True)
        job.local_path.write_bytes(b"bi5")
        return http.ManifestRow(job.source, job.url, str(job.local_path), 200, 3, "", "ok", _utc(fetched))

    return fetch_one


def test_an_early_404_is_requested_again_and_reported_until_then(recorder, monkeypatch):
    """Regression: a Dukascopy day asked for before publication (404) must not be lost for good.

    Run 1 (00:30 UTC) gets 404 for 2026-10-01; the frozen downloader records it as ``missing`` and would never ask
    again (``skip_done``). Run 2 (16:00 UTC, the day is published) must fetch it; until then it is reported.
    """
    # the real frozen downloader, from the first live day (the sealed-period days are not part of this test)
    monkeypatch.setattr(dukascopy, "download", lambda a, **kw: FROZEN_DUKA_DOWNLOAD(a, start=LIVE_DAY, **kw))
    monkeypatch.setattr(dukascopy, "_utc_today", lambda: date(2026, 10, 2))
    _binance_zip("BTC", LIVE_DAY)
    files = [dukascopy.raw_path(SPX_SYM, LIVE_DAY, s) for s in dukascopy.SIDES]
    no_spx_live = ~((pl.col("asset") == "SPX") & (pl.col("session_date") == LIVE_DAY))
    full_holdout = recorder["gold"]["holdout"]

    calls: list[str] = []
    monkeypatch.setattr(http, "_fetch_one", _fake_fetch(set(), calls, "2026-10-02T00:30Z"))
    recorder["gold"]["holdout"] = full_holdout.filter(no_spx_live)  # no raw data -> no gold row
    s1 = U.update("2026-10-02T00:30:00Z", assets=SANDBOX_ASSETS)
    assert sorted(calls) == sorted(dukascopy.url(SPX_SYM, LIVE_DAY, s) for s in dukascopy.SIDES)
    assert s1["download"]["SPX"]["missing"] == 2 and s1["download"]["refetch"]["SPX"]["requested"] == 0
    assert s1["stale_assets"] == ["SPX"] and s1["missing_sessions"] == {"SPX": [LIVE_DAY]}
    assert s1["incomplete_assets"] == ["SPX"]  # BTC has its zip and its session
    assert [(r["file"], r["status"], r["sessions"]) for r in s1["completeness"]["SPX"]["raw_not_ok"]] == [
        (f.name, "missing", [LIVE_DAY]) for f in sorted(files, key=lambda p: p.name)]

    calls.clear()
    monkeypatch.setattr(http, "_fetch_one", _fake_fetch({LIVE_DAY}, calls, "2026-10-02T16:00Z"))
    recorder["gold"]["holdout"] = full_holdout
    s2 = U.update(NOW, assets=SANDBOX_ASSETS)
    assert sorted(calls) == sorted(dukascopy.url(SPX_SYM, LIVE_DAY, s) for s in dukascopy.SIDES)  # asked again
    rf = s2["download"]["refetch"]["SPX"]
    assert (rf["requested"], rf["ok"], rf["missing"], rf["not_ok"]) == (2, 2, 0, [])
    assert s2["download"]["SPX"]["skipped"] == s2["download"]["SPX"]["jobs"]  # the frozen downloader would not
    assert all(f.read_bytes() == b"bi5" for f in files)
    assert all(_manifest_status()[dukascopy.url(SPX_SYM, LIVE_DAY, s)] == "ok" for s in dukascopy.SIDES)
    assert s2["stale_assets"] == [] and s2["missing_sessions"] == {} and s2["incomplete_assets"] == []
    assert s2["check"]["ok"]

    calls.clear()  # run 3: everything settled, nothing is requested again
    s3 = U.update(NOW, assets=SANDBOX_ASSETS)
    assert calls == [] and s3["download"]["refetch"]["SPX"]["requested"] == 0


def test_refetch_selects_only_unsettled_live_window_files(sandbox, monkeypatch):
    monkeypatch.setattr(dukascopy, "_utc_today", lambda: date(2026, 10, 6))
    d = {k: date(2026, 10, k) for k in range(1, 6)}  # Thu 1, Fri 2, Sat 3 (no file), Sun 4, Mon 5
    rows = [
        _duka_row(SPX_SYM, FROZEN_END, "BID", "missing", "2026-10-01T00:10Z"),  # sealed period: never touched
        _duka_row(SPX_SYM, d[1], "BID", "ok", "2026-10-02T16:00Z"),  # settled
        _duka_row(SPX_SYM, d[1], "ASK", "ok", "2026-10-02T02:00Z"),  # exactly SETTLE after the day: settled
        _duka_row(SPX_SYM, d[2], "BID", "missing", "2026-10-03T00:10Z"),  # 404 before publication
        _duka_row(SPX_SYM, d[2], "ASK", "ok", "2026-10-03T00:20Z"),  # fetched 20 min after the day ended
        _duka_row(SPX_SYM, d[4], "BID", "empty", "2026-10-05T09:00Z"),  # empty body
        _duka_row(SPX_SYM, d[4], "ASK", "error", "2026-10-05T09:00Z"),  # retried by the frozen download itself
    ]  # d[5]: never requested -> the frozen download
    http._write_manifest(rows)
    calls: list[str] = []
    monkeypatch.setattr(http, "_fetch_one", _fake_fetch(set(d.values()), calls, "2026-10-06T10:00Z"))

    res = U.refetch_live_window("SPX", d[5])
    want = [dukascopy.url(SPX_SYM, d[2], "BID"), dukascopy.url(SPX_SYM, d[2], "ASK"),
            dukascopy.url(SPX_SYM, d[4], "BID")]
    assert sorted(calls) == sorted(want)
    assert res["jobs"] == 8 and res["requested"] == 3 and res["ok"] == 3 and res["not_ok"] == []
    assert (res["first"], res["end"]) == (LIVE_DAY, d[5]) and C.data_end() == FROZEN_END
    st = _manifest_status()
    assert [st[u] for u in want] == ["ok"] * 3 and st[rows[0].url] == "missing"

    calls.clear()  # the frozen downloader then fetches the rest once: the error and the never-requested day
    with context.live_end(d[5]):
        summary = FROZEN_DUKA_DOWNLOAD("SPX", start=LIVE_DAY)
    assert sorted(calls) == sorted([dukascopy.url(SPX_SYM, d[4], "ASK"), dukascopy.url(SPX_SYM, d[5], "BID"),
                                    dukascopy.url(SPX_SYM, d[5], "ASK")])
    assert summary["jobs"] == 8 and summary["ok"] == 3

    with pytest.raises(ValueError, match="not a Dukascopy asset"):
        U.refetch_live_window("BTC", d[5])


def test_a_missing_session_in_the_middle_is_reported(recorder, monkeypatch):
    """SPX 2026-10-01 missing, 2026-10-02 present: the last session is current (not stale), the gap is reported."""
    monkeypatch.setattr(U, "_wall_clock", lambda: _utc("2026-10-03T17:00Z"))
    d2 = date(2026, 10, 2)
    new = _gold([LIVE_DAY, d2], 4).filter(~((pl.col("asset") == "SPX") & (pl.col("session_date") == LIVE_DAY)))
    sealed_part = recorder["gold"]["holdout"].filter(pl.col("session_date") <= FROZEN_END)
    recorder["gold"]["holdout"] = pl.concat([sealed_part, new])
    state = U.update("2026-10-03T16:00:00Z", assets=SANDBOX_ASSETS, download=False)
    assert state["end"] == d2 and state["last_sessions"] == {"BTC": d2, "SPX": d2} and state["stale_assets"] == []
    assert state["missing_sessions"] == {"SPX": [LIVE_DAY]} and "SPX" in state["incomplete_assets"]
    assert state["completeness"]["SPX"]["scheduled"] == 2 and state["completeness"]["BTC"]["missing_sessions"] == []
    assert state["check"]["ok"]  # nothing sealed differs; the gap is in the proof's report (and so in payloads)
    assert state["check"]["live_after_frozen_end"]["SPX"] == {"rows": 1, "first": d2, "last": d2, "scheduled": 2,
                                                               "missing": [LIVE_DAY]}
    assert U.read_state()["missing_sessions"] == {"SPX": ["2026-10-01"]}


def test_completeness_maps_sessions_to_raw_files(sandbox):
    eur = C.asset("EURUSD").symbol
    _binance_zip("BTC", LIVE_DAY)
    http._write_manifest([
        _duka_row(SPX_SYM, LIVE_DAY, "BID", "ok", "2026-10-02T16:00Z"),
        _duka_row(SPX_SYM, LIVE_DAY, "ASK", "ok", "2026-10-02T16:00Z", write=False),  # recorded ok, file gone
        _duka_row(eur, FROZEN_END, "BID", "ok", "2026-10-02T16:00Z"),
        _duka_row(eur, FROZEN_END, "ASK", "missing", "2026-10-02T16:00Z"),
        _duka_row(eur, LIVE_DAY, "BID", "empty", "2026-10-02T16:00Z"),  # 10-01 ASK: never requested
    ])
    comp = U.completeness(("BTC", "SPX", "EURUSD"), LIVE_DAY)
    assert comp["BTC"] == {"first": LIVE_DAY, "end": LIVE_DAY, "scheduled": 1, "missing_sessions": [],
                           "raw_not_ok": []}
    assert comp["SPX"]["missing_sessions"] == [] and comp["SPX"]["raw_not_ok"] == [
        {"file": f"{LIVE_DAY}_ASK.bi5", "day": LIVE_DAY, "status": "absent", "sessions": [LIVE_DAY]}]
    # the FX session of 10-01 opens 09-30 21:00 UTC (17:00 New York): it needs the 09-30 files too
    assert comp["EURUSD"]["missing_sessions"] == [LIVE_DAY]  # no EURUSD gold in the sandbox
    assert [(r["file"], r["status"]) for r in comp["EURUSD"]["raw_not_ok"]] == [
        (f"{FROZEN_END}_ASK.bi5", "missing"), (f"{LIVE_DAY}_ASK.bi5", "not_requested"),
        (f"{LIVE_DAY}_BID.bi5", "empty")]
    empty = U.completeness(("SPX",), FROZEN_END)["SPX"]  # no live window yet
    assert empty["scheduled"] == 0 and empty["missing_sessions"] == [] and empty["raw_not_ok"] == []
