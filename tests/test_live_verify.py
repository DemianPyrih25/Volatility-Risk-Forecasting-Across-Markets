"""Tests for the verification report (docs/LIVE_SPEC.md §7): offline, synthetic data and stubs only.

Network access is blocked for the whole module; every source is replaced by a stub of ``verify._http_get``. Besides
the helpers, every ``check_*`` wrapper is exercised on both sides of its verdict (PASS and FAIL / SKIPPED), so a
check whose decision rule is weakened fails a test.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest
import requests
import yaml

from volrisk import config as C
from volrisk import holdout
from volrisk.models.base import align_targets, predict_mask_for, to_forecast_frame
from volrisk.models.har import HAR
from volrisk.models.simple import RW
from volrisk.pipeline import FORECAST_MODELS
from volrisk.targets import build_targets
from volrisk_live import paths, schema
from volrisk_live import verify as V


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("network access in an offline test")

    monkeypatch.setattr(requests, "get", refuse)
    monkeypatch.setattr(requests.Session, "request", refuse)
    monkeypatch.setattr(V, "_sleep", lambda s: None)


def _gold(asset: str, n: int, start: str = "2019-01-01", seed: int = 0) -> pd.DataFrame:
    """Gold-like rows of one asset (persistent log-vol, jumps, semivariances)."""
    rng = np.random.default_rng(seed)
    h = np.zeros(n)
    for i in range(1, n):
        h[i] = 0.97 * h[i - 1] + 0.25 * rng.standard_normal()
    rv = np.exp(h) * rng.gamma(4.0, 0.25, n)
    j = np.where(rng.random(n) < 0.15, rv * rng.uniform(0.1, 0.5, n), 0.0)
    share = rng.uniform(0.3, 0.7, n)
    dates = pd.date_range(start, periods=n, freq="D") if asset in C.CRYPTO else pd.bdate_range(start, periods=n)
    return pd.DataFrame({
        "asset": asset, "session_date": dates.astype("datetime64[ms]"), "rv": rv, "bv": rv - j, "c": rv - j, "j": j,
        "rs_pos": rv * share, "rs_neg": rv * (1 - share), "rq": rv**2 * rng.uniform(1.0, 3.0, n),
        "gap": np.zeros(n), "r_cc": rng.standard_normal(n) * np.sqrt(rv), "tv": rv,
    })


def _dev_loader(monkeypatch, n: int = 1700) -> None:
    """``io.load_daily`` -> synthetic BTC rows from 2019 that cover ``LEAK_T0`` (2023-06-30) and a month after."""
    data = _gold("BTC", n, start="2019-01-01", seed=4)
    monkeypatch.setattr(V.io, "load_daily", lambda asset=None, include_holdout=False: data.copy())


class _Constant:
    """Ignores its inputs: the perturbation cannot change it, so a leak test on it proves nothing (vacuous)."""

    name = "CONST"

    def forecast(self, daily, targets, asset, horizon):
        a = align_targets(daily, targets)
        F = np.where(predict_mask_for(a), a["n_t"].to_numpy(float), np.nan)
        return to_forecast_frame(asset, horizon, self.name, daily["session_date"], a["n_t"], F)


# ------------------------------------------------------------------------------------- 6 negative controls
def test_perturb_after_changes_only_later_rows():
    d = _gold("BTC", 100)
    t0 = d["session_date"].iloc[59].date()
    p = V.perturb_after(d, t0)
    assert p.iloc[:60].equals(d.iloc[:60])
    later = p.iloc[60:]
    assert (later["tv"] != d["tv"].iloc[60:]).all()
    assert (np.sign(later["r_cc"]) == np.sign(d["r_cc"].iloc[60:])).all()
    assert (later["session_date"] == d["session_date"].iloc[60:]).all()


@pytest.mark.parametrize("horizon", ["1d", "1w"])
def test_detect_leak_clean_models_and_cheaters(horizon):
    d = _gold("BTC", 700)
    t0 = d["session_date"].iloc[500].date()
    last = d["session_date"].iloc[-1].date()
    for model in (HAR(window=200), RW()):
        r = V.detect_leak(model, d, "BTC", horizon, t0, last_date=last)
        assert not r["leak"], r
        assert r["n_before"] > 100 and r["n_changed_after"] > 0  # the test is not vacuous
    for model in (V.OracleModel(), V.PeekModel()):
        r = V.detect_leak(model, d, "BTC", horizon, t0, last_date=last)
        assert r["leak"] and r["n_changed_before"] >= 1, r
        assert r["first_changed_origin"] <= t0.isoformat()


def test_detect_leak_flags_a_subtle_lookahead():
    """A model that is fine except for a one-session look-ahead in a trailing mean is caught."""

    class Lookahead:
        name = "LOOK"

        def forecast(self, daily, targets, asset, horizon):
            a = align_targets(daily, targets)
            lvl = daily["tv"].rolling(5, min_periods=5).mean().shift(-1).to_numpy()  # includes t+1
            F = np.where(predict_mask_for(a), a["n_t"].to_numpy(float) * lvl, np.nan)
            return to_forecast_frame(asset, horizon, self.name, daily["session_date"], a["n_t"], F)

    d = _gold("BTC", 300)
    r = V.detect_leak(Lookahead(), d, "BTC", "1d", d["session_date"].iloc[200].date(),
                      last_date=d["session_date"].iloc[-1].date())
    assert r["leak"] and r["n_changed_before"] == 1


@pytest.mark.parametrize("horizon", ["1d", "1m"])
def test_label_leak_is_caught_at_its_refit_origin_only(horizon):
    """Finding: a training-label leak in a model refitted every 90 rows escapes a single fixed cut-off; the cut-off
    at its refit origin (what ``leak_t0s`` adds) exposes it at exactly that origin."""
    d = _gold("BTC", 1400, seed=2)
    last = d["session_date"].iloc[-1].date()
    m = V.LabelLeakModel(refit_every=90, window=300)
    sched = m.refit_schedule(d, "BTC")["LABEL-LEAK"]
    r0 = sched[5].date()
    mid = r0 + timedelta(days=45)  # inside the block, after every leaked label window has ended
    t0s = V.leak_t0s(m, d, "BTC", t0=mid)
    assert t0s == [(r0, f"last LABEL-LEAK refit origin <= {mid}", True), (mid, "fixed cut-off", False)]
    at_refit, at_mid = V.detect_leak_multi(m, d, "BTC", horizon, [r0, mid], last_date=last)
    assert at_refit["leak"] and at_refit["first_changed_origin"] == r0.isoformat(), at_refit
    assert not at_mid["leak"] and at_mid["n_changed_after"] > 0, at_mid
    # the same model with a correct purge is clean at the refit origin
    clean = V.detect_leak(V.LabelLeakModel(refit_every=90, window=300, extra=0), d, "BTC", horizon, r0, last_date=last)
    assert not clean["leak"] and clean["n_changed_after"] > 0


def test_refit_schedule_matches_the_frozen_models():
    """The cut-offs of 6a sit on the refit origins the frozen GJR and LightGBM really use."""
    from volrisk.models.garch import GJR
    from volrisk.models.ml import LGBM

    d = _gold("BTC", 700, seed=6)
    g = GJR(window=300, refit_every=60)
    sched = V.refit_schedule(g, d, "BTC")["GJR"]
    assert list(pd.to_datetime(g.paths(d, "BTC").params["origin"])) == sched
    m = LGBM(window=300, refit_every=100, params={"n_estimators": 10})
    m.forecast(d, build_targets(d, "BTC", "1d", d["session_date"].iloc[-1].date()), "BTC", "1d")
    fits = list(pd.to_datetime(m.fit_origins))
    sched = V.refit_schedule(m, d, "BTC")["LGBM"]
    assert len(fits) >= 2 and fits == [s for s in sched if s >= fits[0]]
    assert V.refit_schedule(HAR(), d, "BTC") == {}  # refitted at every origin


def test_leak_t0s_of_combo_cut_at_every_member_refit_origin():
    from volrisk.models.garch import GJR
    from volrisk.models.ml import LGBM

    d = _gold("BTC", 900, seed=1)
    gjr, lgbm = GJR(window=300, refit_every=30), LGBM(window=300, refit_every=90)
    combo = V.ComboForecaster([gjr, HAR(window=200), lgbm])
    t0 = d["session_date"].iloc[700].date()
    got = V.leak_t0s(combo, d, "BTC", t0=t0)
    want = {t0}
    for m in (gjr, lgbm):
        want.add(max(s.date() for s in V.refit_schedule(m, d, "BTC")[m.name] if s.date() <= t0))
    assert {t for t, _, _ in got} == want and len(got) == len(want) >= 2
    assert any("GJR" in w for _, w, _ in got) and any("LGBM" in w for _, w, _ in got)


def test_leak_tasks_cover_every_frozen_model_and_combo():
    tasks = V.leak_tasks()
    assert {t[0] for t in tasks} == set(FORECAST_MODELS) | {"COMBO"}
    assert tasks[:2] == [("MLP", "BTC", "1d", (0,)), ("MLP", "BTC", "1d", (1,))]  # slowest first, split by cut-off
    for m in set(FORECAST_MODELS) - {"MLP"} | {"COMBO"}:
        assert sorted(t[1:3] for t in tasks if t[0] == m) == sorted(V.LEAK_CELLS)


def test_check_leak_frozen_verdicts(monkeypatch):
    _dev_loader(monkeypatch)
    models = {"HAR": lambda: HAR(window=200), "CONST": _Constant, "PEEK": V.PeekModel,
              "LABEL-LEAK": V.LabelLeakModel}
    monkeypatch.setattr(V, "_leak_model", lambda name: models[name]())
    status, ev, _ = V.check_leak_frozen(workers=1, tasks=[("HAR", "BTC", "1d", None)])
    assert status == "PASS" and ev["runs"][0]["t0"] == V.LEAK_T0.isoformat()
    status, ev, reason = V.check_leak_frozen(workers=1, tasks=[("CONST", "BTC", "1d", None)])
    assert status == "FAIL" and "vacuous" in reason  # nothing changed after t0 either: the test proves nothing
    assert V.check_leak_frozen(workers=1, tasks=[("PEEK", "BTC", "1d", None)])[0] == "FAIL"
    # a periodically refitted model with a label leak is run at its refit origin too, and fails there
    status, ev, _ = V.check_leak_frozen(workers=1, tasks=[("LABEL-LEAK", "BTC", "1d", None)])
    assert status == "FAIL" and len(ev["runs"]) == 2
    assert [r["leak"] for r in ev["runs"] if r["t0_is_refit_origin"]] == [True]
    assert V.check_leak_frozen(workers=1, tasks=[])[0] == "FAIL"  # no run at all is not a pass


def test_check_leak_cheaters_needs_the_refit_origin_cut(monkeypatch):
    _dev_loader(monkeypatch)
    monkeypatch.setattr(V, "LEAK_CELLS", (("BTC", "1d"), ("BTC", "1m")))
    status, ev, _ = V.check_leak_cheaters()
    assert status == "PASS", ev["summary"]
    label = [r for r in ev["runs"] if r["model"] == "LABEL-LEAK"]
    assert {r["t0_is_refit_origin"] for r in label} == {True, False}
    assert all(r["leak"] for r in label if r["t0_is_refit_origin"])
    # without the refit-origin cut-offs the label cheat is never tested where it can be seen -> FAIL
    real = V.leak_t0s
    monkeypatch.setattr(V, "leak_t0s", lambda *a, **k: [x for x in real(*a, **k) if not x[2]])
    assert V.check_leak_cheaters()[0] == "FAIL"


def test_noise_control_ranks_noise_last():
    d = _gold("BTC", 900, seed=3)
    last = d["session_date"].iloc[-1].date()
    tg = build_targets(d, "BTC", "1d", last)
    ok = tg["split"].eq("dev") & tg["ybar"].notna()
    rng = np.random.default_rng(1)
    rows = []
    for model, sd in (("HAR", 0.3), ("COMBO", 0.25), ("RW", 0.5)):
        g = tg[ok]
        rows.append(pd.DataFrame({"asset": "BTC", "horizon": "1d", "model": model, "origin": g["origin"],
                                  "n_t": g["n_t"], "F": g["n_t"] * g["ybar"] * np.exp(sd * rng.standard_normal(len(g))),
                                  "split": "dev"}))
    fc = pd.concat(rows, ignore_index=True)
    start = d["session_date"].iloc[400].date()
    res = V.noise_control(d, fc, tg, "BTC", "1d", start, last, reps=300)
    assert res["noise_last"] and not res["noise_in_90_mcs"]
    assert [r["model"] for r in res["ranking"]][-1] == "NOISE"
    assert res["T"] > 400


def test_check_noise_verdict(monkeypatch):
    base = {"noise_rank": 4, "ranking": [{}] * 4, "T": 300, "noise_mcs_pvalue": 0.0}
    empty = pd.DataFrame()

    def run(**over):
        monkeypatch.setattr(V, "noise_control", lambda *a, **k: {**base, "noise_last": True,
                                                                  "noise_in_90_mcs": False, **over})
        return V.check_noise(daily=empty, forecasts=empty, targets=empty)[0]

    assert run() == "PASS"
    assert run(noise_in_90_mcs=True) == "FAIL"
    assert run(noise_last=False) == "FAIL"


# ------------------------------------------------------------------------------------- 4 independent sources
def test_compare_returns_aligns_previous_observation():
    dates = pd.date_range("2024-01-01", periods=50, freq="D")
    lvl = 100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.02, 50)))
    ours = pd.DataFrame({"date": dates, "r": 100 * np.log(lvl / np.roll(lvl, 1)),
                         "prev": dates.to_series().shift(1).values})
    ours.loc[0, "r"] = np.nan
    ref = pd.DataFrame({"date": dates, "level": lvl})
    st = V.compare_returns(ours, ref)
    assert st["n"] == 49 and st["corr"] == pytest.approx(1.0) and st["max_abs_diff_bp"] < 1e-9
    # a day missing in the reference: its next return spans two days there, so that day is not compared
    st2 = V.compare_returns(ours, ref.drop(index=10))
    assert st2["n"] == 47
    lv = V.compare_levels(ref.assign(level=ref["level"] * 1.0001), ref)
    assert lv["corr"] == pytest.approx(1.0) and lv["level_median_abs_diff_bp"] == pytest.approx(1.0, rel=1e-3)


def _candles(levels: dict[date, float]):
    def fake(url, params=None, timeout=60.0, retries=3):
        s, e = date.fromisoformat(params["start"][:10]), date.fromisoformat(params["end"][:10])
        assert (e - s).days + 1 <= V.COINBASE_MAX
        days = [d for d in levels if s <= d <= e]
        return json.dumps([[int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp()), 1, 2, 1, levels[d], 5]
                           for d in reversed(days)]).encode()
    return fake


def test_fetch_coinbase_paginates_within_300_candles(monkeypatch):
    calls = []

    def fake(url, params=None, timeout=60.0, retries=3):
        assert "BTC-USD" in url and params["granularity"] == 86400
        s = date.fromisoformat(params["start"][:10])
        e = date.fromisoformat(params["end"][:10])
        calls.append((s, e))
        assert (e - s).days + 1 <= V.COINBASE_MAX
        days = [s + timedelta(days=k) for k in range((e - s).days + 1)]
        candles = [[int(datetime(x.year, x.month, x.day, tzinfo=UTC).timestamp()), 1, 2, 1, 100 + x.toordinal() % 7, 5]
                   for x in reversed(days)]  # newest first, like Coinbase
        return json.dumps(candles).encode()

    monkeypatch.setattr(V, "_http_get", fake)
    out = V.fetch_coinbase_closes("BTC-USD", date(2024, 1, 1), date(2025, 6, 30))
    assert len(calls) == 2 and calls[0][0] == date(2024, 1, 1) and calls[-1][1] == date(2025, 6, 30)
    assert len(out) == (date(2025, 6, 30) - date(2024, 1, 1)).days + 1
    assert out["date"].is_monotonic_increasing and out.attrs["requests"] == 2
    assert out.loc[0, "level"] == 100 + date(2024, 1, 1).toordinal() % 7


def test_coinbase_error_answer_is_a_source_failure(monkeypatch):
    monkeypatch.setattr(V, "_http_get", lambda *a, **k: b'{"message": "rate limited"}')
    with pytest.raises(requests.RequestException):
        V.fetch_coinbase_closes("ETH-USD", date(2024, 1, 1), date(2024, 1, 5))


def test_check_crypto_verdict_uses_the_correlation_threshold(monkeypatch):
    rng = np.random.default_rng(9)
    dates = pd.date_range(end="2026-09-30", periods=400, freq="D")
    lvl = 30000 * np.exp(np.cumsum(rng.normal(0, 0.03, len(dates))))
    daily = pd.DataFrame({"asset": "BTC", "session_date": dates, "r_cc": 100 * np.log(lvl / np.roll(lvl, 1))})
    daily.loc[0, "r_cc"] = np.nan
    same = {d.date(): float(x * (1 + rng.normal(0, 1e-4))) for d, x in zip(dates, lvl, strict=True)}
    monkeypatch.setattr(V, "_http_get", _candles(same))
    status, ev, _ = V.check_crypto("BTC", daily, "synthetic")
    assert status == "PASS" and ev["corr"] > 0.999 and ev["n"] >= V.MIN_DAYS
    other = 30000 * np.exp(np.cumsum(rng.normal(0, 0.03, len(dates))))  # an unrelated price path
    monkeypatch.setattr(V, "_http_get", _candles({d.date(): float(x) for d, x in zip(dates, other, strict=True)}))
    status, ev, reason = V.check_crypto("BTC", daily, "synthetic")
    assert status == "FAIL" and "correlation" in reason and ev["corr"] < V.CORR_MIN["BTC"]
    monkeypatch.setattr(V, "_http_get", _candles(dict(list(same.items())[:100])))  # too few matched days
    status, _, reason = V.check_crypto("BTC", daily, "synthetic")
    assert status == "FAIL" and "matched days" in reason


def test_parse_ecb_csv():
    csv = (b"KEY,FREQ,TIME_PERIOD,OBS_VALUE,OBS_STATUS\n"
           b"EXR.D.USD.EUR.SP00.A,D,2024-01-02,1.0956,A\n"
           b"EXR.D.USD.EUR.SP00.A,D,2024-01-03,,A\n"
           b"EXR.D.USD.EUR.SP00.A,D,2024-01-04,1.0953,A\n")
    out = V.parse_ecb_csv(csv)
    assert list(out["level"]) == [1.0956, 1.0953]
    assert out["date"].dtype == "datetime64[ns]"


def _bronze(tmp_path: Path, rows: list[tuple[datetime, float, bool]]) -> Path:
    root = tmp_path / "bronze"
    df = pl.DataFrame({"ts": [r[0] for r in rows], "close": [r[1] for r in rows], "is_real": [r[2] for r in rows]},
                      schema={"ts": pl.Datetime("us", "UTC"), "close": pl.Float64, "is_real": pl.Boolean})
    for (year,), g in df.group_by(pl.col("ts").dt.year()):
        p = root / "asset=EURUSD" / f"year={year}" / "part.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        g.sort("ts").write_parquet(p)
    return root


def test_dukascopy_fix_prices_use_frankfurt_time(tmp_path):
    winter, summer = date(2024, 1, 10), date(2024, 7, 10)  # 14:15 CET = 13:15 UTC; 14:15 CEST = 12:15 UTC
    rows = [
        (datetime(2024, 1, 10, 13, 14, tzinfo=UTC), 1.10, True),   # ends 13:15 UTC: the fix minute
        (datetime(2024, 1, 10, 13, 15, tzinfo=UTC), 9.99, True),   # after the fix
        (datetime(2024, 7, 10, 12, 10, tzinfo=UTC), 1.20, True),   # ends 12:11, last real before 12:15 UTC
        (datetime(2024, 7, 10, 12, 14, tzinfo=UTC), 7.77, False),  # flat fill: never a price
        (datetime(2024, 3, 5, 12, 50, tzinfo=UTC), 1.30, True),    # 24 min before the 13:15 fix: too stale
    ]
    out = V.dukascopy_fix_prices([winter, summer, date(2024, 3, 5)], _bronze(tmp_path, rows))
    got = dict(zip(out["date"].dt.date, out["level"], strict=True))
    assert got == {winter: 1.10, summer: 1.20}


def test_eurusd_check_uses_fixing_time(monkeypatch, tmp_path):
    days = pd.bdate_range("2023-01-02", periods=320)
    rng = np.random.default_rng(5)
    fix = 1.1 * np.exp(np.cumsum(rng.normal(0, 0.004, len(days))))
    rows = []
    for d, x in zip(days, fix, strict=True):
        t = V.fix_times_utc([d.date()])[0] - timedelta(minutes=1)
        rows.append((t, float(x) * (1 + rng.normal(0, 2e-5)), True))
    bronze = _bronze(tmp_path, rows)
    ecb = "\n".join(["TIME_PERIOD,OBS_VALUE"] + [f"{d.date()},{x:.4f}" for d, x in zip(days, fix, strict=True)])
    monkeypatch.setattr(V, "_http_get", lambda *a, **k: ecb.encode())
    monkeypatch.setattr(V.context, "require_opened_holdout", lambda: {})
    daily = pd.DataFrame({"asset": "EURUSD", "session_date": days, "r_cc": rng.normal(0, 0.4, len(days))})
    status, ev, reason = V.check_eurusd(daily, "synthetic", bronze_dir=bronze)
    assert status == "PASS", reason
    assert ev["at_fixing_time"]["corr"] > 0.99 and ev["at_fixing_time"]["n"] >= V.MIN_DAYS
    assert abs(ev["session_close_vs_fix"]["corr"]) < 0.5  # unrelated synthetic closes: reported, not judged


def test_network_disabled_skips():
    r = V.run_check("4a", "t", "m", lambda: V.check_crypto("BTC", pd.DataFrame(), "x", network=False))
    assert r["status"] == "SKIPPED" and "network" in r["reason"]


# ------------------------------------------------------------------------------------- 3 raw data
def _binance_zips(tmp_path: Path) -> tuple[Path, dict[str, bytes]]:
    root = tmp_path / "binance" / "BTCUSDT" / "monthly"
    root.mkdir(parents=True)
    remote = {}
    for k in range(3):
        z = root / f"BTCUSDT-1m-2020-0{k + 1}.zip"
        z.write_bytes(f"zip {k}".encode())
        line = f"{hashlib.sha256(z.read_bytes()).hexdigest()}  {z.name}\n"
        (root / (z.name + ".CHECKSUM")).write_text(line)
        remote[V.binance_zip_url(z) + ".CHECKSUM"] = line.encode()
    return root, remote


def test_binance_checksums_against_binance_and_local_copy(monkeypatch, tmp_path):
    root, remote = _binance_zips(tmp_path)
    assert V.binance_zip_url(root / "BTCUSDT-1m-2020-01.zip") == \
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2020-01.zip"
    monkeypatch.setattr(V, "_http_get", lambda url, **k: remote[url])
    status, ev, _ = V.check_binance_checksums(tmp_path / "binance")
    assert status == "PASS" and ev["n_match_binance"] == 3 and ev["n_checksums_downloaded"] == 3
    # a zip edited together with its local CHECKSUM passes the local comparison, not Binance's
    z = root / "BTCUSDT-1m-2020-02.zip"
    z.write_bytes(b"tampered")
    (root / (z.name + ".CHECKSUM")).write_text(f"{hashlib.sha256(b'tampered').hexdigest()}  {z.name}\n")
    status, ev, _ = V.check_binance_checksums(tmp_path / "binance")
    assert status == "FAIL" and ev["n_mismatch_binance"] == 1 and ev["n_mismatch_local_copy"] == 0
    assert ev["n_local_copy_differs_from_binance"] == 1
    # offline: the local comparison alone is never a PASS
    r = V.run_check("3a", "t", "m", lambda: V.check_binance_checksums(tmp_path / "binance", network=False))
    assert r["status"] == "SKIPPED" and "local CHECKSUM copies only" in r["reason"]
    # ... but a zip that does not even match its local CHECKSUM fails offline too
    z.write_bytes(b"tampered again")
    (root / "BTCUSDT-1m-2020-04.zip").write_bytes(b"no checksum")
    status, ev, _ = V.check_binance_checksums(tmp_path / "binance", network=False)
    assert status == "FAIL" and ev["n_mismatch_local_copy"] == 1 and ev["n_without_local_checksum"] == 1


def test_binance_checksum_download_errors_skip(monkeypatch, tmp_path):
    _, remote = _binance_zips(tmp_path)

    def flaky(url, **k):
        if url.endswith("2020-03.zip.CHECKSUM"):
            raise requests.ConnectionError("down")
        return remote[url]

    monkeypatch.setattr(V, "_http_get", flaky)
    r = V.run_check("3a", "t", "m", lambda: V.check_binance_checksums(tmp_path / "binance"))
    assert r["status"] == "SKIPPED" and r["evidence"]["n_download_errors"] == 1


def _manifest(tmp_path: Path, n_each: int = 6) -> tuple[pd.DataFrame, dict[str, bytes]]:
    rows, remote = [], {}
    for src, ext in (("binance", "zip"), ("dukascopy", "bi5")):
        for k in range(n_each):
            body = f"{src}-{k}".encode()
            p = tmp_path / "raw" / src / f"f{k}.{ext}"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(body)
            url = f"http://example.invalid/{src}/f{k}.{ext}"
            remote[url] = body
            rows.append({"source": src, "url": url, "local_path": str(p), "http_status": 200, "bytes": len(body),
                         "sha256": hashlib.sha256(body).hexdigest(), "status": "ok"})
    rows.append({"source": "binance", "url": "http://example.invalid/x.zip.CHECKSUM", "local_path": "",
                 "http_status": 200, "bytes": 5, "sha256": "", "status": "ok"})  # never sampled
    return pd.DataFrame(rows), remote


def test_resample_raw_identical_tampered_and_offline(monkeypatch, tmp_path):
    man, remote = _manifest(tmp_path)

    def serve(url, params=None, timeout=60.0, retries=3):
        return remote[url]

    monkeypatch.setattr(V, "_http_get", serve)
    status, ev, _ = V.resample_raw(sample=6, seed=1, manifest=man, seed_info={"source": "test"})
    assert status == "PASS" and ev["n_identical"] == 6 and ev["by_source"] == {"binance": 3, "dukascopy": 3}
    assert ev["seed"] == 1 and ev["seed_source"] == {"source": "test"} and ev["population"] == {"binance": 6,
                                                                                             "dukascopy": 6}
    assert all(not f["url"].endswith("CHECKSUM") for f in ev["files"])
    # same seed -> same sample; another seed -> another sample
    assert [f["url"] for f in V.resample_raw(sample=6, seed=1, manifest=man)[1]["files"]] == \
           [f["url"] for f in ev["files"]]
    assert [f["url"] for f in V.resample_raw(sample=6, seed=2, manifest=man)[1]["files"]] != \
           [f["url"] for f in ev["files"]]

    # a local copy that differs from what the source serves -> FAIL
    for f in ev["files"]:
        remote[f["url"]] = b"changed at the source"
    status, ev2, reason = V.resample_raw(sample=6, seed=1, manifest=man)
    assert status == "FAIL" and ev2["n_different"] == 6 and "differ" in reason

    # flaky network: failed files are replaced from the same seeded order
    _, remote = _manifest(tmp_path)
    bad = set(list(remote)[:2])

    def flaky(url, params=None, timeout=60.0, retries=3):
        if url in bad:
            raise requests.ConnectionError("down")
        return remote[url]

    monkeypatch.setattr(V, "_http_get", flaky)
    status, ev3, _ = V.resample_raw(sample=6, seed=1, manifest=man)
    assert status == "PASS" and ev3["n_compared"] == 6

    # no network at all -> SKIPPED, never PASS/FAIL
    monkeypatch.setattr(V, "_http_get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("offline")))
    r = V.run_check("3b", "t", "m", lambda: V.resample_raw(sample=6, seed=1, manifest=man))
    assert r["status"] == "SKIPPED" and "could be re-downloaded" in r["reason"]


def test_draw_seed_is_unpredictable_and_recorded(monkeypatch):
    tip = "00" * 28 + "0badf00d"
    monkeypatch.setattr(V, "_http_get", lambda url, **k: tip.encode())
    seed, info = V.draw_seed(True)
    assert seed == 0x0BADF00D and info["block_hash"] == tip and "Bitcoin" in info["source"]
    monkeypatch.setattr(V, "_http_get", lambda url, **k: b"<html>rate limited</html>")  # not a block hash
    assert "operating-system" in V.draw_seed(True)[1]["source"]

    def down(url, **k):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(V, "_http_get", down)
    seeds = {V.draw_seed(True)[0] for _ in range(5)}
    assert len(seeds) > 1 and "operating-system" in V.draw_seed(False)[1]["source"]


def test_raw_manifest_hashes_and_unrecorded_files(tmp_path):
    man, _ = _manifest(tmp_path)
    status, ev, _ = V.check_raw_manifest(man, raw_root=tmp_path / "raw", workers=2)
    assert status == "FAIL" and ev["n_missing"] == 1  # a row recorded as downloaded without a local file
    man = man[man["local_path"] != ""]
    status, ev, _ = V.check_raw_manifest(man, raw_root=tmp_path / "raw", workers=2)
    assert status == "PASS" and ev["n_unchanged"] == 12 and ev["n_unrecorded"] == 0
    (tmp_path / "raw" / "dukascopy" / "f2.bi5").write_bytes(b"edited after download")
    status, ev, _ = V.check_raw_manifest(man, raw_root=tmp_path / "raw", workers=2)
    assert status == "FAIL" and ev["changed"] == ["http://example.invalid/dukascopy/f2.bi5"]
    (tmp_path / "raw" / "dukascopy" / "f2.bi5").write_bytes(b"dukascopy-2")
    (tmp_path / "raw" / "dukascopy" / "extra.bi5").write_bytes(b"added by hand")
    status, ev, reason = V.check_raw_manifest(man, raw_root=tmp_path / "raw", workers=2)
    assert status == "FAIL" and ev["n_unrecorded"] == 1 and "unrecorded" in reason


# ------------------------------------------------------------------------------------- 1-2 seal (stubbed)
def test_seal_and_opening_checks(monkeypatch, tmp_path):
    hashes = {"code": "c" * 64, "config": "f" * 64}
    sealed = tmp_path / "SEALED.json"
    sealed.write_text(json.dumps({"created_utc": "2026-10-02T15:13:33+00:00", "hashes": hashes}), encoding="utf-8")
    entry = {"utc": "2026-10-02T15:20:42+00:00", "n_previous": 0, "seal_sha": holdout.file_sha(sealed),
             "hashes": dict(hashes), "rerun_reason": None}
    monkeypatch.setattr(holdout, "SEALED", sealed)
    monkeypatch.setattr(holdout, "verify_seal", lambda: [])
    monkeypatch.setattr(holdout, "code_sha", lambda src=None: "c" * 64)
    monkeypatch.setattr(holdout, "code_files", lambda src=None: {"a.py": "1"})
    monkeypatch.setattr(holdout, "RESULTS_HOLDOUT", tmp_path / "none")
    monkeypatch.setattr(holdout, "openings", lambda: [entry])
    assert V.check_seal()[0] == "PASS"
    assert V.check_opened_once()[0] == "PASS"

    monkeypatch.setattr(holdout, "code_sha", lambda src=None: "d" * 64)  # code changed after the opening
    assert V.check_seal()[0] == "FAIL"
    monkeypatch.setattr(holdout, "openings", lambda: [entry, {**entry, "n_previous": 1}])  # opened twice
    assert V.check_opened_once()[0] == "FAIL"
    monkeypatch.setattr(holdout, "openings", lambda: [{**entry, "hashes": {**hashes, "config": "0" * 64}}])
    assert V.check_opened_once()[0] == "FAIL"


# ------------------------------------------------------------------------------------- 5 reproducibility
def test_compare_tables(tmp_path):
    a = pd.DataFrame({"model": ["HAR", "COMBO"], "qlike": [0.3, np.nan]})
    a.to_parquet(tmp_path / "eval_leaderboard.parquet", index=False)
    rows = V.compare_tables({"leaderboard": a.copy(), "iv_mcs": pd.DataFrame()}, tmp_path)
    assert all(r["equal"] for r in rows)
    b = a.assign(qlike=[0.3 + 1e-15, np.nan])
    rows = V.compare_tables({"leaderboard": b, "mz": a}, tmp_path)
    assert {r["table"]: r["equal"] for r in rows} == {"leaderboard": False, "mz": False}
    assert V.compare_tables({"leaderboard": b}, tmp_path, rtol=1e-9)[0]["equal"]  # re-run tables: rtol
    assert not V.compare_tables({"leaderboard": a.assign(qlike=[0.31, np.nan])}, tmp_path, rtol=1e-9)[0]["equal"]


def test_compare_keyed_checks_both_directions():
    o = pd.to_datetime(["2025-10-01", "2025-10-02", "2025-10-03"])
    a = pd.DataFrame({"asset": "BTC", "horizon": "1d", "model": "HAR", "origin": o, "n_t": [1, 1, 1],
                      "F": [1.0, 2.0, 3.0], "split": "holdout"})
    keys = ["asset", "horizon", "model", "origin"]
    assert V.compare_keyed(a, a.assign(F=a["F"] * (1 + 1e-12)), keys, ["F", "n_t", "split"], exact=("n_t",))["equal"]
    res = V.compare_keyed(a.iloc[:2], a, keys, ["F"])  # a row deleted from the stored file
    assert not res["equal"] and res["only_rerun"] == 1
    res = V.compare_keyed(a, a.iloc[1:], keys, ["F"])
    assert not res["equal"] and res["only_stored"] == 1
    res = V.compare_keyed(a, a.assign(F=[1.0, 2.0, 3.3]), keys, ["F"])
    assert not res["equal"] and res["differ"] == {"F": 1} and res["max_rel_diff"] == pytest.approx(0.1)
    res = V.compare_keyed(a, a.assign(split=["holdout", "dev", "holdout"]), keys, ["split"], exact=("split",))
    assert not res["equal"] and res["differ"] == {"split": 1}


def test_live_checks_skip_without_live_data_and_fail_on_mismatch(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "LIVE_DAILY_DEV", tmp_path / "missing.parquet")
    r = V.run_check("5a", "t", "m", V.check_live_gold)
    assert r["status"] == "SKIPPED"
    monkeypatch.setattr(paths, "LIVE_RESULTS", tmp_path)
    assert V.run_check("5b", "t", "m", V.check_live_forecasts)["status"] == "SKIPPED"

    (tmp_path / "dev.parquet").write_bytes(b"x")
    monkeypatch.setattr(paths, "LIVE_DAILY_DEV", tmp_path / "dev.parquet")
    monkeypatch.setattr(paths, "LIVE_DAILY_HOLDOUT", tmp_path / "dev.parquet")
    monkeypatch.setattr(paths, "LIVE_STATE", tmp_path / "state.json")
    (tmp_path / "state.json").write_text(json.dumps({"started_utc": "2026-10-02T16:14:47+00:00", "end": "2026-10-01"}))

    class LiveDataMismatch(RuntimeError):
        pass

    def mismatch():
        raise LiveDataMismatch("live rebuild differs from the sealed data")

    monkeypatch.setattr(V, "_live_module", lambda name: SimpleNamespace(check_against_sealed=mismatch))
    r = V.run_check("5a", "t", "m", V.check_live_gold)
    assert r["status"] == "FAIL" and "LiveDataMismatch" in r["reason"]
    ok = {"ok": True, "dev": {"rows_compared": 3}, "holdout": {"rows_compared": 2}}
    monkeypatch.setattr(V, "_live_module", lambda name: SimpleNamespace(check_against_sealed=lambda: ok))
    r = V.run_check("5a", "t", "m", V.check_live_gold)
    assert r["status"] == "PASS" and "3 rows" in r["evidence"]["summary"]
    ev = r["evidence"]  # what was compared, and when it was built
    assert ev["built_utc"] == "2026-10-02T16:14:47+00:00" and "2026-10-02T16:14:47" in ev["summary"]
    assert ev["live_files"][0]["sha256"] == hashlib.sha256(b"x").hexdigest()
    monkeypatch.setattr(V, "_live_module", lambda name: SimpleNamespace(check_against_sealed=lambda: {**ok,
                                                                                                    "ok": False}))
    assert V.run_check("5a", "t", "m", V.check_live_gold)["status"] == "FAIL"


def test_live_forecasts_verdicts(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "LIVE_RESULTS", tmp_path)
    pd.DataFrame({"asset": ["BTC"], "horizon": ["1d"], "model": ["HAR"], "origin": [pd.Timestamp("2026-10-01")],
                  "n_t": [1], "F": [2.0]}).to_parquet(tmp_path / "forecasts.parquet")

    class ReproductionError(RuntimeError):
        pass

    def boom(fc):
        raise ReproductionError("1 of 7 rows differ")

    monkeypatch.setattr(V, "_live_module", lambda name: SimpleNamespace(check_reproduces_holdout=lambda fc: 7))
    r = V.run_check("5b", "t", "m", V.check_live_forecasts)
    assert r["status"] == "PASS" and r["evidence"]["n_sealed_forecasts_reproduced"] == 7
    assert r["evidence"]["file"]["sha256"] == V._sha256_file(tmp_path / "forecasts.parquet")
    monkeypatch.setattr(V, "_live_module", lambda name: SimpleNamespace(check_reproduces_holdout=boom))
    r = V.run_check("5b", "t", "m", V.check_live_forecasts)
    assert r["status"] == "FAIL" and "ReproductionError" in r["reason"]
    monkeypatch.setattr(V, "_live_module", lambda name: SimpleNamespace(check_reproduces_holdout=lambda fc: 0))
    assert V.run_check("5b", "t", "m", V.check_live_forecasts)["status"] == "SKIPPED"


def _pin_ledger(tmp_path: Path, files: dict[str, str], run_id: str = "20261003T070000Z") -> None:
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    payload = {"run_id": run_id, "checks": {"reproduction": {"files": {k: {"sha256": v} for k, v in files.items()}}}}
    (runs / f"{run_id}.json").write_text(schema.canonical_json(payload), encoding="utf-8")
    (tmp_path / "ledger.jsonl").write_text(json.dumps({"run_id": run_id}) + "\n", encoding="utf-8")


def test_eval_tables_verdicts_and_ledger_pins(monkeypatch, tmp_path):
    import volrisk.evaluation.suite as suite

    lb = pd.DataFrame({"asset": ["BTC"], "horizon": ["1d"], "model": ["COMBO"], "qlike_ratio": [0.828], "n": [365]})
    mcs = pd.DataFrame({"model": ["HAR", "COMBO"], "pvalue": [0.2, 1.0]})
    monkeypatch.setattr(suite, "evaluate", lambda fc, tg, mode="dev": {"leaderboard": lb.copy(), "mcs": mcs.copy()})
    folder = tmp_path / "holdout"
    folder.mkdir()
    pd.DataFrame({"F": [1.0]}).to_parquet(folder / "forecasts.parquet")
    pd.DataFrame({"y": [1.0]}).to_parquet(folder / "targets.parquet")
    lb.to_parquet(folder / "eval_leaderboard.parquet", index=False)
    mcs.to_parquet(folder / "eval_mcs.parquet", index=False)
    monkeypatch.setattr(holdout, "RESULTS_HOLDOUT", folder)
    monkeypatch.setattr(V.context, "require_opened_holdout", lambda: {})
    monkeypatch.setattr(paths, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    status, ev, _ = V.check_eval_tables("holdout")
    assert status == "PASS" and "not pinned yet" in ev["summary"] and "internal consistency" in ev["note"]
    # a stored table that does not follow from the stored forecasts
    mcs.assign(pvalue=[0.25, 1.0]).to_parquet(folder / "eval_mcs.parquet", index=False)
    assert V.check_eval_tables("holdout")[0] == "FAIL"
    mcs.to_parquet(folder / "eval_mcs.parquet", index=False)
    # pinned by the first ledger payload: unchanged -> PASS, changed -> FAIL
    name = V._rel(folder / "forecasts.parquet")
    _pin_ledger(tmp_path, {name: V._sha256_file(folder / "forecasts.parquet")})
    status, ev, _ = V.check_eval_tables("holdout")
    assert status == "PASS" and "unchanged since ledger run 20261003T070000Z" in ev["summary"]
    pd.DataFrame({"F": [0.8]}).to_parquet(folder / "forecasts.parquet")  # "improved" after it was pinned
    status, ev, reason = V.check_eval_tables("holdout")
    assert status == "FAIL" and "changed since" in reason


def test_full_holdout_rerun_compares_every_row(monkeypatch, tmp_path):
    from volrisk import pipeline
    from volrisk.evaluation import suite
    from volrisk.risk import suite as risk_suite

    o = pd.to_datetime(["2025-10-01", "2025-10-02"])
    fc = pd.DataFrame({"asset": "BTC", "horizon": "1d", "model": "COMBO", "origin": o, "n_t": [1, 1],
                       "F": [1.0, 2.0], "split": "holdout"})
    tg = pd.DataFrame({"asset": "BTC", "horizon": "1d", "origin": o, "n_t": [1, 1], "y": [1.1, 1.9],
                       "split": "holdout"})
    risk = pd.DataFrame({"asset": "BTC", "model": "COMBO+FHS", "date": o, "var99": [3.0, 4.0]})
    zones = pd.DataFrame({"asset": ["BTC"], "zone": ["green"]})
    folder = tmp_path / "holdout"
    folder.mkdir()
    for name, df in (("forecasts", fc), ("targets", tg), ("risk", risk), ("risk_rolling_zones", zones),
                     ("risk_time_in_zone", zones)):
        df.to_parquet(folder / f"{name}.parquet", index=False)
    monkeypatch.setattr(holdout, "RESULTS_HOLDOUT", folder)
    monkeypatch.setattr(holdout, "frozen", lambda: {"har_star": {"BTC": "HAR"}})
    monkeypatch.setattr(V.context, "require_opened_holdout", lambda: {})
    monkeypatch.setattr(V.io, "load_daily", lambda asset=None, include_holdout=False: pd.DataFrame())
    monkeypatch.setattr(pipeline, "assert_reproduces_dev", lambda f: 5)
    monkeypatch.setattr(suite, "evaluate", lambda f, t, mode="dev": {})
    monkeypatch.setattr(risk_suite, "build_risk", lambda d, f, s: risk.copy())
    monkeypatch.setattr(risk_suite, "evaluate_risk", lambda r, d, f, mode="dev": {"rolling_zones": zones.copy()})
    monkeypatch.setattr(risk_suite, "time_in_green", lambda z, mode: zones.copy())
    monkeypatch.setattr(pipeline, "compute_forecasts", lambda include_holdout, workers: (fc.copy(), tg.copy()))
    status, ev, _ = V.check_full_holdout(workers=1)
    assert status == "PASS" and ev["forecasts"]["equal"] and ev["n_dev_rows_equal_sealed"] == 5
    # the stored file lacks a forecast the frozen code produces (a bad day removed after the run) -> FAIL
    fc.iloc[:1].to_parquet(folder / "forecasts.parquet", index=False)
    status, ev, _ = V.check_full_holdout(workers=1)
    assert status == "FAIL" and ev["forecasts"]["only_rerun"] == 1
    fc.to_parquet(folder / "forecasts.parquet", index=False)
    monkeypatch.setattr(risk_suite, "build_risk", lambda d, f, s: risk.assign(var99=[3.0, 4.1]))
    assert V.check_full_holdout(workers=1)[0] == "FAIL"


def test_rebuild_gold_from_raw_compares_with_the_sealed_tables(monkeypatch, tmp_path):
    from volrisk import bars, measures
    from volrisk.data import binance, dukascopy, implied

    tables = {"dev": pd.DataFrame({"tv": [1.0, 2.0]}), "holdout": pd.DataFrame({"tv": [3.0]}),
              "implied": pd.DataFrame({"iv": [50.0]})}
    sealed = {}
    for k, df in tables.items():
        df.to_parquet(tmp_path / f"{k}.parquet", index=False)
        sealed[k] = tmp_path / f"{k}.parquet"
    monkeypatch.setattr(V.io, "DAILY_DEV", sealed["dev"])
    monkeypatch.setattr(V.io, "DAILY_HOLDOUT", sealed["holdout"])
    monkeypatch.setattr(V.io, "IMPLIED", sealed["implied"])
    monkeypatch.setattr(V, "_sealed_hashes", lambda: {"dev_data": V._sha256_file(sealed["dev"]),
                                                      "holdout_data": V._sha256_file(sealed["holdout"]),
                                                      "implied": V._sha256_file(sealed["implied"])})
    monkeypatch.setattr(V.context, "require_opened_holdout", lambda: {})
    calls = []
    monkeypatch.setattr(binance, "build_bronze", lambda a, **k: calls.append(("bronze", a)))
    monkeypatch.setattr(dukascopy, "build_bronze", lambda a, **k: calls.append(("bronze", a)))
    monkeypatch.setattr(bars, "build_bars", lambda a, **k: calls.append(("bars", a)))
    gold = {"dev": tables["dev"]}

    def build_gold(assets, silver_dir, gold_dir, holdout_dir):
        for d, df in ((gold_dir, gold["dev"]), (holdout_dir, tables["holdout"])):
            Path(d).mkdir(parents=True, exist_ok=True)
            df.to_parquet(Path(d) / "daily.parquet", index=False)

    monkeypatch.setattr(measures, "build_gold", build_gold)
    monkeypatch.setattr(implied, "build_implied", lambda out_path: tables["implied"].to_parquet(out_path, index=False))
    work = tmp_path / "work"
    work.mkdir()
    status, ev, _ = V.check_rebuild_gold(tmp_root=work)
    assert status == "PASS" and all(r["same_bytes"] for r in ev["tables"])
    assert sorted(calls) == sorted([(s, a) for a in C.ASSETS for s in ("bronze", "bars")])
    assert list(work.iterdir()) == []  # the temporary build is deleted
    gold["dev"] = pd.DataFrame({"tv": [1.0, 2.5]})
    status, ev, _ = V.check_rebuild_gold(tmp_root=work)
    assert status == "FAIL" and [r["equal"] for r in ev["tables"]] == [False, True, True]


# ------------------------------------------------------------------------------------- 7 no tuning
@pytest.mark.skipif(not V.SPEC_PATH.exists(), reason="docs/SPEC.md is not in this checkout")
def test_spec_values_equal_frozen_config():
    spec = V.spec_hyperparameters(V.SPEC_PATH.read_text(encoding="utf-8"))
    assert spec["lgbm.num_leaves"] == 15 and spec["mlp.hidden"] == [32, 32]
    assert spec["walk_forward.refit.ml.crypto"] == 90 and spec["har_lags.crypto"] == [1, 7, 30]
    status, ev, _ = V.check_no_tuning()
    assert status == "PASS", [r for r in ev["parameters"] if not r["equal"]]
    assert len(ev["parameters"]) == 23


@pytest.mark.skipif(not V.SPEC_PATH.exists(), reason="docs/SPEC.md is not in this checkout")
def test_no_tuning_detects_a_changed_hyperparameter(tmp_path):
    frozen = yaml.safe_load(C.FROZEN_PATH.read_text(encoding="utf-8"))
    frozen["hyperparameters"]["lgbm"]["num_leaves"] = 7
    p = tmp_path / "frozen.yaml"
    p.write_text(yaml.safe_dump(frozen), encoding="utf-8")
    status, ev, _ = V.check_no_tuning(frozen_path=p, sensitivity_path=tmp_path / "none.csv")
    assert status == "FAIL"
    assert [r["parameter"] for r in ev["parameters"] if not r["equal"]] == ["lgbm.num_leaves"]


# ------------------------------------------------------------------------------------- 8 ledger, timestamps
def _ledger(tmp_path: Path, days: tuple[int, ...] = (3, 4, 5)) -> tuple[Path, Path]:
    """A chain written by the §4 rule: payloads, entry files (bytes hashed by entry_sha256) and ledger.jsonl."""
    runs = tmp_path / "runs"
    runs.mkdir(exist_ok=True)
    prev, lines = schema.ZERO_HASH, []
    for i, day in enumerate(days):
        run_id = f"202610{day:02d}T003000Z"
        text = schema.canonical_json({"run_id": run_id, "forecasts": [{"F": 1.5 + i}]})
        (runs / f"{run_id}.json").write_bytes(text.encode("utf-8"))
        e = {"seq": i, "run_id": run_id, "run_utc": f"2026-10-{day:02d}T00:30:00+00:00",
             "payload": f"runs/{run_id}.json", "payload_sha256": schema.sha256_text(text), "prev_entry_sha256": prev}
        (runs / f"{run_id}.entry").write_bytes(schema.canonical_json(e).encode("utf-8"))
        e["entry_sha256"] = schema.sha256_text(schema.canonical_json(e))
        prev = e["entry_sha256"]
        lines.append(schema.canonical_json(e))
    led = tmp_path / "ledger.jsonl"
    led.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return led, runs


def test_ledger_chain_detects_tampering(tmp_path):
    led, runs = _ledger(tmp_path)
    res = V.verify_ledger_file(led, runs)
    assert res["ok"] and res["n_entry_files_ok"] == 3 and res["days_without_run"] == []
    # edited payload
    p = sorted(runs.glob("*.json"))[1]
    original = p.read_bytes()
    p.write_bytes(original.replace(b"2.5", b"2.4"))
    res = V.verify_ledger_file(led, runs)
    assert not res["ok"] and "entry 1" in res["first_broken"]
    p.write_bytes(original)
    # reordered entries
    lines = led.read_text(encoding="utf-8").splitlines()
    led.write_text("\n".join([lines[0], lines[2], lines[1]]) + "\n", encoding="utf-8")
    assert not V.verify_ledger_file(led, runs)["ok"]
    # removed last entry -> its payload and entry file become orphans
    led.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    res = V.verify_ledger_file(led, runs)
    assert not res["ok"] and "without a ledger entry" in res["first_broken"]
    assert res["orphans"] == ["20261005T003000Z.entry", "20261005T003000Z.json"]
    # edited entry field
    e = json.loads(lines[0])
    e["run_utc"] = "2026-10-03T00:31:00+00:00"
    led.write_text("\n".join([schema.canonical_json(e), *lines[1:]]) + "\n", encoding="utf-8")
    assert "entry_sha256" in V.verify_ledger_file(led, runs)["first_broken"]
    # an entry file rewritten next to an intact ledger line
    led.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (runs / "20261004T003000Z.entry").write_bytes(b"{}")
    assert "entry file" in V.verify_ledger_file(led, runs)["first_broken"]


def _ledger_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(paths, "LEDGER", tmp_path / "ledger.jsonl")
    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "SCORES", tmp_path / "scores.csv")
    monkeypatch.setattr(paths, "RISK_SCORES", tmp_path / "risk_scores.csv")


def test_check_ledger_verdicts(monkeypatch, tmp_path):
    _ledger_paths(monkeypatch, tmp_path)
    chain = {"ok": True, "reason": None, "days_without_run": []}
    monkeypatch.setitem(sys.modules, "volrisk_live.ledger", SimpleNamespace(verify_chain=lambda: chain))
    assert V.run_check("8a", "t", "m", V.check_ledger)["status"] == "SKIPPED"  # nothing recorded yet
    _ledger(tmp_path)
    status, ev, reason = V.check_ledger()
    assert status == "PASS" and reason is None and "UTC days without a run: none" in ev["summary"]
    # the ledger module disagrees -> FAIL even though the independent check passes
    monkeypatch.setitem(sys.modules, "volrisk_live.ledger",
                        SimpleNamespace(verify_chain=lambda: {"ok": False, "reason": "entry 1: proof invalid"}))
    status, _, reason = V.check_ledger()
    assert status == "FAIL" and "proof invalid" in reason
    monkeypatch.setitem(sys.modules, "volrisk_live.ledger", SimpleNamespace(verify_chain=lambda: chain))
    # scores of a run the ledger does not record
    pd.DataFrame({"run_id": ["20261003T003000Z", "20260930T070000Z"]}).to_csv(tmp_path / "scores.csv", index=False)
    status, ev, _ = V.check_ledger()
    assert status == "FAIL" and ev["scored_runs_not_in_ledger"] == {V._rel(tmp_path / "scores.csv"):
                                                                    ["20260930T070000Z"]}
    # ledger deleted while payloads and scores remain -> FAIL, never SKIPPED
    (tmp_path / "ledger.jsonl").unlink()
    status, _, reason = V.check_ledger()
    assert status == "FAIL" and "ledger.jsonl is missing" in reason
    (tmp_path / "scores.csv").unlink()
    assert V.check_ledger()[0] == "FAIL"  # run files alone


def test_check_ledger_flags_days_without_a_run(monkeypatch, tmp_path):
    _ledger_paths(monkeypatch, tmp_path)
    monkeypatch.setitem(sys.modules, "volrisk_live.ledger",
                        SimpleNamespace(verify_chain=lambda: {"ok": True, "days_without_run": ["2026-10-04"]}))
    _ledger(tmp_path, days=(3, 5))  # a run of 2026-10-04 dropped and the chain rebuilt consistently
    r = V.run_check("8a", "t", "m", V.check_ledger)
    assert r["status"] == "PASS" and "2026-10-04" in r["reason"] and "2026-10-04" in r["evidence"]["summary"]
    assert r["evidence"]["days_without_run"] == ["2026-10-04"]


def test_check_ledger_with_the_ledger_module(monkeypatch, tmp_path):
    """The reviewer's rewrite scenario on the real ledger module (sandbox, no stamps)."""
    ledger = pytest.importorskip("volrisk_live.ledger")
    _ledger_paths(monkeypatch, tmp_path)

    def payload(day: int) -> dict:
        rid = f"202610{day:02d}T003000Z"
        return {"schema": schema.SCHEMA, "run_id": rid, "run_utc": f"2026-10-{day:02d}T00:30:00+00:00",
                "frozen": {"code_sha": "x", "seal_ok": True, "sealed_utc": "s", "holdout_opened_utc": "o"},
                "live_code_sha": "y", "data": {}, "forecasts": [], "risk": [], "implied": [], "checks": {}}

    try:
        for d in (3, 5):
            ledger.append(payload(d), stamp=False, check_clock=False)
    except TypeError as e:  # another ledger API: covered by the stub tests above
        pytest.skip(f"ledger.append signature differs: {e}")
    r = V.run_check("8a", "t", "m", V.check_ledger)
    assert r["status"] == "PASS" and "2026-10-04" in r["reason"]
    paths.LEDGER.unlink()  # ledger deleted, payloads left
    assert V.run_check("8a", "t", "m", V.check_ledger)["status"] == "FAIL"


def test_check_anchors_verdicts(monkeypatch, tmp_path):
    _ledger_paths(monkeypatch, tmp_path)
    _ledger(tmp_path)

    def anchors(*rows):
        counts: dict = {}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        res = {"ok": True, "chain_ok": True, "counts": counts, "max_lag_hours": 24.0, "entries": list(rows),
               "note": "n"}
        monkeypatch.setattr(V, "_live_module", lambda name: SimpleNamespace(verify_anchors=lambda: res))

    ok = {"seq": 0, "run_id": "20261003T003000Z", "status": "anchored", "pinned_utc": "2026-10-03T03:00:00Z",
          "proof": "verified", "reason": None}
    anchors(ok, {**ok, "seq": 1, "status": "pending", "pinned_utc": None, "proof": "pending"})
    status, _, reason = V.check_anchors()
    assert status == "PASS" and "1 recent" in reason
    assert V.run_check("8c", "t", "m", lambda: V.check_anchors(network=False))["status"] == "SKIPPED"
    late = {**ok, "seq": 1, "status": "late", "pinned_utc": "2026-10-05T03:00:00Z", "reason": "pinned 50 h late"}
    anchors(ok, late)  # an entry stamped long after its run: the chain may have been rewritten from there
    assert V.check_anchors()[0] == "FAIL"
    anchors(ok, {**late, "pinned_utc": None, "proof": "unstamped"})  # no proof at all after 24 h
    assert V.check_anchors()[0] == "FAIL"
    anchors(ok, {**late, "status": "failed", "proof": "failed"})
    assert V.check_anchors()[0] == "FAIL"
    anchors(ok, {**late, "pinned_utc": None, "proof": "pending"})  # never upgraded: cannot tell yet
    r = V.run_check("8c", "t", "m", V.check_anchors)
    assert r["status"] == "SKIPPED" and "stamp" in r["reason"]
    anchors(ok, {**late, "pinned_utc": None, "proof": "unverified"})  # no explorer answered
    assert V.run_check("8c", "t", "m", V.check_anchors)["status"] == "SKIPPED"


def _stamp(path: Path, heights: tuple[int, ...] = ()) -> None:
    from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
    from opentimestamps.core.op import OpSHA256
    from opentimestamps.core.serialize import StreamSerializationContext
    from opentimestamps.core.timestamp import DetachedTimestampFile, Timestamp

    dtf = DetachedTimestampFile(OpSHA256(), Timestamp(hashlib.sha256(path.read_bytes()).digest()))
    dtf.timestamp.attestations.add(PendingAttestation("https://a.pool.opentimestamps.org"))
    for h in heights:
        dtf.timestamp.attestations.add(BitcoinBlockHeaderAttestation(h))
    with open(str(path) + ".ots", "wb") as f:
        dtf.serialize(StreamSerializationContext(f))


NOW = datetime(2026, 10, 2, 22, 0, tzinfo=UTC)


def test_timestamps_offline_and_stubbed_bitcoin(monkeypatch, tmp_path):
    from volrisk_live import ots

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(paths, "RUNS", runs)
    payload = {"forecasts": [{"asset": "BTC", "window_first": "2026-10-03"}],
               "risk": [{"asset": "SPX", "date": "2026-10-02"}]}
    p = runs / "20261002T003000Z.json"
    p.write_text(schema.canonical_json(payload), encoding="utf-8")
    assert V.first_outcome_utc(payload) == datetime(2026, 10, 2, 20, 0, tzinfo=UTC)  # SPX close 16:00 New York

    assert V.run_check("8b", "t", "m", lambda: V.check_timestamps(network=False, now_utc=NOW))["status"] == "FAIL"
    _stamp(p)
    status, ev, reason = V.check_timestamps(network=False, now_utc=NOW)
    assert status == "PASS" and ev["n_pending"] == 1 and "wait for their Bitcoin attestation" in reason
    # a proof still pending a day after its run has not been upgraded: nothing proves its time yet
    r = V.run_check("8b", "t", "m", lambda: V.check_timestamps(network=False, now_utc=NOW + timedelta(days=1)))
    assert r["status"] == "SKIPPED" and "stamp" in r["reason"]

    _stamp(p, heights=(900000,))
    monkeypatch.setattr(ots, "status_all", lambda: {"counts": {"complete": 1}})
    monkeypatch.setattr(ots, "verify", lambda path: {"status": "verified", "attested_utc": "2026-10-02T03:10:00Z"})
    status, ev, _ = V.check_timestamps(network=True, now_utc=NOW)
    assert status == "PASS" and ev["n_bitcoin_verified"] == 1
    rec = ev["payloads"][0]
    assert rec["proven_by_utc"] == "2026-10-02T05:10:00Z" and rec["before_first_outcome"]
    assert rec["hours_before_first_outcome"] == pytest.approx(14.83, abs=0.01)  # 20:00 - (03:10 + 2 h)
    monkeypatch.setattr(ots, "verify", lambda path: {"status": "verified", "attested_utc": "2026-10-02T21:00:00Z"})
    assert V.check_timestamps(network=True, now_utc=NOW)[0] == "FAIL"  # attested after the outcome was known

    p.write_text(schema.canonical_json({**payload, "edited": True}), encoding="utf-8")
    status, ev, reason = V.check_timestamps(network=False, now_utc=NOW)
    assert status == "FAIL" and "edited after stamping" in reason


def _session_payload(runs: Path) -> Path:
    """A run at 16:32 UTC on 2026-10-02 for the sessions of that day (BTC, EUR/USD and S&P 500 already trading)."""
    rows = [{"asset": a, "horizon": "1d", "window_first": "2026-10-02", "window_last": "2026-10-02"}
            for a in ("BTC", "EURUSD", "SPX")]
    p = runs / "20261002T163200Z.json"
    p.write_text(schema.canonical_json({"run_utc": "2026-10-02T16:32:00+00:00", "forecasts": rows, "risk": []}),
                 encoding="utf-8")
    _stamp(p, heights=(900000,))
    return p


def test_timestamps_bound_has_block_time_slack_and_reports_elapsed_share(monkeypatch, tmp_path):
    """Findings: the attested block time gets the 2-hour block-time allowance, and each target window shows how
    much of it had passed at that proven time."""
    from volrisk_live import ots

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(paths, "RUNS", runs)
    _session_payload(runs)
    monkeypatch.setattr(ots, "status_all", lambda: {"counts": {}})
    # block at 19:30 UTC: + 2 h = 21:30, after the S&P 500 close (20:00) -> not proven before its outcome
    monkeypatch.setattr(ots, "verify", lambda path: {"status": "verified", "attested_utc": "2026-10-02T19:30:00Z"})
    status, ev, reason = V.check_timestamps(network=True, now_utc=NOW)
    assert status == "FAIL" and "is not before the first outcome 2026-10-02T20:00:00Z" in reason
    # block at 17:00 UTC: proven by 19:00, before the close; the share of each session already gone is shown
    monkeypatch.setattr(ots, "verify", lambda path: {"status": "verified", "attested_utc": "2026-10-02T17:00:00Z"})
    status, ev, _ = V.check_timestamps(network=True, now_utc=NOW)
    assert status == "PASS"
    w = {r["asset"]: r for r in ev["payloads"][0]["windows"]}
    assert w["BTC"]["elapsed_at_proof"] == pytest.approx(19 / 24, abs=1e-4)
    assert w["EURUSD"]["open_utc"] == "2026-10-01T21:00:00Z" and w["EURUSD"]["elapsed_at_proof"] == \
        pytest.approx(22 / 24, abs=1e-4)
    assert w["SPX"]["elapsed_at_proof"] == pytest.approx(5.5 / 6.5, abs=1e-4)
    assert not any(r["before_open"] for r in w.values()) and all(r["before_first_close"] for r in w.values())
    assert ev["max_elapsed_next_session"] == pytest.approx(22 / 24, abs=1e-4) and "92%" in ev["summary"]


def test_timestamps_unchecked_bitcoin_claim_is_skipped(monkeypatch, tmp_path):
    """Finding: a proof that claims a Bitcoin attestation which was not checked is never a PASS."""
    from volrisk_live import ots

    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(paths, "RUNS", runs)
    _session_payload(runs)
    monkeypatch.setattr(ots, "status_all", lambda: {"counts": {}})
    monkeypatch.setattr(ots, "verify", lambda path: {"status": "unverified", "attested_utc": None,
                                                     "reason": "no block explorer reachable"})
    r = V.run_check("8b", "t", "m", lambda: V.check_timestamps(network=True, now_utc=NOW))
    assert r["status"] == "SKIPPED" and "not checked" in r["reason"] and "no block explorer" in r["reason"]
    r = V.run_check("8b", "t", "m", lambda: V.check_timestamps(network=False, now_utc=NOW))
    assert r["status"] == "SKIPPED" and "network disabled" in r["reason"]


def test_ex_ante_window_from_the_session_calendar():
    assert V.ex_ante_window("BTC", "1d", date(2026, 10, 1)) == (date(2026, 10, 2), date(2026, 10, 2), 1)
    assert V.ex_ante_window("BTC", "1w", date(2026, 10, 1)) == (date(2026, 10, 2), date(2026, 10, 8), 7)
    assert V.ex_ante_window("SPX", "1d", date(2026, 10, 2)) == (date(2026, 10, 5), date(2026, 10, 5), 1)  # Friday
    first, last, n = V.ex_ante_window("SPX", "1m", date(2026, 10, 1))
    assert first == date(2026, 10, 2) and n <= C.n_max("1m", "SPX") and last <= date(2026, 10, 31)


def test_payload_forecasts_rederived(monkeypatch, tmp_path):
    runs, live = tmp_path / "runs", tmp_path / "live"
    runs.mkdir()
    live.mkdir()
    monkeypatch.setattr(paths, "RUNS", runs)
    monkeypatch.setattr(paths, "LIVE_RESULTS", live)
    monkeypatch.setattr(V, "payload_inputs", lambda items: {})  # input hashes: see the next test
    origin = date(2026, 10, 1)
    rows = []
    for asset, h in (("BTC", "1d"), ("BTC", "1w"), ("SPX", "1m")):
        first, last, n = V.ex_ante_window(asset, h, origin)
        rows.append({"asset": asset, "horizon": h, "model": "COMBO", "origin": origin.isoformat(),
                     "window_first": first.isoformat(), "window_last": last.isoformat(), "n_t": n, "F": 1.5 * n,
                     "vol_ann": 40.0})
    risk = [{"asset": "BTC", "model": "COMBO+FHS", "date": "2026-10-02", "sigma": 1.5, "var99": 3.5, "var975": 3.0,
             "es975": 3.6}]
    p = runs / "20261002T070000Z.json"
    p.write_text(schema.canonical_json({"forecasts": rows, "risk": risk}), encoding="utf-8")
    assert V.run_check("8d", "t", "m", V.check_payload_forecasts)["status"] == "SKIPPED"  # no walk-forward file

    fc = pd.DataFrame(rows)[["asset", "horizon", "model", "origin", "n_t", "F"]]
    fc["origin"] = pd.to_datetime(fc["origin"])
    rk = pd.DataFrame(risk).assign(date=lambda d: pd.to_datetime(d["date"]), r_cc=np.nan, next=True)

    def write(f: pd.DataFrame, r: pd.DataFrame = rk) -> None:
        f.to_parquet(live / "forecasts.parquet", index=False)
        r.to_parquet(live / "risk.parquet", index=False)

    write(pd.concat([fc, fc.assign(model="HAR")], ignore_index=True))
    status, ev, _ = V.check_payload_forecasts()
    assert status == "PASS" and ev["counts"]["equal"] == 3 and ev["counts"]["risk_equal"] == 1
    write(fc.assign(F=fc["F"] * np.array([1.0, 1.0 + 1e-6, 1.0])))  # a recorded number that the models do not give
    status, ev, reason = V.check_payload_forecasts()
    assert status == "FAIL" and ev["counts"]["different"] == 1 and "BTC 1w COMBO" in reason
    write(fc.iloc[:2])
    assert V.check_payload_forecasts()[0] == "FAIL"  # recorded forecast missing from the walk-forward
    write(fc, rk.assign(var99=4.0))
    assert V.check_payload_forecasts()[0] == "FAIL"  # VaR/ES changed
    # realised sessions differ from the schedule (e.g. an outage): reported as not comparable, not as a difference
    write(pd.concat([fc.iloc[[0, 2]], fc.iloc[[1]].assign(n_t=6, F=99.0)], ignore_index=True))
    status, ev, _ = V.check_payload_forecasts()
    assert status == "PASS" and ev["counts"]["n_t_realised_differs"] == 1
    # the source rows a payload used were revised after its run: its values cannot be re-derived -> SKIPPED, not
    # FAIL; with unchanged inputs the same difference is a FAIL
    write(fc.assign(F=fc["F"] * 1.01))
    monkeypatch.setattr(V, "payload_inputs", lambda items: {(p.name, "BTC"): False, (p.name, "SPX"): True})
    r = V.run_check("8d", "t", "m", V.check_payload_forecasts)
    assert r["status"] == "FAIL" and "SPX 1m COMBO" in r["reason"]
    write(fc.assign(F=fc["F"] * np.array([1.01, 1.01, 1.0])))
    r = V.run_check("8d", "t", "m", V.check_payload_forecasts)
    assert r["status"] == "SKIPPED" and "revised after the run" in r["reason"]
    assert r["evidence"]["counts"]["inputs_revised"] == 2
    monkeypatch.setattr(V, "payload_inputs", lambda items: {})
    # a payload window that is not the ex-ante one of the calendar
    write(fc)
    bad = [dict(rows[0], window_first="2026-10-03")] + rows[1:]
    p.write_text(schema.canonical_json({"forecasts": bad, "risk": risk}), encoding="utf-8")
    status, ev, reason = V.check_payload_forecasts()
    assert status == "FAIL" and ev["counts"]["window_mismatch"] == 1 and "ex-ante" in reason


def test_payload_inputs_compare_the_recorded_row_hashes(monkeypatch):
    d = _gold("BTC", 30, start="2026-09-01")

    def rows_sha(rows: pd.DataFrame) -> str:
        return hashlib.sha256(rows.to_csv(index=False).encode()).hexdigest()

    sha = rows_sha(d[d["session_date"] <= pd.Timestamp("2026-09-20")])
    mods = {"update": SimpleNamespace(live_daily=lambda: d.copy()),
            "forecast": SimpleNamespace(normalise_daily=lambda x: x, rows_sha256=rows_sha)}
    monkeypatch.setattr(V, "_live_module", lambda name: mods[name])
    items = [(Path("a.json"), {"data": {"BTC": {"last_session": "2026-09-20", "rows_sha256": sha}}}),
             (Path("b.json"), {"data": {"BTC": {"last_session": "2026-09-21", "rows_sha256": sha}}}),
             (Path("c.json"), {"data": {"BTC": {"rows_sha256": sha}}}), (Path("d.json"), None)]
    assert V.payload_inputs(items) == {("a.json", "BTC"): True, ("b.json", "BTC"): False, ("c.json", "BTC"): None}

    def missing():
        raise FileNotFoundError("no live tables")

    mods["update"] = SimpleNamespace(live_daily=missing)
    assert V.payload_inputs(items) == {}


# ------------------------------------------------------------------------------------- runner and report
def test_run_check_maps_outcomes():
    assert V.run_check("x", "t", "m", lambda: ("PASS", {"n": 1}))["status"] == "PASS"
    r = V.run_check("x", "t", "m", lambda: V._skip("only with --full"))
    assert r["status"] == "SKIPPED" and r["reason"] == "only with --full"

    def offline():
        raise requests.ConnectionError("no route")

    assert V.run_check("x", "t", "m", offline)["status"] == "SKIPPED"

    def broken():
        raise ValueError("bug")

    r = V.run_check("x", "t", "m", broken)
    assert r["status"] == "FAIL" and "ValueError" in r["reason"]


def test_run_all_writes_report(monkeypatch, tmp_path):
    md, js = tmp_path / "verification.md", tmp_path / "verification.json"
    monkeypatch.setattr(paths, "VERIFY_MD", md)
    monkeypatch.setattr(paths, "VERIFY_JSON", js)
    stub = [
        ("1", "Frozen code unchanged", "Hash every file.", lambda: ("PASS", {"summary": "ok", "n": np.int64(46)})),
        ("4a", "BTC vs Coinbase", "Correlate.", lambda: ("FAIL", {"corr": np.float64(0.5), "when": date(2026, 1, 1),
                                                                   "rows": [{"a": 1, "b": float("nan")}]}, "low")),
        ("5e", "Full", "Re-run.", lambda: V._skip("only with --full")),
        ("8a", "Ledger", "Chain.", lambda: ("PASS", {"summary": "3 entries"}, "1 UTC day without a run")),
    ]
    monkeypatch.setattr(V, "plan", lambda *a, **k: stub)
    res = V.run_all(seed=11, now_utc=datetime(2026, 10, 2, 18, 0, tzinfo=UTC))
    assert res["overall"] == "FAIL" and res["counts"] == {"PASS": 2, "FAIL": 1, "SKIPPED": 1}
    assert res["args"]["seed"] == 11 and res["args"]["seed_source"] == {"source": "given (--seed)"}
    data = json.loads(js.read_text(encoding="utf-8"))
    assert data["generated_utc"] == "2026-10-02T18:00:00Z"
    assert [c["status"] for c in data["checks"]] == ["PASS", "FAIL", "SKIPPED", "PASS"]
    assert data["checks"][1]["evidence"]["rows"][0]["b"] is None
    for c in data["checks"]:
        assert {"id", "title", "status", "evidence", "method"} <= set(c)
    text = md.read_text(encoding="utf-8")
    assert "**Overall: FAIL**" in text and "Pass rules" in text and "BTC vs Coinbase" in text
    assert "| a | b |" in text and "`--seed 11` repeats this draw" in text
    assert "| 8a | Ledger | **PASS** | 3 entries (1 UTC day without a run) |" in text  # a PASS note is visible

    stub[1] = ("4a", "BTC vs Coinbase", "Correlate.", lambda: ("PASS", {}))
    assert V.run_all(seed=1, write=False)["overall"] == "PASS"  # SKIPPED never fails the run
    tip = "11" * 28 + "000000ff"  # 64 hex digits
    monkeypatch.setattr(V, "_http_get", lambda url, **k: tip.encode())
    res = V.run_all(write=False)  # no --seed: drawn at run time and recorded
    assert res["args"]["seed"] == 0xFF and res["args"]["seed_source"]["block_hash"] == tip


def test_plan_ids_titles_and_methods():
    ids = [p[0] for p in V.plan()]
    assert ids == ["1", "2", "3a", "3b", "3c", "4a", "4b", "4c", "4d", "5a", "5b", "5c", "5d", "5e", "5f", "5g", "6a",
                   "6b", "6c", "7", "8a", "8b", "8c", "8d"]
    assert all(p[1] and p[2].endswith(".") for p in V.plan())
    for cid in ("5e", "5f", "5g"):
        assert V.run_check(*[p for p in V.plan(full=False) if p[0] == cid][0])["status"] == "SKIPPED"
    methods = {p[0]: p[2] for p in V.plan()}
    assert "internal consistency only" in methods["5d"] and "compares files" in methods["5a"]
    assert "published elsewhere" in methods["8a"] and "2 hours" in methods["8b"]
    offline = {p[0]: p for p in V.plan(network=False)}
    for cid in ("3a", "3b", "8b", "8c"):
        assert "(network disabled)" in offline[cid][2]


def test_no_tuning_is_skipped_without_the_spec(tmp_path):
    with pytest.raises(V.SkipCheck, match="not in this checkout"):
        V.check_no_tuning(spec_path=tmp_path / "SPEC.md")
