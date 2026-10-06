"""Live forecast module (docs/LIVE_SPEC.md §3, §5): live targets with ex-ante windows, the reproduction proofs,
next-session VaR/ES, the payload and tomorrow.md.

The "sealed" baseline is produced by the frozen code path itself (``pipeline.compute_forecasts(include_holdout=True)``
and ``risk.suite.build_risk`` on a synthetic BTC gold table that ends on the frozen data end). The live path then
runs on the same table extended by two sessions with a later live end. Forecasts run in-process with ``W = 200``
and six models (GJR + HARQ + LGBM = the COMBO members), so the walk-forwards take seconds. Nothing under the real
``data/``, ``forecasts/`` or ``reports/`` is read or written: the holdout log, the seal and every output path are
redirected or stubbed.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import types
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from volrisk import config as C
from volrisk import holdout as H
from volrisk import io, pipeline
from volrisk.models.base import align_targets, predict_mask_for
from volrisk.risk import suite as risk_suite
from volrisk.risk import var_es
from volrisk.targets import build_all_targets
from volrisk_live import context, paths, schema
from volrisk_live import forecast as F

W_SMALL = 200
START = date(2024, 9, 1)
FROZEN_END = date(2026, 9, 30)
LIVE_END = date(2026, 10, 2)  # two sessions after the frozen end
FAST = ("RW", "EWMA", "GJR", "HAR", "HARQ", "LGBM")
STAR = {"BTC": "HARQ"}
OPENING = {"utc": "2026-10-02T15:20:42+00:00", "n_previous": 0}
RUN_UTC = datetime(2026, 10, 3, 0, 30, tzinfo=timezone.utc)
CHECK_REPORT = {"ok": True, "frozen_data_end": FROZEN_END, "dev": {"rows_compared": 900},
                "holdout": {"rows_compared": 395}, "implied": {"rows_compared": 1295}}


def _gold(end: date, seed: int = 11) -> pd.DataFrame:
    """Gold-like BTC rows (every calendar day), persistent log-vol with jumps (as tests/test_holdout_run.py)."""
    dates = pd.date_range(START, end, freq="D")
    n = len(dates)
    rng = np.random.default_rng(seed)
    h = np.zeros(n)
    for i in range(1, n):
        h[i] = 0.95 * h[i - 1] + 0.25 * rng.standard_normal()
    rv = np.exp(h) * rng.gamma(4.0, 0.25, n)
    j = np.where(rng.random(n) < 0.15, rv * rng.uniform(0.1, 0.5, n), 0.0)
    share = rng.uniform(0.3, 0.7, n)
    return pd.DataFrame({
        "asset": "BTC",
        "session_date": dates.to_numpy().astype("datetime64[ms]"),
        "rv": rv, "bv": rv - j, "c": rv - j, "j": j,
        "rs_pos": rv * share, "rs_neg": rv * (1 - share),
        "rq": rv**2 * rng.uniform(1.0, 3.0, n),
        "gap": np.zeros(n),
        "r_cc": rng.standard_normal(n) * np.sqrt(rv),
        "tv": rv,
    })


def _implied(gold: pd.DataFrame, seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    iv = 50.0 + 5.0 * rng.standard_normal(len(gold))
    return pd.DataFrame({"asset": "BTC", "origin": gold["session_date"].dt.date, "iv": iv,
                         "iv_var_30d": iv**2 * 30 / 365, "source": "DVOL"})


class _Inline:
    """In-process stand-in for ProcessPoolExecutor (patched config values do not reach spawned workers)."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, *it):
        return list(map(fn, *it))


def _small_risk_cfg(cfg=var_es._risk_cfg()):
    return {**cfg, "fhs_pool": 300, "fhs_min": 100}  # W = 200 leaves ~530 1d forecasts before the live end


@pytest.fixture(scope="module")
def world():
    """Sealed baseline (frozen code path, data to 2026-09-30) and the live run (data to 2026-10-02)."""
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(C, "window", lambda: W_SMALL)
        mp.setattr(var_es, "_risk_cfg", _small_risk_cfg)
        mp.setattr(H, "openings", lambda: [OPENING])
        mp.setattr(pipeline, "ProcessPoolExecutor", _Inline)
        assert C.data_end() == FROZEN_END

        gold_all = _gold(LIVE_END)
        imp_all = _implied(gold_all)
        gold_sealed = gold_all[gold_all["session_date"] <= pd.Timestamp(FROZEN_END)].reset_index(drop=True)
        imp_sealed = imp_all[pd.to_datetime(imp_all["origin"]) <= pd.Timestamp(FROZEN_END)].reset_index(drop=True)

        def load_daily(asset=None, include_holdout=False):
            if include_holdout:
                return gold_sealed.copy()
            return gold_sealed[gold_sealed["session_date"] < pd.Timestamp(C.holdout_start())].reset_index(drop=True)

        def load_implied(asset=None):
            return imp_sealed.copy()

        mp.setattr(io, "load_daily", load_daily)
        mp.setattr(io, "load_implied", load_implied)
        fc_dev, _ = pipeline.compute_forecasts(include_holdout=False, workers=1, models=FAST)
        fc_hold, tg_hold = pipeline.compute_forecasts(include_holdout=True, workers=1, models=FAST)
        risk_hold = risk_suite.build_risk(gold_sealed, fc_hold, STAR)
        sealed_fc = [("holdout", fc_hold), ("dev", fc_dev)]

        run = F.compute(LIVE_END, gold_all, imp_all, workers=1, models=FAST, har_star=STAR, reference_fc=sealed_fc,
                        reference_risk=risk_hold)
        yield types.SimpleNamespace(gold=gold_all, implied=imp_all, gold_sealed=gold_sealed, fc_dev=fc_dev,
                                    fc_hold=fc_hold, tg_hold=tg_hold, risk_hold=risk_hold, sealed_fc=sealed_fc,
                                    run=run, mp=mp)
    finally:
        mp.undo()
    assert C.data_end() == FROZEN_END


@pytest.fixture
def patched(world, monkeypatch):
    """Per-test patches on top of the module ones (the module patch stays active while ``world`` lives)."""
    return monkeypatch


# ------------------------------------------------------------------------------------------------ ex-ante windows
@pytest.mark.parametrize(
    "asset, horizon, origin, n_t, first, last",
    [
        ("SPX", "1w", "2025-12-23", 4, "2025-12-24", "2025-12-30"),  # Christmas closed
        ("EURUSD", "1w", "2025-12-23", 4, "2025-12-24", "2025-12-30"),  # Dec 25 is never an FX session
        ("SPX", "1w", "2026-11-20", 4, "2026-11-23", "2026-11-27"),  # Thanksgiving closed
        ("SPX", "1m", "2026-11-30", 21, "2026-12-01", "2026-12-30"),
        ("SPX", "1d", "2026-11-25", 1, "2026-11-27", "2026-11-27"),
        ("SPX", "1d", "2026-10-02", 1, "2026-10-05", "2026-10-05"),  # Friday -> Monday
        ("EURUSD", "1d", "2025-12-24", 1, "2025-12-26", "2025-12-26"),
        ("EURUSD", "1m", "2026-10-01", 21, "2026-10-02", "2026-10-30"),  # 21 weekdays in (10-01, 10-31]
        ("EURUSD", "1m", "2026-07-28", 22, "2026-07-29", "2026-08-27"),  # 22 = n_max (30 days hold at most 22)
        ("BTC", "1w", "2026-10-01", 7, "2026-10-02", "2026-10-08"),
        ("BTC", "1m", "2026-10-01", 30, "2026-10-02", "2026-10-31"),
        ("BTC", "1d", "2026-10-01", 1, "2026-10-02", "2026-10-02"),
    ],
)
def test_schedule_window_counts_scheduled_sessions(asset, horizon, origin, n_t, first, last):
    w = F.schedule_window(asset, horizon, [pd.Timestamp(origin)])
    assert w["n_t"].tolist() == [n_t]
    assert w["n_t"].iloc[0] <= C.n_max(horizon, asset)
    assert F._iso(w["window_first"].iloc[0]) == first and F._iso(w["window_last"].iloc[0]) == last


def test_schedule_window_caps_at_n_max(monkeypatch):
    """Real calendars reach n_max but cannot exceed it; a smaller cap shows the window is cut at the n-th session."""
    monkeypatch.setattr(C, "n_max", lambda h, a: 3)
    w = F.schedule_window("SPX", "1w", [pd.Timestamp("2026-10-02"), pd.Timestamp("2026-11-25")])
    assert w["n_t"].tolist() == [3, 3]
    assert [F._iso(x) for x in w["window_first"]] == ["2026-10-05", "2026-11-27"]
    assert [F._iso(x) for x in w["window_last"]] == ["2026-10-07", "2026-12-01"]


# ------------------------------------------------------------------------------------------------ live targets
def test_live_targets_keep_frozen_rows_and_mark_live_origins(world):
    tg = world.run.targets
    assert C.data_end() == FROZEN_END  # the live_end override is gone after the call
    with context.live_end(LIVE_END):
        frozen = build_all_targets(world.gold, last_date=LIVE_END)
    assert C.data_end() == FROZEN_END
    rest = ~tg["live"].to_numpy()
    pd.testing.assert_frame_equal(tg.loc[rest, frozen.columns].reset_index(drop=True),
                                  frozen.loc[rest].reset_index(drop=True))

    live = tg[tg["live"]]
    assert (live["split"] == "holdout").all() and live["y"].isna().all() and live["ybar"].isna().all()
    assert live["window_end"].isna().all()
    got = {h: sorted(pd.to_datetime(g["origin"]).dt.date) for h, g in live.groupby("horizon")}
    # 1d: last holdout origin is 10-01 (window 10-02 complete) -> live = {10-02}; 1w: (t, t+7] complete up to
    # t = 09-25; 1m: up to t = 09-02
    assert got["1d"] == [LIVE_END]
    assert got["1w"] == [date(2026, 9, d) for d in range(26, 31)] + [date(2026, 10, 1), LIVE_END]
    assert len(got["1m"]) == 30 and got["1m"][0] == date(2026, 9, 3)
    last = live[pd.to_datetime(live["origin"]) == pd.Timestamp(LIVE_END)].set_index("horizon")
    assert last["n_t"].to_dict() == {"1d": 1, "1w": 7, "1m": 30}  # ex-ante, not the 0 of an empty window
    assert F._iso(last.loc["1m", "window_last"]) == "2026-11-01"
    # the frozen predict mask now selects every live origin
    a = align_targets(world.gold, tg[(tg["horizon"] == "1m")].drop(columns=F.LIVE_COLUMNS))
    assert predict_mask_for(a)[-30:].all()


def test_live_origins_never_train(world):
    """Rows without a complete window (every live origin) carry no target, so no model can train on them."""
    live = world.run.targets[world.run.targets["live"]]
    assert live["ybar"].isna().all() and live["window_end"].isna().all()


# ------------------------------------------------------------------------------------------------ reproduction
def test_live_run_reproduces_every_sealed_forecast(world):
    rep = world.run.reproduction
    n_hold = len(world.fc_hold)
    assert rep["passed"] and rep["rows"] == n_hold  # the holdout file holds the dev rows too
    assert rep["files"] == {"holdout": {"rows": n_hold, "sha256": None, "sealed_as": None},  # in-memory tables
                            "dev": {"rows": len(world.fc_dev), "sha256": None, "sealed_as": None}}
    assert rep["max_abs_diff"] == 0.0 and rep["rtol"] == pipeline.REPRO_RTOL and rep["atol"] == 0.0
    assert rep["new_rows"] == len(world.run.fc) - n_hold > 0
    # the sealed holdout run had no forecast at 2026-09-30 for 1d; the live run has, and at the live origin
    fc = world.run.fc
    at_end = fc[pd.to_datetime(fc["origin"]) == pd.Timestamp(LIVE_END)]
    want = {(m, h) for m in [*FAST, "COMBO"] for h in C.HORIZONS} | {("IV", "1m"), ("IV-cal", "1m")}
    assert set(zip(at_end["model"], at_end["horizon"])) == want
    assert (at_end["F"] > 0).all() and (at_end["split"] == "holdout").all()
    assert at_end.set_index(["model", "horizon"]).loc[("COMBO", "1m"), "n_t"] == 30


def test_refit_points_do_not_move(world):
    """GJR/LGBM refit on fixed grids: the live forecasts at sealed origins are bitwise the sealed ones."""
    keys = ["asset", "horizon", "model", "origin"]
    s = world.fc_hold.set_index(keys)["F"]
    live = world.run.fc.set_index(keys)["F"].reindex(s.index)
    for m in ("GJR", "LGBM", "HARQ", "COMBO", "IV-cal"):
        sel = s.index.get_level_values("model") == m
        assert np.array_equal(live[sel].to_numpy(), s[sel].to_numpy()), m


@pytest.mark.parametrize("fault", ["rel-1e-8", "missing-row", "n_t-changed"])
def test_reproduction_check_rejects_any_difference(world, fault):
    fc = world.run.fc.copy()
    i = fc.index[(fc["model"] == "HARQ") & (fc["split"] == "holdout") & (fc["horizon"] == "1w")][3]
    if fault == "rel-1e-8":
        fc.loc[i, "F"] *= 1 + 1e-8  # above rtol 1e-9, no absolute slack
    elif fault == "missing-row":
        fc = fc.drop(index=i)
    else:
        fc.loc[i, "n_t"] += 1
    with pytest.raises(F.ReproductionError):
        F.reproduction_check(fc, world.sealed_fc)


def test_check_reproduces_holdout_reads_the_files_and_pins_them_by_sha256(world, patched, tmp_path):
    """The holdout-run results are not in SEALED.json (written at the opening, after the seal): the reproduction
    statistics, which go into every payload, must record the SHA-256 of each compared file, and say which ones the
    seal covers. With row counts only, a later swap of the holdout results could not be seen from the ledger."""
    hold, dev = tmp_path / "holdout" / "forecasts.parquet", tmp_path / "forecasts.parquet"
    risk = tmp_path / "holdout" / "risk.parquet"
    hold.parent.mkdir()
    world.fc_hold.to_parquet(hold)
    world.fc_dev.to_parquet(dev)
    world.risk_hold.to_parquet(risk)
    seal = tmp_path / "SEALED.json"
    seal.write_text(json.dumps({"hashes": {"dev_forecasts": H.file_sha(dev), "code": "0" * 64}}), encoding="utf-8")
    patched.setattr(H, "SEALED", seal)
    patched.setattr(F, "results_dir", lambda include_holdout: tmp_path / "holdout")
    patched.setattr(io, "FORECASTS", dev)
    patched.setattr(C, "ROOT", tmp_path)
    assert F.check_reproduces_holdout(world.run.fc) == len(world.fc_hold)
    rep = F.reproduction_check(world.run.fc)
    assert rep["files"] == {
        "holdout/forecasts.parquet": {"rows": len(world.fc_hold), "sha256": H.file_sha(hold), "sealed_as": None},
        "forecasts.parquet": {"rows": len(world.fc_dev), "sha256": H.file_sha(dev), "sealed_as": "dev_forecasts"},
    }
    rr = F.risk_reproduction_check(world.run.risk)
    assert rr["rows"] == len(world.risk_hold)
    assert rr["files"] == {"holdout/risk.parquet": {"rows": len(world.risk_hold), "sha256": H.file_sha(risk),
                                                    "sealed_as": None}}
    # a swapped results file yields a different pin in the next payload, even when it still reproduces
    world.fc_hold.iloc[::-1].to_parquet(hold)
    assert F.reproduction_check(world.run.fc)["files"]["holdout/forecasts.parquet"]["sha256"] != \
        rep["files"]["holdout/forecasts.parquet"]["sha256"]
    patched.setattr(H, "openings", lambda: [])
    with pytest.raises(context.HoldoutNotOpenedError):
        F.check_reproduces_holdout(world.run.fc)


# ------------------------------------------------------------------------------------------------ risk
def test_next_session_risk_reproduces_sealed_risk_and_targets_the_next_session(world):
    rr = world.run.risk_reproduction
    assert rr["passed"] and rr["rows"] == len(world.risk_hold) > 0 and rr["max_abs_diff"] == 0.0
    nxt = F.next_session_risk(world.gold, world.run.fc, STAR, reference=world.risk_hold)
    assert nxt.attrs["reproduction"]["rows"] == len(world.risk_hold)
    assert list(nxt["model"]) == list(var_es.RISK_MODELS)
    assert (pd.to_datetime(nxt["date"]) == pd.Timestamp("2026-10-03")).all()
    assert (nxt[["var99", "var975", "es975", "sigma"]] > 0).all().all()
    assert (nxt["es975"] > nxt["var975"]).all() and (nxt["var99"] > nxt["var975"]).all()
    # COMBO+FHS sigma is the root of the COMBO 1d forecast made at the last session
    fc = world.run.fc
    combo = fc[(fc["model"] == "COMBO") & (fc["horizon"] == "1d") & (pd.to_datetime(fc["origin"]) ==
                                                                      pd.Timestamp(LIVE_END))]["F"].iloc[0]
    assert nxt.set_index("model").loc["COMBO+FHS", "sigma"] == pytest.approx(math.sqrt(combo), rel=1e-12)


def test_risk_placeholder_value_never_enters(world):
    a = F.live_risk_frame(world.gold, world.run.fc, STAR, placeholder=0.0)
    b = F.live_risk_frame(world.gold, world.run.fc, STAR, placeholder=-50.0)
    pd.testing.assert_frame_equal(a, b)
    assert a.loc[a["next"], "r_cc"].isna().all()


def test_risk_reproduction_rejects_a_changed_value(world):
    bad = world.risk_hold.copy()
    bad.loc[bad.index[-1], "var99"] *= 1 + 1e-8
    with pytest.raises(F.ReproductionError):
        F.risk_reproduction_check(world.run.risk, bad)


# ------------------------------------------------------------------------------------------------ payload
@pytest.fixture
def frozen_stub(world, patched, tmp_path):
    sealed = tmp_path / "SEALED.json"
    sealed.write_text(json.dumps({"created_utc": "2026-10-02T14:50:00+00:00"}), encoding="utf-8")
    patched.setattr(H, "SEALED", sealed)
    patched.setattr(H, "verify_seal", lambda: [])
    patched.setattr(F, "_wall_clock", lambda: RUN_UTC + pd.Timedelta(minutes=1))  # payload built 1 min after start
    return patched


def test_build_payload_follows_the_schema(world, frozen_stub):
    run = world.run
    p = F.build_payload(run, RUN_UTC, {"check_against_sealed": {"ok": True}})
    schema.validate_payload(p)
    assert p["run_id"] == "20261003T003000Z" and p["run_utc"] == "2026-10-03T00:30:00+00:00"
    assert p["frozen"]["seal_ok"] and p["frozen"]["code_sha"] == H.code_sha()
    assert p["frozen"]["holdout_opened_utc"] == OPENING["utc"]
    d = p["data"]["BTC"]
    assert d["last_session"] == "2026-10-02" and d["next_session"] == "2026-10-03"
    assert d["n_sessions"] == len(world.gold)
    assert d["rows_sha256"] == F.rows_sha256(world.gold) and not d["stale"]
    assert d["recorded_before_open"] is False  # the crypto session of 10-03 opened at 00:00 UTC
    assert d["recorded_before_close"] is True  # ... and closes at 24:00 UTC
    assert len(p["forecasts"]) == (len(FAST) + 1) * 3 + 2 and p["forecasts"][0]["model"] == "COMBO"
    for r in p["forecasts"]:
        assert r["origin"] == "2026-10-02" and r["window_first"] == "2026-10-03"
        assert r["vol_ann"] == pytest.approx(math.sqrt(r["F"] / r["n_t"] * 365))
    assert {r["horizon"]: r["n_t"] for r in p["forecasts"] if r["model"] == "COMBO"} == {"1d": 1, "1w": 7, "1m": 30}
    assert [r["model"] for r in p["risk"]] == list(var_es.RISK_MODELS)
    assert p["implied"] == [{"asset": "BTC", "origin": "2026-10-02", "iv": world.implied["iv"].iloc[-1],
                             "iv_var_30d": world.implied["iv_var_30d"].iloc[-1], "source": "DVOL"}]
    ck = p["checks"]
    assert ck["live_end"] == "2026-10-02" and ck["reproduction"]["rows"] == len(world.fc_hold)
    assert ck["risk_reproduction"]["rows"] == len(world.risk_hold) and ck["stale_assets"] == []
    assert ck["closed_assets"] == [] and ck["replayed_now"] is False
    # timing flags: machine clock at the payload build (00:31) + the ledger-write margin, not the start (00:30)
    assert ck["computed_utc"] == "2026-10-03T00:31:00+00:00" and ck["timing_utc"] == "2026-10-03T00:32:00+00:00"
    text = schema.canonical_json(p)  # strict JSON (no NaN), deterministic
    assert json.loads(text)["forecasts"][0]["F"] == p["forecasts"][0]["F"]
    assert schema.canonical_json(json.loads(text)) == text


def test_stale_asset_gets_no_forecast(world, frozen_stub):
    run = dataclasses.replace(world.run, end=date(2026, 10, 4))  # data ends 10-02, sessions 10-03/04 missing
    p = F.build_payload(run, datetime(2026, 10, 5, 1, tzinfo=timezone.utc))
    schema.validate_payload(p)
    assert p["forecasts"] == [] and p["risk"] == [] and p["checks"]["stale_assets"] == ["BTC"]
    assert p["data"]["BTC"]["stale"] is True
    assert "No forecast" in F.render_tomorrow(p)


@pytest.mark.parametrize("built, recorded", [("23:58:00", True), ("23:59:30", False)])
def test_asset_whose_next_session_closed_before_recording_gets_no_forecast(world, frozen_stub, built, recorded):
    """A run late in the session day: the data are current (cutoff 10-02, next session 10-03 not stale), but the
    10-03 session closes at 24:00 UTC. If the payload is recorded after that close (the build time plus the 60 s
    ledger margin), its outcome is already known, so no forecast or VaR/ES row may be recorded for it."""
    clock = datetime.fromisoformat(f"2026-10-03T{built}+00:00")
    frozen_stub.setattr(F, "_wall_clock", lambda: clock)
    p = F.build_payload(world.run, datetime(2026, 10, 3, 23, 50, tzinfo=timezone.utc))
    schema.validate_payload(p)
    d, ck = p["data"]["BTC"], p["checks"]
    assert d["stale"] is False and ck["stale_assets"] == []
    assert d["recorded_before_close"] is recorded and d["recorded_before_open"] is False
    md = F.render_tomorrow(p)
    if recorded:
        assert ck["closed_assets"] == [] and len(p["forecasts"]) > 0 and len(p["risk"]) == len(var_es.RISK_MODELS)
        assert "closes 0.0 h after it" in md
    else:
        assert ck["closed_assets"] == ["BTC"] and p["forecasts"] == [] and p["risk"] == []
        assert "had already closed when this payload was recorded (2026-10-04T00:00:30+00:00)" in md
        assert "before this payload was recorded" not in md.split("## BTC")[1].split("## How to verify")[0]


def test_asset_status_flags_sessions_closed_at_recording_on_real_calendars():
    """The reviewer's probe: cutoff 2026-10-01, recorded 2026-10-02 21:30 UTC. The SPX session of 10-02 closed at
    20:00 UTC and the EUR/USD one at 21:00 UTC, so both outcomes are known although the data are not stale."""
    days = pd.to_datetime(["2026-09-29", "2026-09-30", "2026-10-01"]).to_numpy().astype("datetime64[ms]")
    daily = pd.concat([pd.DataFrame({"asset": a, "session_date": days, "tv": [1.0, 2.0, 3.0]})
                       for a in ("EURUSD", "SPX")], ignore_index=True)

    def flags(hh, mm):
        st = F.asset_status(daily, date(2026, 10, 1), datetime(2026, 10, 2, hh, mm, tzinfo=timezone.utc))
        return {a: (s["stale"], s["recorded_before_open"], s["recorded_before_close"]) for a, s in st.items()}

    assert flags(21, 30) == {"EURUSD": (False, False, False), "SPX": (False, False, False)}
    assert flags(20, 30) == {"EURUSD": (False, False, True), "SPX": (False, False, False)}
    assert flags(13, 0) == {"EURUSD": (False, False, True), "SPX": (False, True, True)}  # SPX opens 13:30 UTC


def test_tomorrow_md(world, frozen_stub, tmp_path):
    p = F.build_payload(world.run, RUN_UTC, {"check_against_sealed": {"ok": True}})
    p = json.loads(schema.canonical_json(p))  # a copy: the checks below edit it, world.run stays untouched
    p["checks"]["reproduction"]["files"] = {
        "data/results/holdout/forecasts.parquet": {"rows": 7, "sha256": "d" * 64, "sealed_as": None},
        "data/results/forecasts.parquet": {"rows": 5, "sha256": "9" * 64, "sealed_as": "dev_forecasts"}}
    entry = {"seq": 3, "payload_sha256": "a" * 64, "entry_sha256": "b" * 64, "prev_entry_sha256": "c" * 64}
    out = F.write_tomorrow(p, entry, tmp_path / "tomorrow.md")
    md = out.read_text(encoding="utf-8")
    combo_1d = next(r for r in p["forecasts"] if r["model"] == "COMBO" and r["horizon"] == "1d")
    assert "| **COMBO** (GJR + HARQ + LGBM, primary) | " + f"**{combo_1d['vol_ann']:.1f}**" in md
    assert "**COMBO+FHS** (primary)" in md and "DVOL" in md and "2026-10-02" in md
    assert f"{len(world.fc_hold):,} forecasts" in md and "Ledger entry #3" in md and "b" * 64 in md
    # timing relative to the recording time (00:32 = build 00:31 + margin), with the close
    assert "opened 0.5 h before this payload was recorded and closes 23.5 h after it" in md
    assert "Timing is checked at 2026-10-03T00:32:00+00:00" in md and "Replayed run" not in md
    # the holdout-run results are pinned by hash and not called sealed
    assert ("`data/results/holdout/forecasts.parquet` (7 rows; sha256 `" + "d" * 64 + "`; not in SEALED.json, "
            "pinned by the hash recorded in this payload)") in md
    assert "sealed in SEALED.json as `dev_forecasts`" in md
    assert "sealed VaR/ES" not in md and "sealed dev/holdout run" not in md
    assert "verify" in md and ".ots" in md


# ------------------------------------------------------------------------------------------------ entry point
@pytest.fixture
def live_env(world, frozen_stub, tmp_path):
    """forecast() with stubbed update/ledger modules, outputs redirected to tmp_path, compute() replayed."""
    mp = frozen_stub
    # project: whether the stub ledger claims to be forecasts/ledger.jsonl; clock: the machine clock, advanced by
    # walk_forward during compute()
    calls = types.SimpleNamespace(ledger=[], compute=[], project=False, clock=RUN_UTC + pd.Timedelta(minutes=1),
                                  walk_forward=pd.Timedelta(0))
    state = tmp_path / "state.json"
    state.write_text(json.dumps({"end": "2026-10-02", "last_sessions": {"BTC": "2026-10-02"}}), encoding="utf-8")
    upd = types.ModuleType("volrisk_live.update")
    upd.live_end_date = lambda now: (now - pd.Timedelta(days=1)).date()
    upd.check_against_sealed = lambda: dict(CHECK_REPORT)
    upd.live_daily = lambda: world.gold
    upd.live_implied = lambda: world.implied
    led = types.ModuleType("volrisk_live.ledger")

    def append(payload, stamp=True):
        schema.validate_payload(payload)
        calls.ledger.append((payload, stamp))
        return {"seq": 1, "run_id": payload["run_id"], "payload_sha256": "p" * 64, "entry_sha256": "e" * 64,
                "prev_entry_sha256": schema.ZERO_HASH}

    led.append = append
    led.is_project_ledger = lambda: calls.project

    def compute(end, daily_all, implied, workers=10, **kw):
        calls.compute.append((end, workers))
        calls.clock = calls.clock + calls.walk_forward
        return dataclasses.replace(world.run, end=end)

    import volrisk_live

    for name, mod in (("update", upd), ("ledger", led)):
        mp.setitem(sys.modules, f"volrisk_live.{name}", mod)
        mp.setattr(volrisk_live, name, mod, raising=False)
    mp.setattr(F, "compute", compute)
    mp.setattr(F, "_wall_clock", lambda: calls.clock)
    mp.setattr(paths, "LIVE_STATE", state)
    mp.setattr(paths, "LIVE_RESULTS", tmp_path / "live_forecasts")
    mp.setattr(paths, "TOMORROW_MD", tmp_path / "reports" / "tomorrow.md")
    return calls


def test_forecast_records_one_payload_and_writes_reports(world, live_env, tmp_path):
    run_id = F.forecast(now_utc=RUN_UTC, workers=3, stamp=False)
    assert run_id == "20261003T003000Z"
    assert live_env.compute == [(LIVE_END, 3)]
    (payload, stamp), = live_env.ledger
    assert stamp is False and payload["run_id"] == run_id
    assert payload["checks"]["computed_utc"] == "2026-10-03T00:31:00+00:00"  # the machine clock, next to run_utc
    dc = payload["checks"]["data_consistency"]
    assert dc["ok"] is True and dc["check_against_sealed"] == CHECK_REPORT
    assert dc["state"] == {"end": "2026-10-02", "last_sessions": {"BTC": "2026-10-02"}}
    md = (tmp_path / "reports" / "tomorrow.md").read_text(encoding="utf-8")
    assert md.count("COMBO") >= 2 and "Ledger entry #1" in md and "not stamped in this run" in md
    assert "dev gold 900 rows, holdout gold 395 rows, implied vol 1,295 rows identical, cell by cell" in md
    assert {p.name for p in (tmp_path / "live_forecasts").iterdir()} == {"forecasts.parquet", "targets.parquet",
                                                                          "risk.parquet"}


def test_forecast_refuses_a_future_now(world, live_env):
    with pytest.raises(ValueError, match="future"):
        F.forecast(now_utc=RUN_UTC + pd.Timedelta(minutes=10), stamp=False)
    assert live_env.ledger == [] and live_env.compute == []


def test_forecast_refuses_a_backdated_now_on_the_project_ledger(world, live_env, tmp_path):
    """A past ``now_utc`` would backdate run_utc / run_id (and let one pick afterwards which days to record)."""
    live_env.project = True
    with pytest.raises(ValueError, match="before the machine clock"):
        F.forecast(now_utc=RUN_UTC - pd.Timedelta(minutes=10), stamp=False)
    assert live_env.ledger == [] and live_env.compute == []
    F.forecast(now_utc=RUN_UTC, stamp=False)  # 1 min behind the clock: within FUTURE_SLACK
    (payload, _), = live_env.ledger
    assert payload["checks"]["replayed_now"] is False


def test_forecast_marks_a_replayed_now_in_a_test_sandbox(world, live_env, tmp_path):
    run_id = F.forecast(now_utc=RUN_UTC - pd.Timedelta(minutes=10), stamp=False)
    (payload, _), = live_env.ledger
    assert run_id == "20261003T002000Z" and payload["checks"]["replayed_now"] is True
    assert "**Replayed run:**" in (tmp_path / "reports" / "tomorrow.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("start, end, before_open, before_close", [
    ("2026-10-02T23:55:00", "2026-10-01", False, True),  # 10-03 session opens 00:00, during the walk-forward
    ("2026-10-03T23:55:00", "2026-10-02", False, False),  # ... and closes 24:00, during the walk-forward
])
def test_forecast_timing_flags_use_the_recording_time_not_the_start(world, live_env, tmp_path, start, end,
                                                                     before_open, before_close):
    """The walk-forward takes minutes. A session that opens or closes between the start of the run and the ledger
    write must be judged at the write: flags from ``run_utc`` claimed "recorded before it opens" for a payload
    written after the open, and recorded forecasts whose target session had already closed."""
    t0 = datetime.fromisoformat(start + "+00:00")
    live_env.clock, live_env.walk_forward = t0, pd.Timedelta(minutes=9)
    paths.LIVE_STATE.write_text(json.dumps({"end": end}), encoding="utf-8")
    run_id = F.forecast(now_utc=t0, stamp=False)
    (payload, _), = live_env.ledger
    d, ck = payload["data"]["BTC"], payload["checks"]
    assert d["next_session"] == "2026-10-03" and not d["stale"]  # the stub data end on 10-02
    assert (d["recorded_before_open"], d.get("recorded_before_close")) == (before_open, before_close)
    assert (len(payload["forecasts"]) > 0, len(payload["risk"]) > 0) == (before_close, before_close)
    assert run_id == t0.strftime("%Y%m%dT%H%M%SZ") and payload["run_utc"] == t0.isoformat()  # run_id = start
    built = t0 + pd.Timedelta(minutes=9)
    assert ck["computed_utc"] == built.isoformat() and ck["timing_utc"] == (built + F.RECORD_MARGIN).isoformat()
    md = (tmp_path / "reports" / "tomorrow.md").read_text(encoding="utf-8")
    if before_close:
        assert ck["closed_assets"] == [] and "opened 0.1 h before this payload was recorded" in md
    else:
        assert ck["closed_assets"] == ["BTC"]


def test_forecast_refuses_without_a_current_update(world, live_env):
    paths.LIVE_STATE.write_text(json.dumps({"end": "2026-10-01"}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="run update first"):
        F.forecast(now_utc=RUN_UTC, stamp=False)
    paths.LIVE_STATE.unlink()
    with pytest.raises(RuntimeError, match="update"):
        F.forecast(now_utc=RUN_UTC, stamp=False)
    assert live_env.ledger == [] and live_env.compute == []


def test_forecast_refuses_on_a_broken_seal_or_drifted_data(world, live_env, patched):
    patched.setattr(H, "verify_seal", lambda: ["code"])
    with pytest.raises(H.SealError):
        F.forecast(now_utc=RUN_UTC, stamp=False)
    patched.setattr(H, "verify_seal", lambda: [])

    class LiveDataMismatch(RuntimeError):
        pass

    def drifted():
        raise LiveDataMismatch("holdout gold differs in r_cc")

    patched.setattr(sys.modules["volrisk_live.update"], "check_against_sealed", drifted)
    with pytest.raises(LiveDataMismatch):
        F.forecast(now_utc=RUN_UTC, stamp=False)
    assert live_env.ledger == [] and live_env.compute == []


def test_rows_sha256_is_order_independent_and_value_sensitive(world):
    g = world.gold
    assert F.rows_sha256(g) == F.rows_sha256(g.iloc[::-1])
    h = g.copy()
    h.loc[5, "rv"] = np.nextafter(h.loc[5, "rv"], np.inf)
    assert F.rows_sha256(h) != F.rows_sha256(g)


def test_payload_reports_only_recent_implied_vol(world, frozen_stub):
    old = world.implied[pd.to_datetime(world.implied["origin"]) <= pd.Timestamp("2026-09-24")]
    p = F.build_payload(dataclasses.replace(world.run, implied=old), RUN_UTC)
    assert p["implied"] == []  # newest value is 8 days before the cutoff
    assert "No DVOL value in the last 7 days" in F.render_tomorrow(p)
    recent = world.implied[pd.to_datetime(world.implied["origin"]) <= pd.Timestamp("2026-09-26")]
    q = F.build_payload(dataclasses.replace(world.run, implied=recent), RUN_UTC)
    assert [r["origin"] for r in q["implied"]] == ["2026-09-26"]
