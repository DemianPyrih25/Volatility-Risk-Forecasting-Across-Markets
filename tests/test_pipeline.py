"""Pipeline orchestration (SPEC §11, §12): the dev-reproduction checks of ``freeze`` / ``holdout``, staging of the
holdout outputs, the holdout data-quality report and the frozen specification.

Every heavy stage (walk-forward, evaluation, risk, DQ report) and every holdout primitive (seal, log, unlock) is
stubbed, and all paths point into ``tmp_path``: these tests never read ``data/`` and never unlock the holdout.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import yaml

from volrisk import config as C
from volrisk import holdout, io, pipeline

KEYS = ["asset", "horizon", "model", "origin"]


def _fc(n: int = 6, F: float = 0.02, split: str = "dev", start: str = "2024-01-01", model: str = "HAR") -> pd.DataFrame:
    """Small forecasts frame (EURUSD-1d-sized F, the case where an absolute tolerance would dominate)."""
    origins = pd.date_range(start, periods=n, freq="D").astype("datetime64[ms]")
    return pd.DataFrame({"asset": "EURUSD", "horizon": "1d", "model": model, "origin": origins, "n_t": 1,
                         "F": F * (1.0 + 0.1 * np.arange(n)), "split": split})


def _targets(fc: pd.DataFrame) -> pd.DataFrame:
    t = fc.drop_duplicates(["asset", "horizon", "origin"])[["asset", "horizon", "origin", "n_t", "split"]].copy()
    return t.assign(window_end=t["origin"] + pd.Timedelta(days=1), y=1.0, ybar=1.0)


@pytest.fixture
def results(tmp_path, monkeypatch):
    """Redirect results/reports/seal paths into tmp_path and write a sealed dev forecasts file."""
    res = tmp_path / "data" / "results"
    res.mkdir(parents=True)
    monkeypatch.setattr(C, "RESULTS", res)
    monkeypatch.setattr(C, "REPORTS", tmp_path / "reports")
    monkeypatch.setattr(C, "FIGURES", tmp_path / "reports" / "figures")
    monkeypatch.setattr(C, "TABLES", tmp_path / "reports" / "tables")
    monkeypatch.setattr(io, "FORECASTS", res / "forecasts.parquet")
    monkeypatch.setattr(holdout, "SEALED", tmp_path / "SEALED.json")
    monkeypatch.delenv(io._UNLOCK_ENV, raising=False)
    sealed = _fc()
    sealed.to_parquet(res / "forecasts.parquet", index=False)
    return res


# ------------------------------------------------------------------------------------------ reproduction check
def test_reproduction_check_has_no_absolute_slack(results):
    sealed = pd.read_parquet(io.FORECASTS)
    drifted = sealed.assign(F=sealed["F"] * (1 + 1e-7))
    # the previous check, np.allclose(..., rtol=1e-9), kept numpy's atol=1e-8 and accepted this 1e-7 drift
    assert np.allclose(sealed["F"], drifted["F"], rtol=1e-9)
    with pytest.raises(RuntimeError, match=r"6 of 6 rows differ"):
        pipeline.assert_reproduces_dev(drifted)
    assert pipeline.assert_reproduces_dev(sealed.assign(F=sealed["F"] * (1 + 1e-12))) == 6


def test_reproduction_check_needs_identical_dev_rows(results):
    sealed = pd.read_parquet(io.FORECASTS)
    assert pipeline.assert_reproduces_dev(sealed) == 6
    # origin precision (ms on disk vs ns in memory) does not matter; holdout / dropped rows are not compared
    extra = pd.concat([_fc(3, split="holdout", start="2026-01-01"), _fc(2, split="dropped", start="2025-09-29")])
    rerun = pd.concat([sealed.assign(origin=sealed["origin"].astype("datetime64[ns]")), extra], ignore_index=True)
    assert pipeline.assert_reproduces_dev(rerun) == 6
    with pytest.raises(RuntimeError, match=r"1 of 6 rows differ \(1 missing"):
        pipeline.assert_reproduces_dev(sealed.iloc[1:])
    with pytest.raises(RuntimeError, match=r"1 of 7 rows differ \(1 missing"):
        pipeline.assert_reproduces_dev(pd.concat([sealed, _fc(1, model="LGBM")], ignore_index=True))
    with pytest.raises(RuntimeError, match="duplicate"):
        pipeline.assert_reproduces_dev(pd.concat([sealed, sealed.iloc[:1]], ignore_index=True))
    with pytest.raises(RuntimeError, match="no development forecasts"):
        pipeline.assert_reproduces_dev(sealed.iloc[:0], sealed=sealed.iloc[:0])


# ------------------------------------------------------------------------------------------ forecast stage
class _InlineExecutor:
    def __init__(self, max_workers=None):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, items):
        return [fn(x) for x in items]


def test_compute_forecasts_writes_nothing_and_wires_the_iv_calibration_window(results, monkeypatch):
    import volrisk.evaluation.iv as iv_mod
    import volrisk.models.combo as combo_mod
    import volrisk.targets as targets_mod

    model_fc = _fc().drop(columns="split")
    calls = {}

    def fake_load_daily(include_holdout=False):
        calls["include_holdout"] = include_holdout
        return pd.DataFrame({"asset": ["EURUSD"] * 3, "session_date": pd.date_range("2024-01-01", periods=3)})

    def fake_iv(tg, iv, cal_min=250, cal_max=1000):
        calls["iv"] = (cal_min, cal_max)
        return model_fc.iloc[:0].assign(model="IV-cal")

    cfg = {**C.load(), "evaluation": {**C.load()["evaluation"], "iv_cal_min": 300, "iv_cal_max": 900}}
    monkeypatch.setattr(C, "load", lambda: cfg)
    monkeypatch.setattr(io, "load_daily", fake_load_daily)
    monkeypatch.setattr(io, "load_implied", lambda: pd.DataFrame({"asset": ["EURUSD"], "origin": [pd.Timestamp(0)]}))
    monkeypatch.setattr(targets_mod, "build_all_targets", lambda daily, last: _targets(_fc()))
    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", _InlineExecutor)
    monkeypatch.setattr(pipeline, "_forecast_task", lambda args: model_fc)
    monkeypatch.setattr(combo_mod, "combine", lambda fc: fc.iloc[:0])
    monkeypatch.setattr(iv_mod, "iv_benchmarks", fake_iv)

    fc, tg = pipeline.compute_forecasts(workers=1, models=("HAR",))
    assert calls == {"include_holdout": False, "iv": (300, 900)}  # config values, not iv.py's defaults
    assert list(results.iterdir()) == [results / "forecasts.parquet"]  # nothing new written
    assert (fc["split"] == "dev").all() and len(fc) == 6
    (results / "forecasts.parquet").unlink()
    pipeline.stage_forecast(workers=1, models=("HAR",))
    assert {p.name for p in results.iterdir()} == {"forecasts.parquet", "targets.parquet"}
    pd.testing.assert_frame_equal(pd.read_parquet(results / "forecasts.parquet"), fc, check_dtype=False)


# ------------------------------------------------------------------------------------------ freeze
@pytest.fixture
def freeze_env(results, monkeypatch):
    log: list = []
    monkeypatch.setattr(holdout, "current_hashes", lambda: {"code": "c", "dev_forecasts": "f"})
    monkeypatch.setattr(holdout, "freeze", lambda spec, **kw: log.append(("freeze", spec, kw)) or {"created_utc": "t"})
    monkeypatch.setattr(pipeline, "frozen_spec", lambda: {"spec": 1})
    return log


def test_freeze_refuses_stale_dev_results(freeze_env, monkeypatch):
    sealed = pd.read_parquet(io.FORECASTS)
    stale = sealed.assign(F=np.where(sealed.index == 2, sealed["F"] * 1.001, sealed["F"]))  # e.g. a config edit
    monkeypatch.setattr(pipeline, "compute_forecasts", lambda include_holdout=False, workers=10: (stale, None))
    with pytest.raises(RuntimeError, match=r"1 of 6 rows differ.*\('EURUSD', 'HAR'\): 1"):
        pipeline.stage_freeze(workers=1)
    assert freeze_env == []  # nothing sealed


def test_freeze_seals_when_the_dev_forecasts_reproduce(freeze_env, monkeypatch):
    seen = []

    def fake_compute(include_holdout=False, workers=10):
        seen.append(include_holdout)
        return pd.read_parquet(io.FORECASTS), None

    monkeypatch.setattr(pipeline, "compute_forecasts", fake_compute)
    pipeline.stage_freeze(workers=1)
    pipeline.stage_freeze(workers=1, refreeze_reason="audit fix")
    assert seen == [False, False]  # dev-only pre-flight, never the holdout
    assert freeze_env == [("freeze", {"spec": 1}, {}), ("freeze", {"spec": 1}, {"refreeze_reason": "audit fix"})]


def test_freeze_refuses_files_changed_during_the_preflight(freeze_env, monkeypatch):
    hashes = iter([{"code": "a"}, {"code": "b"}])
    monkeypatch.setattr(holdout, "current_hashes", lambda: next(hashes))
    monkeypatch.setattr(pipeline, "compute_forecasts",
                        lambda include_holdout=False, workers=10: (pd.read_parquet(io.FORECASTS), None))
    with pytest.raises(holdout.SealError, match="changed during the freeze pre-flight"):
        pipeline.stage_freeze(workers=1)
    assert freeze_env == []


def test_freeze_refuses_an_existing_seal_before_any_work(freeze_env, monkeypatch):
    holdout.SEALED.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pipeline, "compute_forecasts", lambda *a, **k: pytest.fail("pre-flight must not run"))
    with pytest.raises(holdout.SealError, match="already exists"):
        pipeline.stage_freeze(workers=1)


# ------------------------------------------------------------------------------------------ holdout run
@pytest.fixture
def hold(results, monkeypatch):
    """Stubbed holdout primitives and stages; ``state`` controls them and records the call order."""
    sealed = pd.read_parquet(io.FORECASTS)
    holdout_rows = _fc(4, split="holdout", start="2025-10-01")
    state = {"calls": [], "openings": [], "seal_bad": [], "pre": sealed, "post": pd.concat([sealed, holdout_rows]),
             "quality_error": None}
    final = pipeline.results_dir(True)

    def fake_compute(include_holdout=False, workers=10):
        state["calls"].append(("compute", include_holdout))
        fc = state["post"] if include_holdout else state["pre"]
        return fc.reset_index(drop=True), _targets(fc)

    def fake_unlock(reason=None):
        state["calls"].append(("unlock", reason))
        return {"utc": "now", "n_previous": len(state["openings"])}

    def fake_stage(kind):
        def run(include_holdout=False, out=None):
            assert include_holdout and out is not None and out != final  # built in staging, not in place
            assert (out / "forecasts.parquet").exists() and (out / "targets.parquet").exists()
            state["calls"].append((kind, out.name.startswith("holdout_staging_")))
            (out / f"{kind}_table.parquet").write_bytes(b"new")
        return run

    def fake_quality(assets=C.ASSETS, include_holdout=False):
        state["calls"].append(("quality", include_holdout))
        if state["quality_error"]:
            raise state["quality_error"]

    monkeypatch.setattr(holdout, "verify_seal", lambda: state["seal_bad"])
    monkeypatch.setattr(holdout, "openings", lambda: state["openings"])
    monkeypatch.setattr(holdout, "unlock", fake_unlock)
    monkeypatch.setattr(pipeline, "compute_forecasts", fake_compute)
    monkeypatch.setattr(pipeline, "stage_evaluate", fake_stage("eval"))
    monkeypatch.setattr(pipeline, "stage_risk", fake_stage("risk"))
    monkeypatch.setattr(pipeline, "stage_quality", fake_quality)
    return state


def _previous_run(final):
    final.mkdir(parents=True)
    (final / "eval_leaderboard.parquet").write_bytes(b"old")
    (final / "forecasts.parquet").write_bytes(b"old")


def _staging_dirs(results):
    return [p for p in results.iterdir() if p.name.startswith("holdout_staging_")]


def test_holdout_preflight_refuses_stale_dev_results_before_the_opening(hold, results):
    pre = hold["pre"]
    hold["pre"] = pre.assign(F=pre["F"] * (1 + 1e-6))  # e.g. lgbm learning_rate edited after `forecast`
    with pytest.raises(RuntimeError, match="did not reproduce"):
        pipeline.stage_holdout(workers=1)
    assert hold["calls"] == [("compute", False)]  # never unlocked, nothing logged, holdout never read
    assert not pipeline.results_dir(True).exists() and not _staging_dirs(results)


def test_holdout_checks_seal_and_openings_before_the_preflight(hold, results):
    hold["seal_bad"] = ["code"]
    with pytest.raises(holdout.SealError, match="seal mismatch"):
        pipeline.stage_holdout(workers=1)
    hold["seal_bad"] = []
    hold["openings"] = [{"utc": "earlier"}]
    with pytest.raises(holdout.SealError, match="rerun-reason"):
        pipeline.stage_holdout(workers=1)
    hold["openings"] = []
    _previous_run(pipeline.results_dir(True))  # results without a log entry count as an opening too
    with pytest.raises(holdout.SealError, match="rerun-reason"):
        pipeline.stage_holdout(workers=1)
    assert hold["calls"] == []


def test_holdout_lookahead_failure_writes_nothing_and_keeps_earlier_results(hold, results):
    final = pipeline.results_dir(True)
    _previous_run(final)
    hold["openings"] = [{"utc": "earlier"}]
    post = hold["post"]
    hold["post"] = post.assign(F=np.where(post["split"] == "dev", post["F"] * 1.01, post["F"]))  # look-ahead
    with pytest.raises(RuntimeError, match="did not reproduce"):
        pipeline.stage_holdout(rerun_reason="retry", workers=1)
    assert hold["calls"] == [("compute", False), ("unlock", "retry"), ("compute", True)]
    assert sorted(p.name for p in final.iterdir()) == ["eval_leaderboard.parquet", "forecasts.parquet"]
    assert (final / "forecasts.parquet").read_bytes() == b"old"  # previously: overwritten by the rejected run
    assert not _staging_dirs(results)


def test_holdout_success_publishes_a_complete_run(hold, results):
    pipeline.stage_holdout(workers=1)
    assert hold["calls"] == [("compute", False), ("unlock", None), ("compute", True), ("eval", True), ("risk", True),
                             ("quality", True)]
    final = pipeline.results_dir(True)
    assert sorted(p.name for p in final.iterdir()) == ["eval_table.parquet", "forecasts.parquet",
                                                       "risk_table.parquet", "targets.parquet"]
    fc = pd.read_parquet(final / "forecasts.parquet")
    assert set(fc["split"]) == {"dev", "holdout"}
    assert not _staging_dirs(results)
    assert pd.read_parquet(io.FORECASTS).equals(hold["pre"])  # sealed dev file untouched


def test_holdout_reopening_never_mixes_runs(hold, results):
    final = pipeline.results_dir(True)
    _previous_run(final)
    hold["openings"] = [{"utc": "earlier"}]
    pipeline.stage_holdout(rerun_reason="logged", workers=1)
    archived = results / "holdout_prev_1"
    assert sorted(p.name for p in archived.iterdir()) == ["eval_leaderboard.parquet", "forecasts.parquet"]
    assert "eval_leaderboard.parquet" not in {p.name for p in final.iterdir()}  # no stale table next to new ones
    assert (final / "eval_table.parquet").read_bytes() == b"new"


def test_holdout_dq_failure_does_not_lose_the_results(hold, results, caplog):
    hold["quality_error"] = ValueError("figure backend")
    with caplog.at_level(logging.ERROR, logger="volrisk"):
        pipeline.stage_holdout(workers=1)
    assert "holdout data-quality report failed" in caplog.text
    assert (pipeline.results_dir(True) / "risk_table.parquet").exists()


def test_holdout_stage_failure_after_the_check_leaves_the_published_dir_alone(hold, results, monkeypatch):
    final = pipeline.results_dir(True)
    _previous_run(final)
    hold["openings"] = [{"utc": "earlier"}]

    def boom(include_holdout=False, out=None):
        raise MemoryError("risk")

    monkeypatch.setattr(pipeline, "stage_risk", boom)
    with pytest.raises(MemoryError):
        pipeline.stage_holdout(rerun_reason="logged", workers=1)
    assert (final / "forecasts.parquet").read_bytes() == b"old"
    assert len(_staging_dirs(results)) == 1  # the partial run stays aside for inspection


# ------------------------------------------------------------------------------------------ DQ report
def test_holdout_dq_report_has_its_own_files(results, monkeypatch):
    import volrisk.quality as quality

    seen = []
    monkeypatch.setattr(quality, "build_report", lambda **kw: seen.append(kw) or "x")
    pipeline.stage_quality(("BTC",))
    pipeline.stage_quality(("BTC",), include_holdout=True)
    assert seen[0] == {"assets": ("BTC",)}  # dev report: quality's own dev defaults
    assert seen[1]["until"] == C.data_end()
    assert seen[1]["out_md"] == C.REPORTS / "data_quality_holdout.md"
    assert seen[1]["fig_dir"] != C.FIGURES and seen[1]["table_dir"] != C.TABLES


def test_holdout_dq_report_stays_sealed_without_the_unlock(results):
    with pytest.raises(io.HoldoutSealedError):
        pipeline.stage_quality(("BTC",), include_holdout=True)
    assert not C.REPORTS.exists()


# ------------------------------------------------------------------------------------------ frozen spec
def test_frozen_spec_records_the_features(results):
    from volrisk.models.features import ml_features

    (results / "har_star.json").write_text('{"BTC": "HARQ"}', encoding="utf-8")
    spec = pipeline.frozen_spec()
    feats = spec["features"]
    assert set(feats) == set(C.ASSETS)
    for a in C.ASSETS:
        assert set(feats[a]) == set(pipeline.FORECAST_MODELS)
    assert feats["SPX"]["HAR"] == ["rv_d", "rv_w", "rv_m", "gap2"]
    assert feats["BTC"]["HAR"] == ["rv_d", "rv_w", "rv_m"]
    assert feats["BTC"]["HARQ"] == ["rv_d", "rq_rv", "rv_w", "rv_m"]
    probe = pd.DataFrame({c: [1.0] for c in ("rv", "bv", "c", "j", "rs_pos", "rs_neg", "rq", "r_cc", "gap")})
    probe["session_date"] = pd.Timestamp("2020-01-06")
    assert feats["EURUSD"]["LGBM"] == list(ml_features(probe, "EURUSD").columns)
    assert "gap2" in feats["EURUSD"]["MLP"] and "dow" not in feats["EURUSD"]["MLP"]
    assert "dow_6" in feats["BTC"]["MLP"] and "gap2" not in feats["BTC"]["MLP"]
    assert not any("iv" in c for m in feats.values() for cols in m.values() for c in cols)  # IV never a feature
    assert yaml.safe_load(yaml.safe_dump(spec, sort_keys=False))["features"] == feats
