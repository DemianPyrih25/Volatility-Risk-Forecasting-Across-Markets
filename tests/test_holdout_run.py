"""The one-shot holdout path at the stage level (SPEC §11): ``pipeline.stage_freeze`` and ``pipeline.stage_holdout``
on a synthetic BTC gold table that spans the dev/holdout boundary.

What is checked here (the seal primitives themselves are unit-tested in ``test_holdout.py``, the loaders in
``test_io.py``):

- the full flow dev forecast -> freeze -> holdout run reproduces the sealed dev forecasts with holdout rows appended
  (the look-ahead test), forecasts every model at holdout origins and publishes only a complete run;
- a dev-forecast mismatch after the unlock aborts before anything is published (the opening stays logged);
- stale dev results are refused by the freeze and by the holdout pre-flight, before the opening is spent;
- a change to any sealed artefact blocks the run before any forecast is computed, with no log entry;
- a second run is refused without a reason; with one it is logged as a deviation and the first results are kept.

Every path the seal, the loaders and the stages touch is redirected to ``tmp_path`` and checked by a guard before
the test body runs. The real ``data/holdout/``, ``SEALED.json``, ``holdout_log.jsonl``, ``config/frozen.yaml``,
``data/results/`` and ``docs/DEVIATIONS.md`` are never read or written. Forecasts run in-process
(``ProcessPoolExecutor`` replaced) with ``W = 200`` so the walk-forward takes seconds; the evaluation, risk and
data-quality stages are replaced by recorders (they are tested in their own modules).
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import polars as pl
import pytest

from volrisk import config as C
from volrisk import holdout as H
from volrisk import io, pipeline

W_SMALL = 200
START = date(2024, 9, 1)  # W_SMALL + 30-session lags/windows before the end of the dev period
FAST_MODELS = ("RW", "EWMA", "GJR", "HAR", "HARQ", "LGBM")  # GJR + HARQ + LGBM = the COMBO members
STAR = {"BTC": "HARQ"}
SOURCES = {
    "__init__.py": "",
    "models/__init__.py": "",
    "models/har.py": "def fit(x):\n    return x\n",
    "report.py": "TITLE = 'results'\n",
}

# Real locations, captured at import (before any test patches them): the guard checks they are left alone.
_REAL = {
    "SEALED.json": H.SEALED,
    "holdout_log.jsonl": H.HOLDOUT_LOG,
    "config/frozen.yaml": C.FROZEN_PATH,
    "data/results/holdout": C.RESULTS / "holdout",
}
_REAL_CONFIG_BYTES = C.CONFIG_PATH.read_bytes()  # copied into the sandbox (the seal hashes the copy)

_CACHE: dict[tuple[bool, tuple[str, ...]], tuple[pd.DataFrame, pd.DataFrame]] = {}


class _InlineExecutor:
    """In-process stand-in for ProcessPoolExecutor (patched config values do not reach spawned workers)."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, *iterables):
        return list(map(fn, *iterables))


def _gold(seed: int = 11) -> pl.DataFrame:
    """Gold-like BTC rows (every calendar day) from START to data_end, persistent log-vol with jumps."""
    dates = pd.date_range(START, C.data_end(), freq="D")
    n = len(dates)
    rng = np.random.default_rng(seed)
    h = np.zeros(n)
    for i in range(1, n):
        h[i] = 0.95 * h[i - 1] + 0.25 * rng.standard_normal()
    rv = np.exp(h) * rng.gamma(4.0, 0.25, n)
    j = np.where(rng.random(n) < 0.15, rv * rng.uniform(0.1, 0.5, n), 0.0)
    share = rng.uniform(0.3, 0.7, n)
    return pl.DataFrame(
        {
            "asset": "BTC",
            "session_date": [d.date() for d in dates],
            "rv": rv,
            "bv": rv - j,
            "c": rv - j,
            "j": j,
            "rs_pos": rv * share,
            "rs_neg": rv * (1 - share),
            "rq": rv**2 * rng.uniform(1.0, 3.0, n),
            "gap": np.zeros(n),
            "r_cc": rng.standard_normal(n) * np.sqrt(rv),
            "tv": rv,
        }
    )


def _implied(gold: pl.DataFrame, seed: int = 12) -> pl.DataFrame:
    """DVOL-like implied vol on every BTC session, holdout period included (as in the real sealed table)."""
    rng = np.random.default_rng(seed)
    iv = 50.0 + 5.0 * rng.standard_normal(gold.height)
    return pl.DataFrame(
        {
            "asset": "BTC",
            "origin": gold["session_date"],
            "iv": iv,
            "iv_var_30d": iv**2 * 30 / 365,
            "source": "DVOL",
        }
    )


def _write_tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode())
    return root


def _inside(p: Path, root: Path) -> bool:
    return Path(p).resolve().is_relative_to(root.resolve())


@pytest.fixture
def sb(tmp_path, monkeypatch):
    """Sandboxed project: synthetic gold/implied, tmp seal files, in-process forecasts, recording fakes."""
    real_state = {k: p.exists() for k, p in _REAL.items()}
    C.load()  # populate the config cache from the real config before CONFIG_PATH is redirected

    root = tmp_path / "proj"
    src = _write_tree(root / "src" / "volrisk", SOURCES)
    cfg = root / "config" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_bytes(_REAL_CONFIG_BYTES)
    data = root / "data"
    gold, results = data / "gold", data / "results"
    for d in (gold, data / "holdout", results):
        d.mkdir(parents=True)
    docs = root / "docs" / "DEVIATIONS.md"
    docs.parent.mkdir()
    docs.write_text("# Deviations log\n", encoding="utf-8")

    monkeypatch.setattr(C, "CONFIG_PATH", cfg)
    monkeypatch.setattr(C, "FROZEN_PATH", root / "config" / "frozen.yaml")
    monkeypatch.setattr(C, "RESULTS", results)
    monkeypatch.setattr(C, "REPORTS", root / "reports")
    monkeypatch.setattr(C, "window", lambda: W_SMALL)
    monkeypatch.setattr(io, "DAILY_DEV", gold / "daily.parquet")
    monkeypatch.setattr(io, "DAILY_HOLDOUT", data / "holdout" / "daily.parquet")
    monkeypatch.setattr(io, "IMPLIED", gold / "implied.parquet")
    monkeypatch.setattr(io, "FORECASTS", results / "forecasts.parquet")
    monkeypatch.setattr(io, "RISK", results / "risk.parquet")
    monkeypatch.setattr(io, "_UNLOCKED", False)  # restored to the locked state after the test
    monkeypatch.delenv(io._UNLOCK_ENV, raising=False)
    monkeypatch.setattr(H, "SRC", src)
    monkeypatch.setattr(H, "SEALED", root / "SEALED.json")
    monkeypatch.setattr(H, "HOLDOUT_LOG", root / "holdout_log.jsonl")
    monkeypatch.setattr(H, "DEVIATIONS", docs)
    monkeypatch.setattr(H, "RESULTS_HOLDOUT", results / "holdout")
    monkeypatch.setattr(pipeline, "ProcessPoolExecutor", _InlineExecutor)

    # Guard: nothing the seal or the stages read or write may still point into the project.
    touched = [*H._sealed_files().values(), H.SEALED, H.HOLDOUT_LOG, H.DEVIATIONS, H.SRC, H.RESULTS_HOLDOUT,
               pipeline.results_dir(True), pipeline.results_dir(False), io.DAILY_DEV, io.DAILY_HOLDOUT]
    outside = [str(p) for p in touched if not _inside(p, tmp_path)]
    assert not outside, f"sandbox leak: {outside}"
    assert H.RESULTS_HOLDOUT == pipeline.results_dir(True)
    assert not io.holdout_unlocked()

    g = _gold()
    h0 = C.holdout_start()
    g.filter(pl.col("session_date") < h0).write_parquet(io.DAILY_DEV)
    g.filter(pl.col("session_date") >= h0).write_parquet(io.DAILY_HOLDOUT)
    _implied(g).write_parquet(io.IMPLIED)
    (results / "har_star.json").write_text(json.dumps(STAR), encoding="utf-8")

    ns = SimpleNamespace(root=root, src=src, results=results, models=FAST_MODELS, fresh=False, hooks={},
                         compute_calls=[], stage_calls=[])
    real_compute = pipeline.compute_forecasts

    def compute(include_holdout=False, workers=10, models=None):
        """The real walk-forward on ns.models (cached per mode unless ns.fresh), then an optional fault hook."""
        unlocked = io.holdout_unlocked()
        ns.compute_calls.append((include_holdout, unlocked))
        if include_holdout and not unlocked:
            raise io.HoldoutSealedError("holdout requested in a locked process")
        key = (bool(include_holdout), tuple(ns.models))
        if ns.fresh or key not in _CACHE:
            _CACHE[key] = real_compute(include_holdout=include_holdout, workers=1, models=ns.models)
        fc, tg = (x.copy() for x in _CACHE[key])
        hook = ns.hooks.get(bool(include_holdout))
        return (hook(fc) if hook else fc), tg

    def fake_evaluate(include_holdout=False, out=None, **kw):
        ns.stage_calls.append(("evaluate", include_holdout, io.holdout_unlocked()))
        out = pipeline.results_dir(include_holdout) if out is None else Path(out)
        fc = pd.read_parquet(out / "forecasts.parquet")
        n = fc.groupby(["split", "model"]).size().rename("n").reset_index()
        io.write_parquet(n, out / "eval_leaderboard.parquet")

    def fake_risk(include_holdout=False, out=None, **kw):
        ns.stage_calls.append(("risk", include_holdout, io.holdout_unlocked()))
        out = pipeline.results_dir(include_holdout) if out is None else Path(out)
        io.write_parquet(pd.DataFrame({"asset": ["BTC"], "har_star": [pipeline._har_star(include_holdout)["BTC"]]}),
                         out / "risk_risk_leaderboard.parquet")

    monkeypatch.setattr(pipeline, "compute_forecasts", compute)
    monkeypatch.setattr(pipeline, "stage_evaluate", fake_evaluate)
    monkeypatch.setattr(pipeline, "stage_risk", fake_risk)
    monkeypatch.setattr(pipeline, "stage_quality", lambda *a, **k: ns.stage_calls.append(("quality", k)))
    yield ns

    for k, p in _REAL.items():
        assert p.exists() == real_state[k], f"the test touched the real {k}"


def _dev_run_and_freeze(sb) -> pd.DataFrame:
    """Dev forecast stage (writes the dev results), dummy dev risk table, then the freeze stage."""
    fc = pipeline.stage_forecast(include_holdout=False, workers=1, models=sb.models)
    io.write_parquet(pd.DataFrame({"asset": ["BTC"], "var99": [1.0]}), io.RISK)
    pipeline.stage_freeze(workers=1)
    return fc


def _dev(fc: pd.DataFrame) -> pd.DataFrame:
    keys = ["asset", "horizon", "model", "origin"]
    d = fc.loc[fc["split"] == "dev", [*keys, "F"]].copy()
    d["origin"] = pd.to_datetime(d["origin"])
    return d.sort_values(keys).reset_index(drop=True)


def _staging_dirs(sb) -> list[Path]:
    return [p for p in sb.results.iterdir() if p.name.startswith("holdout_staging")]


# ------------------------------------------------------------------------------------------------ full flow
@pytest.mark.parametrize(
    "models",
    [pytest.param(FAST_MODELS, id="fast-models"),
     pytest.param(pipeline.FORECAST_MODELS, marks=pytest.mark.slow, id="all-models")],
)
def test_holdout_run_reproduces_dev_and_publishes_a_complete_run(sb, models):
    sb.models, sb.fresh = tuple(models), True  # every walk-forward below is computed from scratch
    dev_fc = _dev_run_and_freeze(sb)

    # dev run: no holdout origin, no holdout-period IV (the loader hides it in a locked process)
    assert set(dev_fc["split"].dropna()) <= {"dev", "dropped"}
    assert pd.to_datetime(dev_fc["origin"]).max().date() <= C.dev_end()
    assert {"COMBO", "IV", "IV-cal"} <= set(dev_fc.loc[dev_fc["split"] == "dev", "model"])
    assert not any(inc for inc, _ in sb.compute_calls)  # dev stage and freeze pre-flight never ask for holdout

    # freeze: frozen spec carries the HAR* choice and the model list; nothing changed since
    assert H.SEALED.exists() and H.verify_seal() == []
    spec = H.frozen()
    assert spec["har_star"] == STAR and spec["forecast_models"] == list(pipeline.FORECAST_MODELS)

    pipeline.stage_holdout(workers=1)

    # holdout data was requested, and only after the unlock
    assert any(inc for inc, _ in sb.compute_calls) and all(unl for inc, unl in sb.compute_calls if inc)
    assert len(H.openings()) == 1 and H.openings()[0]["rerun_reason"] is None
    assert [c[:2] for c in sb.stage_calls if c[0] != "quality"] == [("evaluate", True), ("risk", True)]

    # published run: dev rows reproduce the sealed dev forecasts exactly; every model forecasts the holdout
    out = pipeline.results_dir(True)
    hfc = pd.read_parquet(out / "forecasts.parquet")
    sealed = pd.read_parquet(io.FORECASTS)
    a, b = _dev(sealed), _dev(hfc)
    pd.testing.assert_frame_equal(a[["asset", "horizon", "model", "origin"]], b[["asset", "horizon", "model", "origin"]])
    np.testing.assert_allclose(b["F"], a["F"], rtol=1e-9, atol=0)
    hold = hfc[hfc["split"] == "holdout"]
    want = {(m, h) for m in [*models, "COMBO"] for h in C.HORIZONS} | {("IV", "1m"), ("IV-cal", "1m")}
    assert set(zip(hold["model"], hold["horizon"])) == want
    assert (hold["F"] > 0).all()
    assert pd.to_datetime(hold["origin"]).min().date() >= C.dev_end()
    tg = pd.read_parquet(out / "targets.parquet")
    th = tg[tg["split"] == "holdout"]
    assert pd.to_datetime(th["origin"]).min().date() >= C.dev_end()
    assert pd.to_datetime(th["window_end"]).max().date() <= C.data_end()
    assert pd.to_datetime(tg.loc[tg["split"] == "dev", "window_end"]).max().date() <= C.dev_end()
    assert (out / "eval_leaderboard.parquet").exists() and (out / "risk_risk_leaderboard.parquet").exists()
    assert pd.read_parquet(out / "risk_risk_leaderboard.parquet")["har_star"].tolist() == ["HARQ"]  # frozen HAR*
    assert not _staging_dirs(sb)
    # the holdout run never rewrites a sealed artefact (dev results, gold, config, code)
    assert H.verify_seal() == []


# ------------------------------------------------------------------------------------------------ mismatches
def _perturb(rel: float):
    def hook(fc: pd.DataFrame) -> pd.DataFrame:
        i = fc.index[(fc["split"] == "dev") & (fc["model"] == "HARQ")][-1]
        fc.loc[i, "F"] *= 1 + rel
        return fc

    return hook


def _drop_one_dev_row(fc: pd.DataFrame) -> pd.DataFrame:
    return fc.drop(index=fc.index[(fc["split"] == "dev") & (fc["model"] == "GJR")][0])


@pytest.mark.parametrize("hook", [_perturb(1e-6), _perturb(1e-8), _drop_one_dev_row],
                         ids=["F-changed-1e-6", "F-changed-1e-8", "dev-row-missing"])
def test_dev_mismatch_after_unlock_aborts_before_publishing(sb, hook):
    """A difference that appears only once holdout rows are appended is look-ahead: abort, publish nothing.
    1e-8 relative is above SPEC's rtol 1e-9 (no absolute slack)."""
    _dev_run_and_freeze(sb)
    sb.hooks[True] = hook
    with pytest.raises(RuntimeError):
        pipeline.stage_holdout(workers=1)
    assert len(H.openings()) == 1  # the opening is spent and stays on record
    assert not [c for c in sb.stage_calls if c[0] in ("evaluate", "risk")]
    assert not pipeline.results_dir(True).exists() and not _staging_dirs(sb)
    assert H.verify_seal() == []


def test_stale_dev_results_are_refused_before_the_opening(sb):
    """The holdout pre-flight re-runs the dev walk-forward without holdout data; a difference (stale results,
    library drift) is refused before the opening is logged or any holdout row is read."""
    _dev_run_and_freeze(sb)
    sb.hooks[False] = _perturb(1e-6)
    with pytest.raises(RuntimeError):
        pipeline.stage_holdout(workers=1)
    assert H.openings() == [] and not io.holdout_unlocked()
    assert all(inc is False for inc, _ in sb.compute_calls)
    assert not pipeline.results_dir(True).exists()


def test_freeze_refuses_stale_dev_results(sb):
    pipeline.stage_forecast(include_holdout=False, workers=1, models=sb.models)
    io.write_parquet(pd.DataFrame({"asset": ["BTC"], "var99": [1.0]}), io.RISK)
    sb.hooks[False] = _perturb(1e-6)
    with pytest.raises(RuntimeError):
        pipeline.stage_freeze(workers=1)
    assert not H.SEALED.exists()


# ------------------------------------------------------------------------------------------------ seal
def _tamper(name: str, sb) -> None:
    if name == "new-code-file":
        (sb.src / "models" / "extra.py").write_bytes(b"X = 1\n")
    elif name == "dev_risk-deleted":
        H._sealed_files()["dev_risk"].unlink()
    else:
        p = sb.src / "models" / "har.py" if name == "code" else H._sealed_files()[name]
        p.write_bytes(p.read_bytes() + b"\n# changed after the freeze\n")


@pytest.mark.parametrize("name", ["code", "new-code-file", "config", "frozen", "dev_data", "holdout_data", "implied",
                                  "dev_forecasts", "dev_risk", "dev_risk-deleted"])
def test_changed_sealed_artefact_blocks_the_run_before_any_work(sb, name):
    _dev_run_and_freeze(sb)
    n_calls = len(sb.compute_calls)
    _tamper(name, sb)
    assert H.verify_seal()
    with pytest.raises(H.SealError):
        pipeline.stage_holdout(workers=1)
    assert len(sb.compute_calls) == n_calls  # no walk-forward, not even the dev pre-flight
    assert H.openings() == [] and not H.HOLDOUT_LOG.exists() and not io.holdout_unlocked()
    assert not pipeline.results_dir(True).exists()


def test_holdout_run_without_freeze_is_refused(sb):
    pipeline.stage_forecast(include_holdout=False, workers=1, models=sb.models)
    with pytest.raises(H.SealError):
        pipeline.stage_holdout(workers=1)
    assert not H.HOLDOUT_LOG.exists() and not io.holdout_unlocked()


def test_second_run_needs_a_reason_and_keeps_the_first_results(sb):
    _dev_run_and_freeze(sb)
    pipeline.stage_holdout(workers=1)
    first = pd.read_parquet(pipeline.results_dir(True) / "forecasts.parquet")
    io._UNLOCKED = False  # a new process (restored by the fixture either way)
    n_calls = len(sb.compute_calls)

    with pytest.raises(H.SealError):
        pipeline.stage_holdout(workers=1)
    assert len(sb.compute_calls) == n_calls and len(H.openings()) == 1

    reason = "test-only re-run 7f3c"
    pipeline.stage_holdout(rerun_reason=reason, workers=1)
    log = H.openings()
    assert len(log) == 2 and log[1]["n_previous"] == 1 and log[1]["rerun_reason"] == reason
    assert reason in H.DEVIATIONS.read_text(encoding="utf-8")
    archived = [p for p in sb.results.iterdir() if p.name.startswith("holdout_prev")]
    assert len(archived) == 1
    pd.testing.assert_frame_equal(pd.read_parquet(archived[0] / "forecasts.parquet"), first)
    assert H.verify_seal() == []
