"""Pipeline stages (SPEC §12). Each stage reads the previous stage's parquet and writes its own."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from volrisk import config as C
from volrisk import io

log = logging.getLogger("volrisk")

FORECAST_MODELS = ("RW", "EWMA", "GARCH", "GJR", "HAR", "HAR-CJ", "SHAR", "HARQ", "LGBM", "MLP")
HAR_FAMILY = ("HAR", "HAR-CJ", "SHAR", "HARQ")
REPRO_RTOL = 1e-9  # SPEC §11: the dev forecasts must be reproduced to rtol 1e-9 (no absolute slack)


def results_dir(include_holdout: bool) -> Path:
    """Development results live in data/results/; the one-time holdout run writes to data/results/holdout/."""
    return C.RESULTS / "holdout" if include_holdout else C.RESULTS


# --------------------------------------------------------------------------------------------- data stages
def stage_download(assets=C.ASSETS) -> None:
    from volrisk.data import binance, dukascopy, implied

    for a in assets:
        src = C.asset(a).source
        t0 = time.time()
        summary = binance.download(a) if src == "binance" else dukascopy.download(a)
        log.info("download %s: %s (%.0fs)", a, summary, time.time() - t0)
    log.info("implied: %s", implied.download_all())


def stage_bronze(assets=C.ASSETS) -> None:
    from volrisk.data import binance, dukascopy

    for a in assets:
        mod = binance if C.asset(a).source == "binance" else dukascopy
        log.info("bronze %s: %s", a, mod.build_bronze(a))


def stage_bars(assets=C.ASSETS) -> None:
    from volrisk import bars

    for a in assets:
        log.info("bars %s: %s", a, bars.build_bars(a))


def stage_measures() -> None:
    from volrisk import measures
    from volrisk.data import implied

    log.info("gold: %s", measures.build_gold())
    df = implied.build_implied()
    log.info("implied: %d rows", len(df))


# --------------------------------------------------------------------------------------------- forecasting
def _model(name: str):
    from volrisk.models import garch, har, ml, simple

    return {
        "RW": simple.RW,
        "EWMA": simple.EWMA,
        "GARCH": garch.GARCH,
        "GJR": garch.GJR,
        "HAR": har.HAR,
        "HAR-CJ": har.HARCJ,
        "SHAR": har.SHAR,
        "HARQ": har.HARQ,
        "LGBM": ml.LGBM,
        "MLP": ml.MLP,
    }[name]()


def _forecast_task(args: tuple[str, str, pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    asset, model_name, daily, targets = args
    t0 = time.time()
    model = _model(model_name)
    out = []
    for h in C.HORIZONS:
        tg = targets[(targets["horizon"] == h)].reset_index(drop=True)
        out.append(model.forecast(daily, tg, asset, h))
    df = pd.concat(out, ignore_index=True)
    logging.getLogger("volrisk").info("%s %s: %d forecasts (%.0fs)", asset, model_name, len(df), time.time() - t0)
    return df


def compute_forecasts(
    include_holdout: bool = False, workers: int = 10, models=FORECAST_MODELS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward forecasts of every model plus COMBO and the IV benchmarks, and the §6 targets — in memory,
    nothing is written. ``include_holdout`` needs the holdout unlocked (``io.load_daily``)."""
    from volrisk.evaluation.iv import iv_benchmarks
    from volrisk.models.combo import combine
    from volrisk.targets import build_all_targets

    daily_all = io.load_daily(include_holdout=include_holdout)
    last_date = C.data_end() if include_holdout else C.dev_end()
    targets = build_all_targets(daily_all, last_date)

    assets = [a for a in C.ASSETS if (daily_all["asset"] == a).any()]
    tasks = []
    for a in assets:
        d = daily_all[daily_all["asset"] == a].reset_index(drop=True)
        tg = targets[targets["asset"] == a].reset_index(drop=True)
        tasks += [(a, m, d, tg) for m in models]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parts = list(ex.map(_forecast_task, tasks))
    fc = pd.concat(parts, ignore_index=True)

    fc = pd.concat([fc, combine(fc)], ignore_index=True)
    implied = io.load_implied()
    ev = C.load()["evaluation"]
    iv_parts = []
    for a in assets:
        tg = targets[(targets["asset"] == a) & (targets["horizon"] == "1m")].reset_index(drop=True)
        iv = implied[implied["asset"] == a]
        if len(iv):
            iv_parts.append(iv_benchmarks(tg, iv, cal_min=int(ev["iv_cal_min"]), cal_max=int(ev["iv_cal_max"])))
    if iv_parts:
        fc = pd.concat([fc, *iv_parts], ignore_index=True)

    fc = fc.merge(targets[["asset", "horizon", "origin", "split"]], on=["asset", "horizon", "origin"], how="left")
    return fc, targets


def _write_forecasts(fc: pd.DataFrame, targets: pd.DataFrame, out: Path) -> None:
    io.write_parquet(fc, out / "forecasts.parquet")
    io.write_parquet(targets, out / "targets.parquet")
    log.info("forecasts: %d rows -> %s", len(fc), out)


def stage_forecast(include_holdout: bool = False, workers: int = 10, models=FORECAST_MODELS) -> pd.DataFrame:
    fc, targets = compute_forecasts(include_holdout, workers, models)
    _write_forecasts(fc, targets, results_dir(include_holdout))
    return fc


def assert_reproduces_dev(fc: pd.DataFrame, sealed: pd.DataFrame | None = None) -> int:
    """SPEC §11: the ``dev`` rows of ``fc`` must reproduce the sealed dev forecasts (default ``io.FORECASTS``).

    Same ``(asset, horizon, model, origin)`` keys on both sides and ``F`` equal to ``rtol = 1e-9`` with
    ``atol = 0`` (numpy's default ``atol = 1e-8`` would accept ~1e-7 relative differences on small forecasts
    such as EURUSD 1d). Returns the number of rows compared; raises ``RuntimeError`` otherwise.
    """
    import numpy as np

    keys = ["asset", "horizon", "model", "origin"]

    def dev_rows(df: pd.DataFrame, name: str) -> pd.DataFrame:
        d = df.loc[df["split"] == "dev", [*keys, "F"]].copy()
        d["origin"] = pd.to_datetime(d["origin"]).astype("datetime64[ns]")
        if d.duplicated(keys).any():
            raise RuntimeError(f"{name} dev forecasts contain duplicate {keys} rows")
        return d

    sealed = pd.read_parquet(io.FORECASTS) if sealed is None else sealed
    m = dev_rows(sealed, "sealed").merge(dev_rows(fc, "re-run"), on=keys, how="outer", suffixes=("_sealed", "_rerun"))
    if m.empty:
        raise RuntimeError("no development forecasts to compare — cannot verify reproducibility")
    ok = np.isclose(m["F_sealed"].to_numpy(dtype=float), m["F_rerun"].to_numpy(dtype=float),
                    rtol=REPRO_RTOL, atol=0.0)  # NaN (row missing on one side) compares unequal
    if not ok.all():
        bad = m.loc[~ok]
        missing = int(bad[["F_sealed", "F_rerun"]].isna().any(axis=1).sum())
        by_model = bad.groupby(["asset", "model"]).size().to_dict()
        raise RuntimeError(
            f"the re-run did not reproduce the sealed development forecasts: {len(bad)} of {len(m)} rows differ "
            f"({missing} missing on one side; rtol {REPRO_RTOL:g}, atol 0); by (asset, model): {by_model}"
        )
    return len(m)


# --------------------------------------------------------------------------------------------- evaluation
def _write_tables(tables: dict[str, pd.DataFrame], out: Path, prefix: str) -> None:
    """Write each table as ``{prefix}_{name}.parquet``; tables without columns (e.g. the holdout IV MCS, which
    SPEC §8 does not run) are skipped, because a zero-column parquet cannot be read back by DuckDB."""
    for k, v in tables.items():
        if v.shape[1] == 0:
            log.info("%s_%s: empty table, not written", prefix, k)
            continue
        io.write_parquet(v.reset_index(drop=True), out / f"{prefix}_{k}.parquet")


def stage_evaluate(include_holdout: bool = False, out: Path | None = None) -> dict[str, pd.DataFrame]:
    """``out`` (default ``results_dir(include_holdout)``) holds the input forecasts/targets and gets the tables."""
    import json

    from volrisk.evaluation.suite import evaluate, har_star

    out = results_dir(include_holdout) if out is None else Path(out)
    fc = pd.read_parquet(out / "forecasts.parquet")
    tg = pd.read_parquet(out / "targets.parquet")
    mode = "holdout" if include_holdout else "dev"
    tables = evaluate(fc, tg, mode=mode)
    _write_tables(tables, out, "eval")
    if mode == "dev":
        star = har_star(tables["leaderboard"])
        (out / "har_star.json").write_text(json.dumps(star, indent=2), encoding="utf-8")
        log.info("HAR* per asset (dev, 1d): %s", star)
    return tables


def _har_star(include_holdout: bool) -> dict[str, str]:
    import json

    if include_holdout:
        from volrisk.holdout import frozen

        return frozen()["har_star"]
    return json.loads((C.RESULTS / "har_star.json").read_text(encoding="utf-8"))


def stage_risk(include_holdout: bool = False, out: Path | None = None) -> dict[str, pd.DataFrame]:
    """``out`` (default ``results_dir(include_holdout)``) holds the input forecasts and gets the risk tables."""
    from volrisk.risk.suite import build_risk, evaluate_risk, time_in_green

    out = results_dir(include_holdout) if out is None else Path(out)
    fc = pd.read_parquet(out / "forecasts.parquet")
    daily_all = io.load_daily(include_holdout=include_holdout)
    star = _har_star(include_holdout)
    risk = build_risk(daily_all, fc, star)
    io.write_parquet(risk, out / "risk.parquet")
    mode = "holdout" if include_holdout else "dev"
    tables = evaluate_risk(risk, daily_all, fc, mode=mode)
    tables["time_in_zone"] = time_in_green(tables["rolling_zones"], mode)
    _write_tables(tables, out, "risk")
    return tables


def stage_quality(assets=C.ASSETS, include_holdout: bool = False) -> None:
    """Data-quality report (SPEC §10) on dev data. ``include_holdout`` is only possible inside the unlocked holdout
    run (``quality`` raises ``HoldoutSealedError`` otherwise): it covers every session up to ``data_end`` and is
    written to its own files, leaving the dev report untouched."""
    from volrisk import quality

    kw = {}
    if include_holdout:
        kw = {"until": C.data_end(), "out_md": C.REPORTS / "data_quality_holdout.md",
              "fig_dir": C.FIGURES / "holdout", "table_dir": C.TABLES / "holdout"}
    log.info("data-quality report: %s", quality.build_report(assets=tuple(assets), **kw))


def stage_report(include_holdout: bool | None = None) -> None:
    from volrisk import report

    report.build(include_holdout=include_holdout)


# --------------------------------------------------------------------------------------------- holdout
def feature_spec() -> dict[str, dict[str, list[str]]]:
    """Inputs of every forecast model per asset (SPEC §7), taken from the model code itself."""
    from volrisk.models import har
    from volrisk.models.features import ml_features
    from volrisk.models.ml import mlp_design

    probe = pd.DataFrame({c: [1.0] for c in ("rv", "bv", "c", "j", "rs_pos", "rs_neg", "rq", "r_cc", "gap")})
    probe["session_date"] = pd.Timestamp("2020-01-06")
    out: dict[str, dict[str, list[str]]] = {}
    for a in C.ASSETS:
        ml = ml_features(probe, a)
        out[a] = {
            "RW": ["tv"],
            "EWMA": ["r_cc"],
            "GARCH": ["r_cc"],
            "GJR": ["r_cc"],
            **{m: list(har.regressors(m, a)) for m in HAR_FAMILY},
            "LGBM": [str(c) for c in ml.columns],
            "MLP": [str(c) for c in mlp_design(ml).columns],
        }
    return out


def frozen_spec() -> dict:
    """The final specification recorded in config/frozen.yaml at the freeze (SPEC §11)."""
    import json

    cfg = C.load()
    star = json.loads((C.RESULTS / "har_star.json").read_text(encoding="utf-8"))
    return {
        "forecast_models": list(FORECAST_MODELS),
        "combo_members": cfg["models"]["combo_members"],
        "risk_models": ["HS-250", "RiskMetrics", "GJR+FHS", "HAR*+FHS", "COMBO+FHS", "COMBO+Normal"],
        "primary_risk_model": "COMBO+FHS",
        "reference_forecast_model": "HAR",
        "har_star": star,
        "features": feature_spec(),
        "hyperparameters": {"lgbm": cfg["models"]["lgbm"], "mlp": cfg["models"]["mlp"]},
        "walk_forward": cfg["walk_forward"],
        "har_lags": cfg["har_lags"],
        "jump_alpha": cfg["jumps"]["alpha"],
        "min_coverage": cfg["sessions"]["min_coverage"],
        "risk": cfg["risk"],
        "evaluation": cfg["evaluation"],
        "seed": cfg["seed"],
        "dates": {k: str(v) for k, v in cfg["dates"].items()},
        "hypotheses": ["H1", "H2", "H3", "H4", "H5"],
    }


def stage_freeze(workers: int = 10, refreeze_reason: str | None = None) -> None:
    """Freeze and seal (SPEC §11) — only after a dev-only walk-forward (in memory: no holdout data, nothing written)
    reproduces ``data/results/forecasts.parquet`` with the current code and config, so stale dev results are
    refused here instead of being found after the one-time holdout opening. ``refreeze_reason`` is passed to
    ``holdout.freeze`` (needed for a new seal after an opening)."""
    from volrisk import holdout

    if holdout.SEALED.exists():
        raise holdout.SealError(f"{holdout.SEALED.name} already exists — the specification is frozen")
    before = holdout.current_hashes()
    fc, _ = compute_forecasts(include_holdout=False, workers=workers)
    n = assert_reproduces_dev(fc)
    if holdout.current_hashes() != before:
        raise holdout.SealError("code, config, data or results changed during the freeze pre-flight — run it again")
    log.info("freeze pre-flight: the current code and config reproduce the dev forecasts (%d rows)", n)
    kw = {} if refreeze_reason is None else {"refreeze_reason": refreeze_reason}
    seal = holdout.freeze(frozen_spec(), **kw)
    log.info("frozen; SEALED.json written at %s", seal["created_utc"])


def _move_dir(src: Path, dst: Path, lock_timeout: float = 120.0) -> None:
    """Rename a directory; retried while another process (e.g. the dashboard) holds a file in it open (Windows)."""
    deadline = time.monotonic() + lock_timeout
    while True:
        try:
            os.rename(src, dst)
            return
        except PermissionError:
            if time.monotonic() > deadline:
                raise PermissionError(f"cannot move {src} to {dst}: locked by another process") from None
            time.sleep(0.5)


def _publish_holdout(staging: Path, n_previous: int) -> Path | None:
    """Move a complete holdout run from ``staging`` to ``results_dir(True)``. The results of an earlier opening
    are kept, never mixed in: they move to ``holdout_prev_<n_previous>``. Returns that archive path, if any."""
    final = results_dir(True)
    archived = None
    if final.exists():
        archived = final.with_name(f"{final.name}_prev_{n_previous}")
        k = 1
        while archived.exists():
            archived = final.with_name(f"{final.name}_prev_{n_previous}_{k}")
            k += 1
        _move_dir(final, archived)
    _move_dir(staging, final)
    return archived


def stage_holdout(rerun_reason: str | None = None, workers: int = 10) -> None:
    """The one-time holdout run (SPEC §11): verify seal -> log + unlock -> full walk-forward with holdout data ->
    assert the dev forecasts are reproduced (rtol 1e-9) -> evaluate the holdout split.

    Before the opening is spent, a dev-only pre-flight (no holdout data) re-runs the walk-forward and refuses on any
    difference from the sealed dev forecasts (stale results, library or environment drift). The post-unlock check
    with holdout rows included stays: it is the look-ahead test. Outputs are built in a staging directory and only a
    complete run is moved to ``data/results/holdout/``; a failed run writes nothing there and a re-opening never
    mixes its tables with an earlier opening's (those are kept as ``holdout_prev_<n>``).
    """
    from volrisk import holdout

    # fail fast, before the pre-flight; holdout.unlock() repeats both checks authoritatively
    bad = holdout.verify_seal()
    if bad:
        raise holdout.SealError(f"seal mismatch for {bad}: code/config/data changed after the freeze")
    prior = results_dir(True)
    if not rerun_reason and (holdout.openings() or (prior.is_dir() and any(p.is_file() for p in prior.rglob("*")))):
        raise holdout.SealError("the holdout was already opened; pass --rerun-reason to re-open "
                                "(it will be recorded as a deviation)")
    fc_dev, _ = compute_forecasts(include_holdout=False, workers=workers)
    n = assert_reproduces_dev(fc_dev)
    del fc_dev
    log.info("holdout pre-flight: dev forecasts reproduced without holdout data (%d rows)", n)

    entry = holdout.unlock(rerun_reason)
    log.info("HOLDOUT UNLOCKED at %s (opening #%d)", entry["utc"], entry["n_previous"] + 1)
    fc, targets = compute_forecasts(include_holdout=True, workers=workers)
    n = assert_reproduces_dev(fc)
    log.info("dev forecasts reproduced exactly (%d rows, rtol %g, atol 0)", n, REPRO_RTOL)

    C.RESULTS.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="holdout_staging_", dir=C.RESULTS))
    _write_forecasts(fc, targets, staging)
    stage_evaluate(include_holdout=True, out=staging)
    stage_risk(include_holdout=True, out=staging)
    try:  # descriptive H5 inputs; never lose the holdout results over them
        _write_h5_measures(staging)
    except Exception:
        log.exception("H5 holdout measures failed; the report falls back to dev-window measures for H5")
    archived = _publish_holdout(staging, entry["n_previous"])
    if archived is not None:
        log.info("results of the earlier holdout opening moved to %s", archived)
    log.info("holdout results -> %s", results_dir(True))
    try:  # SPEC §10: the DQ report covers holdout data once unsealed; the unlock is process-local, so it is now
        stage_quality(include_holdout=True)
    except Exception:  # descriptive only: never lose the completed holdout results over it
        log.exception("holdout data-quality report failed (holdout results are complete)")
    try:  # tables, figures and verdicts for dev + holdout (presentation only; results are already published)
        stage_report(include_holdout=True)
    except Exception:
        log.exception("report generation failed (holdout results are complete); re-run `python -m volrisk report`")


def _write_h5_measures(out: Path) -> None:
    """H5 jump/semivariance measures over the holdout window (needs the unlocked process), for the report."""
    from volrisk import hypotheses

    m = hypotheses.daily_measures(io.load_daily(include_holdout=True), C.holdout_start(), C.data_end())
    io.write_parquet(m, out / hypotheses.H5_MEASURES_FILE)
