"""Dashboard (SPEC §12): layout, callback functions and the data layer — offline, no server.

Development results are small slices of the real ``data/results`` files when they exist (copied to tmp_path),
otherwise synthetic tables with the same schemas. Holdout results are always synthetic files in tmp_path: the
sealed holdout and ``data/results/holdout/`` are never read here.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from dash import dcc, html

from volrisk import config as C
from volrisk.dashboard import app as A
from volrisk.dashboard.data import ResultsData, annualisation

MODELS = ["RW", "EWMA", "GARCH", "GJR", "HAR", "HAR-CJ", "SHAR", "HARQ", "LGBM", "MLP", "COMBO"]
RISK_MODELS = ["HS-250", "RiskMetrics", "GJR+FHS", "HAR*+FHS", "COMBO+FHS", "COMBO+Normal"]
N_T = {"1d": 1, "1w": 7, "1m": 30}
REAL_FILES = ("forecasts", "targets", "eval_leaderboard", "eval_dm_har", "eval_mcs", "risk",
              "risk_rolling_zones", "risk_risk_leaderboard", "risk_backtests")
SLICE_FROM = pd.Timestamp("2024-07-01")


# --------------------------------------------------------------------------------------------- synthetic results
def _zone(x: int) -> str:
    return "green" if x <= 4 else ("yellow" if x <= 9 else "red")


def _plus(x: int) -> float:
    return {5: 0.40, 6: 0.50, 7: 0.65, 8: 0.75, 9: 0.85}.get(x, 0.0 if x < 5 else 1.0)


def write_synthetic(d: Path, assets=("BTC", "ETH"), split="dev", start="2024-01-01", n=150,
                    iv_assets=("BTC", "ETH"), lead_dev_days=0, seed=7, bt_window=30) -> Path:
    """Results tables with the pipeline's schemas for ``assets`` labelled ``split``. ``lead_dev_days`` prepends
    rows labelled 'dev' (a holdout run's files hold dev and holdout rows).

    The traffic light follows the generators (``risk.suite``) with a ``bt_window``-observation rolling window:
    ``risk_rolling_zones`` has complete windows over the whole series; ``risk_time_in_zone`` (the H4 measure)
    counts windows *ending* in the split; ``risk_risk_leaderboard`` / ``risk_backtests`` (``backtest_table`` on
    the split rows only) count windows lying entirely *inside* the split, NaN / None when there are none. Like
    the pipeline, a holdout run writes its empty IV-MCS table as a zero-column parquet."""
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    origins = pd.date_range(pd.Timestamp(start) - pd.Timedelta(days=lead_dev_days), periods=n + lead_dev_days,
                            freq="D")
    labels = np.where(origins < pd.Timestamp(start), "dev", split)
    fc, tg = [], []
    for a in assets:
        for h, nt in N_T.items():
            y = nt * np.exp(rng.normal(1.0, 0.5, len(origins)))
            tg.append(pd.DataFrame({"asset": a, "horizon": h, "origin": origins,
                                    "window_end": origins + pd.Timedelta(days=nt), "n_t": nt, "y": y,
                                    "ybar": y / nt, "split": labels}))
            ms = MODELS + (["IV", "IV-cal"] if h == "1m" and a in iv_assets else [])
            for m in ms:
                fc.append(pd.DataFrame({"asset": a, "horizon": h, "model": m, "origin": origins, "n_t": nt,
                                        "F": y * np.exp(rng.normal(0, 0.3, len(origins))), "split": labels}))
    pd.concat(fc).to_parquet(d / "forecasts.parquet", index=False)
    pd.concat(tg).to_parquet(d / "targets.parquet", index=False)

    lb, dm, mcs, ci = [], [], [], []
    mcs_h = ("1d", "1w", "1m") if split == "dev" else ("1d", "1w")
    for a in assets:
        for h in N_T:
            base = 0.3
            q = {m: base * (1.0 if m == "HAR" else rng.uniform(0.8, 1.3)) for m in MODELS}
            for m in MODELS:
                lb.append({"asset": a, "horizon": h, "split": split, "model": m, "qlike": q[m], "mse": 100 * q[m],
                           "qlike_ratio": q[m] / q["HAR"], "n": n})
                if m != "HAR":
                    dm.append({"model": m, "ref": "HAR", "mean_diff": q[m] - q["HAR"], "dm": rng.normal(),
                               "dm_hln": rng.normal(), "pvalue": rng.uniform(), "T": n, "maxlags": 1,
                               "kernel": "bartlett", "pvalue_2n": rng.uniform(), "ratio": q[m] / q["HAR"],
                               "asset": a, "horizon": h})
                    ci.append({"asset": a, "horizon": h, "model": m, "ratio": q[m] / q["HAR"],
                               "lo": 0.9 * q[m] / q["HAR"], "hi": 1.1 * q[m] / q["HAR"], "T": n})
                if h in mcs_h:
                    p = 0.9 if m == "HARQ" else float(rng.uniform())
                    mcs.append({"model": m, "pvalue": p, "in_90": p > 0.10, "in_75": p > 0.25, "asset": a,
                                "horizon": h, "T": n})
    pd.DataFrame(lb).to_parquet(d / "eval_leaderboard.parquet", index=False)
    pd.DataFrame(dm).to_parquet(d / "eval_dm_har.parquet", index=False)
    pd.DataFrame(mcs).to_parquet(d / "eval_mcs.parquet", index=False)
    if split == "holdout":
        pd.DataFrame(ci).to_parquet(d / "eval_ratio_ci.parquet", index=False)

    iva = [a for a in assets if a in iv_assets]
    if iva:
        ivl, ivd, ivm, enc = [], [], [], []
        for a in iva:
            for m in [*MODELS, "IV", "IV-cal"]:
                r = 1.0 if m == "HAR" else float(rng.uniform(0.5, 1.4))
                ivl.append({"asset": a, "horizon": "1m", "split": split, "model": m, "qlike": 0.2 * r,
                            "mse": 10 * r, "qlike_ratio": r, "n": n})
                if m != "IV-cal":
                    ivd.append({"model": m, "ref": "IV-cal", "mean_diff": 0.01, "dm": 1.0, "dm_hln": 1.0,
                                "pvalue": float(rng.uniform()), "T": n, "maxlags": 29, "kernel": "uniform",
                                "pvalue_2n": float(rng.uniform()), "ratio": r / 0.6, "asset": a, "horizon": "1m"})
                ivm.append({"model": m, "pvalue": float(rng.uniform()), "in_90": bool(rng.uniform() > 0.5),
                            "in_75": False, "asset": a, "horizon": "1m", "T": n})
                if m not in ("IV", "IV-cal"):
                    enc.append({"asset": a, "model": m, "T": n, "b": 0.9, "c": float(rng.normal(0, 0.2)),
                                "se_c": 0.1, "t_c": 1.0, "p_c": float(rng.uniform()), "c_ivlag": 0.1,
                                "p_c_ivlag": 0.5})
        pd.DataFrame(ivl).to_parquet(d / "eval_iv_leaderboard.parquet", index=False)
        pd.DataFrame(ivd).to_parquet(d / "eval_iv_dm.parquet", index=False)
        pd.DataFrame(enc).to_parquet(d / "eval_encompassing.parquet", index=False)
        if split == "dev":
            pd.DataFrame(ivm).to_parquet(d / "eval_iv_mcs.parquet", index=False)
        else:  # pipeline._write_tables writes evaluate()'s empty holdout iv_mcs frame as-is
            pd.DataFrame().to_parquet(d / "eval_iv_mcs.parquet", index=False)

    risk, zones, rl, bt, tz = [], [], [], [], []
    in_split = labels == split
    idx = np.flatnonzero(in_split)
    complete = np.arange(len(origins)) >= bt_window - 1  # rolling windows exist once bt_window rows are in
    inside = idx[idx >= idx[0] + bt_window - 1]  # windows lying entirely inside the split (backtest_table)
    for a in assets:
        r = rng.standard_t(4, len(origins)) * 2.0
        for m in RISK_MODELS:
            sigma = 2.0 * np.exp(rng.normal(0, 0.1, len(origins)))
            k = 0.9 if m == "RiskMetrics" else 1.0
            risk.append(pd.DataFrame({"asset": a, "model": m, "date": origins, "r_cc": r, "var99": 2.33 * k * sigma,
                                      "var975": 1.96 * k * sigma, "es975": 2.34 * k * sigma, "sigma": sigma,
                                      "har_member": "HARQ" if m == "HAR*+FHS" else None, "split": labels}))
            exc = np.clip(np.cumsum(rng.integers(-1, 2, len(origins))) % 12, 0, 11)
            zarr = np.array([_zone(int(x)) for x in exc])
            zones.append(pd.DataFrame({"date": origins, "exceptions": exc, "zone": zarr,
                                       "plus_factor": [_plus(int(x)) for x in exc], "asset": a,
                                       "model": m})[complete])
            hit = (-r > 2.33 * k * sigma)[in_split]
            bt_green = float(np.mean(zarr[inside] == "green")) if len(inside) else np.nan
            bt_last = str(zarr[inside[-1]]) if len(inside) else None
            ending = in_split & complete
            tz_green = float(np.mean(zarr[ending] == "green"))
            rl.append({"asset": a, "model": m, "n": int(in_split.sum()), "fz0": 2.0 + rng.normal(0, 0.05),
                       "fz0_diff": 0.0 if m == "HS-250" else float(rng.normal(0, 0.05)),
                       "dm_hln": np.nan if m == "HS-250" else float(rng.normal()),
                       "p_dm": np.nan if m == "HS-250" else float(rng.uniform()),
                       "mcs_p": float(rng.uniform()), "in_90": bool(rng.uniform() > 0.4), "in_75": False,
                       "zone_last": bt_last, "green_share": bt_green})
            for level in ("99", "97.5"):
                bt.append({"asset": a, "model": m, "level": level, "x": int(hit.sum()), "T": len(hit),
                           "rate": float(hit.mean()), "p_binom": 0.5, "p_cc_mc": 0.4, "p_dq": 0.3,
                           "zone_full": "green", "green_share": bt_green, "zone_last": bt_last})
            tz.append({"asset": a, "model": m, "green": tz_green, "yellow": 1 - tz_green, "red": 0.0})
    pd.concat(risk).to_parquet(d / "risk.parquet", index=False)
    pd.concat(zones).to_parquet(d / "risk_rolling_zones.parquet", index=False)
    pd.DataFrame(rl).to_parquet(d / "risk_risk_leaderboard.parquet", index=False)
    pd.DataFrame(bt).to_parquet(d / "risk_backtests.parquet", index=False)
    pd.DataFrame(tz).to_parquet(d / "risk_time_in_zone.parquet", index=False)
    (d / "har_star.json").write_text(json.dumps({a: "HARQ" for a in assets}), encoding="utf-8")
    return d


def copy_real_slices(dst: Path) -> Path:
    """Small slices of the real development results (top-level data/results only; never the holdout)."""
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(C.RESULTS.glob("*.parquet")):
        if f.stem == "eval_losses":
            continue
        df = pd.read_parquet(f)
        if f.stem in ("forecasts", "targets"):
            df = df[df["origin"] >= SLICE_FROM]
        elif f.stem in ("risk", "risk_rolling_zones"):
            df = df[df["date"] >= SLICE_FROM]
        df.to_parquet(dst / f.name, index=False)
    if (C.RESULTS / "har_star.json").exists():
        shutil.copy(C.RESULTS / "har_star.json", dst / "har_star.json")
    return dst


def _real_available() -> bool:
    return all((C.RESULTS / f"{t}.parquet").exists() for t in REAL_FILES)


@pytest.fixture
def dev_dir(tmp_path: Path) -> Path:
    if _real_available():
        return copy_real_slices(tmp_path / "results")
    return write_synthetic(tmp_path / "results")


@pytest.fixture
def reports_dir(tmp_path: Path) -> Path:
    p = tmp_path / "reports"
    (p / "tables").mkdir(parents=True)
    return p


@pytest.fixture
def store(dev_dir: Path, reports_dir: Path) -> ResultsData:
    return ResultsData(dev_dir, reports_dir)


@pytest.fixture
def holdout_store(tmp_path: Path, reports_dir: Path) -> ResultsData:
    """Synthetic dev + holdout results (BTC with implied vol, EURUSD without) in tmp_path."""
    dev = write_synthetic(tmp_path / "hres", assets=("BTC", "EURUSD"), iv_assets=("BTC", "EURUSD"))
    write_synthetic(dev / "holdout", assets=("BTC", "EURUSD"), split="holdout", start="2025-10-01", n=120,
                    iv_assets=("BTC",), lead_dev_days=40, seed=11)
    return ResultsData(dev, reports_dir)


# --------------------------------------------------------------------------------------------- helpers
def _all(component, kind):
    return [c for c in component._traverse() if isinstance(c, kind)]


def _text(c) -> str:
    if c is None:
        return ""
    if isinstance(c, (str, int, float)):
        return str(c)
    if isinstance(c, (list, tuple)):
        return " ".join(_text(x) for x in c)
    return _text(getattr(c, "children", None))


def _split_radio(app):
    return next(r for r in _all(app.layout, dcc.RadioItems) if r.id == "split")


def _table_cells(component) -> tuple[pd.DataFrame, list[str]]:
    """(header -> cell texts of the first html.Table inside ``component``, the row classNames)."""
    table = _all(component, html.Table)[0]
    head = [_text(th).strip() for th in _all(table, html.Th)]
    trs = _all(table.children[1], html.Tr)
    body = [[_text(td).strip() for td in tr.children] for tr in trs]
    return pd.DataFrame(body, columns=head), [tr.className or "" for tr in trs]


def _src(store: ResultsData, split: str, name: str) -> pd.DataFrame:
    """A results file read directly with pandas (independent of the DuckDB layer under test)."""
    return pd.read_parquet(store.dirs()[split] / f"{name}.parquet")


def _expected(rows: pd.DataFrame, src: pd.DataFrame, keys: list[str], col: str) -> np.ndarray:
    """``src[col]`` aligned to ``rows`` on ``keys`` (NaN where the source has no row, e.g. the reference)."""
    s = src.set_index(keys)[col]
    idx = pd.MultiIndex.from_frame(rows[keys]) if len(keys) > 1 else pd.Index(rows[keys[0]])
    return s.reindex(idx).to_numpy()


def _f(x) -> np.ndarray:
    """Floats (bools as 0/1) with NaN for None / NaN / pd.NA."""
    return np.array([np.nan if pd.isna(v) else float(v) for v in x], dtype=float)


def _assert_from(rows, src, keys, pairs):
    for shown, col in pairs:
        np.testing.assert_allclose(_f(rows[shown]), _f(_expected(rows, src, keys, col)), equal_nan=True,
                                   err_msg=f"{shown} is not {col}")


def _iv_asset(store: ResultsData, split: str) -> str:
    return sorted(set(_src(store, split, "eval_encompassing")["asset"]), key=C.ASSETS.index)[0]


# --------------------------------------------------------------------------------------------- layout
def test_create_app_builds_three_tabs(dev_dir, reports_dir):
    app = A.create_app(results_dir=dev_dir, reports_dir=reports_dir)
    tabs = _all(app.layout, dcc.Tabs)
    assert len(tabs) == 1
    assert [t.label for t in tabs[0].children] == ["Forecasts vs realized", "Leaderboard", "VaR breaches"]
    ids = {c.id for c in app.layout._traverse() if getattr(c, "id", None)}
    assert {"asset", "horizon", "split", "models", "risk-models", "forecast-graph", "lb-heatmap", "lb-table",
            "iv-table", "risk-lb", "var-graph", "breach-table", "hypotheses"} <= ids
    horizon = next(r for r in _all(app.layout, dcc.RadioItems) if r.id == "horizon")
    assert [o["value"] for o in horizon.options] == ["1d", "1w", "1m"]
    asset = next(d for d in _all(app.layout, dcc.Dropdown) if d.id == "asset")
    values = [o["value"] for o in asset.options]
    assert values == [a for a in C.ASSETS if a in values] and asset.value == values[0]
    assert A.TITLE in _text(app.layout)
    assert isinstance(app.server.config["VOLRISK_DATA"], ResultsData)


def test_holdout_toggle_disabled_without_holdout_dir(dev_dir, reports_dir):
    app = A.create_app(results_dir=dev_dir, reports_dir=reports_dir)
    radio = _split_radio(app)
    opts = {o["value"]: o for o in radio.options}
    assert radio.value == "dev"
    assert opts["holdout"]["disabled"] is True and opts["dev"]["disabled"] is False
    assert "Holdout disabled" in _text(app.layout)
    assert app.server.config["VOLRISK_DATA"].available_splits() == ["dev"]


def test_holdout_toggle_enabled_with_synthetic_holdout(holdout_store, reports_dir):
    app = A.create_app(results_dir=holdout_store.results_dir, reports_dir=reports_dir)
    opts = {o["value"]: o for o in _split_radio(app).options}
    assert opts["holdout"]["disabled"] is False
    assert holdout_store.available_splits() == ["dev", "holdout"]
    assert holdout_store.available_assets("holdout") == ["BTC", "EURUSD"]


def test_empty_results_dir_renders_placeholders(tmp_path, reports_dir):
    app = A.create_app(results_dir=tmp_path / "nothing", reports_dir=reports_dir)
    assert _split_radio(app).options[1]["disabled"] is True
    s = app.server.config["VOLRISK_DATA"]
    assert s.available_splits() == [] and s.available_assets() == []
    fig = A.forecast_figure(s, None, "1d", ["HAR"], "dev")
    assert len(fig.data) == 0 and fig.layout.annotations
    assert A.update_leaderboard(s, None, "dev")[0].layout.annotations
    assert A.var_figure(s, "BTC", ["HS-250"], "dev").layout.annotations


# --------------------------------------------------------------------------------------------- tab 1
def test_model_options_follow_horizon(store):
    asset = store.available_assets()[0]
    opts_1d, val_1d = A.model_options(store, asset, "1d", "dev", None)
    names = [o["value"] for o in opts_1d]
    assert "IV" not in names and names[0] == "HAR" and names == A.order_models(names)
    assert val_1d == ["HAR", "GARCH", "COMBO"]
    opts_1m, val_1m = A.model_options(store, asset, "1m", "dev", ["COMBO", "IV-cal", "NOPE"])
    assert {"IV", "IV-cal"} <= {o["value"] for o in opts_1m}
    assert val_1m == ["COMBO", "IV-cal"]  # unknown models dropped, display order kept
    _, back = A.model_options(store, asset, "1d", "dev", ["IV-cal"])
    assert back == ["HAR", "GARCH", "COMBO"]  # IV only exists at 1m -> defaults


def test_forecast_figure_traces_and_annualisation(store):
    asset = store.available_assets()[0]
    fig = A.forecast_figure(store, asset, "1w", ["COMBO", "HAR"], "dev")
    assert isinstance(fig, go.Figure)
    assert [t.name for t in fig.data] == ["Realized", "HAR", "COMBO"]
    assert fig.layout.xaxis.rangeslider.visible is True
    ann = annualisation(asset)
    fc = pd.read_parquet(store.results_dir / "forecasts.parquet")
    har = fc[(fc["asset"] == asset) & (fc["horizon"] == "1w") & (fc["model"] == "HAR") & (fc["split"] == "dev")]
    har = har.sort_values("origin")
    np.testing.assert_allclose(np.asarray(fig.data[1].y, dtype=float), np.sqrt(har["F"] / har["n_t"] * ann))
    tg = pd.read_parquet(store.results_dir / "targets.parquet")
    tg = tg[(tg["asset"] == asset) & (tg["horizon"] == "1w") & (tg["split"] == "dev")]
    tg = tg[tg["origin"] >= har["origin"].min()].sort_values("origin")
    np.testing.assert_allclose(np.asarray(fig.data[0].y, dtype=float), np.sqrt(tg["y"] / tg["n_t"] * ann))
    assert fig.data[2].line.color == A.MODEL_COLORS["COMBO"]


def test_palette_mapping_is_the_shared_one():
    assert A.MODEL_COLORS["HAR"] == "#4C78A8" and A.MODEL_COLORS["COMBO"] == "#222222"
    assert A.MODEL_COLORS["IV-cal"] == "#BAB0AC" and A.RISK_COLORS["COMBO+FHS"] == "#222222"
    assert A.RISK_COLORS["HS-250"] == "#9D755D" and tuple(A.HORIZON_ORDER) == ("1d", "1w", "1m")
    assert tuple(A.ASSET_ORDER) == C.ASSETS


# --------------------------------------------------------------------------------------------- tab 2
def test_leaderboard_callback(store):
    asset = store.available_assets()[0]
    fig, summary, table, iv, risk = A.update_leaderboard(store, asset, "dev")
    heat = [t for t in fig.data if t.type == "heatmap"]
    assert len(heat) == 1
    assert list(fig.layout.xaxis.ticktext) == ["1d", "1w", "1m"]
    rows = A.leaderboard_rows(store, asset, "dev")
    models = A.order_models(rows["model"].unique())
    assert np.asarray(heat[0].z).shape == (len(models), 3)
    i, j = models.index("HAR"), 0
    assert np.asarray(heat[0].z)[i, j] == pytest.approx(0.0)  # log(1): HAR is the reference
    mcs = [t for t in fig.data if t.name == "In 90% MCS"]
    assert len(mcs) == 1 and len(mcs[0].x) == int(rows["in_90"].fillna(False).sum())
    head = _text(table)
    assert "DM p" in head and "QLIKE ratio vs HAR" in head and "MCS p" in head
    assert {"dm_p", "dm_hln", "mcs_p", "in_90"} <= set(rows.columns)
    assert rows["horizon"].map(A._horizon_rank).is_monotonic_increasing
    assert "FZ0 diff vs HS-250" in _text(risk) and "Green share" in _text(risk)
    assert "Encompassing c" in _text(iv)  # BTC has implied vol in the dev results
    assert "1d:" in _text(summary)


@pytest.mark.parametrize("which", ["dev", "holdout"])
def test_leaderboard_cells_come_from_their_source_columns(which, store, holdout_store):
    s, split = (store, "dev") if which == "dev" else (holdout_store, "holdout")
    asset = s.available_assets(split)[0]
    rows = A.leaderboard_rows(s, asset, split)
    keys = ["asset", "horizon", "model"]
    _assert_from(rows, _src(s, split, "eval_leaderboard"), keys, [("qlike_ratio", "qlike_ratio"), ("qlike", "qlike")])
    _assert_from(rows, _src(s, split, "eval_dm_har"), keys,
                 [("dm_hln", "dm_hln"), ("dm_p", "pvalue"), ("dm_p_2n", "pvalue_2n")])
    _assert_from(rows, _src(s, split, "eval_mcs"), keys, [("mcs_p", "pvalue"), ("in_90", "in_90")])
    cells, _ = _table_cells(A.leaderboard_table(rows, split))
    assert cells["DM p"].tolist() == [A.fmt_p(p) for p in _expected(rows, _src(s, split, "eval_dm_har"), keys,
                                                                       "pvalue")]
    assert cells["In 90% MCS"].tolist() == [A.fmt_bool(v) for v in _expected(rows, _src(s, split, "eval_mcs"), keys,
                                                                              "in_90")]
    if split == "holdout":
        ci = _src(s, split, "eval_ratio_ci")
        _assert_from(rows, ci, keys, [("ratio_lo", "lo"), ("ratio_hi", "hi")])
        ok = rows["ratio_lo"].notna()
        assert ok.any()
        assert (rows.loc[ok, "ratio_lo"] <= rows.loc[ok, "qlike_ratio"] + 1e-12).all()
        assert (rows.loc[ok, "qlike_ratio"] <= rows.loc[ok, "ratio_hi"] + 1e-12).all()
        assert cells["90% CI low"].tolist() == [A.fmt_num(v) for v in _expected(rows, ci, keys, "lo")]
        assert cells["90% CI high"].tolist() == [A.fmt_num(v) for v in _expected(rows, ci, keys, "hi")]


def test_iv_cells_come_from_their_source_columns(store):
    asset = _iv_asset(store, "dev")
    rows = A.iv_rows(store, asset, "dev")
    assert len(rows) and {"IV", "IV-cal", "COMBO"} <= set(rows["model"])
    _assert_from(rows, _src(store, "dev", "eval_iv_leaderboard").query("horizon == '1m'"), ["asset", "model"],
                 [("qlike_ratio", "qlike_ratio")])
    enc = _src(store, "dev", "eval_encompassing")
    _assert_from(rows, enc, ["asset", "model"], [("enc_c", "c"), ("enc_p", "p_c")])
    ivdm = _src(store, "dev", "eval_iv_dm").query("horizon == '1m'")
    not_ref = rows["model"] != "IV-cal"
    _assert_from(rows[not_ref], ivdm, ["asset", "model"], [("dm_p_ivcal", "pvalue"), ("ratio_vs_ivcal", "ratio")])
    _assert_from(rows, _src(store, "dev", "eval_iv_mcs").query("horizon == '1m'"), ["asset", "model"],
                 [("mcs_p", "pvalue"), ("in_90", "in_90")])
    cells, _ = _table_cells(A.iv_section(store, asset, "dev"))
    assert cells["Encompassing c"].tolist() == [A.fmt_num(v, 3, signed=True)
                                                for v in _expected(rows, enc, ["asset", "model"], "c")]
    assert cells["DM p vs IV-cal"].tolist() == [A.fmt_p(v) for v in rows["dm_p_ivcal"]]


@pytest.mark.parametrize("which", ["dev", "holdout"])
def test_risk_cells_come_from_their_source_columns(which, store, holdout_store):
    s, split = (store, "dev") if which == "dev" else (holdout_store, "holdout")
    asset = s.available_assets(split)[0]
    rows = A.risk_leaderboard_rows(s, asset, split)
    assert len(rows) == 6
    rl = _src(s, split, "risk_risk_leaderboard")
    _assert_from(rows, rl, ["asset", "model"],
                 [("fz0_diff", "fz0_diff"), ("p_dm", "p_dm"), ("fz0", "fz0"),
                  ("mcs_p", "mcs_p"), ("in_90", "in_90"), ("n", "n")])
    _assert_from(rows, _src(s, split, "risk_time_in_zone"), ["asset", "model"], [("green_share", "green")])
    cells, _ = _table_cells(A.risk_leaderboard_table(rows, split))
    assert cells["FZ0 diff vs HS-250"].tolist() == [A.fmt_num(v, 4, signed=True)
                                                    for v in _expected(rows, rl, ["asset", "model"], "fz0_diff")]
    assert cells["Green share (rolling 250)"].tolist() == [A.fmt_pct(v) for v in rows["green_share"]]


def test_risk_leaderboard_green_share_and_zone_are_the_h4_and_var_tab_ones(holdout_store):
    from volrisk import hypotheses as H

    for split in ("dev", "holdout"):
        rl, tz = _src(holdout_store, split, "risk_risk_leaderboard"), _src(holdout_store, split, "risk_time_in_zone")
        h4 = H.h4(rl, time_in_zone=tz, assets=("BTC", "EURUSD")).rows.set_index("unit")
        for asset in ("BTC", "EURUSD"):
            rows = A.risk_leaderboard_rows(holdout_store, asset, split).set_index("model")
            assert rows.at["COMBO+FHS", "green_share"] == pytest.approx(h4.at[asset, "green_COMBO+FHS"])
            assert rows.at["HS-250", "green_share"] == pytest.approx(h4.at[asset, "green_HS-250"])
            z = holdout_store.rolling_zones(asset, list(rows.index), split).sort_values("date")
            last = z.groupby("model").tail(1).set_index("model").reindex(rows.index)
            assert rows["zone_last"].tolist() == last["zone"].tolist()
            assert (pd.to_datetime(rows["zone_date"]) == pd.to_datetime(last["date"])).all()
            assert rows["windows"].tolist() == z.groupby("model").size().reindex(rows.index).astype(float).tolist()
            if split == "holdout":
                assert pd.to_datetime(rows["zone_date"]).min() >= pd.Timestamp("2025-10-01")
    # the holdout leaderboard file's own share (windows entirely inside the holdout) is not the H4 measure:
    # the fixture makes them differ, so a dashboard reading risk_risk_leaderboard.green_share fails above
    m = _src(holdout_store, "holdout", "risk_risk_leaderboard").merge(
        _src(holdout_store, "holdout", "risk_time_in_zone"), on=["asset", "model"])
    assert not np.allclose(m["green_share"], m["green"])
    note = _text(A.risk_leaderboard_table(A.risk_leaderboard_rows(holdout_store, "BTC", "holdout"), "holdout"))
    assert "ending in the holdout sample" in note and "H4" in note


def test_short_holdout_shows_h4_share_not_the_empty_backtest_share(tmp_path, reports_dir):
    """Fewer holdout observations than the rolling window (e.g. SPX with invalid sessions): the leaderboard file
    has no window inside the holdout (NaN / None), yet the windows ending in the holdout exist."""
    dev = write_synthetic(tmp_path / "res", assets=("SPX",), iv_assets=(), bt_window=250, n=300)
    write_synthetic(dev / "holdout", assets=("SPX",), split="holdout", start="2025-10-01", n=120, iv_assets=(),
                    lead_dev_days=300, seed=5, bt_window=250)
    s = ResultsData(dev, reports_dir)
    raw = _src(s, "holdout", "risk_risk_leaderboard")
    assert raw["green_share"].isna().all() and raw["zone_last"].isna().all()
    rows = A.risk_leaderboard_rows(s, "SPX", "holdout")
    tz = _src(s, "holdout", "risk_time_in_zone").set_index("model")["green"]
    np.testing.assert_allclose(rows["green_share"], tz.reindex(rows["model"]).to_numpy())
    assert rows["zone_last"].notna().all() and (rows["windows"] == 120).all()
    cells, _ = _table_cells(A.risk_leaderboard_table(rows, "holdout"))
    assert "—" not in set(cells["Green share (rolling 250)"]) and "—" not in set(cells["Latest zone (99%, last 250 obs.)"])


def test_green_share_falls_back_to_the_leaderboard_without_time_in_zone(tmp_path, reports_dir):
    d = write_synthetic(tmp_path / "res")
    (d / "risk_time_in_zone.parquet").unlink()
    s = ResultsData(d, reports_dir)
    assert s.green_shares("dev")[1] == "risk_leaderboard"
    rows = A.risk_leaderboard_rows(s, "BTC", "dev")
    _assert_from(rows, _src(s, "dev", "risk_risk_leaderboard"), ["asset", "model"], [("green_share", "green_share")])
    assert "no time-in-zone table" in A.risk_leaderboard_note(rows, "dev")


def test_holdout_1m_is_marked_descriptive(holdout_store):
    fig, summary, table, iv, _ = A.update_leaderboard(holdout_store, "BTC", "holdout")
    assert list(fig.layout.xaxis.ticktext) == ["1d", "1w", "1m · descriptive"]
    sub = fig.layout.title.subtitle.text
    assert len(sub) <= 80  # Plotly subtitles do not wrap: it must fit the ≤720 px heatmap column
    heat = next(t for t in fig.data if t.type == "heatmap")
    hover = np.asarray(heat.customdata, dtype=object)
    assert all("descriptive" in str(v) for v in hover[:, 2]) and not any("descriptive" in str(v) for v in hover[:, 0])
    caveat = [p for p in _all(table, html.P) if "caveat" in (p.className or "")]
    assert len(caveat) == 1 and "descriptive" in _text(caveat[0]) and "1d and 1w only" in _text(caveat[0])
    cells, classes = _table_cells(table)
    rows = A.leaderboard_rows(holdout_store, "BTC", "holdout")
    is_1m = (rows["horizon"] == "1m").to_numpy()
    assert (cells.loc[is_1m, "Horizon"] == "1m · descriptive").all()
    assert set(cells.loc[~is_1m, "Horizon"]) == {"1d", "1w"}
    assert all(("descr" in c) == bool(m) for c, m in zip(classes, is_1m, strict=True))
    assert cells.loc[is_1m, "DM p"].ne("—").any()  # the 1m holdout DM p-values are shown, but labelled
    assert "1m (descriptive):" in _text(summary)
    assert "descriptive" in _text(iv)  # 1m IV comparison on the holdout
    # the development sample carries no such marker
    fig_d, summary_d, table_d, iv_d, _ = A.update_leaderboard(holdout_store, "BTC", "dev")
    assert list(fig_d.layout.xaxis.ticktext) == ["1d", "1w", "1m"]
    assert "descriptive" not in _text(table_d) + _text(summary_d) + _text(iv_d)
    assert not any("descr" in c for c in _table_cells(table_d)[1])


def test_verdict_chip_colours():
    def dot(v):
        chip = A._verdict_chip(v)
        return next(c for c in _all(chip, html.Span) if c.className == "dot").style["background"]

    assert dot("supported") == A.ZONE_COLORS["green"] and dot("Supported") == A.ZONE_COLORS["green"]
    assert dot("not supported") == A.ZONE_COLORS["red"]
    assert dot("n/a") == A.NA_COLOR and dot(None) == A.NA_COLOR
    assert "not supported" in _text(A._verdict_chip("not supported"))  # never colour alone


def test_zero_column_results_file_is_treated_as_missing(holdout_store):
    # the holdout run writes its (empty) IV-MCS table as a zero-column parquet, which DuckDB cannot read
    assert ("holdout", "eval_iv_mcs") in holdout_store.skipped and not holdout_store.has("holdout", "eval_iv_mcs")
    iv = holdout_store.iv_comparison("holdout")
    assert len(iv) and iv["mcs_p"].isna().all()
    cells, _ = _table_cells(A.iv_section(holdout_store, "BTC", "holdout"))
    assert "MCS p" not in cells.columns and "Encompassing c" in cells.columns


def test_iv_message_for_eurusd_holdout(holdout_store):
    sec = A.iv_section(holdout_store, "EURUSD", "holdout")
    assert "no free implied-vol benchmark available" in _text(sec).lower()
    assert isinstance(sec, html.P) and not _all(sec, html.Table)
    _, _, _, iv, _ = A.update_leaderboard(holdout_store, "EURUSD", "holdout")
    assert "no free implied-vol benchmark available" in _text(iv).lower()
    btc = A.iv_section(holdout_store, "BTC", "holdout")
    assert _all(btc, html.Table) and "IV-cal" in _text(btc)
    # dev EURUSD has (synthetic) EVZ rows -> a table, not the message
    assert _all(A.iv_section(holdout_store, "EURUSD", "dev"), html.Table)


def test_holdout_views_use_holdout_rows_only(holdout_store):
    fig = A.forecast_figure(holdout_store, "BTC", "1d", ["HAR"], "holdout")
    x = pd.to_datetime(pd.Series(fig.data[1].x))
    assert x.min() >= pd.Timestamp("2025-10-01")  # the holdout file also holds 40 dev rows
    lb = A.leaderboard_rows(holdout_store, "BTC", "holdout")
    assert {"ratio_lo", "ratio_hi"} <= set(lb.columns) and lb["ratio_lo"].notna().any()
    heat_fig = A.leaderboard_figure(holdout_store, "BTC", "holdout")
    mcs = next(t for t in heat_fig.data if t.name == "In 90% MCS")
    assert max(mcs.x) < 2  # holdout MCS only at 1d / 1w
    var = A.var_figure(holdout_store, "BTC", ["HS-250"], "holdout")
    assert pd.to_datetime(pd.Series(var.data[0].x)).min() >= pd.Timestamp("2025-10-01")
    z = A.update_var(holdout_store, "BTC", ["HS-250"], "holdout")[0]
    strip = next(t for t in z.data if t.type == "heatmap")
    assert pd.to_datetime(pd.Series(strip.x)).min() >= pd.Timestamp("2025-10-01")


# --------------------------------------------------------------------------------------------- tab 3
def test_var_callback_traces_and_breaches(store):
    asset = store.available_assets()[0]
    fig, table = A.update_var(store, asset, ["COMBO+FHS", "HS-250"], "dev")
    names = [t.name for t in fig.data]
    assert names[0] == "Daily return"
    for m in ("HS-250", "COMBO+FHS"):
        assert f"{m} −VaR99" in names and f"{m} −VaR97.5" in names and f"{m} plus factor" in names
    assert names.index("HS-250 −VaR99") < names.index("COMBO+FHS −VaR99")  # display order, not click order
    rs = store.risk_series(asset, ["HS-250"], "dev")
    n99 = int((-rs["r_cc"] > rs["var99"]).sum())
    tr = next(t for t in fig.data if t.name and t.name.startswith("HS-250 breaches of VaR99"))
    assert len(tr.x) == n99 and tr.name.endswith(f"({n99})")
    np.testing.assert_allclose(np.asarray(next(t for t in fig.data if t.name == "HS-250 −VaR99").y, dtype=float),
                               -rs.sort_values("date")["var99"].to_numpy())
    strip = next(t for t in fig.data if t.type == "heatmap")
    z = np.asarray(strip.z, dtype=float)
    assert set(np.unique(z[np.isfinite(z)])) <= {0.0, 1.0, 2.0} and list(strip.y) == ["HS-250", "COMBO+FHS"]
    assert {"Green zone (0–4)", "Yellow zone (5–9)", "Red zone (≥10)"} <= set(names)
    rows = A.breach_rows(store, asset, ["HS-250", "COMBO+FHS"], "dev")
    assert len(rows) == 4 and rows.loc[(rows["model"] == "HS-250") & (rows["level"] == "99"), "x"].item() == n99
    assert "Kupiec p" in _text(table)
    empty = A.var_figure(store, asset, [], "dev")
    assert len(empty.data) == 0 and "Select" in empty.layout.annotations[0].text


def test_rolling_zones_query(store):
    asset = store.available_assets()[0]
    z = store.rolling_zones(asset, "COMBO+FHS")
    assert not z.empty and set(z["zone"]) <= {"green", "yellow", "red"} and set(z["model"]) == {"COMBO+FHS"}
    assert z["date"].is_monotonic_increasing


# --------------------------------------------------------------------------------------------- hypotheses
def test_hypotheses_panel_pending_then_verdicts(store, reports_dir):
    pending = A.hypotheses_panel(store, "dev")
    assert "pending" in _text(pending) and "H5" in _text(pending)
    rows = []
    for mode, verdict in (("dev", "supported"), ("holdout", "not supported")):
        for h in ("H1", "H2", "H3", "H4", "H5"):
            rows.append({"mode": mode, "hypothesis": h, "unit": "BTC", "asset": "BTC", "horizon": None,
                         "required": True, "verdict": verdict, "evidence": "x", "note": ""})
            rows.append({"mode": mode, "hypothesis": h, "unit": "overall", "asset": None, "horizon": None,
                         "required": True, "verdict": verdict, "evidence": "rule", "note": ""})
    pd.DataFrame(rows).to_csv(reports_dir / "tables" / "hypotheses.csv", index=False)
    assert len(store.hypotheses("dev")) == 10 and set(store.hypotheses("holdout")["verdict"]) == {"not supported"}
    dev = A.hypotheses_panel(store, "dev")
    assert len(dev) == 5 and "supported" in _text(dev) and "not supported" not in _text(dev)
    assert "BTC:" in _text(dev[0])
    assert "not supported" in _text(A.hypotheses_panel(store, "holdout"))


def test_hypotheses_split_by_file_name(store, reports_dir):
    pd.DataFrame({"hypothesis": ["H1"], "verdict": ["supported"]}).to_csv(
        reports_dir / "tables" / "hypotheses_holdout.csv", index=False)
    assert store.hypotheses("dev").empty
    assert store.hypotheses("holdout")["verdict"].tolist() == ["supported"]


# --------------------------------------------------------------------------------------------- data layer
def test_data_layer_refuses_sealed_holdout_dir():
    with pytest.raises(PermissionError):
        ResultsData(results_dir=C.HOLDOUT)
    with pytest.raises(PermissionError):
        ResultsData(results_dir=C.RESULTS, holdout_dir=C.HOLDOUT / "x")


def test_views_are_read_only_over_results_files(store):
    views = store._views
    assert views and all(split == "dev" for split, _ in views)
    assert ("dev", "forecasts") in views and ("dev", "targets") in views
    lb = store.leaderboard("dev")
    assert set(lb["asset"]) == set(store.available_assets())
    assert (lb.loc[lb["model"] == "HAR", "qlike_ratio"] == 1.0).all()
    assert store.leaderboard("holdout").empty and store.risk_series("BTC", ["HS-250"], "holdout").empty
    assert store.har_star("dev")


def test_annualisation_factors():
    assert (annualisation("BTC"), annualisation("ETH"), annualisation("EURUSD"), annualisation("SPX")) == (
        365, 365, 260, 252)


def test_formatters():
    assert A.fmt_p(0.0004) == "<0.001" and A.fmt_p(float("nan")) == "—" and A.fmt_p(0.0456) == "0.046"
    assert A.fmt_num(-0.1234, 3, signed=True) == "−0.123" and A.fmt_pct(0.925) == "92.5%"
    assert A.zone_color("Green") == A.ZONE_COLORS["green"] and A.zone_color(None) == A.NA_COLOR
    assert len(A.DIVERGING) % 2 == 1  # odd number of stops: the midpoint (ratio = 1) is the neutral grey


def test_main_runs_on_localhost(monkeypatch, dev_dir, reports_dir):
    import dash

    from volrisk.dashboard import __main__ as M

    calls = {}
    monkeypatch.setattr(dash.Dash, "run", lambda self, **kw: calls.update(kw))
    M.main(["--port", "8059", "--results-dir", str(dev_dir), "--reports-dir", str(reports_dir)])
    assert calls == {"host": "127.0.0.1", "port": 8059, "debug": False}
