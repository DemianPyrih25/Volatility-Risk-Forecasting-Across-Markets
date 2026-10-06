"""Static report (tables, figures, results.md) on synthetic result tables with the real schemas, in tmp dirs.

Nothing here reads ``data/``: the gold-daily loader is replaced by a guard, the holdout path is exercised with a
synthetic ``holdout/`` results directory.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex, to_rgb

from volrisk import config as C
from volrisk import hypotheses as H
from volrisk import io
from volrisk import palette as P
from volrisk import report as R
from volrisk.evaluation.leaderboard import leaderboard, losses_frame
from volrisk.risk.backtests import plus_factor

ASSET_ORDER = ("BTC", "ETH", "EURUSD", "SPX")
MODELS =("RW", "EWMA", "GARCH", "GJR", "HAR", "HAR-CJ", "SHAR", "HARQ", "LGBM", "MLP", "COMBO")
RISK_MODELS = ("HS-250", "RiskMetrics", "GJR+FHS", "HAR*+FHS", "COMBO+FHS", "COMBO+Normal")
HORIZON_DAYS = {"1d": 1, "1w": 7, "1m": 30}
DEV_START, DEV_END = pd.Timestamp("2024-06-01"), pd.Timestamp("2025-09-30")
HOLD_END = pd.Timestamp("2026-02-28")
ASSETS = ("BTC", "EURUSD", "SPX")


# --------------------------------------------------------------------------------------------- generator
def _targets_forecasts(assets, iv_assets, end, rng):
    tg, fc = [], []
    for a in assets:
        days = pd.date_range(DEV_START - pd.Timedelta(days=5), end + pd.Timedelta(days=31), freq="D")
        v = np.exp(np.log(2.0) + np.cumsum(rng.normal(0, 0.05, len(days))))
        tv = v * rng.chisquare(4, len(days)) / 4
        for h, nd in HORIZON_DAYS.items():
            for i, o in enumerate(days):
                if o < DEV_START or i + nd >= len(days):
                    continue
                we = days[i + nd]
                split = "dev" if we <= DEV_END else ("holdout" if o >= DEV_END and we <= end else "dropped")
                if split == "dropped":
                    continue
                y = float(tv[i + 1:i + nd + 1].sum())
                tg.append({"asset": a, "horizon": h, "origin": o, "window_end": we, "n_t": nd, "y": y,
                           "ybar": y / nd, "split": split})
                base = v[i] * nd
                for k, m in enumerate(MODELS):
                    fc.append({"asset": a, "horizon": h, "model": m, "origin": o, "n_t": nd,
                               "F": base * np.exp(rng.normal(0.02 * (k - 5), 0.25)), "split": split})
                if h == "1m" and a in iv_assets:
                    for m, s in (("IV", 1.2), ("IV-cal", 1.0)):
                        if m == "IV-cal" and i < 40:
                            continue
                        fc.append({"asset": a, "horizon": h, "model": m, "origin": o, "n_t": nd,
                                   "F": base * s * np.exp(rng.normal(0, 0.15)), "split": split})
    tg = pd.DataFrame(tg).astype({"origin": "datetime64[ms]", "window_end": "datetime64[ms]"})
    fc = pd.DataFrame(fc).astype({"origin": "datetime64[ms]"})
    return tg, fc


def _cell_tables(lb: pd.DataFrame, mode: str, rng) -> dict[str, pd.DataFrame]:
    dm, mcs, ci = [], [], []
    for (a, h), g in lb.groupby(["asset", "horizon"]):
        n_max = C.n_max(h, a)
        for r in g.itertuples():
            pv = float(rng.uniform(0, 1))
            if r.model != "HAR":
                dm.append({"model": r.model, "ref": "HAR", "mean_diff": r.qlike - 0.3, "dm": 1.0, "dm_hln": 1.0,
                           "pvalue": pv, "T": r.n, "maxlags": max(n_max - 1, 4),
                           "kernel": "bartlett" if n_max == 1 else "uniform", "pvalue_2n": pv * 0.9,
                           "ratio": r.qlike_ratio, "asset": a, "horizon": h})
                ci.append({"asset": a, "horizon": h, "model": r.model, "ratio": r.qlike_ratio,
                           "lo": r.qlike_ratio * 0.9, "hi": r.qlike_ratio * 1.1, "T": r.n})
            if mode == "dev" or h != "1m":
                member = r.model in ("HARQ", "LGBM", "COMBO")
                mcs.append({"model": r.model, "pvalue": 0.5 if member else 0.01, "in_90": member,
                            "in_75": member, "asset": a, "horizon": h, "T": r.n})
    out = {"dm_har": pd.DataFrame(dm), "mcs": pd.DataFrame(mcs)}
    if mode == "holdout":
        out["ratio_ci"] = pd.DataFrame(ci)
    return out


def _iv_tables(losses: pd.DataFrame, mode: str, rng) -> dict[str, pd.DataFrame]:
    l1 = losses[losses["horizon"] == "1m"]
    iv_assets = sorted(l1.loc[l1["model"] == "IV", "asset"].unique())
    if not iv_assets:
        return {}
    h1m = l1[l1["asset"].isin(iv_assets)]
    ivlb = leaderboard(h1m, models=[*MODELS, "IV", "IV-cal"])
    dm, mcs, enc, mz = [], [], [], []
    for r in ivlb.itertuples():
        if r.model != "IV-cal":
            dm.append({"model": r.model, "ref": "IV-cal", "mean_diff": 0.05, "dm": 2.0, "dm_hln": 2.0,
                       "pvalue": 0.03, "T": r.n, "maxlags": 29, "kernel": "uniform", "pvalue_2n": 0.02,
                       "ratio": 1.4, "asset": r.asset, "horizon": "1m"})
        mcs.append({"model": r.model, "pvalue": 0.5 if r.model.startswith("IV") else 0.02,
                    "in_90": r.model.startswith("IV"), "in_75": r.model == "IV-cal", "asset": r.asset,
                    "horizon": "1m", "T": r.n})
    for a in iv_assets:
        for m in MODELS:
            enc.append({"asset": a, "model": m, "T": 300, "b": 0.8, "c": float(rng.normal(0, 0.2)),
                        "se_c": 0.15, "t_c": 0.5, "p_c": float(rng.uniform()), "c_ivlag": 0.1, "p_c_ivlag": 0.4})
    for a in sorted(l1["asset"].unique()):
        for m in [*MODELS, *(("IV", "IV-cal") if a in iv_assets else ())]:
            mz.append({"asset": a, "model": m, "T": 400, "T_nonoverlap": 13, "a": -50.0, "b": 1.0,
                       "p_wald_levels": 0.01, "b_log": 1.05, "se_b_log": 0.2, "p_log": 0.7, "a_no": -10.0,
                       "b_no": 0.9, "p_wald_levels_no": 0.4, "b_log_no": 1.0, "se_b_log_no": 0.2, "p_log_no": 0.9})
    return {"iv_leaderboard": ivlb, "iv_dm": pd.DataFrame(dm),
            "iv_mcs": pd.DataFrame(mcs) if mode == "dev" else pd.DataFrame(),
            "encompassing": pd.DataFrame(enc), "mz": pd.DataFrame(mz)}


def _risk(assets, end, mode, rng) -> dict[str, pd.DataFrame]:
    rows = []
    dates = pd.date_range(DEV_START, end, freq="D")
    for a in assets:
        sigma = np.exp(np.cumsum(rng.normal(0, 0.03, len(dates)))) * 1.5
        r = rng.standard_t(5, len(dates)) * sigma / np.sqrt(5 / 3)
        for k, m in enumerate(RISK_MODELS):
            s = sigma * (1.0 + 0.05 * (k - 2))
            rows.append(pd.DataFrame({
                "asset": a, "model": m, "date": dates, "r_cc": r, "var99": 2.33 * s, "var975": 1.96 * s,
                "es975": 2.34 * s, "sigma": s, "har_member": "HARQ" if m == "HAR*+FHS" else None,
                "split": np.where(dates <= DEV_END, "dev", "holdout")}))
    risk = pd.concat(rows, ignore_index=True).astype({"date": "datetime64[ms]"})
    win = risk[risk["split"] == mode]
    bt, es, fz, zones, tiz = [], [], [], [], []
    # every source column gets values that differ from its neighbours, so a swapped report column is visible
    for k, ((a, m), g) in enumerate(win.groupby(["asset", "model"], sort=False)):
        for j, (lvl, col, p) in enumerate((("99", "var99", 0.01), ("97.5", "var975", 0.025))):
            x = int((g["r_cc"] < -g[col]).sum())
            bt.append({"asset": a, "model": m, "level": lvl, "start": g["date"].min(), "end": g["date"].max(),
                       "x": x, "T": len(g), "rate": x / len(g), "lr_uc": 0.5, "p_chi2": 0.5,
                       "p_binom": 0.61 - 0.01 * (k % 6) - 0.1 * j, "lr_ind": 0.2, "p_ind_mc": 0.6, "lr_cc": 0.7,
                       "p_cc_mc": 0.0004 if j == 0 else 0.0234, "dq": 3.0, "p_dq": 0.83 - 0.02 * (k % 6),
                       "power_uc": 0.712 + 0.2 * j, "zone_full": "green" if x < 6 else "yellow",
                       "green_share": 0.9 if m == "COMBO+FHS" else 0.6,
                       "zone_last": "red" if j == 1 and m == "RiskMetrics" else "green"})
        es.append({"asset": a, "model": m, "T": len(g), "z2": -0.1 - 0.01 * (k % 6), "p_z2": 0.3 + 0.01 * (k % 6)})
        diff = 0.0 if m == "HS-250" else (-0.08 if m == "COMBO+FHS" else 0.01 + 0.001 * (k % 6))
        fz.append({"asset": a, "model": m, "n": len(g), "fz0": 2.0 + diff, "fz0_diff": diff,
                   "dm_hln": np.nan if m == "HS-250" else -2.0, "p_dm": np.nan if m == "HS-250" else 0.03 + 0.01 * k,
                   "mcs_p": 0.5, "in_90": m != "RiskMetrics", "in_75": m == "COMBO+FHS"})
        tiz.append({"asset": a, "model": m, "green": 0.95 if m == "COMBO+FHS" else 0.5,
                    "yellow": 0.05 if m == "COMBO+FHS" else 0.4, "red": 0.0 if m == "COMBO+FHS" else 0.1})
    for (a, m), g in risk.groupby(["asset", "model"], sort=False):
        g = g.iloc[250:] if len(g) > 250 else g.iloc[0:0]
        exc = ((np.arange(len(g)) + 7 * RISK_MODELS.index(m) + 3 * ASSET_ORDER.index(a)) % 120) // 10  # 0..11
        zone = np.select([exc <= 4, exc <= 9], ["green", "yellow"], "red")
        zones.append(pd.DataFrame({"date": g["date"].to_numpy(), "exceptions": exc, "zone": zone,
                                   "plus_factor": [plus_factor(int(x)) for x in exc], "asset": a, "model": m}))
    fz = pd.DataFrame(fz)
    bt = pd.DataFrame(bt)
    lb = fz.merge(bt.loc[bt["level"] == "99", ["asset", "model", "zone_last", "green_share"]], on=["asset", "model"])
    return {"risk": risk, "risk_backtests": bt, "risk_es": pd.DataFrame(es), "risk_fz0": fz,
            "risk_risk_leaderboard": lb, "risk_rolling_zones": pd.concat(zones, ignore_index=True),
            "risk_time_in_zone": pd.DataFrame(tiz)}


def write_results(path: Path, mode: str, assets=ASSETS, iv_assets=ASSETS, seed: int = 7) -> None:
    """Synthetic ``data/results``-like directory for ``mode`` (holdout: forecasts/risk include the dev rows)."""
    rng = np.random.default_rng(seed)
    path.mkdir(parents=True, exist_ok=True)
    end = DEV_END if mode == "dev" else HOLD_END
    tg, fc = _targets_forecasts(assets, iv_assets, end, rng)
    losses = losses_frame(fc, tg)
    head = losses[losses["split"] == mode]
    if mode == "dev":
        head = head[head["origin"] >= pd.Timestamp("2025-01-01")]
        pre = losses[(losses["split"] == "dev") & (losses["origin"] < pd.Timestamp("2025-01-01"))]
    lb = leaderboard(head)
    tables = {"forecasts": fc, "targets": tg, "eval_losses": head, "eval_leaderboard": lb}
    tables |= {f"eval_{k}": v for k, v in _cell_tables(lb, mode, rng).items()}
    tables |= {f"eval_{k}": v for k, v in _iv_tables(head, mode, rng).items()}
    if mode == "dev":
        tables["eval_leaderboard_pre2021"] = leaderboard(pre)
    tables |= _risk(assets, end, mode, rng)
    for k, v in tables.items():
        v.reset_index(drop=True).to_parquet(path / f"{k}.parquet")


def synthetic_daily(assets=ASSETS) -> pd.DataFrame:
    rows = []
    dates = pd.date_range(C.dev_eval_start(), C.dev_end(), freq="D")
    for a in assets:
        share = 0.2 if a in C.CRYPTO else (0.05 if a == "EURUSD" else 0.1)
        j = np.where(np.arange(len(dates)) % int(1 / share) == 0, 0.5, 0.0)
        rows.append(pd.DataFrame({"asset": a, "session_date": dates, "rv": 1.0, "j": j,
                                  "rs_neg": 0.55 if a in C.CRYPTO else 0.5}))
    return pd.concat(rows, ignore_index=True)


@pytest.fixture(autouse=True)
def no_real_gold(monkeypatch):
    def guard(*args, **kwargs):
        raise AssertionError("tests must not read the real gold table")

    monkeypatch.setattr(io, "load_daily", guard)


@pytest.fixture(scope="module")
def dev_build(tmp_path_factory):
    root = tmp_path_factory.mktemp("dev")
    write_results(root / "results", "dev")
    before = sorted(p.relative_to(root) for p in (root / "results").rglob("*"))
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(io, "load_daily", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real gold read")))
        written = R.build(results_dir=root / "results", reports_dir=root / "reports", daily=synthetic_daily())
    after = sorted(p.relative_to(root) for p in (root / "results").rglob("*"))
    return root, written, before, after


@pytest.fixture(scope="module")
def holdout_build(tmp_path_factory):
    root = tmp_path_factory.mktemp("hold")
    write_results(root / "results", "dev")
    # holdout: EURUSD has no implied-vol benchmark (EVZ discontinued)
    write_results(root / "results" / "holdout", "holdout", iv_assets=("BTC", "SPX"), seed=11)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(io, "load_daily", lambda *a, **k: (_ for _ in ()).throw(AssertionError("real gold read")))
        written = R.build(results_dir=root / "results", reports_dir=root / "reports", daily=synthetic_daily())
    return root, written


# --------------------------------------------------------------------------------------------- tests
DEV_TABLES = ("leaderboard_dev", "dm_har_dev", "iv_1m_dev", "mz_dev", "encompassing_dev", "risk_backtests_dev",
              "risk_leaderboard_dev", "hypotheses_dev", "leaderboard_pre2021_dev")
FIGURES = ("leaderboard_heatmap", "cum_qlike_vs_har", "iv_vs_models_1m", "h5_jumps",
           *(f"forecast_vs_realized_{a}" for a in ASSETS), *(f"var_breaches_{a}" for a in ASSETS))


def test_dev_build_writes_every_table_and_figure(dev_build):
    root, written, _, _ = dev_build
    rep = root / "reports"
    for t in DEV_TABLES:
        for ext in ("md", "csv"):
            p = rep / "tables" / f"{t}.{ext}"
            assert p.exists() and p.stat().st_size > 0, p
            assert written[f"tables/{t}.{ext}"] == p
    for f in FIGURES:
        p = rep / "figures" / f"{f}.png"
        assert p.exists(), p
        assert p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert written[f"figures/{f}.png"] == p
    assert not any("holdout" in k for k in written)
    assert (rep / "results.md").exists()


def test_results_md_embeds_existing_figures_and_all_tables(dev_build):
    root, written, _, _ = dev_build
    md = (root / "reports" / "results.md").read_text(encoding="utf-8")
    links = re.findall(r"!\[[^\]]*\]\(\./figures/([^)]+)\)", md)
    assert set(links) == {f"{f}.png" for f in FIGURES}
    for name in links:
        assert (root / "reports" / "figures" / name).exists()
    for title in ("QLIKE ratio vs HAR (development)", "Diebold–Mariano vs HAR", "Mincer–Zarnowitz",
                  "Encompassing", "VaR backtests (development)", "Risk leaderboard: FZ0, ES test and traffic light",
                  "Hypotheses H1–H5", "before 2021"):
        assert title in md
    assert "ratio < 1" in md.lower() or "ratio below 1" in md.lower()
    assert "**sealed**" in md


def test_build_never_writes_into_results(dev_build):
    root, written, before, after = dev_build
    assert before == after
    assert all(p.is_relative_to(root / "reports") for p in written.values())


def test_leaderboard_bold_marks_mcs_members_and_n_footer(dev_build):
    root, *_ = dev_build
    md = (root / "reports" / "tables" / "leaderboard_dev.md").read_text(encoding="utf-8")
    rows = {line.split("|")[1].strip(): line for line in md.splitlines() if line.startswith("| ")}
    assert re.search(r"\*\*\d\.\d{3}\*\*", rows["HARQ"])  # member of every synthetic MCS
    assert "**" not in rows["GARCH"]  # never a member
    assert "**" not in rows["HAR"]
    assert re.fullmatch(r"\| n( \| [\d,]+)+ \|", rows["n"])
    csv = pd.read_csv(root / "reports" / "tables" / "leaderboard_dev.csv")
    assert {"qlike_ratio", "in_mcs90", "mcs_p", "n"} <= set(csv.columns)
    assert csv.loc[csv["model"] == "HARQ", "in_mcs90"].all()


def test_leaderboard_table_formatting_unit():
    lb = pd.DataFrame([
        {"asset": a, "horizon": h, "split": "dev", "model": m, "qlike": 0.3, "mse": 1.0, "qlike_ratio": r, "n": n}
        for a in ("SPX", "BTC") for h, n in (("1m", 300), ("1d", 1733), ("1w", 900))
        for m, r in (("HARQ", 0.94759), ("HAR", 1.0), ("GJR", 1.2391))
    ])
    mcs = pd.DataFrame([{"asset": "BTC", "horizon": "1d", "model": "HARQ", "pvalue": 0.4, "in_90": True,
                         "in_75": True, "T": 1733},
                        {"asset": "BTC", "horizon": "1d", "model": "GJR", "pvalue": 0.0, "in_90": False,
                         "in_75": False, "T": 1733}])
    t = R.leaderboard_table(lb, mcs, name="x", title="x", caption="x")
    header = t.body.splitlines()[0]
    assert header == "| model | BTC 1d | BTC 1w | BTC 1m | SPX 1d | SPX 1w | SPX 1m |"  # asset and horizon order
    lines = t.body.splitlines()
    assert lines[2].startswith("| HAR |") and lines[3].startswith("| HARQ |")  # palette model order
    assert "| **0.948** |" in lines[3]  # BTC 1d HARQ in MCS: bold, 3 decimals
    assert lines[3].count("**") == 2  # only that one cell
    assert "| 1.239 |" in lines[4]
    assert lines[-1] == "| n | 1,733 | 900 | 300 | 1,733 | 900 | 300 |"
    assert list(t.data["horizon"].unique()) == ["1d", "1w", "1m"]


def test_dm_table_orders_horizons_and_formats_p():
    dm = pd.DataFrame([
        {"model": "GJR", "ref": "HAR", "dm_hln": 3.35, "pvalue": p, "pvalue_2n": 0.2, "T": 100, "maxlags": 7,
         "kernel": "bartlett", "ratio": 1.2, "asset": "ETH", "horizon": h, "mean_diff": 0.1}
        for h, p in (("1m", 0.0004), ("1d", 0.0512), ("1w", 0.5))
    ])
    t = R.dm_table(dm, name="x", title="x", caption="x")
    lines = t.body.splitlines()
    assert lines[2].startswith("| model | 1d DM | 1d p | 1d p (2n lags) | 1w DM")
    gjr = next(line for line in lines if line.startswith("| GJR"))
    assert gjr == "| GJR | +3.35 | 0.051 | 0.200 | +3.35 | 0.500 | 0.200 | +3.35 | <0.001 | 0.200 |"


def test_markdown_escaping_of_model_names():
    assert R.esc("HAR*+FHS") == "HAR\\*+FHS"
    assert R.esc("a|b") == "a\\|b"


def test_risk_tables_list_every_asset_and_model(dev_build):
    root, *_ = dev_build
    for t in ("risk_backtests_dev", "risk_leaderboard_dev"):
        md = (root / "reports" / "tables" / f"{t}.md").read_text(encoding="utf-8")
        for a in ASSETS:
            assert f"**{a}** (2024-06-01 → 2025-09-30, T = 487)" in md
        for m in RISK_MODELS:
            assert R.esc(m) in md
    var = pd.read_csv(root / "reports" / "tables" / "risk_backtests_dev.csv")
    assert len(var) == len(ASSETS) * len(RISK_MODELS) * 2  # one row per level
    assert {"x", "T", "p_binom", "p_cc_mc", "p_dq", "power_uc", "zone_full", "zone_latest", "plus_factor_latest",
            "exceptions_latest"} <= set(var)
    lb = pd.read_csv(root / "reports" / "tables" / "risk_leaderboard_dev.csv")
    assert len(lb) == len(ASSETS) * len(RISK_MODELS)
    assert {"fz0", "fz0_diff", "p_dm", "in_mcs90", "z2", "p_z2", "green_share", "zone_latest"} <= set(lb)


def test_risk_tables_show_their_source_values(dev_build):
    """Every cell of the built risk tables against the synthetic source parquet (real schemas)."""
    root, *_ = dev_build
    src = root / "results"
    bt = pd.read_parquet(src / "risk_backtests.parquet")
    fz = pd.read_parquet(src / "risk_fz0.parquet")
    es = pd.read_parquet(src / "risk_es.parquet")
    tiz = pd.read_parquet(src / "risk_time_in_zone.parquet")
    zones = pd.read_parquet(src / "risk_rolling_zones.parquet")
    var = md_tables((root / "reports" / "tables" / "risk_backtests_dev.md").read_text(encoding="utf-8"))
    lbt = md_tables((root / "reports" / "tables" / "risk_leaderboard_dev.md").read_text(encoding="utf-8"))
    assert len(var) == len(lbt) == len(ASSETS)
    for a, vt, lt in zip(ASSETS, var, lbt, strict=True):
        level = ""
        for row in vt:
            level = row["level"] or level
            lvl = {"99%": "99", "97.5%": "97.5"}[level]
            m = unesc(row["model"])
            b = bt[(bt["asset"] == a) & (bt["model"] == m) & (bt["level"] == lvl)].iloc[0]
            assert row["breaches (rate)"] == f"{b['x']} ({100 * b['rate']:.1f}%)"
            assert row["binom p"] == H.fmt_p(b["p_binom"])
            assert row["CC p"] == H.fmt_p(b["p_cc_mc"])
            assert row["DQ p"] == H.fmt_p(b["p_dq"])
            assert row["UC power"] == H.fmt_share(b["power_uc"])
            assert row["zone, whole window"] == b["zone_full"]
            if lvl == "99":
                z = zones[(zones["asset"] == a) & (zones["model"] == m)].sort_values("date").iloc[-1]
                assert row["latest zone (last 250 obs.)"] == z["zone"]
                assert row["plus factor (latest)"] == ("0.00" if z["plus_factor"] == 0
                                                       else H.fmt_num(z["plus_factor"], 2, signed=True))
            else:
                assert row["latest zone (last 250 obs.)"] == b["zone_last"]
                assert row["plus factor (latest)"] == "—"
        assert [r["level"] for r in vt if r["level"]] == ["99%", "97.5%"]
        assert [unesc(r["model"]) for r in lt] == list(RISK_MODELS)
        for row in lt:
            m = unesc(row["model"])
            f = fz[(fz["asset"] == a) & (fz["model"] == m)].iloc[0]
            e = es[(es["asset"] == a) & (es["model"] == m)].iloc[0]
            g = tiz[(tiz["asset"] == a) & (tiz["model"] == m)].iloc[0]
            z = zones[(zones["asset"] == a) & (zones["model"] == m)].sort_values("date").iloc[-1]
            assert row["FZ0 Δ vs HS-250"] == ("0 (ref.)" if m == "HS-250" else H.fmt_num(f["fz0_diff"], signed=True))
            assert row["DM p"] == H.fmt_p(f["p_dm"])
            assert row["90% MCS"] == ("yes" if f["in_90"] else "no")
            assert row["ES Z2"] == H.fmt_num(e["z2"], signed=True)
            assert row["Z2 p"] == H.fmt_p(e["p_z2"])
            assert row["green share (rolling 250)"] == H.fmt_share(g["green"])
            assert row["latest zone (99%, last 250 obs.)"] == z["zone"]


def test_hypotheses_table_dev(dev_build):
    root, *_ = dev_build
    csv = pd.read_csv(root / "reports" / "tables" / "hypotheses_dev.csv", keep_default_na=False)
    assert set(csv["hypothesis"]) == {"H1", "H2", "H3", "H4", "H5"}
    assert set(csv["verdict"]) <= set(H.VERDICTS)
    md = (root / "reports" / "tables" / "hypotheses_dev.md").read_text(encoding="utf-8")
    assert "| **H4** |" in md
    # synthetic risk tables: COMBO+FHS better FZ0 and greener than HS-250 for the three assets, ETH missing
    h4 = csv[(csv["hypothesis"] == "H4")].set_index("unit")["verdict"]
    assert h4["BTC"] == H.SUPPORTED and h4["ETH"] == H.NA and h4["overall"] == H.NA


def test_holdout_autodetected_with_ci_and_shading(holdout_build):
    root, written = holdout_build
    rep = root / "reports"
    for t in ("leaderboard_holdout", "dm_har_holdout", "iv_1m_holdout", "mz_holdout", "encompassing_holdout",
              "risk_backtests_holdout", "risk_leaderboard_holdout", "hypotheses_holdout"):
        assert (rep / "tables" / f"{t}.md").exists(), t
        assert (rep / "tables" / f"{t}.csv").exists(), t
    assert not (rep / "tables" / "leaderboard_pre2021_holdout.md").exists()
    assert (rep / "figures" / "leaderboard_heatmap_holdout.png").exists()
    md = (rep / "tables" / "leaderboard_holdout.md").read_text(encoding="utf-8")
    assert re.search(r"\d\.\d{3} \[\d\.\d{3}, \d\.\d{3}\]", md)  # 90% bootstrap CI
    rows = {line.split("|")[1].strip(): line for line in md.splitlines() if line.startswith("| ")}
    cells = [c.strip() for c in rows["HARQ"].split("|")[2:-1]]
    header = [c.strip() for c in md.splitlines()[next(i for i, x in enumerate(md.splitlines())
                                                      if x.startswith("| model"))].split("|")[2:-1]]
    for h, c in zip(header, cells, strict=True):
        assert c.startswith("**") == (not h.endswith("1m")), (h, c)  # no holdout MCS at 1m
    res_md = (rep / "results.md").read_text(encoding="utf-8")
    assert "opened" in res_md and "Hypotheses H1–H5 (holdout)" in res_md
    csv = pd.read_csv(rep / "tables" / "leaderboard_holdout.csv")
    assert {"ci90_lo", "ci90_hi"} <= set(csv.columns)


def test_holdout_without_eurusd_implied_vol(holdout_build):
    root, _ = holdout_build
    md = (root / "reports" / "tables" / "iv_1m_holdout.md").read_text(encoding="utf-8")
    eur = [line for line in md.splitlines() if line.startswith("| EURUSD")]
    assert len(eur) == 1 and "no free benchmark available" in eur[0]
    dev_md = (root / "reports" / "tables" / "iv_1m_dev.md").read_text(encoding="utf-8")
    assert "no free benchmark available" not in dev_md
    hv = pd.read_csv(root / "reports" / "tables" / "hypotheses_holdout.csv", keep_default_na=False).set_index(["hypothesis", "unit"])
    assert hv.loc[("H3", "EURUSD"), "verdict"] == H.NA
    assert hv.loc[("H2", "BTC 1m"), "verdict"] == H.NA  # holdout MCS at 1d / 1w only


def test_explicit_holdout_flag(tmp_path):
    write_results(tmp_path / "results", "dev", assets=("ETH",), iv_assets=())
    with pytest.raises(FileNotFoundError):
        R.build(include_holdout=True, results_dir=tmp_path / "results", reports_dir=tmp_path / "rep",
                daily=synthetic_daily(("ETH",)))
    written = R.build(include_holdout=False, results_dir=tmp_path / "results", reports_dir=tmp_path / "rep",
                      daily=synthetic_daily(("ETH",)))
    assert "figures/iv_vs_models_1m.png" not in written  # no implied vol at all
    md = (tmp_path / "rep" / "tables" / "iv_1m_dev.md").read_text(encoding="utf-8")
    assert "| ETH | no implied-vol benchmark available |" in md
    assert R.holdout_available(tmp_path / "results") is False
    (tmp_path / "results" / "holdout").mkdir()
    pd.DataFrame({"asset": ["ETH"]}).to_parquet(tmp_path / "results" / "holdout" / "eval_leaderboard.parquet")
    assert R.holdout_available(tmp_path / "results") is True


def test_palette_contract():
    assert P.MODEL_COLORS == {
        "HAR": "#4C78A8", "HARQ": "#72B7B2", "HAR-CJ": "#54A24B", "SHAR": "#88D27A", "GARCH": "#E45756",
        "GJR": "#FF9D98", "EWMA": "#B279A2", "RW": "#9D755D", "LGBM": "#F58518", "MLP": "#FFBF79",
        "COMBO": "#222222", "IV": "#7F7F7F", "IV-cal": "#BAB0AC",
    }
    assert P.RISK_COLORS == {"HS-250": "#9D755D", "RiskMetrics": "#B279A2", "GJR+FHS": "#E45756",
                             "HAR*+FHS": "#4C78A8", "COMBO+FHS": "#222222", "COMBO+Normal": "#F58518"}
    assert set(P.ZONE_COLORS) == {"green", "yellow", "red"}
    assert P.HORIZON_ORDER == C.HORIZONS == ("1d", "1w", "1m")
    assert P.ASSET_ORDER == C.ASSETS
    assert P.order_horizons(["1m", "1d", "1w", "1d"]) == ["1d", "1w", "1m"]
    assert P.order_assets(["SPX", "XYZ", "BTC"]) == ["BTC", "SPX", "XYZ"]
    assert P.order_models(["COMBO", "HAR", "GJR"]) == ["HAR", "GJR", "COMBO"]
    for m in (*MODELS, "IV", "IV-cal"):
        assert m in P.MODEL_COLORS
    for m in RISK_MODELS:
        assert m in P.RISK_COLORS
    assert all(re.fullmatch(r"#[0-9A-F]{6}", c) for c in
               (*P.MODEL_COLORS.values(), *P.RISK_COLORS.values(), *P.ZONE_COLORS.values()))


# --------------------------------------------------------------------------------------------- cell values
def md_tables(md: str) -> list[list[dict[str, str]]]:
    """Every Markdown table in ``md`` as a list of {header: cell} rows (separator line dropped)."""
    out, block = [], []
    for line in [*md.splitlines(), ""]:
        if line.startswith("| "):
            block.append([c.strip() for c in line.strip()[2:-2].split(" | ")])
        elif block:
            header, rows = block[0], block[2:]
            assert all(len(r) == len(header) for r in rows), block
            out.append([dict(zip(header, r, strict=True)) for r in rows])
            block = []
    return out


def unesc(s: str) -> str:
    return s.replace("\\*", "*").replace("\\_", "_").replace("\\|", "|").replace("\\\\", "\\")


def test_md_tables_parser():
    t = md_tables("x\n\n| a | b |\n| :--- | ---: |\n| 1 | **2** |\n| 3 | — |\n\ntext\n\n| c |\n| --- |\n| HAR\\*+FHS |\n")
    assert t == [[{"a": "1", "b": "**2**"}, {"a": "3", "b": "—"}], [{"c": "HAR\\*+FHS"}]]
    assert unesc(t[1][0]["c"]) == "HAR*+FHS"


def test_annualisation_and_vol_units():
    assert {a: R._ann(a) for a in ASSET_ORDER} == {"BTC": 365, "ETH": 365, "EURUSD": 260, "SPX": 252}
    # SPEC §2.4: iv_var_30d = iv² · 30/365 (%²) over a 30-session crypto month -> back to the quoted iv
    iv = pd.Series([40.0, 65.0, 120.0])
    F = iv ** 2 * 30 / 365
    np.testing.assert_allclose(R._vol(F, pd.Series([30, 30, 30]), R._ann("BTC")), iv)
    # 1d: 4 %²/day -> sqrt(4 · 252) % per year; a 22-session month of 4 %²/day gives the same daily vol
    np.testing.assert_allclose(R._vol(pd.Series([4.0, 88.0]), pd.Series([1, 22]), 252), [np.sqrt(1008)] * 2)


def write_tables(path: Path, **tables: pd.DataFrame) -> R.Results:
    path.mkdir(parents=True, exist_ok=True)
    for k, v in tables.items():
        v.to_parquet(path / f"{k}.parquet")
    return R.Results("dev", path)


def unit_risk_inputs() -> dict[str, pd.DataFrame]:
    """Two risk models of one asset; every source column has its own distinct value."""
    start, end = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-09-30")
    bt = pd.DataFrame([
        {"asset": "ETH", "model": m, "level": lvl, "start": start, "end": end, "x": x, "T": 1004, "rate": x / 1004,
         "p_binom": pb, "p_cc_mc": pc, "p_dq": pq, "power_uc": pw, "zone_full": zf, "green_share": gs,
         "zone_last": zl}
        for m, lvl, x, pb, pc, pq, pw, zf, gs, zl in (
            ("HS-250", "99", 19, 0.0123, 0.0456, 0.0789, 0.811, "yellow", 0.41, "green"),
            ("HS-250", "97.5", 37, 0.323, 0.169, 0.0031, 0.997, "green", 0.83, "yellow"),
            ("COMBO+FHS", "99", 8, 0.512, 0.634, 0.745, 0.811, "green", 0.88, "green"),
            ("COMBO+FHS", "97.5", 21, 0.211, 0.322, 0.433, 0.997, "green", 0.11, None),
        )
    ])
    fz = pd.DataFrame([
        {"asset": "ETH", "model": "HS-250", "n": 1004, "fz0": 2.3627, "fz0_diff": 0.0, "dm_hln": np.nan,
         "p_dm": np.nan, "mcs_p": 0.1056, "in_90": True, "in_75": False},
        {"asset": "ETH", "model": "COMBO+FHS", "n": 1004, "fz0": 2.2379, "fz0_diff": -0.1248, "dm_hln": -2.23,
         "p_dm": 0.0258, "mcs_p": 1.0, "in_90": False, "in_75": True},
    ])
    es = pd.DataFrame([{"asset": "ETH", "model": "HS-250", "T": 1004, "z2": -0.2565, "p_z2": 0.2359},
                       {"asset": "ETH", "model": "COMBO+FHS", "T": 1004, "z2": 0.1708, "p_z2": 0.0187}])
    tiz = pd.DataFrame([{"asset": "ETH", "model": "HS-250", "green": 0.4192, "yellow": 0.5269, "red": 0.0539},
                        {"asset": "ETH", "model": "COMBO+FHS", "green": 0.8874, "yellow": 0.1126, "red": 0.0}])
    zones = pd.DataFrame([
        # HS-250: the latest window inside the backtest window ends 2025-09-30 (6 exceptions, +0.50); the
        # 2025-10-01 row lies after the window (a file holding the whole series) and must be ignored
        {"date": pd.Timestamp("2025-09-29"), "exceptions": 4, "zone": "green", "plus_factor": 0.0, "asset": "ETH",
         "model": "HS-250"},
        {"date": pd.Timestamp("2025-09-30"), "exceptions": 6, "zone": "yellow", "plus_factor": 0.5, "asset": "ETH",
         "model": "HS-250"},
        {"date": pd.Timestamp("2025-10-01"), "exceptions": 10, "zone": "red", "plus_factor": 1.0, "asset": "ETH",
         "model": "HS-250"},
        {"date": pd.Timestamp("2025-09-30"), "exceptions": 2, "zone": "green", "plus_factor": 0.0, "asset": "ETH",
         "model": "COMBO+FHS"},
    ])
    return {"risk_backtests": bt, "risk_fz0": fz, "risk_es": es, "risk_time_in_zone": tiz,
            "risk_rolling_zones": zones}


def test_var_backtest_table_cells(tmp_path):
    res = write_tables(tmp_path, **unit_risk_inputs())
    t = R.var_backtest_table(res, ["ETH", "SPX"], name="x", title="x", caption="x")
    assert t.body.startswith("**ETH** (2023-01-01 → 2025-09-30, T = 1,004)")
    assert "**SPX**: _no risk results._" in t.body
    (rows,) = md_tables(t.body)
    assert [(r["level"], unesc(r["model"])) for r in rows] == [
        ("99%", "HS-250"), ("", "COMBO+FHS"), ("97.5%", "HS-250"), ("", "COMBO+FHS")]
    hs99, combo99, hs975, combo975 = rows
    assert hs99 == {"level": "99%", "model": "HS-250", "breaches (rate)": "19 (1.9%)", "binom p": "0.012",
                    "CC p": "0.046", "DQ p": "0.079", "UC power": "81.1%", "zone, whole window": "yellow",
                    "latest zone (last 250 obs.)": "yellow", "plus factor (latest)": "+0.50"}
    assert combo99["breaches (rate)"] == "8 (0.8%)" and combo99["binom p"] == "0.512"
    assert combo99["CC p"] == "0.634" and combo99["DQ p"] == "0.745"
    assert combo99["latest zone (last 250 obs.)"] == "green" and combo99["plus factor (latest)"] == "0.00"
    # 97.5%: latest zone from the backtests' own rolling window; the Basel plus factor exists at 99% only
    assert hs975 == {"level": "97.5%", "model": "HS-250", "breaches (rate)": "37 (3.7%)", "binom p": "0.323",
                     "CC p": "0.169", "DQ p": "0.003", "UC power": "99.7%", "zone, whole window": "green",
                     "latest zone (last 250 obs.)": "yellow", "plus factor (latest)": "—"}
    assert combo975["latest zone (last 250 obs.)"] == "—"  # fewer than 250 observations
    d = t.data.set_index(["model", "level"])
    assert d.loc[("HS-250", "99"), "plus_factor_latest"] == 0.5
    assert d.loc[("HS-250", "99"), "exceptions_latest"] == 6
    assert d.loc[("HS-250", "97.5"), "power_uc"] == pytest.approx(0.997)


def test_latest_zone_falls_back_to_backtests_without_rolling_zones(tmp_path):
    inp = unit_risk_inputs()
    del inp["risk_rolling_zones"]
    res = write_tables(tmp_path, **inp)
    (rows,) = md_tables(R.var_backtest_table(res, ["ETH"], name="x", title="x", caption="x").body)
    assert rows[0]["latest zone (last 250 obs.)"] == "green"  # zone_last of the 99% backtest
    assert rows[0]["plus factor (latest)"] == "—"


def test_risk_leaderboard_table_cells(tmp_path):
    res = write_tables(tmp_path, **unit_risk_inputs())
    t = R.risk_leaderboard_table(res, ["ETH"], name="x", title="x", caption="x")
    (rows,) = md_tables(t.body)
    assert rows == [
        {"model": "HS-250", "FZ0 Δ vs HS-250": "0 (ref.)", "DM p": "—", "90% MCS": "yes", "ES Z2": "−0.257",
         "Z2 p": "0.236", "green share (rolling 250)": "41.9%", "latest zone (99%, last 250 obs.)": "yellow"},
        # FZ0 as a difference (DEVIATIONS), never the level 2.238; MCS from in_90, not in_75
        {"model": "COMBO+FHS", "FZ0 Δ vs HS-250": "−0.125", "DM p": "0.026", "90% MCS": "no", "ES Z2": "+0.171",
         "Z2 p": "0.019", "green share (rolling 250)": "88.7%", "latest zone (99%, last 250 obs.)": "green"},
    ]
    d = t.data.set_index("model")
    assert d.loc["COMBO+FHS", "fz0"] == pytest.approx(2.2379)
    assert d.loc["COMBO+FHS", "fz0_diff"] == pytest.approx(-0.1248)


def test_green_share_falls_back_to_the_99_backtest(tmp_path):
    inp = unit_risk_inputs()
    del inp["risk_time_in_zone"]
    res = write_tables(tmp_path, **inp)
    g = R._green(res).set_index("model")["green"]
    assert g.to_dict() == {"HS-250": 0.41, "COMBO+FHS": 0.88}  # 99% rows, not the 97.5% ones
    (rows,) = md_tables(R.risk_leaderboard_table(res, ["ETH"], name="x", title="x", caption="x").body)
    assert [r["green share (rolling 250)"] for r in rows] == ["41.0%", "88.0%"]


def test_mz_table_cells():
    mz = pd.DataFrame([{"asset": "BTC", "model": "HAR", "T": 1450, "T_nonoverlap": 48, "a": -12.34, "b": 0.877,
                        "p_wald_levels": 0.0123, "b_log": 1.111, "se_b_log": 0.222, "p_log": 0.333,
                        "a_no": -3.0, "b_no": 0.9, "p_wald_levels_no": 0.4, "b_log_no": 0.944, "se_b_log_no": 0.25,
                        "p_log_no": 0.455}])
    (rows,) = md_tables(R.mz_table(mz, name="x", title="x", caption="x").body)
    # inference on the log regression; the levels Wald test is descriptive only (DEVIATIONS)
    assert rows == [{"asset": "BTC", "model": "HAR", "T": "1,450", "log b": "1.111", "s.e.": "0.222",
                     "p (b = 1)": "0.333", "T non-overl.": "48", "log b non-overl.": "0.944", "p non-overl.": "0.455",
                     "levels a †": "−12.3", "levels b †": "0.877", "levels p (a = 0, b = 1) †": "0.012"}]


def test_encompassing_table_cells():
    enc = pd.DataFrame([{"asset": "SPX", "model": "COMBO", "T": 1200, "b": 0.812, "c": 0.234, "se_c": 0.067,
                         "t_c": 3.49, "p_c": 0.0004, "c_ivlag": -0.199, "p_c_ivlag": 0.021}])
    (rows,) = md_tables(R.encompassing_table(enc, name="x", title="x", caption="x").body)
    assert rows == [{"asset": "SPX", "model": "**COMBO**", "T": "1,200", "b (log IV)": "0.812",
                     "c (log F)": "+0.234", "s.e. (c)": "0.067", "p (c = 0)": "<0.001", "c with IV t−1": "−0.199",
                     "p with IV t−1": "0.021"}]


def test_iv_table_cells():
    lb = pd.DataFrame([{"asset": "BTC", "horizon": "1m", "split": "dev", "model": m, "qlike": q, "mse": 1.0,
                        "qlike_ratio": r, "n": 640}
                       for m, q, r in (("HAR", 0.31, 1.0), ("COMBO", 0.29, 0.935), ("IV", 0.4, 1.29),
                                       ("IV-cal", 0.33, 1.064))])
    dm = pd.DataFrame([{"asset": "BTC", "horizon": "1m", "model": m, "ref": "IV-cal", "ratio": r, "dm_hln": d,
                        "pvalue": p, "pvalue_2n": p2, "T": 640, "maxlags": 29, "kernel": "uniform", "mean_diff": 0.0}
                       for m, r, d, p, p2 in (("HAR", 0.94, -1.11, 0.267, 0.301),
                                              ("COMBO", 0.879, -2.02, 0.044, 0.061),
                                              ("IV", 1.212, 3.3, 0.001, 0.0005))])
    mcs = pd.DataFrame([{"asset": "BTC", "horizon": "1m", "model": m, "pvalue": p, "in_90": i90, "in_75": i75,
                         "T": 640}
                        for m, p, i90, i75 in (("HAR", 0.2, True, False), ("COMBO", 1.0, True, True),
                                               ("IV", 0.01, False, True), ("IV-cal", 0.15, True, False))])
    t = R.iv_table(lb, dm, mcs, ["BTC", "EURUSD"], name="x", title="x", caption="x")
    (rows,) = md_tables(t.body)
    by = {unesc(r["model"]).strip("*"): r for r in rows[:-1]}
    assert by["HAR"] == {"asset": "BTC", "model": "HAR", "QLIKE ratio vs HAR": "1.000",
                         "QLIKE ratio vs IV-cal": "0.940", "DM vs IV-cal": "−1.11", "p": "0.267", "p (2n lags)": "0.301",
                         "90% MCS": "yes", "n": "640"}
    assert by["COMBO"]["QLIKE ratio vs HAR"] == "0.935" and by["COMBO"]["QLIKE ratio vs IV-cal"] == "0.879"
    assert by["COMBO"]["p"] == "0.044" and by["COMBO"]["p (2n lags)"] == "0.061"
    assert by["IV"]["90% MCS"] == "no"  # in_90, although in_75 is True
    assert by["IV"]["p"] == "0.001" and by["IV"]["p (2n lags)"] == "<0.001"
    assert by["IV-cal"]["QLIKE ratio vs IV-cal"] == "1.000" and by["IV-cal"]["DM vs IV-cal"] == "—"
    assert by["IV-cal"]["90% MCS"] == "yes"  # in_90 (in_75 is False)
    assert rows[-1]["asset"] == "EURUSD" and "no free benchmark" in rows[-1]["model"]


def test_iv_figure_keeps_ivcal_visible(monkeypatch, tmp_path):
    """IV-cal (the reference of the DM test vs implied vol) is the lightest model colour: the realized area must
    stay a light wash under it, and IV-cal must carry non-colour cues (dashes, heavier stroke, surface ring)."""

    def contrast(a: str, b: str) -> float:
        la, lb = sorted((R._luminance(to_rgb(a)), R._luminance(to_rgb(b))), reverse=True)
        return (la + 0.05) / (lb + 0.05)

    assert contrast(P.MODEL_COLORS["IV-cal"], P.REALIZED_WASH) >= 1.75  # 1.32 on the old #CFCCC5 fill
    assert contrast(P.MODEL_COLORS["IV"], P.REALIZED_WASH) >= 3.0
    captured = {}
    save = R._save

    def spy(fig, path):
        captured["fig"] = fig
        return save(fig, path)

    monkeypatch.setattr(R, "_save", spy)
    tg, fc = _targets_forecasts(("BTC",), ("BTC",), DEV_END, np.random.default_rng(3))
    assert R.fig_iv_vs_models(fc, tg, tmp_path / "iv.png", None) is not None
    ax = captured["fig"].axes[0]
    (wash,) = ax.collections
    assert to_hex(wash.get_facecolor()[0]).upper() == P.REALIZED_WASH
    lines = {ln.get_label(): ln for ln in ax.get_lines()}
    ivcal, iv, combo = lines["IV-cal"], lines["IV"], lines["COMBO"]
    assert to_hex(ivcal.get_color()).upper() == P.MODEL_COLORS["IV-cal"]
    assert ivcal.get_linestyle() == "--" and ivcal.get_linewidth() >= 2.0 > iv.get_linewidth()
    assert ivcal.get_path_effects()
    assert wash.get_zorder() < iv.get_zorder() < ivcal.get_zorder() < combo.get_zorder()
    # plotted values are annualised vols √(F / n_t × 365) at each origin; the wash is √(y / n_t × 365)
    f = fc[(fc["asset"] == "BTC") & (fc["horizon"] == "1m")].pivot_table(index="origin", columns="model",
                                                                          values="F")
    t = tg[(tg["asset"] == "BTC") & (tg["horizon"] == "1m")].set_index("origin")
    d = f.join(t[["y", "n_t"]], how="inner")
    d = d[d["IV"].notna()]
    np.testing.assert_allclose(iv.get_ydata(), np.sqrt(d["IV"] / d["n_t"] * 365))
    np.testing.assert_allclose(combo.get_ydata(), np.sqrt(d["COMBO"] / d["n_t"] * 365))
    top = wash.get_paths()[0].vertices[:, 1]
    assert top.max() == pytest.approx(float(np.sqrt(d["y"] / d["n_t"] * 365).max()))
