"""Mechanical H1–H5 decision rules (SPEC §0) on hand-made inputs: verdicts must flip exactly at the thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volrisk import hypotheses as H

ASSETS = ("BTC", "ETH", "EURUSD", "SPX")


# --------------------------------------------------------------------------------------------- builders
def dm_rows(asset: str, garch=(1.2, 0.01), gjr=(1.2, 0.01), horizon: str = "1d", extra: dict | None = None):
    rows = [
        {"asset": asset, "horizon": horizon, "model": "GARCH", "ratio": garch[0], "pvalue": garch[1],
         "dm_hln": 2.0, "pvalue_2n": garch[1]},
        {"asset": asset, "horizon": horizon, "model": "GJR", "ratio": gjr[0], "pvalue": gjr[1],
         "dm_hln": 2.0, "pvalue_2n": gjr[1]},
    ]
    for m, (r, p) in (extra or {}).items():
        rows.append({"asset": asset, "horizon": horizon, "model": m, "ratio": r, "pvalue": p, "dm_hln": -1.0,
                     "pvalue_2n": p})
    return rows


def mcs_rows(asset: str, horizon: str, members: list[str], others=("GARCH", "GJR", "LGBM", "HAR")):
    rows = [{"asset": asset, "horizon": horizon, "model": m, "pvalue": 0.5, "in_90": True, "in_75": True, "T": 500}
            for m in members]
    rows += [{"asset": asset, "horizon": horizon, "model": m, "pvalue": 0.01, "in_90": False, "in_75": False,
              "T": 500} for m in others if m not in members]
    return rows


def risk_lb(asset: str, fz0_diff: float, green_combo: float, green_hs: float, p_dm: float = 0.03):
    return [
        {"asset": asset, "model": "HS-250", "n": 500, "fz0": 2.0, "fz0_diff": 0.0, "dm_hln": np.nan, "p_dm": np.nan,
         "mcs_p": 0.2, "in_90": True, "in_75": False, "zone_last": "green", "green_share": green_hs},
        {"asset": asset, "model": "COMBO+FHS", "n": 500, "fz0": 2.0 + fz0_diff, "fz0_diff": fz0_diff,
         "dm_hln": -2.0, "p_dm": p_dm, "mcs_p": 1.0, "in_90": True, "in_75": True, "zone_last": "green",
         "green_share": green_combo},
    ]


def verdict(res: H.HypothesisResult, unit: str) -> str:
    return res.rows.set_index("unit").loc[unit, "verdict"]


# --------------------------------------------------------------------------------------------- formatting
def test_number_formatting():
    assert H.fmt_p(0.0004) == "<0.001"
    assert H.fmt_p(0.001) == "0.001"
    assert H.fmt_p(0.0494) == "0.049"
    assert H.fmt_p(np.nan) == "—"
    assert H.fmt_p(None) == "—"
    assert H.fmt_num(0.94759) == "0.948"
    assert H.fmt_num(-0.0931, signed=True) == "−0.093"
    assert H.fmt_num(2.0, 2, signed=True) == "+2.00"
    assert H.fmt_share(0.1482) == "14.8%"


def test_overall_verdict_logic():
    S, N, A = H.SUPPORTED, H.NOT_SUPPORTED, H.NA
    assert H.overall_verdict([S, S, S, S]) == S
    assert H.overall_verdict([S, S, A, A]) == A  # incomplete evidence
    assert H.overall_verdict([S, N, A, A]) == N  # one failure decides
    assert H.overall_verdict([]) == A


# --------------------------------------------------------------------------------------------- H1
@pytest.mark.parametrize(
    ("garch", "gjr", "expected"),
    [
        ((1.20, 0.049), (1.10, 0.001), H.SUPPORTED),
        ((1.20, 0.051), (1.10, 0.001), H.NOT_SUPPORTED),  # p just above 5%
        ((1.20, 0.010), (1.10, 0.050), H.NOT_SUPPORTED),  # p = 5% is not < 5%
        ((1.01, 0.010), (1.10, 0.010), H.SUPPORTED),
        ((0.99, 0.010), (1.10, 0.010), H.NOT_SUPPORTED),  # ratio below 1 (GARCH beats HAR)
        ((1.00, 0.010), (1.10, 0.010), H.NOT_SUPPORTED),  # ratio = 1 is not > 1
    ],
)
def test_h1_thresholds(garch, gjr, expected):
    res = H.h1(pd.DataFrame(dm_rows("BTC", garch, gjr)), assets=["BTC"])
    assert verdict(res, "BTC") == expected
    assert res.overall == expected


def test_h1_overall_needs_all_four_assets():
    rows = [r for a in ASSETS for r in dm_rows(a)]
    assert H.h1(pd.DataFrame(rows)).overall == H.SUPPORTED
    partial = pd.DataFrame([r for a in ("BTC", "ETH") for r in dm_rows(a)])
    res = H.h1(partial)
    assert res.overall == H.NA
    assert verdict(res, "EURUSD") == H.NA and verdict(res, "SPX") == H.NA
    assert "EURUSD" in res.note and "SPX" in res.note
    failing = pd.DataFrame(dm_rows("BTC") + dm_rows("ETH", garch=(1.1, 0.08)))
    assert H.h1(failing).overall == H.NOT_SUPPORTED  # a failure decides even with missing assets


def test_h1_uses_only_the_1d_horizon():
    rows = dm_rows("BTC", horizon="1w")  # strong evidence, wrong horizon
    res = H.h1(pd.DataFrame(rows), assets=["BTC"])
    assert verdict(res, "BTC") == H.NA


def test_h1_empty_input_is_na():
    res = H.h1(pd.DataFrame(), assets=ASSETS)
    assert set(res.rows["verdict"]) == {H.NA}
    assert res.overall == H.NA


# --------------------------------------------------------------------------------------------- H2
def test_h2_mcs_without_har_family_fails_the_cell():
    rows = (mcs_rows("BTC", "1d", ["LGBM", "COMBO"]) + mcs_rows("BTC", "1w", ["SHAR"])
            + mcs_rows("BTC", "1m", ["HARQ", "LGBM"]))
    dm = pd.DataFrame(dm_rows("BTC", extra={"LGBM": (0.85, 0.10), "MLP": (0.9, 0.5)}))
    res = H.h2(pd.DataFrame(rows), dm, assets=["BTC"])
    assert verdict(res, "BTC 1d") == H.NOT_SUPPORTED
    assert verdict(res, "BTC 1w") == H.SUPPORTED
    assert verdict(res, "BTC 1m") == H.SUPPORTED
    assert res.overall == H.NOT_SUPPORTED
    r = res.rows.set_index("unit").loc["BTC 1d"]
    assert r["har_in_mcs"] == ""
    assert r["ratio_LGBM"] == pytest.approx(0.85) and r["p_LGBM"] == pytest.approx(0.10)  # reported alongside


@pytest.mark.parametrize("member", ["HAR", "HARQ", "HAR-CJ", "SHAR"])
def test_h2_any_har_family_member_counts(member):
    rows = [r for h in ("1d", "1w", "1m") for r in mcs_rows("ETH", h, [member, "LGBM"])]
    res = H.h2(pd.DataFrame(rows), None, assets=["ETH"])
    assert res.overall == H.SUPPORTED


def test_h2_holdout_horizons_exclude_1m():
    rows = [r for a in ASSETS for h in ("1d", "1w") for r in mcs_rows(a, h, ["HAR"])]
    res = H.h2(pd.DataFrame(rows), None, horizons=("1d", "1w"))
    assert res.overall == H.SUPPORTED  # 8 inferential cells; 1m holdout cells are descriptive
    one_m = res.rows[res.rows["horizon"] == "1m"]
    assert (~one_m["required"]).all() and (one_m["verdict"] == H.NA).all()
    dev = H.h2(pd.DataFrame(rows), None)  # dev rule: the missing 1m cells leave the verdict open
    assert dev.overall == H.NA


# --------------------------------------------------------------------------------------------- H3
def enc(asset, c, p, model="COMBO"):
    return {"asset": asset, "model": model, "T": 1000, "b": 0.8, "c": c, "se_c": 0.1, "t_c": c / 0.1, "p_c": p,
            "c_ivlag": c, "p_c_ivlag": p}


@pytest.mark.parametrize(
    ("c", "p", "expected"),
    [(0.10, 0.049, H.SUPPORTED), (0.10, 0.051, H.NOT_SUPPORTED), (-0.10, 0.001, H.NOT_SUPPORTED),
     (0.0, 0.001, H.NOT_SUPPORTED)],
)
def test_h3_thresholds(c, p, expected):
    res = H.h3(pd.DataFrame([enc("SPX", c, p), enc("SPX", 0.5, 0.001, "HAR")]), assets=["SPX"])
    assert verdict(res, "SPX") == expected


def test_h3_rule_covers_spx_btc_eth_only():
    rows = [enc(a, 0.2, 0.01) for a in ("SPX", "BTC", "ETH")] + [enc("EURUSD", -0.3, 0.001)]
    res = H.h3(pd.DataFrame(rows))
    assert res.overall == H.SUPPORTED  # EURUSD failing does not count
    eur = res.rows.set_index("unit").loc["EURUSD"]
    assert not eur["required"] and eur["verdict"] == H.NOT_SUPPORTED
    # holdout: no EURUSD implied vol at all -> n/a row, overall unaffected
    res2 = H.h3(pd.DataFrame(rows[:3]))
    assert verdict(res2, "EURUSD") == H.NA and res2.overall == H.SUPPORTED
    # a required asset without IV leaves the verdict open
    res3 = H.h3(pd.DataFrame(rows[:2]))
    assert verdict(res3, "ETH") == H.NA and res3.overall == H.NA


# --------------------------------------------------------------------------------------------- H4
@pytest.mark.parametrize(
    ("diff", "g1", "g0", "expected"),
    [
        (-0.01, 0.90, 0.80, H.SUPPORTED),
        (+0.01, 0.90, 0.80, H.NOT_SUPPORTED),  # FZ0 worse than HS-250
        (0.00, 0.90, 0.80, H.NOT_SUPPORTED),  # zero difference is not < 0
        (-0.01, 0.80, 0.80, H.NOT_SUPPORTED),  # green-share tie is not "higher"
        (-0.01, 0.79, 0.80, H.NOT_SUPPORTED),
    ],
)
def test_h4_thresholds(diff, g1, g0, expected):
    res = H.h4(pd.DataFrame(risk_lb("BTC", diff, g1, g0)), assets=["BTC"])
    assert verdict(res, "BTC") == expected


def test_h4_prefers_time_in_zone_and_accepts_fz0_plus_backtests():
    lb = pd.DataFrame(risk_lb("ETH", -0.1, 0.5, 0.9))  # leaderboard green share says "lower"
    tiz = pd.DataFrame([{"asset": "ETH", "model": "COMBO+FHS", "green": 0.9, "yellow": 0.1, "red": 0.0},
                        {"asset": "ETH", "model": "HS-250", "green": 0.4, "yellow": 0.5, "red": 0.1}])
    res = H.h4(lb, time_in_zone=tiz, assets=["ETH"])
    assert verdict(res, "ETH") == H.SUPPORTED
    assert res.rows.iloc[0]["green_source"] == "time_in_zone"
    fz = lb.drop(columns=["zone_last", "green_share"])
    bt = pd.DataFrame([
        {"asset": "ETH", "model": "COMBO+FHS", "level": "99", "green_share": 0.7},
        {"asset": "ETH", "model": "HS-250", "level": "99", "green_share": 0.6},
        {"asset": "ETH", "model": "COMBO+FHS", "level": "97.5", "green_share": 0.0},
        {"asset": "ETH", "model": "HS-250", "level": "97.5", "green_share": 1.0},
    ])
    res2 = H.h4(None, fz0=fz, backtests=bt, assets=["ETH"])
    assert verdict(res2, "ETH") == H.SUPPORTED  # uses the 99% green share only
    assert res2.rows.iloc[0]["green_source"] == "backtests_99"


def test_h4_missing_assets():
    res = H.h4(pd.DataFrame(risk_lb("BTC", -0.1, 1.0, 0.9)))
    assert verdict(res, "BTC") == H.SUPPORTED
    assert res.overall == H.NA
    assert verdict(res, "SPX") == H.NA


# --------------------------------------------------------------------------------------------- H5
def synthetic_daily(spec: dict[str, tuple[float, float, float]], n: int = 400) -> pd.DataFrame:
    """Per asset (j/rv on jump days, jump-day share, rs_neg/rv) -> deterministic gold-like rows."""
    rows = []
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    for a, (jr, share, neg) in spec.items():
        n_jump = int(round(share * n))
        for i, d in enumerate(dates):
            rv = 2.0
            j = jr * rv if i < n_jump else 0.0
            rows.append({"asset": a, "session_date": d, "rv": rv, "j": j, "rs_neg": neg * rv})
    return pd.DataFrame(rows)


def lb_1d(gains: dict[str, tuple[float, float]]) -> pd.DataFrame:
    rows = []
    for a, (g_cj, g_shar) in gains.items():
        for m, g in (("HAR", 0.0), ("HAR-CJ", g_cj), ("SHAR", g_shar)):
            rows.append({"asset": a, "horizon": "1d", "split": "dev", "model": m, "qlike": 0.3,
                         "mse": 1.0, "qlike_ratio": 1.0 - g, "n": 400})
    return pd.DataFrame(rows)


def test_daily_measures_definitions():
    d = synthetic_daily({"BTC": (0.5, 0.25, 0.6)}, n=100)
    m = H.daily_measures(d).set_index("asset").loc["BTC"]
    assert m["jump_share"] == pytest.approx(0.25)
    assert m["j_rv"] == pytest.approx(0.25 * 0.5)  # mean of daily J/RV
    assert m["rsneg_rv"] == pytest.approx(0.6)
    assert m["n_sessions"] == 100
    # window filter and non-positive RV skipped
    d.loc[0, "rv"] = 0.0
    m2 = H.daily_measures(d, start="2021-01-02", end="2021-01-11")
    assert int(m2.iloc[0]["n_sessions"]) == 10


def test_h5_supported_and_flip():
    daily = synthetic_daily({"BTC": (0.3, 0.2, 0.52), "ETH": (0.3, 0.25, 0.53), "EURUSD": (0.2, 0.05, 0.50)})
    gains = {"BTC": (0.02, 0.03), "ETH": (0.01, 0.02), "EURUSD": (0.00, 0.01)}
    res = H.h5(daily, lb_1d(gains))
    assert verdict(res, "BTC") == H.SUPPORTED and verdict(res, "ETH") == H.SUPPORTED
    assert res.overall == H.SUPPORTED
    assert verdict(res, "EURUSD") == H.NA and res.rows.set_index("unit").loc["EURUSD", "note"] == "reference asset"
    # ETH's SHAR gain equal to EURUSD's -> not larger -> not supported
    gains["ETH"] = (0.01, 0.01)
    res2 = H.h5(daily, lb_1d(gains))
    assert verdict(res2, "ETH") == H.NOT_SUPPORTED and res2.overall == H.NOT_SUPPORTED
    assert "gain_SHAR" in res2.rows.set_index("unit").loc["ETH", "note"]


def test_h5_without_eurusd_is_na():
    daily = synthetic_daily({"BTC": (0.3, 0.2, 0.52), "ETH": (0.3, 0.25, 0.53)})
    res = H.h5(daily, lb_1d({"BTC": (0.02, 0.03), "ETH": (0.01, 0.02)}))
    assert res.overall == H.NA
    assert "EURUSD" in res.note
    assert np.isfinite(res.rows.set_index("unit").loc["BTC", "j_rv"])  # evidence still reported


# --------------------------------------------------------------------------------------------- evidence values
def test_evidence_numbers_come_from_the_right_rows_and_columns():
    r1 = H.h1(pd.DataFrame(dm_rows("BTC", garch=(1.21, 0.012), gjr=(1.13, 0.034))), assets=["BTC"]).rows.iloc[0]
    assert (r1["ratio_GARCH"], r1["p_GARCH"], r1["ratio_GJR"], r1["p_GJR"]) == (1.21, 0.012, 1.13, 0.034)
    assert r1["evidence"] == "GARCH ratio 1.210 (DM p 0.012); GJR ratio 1.130 (DM p 0.034)"
    r3 = H.h3(pd.DataFrame([enc("BTC", 0.21, 0.004), enc("BTC", -0.5, 0.3, "HAR")]), assets=["BTC"]).rows.iloc[0]
    assert (r3["c_COMBO"], r3["p_COMBO"], r3["c_HAR"], r3["p_HAR"]) == (0.21, 0.004, -0.5, 0.3)
    assert r3["evidence"] == "COMBO c +0.210 (p 0.004); HAR c −0.500 (p 0.300)"
    # H4: the FZ0 *difference* (not the level 2 + diff) and the DM p of COMBO+FHS (HS-250 has none)
    r4 = H.h4(pd.DataFrame(risk_lb("ETH", -0.0731, 0.887, 0.419, p_dm=0.0258)), assets=["ETH"]).rows.iloc[0]
    assert r4["fz0_diff"] == -0.0731 and r4["p_dm"] == 0.0258
    assert r4["evidence"] == "FZ0 diff −0.073 (DM p 0.026); green share 88.7% vs HS-250 41.9%"
    r2 = H.h2(pd.DataFrame(mcs_rows("BTC", "1d", ["HARQ", "LGBM"])),
              pd.DataFrame(dm_rows("BTC", extra={"LGBM": (0.912, 0.071), "MLP": (1.043, 0.52)})),
              assets=["BTC"]).rows.set_index("unit").loc["BTC 1d"]
    assert (r2["ratio_LGBM"], r2["p_LGBM"], r2["ratio_MLP"], r2["p_MLP"]) == (0.912, 0.071, 1.043, 0.52)
    assert r2["har_in_mcs"] == "HARQ" and r2["mcs_size"] == 2


# --------------------------------------------------------------------------------------------- verdicts()
def test_verdicts_reads_a_results_dir(tmp_path):
    pd.DataFrame([r for a in ASSETS for r in dm_rows(a, extra={"LGBM": (0.9, 0.2), "MLP": (0.95, 0.6)})]) \
        .to_parquet(tmp_path / "eval_dm_har.parquet")
    pd.DataFrame([r for a in ASSETS for h in ("1d", "1w", "1m") for r in mcs_rows(a, h, ["HARQ"])]) \
        .to_parquet(tmp_path / "eval_mcs.parquet")
    pd.DataFrame([enc(a, 0.2, 0.01) for a in ("SPX", "BTC", "ETH")]).to_parquet(tmp_path / "eval_encompassing.parquet")
    pd.DataFrame([r for a in ASSETS for r in risk_lb(a, -0.05, 0.9, 0.5)]) \
        .to_parquet(tmp_path / "risk_risk_leaderboard.parquet")
    lb_1d({a: (0.0, 0.0) for a in ASSETS}).to_parquet(tmp_path / "eval_leaderboard.parquet")
    daily = synthetic_daily({a: (0.3, 0.1, 0.5) for a in ASSETS})
    v = H.verdicts(tmp_path, "dev", daily=daily)
    assert list(v.columns) == ["mode", "hypothesis", "unit", "asset", "horizon", "required", "verdict", "evidence",
                               "note"]
    overall = v[v["unit"] == "overall"].set_index("hypothesis")["verdict"]
    assert overall.to_dict() == {"H1": H.SUPPORTED, "H2": H.SUPPORTED, "H3": H.SUPPORTED, "H4": H.SUPPORTED,
                                 "H5": H.NOT_SUPPORTED}  # equal measures are not "larger"
    assert set(v["verdict"]) <= set(H.VERDICTS)
    hv = H.verdicts(tmp_path, "holdout", daily=daily)
    h2_cells = hv[(hv["hypothesis"] == "H2") & (hv["unit"] != "overall")]
    assert set(h2_cells.loc[h2_cells["horizon"] == "1m", "verdict"]) == {H.NA}


def test_verdicts_missing_files_are_na(tmp_path):
    v = H.verdicts(tmp_path, "dev", daily=pd.DataFrame(columns=["asset", "session_date", "rv", "j", "rs_neg"]))
    assert set(v["verdict"]) == {H.NA}


def test_verdicts_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError):
        H.verdicts(tmp_path, "live")
