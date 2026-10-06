"""Tests for the risk module (SPEC §9): VaR/ES models, VaR backtests, Z2 ES backtest, FZ0 scoring."""

from __future__ import annotations

import copy
import math

import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import binom, binomtest, chi2, norm
from scipy.stats import t as student_t

from volrisk import config as C
from volrisk.evaluation.dm import dm_test
from volrisk.models import simple
from volrisk.risk import backtests as bt
from volrisk.risk import es_tests as es
from volrisk.risk import scoring, var_es

SEED = 20261002


# --------------------------------------------------------------------------------------------- helpers
def _std_t(nu: float) -> tuple[float, float, float]:
    """(scale to unit variance, VaR97.5, ES97.5) of a unit-variance Student t (positive numbers)."""
    sc = math.sqrt((nu - 2) / nu)
    q = student_t.ppf(0.025, nu)
    es975 = sc * student_t.pdf(q, nu) / 0.025 * (nu + q * q) / (nu - 1)
    return sc, -q * sc, es975


def _series(n: int, seed: int = 0, start: str = "2015-01-01"):
    """Returns r = sigma·z with z ~ N(0,1) and the true variance indexed by date (sigma2 known)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    s2 = np.exp(0.6 * np.sin(np.arange(n) / 40.0))
    r = pd.Series(np.sqrt(s2) * rng.standard_normal(n), index=dates)
    return r, pd.Series(s2, index=dates)


def _synthetic(n: int = 1300, fc_start: int = 300, combo: bool = True, seed: int = 1):
    """Gold-like daily rows (polars-style datetime64[ms]) and a 1d forecast table whose F at origin i is the
    true variance of session i+1 (scaled per model). Includes noise rows that must be filtered out."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-01", periods=n)
    s2 = np.exp(0.5 * np.sin(np.arange(n) / 50.0))
    r = np.sqrt(s2) * rng.standard_normal(n)
    r[0] = np.nan  # first session has no previous close
    daily = pd.DataFrame({"asset": "SPX", "session_date": dates.astype("datetime64[ms]"), "r_cc": r})
    origins = dates[fc_start:].astype("datetime64[ms]")  # includes the last session (no target -> dropped)
    f_true = np.r_[s2[fc_start + 1 :], 1.0]
    scales = {"GJR": 1.1, "HARQ": 0.9, "LGBM": 1.0, "HAR": 1.2}
    if combo:
        scales["COMBO"] = (1.1 + 0.9 + 1.0) / 3
    parts = [
        pd.DataFrame({"asset": "SPX", "horizon": "1d", "model": m, "origin": origins, "n_t": 1, "F": f_true * k})
        for m, k in scales.items()
    ]
    parts.append(parts[0].assign(horizon="1w", F=99.0))  # other horizon
    parts.append(parts[0].assign(asset="BTC", F=99.0))  # other asset
    return daily, pd.concat(parts, ignore_index=True), dates, s2


# --------------------------------------------------------------------------------------------- var_es: FHS
def test_fhs_uses_only_past_z():
    r, s2 = _series(1600, seed=3)
    base = var_es.fhs(r, s2, pool=1000, min_pool=500)
    k = 1100
    r2, s22 = r.copy(), s2.copy()
    r2.iloc[k:] = r2.iloc[k:] * 4.0 - 1.0
    s22.iloc[k + 1 :] *= 2.0  # forecasts for later targets
    pert = var_es.fhs(r2, s22, pool=1000, min_pool=500)
    cut = r.index[k]
    pd.testing.assert_frame_equal(base.loc[:cut], pert.loc[:cut])  # VaR at date k uses z strictly before k
    assert not np.allclose(base.loc[r.index[k + 1] :, "es975"], pert.loc[r.index[k + 1] :, "es975"])


def test_fhs_pool_matches_manual_computation():
    r, s2 = _series(1400, seed=4)
    out = var_es.fhs(r, s2, pool=1000, min_pool=500)
    assert out["pool_n"].iloc[0] == 500 and out.index[0] == r.index[500]
    assert out["pool_n"].max() == 1000
    d = r.index[1300]
    z = (r / np.sqrt(s2)).loc[: r.index[1299]].to_numpy()[-1000:]
    q99, q975 = np.quantile(z, [0.01, 0.025], method="linear")
    sig = math.sqrt(s2[d])
    row = out.loc[d]
    assert row["sigma"] == pytest.approx(sig)
    assert row["var99"] == pytest.approx(-sig * q99)
    assert row["var975"] == pytest.approx(-sig * q975)
    assert row["es975"] == pytest.approx(-sig * z[z <= q975].mean())
    pools = var_es.fhs_pools(r, s2, [d], pool=1000)
    np.testing.assert_array_equal(pools[0], z)


def test_fhs_standard_normal_z_recovers_normal_multipliers():
    r, s2 = _series(8000, seed=5)
    out = var_es.fhs(r, s2)
    ratio = out[["var99", "var975", "es975"]].div(out["sigma"], axis=0).mean()
    assert ratio["var99"] == pytest.approx(norm.ppf(0.99), abs=0.08)
    assert ratio["var975"] == pytest.approx(norm.ppf(0.975), abs=0.05)
    assert ratio["es975"] == pytest.approx(var_es.ES975_NORMAL, abs=0.08)
    assert (out["es975"] >= out["var975"]).all()


def test_fhs_rejects_nonpositive_variance():
    r, s2 = _series(600)
    s2.iloc[10] = 0.0
    with pytest.raises(ValueError):
        var_es.fhs(r, s2)


# --------------------------------------------------------------------------------------------- var_es: others
def test_historical_sim_matches_manual_window():
    r, _ = _series(400, seed=6)
    r.iloc[0] = np.nan
    out = var_es.historical_sim(r, window=250)
    valid = r.dropna()
    assert out.index[0] == valid.index[250] and (out["pool_n"] == 250).all()
    d = valid.index[300]
    w = valid.loc[: valid.index[299]].to_numpy()[-250:]
    q99, q975 = np.quantile(w, [0.01, 0.025], method="linear")
    row = out.loc[d]
    assert row["var99"] == pytest.approx(-q99)
    assert row["var975"] == pytest.approx(-q975)
    assert row["es975"] == pytest.approx(-w[w <= q975].mean())
    assert row["sigma"] == pytest.approx(w.std(ddof=1))
    windows = var_es.hs_windows(r, [d], 250)
    np.testing.assert_array_equal(windows[0], w)


def test_riskmetrics_recursion_and_normal_quantiles():
    r, _ = _series(300, seed=7)
    out = var_es.riskmetrics(r, lam=0.94, init=250)
    x = r.to_numpy()
    s = np.var(x[:250], ddof=1)
    assert out["sigma"].iloc[0] ** 2 == pytest.approx(s)
    s = 0.94 * s + 0.06 * x[250] ** 2
    assert out["sigma"].iloc[1] ** 2 == pytest.approx(s)
    assert out.index[0] == r.index[250]
    np.testing.assert_allclose(out["var99"] / out["sigma"], norm.ppf(0.99))


def test_normal_multipliers_from_scipy():
    out = var_es.normal_from_sigma2(pd.Series([4.0, 1.0], index=pd.bdate_range("2024-01-01", periods=2)))
    assert out["var99"].iloc[1] == pytest.approx(2.326, abs=5e-4)
    assert out["var975"].iloc[1] == pytest.approx(1.960, abs=5e-4)
    assert out["es975"].iloc[1] == pytest.approx(2.338, abs=5e-4)
    assert out["sigma"].iloc[0] == pytest.approx(2.0)
    assert out["es975"].iloc[1] == pytest.approx(norm.pdf(norm.ppf(0.025)) / 0.025)


# --------------------------------------------------------------------------------------------- risk_frame
def test_risk_frame_models_alignment_and_common_window():
    daily, fc, dates, s2 = _synthetic()
    out = var_es.risk_frame("SPX", daily, fc, har_star="HARQ")
    assert list(out.columns) == var_es.RISK_COLUMNS
    assert tuple(out["model"].unique()) == var_es.RISK_MODELS
    counts = out.groupby("model")["date"].count()
    assert counts.nunique() == 1
    # z exist for targets 301.. ; FHS pool reaches 500 at session 801; forecast at the last session is dropped
    assert out["date"].min() == dates[801] and out["date"].max() == dates[-1]
    assert out["date"].dtype == np.dtype("datetime64[ms]")
    pos = pd.DatetimeIndex(out["date"]).as_unit("ns")
    idx = pd.Index(dates).get_indexer(pos)
    sig_true = np.sqrt(s2[idx])
    m = out["model"].to_numpy()
    np.testing.assert_allclose(out.loc[m == "COMBO+Normal", "sigma"], sig_true[m == "COMBO+Normal"])
    np.testing.assert_allclose(out.loc[m == "GJR+FHS", "sigma"], np.sqrt(1.1) * sig_true[m == "GJR+FHS"])
    np.testing.assert_allclose(out.loc[m == "HAR*+FHS", "sigma"], np.sqrt(0.9) * sig_true[m == "HAR*+FHS"])
    np.testing.assert_allclose(out["r_cc"], daily["r_cc"].to_numpy()[idx])
    assert (out.loc[m == "HAR*+FHS", "har_member"] == "HARQ").all()
    assert out.loc[m != "HAR*+FHS", "har_member"].isna().all()
    assert (out["split"] == "dev").all()
    assert ((out["var99"] > 0) & (out["es975"] >= out["var975"])).all()


def test_risk_frame_start_date_and_combo_fallback():
    daily, fc, dates, _ = _synthetic()
    with_combo = var_es.risk_frame("SPX", daily, fc, "HARQ", start_date=dates[900])
    assert with_combo["date"].min() == dates[900]
    no_combo = var_es.risk_frame("SPX", daily, fc[fc["model"] != "COMBO"], "HARQ", start_date=dates[900])
    pd.testing.assert_frame_equal(with_combo, no_combo)
    # FHS pools are built from all earlier z: trimming does not change measures
    full = var_es.risk_frame("SPX", daily, fc, "HARQ")
    pd.testing.assert_frame_equal(
        with_combo.reset_index(drop=True), full[full["date"] >= dates[900]].reset_index(drop=True)
    )


def test_risk_frame_riskmetrics_is_the_ewma_forecaster(monkeypatch):
    """RiskMetrics = EWMA forecaster (λ = 0.94, 250-return burn-in) re-indexed to the target date, whatever
    the HS window is."""
    daily, fc, dates, _ = _synthetic()
    r = daily["r_cc"].to_numpy()
    ewma = simple.ewma_variance(r, 0.94, 250)  # row i: forecast made at session i for session i+1
    target = pd.Series(ewma[:-1], index=dates[1:]).dropna()
    first = dates[1 + 250]  # r[0] is NaN: the burn-in is r[1..250]

    def check(out: pd.DataFrame, start) -> None:
        rm = out[out["model"] == "RiskMetrics"]
        idx = pd.DatetimeIndex(rm["date"]).as_unit("ns")
        assert idx[0] == start
        s2 = pd.Series(rm["sigma"].to_numpy() ** 2, index=idx)
        both = idx.intersection(target.index)
        assert len(both) >= len(idx) - 1
        np.testing.assert_allclose(s2[both], target[both], rtol=1e-12)
        if start == first:
            assert s2[first] == pytest.approx(np.var(r[1:251], ddof=1), rel=1e-12)

    check(var_es.risk_frame("SPX", daily, fc, "HARQ", common=False), first)
    check(var_es.risk_frame("SPX", daily, fc, "HARQ"), dates[801])
    cfg = copy.deepcopy(C.load())
    cfg["risk"]["hs_window"] = 300  # robustness run: only HS moves
    monkeypatch.setattr(C, "load", lambda: cfg)
    out = var_es.risk_frame("SPX", daily, fc, "HARQ", common=False)
    check(out, first)
    assert out.loc[out["model"] == "HS-250", "date"].min() == dates[301]


def test_risk_frame_split_holdout():
    daily, fc, dates, _ = _synthetic(n=1300)
    shift = pd.Timestamp("2025-10-01") - dates[1100]
    daily = daily.assign(session_date=(daily["session_date"] + shift))
    fc = fc.assign(origin=fc["origin"] + shift)
    out = var_es.risk_frame("SPX", daily, fc, "HARQ")
    hold = out["date"] >= pd.Timestamp("2025-10-01")
    assert hold.any() and (~hold).any()
    assert (out.loc[hold, "split"] == "holdout").all() and (out.loc[~hold, "split"] == "dev").all()


# --------------------------------------------------------------------------------------------- Kupiec
def test_kupiec_hand_computed():
    h = np.zeros(250, dtype=int)
    h[[3, 40, 77, 120, 150, 190, 220, 249]] = 1
    res = bt.kupiec(h, 0.01)
    pi = 8 / 250
    lr = -2 * (242 * math.log(0.99) + 8 * math.log(0.01)) + 2 * (242 * math.log(1 - pi) + 8 * math.log(pi))
    assert res["x"] == 8 and res["T"] == 250 and res["rate"] == pytest.approx(0.032)
    assert res["lr_uc"] == pytest.approx(lr) and res["lr_uc"] == pytest.approx(7.73355, abs=1e-4)
    assert res["p_chi2"] == pytest.approx(0.0054204, abs=1e-6)
    assert res["p_binom"] == pytest.approx(binomtest(8, 250, 0.01).pvalue)


def test_kupiec_zero_breaches_uses_0ln0():
    res = bt.kupiec(np.zeros(250, dtype=int), 0.01)
    assert res["lr_uc"] == pytest.approx(-2 * 250 * math.log(0.99))
    assert np.isfinite(res["p_chi2"]) and res["p_binom"] == pytest.approx(binomtest(0, 250, 0.01).pvalue)


def test_hits_definition():
    np.testing.assert_array_equal(bt.hits([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]), [0, 0, 1])
    with pytest.raises(ValueError):
        bt.hits([1.0, np.nan], [1.0, 1.0])


# --------------------------------------------------------------------------------------------- Christoffersen
def test_christoffersen_counts_and_statistic():
    h = np.array([0, 1, 1, 0, 0, 1, 0])
    res = bt.christoffersen(h, 0.05, mc_reps=2000, seed=SEED)
    assert (res["n00"], res["n01"], res["n10"], res["n11"]) == (1, 2, 2, 1)
    n00, n01, n10, n11 = 1, 2, 2, 1
    p01, p11, pi = n01 / (n00 + n01), n11 / (n10 + n11), (n01 + n11) / 6
    ll1 = n00 * math.log(1 - p01) + n01 * math.log(p01) + n10 * math.log(1 - p11) + n11 * math.log(p11)
    ll0 = (n00 + n10) * math.log(1 - pi) + (n01 + n11) * math.log(pi)
    llp = (n00 + n10) * math.log(0.95) + (n01 + n11) * math.log(0.05)
    assert res["lr_ind"] == pytest.approx(2 * (ll1 - ll0))
    assert res["lr_cc"] == pytest.approx(2 * (ll1 - llp))
    assert 0 <= res["p_ind_mc"] <= 1 and 0 <= res["p_cc_mc"] <= 1


def test_christoffersen_zero_breaches_is_na():
    res = bt.christoffersen(np.zeros(300, dtype=int), 0.01, mc_reps=500, seed=SEED)
    assert res["n00"] == 299 and res["n01"] == res["n10"] == res["n11"] == 0
    for k in ("lr_ind", "lr_cc", "p_ind_mc", "p_cc_mc"):
        assert np.isnan(res[k])


def test_christoffersen_n11_zero_is_finite():
    h = np.zeros(500, dtype=int)
    h[[10, 100, 200, 300, 499]] = 1  # isolated breaches, one at the very end
    res = bt.christoffersen(h, 0.01, mc_reps=2000, seed=SEED)
    assert res["n11"] == 0
    for k in ("lr_ind", "lr_cc", "p_ind_mc", "p_cc_mc"):
        assert np.isfinite(res[k])
    # single breach at t=0: no hit in the transition sample, still finite
    h1 = np.zeros(100, dtype=int)
    h1[0] = 1
    res1 = bt.christoffersen(h1, 0.01, mc_reps=500, seed=SEED)
    assert res1["n10"] == 1 and np.isfinite(res1["lr_ind"]) and np.isfinite(res1["lr_cc"])


def test_christoffersen_consecutive_breaches_raise_lr_ind():
    spread = np.zeros(500, dtype=int)
    spread[np.arange(10) * 50 + 7] = 1
    clustered = np.zeros(500, dtype=int)
    clustered[[50, 51, 150, 151, 250, 251, 350, 351, 450, 451]] = 1
    a = bt.christoffersen(spread, 0.02, mc_reps=5000, seed=SEED)
    b = bt.christoffersen(clustered, 0.02, mc_reps=5000, seed=SEED)
    assert b["lr_ind"] > a["lr_ind"] + 5
    assert b["p_ind_mc"] < 0.01 < a["p_ind_mc"]
    assert b["lr_cc"] > a["lr_cc"]


def test_christoffersen_mc_is_seeded():
    h = (np.random.default_rng(0).random(400) < 0.02).astype(int)
    assert bt.christoffersen(h, 0.01, seed=7, mc_reps=3000) == bt.christoffersen(h, 0.01, seed=7, mc_reps=3000)


# --------------------------------------------------------------------------------------------- DQ
def _markov_hits(T: int, p01: float, p11: float, rng) -> np.ndarray:
    h = np.zeros(T, dtype=int)
    u = rng.random(T)
    for t in range(1, T):
        h[t] = u[t] < (p11 if h[t - 1] else p01)
    return h


def test_dq_rejects_clustered_hits_not_iid():
    rng = np.random.default_rng(SEED)
    T, p = 1500, 0.01
    var = 2.0 + 0.3 * rng.standard_normal(T)
    clustered = _markov_hits(T, 0.004, 0.6, rng)  # stationary rate ≈ 1%
    iid = (rng.random(T) < p).astype(int)
    assert bt.dq_test(clustered, p, var)["p"] < 0.001
    assert bt.dq_test(iid, p, var)["p"] > 0.05


def test_dq_rejects_hits_predictable_from_var():
    rng = np.random.default_rng(SEED + 1)
    T, p = 2000, 0.025
    var = np.exp(0.5 * rng.standard_normal(T))
    prob = np.where(var < np.median(var), 0.045, 0.005)  # unconditional rate 2.5%, conditional not
    h = (rng.random(T) < prob).astype(int)
    assert bt.dq_test(h, p, var)["p"] < 0.01


def test_dq_handles_constant_var():
    rng = np.random.default_rng(3)
    h = (rng.random(800) < 0.01).astype(int)
    res = bt.dq_test(h, 0.01, np.full(800, 2.3))
    assert np.isfinite(res["dq"]) and res["p"] == pytest.approx(chi2.sf(res["dq"], 6))


def test_dq_hand_computed():
    T, p, lags = 30, 0.05, 4
    h = np.zeros(T, dtype=int)
    h[[5, 6, 17, 25]] = 1
    var = 1.5 + 0.1 * np.cos(np.arange(T))
    hit = h - p
    X = np.array([[1.0, hit[t - 1], hit[t - 2], hit[t - 3], hit[t - 4], var[t]] for t in range(lags, T)])
    y = hit[lags:]
    dq = y @ X @ np.linalg.inv(X.T @ X) @ X.T @ y / (p * (1 - p))
    res = bt.dq_test(h, p, var)
    assert res["dq"] == pytest.approx(dq, rel=1e-10)
    assert res["p"] == pytest.approx(chi2.sf(dq, 6), rel=1e-10)


def test_dq_rank_deficient_keeps_six_df():
    """Zero breaches, or breaches only in the last 4 observations, make lag columns constant (rank(X) < 6);
    the degrees of freedom must not follow the rank."""
    rng = np.random.default_rng(SEED)
    T, p = 1190, 0.01
    var = 2.0 + 0.1 * rng.standard_normal(T)
    zero = bt.dq_test(np.zeros(T, dtype=int), p, var)
    assert zero["dq"] == pytest.approx((T - 4) * p / (1 - p))  # Hit = -p lies in span(X): DQ = Hit'Hit/(p(1-p))
    assert zero["p"] == pytest.approx(chi2.sf(zero["dq"], 6))
    ps = []
    for pos in (595, 1187, 1189):  # rank(X) = 6, 4, 2
        h = np.zeros(T, dtype=int)
        h[pos] = 1
        res = bt.dq_test(h, p, var)
        assert res["p"] == pytest.approx(chi2.sf(res["dq"], 6))
        ps.append(res["p"])
    assert min(ps) > 0.05 and max(ps) / min(ps) < 1.5  # ≈ same statistic -> same decision (rank-df: ~19x)


# --------------------------------------------------------------------------------------------- traffic light
@pytest.mark.parametrize(("N", "p", "expected"), [(250, 0.01, (4, 9)), (365, 0.01, (6, 12)), (250, 0.025, (10, 16))])
def test_traffic_light_zones(N, p, expected):
    assert bt.traffic_light_zones(N, p) == expected


def test_zone_and_plus_factor():
    assert [bt.zone(x, 250, 0.01) for x in (0, 4, 5, 9, 10)] == ["green", "green", "yellow", "yellow", "red"]
    assert [bt.plus_factor(x) for x in range(0, 13)] == [0, 0, 0, 0, 0, 0.40, 0.50, 0.65, 0.75, 0.85, 1.0, 1.0, 1.0]


def test_rolling_traffic_light():
    dates = pd.bdate_range("2022-01-03", periods=300)
    h = pd.Series(0, index=dates)
    h.iloc[:5] = 1
    h.iloc[260:270] = 1
    out = bt.rolling_traffic_light(h, window=250, p=0.01)
    assert len(out) == 51 and out.index[0] == dates[249]
    first = out.iloc[0]
    assert (first["exceptions"], first["zone"], first["plus_factor"]) == (5, "yellow", 0.40)
    assert out.loc[dates[250], "zone"] == "green" and out.loc[dates[250], "plus_factor"] == 0.0
    last = out.iloc[-1]
    assert (last["exceptions"], last["zone"], last["plus_factor"]) == (10, "red", 1.0)
    other = bt.rolling_traffic_light(h, window=250, p=0.025)
    assert other["plus_factor"].isna().all() and other.iloc[-1]["zone"] == "green"


def test_uc_power_matches_exact_rejection_probability():
    T, p = 250, 0.01
    reject = np.array([binomtest(x, T, p).pvalue < 0.05 for x in range(T + 1)])
    exact = binom.pmf(np.arange(T + 1), T, 2 * p)[reject].sum()
    assert bt.uc_power(T, p, reps=20000, seed=SEED) == pytest.approx(exact, abs=0.015)
    assert bt.uc_power(250, p, seed=SEED) < bt.uc_power(1000, p, seed=SEED)


def test_backtest_table_on_risk_frame():
    daily, fc, _, _ = _synthetic()
    risk = var_es.risk_frame("SPX", daily, fc, "HARQ")
    tab = bt.backtest_table(risk, mc_reps=300, seed=SEED, window=250)
    assert len(tab) == 12 and set(tab["level"]) == {"99", "97.5"}
    for col in ("x", "T", "lr_uc", "p_binom", "p_cc_mc", "dq", "p_dq", "power_uc", "zone_full", "green_share"):
        assert col in tab.columns
    assert (tab["T"] == 499).all()
    # hits are on the loss L = -r_cc, recomputed independently from the risk frame
    cols = {"99": "var99", "97.5": "var975"}
    on_gains_differs = False
    for row in tab.itertuples():
        g = risk[risk["model"] == row.model].sort_values("date")
        r, v = g["r_cc"].to_numpy(), g[cols[row.level]].to_numpy()
        h = (-r > v).astype(int)
        assert row.x == h.sum()
        assert row.lr_uc == pytest.approx(bt.kupiec(h, 0.01 if row.level == "99" else 0.025)["lr_uc"])
        on_gains_differs |= row.x != int(np.sum(r > v))
    assert on_gains_differs


def test_var_backtests_counts_losses_not_gains():
    r = np.zeros(300)
    r[[10, 50]] = -3.0  # losses beyond VaR -> hits
    r[[100, 200, 250]] = 3.0  # gains beyond VaR -> no hits
    var = 2.0 + 0.01 * np.cos(np.arange(300))
    res = bt.var_backtests(r, var, 0.01, mc_reps=200, seed=SEED, window=250)
    assert res["x"] == 2 and res["T"] == 300


# --------------------------------------------------------------------------------------------- Z2
def test_student_t_es_formula():
    nu = 5.0
    sc, v, e = _std_t(nu)
    q = student_t.ppf(0.025, nu)
    num = -quad(lambda x: x * student_t.pdf(x, nu), -np.inf, q)[0] / 0.025 * sc
    assert e == pytest.approx(num, rel=1e-8) and v == pytest.approx(1.99116, abs=1e-4)


def test_z2_mean_zero_under_true_model_and_negative_when_es_understated():
    rng = np.random.default_rng(SEED)
    nu, T, reps = 5.0, 1000, 300
    sc, v1, e1 = _std_t(nu)
    sigma = np.exp(0.4 * np.sin(np.arange(T) / 30.0))
    var975, es975 = v1 * sigma, e1 * sigma
    z_true, z_low = [], []
    for _ in range(reps):
        r = sigma * sc * rng.standard_t(nu, T)
        z_true.append(es.acerbi_szekely_z2(r, var975, es975))
        z_low.append(es.acerbi_szekely_z2(r, var975, 0.8 * es975))
    assert abs(np.mean(z_true)) < 0.05  # SE of the mean ≈ 0.012
    assert np.mean(z_low) < -0.2 and np.mean(z_low) == pytest.approx(1 - 1 / 0.8, abs=0.05)


def test_z2_pvalue_normal_model():
    rng = np.random.default_rng(SEED)
    T = 1500
    sigma = np.exp(0.3 * np.sin(np.arange(T) / 25.0))
    v, e = var_es.Z975 * sigma, var_es.ES975_NORMAL * sigma
    sampler = es.normal_sampler(sigma)
    r_ok = sigma * rng.standard_normal(T)
    p_ok = es.z2_pvalue(r_ok, v, e, sampler, reps=2000, seed=1)
    assert p_ok > 0.05
    assert p_ok == es.z2_pvalue(r_ok, v, e, sampler, reps=2000, seed=1)
    r_hot = 1.25 * sigma * rng.standard_normal(T)  # the model understates volatility -> ES understated
    assert es.acerbi_szekely_z2(r_hot, v, e) < 0
    assert es.z2_pvalue(r_hot, v, e, sampler, reps=2000, seed=1) < 0.05


def test_samplers_draw_from_the_right_sets():
    rng = np.random.default_rng(0)
    pools = [rng.standard_normal(n) for n in (5, 7, 3)]
    sigma = np.array([1.0, 2.0, 0.5])
    draws = es.fhs_sampler(sigma, pools)(np.random.default_rng(1), 50)
    assert draws.shape == (50, 3)
    for t in range(3):
        assert np.isin(draws[:, t] / sigma[t], pools[t]).all()
    hs = es.hs_sampler(pools)(np.random.default_rng(1), 400)
    for t in range(3):
        assert set(hs[:, t]) == set(pools[t])  # every element reachable, nothing else
    assert es.normal_sampler(sigma)(np.random.default_rng(1), 10).shape == (10, 3)


def test_sampler_for_each_risk_model():
    daily, fc, _, _ = _synthetic()
    risk = var_es.risk_frame("SPX", daily, fc, "HARQ")
    r = var_es.session_returns(daily)
    for model, rows in risk.groupby("model", sort=False):
        rows = rows.sort_values("date")
        sampler = es.sampler_for(model, rows, daily, fc)
        draws = sampler(np.random.default_rng(2), 4)
        assert draws.shape == (4, len(rows))
        p = es.z2_pvalue(rows["r_cc"], rows["var975"], rows["es975"], sampler, reps=200, seed=3)
        assert 0 <= p <= 1
        if model == "HS-250":
            assert np.isin(draws[:, -1], var_es.hs_windows(r, rows["date"].iloc[-1:])[0]).all()
        if model == "COMBO+FHS":
            s2 = var_es.sigma2_by_target(daily, fc, "COMBO")
            pools = var_es.fhs_pools(r, s2, rows["date"])
            assert np.isin(draws[:, -1] / rows["sigma"].iloc[-1], pools[-1]).all()


# --------------------------------------------------------------------------------------------- FZ0
def test_fz0_hand_values_and_domain():
    out = scoring.fz0([-3.0, 1.0], [2.0, 2.0], [2.5, 2.5])
    base = 0.8 + math.log(2.5) - 1
    np.testing.assert_allclose(out, [16.0 + base, base])
    with pytest.raises(ValueError):
        scoring.fz0([0.0], [2.0], [1.5])  # ES < VaR
    with pytest.raises(ValueError):
        scoring.fz0([0.0], [-1.0], [1.0])  # VaR <= 0


def test_fz0_expected_score_minimised_at_truth():
    nu = 5.0
    sc, v, e = _std_t(nu)
    y = sc * np.random.default_rng(SEED).standard_t(nu, 1_000_000)
    ones = np.ones_like(y)

    def score(kv: float, ke: float) -> float:
        return float(scoring.fz0(y, kv * v * ones, ke * e * ones).mean())

    truth = score(1.0, 1.0)
    for kv, ke in [(1.15, 1.0), (0.85, 1.0), (1.0, 1.15), (1.0, 0.85), (1.15, 1.15), (0.85, 0.85), (1.1, 0.9)]:
        assert score(kv, ke) > truth


def test_add_fz0_column():
    daily, fc, _, _ = _synthetic()
    risk = scoring.add_fz0(var_es.risk_frame("SPX", daily, fc, "HARQ"))
    assert np.isfinite(risk["fz0"]).all()


def _fz0_frame(scale: float = 1.0, T: int = 1500, seed: int = SEED) -> pd.DataFrame:
    """Risk rows of one asset with a strongly time-varying true variance: HS-250, the true Normal model
    (named 'COMBO+Normal') and a Normal model that understates sigma by 40% ('bad'); returns scaled by ``scale``."""
    rng = np.random.default_rng(seed)
    n = T + 250
    dates = pd.bdate_range("2015-01-01", periods=n)
    s2 = pd.Series(np.exp(1.2 * np.sin(np.arange(n) / 60.0)), index=dates)
    r = pd.Series(np.sqrt(s2) * rng.standard_normal(n), index=dates)
    models = {
        "HS-250": var_es.historical_sim(r, 250),
        "COMBO+Normal": var_es.normal_from_sigma2(s2),
        "bad": var_es.normal_from_sigma2(0.36 * s2),
    }
    parts = [
        pd.DataFrame(
            {
                "asset": "SPX",
                "model": m,
                "date": x.index,
                "r_cc": scale * r.reindex(x.index).to_numpy(),
                "var975": scale * x["var975"].to_numpy(),
                "es975": scale * x["es975"].to_numpy(),
            }
        )
        for m, x in models.items()
    ]
    return pd.concat(parts, ignore_index=True)


def test_fz0_summary_ranks_true_model_and_is_scale_free():
    risk = _fz0_frame()
    s = scoring.fz0_summary(risk, mcs_reps=500, seed=SEED).set_index("model")
    assert list(s.columns) == [c for c in scoring.FZ0_SUMMARY_COLUMNS if c != "model"]
    assert list(s.index) == ["HS-250", "COMBO+Normal", "bad"] and (s["n"] == 1500).all()
    hs, true, bad = s.loc["HS-250"], s.loc["COMBO+Normal"], s.loc["bad"]
    assert hs["fz0_diff"] == 0 and np.isnan(hs["dm_hln"]) and np.isnan(hs["p_dm"])
    assert true["fz0_diff"] < 0 and true["dm_hln"] < 0 and true["p_dm"] < 0.05
    assert true["mcs_p"] == 1.0 and true["in_90"] and not bad["in_90"] and bad["fz0_diff"] > 0
    # independent recomputation on the common dates
    wide = scoring.fz0_wide(risk)
    rows = risk[(risk["model"] == "COMBO+Normal") & risk["date"].isin(wide.index)]
    assert true["fz0"] == pytest.approx(scoring.fz0(rows["r_cc"], rows["var975"], rows["es975"]).mean())
    ref = dm_test(wide["COMBO+Normal"], wide["HS-250"], 1)
    assert true["dm_hln"] == pytest.approx(ref["dm_hln"]) and true["fz0_diff"] == pytest.approx(ref["mean_diff"])
    # FZ0(c·y, c·v, c·e) = FZ0 + ln c (E[FZ0] = ln ES at the truth): the comparison must not depend on c
    scaled = scoring.fz0_summary(_fz0_frame(scale=0.35), mcs_reps=500, seed=SEED).set_index("model")
    np.testing.assert_allclose(scaled["fz0"], s["fz0"] + math.log(0.35), rtol=1e-9)
    np.testing.assert_allclose(scaled[["fz0_diff", "dm_hln", "p_dm"]], s[["fz0_diff", "dm_hln", "p_dm"]], atol=1e-9)
    np.testing.assert_allclose(scaled["mcs_p"], s["mcs_p"], atol=2 / 500)


def test_fz0_summary_on_risk_frame_and_leaderboard():
    daily, fc, _, _ = _synthetic()
    full = var_es.risk_frame("SPX", daily, fc, "HARQ", common=False)
    common = var_es.risk_frame("SPX", daily, fc, "HARQ")
    wide = scoring.fz0_wide(full)
    assert list(wide.columns) == list(var_es.RISK_MODELS)
    np.testing.assert_array_equal(wide.index.to_numpy(), np.sort(common["date"].unique()))
    two = pd.concat([common, common.assign(asset="BTC")], ignore_index=True)
    s = scoring.fz0_summary(two, mcs_reps=200, seed=SEED)
    assert list(s.columns) == scoring.FZ0_SUMMARY_COLUMNS and len(s) == 12
    assert set(s["asset"]) == {"SPX", "BTC"} and s["mcs_p"].between(0, 1).all()
    tab = bt.backtest_table(two, mc_reps=200, seed=SEED, window=250)
    lb = scoring.risk_leaderboard(s, tab)
    assert list(lb.columns) == scoring.RISK_LEADERBOARD_COLUMNS and len(lb) == 12
    b99 = tab[tab["level"] == "99"].set_index(["asset", "model"])
    for row in lb.itertuples():
        assert row.zone_last == b99.at[(row.asset, row.model), "zone_last"]
        assert row.green_share == b99.at[(row.asset, row.model), "green_share"]
    with pytest.raises(ValueError, match="splits"):
        scoring.fz0_summary(pd.concat([common, common.assign(split="holdout")]))


# --------------------------------------------------------------------------------------------- slow
@pytest.mark.slow
@pytest.mark.parametrize(("T", "p"), [(250, 0.01), (250, 0.025), (1190, 0.01), (1730, 0.01), (1730, 0.025)])
def test_uc_size_within_two_points(T, p):
    rng = np.random.default_rng(SEED)
    paths = rng.random((4000, T)) < p
    rej = np.mean([bt.kupiec(h.astype(int), p)["p_binom"] < 0.05 for h in paths])
    assert abs(rej - 0.05) <= 0.02
    assert abs(bt.uc_power(T, p, reps=20000, seed=SEED, rate=p) - 0.05) <= 0.02


@pytest.mark.slow
def test_christoffersen_mc_size():
    rng = np.random.default_rng(SEED + 5)
    T, p, n = 500, 0.025, 400
    rej_ind, rej_cc = [], []
    for i in range(n):
        h = (rng.random(T) < p).astype(int)
        res = bt.christoffersen(h, p, mc_reps=2000, seed=i)
        rej_ind.append(res["p_ind_mc"] < 0.05)
        rej_cc.append(res["p_cc_mc"] < 0.05)
    assert np.mean(rej_ind) <= 0.08 and np.mean(rej_cc) <= 0.08
