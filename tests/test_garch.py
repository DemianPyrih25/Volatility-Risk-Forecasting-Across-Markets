"""Tests for the GARCH/GJR walk-forward forecasters (SPEC §6, §7 models 3–4). Synthetic data only."""

from __future__ import annotations

import logging
import time
import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from volrisk import config as C
from volrisk.models import garch as G
from volrisk.models.base import FORECAST_COLUMNS

TRUE = {"omega": 0.02, "alpha": 0.03, "gamma": 0.10, "beta": 0.88, "nu": 6.0}  # persistence 0.96, var 0.5


def simulate_gjr(n: int, seed: int, omega=0.02, alpha=0.03, gamma=0.10, beta=0.88, nu=6.0, burn=500):
    """GJR-GARCH(1,1) with standardised Student-t shocks (percent returns)."""
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, size=n + burn) * np.sqrt((nu - 2) / nu)
    r = np.empty(n + burn)
    s2 = omega / (1 - alpha - gamma / 2 - beta)
    for t in range(n + burn):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + (alpha + gamma * (r[t] < 0)) * r[t] ** 2 + beta * s2
    return r[burn:]


def simulate_igarch(n: int, seed: int, o: int, omega=0.01, nu=4.0, burn=500):
    """(GJR-)GARCH(1,1)-t on the IGARCH boundary ``α + γ/2 + β = 1``: arch's MLE often lands on it."""
    alpha, gamma = (0.05, 0.06) if o else (0.08, 0.0)
    beta = 1.0 - alpha - gamma / 2
    rng = np.random.default_rng(seed)
    z = rng.standard_t(nu, size=n + burn) * np.sqrt((nu - 2) / nu)
    r = np.empty(n + burn)
    s2 = 1.0
    for t in range(n + burn):
        r[t] = np.sqrt(s2) * z[t]
        s2 = omega + (alpha + gamma * (r[t] < 0)) * r[t] ** 2 + beta * s2
    return r[burn:]


def make_daily(r: np.ndarray, asset: str) -> pd.DataFrame:
    """Gold-like rows of one asset; the first session has no previous close (r_cc NaN), like §5.3."""
    n = len(r) + 1
    dates = pd.date_range("2012-01-02", periods=n, freq="D" if asset in C.CRYPTO else "B")
    return pd.DataFrame({"asset": asset, "session_date": dates, "r_cc": np.r_[np.nan, r]})


def make_targets(daily: pd.DataFrame, horizon: str, n: int, dropped=()) -> pd.DataFrame:
    """§6-like targets with a constant n_t; origins whose window runs past the data are 'dropped'."""
    dates = daily["session_date"].to_numpy()
    i = np.arange(len(dates))
    ok = i + n < len(dates)
    window_end = np.full(len(dates), np.datetime64("NaT", "ns")).astype(dates.dtype)
    window_end[ok] = dates[i[ok] + n]
    split = np.where(ok, "dev", "dropped").astype(object)
    split[list(dropped)] = "dropped"
    return pd.DataFrame(
        {
            "asset": daily["asset"].iloc[0],
            "horizon": horizon,
            "origin": dates,
            "window_end": window_end,
            "n_t": n,
            "y": 1.0,
            "ybar": 1.0 / n,
            "split": split,
        }
    )


def manual_F(r: np.ndarray, a: int, s: int, params: np.ndarray, o: int, n: int) -> float:
    """Σ_{k=1..n} σ²_{s+k|s} by hand: arch's backcast and (GJR-)GARCH filter from row ``a`` to ``s``, then
    the analytic k-step recursion ``σ²_{s+k+1|s} = ω + π σ²_{s+k|s}``, ``π = α + γ/2 + β`` (also at π = 1)."""
    omega, alpha, beta = params[0], params[1], params[2 + o]
    gamma = params[2] if o else 0.0
    x = r[a : s + 1]
    w = 0.94 ** np.arange(75)
    backcast = np.sum(w / w.sum() * x[:75] ** 2)
    s2 = omega + (alpha + 0.5 * gamma + beta) * backcast  # σ²_a
    for xt in x:
        s2 = omega + (alpha + gamma * (xt < 0)) * xt**2 + beta * s2  # ends at σ²_{s+1|s}
    pers = alpha + 0.5 * gamma + beta
    total = 0.0
    for _ in range(n):
        total += s2
        s2 = omega + pers * s2
    return float(total)


def refit_index(seen: list, am) -> int:
    """0-based refit number of the arch model ``am`` (one model per block; kept alive so ids stay unique)."""
    for k, m in enumerate(seen):
        if m is am:
            return k
    seen.append(am)
    return len(seen) - 1


@pytest.fixture(autouse=True)
def _fresh_cache():
    G.clear_cache()
    yield
    G.clear_cache()


# --------------------------------------------------------------------------------------------- estimation


def test_gjr_parameter_recovery():
    r = simulate_gjr(2500, seed=11)
    out = G.fit_with_retries(G.make_model(r, 1), len(r), o=1)
    assert out.status == "ok" and out.attempts == 1
    omega, alpha, gamma, beta, nu = out.params
    pers = G.persistence(out.params, 1)
    # tolerances ≈ 3 Monte Carlo SDs at n = 2500 (see the slow test)
    assert abs(pers - 0.96) < 0.05
    assert abs(gamma - TRUE["gamma"]) < 0.07
    assert abs(beta - TRUE["beta"]) < 0.085
    assert 0 <= alpha < 0.08
    assert 4.5 < nu < 9.0
    assert abs(omega / (1 - pers) - 0.5) < 0.16


@pytest.mark.slow
def test_gjr_parameter_recovery_monte_carlo():
    est = []
    for seed in range(20):
        r = simulate_gjr(2500, seed=1000 + seed)
        p = G.fit_with_retries(G.make_model(r, 1), len(r), o=1).params
        est.append([p[2], p[3], p[4], G.persistence(p, 1), p[0] / (1 - G.persistence(p, 1))])
    gamma, beta, nu, pers, uvar = np.mean(est, axis=0)
    assert abs(pers - 0.96) < 0.01
    assert abs(gamma - TRUE["gamma"]) < 0.02
    assert abs(beta - TRUE["beta"]) < 0.02
    assert abs(nu - TRUE["nu"]) < 0.8
    assert abs(uvar - 0.5) < 0.05


def test_arch_last_obs_is_exclusive():
    """arch documents Python slice semantics y[first_obs:last_obs] for the estimation sample."""
    r = simulate_gjr(1500, seed=3)
    am = G.make_model(r, 0)
    res = am.fit(first_obs=200, last_obs=1200, disp="off")
    assert res.nobs == 1000 and res.fit_start == 200 and res.fit_stop == 1200
    ref = G.make_model(r[200:1200], 0).fit(disp="off")
    np.testing.assert_allclose(res.params.to_numpy(), ref.params.to_numpy(), rtol=1e-10)

    after = r.copy()
    after[1200:] *= 5.0  # observation `last_obs` itself and later: not used
    res_after = G.make_model(after, 0).fit(first_obs=200, last_obs=1200, disp="off")
    np.testing.assert_array_equal(res_after.params.to_numpy(), res.params.to_numpy())

    inside = r.copy()
    inside[1199] *= 5.0  # observation last_obs-1: used
    res_inside = G.make_model(inside, 0).fit(first_obs=200, last_obs=1200, disp="off")
    assert not np.allclose(res_inside.params.to_numpy(), res.params.to_numpy())


def test_refit_window_ends_at_origin_inclusive():
    W, R = 400, 50
    r = simulate_gjr(700, seed=5)
    paths = G.walk_forward_variances(r, 1, W, R, 5)
    for j, row in zip(paths.refit_rows[:3], paths.params.itertuples(index=False)):
        ref = G.make_model(r[j - W + 1 : j + 1], 1).fit(disp="off")
        np.testing.assert_allclose(np.asarray(row[4:], dtype=float), ref.params.to_numpy(), rtol=1e-10)


def gjr_params(o: int, pers: float, alpha=0.04, gamma=0.06, omega=0.01, nu=6.0) -> np.ndarray:
    """Parameter vector with persistence ``pers`` (β solved for)."""
    gamma = gamma if o else 0.0
    return np.array([omega, alpha, *([gamma] if o else []), pers - alpha - gamma / 2, nu])


@pytest.mark.parametrize("o", [0, 1])
def test_valid_params_is_arch_parameter_space(o):
    # arch's own constraint is α + γ/2 + β ≤ 1: the IGARCH boundary is feasible
    am = G.make_model(simulate_gjr(300, seed=2), o)
    a, b = am.volatility.constraints()
    boundary = np.array([0.01, 0.25, *([0.5] if o else []), 0.5 if o else 0.75])  # π = 1.0 exactly
    assert np.all(a @ boundary - b >= 0)

    for pers in (0.97, 1.0, 1 + 2e-7, 1 + 4e-6):  # interior, boundary, SLSQP overshoots of converged fits
        assert G.valid_params(gjr_params(o, pers), o), pers
    assert not G.valid_params(gjr_params(o, 1 + 1e-3), o)  # explosive, beyond optimizer tolerance
    bad = {"omega": (0, 0.0), "alpha": (1, -1e-12), "beta": (2 + o, -1e-12), "nu": (-1, 2.0)}
    for name, (i, value) in bad.items():  # box bounds are exact
        p = gjr_params(o, 0.97)
        p[i] = value
        assert not G.valid_params(p, o), name
    p = gjr_params(o, 0.97)
    p[0] = np.nan
    assert not G.valid_params(p, o) and not G.valid_params(None, o) and not G.valid_params(p[:-1], o)
    if o:  # α + γ ≥ 0 is a linear constraint as well
        assert G.valid_params(gjr_params(o, 0.97, alpha=0.04, gamma=-0.04 - 1e-9), o)
        assert not G.valid_params(gjr_params(o, 0.97, alpha=0.04, gamma=-0.05), o)


@pytest.mark.parametrize("excess", [0.0, 2e-7, 4e-6])
def test_converged_boundary_fit_is_used_unchanged(monkeypatch, excess):
    """A converged arch fit on (or within tolerance past) the IGARCH boundary is 'ok' at the first attempt."""
    am = G.make_model(simulate_gjr(600, seed=31), 1)
    boundary = gjr_params(1, 1.0 + excess)
    calls = []

    def fake(am_, last_obs, starting_values, options):
        calls.append(starting_values)
        return SimpleNamespace(params=pd.Series(boundary), loglikelihood=-700.0, convergence_flag=0)

    monkeypatch.setattr(G, "_fit_once", fake)
    for prev in (None, gjr_params(1, 0.96)):
        calls.clear()
        out = G.fit_with_retries(am, 600, 1, prev)
        assert (out.status, out.attempts, len(calls)) == ("ok", 1, 1)
        np.testing.assert_array_equal(out.params, boundary)
        assert out.loglik == -700.0


def test_explosive_converged_fit_falls_back(monkeypatch):
    """Converged but outside arch's space (beyond tolerance): retried, then previous parameters / failed."""
    am = G.make_model(simulate_gjr(600, seed=32), 1)
    explosive = gjr_params(1, 1.03)
    starts = []

    def fake(am_, last_obs, starting_values, options):
        starts.append(starting_values)
        return SimpleNamespace(params=pd.Series(explosive), loglikelihood=-650.0, convergence_flag=0)

    monkeypatch.setattr(G, "_fit_once", fake)
    prev = gjr_params(1, 1.0 + 1e-7, alpha=0.03, gamma=-0.03 - 1e-9)  # an accepted boundary fit
    out = G.fit_with_retries(am, 600, 1, prev)
    assert (out.status, out.attempts, len(starts)) == ("fallback", 3, 3)
    np.testing.assert_array_equal(out.params, prev)
    assert np.isnan(out.loglik)
    np.testing.assert_array_equal(starts[1], G._start_from(prev, 1))  # retry from the previous fit
    assert G.fit_with_retries(am, 600, 1).status == "failed"


@pytest.mark.parametrize("o", [0, 1])
def test_boundary_start_values_are_not_discarded_by_arch(o):
    """arch drops infeasible starting values without tolerance; ``_start_from`` keeps a boundary fit usable."""
    from arch.utility.exceptions import StartingValueWarning

    r = simulate_gjr(600, seed=33)
    am = G.make_model(r, o)
    prev = gjr_params(o, 1.0 + 1e-7, alpha=0.03, gamma=-0.03 - 1e-9 if o else 0.0)
    assert G.valid_params(prev, o)
    with pytest.warns(StartingValueWarning):
        am.fit(disp="off", starting_values=prev)

    sv = G._start_from(prev, o)
    assert G.persistence(sv, o) == pytest.approx(0.999, abs=1e-12)
    assert sv[0] == prev[0] and sv[-1] == prev[-1] and sv[1] + (sv[2] if o else 0.0) >= 0
    a, b = am.volatility.constraints()
    assert np.all(a @ sv[:-1] - b >= 0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", StartingValueWarning)
        am.fit(disp="off", starting_values=sv)
    interior = gjr_params(o, 0.96)
    np.testing.assert_array_equal(G._start_from(interior, o), interior)


@pytest.mark.parametrize("o", [0, 1])
def test_igarch_boundary_mles_are_forecast_unmocked(o):
    """On IGARCH data arch's MLE sits on α+γ/2+β = 1 at many refits. These converged fits are used as estimated
    (a strict ``< 1`` rule turned this exact walk-forward into retries, fallbacks and four 'failed' blocks)."""
    W, R, H = 500, 25, 22
    r = simulate_igarch(1000, seed=5, o=o)
    paths = G.walk_forward_variances(r, o, W, R, H)
    names = G.param_names(o)
    est = paths.params[names].to_numpy(float)
    pers = np.array([G.persistence(p, o) for p in est])
    on_boundary = np.flatnonzero(np.abs(pers - 1) <= G.FEASIBILITY_TOL)
    assert len(on_boundary) >= 10  # the data exercise the boundary
    assert paths.counts["ok"] == len(paths.params) and (paths.params["attempts"] == 1).all()
    assert np.isfinite(paths.variances[W - 1 :]).all() and (paths.variances[W - 1 :] > 0).all()

    for i in on_boundary[:3]:  # arch's own estimate on the refit window, unchanged
        j = int(paths.refit_rows[i])
        ref = G.make_model(r[j - W + 1 : j + 1], o).fit(disp="off")
        np.testing.assert_allclose(est[i], ref.params.to_numpy(), rtol=1e-10)
    i = int(on_boundary[0])
    j = int(paths.refit_rows[i])
    s = min(j + R // 2, len(r) - 1)  # mid-block origin forecast with persistence-1 parameters
    for n in (1, 5, H):
        F = float(np.sum(paths.variances[s, :n]))
        assert F == pytest.approx(manual_F(r, j - W + 1, s, est[i], o, n), rel=1e-10)


# --------------------------------------------------------------------------------------------- walk-forward


@pytest.mark.parametrize("o", [0, 1])
def test_no_lookahead_core(o):
    W, R, H = 500, 25, 22
    r = simulate_gjr(1100, seed=21)
    base = G.walk_forward_variances(r, o, W, R, H)
    rng = np.random.default_rng(99)
    for s in (W - 1 + 2 * R, W - 1 + 2 * R + 7):  # a refit origin and a mid-block origin
        pert = r.copy()
        pert[s + 1 :] = 3.0 * pert[s + 1 :] + rng.normal(0.0, 1.0, len(r) - s - 1)
        alt = G.walk_forward_variances(pert, o, W, R, H)
        # bitwise identical up to and including origin s; different afterwards
        np.testing.assert_array_equal(alt.variances[: s + 1], base.variances[: s + 1])
        assert not np.allclose(alt.variances[s + 1], base.variances[s + 1])
    # a strictly real-time run that never sees data after s reproduces the forecasts at s
    s = W - 1 + 3 * R + 11
    real_time = G.walk_forward_variances(r[: s + 1], o, W, R, H)
    np.testing.assert_array_equal(real_time.variances, base.variances[: s + 1])


def test_no_lookahead_forecast_frame():
    asset, W, R = "SPX", 400, 21
    daily = make_daily(simulate_gjr(800, seed=8), asset)
    targets = {h: make_targets(daily, h, n) for h, n in (("1d", 1), ("1m", 22))}
    base = G.forecast_all_horizons(daily, targets, asset, "GJR", window=W, refit_every=R)
    s_date = daily["session_date"].iloc[W + 2 * R + 5]
    pert = daily.copy()
    later = pert["session_date"] > s_date
    pert.loc[later, "r_cc"] = -2.0 * pert.loc[later, "r_cc"]
    alt = G.forecast_all_horizons(pert, targets, asset, "GJR", window=W, refit_every=R)
    for d in (daily, pert):  # the perturbed run has an IGARCH-boundary MLE (refit row 778): used, no fallback
        paths = G.variance_paths(d, asset, "GJR", window=W, refit_every=R)
        assert paths.counts["ok"] == len(paths.params)
    for h in targets:
        b = base[h][base[h]["origin"] <= s_date]
        a = alt[h][alt[h]["origin"] <= s_date]
        assert len(b) > 2 * R
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        assert not np.allclose(alt[h]["F"], base[h]["F"])


@pytest.mark.parametrize("model,o", [("GARCH", 0), ("GJR", 1)])
def test_cumulative_sum_matches_manual_recursion(model, o):
    asset, W, R = "SPX", 500, 21
    daily = make_daily(simulate_gjr(900, seed=4), asset)
    horizons = {"1d": 1, "1w": 5, "1m": 22}
    targets = {h: make_targets(daily, h, n) for h, n in horizons.items()}
    frames = G.forecast_all_horizons(daily, targets, asset, model, window=W, refit_every=R)
    paths = G.variance_paths(daily, asset, model, window=W, refit_every=R)
    r = daily["r_cc"].to_numpy()
    for s in (W, W + 2 * R + 9, len(daily) - 40):  # first origin (row 0 is NaN), mid-block, late
        j = int(paths.refit_rows[paths.refit_rows <= s].max())
        params = paths.params.loc[paths.params["row"] == j, G.param_names(o)].to_numpy(float)[0]
        origin = daily["session_date"].iloc[s]
        one_step = manual_F(r, j - W + 1, s, params, o, 1)
        for h, n in horizons.items():
            F = frames[h].loc[frames[h]["origin"] == origin, "F"].item()
            assert F == pytest.approx(manual_F(r, j - W + 1, s, params, o, n), rel=1e-10)
            if n > 1:  # a sum of per-step forecasts, not n × the 1-step forecast
                assert F != pytest.approx(n * one_step, rel=1e-6)


@pytest.mark.parametrize("asset", ["SPX", "BTC"])
def test_refit_cadence_from_config(asset):
    W = 300
    daily = make_daily(simulate_gjr(1000, seed=12), asset)  # 1001 rows, row 0 has no return
    R = C.refit_every("garch", asset)
    assert R == (30 if asset in C.CRYPTO else 21)
    paths = G.GARCH(window=W).paths(daily, asset)
    first = W  # row W holds the W-th return
    expected = np.arange(first, len(daily), R)
    np.testing.assert_array_equal(paths.refit_rows, expected)
    assert len(paths.params) == int(np.ceil((len(daily) - first) / R))
    assert paths.counts["ok"] + paths.counts["retry"] == len(expected)
    assert np.isnan(paths.variances[:first]).all() and np.isfinite(paths.variances[first:]).all()


def test_forecast_frame_contract_and_shared_fits(monkeypatch):
    asset, W = "BTC", 300
    daily = make_daily(simulate_gjr(700, seed=13), asset)
    hz = {h: C.n_max(h, asset) for h in C.HORIZONS}
    dropped = range(450, 470)  # e.g. windows crossing the dev/holdout boundary
    targets = {h: make_targets(daily, h, n, dropped=dropped) for h, n in hz.items()}

    calls = []
    real = G.walk_forward_variances
    monkeypatch.setattr(G, "walk_forward_variances", lambda *a, **k: calls.append(1) or real(*a, **k))

    model = G.GJR(window=W)
    frames = {h: model.forecast(daily, targets[h], asset, h) for h in C.HORIZONS}
    assert len(calls) == 1  # one walk-forward shared by 1d/1w/1m
    status = model.paths(daily, asset).params["status"]  # incl. a persistence-1 MLE at row 690
    assert (status == "ok").all() and len(calls) == 1

    for h, n in hz.items():
        f = frames[h]
        assert list(f.columns) == FORECAST_COLUMNS
        assert (f["model"] == "GJR").all() and (f["horizon"] == h).all() and (f["asset"] == asset).all()
        assert (f["F"] > 0).all() and np.isfinite(f["F"]).all()
        assert (f["n_t"] == n).all() and f["n_t"].dtype == np.int64
        tg = targets[h].set_index("origin")
        assert (tg.loc[f["origin"], "split"] == "dev").all()
        assert f["origin"].min() == daily["session_date"].iloc[W]  # first origin with W returns
        expected = targets[h][(targets[h]["split"] == "dev") & (targets[h]["origin"] >= f["origin"].min())]
        assert len(f) == len(expected)
    # cumulative sums are increasing in the horizon at common origins
    m = frames["1d"].merge(frames["1w"], on="origin").merge(frames["1m"], on="origin")
    assert len(m) > 0 and (m["F_x"] < m["F_y"]).all() and (m["F_y"] < m["F"]).all()

    # origins missing from the targets get no forecast; an empty targets frame gives an empty result
    sparse = targets["1w"].iloc[::3]
    f = model.forecast(daily, sparse, asset, "1w")
    assert set(f["origin"]) <= set(sparse["origin"]) and len(f) > 0
    empty = model.forecast(daily, targets["1d"].iloc[:0], asset, "1d")
    assert list(empty.columns) == FORECAST_COLUMNS and empty.empty
    assert len(calls) == 1


def test_convergence_failure_falls_back_to_previous_params(monkeypatch, caplog):
    W, R = 300, 50
    r = simulate_gjr(500, seed=17)
    real = G._fit_once
    seen: list = []

    def flaky(am, last_obs, starting_values, options):
        block = refit_index(seen, am)
        if block == 1:
            raise RuntimeError("optimizer blew up")  # every attempt of the 2nd refit fails
        res = real(am, last_obs, starting_values, options)
        if block == 2 and starting_values is None:  # 3rd refit: default start does not converge
            return SimpleNamespace(params=res.params, loglikelihood=res.loglikelihood, convergence_flag=9)
        return res

    monkeypatch.setattr(G, "_fit_once", flaky)
    with caplog.at_level(logging.WARNING, logger="volrisk.models.garch"):
        daily = make_daily(r, "SPX")
        paths = G.variance_paths(daily, "SPX", "GJR", window=W, refit_every=R, steps=5)
    st = paths.params["status"].tolist()
    assert len(st) == 5 and st[:3] == ["ok", "fallback", "retry"] and set(st[3:]) <= {"ok", "retry"}
    assert paths.counts["fallback"] == 1 and paths.counts["failed"] == 0
    names = G.param_names(1)
    np.testing.assert_array_equal(paths.params.loc[1, names], paths.params.loc[0, names])
    assert np.isfinite(paths.variances[W:]).all() and (paths.variances[W:] > 0).all()
    assert any("fallback 1" in rec.getMessage() for rec in caplog.records)


def test_first_refit_failure_leaves_block_empty(monkeypatch):
    W, R = 300, 50
    r = simulate_gjr(440, seed=19)  # refit positions 299, 349, 399
    real = G._fit_once
    seen: list = []

    def flaky(am, last_obs, starting_values, options):
        if refit_index(seen, am) == 0:
            raise RuntimeError("no luck")
        return real(am, last_obs, starting_values, options)

    monkeypatch.setattr(G, "_fit_once", flaky)
    paths = G.walk_forward_variances(r, 1, W, R, 3)
    status = paths.params["status"].tolist()
    assert len(status) == 3 and status[0] == "failed" and set(status[1:]) <= {"ok", "retry"}
    assert paths.counts["failed"] == 1
    assert np.isnan(paths.variances[: W - 1 + R]).all()
    assert np.isfinite(paths.variances[W - 1 + R :]).all()


def test_runtime_one_asset_3000_obs():
    asset = "SPX"
    daily = make_daily(simulate_gjr(3000, seed=23), asset)
    targets = {h: make_targets(daily, h, C.n_max(h, asset)) for h in C.HORIZONS}
    t0 = time.perf_counter()
    frames = G.forecast_all_horizons(daily, targets, asset, "GJR", window=1000, refit_every=21)
    elapsed = time.perf_counter() - t0
    print(f"\nGJR walk-forward, 3000 obs, W=1000, refit every 21, 3 horizons: {elapsed:.2f}s")
    assert all(len(f) > 1900 for f in frames.values())
    assert elapsed < 60
