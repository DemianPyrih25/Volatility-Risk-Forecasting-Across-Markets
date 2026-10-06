"""Tests for the LightGBM / MLP forecasters (SPEC §6 refit cadence, §7 models 9–10). Offline, seeded."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch

from volrisk import config as C
from volrisk.models.base import FORECAST_COLUMNS
from volrisk.models.features import VARIANCE_FEATURES, ml_features
from volrisk.models.ml import LGBM, MLP, anchored_walk_forward, mlp_design
from volrisk.targets import build_targets

W_TEST = 300  # small rolling window so the synthetic walk-forward is quick


# ------------------------------------------------------------------------------------------- helpers


def synth_daily(n: int = 1300, asset: str = "BTC", seed: int = 7, start: str = "2019-01-01") -> pd.DataFrame:
    """Gold-like daily rows of one asset with persistent (log-AR(1)) variance; crypto = every calendar day."""
    rng = np.random.default_rng(seed)
    crypto = asset in C.CRYPTO
    dates = pd.date_range(start, periods=n, freq="D" if crypto else "B")
    h = np.zeros(n)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        h[t] = 0.97 * h[t - 1] + 0.2 * eps[t]
    sig2 = (4.0 if crypto else 0.8) * np.exp(h)
    rv = sig2 * rng.chisquare(30, n) / 30
    bv = rv * rng.uniform(0.75, 1.0, n)
    j = np.where(rng.uniform(size=n) < 0.1, rv - bv, 0.0)
    rs_pos = rv * rng.uniform(0.3, 0.7, n)
    gap = np.zeros(n) if crypto else np.sqrt(0.2 * sig2) * rng.standard_normal(n)
    return pd.DataFrame(
        {
            "asset": asset,
            "session_date": dates.astype("datetime64[ms]"),  # what polars Date -> pandas yields
            "rv": rv,
            "bv": bv,
            "j": j,
            "c": rv - j,
            "rs_pos": rs_pos,
            "rs_neg": rv - rs_pos,
            "rq": rv**2 * rng.uniform(1.0, 3.0, n),
            "gap": gap,
            "r_cc": gap + np.sqrt(rv) * rng.standard_normal(n),
            "tv": gap**2 + rv,
        }
    )


def synth_targets(daily: pd.DataFrame, asset: str, horizon: str) -> pd.DataFrame:
    """SPEC §6 targets for one (asset, horizon); incomplete windows -> NaN target, NaT end, split 'dropped'."""
    d = daily["session_date"].to_numpy().astype("datetime64[D]")
    tv = daily["tv"].to_numpy()
    n, n_max, days = len(d), C.n_max(horizon, asset), np.timedelta64(C.horizon_days(horizon), "D")
    dev_end = np.datetime64(C.dev_end(), "D")
    rows = []
    for i in range(n):
        lo = i + 1
        if horizon == "1d":
            hi, complete = min(i + 2, n), i + 1 < n
        else:
            hi = min(int(np.searchsorted(d, d[i] + days, side="right")), lo + n_max)
            complete = d[-1] >= d[i] + days
        if complete and hi > lo:
            we, y = d[hi - 1], tv[lo:hi].sum()
            split = "dev" if we <= dev_end else "dropped"
        else:
            we, y, split = np.datetime64("NaT", "D"), np.nan, "dropped"
        rows.append((asset, horizon, d[i], we, hi - lo, y, split))
    t = pd.DataFrame(rows, columns=["asset", "horizon", "origin", "window_end", "n_t", "y", "split"])
    t["origin"] = t["origin"].astype("datetime64[ms]")
    t["window_end"] = t["window_end"].astype("datetime64[ms]")
    t["ybar"] = t["y"] / t["n_t"].where(t["n_t"] > 0)
    return t[["asset", "horizon", "origin", "window_end", "n_t", "y", "ybar", "split"]]


def holdout_sample(asset: str, start: str, seed: int) -> pd.DataFrame:
    """Synthetic gold rows from ``start`` through the last holdout session (``data_end``)."""
    end = pd.Timestamp(C.data_end())
    n = (end - pd.Timestamp(start)).days + 1 if asset in C.CRYPTO else len(pd.bdate_range(start, end))
    daily = synth_daily(n, asset, seed=seed, start=start)
    assert daily["session_date"].iloc[-1] == end
    return daily


def qlike(y: np.ndarray, f: np.ndarray) -> np.ndarray:
    """QLIKE up to constants on the log link: y·e^{−f} + f."""
    return y * np.exp(-f) + f


def expected_oos_start(daily: pd.DataFrame, targets_by_h: dict, asset: str, window: int) -> int:
    """Brute-force SPEC §6 OOS start: first row with ``window`` training rows (finite features and target,
    ``window_end <= origin``) for every horizon."""
    assert set(targets_by_h) == set(C.HORIZONS)
    x_ok = ml_features(daily, asset).notna().all(axis=1).to_numpy()
    d = daily["session_date"].to_numpy()
    starts = []
    for tg in targets_by_h.values():
        ok = x_ok & tg["ybar"].notna().to_numpy()
        counts = ((tg["window_end"].to_numpy()[None, :] <= d[:, None]) & ok[None, :]).sum(axis=1)
        assert (counts >= window).any()
        starts.append(int(np.argmax(counts >= window)))
    return max(starts)


class _Spy:
    """Mixin recording training labels and which fitted model served each forecast origin."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.sizes: list[int] = []
        self.labels: list[np.ndarray] = []
        self.models: list[object] = []
        self.served: list[int] = []

    def fit(self, X, y, *a, **k):
        m = super().fit(X, y, *a, **k)
        self.sizes.append(len(y))
        self.labels.append(np.array(y, copy=True))
        self.models.append(m)
        return m

    def predict(self, model, X):
        self.served.append(next(i for i, m in enumerate(self.models) if m is model))
        return super().predict(model, X)


class _MeanFit:
    """Training-mean stand-in for the learner: exercises the real walk-forward driver at negligible cost."""

    def fit(self, X, y, *a, **k):
        self.n_fits += 1
        return [float(np.mean(y))]  # a fresh object per fit, so the spy can tell fits apart

    def predict(self, model, X):
        return np.full(len(X), model[0])


class SpyLGBM(_Spy, LGBM):
    pass


class SpyMLP(_Spy, MLP):
    pass


class SpyMean(_Spy, _MeanFit, LGBM):
    pass


def check_schedule(
    m: _Spy, daily: pd.DataFrame, tg: pd.DataFrame, out: pd.DataFrame, asset: str, window: int, refit: int
) -> np.ndarray:
    """SPEC §6 walk-forward invariants of one ``forecast`` call; returns the fitted rows.

    Refits sit on the session grid ``start + m·refit``; every origin is served by the latest grid refit (so a
    fit is never ``refit`` or more sessions old); each fit uses exactly the last ``window`` rows eligible at its
    origin (finite features and target, ``window_end <= origin``).
    """
    d = daily["session_date"].to_numpy()
    start = int(np.searchsorted(d, np.datetime64(m.oos_start)))
    fit_rows = np.searchsorted(d, m.fit_origins)
    assert len(fit_rows) == m.n_fits == len(m.labels) > 0
    assert fit_rows[0] == start and ((fit_rows - start) % refit == 0).all()
    rows = np.searchsorted(d, out["origin"].to_numpy())
    assert rows[0] >= start
    np.testing.assert_array_equal(fit_rows[np.asarray(m.served)], start + (rows - start) // refit * refit)
    ok = ml_features(daily, asset).notna().all(axis=1).to_numpy() & tg["ybar"].notna().to_numpy()
    for lab, j in zip(m.labels, fit_rows):
        elig = ok & (tg["window_end"].to_numpy() <= d[j])
        np.testing.assert_array_equal(lab, tg["ybar"].to_numpy()[elig][-window:])
    return fit_rows


@pytest.fixture(scope="module")
def btc():
    daily = synth_daily(1300, "BTC", seed=7)
    return daily, {h: synth_targets(daily, "BTC", h) for h in C.HORIZONS}


@pytest.fixture(scope="module")
def spx():
    daily = synth_daily(1300, "SPX", seed=11)
    return daily, {h: synth_targets(daily, "SPX", h) for h in C.HORIZONS}


@pytest.fixture(scope="module")
def lgbm_btc(btc):
    """One LightGBM walk-forward per horizon on the BTC sample: {horizon: (spy model, forecasts)}."""
    daily, targets = btc
    runs = {}
    for h in C.HORIZONS:
        m = SpyLGBM(window=W_TEST)
        runs[h] = (m, m.forecast(daily, targets[h], "BTC", h))
    return runs


def _check_frame(out: pd.DataFrame, targets: pd.DataFrame, asset: str, horizon: str, model: str) -> None:
    assert list(out.columns) == FORECAST_COLUMNS
    assert len(out) > 0
    assert (out["asset"] == asset).all() and (out["horizon"] == horizon).all() and (out["model"] == model).all()
    assert np.isfinite(out["F"]).all() and (out["F"] > 0).all()
    assert out["n_t"].dtype == np.int64
    m = out.merge(targets, on="origin", how="left", validate="1:1")
    assert (m["n_t_x"] == m["n_t_y"]).all()
    assert (m["split"] == "dev").all()
    assert out["origin"].is_monotonic_increasing


# ------------------------------------------------------------------------------------------- LightGBM


def test_lgbm_params_match_spec():
    p, n_rounds = LGBM().lgb_params()
    assert n_rounds == 300
    assert p == {
        "objective": "gamma",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 50,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "n_jobs": 4,
        "random_state": C.seed(),
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    rng = np.random.default_rng(0)
    booster = LGBM().fit(rng.normal(size=(200, 3)), rng.gamma(2.0, 1.0, 200))
    # the aliases resolve to the intended native LightGBM parameters
    resolved = booster.model_to_string()
    for line in ("[objective: gamma]", "[num_iterations: 300]", "[learning_rate: 0.03]", "[num_leaves: 15]",
                 "[min_data_in_leaf: 50]", "[feature_fraction: 0.8]", "[lambda_l2: 1]",
                 f"[seed: {C.seed()}]", "[deterministic: 1]", "[force_row_wise: 1]", "[num_threads: 4]"):
        assert line in resolved, line


def test_gamma_derivatives_equal_numerical_qlike_derivatives():
    """LightGBM's gamma objective uses grad 1 − y·e^{−f}, hess y·e^{−f}: the derivatives of y·e^{−f} + f."""
    rng = np.random.default_rng(1)
    y = rng.gamma(2.0, 1.5, 500)
    f = rng.normal(0.5, 1.0, 500)
    h = 1e-4
    g_num = (qlike(y, f + h) - qlike(y, f - h)) / (2 * h)
    h_num = (qlike(y, f + h) - 2 * qlike(y, f) + qlike(y, f - h)) / h**2
    np.testing.assert_allclose(g_num, 1 - y * np.exp(-f), rtol=1e-6, atol=1e-7)
    np.testing.assert_allclose(h_num, y * np.exp(-f), rtol=1e-4, atol=1e-5)


def test_lgbm_gamma_objective_equals_numerical_qlike_objective():
    """Boosting with built-in 'gamma' == boosting with numerically differentiated QLIKE from log(mean y).

    feature_fraction is switched off here: LightGBM draws column subsets from a differently advanced RNG for
    custom objectives, which would make the two (equally valid) ensembles differ.
    """
    rng = np.random.default_rng(2)
    n = 800
    X = rng.normal(size=(n, 5))
    y = np.exp(0.6 * X[:, 0]) * rng.gamma(2.0, 0.5, n)
    h = 1e-4

    def numerical_qlike(preds, ds):
        yy = ds.get_label().astype(float)
        g = (qlike(yy, preds + h) - qlike(yy, preds - h)) / (2 * h)
        hess = (qlike(yy, preds + h) - 2 * qlike(yy, preds) + qlike(yy, preds - h)) / h**2
        return g, hess

    f0 = math.log(y.mean())  # LightGBM's boost-from-average for gamma
    builtin = LGBM(params={"colsample_bytree": 1.0})
    custom = LGBM(params={"colsample_bytree": 1.0, "objective": numerical_qlike})
    p_builtin = builtin.predict(builtin.fit(X, y), X)
    raw = custom.fit(X, y, init_score=np.full(n, f0)).predict(X)  # custom objective -> raw score
    np.testing.assert_allclose(np.exp(raw + f0), p_builtin, rtol=1e-6)
    assert np.corrcoef(np.log(p_builtin), X[:, 0])[0, 1] > 0.8  # and it learns the signal


def test_lgbm_constant_features_predict_mean():
    """With no usable split the gamma model is its QLIKE-optimal constant: exactly mean(ybar)."""
    rng = np.random.default_rng(3)
    y = rng.gamma(2.0, 0.5, 1000)
    m = LGBM()
    pred = m.predict(m.fit(np.ones((1000, 11)), y), np.ones((5, 11)))
    np.testing.assert_allclose(pred, y.mean(), rtol=1e-6)


def test_lgbm_noise_features_predict_about_mean():
    rng = np.random.default_rng(4)
    y = rng.gamma(2.0, 0.5, 1000)
    m = LGBM()
    pred = m.predict(m.fit(rng.normal(size=(1000, 11)), y), rng.normal(size=(2000, 11)))
    assert abs(pred.mean() / y.mean() - 1) < 0.15
    assert abs(np.median(pred) / y.mean() - 1) < 0.2


def test_lgbm_rejects_nonpositive_label():
    X = np.zeros((100, 2))
    with pytest.raises(ValueError, match="> 0"):
        LGBM().fit(X, np.r_[np.ones(99), 0.0])


# ------------------------------------------------------------------------------------------- MLP


def test_mlp_design_transforms():
    daily = synth_daily(80, "SPX", seed=1)
    feat = ml_features(daily, "SPX")
    X = mlp_design(feat)
    assert "dow" not in X.columns
    assert [f"dow_{d}" for d in range(7)] == [c for c in X.columns if c.startswith("dow_")]
    np.testing.assert_array_equal(X.filter(like="dow_").to_numpy().argmax(axis=1), feat["dow"].to_numpy())
    assert (X.filter(like="dow_").sum(axis=1) == 1).all()
    for c in VARIANCE_FEATURES:
        if c in feat.columns:
            np.testing.assert_allclose(X[c], np.log(feat[c] + 1e-8))
    for c in ("r_cc", "r_cc_neg", "j_share"):
        np.testing.assert_array_equal(X[c], feat[c])
    assert "gap2" in X.columns  # SPX/EURUSD only
    assert "gap2" not in mlp_design(ml_features(synth_daily(80, "BTC"), "BTC")).columns


def test_mlp_standardisation_and_bias_init():
    rng = np.random.default_rng(5)
    X = rng.normal(3.0, 2.0, size=(400, 4))
    X[:, 3] = 1.0  # constant column must not blow up the standardisation
    y = rng.gamma(2.0, 0.7, 400)
    m = MLP(epochs=0, n_seeds=2)
    fit = m.fit(X, y)
    np.testing.assert_allclose(fit.mu, X.mean(axis=0))
    np.testing.assert_allclose(fit.sd[:3], X[:, :3].std(axis=0))
    assert fit.sd[3] == 1.0
    for net in fit.nets:
        assert net[-1].bias.item() == pytest.approx(math.log(y.mean()), rel=1e-6)
    sizes = [layer.out_features for layer in fit.nets[0] if isinstance(layer, torch.nn.Linear)]
    assert sizes == [32, 32, 1]
    assert sum(isinstance(layer, torch.nn.SiLU) for layer in fit.nets[0]) == 2
    # predictions are row-wise: the training-window scaling, not the prediction batch, defines the inputs
    p_all = m.predict(fit, X[:50])
    np.testing.assert_allclose(m.predict(fit, X[7:8])[0], p_all[7], rtol=1e-6)


def test_mlp_loss_decreases_and_learns_signal():
    rng = np.random.default_rng(6)
    X = rng.normal(size=(1000, 6))
    y = np.exp(0.7 * X[:, 0]) * rng.gamma(2.0, 0.5, 1000)
    m = MLP(n_seeds=2)
    fit = m.fit(X, y)
    assert fit.losses.shape == (2, 400)
    assert np.all(fit.losses[:, -1] < fit.losses[:, 0] - 0.05)
    assert np.all(fit.losses[:, -50:].mean(axis=1) < fit.losses[:, :50].mean(axis=1))
    # loss is QLIKE: its constant optimum (f = log mean y) is beaten by the fitted model
    assert np.all(fit.losses[:, -1] < np.mean(qlike(y, np.full_like(y, math.log(y.mean())))))
    pred = m.predict(fit, X)
    assert np.corrcoef(np.log(pred), X[:, 0])[0, 1] > 0.9


def test_mlp_constant_features_predict_mean():
    rng = np.random.default_rng(7)
    y = rng.gamma(2.0, 0.5, 1000)
    m = MLP(n_seeds=2)
    pred = m.predict(m.fit(np.ones((1000, 11)), y), np.ones((3, 11)))
    np.testing.assert_allclose(pred, y.mean(), rtol=0.01)


def test_mlp_noise_features_predict_about_mean():
    rng = np.random.default_rng(8)
    y = rng.gamma(2.0, 0.5, 1000)
    m = MLP()  # full SPEC configuration: 400 epochs, 5 seeds
    pred = m.predict(m.fit(rng.normal(size=(1000, 11)), y), rng.normal(size=(2000, 11)))
    assert abs(pred.mean() / y.mean() - 1) < 0.15
    assert abs(np.median(pred) / y.mean() - 1) < 0.25


def test_mlp_seed_average_in_variance_space():
    rng = np.random.default_rng(9)
    X = rng.normal(size=(300, 3))
    y = rng.gamma(2.0, 1.0, 300)
    m = MLP(epochs=30, n_seeds=3)
    fit = m.fit(X, y)
    xt = torch.from_numpy(((X[:10] - fit.mu) / fit.sd).astype(np.float32))
    with torch.no_grad():
        f = np.stack([net(xt).squeeze(1).numpy() for net in fit.nets]).astype(float)
    np.testing.assert_allclose(m.predict(fit, X[:10]), np.exp(f).mean(axis=0), rtol=1e-7)
    assert not np.allclose(f[0], f[1])  # seeds differ: SEED+k
    # each seed reproduces in isolation: seed k of a 3-seed fit == the single-seed fit seeded SEED+k
    solo = MLP(epochs=30, n_seeds=1, seed=C.seed() + 2).fit(X, y)
    np.testing.assert_array_equal(solo.losses[0], fit.losses[2])


def test_mlp_restores_torch_global_state():
    torch.manual_seed(123)
    rng_state = torch.get_rng_state()
    n_threads = torch.get_num_threads()
    det = torch.are_deterministic_algorithms_enabled()
    rng = np.random.default_rng(10)
    MLP(epochs=5, n_seeds=2).fit(rng.normal(size=(50, 3)), rng.gamma(2.0, 1.0, 50))
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert torch.get_num_threads() == n_threads
    assert torch.are_deterministic_algorithms_enabled() == det


# ------------------------------------------------------------------------------------------- walk-forward


@pytest.mark.parametrize("horizon", C.HORIZONS)
def test_lgbm_walk_forward_end_to_end(btc, lgbm_btc, horizon):
    daily, targets = btc
    m, out = lgbm_btc[horizon]
    _check_frame(out, targets[horizon], "BTC", horizon, "LGBM")
    refit = C.refit_every("ml", "BTC")
    assert refit == 90
    # refit cadence: dev origins are contiguous sessions, so a new fit every `refit` origins, each on W rows
    assert m.n_fits == len(m.sizes) == math.ceil(len(out) / refit)
    assert set(m.sizes) == {W_TEST}
    np.testing.assert_array_equal(m.served, np.arange(len(out)) // refit)
    check_schedule(m, daily, targets[horizon], out, "BTC", W_TEST, refit)
    # first origin = the asset's OOS start, set by 1m for every horizon: rv_m needs 30 sessions (row 29) and
    # the 1m window of row s ends 30 days later, so W rows are eligible from row 29 + W + 29 on
    first = daily.index[daily["session_date"] == out["origin"].iloc[0]][0]
    assert first == 29 + W_TEST + 29 == expected_oos_start(daily, targets, "BTC", W_TEST)
    assert out["origin"].iloc[0] == m.oos_start
    # forecasts track the persistent variance
    m_ = out.merge(targets[horizon], on="origin")
    assert np.corrcoef(np.log(m_["F"]), np.log(m_["y"]))[0, 1] > 0.3


def test_oos_start_and_refit_dates_are_per_asset(lgbm_btc):
    """SPEC §6: one OOS start per asset and one refit schedule, whichever horizon is forecast."""
    runs = list(lgbm_btc.values())
    first = {out["origin"].iloc[0] for _, out in runs}
    assert len(first) == 1 and first == {m.oos_start for m, _ in runs}
    grids = [m.fit_origins for m, _ in runs]
    n = min(len(g) for g in grids)  # 1m may skip a last block whose origins all lack a complete window
    assert n >= 5 and all(len(g) - n <= 1 for g in grids)
    for g in grids[1:]:
        np.testing.assert_array_equal(g[:n], grids[0][:n])


def test_lgbm_default_window_and_spx_cadence(spx):
    daily, targets = spx
    m = SpyLGBM()  # W from config (1000)
    out = m.forecast(daily, targets["1d"], "SPX", "1d")
    _check_frame(out, targets["1d"], "SPX", "1d", "LGBM")
    assert set(m.sizes) == {C.window()} == {1000}
    assert C.refit_every("ml", "SPX") == 63
    assert m.n_fits == math.ceil(len(out) / 63)
    np.testing.assert_array_equal(m.served, np.arange(len(out)) // 63)
    check_schedule(m, daily, targets["1d"], out, "SPX", 1000, 63)
    first = daily.index[daily["session_date"] == out["origin"].iloc[0]][0]
    # rv_m (22 sessions) is first finite at row 21: 1d alone would start at 21 + 1000, but the asset's OOS
    # start waits until 1m (window of up to 22 sessions) also has 1000 eligible rows
    assert first == expected_oos_start(daily, targets, "SPX", 1000) > 21 + 1000
    assert len(ml_features(daily, "SPX").columns) == 12  # gap² included for SPX


@pytest.mark.parametrize(("asset", "start"), [("BTC", "2023-10-01"), ("SPX", "2022-06-01")])
def test_holdout_run_keeps_session_cadence(asset, start):
    """SPEC §6: during the holdout run the refits keep the session cadence across the 'dropped' origins
    between dev and holdout — the boundary gap is one ordinary step, no fit is ever ``R`` sessions old — and
    every horizon re-estimates on the same dates."""
    daily = holdout_sample(asset, start, seed=13)
    R = C.refit_every("ml", asset)
    tgs = {h: build_targets(daily, asset, h, C.data_end()) for h in C.HORIZONS}
    grids, starts = {}, set()
    for h, tg in tgs.items():
        m = SpyMean(window=W_TEST)
        out = m.forecast(daily, tg, asset, h)
        assert set(out.merge(tg, on="origin")["split"]) == {"dev", "holdout"}
        grids[h] = check_schedule(m, daily, tg, out, asset, W_TEST, R)
        assert (np.diff(grids[h]) == R).all()  # contiguous grid through the dropped boundary origins
        starts.add(m.oos_start)
    d = daily["session_date"]
    assert starts == {d.iloc[expected_oos_start(daily, tgs, asset, W_TEST)]}
    boundary = np.searchsorted(d.to_numpy(), np.datetime64(C.dev_end()))
    assert grids["1m"][0] < boundary < grids["1m"][-1]
    n = min(len(g) for g in grids.values())
    for g in grids.values():
        np.testing.assert_array_equal(g[:n], grids["1d"][:n])


@pytest.mark.parametrize(
    "make",
    [lambda: LGBM(window=W_TEST), lambda: MLP(window=W_TEST, epochs=40, n_seeds=2)],
    ids=["LGBM", "MLP"],
)
def test_holdout_run_reproduces_dev_forecasts(make):
    """SPEC §11: re-running the walk-forward with the holdout rows reproduces every dev forecast exactly."""
    daily = holdout_sample("BTC", "2023-10-01", seed=17)
    dev = daily[daily["session_date"] < pd.Timestamp(C.holdout_start())].reset_index(drop=True)
    a = make().forecast(dev, build_targets(dev, "BTC", "1m", C.dev_end()), "BTC", "1m")
    tg = build_targets(daily, "BTC", "1m", C.data_end())
    b = make().forecast(daily, tg, "BTC", "1m").merge(tg[["origin", "split"]], on="origin")
    assert (b["split"] == "holdout").sum() > 300
    b_dev = b.loc[b["split"] == "dev", FORECAST_COLUMNS].reset_index(drop=True)
    assert len(a) > 300
    pd.testing.assert_frame_equal(a, b_dev, check_exact=True)


def test_incomplete_window_inside_sample_does_not_freeze_training(btc):
    """Rows without a complete window inside the sample (NaT window_end, NaN target — e.g. a non-finite tv)
    never train, and later refits still use the latest eligible rows."""
    daily, targets = btc
    tg = targets["1w"].copy()
    tg.loc[600:602, "window_end"] = pd.NaT
    tg.loc[600:602, ["y", "ybar"]] = np.nan
    tg.loc[600:602, "split"] = "dropped"
    m = SpyMean(window=W_TEST)
    out = m.forecast(daily, tg, "BTC", "1w")
    fit_rows = check_schedule(m, daily, tg, out, "BTC", W_TEST, C.refit_every("ml", "BTC"))
    assert (fit_rows > 602).sum() >= 3
    assert not out["origin"].isin(daily["session_date"].iloc[600:603]).any()


def test_anchored_walk_forward_edge_cases():
    T = 50
    X = np.ones((T, 2))
    y = np.ones(T)
    days = np.arange(T).astype("datetime64[D]")
    args = (X, y, days + np.timedelta64(1, "D"), days, np.ones(T, bool), 10)
    fit, pred = (lambda X, y: y.mean()), (lambda m, X: np.full(len(X), m))
    out, rows = anchored_walk_forward(*args, 5, None, fit, pred)
    assert np.isnan(out).all() and rows.size == 0  # never enough history
    out, rows = anchored_walk_forward(*args, 5, 12, fit, pred)
    assert rows.tolist() == list(range(12, T, 5)) and np.isnan(out[:12]).all() and (out[12:] == 1).all()
    with pytest.raises(ValueError, match="refit_every"):
        anchored_walk_forward(*args, 0, 12, fit, pred)


def test_lgbm_refit_injectable(btc):
    daily, targets = btc
    m = SpyLGBM(window=W_TEST, refit_every=25)
    out = m.forecast(daily, targets["1w"], "BTC", "1w")
    assert m.n_fits == math.ceil(len(out) / 25)
    np.testing.assert_array_equal(m.served, np.arange(len(out)) // 25)


def test_lgbm_deterministic(btc):
    daily, targets = btc
    a = LGBM(window=W_TEST).forecast(daily, targets["1m"], "BTC", "1m")
    b = LGBM(window=W_TEST).forecast(daily, targets["1m"], "BTC", "1m")
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_lgbm_no_lookahead(btc):
    """Changing data after session k leaves every forecast made at origins before k unchanged (purge)."""
    daily, targets = btc
    k = 900
    shocked = daily.copy()
    cols = ["rv", "bv", "j", "c", "rs_pos", "rs_neg", "tv"]
    shocked.loc[k:, cols] *= 5.0
    shocked.loc[k:, "rq"] *= 25.0
    t_shocked = synth_targets(shocked, "BTC", "1w")
    a = LGBM(window=W_TEST).forecast(daily, targets["1w"], "BTC", "1w")
    b = LGBM(window=W_TEST).forecast(shocked, t_shocked, "BTC", "1w")
    cut = daily["session_date"].iloc[k]
    pd.testing.assert_frame_equal(
        a[a["origin"] < cut].reset_index(drop=True), b[b["origin"] < cut].reset_index(drop=True), check_exact=True
    )
    assert not np.allclose(a.loc[a["origin"] >= cut, "F"], b.loc[b["origin"] >= cut, "F"])


@pytest.mark.parametrize(("asset", "horizon"), [("BTC", "1w"), ("SPX", "1m")])
def test_mlp_walk_forward_end_to_end(btc, spx, asset, horizon):
    daily, targets = (btc if asset == "BTC" else spx)
    m = SpyMLP(window=W_TEST, epochs=40, n_seeds=2)
    out = m.forecast(daily, targets[horizon], asset, horizon)
    _check_frame(out, targets[horizon], asset, horizon, "MLP")
    refit = C.refit_every("ml", asset)
    assert m.n_fits == math.ceil(len(out) / refit)
    assert set(m.sizes) == {W_TEST}
    np.testing.assert_array_equal(m.served, np.arange(len(out)) // refit)
    check_schedule(m, daily, targets[horizon], out, asset, W_TEST, refit)
    again = MLP(window=W_TEST, epochs=40, n_seeds=2).forecast(daily, targets[horizon], asset, horizon)
    pd.testing.assert_frame_equal(out, again, check_exact=True)  # deterministic across instances


@pytest.mark.slow
@pytest.mark.parametrize("cls", [LGBM, MLP])
def test_full_spec_config_beats_unconditional_mean(cls):
    """Full SPEC settings (W=1000, crypto refit 90, 400 epochs × 5 seeds): with persistent variance both
    learners must clearly beat the QLIKE-optimal constant of the same rolling training window."""
    daily = synth_daily(1700, "BTC", seed=21)
    targets = synth_targets(daily, "BTC", "1d")
    m = cls()
    out = m.forecast(daily, targets, "BTC", "1d")
    _check_frame(out, targets, "BTC", "1d", cls.name)
    assert m.n_fits == math.ceil(len(out) / 90)
    # 1d: the W training rows at origin i are i−W..i−1, so the constant forecast is a lagged rolling mean
    bench = targets.assign(F_bench=targets["n_t"] * targets["ybar"].shift(1).rolling(C.window()).mean())
    e = out.merge(bench, on="origin")
    assert e["F_bench"].notna().all()
    ql = lambda F: np.mean(e["y"] / F - np.log(e["y"] / F) - 1)  # noqa: E731
    assert ql(e["F"]) < 0.5 * ql(e["F_bench"])  # observed ≈ 0.15–0.17
