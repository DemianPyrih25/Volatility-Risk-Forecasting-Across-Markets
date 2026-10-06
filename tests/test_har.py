"""Tests for the HAR family (SPEC §7, models 5–8): purge, coefficient recovery, insanity filter, F > 0."""

from __future__ import annotations

import time
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from volrisk import config as C
from volrisk.models import har
from volrisk.models.base import FORECAST_COLUMNS, align_targets
from volrisk.models.features import har_frame
from volrisk.models.har import HAR, HAR_MODELS, HARCJ, HARQ, SHAR, design, ols_fit, ols_predict, regressors
from volrisk.targets import build_all_targets, build_targets


def _dates(asset: str, start: str, n: int) -> pd.DatetimeIndex:
    if asset in C.CRYPTO:
        return pd.date_range(start, periods=n, freq="D")
    d = pd.bdate_range(start, periods=n + n // 50 + 10)
    d = d[~(((d.month == 12) & (d.day == 25)) | ((d.month == 1) & (d.day == 1)))]
    return d[:n]


def _gold(asset: str, n: int, start: str = "2018-01-01", seed: int = 0) -> pd.DataFrame:
    """Gold-like rows of one asset: persistent log-vol, jumps, semivariances, quarticity, gap, r_cc."""
    rng = np.random.default_rng(seed)
    h = np.zeros(n)
    for i in range(1, n):
        h[i] = 0.97 * h[i - 1] + 0.25 * rng.standard_normal()
    rv = np.exp(h) * rng.gamma(4.0, 0.25, n)
    j = np.where(rng.random(n) < 0.15, rv * rng.uniform(0.1, 0.5, n), 0.0)
    share = rng.uniform(0.3, 0.7, n)
    gap = rng.standard_normal(n) * np.sqrt(0.2 * rv) if asset in ("SPX", "EURUSD") else np.zeros(n)
    return pd.DataFrame(
        {
            "asset": asset,
            "session_date": _dates(asset, start, n).astype("datetime64[ms]"),
            "rv": rv,
            "bv": rv - j,
            "c": rv - j,
            "j": j,
            "rs_pos": rv * share,
            "rs_neg": rv * (1 - share),
            "rq": rv**2 * rng.uniform(1.0, 3.0, n),
            "gap": gap,
            "r_cc": rng.standard_normal(n) * np.sqrt(rv + gap**2),
            "tv": gap**2 + rv,
        }
    )


def _last(d: pd.DataFrame) -> date:
    return d["session_date"].iloc[-1].date()


def test_regressors_and_design():
    assert regressors("HAR", "BTC") == ["rv_d", "rv_w", "rv_m"]
    assert regressors("HAR", "SPX")[-1] == "gap2" and regressors("HARQ", "EURUSD")[-1] == "gap2"
    assert len(regressors("HAR-CJ", "ETH")) == 6
    assert regressors("SHAR", "BTC") == ["rs_pos", "rs_neg", "rv_w", "rv_m"]
    assert regressors("HARQ", "BTC") == ["rv_d", "rq_rv", "rv_w", "rv_m"]
    assert [m.name for m in HAR_MODELS] == ["HAR", "HAR-CJ", "SHAR", "HARQ"]
    d = _gold("BTC", 60)
    X = design(d, "BTC", "HAR")
    assert X.shape == (60, 4) and (X[:, 0] == 1).all()
    assert np.isnan(X[28]).any() and np.isfinite(X[29]).all()  # crypto monthly lag = 30 sessions
    np.testing.assert_allclose(X[40, 3], d["rv"].iloc[11:41].mean(), rtol=1e-12)


def test_har_recovers_coefficients():
    """Simulated HAR(1,5,22) with a gap² term: OLS on the design recovers the parameters."""
    rng = np.random.default_rng(7)
    n, burn = 60_000, 500
    b0, bd, bw, bm, bg, s2_gap = 0.05, 0.35, 0.30, 0.20, 0.30, 0.15
    rv = np.full(n, 1.0)
    gap = rng.standard_normal(n) * np.sqrt(s2_gap)
    eps = rng.gamma(20.0, 0.05, n)  # mean 1
    for t in range(22, n - 1):
        mu = b0 + bd * rv[t] + bw * rv[t - 4 : t + 1].mean() + bm * rv[t - 21 : t + 1].mean() + bg * gap[t] ** 2
        rv[t + 1] = mu * eps[t + 1]
    d = pd.DataFrame(
        {
            "session_date": pd.bdate_range("1780-01-01", periods=n).astype("datetime64[ms]"),
            "rv": rv,
            "c": rv,
            "j": 0.0,
            "rs_pos": rv / 2,
            "rs_neg": rv / 2,
            "rq": rv**2,
            "gap": gap,
            "tv": gap**2 + rv,
        }
    ).iloc[burn:].reset_index(drop=True)
    X = design(d, "SPX", "HAR")
    ybar = np.r_[d["tv"].to_numpy()[1:], np.nan]  # 1d target = next session's tv
    ok = np.isfinite(X).all(axis=1) & np.isfinite(ybar)
    beta = ols_fit(X[ok], ybar[ok]).beta
    # E[tv_{t+1} | t] = (b0 + E gap²) + bd RV_t + bw RV_w + bm RV_m + bg gap_t²
    np.testing.assert_allclose(beta, [b0 + s2_gap, bd, bw, bm, bg], atol=0.05)

    # the walk-forward forecasts track the true conditional mean
    sub = d.iloc[-1600:].reset_index(drop=True)
    sub["asset"] = "SPX"
    tg = build_targets(sub, "SPX", "1d", _last(sub))
    fc = HAR().forecast(sub, tg, "SPX", "1d")
    hf = har_frame(sub, "SPX").set_index(sub["session_date"])
    mu = s2_gap + b0 + bd * hf["rv_d"] + bw * hf["rv_w"] + bm * hf["rv_m"] + bg * hf["gap2"]
    true = mu.reindex(fc["origin"]).to_numpy()
    assert len(fc) > 500
    assert np.corrcoef(fc["F"], true)[0, 1] > 0.97


def test_ols_insanity_filter_unit():
    rng = np.random.default_rng(1)
    X = np.column_stack([np.ones(200), rng.uniform(1, 2, 200)])
    y = 0.5 + 2.0 * X[:, 1] + 0.01 * rng.standard_normal(200)
    fit = ols_fit(X, y)
    inside = ols_predict(fit, np.array([[1.0, 1.5]]))
    assert inside[0] == pytest.approx(0.5 + 3.0, abs=0.01)
    outside = ols_predict(fit, np.array([[1.0, 50.0], [1.0, -50.0]]))
    np.testing.assert_allclose(outside, y.mean(), rtol=1e-15)
    # rank-deficient window (no jumps at all) still fits via the minimum-norm solution
    Xz = np.column_stack([X, np.zeros(200)])
    assert np.isfinite(ols_fit(Xz, y).beta).all() and ols_fit(Xz, y).beta[2] == 0.0


def test_insanity_filter_end_to_end():
    d = _gold("BTC", 300, seed=2)
    W = 100
    i = 250  # origin whose rq is an extreme outlier; rq never enters the target
    d.loc[i, "rq"] = 1e14
    t = build_targets(d, "BTC", "1d", _last(d))
    fc = HARQ(window=W).forecast(d, t, "BTC", "1d").set_index("origin")
    X = design(d, "BTC", "HARQ")
    ybar = t["ybar"].to_numpy()
    rows = np.arange(i - W, i)  # 1d: rows s <= i-1 are eligible; the last W of them
    raw = X[i] @ ols_fit(X[rows], ybar[rows]).beta
    assert not (ybar[rows].min() <= raw <= ybar[rows].max())  # the filter really had to act
    assert fc.loc[d["session_date"].iloc[i], "F"] == pytest.approx(ybar[rows].mean(), rel=1e-12)


def _instrument(monkeypatch, X: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Record the design rows of every OLS fit and prediction (rv_d is continuous -> unique row key)."""
    row_of = {v: k for k, v in enumerate(X[:, 1])}
    events: list[tuple[str, np.ndarray]] = []
    real_fit, real_predict = har.ols_fit, har.ols_predict

    def fit(Xw, yw):
        events.append(("fit", np.array([row_of[v] for v in Xw[:, 1]])))
        return real_fit(Xw, yw)

    def predict(m, Xi):
        events.append(("predict", np.array([row_of[Xi[0, 1]]])))
        return real_predict(m, Xi)

    monkeypatch.setattr(har, "ols_fit", fit)
    monkeypatch.setattr(har, "ols_predict", predict)
    return events


@pytest.mark.parametrize(
    "asset,horizon,gap",
    [("BTC", "1w", False), ("BTC", "1m", False), ("SPX", "1w", False), ("SPX", "1m", False), ("SPX", "1d", False),
     ("SPX", "1w", True)],
)
def test_purge_training_rows(monkeypatch, asset, horizon, gap):
    """Instrumented fit: every training row's window ended by the origin; the set is the last W eligible.

    ``gap``: 9 days without a valid session -> an empty 1w window (n_t = 0) in the middle of the sample,
    which must not freeze the training window afterwards.
    """
    d = _gold(asset, 270, start="2024-01-01", seed=5)
    if gap:
        sd = d["session_date"]
        d = d[(sd < "2024-06-01") | (sd > "2024-06-09")].reset_index(drop=True)
    t = build_targets(d, asset, horizon, _last(d))
    if gap:
        assert t.loc[t["origin"] == pd.Timestamp("2024-05-31"), "n_t"].item() == 0
    W = 60
    X = design(d, asset, "HAR")
    events = _instrument(monkeypatch, X)
    fc = HAR(window=W).forecast(d, t, asset, horizon)
    assert len(fc) > 50
    we = t["window_end"].to_numpy()
    ybar = t["ybar"].to_numpy()
    origin = d["session_date"].to_numpy()
    assert [e[0] for e in events] == ["fit", "predict"] * len(fc)  # daily refit
    for k in range(len(fc)):
        rows, i = events[2 * k][1], int(events[2 * k + 1][1][0])
        assert origin[i] == fc["origin"].iloc[k]
        assert len(rows) == W
        assert (we[rows] <= origin[i]).all()
        eligible = np.flatnonzero((we <= origin[i]) & np.isfinite(ybar) & np.isfinite(X).all(axis=1))
        np.testing.assert_array_equal(rows, eligible[-W:])


def test_row_without_window_end_never_trains(monkeypatch):
    """Inconsistent input: a finite ybar with a NaT window_end. The row's window is not known to have ended,
    so it must never train (it would leak its own target at its origin), and the window must keep rolling."""
    d = _gold("SPX", 270, start="2024-01-01", seed=6)
    t = build_targets(d, "SPX", "1w", _last(d))
    r = 150
    t.loc[r, "window_end"] = pd.NaT
    assert np.isfinite(t.loc[r, "ybar"]) and t.loc[r, "split"] == "dev"
    W = 60
    X = design(d, "SPX", "HAR")
    events = _instrument(monkeypatch, X)
    fc = HAR(window=W).forecast(d, t, "SPX", "1w")
    fits = [rows for kind, rows in events if kind == "fit"]
    assert len(fits) == len(fc) > 100
    assert all(r not in rows for rows in fits)
    we, origin = t["window_end"].to_numpy(), d["session_date"].to_numpy()
    for k in range(len(fc)):
        rows, i = events[2 * k][1], int(events[2 * k + 1][1][0])
        assert len(rows) == W and (we[rows] <= origin[i]).all()
    assert fits[-1].max() > r + 100  # not frozen at the NaT row


def test_first_origin_is_the_asset_oos_start(monkeypatch):
    """SPEC §6 OOS start: with targets from build_all_targets (W = 1000) every HAR-family model and horizon starts
    at the asset's start, BTC from 2018-01-01 -> 2020-11-24, where the 1m window holds exactly rows 29..1028."""
    d = _gold("BTC", 1130, start="2018-01-01", seed=3)
    W = 1000
    t = build_all_targets(d, _last(d), window=W)
    start = pd.Timestamp("2020-11-24")
    for h in C.HORIZONS:
        th = t[t["horizon"] == h].reset_index(drop=True)
        for M in HAR_MODELS if h == "1m" else (HAR,):
            fc = M(window=W).forecast(d, th, "BTC", h)
            assert fc["origin"].iloc[0] == start, (M.name, h)
            assert set(fc["origin"]) == set(th.loc[th["split"] == "dev", "origin"]), (M.name, h)
    # on its own a horizon would start earlier (1d: W rows 29..1028 have ended by row 1029 = 2020-10-26)
    fc1 = HAR(window=W).forecast(d, build_targets(d, "BTC", "1d", _last(d)), "BTC", "1d")
    assert fc1["origin"].iloc[0] == pd.Timestamp("2020-10-26")
    # the first 1m fit uses exactly the rows that make the start
    th = t[t["horizon"] == "1m"].reset_index(drop=True)
    events = _instrument(monkeypatch, design(d, "BTC", "HAR"))
    HAR(window=W).forecast(d, th, "BTC", "1m")
    np.testing.assert_array_equal(events[0][1], np.arange(29, 29 + W))
    assert d["session_date"].iloc[events[1][1][0]] == start


@pytest.mark.parametrize("asset", ["BTC", "SPX"])
def test_all_models_positive(asset):
    d = _gold(asset, 1400, seed=11)
    for h in C.HORIZONS:
        t = build_targets(d, asset, h, _last(d))
        for M in HAR_MODELS:
            fc = M().forecast(d, t, asset, h)
            assert list(fc.columns) == FORECAST_COLUMNS
            assert len(fc) > 250 and (fc["model"] == M.name).all()
            assert np.isfinite(fc["F"]).all() and (fc["F"] > 0).all()
            n_t = t.set_index("origin").loc[fc["origin"], "n_t"].to_numpy()
            np.testing.assert_array_equal(fc["n_t"].to_numpy(), n_t)


def test_f_scales_by_sessions_present_after_a_dropped_session():
    """SPEC §6/§7: ``F = n_t·ŷbar`` with ``n_t`` = sessions present in the window. A session dropped for coverage
    (SPX/EURUSD, §4.3) inside the window lowers ``n_t`` by one. That is the count every other model scales by (RW,
    EWMA, GARCH steps, ML), and the dropped session's price move stays in ``y`` via the next session's gap (§5.2).
    On dev data this is SPX 2023-11-24 only (25 headline 1w/1m origins). Scaling by the calendar count would move
    the SPX QLIKE ratios by at most 0.002."""
    full = _gold("SPX", 270, start="2024-01-01", seed=8)
    drop = pd.Timestamp("2024-07-10")
    d = full[full["session_date"] != drop].reset_index(drop=True)
    assert len(d) == len(full) - 1
    W = 60
    for h in ("1w", "1m"):
        n_full = build_targets(full, "SPX", h, _last(full)).set_index("origin")["n_t"]
        t = build_targets(d, "SPX", h, _last(d))
        fc = HAR(window=W).forecast(d, t, "SPX", h).set_index("origin")
        ybar_hat = pd.Series(HAR(window=W).predict_ybar(d, align_targets(d, t), "SPX"), index=d["session_date"])
        n_t = t.set_index("origin")["n_t"]
        hit = fc.index[(fc.index < drop) & (fc.index >= drop - pd.Timedelta(days=C.horizon_days(h)))]
        assert len(hit) >= 4, h
        np.testing.assert_array_equal(n_t.loc[hit], n_full.loc[hit] - 1)
        np.testing.assert_array_equal(fc["n_t"], n_t.loc[fc.index])
        np.testing.assert_allclose(fc["F"], n_t.loc[fc.index] * ybar_hat.loc[fc.index], rtol=1e-15)


def test_dev_forecasts_reproduced_with_holdout_data():
    """Adding holdout sessions must not change any dev forecast (SPEC §11 reproducibility)."""
    start = (C.data_end() - timedelta(days=899)).isoformat()
    full = _gold("ETH", 900, start=start, seed=4)
    assert _last(full) == C.data_end()
    dev = full[full["session_date"] < pd.Timestamp(C.holdout_start())].reset_index(drop=True)
    for h in ("1d", "1m"):
        td = build_targets(dev, "ETH", h, C.dev_end())
        th = build_targets(full, "ETH", h, C.data_end())
        for M in (HAR, SHAR):
            a = M(window=200).forecast(dev, td, "ETH", h)
            b = M(window=200).forecast(full, th, "ETH", h)
            b = b[b["origin"] <= a["origin"].max()].reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_exact=True)
            assert (th.set_index("origin").loc[b["origin"], "split"] == "dev").all()


@pytest.mark.slow
def test_runtime_full_size_asset():
    d = _gold("BTC", 3200, seed=9)
    t0 = time.perf_counter()
    n = 0
    for h in C.HORIZONS:
        t = build_targets(d, "BTC", h, _last(d))
        for M in (HAR, HARCJ, SHAR, HARQ):
            n += len(M().forecast(d, t, "BTC", h))
    elapsed = time.perf_counter() - t0
    assert n > 20_000
    assert elapsed < 60, elapsed
