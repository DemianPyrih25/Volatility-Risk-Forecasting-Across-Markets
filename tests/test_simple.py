"""Tests for the RW and EWMA baselines (SPEC §7, models 1–2) on hand-computed series."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from volrisk import config as C
from volrisk.models.base import FORECAST_COLUMNS
from volrisk.models.simple import EWMA, RW, ewma_variance, trailing_mean
from volrisk.targets import build_all_targets, build_targets


def _frame(asset: str, dates, tv=None, r_cc=None) -> pd.DataFrame:
    n = len(dates)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "asset": asset,
            "session_date": pd.DatetimeIndex(dates).astype("datetime64[ms]"),
            "tv": rng.gamma(2.0, 0.5, n) if tv is None else np.asarray(tv, dtype=float),
            "r_cc": rng.standard_normal(n) if r_cc is None else np.asarray(r_cc, dtype=float),
        }
    )


def _last(d: pd.DataFrame) -> date:
    return d["session_date"].iloc[-1].date()


def test_trailing_mean():
    np.testing.assert_allclose(trailing_mean(np.arange(1.0, 7.0), 3), [np.nan, np.nan, 2, 3, 4, 5])
    assert np.isnan(trailing_mean(np.ones(2), 5)).all()


def test_rw_hand_computed_spx():
    d = _frame("SPX", pd.bdate_range("2024-01-08", periods=12), tv=np.arange(1.0, 13.0))
    # 1w (n_max = 5): complete windows for origins up to 2024-01-16 (row 6); 5 rows of history from row 4
    t = build_targets(d, "SPX", "1w", _last(d))
    fc = RW().forecast(d, t, "SPX", "1w")
    assert list(fc.columns) == FORECAST_COLUMNS and (fc["model"] == "RW").all()
    assert list(fc["origin"]) == list(d["session_date"].iloc[4:7])
    np.testing.assert_allclose(fc["F"], [15.0, 20.0, 25.0], rtol=1e-15)  # 5·mean(1..5), 5·mean(2..6), …
    assert (fc["n_t"] == 5).all()
    # 1d: F = tv_t for every origin with a next session
    t1 = build_targets(d, "SPX", "1d", _last(d))
    fc1 = RW().forecast(d, t1, "SPX", "1d")
    np.testing.assert_allclose(fc1["F"], np.arange(1.0, 12.0), rtol=1e-15)


def test_rw_uses_n_t_and_n_max_crypto():
    d = _frame("BTC", pd.date_range("2024-01-01", periods=80))
    tv = d["tv"].to_numpy()
    t = build_targets(d, "BTC", "1m", _last(d))
    fc = RW().forecast(d, t, "BTC", "1m").set_index("origin")
    i = 40
    assert fc.loc[d["session_date"].iloc[i], "F"] == pytest.approx(30 * tv[i - 29 : i + 1].mean(), rel=1e-14)
    assert fc.index.min() == d["session_date"].iloc[29]
    assert fc.index.max() == d["session_date"].iloc[len(d) - 31]


def test_ewma_recursion_hand_computed():
    r = np.array([1.0, -2.0, 3.0, 0.5, -1.0, 2.0])
    s2 = ewma_variance(r, lam=0.94, n_init=3)
    init = 19.0 / 3.0  # sample variance of (1, -2, 3): Σ(x - 2/3)² / 2 = (114/9) / 2
    e3 = 0.94 * init + 0.06 * 0.25
    e4 = 0.94 * e3 + 0.06 * 1.0
    e5 = 0.94 * e4 + 0.06 * 4.0
    assert np.isnan(s2[:3]).all()
    np.testing.assert_allclose(s2[3:], [e3, e4, e5], rtol=1e-14)
    np.testing.assert_allclose(s2[3:], [5.968333333, 5.670233333, 5.570019333], rtol=1e-9)


def test_ewma_init_default_250():
    rng = np.random.default_rng(3)
    r = rng.standard_normal(300) * 1.5
    s2 = ewma_variance(r)
    assert np.isnan(s2[:250]).all() and np.isfinite(s2[250:]).all()
    assert s2[250] == pytest.approx(0.94 * np.var(r[:250], ddof=1) + 0.06 * r[250] ** 2, rel=1e-14)
    # a leading missing r_cc (first session has no previous close) shifts the start by one row
    r2 = r.copy()
    r2[0] = np.nan
    s2b = ewma_variance(r2)
    assert np.isnan(s2b[:251]).all()
    assert s2b[251] == pytest.approx(0.94 * np.var(r2[1:251], ddof=1) + 0.06 * r2[251] ** 2, rel=1e-14)
    # an interior missing return leaves the variance unchanged
    r3 = r.copy()
    r3[270] = np.nan
    s2c = ewma_variance(r3)
    assert s2c[270] == s2c[269] == s2[269]
    # too short to initialise
    assert np.isnan(ewma_variance(r[:250])).all()


def test_ewma_forecast_scaled_by_n_t():
    d = _frame("BTC", pd.date_range("2023-01-01", periods=320))
    s2 = ewma_variance(d["r_cc"].to_numpy(), 0.94, 250)
    m = EWMA()
    assert m.lam == C.load()["risk"]["ewma_lambda"] == 0.94
    for h, n in (("1d", 1), ("1w", 7)):
        t = build_targets(d, "BTC", h, _last(d))
        fc = m.forecast(d, t, "BTC", h)
        assert (fc["model"] == "EWMA").all() and (fc["n_t"] == n).all()
        assert fc["origin"].iloc[0] == d["session_date"].iloc[250]
        assert fc["origin"].iloc[-1] == d["session_date"].iloc[len(d) - 1 - n]
        np.testing.assert_allclose(fc["F"], n * s2[250 : len(d) - n], rtol=1e-15)


def test_first_origin_is_the_asset_oos_start():
    """SPEC §6: RW (history from row n_max−1) and EWMA (from row 250) could start far earlier, but with targets
    from build_all_targets they share the asset's OOS start (BTC from 2018-01-01, W = 1000 -> 2020-11-24)."""
    d = _frame("BTC", pd.date_range("2018-01-01", periods=1130))
    t = build_all_targets(d, _last(d), window=1000)
    for h in C.HORIZONS:
        th = t[t["horizon"] == h].reset_index(drop=True)
        dev = th.loc[th["split"] == "dev", "origin"]
        for M in (RW, EWMA):
            fc = M().forecast(d, th, "BTC", h)
            assert fc["origin"].iloc[0] == pd.Timestamp("2020-11-24"), (M.name, h)
            assert list(fc["origin"]) == list(dev), (M.name, h)
        # the start only masks origins: values are those of the unmasked run
        full = RW().forecast(d, build_targets(d, "BTC", h, _last(d)), "BTC", h)
        assert full["origin"].iloc[0] < pd.Timestamp("2020-11-24")
        pd.testing.assert_frame_equal(
            RW().forecast(d, th, "BTC", h), full[full["origin"] >= pd.Timestamp("2020-11-24")].reset_index(drop=True)
        )


def test_only_evaluation_split_forecast_and_positive():
    full = _frame("ETH", pd.date_range("2024-09-01", "2026-09-30"))
    for h in C.HORIZONS:
        t = build_targets(full, "ETH", h, C.data_end())
        keep = set(t.loc[t["split"].isin(["dev", "holdout"]), "origin"])
        for M in (RW, EWMA):
            fc = M().forecast(full, t, "ETH", h)
            assert len(fc) > 100
            assert set(fc["origin"]) <= keep
            assert (fc["F"] > 0).all() and np.isfinite(fc["F"]).all()
