"""Tests for the risk suite (``volrisk.risk.suite``): rolling Basel zones and the H4 green-zone measure.

The holdout H4 green measure is the 99% Basel zone of the holdout observations at the actual N (SPEC §9
"Holdout summary uses each asset's actual N"; SPEC §0 evaluates the holdout separately). Rolling windows ending
in the holdout still contain development observations and must not decide it.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from volrisk import config as C
from volrisk.risk import suite
from volrisk.risk.backtests import backtest_table, traffic_light_zones
from volrisk.risk.scoring import RISK_LEADERBOARD_COLUMNS

HOLD = pd.Timestamp("2025-10-01")  # config holdout_start (asserted below)


def _risk(n_dev: int, n_hold: int, breaches: dict[str, tuple[list[int], list[int]]], asset: str = "SPX"):
    """Risk frame (RISK_COLUMNS) with VaR99 = 1 and r_cc = -2 on breach days, 0.1 otherwise, on calendar days
    (365 holdout days = the crypto holdout 2025-10-01 … 2026-09-30).

    ``breaches[model] = (dev positions, holdout positions)``, both counted from the start of their split.
    """
    dev = pd.date_range(end=HOLD - pd.Timedelta(days=1), periods=n_dev)
    hold = pd.date_range(start=HOLD, periods=n_hold)
    assert n_hold == 0 or hold[-1] <= pd.Timestamp(C.data_end())
    dates = dev.append(hold)
    parts = []
    for model, (d_idx, h_idx) in breaches.items():
        r = np.full(len(dates), 0.1)
        r[np.asarray(d_idx, dtype=int)] = -2.0
        r[n_dev + np.asarray(h_idx, dtype=int)] = -2.0
        parts.append(pd.DataFrame({
            "asset": asset, "model": model, "date": dates.astype("datetime64[ms]"), "r_cc": r, "var99": 1.0,
            "var975": 0.8, "es975": 1.1, "sigma": 0.5, "har_member": None,
            "split": np.where(dates < HOLD, "dev", "holdout"),
        }))
    return pd.concat(parts, ignore_index=True)


def _bt99(risk: pd.DataFrame) -> pd.DataFrame:
    bt = backtest_table(risk, mc_reps=200, seed=1, window=250)
    return bt[bt["level"] == "99"].set_index(["asset", "model"])


def test_config_dates():
    assert pd.Timestamp(C.holdout_start()) == HOLD
    assert pd.Timestamp(C.dev_eval_start()) <= HOLD - pd.offsets.BDay(600)


# --------------------------------------------------------------------------------------------- rolling zones
def test_rolling_zones_carry_the_hit_of_the_window_end():
    risk = _risk(400, 0, {"M": ([10, 260, 261, 399], [])})
    z = suite.rolling_zones(risk).sort_values("date").reset_index(drop=True)
    g = risk.sort_values("date").reset_index(drop=True)
    breach = (-g["r_cc"] > g["var99"]).astype(int).to_numpy()
    assert len(z) == 400 - 249
    np.testing.assert_array_equal(pd.DatetimeIndex(z["date"]), pd.DatetimeIndex(g["date"].iloc[249:]))
    np.testing.assert_array_equal(z["hit"].to_numpy(), breach[249:])
    np.testing.assert_array_equal(z["exceptions"].to_numpy(),
                                  pd.Series(breach).rolling(250).sum().dropna().astype(int).to_numpy())


# --------------------------------------------------------------------------------------------- dev: unchanged
def test_time_in_green_dev_is_the_rolling_share_inside_dev():
    risk = _risk(400, 0, {"COMBO+FHS": (list(range(100, 106)), []), "HS-250": ([5, 300], [])})
    tiz = suite.time_in_green(suite.rolling_zones(risk), "dev").set_index(["asset", "model"])
    assert list(tiz.reset_index().columns) == suite.TIME_IN_ZONE_COLUMNS
    bt = _bt99(risk)
    for key in tiz.index:
        assert tiz.at[key, "green"] == pytest.approx(bt.at[key, "green_share"])
        assert tiz.loc[key, ["green", "yellow", "red"]].sum() == pytest.approx(1.0)
    assert 0 < tiz.at[("SPX", "COMBO+FHS"), "green"] < 1  # the fixture mixes zones
    assert (tiz["windows"] == 400 - 249).all() and (tiz["n_obs"] == 250).all() and (tiz["basis"] == "rolling").all()


# --------------------------------------------------------------------------------------------- holdout
def test_time_in_green_holdout_ignores_development_hits():
    """COMBO+FHS: 8 breaches in the last 100 dev days, none in 251 holdout days; HS-250: 4 holdout breaches.
    Rolling windows ending in the holdout would put COMBO+FHS mostly in yellow (dev breaches) — the holdout
    measure must not see them: both are green at the actual N (a tie)."""
    combo_dev = list(range(205, 213))
    risk = _risk(300, 251, {"COMBO+FHS": (combo_dev, []), "HS-250": ([], [10, 50, 90, 130])})
    rolling = suite.rolling_zones(risk)
    tiz = suite.time_in_green(rolling, "holdout").set_index("model")
    assert tiz.loc["COMBO+FHS", ["green", "yellow", "red"]].tolist() == [1.0, 0.0, 0.0]
    assert tiz.loc["HS-250", ["green", "yellow", "red"]].tolist() == [1.0, 0.0, 0.0]  # 4 <= 4 at N = 251
    assert (tiz["n_obs"] == 251).all() and (tiz["windows"] == 1).all() and (tiz["basis"] == "actual_n").all()
    # the old measure (windows ending in the holdout) was driven by the dev breaches
    r = rolling[(rolling["model"] == "COMBO+FHS") & (rolling["date"] >= HOLD)]
    assert (r["zone"] == "green").mean() < 0.5
    # changing only development hits leaves the holdout measure unchanged
    clean = _risk(300, 251, {"COMBO+FHS": ([], []), "HS-250": (list(range(0, 300, 30)), [10, 50, 90, 130])})
    pd.testing.assert_frame_equal(suite.time_in_green(suite.rolling_zones(clean), "holdout").set_index("model"), tiz)


@pytest.mark.parametrize(("n_hold", "x", "expected"), [(246, 4, "green"), (246, 5, "yellow"), (251, 5, "yellow"),
                                                        (365, 6, "green"), (365, 7, "yellow"), (259, 10, "red")])
def test_time_in_green_holdout_uses_the_actual_n_and_equals_zone_full(n_hold, x, expected):
    """Thresholds at the actual N (N = 365: green up to 6, unlike N = 250), defined also for N < 250 where the
    in-holdout rolling green share is NaN; identical to the holdout backtests' ``zone_full`` at 99%."""
    pos = list(np.linspace(3, n_hold - 3, x).astype(int))
    risk = _risk(300, n_hold, {"COMBO+FHS": ([], pos), "HS-250": ([], [])})
    tiz = suite.time_in_green(suite.rolling_zones(risk), "holdout").set_index(["asset", "model"])
    bt = _bt99(risk[risk["split"] == "holdout"])
    key = ("SPX", "COMBO+FHS")
    assert bt.at[key, "x"] == x and tiz.at[key, "n_obs"] == n_hold
    assert bt.at[key, "zone_full"] == expected
    assert tiz.loc[key, ["green", "yellow", "red"]].tolist() == [float(z == expected) for z in ("green", "yellow", "red")]
    if n_hold < 250:
        assert np.isnan(bt.at[key, "green_share"])
    if n_hold == 365:
        assert traffic_light_zones(250, 0.01)[0] == 4 < traffic_light_zones(365, 0.01)[0] == 6


def test_time_in_green_holdout_needs_hits_and_valid_mode():
    risk = _risk(300, 30, {"M": ([], [1])})
    rolling = suite.rolling_zones(risk)
    with pytest.raises(ValueError, match="hit"):
        suite.time_in_green(rolling.drop(columns="hit"), "holdout")
    suite.time_in_green(rolling.drop(columns="hit"), "dev")  # dev does not need the hits
    with pytest.raises(ValueError, match="unknown mode"):
        suite.time_in_green(rolling, "test")


# --------------------------------------------------------------------------------------------- evaluate_risk
def _daily_and_forecasts(n: int = 1300, hold_at: int = 1100, seed: int = 3):
    """Gold-like SPX rows whose session ``hold_at`` is the first holdout session, and 1d forecasts (F = true
    variance times model-specific noise; FHS is scale-free, so a pure rescaling would tie the FHS models)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=HOLD, periods=hold_at + 1).append(pd.bdate_range(HOLD, periods=n - hold_at)[1:])
    s2 = np.exp(0.5 * np.sin(np.arange(n) / 50.0))
    r = np.sqrt(s2) * rng.standard_normal(n)
    r[0] = np.nan
    daily = pd.DataFrame({"asset": "SPX", "session_date": dates.astype("datetime64[ms]"), "r_cc": r})
    origins = dates[300:].astype("datetime64[ms]")
    f = np.r_[s2[301:], 1.0]
    fc = pd.concat([pd.DataFrame({"asset": "SPX", "horizon": "1d", "model": m, "origin": origins, "n_t": 1,
                                  "F": f * k * np.exp(0.2 * rng.standard_normal(len(f)))})
                    for m, k in {"GJR": 1.1, "HARQ": 0.9, "LGBM": 1.0, "HAR": 1.2, "COMBO": 1.0}.items()],
                   ignore_index=True)
    return daily, fc


@pytest.fixture
def fast(monkeypatch):
    cfg = copy.deepcopy(C.load())
    cfg["risk"]["mc_reps"] = 200
    cfg["evaluation"]["mcs_reps"] = 200
    monkeypatch.setattr(C, "load", lambda: cfg)
    monkeypatch.setattr(suite, "es_table", lambda risk, daily_all, forecasts: pd.DataFrame())


def test_evaluate_risk_holdout_publishes_one_actual_n_green_share(fast):
    daily, fc = _daily_and_forecasts()
    risk = suite.build_risk(daily, fc, {"SPX": "HARQ"})
    assert (risk["split"] == "dev").any() and (risk["split"] == "holdout").any()
    t = suite.evaluate_risk(risk, daily, fc, mode="holdout")
    n_hold = int((risk["split"] == "holdout").sum() / risk["model"].nunique())
    assert n_hold < 250
    tiz = t["time_in_zone"].set_index(["asset", "model"])
    bt = t["backtests"]
    bt99 = bt[bt["level"] == "99"].set_index(["asset", "model"])
    lb = t["risk_leaderboard"].set_index(["asset", "model"])
    assert set(tiz.index) == set(bt99.index) == set(lb.index)
    for key in tiz.index:
        assert tiz.at[key, "n_obs"] == bt99.at[key, "T"] == n_hold
        assert tiz.at[key, "green"] == float(bt99.at[key, "zone_full"] == "green")
        assert tiz.at[key, bt99.at[key, "zone_full"]] == 1.0
        assert lb.at[key, "green_share"] == tiz.at[key, "green"]  # one holdout green share, not NaN
        assert np.isnan(bt99.at[key, "green_share"])  # in-holdout rolling share: N < 250
    # rolling zones stay the whole series for display; the pipeline's recomputation gives the same table
    assert t["rolling_zones"]["date"].min() < HOLD
    pd.testing.assert_frame_equal(suite.time_in_green(t["rolling_zones"], "holdout"), t["time_in_zone"])
    assert list(t["risk_leaderboard"].columns) == RISK_LEADERBOARD_COLUMNS


def test_evaluate_risk_dev_green_share_unchanged(fast):
    daily, fc = _daily_and_forecasts()
    risk = suite.build_risk(daily, fc, {"SPX": "HARQ"})
    dev = risk[risk["split"] == "dev"]
    t = suite.evaluate_risk(dev, daily, fc, mode="dev")
    tiz = t["time_in_zone"].set_index(["asset", "model"])
    lb = t["risk_leaderboard"].set_index(["asset", "model"])
    np.testing.assert_allclose(tiz["green"], lb.loc[tiz.index, "green_share"])
    assert (tiz["basis"] == "rolling").all()


def test_evaluate_risk_holdout_refuses_holdout_dates_without_a_complete_window(fast):
    daily, fc = _daily_and_forecasts()
    risk = suite.build_risk(daily, fc, {"SPX": "HARQ"})
    short = risk[risk["date"] >= HOLD - pd.offsets.BDay(100)]  # too little history before the holdout
    with pytest.raises(ValueError, match="complete rolling window"):
        suite.evaluate_risk(short, daily, fc, mode="holdout")
