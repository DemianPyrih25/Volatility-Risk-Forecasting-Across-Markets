"""Tests for volrisk.evaluation (SPEC §7 IV benchmarks, §8). Offline, synthetic and seeded."""

from __future__ import annotations

import functools
import inspect
import math
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
import polars as pl
import pytest
from arch.bootstrap import MCS, StationaryBootstrap
from scipy import stats
from scipy.signal import lfilter

import volrisk.evaluation.bootstrap as bootstrap_mod
import volrisk.evaluation.mcs as mcs_mod
import volrisk.evaluation.suite as suite
from volrisk import config as C
from volrisk.evaluation.bootstrap import qlike_ratio_ci
from volrisk.evaluation.dm import default_maxlags, dm_test, dm_vs_ref, hln_factor
from volrisk.evaluation.iv import IV_MODELS, iv_benchmarks
from volrisk.evaluation.leaderboard import LEADERBOARD_COLUMNS, common_dates, leaderboard, losses_frame
from volrisk.evaluation.losses import mse, qlike
from volrisk.evaluation.mcs import block_size, run_mcs
from volrisk.evaluation.mz import encompassing, hac_lag, mincer_zarnowitz, mz_log, non_overlapping
from volrisk.models.base import FORECAST_COLUMNS

# ---------------------------------------------------------------------------------------------- losses


def test_qlike_nonnegative_and_minimised_at_target():
    rng = np.random.default_rng(0)
    y = rng.lognormal(0.0, 1.0, 200)
    grid = np.exp(np.linspace(-3, 3, 601))
    L = np.array([qlike(y, y * g) for g in grid])  # (grid, obs)
    assert np.all(L >= 0)
    assert np.allclose(qlike(y, y), 0.0)
    assert np.all(np.argmin(L, axis=0) == np.argmin(np.abs(np.log(grid))))
    # under-prediction is penalised more than over-prediction by the same factor
    assert np.all(qlike(y, y / 2) > qlike(y, y * 2))


def test_mse_and_input_validation():
    assert np.allclose(mse([1.0, 2.0], [2.0, 2.0]), [1.0, 0.0])
    for y, F in [([1.0], [0.0]), ([1.0], [-1.0]), ([0.0], [1.0]), ([np.nan], [1.0]), ([1.0], [np.inf])]:
        with pytest.raises(ValueError):
            qlike(y, F)
        with pytest.raises(ValueError):
            mse(y, F)


# ---------------------------------------------------------------------------------------------- DM


def _ma(rng: np.random.Generator, theta: list[float], T: int) -> np.ndarray:
    e = rng.standard_normal(T + len(theta) - 1)
    return np.convolve(e, theta, mode="valid")


def test_dm_bandwidth_and_hln_factor():
    assert default_maxlags(100, 1) == 4
    assert default_maxlags(1000, 1) == 6  # ⌊4·10^{2/9}⌋ = ⌊6.67⌋
    assert default_maxlags(1000, 22) == 21
    assert default_maxlags(1500, 30) == 29
    assert hln_factor(500, 1) == pytest.approx(math.sqrt(499 / 500))
    assert hln_factor(500, 5) == pytest.approx(math.sqrt((501 - 10 + 20 / 500) / 500))


def test_dm_matches_hand_computed_bartlett_hac():
    rng = np.random.default_rng(1)
    T = 400
    la = rng.standard_normal(T) ** 2 + _ma(rng, [1.0, 0.6, 0.3], T) * 0.2
    lb = rng.standard_normal(T) ** 2
    L = default_maxlags(T, 5)
    r = dm_test(la, lb, n_max=5, maxlags=L)  # explicit maxlags -> Bartlett
    d = la - lb
    u = d - d.mean()
    gam = [u @ u / T] + [u[j:] @ u[:-j] / T for j in range(1, L + 1)]
    lrv = gam[0] + 2 * sum((1 - j / (L + 1)) * gam[j] for j in range(1, L + 1))
    dm = d.mean() / math.sqrt(lrv / T)
    assert r["T"] == T and r["maxlags"] == L
    assert r["mean_diff"] == pytest.approx(d.mean(), rel=1e-12)
    assert r["dm"] == pytest.approx(dm, rel=1e-10)
    assert r["dm_hln"] == pytest.approx(dm * hln_factor(T, 5), rel=1e-10)
    assert r["pvalue"] == pytest.approx(2 * stats.t.sf(abs(r["dm_hln"]), T - 1), rel=1e-10)
    assert dm_test(la, lb, n_max=5, maxlags=10)["maxlags"] == 10
    assert r["kernel"] == "bartlett"


def test_dm_overlapping_default_uses_truncated_uniform_kernel():
    rng = np.random.default_rng(11)
    T = 400
    la = rng.standard_normal(T) ** 2 + _ma(rng, [1.0] * 5, T) * 0.2
    lb = rng.standard_normal(T) ** 2
    r = dm_test(la, lb, n_max=5)
    d = la - lb
    u = d - d.mean()
    lrv = u @ u / T + 2 * sum(u[j:] @ u[:-j] / T for j in range(1, 5))
    assert r["kernel"] == "uniform" and r["maxlags"] == 4
    assert r["dm"] == pytest.approx(d.mean() / math.sqrt(lrv / T), rel=1e-10)
    assert dm_test(la, lb, n_max=1)["kernel"] == "bartlett"


def test_dm_overlapping_falls_back_to_bartlett_with_documented_lags():
    # d_t = 0.2 + (−1)^t + 0.3·e_t: γ0 + 2γ1 ≈ 0.09 − 1 < 0, so the 1-lag uniform estimate (n_max = 2) is not
    # positive and the Bartlett fallback with max(n_max − 1, ⌊4(T/100)^{2/9}⌋) lags must be used.
    rng = np.random.default_rng(12)
    T, n_max = 400, 2
    lb = rng.exponential(1.0, T)
    la = lb + 0.2 + (-1.0) ** np.arange(T) + 0.3 * rng.standard_normal(T)
    d = la - lb
    u = d - d.mean()
    assert u @ u / T + 2 * (u[1:] @ u[:-1] / T) < 0  # precondition: uniform LRV not positive
    L = default_maxlags(T, n_max)
    assert L == max(n_max - 1, math.floor(4 * (T / 100) ** (2 / 9))) == 5
    gam = [u @ u / T] + [u[j:] @ u[:-j] / T for j in range(1, L + 1)]
    lrv = gam[0] + 2 * sum((1 - j / (L + 1)) * gam[j] for j in range(1, L + 1))
    r = dm_test(la, lb, n_max=n_max)
    assert r["kernel"] == "bartlett" and r["maxlags"] == L
    assert r["dm"] == pytest.approx(d.mean() / math.sqrt(lrv / T), rel=1e-10)
    assert r["dm_hln"] == pytest.approx(r["dm"] * hln_factor(T, n_max), rel=1e-12)
    assert r["dm"] == pytest.approx(dm_test(la, lb, n_max=n_max, maxlags=L)["dm"], rel=1e-12)


def test_dm_detects_a_real_difference():
    rng = np.random.default_rng(2)
    T = 500
    lb = rng.exponential(1.0, T)
    la = lb + 0.25 + _ma(rng, [1.0, 0.5], T) * 0.5
    r = dm_test(la, lb, n_max=1)
    assert r["mean_diff"] > 0 and r["dm_hln"] > 0 and r["pvalue"] < 0.001
    r2 = dm_test(lb, la, n_max=1)
    assert r2["dm_hln"] == pytest.approx(-r["dm_hln"]) and r2["pvalue"] == pytest.approx(r["pvalue"])


def test_dm_identical_losses_and_validation():
    x = np.random.default_rng(3).exponential(1.0, 100)
    r = dm_test(x, x.copy(), n_max=1)
    assert r["dm"] == 0.0 and r["pvalue"] == 1.0
    with pytest.raises(ValueError):
        dm_test(x, x[:-1], n_max=1)
    with pytest.raises(ValueError):
        dm_test(np.r_[x[:-1], np.nan], x, n_max=1)


def test_dm_vs_ref_frame():
    rng = np.random.default_rng(4)
    wide = pd.DataFrame({"HAR": rng.exponential(1, 300), "GJR": rng.exponential(1.3, 300)})
    out = dm_vs_ref(wide, "HAR", n_max=1)
    assert list(out["model"]) == ["GJR"] and (out["ref"] == "HAR").all()
    assert out.loc[0, "dm_hln"] == pytest.approx(dm_test(wide["GJR"], wide["HAR"], 1)["dm_hln"])


# Size under H0 (E d = 0). n_max=1: iid differential. n_max=5: MA(4) with geometrically decaying weights
# 0.3^j. Flat MA(n_max-1) weights mimic the overlap of cumulative windows; the truncated uniform kernel keeps
# the test close to nominal there, where Bartlett with the same bandwidth was ≈10–12%.
_SIZE_DGPS = {1: [1.0], 5: [0.3**j for j in range(5)]}


def _dm_size(theta: list[float], n_max: int, T: int = 500, sims: int = 2000, seed: int = 20261002) -> float:
    rng = np.random.default_rng(seed)
    zeros = np.zeros(T)
    return float(np.mean([dm_test(_ma(rng, theta, T), zeros, n_max)["pvalue"] < 0.05 for _ in range(sims)]))


@pytest.mark.slow
@pytest.mark.parametrize("n_max", [1, 5])
def test_dm_size_close_to_nominal(n_max):
    size = _dm_size(_SIZE_DGPS[n_max], n_max)
    assert 0.03 <= size <= 0.07, size


@pytest.mark.slow
@pytest.mark.parametrize("n_max,T", [(5, 500), (22, 1000)])
def test_dm_size_flat_ma_overlap(n_max, T):
    size = _dm_size([1.0] * n_max, n_max, T=T)
    assert 0.03 <= size <= 0.08, size


# ---------------------------------------------------------------------------------------------- MCS


def test_block_size_rule():
    assert block_size(1000, 1) == 10
    assert block_size(1001, 1) == 11
    assert block_size(125, 1) == 5
    assert block_size(27, 1) == 3
    assert block_size(1000, 22) == 22
    assert all(block_size(T, 1) == math.ceil(round(T ** (1 / 3), 9)) for T in range(1, 3000))


def _mcs_losses(T: int = 600) -> pd.DataFrame:
    """Three equally good models and a GARCH that is 0.4 worse on average."""
    rng = np.random.default_rng(5)
    common = rng.exponential(1.0, T)
    losses = pd.DataFrame({m: common + 0.5 * rng.exponential(1.0, T) for m in ("HAR", "HARQ", "SHAR")})
    losses["GARCH"] = common + 0.5 * rng.exponential(1.0, T) + 0.4
    return losses


def _spy(monkeypatch: pytest.MonkeyPatch, module, name: str) -> list[dict]:
    """Wrap ``module.name`` (an arch class) to record its bound constructor arguments, defaults applied."""
    real = getattr(module, name)
    sig = inspect.signature(real)
    calls: list[dict] = []

    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append(dict(bound.arguments))
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return calls


def test_mcs_excludes_worse_model_and_keeps_equal_models():
    losses = _mcs_losses()
    out = run_mcs(losses, n_max=1, reps=1000, seed=7)
    assert list(out.columns) == ["model", "pvalue", "in_90", "in_75"]
    assert list(out["model"]) == list(losses.columns)
    o = out.set_index("model")
    assert o.loc["GARCH", "pvalue"] < 0.01 and not o.loc["GARCH", "in_90"]
    assert o.loc[["HAR", "HARQ", "SHAR"], "in_90"].all()
    assert o["pvalue"].max() == 1.0
    assert (out["in_90"] == (out["pvalue"] >= 0.10)).all() and (out["in_75"] == (out["pvalue"] >= 0.25)).all()
    # deterministic under a fixed seed
    pd.testing.assert_frame_equal(out, run_mcs(losses, n_max=1, reps=1000, seed=7))


@pytest.mark.parametrize(("n_max", "expected_block"), [(1, 9), (22, 22)])  # ⌈600^{1/3}⌉ = 9
def test_mcs_passes_spec_settings_to_arch(monkeypatch, n_max, expected_block):
    losses = _mcs_losses()
    calls = _spy(monkeypatch, mcs_mod, "MCS")
    run_mcs(losses, n_max=n_max, reps=50)
    run_mcs(losses, n_max=n_max, reps=50, seed=7)
    assert len(calls) == 2
    default_seed, explicit_seed = calls
    assert default_seed["block_size"] == block_size(len(losses), n_max) == expected_block
    assert default_seed["method"] == "R" and default_seed["bootstrap"] == "stationary"
    assert default_seed["size"] == 0.10 and default_seed["reps"] == 50
    assert default_seed["seed"] == C.seed() and explicit_seed["seed"] == 7
    pd.testing.assert_frame_equal(default_seed["losses"], losses)
    params = inspect.signature(run_mcs).parameters
    assert params["reps"].default == 5000 and params["size"].default == 0.10 and params["seed"].default is None


def test_mcs_matches_direct_arch_call_with_spec_arguments():
    losses = _mcs_losses()
    T, n_max = len(losses), 5
    out = run_mcs(losses, n_max=n_max, reps=300)
    ref = MCS(losses, 0.10, reps=300, block_size=max(n_max, math.ceil(T ** (1 / 3))), method="R",
              bootstrap="stationary", seed=C.seed())
    ref.compute()
    expected = ref.pvalues["Pvalue"].reindex(losses.columns).to_numpy()
    np.testing.assert_allclose(out["pvalue"].to_numpy(), expected, rtol=1e-12, atol=0)


def test_mcs_edge_cases():
    x = pd.DataFrame({"HAR": np.random.default_rng(6).exponential(1, 50)})
    out = run_mcs(x, n_max=1, reps=100, seed=1)
    assert out.loc[0, "pvalue"] == 1.0 and out.loc[0, "in_90"] and out.loc[0, "in_75"]
    with pytest.raises(ValueError):
        run_mcs(pd.DataFrame({"A": [1.0, np.nan, 2.0], "B": [1.0, 1.0, 1.0]}), n_max=1, reps=10, seed=1)


# ---------------------------------------------------------------------------------------------- MZ / encompassing


def _latent_logvar(rng: np.random.Generator, T: int, phi: float = 0.98, sd: float = 0.5) -> np.ndarray:
    h = np.empty(T)
    h[0] = 0.0
    eps = rng.standard_normal(T) * sd * math.sqrt(1 - phi**2)
    for t in range(1, T):
        h[t] = phi * h[t - 1] + eps[t]
    return h


def test_mincer_zarnowitz_unbiased_and_biased_forecast():
    rng = np.random.default_rng(8)
    T = 3000
    F = np.exp(_latent_logvar(rng, T))
    y = F * rng.gamma(5.0, 1 / 5.0, T)  # E[y | F] = F
    r = mincer_zarnowitz(y, F, lag=10)
    assert set(r) == {"a", "b", "se_a", "se_b", "wald", "p_wald"}
    assert abs(r["b"] - 1) < 0.1 and abs(r["a"]) < 0.1
    assert r["p_wald"] > 0.05
    biased = mincer_zarnowitz(y, 1.5 * F, lag=10)
    assert biased["b"] == pytest.approx(r["b"] / 1.5) and biased["p_wald"] < 1e-6


def test_mz_log_slope_one():
    rng = np.random.default_rng(9)
    T = 3000
    F = np.exp(_latent_logvar(rng, T))
    y = F * rng.gamma(5.0, 1 / 5.0, T)
    r = mz_log(y, F, lag=10)
    assert abs(r["b"] - 1) < 0.1 and r["p_wald"] > 0.05
    assert r["a"] < 0  # E log(eta) < 0 for mean-one noise
    assert mz_log(y, F**0.5, lag=10)["p_wald"] < 1e-6
    with pytest.raises(ValueError):
        mz_log(np.r_[y[:-1], 0.0], F, lag=10)


def test_encompassing_detects_information_beyond_iv():
    rng = np.random.default_rng(10)
    T = 2000
    h = _latent_logvar(rng, T)
    y = np.exp(h) * rng.gamma(3.0, 1 / 3.0, T)
    iv = np.exp(h + 0.4 * rng.standard_normal(T))
    informative = np.exp(h + 0.4 * rng.standard_normal(T))  # independent second signal of h
    noise = np.exp(0.5 * rng.standard_normal(T))
    r = encompassing(y, iv, informative, lag=60)
    assert set(r) == {"b", "c", "se_c", "t_c", "p_c"}
    assert r["c"] > 0 and r["p_c"] < 0.01 and r["t_c"] == pytest.approx(r["c"] / r["se_c"])
    r0 = encompassing(y, iv, noise, lag=60)
    assert r0["p_c"] > 0.05 and r0["b"] > 0.5


def test_non_overlapping():
    df = pd.DataFrame({"origin": pd.date_range("2020-01-01", periods=100, freq="D"), "x": range(100)})
    out = non_overlapping(df.sample(frac=1.0, random_state=0), 22)
    assert list(out["x"]) == [0, 22, 44, 66, 88]
    assert len(non_overlapping(df, 30)) == 4


def test_hac_lag_and_lag_guard():
    assert hac_lag(30) == 60 and hac_lag(22) == 44
    assert hac_lag(30, overlapping=False) == 0 and hac_lag(22, overlapping=False) == 0
    with pytest.raises(ValueError):
        hac_lag(0)
    # a non-overlapping crypto subsample (every 30th of ~1700 origins) has T = 57 < 2·n_max = 60
    rng = np.random.default_rng(15)
    T = 57
    F = np.exp(rng.standard_normal(T))
    y = F * rng.gamma(5.0, 0.2, T)
    iv = F * np.exp(0.3 * rng.standard_normal(T))
    fits = {
        "mz": lambda lag: mincer_zarnowitz(y, F, lag),
        "mz_log": lambda lag: mz_log(y, F, lag),
        "encompassing": lambda lag: encompassing(y, iv, F, lag),
    }
    for fit in fits.values():
        for bad in (hac_lag(30), T, -1):
            with pytest.raises(ValueError, match="HAC lag"):
                fit(bad)
        with pytest.warns(RuntimeWarning, match="T/4"):
            fit(15)  # 15 > 57/4
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            fit(14)
            fit(hac_lag(30, overlapping=False))


def _overlap_1m(rng: np.random.Generator, T: int, h: int = 30, phi: float = 0.97, sd: float = 0.25,
                burn: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Overlapping sums over (t, t+h] of daily variance ``exp(x)·χ²₁`` with AR(1) ``x``.

    Returns ``(y, F)`` with ``F = E[y | x_t]`` exactly, so every MZ / encompassing H0 holds.
    """
    x = lfilter([1.0], [1.0, -phi], sd * rng.standard_normal(T + h + burn))[burn:]
    cs = np.r_[0.0, np.cumsum(np.exp(x) * rng.chisquare(1, x.size))]
    t = np.arange(T)
    k = np.arange(1, h + 1)
    var_k = sd**2 * (1 - phi ** (2 * k)) / (1 - phi**2)
    return cs[t + h + 1] - cs[t + 1], np.exp(np.outer(x[:T], phi**k) + 0.5 * var_k).sum(1)


@functools.lru_cache(maxsize=1)
def _mz_rejection_rates(T: int = 1700, h: int = 30, sims: int = 400, seed: int = 20261002) -> dict[str, float]:
    """Rejection rates at 5% under H0: full overlapping sample with ``hac_lag(h)`` and the non-overlapping
    subsample (every h-th origin, T = 57) with ``hac_lag(h, overlapping=False)``. Encompassing uses IV = F and
    a model forecast ``F·exp(0.3 z)`` with no extra information (c = 0)."""
    rng = np.random.default_rng(seed)
    idx = np.arange(0, T, h)
    full, sub = hac_lag(h), hac_lag(h, overlapping=False)
    rej: dict[str, int] = {}
    for _ in range(sims):
        y, F = _overlap_1m(rng, T, h)
        model = F * np.exp(0.3 * rng.standard_normal(T))
        tests = {
            "levels_full": mincer_zarnowitz(y, F, full)["p_wald"],
            "log_full": mz_log(y, F, full)["p_wald"],
            "enc_full": encompassing(y, F, model, full)["p_c"],
            "levels_sub": mincer_zarnowitz(y[idx], F[idx], sub)["p_wald"],
            "log_sub": mz_log(y[idx], F[idx], sub)["p_wald"],
            "enc_sub": encompassing(y[idx], F[idx], model[idx], sub)["p_c"],
        }
        for key, p in tests.items():
            rej[key] = rej.get(key, 0) + int(p < 0.05)
    return {key: n / sims for key, n in rej.items()}


@pytest.mark.slow
def test_mz_and_encompassing_size_on_overlapping_1m_targets():
    r = _mz_rejection_rates()
    assert 0.02 <= r["enc_full"] <= 0.10, r  # H3 test: ≈ 6%
    assert 0.02 <= r["enc_sub"] <= 0.12, r  # lag 0 on the subsample: ≈ 7–9% (lag 60 gave ≈ 37%)
    assert 0.02 <= r["log_sub"] <= 0.12, r
    assert r["log_full"] <= 0.16, r  # mildly oversized (≈ 11%), documented in mz.py


@pytest.mark.slow
@pytest.mark.xfail(strict=True, reason="levels MZ Wald (χ², Bartlett HAC lag 2·n_max) rejects ≈ 35% of true H0 "
                   "on overlapping 1m variance targets (≈ 23% on the lag-0 non-overlapping subsample); "
                   "reported as descriptive, see mz.py")
def test_mz_levels_size_on_overlapping_1m_targets():
    r = _mz_rejection_rates()
    assert 0.03 <= r["levels_full"] <= 0.10, r


# ---------------------------------------------------------------------------------------------- IV benchmarks


def _iv_inputs(n: int = 2000, seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crypto-like daily origins with 30-day windows; the last 30 windows are incomplete (dropped)."""
    rng = np.random.default_rng(seed)
    origins = [date(2018, 1, 1) + timedelta(days=i) for i in range(n)]
    complete = np.arange(n) < n - 30
    targets = pd.DataFrame(
        {
            "asset": "BTC",
            "horizon": "1m",
            "origin": origins,
            "window_end": [o + timedelta(days=30) if c else None for o, c in zip(origins, complete)],
            "n_t": 30,
            "y": np.where(complete, rng.lognormal(3.0, 0.5, n), np.nan),
            "split": np.where(complete, "dev", "dropped"),
        }
    )
    has_iv = (np.arange(n) % 7 != 3) & ~((np.arange(n) >= 500) & (np.arange(n) < 560))
    implied = pd.DataFrame(
        {
            "asset": "BTC",
            "origin": [o for o, k in zip(origins, has_iv) if k],
            "iv": 0.0,
            "iv_var_30d": rng.lognormal(3.2, 0.3, int(has_iv.sum())),
            "source": "DVOL",
        }
    )
    return targets, implied


def _b_hat(fc: pd.DataFrame, implied: pd.DataFrame) -> pd.Series:
    cal = fc[fc["model"] == "IV-cal"].set_index("origin")["F"]
    iv = implied.set_index("origin")["iv_var_30d"]
    return cal / iv.reindex(cal.index)


def test_iv_raw_never_forward_fills():
    targets, implied = _iv_inputs()
    fc = iv_benchmarks(targets, implied)
    assert list(fc.columns) == FORECAST_COLUMNS
    assert set(fc["model"]) == {"IV", "IV-cal"}
    raw = fc[fc["model"] == "IV"].set_index("origin")
    expected = set(implied["origin"]) & set(targets.loc[targets["split"] == "dev", "origin"])
    assert set(raw.index) == expected
    assert np.allclose(raw["F"], implied.set_index("origin")["iv_var_30d"].reindex(raw.index))
    assert set(fc.loc[fc["model"] == "IV-cal", "origin"]) <= expected
    assert (fc["n_t"] == 30).all() and (fc["horizon"] == "1m").all() and (fc["asset"] == "BTC").all()


def test_iv_cal_respects_min_and_max_calibration_windows():
    targets, implied = _iv_inputs()
    fc = iv_benchmarks(targets, implied, cal_min=250, cal_max=1000)
    b = _b_hat(fc, implied)
    m = targets.merge(implied[["origin", "iv_var_30d"]], on="origin", how="inner")
    m = m[m["y"].notna()].sort_values("origin")
    ratio = (m["y"] / m["iv_var_30d"]).to_numpy()
    we = m["window_end"].to_numpy()

    raw_origins = sorted(fc.loc[fc["model"] == "IV", "origin"])
    n_elig = {t: int(np.sum(we <= t)) for t in raw_origins}
    # an IV-cal forecast exists exactly when >= cal_min calibration origins have completed windows
    assert set(b.index) == {t for t in raw_origins if n_elig[t] >= 250}
    first = min(b.index)
    assert n_elig[first] >= 250 and max(n_elig[t] for t in raw_origins if t < first) < 250
    # brute-force mean of the last <= 1000 eligible ratios, early (< 1000) and late (> 1000 available)
    late = [t for t in b.index if n_elig[t] > 1000]
    assert late, "fixture must exercise the cal_max cap"
    for t in [first, sorted(b.index)[300], late[0], late[-1]]:
        k = n_elig[t]
        assert b[t] == pytest.approx(ratio[max(0, k - 1000) : k].mean(), rel=1e-12)


def test_iv_cal_uses_only_past_windows():
    targets, implied = _iv_inputs()
    base = _b_hat(iv_benchmarks(targets, implied), implied)
    t0 = sorted(base.index)[400]
    future = pd.to_datetime(targets["window_end"]) > pd.Timestamp(t0)
    pert = targets.copy()
    pert.loc[future & pert["y"].notna(), "y"] *= 5.0
    after = _b_hat(iv_benchmarks(pert, implied), implied)
    early = [t for t in base.index if t <= t0]
    pd.testing.assert_series_equal(base.loc[early], after.loc[early])
    later = [t for t in base.index if t > t0 + timedelta(days=31)]
    assert (after.loc[later] > base.loc[later]).all()


def test_iv_benchmarks_validation():
    targets, implied = _iv_inputs(n=400)
    with pytest.raises(ValueError):
        iv_benchmarks(targets.assign(horizon="1w"), implied)
    with pytest.raises(ValueError):
        iv_benchmarks(targets, implied.assign(asset="ETH"))
    # too short for calibration: only raw IV rows
    fc = iv_benchmarks(targets, implied, cal_min=500, cal_max=1000)
    assert set(fc["model"]) == {"IV"}


# ---------------------------------------------------------------------------------------------- bootstrap CI


def test_qlike_ratio_ci_covers_one_for_equal_models():
    rng = np.random.default_rng(12)
    T = 500
    common = rng.exponential(1.0, T)
    lm, lr = common + rng.exponential(0.5, T), common + rng.exponential(0.5, T)
    ratio, lo, hi = qlike_ratio_ci(lm, lr, n_max=1, reps=2000, seed=3)
    assert ratio == pytest.approx(lm.mean() / lr.mean())
    assert lo < 1.0 < hi and lo < ratio < hi
    assert (ratio, lo, hi) == qlike_ratio_ci(lm, lr, n_max=1, reps=2000, seed=3)
    _, lo2, _ = qlike_ratio_ci(lr * 1.5 + 0.1, lr, n_max=1, reps=2000, seed=3)
    assert lo2 > 1.0


def test_qlike_ratio_ci_resamples_jointly():
    lr = np.random.default_rng(13).exponential(1.0, 300)
    ratio, lo, hi = qlike_ratio_ci(2.0 * lr, lr, n_max=5, reps=500, seed=1)
    assert ratio == pytest.approx(2.0) and lo == pytest.approx(2.0) and hi == pytest.approx(2.0)


def _ci_losses(T: int = 300) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(16)
    lr = rng.exponential(1.0, T)
    return lr + rng.exponential(0.3, T), lr


@pytest.mark.parametrize(("n_max", "expected_block"), [(1, 7), (22, 22)])  # ⌈300^{1/3}⌉ = 7
def test_qlike_ratio_ci_passes_spec_settings_to_arch(monkeypatch, n_max, expected_block):
    lm, lr = _ci_losses()
    calls = _spy(monkeypatch, bootstrap_mod, "StationaryBootstrap")
    qlike_ratio_ci(lm, lr, n_max=n_max, reps=50)
    qlike_ratio_ci(lm, lr, n_max=n_max, reps=50, seed=9)
    assert len(calls) == 2
    default_seed, explicit_seed = calls
    assert default_seed["block_size"] == block_size(len(lm), n_max) == expected_block
    assert default_seed["seed"] == C.seed() and explicit_seed["seed"] == 9
    # joint resampling: both series go into the same bootstrap as positional data
    args = default_seed["args"]
    assert len(args) == 2 and np.array_equal(args[0], lm) and np.array_equal(args[1], lr)
    assert not default_seed["kwargs"]
    params = inspect.signature(qlike_ratio_ci).parameters
    assert params["reps"].default == 5000 and params["level"].default == 0.90 and params["seed"].default is None


def test_qlike_ratio_ci_matches_direct_arch_call_with_spec_arguments():
    lm, lr = _ci_losses()
    T, n_max, reps = len(lm), 5, 400
    ratio, lo, hi = qlike_ratio_ci(lm, lr, n_max=n_max, reps=reps)
    bs = StationaryBootstrap(max(n_max, math.ceil(T ** (1 / 3))), lm, lr, seed=C.seed())
    draws = bs.apply(lambda a, b: np.array([a.mean() / b.mean()]), reps).ravel()
    assert ratio == pytest.approx(lm.mean() / lr.mean(), rel=1e-12)
    assert [lo, hi] == pytest.approx(np.quantile(draws, [0.05, 0.95]), rel=1e-12)  # same draws up to rounding


# ---------------------------------------------------------------------------------------------- leaderboard


def _lb_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(14)
    n = 300
    origins = [date(2024, 1, 1) + timedelta(days=i) for i in range(n)]
    split = np.array(["dev"] * 200 + ["holdout"] * 90 + ["dropped"] * 10)
    y = rng.lognormal(0.0, 0.7, n)
    y[-1] = np.nan
    targets = pd.DataFrame(
        {"asset": "BTC", "horizon": "1d", "origin": origins, "window_end": origins, "n_t": 1, "y": y, "split": split}
    )
    rows = []
    for model, scale, start in (("HAR", 1.0, 0), ("GARCH", 1.4, 50), ("LGBM", 1.1, 0)):
        F = np.where(np.isnan(y), 1.0, y) * rng.lognormal(0.0, 0.3 * scale, n)
        rows.append(
            pd.DataFrame(
                {"asset": "BTC", "horizon": "1d", "model": model, "origin": origins[start:], "n_t": 1,
                 "F": F[start:], "split": "WRONG"}
            )
        )
    forecasts = pd.concat(rows, ignore_index=True)
    forecasts["origin"] = pd.to_datetime(forecasts["origin"])  # different dtype than targets on purpose
    return forecasts, targets


def test_losses_frame_join_and_split():
    forecasts, targets = _lb_inputs()
    lf = losses_frame(forecasts, targets)
    assert list(lf.columns) == ["asset", "horizon", "model", "origin", "split", "n_t", "y", "F", "qlike", "mse"]
    assert set(lf["split"]) == {"dev", "holdout"}  # dropped and unobserved targets are not evaluated
    assert len(lf) == 290 + 240 + 290
    assert np.allclose(lf["qlike"], qlike(lf["y"], lf["F"])) and np.allclose(lf["mse"], (lf["y"] - lf["F"]) ** 2)
    # polars input gives the same frame
    pd.testing.assert_frame_equal(lf, losses_frame(pl.from_pandas(forecasts), pl.from_pandas(targets)))
    bad = forecasts.assign(n_t=2)
    with pytest.raises(ValueError):
        losses_frame(bad, targets)


def test_common_dates_and_leaderboard():
    forecasts, targets = _lb_inputs()
    lf = losses_frame(forecasts, targets)
    dev = lf[lf["split"] == "dev"]
    wide = common_dates(dev, ["HAR", "GARCH", "LGBM"])
    assert list(wide.columns) == ["HAR", "GARCH", "LGBM"] and len(wide) == 150
    assert wide.index.min() == pd.Timestamp("2024-02-20")  # first GARCH origin (day 50)
    assert len(common_dates(dev, ["HAR", "LGBM"])) == 200
    with pytest.raises(KeyError):
        common_dates(dev, ["HAR", "MLP"])

    lb = leaderboard(lf, ref="HAR")
    assert list(lb.columns) == LEADERBOARD_COLUMNS
    assert len(lb) == 6
    d = lb[lb["split"] == "dev"].set_index("model")
    assert (d["n"] == 150).all()
    assert d.loc["HAR", "qlike_ratio"] == pytest.approx(1.0)
    assert d.loc["GARCH", "qlike"] == pytest.approx(wide["GARCH"].mean())
    assert d.loc["LGBM", "qlike_ratio"] == pytest.approx(wide["LGBM"].mean() / wide["HAR"].mean())
    assert d.loc["HAR", "mse"] == pytest.approx(common_dates(dev, ["HAR", "GARCH", "LGBM"], "mse")["HAR"].mean())
    h = lb[lb["split"] == "holdout"]
    assert (h["n"] == 90).all()

    sub = leaderboard(lf, ref="HAR", models=["HAR", "LGBM"])
    assert set(sub["model"]) == {"HAR", "LGBM"} and (sub.loc[sub["split"] == "dev", "n"] == 200).all()
    no_ref = leaderboard(lf, ref="HAR", models=["GARCH", "LGBM"])
    assert no_ref["qlike_ratio"].isna().all()


def test_leaderboard_keeps_iv_benchmarks_out_unless_requested():
    forecasts, targets = _lb_inputs()
    iv = forecasts[forecasts["model"] == "HAR"].iloc[100:160].assign(model="IV-cal")
    lf = losses_frame(pd.concat([forecasts, iv], ignore_index=True), targets)
    default = leaderboard(lf, ref="HAR")
    assert "IV-cal" not in set(default["model"])
    assert (default.loc[default["split"] == "dev", "n"] == 150).all()  # IV subsample did not shrink it
    with_iv = leaderboard(lf, ref="IV-cal", models=["HAR", "IV-cal"])
    d = with_iv[with_iv["split"] == "dev"].set_index("model")
    assert (d["n"] == 60).all() and d.loc["IV-cal", "qlike_ratio"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------------------------- suite.evaluate


_SUITE_SPEC = {"SPX": ("2019-01-02", "2019-06-03"), "BTC": ("2020-11-25", "2021-12-28")}  # (OOS start, IV start)


def _suite_inputs(holdout_from: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """1m forecasts/targets for SPX (IV from mid-2019, i.e. before ``dev_eval_start``) and BTC (OOS from late
    2020, IV from 2021-12-28). Origins from ``holdout_from`` on are labelled ``holdout`` (synthetic data)."""
    rng = np.random.default_rng(17)
    tg, fc = [], []
    for asset, (start, iv_start) in _SUITE_SPEC.items():
        origins = pd.date_range(start, "2022-12-30", freq="D" if asset in C.CRYPTO else "B")
        T, n_t = len(origins), C.n_max("1m", asset)
        level = n_t * np.exp(0.5 * np.sin(np.arange(T) / 40.0))
        split = np.where(origins >= pd.Timestamp(holdout_from), "holdout", "dev") if holdout_from else "dev"
        tg.append(pd.DataFrame({"asset": asset, "horizon": "1m", "origin": origins,
                                "window_end": origins + pd.Timedelta(days=30), "n_t": n_t,
                                "y": level * rng.gamma(4.0, 0.25, T), "split": split}))
        for model, noise in (("HAR", 0.3), ("GJR", 0.45), ("COMBO", 0.25), ("IV", 0.35), ("IV-cal", 0.3)):
            keep = origins >= pd.Timestamp(iv_start) if model in IV_MODELS else np.ones(T, bool)
            fc.append(pd.DataFrame({"asset": asset, "horizon": "1m", "model": model, "origin": origins[keep],
                                    "n_t": n_t, "F": (level * np.exp(noise * rng.standard_normal(T)))[keep]}))
    return pd.concat(fc, ignore_index=True), pd.concat(tg, ignore_index=True)


@pytest.fixture
def fast_eval_cfg(monkeypatch):
    monkeypatch.setattr(suite, "_eval_cfg", lambda: {"mcs_size": 0.10, "mcs_reps": 50, "boot_reps": 50})


def _n_iv_origins(asset: str, lo: pd.Timestamp | None = None) -> int:
    """Number of the asset's IV origins (every model has a forecast there) from ``lo`` on."""
    o = pd.date_range(_SUITE_SPEC[asset][1], "2022-12-30", freq="D" if asset in C.CRYPTO else "B")
    return int((o >= lo).sum()) if lo is not None else len(o)


def test_evaluate_iv_tables_use_each_assets_full_dev_iv_subsample(fast_eval_cfg):
    """SPEC §8: IV-inclusive QLIKE/DM/MCS on the IV subsample (SPX: full OOS, incl. pre-2021 origins);
    encompassing (H3) and MZ stay on the headline window (SPEC §0)."""
    fc, tg = _suite_inputs()
    out = suite.evaluate(fc, tg, mode="dev")
    start = pd.Timestamp(C.dev_eval_start())
    n_full = {a: _n_iv_origins(a) for a in _SUITE_SPEC}
    n_head = {a: _n_iv_origins(a, lo=start) for a in _SUITE_SPEC}
    assert n_full["SPX"] > n_head["SPX"] and n_full["BTC"] == n_head["BTC"]

    assert (out["losses"]["origin"] >= start).all()
    ivl = out["iv_leaderboard"]
    assert set(ivl["model"]) == {"HAR", "GJR", "COMBO", "IV", "IV-cal"}
    assert ivl.groupby("asset")["n"].unique().map(list).to_dict() == {a: [n] for a, n in n_full.items()}
    assert ivl.loc[ivl["model"] == "HAR", "qlike_ratio"].eq(1.0).all()
    assert out["iv_dm"].groupby("asset")["T"].unique().map(list).to_dict() == {a: [n] for a, n in n_full.items()}
    assert (out["iv_dm"]["ref"] == "IV-cal").all()
    assert out["iv_mcs"].groupby("asset")["T"].unique().map(list).to_dict() == {a: [n] for a, n in n_full.items()}
    assert set(out["iv_mcs"]["model"]) == {"HAR", "GJR", "COMBO", "IV", "IV-cal"}

    enc = out["encompassing"]
    assert set(enc["model"]) == {"HAR", "GJR", "COMBO"}
    assert enc.groupby("asset")["T"].unique().map(list).to_dict() == {a: [n] for a, n in n_head.items()}
    mz = out["mz"].set_index(["asset", "model"])
    assert mz.loc[("SPX", "IV"), "T"] == n_head["SPX"] and mz.loc[("SPX", "IV-cal"), "T"] == n_head["SPX"]
    # the headline leaderboard (no IV) is unchanged: every SPX origin from dev_eval_start
    lb = out["leaderboard"].set_index(["asset", "model"])
    assert lb.loc[("SPX", "HAR"), "n"] == pd.date_range(start, "2022-12-30", freq="B").size


def test_evaluate_pre2021_robustness_table_is_spx_eurusd_only(fast_eval_cfg):
    """SPEC §6: 'Pre-2021 OOS (SPX/EURUSD) is a robustness table' — the ~37 late-2020 crypto origins are left out."""
    fc, tg = _suite_inputs()
    out = suite.evaluate(fc, tg, mode="dev")
    pre = out["leaderboard_pre2021"]
    start = pd.Timestamp(C.dev_eval_start())
    assert set(pre["asset"]) == {"SPX"}
    assert "IV" not in set(pre["model"]) and "IV-cal" not in set(pre["model"])
    assert (pre["n"] == int(pd.date_range("2019-01-02", start - pd.Timedelta(days=1), freq="B").size)).all()
    # crypto-only development data: no robustness table at all
    btc = suite.evaluate(fc[fc["asset"] == "BTC"], tg[tg["asset"] == "BTC"], mode="dev")
    assert "leaderboard_pre2021" not in btc


def test_evaluate_holdout_iv_tables_use_holdout_split_only(fast_eval_cfg):
    fc, tg = _suite_inputs(holdout_from="2022-01-03")
    out = suite.evaluate(fc, tg, mode="holdout")
    lo = pd.Timestamp("2022-01-03")
    n_hold = {a: _n_iv_origins(a, lo=lo) for a in _SUITE_SPEC}
    ivl = out["iv_leaderboard"]
    assert (ivl["split"] == "holdout").all()
    assert ivl.groupby("asset")["n"].unique().map(list).to_dict() == {a: [n] for a, n in n_hold.items()}
    assert out["iv_dm"].groupby("asset")["T"].unique().map(list).to_dict() == {a: [n] for a, n in n_hold.items()}
    assert out["iv_mcs"].empty and "leaderboard_pre2021" not in out
    assert (out["losses"]["split"] == "holdout").all()
    assert out["encompassing"].groupby("asset")["T"].unique().map(list).to_dict() == {
        a: [n] for a, n in n_hold.items()}
