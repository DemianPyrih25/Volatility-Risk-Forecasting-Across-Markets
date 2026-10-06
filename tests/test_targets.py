"""Tests for volrisk.targets (SPEC §6): windows, n_t, y/ybar and the dev/holdout split."""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest

from volrisk import config as C
from volrisk.targets import (
    TARGET_COLUMNS,
    apply_oos_start,
    build_all_targets,
    build_targets,
    default_warmup,
    eligible_counts,
    oos_start,
)


def _daily(asset: str, dates, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    d = pd.DatetimeIndex(dates).astype("datetime64[ms]")
    return pd.DataFrame({"asset": asset, "session_date": d, "tv": rng.gamma(2.0, 0.5, len(d))})


def _crypto_dates(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(start, end, freq="D")


def _xnys_dates(start: str, end: str) -> pd.DatetimeIndex:
    with warnings.catch_warnings():  # exchange_calendars emits numpy deprecation noise on import
        warnings.simplefilter("ignore", DeprecationWarning)
        import exchange_calendars as xcals

        cal = xcals.get_calendar("XNYS", start="2012-01-01", end="2026-12-31")
        return cal.sessions_in_range(start, end).tz_localize(None)


def _fx_dates(start: str, end: str) -> pd.DatetimeIndex:
    d = pd.bdate_range(start, end)
    return d[~(((d.month == 12) & (d.day == 25)) | ((d.month == 1) & (d.day == 1)))]


def _row(t: pd.DataFrame, origin: str) -> pd.Series:
    r = t[t["origin"] == pd.Timestamp(origin)]
    assert len(r) == 1, origin
    return r.iloc[0]


HOLDOUT_MODE = C.data_end()
DEV_MODE = C.dev_end()


def test_columns_and_sums_crypto():
    daily = _daily("BTC", _crypto_dates("2024-01-01", "2024-06-30"))
    tv = daily["tv"].to_numpy()
    for h, n in (("1d", 1), ("1w", 7), ("1m", 30)):
        t = build_targets(daily, "BTC", h, date(2024, 6, 30))
        assert list(t.columns) == TARGET_COLUMNS
        assert len(t) == len(daily)
        assert t["origin"].dtype == daily["session_date"].dtype
        assert t["window_end"].dtype == daily["session_date"].dtype
        full = t[t["split"] == "dev"]
        assert (full["n_t"] == n).all()
        # every origin with a full window ahead in the sample is complete
        assert len(full) == len(daily) - n
        i = 10
        r = t.iloc[i]
        assert r["window_end"] == daily["session_date"].iloc[i + n]
        assert r["y"] == pytest.approx(tv[i + 1 : i + 1 + n].sum(), rel=1e-14)
        assert r["ybar"] == pytest.approx(tv[i + 1 : i + 1 + n].mean(), rel=1e-14)
        tail = t.iloc[len(daily) - n :]
        assert (tail["split"] == "dropped").all()
        assert tail["y"].isna().all() and tail["ybar"].isna().all() and tail["window_end"].isna().all()


def test_spx_holiday_week_and_caps():
    daily = _daily("SPX", _xnys_dates("2012-01-03", "2025-09-30"))
    t1w = build_targets(daily, "SPX", "1w", DEV_MODE)
    # Thanksgiving 2024: Mon 25, Tue 26, Wed 27, (Thu 28 closed), Fri 29 half-day -> 4 sessions
    r = _row(t1w, "2024-11-22")
    assert r["n_t"] == 4 and r["window_end"] == pd.Timestamp("2024-11-29")
    assert _row(t1w, "2024-11-15")["n_t"] == 5
    tv = daily.set_index("session_date")["tv"]
    assert r["y"] == pytest.approx(tv.loc["2024-11-23":"2024-11-29"].sum(), rel=1e-14)
    assert r["ybar"] == pytest.approx(r["y"] / 4, rel=1e-14)
    t1m = build_targets(daily, "SPX", "1m", DEV_MODE)
    ok = t1m["split"] == "dev"
    assert t1m.loc[ok, "n_t"].max() == C.n_max("1m", "SPX") == 22
    assert t1w.loc[t1w["split"] == "dev", "n_t"].max() == 5
    t1d = build_targets(daily, "SPX", "1d", DEV_MODE)
    # 1d = the next session, across the weekend and the holiday
    assert _row(t1d, "2024-11-27")["window_end"] == pd.Timestamp("2024-11-29")
    assert (t1d.loc[t1d["split"] == "dev", "n_t"] == 1).all()


def test_fx_caps_and_christmas():
    daily = _daily("EURUSD", _fx_dates("2012-01-02", "2025-09-30"))
    t1w = build_targets(daily, "EURUSD", "1w", DEV_MODE)
    t1m = build_targets(daily, "EURUSD", "1m", DEV_MODE)
    assert t1w.loc[t1w["split"] == "dev", "n_t"].max() == 5
    assert t1m.loc[t1m["split"] == "dev", "n_t"].max() == 22
    assert _row(t1w, "2023-12-22")["n_t"] == 4  # Dec 25 2023 (Monday) is not a session


def test_n_t_above_cap_raises():
    # weekend rows in an SPX table are a calendar bug: a 1w window would hold 7 sessions
    daily = _daily("SPX", _crypto_dates("2024-01-01", "2024-03-31"))
    with pytest.raises(ValueError, match="n_max"):
        build_targets(daily, "SPX", "1w", DEV_MODE)
    build_targets(daily, "SPX", "1d", DEV_MODE)  # 1d is always one row


def test_empty_window_is_dropped():
    d = _xnys_dates("2024-01-02", "2024-06-28")
    d = d[(d < "2024-03-01") | (d > "2024-03-12")]  # 10 days without a valid session
    daily = _daily("SPX", d)
    t = build_targets(daily, "SPX", "1w", date(2024, 6, 28))
    r = _row(t, "2024-02-29")
    assert r["n_t"] == 0 and r["split"] == "dropped" and np.isnan(r["y"]) and pd.isna(r["window_end"])
    assert _row(t, "2024-03-13")["split"] == "dev"


def test_split_boundaries_holdout_mode():
    daily = _daily("BTC", _crypto_dates("2025-01-01", "2026-09-30"))
    t = {h: build_targets(daily, "BTC", h, HOLDOUT_MODE) for h in C.HORIZONS}
    cases = {
        "1d": [
            ("2025-09-29", "dev", "2025-09-30"),
            ("2025-09-30", "holdout", "2025-10-01"),
            ("2026-09-29", "holdout", "2026-09-30"),
            ("2026-09-30", "dropped", None),
        ],
        "1w": [
            ("2025-09-23", "dev", "2025-09-30"),
            ("2025-09-24", "dropped", "2025-10-01"),  # crosses the boundary: complete but dropped
            ("2025-09-29", "dropped", "2025-10-06"),
            ("2025-09-30", "holdout", "2025-10-07"),
            ("2026-09-23", "holdout", "2026-09-30"),
            ("2026-09-24", "dropped", None),  # runs past the end of data
        ],
        "1m": [
            ("2025-08-31", "dev", "2025-09-30"),
            ("2025-09-01", "dropped", "2025-10-01"),
            ("2025-09-30", "holdout", "2025-10-30"),
            ("2026-08-31", "holdout", "2026-09-30"),
            ("2026-09-01", "dropped", None),
        ],
    }
    for h, rows in cases.items():
        for origin, split, we in rows:
            r = _row(t[h], origin)
            assert r["split"] == split, (h, origin)
            if we is None:
                assert pd.isna(r["window_end"]) and np.isnan(r["y"]) and np.isnan(r["ybar"])
            else:
                assert r["window_end"] == pd.Timestamp(we), (h, origin)
                assert np.isfinite(r["y"]) and r["y"] > 0
    for h in C.HORIZONS:
        th = t[h]
        dev, hold = th[th["split"] == "dev"], th[th["split"] == "holdout"]
        assert (dev["window_end"] <= pd.Timestamp(C.dev_end())).all()
        assert (hold["origin"] >= pd.Timestamp("2025-09-30")).all()
        assert (hold["window_end"] >= pd.Timestamp(C.holdout_start())).all()
        assert (hold["window_end"] <= pd.Timestamp(C.data_end())).all()


def test_dev_mode_matches_holdout_mode_on_dev_rows():
    full = _daily("ETH", _crypto_dates("2024-06-01", "2026-09-30"), seed=3)
    dev = full[full["session_date"] < pd.Timestamp(C.holdout_start())].reset_index(drop=True)
    for h in C.HORIZONS:
        td = build_targets(dev, "ETH", h, DEV_MODE)
        th = build_targets(full, "ETH", h, HOLDOUT_MODE)
        assert (td["split"] != "holdout").all()
        a = td[td["split"] == "dev"].reset_index(drop=True)
        b = th[th["split"] == "dev"].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)
        # dev-mode windows that would reach past 2025-09-30 are incomplete, hence null
        drop = td[td["split"] == "dropped"]
        assert drop["y"].isna().all()
        assert len(drop) == C.n_max(h, "ETH")


def test_1d_next_row_must_be_within_last_date():
    daily = _daily("BTC", _crypto_dates("2025-09-01", "2025-10-10"))
    t = build_targets(daily, "BTC", "1d", DEV_MODE)
    r = _row(t, "2025-09-30")
    assert r["split"] == "dropped" and np.isnan(r["y"])
    assert (t.loc[t["origin"] >= pd.Timestamp("2025-09-30"), "split"] == "dropped").all()


def test_spx_boundary_with_missing_last_dev_session():
    d = _xnys_dates("2025-06-02", "2026-09-30")
    daily = _daily("SPX", d)
    t = build_targets(daily, "SPX", "1d", HOLDOUT_MODE)
    assert _row(t, "2025-09-29")["split"] == "dev"
    assert _row(t, "2025-09-30")["split"] == "holdout"
    t1w = build_targets(daily, "SPX", "1w", HOLDOUT_MODE)
    assert _row(t1w, "2025-09-26")["split"] == "dropped"  # (09-26, 10-03] crosses the boundary
    assert _row(t1w, "2025-09-23")["split"] == "dev"
    # 2025-09-30 invalid (dropped from gold): the literal SPEC threshold (origin >= 2025-09-30) still applies,
    # so origin 2025-09-29, whose 1d window is 2025-10-01, crosses the boundary and is dropped (its y is kept)
    gap = daily[daily["session_date"] != pd.Timestamp("2025-09-30")].reset_index(drop=True)
    t = build_targets(gap, "SPX", "1d", HOLDOUT_MODE)
    r = _row(t, "2025-09-29")
    assert r["split"] == "dropped" and r["window_end"] == pd.Timestamp("2025-10-01") and np.isfinite(r["y"])
    assert _row(t, "2025-09-26")["split"] == "dev"
    assert _row(t, "2025-10-01")["split"] == "holdout"
    t1w = build_targets(gap, "SPX", "1w", HOLDOUT_MODE)
    assert _row(t1w, "2025-09-23")["split"] == "dev"  # (09-23, 09-30] now ends on 09-29
    assert _row(t1w, "2025-09-29")["split"] == "dropped"
    assert _row(t1w, "2025-10-01")["split"] == "holdout"
    assert (t1w.loc[t1w["split"] == "holdout", "origin"] >= pd.Timestamp(C.dev_end())).all()


def test_rejects_unsorted_or_mixed_input():
    daily = _daily("BTC", _crypto_dates("2024-01-01", "2024-02-01"))
    with pytest.raises(ValueError, match="increasing"):
        build_targets(daily.iloc[::-1].reset_index(drop=True), "BTC", "1d", DEV_MODE)
    with pytest.raises(ValueError, match="other assets"):
        build_targets(daily, "ETH", "1d", DEV_MODE)


def test_build_all_targets():
    parts = [
        _daily("SPX", _xnys_dates("2024-01-02", "2024-12-31"), seed=1),
        _daily("BTC", _crypto_dates("2024-01-01", "2024-12-31"), seed=2),
        _daily("EURUSD", _fx_dates("2024-01-02", "2024-12-31"), seed=4),
    ]
    daily_all = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=0)  # unsorted input
    last, W = date(2024, 12, 31), 100
    t = build_all_targets(daily_all, last, window=W)
    assert list(t.columns) == TARGET_COLUMNS
    assert len(t) == 3 * sum(len(p) for p in parts)
    assert list(pd.unique(t["asset"])) == ["BTC", "EURUSD", "SPX"]
    for (a, h), g in t.groupby(["asset", "horizon"]):
        assert g["origin"].is_monotonic_increasing and g["origin"].is_unique
        assert g.loc[g["split"] == "dev", "n_t"].max() <= C.n_max(h, a)
    assert set(t["split"]) <= {"dev", "holdout", "dropped"}
    assert (t.loc[t["split"] != "dropped", "ybar"] > 0).all()
    # one OOS start per asset shared by every horizon; earlier origins are dropped but keep their targets
    for p in parts:
        a = p["asset"].iloc[0]
        ta = t[t["asset"] == a]
        start = oos_start(ta, W)
        assert start is not None
        for h in C.HORIZONS:
            g = ta[ta["horizon"] == h].reset_index(drop=True)
            raw = build_targets(p, a, h, last)
            assert g.loc[g["split"] != "dropped", "origin"].min() == start, (a, h)
            pd.testing.assert_frame_equal(g.drop(columns="split"), raw.drop(columns="split"))
            keep = (g["origin"] >= start).to_numpy()
            assert (g.loc[~keep, "split"] == "dropped").all()
            np.testing.assert_array_equal(g.loc[keep, "split"].to_numpy(), raw.loc[keep, "split"].to_numpy())


def test_oos_start_crypto_pinned():
    """BTC from 2018-01-01, W = 1000, monthly lag 30: the first trainable row is 29, so rows 29..1028 must have
    ended windows. At 1m row 1028's window ends on row 1058 = 2020-11-24, the start for every horizon."""
    daily = _daily("BTC", _crypto_dates("2018-01-01", "2021-03-31"))
    last, W = date(2021, 3, 31), 1000
    assert default_warmup("BTC") == 29 and default_warmup("SPX") == default_warmup("EURUSD") == 21
    one = {h: build_targets(daily, "BTC", h, last) for h in C.HORIZONS}
    # per-horizon starts differ (what each model would do on its own) ...
    assert oos_start(one["1d"], W) == pd.Timestamp("2020-10-26")
    assert oos_start(one["1w"], W) == pd.Timestamp("2020-11-01")
    assert oos_start(one["1m"], W) == pd.Timestamp("2020-11-24")
    n1m = eligible_counts(one["1m"], 29)
    i = int(np.flatnonzero(one["1m"]["origin"] == pd.Timestamp("2020-11-24"))[0])
    assert n1m[i] == W and n1m[i - 1] == W - 1
    # ... the asset's OOS start is the latest of them, applied to every horizon
    t = build_all_targets(daily, last, window=W)
    assert oos_start(t, W) == pd.Timestamp("2020-11-24")
    for h in C.HORIZONS:
        g = t[t["horizon"] == h]
        assert g.loc[g["split"] != "dropped", "origin"].min() == pd.Timestamp("2020-11-24"), h
    pre = t[t["origin"] < pd.Timestamp("2020-11-24")]
    assert (pre["split"] == "dropped").all() and np.isfinite(pre["ybar"]).all()


def test_eligible_counts_and_oos_start_brute_force():
    d = _xnys_dates("2019-01-02", "2020-12-31")
    d = d[(d < "2020-03-02") | (d > "2020-03-12")]  # 10 days without a valid session -> an empty 1w window
    daily = _daily("SPX", d, seed=6)
    last, W, wu = date(2020, 12, 31), 300, default_warmup("SPX")
    tb = {h: build_targets(daily, "SPX", h, last) for h in C.HORIZONS}
    assert (tb["1w"]["n_t"] == 0).any()
    firsts = []
    for h, g in tb.items():
        o = g["origin"].to_numpy()
        we = g["window_end"].to_numpy()
        ok = np.isfinite(g["ybar"].to_numpy()) & ~np.isnat(we)
        ok[:wu] = False
        brute = np.array([int(np.sum(ok & (we <= x))) for x in o])
        np.testing.assert_array_equal(eligible_counts(g, wu), brute)
        firsts.append(o[np.argmax(brute >= W)])
    assert oos_start(build_all_targets(daily, last, window=W), W) == pd.Timestamp(max(firsts))
    with pytest.raises(ValueError, match="increasing"):
        eligible_counts(pd.concat(tb.values()), wu)


def test_oos_start_same_in_dev_and_holdout_mode():
    full = _daily("ETH", _crypto_dates("2024-01-01", "2026-09-30"), seed=8)
    dev = full[full["session_date"] < pd.Timestamp(C.holdout_start())].reset_index(drop=True)
    W = 300
    td = build_all_targets(dev, DEV_MODE, window=W)
    th = build_all_targets(full, HOLDOUT_MODE, window=W)
    # rows 29 .. 29+W-1 trainable; at 1m the last of them has its window end 30 days later
    expected = pd.Timestamp(np.datetime64("2024-01-01", "D") + np.timedelta64(29 + W - 1 + 30, "D"))
    assert oos_start(td, W) == oos_start(th, W) == expected
    for h in C.HORIZONS:
        a = td[(td["horizon"] == h) & (td["split"] == "dev")].reset_index(drop=True)
        b = th[(th["horizon"] == h) & (th["split"] == "dev")].reset_index(drop=True)
        assert len(a) > 100
        pd.testing.assert_frame_equal(a, b)
    assert (th.loc[th["split"] == "holdout", "origin"] >= pd.Timestamp(C.dev_end())).all()


def test_no_oos_start_when_sample_too_short():
    daily = _daily("BTC", _crypto_dates("2024-01-01", "2024-03-31"))
    t = build_all_targets(daily, date(2024, 3, 31), window=1000)
    assert oos_start(t, 1000) is None
    assert (t["split"] == "dropped").all() and np.isfinite(t["ybar"]).sum() > 0
    assert oos_start(t.iloc[0:0], 10) is None
    with pytest.raises(ValueError, match="one asset"):
        oos_start(pd.concat([t, t.assign(asset="ETH")]), 10)
    # W = 0 (or anything already reached at the first origin) changes nothing
    raw = pd.concat([build_targets(daily, "BTC", h, date(2024, 3, 31)) for h in C.HORIZONS], ignore_index=True)
    pd.testing.assert_frame_equal(apply_oos_start(raw, oos_start(raw, 0)), raw)
