"""Implied-volatility benchmarks at the 1m horizon (SPEC §7, IV bullet; §8).

- ``IV``:     ``F_t = iv_var_30d_t``.
- ``IV-cal``: ``F_t = b̂_t·iv_var_30d_t`` with ``b̂_t = mean(y_s / iv_var_s)`` over origins ``s`` whose target
  window has ended (``window_end_s ≤ t``) and where both ``y_s`` and ``iv_var_s`` exist; the last
  ``cal_max`` such origins are used and no forecast is made before ``cal_min`` are available.

IV is never forward-filled: an origin without an IV value gets no forecast and never enters a calibration.
Forecasts are made for origins in the ``dev``/``holdout`` splits, like every other model (``models.base``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volrisk.models.base import FORECAST_COLUMNS, to_forecast_frame

IV_MODELS = ("IV", "IV-cal")


def _day(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s).to_numpy().astype("datetime64[D]")


def iv_benchmarks(
    targets_1m: pd.DataFrame,
    implied: pd.DataFrame,
    cal_min: int = 250,
    cal_max: int = 1000,
) -> pd.DataFrame:
    """``IV`` and ``IV-cal`` forecasts (FORECAST_COLUMNS) for one asset at horizon ``1m``.

    targets_1m: §6 targets of ONE asset at horizon 1m (``asset, horizon, origin, window_end, n_t, y, split``);
    implied: ONE asset's implied rows (``origin, iv_var_30d``; §2.4).
    """
    if not 1 <= cal_min <= cal_max:
        raise ValueError(f"need 1 <= cal_min <= cal_max, got {cal_min}, {cal_max}")
    assets = targets_1m["asset"].unique()
    if len(assets) != 1:
        raise ValueError(f"targets must hold exactly one asset, got {list(assets)}")
    asset = str(assets[0])
    if "horizon" in targets_1m and not (targets_1m["horizon"] == "1m").all():
        raise ValueError("IV benchmarks are defined at the 1m horizon only")
    if "asset" in implied and not (implied["asset"] == asset).all():
        raise ValueError(f"implied rows must belong to {asset}")

    iv = pd.DataFrame({"_key": _day(implied["origin"]), "iv_var": implied["iv_var_30d"].astype(float)})
    iv = iv[np.isfinite(iv["iv_var"]) & (iv["iv_var"] > 0)]
    if iv["_key"].duplicated().any():
        raise ValueError("implied has duplicate origins")

    t = targets_1m.assign(_key=_day(targets_1m["origin"])).sort_values("_key", kind="stable")
    t = t.merge(iv, on="_key", how="left").reset_index(drop=True)  # left join: no fill of missing IV
    origin_d = t["_key"].to_numpy()
    window_end_d = _day(t["window_end"])
    y = t["y"].to_numpy(dtype=float)
    ivv = t["iv_var"].to_numpy(dtype=float)
    has_iv = np.isfinite(ivv)

    # calibration pool: completed windows with both y and IV observed (NaT window_end never qualifies)
    cal = has_iv & np.isfinite(y) & (y > 0) & ~np.isnat(window_end_d)
    cal_end = window_end_d[cal]
    cal_ratio = y[cal] / ivv[cal]

    want = has_iv.copy()
    if "split" in t:
        want &= t["split"].isin(["dev", "holdout"]).to_numpy()

    b_hat = np.full(len(t), np.nan)
    for i in np.flatnonzero(want):
        idx = np.flatnonzero(cal_end <= origin_d[i])  # cal rows are in origin order
        if idx.size >= cal_min:
            b_hat[i] = cal_ratio[idx[-cal_max:]].mean()

    sel = t[want]
    raw = to_forecast_frame(asset, "1m", "IV", sel["origin"], sel["n_t"], ivv[want])
    calf = to_forecast_frame(asset, "1m", "IV-cal", sel["origin"], sel["n_t"], (b_hat * ivv)[want])
    return pd.concat([raw, calf], ignore_index=True)[FORECAST_COLUMNS]
