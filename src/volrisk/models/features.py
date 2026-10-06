"""Strictly trailing regressors built from the gold daily table (SPEC §7).

All functions take the gold rows of ONE asset (pandas, sorted by session_date) and return a frame aligned
row-for-row with the input. Rolling windows count sessions (rows), not calendar days, and include the
current session ``t`` — everything at row ``t`` is known at the close of session ``t``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volrisk import config as C

GAP_ASSETS = ("SPX", "EURUSD")


def uses_gap(asset: str) -> bool:
    return asset in GAP_ASSETS


def har_frame(daily: pd.DataFrame, asset: str) -> pd.DataFrame:
    """HAR-family building blocks: daily/weekly/monthly RV, C, J plus RS±, √RQ·RV and gap²."""
    _, w, m = C.har_lags(asset)
    rv, c, j = daily["rv"], daily["c"], daily["j"]
    out = pd.DataFrame(index=daily.index)
    for name, s in (("rv", rv), ("c", c), ("j", j)):
        out[f"{name}_d"] = s
        out[f"{name}_w"] = s.rolling(w, min_periods=w).mean()
        out[f"{name}_m"] = s.rolling(m, min_periods=m).mean()
    out["rs_pos"] = daily["rs_pos"]
    out["rs_neg"] = daily["rs_neg"]
    out["rq_rv"] = np.sqrt(daily["rq"]) * rv
    out["gap2"] = daily["gap"] ** 2
    return out


def ml_features(daily: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Feature matrix shared by LightGBM and the MLP (levels; transforms are model-specific)."""
    h = har_frame(daily, asset)
    rv = daily["rv"]
    out = pd.DataFrame(
        {
            "rv": rv,
            "rv_w": h["rv_w"],
            "rv_m": h["rv_m"],
            "bv": daily["bv"],
            "j_share": np.where(rv > 0, daily["j"] / rv, 0.0),
            "rs_pos": daily["rs_pos"],
            "rs_neg": daily["rs_neg"],
            "rq": daily["rq"],
            "r_cc": daily["r_cc"],
            "r_cc_neg": np.minimum(daily["r_cc"], 0.0),
            "dow": pd.to_datetime(daily["session_date"]).dt.dayofweek.astype("int64"),
        },
        index=daily.index,
    )
    if uses_gap(asset):
        out["gap2"] = daily["gap"] ** 2
    return out


# Variance-type columns (positive, heavy-tailed) — the MLP log-transforms these.
VARIANCE_FEATURES = ("rv", "rv_w", "rv_m", "bv", "rs_pos", "rs_neg", "rq", "gap2")
