"""Walk-forward machinery shared by every forecaster (SPEC §6–7).

Purged rolling window: at origin ``t`` a training row ``s`` is eligible only if its target window has ended
(``window_end_s <= t``) and its target is observed; the last ``W`` eligible rows are used.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np
import pandas as pd

FORECAST_COLUMNS = ["asset", "horizon", "model", "origin", "n_t", "F"]


class Forecaster(Protocol):
    name: str

    def forecast(self, daily: pd.DataFrame, targets: pd.DataFrame, asset: str, horizon: str) -> pd.DataFrame:
        """Return FORECAST_COLUMNS for every forecastable origin of one (asset, horizon)."""
        ...


def align_targets(daily: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Targets re-indexed to the rows of ``daily`` (origin == session_date); missing origins get NaN."""
    t = targets.set_index("origin")
    a = t.reindex(pd.Index(daily["session_date"].to_numpy(), name="origin"))
    a = a.reset_index()
    a.index = daily.index
    return a


def eligible_end(window_end: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """For each origin i, the number of leading rows whose target window has ended by origins[i].

    ``window_end`` is non-decreasing in the row order (rows = sessions sorted by date), so the purged set at
    origin i is a prefix ``[0, k_i)`` of the rows. Rows without a window (NaT: incomplete at the end of the
    sample, or an empty window after a data outage) are stamped with their own origin so they cannot freeze the
    envelope; ``direct_walk_forward`` excludes them from training separately. Returns k (exclusive end) per origin.
    """
    org = origins.astype("datetime64[D]")
    we = window_end.astype("datetime64[D]")
    we = np.where(np.isnat(we), org, we).astype("int64")
    we_mono = np.maximum.accumulate(we)  # defensive: monotone envelope
    return np.searchsorted(we_mono, org.astype("int64"), side="right")


def direct_walk_forward(
    X: np.ndarray,
    ybar: np.ndarray,
    window_end: np.ndarray,
    origins: np.ndarray,
    predict_mask: np.ndarray,
    window: int,
    refit_every: int,
    fit: Callable[[np.ndarray, np.ndarray], object],
    predict: Callable[[object, np.ndarray], np.ndarray],
) -> np.ndarray:
    """Generic purged, rolling walk-forward for direct (per-session-mean) models.

    X: (T, k) features at each origin row; ybar: (T,) target mean per session (NaN if unobserved);
    window_end/origins: datetime64[D] arrays per row; predict_mask: rows for which a forecast is wanted.
    Returns ``ybar_hat`` (T,) with NaN where no forecast was made (not enough eligible history).
    ``refit_every`` counts forecast origins; models with a session-based cadence (ML) use their own driver.
    """
    T = len(ybar)
    y_ok = np.isfinite(ybar) & np.all(np.isfinite(X), axis=1) & ~np.isnat(window_end.astype("datetime64[D]"))
    k_end = eligible_end(window_end, origins)
    obs_idx = np.flatnonzero(y_ok)
    out = np.full(T, np.nan)
    model = None
    since_refit = refit_every  # force a fit at the first forecast origin
    for i in np.flatnonzero(predict_mask):
        if not np.all(np.isfinite(X[i])):
            continue
        # eligible observed rows: obs_idx < k_end[i]
        n_elig = np.searchsorted(obs_idx, k_end[i], side="left")
        if n_elig < window:
            continue
        if model is None or since_refit >= refit_every:
            rows = obs_idx[n_elig - window : n_elig]
            model = fit(X[rows], ybar[rows])
            since_refit = 0
        out[i] = predict(model, X[i : i + 1])[0]
        since_refit += 1
    return out


def to_forecast_frame(
    asset: str, horizon: str, model: str, origins: pd.Series, n_t: pd.Series, F: np.ndarray
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "asset": asset,
            "horizon": horizon,
            "model": model,
            "origin": origins.to_numpy(),
            "n_t": n_t.to_numpy(),
            "F": np.asarray(F, dtype=float),
        }
    )
    df = df[np.isfinite(df["F"])].reset_index(drop=True)
    if (df["F"] <= 0).any():
        bad = df[df["F"] <= 0].head()
        raise ValueError(f"{model} {asset} {horizon}: non-positive forecasts\n{bad}")
    df["n_t"] = df["n_t"].astype("int64")
    return df


def predict_mask_for(targets_aligned: pd.DataFrame) -> np.ndarray:
    """Forecast every origin whose window lies in the dev or holdout evaluation split."""
    return targets_aligned["split"].isin(["dev", "holdout"]).to_numpy()
