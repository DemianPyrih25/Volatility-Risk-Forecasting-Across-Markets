"""HAR family: HAR, HAR-CJ, SHAR, HARQ (SPEC §7, models 5–8).

Direct per-horizon OLS of the per-session target mean ``ybar`` on strictly trailing regressors
(``features.har_frame``; ``+ gap²`` for SPX/EURUSD), estimated with ``numpy.linalg.lstsq`` on the purged
rolling window of ``W`` rows (``base.direct_walk_forward``) and refitted at every origin. Insanity filter:
a prediction outside ``[min, max]`` of the training targets is replaced by the training mean. ``F = n_t·ŷbar``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volrisk import config as C
from volrisk.models.base import align_targets, direct_walk_forward, predict_mask_for, to_forecast_frame
from volrisk.models.features import har_frame, uses_gap

# Regressors per model (columns of features.har_frame), before the optional gap² term.
SPECS: dict[str, tuple[str, ...]] = {
    "HAR": ("rv_d", "rv_w", "rv_m"),
    "HAR-CJ": ("c_d", "c_w", "c_m", "j_d", "j_w", "j_m"),
    "SHAR": ("rs_pos", "rs_neg", "rv_w", "rv_m"),
    "HARQ": ("rv_d", "rq_rv", "rv_w", "rv_m"),
}


def regressors(model: str, asset: str) -> list[str]:
    """Regressor names (without the intercept) of a HAR-family model for one asset."""
    cols = list(SPECS[model])
    if uses_gap(asset):
        cols.append("gap2")
    return cols


def design(daily: pd.DataFrame, asset: str, model: str, frame: pd.DataFrame | None = None) -> np.ndarray:
    """Design matrix ``[1, regressors]`` aligned with the rows of ``daily`` (NaN until the lags exist)."""
    h = har_frame(daily, asset) if frame is None else frame
    X = h[regressors(model, asset)].to_numpy(dtype=float)
    return np.column_stack([np.ones(len(X)), X])


@dataclass(frozen=True)
class OLSFit:
    beta: np.ndarray
    y_min: float
    y_max: float
    y_mean: float


def ols_fit(X: np.ndarray, y: np.ndarray) -> OLSFit:
    """Least squares (minimum-norm if rank deficient, e.g. a window without jumps) + training-target range."""
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return OLSFit(beta=beta, y_min=float(y.min()), y_max=float(y.max()), y_mean=float(y.mean()))


def ols_predict(fit: OLSFit, X: np.ndarray) -> np.ndarray:
    """Prediction with the insanity filter (outside the training-target range -> training mean)."""
    yhat = X @ fit.beta
    sane = (yhat >= fit.y_min) & (yhat <= fit.y_max)
    return np.where(sane, yhat, fit.y_mean)


class _HARFamily:
    name: str

    def __init__(self, window: int | None = None, refit_every: int = 1):
        self.window = C.window() if window is None else int(window)
        self.refit_every = int(refit_every)

    def predict_ybar(
        self, daily: pd.DataFrame, aligned: pd.DataFrame, asset: str, frame: pd.DataFrame | None = None
    ) -> np.ndarray:
        """Walk-forward ``ŷbar`` per row of ``daily`` (NaN where no forecast is made)."""
        X = design(daily, asset, self.name, frame)
        origins = daily["session_date"].to_numpy().astype("datetime64[D]")
        we = pd.to_datetime(aligned["window_end"]).to_numpy().astype("datetime64[D]")
        no_window = np.isnat(we)
        # A row without a complete window never trains, whatever its ybar says (no target leak)...
        ybar = np.where(no_window, np.nan, aligned["ybar"].to_numpy(dtype=float))
        # ...and is stamped with its own origin: base.eligible_end maps NaT to "never" and carries it forward
        # through its monotone envelope, so an interior empty window (n_t = 0) would otherwise freeze the
        # training window for every later origin.
        we = np.where(no_window, origins, we)
        return direct_walk_forward(
            X,
            ybar,
            we,
            origins,
            predict_mask_for(aligned),
            self.window,
            self.refit_every,
            fit=ols_fit,
            predict=ols_predict,
        )

    def forecast(self, daily: pd.DataFrame, targets: pd.DataFrame, asset: str, horizon: str) -> pd.DataFrame:
        a = align_targets(daily, targets)
        ybar_hat = self.predict_ybar(daily, a, asset)
        F = a["n_t"].to_numpy(dtype=float) * ybar_hat
        return to_forecast_frame(asset, horizon, self.name, daily["session_date"], a["n_t"], F)


class HAR(_HARFamily):
    name = "HAR"


class HARCJ(_HARFamily):
    name = "HAR-CJ"


class SHAR(_HARFamily):
    name = "SHAR"


class HARQ(_HARFamily):
    name = "HARQ"


HAR_MODELS: tuple[type[_HARFamily], ...] = (HAR, HARCJ, SHAR, HARQ)
