"""Forecast losses on variances (SPEC §8).

``y`` is the realised cumulative target variance over the window and ``F`` the forecast of it (both %²).
QLIKE is the primary loss, MSE the secondary one. Both require ``F > 0`` and ``y > 0``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _check(y: ArrayLike, F: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    F = np.asarray(F, dtype=float)
    if y.shape != F.shape:
        raise ValueError(f"y and F have different shapes: {y.shape} vs {F.shape}")
    if not np.all(np.isfinite(F) & (F > 0)):
        raise ValueError("forecasts must be strictly positive and finite (F > 0)")
    if not np.all(np.isfinite(y) & (y > 0)):
        raise ValueError("targets must be strictly positive and finite (y > 0)")
    return y, F


def qlike(y: ArrayLike, F: ArrayLike) -> np.ndarray:
    """QLIKE ``y/F - ln(y/F) - 1`` (>= 0, zero iff ``F == y``)."""
    y, F = _check(y, F)
    r = y / F
    return r - np.log(r) - 1.0


def mse(y: ArrayLike, F: ArrayLike) -> np.ndarray:
    """Squared error ``(y - F)²``."""
    y, F = _check(y, F)
    return (y - F) ** 2
