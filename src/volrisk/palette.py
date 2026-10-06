"""Shared colour mapping and display order for the static report and the dashboard.

One entity keeps one colour everywhere (a filter never repaints the survivors). The model and risk-model
colours are fixed project-wide; ``COMBO`` / ``COMBO+FHS`` are near-black on purpose (the primary model is the
emphasis series) and the implied-vol benchmarks are greys. Colour is never the only identity channel: every
chart carries a legend and direct labels, and every table has the same numbers.

Zone colours are a colour-blind-checked green / yellow / red triple (worst all-pairs CVD ΔE ≈ 11.7 under
deuteranopia); they are always shown together with the zone name, because the yellow is below 3:1 contrast on
a light surface.
"""

from __future__ import annotations

from collections.abc import Iterable

# --------------------------------------------------------------------------------------------- entities
MODEL_COLORS: dict[str, str] = {
    "HAR": "#4C78A8",
    "HARQ": "#72B7B2",
    "HAR-CJ": "#54A24B",
    "SHAR": "#88D27A",
    "GARCH": "#E45756",
    "GJR": "#FF9D98",
    "EWMA": "#B279A2",
    "RW": "#9D755D",
    "LGBM": "#F58518",
    "MLP": "#FFBF79",
    "COMBO": "#222222",
    "IV": "#7F7F7F",
    "IV-cal": "#BAB0AC",
}

RISK_COLORS: dict[str, str] = {
    "HS-250": "#9D755D",
    "RiskMetrics": "#B279A2",
    "GJR+FHS": "#E45756",
    "HAR*+FHS": "#4C78A8",
    "COMBO+FHS": "#222222",
    "COMBO+Normal": "#F58518",
}

ZONE_COLORS: dict[str, str] = {
    "green": "#2CA02C",
    "yellow": "#F0C419",
    "red": "#B2182B",
}

# Display order: horizons by length (never alphabetical), assets as in volrisk.config.ASSETS.
HORIZON_ORDER: tuple[str, ...] = ("1d", "1w", "1m")
ASSET_ORDER: tuple[str, ...] = ("BTC", "ETH", "EURUSD", "SPX")
MODEL_ORDER: tuple[str, ...] = tuple(MODEL_COLORS)
RISK_ORDER: tuple[str, ...] = tuple(RISK_COLORS)
ZONE_ORDER: tuple[str, ...] = tuple(ZONE_COLORS)

HAR_FAMILY: tuple[str, ...] = ("HAR", "HARQ", "HAR-CJ", "SHAR")

# Annualisation factors for display only (SPEC preamble: never used in a comparison).
ANNUALISATION: dict[str, int] = {"BTC": 365, "ETH": 365, "EURUSD": 260, "SPX": 252}

# --------------------------------------------------------------------------------------------- chart chrome
SURFACE = "#FFFFFF"
INK = "#0B0B0B"  # primary text
INK2 = "#52514E"  # secondary text, axis labels
MUTED = "#898781"  # tick labels, notes
GRID = "#E1E0D9"  # hairline gridlines
AXIS = "#C3C2B7"  # baseline / axis
REALIZED_COLOR = "#CFCCC5"  # realized-volatility series colour (not a model); the dashboard fills with it at alpha
REALIZED_WASH = "#EFEDE9"  # opaque realized-volatility area behind the 1m forecast lines in the report: light
#                           enough that every model line stays visible on it (IV-cal #BAB0AC 1.8:1, IV 3.4:1)
HOLDOUT_SHADE = "#EEF3FA"  # background band marking the holdout period
NEUTRAL_MID = "#F0EFEC"  # diverging-scale midpoint (ratio = 1)
DIVERGING = (  # blue arm = better than the reference (ratio < 1), red arm = worse
    "#184F95", "#5598E7", "#B7D3F6", NEUTRAL_MID, "#F6BDB8", "#E4675F", "#A3201D",
)


# --------------------------------------------------------------------------------------------- helpers
def _ordered(values: Iterable[str], order: tuple[str, ...]) -> list[str]:
    """Unique values in ``order`` first, then unknown values in first-seen order."""
    seen = list(dict.fromkeys(str(v) for v in values))
    rank = {k: i for i, k in enumerate(order)}
    known = sorted((v for v in seen if v in rank), key=rank.__getitem__)
    return known + [v for v in seen if v not in rank]


def order_horizons(values: Iterable[str]) -> list[str]:
    """Horizons present in ``values`` in display order 1d, 1w, 1m."""
    return _ordered(values, HORIZON_ORDER)


def order_assets(values: Iterable[str]) -> list[str]:
    """Assets present in ``values`` in display order BTC, ETH, EURUSD, SPX."""
    return _ordered(values, ASSET_ORDER)


def order_models(values: Iterable[str]) -> list[str]:
    """Forecast models present in ``values`` in the order of ``MODEL_COLORS``."""
    return _ordered(values, MODEL_ORDER)


def order_risk_models(values: Iterable[str]) -> list[str]:
    """Risk models present in ``values`` in the order of ``RISK_COLORS``."""
    return _ordered(values, RISK_ORDER)


def horizon_rank(h: str) -> int:
    """Sort key for a horizon (unknown horizons sort last)."""
    return HORIZON_ORDER.index(h) if h in HORIZON_ORDER else len(HORIZON_ORDER)


def asset_rank(a: str) -> int:
    """Sort key for an asset (unknown assets sort last)."""
    return ASSET_ORDER.index(a) if a in ASSET_ORDER else len(ASSET_ORDER)


def model_color(name: str, default: str = MUTED) -> str:
    """Colour of a forecast model or a risk model."""
    return MODEL_COLORS.get(name) or RISK_COLORS.get(name) or default
