"""Dash dashboard (SPEC §12): a thin, read-only viewer of the pipeline results.

No model fitting, no downloads, no live data: every number comes from the results parquet files through
:class:`volrisk.dashboard.data.ResultsData`. The callback bodies are plain functions of a ``ResultsData`` and the
control values (``forecast_figure``, ``update_leaderboard``, ``update_var``, ...), so they can be imported and
called directly; :func:`create_app` only wires them to the Dash components.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from plotly.subplots import make_subplots

from volrisk.dashboard.data import ResultsData, annualisation

try:
    from volrisk import palette as _palette
    from volrisk.palette import ASSET_ORDER, HORIZON_ORDER, MODEL_COLORS, RISK_COLORS, ZONE_COLORS
except ImportError:  # pragma: no cover - the shared palette module is maintained with the static report
    _palette = None
    MODEL_COLORS = {
        "HAR": "#4C78A8", "HARQ": "#72B7B2", "HAR-CJ": "#54A24B", "SHAR": "#88D27A", "GARCH": "#E45756",
        "GJR": "#FF9D98", "EWMA": "#B279A2", "RW": "#9D755D", "LGBM": "#F58518", "MLP": "#FFBF79",
        "COMBO": "#222222", "IV": "#7F7F7F", "IV-cal": "#BAB0AC",
    }
    RISK_COLORS = {
        "HS-250": "#9D755D", "RiskMetrics": "#B279A2", "GJR+FHS": "#E45756", "HAR*+FHS": "#4C78A8",
        "COMBO+FHS": "#222222", "COMBO+Normal": "#F58518",
    }
    ZONE_COLORS = {"green": "#2CA02C", "yellow": "#F0C419", "red": "#B2182B"}
    HORIZON_ORDER = ("1d", "1w", "1m")
    ASSET_ORDER = ("BTC", "ETH", "EURUSD", "SPX")


def _chrome(name: str, default: Any) -> Any:
    """Chart-chrome constant shared with the report when the palette module defines it."""
    return getattr(_palette, name, default) if _palette is not None else default


# --------------------------------------------------------------------------------------------- tokens
SURFACE = _chrome("SURFACE", "#FFFFFF")
INK = _chrome("INK", "#0B0B0B")
INK2 = _chrome("INK2", "#52514E")
MUTED = _chrome("MUTED", "#898781")
GRID = _chrome("GRID", "#E1E0D9")
AXIS = _chrome("AXIS", "#C3C2B7")
REALIZED_COLOR = _chrome("REALIZED_COLOR", "#CFCCC5")
NEUTRAL_MID = _chrome("NEUTRAL_MID", "#F0EFEC")
DIVERGING = tuple(
    _chrome("DIVERGING", ("#184F95", "#5598E7", "#B7D3F6", "#F0EFEC", "#F6BDB8", "#E4675F", "#A3201D"))
)
PAGE = "#F6F6F3"
BORDER = "#E4E3DD"
RETURN_COLOR = "#BDBBB4"  # daily-return bars behind the VaR lines (not a model)
NA_COLOR = "#B5B3AC"
FONT = '-apple-system, "Segoe UI", system-ui, Roboto, "Helvetica Neue", Arial, sans-serif'

TITLE = "Volatility & risk forecasting"
DESCRIPTION = (
    "Out-of-sample variance forecasts (HAR family, GARCH, machine learning, implied vol) and VaR/ES backtests "
    "for BTC, ETH, EUR/USD and the S&P 500 — a read-only view of the pipeline results."
)
ASSET_LABELS = {"BTC": "Bitcoin (BTC)", "ETH": "Ether (ETH)", "EURUSD": "EUR/USD", "SPX": "S&P 500 (SPX)"}
HORIZON_LABELS = {"1d": "1 day", "1w": "1 week", "1m": "1 month"}
SPLIT_LABELS = {"dev": "development", "holdout": "holdout"}
DEFAULT_MODELS = ("HAR", "GARCH", "COMBO")
DEFAULT_RISK_MODELS = ("HS-250", "COMBO+FHS")
REF_MODEL = "HAR"
ZONE_CODES = {"green": 0, "yellow": 1, "red": 2}
ZONE_RULES = {"green": "0–4", "yellow": "5–9", "red": "≥10"}  # exceptions in 250 days at 99% (Basel)
DESCRIPTIVE_HOLDOUT_HORIZON = "1m"  # SPEC §8: "MCS at 1d and 1w only; 1m holdout is descriptive"
HOLDOUT_1M_NOTE = (
    "Holdout: the 1-month rows are descriptive (SPEC §8). The holdout year holds only about a dozen "
    "non-overlapping 1-month windows, so the 1m QLIKE ratios, bootstrap intervals and DM-HLN p-values below are "
    "shown for information and enter no hypothesis verdict; the holdout model confidence set is computed at 1d "
    "and 1w only."
)
LEADERBOARD_NOTE = (
    "Mean QLIKE of each model divided by HAR's on the dates where every model has a forecast (<1 = more accurate "
    "than HAR). ★ marks the 90% model confidence set (MCS). DM-HLN p: two-sided test of equal QLIKE vs HAR."
)
IV_HOLDOUT_NOTE = (
    "Holdout: these 1-month comparisons are descriptive (SPEC §8); the encompassing coefficient c of COMBO still "
    "gives the low-power holdout check of H3 (SPEC §0). No IV-inclusive MCS is computed on the holdout."
)
IV_MESSAGE_EURUSD = (
    "No free implied-vol benchmark available for EUR/USD in the holdout: CBOE EVZ was discontinued in "
    "March 2025 and is used only up to 2023-12-31."
)
STATEMENTS = {
    "H1": "HAR-type models beat GARCH-type models on QLIKE at 1d for all four assets.",
    "H2": "Machine learning (LightGBM, MLP) is not significantly better than the best HAR model.",
    "H3": "At 1m, model forecasts contain information beyond implied vol (SPX, BTC, ETH).",
    "H4": "COMBO+FHS has a lower FZ0 score than HS-250 and spends more time in the Basel green zone.",
    "H5": "Jumps and downside semivariance matter more in crypto than in EUR/USD (descriptive).",
}
GRAPH_CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"], "responsive": True}


# --------------------------------------------------------------------------------------------- helpers
def _ordered(values: Iterable[str], order: Sequence[str]) -> list[str]:
    seen = list(dict.fromkeys(str(v) for v in values))
    rank = {k: i for i, k in enumerate(order)}
    return sorted((v for v in seen if v in rank), key=rank.__getitem__) + [v for v in seen if v not in rank]


def order_models(values: Iterable[str]) -> list[str]:
    return _ordered(values, tuple(MODEL_COLORS))


def order_risk_models(values: Iterable[str]) -> list[str]:
    return _ordered(values, tuple(RISK_COLORS))


def order_horizons(values: Iterable[str]) -> list[str]:
    return _ordered(values, HORIZON_ORDER)


def _horizon_rank(h: str) -> int:
    return HORIZON_ORDER.index(h) if h in HORIZON_ORDER else len(HORIZON_ORDER)


def model_color(name: str) -> str:
    return MODEL_COLORS.get(name) or RISK_COLORS.get(name) or MUTED


def zone_color(zone: Any) -> str:
    z = str(zone).strip().lower()
    for key, col in ZONE_COLORS.items():
        if str(key).lower() == z:
            return col
    return NA_COLOR


def _missing(x: Any) -> bool:
    """None, NaN, NaT or pandas NA (scalars only)."""
    try:
        return bool(pd.isna(x)) if np.ndim(x) == 0 else False
    except (TypeError, ValueError):
        return False


def _finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def fmt_num(x: Any, digits: int = 3, signed: bool = False) -> str:
    if not _finite(x):
        return "—"
    s = f"{float(x):+,.{digits}f}" if signed else f"{float(x):,.{digits}f}"
    return s.replace("-", "−")


def fmt_p(p: Any) -> str:
    if not _finite(p):
        return "—"
    return "<0.001" if float(p) < 0.001 else f"{float(p):.3f}"


def fmt_pct(x: Any, digits: int = 1) -> str:
    return "—" if not _finite(x) else f"{100 * float(x):.{digits}f}%"


def fmt_int(x: Any) -> str:
    return "—" if not _finite(x) else f"{int(x):,}"


def fmt_bool(x: Any) -> str:
    if _missing(x):
        return "—"
    return "yes" if bool(x) else "no"


def asset_label(asset: str) -> str:
    return ASSET_LABELS.get(asset, asset)


def _hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


# --------------------------------------------------------------------------------------------- figure chrome
def _style(fig: go.Figure, title: str, subtitle: str = "", height: int = 480) -> go.Figure:
    fig.update_layout(
        template="none",
        height=height,
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT, size=12, color=INK2),
        title=dict(
            text=title,
            subtitle=dict(text=subtitle, font=dict(size=12, color=MUTED)),
            x=0,
            xref="container",
            xanchor="left",
            y=0.985,
            yanchor="top",
            font=dict(size=15, color=INK),
        ),
        margin=dict(l=64, r=24, t=96, b=40),
        legend=dict(orientation="h", x=0, xanchor="left", y=1.0, yanchor="bottom", font=dict(size=12, color=INK2),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS, font=dict(family=FONT, size=12, color=INK)),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=True, linecolor=AXIS, linewidth=1, ticks="outside",
                     tickcolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False,
                     tickfont=dict(color=MUTED), title_font=dict(color=INK2, size=12))
    return fig


def empty_figure(message: str, height: int = 360) -> go.Figure:
    fig = go.Figure()
    _style(fig, "", height=height)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=14, color=MUTED))
    return fig


# --------------------------------------------------------------------------------------------- tables
Column = tuple[str, str, Callable[[Any], Any] | None, bool]  # key, label, formatter, numeric


def html_table(df: pd.DataFrame, columns: Sequence[Column], row_class: Callable[[pd.Series], str] | None = None,
               scroll: bool = False) -> html.Div:
    """Plain HTML table (the table-view twin of every chart); numbers right-aligned in tabular figures."""
    cols = [c for c in columns if c[0] in df.columns]
    head = html.Thead(html.Tr([html.Th(label, className="num" if num else None) for _, label, _, num in cols]))
    body = []
    for _, r in df.iterrows():
        cells = []
        for key, _, fmt, num in cols:
            v = r[key]
            cells.append(html.Td(fmt(v) if fmt else ("—" if _missing(v) else str(v)),
                                 className="num" if num else None))
        body.append(html.Tr(cells, className=row_class(r) if row_class else None))
    table = html.Table([head, html.Tbody(body)], className="tbl")
    return html.Div(table, className="scroll" if scroll else "tblwrap")


def status_chip(label: str, color: str, title: str | None = None) -> html.Span:
    """Coloured dot + text label (state is never shown by colour alone)."""
    return html.Span([html.Span(className="dot", style={"background": color}), html.Span(label)],
                     className="chip", title=title)


def _zone_cell(z: Any) -> Any:
    if _missing(z):
        return "—"
    return status_chip(str(z), zone_color(z))


# --------------------------------------------------------------------------------------------- control callbacks
def model_options(store: ResultsData, asset: str | None, horizon: str | None, split: str | None,
                  current: Sequence[str] | None = None) -> tuple[list[dict], list[str]]:
    """Forecast-model options for (asset, horizon, split) and the selection kept from ``current``."""
    if not asset or not horizon or not split:
        return [], []
    avail = order_models(store.forecast_models(asset, horizon, split))
    keep = [m for m in (current or []) if m in avail]
    if not keep:
        keep = [m for m in DEFAULT_MODELS if m in avail] or avail[:3]
    return [{"label": m, "value": m} for m in avail], order_models(keep)


def risk_model_options(store: ResultsData, asset: str | None, split: str | None,
                       current: Sequence[str] | None = None) -> tuple[list[dict], list[str]]:
    if not asset or not split:
        return [], []
    avail = order_risk_models(store.risk_models(asset, split))
    keep = [m for m in (current or []) if m in avail]
    if not keep:
        keep = [m for m in DEFAULT_RISK_MODELS if m in avail] or avail[:2]
    return [{"label": m, "value": m} for m in avail], order_risk_models(keep)


# --------------------------------------------------------------------------------------------- tab 1: forecasts
def forecast_figure(store: ResultsData, asset: str | None, horizon: str | None, models: Sequence[str] | None,
                    split: str | None) -> go.Figure:
    """Annualised forecast vs realized volatility (display only) with a range slider."""
    if not asset or not horizon or not split:
        return empty_figure("No results to show — run the pipeline first.")
    ms = order_models(models or [])
    fc = store.forecasts_vs_realized(asset, horizon, ms, split)
    real = store.realized(asset, horizon, split)
    if fc.empty and real.empty:
        return empty_figure(f"No {SPLIT_LABELS.get(split, split)} forecasts for {asset} at {horizon}.")
    ann = annualisation(asset)
    fig = go.Figure()
    if not real.empty:
        fig.add_trace(go.Scatter(
            x=real["origin"], y=real["vol_realized"], name="Realized", mode="lines",
            line=dict(color=REALIZED_COLOR, width=0.8), fill="tozeroy", fillcolor=_hex_rgba(REALIZED_COLOR, 0.55),
            hovertemplate="%{y:.1f}%",
        ))
    width = 1.6 if horizon == "1d" else 2.0
    for m in ms:
        g = fc[fc["model"] == m]
        if g.empty:
            continue
        fig.add_trace(go.Scatter(
            x=g["origin"], y=g["vol_forecast"], name=m, mode="lines",
            line=dict(color=model_color(m), width=width, shape="linear"), hovertemplate="%{y:.1f}%",
        ))
    window = "next session" if horizon == "1d" else f"sessions in (t, t+{7 if horizon == '1w' else 30} days]"
    _style(
        fig,
        f"{asset_label(asset)} · {HORIZON_LABELS.get(horizon, horizon)} ahead — forecast vs realized volatility",
        f"Annualised for display only (√(variance per session × {ann})), % per year · plotted at the forecast "
        f"origin t; realized = target over the {window} · {SPLIT_LABELS.get(split, split)} sample",
        height=540,
    )
    fig.update_layout(hovermode="x unified", margin=dict(t=110))
    fig.update_yaxes(title_text="Annualised volatility (%)", rangemode="tozero", ticksuffix="%")
    fig.update_xaxes(
        rangeslider=dict(visible=True, thickness=0.07, bgcolor=PAGE, bordercolor=GRID, borderwidth=1),
        rangeselector=dict(
            buttons=[dict(count=3, label="3m", step="month", stepmode="backward"),
                     dict(count=6, label="6m", step="month", stepmode="backward"),
                     dict(count=1, label="1y", step="year", stepmode="backward"),
                     dict(step="all", label="All")],
            x=1, xanchor="right", y=1.0, yanchor="bottom", bgcolor=PAGE, activecolor=GRID,
            bordercolor=BORDER, borderwidth=1, font=dict(color=INK2, size=11),
        ),
        hoverformat="%Y-%m-%d",
    )
    return fig


# --------------------------------------------------------------------------------------------- tab 2: leaderboard
def leaderboard_rows(store: ResultsData, asset: str, split: str) -> pd.DataFrame:
    """Leaderboard rows of one asset: horizon order 1d/1w/1m, then QLIKE ratio ascending."""
    lb = store.leaderboard(split)
    if lb.empty:
        return lb
    lb = lb[lb["asset"] == asset].copy()
    lb["_h"] = lb["horizon"].map(_horizon_rank)
    return lb.sort_values(["_h", "qlike_ratio", "model"]).drop(columns="_h").reset_index(drop=True)


def leaderboard_figure(store: ResultsData, asset: str | None, split: str | None,
                       rows: pd.DataFrame | None = None) -> go.Figure:
    """Heatmap of the QLIKE ratio vs HAR (model × horizon) with 90% MCS markers."""
    if not asset or not split:
        return empty_figure("No results to show — run the pipeline first.")
    lb = leaderboard_rows(store, asset, split) if rows is None else rows
    if lb.empty:
        return empty_figure(f"No {SPLIT_LABELS.get(split, split)} leaderboard for {asset}.")
    models = order_models(lb["model"].unique())
    horizons = order_horizons(lb["horizon"].unique())
    ratio = lb.pivot_table(index="model", columns="horizon", values="qlike_ratio", aggfunc="first")
    ratio = ratio.reindex(index=models, columns=horizons)
    z = np.log(ratio.to_numpy(dtype=float))
    finite = np.abs(z[np.isfinite(z)])
    bound = float(np.clip(finite.max() if finite.size else math.log(1.1), math.log(1.1), math.log(2.0)))
    text = [[fmt_num(v) for v in row] for row in ratio.to_numpy(dtype=float)]
    has_ci = "ratio_lo" in lb.columns and lb["ratio_lo"].notna().any()
    if has_ci:
        lo = lb.pivot_table(index="model", columns="horizon", values="ratio_lo", aggfunc="first")
        hi = lb.pivot_table(index="model", columns="horizon", values="ratio_hi", aggfunc="first")
        lo, hi = lo.reindex(index=models, columns=horizons), hi.reindex(index=models, columns=horizons)
        ci = [[f"90% CI [{fmt_num(a)}, {fmt_num(b)}]" if _finite(a) else "" for a, b in zip(ra, rb, strict=True)]
              for ra, rb in zip(lo.to_numpy(dtype=float), hi.to_numpy(dtype=float), strict=True)]
    else:
        ci = [["" for _ in horizons] for _ in models]
    if split == "holdout" and DESCRIPTIVE_HOLDOUT_HORIZON in horizons:
        j = horizons.index(DESCRIPTIVE_HOLDOUT_HORIZON)
        for row in ci:
            row[j] = (row[j] + "<br>" if row[j] else "") + "1m holdout: descriptive only (SPEC §8)"
    stops = np.linspace(0, 1, len(DIVERGING))
    ticks = [r for r in (0.5, 0.67, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0) if abs(math.log(r)) <= bound + 1e-9]
    fig = go.Figure(go.Heatmap(
        x=list(range(len(horizons))), y=list(range(len(models))), z=z, zmin=-bound, zmax=bound, zmid=0,
        colorscale=[[float(s), c] for s, c in zip(stops, DIVERGING, strict=True)],
        text=text, texttemplate="%{text}", textfont=dict(size=12), customdata=np.array(ci, dtype=object),
        xgap=2, ygap=2, name="QLIKE ratio vs HAR",
        hovertemplate="%{text} × HAR's mean QLIKE<br>%{customdata}<extra></extra>",
        colorbar=dict(title=dict(text="QLIKE ratio<br>vs HAR", font=dict(size=11, color=INK2)),
                      tickvals=[math.log(t) for t in ticks], ticktext=[f"{t:g}" for t in ticks],
                      tickfont=dict(color=MUTED, size=11), outlinewidth=0, thickness=12, len=0.8),
    ))
    if "in_90" in lb.columns:
        mcs = lb[lb["in_90"].fillna(False).astype(bool)]
        xs = [horizons.index(h) + 0.36 for h in mcs["horizon"]]
        ys = [models.index(m) for m in mcs["model"]]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers", name="In 90% MCS",
            marker=dict(symbol="star", size=11, color=INK, line=dict(color=SURFACE, width=1.5)),
            customdata=np.column_stack([mcs["model"], mcs["horizon"], [fmt_p(p) for p in mcs["mcs_p"]]])
            if len(mcs) else None,
            hovertemplate="%{customdata[0]} at %{customdata[1]}: in the 90% MCS (p = %{customdata[2]})<extra></extra>",
            showlegend=True,
        ))
    # Plotly titles do not wrap: keep the subtitle short (it must fit the ≤720 px heatmap column); the longer
    # explanations and the holdout 1m caveat are wrapping HTML notes next to / under the chart.
    sub = "Mean QLIKE ÷ HAR's (<1 = better than HAR, blue) · ★ = in the 90% MCS"
    _style(fig, f"{asset_label(asset)} — forecast accuracy by horizon", sub, height=170 + 34 * len(models))
    fig.update_layout(margin=dict(l=80, r=24, t=96, b=40),
                      legend=dict(y=0, yref="container", yanchor="bottom", x=0))
    ticktext = [f"{h} · descriptive" if split == "holdout" and h == DESCRIPTIVE_HOLDOUT_HORIZON else h
                for h in horizons]
    fig.update_xaxes(tickvals=list(range(len(horizons))), ticktext=ticktext, side="top", showline=False, ticks="",
                     tickfont=dict(color=INK2, size=12), range=[-0.5, len(horizons) - 0.5])
    fig.update_yaxes(tickvals=list(range(len(models))),
                     ticktext=[f"{m} (ref)" if m == REF_MODEL else m for m in models], autorange="reversed",
                     showgrid=False, ticks="", tickfont=dict(color=INK2, size=12))
    return fig


LEADERBOARD_COLUMNS: list[Column] = [
    ("horizon_label", "Horizon", None, False),
    ("model", "Model", None, False),
    ("qlike_ratio", "QLIKE ratio vs HAR", fmt_num, True),
    ("ratio_lo", "90% CI low", fmt_num, True),
    ("ratio_hi", "90% CI high", fmt_num, True),
    ("qlike", "Mean QLIKE", lambda v: fmt_num(v, 4), True),
    ("mse", "Mean MSE (%⁴)", lambda v: fmt_num(v, 1), True),
    ("dm_hln", "DM-HLN vs HAR", lambda v: fmt_num(v, 2, signed=True), True),
    ("dm_p", "DM p", fmt_p, True),
    ("dm_p_2n", "DM p (2·n_max lags)", fmt_p, True),
    ("mcs_p", "MCS p", fmt_p, True),
    ("in_90", "In 90% MCS", fmt_bool, False),
    ("n", "Origins", fmt_int, True),
]


def is_descriptive(horizon: Any, split: str | None) -> bool:
    """True for leaderboard cells that are descriptive only (the 1m holdout, SPEC §8)."""
    return split == "holdout" and str(horizon) == DESCRIPTIVE_HOLDOUT_HORIZON


def _leaderboard_row_class(r: pd.Series, split: str | None) -> str:
    cls = ["ref"] if r["model"] == REF_MODEL else []
    if is_descriptive(r["horizon"], split):
        cls.append("descr")
    return " ".join(cls)


def leaderboard_table(rows: pd.DataFrame, split: str | None = None) -> html.Div:
    """Leaderboard table; on the holdout the 1m rows are labelled descriptive (SPEC §8) under a wrapping note."""
    if rows.empty:
        return html.P("No leaderboard table in these results.", className="note")
    rows = rows.assign(horizon_label=[f"{h} · descriptive" if is_descriptive(h, split) else str(h)
                                      for h in rows["horizon"]])
    cols = LEADERBOARD_COLUMNS if rows.get("ratio_lo", pd.Series(dtype=float)).notna().any() else [
        c for c in LEADERBOARD_COLUMNS if c[0] not in ("ratio_lo", "ratio_hi")]
    descr = split == "holdout" and (rows["horizon"].astype(str) == DESCRIPTIVE_HOLDOUT_HORIZON).any()
    return html.Div([
        html.P(HOLDOUT_1M_NOTE, className="note caveat") if descr else None,
        html_table(rows, cols, row_class=lambda r: _leaderboard_row_class(r, split), scroll=True),
    ])


def leaderboard_summary(rows: pd.DataFrame, split: str | None = None) -> html.Div:
    """Per-horizon best model and 90% MCS members (the text twin of the heatmap) with the chart's key."""
    if rows.empty:
        return html.Div()
    items = []
    for h in order_horizons(rows["horizon"].unique()):
        g = rows[rows["horizon"] == h].sort_values("qlike_ratio")
        best = g.iloc[0]
        members = order_models(g.loc[g["in_90"].fillna(False).astype(bool), "model"]) if "in_90" in g else []
        descr = is_descriptive(h, split)
        items.append(html.Li([
            html.Strong(f"{h}{' (descriptive)' if descr else ''}: "),
            f"lowest QLIKE {best['model']} ({fmt_num(best['qlike_ratio'])} × HAR); ",
            f"90% MCS: {', '.join(members) if members else 'not computed'}",
            html.Span(f" · {fmt_int(best['n'])} origins", className="muted"),
        ]))
    return html.Div([html.H3("By horizon"), html.Ul(items, className="plain"),
                     html.P(LEADERBOARD_NOTE, className="note")], className="side")


def iv_message(store: ResultsData, asset: str, split: str, rows: pd.DataFrame | None = None) -> str | None:
    """Why the 1m implied-vol comparison is empty for (asset, split), or None when there are rows."""
    if asset == "EURUSD" and split == "holdout":
        return IV_MESSAGE_EURUSD
    if rows is None:
        rows = iv_rows(store, asset, split)
    if rows.empty:
        if asset == "EURUSD":
            return "No free implied-vol benchmark available for EUR/USD in these results (EVZ series missing)."
        return f"No implied-vol benchmark rows for {asset} in the {SPLIT_LABELS.get(split, split)} results."
    return None


def iv_rows(store: ResultsData, asset: str, split: str) -> pd.DataFrame:
    iv = store.iv_comparison(split)
    if iv.empty:
        return iv
    iv = iv[iv["asset"] == asset].copy()
    iv.loc[iv["model"] == "IV-cal", "ratio_vs_ivcal"] = 1.0  # the reference of the DM-vs-IV-cal column
    return iv.sort_values(["qlike_ratio", "model"]).reset_index(drop=True)


IV_COLUMNS: list[Column] = [
    ("model", "Model", None, False),
    ("qlike_ratio", "QLIKE ratio vs HAR", fmt_num, True),
    ("ratio_vs_ivcal", "QLIKE ratio vs IV-cal", fmt_num, True),
    ("dm_p_ivcal", "DM p vs IV-cal", fmt_p, True),
    ("mcs_p", "MCS p", fmt_p, True),
    ("in_90", "In 90% MCS", fmt_bool, False),
    ("enc_c", "Encompassing c", lambda v: fmt_num(v, 3, signed=True), True),
    ("enc_p", "p (c = 0)", fmt_p, True),
    ("n", "Origins", fmt_int, True),
]


def iv_section(store: ResultsData, asset: str | None, split: str | None) -> Any:
    if not asset or not split:
        return html.P("No results.", className="note")
    rows = iv_rows(store, asset, split)
    msg = iv_message(store, asset, split, rows)
    if msg:
        return html.P(msg, className="note iv-message")
    cols = IV_COLUMNS if rows["mcs_p"].notna().any() else [c for c in IV_COLUMNS if c[0] not in ("mcs_p", "in_90")]
    table = html_table(rows, cols, row_class=lambda r: "ref" if r["model"] in ("IV", "IV-cal") else "")
    if split == "holdout":
        return html.Div([html.P(IV_HOLDOUT_NOTE, className="note caveat"), table])
    return table


def risk_leaderboard_rows(store: ResultsData, asset: str, split: str) -> pd.DataFrame:
    """Risk leaderboard of one asset (FZ0 diff, DM p and MCS from ``risk_risk_leaderboard``) with the rolling
    traffic light taken from the same sources as H4 and the VaR tab, not from the leaderboard file:

    - ``green_share``: share of rolling 250-observation windows ending in the split that are green
      (``risk_time_in_zone``, the H4 measure; see :meth:`ResultsData.green_shares`). The leaderboard file's own
      ``green_share`` covers only windows lying entirely inside the split — on the holdout that is at most a
      few windows (NaN when the split has fewer than 250 observations).
    - ``zone_last`` / ``zone_date``: zone of the last rolling window of the split (the VaR tab's strip);
      ``windows``: number of rolling windows ending in the split.
    """
    rl = store.risk_leaderboard(split)
    if rl.empty:
        return rl
    rl = rl[rl["asset"] == asset].copy()
    green, source = store.green_shares(split)
    if source:
        g = green[green["asset"] == asset].drop_duplicates("model").set_index("model")["green"]
        rl["green_share"] = rl["model"].map(g).astype(float)
    rl["green_source"] = source
    zones = store.latest_zones(asset, rl["model"].tolist(), split)
    # an all-null zone_last (no window inside a short split) comes back from DuckDB as a nullable int column
    zl = rl["zone_last"] if "zone_last" in rl else pd.Series(None, index=rl.index)
    rl["zone_last"] = [None if _missing(v) else str(v) for v in zl]
    rl["zone_date"] = pd.NaT
    rl["windows"] = np.nan
    if not zones.empty:
        z = zones.set_index("model")
        has = rl["model"].isin(z.index)
        rl.loc[has, "zone_last"] = rl.loc[has, "model"].map(z["zone"]).astype(str)
        rl["zone_date"] = pd.to_datetime(rl["model"].map(z["date"]))
        rl["windows"] = rl["model"].map(z["windows"]).astype(float)
    order = {m: i for i, m in enumerate(order_risk_models(rl["model"]))}
    return rl.sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)


RISK_COLUMNS: list[Column] = [
    ("model", "Risk model", None, False),
    ("fz0_diff", "FZ0 diff vs HS-250", lambda v: fmt_num(v, 4, signed=True), True),
    ("p_dm", "DM p vs HS-250", fmt_p, True),
    ("fz0", "Mean FZ0", lambda v: fmt_num(v, 4), True),
    ("mcs_p", "MCS p", fmt_p, True),
    ("in_90", "In 90% MCS", fmt_bool, False),
    ("zone_last", "Latest zone (99%, last 250 obs.)", _zone_cell, False),
    ("green_share", "Green share (rolling 250)", fmt_pct, True),
    ("windows", "Windows", fmt_int, True),
    ("n", "Days", fmt_int, True),
]


def risk_leaderboard_note(rows: pd.DataFrame, split: str | None) -> str:
    """What the traffic-light columns cover (the dates differ per asset and split)."""
    sample = SPLIT_LABELS.get(split or "dev", split or "dev")
    dates = pd.to_datetime(rows["zone_date"]).dropna() if "zone_date" in rows else pd.Series(dtype="datetime64[ns]")
    when = f" ending on {dates.max():%Y-%m-%d}, the last date of the {sample} sample" if len(dates) else ""
    source = str(rows["green_source"].iloc[0]) if "green_source" in rows and len(rows) else ""
    if source == "time_in_zone":
        spans = " (early windows include development observations)" if split == "holdout" else ""
        share = (f"Green share: share of the rolling 250-observation windows ending in the {sample} sample{spans} "
                 "that are in the Basel green zone at 99% — the measure H4 is decided on.")
    else:
        share = (f"Green share: share of the rolling 250-observation windows inside the {sample} sample that are "
                 "in the Basel green zone at 99% (no time-in-zone table in these results).")
    return (f"{share} Windows: number of rolling windows ending in the sample. Latest zone: the window of 250 "
            f"observations{when}. Lower FZ0 is better; HS-250 is the reference (FZ0 diff 0).")


def risk_leaderboard_table(rows: pd.DataFrame, split: str | None = None) -> Any:
    if rows.empty:
        return html.P("No risk leaderboard in these results.", className="note")
    return html.Div([
        html_table(rows, RISK_COLUMNS, row_class=lambda r: "ref" if r["model"] == "COMBO+FHS" else ""),
        html.P(risk_leaderboard_note(rows, split), className="note risk-note"),
    ])


def update_leaderboard(store: ResultsData, asset: str | None, split: str | None) -> tuple:
    """Leaderboard-tab callback: (heatmap figure, per-horizon summary, table, IV section, risk table)."""
    if not asset or not split:
        empty = html.P("No results.", className="note")
        return empty_figure("No results to show — run the pipeline first."), html.Div(), empty, empty, empty
    rows = leaderboard_rows(store, asset, split)
    return (
        leaderboard_figure(store, asset, split, rows),
        leaderboard_summary(rows, split),
        leaderboard_table(rows, split),
        iv_section(store, asset, split),
        risk_leaderboard_table(risk_leaderboard_rows(store, asset, split), split),
    )


# --------------------------------------------------------------------------------------------- tab 3: VaR
def var_figure(store: ResultsData, asset: str | None, risk_models: Sequence[str] | None,
               split: str | None) -> go.Figure:
    """Daily returns with −VaR99/−VaR97.5, breach markers, the rolling-250 traffic light and the plus factor."""
    if not asset or not split:
        return empty_figure("No results to show — run the pipeline first.")
    ms = order_risk_models(risk_models or [])
    if not ms:
        return empty_figure("Select one or more risk models.")
    rs = store.risk_series(asset, ms, split)
    if rs.empty:
        return empty_figure(f"No {SPLIT_LABELS.get(split, split)} VaR results for {asset}.")
    ms = [m for m in ms if (rs["model"] == m).any()]
    zones = store.rolling_zones(asset, ms, split)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, row_heights=[0.64, 0.14, 0.22], vertical_spacing=0.07,
        subplot_titles=("Daily return and −VaR, % (one unit long)",
                        "Rolling 250-day Basel traffic light (VaR99 exceptions)",
                        "Basel plus factor (rolling 250 days)"),
    )
    ret = rs.drop_duplicates("date").sort_values("date")
    fig.add_trace(go.Bar(x=ret["date"], y=ret["r_cc"], name="Daily return", marker=dict(color=RETURN_COLOR,
                         line=dict(width=0)), hovertemplate="return %{y:.2f}%<extra></extra>"), row=1, col=1)
    for m in ms:
        g = rs[rs["model"] == m].sort_values("date")
        c = model_color(m)
        loss = -g["r_cc"]
        b99 = g[loss > g["var99"]]
        b975 = g[(loss > g["var975"]) & ~(loss > g["var99"])]
        fig.add_trace(go.Scatter(x=g["date"], y=-g["var99"], mode="lines", name=f"{m} −VaR99", legendgroup=m,
                                 line=dict(color=c, width=1.6), hovertemplate=f"{m} −VaR99 %{{y:.2f}}%<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=g["date"], y=-g["var975"], mode="lines", name=f"{m} −VaR97.5", legendgroup=m,
                                 line=dict(color=c, width=1.2, dash="dot"),
                                 hovertemplate=f"{m} −VaR97.5 %{{y:.2f}}%<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=b99["date"], y=b99["r_cc"], mode="markers", legendgroup=m,
                                 name=f"{m} breaches of VaR99 ({len(b99)})",
                                 marker=dict(symbol="circle", size=9, color=c, line=dict(color=SURFACE, width=2)),
                                 hovertemplate=f"{m}: VaR99 breach, return %{{y:.2f}}%<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=b975["date"], y=b975["r_cc"], mode="markers", legendgroup=m,
                                 name=f"{m} breaches of VaR97.5 only ({len(b975)})",
                                 marker=dict(symbol="circle-open", size=8, color=c, line=dict(color=c, width=2)),
                                 hovertemplate=f"{m}: VaR97.5 breach, return %{{y:.2f}}%<extra></extra>"),
                      row=1, col=1)
    if not zones.empty:
        zm = [m for m in ms if (zones["model"] == m).any()]
        zone = zones.pivot(index="model", columns="date", values="zone").reindex(zm)
        exc = zones.pivot(index="model", columns="date", values="exceptions").reindex(zm)
        pf = zones.pivot(index="model", columns="date", values="plus_factor").reindex(zm)
        z = zone.apply(lambda col: col.map(lambda v: ZONE_CODES.get(str(v).lower(), np.nan))).to_numpy(dtype=float)
        g_, y_, r_ = (zone_color(k) for k in ("green", "yellow", "red"))
        fig.add_trace(go.Heatmap(
            x=zone.columns, y=zm, z=z, zmin=-0.5, zmax=2.5, showscale=False, ygap=3, name="Traffic light",
            colorscale=[[0, g_], [1 / 3, g_], [1 / 3, y_], [2 / 3, y_], [2 / 3, r_], [1, r_]],
            text=zone.to_numpy(dtype=object),
            customdata=np.dstack([exc.to_numpy(dtype=float), pf.to_numpy(dtype=float)]),
            hovertemplate="%{y} · %{x|%Y-%m-%d}<br>%{text} zone: %{customdata[0]:.0f} exceptions in the last 250 days"
                          "<br>plus factor +%{customdata[1]:.2f}<extra></extra>",
        ), row=2, col=1)
        for m in zm:
            g = zones[zones["model"] == m]
            fig.add_trace(go.Scatter(x=g["date"], y=g["plus_factor"], mode="lines", name=f"{m} plus factor",
                                     legendgroup=m, showlegend=False,
                                     line=dict(color=model_color(m), width=2, shape="hv"),
                                     hovertemplate=f"{m} plus factor +%{{y:.2f}}<extra></extra>"), row=3, col=1)
        for k in ("green", "yellow", "red"):
            fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", legendgroup="zones", hoverinfo="skip",
                                     name=f"{k.capitalize()} zone ({ZONE_RULES[k]})",
                                     marker=dict(symbol="square", size=11, color=zone_color(k))), row=1, col=1)
    n_days = ret["date"].nunique()
    _style(fig, f"{asset_label(asset)} — VaR breaches and Basel traffic light",
           f"Hypothetical 1-unit long position, μ = 0 · filled dots: loss beyond VaR99, open circles: beyond VaR97.5 "
           f"only · {n_days:,} days, {SPLIT_LABELS.get(split, split)} sample", height=820)
    fig.update_layout(hovermode="closest", bargap=0, margin=dict(l=104, r=24, t=96, b=40),
                      legend=dict(orientation="v", x=1.02, xanchor="left", y=1.0, yanchor="top",
                                  traceorder="grouped", tracegroupgap=14, font=dict(size=11, color=INK2)))
    fig.update_annotations(font=dict(size=12, color=INK2), xanchor="left", x=0)
    fig.update_yaxes(title_text="% per day", ticksuffix="%", row=1, col=1)
    fig.update_yaxes(showgrid=False, ticks="", tickfont=dict(color=INK2, size=11), row=2, col=1)
    fig.update_yaxes(title_text="plus factor", range=[-0.03, 1.05], tickvals=[0, 0.4, 0.75, 1.0], row=3, col=1)
    fig.update_xaxes(showticklabels=True, hoverformat="%Y-%m-%d", row=3, col=1)
    return fig


def breach_rows(store: ResultsData, asset: str, risk_models: Sequence[str] | None, split: str) -> pd.DataFrame:
    """Breach counts per (model, level) from the VaR series, joined with the backtest p-values."""
    ms = order_risk_models(risk_models or [])
    rs = store.risk_series(asset, ms, split)
    cols = ["model", "level", "x", "T", "rate", "expected", "p_binom", "p_cc_mc", "p_dq", "zone_full"]
    if rs.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for m in [m for m in ms if (rs["model"] == m).any()]:
        g = rs[rs["model"] == m]
        for level, col, p in (("99", "var99", 0.01), ("97.5", "var975", 0.025)):
            x = int((-g["r_cc"] > g[col]).sum())
            rows.append({"model": m, "level": level, "x": x, "T": len(g), "rate": x / len(g), "expected": p})
    out = pd.DataFrame(rows)
    bt = store.backtests(split)
    if not bt.empty:
        bt = bt[bt["asset"] == asset].assign(level=lambda d: d["level"].astype(str))
        keep = [c for c in ("model", "level", "p_binom", "p_cc_mc", "p_dq", "zone_full") if c in bt.columns]
        out = out.merge(bt[keep], on=["model", "level"], how="left")
    for c in cols:
        if c not in out:
            out[c] = np.nan
    return out[cols]


BREACH_COLUMNS: list[Column] = [
    ("model", "Risk model", None, False),
    ("level", "VaR level (%)", None, False),
    ("x", "Breaches", fmt_int, True),
    ("T", "Days", fmt_int, True),
    ("rate", "Breach rate", lambda v: fmt_pct(v, 2), True),
    ("expected", "Expected", lambda v: fmt_pct(v, 1), True),
    ("p_binom", "Kupiec p (binomial)", fmt_p, True),
    ("p_cc_mc", "Christoffersen p (cc)", fmt_p, True),
    ("p_dq", "DQ p", fmt_p, True),
    ("zone_full", "Zone (whole window)", _zone_cell, False),
]


def update_var(store: ResultsData, asset: str | None, risk_models: Sequence[str] | None, split: str | None) -> tuple:
    """VaR-tab callback: (figure, breach table)."""
    fig = var_figure(store, asset, risk_models, split)
    if not asset or not split:
        return fig, html.Div()
    rows = breach_rows(store, asset, risk_models, split)
    table = html_table(rows, BREACH_COLUMNS) if not rows.empty else html.P("No VaR series selected.", className="note")
    return fig, table


# --------------------------------------------------------------------------------------------- hypotheses
VERDICT_COLORS = {"supported": "green", "not supported": "red"}


def _verdict_chip(v: Any) -> html.Span:
    s = "n/a" if _missing(v) else str(v)
    zone = VERDICT_COLORS.get(s.lower())
    return status_chip(s, zone_color(zone) if zone else NA_COLOR)


def hypotheses_panel(store: ResultsData, split: str | None) -> list:
    """H1–H5 verdicts of ``split`` from reports/tables/hypotheses*.csv (the report stage computes them)."""
    split = split or "dev"
    df = store.hypotheses(split)
    if df.empty:
        items = [html.Li([html.Strong(h), f" {s}", html.Span(" · pending", className="muted")])
                 for h, s in STATEMENTS.items()]
        return [html.P(f"No {SPLIT_LABELS.get(split, split)} verdicts yet — they appear once the report stage "
                       "has written reports/tables/hypotheses*.csv.", className="note"),
                html.Ul(items, className="plain")]
    if not {"hypothesis", "verdict"} <= set(df.columns):
        return [html_table(df, [(c, c, None, False) for c in df.columns])]
    out = []
    for h in sorted(df["hypothesis"].astype(str).unique()):
        g = df[df["hypothesis"].astype(str) == h]
        if "unit" in g.columns:
            ov = g[g["unit"].astype(str) == "overall"]
            units = g[g["unit"].astype(str) != "overall"]
        else:
            ov, units = g.head(1), g.iloc[0:0]
        overall = ov["verdict"].iloc[0] if len(ov) else (g["verdict"].iloc[0] if len(g) else None)
        if "required" in units.columns:
            units = units[units["required"].astype(str).str.lower().isin(("true", "1"))]
        chips = [html.Span([f"{u}: ", _verdict_chip(v)], className="unit")
                 for u, v in zip(units.get("unit", []), units.get("verdict", []), strict=False)]
        note = ov["note"].iloc[0] if len(ov) and "note" in ov.columns else None
        out.append(html.Div([
            html.Div(h, className="hid"),
            html.Div([
                html.Div(STATEMENTS.get(h, "")),
                html.Div(chips, className="units") if chips else None,
                html.Div(str(note), className="muted") if isinstance(note, str) and note.strip() else None,
            ], className="hbody"),
            html.Div(_verdict_chip(overall), className="hverdict"),
        ], className="hrow"))
    return out


# --------------------------------------------------------------------------------------------- layout
def _control(label: str, component: Any, note: Any = None, cls: str = "") -> html.Div:
    return html.Div([html.Label(label, className="ctl"), component, note], className=f"control {cls}".strip())


def _card(title: str | None, sub: str | None, *children: Any, cls: str = "") -> html.Section:
    head = ([html.H2(title)] if title else []) + ([html.P(sub, className="sub")] if sub else [])
    return html.Section(head + list(children), className=f"card {cls}".strip())


def _meta_line(store: ResultsData, splits: list[str], assets: list[str]) -> str:
    parts = []
    t = store.mtime("dev")
    parts.append(f"Development results: {store.results_dir.as_posix()}"
                 + (f" (updated {t:%Y-%m-%d %H:%M})" if t is not None else " (none found)"))
    parts.append(f"assets: {', '.join(assets) if assets else 'none'}")
    parts.append("holdout: results available" if "holdout" in splits else "holdout: sealed, no results yet")
    return " · ".join(parts)


def build_layout(store: ResultsData) -> html.Div:
    splits = store.available_splits()
    assets = store.available_assets()
    holdout_ready = "holdout" in splits
    split0 = "dev" if "dev" in splits else (splits[0] if splits else "dev")
    asset0 = (store.available_assets(split0) or assets or [None])[0]
    split_note = None if holdout_ready else html.Div(
        "Holdout disabled until the one-time holdout run writes data/results/holdout/.",
        className="note", id="split-note")
    tab_style = {"padding": "8px 16px", "border": "none", "borderBottom": f"2px solid {BORDER}",
                 "background": "transparent", "color": INK2, "fontWeight": 500}
    tab_selected = {**tab_style, "borderBottom": f"2px solid {INK}", "color": INK, "fontWeight": 600}
    controls = html.Div([
        _control("Asset", dcc.Dropdown(id="asset", options=[{"label": asset_label(a), "value": a} for a in assets],
                                       value=asset0, clearable=False, searchable=False)),
        _control("Horizon (forecasts)", dcc.RadioItems(
            id="horizon", options=[{"label": HORIZON_LABELS[h], "value": h} for h in HORIZON_ORDER], value="1d",
            inline=True, className="radio")),
        _control("Sample", dcc.RadioItems(
            id="split", inline=True, className="radio", value=split0,
            options=[{"label": "Development", "value": "dev", "disabled": "dev" not in splits},
                     {"label": "Holdout", "value": "holdout", "disabled": not holdout_ready}]), split_note),
        _control("Forecast models", dcc.Dropdown(id="models", multi=True, value=list(DEFAULT_MODELS),
                                                 placeholder="Select forecast models"), cls="wide"),
        _control("Risk models (VaR tab)", dcc.Dropdown(id="risk-models", multi=True,
                                                       value=list(DEFAULT_RISK_MODELS),
                                                       placeholder="Select risk models"), cls="wide"),
    ], className="controls")
    tabs = dcc.Tabs(id="tabs", value="forecasts", className="tabs", children=[
        dcc.Tab(label="Forecasts vs realized", value="forecasts", style=tab_style, selected_style=tab_selected,
                children=[_card(
                    None, None,
                    dcc.Graph(id="forecast-graph", config=GRAPH_CONFIG, figure=empty_figure("Loading…")),
                    html.P("Forecasts F are cumulative variances over the horizon window (%²); the chart shows "
                           "√(F/n·ann) and √(y/n·ann) for readability only — every comparison in the project uses "
                           "the variances themselves (QLIKE).", className="note"))]),
        dcc.Tab(label="Leaderboard", value="leaderboard", style=tab_style, selected_style=tab_selected,
                children=[
                    _card("Forecast leaderboard",
                          "QLIKE ratio vs HAR per model and horizon, DM-HLN test vs HAR and the 90% model confidence "
                          "set (MCS) — headline evaluation window.",
                          html.Div([dcc.Graph(id="lb-heatmap", config=GRAPH_CONFIG,
                                               figure=empty_figure("Loading…")),
                                    html.Div(id="lb-summary")], className="lbgrid"),
                          html.Div(id="lb-table")),
                    _card("1-month implied-vol comparison",
                          "On each asset's implied-vol subsample: ratios vs HAR and vs calibrated IV (IV-cal), DM p vs "
                          "IV-cal, IV-inclusive MCS and the encompassing coefficient c of each model beyond IV (H3).",
                          html.Div(id="iv-table")),
                    _card("Risk leaderboard",
                          "FZ0 joint VaR/ES score at 97.5% (lower is better): mean difference vs HS-250 with DM-HLN, "
                          "MCS membership, latest rolling-250 Basel zone and the share of green windows (H4).",
                          html.Div(id="risk-lb")),
                ]),
        dcc.Tab(label="VaR breaches", value="var", style=tab_style, selected_style=tab_selected,
                children=[_card(None, None,
                                dcc.Graph(id="var-graph", config=GRAPH_CONFIG, figure=empty_figure("Loading…")),
                                html.H3("Breaches and backtests"),
                                html.Div(id="breach-table"))]),
    ])
    hyp = _card("Pre-registered hypotheses (H1–H5)",
                "Mechanical verdicts at the 5% level from the report tables (SPEC §0); the holdout verdict is a "
                "low-power confirmation.", html.Div(id="hypotheses", children=hypotheses_panel(store, split0)),
                cls="hyp")
    return html.Div([
        html.Header([html.H1(TITLE), html.P(DESCRIPTION, className="lede"),
                     html.P(_meta_line(store, splits, assets), className="meta")]),
        controls,
        tabs,
        hyp,
        html.Footer("Read-only viewer · no model fitting, downloads or live data · annualisation is for display only.",
                    className="muted"),
    ], className="page")


INDEX_STRING = """<!DOCTYPE html>
<html lang="en">
<head>
{%metas%}
<title>{%title%}</title>
{%favicon%}
{%css%}
<style>
:root { color-scheme: light; --page: #F6F6F3; --surface: #FFFFFF; --ink: #0B0B0B; --ink2: #52514E;
        --muted: #898781; --grid: #E1E0D9; --axis: #C3C2B7; --border: #E4E3DD; --ref: #F7F7F4;
        --Dash-Fill-Interactive-Strong: #2F2E2B; --Dash-Text-Primary: #0B0B0B; }
html, body { background: var(--page); }
body { margin: 0; color: var(--ink); font-family: -apple-system, "Segoe UI", system-ui, Roboto, "Helvetica Neue",
       Arial, sans-serif; font-size: 14px; line-height: 1.45; }
.page { max-width: 1360px; margin: 0 auto; padding: 20px 24px 32px; }
header h1 { font-size: 22px; font-weight: 650; margin: 0 0 4px; letter-spacing: -0.01em; }
header .lede { margin: 0; color: var(--ink2); }
header .meta { margin: 6px 0 0; color: var(--muted); font-size: 12px; }
.controls { display: grid; grid-template-columns: minmax(170px, 210px) auto auto minmax(200px, 1fr) minmax(200px, 1fr);
            gap: 12px 20px; align-items: start; background: var(--surface); border: 1px solid var(--border);
            border-radius: 8px; padding: 12px 16px; margin: 16px 0 8px; }
@media (max-width: 1180px) { .controls { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
.control .ctl { display: block; font-size: 12px; font-weight: 600; color: var(--ink2); margin-bottom: 6px; }
.radio label { margin-right: 14px; white-space: nowrap; cursor: pointer; }
.radio input { margin-right: 5px; }
.note { color: var(--muted); font-size: 12px; margin: 6px 0 0; }
.control .note { max-width: 250px; }
.muted { color: var(--muted); }
.tabs { margin-top: 8px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 14px 16px;
        margin-top: 14px; }
.card h2 { font-size: 15px; font-weight: 600; margin: 0 0 2px; }
.card h3 { font-size: 13px; font-weight: 600; margin: 14px 0 6px; color: var(--ink2); }
.card .sub { color: var(--ink2); font-size: 12px; margin: 0 0 10px; }
.lbgrid { display: grid; grid-template-columns: minmax(0, 720px) minmax(240px, 1fr); gap: 16px; align-items: start; }
@media (max-width: 1100px) { .lbgrid { grid-template-columns: 1fr; } }
.side h3 { margin-top: 8px; }
ul.plain { list-style: none; padding: 0; margin: 0; }
ul.plain li { padding: 5px 0; border-bottom: 1px solid var(--grid); font-size: 13px; }
.tblwrap { overflow-x: auto; }
.scroll { max-height: 460px; overflow: auto; margin-top: 10px; border-top: 1px solid var(--grid); }
table.tbl { border-collapse: collapse; width: 100%; font-size: 12.5px; }
table.tbl th { text-align: left; font-weight: 600; color: var(--ink2); padding: 7px 8px;
               border-bottom: 1px solid var(--axis); background: var(--surface); position: sticky; top: 0;
               white-space: nowrap; }
table.tbl td { padding: 5px 8px; border-bottom: 1px solid var(--grid); white-space: nowrap; }
table.tbl .num { text-align: right; font-variant-numeric: tabular-nums; }
table.tbl tr.ref td { background: var(--ref); font-weight: 600; }
table.tbl tr.descr td { color: var(--ink2); font-style: italic; }
.note.caveat { color: var(--ink2); font-size: 12.5px; margin: 10px 0 4px; padding: 6px 10px;
               border-left: 3px solid var(--axis); background: var(--ref); max-width: 960px; }
table.tbl tbody tr:hover td { background: #F1F0EC; }
.chip { display: inline-flex; align-items: center; gap: 6px; }
.dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; flex: none; }
.hrow { display: grid; grid-template-columns: 40px minmax(0, 1fr) 140px; gap: 12px; padding: 8px 0;
        border-bottom: 1px solid var(--grid); align-items: start; }
.hid { font-weight: 700; }
.hverdict { font-weight: 600; }
.units { display: flex; flex-wrap: wrap; gap: 4px 14px; margin-top: 4px; font-size: 12px; color: var(--ink2); }
.iv-message { font-size: 13px; color: var(--ink2); }
footer { margin-top: 18px; font-size: 12px; }
</style>
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>"""


# --------------------------------------------------------------------------------------------- app
def create_app(results_dir: str | Path | None = None, reports_dir: str | Path | None = None,
               holdout_dir: str | Path | None = None) -> dash.Dash:
    """Build the dashboard over ``results_dir`` (default ``data/results``; holdout results are read from
    ``<results_dir>/holdout`` when present) and ``reports_dir`` (default ``reports``)."""
    store = ResultsData(results_dir, reports_dir, holdout_dir)
    app = dash.Dash(__name__, title="Volatility & risk dashboard", update_title=None)
    app.index_string = INDEX_STRING
    app.layout = build_layout(store)
    app.server.config["VOLRISK_DATA"] = store

    @app.callback(Output("models", "options"), Output("models", "value"),
                  Input("asset", "value"), Input("horizon", "value"), Input("split", "value"),
                  State("models", "value"))
    def _models(asset, horizon, split, current):
        return model_options(store, asset, horizon, split, current)

    @app.callback(Output("risk-models", "options"), Output("risk-models", "value"),
                  Input("asset", "value"), Input("split", "value"), State("risk-models", "value"))
    def _risk_models(asset, split, current):
        return risk_model_options(store, asset, split, current)

    @app.callback(Output("forecast-graph", "figure"),
                  Input("asset", "value"), Input("horizon", "value"), Input("models", "value"),
                  Input("split", "value"))
    def _forecasts(asset, horizon, models, split):
        return forecast_figure(store, asset, horizon, models, split)

    @app.callback(Output("lb-heatmap", "figure"), Output("lb-summary", "children"), Output("lb-table", "children"),
                  Output("iv-table", "children"), Output("risk-lb", "children"),
                  Input("asset", "value"), Input("split", "value"))
    def _leaderboard(asset, split):
        return update_leaderboard(store, asset, split)

    @app.callback(Output("var-graph", "figure"), Output("breach-table", "children"),
                  Input("asset", "value"), Input("risk-models", "value"), Input("split", "value"))
    def _var(asset, risk_models, split):
        return update_var(store, asset, risk_models, split)

    @app.callback(Output("hypotheses", "children"), Input("split", "value"))
    def _hypotheses(split):
        return hypotheses_panel(store, split)

    return app
