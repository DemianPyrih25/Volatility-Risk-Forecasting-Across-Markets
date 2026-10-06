"""Static report (SPEC §12): Markdown/CSV tables, PNG figures and ``reports/results.md``.

Inputs are the tables written by ``pipeline.stage_evaluate`` / ``stage_risk`` in ``data/results/`` (development)
and, once the holdout has been opened by ``python -m volrisk holdout --unlock-holdout``, the same file names in
``data/results/holdout/``. The report only *reads* result tables: it never opens ``data/holdout/`` and never
writes into a results directory. Holdout figures use the holdout run's ``forecasts`` / ``risk`` files, which
contain the reproduced development rows as well, and shade the holdout period.

Everything is asset-agnostic: any subset of BTC, ETH, EURUSD, SPX is handled, missing assets show as "n/a".
"""

from __future__ import annotations

import logging
import math
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import dates as mdates
from matplotlib import patheffects as pe
from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator, PercentFormatter

from volrisk import config as C
from volrisk import hypotheses as H
from volrisk import palette as P
from volrisk.hypotheses import fmt_num, fmt_p, fmt_share

log = logging.getLogger(__name__)

DPI = 150
IV_MODELS = ("IV", "IV-cal")
FIG_MODELS_1D = ("HAR", "GJR", "COMBO")
CUM_MODELS = ("GJR", "LGBM", "MLP", "COMBO")
CUM_STYLES = {"GJR": "-", "LGBM": (0, (5, 2)), "MLP": (0, (1.5, 1.5)), "COMBO": "-"}
NEUTRAL_BAR = "#5E6A75"
REALIZED_DOT = "#A9A69F"  # realized-volatility dots behind the 1d forecasts (not a model colour)
# 1m implied-vol chart: IV-cal (the DM reference) is the lightest model colour (2.1:1 even on white), so it gets a
# heavier dashed stroke with a surface ring: its identity never rests on contrast against the realized wash.
IV_LINE_STYLES = {"IV": {"lw": 1.3, "ls": "-", "zorder": 3}, "IV-cal": {"lw": 2.2, "ls": (0, (4, 1.6)), "zorder": 4},
                  "COMBO": {"lw": 1.8, "ls": "-", "zorder": 5}}
VERDICT_MARK = {H.SUPPORTED: "✓ supported", H.NOT_SUPPORTED: "✗ not supported", H.NA: "n/a"}

RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": P.INK,
    "axes.labelsize": 9,
    "axes.labelcolor": P.INK2,
    "axes.edgecolor": P.AXIS,
    "axes.linewidth": 0.8,
    "axes.facecolor": P.SURFACE,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": P.GRID,
    "grid.linewidth": 0.7,
    "grid.linestyle": "-",
    "xtick.color": P.AXIS,
    "ytick.color": P.AXIS,
    "xtick.labelcolor": P.INK2,
    "ytick.labelcolor": P.INK2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "legend.frameon": False,
    "figure.facecolor": P.SURFACE,
    "text.color": P.INK,
    "lines.solid_capstyle": "round",
    "lines.solid_joinstyle": "round",
}


# ============================================================================================= inputs
@dataclass
class Results:
    """Lazy reader for one mode's result tables (``eval_*``, ``risk_*``, ``forecasts``, ``targets``, ``risk``)."""

    mode: str
    path: Path
    _cache: dict = field(default_factory=dict, repr=False)

    def __getitem__(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            df = H.read_table(self.path, name)
            for col in ("origin", "date", "window_end", "start", "end"):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col]).astype("datetime64[ns]")
            self._cache[name] = df
        return self._cache[name]


def holdout_available(results_dir: Path | str | None = None) -> bool:
    """True when the one-time holdout run has written its evaluation tables."""
    rd = Path(results_dir) if results_dir is not None else C.RESULTS
    return (rd / "holdout" / "eval_leaderboard.parquet").exists()


def _assets(*frames: pd.DataFrame) -> list[str]:
    seen: list[str] = []
    for f in frames:
        if not f.empty and "asset" in f:
            seen += [str(a) for a in f["asset"].unique()]
    return P.order_assets(seen)


def _model_rank(m: str) -> int:
    if m in P.MODEL_ORDER:
        return P.MODEL_ORDER.index(m)
    if m in P.RISK_ORDER:
        return P.RISK_ORDER.index(m)
    return 99


def _sort(df: pd.DataFrame) -> pd.DataFrame:
    """Rows in display order: asset, horizon (1d, 1w, 1m), model."""
    if df.empty:
        return df
    d = df.copy()
    keys = []
    for col, fn in (("asset", P.asset_rank), ("horizon", P.horizon_rank), ("model", _model_rank)):
        if col in d:
            d[f"_{col}"] = d[col].astype(str).map(fn)
            keys.append(f"_{col}")
    return d.sort_values(keys, kind="stable").drop(columns=keys).reset_index(drop=True)


def _lookup(df: pd.DataFrame, **eq) -> pd.Series | None:
    if df.empty or any(k not in df.columns for k in eq):
        return None
    m = np.ones(len(df), dtype=bool)
    for k, v in eq.items():
        m &= (df[k].astype(str) == str(v)).to_numpy()
    sub = df[m]
    return None if sub.empty else sub.iloc[0]


def _val(r: pd.Series | None, col: str):
    return None if r is None or col not in r.index else r[col]


def _ann(asset: str) -> int:
    if asset in P.ANNUALISATION:
        return P.ANNUALISATION[asset]
    try:
        return {"crypto": 365, "fx": 260, "xnys": 252}[C.clock(asset)]
    except (KeyError, TypeError):
        return 252


# ============================================================================================= markdown
def esc(s) -> str:
    """Escape Markdown table metacharacters in a plain-text cell (model names such as HAR*+FHS)."""
    return str(s).replace("\\", "\\\\").replace("|", "\\|").replace("*", "\\*").replace("_", "\\_")


def bold(s: str) -> str:
    return f"**{s}**" if s and s != "—" else s


def md_table(header: Sequence[str], rows: Sequence[Sequence[str]], align: Sequence[str] | None = None) -> str:
    """GitHub-flavoured Markdown table; first column left-aligned, the rest right-aligned by default."""
    align = list(align) if align is not None else ["l"] + ["r"] * (len(header) - 1)
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(str(h) for h in header) + " |", "| " + " | ".join(sep[a] for a in align) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("\n", " ") for c in r) + " |")
    return "\n".join(out)


def _n(x) -> str:
    try:
        return f"{int(x):,}"
    except (TypeError, ValueError):
        return "—"


def _verdict(v: str) -> str:
    return VERDICT_MARK.get(v, v)


@dataclass
class Table:
    name: str
    title: str
    caption: str
    body: str
    data: pd.DataFrame

    def markdown(self, level: int = 1) -> str:
        return f"{'#' * level} {self.title}\n\n{self.caption}\n\n{self.body}\n"


NO_RESULTS = "_No results in this run._"


# ============================================================================================= tables
def _cells(lb: pd.DataFrame) -> list[tuple[str, str]]:
    cells = []
    for a in P.order_assets(lb["asset"]):
        for h in P.order_horizons(lb.loc[lb["asset"] == a, "horizon"]):
            cells.append((a, h))
    return cells


def leaderboard_table(
    lb: pd.DataFrame, mcs: pd.DataFrame, ci: pd.DataFrame | None = None, *, name: str, title: str, caption: str
) -> Table:
    """QLIKE ratio vs HAR: models × (asset, horizon); bold = in the 90% MCS; optional bootstrap CI; n footer."""
    if lb.empty:
        return Table(name, title, caption, NO_RESULTS, pd.DataFrame())
    cells = _cells(lb)
    models = P.order_models(lb["model"])
    in90 = set()
    if not mcs.empty and {"asset", "horizon", "model", "in_90"} <= set(mcs.columns):
        in90 = {(str(r.asset), str(r.horizon), str(r.model)) for r in mcs.itertuples() if bool(r.in_90)}
    has_ci = ci is not None and not ci.empty
    rows = []
    for m in models:
        rec = [esc(m)]
        for a, h in cells:
            r = _lookup(lb, asset=a, horizon=h, model=m)
            if r is None:
                rec.append("—")
                continue
            txt = fmt_num(r["qlike_ratio"])
            if has_ci and m != "HAR":
                c = _lookup(ci, asset=a, horizon=h, model=m)
                if c is not None:
                    txt += f" [{fmt_num(c['lo'])}, {fmt_num(c['hi'])}]"
            rec.append(bold(txt) if (a, h, m) in in90 else txt)
        rows.append(rec)
    n_row = ["n"] + [_n(_val(_lookup(lb, asset=a, horizon=h), "n")) for a, h in cells]
    body = md_table(["model", *[f"{a} {h}" for a, h in cells]], [*rows, n_row])

    data = lb.copy()
    if not mcs.empty and {"asset", "horizon", "model", "pvalue"} <= set(mcs.columns):
        mc = mcs[["asset", "horizon", "model", "pvalue", "in_90", "in_75"]].rename(
            columns={"pvalue": "mcs_p", "in_90": "in_mcs90", "in_75": "in_mcs75"})
        data = data.merge(mc, on=["asset", "horizon", "model"], how="left")
    if has_ci:
        data = data.merge(ci[["asset", "horizon", "model", "lo", "hi"]].rename(
            columns={"lo": "ci90_lo", "hi": "ci90_hi"}), on=["asset", "horizon", "model"], how="left")
    return Table(name, title, caption, body, _sort(data))


def dm_table(dm: pd.DataFrame, *, name: str, title: str, caption: str) -> Table:
    """DM-HLN vs HAR per asset: statistic, p-value (default kernel) and the 2·n_max Bartlett p-value."""
    if dm.empty:
        return Table(name, title, caption, NO_RESULTS, pd.DataFrame())
    parts = []
    for a in P.order_assets(dm["asset"]):
        g = dm[dm["asset"] == a]
        hs = P.order_horizons(g["horizon"])
        header = ["model"]
        for h in hs:
            header += [f"{h} DM", f"{h} p", f"{h} p (2n lags)"]
        rows = []
        for m in P.order_models(g["model"]):
            rec = [esc(m)]
            for h in hs:
                r = _lookup(g, horizon=h, model=m)
                rec += [fmt_num(_val(r, "dm_hln"), 2, signed=True), fmt_p(_val(r, "pvalue")),
                        fmt_p(_val(r, "pvalue_2n"))]
            rows.append(rec)
        t_row, k_row = ["T"], ["kernel (lags)"]
        for h in hs:
            r = _lookup(g, horizon=h)
            t_row += [_n(_val(r, "T")), "", ""]
            k_row += [f"{_val(r, 'kernel')} ({_val(r, 'maxlags')})" if r is not None else "—", "", ""]
        parts.append(f"**{a}**\n\n" + md_table(header, [*rows, t_row, k_row]))
    keep = [c for c in ("asset", "horizon", "model", "ref", "ratio", "mean_diff", "dm_hln", "pvalue", "pvalue_2n",
                        "T", "maxlags", "kernel") if c in dm.columns]
    return Table(name, title, caption, "\n\n".join(parts), _sort(dm[keep]))


def iv_table(
    iv_lb: pd.DataFrame, iv_dm: pd.DataFrame, iv_mcs: pd.DataFrame, assets: Sequence[str], *, name: str,
    title: str, caption: str,
) -> Table:
    """1m comparison with implied vol on each asset's IV subsample; a row per asset without a benchmark."""
    header = ["asset", "model", "QLIKE ratio vs HAR", "QLIKE ratio vs IV-cal", "DM vs IV-cal", "p", "p (2n lags)",
              "90% MCS", "n"]
    rows, data = [], []
    has_mcs = not iv_mcs.empty and "in_90" in iv_mcs.columns
    for a in assets:
        g = iv_lb[iv_lb["asset"] == a] if not iv_lb.empty else iv_lb
        if g.empty:
            msg = ("no free benchmark available (EVZ discontinued; used only up to 2023-12-31)"
                   if a == "EURUSD" else "no implied-vol benchmark available")
            rows.append([a, msg, *["—"] * (len(header) - 2)])
            data.append({"asset": a, "horizon": "1m", "model": None, "note": msg})
            continue
        for i, m in enumerate(P.order_models(g["model"])):
            r = _lookup(g, model=m)
            d = _lookup(iv_dm, asset=a, model=m) if not iv_dm.empty else None
            mc = _lookup(iv_mcs, asset=a, model=m) if has_mcs else None
            ratio_ivcal = _val(d, "ratio") if d is not None else (1.0 if m == "IV-cal" else None)
            in_mcs = "—" if mc is None else ("yes" if bool(mc["in_90"]) else "no")
            label = bold(esc(m)) if m in IV_MODELS else esc(m)
            rows.append([a if i == 0 else "", label, fmt_num(r["qlike_ratio"]), fmt_num(ratio_ivcal),
                         fmt_num(_val(d, "dm_hln"), 2, signed=True), fmt_p(_val(d, "pvalue")),
                         fmt_p(_val(d, "pvalue_2n")), in_mcs, _n(r["n"])])
            data.append({"asset": a, "horizon": "1m", "model": m, "qlike": r["qlike"],
                         "qlike_ratio_vs_har": r["qlike_ratio"], "qlike_ratio_vs_ivcal": ratio_ivcal,
                         "dm_vs_ivcal": _val(d, "dm_hln"), "p_vs_ivcal": _val(d, "pvalue"),
                         "p2n_vs_ivcal": _val(d, "pvalue_2n"), "mcs_p": _val(mc, "pvalue"),
                         "in_mcs90": _val(mc, "in_90"), "n": r["n"], "note": None})
    body = md_table(header, rows, ["l", "l"] + ["r"] * (len(header) - 2)) if rows else NO_RESULTS
    return Table(name, title, caption, body, pd.DataFrame(data))


def mz_table(mz: pd.DataFrame, *, name: str, title: str, caption: str) -> Table:
    if mz.empty:
        return Table(name, title, caption, NO_RESULTS, pd.DataFrame())
    header = ["asset", "model", "T", "log b", "s.e.", "p (b = 1)", "T non-overl.", "log b non-overl.",
              "p non-overl.", "levels a †", "levels b †", "levels p (a = 0, b = 1) †"]
    rows, prev = [], None
    d = _sort(mz)
    for r in d.itertuples(index=False):
        rr = r._asdict()
        rows.append([rr["asset"] if rr["asset"] != prev else "", esc(rr["model"]), _n(rr.get("T")),
                     fmt_num(rr.get("b_log")), fmt_num(rr.get("se_b_log")), fmt_p(rr.get("p_log")),
                     _n(rr.get("T_nonoverlap")), fmt_num(rr.get("b_log_no")), fmt_p(rr.get("p_log_no")),
                     fmt_num(rr.get("a"), 1), fmt_num(rr.get("b")), fmt_p(rr.get("p_wald_levels"))])
        prev = rr["asset"]
    body = md_table(header, rows, ["l", "l"] + ["r"] * (len(header) - 2))
    return Table(name, title, caption, body, d)


def encompassing_table(enc: pd.DataFrame, *, name: str, title: str, caption: str) -> Table:
    if enc.empty:
        return Table(name, title, caption, NO_RESULTS, pd.DataFrame())
    header = ["asset", "model", "T", "b (log IV)", "c (log F)", "s.e. (c)", "p (c = 0)", "c with IV t−1",
              "p with IV t−1"]
    rows, prev = [], None
    d = _sort(enc)
    for r in d.itertuples(index=False):
        rr = r._asdict()
        label = bold(esc(rr["model"])) if rr["model"] in ("COMBO", "HAR") else esc(rr["model"])
        rows.append([rr["asset"] if rr["asset"] != prev else "", label, _n(rr.get("T")), fmt_num(rr.get("b")),
                     fmt_num(rr.get("c"), signed=True), fmt_num(rr.get("se_c")), fmt_p(rr.get("p_c")),
                     fmt_num(rr.get("c_ivlag"), signed=True), fmt_p(rr.get("p_c_ivlag"))])
        prev = rr["asset"]
    body = md_table(header, rows, ["l", "l"] + ["r"] * (len(header) - 2))
    return Table(name, title, caption, body, d)


def _green(res: Results) -> pd.DataFrame:
    """Green-zone share per (asset, model) from ``risk_time_in_zone``: dev = share of rolling-250 windows in the
    green zone; holdout = 1 if the zone over the holdout observations (thresholds at the actual N) is green, else 0
    (see DEVIATIONS); falls back to the 99% backtest green share."""
    tiz = res["risk_time_in_zone"]
    if not tiz.empty and "green" in tiz:
        return tiz[["asset", "model", "green"]]
    bt = res["risk_backtests"]
    if not bt.empty and "green_share" in bt:
        b = bt[bt["level"].astype(str) == "99"]
        return b[["asset", "model", "green_share"]].rename(columns={"green_share": "green"})
    return pd.DataFrame(columns=["asset", "model", "green"])


RISK_LEVELS = (("99", "99%"), ("97.5", "97.5%"))
LATEST_COLS = ["asset", "model", "zone_date", "zone_latest", "exceptions_latest", "plus_factor_latest"]


def _latest_zones(res: Results) -> pd.DataFrame:
    """Per (asset, model): Basel zone, exceptions and plus factor of the last rolling 250-observation window at
    99% that ends inside the split's backtest window (``risk_rolling_zones`` holds the whole series; the holdout
    file includes the development dates). Same window as the dashboard's 'latest zone' and the figures' strip.
    Without rolling zones, the backtests' ``zone_last`` (99%) is used and the plus factor is unknown."""
    z, bt = res["risk_rolling_zones"], res["risk_backtests"]
    bt99 = bt[bt["level"].astype(str) == "99"] if not bt.empty and "level" in bt else pd.DataFrame()
    if not z.empty and {"asset", "model", "date", "zone"} <= set(z.columns):
        z = z.sort_values("date")
        if not bt99.empty and {"start", "end"} <= set(bt99.columns):
            w = bt99[["asset", "model", "start", "end"]].drop_duplicates(["asset", "model"])
            z = z.merge(w, on=["asset", "model"], how="inner")
            z = z[(z["date"] >= z["start"]) & (z["date"] <= z["end"])]
        last = z.groupby(["asset", "model"], sort=False).tail(1)
        out = pd.DataFrame({
            "asset": last["asset"].astype(str), "model": last["model"].astype(str), "zone_date": last["date"],
            "zone_latest": last["zone"],
            "exceptions_latest": last["exceptions"] if "exceptions" in last else np.nan,
            "plus_factor_latest": last["plus_factor"] if "plus_factor" in last else np.nan,
        })
        return out.reset_index(drop=True)
    if not bt99.empty and "zone_last" in bt99:
        b = bt99.dropna(subset=["zone_last"])
        return pd.DataFrame({"asset": b["asset"].astype(str), "model": b["model"].astype(str),
                             "zone_date": b["end"] if "end" in b else pd.NaT, "zone_latest": b["zone_last"],
                             "exceptions_latest": np.nan, "plus_factor_latest": np.nan}).reset_index(drop=True)
    return pd.DataFrame(columns=LATEST_COLS)


def _risk_models(a: str, *frames: pd.DataFrame) -> list[str]:
    names: list[str] = []
    for src in frames:
        if not src.empty and {"asset", "model"} <= set(src.columns):
            names += [str(m) for m in src.loc[src["asset"] == a, "model"]]
    return P.order_risk_models(names)


def _risk_window(bt: pd.DataFrame, a: str) -> str:
    """'(start → end, T = n)' of an asset's backtest window (99% rows)."""
    r0 = _lookup(bt, asset=a, level="99")
    if r0 is None:
        return ""
    return (f" ({pd.Timestamp(r0['start']):%Y-%m-%d} → {pd.Timestamp(r0['end']):%Y-%m-%d}, "
            f"T = {_n(r0['T'])})")


def _plus(x) -> str:
    """Basel plus factor: '0.00' in the green zone, '+0.40' … '+1.00' above."""
    return fmt_num(x, 2, signed=H._finite(x) and float(x) > 0)


def _zone(z) -> str:
    """Zone name; '—' when missing (None / NaN, e.g. fewer than 250 observations in the window)."""
    if z is None or (isinstance(z, float) and not math.isfinite(z)) or str(z) in ("", "nan", "None"):
        return "—"
    return str(z)


def var_backtest_table(res: Results, assets: Sequence[str], *, name: str, title: str, caption: str) -> Table:
    """VaR backtests per asset: one row per level × risk model with coverage tests, the UC power, the Basel zone
    over the whole window, and the latest rolling-250 zone (+ Basel plus factor at 99%)."""
    bt = res["risk_backtests"]
    if bt.empty:
        return Table(name, title, caption, NO_RESULTS, pd.DataFrame())
    latest = _latest_zones(res)
    header = ["level", "model", "breaches (rate)", "binom p", "CC p", "DQ p", "UC power",
              "zone, whole window", "latest zone (last 250 obs.)", "plus factor (latest)"]
    parts, data = [], []
    for a in assets:
        models = _risk_models(a, bt)
        if not models:
            parts.append(f"**{a}**: _no risk results._")
            continue
        rows = []
        for lvl, label in RISK_LEVELS:
            first = True
            for m in models:
                b = _lookup(bt, asset=a, model=m, level=lvl)
                if b is None:
                    continue
                z = _lookup(latest, asset=a, model=m) if lvl == "99" else None
                zone_latest = _val(z, "zone_latest") if lvl == "99" else _val(b, "zone_last")
                pf = _val(z, "plus_factor_latest")
                rows.append([label if first else "", esc(m), f"{int(b['x'])} ({100 * float(b['rate']):.1f}%)",
                             fmt_p(b["p_binom"]), fmt_p(b["p_cc_mc"]), fmt_p(b["p_dq"]),
                             fmt_share(_val(b, "power_uc")), _zone(b["zone_full"]),
                             _zone(zone_latest),
                             _plus(pf) if lvl == "99" else "—"])
                first = False
                data.append({"asset": a, "model": m, "level": lvl,
                             **{k: _val(b, k) for k in ("start", "end", "T", "x", "rate", "p_binom", "p_cc_mc",
                                                        "p_dq", "power_uc", "zone_full")},
                             "zone_latest": zone_latest,
                             "exceptions_latest": _val(z, "exceptions_latest") if lvl == "99" else None,
                             "plus_factor_latest": pf if lvl == "99" else None,
                             "zone_date": _val(z, "zone_date") if lvl == "99" else _val(b, "end")})
        parts.append(f"**{a}**{_risk_window(bt, a)}\n\n"
                     + md_table(header, rows, ["l", "l", "r", "r", "r", "r", "r", "l", "l", "r"]))
    return Table(name, title, caption, "\n\n".join(parts), pd.DataFrame(data))


def risk_leaderboard_table(res: Results, assets: Sequence[str], *, name: str, title: str, caption: str) -> Table:
    """Risk leaderboard per asset (SPEC §9): mean FZ0 difference vs HS-250 with its DM p, 90% MCS membership,
    the Acerbi–Székely ES test, the rolling-250 green share and the latest zone (99%)."""
    bt, es, fz = res["risk_backtests"], res["risk_es"], res["risk_fz0"]
    if fz.empty and not res["risk_risk_leaderboard"].empty:
        fz = res["risk_risk_leaderboard"]
    if fz.empty and es.empty:
        return Table(name, title, caption, NO_RESULTS, pd.DataFrame())
    green, latest = _green(res), _latest_zones(res)
    header = ["model", "FZ0 Δ vs HS-250", "DM p", "90% MCS", "ES Z2", "Z2 p", "green share (rolling 250)",
              "latest zone (99%, last 250 obs.)"]
    parts, data = [], []
    for a in assets:
        models = _risk_models(a, fz, es)
        if not models:
            parts.append(f"**{a}**: _no risk results._")
            continue
        rows = []
        for m in models:
            e = _lookup(es, asset=a, model=m)
            f = _lookup(fz, asset=a, model=m)
            g = _lookup(green, asset=a, model=m)
            z = _lookup(latest, asset=a, model=m)
            in90 = _val(f, "in_90")
            rows.append([esc(m), "0 (ref.)" if m == "HS-250" else fmt_num(_val(f, "fz0_diff"), signed=True),
                         fmt_p(_val(f, "p_dm")), "—" if in90 is None else ("yes" if bool(in90) else "no"),
                         fmt_num(_val(e, "z2"), signed=True), fmt_p(_val(e, "p_z2")), fmt_share(_val(g, "green")),
                         _zone(_val(z, "zone_latest"))])
            data.append({"asset": a, "model": m, "fz0": _val(f, "fz0"), "fz0_diff": _val(f, "fz0_diff"),
                         "dm_hln": _val(f, "dm_hln"), "p_dm": _val(f, "p_dm"), "mcs_p": _val(f, "mcs_p"),
                         "in_mcs90": in90, "z2": _val(e, "z2"), "p_z2": _val(e, "p_z2"),
                         "green_share": _val(g, "green"), "zone_latest": _val(z, "zone_latest"),
                         "plus_factor_latest": _val(z, "plus_factor_latest"), "zone_date": _val(z, "zone_date")})
        parts.append(f"**{a}**{_risk_window(bt, a)}\n\n"
                     + md_table(header, rows, ["l", "r", "r", "l", "r", "r", "r", "l"]))
    return Table(name, title, caption, "\n\n".join(parts), pd.DataFrame(data))


def _asset_cell(r: H.HypothesisResult, asset: str) -> str:
    rows = r.rows[r.rows["asset"] == asset] if not r.rows.empty else r.rows
    if rows.empty:
        return "—"
    if r.hypothesis == "H2":
        req = rows[rows["required"]]
        v = H.overall_verdict(req["verdict"])
        fails = list(req.loc[req["verdict"] == H.NOT_SUPPORTED, "horizon"])
        if v == H.NOT_SUPPORTED:
            return f"{_verdict(v)} ({', '.join(fails)})"
        ok = int((req["verdict"] == H.SUPPORTED).sum())
        return f"{_verdict(v)} ({ok}/{len(req)} cells)" if v == H.SUPPORTED else _verdict(v)
    x = rows.iloc[0]
    if not bool(x["required"]):
        if r.hypothesis == "H5":
            return "reference" if asset == H.H5_REFERENCE else "—"
        return f"({x['verdict']}; outside rule)" if x["verdict"] != H.NA else "n/a"
    return _verdict(x["verdict"])


def hypotheses_table(res: dict[str, H.HypothesisResult], assets: Sequence[str], mode: str, *, name: str,
                     title: str, caption: str) -> Table:
    header = ["", "rule (5% level)", *assets, "overall"]
    rows = [[f"**{k}**", r.rule, *[_asset_cell(r, a) for a in assets], bold(_verdict(r.overall))]
            for k, r in res.items()]
    parts = [md_table(header, rows, ["l", "l"] + ["c"] * (len(assets) + 1))]
    notes = [f"- **{k}**: {r.note}" for k, r in res.items() if r.note]
    if notes:
        parts.append("Notes:\n\n" + "\n".join(notes))
    for k, r in res.items():
        det = r.rows
        if det.empty:
            continue
        drows = [[esc(x["unit"]), _verdict(x["verdict"]) + ("" if x["required"] else " (not in rule)"),
                  x["evidence"], x["note"] or ""] for _, x in det.iterrows()]
        parts.append(f"#### {k}: {r.statement}\n\n" + md_table(["unit", "verdict", "evidence", "note"], drows,
                                                                ["l", "l", "l", "l"]))
    return Table(name, title, caption, "\n\n".join(parts), H.combine(res, mode))


# ============================================================================================= figures
def _fig(w: float, h: float) -> Figure:
    return Figure(figsize=(w, h), dpi=DPI, layout="constrained")


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(fig)
    fig.savefig(path, dpi=DPI, facecolor=P.SURFACE)
    return path


def _title(fig: Figure, title: str, subtitle: str | None = None) -> None:
    """Left-aligned title on top and a wrapped explanatory note at the bottom (both inside the layout)."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold", color=P.INK)
    if subtitle:
        width = max(40, int(fig.get_figwidth() * 14.5))
        fig.supxlabel(textwrap.fill(subtitle, width), x=0.01, ha="left", fontsize=8.5, color=P.INK2,
                      linespacing=1.3)


def _date_axis(ax) -> None:
    loc = mdates.AutoDateLocator(minticks=4, maxticks=9)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def _shade(ax, span: tuple[pd.Timestamp, pd.Timestamp] | None, label: bool = True) -> None:
    if span is None:
        return
    ax.axvspan(span[0], span[1], color=P.HOLDOUT_SHADE, zorder=0, lw=0)
    if label:
        ax.text(span[0], 1.0, " holdout", transform=ax.get_xaxis_transform(), ha="left", va="top", fontsize=8,
                color=P.INK2)


def _luminance(rgb) -> float:
    r, g, b = (c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _end_labels(ax, ends: list[tuple[pd.Timestamp, float, str]], min_frac: float = 0.075) -> None:
    """Direct labels at the right end of lines, spread vertically with a thin leader line when they collide."""
    if not ends:
        return
    lo, hi = ax.get_ylim()
    x0, x1 = ax.get_xlim()
    logy = ax.get_yscale() == "log"
    fwd = np.log10 if logy else (lambda v: v)
    span = fwd(hi) - fwd(lo)
    items = sorted((float(fwd(y)), mdates.date2num(x), y, t) for x, y, t in ends
                   if np.isfinite(y) and (y > 0 or not logy))
    pos: list[float] = []
    for ty, *_ in items:
        pos.append(max(ty, pos[-1] + min_frac * span) if pos else ty)
    top = fwd(hi) - 0.03 * span
    if pos and pos[-1] > top:
        pos = [p - (pos[-1] - top) for p in pos]
    for p, (ty, xn, y, t) in zip(pos, items, strict=True):
        moved = abs(p - ty) > 1e-9 * max(1.0, abs(span)) or xn < x1 - 0.002 * (x1 - x0)
        ax.annotate(t, xy=(xn, y), xytext=(1.03, 10 ** p if logy else p), textcoords=("axes fraction", "data"),
                    ha="left", va="center", fontsize=8, color=P.INK, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=P.MUTED, lw=0.6, shrinkA=1, shrinkB=1) if moved else None)


def fig_leaderboard_heatmap(lb: pd.DataFrame, mcs: pd.DataFrame, path: Path, title: str, note: str = "") -> Path | None:
    if lb.empty:
        return None
    cells = _cells(lb)
    models = P.order_models(lb["model"])
    M = np.full((len(models), len(cells)), np.nan)
    mark = np.zeros_like(M, dtype=bool)
    in90 = set()
    if not mcs.empty and "in_90" in mcs:
        in90 = {(str(r.asset), str(r.horizon), str(r.model)) for r in mcs.itertuples() if bool(r.in_90)}
    for i, m in enumerate(models):
        for j, (a, h) in enumerate(cells):
            r = _lookup(lb, asset=a, horizon=h, model=m)
            if r is not None and np.isfinite(r["qlike_ratio"]) and r["qlike_ratio"] > 0:
                M[i, j] = r["qlike_ratio"]
                mark[i, j] = (a, h, m) in in90
    with rc_context(RC):
        fig = _fig(max(6.8, 1.6 + 0.66 * len(cells)), 1.9 + 0.34 * len(models))
        ax = fig.add_subplot()
        L = np.log(M)
        vmax = float(np.clip(np.nanmax(np.abs(L)) if np.isfinite(L).any() else np.log(1.2), np.log(1.1),
                             np.log(1.6)))
        cmap = LinearSegmentedColormap.from_list("ratio", P.DIVERGING).with_extremes(bad=P.SURFACE)
        im = ax.imshow(np.ma.masked_invalid(L), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto",
                       interpolation="nearest")
        for i in range(len(models)):
            for j in range(len(cells)):
                if not np.isfinite(M[i, j]):
                    ax.text(j, i, "—", ha="center", va="center", fontsize=8, color=P.MUTED)
                    continue
                rgb = cmap((L[i, j] + vmax) / (2 * vmax))
                ink = "#FFFFFF" if _luminance(rgb) < 0.32 else P.INK
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8, color=ink,
                        fontweight="bold" if mark[i, j] else "normal")
                if mark[i, j]:
                    ax.plot(j + 0.38, i - 0.30, marker="o", ms=3.2, color=ink, mec="none")
        assets = [a for a, _ in cells]
        bounds = [j for j in range(1, len(cells)) if assets[j] != assets[j - 1]]
        for b in bounds:
            ax.axvline(b - 0.5, color=P.SURFACE, lw=4)
        ax.set_xticks(range(len(cells)), [h for _, h in cells])
        ax.set_yticks(range(len(models)), models)
        ax.tick_params(length=0)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        starts = [0, *bounds]
        ends = [*bounds, len(cells)]
        for s0, e0 in zip(starts, ends, strict=True):
            ax.annotate(assets[s0], xy=((s0 + e0 - 1) / 2, -0.62), xycoords="data", ha="center", va="bottom",
                        fontsize=9.5, fontweight="bold", color=P.INK, annotation_clip=False)
        ax.set_ylim(len(models) - 0.5, -0.5)
        ticks = [t for t in (0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5) if abs(np.log(t)) <= vmax + 1e-12]
        cb = fig.colorbar(im, ax=ax, ticks=np.log(ticks), shrink=0.85, pad=0.015, aspect=30)
        cb.ax.set_yticklabels([f"{t:g}" for t in ticks])
        cb.set_label("QLIKE ratio vs HAR (log scale)", color=P.INK2)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0)
        _title(fig, title, note or "Mean QLIKE of the model ÷ mean QLIKE of HAR on common dates. "
               "Below 1 (blue) = lower loss than HAR. Bold value with dot = in the 90% model confidence set.")
        return _save(fig, path)


def _vol(F: pd.Series, n_t: pd.Series, ann: int) -> pd.Series:
    return np.sqrt(F.astype(float) / n_t.astype(float) * ann)


def fig_forecast_vs_realized(asset: str, fc: pd.DataFrame, tg: pd.DataFrame, path: Path,
                             span: tuple[pd.Timestamp, pd.Timestamp] | None) -> Path | None:
    f = fc[(fc["asset"] == asset) & (fc["horizon"] == "1d") & fc["model"].isin(FIG_MODELS_1D)
           & fc["split"].isin(["dev", "holdout"])]
    t = tg[(tg["asset"] == asset) & (tg["horizon"] == "1d") & tg["split"].isin(["dev", "holdout"])]
    if f.empty or t.empty:
        return None
    t = t[["origin", "window_end", "y", "n_t"]]
    lo = pd.Timestamp(C.dev_end()) - pd.Timedelta(days=730)
    t = t[t["window_end"] >= lo].sort_values("window_end")
    if t.empty:
        return None
    ann = _ann(asset)
    w = f.pivot_table(index="origin", columns="model", values="F")
    d = t.set_index("origin").join(w, how="left").sort_values("window_end")
    with rc_context(RC):
        fig = _fig(9, 3.8)
        ax = fig.add_subplot()
        _shade(ax, span)
        rv = _vol(d["y"], d["n_t"], ann)
        ax.set_yscale("log")
        floor = max(float(np.nanmin(rv[rv > 0])) * 0.8, 1e-3) if (rv > 0).any() else 1.0
        ax.scatter(d["window_end"], rv.where(rv > 0), s=7, color=REALIZED_DOT, lw=0, zorder=1,
                   label="realized (next session)")
        ends = []
        for m in FIG_MODELS_1D:
            if m not in d:
                continue
            v = _vol(d[m], d["n_t"], ann)
            ax.plot(d["window_end"], v, color=P.MODEL_COLORS[m], lw=1.6 if m == "COMBO" else 1.2, label=m,
                    zorder=3 if m == "COMBO" else 2)
            ok = v.notna()
            if ok.any():
                ends.append((d.loc[ok, "window_end"].iloc[-1], float(v[ok].iloc[-1]), m))
        ax.set_ylim(floor, float(np.nanmax(rv)) * 1.15 if np.isfinite(np.nanmax(rv)) else None)
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_ylabel("annualised volatility, % (log scale)")
        _date_axis(ax)
        ax.set_xlim(d["window_end"].min(), d["window_end"].max())
        _end_labels(ax, ends)
        ax.legend(loc="lower left", ncols=4, bbox_to_anchor=(0, 1.0), handlelength=1.6, markerscale=1.6)
        _title(fig, f"{asset}: 1-day volatility forecasts vs realized",
               f"√(F × {ann}) for the forecasts and √(y × {ann}) for the realized next-session variance "
               "(gap² + RV), plotted at the target session; last two development years"
               + (" and the holdout (shaded)." if span is not None else "."))
        return _save(fig, path)


def _grid(n: int) -> tuple[int, int]:
    return (1, n) if n <= 2 else (math.ceil(n / 2), 2)


def fig_cum_qlike(losses: pd.DataFrame, path: Path, span: tuple[pd.Timestamp, pd.Timestamp] | None) -> Path | None:
    l1 = losses[(losses["horizon"] == "1d") & losses["model"].isin(["HAR", *CUM_MODELS])] if not losses.empty \
        else losses
    if l1.empty:
        return None
    assets = P.order_assets(l1["asset"])
    nr, nc = _grid(len(assets))
    with rc_context(RC):
        fig = _fig(9.5 if nc > 1 else 8.0, 3.0 * nr + 0.9)
        axes = fig.subplots(nr, nc, squeeze=False)
        for k, a in enumerate(assets):
            ax = axes[k // nc][k % nc]
            w = l1[l1["asset"] == a].pivot_table(index="origin", columns="model", values="qlike").sort_index()
            if "HAR" not in w:
                ax.set_visible(False)
                continue
            models = [m for m in CUM_MODELS if m in w]
            w = w[["HAR", *models]].dropna()
            _shade(ax, span, label=k == 0)
            ax.axhline(0, color=P.AXIS, lw=1.0, zorder=1)
            ends = []
            for m in models:
                cum = (w[m] - w["HAR"]).cumsum()
                ax.plot(cum.index, cum.to_numpy(), color=P.MODEL_COLORS[m], lw=2.0 if m == "COMBO" else 1.5,
                        ls=CUM_STYLES[m], label=m, zorder=3 if m == "COMBO" else 2)
                ends.append((cum.index[-1], float(cum.iloc[-1]), m))
            ax.set_title(a, loc="left")
            _date_axis(ax)
            ax.set_xlim(w.index.min(), w.index.max())
            ax.margins(y=0.08)
            _end_labels(ax, ends)
            if k % nc == 0:
                ax.set_ylabel("Σ (QLIKE model − QLIKE HAR)")
        for k in range(len(assets), nr * nc):
            axes[k // nc][k % nc].set_visible(False)
        handles = [Line2D([], [], color=P.MODEL_COLORS[m], lw=2.0 if m == "COMBO" else 1.5, ls=CUM_STYLES[m])
                   for m in CUM_MODELS]
        fig.legend(handles, CUM_MODELS, loc="outside upper right", ncols=4)
        _title(fig, "Cumulative QLIKE difference vs HAR at 1 day",
               "Falling line = the model is accumulating lower loss than HAR (below 0 = better than HAR so far). "
               "Common origins of HAR, GJR, LGBM, MLP and COMBO" + ("; holdout shaded." if span else "."))
        return _save(fig, path)


def _zone_runs(dates: pd.Series, zones: pd.Series):
    """Consecutive runs of one zone as (start, end, zone)."""
    if dates.empty:
        return []
    d = dates.reset_index(drop=True)
    z = zones.reset_index(drop=True)
    brk = (z != z.shift()).cumsum()
    out = []
    for _, idx in z.groupby(brk).groups.items():
        i0, i1 = idx[0], idx[-1]
        end = d[i1 + 1] if i1 + 1 < len(d) else d[i1] + pd.Timedelta(days=1)
        out.append((d[i0], end, z[i0]))
    return out


def fig_var_breaches(asset: str, risk: pd.DataFrame, zones: pd.DataFrame, path: Path,
                     span: tuple[pd.Timestamp, pd.Timestamp] | None) -> Path | None:
    r = risk[(risk["asset"] == asset) & risk["model"].isin(["COMBO+FHS", "HS-250"])] if not risk.empty else risk
    if r.empty:
        return None
    w = r.pivot_table(index="date", columns="model", values="var99").sort_index()
    ret = r.drop_duplicates("date").set_index("date")["r_cc"].sort_index()
    with rc_context(RC):
        fig = _fig(9.5, 4.8)
        ax, axz = fig.subplots(2, 1, sharex=True, height_ratios=[5, 1.15])
        _shade(ax, span)
        _shade(axz, span, label=False)
        ax.vlines(ret.index, 0, ret.to_numpy(), color="#A9A69F", lw=0.6, zorder=1, label="daily return r_cc")
        legend = [Line2D([], [], color="#A9A69F", lw=1.2, label="daily close-to-close return")]
        for m in ("HS-250", "COMBO+FHS"):
            if m not in w:
                continue
            var = w[m]
            ax.plot(var.index, -var.to_numpy(), color=P.RISK_COLORS[m], lw=1.2,
                    drawstyle="steps-mid", zorder=3 if m == "COMBO+FHS" else 2)
            br = ret.reindex(var.index)
            hit = br < -var
            mk = "o" if m == "COMBO+FHS" else "D"
            ax.scatter(br.index[hit], br[hit], s=34 if m == "COMBO+FHS" else 28, marker=mk,
                       color=P.RISK_COLORS[m], edgecolors=P.SURFACE, linewidths=1.5, zorder=5)
            legend.append(Line2D([], [], color=P.RISK_COLORS[m], lw=1.6, label=f"−VaR99 {m}"))
            legend.append(Line2D([], [], color=P.RISK_COLORS[m], marker=mk, ls="none", ms=6, mec=P.SURFACE,
                                 label=f"{m} breach ({int(hit.sum())})"))
        ax.axhline(0, color=P.AXIS, lw=0.8)
        ax.set_ylabel("return, %")
        ax.legend(handles=legend, loc="lower left", bbox_to_anchor=(0, 1.0), ncols=3, fontsize=8)
        # traffic-light strip
        z = zones[(zones["asset"] == asset)] if not zones.empty else zones
        ylabels = []
        for k, m in enumerate(("HS-250", "COMBO+FHS")):
            zm = z[z["model"] == m].sort_values("date") if not z.empty else z
            ylabels.append(m)
            for s0, e0, zone in _zone_runs(zm["date"], zm["zone"]) if not zm.empty else []:
                axz.broken_barh([(mdates.date2num(s0), mdates.date2num(e0) - mdates.date2num(s0))], (k - 0.4, 0.8),
                                color=P.ZONE_COLORS.get(zone, P.MUTED), lw=0)
        axz.set_yticks([0, 1], ylabels)
        axz.set_ylim(-0.6, 1.6)
        axz.grid(False)
        axz.tick_params(axis="y", length=0)
        axz.set_title("Basel zone, rolling 250 observations at 99% (blank = fewer than 250 observations)",
                      loc="left", fontsize=8.5, fontweight="normal", color=P.INK2)
        axz.legend(handles=[Patch(color=P.ZONE_COLORS[k], label=k) for k in P.ZONE_ORDER], loc="upper left",
                   bbox_to_anchor=(1.0, 1.15), fontsize=8, handlelength=1.0)
        _date_axis(axz)
        ax.set_xlim(ret.index.min(), ret.index.max())
        _title(fig, f"{asset}: 99% Value-at-Risk breaches, COMBO+FHS vs HS-250",
               "A breach is a day whose loss exceeds the VaR forecast made the day before (expected 1% of days). "
               + ("Holdout shaded." if span else ""))
        return _save(fig, path)


def fig_iv_vs_models(fc: pd.DataFrame, tg: pd.DataFrame, path: Path,
                     span: tuple[pd.Timestamp, pd.Timestamp] | None) -> Path | None:
    f = fc[(fc["horizon"] == "1m") & fc["split"].isin(["dev", "holdout"])] if not fc.empty else fc
    iv_assets = P.order_assets(f.loc[f["model"].isin(IV_MODELS), "asset"]) if not f.empty else []
    if not iv_assets:
        return None
    with rc_context(RC):
        fig = _fig(9.5, 2.7 * len(iv_assets) + 1.0)
        axes = fig.subplots(len(iv_assets), 1, squeeze=False)[:, 0]
        for ax, a in zip(axes, iv_assets, strict=True):
            fa = f[(f["asset"] == a) & f["model"].isin(["IV", "IV-cal", "COMBO"])]
            w = fa.pivot_table(index="origin", columns="model", values="F").sort_index()
            t = tg[(tg["asset"] == a) & (tg["horizon"] == "1m")].set_index("origin")[["y", "n_t"]]
            d = w.join(t, how="inner")
            d = d[d["IV"].notna()] if "IV" in d else d
            if d.empty:
                ax.set_visible(False)
                continue
            ann = _ann(a)
            _shade(ax, span, label=ax is axes[0])
            rv = _vol(d["y"], d["n_t"], ann)
            # wash below the gridlines (zorder 0.5) so values stay readable inside it
            ax.fill_between(d.index, 0, rv.to_numpy(), color=P.REALIZED_WASH, lw=0, zorder=0.4,
                            label="realized over the next 30 days")
            ends = []
            for m in ("IV", "IV-cal", "COMBO"):
                if m not in d:
                    continue
                v = _vol(d[m], d["n_t"], ann)
                st = IV_LINE_STYLES[m]
                ring = None
                if m == "IV-cal":  # 2px surface ring keeps the light dashed line distinct on the wash and on IV
                    ring = [pe.Stroke(linewidth=st["lw"] + 2.0, foreground=P.SURFACE), pe.Normal()]
                ax.plot(d.index, v.to_numpy(), color=P.MODEL_COLORS[m], lw=st["lw"], ls=st["ls"], label=m,
                        zorder=st["zorder"], path_effects=ring)
                ok = v.notna()
                if ok.any():
                    ends.append((v.index[ok][-1], float(v[ok].iloc[-1]), m))
            ax.set_title(a, loc="left")
            ax.set_ylabel("annualised vol, %")
            ax.set_ylim(0, None)
            _date_axis(ax)
            ax.set_xlim(d.index.min(), d.index.max())
            _end_labels(ax, ends)
        handles = [Patch(facecolor=P.REALIZED_WASH, edgecolor=P.AXIS, lw=0.6, label="realized, next 30 days")] + [
            Line2D([], [], color=P.MODEL_COLORS[m], lw=IV_LINE_STYLES[m]["lw"], ls=IV_LINE_STYLES[m]["ls"], label=m)
            for m in ("IV", "IV-cal", "COMBO")]
        fig.legend(handles=handles, loc="outside upper right", ncols=4)
        _title(fig, "1-month volatility: implied vs model forecasts",
               "Plotted at the forecast origin: √(F / n × ann) for IV, calibrated IV (dashed; the reference of the "
               "DM test vs implied vol) and COMBO; shaded area = realized volatility over the following window "
               "(√(y / n × ann)). Each asset's IV sample: DVOL from 2021-03, VIX over the full OOS window, EVZ to 2023-12-31.")
        return _save(fig, path)


def fig_h5(r5: H.HypothesisResult, path: Path) -> Path | None:
    rows = r5.rows
    if rows.empty:
        return None
    rows = rows[np.isfinite(rows[["j_rv", "jump_share", "rsneg_rv"]].astype(float)).any(axis=1)]
    if rows.empty:
        return None
    assets = list(rows["asset"])
    panels = [("j_rv", "J / RV (mean daily share)"), ("jump_share", "Jump-day share"),
              ("rsneg_rv", "RS⁻ / RV (mean daily share)")]
    with rc_context(RC):
        fig = _fig(10.5, 3.4)
        axes = fig.subplots(1, 4)
        x = np.arange(len(assets))
        for ax, (col, ttl) in zip(axes[:3], panels, strict=True):
            v = rows[col].astype(float).to_numpy()
            ax.bar(x, v, width=0.45, color=NEUTRAL_BAR)
            for xi, vi in zip(x, v, strict=True):
                if np.isfinite(vi):
                    ax.annotate(f"{100 * vi:.1f}%", (xi, vi), xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=8, color=P.INK)
            ax.set_xticks(x, assets)
            ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=None))
            ax.set_title(ttl, loc="left", fontsize=9)
            ax.margins(y=0.15)
            ax.grid(axis="x", visible=False)
        ax = axes[3]
        bw = 0.3
        for k, m in enumerate(("HAR-CJ", "SHAR")):
            v = rows[f"gain_{m}"].astype(float).to_numpy()
            xs = x + (k - 0.5) * bw
            ax.bar(xs, v, width=bw - 0.03, color=P.MODEL_COLORS[m], label=m)
            for xi, vi in zip(xs, v, strict=True):
                if np.isfinite(vi):
                    ax.annotate(f"{100 * vi:+.1f}%".replace("-", "−"), (xi, vi),
                                xytext=(0, 2 if vi >= 0 else -2), textcoords="offset points", ha="center",
                                va="bottom" if vi >= 0 else "top", fontsize=7.5, color=P.INK)
        ax.axhline(0, color=P.AXIS, lw=0.9)
        ax.set_xticks(x, assets)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=None))
        ax.set_title("1d QLIKE gain vs HAR", loc="left", fontsize=9)
        ax.margins(y=0.2)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="best", fontsize=8)
        _title(fig, "H5: jumps and downside semivariance by asset",
               "Gold daily measures over the evaluation window; H5 expects every bar for BTC and ETH to exceed "
               "EURUSD's. Gain = 1 − QLIKE ratio vs HAR at 1d: positive = the jump / semivariance HAR beats "
               "plain HAR.")
        return _save(fig, path)


# ============================================================================================= build
def _captions() -> dict[str, str]:
    return {
        "leaderboard": (
            "Mean QLIKE of each model divided by the mean QLIKE of HAR, per asset and horizon, on the dates where "
            "every model has a forecast. **Ratio < 1 = lower loss than HAR (better)**; 1.000 is HAR itself. "
            "**Bold** = the model is in the 90% model confidence set (MCS) of that cell. The last row is the number "
            "of forecast origins (n). QLIKE levels are never compared across cells."),
        "leaderboard_holdout": (
            "Holdout (2025-10-01 → 2026-09-30, low-power confirmation): QLIKE ratio vs HAR with its 90% "
            "stationary-bootstrap percentile interval in brackets. **Bold** = in the 90% MCS; the holdout MCS runs at "
            "1d and 1w only, so the 1m column is descriptive."),
        "dm": (
            "Diebold–Mariano test with the Harvey–Leybourne–Newbold correction on the QLIKE loss difference "
            "model − HAR. **Negative DM = the model has lower loss than HAR.** p is two-sided (Student t, T − 1 "
            "df); the long-run variance uses Bartlett lags ⌊4(T/100)^(2/9)⌋ at 1d and the truncated uniform kernel "
            "with n_max − 1 lags at 1w/1m (DEVIATIONS). *p (2n lags)* is the robustness check with 2·n_max "
            "Bartlett lags."),
        "iv": (
            "1-month horizon on each asset's implied-vol subsample (no forward fill). *QLIKE ratio vs HAR* compares "
            "everything with HAR on that subsample; *QLIKE ratio vs IV-cal* and the DM test compare each model with "
            "the calibrated implied-vol benchmark (**negative DM = better than IV-cal**). *90% MCS* = member of the "
            "IV-inclusive model confidence set (development only). IV = raw implied variance over 30 days; IV-cal = "
            "IV × a rolling bias factor b̂ estimated on completed windows only."),
        "mz": (
            "Mincer–Zarnowitz regressions at 1m. Inference is on the log regression log y = a + b log F "
            "(H0: b = 1; Newey–West with 2·n_max lags); the non-overlapping columns re-run it on every n_max-th "
            "origin (HAC 0) as a robustness check. † The levels regression y = a + b F and its Wald test of "
            "a = 0, b = 1 are **descriptive only**: on overlapping monthly targets they reject far too often "
            "(DEVIATIONS). b close to 1 = forecasts move one-for-one with the realized variance."),
        "encompassing": (
            "Encompassing at 1m: log y = a + b log IV + c log F_model (Newey–West, 2·n_max lags). "
            "**c > 0 with p < 0.05 = the model adds information beyond implied vol** (H3 uses COMBO; HAR is the "
            "secondary check; both in bold). The last two columns replace IV by the previous origin's IV "
            "(IV_{t−1}) as a robustness check."),
        "var": (
            "One-day 99% and 97.5% VaR of a long unit position over the backtest window in brackets (T days). "
            "*breaches* = days with loss > VaR (rate in brackets; expected 1% and 2.5%); *binom p* = exact two-sided "
            "binomial test of the rate; *CC p* = Christoffersen conditional coverage (Monte Carlo); *DQ p* = "
            "Engle–Manganelli dynamic quantile; *UC power* = simulated rejection rate of the unconditional-coverage "
            "test at 5% when the true breach rate is twice the nominal one, for this T (low power = a pass says "
            "little). *zone, whole window* = Basel traffic light of all T days (cut-offs scaled to N = T); "
            "*latest zone* = traffic light of the last 250 observations of the window (the primary rolling display; "
            "as in the dashboard) and, at 99%, the Basel plus factor it implies (5 exceptions +0.40 … ≥ 10 +1.00)."),
        "risk_lb": (
            "Risk leaderboard (SPEC §9). *FZ0 Δ* = mean FZ0 score (VaR97.5 and ES97.5 jointly) minus HS-250's "
            "(**negative = better than HS-250**; a difference, not a ratio — DEVIATIONS) with its DM-HLN p-value; "
            "*90% MCS* = member of the model confidence set on FZ0 across the six risk models; *ES Z2* = "
            "Acerbi–Székely test of ES97.5 (negative = ES understated; p from Monte Carlo); *green share* = share "
            "of rolling 250-observation windows ending in the window that are in the Basel green zone at 99% (the "
            "H4 measure); *latest zone* = zone of the last such window."),
        "hypotheses": (
            "Mechanical verdicts of the pre-registered decision rules (SPEC §0, 5% level). Per asset: ✓ the rule "
            "holds, ✗ it fails, n/a = no results yet for that asset. Overall: one failing required unit makes the "
            "hypothesis *not supported*; all required units must hold for *supported*; otherwise *n/a*. "
            "Evidence numbers per unit follow the summary."),
        "pre2021": (
            "Robustness: the same QLIKE ratio vs HAR on development origins **before** 2021-01-01 (pre-headline "
            "out-of-sample window; SPX and EUR/USD only — crypto forecasts start in late 2020). No MCS is computed here."),
    }


def build(
    include_holdout: bool | None = None,
    results_dir: Path | str | None = None,
    reports_dir: Path | str | None = None,
    *,
    daily: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Write the tables, figures and ``results.md``; return {relative path: absolute path} of every file.

    include_holdout: None = auto-detect ``<results_dir>/holdout/eval_leaderboard.parquet``.
    results_dir: development results (default ``data/results``); the holdout run's tables are read from its
    ``holdout/`` subdirectory. reports_dir: output root (default ``reports``).
    daily: optional gold daily rows (or precomputed H5 measures) for H5; default ``volrisk.io.load_daily()``.
    """
    rd = Path(results_dir) if results_dir is not None else C.RESULTS
    out = Path(reports_dir) if reports_dir is not None else C.REPORTS
    if include_holdout is None:
        include_holdout = holdout_available(rd)
    elif include_holdout and not holdout_available(rd):
        raise FileNotFoundError(f"no holdout results under {rd / 'holdout'}")
    tables_dir, figs_dir = out / "tables", out / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    dev = Results("dev", rd)
    hold = Results("holdout", rd / "holdout") if include_holdout else None
    cap = _captions()
    written: dict[str, Path] = {}
    tables: dict[str, Table] = {}

    def emit(t: Table) -> Table:
        md = tables_dir / f"{t.name}.md"
        md.write_text(t.markdown(), encoding="utf-8")
        csv = tables_dir / f"{t.name}.csv"
        t.data.to_csv(csv, index=False, encoding="utf-8")
        written[f"tables/{t.name}.md"] = md
        written[f"tables/{t.name}.csv"] = csv
        tables[t.name] = t
        return t

    modes = [dev] + ([hold] if hold is not None else [])
    assets = _assets(*[r[n] for r in modes for n in ("eval_leaderboard", "risk_risk_leaderboard", "risk_fz0")])
    hyp: dict[str, dict[str, H.HypothesisResult]] = {}
    for res in modes:
        sfx = res.mode
        lab = "development" if sfx == "dev" else "holdout"
        ci = res["eval_ratio_ci"] if sfx == "holdout" else None
        emit(leaderboard_table(res["eval_leaderboard"], res["eval_mcs"], ci, name=f"leaderboard_{sfx}",
                               title=f"QLIKE ratio vs HAR ({lab})",
                               caption=cap["leaderboard"] if sfx == "dev" else cap["leaderboard_holdout"]))
        emit(dm_table(res["eval_dm_har"], name=f"dm_har_{sfx}", title=f"Diebold–Mariano vs HAR ({lab})",
                      caption=cap["dm"]))
        emit(iv_table(res["eval_iv_leaderboard"], res["eval_iv_dm"], res["eval_iv_mcs"], assets,
                      name=f"iv_1m_{sfx}", title=f"1-month forecasts vs implied volatility ({lab})",
                      caption=cap["iv"]))
        emit(mz_table(res["eval_mz"], name=f"mz_{sfx}", title=f"Mincer–Zarnowitz at 1m ({lab})", caption=cap["mz"]))
        emit(encompassing_table(res["eval_encompassing"], name=f"encompassing_{sfx}",
                                title=f"Encompassing vs implied vol at 1m ({lab})", caption=cap["encompassing"]))
        emit(var_backtest_table(res, assets, name=f"risk_backtests_{sfx}", title=f"VaR backtests ({lab})",
                                caption=cap["var"]))
        emit(risk_leaderboard_table(res, assets, name=f"risk_leaderboard_{sfx}",
                                    title=f"Risk leaderboard: FZ0, ES test and traffic light ({lab})",
                                    caption=cap["risk_lb"]))
        hyp[sfx] = H.run_all(res.path, sfx, daily=daily)
        emit(hypotheses_table(hyp[sfx], list(C.ASSETS) if not assets else P.order_assets([*C.ASSETS, *assets]),
                              sfx, name=f"hypotheses_{sfx}", title=f"Hypotheses H1–H5 ({lab})",
                              caption=cap["hypotheses"]))
    emit(leaderboard_table(dev["eval_leaderboard_pre2021"], pd.DataFrame(), name="leaderboard_pre2021_dev",
                           title="Robustness: QLIKE ratio vs HAR before 2021 (development)", caption=cap["pre2021"]))

    # ------------------------------------------------------------------ figures
    span = (pd.Timestamp(C.holdout_start()), pd.Timestamp(C.data_end())) if hold is not None else None
    src = hold if hold is not None else dev

    def pick(name: str) -> pd.DataFrame:
        df = src[name]
        return df if not df.empty else dev[name]

    figs: dict[str, Path | None] = {}
    figs["leaderboard_heatmap"] = fig_leaderboard_heatmap(
        dev["eval_leaderboard"], dev["eval_mcs"], figs_dir / "leaderboard_heatmap.png",
        "QLIKE ratio vs HAR (development, headline window)")
    if hold is not None:
        figs["leaderboard_heatmap_holdout"] = fig_leaderboard_heatmap(
            hold["eval_leaderboard"], hold["eval_mcs"], figs_dir / "leaderboard_heatmap_holdout.png",
            "QLIKE ratio vs HAR (holdout)",
            "Holdout window. Below 1 (blue) = lower loss than HAR. Bold value with dot = in the 90% MCS "
            "(1d and 1w only; 1m is descriptive).")
    fc, tg = pick("forecasts"), pick("targets")
    for a in _assets(fc):
        figs[f"forecast_vs_realized_{a}"] = fig_forecast_vs_realized(
            a, fc, tg, figs_dir / f"forecast_vs_realized_{a}.png", span)
    losses = dev["eval_losses"]
    if hold is not None and not hold["eval_losses"].empty:
        losses = pd.concat([losses, hold["eval_losses"][hold["eval_losses"]["split"] == "holdout"]],
                           ignore_index=True)
    figs["cum_qlike_vs_har"] = fig_cum_qlike(losses, figs_dir / "cum_qlike_vs_har.png", span)
    risk, zones = pick("risk"), pick("risk_rolling_zones")
    for a in _assets(risk):
        figs[f"var_breaches_{a}"] = fig_var_breaches(a, risk, zones, figs_dir / f"var_breaches_{a}.png", span)
    figs["iv_vs_models_1m"] = fig_iv_vs_models(fc, tg, figs_dir / "iv_vs_models_1m.png", span)
    figs["h5_jumps"] = fig_h5(hyp["dev"]["H5"], figs_dir / "h5_jumps.png")
    for k, p in figs.items():
        if p is not None:
            written[f"figures/{k}.png"] = p

    results_md = out / "results.md"
    results_md.write_text(_results_md(tables, figs, hyp, assets, hold is not None, rd), encoding="utf-8")
    written["results.md"] = results_md
    log.info("report: %d files under %s (holdout %s)", len(written), out, "included" if hold else "not included")
    return written


# ============================================================================================= results.md
def _img(figs: dict[str, Path | None], key: str, alt: str) -> str:
    p = figs.get(key)
    return f"![{alt}](./figures/{p.name})" if p is not None else f"_Figure {key} not available in this run._"


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(C.ROOT).as_posix()
    except ValueError:
        return Path(p).name


def _section(t: Table | None, level: int = 3) -> str:
    if t is None:
        return ""
    return f"{'#' * level} {t.title}\n\n{t.caption}\n\n{t.body}\n"


def _results_md(tables: dict[str, Table], figs: dict[str, Path | None], hyp: dict, assets: list[str],
                has_holdout: bool, rd: Path) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    missing = [a for a in C.ASSETS if a not in assets]
    lb = tables.get("leaderboard_dev")
    lines = [
        "# Results",
        "",
        f"_Generated by `volrisk.report.build()` on {now} from `{_rel(rd)}/`"
        + (f" and `{_rel(rd)}/holdout/`" if has_holdout else "") + ". Do not edit by hand: rerun "
        "`uv run python -m volrisk report`. Every table is also in `tables/` as Markdown and CSV._",
        "",
        f"**Assets in this run:** {', '.join(assets) if assets else 'none'}."
        + (f" Not yet run: {', '.join(missing)} (their cells show n/a)." if missing else ""),
        "",
        f"**Windows.** Development headline evaluation: forecast origins from {C.dev_eval_start()} to the last "
        f"development origin (window ending by {C.dev_end()}), common dates of all models. Risk backtests: from "
        f"the later of {C.dev_eval_start()} and the day the FHS pool reaches 500 standardized returns, to "
        f"{C.dev_end()}. Holdout: sessions {C.holdout_start()} → {C.data_end()} — "
        + ("**opened; results below are published unedited.**" if has_holdout
           else "**sealed** (holdout tables appear here automatically after the one-time holdout run)."),
        "",
        "**How to read.** Variance forecasts are scored with QLIKE (lower is better) and reported relative to "
        "HAR: a ratio below 1 means lower loss than HAR. Bold = inside the 90% model confidence set (models "
        "statistically indistinguishable from the best). p-values are two-sided; '<0.001' means below 0.001. "
        "Volatility in figures is annualised for display only (×365 crypto, ×252 SPX, ×260 EURUSD).",
        "",
        "## 1. Hypotheses",
        "",
        _section(tables.get("hypotheses_dev")),
    ]
    if has_holdout:
        lines += [_section(tables.get("hypotheses_holdout"))]
    lines += [
        "## 2. Forecast accuracy",
        "",
        _img(figs, "leaderboard_heatmap", "QLIKE ratio vs HAR heatmap"),
        "",
        "Heatmap of the development leaderboard below: blue cells beat HAR, red cells lose to it; bold values "
        "with a dot are in the 90% MCS.",
        "",
        _section(lb),
    ]
    if has_holdout:
        lines += [_img(figs, "leaderboard_heatmap_holdout", "Holdout QLIKE ratio heatmap"), "",
                  _section(tables.get("leaderboard_holdout"))]
    lines += [_section(tables.get("dm_har_dev"))]
    if has_holdout:
        lines += [_section(tables.get("dm_har_holdout"))]
    lines += [
        "### Losses through time",
        "",
        "Cumulative QLIKE difference vs HAR at 1d: a line that keeps falling is a model that keeps beating HAR; "
        "a step up is a day on which it lost badly.",
        "",
        _img(figs, "cum_qlike_vs_har", "Cumulative QLIKE difference vs HAR"),
        "",
        "One-day forecasts of HAR, GJR and COMBO against the realized next-session volatility (grey) for the last "
        "two development years" + (" and the holdout." if has_holdout else "."),
        "",
    ]
    for k in sorted((k for k in figs if k.startswith("forecast_vs_realized_") and figs[k] is not None),
                    key=lambda k: P.asset_rank(k.rsplit("_", 1)[-1])):
        lines += [_img(figs, k, f"1-day forecasts vs realized, {k.rsplit('_', 1)[-1]}"), ""]
    lines += [_section(tables.get("leaderboard_pre2021_dev")), "## 3. Implied volatility at 1 month", "",
              _img(figs, "iv_vs_models_1m", "Implied vs model 1-month volatility"), "",
              _section(tables.get("iv_1m_dev"))]
    if has_holdout:
        lines += [_section(tables.get("iv_1m_holdout"))]
    lines += [_section(tables.get("encompassing_dev"))]
    if has_holdout:
        lines += [_section(tables.get("encompassing_holdout"))]
    lines += [_section(tables.get("mz_dev"))]
    if has_holdout:
        lines += [_section(tables.get("mz_holdout"))]
    lines += ["## 4. Value-at-Risk and Expected Shortfall", "", _section(tables.get("risk_leaderboard_dev"))]
    if has_holdout:
        lines += [_section(tables.get("risk_leaderboard_holdout"))]
    lines += [_section(tables.get("risk_backtests_dev"))]
    if has_holdout:
        lines += [_section(tables.get("risk_backtests_holdout"))]
    lines += ["Daily returns with the 99% VaR of COMBO+FHS (primary) and HS-250, breaches marked, and the rolling "
              "Basel zone below each panel (zone names in the legend; colour is never the only cue).", ""]
    for k in sorted((k for k in figs if k.startswith("var_breaches_") and figs[k] is not None),
                    key=lambda k: P.asset_rank(k.rsplit("_", 1)[-1])):
        lines += [_img(figs, k, f"VaR breaches, {k.rsplit('_', 1)[-1]}"), ""]
    lines += [
        "## 5. Jumps and downside semivariance (H5)",
        "",
        _img(figs, "h5_jumps", "Jump and semivariance shares by asset"),
        "",
        "J/RV = mean daily share of realized variance attributed to significant jumps (BNS ratio test at 99.9%); "
        "jump-day share = share of sessions with a significant jump; RS⁻/RV = mean daily share of downside "
        "semivariance; gains = 1 − QLIKE ratio vs HAR at 1d (positive = better than HAR). H5 evidence per asset is "
        "in the H5 table of section 1.",
        "",
    ]
    return "\n".join(lines).replace("\n\n\n", "\n\n")
