"""Mechanical verdicts for the pre-registered hypotheses H1–H5 (SPEC §0, decision rules at the 5% level).

Every rule is a pure function of the evaluation / risk tables written by ``pipeline.stage_evaluate`` and
``pipeline.stage_risk`` (and, for H5, of the gold daily measures). Each returns a :class:`HypothesisResult`
with one row per asset (H2: per asset × horizon cell) holding the evidence numbers and a verdict

- ``"supported"`` — the rule holds for that unit,
- ``"not supported"`` — the rule fails for that unit,
- ``"n/a"`` — the inputs for that unit are missing (asset not run yet, no implied vol in the holdout, …),

plus an overall verdict over the units the rule requires: ``"not supported"`` as soon as one required unit
fails, ``"supported"`` when every required unit holds, otherwise ``"n/a"`` (incomplete evidence).

Rules (SPEC §0):

- H1: at 1d, both GARCH and GJR have QLIKE ratio vs HAR > 1 with DM-HLN p < 0.05 — all four assets.
- H2: in every horizon cell the 90% MCS contains at least one of HAR, HAR-CJ, SHAR, HARQ (the DM of LGBM/MLP vs
  HAR is reported alongside) — all 12 cells. The holdout MCS exists at 1d and 1w only (SPEC §8: 1m holdout is
  descriptive), so the holdout verdict covers those 8 cells.
- H3: at 1m, the encompassing coefficient ``c`` of COMBO is > 0 with p < 0.05 (HAR secondary) — SPX, BTC, ETH.
- H4: COMBO+FHS has ``fz0_diff < 0`` vs HS-250 and a strictly higher rolling-250 green-zone share than HS-250
  (DM p reported) — all four assets.
- H5 (descriptive): J/RV, jump-day share, RS⁻/RV and the 1d QLIKE gains (1 − ratio vs HAR) of HAR-CJ and SHAR
  are larger for both BTC and ETH than for EURUSD.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from volrisk import config as C
from volrisk.palette import ASSET_ORDER, HAR_FAMILY, HORIZON_ORDER, asset_rank

log = logging.getLogger(__name__)

SUPPORTED = "supported"
NOT_SUPPORTED = "not supported"
NA = "n/a"
VERDICTS = (SUPPORTED, NOT_SUPPORTED, NA)
ALPHA = 0.05

H3_ASSETS = ("SPX", "BTC", "ETH")
H5_CRYPTO = ("BTC", "ETH")
H5_REFERENCE = "EURUSD"
H5_METRICS = ("j_rv", "jump_share", "rsneg_rv", "gain_HAR-CJ", "gain_SHAR")
H5_MEASURES_FILE = "h5_measures.parquet"

STATEMENTS = {
    "H1": "HAR-type models beat GARCH-type models on QLIKE at 1d for all four assets.",
    "H2": "Machine learning (LightGBM, MLP) is not significantly better than the best HAR model.",
    "H3": "At 1m, model forecasts contain information beyond implied vol (SPX, BTC, ETH).",
    "H4": "COMBO+FHS has a lower FZ0 score than HS-250 and spends more time in the Basel green zone.",
    "H5": "Jumps and downside semivariance matter more in crypto than in EUR/USD (descriptive).",
}
RULES = {
    "H1": "at 1d, GARCH and GJR both have QLIKE ratio vs HAR > 1 with DM-HLN p < 0.05; all four assets",
    "H2": "every asset × horizon cell: the 90% MCS contains HAR, HARQ, HAR-CJ or SHAR; all 12 cells",
    "H3": "at 1m, encompassing coefficient c of COMBO > 0 with p < 0.05; SPX, BTC and ETH",
    "H4": "COMBO+FHS fz0_diff vs HS-250 < 0 and rolling-250 green share > HS-250's; all four assets",
    "H5": "J/RV, jump-day share, RS⁻/RV and 1d QLIKE gains of HAR-CJ and SHAR vs HAR larger for BTC and ETH "
    "than for EURUSD",
}


# --------------------------------------------------------------------------------------------- formatting
def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def fmt_num(x, digits: int = 3, signed: bool = False) -> str:
    """Fixed-decimal number; '—' for missing values."""
    if not _finite(x):
        return "—"
    s = f"{float(x):+.{digits}f}" if signed else f"{float(x):.{digits}f}"
    return s.replace("-", "−")


def fmt_p(p) -> str:
    """p-value with 3 decimals, '<0.001' below that; '—' when missing."""
    if not _finite(p):
        return "—"
    return "<0.001" if float(p) < 0.001 else f"{float(p):.3f}"


def fmt_share(x, signed: bool = False) -> str:
    """Share as a percentage with one decimal ('−' for negatives)."""
    if not _finite(x):
        return "—"
    s = f"{100 * float(x):+.1f}%" if signed else f"{100 * float(x):.1f}%"
    return s.replace("-", "−")


# --------------------------------------------------------------------------------------------- result type
@dataclass
class HypothesisResult:
    """Verdict of one hypothesis: per-unit evidence rows (``unit``, evidence columns, ``required``,
    ``verdict``, ``evidence``, ``note``) and the overall verdict over the required units."""

    hypothesis: str
    rows: pd.DataFrame
    overall: str
    note: str = ""
    statement: str = field(default="")
    rule: str = field(default="")

    def __post_init__(self) -> None:
        self.statement = self.statement or STATEMENTS.get(self.hypothesis, "")
        self.rule = self.rule or RULES.get(self.hypothesis, "")


def overall_verdict(verdicts: Iterable[str]) -> str:
    """'not supported' if any unit fails, 'supported' if all hold, else 'n/a' (missing evidence)."""
    v = list(verdicts)
    if any(x == NOT_SUPPORTED for x in v):
        return NOT_SUPPORTED
    if v and all(x == SUPPORTED for x in v):
        return SUPPORTED
    return NA


def _finish(hyp: str, rows: list[dict], note: str = "", missing_note: str | None = None) -> HypothesisResult:
    df = pd.DataFrame(rows)
    req = df.loc[df["required"], "verdict"] if len(df) else pd.Series(dtype=object)
    overall = overall_verdict(req)
    missing = list(df.loc[df["required"] & (df["verdict"] == NA), "unit"]) if len(df) else []
    if overall == NA and missing:
        extra = missing_note or f"missing evidence: {', '.join(missing)}"
        note = f"{note}; {extra}" if note else extra
    return HypothesisResult(hyp, df, overall, note)


def _empty(df: pd.DataFrame | None, cols: Sequence[str]) -> bool:
    return df is None or df.empty or any(c not in df.columns for c in cols)


def _row(df: pd.DataFrame | None, **eq) -> pd.Series | None:
    """First row of ``df`` matching every ``column=value``; None when there is none (or no such column)."""
    if df is None or df.empty or any(k not in df.columns for k in eq):
        return None
    m = np.ones(len(df), dtype=bool)
    for k, v in eq.items():
        m &= (df[k].astype(str) == str(v)).to_numpy()
    sub = df[m]
    return None if sub.empty else sub.iloc[0]


def _get(r: pd.Series | None, col: str) -> float:
    """Float value of ``r[col]``; NaN when the row or column is missing or not numeric."""
    if r is None or col not in r.index:
        return np.nan
    v = r[col]
    return float(v) if _finite(v) else np.nan


# --------------------------------------------------------------------------------------------- H1
def h1(dm_har: pd.DataFrame | None, assets: Sequence[str] = ASSET_ORDER, alpha: float = ALPHA) -> HypothesisResult:
    """H1 from the DM-vs-HAR table (``eval_dm_har``: asset, horizon, model, ratio, dm_hln, pvalue, pvalue_2n)."""
    ok = not _empty(dm_har, ("asset", "horizon", "model", "ratio", "pvalue"))
    rows = []
    for a in assets:
        rec: dict = {"unit": a, "asset": a, "required": True}
        parts, passes = [], []
        for m in ("GARCH", "GJR"):
            r = _row(dm_har, asset=a, horizon="1d", model=m) if ok else None
            ratio = _get(r, "ratio")
            p = _get(r, "pvalue")
            rec[f"ratio_{m}"], rec[f"dm_{m}"], rec[f"p_{m}"] = ratio, _get(r, "dm_hln"), p
            rec[f"p2n_{m}"] = _get(r, "pvalue_2n")
            if _finite(ratio) and _finite(p):
                passes.append(ratio > 1 and p < alpha)
                parts.append(f"{m} ratio {fmt_num(ratio)} (DM p {fmt_p(p)})")
        if len(passes) < 2 and all(passes):
            verdict, note = NA, f"no 1d DM result for {a}"
        else:
            verdict, note = (SUPPORTED if all(passes) else NOT_SUPPORTED), ""
        rec.update(verdict=verdict, evidence="; ".join(parts) or "—", note=note)
        rows.append(rec)
    return _finish("H1", rows)


# --------------------------------------------------------------------------------------------- H2
def h2(
    mcs: pd.DataFrame | None,
    dm_har: pd.DataFrame | None = None,
    assets: Sequence[str] = ASSET_ORDER,
    horizons: Sequence[str] = HORIZON_ORDER,
) -> HypothesisResult:
    """H2 from the MCS table (``eval_mcs``: asset, horizon, model, pvalue, in_90) and, alongside, the DM of
    LGBM / MLP vs HAR. Cells outside ``horizons`` (the holdout has no 1m MCS) are shown as 'n/a' and are not
    required."""
    ok = not _empty(mcs, ("asset", "horizon", "model", "in_90"))
    ok_dm = not _empty(dm_har, ("asset", "horizon", "model", "ratio", "pvalue"))
    rows = []
    for a in assets:
        for h in HORIZON_ORDER:
            required = h in horizons
            rec: dict = {"unit": f"{a} {h}", "asset": a, "horizon": h, "required": required}
            g = mcs[(mcs["asset"] == a) & (mcs["horizon"] == h)] if ok else pd.DataFrame()
            for m in ("LGBM", "MLP"):
                r = _row(dm_har, asset=a, horizon=h, model=m) if ok_dm else None
                rec[f"ratio_{m}"], rec[f"p_{m}"] = _get(r, "ratio"), _get(r, "pvalue")
            ml = "; ".join(f"{m} {fmt_num(rec[f'ratio_{m}'])} (p {fmt_p(rec[f'p_{m}'])})" for m in ("LGBM", "MLP"))
            if g.empty:
                note = "no MCS at this horizon in this mode (SPEC §8)" if not required else f"no MCS result for {a} {h}"
                rec.update(har_in_mcs="", mcs_size=np.nan, best_har_mcs_p=np.nan, verdict=NA, evidence="—",
                           note=note)
            else:
                members = [str(m) for m in g.loc[g["in_90"].astype(bool), "model"]]
                har_in = [m for m in HAR_FAMILY if m in members]
                har_p = g.loc[g["model"].isin(HAR_FAMILY), "pvalue"] if "pvalue" in g else pd.Series(dtype=float)
                rec.update(
                    har_in_mcs=", ".join(har_in),
                    mcs_size=len(members),
                    best_har_mcs_p=float(har_p.max()) if len(har_p) else np.nan,
                    verdict=SUPPORTED if har_in else NOT_SUPPORTED,
                    evidence=(f"HAR family in MCS: {', '.join(har_in) if har_in else 'none'} "
                              f"({len(members)} members); {ml}"),
                    note="" if required else "descriptive cell (not part of this mode's rule)",
                )
                if not required:
                    rec["verdict"] = NA
            rows.append(rec)
    note = "" if tuple(horizons) == HORIZON_ORDER else f"evaluated on horizons {', '.join(horizons)} only"
    return _finish("H2", rows, note)


# --------------------------------------------------------------------------------------------- H3
def h3(
    encompassing: pd.DataFrame | None,
    assets: Sequence[str] = ASSET_ORDER,
    required: Sequence[str] = H3_ASSETS,
    alpha: float = ALPHA,
) -> HypothesisResult:
    """H3 from the encompassing table (``eval_encompassing``: asset, model, c, se_c, p_c, c_ivlag, p_c_ivlag)."""
    ok = not _empty(encompassing, ("asset", "model", "c", "p_c"))
    rows = []
    for a in assets:
        rec: dict = {"unit": a, "asset": a, "required": a in required}
        for m in ("COMBO", "HAR"):
            r = _row(encompassing, asset=a, model=m) if ok else None
            rec[f"c_{m}"], rec[f"p_{m}"] = _get(r, "c"), _get(r, "p_c")
            rec[f"c_ivlag_{m}"], rec[f"p_ivlag_{m}"] = _get(r, "c_ivlag"), _get(r, "p_c_ivlag")
            if m == "COMBO":
                rec["T"] = _get(r, "T")
        c, p = rec["c_COMBO"], rec["p_COMBO"]
        if not (_finite(c) and _finite(p)):
            verdict, evidence = NA, "—"
            note = "no implied-vol benchmark / encompassing result"
        else:
            verdict = SUPPORTED if (c > 0 and p < alpha) else NOT_SUPPORTED
            evidence = (f"COMBO c {fmt_num(c, signed=True)} (p {fmt_p(p)}); "
                        f"HAR c {fmt_num(rec['c_HAR'], signed=True)} (p {fmt_p(rec['p_HAR'])})")
            note = "" if a in required else "outside the H3 rule (SPX, BTC, ETH); shown for information"
        rec.update(verdict=verdict, evidence=evidence, note=note)
        rows.append(rec)
    return _finish("H3", rows)


# --------------------------------------------------------------------------------------------- H4
def _green_shares(
    risk_leaderboard: pd.DataFrame | None, backtests: pd.DataFrame | None, time_in_zone: pd.DataFrame | None
) -> tuple[pd.DataFrame | None, str]:
    """(asset, model, green) and its source: ``risk_time_in_zone`` first (dev: share of rolling-250 windows in the
    green zone; holdout: 1/0 for the zone over the holdout observations at the actual N — see DEVIATIONS), then the
    risk leaderboard / 99% backtest green share."""
    if not _empty(time_in_zone, ("asset", "model", "green")):
        return time_in_zone[["asset", "model", "green"]], "time_in_zone"
    if not _empty(risk_leaderboard, ("asset", "model", "green_share")):
        return risk_leaderboard[["asset", "model", "green_share"]].rename(columns={"green_share": "green"}), \
            "risk_leaderboard"
    if not _empty(backtests, ("asset", "model", "level", "green_share")):
        bt = backtests[backtests["level"].astype(str) == "99"]
        return bt[["asset", "model", "green_share"]].rename(columns={"green_share": "green"}), "backtests_99"
    return None, ""


def h4(
    risk_leaderboard: pd.DataFrame | None = None,
    *,
    fz0: pd.DataFrame | None = None,
    backtests: pd.DataFrame | None = None,
    time_in_zone: pd.DataFrame | None = None,
    assets: Sequence[str] = ASSET_ORDER,
) -> HypothesisResult:
    """H4 from the risk leaderboard (``risk_risk_leaderboard``) or ``risk_fz0`` for the FZ0 difference / DM p,
    and the rolling-250 green share from ``risk_time_in_zone`` (preferred), the leaderboard or the 99% backtests.
    A tie in the green share is not 'higher' (not supported)."""
    fz = risk_leaderboard if not _empty(risk_leaderboard, ("asset", "model", "fz0_diff")) else fz0
    ok = not _empty(fz, ("asset", "model", "fz0_diff"))
    green, source = _green_shares(risk_leaderboard, backtests, time_in_zone)
    rows = []
    for a in assets:
        rec: dict = {"unit": a, "asset": a, "required": True, "green_source": source}
        r = _row(fz, asset=a, model="COMBO+FHS") if ok else None
        ref = _row(fz, asset=a, model="HS-250") if ok else None
        rec["fz0_diff"] = _get(r, "fz0_diff") if ref is not None else np.nan
        rec["p_dm"] = _get(r, "p_dm")
        rec["green_COMBO+FHS"] = _get(_row(green, asset=a, model="COMBO+FHS"), "green")
        rec["green_HS-250"] = _get(_row(green, asset=a, model="HS-250"), "green")
        d, g1, g0 = rec["fz0_diff"], rec["green_COMBO+FHS"], rec["green_HS-250"]
        if not (_finite(d) and _finite(g1) and _finite(g0)):
            verdict, evidence, note = NA, "—", f"no risk results for {a}"
        else:
            verdict = SUPPORTED if (d < 0 and g1 > g0) else NOT_SUPPORTED
            evidence = (f"FZ0 diff {fmt_num(d, signed=True)} (DM p {fmt_p(rec['p_dm'])}); green share "
                        f"{fmt_share(g1)} vs HS-250 {fmt_share(g0)}")
            note = "green shares tie (not higher)" if g1 == g0 else ""
        rec.update(verdict=verdict, evidence=evidence, note=note)
        rows.append(rec)
    return _finish("H4", rows)


# --------------------------------------------------------------------------------------------- H5
def daily_measures(daily: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    """Per-asset jump and downside-semivariance shares over sessions in [start, end] (gold daily, SPEC §5.3):
    ``j_rv`` = mean of daily J/RV, ``jump_share`` = share of sessions with a significant jump (J > 0),
    ``rsneg_rv`` = mean of daily RS⁻/RV. Sessions with non-positive RV are skipped."""
    cols = ["asset", "n_sessions", "j_rv", "jump_share", "rsneg_rv", "start", "end"]
    if daily is None or daily.empty:
        return pd.DataFrame(columns=cols)
    d = daily
    dates = pd.to_datetime(d["session_date"])
    m = np.ones(len(d), dtype=bool)
    if start is not None:
        m &= (dates >= pd.Timestamp(start)).to_numpy()
    if end is not None:
        m &= (dates <= pd.Timestamp(end)).to_numpy()
    d = d[m & np.isfinite(d["rv"].astype(float)).to_numpy() & (d["rv"] > 0).to_numpy()]
    rows = []
    for a, g in d.groupby("asset", sort=False):
        rv = g["rv"].astype(float)
        sd = pd.to_datetime(g["session_date"])
        rows.append({
            "asset": a,
            "n_sessions": len(g),
            "j_rv": float((g["j"] / rv).mean()),
            "jump_share": float((g["j"] > 0).mean()),
            "rsneg_rv": float((g["rs_neg"] / rv).mean()),
            "start": sd.min(),
            "end": sd.max(),
        })
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("asset", key=lambda s: s.map(asset_rank)).reset_index(drop=True)


def h5(
    daily: pd.DataFrame | None,
    leaderboard: pd.DataFrame | None,
    assets: Sequence[str] = ASSET_ORDER,
    start=None,
    end=None,
    crypto: Sequence[str] = H5_CRYPTO,
    reference: str = H5_REFERENCE,
) -> HypothesisResult:
    """H5 (descriptive). ``daily`` is either the gold daily table (measures computed over [start, end]) or a
    precomputed :func:`daily_measures` frame; ``leaderboard`` is ``eval_leaderboard`` (1d QLIKE ratios)."""
    if daily is not None and {"j_rv", "jump_share", "rsneg_rv"} <= set(daily.columns):
        meas = daily
    else:
        meas = daily_measures(daily, start, end) if daily is not None else pd.DataFrame()
    ok_lb = not _empty(leaderboard, ("asset", "horizon", "model", "qlike_ratio"))
    vals: dict[str, dict] = {}
    for a in assets:
        rec: dict = {"unit": a, "asset": a}
        r = _row(meas, asset=a)
        for k in ("j_rv", "jump_share", "rsneg_rv", "n_sessions"):
            rec[k] = _get(r, k)
        for m in ("HAR-CJ", "SHAR"):
            lr = _row(leaderboard, asset=a, horizon="1d", model=m) if ok_lb else None
            rec[f"gain_{m}"] = 1.0 - _get(lr, "qlike_ratio")
        vals[a] = rec
    ref = vals.get(reference)
    ref_ok = ref is not None and all(_finite(ref[k]) for k in H5_METRICS)
    rows = []
    for a in assets:
        rec = vals[a] | {"required": a in crypto}
        own_ok = all(_finite(rec[k]) for k in H5_METRICS)
        rec["evidence"] = (
            f"J/RV {fmt_share(rec['j_rv'])}, jump days {fmt_share(rec['jump_share'])}, "
            f"RS⁻/RV {fmt_share(rec['rsneg_rv'])}, 1d QLIKE gain vs HAR: HAR-CJ "
            f"{fmt_share(rec['gain_HAR-CJ'], signed=True)}, SHAR {fmt_share(rec['gain_SHAR'], signed=True)}"
        ) if own_ok else "—"
        if a not in crypto:
            rec["n_larger"] = np.nan
            rec["verdict"] = NA
            rec["note"] = "reference asset" if a == reference else "not part of the H5 comparison"
        elif not own_ok:
            rec.update(n_larger=np.nan, verdict=NA, note=f"no measures for {a}")
        elif not ref_ok:
            rec.update(n_larger=np.nan, verdict=NA, note=f"no {reference} reference measures")
        else:
            larger = [k for k in H5_METRICS if rec[k] > ref[k]]
            rec["n_larger"] = len(larger)
            rec["verdict"] = SUPPORTED if len(larger) == len(H5_METRICS) else NOT_SUPPORTED
            smaller = [k for k in H5_METRICS if k not in larger]
            rec["note"] = f"{len(larger)}/{len(H5_METRICS)} larger than {reference}" + (
                f" (not larger: {', '.join(smaller)})" if smaller else "")
        rows.append(rec)
    missing = None if ref_ok else f"missing evidence: no {reference} reference measures"
    return _finish("H5", rows, "descriptive", missing_note=missing)


# --------------------------------------------------------------------------------------------- loading
def read_table(results_dir: Path, name: str) -> pd.DataFrame:
    """A results parquet as pandas, or an empty frame when it is missing or has no columns."""
    p = Path(results_dir) / f"{name}.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    return df if len(df.columns) else pd.DataFrame()


def _default_dir(mode: str) -> Path:
    return C.RESULTS / "holdout" if mode == "holdout" else C.RESULTS


def _h5_inputs(results_dir: Path, mode: str, daily: pd.DataFrame | None) -> tuple[pd.DataFrame | None, str]:
    """Daily measures for H5 and a note on their window. The report never opens the sealed holdout gold
    table: in holdout mode it uses ``h5_measures.parquet`` from the holdout results when the holdout run wrote
    it, else the development window."""
    start, end = C.dev_eval_start(), C.dev_end()
    if daily is not None:
        if {"j_rv", "jump_share", "rsneg_rv"} <= set(daily.columns):
            return daily, ""
        if mode == "dev":
            return daily_measures(daily, start, end), ""
        return daily_measures(daily, C.holdout_start(), C.data_end()), "jump measures over the holdout window"
    if mode == "holdout":
        pre = read_table(results_dir, H5_MEASURES_FILE.removesuffix(".parquet"))
        if not pre.empty:
            return pre, "jump measures over the holdout window"
    try:
        from volrisk import io

        dev = io.load_daily()  # development gold only; never the holdout
    except (FileNotFoundError, OSError) as e:
        log.warning("H5: gold daily table unavailable (%s)", e)
        return None, "gold daily table unavailable"
    note = "" if mode == "dev" else "jump measures over the development window (holdout gold table not read)"
    return daily_measures(dev, start, end), note


def run_all(
    results_dir: Path | str | None = None, mode: str = "dev", daily: pd.DataFrame | None = None
) -> dict[str, HypothesisResult]:
    """All five hypotheses for one mode from the tables in ``results_dir`` (the directory holding that mode's
    ``eval_*`` / ``risk_*`` files: ``data/results`` for dev, ``data/results/holdout`` for the holdout).
    ``daily`` overrides the gold daily table used by H5 (raw daily rows or precomputed measures)."""
    if mode not in ("dev", "holdout"):
        raise ValueError(f"unknown mode {mode!r}")
    rd = Path(results_dir) if results_dir is not None else _default_dir(mode)
    t = {n: read_table(rd, n) for n in ("eval_dm_har", "eval_mcs", "eval_encompassing", "eval_leaderboard",
                                         "risk_risk_leaderboard", "risk_fz0", "risk_backtests",
                                         "risk_time_in_zone")}
    assets = list(ASSET_ORDER) + sorted(
        {str(a) for a in t["eval_leaderboard"].get("asset", pd.Series(dtype=object))} - set(ASSET_ORDER)
    )
    res = {
        "H1": h1(t["eval_dm_har"], assets),
        "H2": h2(t["eval_mcs"], t["eval_dm_har"], assets,
                 horizons=HORIZON_ORDER if mode == "dev" else ("1d", "1w")),
        "H3": h3(t["eval_encompassing"], assets),
        "H4": h4(t["risk_risk_leaderboard"], fz0=t["risk_fz0"], backtests=t["risk_backtests"],
                 time_in_zone=t["risk_time_in_zone"], assets=assets),
    }
    meas, note = _h5_inputs(rd, mode, daily)
    r5 = h5(meas, t["eval_leaderboard"], assets)
    if note:
        r5.note = f"{r5.note}; {note}"
    res["H5"] = r5
    if mode == "holdout":
        res["H2"].note = "; ".join(x for x in (res["H2"].note, "holdout MCS at 1d and 1w only (SPEC §8)") if x)
        res["H2"].rule = RULES["H2"].replace("all 12 cells", "the 8 cells at 1d and 1w (no 1m MCS in the holdout)")
    return res


def combine(results: dict[str, HypothesisResult], mode: str = "dev") -> pd.DataFrame:
    """Long verdict table: one row per (hypothesis, unit) plus an 'overall' row per hypothesis."""
    rows = []
    for hyp, r in results.items():
        for _, x in r.rows.iterrows():
            rows.append({"mode": mode, "hypothesis": hyp, "unit": x["unit"], "asset": x.get("asset"),
                         "horizon": x.get("horizon", None), "required": bool(x["required"]),
                         "verdict": x["verdict"], "evidence": x["evidence"], "note": x["note"]})
        rows.append({"mode": mode, "hypothesis": hyp, "unit": "overall", "asset": None, "horizon": None,
                     "required": True, "verdict": r.overall, "evidence": r.rule, "note": r.note})
    return pd.DataFrame(rows, columns=["mode", "hypothesis", "unit", "asset", "horizon", "required", "verdict",
                                       "evidence", "note"])


def verdicts(results_dir: Path | str | None = None, mode: str = "dev", daily: pd.DataFrame | None = None
             ) -> pd.DataFrame:
    """H1–H5 verdicts for ``mode`` in {'dev', 'holdout'} as a long table (see :func:`combine`).

    ``results_dir`` is the directory with that mode's tables (default ``data/results`` for dev and
    ``data/results/holdout`` for the holdout)."""
    return combine(run_all(results_dir, mode, daily), mode)
