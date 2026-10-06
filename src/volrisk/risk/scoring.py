"""FZ0 joint (VaR, ES) scoring function of Patton, Ziegel & Chen, and its model comparison (SPEC §9).

Return space: ``y = r_cc``, ``v = -VaR97.5``, ``e = -ES97.5`` with ``e <= v < 0``:
``FZ0 = -(1/(α e))·1{y <= v}(v - y) + v/e + ln(-e) - 1``. Lower is better.

The comparison runs per asset on common dates: MCS over the six risk models, and DM-HLN against ``HS-250``,
both with ``n_max = 1``. FZ0 is not scale-free: ``FZ0(c·y, c·v, c·e) = FZ0(y, v, e) + ln c``, and at the
true model ``E[FZ0] = ln ES``. So a mean FZ0 can be near zero or negative, and a *ratio* of means can
point the wrong way. The headline metric is therefore the mean FZ0 **difference** vs HS-250, which is
scale-free (deviation from the SPEC's "FZ0 ratio").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from volrisk import config as C
from volrisk.evaluation.dm import dm_vs_ref
from volrisk.evaluation.mcs import run_mcs
from volrisk.risk.var_es import RISK_MODELS

REF_MODEL = "HS-250"
FZ0_SUMMARY_COLUMNS = ["asset", "model", "n", "fz0", "fz0_diff", "dm_hln", "p_dm", "mcs_p", "in_90", "in_75"]
RISK_LEADERBOARD_COLUMNS = [*FZ0_SUMMARY_COLUMNS, "zone_last", "green_share"]
_TOL = 1e-12  # relative slack for e <= v (a tail mean can exceed its quantile by rounding)


def fz0(r, var975, es975, alpha: float = 0.025) -> np.ndarray:
    """Per-observation FZ0 loss; VaR and ES are positive numbers (loss convention)."""
    y = np.asarray(r, dtype=float)
    v = -np.asarray(var975, dtype=float)
    e = -np.asarray(es975, dtype=float)
    if not (y.shape == v.shape == e.shape):
        raise ValueError("r, var975 and es975 must have the same shape")
    if not (np.all(v < 0) and np.all(e <= v + _TOL * np.abs(v))):
        raise ValueError("FZ0 needs 0 < VaR <= ES (e <= v < 0 in return space)")
    return -(1.0 / (alpha * e)) * (y <= v) * (v - y) + v / e + np.log(-e) - 1.0


def add_fz0(risk: pd.DataFrame, alpha: float = 0.025) -> pd.DataFrame:
    """Copy of a risk frame (RISK_COLUMNS) with an ``fz0`` column, ready for MCS / DM-HLN (n_max = 1)."""
    out = risk.copy()
    out["fz0"] = fz0(out["r_cc"], out["var975"], out["es975"], alpha)
    return out


def fz0_wide(risk: pd.DataFrame, alpha: float = 0.025) -> pd.DataFrame:
    """Date × model FZ0 losses of ONE asset, restricted to the dates on which every model is scored.

    Columns follow the order of ``RISK_MODELS``; any other model name comes after them.
    """
    if "asset" in risk.columns and risk["asset"].nunique() > 1:
        raise ValueError("fz0_wide expects the rows of one asset")
    wide = add_fz0(risk, alpha).pivot(index="date", columns="model", values="fz0")
    cols = [m for m in RISK_MODELS if m in wide.columns] + [m for m in wide.columns if m not in RISK_MODELS]
    out = wide[cols].dropna().sort_index()
    out.columns.name = None
    return out


def fz0_summary(
    risk: pd.DataFrame,
    ref: str = REF_MODEL,
    mcs_reps: int | None = None,
    seed: int | None = None,
    alpha: float = 0.025,
) -> pd.DataFrame:
    """FZ0 comparison of the risk models per asset (FZ0_SUMMARY_COLUMNS).

    For every asset, on common dates: the mean FZ0 (``fz0``), ``fz0_diff = mean(FZ0_model - FZ0_ref)``
    (negative means better than ``ref``), DM-HLN vs ``ref`` (``dm_hln``, ``p_dm``), and MCS p-values and
    90%/75% membership. Both tests use ``n_max = 1``; the MCS settings come from the ``evaluation`` config
    (``mcs_reps`` overrides the replications; ``seed`` defaults to the project seed). The caller selects
    the window beforehand (one split, e.g. dev dates >= 2021-01-01), as for ``backtest_table``.
    """
    if "split" in risk.columns and risk["split"].nunique() > 1:
        raise ValueError("risk frame mixes splits; select dev or holdout rows first")
    ev = C.load()["evaluation"]
    reps = int(ev["mcs_reps"]) if mcs_reps is None else mcs_reps
    rows = []
    for asset, g in risk.groupby("asset", sort=False):
        wide = fz0_wide(g, alpha)
        if ref not in wide.columns:
            raise KeyError(f"{asset}: reference model {ref!r} not in {list(wide.columns)}")
        if wide.empty:
            raise ValueError(f"{asset}: no common dates across the risk models")
        dm = dm_vs_ref(wide, ref, n_max=1).set_index("model")
        mcs = run_mcs(wide, n_max=1, size=float(ev["mcs_size"]), reps=reps, seed=seed).set_index("model")
        mean = wide.mean()
        for m in wide.columns:
            rows.append(
                {
                    "asset": asset,
                    "model": m,
                    "n": len(wide),
                    "fz0": float(mean[m]),
                    "fz0_diff": float((wide[m] - wide[ref]).mean()),
                    "dm_hln": float(dm.at[m, "dm_hln"]) if m != ref else np.nan,
                    "p_dm": float(dm.at[m, "pvalue"]) if m != ref else np.nan,
                    "mcs_p": float(mcs.at[m, "pvalue"]),
                    "in_90": bool(mcs.at[m, "in_90"]),
                    "in_75": bool(mcs.at[m, "in_75"]),
                }
            )
    return pd.DataFrame(rows, columns=FZ0_SUMMARY_COLUMNS)


def risk_leaderboard(summary: pd.DataFrame, backtests: pd.DataFrame, level: str = "99") -> pd.DataFrame:
    """Leaderboard "risk" column (RISK_LEADERBOARD_COLUMNS): ``fz0_summary`` joined with the current
    rolling traffic-light zone and the green-zone share at ``level`` (Basel: 99%) from ``backtest_table``.
    Both inputs must cover the same window."""
    bt = backtests.loc[backtests["level"] == level, ["asset", "model", "zone_last", "green_share"]]
    if bt.duplicated(["asset", "model"]).any():
        raise ValueError("backtest table has duplicate (asset, model) rows at this level")
    out = summary.merge(bt, on=["asset", "model"], how="left", validate="one_to_one")
    return out[RISK_LEADERBOARD_COLUMNS]
