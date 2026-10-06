"""Forecast-evaluation suite over one split (SPEC §8): the tables behind the leaderboard, DM, MCS, MZ and
encompassing results. Pure library code — ``pipeline.stage_evaluate`` handles I/O.

Development mode evaluates origins in the ``dev`` split from ``dev_eval_start`` (headline) and keeps earlier
SPX/EURUSD origins as a robustness window; the IV-inclusive 1m QLIKE/DM/MCS use every dev origin of each asset's
IV subsample. Holdout mode evaluates the ``holdout`` split (MCS at 1d/1w only, bootstrap CIs for the QLIKE ratio
vs HAR).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from volrisk import config as C
from volrisk.evaluation.bootstrap import qlike_ratio_ci
from volrisk.evaluation.dm import dm_vs_ref
from volrisk.evaluation.iv import IV_MODELS
from volrisk.evaluation.leaderboard import common_dates, leaderboard, losses_frame
from volrisk.evaluation.mcs import run_mcs
from volrisk.evaluation.mz import encompassing, hac_lag, mincer_zarnowitz, mz_log, non_overlapping

log = logging.getLogger(__name__)

REF = "HAR"
IV_REF = "IV-cal"
HAR_FAMILY = ("HAR", "HAR-CJ", "SHAR", "HARQ")


def _eval_cfg() -> dict:
    return C.load()["evaluation"]


def select_split(losses: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(headline, robustness) loss rows for ``mode`` in {'dev', 'holdout'}."""
    if mode == "dev":
        dev = losses[losses["split"] == "dev"]
        start = pd.Timestamp(C.dev_eval_start())
        return dev[dev["origin"] >= start], dev[dev["origin"] < start]
    if mode == "holdout":
        return losses[losses["split"] == "holdout"], losses.iloc[0:0]
    raise ValueError(f"unknown mode {mode!r}")


def _cells(losses: pd.DataFrame):
    for (asset, horizon), g in losses.groupby(["asset", "horizon"], sort=True):
        yield asset, horizon, g, C.n_max(horizon, asset)


def _models(g: pd.DataFrame, with_iv: bool = False) -> list[str]:
    present = list(dict.fromkeys(g["model"]))
    return [m for m in present if with_iv or m not in IV_MODELS]


def mcs_table(losses: pd.DataFrame, with_iv: bool = False, horizons=C.HORIZONS) -> pd.DataFrame:
    cfg = _eval_cfg()
    rows = []
    for asset, horizon, g, n_max in _cells(losses):
        if horizon not in horizons:
            continue
        wide = common_dates(g, _models(g, with_iv), "qlike")
        if wide.shape[1] < 2 or len(wide) < 30:
            continue
        res = run_mcs(wide, n_max, size=cfg["mcs_size"], reps=cfg["mcs_reps"], seed=C.seed())
        rows.append(res.assign(asset=asset, horizon=horizon, T=len(wide)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def dm_table(losses: pd.DataFrame, ref: str = REF, with_iv: bool = False) -> pd.DataFrame:
    """DM-HLN vs ``ref`` per cell (default kernel) plus the 2·n_max Bartlett robustness p-value."""
    rows = []
    for asset, horizon, g, n_max in _cells(losses):
        ms = _models(g, with_iv)
        if ref not in ms:
            continue
        wide = common_dates(g, ms, "qlike")
        if len(wide) < 30:
            continue
        main = dm_vs_ref(wide, ref, n_max)
        rob = dm_vs_ref(wide, ref, n_max, maxlags=2 * n_max)[["model", "pvalue"]]
        main = main.merge(rob.rename(columns={"pvalue": "pvalue_2n"}), on="model")
        main["ratio"] = (wide[main["model"]].mean() / wide[ref].mean()).to_numpy()
        rows.append(main.assign(asset=asset, horizon=horizon))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def iv_frame(losses_1m: pd.DataFrame) -> pd.DataFrame:
    """Per-origin IV variance (model 'IV' forecast) for joining onto model rows at 1m."""
    iv = losses_1m[losses_1m["model"] == "IV"][["asset", "origin", "F"]].rename(columns={"F": "iv_var"})
    iv = iv.sort_values(["asset", "origin"])
    iv["iv_var_lag1"] = iv.groupby("asset")["iv_var"].shift(1)  # robustness: IV_{t-1} (previous IV origin)
    return iv


def mz_table(losses: pd.DataFrame) -> pd.DataFrame:
    """Mincer–Zarnowitz at 1m for every model incl. IV benchmarks: levels (descriptive) and log (inferential),
    full overlapping sample (HAC 2·n_max) and non-overlapping subsample (HAC 0)."""
    rows = []
    l1m = losses[losses["horizon"] == "1m"]
    for (asset, model), g in l1m.groupby(["asset", "model"], sort=True):
        n_max = C.n_max("1m", asset)
        g = g.sort_values("origin")
        if len(g) < 60:
            continue
        sub = non_overlapping(g, step=n_max)
        rec = {"asset": asset, "model": model, "T": len(g), "T_nonoverlap": len(sub)}
        for tag, df, lag in (("", g, hac_lag(n_max)), ("_no", sub, hac_lag(n_max, overlapping=False))):
            if len(df) < 10:
                continue
            lv = mincer_zarnowitz(df["y"], df["F"], lag)
            lg = mz_log(df["y"], df["F"], lag)
            rec.update({f"a{tag}": lv["a"], f"b{tag}": lv["b"], f"p_wald_levels{tag}": lv["p_wald"]})
            rec.update({f"b_log{tag}": lg["b"], f"se_b_log{tag}": lg["se_b"], f"p_log{tag}": lg["p_wald"]})
        rows.append(rec)
    return pd.DataFrame(rows)


def encompassing_table(losses: pd.DataFrame) -> pd.DataFrame:
    """H3: log y = a + b log IVvar + c log F_model at 1m on the IV subsample (H0: c = 0); plus IV_{t-1}."""
    l1m = losses[losses["horizon"] == "1m"]
    iv = iv_frame(l1m)
    if iv.empty:
        return pd.DataFrame()
    rows = []
    models = [m for m in dict.fromkeys(l1m["model"]) if m not in IV_MODELS]
    for asset in sorted(iv["asset"].unique()):
        n_max = C.n_max("1m", asset)
        for model in models:
            g = l1m[(l1m["asset"] == asset) & (l1m["model"] == model)].merge(iv, on=["asset", "origin"])
            if len(g) < 60:
                continue
            res = encompassing(g["y"], g["iv_var"], g["F"], hac_lag(n_max))
            rec = {"asset": asset, "model": model, "T": len(g), **res}
            g1 = g.dropna(subset=["iv_var_lag1"])
            if len(g1) >= 60:
                rob = encompassing(g1["y"], g1["iv_var_lag1"], g1["F"], hac_lag(n_max))
                rec.update({"c_ivlag": rob["c"], "p_c_ivlag": rob["p_c"]})
            rows.append(rec)
    return pd.DataFrame(rows)


def ratio_ci_table(losses: pd.DataFrame, ref: str = REF) -> pd.DataFrame:
    """Holdout headline: QLIKE ratio vs ``ref`` with a stationary-bootstrap 90% CI per cell and model."""
    cfg = _eval_cfg()
    rows = []
    for asset, horizon, g, n_max in _cells(losses):
        ms = _models(g)
        if ref not in ms:
            continue
        wide = common_dates(g, ms, "qlike")
        if len(wide) < 30:
            continue
        for m in ms:
            if m == ref:
                continue
            ratio, lo, hi = qlike_ratio_ci(wide[m], wide[ref], n_max, reps=cfg["boot_reps"], seed=C.seed())
            rows.append({"asset": asset, "horizon": horizon, "model": m, "ratio": ratio, "lo": lo, "hi": hi,
                         "T": len(wide)})
    return pd.DataFrame(rows)


def har_star(leader: pd.DataFrame) -> dict[str, str]:
    """HAR-family member with the lowest mean QLIKE at 1d per asset (SPEC §9, risk model 4)."""
    d = leader[(leader["horizon"] == "1d") & (leader["model"].isin(HAR_FAMILY))]
    return {a: g.sort_values("qlike")["model"].iloc[0] for a, g in d.groupby("asset")}


def evaluate(forecasts: pd.DataFrame, targets: pd.DataFrame, mode: str = "dev") -> dict[str, pd.DataFrame]:
    """All evaluation tables for ``mode``. Keys: losses, leaderboard, dm_har, mcs, iv_leaderboard, iv_dm,
    iv_mcs, mz, encompassing, and (dev) leaderboard_pre2021 / (holdout) ratio_ci."""
    losses = losses_frame(forecasts, targets)
    head, rob = select_split(losses, mode)
    if head.empty:
        raise ValueError(f"no {mode} losses to evaluate")
    out: dict[str, pd.DataFrame] = {"losses": head}
    out["leaderboard"] = leaderboard(head)
    out["dm_har"] = dm_table(head)
    out["mcs"] = mcs_table(head, horizons=C.HORIZONS if mode == "dev" else ("1d", "1w"))
    # SPEC §8: IV-inclusive QLIKE/DM/MCS run on each asset's IV subsample (crypto ≈ 2022+, SPX full OOS, EURUSD
    # OOS → 2023-12-31), i.e. on every dev origin, not only the headline window; common dates with IV/IV-cal cut
    # each asset to its subsample. Encompassing (H3 rule, SPEC §0) and MZ stay on the headline window.
    iv_pool = losses[losses["split"] == "dev"] if mode == "dev" else head
    iv_assets = iv_pool.loc[iv_pool["model"].isin(IV_MODELS), "asset"].unique()
    if len(iv_assets):
        h1m = iv_pool[(iv_pool["horizon"] == "1m") & iv_pool["asset"].isin(iv_assets)]
        all_models = list(dict.fromkeys(h1m["model"]))
        out["iv_leaderboard"] = leaderboard(h1m, models=all_models)
        out["iv_dm"] = dm_table(h1m, ref=IV_REF, with_iv=True)
        out["iv_mcs"] = mcs_table(h1m, with_iv=True) if mode == "dev" else pd.DataFrame()
        out["encompassing"] = encompassing_table(head)
    out["mz"] = mz_table(head)
    # SPEC §6: the pre-2021 OOS robustness table covers SPX/EURUSD; crypto OOS starts in late 2020 (≈ 37 origins)
    pre = rob[~rob["asset"].isin(C.CRYPTO)]
    if mode == "dev" and not pre.empty:
        out["leaderboard_pre2021"] = leaderboard(pre)
    if mode == "holdout":
        out["ratio_ci"] = ratio_ci_table(head)
    for k, v in out.items():
        log.info("evaluation[%s] %s: %d rows", mode, k, len(v))
    return out
