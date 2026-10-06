"""Risk suite over one split (SPEC §9): VaR/ES of the six risk models, VaR backtests, the Acerbi–Székely ES
test, FZ0 ranking and rolling Basel traffic lights. Pure library code — ``pipeline.stage_risk`` handles I/O.
"""

from __future__ import annotations

import logging

import pandas as pd

from volrisk import config as C
from volrisk.risk.backtests import backtest_table, hits, rolling_traffic_light, zone
from volrisk.risk.es_tests import acerbi_szekely_z2, sampler_for, z2_pvalue
from volrisk.risk.scoring import fz0_summary, risk_leaderboard
from volrisk.risk.var_es import ALPHA_99, risk_frame

log = logging.getLogger(__name__)

ZONES = ("green", "yellow", "red")
TIME_IN_ZONE_COLUMNS = ["asset", "model", *ZONES, "windows", "n_obs", "basis"]


def _risk_cfg() -> dict:
    return C.load()["risk"]


def build_risk(daily_all: pd.DataFrame, forecasts: pd.DataFrame, har_star: dict[str, str]) -> pd.DataFrame:
    """Long VaR/ES frame (RISK_COLUMNS) for every asset, from the dev evaluation start onwards."""
    fc1d = forecasts[forecasts["horizon"] == "1d"]
    frames = []
    for asset, member in har_star.items():
        d = daily_all[daily_all["asset"] == asset].reset_index(drop=True)
        f = fc1d[fc1d["asset"] == asset]
        frames.append(risk_frame(asset, d, f, member, start_date=C.dev_eval_start()))
    return pd.concat(frames, ignore_index=True)


def es_table(risk: pd.DataFrame, daily_all: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    """Acerbi–Székely Z2 and its Monte Carlo p-value per (asset, model) on the given window."""
    cfg = _risk_cfg()
    fc1d = forecasts[forecasts["horizon"] == "1d"]
    rows = []
    for (asset, model), g in risk.groupby(["asset", "model"], sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        d = daily_all[daily_all["asset"] == asset].reset_index(drop=True)
        sampler = sampler_for(model, g, d, fc1d[fc1d["asset"] == asset])
        z2 = acerbi_szekely_z2(g["r_cc"], g["var975"], g["es975"])
        p = z2_pvalue(g["r_cc"], g["var975"], g["es975"], sampler, reps=cfg["mc_reps"], seed=C.seed())
        rows.append({"asset": asset, "model": model, "T": len(g), "z2": z2, "p_z2": p})
    return pd.DataFrame(rows)


def rolling_zones(risk: pd.DataFrame) -> pd.DataFrame:
    """Rolling 250-observation Basel traffic light at 99% for every (asset, model), for plots/dashboard.

    One row per complete window, dated by its last observation; ``hit`` is the 99% breach on that date
    (``time_in_green`` rebuilds the holdout's own hits from it)."""
    cfg = _risk_cfg()
    frames = []
    for (asset, model), g in risk.groupby(["asset", "model"], sort=False):
        g = g.sort_values("date")
        h = pd.Series(hits(-g["r_cc"].to_numpy(), g["var99"].to_numpy()), index=pd.DatetimeIndex(g["date"]))
        tl = rolling_traffic_light(h, window=cfg["traffic_window"], p=ALPHA_99)
        tl["hit"] = h.to_numpy()[len(h) - len(tl) :]  # complete windows drop exactly the first window-1 dates
        frames.append(tl.reset_index().rename(columns={"index": "date"}).assign(asset=asset, model=model))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def evaluate_risk(
    risk: pd.DataFrame, daily_all: pd.DataFrame, forecasts: pd.DataFrame, mode: str = "dev"
) -> dict[str, pd.DataFrame]:
    """Backtest tables for ``mode`` in {'dev', 'holdout'} (the window is the split's dates).

    ``rolling_zones`` covers the whole series (dev + holdout when present) for display; ``time_in_zone`` is the
    H4 green-zone measure of the split (see ``time_in_green``). On the holdout the leaderboard's ``green_share``
    is that same measure, so the published tables carry a single holdout green share."""
    cfg = _risk_cfg()
    win = risk[risk["split"] == mode]
    if win.empty:
        raise ValueError(f"no {mode} risk rows")
    bt = backtest_table(win, mc_reps=cfg["mc_reps"], seed=C.seed(), window=cfg["traffic_window"])
    fz = fz0_summary(win, seed=C.seed())
    rolling = rolling_zones(risk)
    tiz = time_in_green(rolling, mode)
    lb = risk_leaderboard(fz, bt)
    if mode == "holdout":
        n = win.groupby(["asset", "model"]).size().rename("n_split")
        chk = tiz.set_index(["asset", "model"])["n_obs"].reindex(n.index)
        if not chk.eq(n).all():
            raise ValueError(
                "holdout risk rows without a complete rolling window (or outside the holdout dates): "
                f"{n[~chk.eq(n)].to_dict()} rows vs {chk[~chk.eq(n)].to_dict()} rebuilt hits"
            )
        g = tiz[["asset", "model", "green"]].rename(columns={"green": "green_share"})
        lb = lb.drop(columns="green_share").merge(g, on=["asset", "model"], how="left")[list(lb.columns)]
    out = {
        "backtests": bt,
        "es": es_table(win, daily_all, forecasts),
        "fz0": fz,
        "risk_leaderboard": lb,
        "rolling_zones": rolling,  # whole series (dev + holdout when present) for display
        "time_in_zone": tiz,
    }
    for k, v in out.items():
        log.info("risk[%s] %s: %d rows", mode, k, len(v))
    return out


def time_in_green(rolling: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Green-zone measure of H4 per (asset, model) at 99%: shares of zones (TIME_IN_ZONE_COLUMNS).

    - ``dev``: share of the rolling 250-observation windows ending in [dev_eval_start, dev_end] in each zone
      (``basis = 'rolling'``; ``windows`` = number of windows, ``n_obs`` = window length). The dev risk series
      starts at dev_eval_start, so every window lies inside the dev split.
    - ``holdout``: the Basel zone of ALL holdout observations at the actual N (SPEC §9 "Holdout summary uses
      each asset's actual N"; SPEC §0 evaluates the holdout separately), as a one-hot share
      (``basis = 'actual_n'``, ``windows = 1``, ``n_obs = N``). It equals ``zone_full`` at 99% of the holdout
      backtests. Rolling windows ending in the holdout are not used: they still contain development
      observations (on average ≈ 50% for SPX/EURUSD and ≈ 34% for crypto), and the windows lying entirely
      inside the holdout are 1–2 for SPX (none if N < 250). The holdout hits are read from the ``hit`` column of
      ``rolling``, which must be built over the whole series so every holdout date closes a window.
    """
    cols = TIME_IN_ZONE_COLUMNS
    if mode == "dev":
        start, end = pd.Timestamp(C.dev_eval_start()), pd.Timestamp(C.dev_end())
    elif mode == "holdout":
        start, end = pd.Timestamp(C.holdout_start()), pd.Timestamp(C.data_end())
    else:
        raise ValueError(f"unknown mode {mode!r}")
    if rolling.empty:
        return pd.DataFrame(columns=cols)
    r = rolling[(rolling["date"] >= start) & (rolling["date"] <= end)]
    if r.empty:
        return pd.DataFrame(columns=cols)
    if mode == "dev":
        tab = r.groupby(["asset", "model"])["zone"].value_counts(normalize=True).unstack(fill_value=0.0)
        for z in ZONES:
            if z not in tab:
                tab[z] = 0.0
        tab = tab[list(ZONES)].rename_axis(columns=None)
        tab["windows"] = r.groupby(["asset", "model"]).size()
        tab["n_obs"] = int(_risk_cfg()["traffic_window"])
        tab["basis"] = "rolling"
        return tab.reset_index()[cols]
    if "hit" not in r.columns:
        raise ValueError("holdout time-in-zone needs the 'hit' column of rolling_zones()")
    rows = []
    for (asset, model), g in r.groupby(["asset", "model"], sort=True):
        n, x = len(g), int(g["hit"].sum())
        z = zone(x, n, ALPHA_99)
        rows.append({"asset": asset, "model": model, **{k: float(k == z) for k in ZONES},
                     "windows": 1, "n_obs": n, "basis": "actual_n"})
    return pd.DataFrame(rows, columns=cols)

