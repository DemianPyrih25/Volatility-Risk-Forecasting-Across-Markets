"""Per-origin losses, common-date alignment and the QLIKE-ratio leaderboard (SPEC §8).

The leaderboard metric is ``mean QLIKE(model) / mean QLIKE(ref)`` per (asset, horizon, split) on the dates
where every compared model has a forecast; QLIKE levels are never compared across cells. IV benchmarks
exist only on the IV subsample, so they are left out unless requested (IV-inclusive run via ``models=``).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import polars as pl

from volrisk.evaluation.iv import IV_MODELS
from volrisk.evaluation.losses import mse, qlike

KEYS = ["asset", "horizon", "origin"]
LOSS_COLUMNS = ["asset", "horizon", "model", "origin", "split", "n_t", "y", "F", "qlike", "mse"]
LEADERBOARD_COLUMNS = ["asset", "horizon", "split", "model", "qlike", "mse", "qlike_ratio", "n"]


def _pandas(df: pd.DataFrame | pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas() if isinstance(df, pl.DataFrame) else df


def _origin_ns(s: pd.Series) -> pd.Series:
    # forecasts/targets may carry dates as datetime.date, datetime64[ms] or [ns]; join on one dtype
    return pd.to_datetime(s).astype("datetime64[ns]")


def losses_frame(
    forecasts: pd.DataFrame | pl.DataFrame, targets: pd.DataFrame | pl.DataFrame
) -> pd.DataFrame:
    """QLIKE and MSE per (asset, horizon, model, origin): forecasts inner-joined to targets on
    (asset, horizon, origin), ``split`` taken from the targets.

    Rows whose target is unobserved or whose window falls in the ``dropped`` split are not evaluated.
    ``origin`` is returned as ``datetime64[ns]``.
    """
    f = _pandas(forecasts)
    t = _pandas(targets)
    f = f[["asset", "horizon", "model", "origin", "n_t", "F"]].assign(origin=lambda d: _origin_ns(d["origin"]))
    t = t[["asset", "horizon", "origin", "n_t", "y", "split"]].assign(origin=lambda d: _origin_ns(d["origin"]))
    if t.duplicated(KEYS).any():
        raise ValueError("targets have duplicate (asset, horizon, origin) rows")
    if f.duplicated([*KEYS, "model"]).any():
        raise ValueError("forecasts have duplicate (asset, horizon, model, origin) rows")
    m = f.merge(t, on=KEYS, how="inner", suffixes=("", "_target"))
    if (m["n_t"] != m["n_t_target"]).any():
        bad = m.loc[m["n_t"] != m["n_t_target"], [*KEYS, "model", "n_t", "n_t_target"]].head()
        raise ValueError(f"forecast n_t disagrees with target n_t:\n{bad}")
    m = m[np.isfinite(m["y"].astype(float)) & (m["split"] != "dropped")].drop(columns="n_t_target")
    m = m.sort_values(["asset", "horizon", "model", "origin"], kind="stable").reset_index(drop=True)
    m["qlike"] = qlike(m["y"], m["F"])
    m["mse"] = mse(m["y"], m["F"])
    return m[LOSS_COLUMNS]


def common_dates(losses: pd.DataFrame, models: Sequence[str], loss: str = "qlike") -> pd.DataFrame:
    """Wide ``origin × model`` frame of ``loss`` for ONE (asset, horizon), keeping only origins at which
    every model in ``models`` has a loss. Columns are in the order of ``models``."""
    for col in ("asset", "horizon"):
        if col in losses and losses[col].nunique() > 1:
            raise ValueError(f"common_dates expects one {col}, got {sorted(losses[col].unique())}")
    models = list(models)
    missing = sorted(set(models) - set(losses["model"].unique()))
    if missing:
        raise KeyError(f"models without any loss rows: {missing}")
    sub = losses[losses["model"].isin(models)]
    wide = sub.pivot(index="origin", columns="model", values=loss)
    wide = wide[models].dropna(how="any").sort_index()
    wide.columns.name = None
    return wide


def leaderboard(
    losses: pd.DataFrame, ref: str = "HAR", models: Sequence[str] | None = None
) -> pd.DataFrame:
    """Mean QLIKE, mean MSE, QLIKE ratio vs ``ref`` and ``n`` per (asset, horizon, split, model), all on the
    common dates of the cell. ``models`` sets the comparison set; the default is every model in the cell
    except the IV benchmarks (they only exist on the IV subsample; list them explicitly for the IV-inclusive
    comparison). ``qlike_ratio`` is NaN when ``ref`` is not in the set."""
    rows = []
    for (asset, horizon, split), g in losses.groupby(["asset", "horizon", "split"], sort=True):
        present = list(dict.fromkeys(g["model"]))
        if models is None:
            ms = [m for m in present if m not in IV_MODELS]
        else:
            ms = [m for m in models if m in present]
        if not ms:
            continue
        q = common_dates(g, ms, "qlike")
        e = common_dates(g, ms, "mse").loc[q.index]
        mq, me = q.mean(), e.mean()
        ref_q = mq[ref] if ref in ms else np.nan
        for m in ms:
            rows.append(
                {
                    "asset": asset,
                    "horizon": horizon,
                    "split": split,
                    "model": m,
                    "qlike": mq[m],
                    "mse": me[m],
                    "qlike_ratio": mq[m] / ref_q,
                    "n": len(q),
                }
            )
    return pd.DataFrame(rows, columns=LEADERBOARD_COLUMNS)
