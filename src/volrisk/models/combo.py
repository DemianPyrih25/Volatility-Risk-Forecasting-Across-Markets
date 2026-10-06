"""Forecast combination COMBO (SPEC §7 model 11; members fixed by ``models.combo_members`` in the config).

COMBO is the equal-weight arithmetic mean of the members' cumulative variance forecasts ``F`` per
(asset, horizon, origin), formed only where **every** member has a forecast. Variances are averaged — never
volatilities or quantiles (SPEC §9).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from volrisk import config as C
from volrisk.models.base import FORECAST_COLUMNS

KEYS = ["asset", "horizon", "origin"]


def combo_members() -> list[str]:
    return [str(m) for m in C.load()["models"]["combo_members"]]


def combine(forecasts: pd.DataFrame, members: Sequence[str] | None = None, name: str = "COMBO") -> pd.DataFrame:
    """Equal-weight mean of ``F`` across ``members`` on the (asset, horizon, origin) keys they all cover.

    ``forecasts`` holds FORECAST_COLUMNS (other models are ignored). Members must agree on ``n_t`` (and on
    ``split`` when that column is present, in which case it is carried through). Returns FORECAST_COLUMNS
    (+ ``split``) with ``model = name``, sorted by the keys.
    """
    members = combo_members() if members is None else [str(m) for m in members]
    if not members or len(set(members)) != len(members):
        raise ValueError(f"combine: members must be non-empty and unique, got {members}")
    extra = ["split"] if "split" in forecasts.columns else []
    cols = FORECAST_COLUMNS + extra
    sub = forecasts.loc[forecasts["model"].isin(members), cols]
    sub = sub[np.isfinite(sub["F"].to_numpy(dtype=float))]
    if (sub["F"] <= 0).any():
        raise ValueError("combine: member forecasts must be > 0")
    if sub.duplicated(KEYS + ["model"]).any():
        raise ValueError("combine: duplicate (asset, horizon, origin, model) rows")
    if sub.empty:
        return sub.iloc[0:0].assign(model=name, n_t=sub["n_t"].astype("int64"))[cols].reset_index(drop=True)

    wide = sub.pivot(index=KEYS, columns="model", values=["F", "n_t", *extra])
    F = wide["F"].reindex(columns=members)
    full = F.notna().all(axis=1).to_numpy()
    F = F[full]
    n_t = wide["n_t"].reindex(columns=members)[full]
    if (n_t.nunique(axis=1) != 1).any():
        bad = n_t[n_t.nunique(axis=1) != 1].head()
        raise ValueError(f"combine: members disagree on n_t\n{bad}")
    out = pd.DataFrame(index=F.index)
    out["model"] = name
    out["n_t"] = n_t[members[0]].astype("int64")
    out["F"] = F.to_numpy(dtype=float).sum(axis=1) / len(members)
    if extra:
        sp = wide["split"].reindex(columns=members)[full]
        if (sp.nunique(axis=1) != 1).any():
            raise ValueError("combine: members disagree on split")
        out["split"] = sp[members[0]]
    out = out.reset_index().sort_values(KEYS, kind="stable").reset_index(drop=True)
    return out[cols]
