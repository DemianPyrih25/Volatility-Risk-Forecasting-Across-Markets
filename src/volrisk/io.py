"""Storage helpers and the dev/holdout loaders (SPEC §5.3, §11).

The development loader never touches ``data/holdout/``. Holdout rows can only be read through
``load_daily(include_holdout=True)`` in a process where ``volrisk.holdout.unlock`` has verified the seal and logged
the opening. The unlock is a process-local flag: it is never read from or written to the environment, so it cannot be
granted by setting a variable by hand and is not inherited by child processes. ``load_implied`` likewise hides
holdout-period implied vol (origins from ``holdout_start``) until the process is unlocked.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import polars as pl

from volrisk import config as C

DAILY_DEV = C.GOLD / "daily.parquet"
DAILY_HOLDOUT = C.HOLDOUT / "daily.parquet"
IMPLIED = C.GOLD / "implied.parquet"
FORECASTS = C.RESULTS / "forecasts.parquet"
RISK = C.RESULTS / "risk.parquet"

# Retired environment-variable token. It is deliberately ignored: an inheritable variable that anyone can set
# skipped the seal check and the holdout log. Kept only so that tests can assert that setting it has no effect.
_UNLOCK_ENV = "VOLRISK_HOLDOUT_UNLOCKED"

_UNLOCKED = False  # set only by volrisk.holdout.unlock (via _unlock_for_this_process) after the seal is verified


class HoldoutSealedError(RuntimeError):
    pass


def ensure_dirs() -> None:
    for p in (C.RAW, C.BRONZE, C.SILVER, C.GOLD, C.HOLDOUT, C.RESULTS, C.FIGURES, C.TABLES):
        p.mkdir(parents=True, exist_ok=True)


def write_parquet(df: pl.DataFrame | pd.DataFrame, path: Path, lock_timeout: float = 120.0) -> None:
    """Atomic parquet write (write to .part, then rename).

    On Windows the rename fails while another process (e.g. the dashboard) holds the target open, so the
    rename is retried for up to ``lock_timeout`` seconds before giving up (the .part file is kept).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    if isinstance(df, pl.DataFrame):
        df.write_parquet(tmp)
    else:
        df.to_parquet(tmp, index=False)
    deadline = time.monotonic() + lock_timeout
    while True:
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if time.monotonic() > deadline:
                raise PermissionError(f"{path} is locked by another process; new data kept in {tmp}") from None
            time.sleep(0.5)


def holdout_unlocked() -> bool:
    """True only in the process in which ``volrisk.holdout.unlock`` verified the seal and logged the opening."""
    return _UNLOCKED


def _unlock_for_this_process() -> None:
    """Only called by ``volrisk.holdout.unlock`` after the seal has been verified and the opening logged."""
    global _UNLOCKED
    _UNLOCKED = True


def load_daily(asset: str | None = None, include_holdout: bool = False) -> pd.DataFrame:
    """Gold daily measures (SPEC §5.3) as pandas, sorted by (asset, session_date).

    Development code must call this with ``include_holdout=False`` (the default).
    """
    df = pl.read_parquet(DAILY_DEV)
    if df.height and df["session_date"].max() >= C.holdout_start():
        raise HoldoutSealedError("dev gold table contains holdout sessions — split is broken")
    if include_holdout:
        if not holdout_unlocked():
            raise HoldoutSealedError(
                "holdout is sealed; it can only be opened by `python -m volrisk holdout --unlock-holdout`"
            )
        df = pl.concat([df, pl.read_parquet(DAILY_HOLDOUT)], how="vertical_relaxed")
    if asset is not None:
        df = df.filter(pl.col("asset") == asset)
    return df.sort(["asset", "session_date"]).to_pandas()


def load_implied(asset: str | None = None) -> pd.DataFrame:
    """Gold implied-vol table (SPEC §2.4) as pandas, sorted by (asset, origin).

    The table keeps holdout-period IV (it is part of the seal); rows with ``origin >= holdout_start`` are returned
    only in a process unlocked by ``volrisk.holdout``, so development code sees development-period IV only.
    """
    df = pl.read_parquet(IMPLIED)
    if not holdout_unlocked():
        df = df.filter(pl.col("origin") < C.holdout_start())
    if asset is not None:
        df = df.filter(pl.col("asset") == asset)
    return df.sort(["asset", "origin"]).to_pandas()
