"""Runtime context for live runs (docs/LIVE_SPEC.md §0).

``live_end`` moves the frozen package's data end date forward **in memory only**; every frozen module reads it
through ``volrisk.config.data_end()``, so calendars, builders and split rules apply unchanged. No frozen file is
modified, and the original function is restored on exit.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date

from volrisk import config as C
from volrisk import holdout, io


class HoldoutNotOpenedError(RuntimeError):
    pass


@contextmanager
def live_end(end: date):
    """Temporarily make ``volrisk.config.data_end()`` return ``end`` (must not precede the frozen end)."""
    frozen = C.data_end
    if end < frozen():
        raise ValueError(f"live end {end} precedes the frozen data end {frozen()}")
    C.data_end = lambda: end  # type: ignore[assignment]
    try:
        yield end
    finally:
        C.data_end = frozen  # type: ignore[assignment]


def frozen_data_end() -> date:
    """The sealed data end date (2026-09-30), whatever override is active."""
    return C._d(C.load()["dates"]["data_end"])


def require_opened_holdout() -> dict:
    """The holdout must already have been opened once (logged); returns the first opening entry.

    Live code reads holdout-period data as history *after* that logged opening. It never calls
    ``holdout.unlock`` (which would log a new opening).
    """
    opened = holdout.openings()
    if not opened:
        raise HoldoutNotOpenedError("the holdout has not been opened yet; live forecasting needs the frozen models "
                                    "to have been evaluated on it first (python -m volrisk holdout --unlock-holdout)")
    return opened[0]


@contextmanager
def history_access():
    """Allow frozen loaders to return holdout-period rows in this process, after checking the logged opening."""
    require_opened_holdout()
    before = io._UNLOCKED
    io._UNLOCKED = True
    try:
        yield
    finally:
        io._UNLOCKED = before
