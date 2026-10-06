"""Command line: ``uv run python -m volrisk <stage> [options]`` (SPEC §12)."""

from __future__ import annotations

import argparse
import logging
import sys

from volrisk import config as C
from volrisk import io, pipeline

log = logging.getLogger("volrisk")

STAGES = ("download", "bronze", "bars", "measures", "forecast", "evaluate", "risk", "quality", "report", "freeze",
          "holdout", "data", "all")

# Stages whose outputs are cached by a config-hash build marker (SPEC §12; see DEVIATIONS). Every other stage
# recomputes on each run, so ``--force`` changes nothing there. Downloads never re-fetch files that are already
# ``ok`` in the manifest (SPEC §2.1, raw data is immutable), with or without ``--force``.
CACHED_STAGES = ("bars",)


def _stage_bars(assets: tuple[str, ...], force: bool) -> None:
    """``pipeline.stage_bars``; with ``force`` every asset is rebuilt even when its build marker is current."""
    if not force:
        pipeline.stage_bars(assets)
        return
    from volrisk import bars

    for a in assets:
        log.info("bars %s (forced): %s", a, bars.build_bars(a, force=True))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="volrisk", description=__doc__)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("--assets", nargs="*", default=list(C.ASSETS), choices=list(C.ASSETS))
    p.add_argument("--workers", type=int, default=10, help="processes for the forecast stage")
    p.add_argument("--force", action="store_true",
                   help="recompute even if cached outputs are current for the config hash (SPEC §12); only the "
                        "bars stage caches, every other stage always recomputes")
    p.add_argument("--unlock-holdout", action="store_true", help="required by the one-time holdout run")
    p.add_argument("--rerun-reason", default=None, help="re-open the holdout (logged as a deviation)")
    p.add_argument("-v", "--verbose", action="store_true")
    a = p.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if a.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    io.ensure_dirs()
    assets = tuple(a.assets)
    s = a.stage
    if a.force and s not in (*CACHED_STAGES, "data", "all"):
        log.info("--force has no effect on stage %r: only %s cache outputs, the other stages always recompute",
                 s, "/".join(CACHED_STAGES))

    if s in ("download", "data", "all"):
        pipeline.stage_download(assets)
    if s in ("bronze", "data", "all"):
        pipeline.stage_bronze(assets)
    if s in ("bars", "data", "all"):
        _stage_bars(assets, force=a.force)
    if s in ("measures", "data", "all"):
        pipeline.stage_measures()
    if s in ("forecast", "all"):
        pipeline.stage_forecast(workers=a.workers)
    if s in ("evaluate", "all"):
        pipeline.stage_evaluate()
    if s in ("risk", "all"):
        pipeline.stage_risk()
    if s in ("quality", "all"):
        pipeline.stage_quality(assets)
    if s in ("report", "all"):
        pipeline.stage_report()
    if s == "freeze":
        pipeline.stage_freeze(workers=a.workers)
    if s == "holdout":
        if not a.unlock_holdout:
            p.error("the holdout run requires --unlock-holdout")
        pipeline.stage_holdout(rerun_reason=a.rerun_reason, workers=a.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
