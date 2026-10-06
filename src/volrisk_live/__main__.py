"""``uv run python -m volrisk_live {update|forecast|score|verify|stamp|dashboard|daily}`` (docs/LIVE_SPEC.md §8)."""

from __future__ import annotations

import sys

from volrisk_live.cli import main

if __name__ == "__main__":
    sys.exit(main(open_browser=True))
