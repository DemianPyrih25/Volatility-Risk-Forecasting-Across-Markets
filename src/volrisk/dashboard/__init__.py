"""Read-only Dash viewer of the pipeline results (SPEC §12): ``uv run python -m volrisk.dashboard``."""

from __future__ import annotations

__all__ = ["create_app"]


def create_app(*args, **kwargs):
    """Build the dashboard (see :func:`volrisk.dashboard.app.create_app`); Dash is imported lazily."""
    from volrisk.dashboard.app import create_app as _create_app

    return _create_app(*args, **kwargs)
