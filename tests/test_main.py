"""CLI wiring (``python -m volrisk``, SPEC §12). Every stage is stubbed: these tests never touch ``data/``."""

from __future__ import annotations

import pytest

from volrisk import __main__ as cli
from volrisk import bars, io, pipeline

STAGE_FUNCS = ("stage_download", "stage_bronze", "stage_bars", "stage_measures", "stage_forecast", "stage_evaluate",
               "stage_risk", "stage_quality", "stage_report", "stage_freeze", "stage_holdout")


@pytest.fixture
def calls(monkeypatch):
    """Replace every pipeline stage and ``bars.build_bars`` by recorders; return the call log."""
    log: list[tuple] = []
    for name in STAGE_FUNCS:
        monkeypatch.setattr(pipeline, name, lambda *a, _n=name, **k: log.append((_n, a, k)))

    def fake_build_bars(asset, *args, **kwargs):
        log.append(("build_bars", (asset, *args), kwargs))
        return {"asset": asset, "skipped": False}

    monkeypatch.setattr(bars, "build_bars", fake_build_bars)
    monkeypatch.setattr(io, "ensure_dirs", lambda: None)
    return log


def _names(log):
    return [c[0] for c in log]


def test_force_flag_is_accepted_and_reaches_build_bars(calls):
    # before the fix: argparse exited with 2 ("unrecognized arguments: --force")
    assert cli.main(["bars", "--force", "--assets", "BTC", "SPX"]) == 0
    assert calls == [("build_bars", ("BTC",), {"force": True}), ("build_bars", ("SPX",), {"force": True})]


def test_bars_without_force_keeps_the_cached_pipeline_path(calls):
    assert cli.main(["bars", "--assets", "ETH"]) == 0
    assert calls == [("stage_bars", (("ETH",),), {})]


def test_data_force_rebuilds_bars_for_every_asset(calls):
    assert cli.main(["data", "--force"]) == 0
    assert _names(calls) == ["stage_download", "stage_bronze", *["build_bars"] * 4, "stage_measures"]
    assert [c[1][0] for c in calls if c[0] == "build_bars"] == list(cli.C.ASSETS)
    assert all(c[2] == {"force": True} for c in calls if c[0] == "build_bars")


def test_force_is_a_logged_no_op_for_stages_without_a_cache(calls, caplog):
    with caplog.at_level("INFO", logger="volrisk"):
        assert cli.main(["forecast", "--force", "--workers", "3"]) == 0
    assert calls == [("stage_forecast", (), {"workers": 3})]
    assert "--force has no effect on stage 'forecast'" in caplog.text


def test_force_does_not_open_the_holdout(calls):
    with pytest.raises(SystemExit) as e:
        cli.main(["holdout", "--force"])
    assert e.value.code == 2
    assert "stage_holdout" not in _names(calls)


def test_all_never_freezes_or_opens_the_holdout(calls):
    assert cli.main(["all", "--force"]) == 0
    assert "stage_freeze" not in _names(calls)
    assert "stage_holdout" not in _names(calls)
