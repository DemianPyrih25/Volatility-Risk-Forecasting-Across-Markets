"""Dev/holdout loaders and the holdout unlock (SPEC §5.3, §11) on synthetic parquet files in tmp_path.

The real ``data/holdout/`` is never read: every path the loaders and the seal use is redirected to tmp_path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest

from volrisk import config as C
from volrisk import holdout as H
from volrisk import io

H0 = C.holdout_start()
LAST_DEV = H0 - timedelta(days=1)


@pytest.fixture
def sealed(tmp_path: Path, monkeypatch) -> Path:
    """Synthetic dev/holdout/implied tables, a locked process and no unlock variable in the environment."""
    dev, hold, implied = tmp_path / "dev.parquet", tmp_path / "holdout.parquet", tmp_path / "implied.parquet"
    pl.DataFrame({"asset": ["SPX", "SPX", "BTC"], "session_date": [LAST_DEV - timedelta(days=1), LAST_DEV, LAST_DEV],
                  "tv": [1.0, 2.0, 3.0]}).write_parquet(dev)
    pl.DataFrame({"asset": ["SPX", "BTC"], "session_date": [H0, H0], "tv": [4.0, 5.0]}).write_parquet(hold)
    pl.DataFrame({"asset": ["SPX", "SPX", "SPX", "BTC"], "origin": [LAST_DEV - timedelta(days=1), LAST_DEV, H0, H0],
                  "iv": [10.0, 11.0, 12.0, 50.0], "iv_var_30d": [8.2, 9.9, 11.8, 205.5],
                  "source": ["VIX", "VIX", "VIX", "DVOL"]}).write_parquet(implied)
    monkeypatch.setattr(io, "DAILY_DEV", dev)
    monkeypatch.setattr(io, "DAILY_HOLDOUT", hold)
    monkeypatch.setattr(io, "IMPLIED", implied)
    monkeypatch.setattr(io, "_UNLOCKED", False)  # restored after the test, whatever the test unlocks
    monkeypatch.delenv(io._UNLOCK_ENV, raising=False)
    return tmp_path


def test_dev_loader_refuses_holdout_rows_in_gold(sealed: Path, monkeypatch):
    bad = sealed / "bad_dev.parquet"
    pl.DataFrame({"asset": ["SPX"], "session_date": [H0], "tv": [1.0]}).write_parquet(bad)
    monkeypatch.setattr(io, "DAILY_DEV", bad)
    with pytest.raises(io.HoldoutSealedError, match="split is broken"):
        io.load_daily()


def test_sealed_process_reads_dev_only(sealed: Path):
    assert not io.holdout_unlocked()
    df = io.load_daily()
    assert len(df) == 3 and df["session_date"].max().date() == LAST_DEV
    with pytest.raises(io.HoldoutSealedError):
        io.load_daily(include_holdout=True)


def test_environment_variable_does_not_unlock(sealed: Path, monkeypatch):
    """Setting the retired token by hand must not open the holdout (no seal check, no log entry)."""
    monkeypatch.setenv(io._UNLOCK_ENV, "1")
    assert not io.holdout_unlocked()
    with pytest.raises(io.HoldoutSealedError):
        io.load_daily(include_holdout=True)
    assert io.load_implied()["origin"].max().date() == LAST_DEV


def test_unlock_is_process_local(sealed: Path):
    io._unlock_for_this_process()
    assert io.holdout_unlocked()
    assert io._UNLOCK_ENV not in os.environ  # nothing a child process could inherit
    df = io.load_daily(include_holdout=True)
    assert len(df) == 5 and df["session_date"].max().date() == H0
    assert list(df["asset"]) == sorted(df["asset"])  # still sorted by (asset, session_date)
    child = subprocess.run([sys.executable, "-c", "from volrisk import io; print(io.holdout_unlocked())"],
                           capture_output=True, text=True, check=True)
    assert child.stdout.strip() == "False"


def test_load_implied_hides_holdout_iv_until_unlocked(sealed: Path):
    iv = io.load_implied()
    assert [d.date() for d in iv["origin"]] == [LAST_DEV - timedelta(days=1), LAST_DEV]  # last dev origin kept
    assert io.load_implied("BTC").empty
    io._unlock_for_this_process()
    iv = io.load_implied()
    assert len(iv) == 4 and list(iv["asset"]) == ["BTC", "SPX", "SPX", "SPX"]
    assert io.load_implied("BTC")["origin"].max().date() == H0


def _seal_in_tmp(tmp: Path, monkeypatch) -> None:
    """Redirect every artefact the seal hashes and logs to tmp, then seal the synthetic state."""
    for name in ("config.yaml", "frozen.yaml", "forecasts.parquet", "risk.parquet"):
        (tmp / name).write_text(name, encoding="utf-8")
    monkeypatch.setattr(C, "CONFIG_PATH", tmp / "config.yaml")
    monkeypatch.setattr(C, "FROZEN_PATH", tmp / "frozen.yaml")
    monkeypatch.setattr(io, "FORECASTS", tmp / "forecasts.parquet")
    monkeypatch.setattr(io, "RISK", tmp / "risk.parquet")
    monkeypatch.setattr(H, "code_sha", lambda src=None: "code")
    monkeypatch.setattr(H, "SEALED", tmp / "SEALED.json")
    monkeypatch.setattr(H, "HOLDOUT_LOG", tmp / "holdout_log.jsonl")
    monkeypatch.setattr(H, "DEVIATIONS", tmp / "DEVIATIONS.md")
    monkeypatch.setattr(H, "RESULTS_HOLDOUT", tmp / "results_holdout")  # never the project's real holdout results
    H.SEALED.write_text(json.dumps({"created_utc": "x", "hashes": H.current_hashes()}), encoding="utf-8")


def test_only_the_logged_seal_checked_unlock_opens_the_holdout(sealed: Path, monkeypatch):
    _seal_in_tmp(sealed, monkeypatch)
    (sealed / "forecasts.parquet").write_text("changed after the freeze", encoding="utf-8")
    with pytest.raises(H.SealError, match="seal mismatch"):
        H.unlock()
    assert not io.holdout_unlocked() and not H.HOLDOUT_LOG.exists()

    (sealed / "forecasts.parquet").write_text("forecasts.parquet", encoding="utf-8")
    entry = H.unlock()
    assert io.holdout_unlocked() and H.openings() == [entry]
    assert io.load_daily(include_holdout=True)["session_date"].max().date() == H0
