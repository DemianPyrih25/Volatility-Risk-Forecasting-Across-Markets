"""Seal mechanics of volrisk.holdout (SPEC §11) on a synthetic sandbox.

Every path the module writes or hashes (source tree, config, SEALED.json, holdout_log.jsonl, DEVIATIONS.md, the
sealed data files and data/results/holdout/) is redirected to tmp_path. The real holdout files are never opened.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from volrisk import config as C
from volrisk import holdout as H
from volrisk import io

SOURCES = {
    "__init__.py": "",
    "models/__init__.py": "",
    "models/har.py": "def fit(x):\n    return x\n",
    "report.py": "TITLE = 'results'\n",
    "quality.py": "LIMIT = 0.25\n",
    "dashboard/__init__.py": "",
    "dashboard/app.py": "PORT = 8050\n",
}


def _write_tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode())
    return root


def _append(p: Path, text: bytes = b"X = 1\n") -> None:
    p.write_bytes(p.read_bytes() + text)


def _to_crlf(p: Path) -> None:
    p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    src = _write_tree(tmp_path / "src" / "volrisk", SOURCES)
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "har.cpython-313.pyc").write_bytes(b"\x00bytecode")
    cfg = tmp_path / "config" / "config.yaml"
    cfg.parent.mkdir()
    cfg.write_bytes(b"seed: 20261002\nwindow: 1000\n")
    data = {k: tmp_path / "data" / f"{k}.parquet" for k in
            ("holdout_data", "dev_data", "implied", "dev_forecasts", "dev_risk")}
    for k, p in data.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"PAR1\n" + k.encode() + b"\nPAR1")
    docs = tmp_path / "docs" / "DEVIATIONS.md"
    docs.parent.mkdir()
    docs.write_text("# Deviations log\n", encoding="utf-8")
    unlocked: list[int] = []

    monkeypatch.setattr(H, "SRC", src)
    monkeypatch.setattr(H, "SEALED", tmp_path / "SEALED.json")
    monkeypatch.setattr(H, "HOLDOUT_LOG", tmp_path / "holdout_log.jsonl")
    monkeypatch.setattr(H, "DEVIATIONS", docs)
    monkeypatch.setattr(H, "RESULTS_HOLDOUT", tmp_path / "data" / "results" / "holdout")
    monkeypatch.setattr(C, "FROZEN_PATH", tmp_path / "config" / "frozen.yaml")
    monkeypatch.setattr(H, "_sealed_files", lambda: {"config": cfg, "frozen": C.FROZEN_PATH, **data})
    monkeypatch.setattr(io, "_unlock_for_this_process", lambda: unlocked.append(1))
    return {"root": tmp_path, "src": src, "config": cfg, "data": data, "unlocked": unlocked}


FROZEN = {"forecast_models": ["HAR", "GJR"], "seed": 20261002}


# ------------------------------------------------------------------------------------------------ code hash
@pytest.mark.parametrize("rel", ["report.py", "quality.py", "dashboard/app.py", "models/har.py"])
def test_editing_any_source_file_changes_the_code_hash(tmp_path, rel):
    """SPEC §11 seals every file under src/volrisk/; presentation modules are not exempt."""
    src = _write_tree(tmp_path / "volrisk", SOURCES)
    before = H.code_sha(src)
    _append(src / rel)
    assert H.code_sha(src) != before


@pytest.mark.parametrize("rel", ["models/report.py", "models/dashboard/__init__.py", "models/params.yaml",
                                 "models/har.cp313-win_amd64.pyd", "notes.txt"])
def test_adding_any_file_changes_the_code_hash(tmp_path, rel):
    """New files count too, whatever their name or type (no name-based or *.py-only filter)."""
    src = _write_tree(tmp_path / "volrisk", SOURCES)
    before = H.code_sha(src)
    _write_tree(src, {rel: "X = 1\n"})
    assert H.code_sha(src) != before
    assert rel in H.code_files(src)


def test_pycache_is_not_sealed(tmp_path):
    src = _write_tree(tmp_path / "volrisk", SOURCES)
    before = H.code_sha(src)
    _write_tree(src, {"__pycache__/har.cpython-313.pyc": "a", "models/__pycache__/har.cpython-313.pyc": "b"})
    assert H.code_sha(src) == before


def test_line_endings_do_not_change_the_code_hash(tmp_path):
    """A Git autocrlf checkout rewrites LF as CRLF; Python reads both the same, so the seal must not break."""
    src = _write_tree(tmp_path / "volrisk", SOURCES)
    before = H.code_sha(src)
    _to_crlf(src / "models" / "har.py")
    assert H.code_sha(src) == before
    _append(src / "models" / "har.py", b"\r\n")  # a real edit is still detected
    assert H.code_sha(src) != before


def test_code_files_cover_the_real_package():
    """Every file of the installed package outside __pycache__ is hashed, presentation modules included."""
    expected = set()
    for dirpath, dirnames, filenames in os.walk(H.SRC):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        expected |= {(Path(dirpath) / f).relative_to(H.SRC).as_posix() for f in filenames}
    got = H.code_files()
    assert expected <= set(got)
    assert {"report.py", "quality.py", "holdout.py", "dashboard/app.py"} <= set(got)
    assert not any("__pycache__" in k for k in got)


# ------------------------------------------------------------------------------------------------ sealed files
def test_dev_gold_table_is_sealed():
    """The holdout walk-forward starts from data/gold/daily.parquet; its last session is invisible to the
    dev-forecast reproduction check, so the file itself must be sealed."""
    assert H._sealed_files()["dev_data"] == io.DAILY_DEV
    assert H._sealed_files()["holdout_data"] == io.DAILY_HOLDOUT


def test_results_holdout_path_matches_the_pipeline():
    from volrisk import pipeline

    assert H.RESULTS_HOLDOUT == pipeline.results_dir(True)


def test_edit_to_dev_gold_table_breaks_the_seal(sandbox):
    H.freeze(FROZEN)
    assert H.verify_seal() == []
    _append(sandbox["data"]["dev_data"], b"tampered")
    assert H.verify_seal() == ["dev_data"]
    with pytest.raises(H.SealError, match="dev_data"):
        H.unlock()
    assert not H.HOLDOUT_LOG.exists() and not sandbox["unlocked"]


def test_artefact_missing_from_an_older_seal_is_reported(sandbox):
    H.freeze(FROZEN)
    seal = json.loads(H.SEALED.read_text(encoding="utf-8"))
    del seal["hashes"]["dev_data"]
    H.SEALED.write_text(json.dumps(seal), encoding="utf-8")
    assert H.verify_seal() == ["dev_data"]


@pytest.mark.parametrize("rel", ["report.py", "quality.py", "dashboard/app.py"])
def test_presentation_edit_after_freeze_blocks_unlock(sandbox, rel):
    H.freeze(FROZEN)
    _append(sandbox["src"] / rel)
    assert H.verify_seal() == ["code"]
    assert H.changed_code_files() == [rel]
    with pytest.raises(H.SealError, match=rel.replace(".", r"\.")):
        H.unlock()
    assert not H.HOLDOUT_LOG.exists() and not sandbox["unlocked"]


def test_crlf_conversion_of_text_files_keeps_the_seal(sandbox):
    H.freeze(FROZEN)
    _to_crlf(sandbox["src"] / "models" / "har.py")
    _to_crlf(sandbox["config"])
    _to_crlf(C.FROZEN_PATH)
    assert H.verify_seal() == []
    _to_crlf(sandbox["data"]["implied"])  # binary files are hashed byte for byte
    assert H.verify_seal() == ["implied"]


# ------------------------------------------------------------------------------------------------ openings
def test_first_opening_logs_the_seal_hash_and_unlocks(sandbox):
    H.freeze(FROZEN)
    entry = H.unlock()
    assert sandbox["unlocked"] == [1]
    assert entry["n_previous"] == 0 and entry["rerun_reason"] is None
    assert entry["seal_sha"] == H.file_sha(H.SEALED)
    assert H.openings() == [entry]
    with pytest.raises(H.SealError, match="already opened"):
        H.unlock()
    entry2 = H.unlock(rerun_reason="crash in the evaluation stage")
    assert entry2["n_previous"] == 1
    assert "holdout re-opened (opening #2). Reason: crash in the evaluation stage" in H.DEVIATIONS.read_text("utf-8")


def test_freeze_after_an_opening_needs_a_reason(sandbox):
    H.freeze(FROZEN)
    H.unlock()
    frozen_before = C.FROZEN_PATH.read_bytes()
    H.SEALED.unlink()
    with pytest.raises(H.SealError, match="already opened"):
        H.freeze({**FROZEN, "forecast_models": ["HAR"]})
    assert C.FROZEN_PATH.read_bytes() == frozen_before and not H.SEALED.exists()
    H.freeze({**FROZEN, "forecast_models": ["HAR"]}, refreeze_reason="bug fix in the risk stage")
    assert H.SEALED.exists()
    text = H.DEVIATIONS.read_text("utf-8")
    assert "re-frozen after the holdout was opened" in text and "bug fix in the risk stage" in text
    with pytest.raises(H.SealError, match="already opened"):  # the re-opening itself still needs its own reason
        H.unlock()


def test_removed_log_is_caught_by_existing_holdout_results(sandbox):
    """Deleting SEALED.json and holdout_log.jsonl must not allow a silent 'first' opening."""
    H.RESULTS_HOLDOUT.mkdir(parents=True)  # an empty directory is not evidence of an opening
    H.freeze(FROZEN)
    (H.RESULTS_HOLDOUT / "forecasts.parquet").write_bytes(b"PAR1")
    with pytest.raises(H.SealError, match="no entry"):
        H.unlock()
    assert not sandbox["unlocked"]
    H.SEALED.unlink()
    with pytest.raises(H.SealError, match="earlier holdout run"):
        H.freeze(FROZEN)
    H.freeze(FROZEN, refreeze_reason="seal file lost")
    entry = H.unlock(rerun_reason="log file lost")
    assert sandbox["unlocked"] == [1] and entry["n_previous"] == 0
    assert "earlier holdout results exist without a log entry" in H.DEVIATIONS.read_text("utf-8")


def test_freeze_refuses_existing_seal_and_missing_artefacts(sandbox):
    sandbox["data"]["dev_risk"].unlink()
    with pytest.raises(H.SealError, match="dev_risk"):
        H.freeze(FROZEN)
    assert not C.FROZEN_PATH.exists() and not H.SEALED.exists()
    sandbox["data"]["dev_risk"].write_bytes(b"PAR1")
    H.freeze(FROZEN)
    with pytest.raises(H.SealError, match="already exists"):
        H.freeze(FROZEN)
    assert H.frozen() == FROZEN
