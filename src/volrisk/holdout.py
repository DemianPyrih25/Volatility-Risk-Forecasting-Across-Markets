"""Sealed holdout: freeze, seal and the one-time unlock (SPEC §11). Hashes only — no Git.

``freeze`` records the final specification (``config/frozen.yaml``) and SHA-256 hashes in ``SEALED.json``. It
hashes every file under ``src/volrisk/``, the config, the dev and holdout gold tables, the implied-vol table and the
development results. ``unlock`` re-verifies every hash and refuses a second opening unless a reason is given (logged
as a deviation). It then appends to ``holdout_log.jsonl`` and only after that unlocks the holdout loader for the
current process. A new seal after an opening needs a reason too, which is also recorded as a deviation.

Text files (``.py``, ``.yaml``) are hashed with CRLF normalised to LF. A Git line-ending conversion therefore does
not break the seal; it cannot change what Python or YAML reads.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from volrisk import config as C
from volrisk import io

SEALED = C.ROOT / "SEALED.json"
HOLDOUT_LOG = C.ROOT / "holdout_log.jsonl"
DEVIATIONS = C.ROOT / "docs" / "DEVIATIONS.md"
SRC = C.ROOT / "src" / "volrisk"
RESULTS_HOLDOUT = C.RESULTS / "holdout"  # where the holdout run writes (pipeline.results_dir(True))

_SKIP_DIRS = ("__pycache__",)  # bytecode that Python rewrites on every run; everything else is sealed
_TEXT_SUFFIXES = (".py", ".yaml", ".yml")


class SealError(RuntimeError):
    pass


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def text_sha(path: Path) -> str:
    """SHA-256 of a text file with CRLF normalised to LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _content_sha(path: Path) -> str:
    return text_sha(path) if path.suffix.lower() in _TEXT_SUFFIXES else file_sha(path)


def code_files(src: Path | None = None) -> dict[str, str]:
    """SHA-256 of every file under src/volrisk/ by relative path (SPEC §11); only ``__pycache__/`` is skipped."""
    src = src or SRC
    out = {}
    for p in src.rglob("*"):
        rel = p.relative_to(src).as_posix()
        if p.is_file() and not any(part in _SKIP_DIRS for part in rel.split("/")[:-1]):
            out[rel] = _content_sha(p)
    return dict(sorted(out.items()))


def code_sha(src: Path | None = None) -> str:
    """One SHA-256 over (relative path, file hash) of every file under src/volrisk/."""
    h = hashlib.sha256()
    for rel, sha in code_files(src).items():
        h.update(f"{rel}\0{sha}\n".encode())
    return h.hexdigest()


def _sealed_files() -> dict[str, Path]:
    return {
        "config": C.CONFIG_PATH,
        "frozen": C.FROZEN_PATH,
        "holdout_data": io.DAILY_HOLDOUT,
        "dev_data": io.DAILY_DEV,  # not listed in SPEC §11, but the holdout walk-forward starts from it
        "implied": io.IMPLIED,
        "dev_forecasts": io.FORECASTS,
        "dev_risk": io.RISK,
    }


def current_hashes() -> dict[str, str]:
    out = {"code": code_sha()}
    for k, p in _sealed_files().items():
        out[k] = _content_sha(p) if p.exists() else "MISSING"
    return out


def _holdout_results_exist() -> bool:
    return RESULTS_HOLDOUT.is_dir() and any(p.is_file() for p in RESULTS_HOLDOUT.rglob("*"))


def _opening_evidence() -> str:
    """Why the holdout counts as opened already ('' if it does not)."""
    n = len(openings())
    if n:
        return f"{HOLDOUT_LOG.name} records {n} opening(s)"
    if _holdout_results_exist():
        return f"{RESULTS_HOLDOUT} holds results of an earlier holdout run"
    return ""


def _log_deviation(text: str, utc: str) -> None:
    with open(DEVIATIONS, "a", encoding="utf-8") as f:
        f.write(f"\n- {utc}: {text}\n")


def freeze(frozen: dict, refreeze_reason: str | None = None) -> dict:
    """Write config/frozen.yaml and SEALED.json. Refuses to overwrite an existing seal.

    After the holdout was opened, a new seal is refused unless ``refreeze_reason`` is given; the re-freeze is then
    recorded in DEVIATIONS.md.
    """
    if SEALED.exists():
        raise SealError(f"{SEALED.name} already exists — the specification is frozen")
    opened = _opening_evidence()
    if opened and not refreeze_reason:
        raise SealError(
            f"the holdout was already opened ({opened}); a new seal after an opening needs "
            "holdout.freeze(..., refreeze_reason='...'), which is recorded as a deviation"
        )
    with open(C.FROZEN_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(frozen, f, sort_keys=False)
    hashes = current_hashes()
    missing = [k for k, v in hashes.items() if v == "MISSING"]
    if missing:
        C.FROZEN_PATH.unlink()
        raise SealError(f"cannot seal, missing artefacts: {missing}")
    seal = {"created_utc": _now(), "hashes": hashes, "code_files": code_files()}
    SEALED.write_text(json.dumps(seal, indent=2), encoding="utf-8")
    if opened:
        _log_deviation(f"specification re-frozen after the holdout was opened ({opened}). Reason: {refreeze_reason}",
                       seal["created_utc"])
    return seal


def _load_seal() -> dict:
    if not SEALED.exists():
        raise SealError("no SEALED.json — run `python -m volrisk freeze` first")
    return json.loads(SEALED.read_text(encoding="utf-8"))


def verify_seal() -> list[str]:
    """Names of sealed artefacts whose current hash differs from SEALED.json (or that only one side has)."""
    sealed = _load_seal()["hashes"]
    now = current_hashes()
    return [k for k in [*sealed, *(k for k in now if k not in sealed)] if sealed.get(k) != now.get(k)]


def changed_code_files() -> list[str]:
    """Files under src/volrisk/ that were added, removed or changed since the freeze."""
    sealed = _load_seal().get("code_files", {})
    now = code_files()
    return sorted(k for k in sealed.keys() | now.keys() if sealed.get(k) != now.get(k))


def openings() -> list[dict]:
    if not HOLDOUT_LOG.exists():
        return []
    return [json.loads(line) for line in HOLDOUT_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]


def unlock(rerun_reason: str | None = None) -> dict:
    """Verify the seal, log the opening and unlock the holdout loader for this process."""
    bad = verify_seal()
    if bad:
        detail = f" (changed files under src/volrisk/: {changed_code_files()})" if "code" in bad else ""
        raise SealError(f"seal mismatch for {bad}{detail}: code/config/data changed after the freeze")
    previous = openings()
    orphan = not previous and _holdout_results_exist()
    if not rerun_reason:
        if previous:
            raise SealError(
                f"the holdout was already opened at {previous[0]['utc']}; pass --rerun-reason to re-open "
                "(it will be recorded as a deviation)"
            )
        if orphan:
            raise SealError(
                f"{RESULTS_HOLDOUT} already holds holdout results but {HOLDOUT_LOG.name} has no entry; pass "
                "--rerun-reason to re-open (it will be recorded as a deviation)"
            )
    entry = {"utc": _now(), "n_previous": len(previous), "seal_sha": file_sha(SEALED), "hashes": current_hashes(),
             "rerun_reason": rerun_reason}
    with open(HOLDOUT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    if rerun_reason:
        note = "; earlier holdout results exist without a log entry" if orphan else ""
        _log_deviation(f"holdout re-opened (opening #{len(previous) + 1}{note}). Reason: {rerun_reason}", entry["utc"])
    io._unlock_for_this_process()
    return entry


def frozen() -> dict:
    with open(C.FROZEN_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
