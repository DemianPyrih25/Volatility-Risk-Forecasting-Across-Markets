"""Hash-chained, append-only forward-test ledger (docs/LIVE_SPEC.md §4).

Files under ``forecasts/``:

- ``runs/<run_id>.json``: ``schema.canonical_json(payload)`` as UTF-8 bytes without a trailing newline (so a Git
  line-ending conversion cannot change it).
- ``ledger.jsonl``: one canonical-JSON entry per line, ``{seq, run_id, run_utc, payload, payload_sha256,
  prev_entry_sha256, entry_sha256}``, with ``entry_sha256 = sha256(canonical_json(entry without entry_sha256))`` and
  ``prev_entry_sha256`` the previous entry's hash (``"0"*64`` for the first).
- ``runs/<run_id>.entry``: exactly the bytes that ``entry_sha256`` hashes (``sha256(file) == entry_sha256``). It
  commits to its payload and, through ``prev_entry_sha256``, to every earlier entry and payload.
- ``runs/<run_id>.json.ots`` / ``runs/<run_id>.entry.ots``: OpenTimestamps proofs of both files (``ots.py``). They live
  next to the files, so stamping and upgrading never touch the chain.

What is detected, and how:

- ``verify_chain`` (offline): an edited payload; an edited, removed, inserted or reordered ledger line; an entry file
  that differs from its line; a proof that no longer matches its file (a payload or entry rewritten after it was
  stamped); a payload or entry file that no entry records (a removed last entry whose files were left behind).
- A run removed *with* all its files, and the later entries rewritten consistently (new seq / prev / hashes, new
  entry files, their old proofs deleted), passes ``verify_chain``. ``verify_anchors`` catches it (network): each
  rewritten entry file has to be stamped again, so its Bitcoin time falls long after its ``run_utc``. An entry is
  pinned by the earliest verified Bitcoin time of its own entry file or of any later one; a pin more than
  ``MAX_ANCHOR_LAG`` after the run is reported.
- Not detectable from these files: dropping the most recent run(s) with all their files before a later run links to
  them; the chain then just ends earlier. Only a copy of the head hash published elsewhere (e.g. the owner's Git push
  of ``forecasts/``) proves that such a run existed. A dropped day that later runs follow stays visible in
  ``verify_chain()['days_without_run']``.
- ``append`` refuses, on the project ledger, a ``run_utc`` that is not the machine's current time (more than
  ``MAX_CLOCK_SKEW`` ahead or ``MAX_RUN_AGE`` behind): a run is recorded when it is made, never post- or backdated.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from volrisk_live import ots
from volrisk_live import paths as P
from volrisk_live import schema as S

ENTRY_KEYS = ("seq", "run_id", "run_utc", "payload", "payload_sha256", "prev_entry_sha256", "entry_sha256")
ENTRY_SUFFIX = ".entry"  # runs/<run_id>.entry, outside the runs/*.json payload glob
RUN_ID_FORMAT = "%Y%m%dT%H%M%SZ"
_RUN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")
_LOCK_STALE = 600.0  # seconds after which a leftover lock file (crashed writer) is ignored

PROJECT_LEDGER = P.LEDGER  # the real forecasts/ledger.jsonl, captured at import (tests redirect paths.LEDGER later)
MAX_CLOCK_SKEW = timedelta(minutes=5)  # run_utc may lead the machine clock by this much
MAX_RUN_AGE = timedelta(hours=3)  # ... and lag it by this much (the walk-forward takes minutes, not hours)
MAX_ANCHOR_LAG = timedelta(hours=24)  # an entry must be pinned in Bitcoin within this time after its run
BLOCK_TIME_SLACK = timedelta(hours=2)  # Bitcoin block times may run up to ~2 h ahead of real time


class LedgerError(RuntimeError):
    pass


# ---- helpers --------------------------------------------------------------------------------------------------

def new_run_id(now_utc: datetime | None = None) -> str:
    """``YYYYMMDDTHHMMSSZ`` of ``now_utc`` (default: now), in UTC."""
    return _as_utc(now_utc or datetime.now(timezone.utc)).strftime(RUN_ID_FORMAT)


def _as_utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _wall_clock() -> datetime:
    """The machine's UTC clock (patched in tests)."""
    return datetime.now(timezone.utc)


def _check_run_id(run_id, run_utc) -> str | None:
    """Reason why (run_id, run_utc) is not a valid pair, or None."""
    if not isinstance(run_id, str) or not _RUN_ID_RE.match(run_id):
        return f"run_id {run_id!r} is not YYYYMMDDTHHMMSSZ"
    try:
        utc = _as_utc(datetime.fromisoformat(str(run_utc)))
    except ValueError:
        return f"run_utc {run_utc!r} is not an ISO time"
    if utc.strftime(RUN_ID_FORMAT) != run_id:
        return f"run_id {run_id} does not match run_utc {run_utc}"
    return None


def is_project_ledger() -> bool:
    """True when ``paths.LEDGER`` is the project's ``forecasts/ledger.jsonl`` (not a test sandbox)."""
    return Path(P.LEDGER).resolve() == Path(PROJECT_LEDGER).resolve()


def check_run_time(run_utc, now_utc: datetime | None = None) -> str | None:
    """Reason why ``run_utc`` is not the current time (``now_utc``, default the machine clock), or None."""
    utc = _as_utc(datetime.fromisoformat(str(run_utc)))
    now = _as_utc(now_utc or _wall_clock())
    clock = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    if utc > now + MAX_CLOCK_SKEW:
        return f"run_utc {run_utc} is in the future (machine clock {clock}); a run is never post-dated"
    if utc < now - MAX_RUN_AGE:
        return (f"run_utc {run_utc} is more than {MAX_RUN_AGE.total_seconds() / 3600:g} h before the machine clock "
                f"{clock}; a run is recorded when it is made, never backdated")
    return None


def payload_bytes(payload: dict) -> bytes:
    """The exact bytes stored for a payload: canonical JSON, UTF-8."""
    return S.canonical_json(payload).encode("utf-8")


def entry_bytes(entry: dict) -> bytes:
    """Bytes of ``runs/<run_id>.entry``: canonical JSON of the entry without ``entry_sha256`` (and without the
    ``stamp`` results that ``append`` returns), so ``sha256(entry_bytes(e)) == e['entry_sha256']``."""
    return S.canonical_json({k: v for k, v in entry.items() if k in ENTRY_KEYS and k != "entry_sha256"}).encode("utf-8")


def entry_hash(entry: dict) -> str:
    return hashlib.sha256(entry_bytes(entry)).hexdigest()


def payload_rel(run_id: str) -> str:
    return f"runs/{run_id}.json"


def _run_id_of(entry_or_run_id: dict | str) -> str:
    return entry_or_run_id["run_id"] if isinstance(entry_or_run_id, dict) else entry_or_run_id


def payload_path(entry_or_run_id: dict | str) -> Path:
    return P.RUNS / f"{_run_id_of(entry_or_run_id)}.json"


def entry_path(entry_or_run_id: dict | str) -> Path:
    return P.RUNS / f"{_run_id_of(entry_or_run_id)}{ENTRY_SUFFIX}"


def _replace(tmp: Path, target: Path, lock_timeout: float = 60.0) -> None:
    """os.replace, retried while another process (dashboard, editor) holds the target open on Windows."""
    deadline = time.monotonic() + lock_timeout
    while True:
        try:
            os.replace(tmp, target)
            return
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    _replace(tmp, path)


@contextmanager
def _lock(timeout: float = 60.0):
    """Exclusive writer lock (lock file created with O_EXCL); a lock older than _LOCK_STALE s is ignored."""
    P.LEDGER.parent.mkdir(parents=True, exist_ok=True)
    lock = P.LEDGER.with_name(P.LEDGER.name + ".lock")
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > _LOCK_STALE:
                    lock.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() > deadline:
                raise LedgerError(f"ledger is locked by another writer ({lock})") from None
            time.sleep(0.2)
    try:
        os.write(fd, f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}".encode())
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _lines() -> list[str]:
    """Ledger lines (LF or CRLF); the final newline ends the last entry, any other empty line is an error."""
    if not P.LEDGER.exists():
        return []
    raw = P.LEDGER.read_bytes().decode("utf-8").split("\n")
    if raw and raw[-1] == "":
        raw.pop()
    return [ln[:-1] if ln.endswith("\r") else ln for ln in raw]


# ---- reading --------------------------------------------------------------------------------------------------

def entries() -> list[dict]:
    """All ledger entries in file order (raises LedgerError on an unparsable line; see ``verify_chain``)."""
    out = []
    for i, line in enumerate(_lines()):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise LedgerError(f"ledger line {i} is not valid JSON: {e}") from None
    return out


def load_payload(entry: dict) -> dict:
    """The payload recorded by ``entry``; raises LedgerError if the file is missing or its hash differs."""
    path = payload_path(entry)
    if not path.exists():
        raise LedgerError(f"payload {path.name} is missing")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["payload_sha256"]:
        raise LedgerError(f"payload {path.name} does not match payload_sha256 of entry {entry.get('seq')}")
    return json.loads(data.decode("utf-8"))


def _check_entry(i: int, line: str, prev: str, prev_run_id: str | None) -> tuple[dict | None, str | None]:
    """(entry, None) if line ``i`` is a valid link after ``prev``; else (entry or None, reason)."""
    try:
        e = json.loads(line)
    except json.JSONDecodeError as err:
        return None, f"entry {i}: not valid JSON ({err.msg})"
    if not isinstance(e, dict) or sorted(e) != sorted(ENTRY_KEYS):
        return None, f"entry {i}: keys {sorted(e) if isinstance(e, dict) else type(e).__name__} != {list(ENTRY_KEYS)}"
    if e["entry_sha256"] != entry_hash(e):
        return e, f"entry {i}: entry_sha256 does not match the entry content (entry edited)"
    if line != S.canonical_json(e):
        return e, f"entry {i}: line is not canonical JSON (entry edited)"
    if e["seq"] != i:
        return e, f"entry {i}: seq is {e['seq']}, expected {i} (entry removed, inserted or reordered)"
    if e["prev_entry_sha256"] != prev:
        return e, f"entry {i}: prev_entry_sha256 does not link to entry {i - 1} (entry removed, inserted or reordered)"
    why = _check_run_id(e["run_id"], e["run_utc"])
    if why:
        return e, f"entry {i}: {why}"
    if prev_run_id is not None and e["run_id"] <= prev_run_id:
        return e, f"entry {i}: run_id {e['run_id']} not after {prev_run_id} (entries reordered)"
    if e["payload"] != payload_rel(e["run_id"]):
        return e, f"entry {i}: payload path {e['payload']!r} != {payload_rel(e['run_id'])!r}"
    path = payload_path(e)
    if not path.exists():
        return e, f"entry {i}: payload file {e['payload']} is missing"
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != e["payload_sha256"]:
        return e, f"entry {i}: payload file {e['payload']} does not match payload_sha256 (payload edited)"
    try:
        payload = json.loads(data.decode("utf-8"))
        S.validate_payload(payload)
    except (ValueError, TypeError, KeyError, AttributeError) as err:
        return e, f"entry {i}: payload is not a valid {S.SCHEMA} payload ({err})"
    if payload_bytes(payload) != data:
        return e, f"entry {i}: payload file is not canonical JSON"
    if (payload["run_id"], payload["run_utc"]) != (e["run_id"], e["run_utc"]):
        return e, f"entry {i}: payload run_id/run_utc differ from the entry"
    epath = entry_path(e)
    if not epath.exists():
        return e, f"entry {i}: entry file runs/{epath.name} is missing"
    if epath.read_bytes() != entry_bytes(e):
        return e, f"entry {i}: entry file runs/{epath.name} differs from the ledger line (entry rewritten)"
    for f, what in ((path, "payload"), (epath, "entry")):
        st = ots.status(f)
        if st["status"] == "invalid":
            return e, (f"entry {i}: the timestamp proof of the {what} file runs/{f.name} does not match it "
                       f"({st['reason']}; {what} rewritten after stamping)")
    return e, None


def _days_without_run(good: list[dict]) -> list[str]:
    """UTC dates between the first and the last run on which no run was recorded."""
    days = {date(int(e["run_id"][:4]), int(e["run_id"][4:6]), int(e["run_id"][6:8])) for e in good}
    if len(days) < 2:
        return []
    first, last = min(days), max(days)
    return [d.isoformat() for d in (first + timedelta(n) for n in range((last - first).days + 1)) if d not in days]


def _scan() -> tuple[list[dict], dict]:
    """Walk the chain; returns the valid prefix of entries and the verify_chain result."""
    lines = _lines()
    res = {"ok": True, "n": len(lines), "first_bad": None, "reason": None, "head": S.ZERO_HASH, "orphans": [],
           "days_without_run": []}
    good: list[dict] = []
    prev, prev_run_id = S.ZERO_HASH, None
    for i, line in enumerate(lines):
        e, why = _check_entry(i, line, prev, prev_run_id)
        if why:
            res.update(ok=False, first_bad=i, reason=why, head=prev, days_without_run=_days_without_run(good))
            return good, res
        good.append(e)
        prev, prev_run_id = e["entry_sha256"], e["run_id"]
    res.update(head=prev, days_without_run=_days_without_run(good))
    recorded = {p.name for e in good for p in (payload_path(e), entry_path(e))}
    files = (list(P.RUNS.glob("*.json")) + list(P.RUNS.glob(f"*{ENTRY_SUFFIX}"))) if P.RUNS.exists() else []
    orphans = sorted(p.name for p in files if p.name not in recorded)
    if orphans:
        res.update(ok=False, first_bad=len(lines), orphans=orphans,
                   reason=f"file(s) not recorded in the ledger (last entry removed?): {', '.join(orphans)}")
    return good, res


def verify_chain() -> dict:
    """Recompute every hash and link (offline): {ok, n, first_bad, reason, head, orphans, days_without_run}.

    ``first_bad`` is the 0-based line of the first broken entry (``n`` when a payload or entry file has no entry);
    ``head`` is the last valid entry hash; ``days_without_run`` lists the UTC dates between the first and the last
    valid run without a run. A chain rewritten consistently after a removed run passes this check: see
    ``verify_anchors``.
    """
    return _scan()[1]


def _parse_utc(text: str | None) -> datetime | None:
    return _as_utc(datetime.fromisoformat(text)) if text else None


def verify_anchors(now_utc: datetime | None = None, max_lag: timedelta = MAX_ANCHOR_LAG, timeout: float = 20) -> dict:
    """Bitcoin time at which each entry's place in the chain was fixed (network; run ``ots.upgrade_all`` first).

    Every entry file is checked with ``ots.verify`` (block header, proof of work, merkle root). An entry is pinned at
    the earliest verified Bitcoin time of its own entry file or any later one (a later entry file commits to it).
    Per entry ``status``: ``anchored`` (pinned within ``max_lag`` after ``run_utc``), ``pending`` (not pinned yet, run
    younger than ``max_lag``), ``late`` (pinned later or not at all: a rewrite of the chain from this entry on cannot be
    excluded, e.g. a removed earlier run, or the stamp failed at run time), ``failed`` (its proof contradicts the file
    or the chain, or the Bitcoin time precedes ``run_utc`` by more than ``BLOCK_TIME_SLACK``: post-dated).

    Returns {ok, chain_ok, n, counts, max_lag_hours, entries, note}; ``ok`` = chain intact and no ``late`` / ``failed``.
    """
    good, chain = _scan()
    now = _as_utc(now_utc or _wall_clock())
    cache: dict = {}
    rows = []
    for e in good:
        try:
            v = ots.verify(entry_path(e), timeout=timeout, _cache=cache)
        except Exception as err:  # noqa: BLE001 - reported per entry, never raised
            v = {"status": "error", "attested_utc": None, "reason": f"{type(err).__name__}: {err}"}
        rows.append({"seq": e["seq"], "run_id": e["run_id"], "run_utc": e["run_utc"], "proof": v.get("status"),
                     "attested_utc": v.get("attested_utc") if v.get("status") == "verified" else None,
                     "proof_reason": v.get("reason")})
    pin, pin_by = None, None
    for r in reversed(rows):  # pinned = earliest verified time of this entry file or any later one
        own = _parse_utc(r["attested_utc"])
        if own is not None and (pin is None or own <= pin):
            pin, pin_by = own, r["run_id"]
        run = _as_utc(datetime.fromisoformat(r["run_utc"]))
        r["pinned_utc"] = pin.strftime("%Y-%m-%dT%H:%M:%SZ") if pin else None
        r["pinned_by"] = pin_by
        r["lag_hours"] = round((pin - run).total_seconds() / 3600, 2) if pin else None
        if r["proof"] in ("failed", "invalid"):
            r["status"], r["reason"] = "failed", f"entry proof {r['proof']}: {r['proof_reason']}"
        elif pin is not None and pin < run - BLOCK_TIME_SLACK:
            r["status"], r["reason"] = "failed", "Bitcoin time precedes run_utc: run time post-dated"
        elif pin is not None and pin - run <= max_lag:
            r["status"], r["reason"] = "anchored", None
        elif pin is None and now - run <= max_lag:
            r["status"], r["reason"] = "pending", "no verified Bitcoin attestation yet"
        else:
            r["status"] = "late"
            r["reason"] = (f"first pinned {r['lag_hours']} h after the run" if pin else "not pinned in Bitcoin") + \
                f" (limit {max_lag.total_seconds() / 3600:g} h): a rewrite of the chain from here on is not excluded"
        r.pop("proof_reason")
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"ok": bool(chain["ok"]) and not counts.get("late") and not counts.get("failed"), "chain_ok": chain["ok"],
            "n": len(rows), "counts": counts, "max_lag_hours": max_lag.total_seconds() / 3600, "entries": rows,
            "note": "runs dropped from the end of the ledger together with all their files are not detectable here; "
                    "only a head hash published elsewhere (e.g. a Git push of forecasts/) proves they existed"}


# ---- writing --------------------------------------------------------------------------------------------------

def _write_new(path: Path, data: bytes, created: list[Path]) -> None:
    """Write ``path`` unless it already holds exactly ``data`` (left over from a crashed append)."""
    if path.exists():
        if path.read_bytes() != data:
            raise LedgerError(f"{path.name} exists with different content")
        return
    _atomic_write(path, data)
    created.append(path)


def append(payload: dict, stamp: bool = True, check_clock: bool | None = None) -> dict:
    """Record ``payload`` (schema ``volrisk-live/1``): write ``runs/<run_id>.json`` and ``runs/<run_id>.entry``,
    append the chained entry to ``ledger.jsonl`` (atomic replace under a writer lock), then, if ``stamp``,
    OpenTimestamps-stamp both files.

    Refuses to extend a chain that does not verify, a duplicate or non-increasing ``run_id``, or a payload / entry file
    that already exists with other content. ``check_clock``: ``run_utc`` must be the machine's current time
    (``check_run_time``); always enforced on the project ledger (``False`` is refused there), off by default for any
    other ledger (test sandboxes replaying fixed times).

    Returns the written entry plus ``stamp`` / ``entry_stamp`` (the ``ots.stamp`` results, or None); these keys are not
    part of the ledger. A failed stamp never fails the append: the file stays ``unstamped`` and ``ots.upgrade_all``
    retries it.
    """
    project = is_project_ledger()
    if check_clock is False and project:
        raise LedgerError("the run-time check cannot be switched off for the project ledger (forecasts/ledger.jsonl)")
    enforce = project if check_clock is None else check_clock
    S.validate_payload(payload)
    text = S.canonical_json(payload)
    clean = json.loads(text)
    run_id, run_utc = clean["run_id"], clean["run_utc"]
    why = _check_run_id(run_id, run_utc)
    if why:
        raise ValueError(why)
    data = text.encode("utf-8")
    path, epath = payload_path(run_id), entry_path(run_id)
    with _lock():
        if enforce:
            why = check_run_time(run_utc)
            if why:
                raise LedgerError(why)
        good, chain = _scan()
        if chain["first_bad"] is not None and chain["first_bad"] < chain["n"]:
            raise LedgerError(f"ledger chain is broken, refusing to append: {chain['reason']}")
        stray = [o for o in chain["orphans"] if o not in (path.name, epath.name)]
        if stray:
            raise LedgerError(f"ledger does not verify, refusing to append: file(s) without an entry: "
                              f"{', '.join(stray)}")
        if any(e["run_id"] == run_id for e in good):
            raise LedgerError(f"run_id {run_id} is already in the ledger")
        if good and run_id <= good[-1]["run_id"]:
            raise LedgerError(f"run_id {run_id} is not after the last entry {good[-1]['run_id']}")
        entry = {"seq": len(good), "run_id": run_id, "run_utc": run_utc, "payload": payload_rel(run_id),
                 "payload_sha256": hashlib.sha256(data).hexdigest(), "prev_entry_sha256": chain["head"]}
        entry["entry_sha256"] = entry_hash(entry)
        line = (S.canonical_json(entry) + "\n").encode("utf-8")
        created: list[Path] = []
        try:
            _write_new(path, data, created)
            _write_new(epath, entry_bytes(entry), created)
            old = P.LEDGER.read_bytes() if P.LEDGER.exists() else b""
            if old and not old.endswith(b"\n"):
                raise LedgerError("ledger.jsonl does not end with a newline")
            _atomic_write(P.LEDGER, old + line)
        except BaseException:
            for p in created:  # no entry -> no new files, so runs/ never holds an unrecorded payload or entry file
                p.unlink(missing_ok=True)
            raise
    results: dict[str, dict | None] = {"stamp": None, "entry_stamp": None}
    if stamp:
        for key, f in (("stamp", path), ("entry_stamp", epath)):
            try:
                results[key] = ots.stamp(f)
            except Exception as e:  # noqa: BLE001 - stamping is best effort; ots.stamp itself should not raise
                results[key] = {"status": "unstamped", "calendars_ok": [], "reason": f"{type(e).__name__}: {e}"}
    return {**entry, **results}


def head() -> dict | None:
    """The last entry, or None for an empty ledger."""
    es = entries()
    return es[-1] if es else None
