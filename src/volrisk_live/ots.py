"""OpenTimestamps proofs for ledger payloads (docs/LIVE_SPEC.md §4).

Built on the ``opentimestamps`` core library (the ``ots`` CLI does not run on Windows). The ``.ots`` files use the
standard detached-timestamp format, so https://opentimestamps.org and the reference client read them as well.

- ``stamp(path)``: SHA-256 of the file, a random 16-byte nonce appended and hashed again (the calendars learn
  nothing about the file; only this 32-byte commitment leaves the machine), submitted in parallel to public
  calendars; their answers are merged into ``path.ots``.
- ``upgrade(path)``: asks every pending calendar (whitelisted URIs only) for the Bitcoin attestation, which exists a
  few hours after stamping.
- ``verify(path)``: checks the file digest against the proof and every Bitcoin attestation against the block served
  by a public explorer (blockstream.info, fallback mempool.space): the raw 80-byte header hashes to the block hash at
  that height and carries valid proof of work, its merkle root equals the attested message, and the other explorer
  reports the same block hash. Returns the attested block time.

Offline status of a proof (``status``, ``status_all``), derived from the files only (never stored in the ledger, whose
entries are immutable): ``unstamped`` (no .ots), ``partial`` (fewer than ``min_ok`` distinct calendars), ``pending``
(waiting for Bitcoin), ``upgraded`` (the proof *claims* a Bitcoin attestation), ``invalid`` (unreadable proof, or the
file changed after stamping). None of these checks a Bitcoin block: anyone can write an attestation into a proof
file. Only ``verify`` / ``verify_all`` status ``verified`` means Bitcoin-attested (``STATUS_TEXT`` has the wording).
Network failures never raise out of stamp / upgrade / verify / upgrade_all / status_all / verify_all.

Files covered: the payloads ``runs/*.json`` and the ledger entry files ``runs/*.entry`` (``ledger.py``), whose proofs
pin each entry's place in the hash chain.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path

import requests
from opentimestamps.calendar import DEFAULT_CALENDAR_WHITELIST, RemoteCalendar
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
from opentimestamps.core.op import OpAppend, OpSHA256
from opentimestamps.core.serialize import StreamDeserializationContext, StreamSerializationContext
from opentimestamps.core.timestamp import DetachedTimestampFile, Timestamp

from volrisk_live import paths as P

DEFAULT_CALENDARS = (
    "https://a.pool.opentimestamps.org",
    "https://b.pool.opentimestamps.org",
    "https://a.pool.eternitywall.com",
    "https://alice.btc.calendar.opentimestamps.org",
)
EXPLORERS = ("https://blockstream.info/api", "https://mempool.space/api")  # Esplora API, same endpoints
CALENDAR_WHITELIST = DEFAULT_CALENDAR_WHITELIST  # pending-attestation URIs that upgrade() may contact
USER_AGENT = "volrisk-live (python-opentimestamps)"
NONCE_BYTES = 16
# Bitcoin mainnet proof-of-work limit (difficulty 1, bits 0x1d00ffff); a header claiming an easier target is invalid.
POW_LIMIT = 0xFFFF << 208
PAYLOAD_GLOB = "*.json"  # forecast payloads
ENTRY_GLOB = "*.entry"  # ledger entry files (ledger.ENTRY_SUFFIX)

# Plain wording for every status of ``status`` / ``stamp`` / ``upgrade`` (offline) and ``verify`` (network).
STATUS_TEXT = {
    "verified": "Bitcoin-attested (block header, proof of work and merkle root checked)",
    "upgraded": "claims a Bitcoin attestation, not yet verified against the block chain",
    "unverified": "claims a Bitcoin attestation, no block explorer reachable to check it",
    "pending": "submitted to the calendars, awaiting a Bitcoin attestation",
    "partial": "only one calendar answered",
    "unstamped": "no proof yet",
    "invalid": "proof unreadable or does not match the file",
    "failed": "contradicted by the file or by the Bitcoin block",
}


# ---- proof files ----------------------------------------------------------------------------------------------

def ots_path(path: Path | str) -> Path:
    return Path(str(path) + ".ots")


def file_digest(path: Path | str) -> bytes:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.digest()


def read_proof(ots_file: Path | str) -> DetachedTimestampFile:
    with open(ots_file, "rb") as f:
        return DetachedTimestampFile.deserialize(StreamDeserializationContext(f))


def proof_bytes(detached: DetachedTimestampFile) -> bytes:
    buf = BytesIO()
    detached.serialize(StreamSerializationContext(buf))
    return buf.getvalue()


def write_proof(detached: DetachedTimestampFile, ots_file: Path | str, lock_timeout: float = 60.0) -> None:
    """Atomic write (.part, then rename; the rename is retried while another process holds the target open)."""
    ots_file = Path(ots_file)
    tmp = ots_file.with_name(ots_file.name + ".part")
    tmp.write_bytes(proof_bytes(detached))
    deadline = time.monotonic() + lock_timeout
    while True:
        try:
            os.replace(tmp, ots_file)
            return
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)


def _walk(ts: Timestamp):
    yield ts
    for sub in ts.ops.values():
        yield from _walk(sub)


def pending_calendars(ts: Timestamp) -> list[str]:
    return sorted({a.uri for _, a in ts.all_attestations() if isinstance(a, PendingAttestation)})


def bitcoin_heights(ts: Timestamp) -> list[int]:
    return sorted({a.height for _, a in ts.all_attestations() if isinstance(a, BitcoinBlockHeaderAttestation)})


def _proof_status(ts: Timestamp, min_ok: int) -> str:
    if bitcoin_heights(ts):  # a claim only: verify() checks it against the block
        return "upgraded"
    n = len(pending_calendars(ts))
    return "pending" if n >= min_ok else ("partial" if n else "unstamped")


def _err(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:300]


def _load_checked(path: Path) -> tuple[DetachedTimestampFile | None, str | None]:
    """(proof, None) if path.ots exists, is readable and commits to the current file bytes; else (None, reason)."""
    out = ots_path(path)
    if not out.exists():
        return None, "unstamped"
    try:
        detached = read_proof(out)
    except Exception as e:  # noqa: BLE001 - any parse failure means an unusable proof
        return None, f"unreadable proof: {_err(e)}"
    if not isinstance(detached.file_hash_op, OpSHA256) or detached.file_digest != file_digest(path):
        return None, "file digest differs from the proof (file changed after stamping)"
    return detached, None


def status(path: Path | str, min_ok: int = 2) -> dict:
    """Offline status of one file's proof: status, pending calendars, claimed Bitcoin block heights.

    Never ``verified``: an ``upgraded`` proof only claims a Bitcoin attestation (see ``verify``)."""
    path = Path(path)
    try:
        detached, why = _load_checked(path)
    except OSError as e:
        detached, why = None, f"file unreadable: {_err(e)}"
    if detached is None:
        return {"status": "unstamped" if why == "unstamped" else "invalid", "reason": None if why == "unstamped"
                else why, "pending_calendars": [], "block_heights": []}
    ts = detached.timestamp
    return {"status": _proof_status(ts, min_ok), "reason": None, "pending_calendars": pending_calendars(ts),
            "block_heights": bitcoin_heights(ts)}


# ---- stamp / upgrade ------------------------------------------------------------------------------------------

def _submit_all(msg: bytes, calendars, timeout: float) -> tuple[list[tuple[str, Timestamp]], dict[str, str]]:
    """Submit ``msg`` to every calendar in parallel; returns the answers within ``timeout`` and the errors."""
    def one(url):
        return RemoteCalendar(url, user_agent=USER_AGENT).submit(msg, timeout=timeout)

    ok, errors = [], {}
    ex = ThreadPoolExecutor(max_workers=max(1, len(calendars)))
    try:
        futs = {ex.submit(one, url): url for url in calendars}
        done, _ = wait(futs, timeout=timeout + 2)
        for fut, url in futs.items():
            if fut not in done:
                errors[url] = "timeout"
            elif fut.exception() is not None:
                errors[url] = _err(fut.exception())
            else:
                ok.append((url, fut.result()))
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return ok, errors


def stamp(path: Path | str, calendars=DEFAULT_CALENDARS, min_ok: int = 2, timeout: float = 20) -> dict:
    """Timestamp ``path`` on public calendars and write ``path.ots`` (LIVE_SPEC §4).

    status: ``pending`` (>= min_ok distinct calendars hold the commitment), ``partial`` (fewer; the proof is
    written and a later call adds a fresh branch), ``upgraded``, ``unstamped`` (no calendar answered, nothing
    written) or ``invalid`` (an existing proof is unreadable or no longer matches the file; it is never overwritten).
    A file that already has a pending or upgraded proof is not resubmitted.
    """
    path = Path(path)
    out = ots_path(path)
    res = {"status": "unstamped", "calendars_ok": [], "errors": {}, "pending_calendars": [], "ots": str(out)}
    try:
        digest = file_digest(path)
    except OSError as e:
        return {**res, "status": "invalid", "reason": _err(e)}
    existing = None
    if out.exists():
        existing, why = _load_checked(path)
        if existing is None:
            return {**res, "status": "invalid", "reason": why}
        st = _proof_status(existing.timestamp, min_ok)
        if st in ("pending", "upgraded"):
            return {**res, "status": st, "pending_calendars": pending_calendars(existing.timestamp), "already": True}

    detached = DetachedTimestampFile(OpSHA256(), Timestamp(digest))
    # Same construction as the reference client for one file: digest -> append(nonce) -> sha256 = commitment.
    commitment = detached.timestamp.ops.add(OpAppend(os.urandom(NONCE_BYTES))).ops.add(OpSHA256())
    answers, errors = _submit_all(commitment.msg, list(calendars), timeout)
    ok = []
    for url, ts in answers:
        try:
            commitment.merge(ts)
            ok.append(url)
        except Exception as e:  # noqa: BLE001 - a malformed answer is just a failed calendar
            errors[url] = _err(e)
    res.update(calendars_ok=ok, errors=errors)
    if not ok:
        if existing is not None:
            return {**res, "status": _proof_status(existing.timestamp, min_ok),
                    "pending_calendars": pending_calendars(existing.timestamp)}
        return res
    if existing is not None:  # a partial proof: keep its branch and add the new one
        existing.timestamp.merge(detached.timestamp)
        detached = existing
    try:
        write_proof(detached, out)
    except OSError as e:  # report what is on disk, not what failed to be written
        return {**res, "status": status(path, min_ok)["status"], "reason": f"could not write {out.name}: {_err(e)}"}
    return {**res, "status": _proof_status(detached.timestamp, min_ok),
            "pending_calendars": pending_calendars(detached.timestamp)}


def upgrade(path: Path | str, timeout: float = 20) -> dict:
    """Fetch Bitcoin attestations for the pending parts of ``path.ots`` (rewritten only when it gained one).

    status: ``upgraded`` (with ``block_height``, the lowest claimed height; ``verify`` checks it) or ``pending``;
    ``unstamped`` / ``invalid`` when there is no usable proof.
    """
    path = Path(path)
    out = ots_path(path)
    res = {"status": "pending", "block_height": None, "changed": False, "errors": {}}
    try:
        detached, why = _load_checked(path)
    except OSError as e:
        return {**res, "status": "invalid", "reason": _err(e)}
    if detached is None:
        return {**res, "status": "unstamped" if why == "unstamped" else "invalid",
                "reason": None if why == "unstamped" else why}
    ts = detached.timestamp
    if bitcoin_heights(ts):
        return {**res, "status": "upgraded", "block_height": bitcoin_heights(ts)[0]}

    targets = [(sub, a) for sub in _walk(ts) for a in list(sub.attestations) if isinstance(a, PendingAttestation)]
    for sub, att in targets:
        if att.uri not in CALENDAR_WHITELIST:
            res["errors"][att.uri] = "calendar not in whitelist"
            continue
        try:
            answer = RemoteCalendar(att.uri, user_agent=USER_AGENT).get_timestamp(sub.msg, timeout=timeout)
            before = {a for _, a in sub.all_attestations()}
            sub.merge(answer)
            if {a for _, a in sub.all_attestations()} != before:
                res["changed"] = True
        except Exception as e:  # noqa: BLE001 - not yet in a block (404), offline, or a bad answer
            res["errors"][att.uri] = _err(e)
    if res["changed"]:
        try:
            write_proof(detached, out)
        except OSError as e:  # the file on disk is still the pending proof
            return {**res, "changed": False, "reason": f"could not write {out.name}: {_err(e)}"}
    heights = bitcoin_heights(ts)
    return {**res, "status": "upgraded" if heights else "pending", "block_height": heights[0] if heights else None}


# ---- Bitcoin verification -------------------------------------------------------------------------------------

def _http_get(url: str, timeout: float) -> str:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
    r.raise_for_status()
    return r.text


def _sha256d(b: bytes) -> bytes:
    return hashlib.sha256(hashlib.sha256(b).digest()).digest()


def _target(bits: int) -> int:
    exp, mant = bits >> 24, bits & 0x007FFFFF
    if bits & 0x00800000:  # negative compact target: invalid
        return 0
    return mant << (8 * (exp - 3)) if exp >= 3 else mant >> (8 * (3 - exp))


def check_header(header: bytes, block_hash: str) -> dict:
    """Parse an 80-byte Bitcoin block header and check it against the claimed block hash (display hex).

    Byte order: hashes are serialised little-endian ("internal" order) and printed reversed by explorers, so
    ``sha256d(header)[::-1].hex() == block_hash`` and the explorer's ``merkle_root`` is ``header[36:68][::-1]``.
    """
    if len(header) != 80:
        return {"hash_ok": False, "pow_ok": False, "merkle_root": None, "time": None}
    h = _sha256d(header)
    bits = int.from_bytes(header[72:76], "little")
    target = _target(bits)
    return {
        "hash_ok": h[::-1].hex() == block_hash.lower(),
        "pow_ok": 0 < target <= POW_LIMIT and int.from_bytes(h, "little") <= target,
        "merkle_root": header[36:68],  # internal byte order = what a BitcoinBlockHeaderAttestation commits to
        "time": int.from_bytes(header[68:72], "little"),
    }


def fetch_block(height: int, timeout: float = 20, explorers=EXPLORERS) -> dict:
    """Block hash, JSON summary and raw header at ``height`` from the first explorer that answers; the other
    explorers are asked for the hash at that height as a cross-check. Never raises (``error`` on failure)."""
    errors = []
    for i, base in enumerate(explorers):
        try:
            block_hash = _http_get(f"{base}/block-height/{height}", timeout).strip().lower()
            info = json.loads(_http_get(f"{base}/block/{block_hash}", timeout))
            header = bytes.fromhex(_http_get(f"{base}/block/{block_hash}/header", timeout).strip())
        except Exception as e:  # noqa: BLE001
            errors.append(f"{base}: {_err(e)}")
            continue
        agree, disagree = [], []
        for other in explorers[:i] + explorers[i + 1:]:
            try:
                (agree if _http_get(f"{other}/block-height/{height}", timeout).strip().lower() == block_hash
                 else disagree).append(other)
            except Exception:  # noqa: BLE001 - an unreachable second explorer is not evidence either way
                pass
        return {"explorer": base, "block_hash": block_hash, "info": info, "header": header.hex(),
                "confirmed_by": agree, "disagreed_by": disagree, "error": None}
    return {"explorer": None, "block_hash": None, "info": None, "header": None, "confirmed_by": [],
            "disagreed_by": [], "error": "; ".join(errors)}


def check_attestation(msg: bytes, height: int, block: dict) -> dict:
    """Pure check of one Bitcoin attestation (``msg`` = attested message, internal byte order) against a block
    fetched by ``fetch_block``. ``merkle_ok``: msg equals the block's merkle root (explorer JSON, byte-reversed, and
    the raw header); ``header_ok``: the header hashes to the block hash, has valid proof of work and matches the
    JSON height/time, and no explorer reports a different hash at that height."""
    out = {"chain": "bitcoin", "height": height, "block_hash": block.get("block_hash"), "block_time_utc": None,
           "merkle_ok": None, "header_ok": None, "explorer": block.get("explorer"),
           "confirmed_by": block.get("confirmed_by", []), "error": block.get("error")}
    if block.get("error") or block.get("header") is None:
        return out
    info = block["info"] or {}
    hdr = check_header(bytes.fromhex(block["header"]), block["block_hash"])
    json_root = bytes.fromhex(info["merkle_root"])[::-1] if info.get("merkle_root") else None  # printed big-endian
    out["merkle_ok"] = len(msg) == 32 and msg == hdr["merkle_root"] and json_root in (None, msg)
    out["header_ok"] = bool(hdr["hash_ok"] and hdr["pow_ok"] and info.get("height") == height
                            and info.get("timestamp") == hdr["time"] and not block.get("disagreed_by"))
    if hdr["time"] is not None:
        out["block_time_utc"] = datetime.fromtimestamp(hdr["time"], tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return out


def verify(path: Path | str, timeout: float = 20, explorers=EXPLORERS, _cache: dict | None = None) -> dict:
    """Verify ``path.ots`` against ``path`` and the Bitcoin chain (read-only; run ``upgrade`` first).

    status: ``verified`` (digest ok and >= 1 Bitcoin attestation checked against the block header),
    ``pending`` (only calendar attestations so far), ``unverified`` (Bitcoin attestation present but no explorer
    reachable), ``failed`` (file digest or every Bitcoin attestation contradicts the proof/chain),
    ``unstamped`` or ``invalid``. ``attested_utc``: the earliest verified block time.
    """
    path = Path(path)
    out = ots_path(path)
    res = {"file": path.name, "digest_ok": False, "status": "unstamped", "attestations": [],
           "pending_calendars": [], "attested_utc": None, "reason": None}
    if not path.exists():
        return {**res, "status": "invalid", "reason": "file missing"}
    if not out.exists():
        return res
    try:
        detached = read_proof(out)
    except Exception as e:  # noqa: BLE001
        return {**res, "status": "invalid", "reason": f"unreadable proof: {_err(e)}"}
    ts = detached.timestamp
    res["pending_calendars"] = pending_calendars(ts)
    res["digest_ok"] = isinstance(detached.file_hash_op, OpSHA256) and detached.file_digest == file_digest(path)
    if not res["digest_ok"]:
        return {**res, "status": "failed", "reason": "file digest differs from the proof"}
    cache = {} if _cache is None else _cache
    for msg, att in sorted(ts.all_attestations(), key=lambda x: getattr(x[1], "height", -1)):
        if not isinstance(att, BitcoinBlockHeaderAttestation):
            continue
        if att.height not in cache or cache[att.height].get("error"):
            cache[att.height] = fetch_block(att.height, timeout, explorers)
        res["attestations"].append(check_attestation(msg, att.height, cache[att.height]))
    good = [a for a in res["attestations"] if a["merkle_ok"] and a["header_ok"]]
    bad = [a for a in res["attestations"] if a["merkle_ok"] is False or a["header_ok"] is False]
    if good:
        res["status"] = "verified"
        res["attested_utc"] = min(a["block_time_utc"] for a in good)
    elif bad:
        res.update(status="failed", reason="Bitcoin attestation does not match the block header")
    elif res["attestations"]:
        res.update(status="unverified", reason="no block explorer reachable")
    else:
        res["status"] = "pending"
    return res


# ---- whole runs/ directory ------------------------------------------------------------------------------------

def _files(runs_dir: Path | None, pattern: str = PAYLOAD_GLOB) -> list[Path]:
    runs_dir = Path(runs_dir) if runs_dir is not None else P.RUNS
    return sorted(runs_dir.glob(pattern)) if runs_dir.exists() else []


def _summary(files: dict) -> dict:
    return {"n": len(files), "counts": dict(Counter(f["status"] for f in files.values())), "files": files}


def status_all(runs_dir: Path | None = None, min_ok: int = 2, pattern: str = PAYLOAD_GLOB) -> dict:
    """Offline status of every payload's proof (``pattern=ENTRY_GLOB``: of every ledger entry file):
    {n, counts, files: {name: status dict}}. ``upgraded`` is a claim, not a verification (see ``verify_all``)."""
    return _summary({p.name: status(p, min_ok) for p in _files(runs_dir, pattern)})


def _upgrade_files(paths: list[Path], min_ok: int, timeout: float, stamp_missing: bool, calendars) -> dict:
    files = {}
    for p in paths:
        try:
            st = status(p, min_ok)["status"]
            r = {"status": st}
            if st in ("unstamped", "partial") and stamp_missing:
                r = stamp(p, calendars=calendars, min_ok=min_ok, timeout=timeout)
            if r["status"] in ("pending", "partial"):
                up = upgrade(p, timeout=timeout)
                if up["status"] == "upgraded":
                    r = {**r, **up}
            files[p.name] = {"status": r["status"], "block_height": r.get("block_height")}
        except Exception as e:  # noqa: BLE001 - one bad file never stops the others
            files[p.name] = {"status": "error", "block_height": None, "reason": _err(e)}
    return _summary(files)


def upgrade_all(runs_dir: Path | None = None, min_ok: int = 2, timeout: float = 20, stamp_missing: bool = True,
                calendars=DEFAULT_CALENDARS, entries: bool = True) -> dict:
    """Retry missing/partial stamps and upgrade pending ones (called by ``daily``); never raises.

    Payloads at the top level {n, counts, files}; with ``entries`` the ledger entry files too, under ``entries``."""
    out = _upgrade_files(_files(runs_dir, PAYLOAD_GLOB), min_ok, timeout, stamp_missing, calendars)
    if entries:
        out["entries"] = _upgrade_files(_files(runs_dir, ENTRY_GLOB), min_ok, timeout, stamp_missing, calendars)
    return out


def verify_all(runs_dir: Path | None = None, timeout: float = 20, explorers=EXPLORERS,
               pattern: str = PAYLOAD_GLOB) -> dict:
    """``verify`` for every payload (``pattern=ENTRY_GLOB``: every ledger entry file; block lookups shared):
    {n, counts, files: {name: verify dict}}. Only status ``verified`` means Bitcoin-attested."""
    cache: dict = {}
    files = {}
    for p in _files(runs_dir, pattern):
        try:
            files[p.name] = verify(p, timeout=timeout, explorers=explorers, _cache=cache)
        except Exception as e:  # noqa: BLE001
            files[p.name] = {"file": p.name, "status": "error", "reason": _err(e), "attestations": []}
    return _summary(files)
