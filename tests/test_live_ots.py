"""OpenTimestamps layer of the live ledger (docs/LIVE_SPEC.md §4), fully offline.

Calendars are replaced by an in-memory fake that answers like the real ones (a timestamp from the submitted
commitment to a PendingAttestation; later a path to a BitcoinBlockHeaderAttestation). Block explorers are replaced
by a fake ``_http_get``. The Bitcoin byte-order check runs on the real genesis block header.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from opentimestamps.calendar import CommitmentNotFoundError
from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
from opentimestamps.core.op import OpAppend, OpPrepend, OpSHA256
from opentimestamps.core.timestamp import DetachedTimestampFile, Timestamp

from volrisk_live import ots

# Real Bitcoin genesis block (height 0): raw header, hash and merkle root as printed by every explorer.
GENESIS_HEADER = bytes.fromhex(
    "01000000" + "00" * 32
    + "3ba3edfd7a7b12b27ac72c3e67768f617fc81bc3888a51323a9fb8aa4b1e5e4a"
    + "29ab5f49" + "ffff001d" + "1dac2b7c")
GENESIS_HASH = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"
GENESIS_MERKLE = "4a5e1e4baab89f3a32518a88c31bc87f618f76673e2cc77ab2127b7afdeda33b"
GENESIS_TIME = 1231006505  # 2009-01-03T18:15:05Z

POOLS = {  # submission URL -> calendar URI of the pending attestation it returns (as the real pools do)
    "https://a.pool.opentimestamps.org": "https://alice.btc.calendar.opentimestamps.org",
    "https://b.pool.opentimestamps.org": "https://bob.btc.calendar.opentimestamps.org",
    "https://a.pool.eternitywall.com": "https://finney.calendar.eternitywall.com",
    "https://alice.btc.calendar.opentimestamps.org": "https://alice.btc.calendar.opentimestamps.org",
}
ALICE = "https://alice.btc.calendar.opentimestamps.org"
BOB = "https://bob.btc.calendar.opentimestamps.org"
HEIGHT = 915_000
HEIGHTS = {BOB: HEIGHT, ALICE: HEIGHT + 1, "https://finney.calendar.eternitywall.com": HEIGHT + 2}
REGTEST_POW_LIMIT = 0x7FFFFF << 232  # bits 0x207fffff: every other hash qualifies


class FakeCalendar:
    down: set = set()  # URLs/URIs that fail
    mined: set = set()  # (calendar URI, commitment) pairs already anchored in a block
    submitted: list = []
    queried: list = []

    def __init__(self, url, user_agent=None):
        self.url = url

    def submit(self, digest, timeout=None):
        if self.url in FakeCalendar.down:
            raise OSError(f"{self.url} unreachable")
        FakeCalendar.submitted.append((self.url, digest))
        ts = Timestamp(digest)
        leaf = ts.ops.add(OpPrepend(hashlib.sha256(self.url.encode()).digest()[:8])).ops.add(OpSHA256())
        leaf.attestations.add(PendingAttestation(POOLS[self.url]))
        return ts

    def get_timestamp(self, commitment, timeout=None):
        FakeCalendar.queried.append((self.url, commitment))
        if self.url in FakeCalendar.down:
            raise OSError(f"{self.url} unreachable")
        if (self.url, commitment) not in FakeCalendar.mined:
            raise CommitmentNotFoundError("Pending confirmation in Bitcoin blockchain")
        ts = Timestamp(commitment)
        root = ts.ops.add(OpAppend(b"\x11" * 32)).ops.add(OpSHA256()).ops.add(OpPrepend(b"\x22" * 32)) \
            .ops.add(OpSHA256())
        root.attestations.add(BitcoinBlockHeaderAttestation(HEIGHTS[self.url]))
        return ts


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    FakeCalendar.down, FakeCalendar.mined = set(), set()
    FakeCalendar.submitted, FakeCalendar.queried = [], []
    monkeypatch.setattr(ots, "RemoteCalendar", FakeCalendar)

    def no_http(url, timeout):
        raise AssertionError(f"unexpected HTTP call {url}")
    monkeypatch.setattr(ots, "_http_get", no_http)


@pytest.fixture
def payload(tmp_path) -> Path:
    p = tmp_path / "runs" / "20261002T060000Z.json"
    p.parent.mkdir()
    p.write_bytes(b'{"schema":"volrisk-live/1","x":1.5}')
    return p


def _mine(*paths: Path, uris=None) -> None:
    """Anchor the pending commitments of these proofs (all calendars, or ``uris``); the calendar then answers
    get_timestamp with a path to a Bitcoin block header attestation."""
    for path in paths:
        ts = ots.read_proof(ots.ots_path(path)).timestamp
        FakeCalendar.mined.update((a.uri, sub.msg) for sub in ots._walk(ts) for a in sub.attestations
                                  if isinstance(a, PendingAttestation) and (uris is None or a.uri in uris))


def _attested_msg(path: Path) -> bytes:
    msgs = [m for m, a in ots.read_proof(ots.ots_path(path)).timestamp.all_attestations()
            if isinstance(a, BitcoinBlockHeaderAttestation)]
    assert len(set(msgs)) == 1
    return msgs[0]


def _synthetic_block(merkle_internal: bytes, ntime: int = 1_790_000_000) -> tuple[bytes, str]:
    """An 80-byte header with the given merkle root and easy (regtest) bits; nonce ground until PoW holds."""
    for nonce in range(10_000):
        h = (b"\x00\x00\x00\x20" + b"\x33" * 32 + merkle_internal + ntime.to_bytes(4, "little")
             + (0x207FFFFF).to_bytes(4, "little") + nonce.to_bytes(4, "little"))
        d = hashlib.sha256(hashlib.sha256(h).digest()).digest()
        if int.from_bytes(d, "little") <= REGTEST_POW_LIMIT:
            return h, d[::-1].hex()
    raise AssertionError("no nonce found")


def _explorer(blocks: dict[int, tuple[bytes, str, str]], disagree: bool = False, reversed_root: bool = True):
    """Fake Esplora API: blocks[height] = (header, hash, merkle_root_hex as the explorer prints it)."""
    def get(url, timeout):
        for base in ots.EXPLORERS:
            if url.startswith(base):
                path = url[len(base):]
                break
        else:
            raise AssertionError(url)
        for height, (header, bhash, root_hex) in blocks.items():
            if path == f"/block-height/{height}":
                if disagree and base != ots.EXPLORERS[0]:
                    return "00" * 32
                return bhash
            if path == f"/block/{bhash}":
                return json.dumps({"id": bhash, "height": height, "merkle_root": root_hex,
                                   "timestamp": int.from_bytes(header[68:72], "little")})
            if path == f"/block/{bhash}/header":
                return header.hex()
        raise OSError(f"404 {url}")
    return get


# ---- stamp ----------------------------------------------------------------------------------------------------

def test_stamp_writes_standard_proof_and_round_trips(payload):
    r = ots.stamp(payload)
    assert r["status"] == "pending"
    assert sorted(r["calendars_ok"]) == sorted(ots.DEFAULT_CALENDARS)
    assert r["pending_calendars"] == sorted(set(POOLS.values()))  # 3 distinct calendars (alice twice)
    raw = ots.ots_path(payload).read_bytes()
    assert raw.startswith(DetachedTimestampFile.HEADER_MAGIC)  # standard detached-timestamp file
    proof = ots.read_proof(ots.ots_path(payload))
    assert isinstance(proof.file_hash_op, OpSHA256)
    assert proof.file_digest == hashlib.sha256(payload.read_bytes()).digest()
    assert ots.proof_bytes(proof) == raw  # deserialize -> serialize is byte-identical

    # Privacy: calendars only see sha256(file digest || 16-byte random nonce), never the digest itself.
    digest = proof.file_digest
    sent = {m for _, m in FakeCalendar.submitted}
    assert len(sent) == 1 and digest not in sent
    (op, sub), = proof.timestamp.ops.items()
    assert isinstance(op, OpAppend) and len(op[0]) == ots.NONCE_BYTES
    assert hashlib.sha256(digest + op[0]).digest() in sent


def test_stamp_nonce_differs_per_stamp(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_bytes(b"same")
    b.write_bytes(b"same")
    ots.stamp(a)
    ots.stamp(b)
    assert len({m for _, m in FakeCalendar.submitted}) == 2


def test_stamp_all_calendars_down_writes_nothing(payload):
    FakeCalendar.down = set(ots.DEFAULT_CALENDARS)
    r = ots.stamp(payload)
    assert r["status"] == "unstamped" and r["calendars_ok"] == []
    assert set(r["errors"]) == set(ots.DEFAULT_CALENDARS)
    assert not ots.ots_path(payload).exists()
    assert ots.status(payload)["status"] == "unstamped"


def test_partial_stamp_is_completed_by_a_later_stamp(payload):
    FakeCalendar.down = set(ots.DEFAULT_CALENDARS) - {"https://alice.btc.calendar.opentimestamps.org"}
    r = ots.stamp(payload)
    assert r["status"] == "partial" and r["pending_calendars"] == ["https://alice.btc.calendar.opentimestamps.org"]
    assert ots.ots_path(payload).exists()
    FakeCalendar.down = set()
    r2 = ots.stamp(payload)
    assert r2["status"] == "pending"
    proof = ots.read_proof(ots.ots_path(payload))
    assert len(proof.timestamp.ops) == 2  # two nonce branches, both kept
    assert ots.proof_bytes(proof) == ots.ots_path(payload).read_bytes()
    before = ots.ots_path(payload).read_bytes()
    r3 = ots.stamp(payload)  # already pending: no resubmission, file untouched
    assert r3["status"] == "pending" and r3.get("already") and ots.ots_path(payload).read_bytes() == before


def test_changed_file_is_never_restamped(payload):
    ots.stamp(payload)
    proof = ots.ots_path(payload).read_bytes()
    payload.write_bytes(payload.read_bytes().replace(b"1.5", b"9.5"))
    r = ots.stamp(payload)
    assert r["status"] == "invalid" and "changed" in r["reason"]
    assert ots.ots_path(payload).read_bytes() == proof
    assert ots.status(payload)["status"] == "invalid"
    v = ots.verify(payload)
    assert v["status"] == "failed" and v["digest_ok"] is False


# ---- upgrade --------------------------------------------------------------------------------------------------

def test_upgrade_pending_then_upgraded(payload):
    ots.stamp(payload)
    before = ots.ots_path(payload).read_bytes()
    r = ots.upgrade(payload)
    assert r["status"] == "pending" and r["block_height"] is None and not r["changed"]
    assert ots.ots_path(payload).read_bytes() == before  # nothing new: proof untouched
    assert len(FakeCalendar.queried) == 4  # one query per pending attestation (alice holds two commitments)

    _mine(payload)
    r = ots.upgrade(payload)
    assert r == {**r, "status": "upgraded", "block_height": HEIGHT, "changed": True}
    proof = ots.read_proof(ots.ots_path(payload))
    assert ots.bitcoin_heights(proof.timestamp) == [HEIGHT, HEIGHT + 1, HEIGHT + 2]
    assert proof.file_digest == hashlib.sha256(payload.read_bytes()).digest()
    assert ots.status(payload)["status"] == "upgraded"

    FakeCalendar.queried = []
    assert ots.upgrade(payload)["status"] == "upgraded"
    assert FakeCalendar.queried == []  # upgraded proofs need no network


def test_upgrade_contacts_only_whitelisted_calendars(payload):
    digest = hashlib.sha256(payload.read_bytes()).digest()
    det = DetachedTimestampFile(OpSHA256(), Timestamp(digest))
    leaf = det.timestamp.ops.add(OpAppend(b"\x00" * 16)).ops.add(OpSHA256())
    leaf.attestations.add(PendingAttestation("https://evil.example.com"))
    ots.write_proof(det, ots.ots_path(payload))
    r = ots.upgrade(payload)
    assert r["status"] == "pending" and r["errors"] == {"https://evil.example.com": "calendar not in whitelist"}
    assert FakeCalendar.queried == []


def test_calendar_errors_never_raise(payload, monkeypatch):
    ots.stamp(payload)
    FakeCalendar.down = set(POOLS.values())
    r = ots.upgrade(payload)
    assert r["status"] == "pending" and len(r["errors"]) == 3

    class Broken:
        def __init__(self, *a, **k):
            raise RuntimeError("boom")
    monkeypatch.setattr(ots, "RemoteCalendar", Broken)
    assert ots.upgrade(payload)["status"] == "pending"
    other = payload.with_name("x.json")
    other.write_bytes(b"x")
    assert ots.stamp(other)["status"] == "unstamped"


# ---- Bitcoin verification -------------------------------------------------------------------------------------

def test_merkle_byte_order_on_the_real_genesis_header():
    h = ots.check_header(GENESIS_HEADER, GENESIS_HASH)
    assert h["hash_ok"] and h["pow_ok"] and h["time"] == GENESIS_TIME
    # The header (and an attestation) carries the merkle root in internal order = explorer hex reversed.
    assert h["merkle_root"] == bytes.fromhex(GENESIS_MERKLE)[::-1]
    block = {"explorer": "x", "block_hash": GENESIS_HASH, "header": GENESIS_HEADER.hex(), "confirmed_by": [],
             "disagreed_by": [], "error": None,
             "info": {"id": GENESIS_HASH, "height": 0, "merkle_root": GENESIS_MERKLE, "timestamp": GENESIS_TIME}}
    good = ots.check_attestation(bytes.fromhex(GENESIS_MERKLE)[::-1], 0, block)
    assert good["merkle_ok"] and good["header_ok"] and good["block_time_utc"] == "2009-01-03T18:15:05Z"
    wrong_order = ots.check_attestation(bytes.fromhex(GENESIS_MERKLE), 0, block)
    assert wrong_order["merkle_ok"] is False
    assert ots.check_header(GENESIS_HEADER, GENESIS_HASH[::-1])["hash_ok"] is False
    tampered = GENESIS_HEADER[:68] + (GENESIS_TIME + 1).to_bytes(4, "little") + GENESIS_HEADER[72:]
    t = ots.check_header(tampered, GENESIS_HASH)
    assert not t["hash_ok"] and not t["pow_ok"]


def _upgraded_proof(payload):
    ots.stamp(payload)
    _mine(payload, uris={BOB})  # only bob has anchored so far: one Bitcoin attestation
    assert ots.upgrade(payload)["status"] == "upgraded"
    return _attested_msg(payload)


def test_verify_upgraded_proof_against_synthetic_block(payload, monkeypatch):
    msg = _upgraded_proof(payload)
    header, bhash = _synthetic_block(msg)
    monkeypatch.setattr(ots, "POW_LIMIT", REGTEST_POW_LIMIT)
    monkeypatch.setattr(ots, "_http_get", _explorer({HEIGHT: (header, bhash, msg[::-1].hex())}))
    v = ots.verify(payload)
    assert v["status"] == "verified" and v["digest_ok"]
    (a,) = v["attestations"]
    assert a["chain"] == "bitcoin" and a["height"] == HEIGHT and a["merkle_ok"] and a["header_ok"]
    assert a["block_hash"] == bhash and a["confirmed_by"] == [ots.EXPLORERS[1]]
    assert a["block_time_utc"] == "2026-09-21T14:13:20Z" == v["attested_utc"]


def test_verify_rejects_wrong_byte_order_and_bad_blocks(payload, monkeypatch):
    msg = _upgraded_proof(payload)
    header, bhash = _synthetic_block(msg)
    monkeypatch.setattr(ots, "POW_LIMIT", REGTEST_POW_LIMIT)
    # explorer JSON merkle root not reversed -> mismatch
    monkeypatch.setattr(ots, "_http_get", _explorer({HEIGHT: (header, bhash, msg.hex())}))
    v = ots.verify(payload)
    assert v["status"] == "failed" and v["attestations"][0]["merkle_ok"] is False
    # block whose merkle root is something else
    other, ohash = _synthetic_block(b"\x44" * 32)
    monkeypatch.setattr(ots, "_http_get", _explorer({HEIGHT: (other, ohash, (b"\x44" * 32)[::-1].hex())}))
    assert ots.verify(payload)["status"] == "failed"
    # second explorer reports another block at that height
    monkeypatch.setattr(ots, "_http_get", _explorer({HEIGHT: (header, bhash, msg[::-1].hex())}, disagree=True))
    v = ots.verify(payload)
    assert v["status"] == "failed" and v["attestations"][0]["header_ok"] is False
    # an easy-target header is rejected under the mainnet proof-of-work limit
    monkeypatch.setattr(ots, "POW_LIMIT", 0xFFFF << 208)
    monkeypatch.setattr(ots, "_http_get", _explorer({HEIGHT: (header, bhash, msg[::-1].hex())}))
    v = ots.verify(payload)
    assert v["status"] == "failed" and v["attestations"][0]["merkle_ok"] and not v["attestations"][0]["header_ok"]


def test_verify_offline_and_pending_statuses(payload, monkeypatch):
    assert ots.verify(payload)["status"] == "unstamped"
    ots.stamp(payload)
    v = ots.verify(payload)
    assert v["status"] == "pending" and v["digest_ok"] and len(v["pending_calendars"]) == 3
    _mine(payload)
    assert ots.upgrade(payload)["status"] == "upgraded"

    def down(url, timeout):
        raise OSError("offline")
    monkeypatch.setattr(ots, "_http_get", down)
    v = ots.verify(payload)
    assert v["status"] == "unverified" and v["attestations"][0]["merkle_ok"] is None
    ots.ots_path(payload).write_bytes(b"garbage")
    assert ots.verify(payload)["status"] == "invalid"


# ---- whole directory ------------------------------------------------------------------------------------------

def test_status_all_and_upgrade_all(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    names = ["20261001T060000Z.json", "20261002T060000Z.json", "20261003T060000Z.json"]
    for n in names:
        (runs / n).write_bytes(n.encode())
    ots.stamp(runs / names[0])
    (runs / (names[2] + ".ots")).write_bytes(b"not a proof")
    s = ots.status_all(runs)
    assert s["n"] == 3 and s["counts"] == {"pending": 1, "unstamped": 1, "invalid": 1}

    _mine(runs / names[0])  # the first payload is now anchored; the second gets stamped by upgrade_all
    u = ots.upgrade_all(runs)
    assert u["files"][names[0]] == {"status": "upgraded", "block_height": HEIGHT}
    assert u["files"][names[1]]["status"] == "pending"
    assert u["files"][names[2]]["status"] == "invalid"  # a broken proof is reported, never overwritten
    assert (runs / (names[2] + ".ots")).read_bytes() == b"not a proof"

    def down(url, timeout):
        raise OSError("offline")
    monkeypatch.setattr(ots, "_http_get", down)
    va = ots.verify_all(runs)
    assert va["counts"] == {"unverified": 1, "pending": 1, "invalid": 1}


def test_upgrade_all_covers_ledger_entry_files(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "20261002T060000Z.json").write_bytes(b'{"x":1}')
    (runs / "20261002T060000Z.entry").write_bytes(b'{"seq":0}')  # ledger entry file (ledger.ENTRY_SUFFIX)
    u = ots.upgrade_all(runs)
    assert u["n"] == 1 and u["counts"] == {"pending": 1}  # payloads stay at the top level (cli logs these)
    assert u["entries"]["n"] == 1 and u["entries"]["counts"] == {"pending": 1}
    assert ots.status_all(runs)["n"] == 1
    assert ots.status_all(runs, pattern=ots.ENTRY_GLOB)["counts"] == {"pending": 1}
    _mine(runs / "20261002T060000Z.entry")
    u = ots.upgrade_all(runs, entries=False)
    assert "entries" not in u and ots.status(runs / "20261002T060000Z.entry")["status"] == "pending"
    assert ots.upgrade_all(runs)["entries"]["files"]["20261002T060000Z.entry"]["status"] == "upgraded"


def test_forged_bitcoin_attestation_is_only_a_claim(payload, monkeypatch):
    """A Bitcoin attestation written straight into a proof (no calendar, no block) must never read as proof
    offline: the offline status is 'upgraded' (a claim); only verify() against the block can say 'verified'."""
    det = DetachedTimestampFile(OpSHA256(), Timestamp(hashlib.sha256(payload.read_bytes()).digest()))
    det.timestamp.attestations.add(BitcoinBlockHeaderAttestation(HEIGHT))
    ots.write_proof(det, ots.ots_path(payload))

    st = ots.status(payload)
    assert st["status"] == "upgraded" and st["block_heights"] == [HEIGHT]
    assert ots.status_all(payload.parent)["counts"] == {"upgraded": 1}
    assert "not yet verified" in ots.STATUS_TEXT["upgraded"] and "Bitcoin-attested" not in ots.STATUS_TEXT["upgraded"]
    assert [k for k, v in ots.STATUS_TEXT.items() if "Bitcoin-attested" in v] == ["verified"]
    assert ots.stamp(payload)["status"] == "upgraded"  # not resubmitted, but not called verified either

    # the block at that height has another merkle root: verify() rejects the forged attestation
    header, bhash = _synthetic_block(b"\x55" * 32)
    monkeypatch.setattr(ots, "POW_LIMIT", REGTEST_POW_LIMIT)
    monkeypatch.setattr(ots, "_http_get", _explorer({HEIGHT: (header, bhash, (b"\x55" * 32)[::-1].hex())}))
    v = ots.verify(payload)
    assert v["status"] == "failed" and v["attested_utc"] is None


def test_every_status_has_plain_wording(payload):
    seen = {ots.status(payload)["status"], ots.verify(payload)["status"]}  # unstamped
    ots.stamp(payload)
    seen |= {ots.status(payload)["status"], ots.verify(payload)["status"]}  # pending
    assert seen <= set(ots.STATUS_TEXT)
    assert {"unstamped", "partial", "pending", "upgraded", "invalid", "verified", "unverified", "failed"} == \
        set(ots.STATUS_TEXT)
