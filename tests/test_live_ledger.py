"""Hash-chained forward-test ledger (docs/LIVE_SPEC.md §4) on a tmp forecasts/ directory, offline.

Covers the exact entry/hash format, canonical payload bytes, the entry files that commit each stamp to the chain,
tamper detection (edited payload, edited entry, removed / reordered / appended-out-of-order entries, a removed last
entry, a removed run with a consistently re-hashed tail), the Bitcoin anchor check (``verify_anchors``, block
verification faked), the run-time guard of ``append`` and that stamping never changes the chain. The real
``forecasts/`` directory is never touched; calendars are faked.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from volrisk_live import ledger as L
from volrisk_live import ots
from volrisk_live import paths as P
from volrisk_live import schema as S

T0 = datetime(2026, 10, 2, 6, 0, 0, tzinfo=timezone.utc)


def make_payload(run_utc: datetime = T0, f: float = 4.25) -> dict:
    return {
        "schema": S.SCHEMA,
        "run_id": run_utc.strftime("%Y%m%dT%H%M%SZ"),
        "run_utc": run_utc.isoformat(),
        "frozen": {"code_sha": "ab" * 32, "seal_ok": True, "sealed_utc": "2026-07-01T00:00:00+00:00",
                   "holdout_opened_utc": "2026-10-02T05:00:00+00:00"},
        "live_code_sha": "cd" * 32,
        "data": {"BTC": {"last_session": date(2026, 10, 1), "n_sessions": np.int64(2100), "rows_sha256": "ef" * 32}},
        "forecasts": [{"asset": "BTC", "horizon": "1d", "model": "HAR", "origin": date(2026, 10, 1),
                       "window_first": date(2026, 10, 2), "window_last": date(2026, 10, 2), "n_t": np.int64(1),
                       "F": np.float64(f), "vol_ann": float("nan")}],
        "risk": [{"asset": "BTC", "model": "FHS", "date": pd.Timestamp("2026-10-02"), "sigma": 2.1, "var99": 5.0,
                  "var975": 4.1, "es975": 5.3}],
        "implied": [{"asset": "BTC", "origin": date(2026, 10, 1), "iv": 45.0, "iv_var_30d": 1.2, "source": "dvol"}],
        "checks": {"gold_matches_sealed": True, "reproduced": 1234, "note": "Prüfung ✓"},
    }


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "FORECASTS", tmp_path / "forecasts")
    monkeypatch.setattr(P, "RUNS", tmp_path / "forecasts" / "runs")
    monkeypatch.setattr(P, "LEDGER", tmp_path / "forecasts" / "ledger.jsonl")

    class NoNetwork:
        def __init__(self, *a, **k):
            raise AssertionError("network call in an offline test")
    monkeypatch.setattr(ots, "RemoteCalendar", NoNetwork)
    return tmp_path


@pytest.fixture
def calendars(monkeypatch):
    """Fake OpenTimestamps calendars: every submission gets a pending attestation (two distinct calendars)."""
    from opentimestamps.core.notary import PendingAttestation
    from opentimestamps.core.op import OpSHA256
    from opentimestamps.core.timestamp import Timestamp

    class Cal:
        def __init__(self, url, user_agent=None):
            self.url = url

        def submit(self, digest, timeout=None):
            ts = Timestamp(digest)
            ts.ops.add(OpSHA256()).attestations.add(PendingAttestation("https://alice.btc.calendar.opentimestamps.org"
                                                                       if "a." in self.url else
                                                                       "https://bob.btc.calendar.opentimestamps.org"))
            return ts
    monkeypatch.setattr(ots, "RemoteCalendar", Cal)


def _append_n(n: int) -> list[dict]:
    return [L.append(make_payload(T0 + timedelta(days=i), f=1.0 + i), stamp=False) for i in range(n)]


def _lines() -> list[str]:
    return P.LEDGER.read_text(encoding="utf-8").splitlines()


def _write_lines(lines: list[str]) -> None:
    P.LEDGER.write_bytes(("\n".join(lines) + "\n").encode("utf-8") if lines else b"")


def _names() -> list[str]:
    return sorted(p.name for p in P.RUNS.iterdir())


# ---- format ---------------------------------------------------------------------------------------------------

def test_canonical_payload_bytes_are_stable():
    p = make_payload()
    shuffled = {k: p[k] for k in reversed(list(p))}
    b = L.payload_bytes(p)
    assert b == L.payload_bytes(shuffled)  # key order does not matter
    text = b.decode("utf-8")
    assert "\n" not in text and "\r" not in text and not text.endswith(" ")
    assert '"vol_ann":null' in text and '"F":4.25' in text and '"date":"2026-10-02T00:00:00"' in text
    assert "Prüfung ✓" in text  # UTF-8, not \\u escapes
    reloaded = json.loads(text)
    assert L.payload_bytes(reloaded) == b  # canonical(load(canonical(x))) == canonical(x)

    e = L.append(p, stamp=False)
    path = L.payload_path(e)
    assert path.read_bytes() == b  # exact bytes on disk (binary write: no CRLF on Windows)
    assert e["payload_sha256"] == hashlib.sha256(b).hexdigest()
    assert L.load_payload(e) == reloaded


def test_entry_format_and_chain_match_the_spec():
    es = _append_n(3)
    assert [e["seq"] for e in es] == [0, 1, 2]
    assert es[0]["prev_entry_sha256"] == "0" * 64
    assert [e["prev_entry_sha256"] for e in es[1:]] == [e["entry_sha256"] for e in es[:-1]]
    on_disk = L.entries()
    assert on_disk == [{k: v for k, v in e.items() if k not in ("stamp", "entry_stamp")} for e in es]
    for e in on_disk:
        assert list(sorted(e)) == sorted(L.ENTRY_KEYS)
        assert e["payload"] == f"runs/{e['run_id']}.json" and e["run_id"] == e["run_utc"][:10].replace("-", "") + \
            "T" + e["run_utc"][11:19].replace(":", "") + "Z"
        # independent recomputation of the documented formula
        body = json.dumps({k: v for k, v in e.items() if k != "entry_sha256"}, sort_keys=True,
                          separators=(",", ":"), ensure_ascii=False)
        assert e["entry_sha256"] == hashlib.sha256(body.encode("utf-8")).hexdigest()
        # the entry file holds exactly those bytes: sha256(runs/<run_id>.entry) == entry_sha256
        assert L.entry_path(e).read_bytes() == body.encode("utf-8")
        assert hashlib.sha256(L.entry_path(e).read_bytes()).hexdigest() == e["entry_sha256"]
    assert ots.ENTRY_GLOB == "*" + L.ENTRY_SUFFIX
    v = L.verify_chain()
    assert v == {"ok": True, "n": 3, "first_bad": None, "reason": None, "head": es[-1]["entry_sha256"],
                 "orphans": [], "days_without_run": []}
    assert L.head()["seq"] == 2


def test_empty_ledger_verifies():
    assert L.entries() == [] and L.head() is None
    assert L.verify_chain()["ok"] and L.verify_chain()["n"] == 0


# ---- tamper detection -----------------------------------------------------------------------------------------

def test_edited_payload_is_detected():
    es = _append_n(3)
    path = L.payload_path(es[1])
    path.write_bytes(path.read_bytes().replace(b'"F":2.0', b'"F":1.9'))
    v = L.verify_chain()
    assert not v["ok"] and v["first_bad"] == 1 and "payload edited" in v["reason"]
    assert v["head"] == es[0]["entry_sha256"]  # the last valid entry
    with pytest.raises(L.LedgerError):
        L.load_payload(es[1])
    with pytest.raises(L.LedgerError, match="broken"):
        L.append(make_payload(T0 + timedelta(days=5)), stamp=False)


def test_deleted_payload_is_detected():
    es = _append_n(2)
    L.payload_path(es[0]).unlink()
    v = L.verify_chain()
    assert v["first_bad"] == 0 and "missing" in v["reason"]


def test_edited_entry_is_detected():
    _append_n(3)
    lines = _lines()
    e = json.loads(lines[1])
    e["run_utc"] = e["run_utc"].replace("06:00:00", "05:00:00")  # backdate, hash not recomputed
    _write_lines([lines[0], S.canonical_json(e), lines[2]])
    v = L.verify_chain()
    assert not v["ok"] and v["first_bad"] == 1 and "entry edited" in v["reason"]


def test_edited_and_rehashed_entry_is_caught_by_its_entry_file_and_the_next_link():
    es = _append_n(3)
    lines = _lines()
    e = json.loads(lines[1])
    p = json.loads(L.payload_path(es[1]).read_bytes())
    p["forecasts"][0]["F"] = 0.5  # forge payload and entry consistently
    data = L.payload_bytes(p)
    L.payload_path(es[1]).write_bytes(data)
    e["payload_sha256"] = hashlib.sha256(data).hexdigest()
    e["entry_sha256"] = L.entry_hash(e)
    _write_lines([lines[0], S.canonical_json(e), lines[2]])
    v = L.verify_chain()
    assert v["first_bad"] == 1 and "entry file" in v["reason"] and "differs" in v["reason"]
    L.entry_path(e).write_bytes(L.entry_bytes(e))  # forge the entry file too
    v = L.verify_chain()
    assert v["first_bad"] == 2 and "prev_entry_sha256" in v["reason"]


def test_entry_file_missing_edited_or_orphaned_is_detected():
    es = _append_n(2)
    epath = L.entry_path(es[0])
    good = epath.read_bytes()
    epath.unlink()
    v = L.verify_chain()
    assert v["first_bad"] == 0 and "entry file" in v["reason"] and "missing" in v["reason"]
    epath.write_bytes(good.replace(b'"seq":0', b'"seq":7'))
    assert "differs" in L.verify_chain()["reason"]
    epath.write_bytes(good)
    assert L.verify_chain()["ok"]
    stray = P.RUNS / f"20261001T060000Z{L.ENTRY_SUFFIX}"  # an entry file whose ledger line was removed
    stray.write_bytes(b"{}")
    v = L.verify_chain()
    assert not v["ok"] and v["orphans"] == [stray.name] and v["first_bad"] == 2


def test_removed_middle_entry_is_detected():
    _append_n(4)
    lines = _lines()
    _write_lines(lines[:1] + lines[2:])
    v = L.verify_chain()
    assert v["first_bad"] == 1 and "removed, inserted or reordered" in v["reason"]


def test_reordered_entries_are_detected():
    _append_n(3)
    a, b, c = _lines()
    _write_lines([a, c, b])
    v = L.verify_chain()
    assert v["first_bad"] == 1 and "seq" in v["reason"]


def test_removed_last_entry_is_detected_by_its_orphan_files():
    es = _append_n(3)
    _write_lines(_lines()[:2])
    v = L.verify_chain()
    assert not v["ok"] and v["first_bad"] == 2
    assert v["orphans"] == sorted([L.payload_path(es[2]).name, L.entry_path(es[2]).name])
    with pytest.raises(L.LedgerError, match="without an entry"):
        L.append(make_payload(T0 + timedelta(days=9)), stamp=False)


def test_garbage_and_blank_lines_are_detected():
    _append_n(2)
    lines = _lines()
    _write_lines([lines[0], "", lines[1]])
    assert L.verify_chain()["first_bad"] == 1
    _write_lines([lines[0], "{not json"])
    v = L.verify_chain()
    assert v["first_bad"] == 1 and "JSON" in v["reason"]
    with pytest.raises(L.LedgerError):
        L.entries()


def test_crlf_conversion_does_not_break_the_chain():
    _append_n(2)
    P.LEDGER.write_bytes(P.LEDGER.read_bytes().replace(b"\n", b"\r\n"))  # e.g. a Git autocrlf checkout
    assert L.verify_chain()["ok"]
    L.append(make_payload(T0 + timedelta(days=3)), stamp=False)
    v = L.verify_chain()
    assert v["ok"] and v["n"] == 3


# ---- a removed run with a consistently rewritten tail (review finding) ---------------------------------------

class FakeBitcoin:
    """Stands in for ``ots.verify``: a proof file is 'verified' at the Bitcoin time recorded when it was written
    (keyed by the exact .ots bytes, so a re-stamped file gets the time of the re-stamp); real offline digest check."""

    def __init__(self):
        self.times: dict[str, str] = {}

    def mine(self, path, when: datetime) -> None:
        self.times[hashlib.sha256(ots.ots_path(path).read_bytes()).hexdigest()] = when.strftime("%Y-%m-%dT%H:%M:%SZ")

    def verify(self, path, timeout=20, _cache=None, **kw):
        st = ots.status(path)["status"]
        if st in ("unstamped", "invalid"):
            return {"status": st, "attested_utc": None, "reason": None}
        t = self.times.get(hashlib.sha256(ots.ots_path(path).read_bytes()).hexdigest())
        return {"status": "verified", "attested_utc": t, "reason": None} if t else \
            {"status": "pending", "attested_utc": None, "reason": None}


def _rewrite_without(seq: int) -> list[dict]:
    """The attack of the review: drop entry ``seq``, renumber and re-hash every later entry consistently."""
    kept = [json.loads(ln) for ln in _lines() if json.loads(ln)["seq"] != seq]
    prev = S.ZERO_HASH
    for i, e in enumerate(kept):
        e["seq"], e["prev_entry_sha256"] = i, prev
        e["entry_sha256"] = L.entry_hash(e)
        prev = e["entry_sha256"]
    _write_lines([S.canonical_json(e) for e in kept])
    return kept


def test_deleted_run_with_rehashed_tail_is_detected(calendars, monkeypatch):
    btc = FakeBitcoin()
    monkeypatch.setattr(ots, "verify", btc.verify)
    es = [L.append(make_payload(T0 + timedelta(days=i), f=1.0 + i)) for i in range(4)]
    for e in es:  # every entry file was anchored one hour after its run
        assert e["stamp"]["status"] == "pending" and e["entry_stamp"]["status"] == "pending"
        btc.mine(L.entry_path(e), datetime.fromisoformat(e["run_utc"]) + timedelta(hours=1))
    honest = L.verify_anchors(now_utc=T0 + timedelta(days=10))
    assert honest["ok"] and honest["counts"] == {"anchored": 4}
    assert [r["lag_hours"] for r in honest["entries"]] == [1.0] * 4

    # drop run #1 (a bad day) with all its files, then rewrite the later entries consistently
    victim = es[1]
    for f in (L.payload_path(victim), L.entry_path(victim)):
        f.unlink()
        ots.ots_path(f).unlink()
    kept = _rewrite_without(1)
    v = L.verify_chain()
    assert not v["ok"] and v["first_bad"] == 1 and "entry file" in v["reason"]  # old entry files still there

    for e in kept[1:]:  # forge the entry files too: their timestamp proofs no longer match
        L.entry_path(e).write_bytes(L.entry_bytes(e))
    v = L.verify_chain()
    assert not v["ok"] and v["first_bad"] == 1 and "timestamp proof of the entry file" in v["reason"]

    # delete those proofs and stamp the forged entry files again, ten days later
    for e in kept[1:]:
        ots.ots_path(L.entry_path(e)).unlink()
    v = L.verify_chain()
    assert v["ok"] and v["n"] == 3  # offline, a fully rewritten tail is consistent ...
    assert v["days_without_run"] == [(T0 + timedelta(days=1)).date().isoformat()]  # ... but leaves a gap
    tamper = T0 + timedelta(days=10)
    ots.upgrade_all()
    for e in kept[1:]:
        assert ots.status(L.entry_path(e))["status"] == "pending"
        btc.mine(L.entry_path(e), tamper + timedelta(hours=1))
    a = L.verify_anchors(now_utc=tamper + timedelta(hours=2))
    assert not a["ok"] and a["chain_ok"] and a["counts"] == {"anchored": 1, "late": 2}
    late = [r for r in a["entries"] if r["status"] == "late"]
    assert [r["run_id"] for r in late] == [es[2]["run_id"], es[3]["run_id"]]
    assert late[0]["lag_hours"] == 8 * 24 + 1 and "rewrite of the chain" in late[0]["reason"]


def test_truncated_tail_is_not_detected_but_leaves_a_gap():
    """Documented limitation: the most recent run dropped with all its files leaves a shorter, valid chain. Only a
    head hash published elsewhere proves it existed; a later run makes the missing day visible."""
    es = _append_n(3)
    for f in (L.payload_path(es[2]), L.entry_path(es[2])):
        f.unlink()
    _write_lines(_lines()[:2])
    v = L.verify_chain()
    assert v["ok"] and v["n"] == 2 and v["days_without_run"] == []
    L.append(make_payload(T0 + timedelta(days=3)), stamp=False)
    assert L.verify_chain()["days_without_run"] == [(T0 + timedelta(days=2)).date().isoformat()]


def test_verify_anchors_statuses(calendars, monkeypatch):
    btc = FakeBitcoin()
    monkeypatch.setattr(ots, "verify", btc.verify)
    t = [T0, T0 + timedelta(hours=6), T0 + timedelta(hours=30)]
    es = [L.append(make_payload(x), stamp=i > 0) for i, x in enumerate(t)]  # run 0: stamping failed (offline)
    btc.mine(L.entry_path(es[1]), t[1] + timedelta(hours=1))  # pins run 1 and, through its prev hash, run 0
    a = L.verify_anchors(now_utc=t[2] + timedelta(hours=1))
    st = {r["seq"]: r for r in a["entries"]}
    assert st[0]["proof"] == "unstamped" and st[0]["status"] == "anchored" and st[0]["lag_hours"] == 7.0
    assert st[0]["pinned_by"] == es[1]["run_id"]
    assert st[1]["status"] == "anchored" and st[2]["status"] == "pending"
    assert a["ok"] and a["counts"] == {"anchored": 2, "pending": 1}

    a = L.verify_anchors(now_utc=t[2] + timedelta(hours=25))  # still not in Bitcoin after the limit
    assert not a["ok"] and a["entries"][2]["status"] == "late" and "not pinned" in a["entries"][2]["reason"]

    btc.mine(L.entry_path(es[2]), t[2] - timedelta(hours=4))  # Bitcoin saw it 4 h before its claimed run time
    a = L.verify_anchors(now_utc=t[2] + timedelta(hours=25))
    assert a["entries"][2]["status"] == "failed" and "post-dated" in a["entries"][2]["reason"]

    monkeypatch.setattr(ots, "verify", lambda p, **kw: {"status": "failed", "attested_utc": None, "reason": "x"})
    a = L.verify_anchors(now_utc=t[2])
    assert a["counts"] == {"failed": 3} and not a["ok"]


# ---- append rules ---------------------------------------------------------------------------------------------

def test_append_rejects_bad_payloads_and_order():
    L.append(make_payload(T0), stamp=False)
    bad = make_payload(T0 + timedelta(days=1))
    del bad["risk"]
    with pytest.raises(ValueError, match="misses"):
        L.append(bad, stamp=False)
    mismatch = make_payload(T0 + timedelta(days=1))
    mismatch["run_id"] = "20261003T060001Z"
    with pytest.raises(ValueError, match="does not match"):
        L.append(mismatch, stamp=False)
    with pytest.raises(L.LedgerError, match="already"):
        L.append(make_payload(T0), stamp=False)
    with pytest.raises(L.LedgerError, match="not after"):
        L.append(make_payload(T0 - timedelta(hours=1)), stamp=False)
    assert L.verify_chain()["ok"] and L.verify_chain()["n"] == 1


def test_project_ledger_refuses_post_and_backdated_runs(monkeypatch):
    monkeypatch.setattr(L, "PROJECT_LEDGER", P.LEDGER)  # treat the sandbox as forecasts/ledger.jsonl
    clock = {"now": T0 + timedelta(minutes=1)}
    monkeypatch.setattr(L, "_wall_clock", lambda: clock["now"])
    assert L.is_project_ledger()
    L.append(make_payload(T0), stamp=False)

    with pytest.raises(L.LedgerError, match="future"):  # e.g. `forecast --now 2027-01-01` on the real ledger
        L.append(make_payload(datetime(2027, 1, 1, tzinfo=timezone.utc)), stamp=False)
    with pytest.raises(L.LedgerError, match="cannot be switched off"):
        L.append(make_payload(datetime(2027, 1, 1, tzinfo=timezone.utc)), stamp=False, check_clock=False)
    assert _names() == ["20261002T060000Z.entry", "20261002T060000Z.json"] and L.verify_chain()["n"] == 1

    clock["now"] = T0 + timedelta(days=1)
    with pytest.raises(L.LedgerError, match="backdated"):
        L.append(make_payload(T0 + timedelta(days=1) - L.MAX_RUN_AGE - timedelta(minutes=1)), stamp=False)
    e = L.append(make_payload(T0 + timedelta(days=1) - timedelta(hours=2)), stamp=False)  # a slow run: fine
    assert e["seq"] == 1  # the refused future run did not block the next real one
    e = L.append(make_payload(clock["now"] + timedelta(minutes=4)), stamp=False)  # small clock skew: fine
    assert e["seq"] == 2 and L.verify_chain()["ok"]


def test_clock_check_is_off_by_default_only_for_other_ledgers(monkeypatch):
    assert not L.is_project_ledger()
    assert L.PROJECT_LEDGER == P.ROOT / "forecasts" / "ledger.jsonl"
    future = make_payload(datetime(2027, 1, 1, tzinfo=timezone.utc))
    with pytest.raises(L.LedgerError, match="future"):
        L.append(future, stamp=False, check_clock=True)
    assert L.append(future, stamp=False)["seq"] == 0  # test sandboxes replay fixed times
    assert L.check_run_time(T0.isoformat(), now_utc=T0 + L.MAX_RUN_AGE) is None
    assert "backdated" in L.check_run_time(T0.isoformat(), now_utc=T0 + L.MAX_RUN_AGE + timedelta(seconds=1))


def test_failed_ledger_write_leaves_no_orphan(monkeypatch):
    L.append(make_payload(T0), stamp=False)
    real = L._atomic_write

    def fail_on_ledger(path, data):
        if path == P.LEDGER:
            raise OSError("disk full")
        real(path, data)
    monkeypatch.setattr(L, "_atomic_write", fail_on_ledger)
    with pytest.raises(OSError):
        L.append(make_payload(T0 + timedelta(days=1)), stamp=False)
    assert _names() == ["20261002T060000Z.entry", "20261002T060000Z.json"]
    assert L.verify_chain()["ok"]
    assert not P.LEDGER.with_name("ledger.jsonl.lock").exists()


def test_leftover_identical_files_are_recovered_different_ones_refused():
    p = make_payload(T0)
    P.RUNS.mkdir(parents=True)
    L.payload_path(p["run_id"]).write_bytes(L.payload_bytes(p))  # crash after the payload write
    e = L.append(p, stamp=False)
    assert e["seq"] == 0 and L.verify_chain()["ok"]
    q = make_payload(T0 + timedelta(days=1))
    L.payload_path(q["run_id"]).write_bytes(b"{}")
    with pytest.raises(L.LedgerError):
        L.append(q, stamp=False)
    L.payload_path(q["run_id"]).unlink()
    L.entry_path(q["run_id"]).write_bytes(b"{}")  # an entry file that is not this entry
    with pytest.raises(L.LedgerError, match="different content"):
        L.append(q, stamp=False)
    assert _names() == ["20261002T060000Z.entry", "20261002T060000Z.json", "20261003T060000Z.entry"]


def test_stale_lock_is_ignored_and_live_lock_blocks(monkeypatch):
    P.RUNS.mkdir(parents=True)
    lock = P.LEDGER.with_name("ledger.jsonl.lock")
    lock.write_text("123")
    monkeypatch.setattr(L, "_LOCK_STALE", 0.0)
    L.append(make_payload(T0), stamp=False)
    assert not lock.exists()
    lock.write_text("123")
    monkeypatch.setattr(L, "_LOCK_STALE", 3600.0)
    with pytest.raises(L.LedgerError, match="locked"):
        with L._lock(timeout=0.3):
            pass


# ---- stamping ---------------------------------------------------------------------------------------------------

def test_stamp_result_is_returned_but_never_written_to_the_chain(monkeypatch):
    seen = []

    def fake_stamp(path, **kw):
        seen.append(path)
        return {"status": "pending", "calendars_ok": ["a", "b"]}
    monkeypatch.setattr(ots, "stamp", fake_stamp)
    e = L.append(make_payload(T0))
    assert seen == [L.payload_path(e), L.entry_path(e)]
    assert e["stamp"]["status"] == "pending" and e["entry_stamp"]["status"] == "pending"
    assert "stamp" not in L.entries()[0] and "entry_stamp" not in L.entries()[0]

    def broken_stamp(path, **kw):
        raise RuntimeError("calendar library exploded")
    monkeypatch.setattr(ots, "stamp", broken_stamp)
    e2 = L.append(make_payload(T0 + timedelta(days=1)))
    assert e2["stamp"]["status"] == "unstamped" and e2["entry_stamp"]["status"] == "unstamped"
    assert L.verify_chain()["ok"]


def test_real_ots_stamp_with_fake_calendars(calendars):
    entry_lines = []
    for i in range(2):
        e = L.append(make_payload(T0 + timedelta(days=i)))
        assert e["stamp"]["status"] == "pending" and e["entry_stamp"]["status"] == "pending"
        entry_lines.append(_lines()[-1])
    assert ots.status_all()["counts"] == {"pending": 2}  # default runs dir = paths.RUNS, payloads only
    assert ots.status_all(pattern=ots.ENTRY_GLOB)["counts"] == {"pending": 2}
    assert ots.verify(L.payload_path(e))["digest_ok"] and ots.verify(L.entry_path(e))["digest_ok"]
    proof = ots.read_proof(ots.ots_path(L.entry_path(e)))
    assert proof.file_digest.hex() == e["entry_sha256"]  # the entry proof commits to the entry hash itself
    assert _lines() == entry_lines and L.verify_chain()["ok"]  # stamping changed nothing in the chain
