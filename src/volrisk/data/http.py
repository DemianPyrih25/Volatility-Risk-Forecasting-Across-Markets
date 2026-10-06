"""Resumable, retrying bulk downloader with a parquet manifest (SPEC §2.1)."""

from __future__ import annotations

import hashlib
import logging
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests

from volrisk import config as C

log = logging.getLogger(__name__)

MANIFEST = C.RAW / "manifest.parquet"
USER_AGENT = "Mozilla/5.0 (volrisk research downloader)"
_MANIFEST_SCHEMA = {
    "source": pl.Utf8,
    "url": pl.Utf8,
    "local_path": pl.Utf8,
    "http_status": pl.Int64,
    "bytes": pl.Int64,
    "sha256": pl.Utf8,
    "status": pl.Utf8,
    "fetched_at": pl.Datetime("us", "UTC"),
}
_lock = threading.Lock()


@dataclass
class Job:
    source: str
    url: str
    local_path: Path


@dataclass
class ManifestRow:
    source: str
    url: str
    local_path: str
    http_status: int
    bytes: int
    sha256: str
    status: str  # ok | empty | missing | error
    fetched_at: datetime


def read_manifest() -> pl.DataFrame:
    if MANIFEST.exists():
        return pl.read_parquet(MANIFEST)
    return pl.DataFrame(schema=_MANIFEST_SCHEMA)


def _write_manifest(rows: list[ManifestRow]) -> None:
    if not rows:
        return
    new = pl.DataFrame([asdict(r) for r in rows], schema=_MANIFEST_SCHEMA)
    with _lock:
        old = read_manifest()
        merged = pl.concat([old.filter(~pl.col("url").is_in(new["url"].to_list())), new], how="vertical")
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        tmp = MANIFEST.with_suffix(".parquet.part")
        merged.write_parquet(tmp)
        os.replace(tmp, MANIFEST)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_one(job: Job, session: requests.Session, retries: int, timeout: float) -> ManifestRow:
    now = lambda: datetime.now(timezone.utc)  # noqa: E731
    status_code = -1
    for attempt in range(retries + 1):
        try:
            resp = session.get(job.url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            status_code = resp.status_code
            if status_code == 404:
                return ManifestRow(job.source, job.url, str(job.local_path), 404, 0, "", "missing", now())
            if status_code == 200:
                body = resp.content
                job.local_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = job.local_path.with_suffix(job.local_path.suffix + ".part")
                tmp.write_bytes(body)
                os.replace(tmp, job.local_path)
                status = "ok" if body else "empty"
                digest = hashlib.sha256(body).hexdigest()
                return ManifestRow(job.source, job.url, str(job.local_path), 200, len(body), digest, status, now())
        except requests.RequestException as exc:  # timeouts, resets
            log.debug("attempt %d failed for %s: %s", attempt, job.url, exc)
        time.sleep(min(60.0, (2**attempt) * 0.5 + random.uniform(0, 0.5)))
    return ManifestRow(job.source, job.url, str(job.local_path), status_code, 0, "", "error", now())


def fetch_many(
    jobs: list[Job],
    workers: int = 16,
    retries: int = 5,
    timeout: float = 60.0,
    skip_done: bool = True,
    flush_every: int = 200,
) -> list[ManifestRow]:
    """Download ``jobs`` concurrently. Files already recorded as ok/empty/missing are skipped.

    Returns the manifest rows produced by this call (also upserted into ``data/raw/manifest.parquet``).
    """
    if skip_done:
        done = read_manifest().filter(pl.col("status").is_in(["ok", "empty", "missing"]))
        done_urls = set(done["url"].to_list())
        jobs = [j for j in jobs if j.url not in done_urls or not _exists_if_ok(j, done)]
    rows: list[ManifestRow] = []
    pending: list[ManifestRow] = []
    if not jobs:
        return rows
    with requests.Session() as session:
        adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_fetch_one, j, session, retries, timeout) for j in jobs]
            for n, fut in enumerate(as_completed(futs), 1):
                r = fut.result()
                rows.append(r)
                pending.append(r)
                if len(pending) >= flush_every:
                    _write_manifest(pending)
                    pending = []
                if n % 500 == 0:
                    log.info("downloaded %d/%d", n, len(jobs))
    _write_manifest(pending)
    n_err = sum(r.status == "error" for r in rows)
    if n_err:
        log.warning("%d/%d downloads failed (status=error); re-run to retry", n_err, len(rows))
    return rows


def _exists_if_ok(job: Job, done: pl.DataFrame) -> bool:
    st = done.filter(pl.col("url") == job.url)["status"]
    if st.len() and st[0] == "ok":
        return job.local_path.exists()
    return True
