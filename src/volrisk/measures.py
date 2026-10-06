"""Realized measures and the gold daily table (SPEC §5; validity rules §4.3; SPX sample-start probe §1).

Per session ``t`` the price path is ``[p_open, non-null 5-minute bar prices in bar_idx order]`` and every
measure is built from the percent log returns along that path; the overnight/weekend gap never enters an
intraday measure. The kernel works on one flat, session-sorted return array with integer session codes,
so a full asset history (~3,200 sessions x 288 bars) is processed in a single vectorised pass.

Validity (§4.3): an EURUSD/SPX session is ``valid`` iff it has prices, ``coverage >= 0.80`` and ``M >= 5``;
only valid sessions are kept. A crypto session is ``valid`` iff it has prices (crypto is never dropped
for coverage), and ``flag_partial`` marks ``coverage < 0.80`` or ``M < 5``. ``t-1`` is the previous kept
(= valid) session, so ``gap``/``r_cc`` of a crypto session after a partial one use the partial close.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

from volrisk import config as C
from volrisk.io import write_parquet

log = logging.getLogger(__name__)

MU43 = 2 ** (2 / 3) * math.gamma(7 / 6) / math.gamma(1 / 2)  # E|Z|^{4/3}
THETA = math.pi**2 / 4 + math.pi - 5  # BNS asymptotic variance constant of (RV - BV) / IQ
MIN_RETURNS = 5  # tq needs M >= 5 (SPEC §5.3)
SPX_PROBE_MINUTES = 0.95  # SPEC §1: a session is complete with >= 95% real RTH minutes ...
SPX_PROBE_SHARE = 0.95  # ... and the model sample starts in the first month with >= 95% such sessions

GOLD_SCHEMA: dict[str, pl.DataType] = {
    "asset": pl.String,
    "session_date": pl.Date,
    "n_sched": pl.Int32,
    "n_real_bars": pl.Int32,
    "coverage": pl.Float64,
    "valid": pl.Boolean,
    "flag_partial": pl.Boolean,
    "M": pl.Int32,
    "p_open": pl.Float64,
    "p_close": pl.Float64,
    "gap": pl.Float64,
    "r_cc": pl.Float64,
    "rv": pl.Float64,
    "bv": pl.Float64,
    "tq": pl.Float64,
    "rq": pl.Float64,
    "rs_pos": pl.Float64,
    "rs_neg": pl.Float64,
    "z_jump": pl.Float64,
    "j": pl.Float64,
    "c": pl.Float64,
    "tv": pl.Float64,
}
GOLD_COLUMNS = list(GOLD_SCHEMA)
MEASURE_KEYS = ("rv", "bv", "tq", "rq", "rs_pos", "rs_neg", "z_jump", "j", "c", "M")


def min_coverage() -> float:
    return float(C.load()["sessions"]["min_coverage"])


def _bar_minutes() -> int:
    return int(C.load()["sessions"]["bar_minutes"])


def jump_critical_value() -> float:
    """Phi^{-1}(alpha) for the one-sided jump test (alpha = 0.999 -> 3.090)."""
    return float(norm.ppf(float(C.load()["jumps"]["alpha"])))


# --------------------------------------------------------------------------------------------------------
# Intraday returns and the measure kernel (SPEC §5.1–5.2)


def intraday_returns(bars_one_session_prices: np.ndarray, p_open: float) -> np.ndarray:
    """Percent log returns along ``[p_open, non-null bar prices]``; null bars are spanned by one return."""
    p = np.asarray(bars_one_session_prices, dtype=float)
    path = np.concatenate([[float(p_open)], p[~np.isnan(p)]])
    return 100.0 * np.diff(np.log(path))


def _grouped_measures(r: np.ndarray, gid: np.ndarray, n: int) -> dict[str, np.ndarray]:
    """All §5.2 measures for ``n`` sessions from returns ``r`` sorted by session code ``gid`` (0..n-1).

    Within a session the returns must be in path order. Lagged products only pair returns of the same
    session (``gid[i] == gid[i-k]``; because ``gid`` is sorted, equal codes at lag 4 imply equal at lag 2).
    """
    r = np.asarray(r, dtype=float)
    gid = np.asarray(gid, dtype=np.intp)
    a = np.abs(r)
    r2 = r * r
    M = np.bincount(gid, minlength=n)
    rv = np.bincount(gid, r2, n)
    rs_pos = np.bincount(gid, np.where(r > 0, r2, 0.0), n)
    rs_neg = np.bincount(gid, np.where(r < 0, r2, 0.0), n)
    rq_sum = np.bincount(gid, r2 * r2, n)

    same2 = gid[2:] == gid[:-2]
    bv_sum = np.bincount(gid[2:][same2], (a[2:] * a[:-2])[same2], n)
    a43 = a ** (4 / 3)
    same4 = gid[4:] == gid[:-4]
    tq_sum = np.bincount(gid[4:][same4], (a43[4:] * a43[2:-2] * a43[:-4])[same4], n)

    Mf = M.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        bv = (math.pi / 2) * (Mf / (Mf - 2)) * bv_sum
        tq = Mf * MU43**-3 * (Mf / (Mf - 4)) * tq_sum
        rq = (Mf / 3) * rq_sum
        # bv == 0 implies tq == 0 (every tq product contains a bv product); max(1, 0/0) is taken as 1.
        ratio = np.where(bv > 0, tq / bv**2, 1.0)
        # N(0,1) only under a Brownian semimartingale: fat-tailed intraday returns (realized kurtosis well
        # above 3) shift z upward without any discrete jump, so the jump-day share depends on the sampling
        # interval. The 5-minute test is kept as pre-registered (SPEC §5.2).
        z = np.sqrt(Mf) * ((rv - bv) / rv) / np.sqrt(THETA * np.maximum(1.0, ratio))

    ok = M >= MIN_RETURNS
    bv = np.where(ok, bv, rv)
    tq = np.where(ok, tq, rq)
    z = np.where(ok & (rv > 0), z, 0.0)  # flat path (rv == 0): no jump evidence
    j = np.where(z > jump_critical_value(), np.maximum(rv - bv, 0.0), 0.0)
    return {
        "rv": rv,
        "bv": bv,
        "tq": tq,
        "rq": rq,
        "rs_pos": rs_pos,
        "rs_neg": rs_neg,
        "z_jump": z,
        "j": j,
        "c": rv - j,
        "M": M,
    }


def session_measures(r: np.ndarray) -> dict:
    """rv, bv, tq, rq, rs_pos, rs_neg, z_jump, j, c, M of one session's intraday returns (SPEC §5.2).

    With ``M < 5`` the fallbacks of §5.3 apply: ``bv := rv``, ``tq := rq``, ``z_jump := 0``, ``j := 0``.
    """
    r = np.asarray(r, dtype=float)
    out = _grouped_measures(r, np.zeros(r.size, dtype=np.intp), 1)
    return {k: int(v[0]) if k == "M" else float(v[0]) for k, v in out.items()}


# --------------------------------------------------------------------------------------------------------
# Session table, validity and the gold rows (SPEC §4.3, §5.3)


def _session_table(bars: pl.DataFrame, sessions: pl.DataFrame, asset: str) -> pl.DataFrame:
    """All sessions of ``asset`` with their measures and ``valid`` (= kept), ``flag_partial``, ``reason``."""
    crypto = C.clock(asset) == "crypto"
    s = sessions.filter(pl.col("asset") == asset) if "asset" in sessions.columns else sessions
    if "coverage" not in s.columns:
        s = s.with_columns(coverage=pl.col("n_real_bars") / pl.col("n_sched"))
    s = (
        s.sort("session_date")
        .with_columns(pl.col("p_open").fill_nan(None), pl.col("p_close").fill_nan(None))
        .with_row_index("sid")
        .with_columns(has_prices=pl.col("p_open").is_not_null() & pl.col("p_close").is_not_null())
    )

    b = bars.filter(pl.col("asset") == asset) if "asset" in bars.columns else bars
    b = (
        b.select("session_date", "bar_idx", pl.col("price").cast(pl.Float64).fill_nan(None))
        .drop_nulls("price")
        .join(
            s.filter(pl.col("has_prices")).select("session_date", "sid", "p_open"),
            on="session_date",
            how="inner",
        )
        .sort("sid", "bar_idx")
    )
    sid = b["sid"].to_numpy().astype(np.intp)
    lp = np.log(b["price"].to_numpy())
    prev = np.empty_like(lp)
    prev[1:] = lp[:-1]
    first = np.ones(sid.size, dtype=bool)
    first[1:] = sid[1:] != sid[:-1]
    prev[first] = np.log(b["p_open"].to_numpy()[first])  # the path of every session starts at p_open
    m = _grouped_measures(100.0 * (lp - prev), sid, s.height)
    s = s.with_columns([pl.Series(k, v) for k, v in m.items()])

    cov_ok = pl.col("coverage").cast(pl.Float64).fill_nan(None).fill_null(0.0) >= min_coverage()
    m_ok = pl.col("M") >= MIN_RETURNS
    if crypto:
        # never dropped for coverage (lags need a continuous calendar): valid = has prices, partial is a flag
        valid, flag = pl.col("has_prices"), ~cov_ok | ~m_ok
    else:
        valid, flag = pl.col("has_prices") & cov_ok & m_ok, pl.lit(False)
    return s.with_columns(
        valid=valid,
        flag_partial=flag,
        reason=pl.when(~pl.col("has_prices"))
        .then(pl.lit("no_prices"))
        .when(~cov_ok)
        .then(pl.lit("low_coverage"))
        .when(~m_ok)
        .then(pl.lit("few_returns")),
    )


def _gold_rows(table: pl.DataFrame, asset: str) -> pl.DataFrame:
    """Kept sessions -> gold rows: crypto rq carry-over, gap / r_cc vs the previous kept close, tv."""
    t = table.filter(pl.col("valid")).sort("session_date")
    if C.clock(asset) == "crypto":
        # HARQ stability: a partial session inherits the previous kept session's (possibly inherited) rq.
        t = t.with_columns(
            rq=pl.when(pl.col("flag_partial") & (pl.int_range(pl.len()) > 0))
            .then(None)
            .otherwise(pl.col("rq"))
            .forward_fill()
        )
    prev_close = pl.col("p_close").shift(1).log()
    t = (
        t.with_columns(
            gap=100.0 * (pl.col("p_open").log() - prev_close),
            r_cc=100.0 * (pl.col("p_close").log() - prev_close),
        )
        .filter(pl.col("gap").is_not_null())  # first kept session has no previous close
        .with_columns(tv=pl.col("gap") ** 2 + pl.col("rv"), asset=pl.lit(asset))
    )
    return t.select([pl.col(k).cast(dt) for k, dt in GOLD_SCHEMA.items()])


def daily_measures(bars: pl.DataFrame, sessions: pl.DataFrame, asset: str) -> pl.DataFrame:
    """Gold rows (SPEC §5.3 columns) of one asset from its silver bars and sessions, sorted by session_date.

    EURUSD/SPX keep valid sessions only; crypto keeps every session with prices and flags partial ones.
    ``gap``/``r_cc`` use the previous kept session's ``p_close``; the first kept session is dropped.
    """
    return _gold_rows(_session_table(bars, sessions, asset), asset)


def rth_minute_share(bars: pl.DataFrame, sessions: pl.DataFrame) -> pl.DataFrame:
    """``sessions`` of one asset plus ``minute_share`` = real minutes in its bars / ``bar_minutes * n_sched``.

    ``n_real_min`` of the silver bars counts the real minutes inside the session window, so for SPX this is
    the share of real RTH minutes (SPEC §1). A session without bar rows has share 0.
    """
    real = bars.group_by("session_date").agg(_real_min=pl.col("n_real_min").cast(pl.Int64).sum())
    return (
        sessions.join(real, on="session_date", how="left")
        .with_columns(minute_share=pl.col("_real_min").fill_null(0) / (_bar_minutes() * pl.col("n_sched")))
        .drop("_real_min")
    )


def spx_model_start(sessions: pl.DataFrame, bars: pl.DataFrame) -> date:
    """First day of the first month with >= 95% of sessions having >= 95% real RTH minutes (SPEC §1 probe).

    ``sessions`` are the scheduled XNYS sessions to probe (every session counts in the month's denominator);
    the real-minute counts come from the 5-minute ``bars`` (``n_real_min``), not from bar coverage, which
    would count a bar with a single real minute as complete.
    """
    months = (
        rth_minute_share(bars, sessions)
        .select(
            month=pl.col("session_date").dt.truncate("1mo"),
            full=pl.col("minute_share").fill_nan(None).fill_null(0.0) >= SPX_PROBE_MINUTES,
        )
        .group_by("month")
        .agg(share=pl.col("full").mean())
        .filter(pl.col("share") >= SPX_PROBE_SHARE)
        .sort("month")
    )
    if months.is_empty():
        raise ValueError("SPX start probe: no month has >= 95% of sessions with >= 95% real RTH minutes")
    return months["month"][0]


# --------------------------------------------------------------------------------------------------------
# Gold build (SPEC §5.3 split, §11 seal)


def _silver_path(silver_dir: Path, kind: str, asset: str) -> Path:
    return silver_dir / kind / f"asset={asset}" / "part.parquet"


def sample_start(asset: str, sessions: pl.DataFrame, bars: pl.DataFrame) -> date:
    """Modelling start: config start (BTC/ETH 2018, EURUSD 2012) or the SPX probe.

    The probe only sees dev sessions on or after the config start, so holdout data can never move it.
    """
    start = C.asset(asset).start
    if C.clock(asset) != "xnys":
        return start
    dev = (pl.col("session_date") >= start) & (pl.col("session_date") < C.holdout_start())
    return max(start, spx_model_start(sessions.filter(dev), bars.filter(dev)))


def _write_merged(new: pl.DataFrame, path: Path, assets: Sequence[str], split: pl.Expr) -> None:
    """Write ``new`` to ``path``, carrying over the existing rows of assets that were not rebuilt.

    A rebuild of every asset in ``C.ASSETS`` replaces the file without reading it. On a partial rebuild
    the other assets' rows are copied unchanged (no statistic is computed from them); ``split`` re-asserts
    the file's dev/holdout side.
    """
    parts = [new]
    if path.exists() and not set(C.ASSETS) <= set(assets):
        old = pl.scan_parquet(path)
        missing = set(GOLD_COLUMNS) - set(old.collect_schema().names())
        if missing:
            raise ValueError(f"{path}: existing file lacks columns {sorted(missing)}; rebuild all assets")
        parts.append(
            old.filter(~pl.col("asset").is_in(list(assets)) & split)
            .select([pl.col(k).cast(dt) for k, dt in GOLD_SCHEMA.items()])
            .collect()
        )
    write_parquet(pl.concat(parts, how="vertical").sort("asset", "session_date"), path)


def build_gold(
    assets: Sequence[str] = C.ASSETS,
    silver_dir: Path | None = None,
    gold_dir: Path | None = None,
    holdout_dir: Path | None = None,
) -> dict[str, dict]:
    """Build ``gold/daily.parquet`` (dev) and ``holdout/daily.parquet`` for ``assets``; return a summary.

    Rows of other assets already in the two files are kept, so a single asset can be rebuilt. ``t-1`` of
    the first in-sample session is the previous valid session in silver, even if it precedes the start.
    The summary (and the log) holds dev diagnostics but only row counts for the holdout split.
    """
    silver_dir = Path(silver_dir) if silver_dir is not None else C.SILVER
    gold_dir = Path(gold_dir) if gold_dir is not None else C.GOLD
    holdout_dir = Path(holdout_dir) if holdout_dir is not None else C.HOLDOUT
    h0, end = C.holdout_start(), C.data_end()
    assets = tuple(dict.fromkeys(assets))

    frames: list[pl.DataFrame] = []
    summary: dict[str, dict] = {}
    for a in assets:
        upto_end = pl.col("session_date") <= end
        bars = pl.read_parquet(_silver_path(silver_dir, "bars5m", a)).filter(upto_end)
        sessions = pl.read_parquet(_silver_path(silver_dir, "sessions", a)).filter(upto_end)
        start = sample_start(a, sessions, bars)
        # measures on all silver sessions, so the first in-sample session has its t-1; then cut at start
        table = _session_table(bars, sessions, a)
        daily = _gold_rows(table, a).filter(pl.col("session_date") >= start)
        frames.append(daily)

        dev_tab = table.filter((pl.col("session_date") >= start) & (pl.col("session_date") < h0))
        dropped = dev_tab.filter(~pl.col("valid"))["reason"].value_counts(sort=True)
        info = {
            "start": start,
            "dev_rows": int((daily["session_date"] < h0).sum()),
            "holdout_rows": int((daily["session_date"] >= h0).sum()),
            "dev_dropped": {r: int(n) for r, n in dropped.iter_rows()},
            "dev_flag_partial": int(dev_tab.filter(pl.col("valid") & pl.col("flag_partial")).height),
        }
        summary[a] = info
        log.info(
            "%s: start %s, dev rows %d (dropped %s, flagged partial %d), holdout rows %d",
            a,
            start,
            info["dev_rows"],
            info["dev_dropped"],
            info["dev_flag_partial"],
            info["holdout_rows"],
        )

    gold = pl.concat([pl.DataFrame(schema=GOLD_SCHEMA), *frames], how="vertical")
    for split, out in ((pl.col("session_date") < h0, gold_dir), (pl.col("session_date") >= h0, holdout_dir)):
        _write_merged(gold.filter(split), out / "daily.parquet", assets, split)
    return summary
