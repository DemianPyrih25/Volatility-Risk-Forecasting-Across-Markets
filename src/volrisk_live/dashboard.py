"""Live dashboard (docs/LIVE_SPEC.md §8): the frozen dashboard plus the tabs *Tomorrow*, *Forward test* and
*Verification*.

``uv run python -m volrisk_live.dashboard [--port 8050] [--no-browser]`` (or ``python -m volrisk_live dashboard``).
The frozen app (:func:`volrisk.dashboard.app.create_app`) is built unchanged and the three tabs are added to its
``dcc.Tabs(id="tabs")`` in memory — no frozen file is touched, and the frozen tabs and controls keep working.

Like the frozen app this is a read-only viewer: no model fitting, no downloads, no network. The new tabs read the
ledger and payloads under ``forecasts/``, the score files of ``volrisk_live.score`` (through its own loaders and
tables, so every number equals ``reports/live/forward_test.md``), ``reports/live/verification.json`` and, for the
trailing realised volatility, the live gold rebuild under ``data/live/``. The callback bodies are plain functions
of a :class:`LiveData` (``tomorrow_view``, ``forward_view``, ``verification_view``), so the tests call them
directly; :func:`create_app` only wires them to the components.

Claims about time are made conservatively. "Recorded before the session opened / closed" uses the payload's write time
(``cli.recorded_utc``: the later of ``run_utc`` and ``checks.computed_utc``) against the frozen session calendar.
Forecasts written after their first target session had closed are flagged, never presented as forward-test
evidence. The Timestamp chip is green only when verification 8b checked the proof's Bitcoin attestation against the
block chain (``reports/live/verification.json``), never from the proof file alone.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import threading
import webbrowser
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots

from volrisk import palette as P
from volrisk.dashboard import app as A
from volrisk.dashboard.data import annualisation
from volrisk_live import cli as CLI
from volrisk_live import paths
from volrisk_live import score as SC
from volrisk_live.cli import normalise_checks, ots_path

HOST = "127.0.0.1"
TITLE = "Volatility & risk dashboard — live"
PRIMARY = "COMBO"  # primary forecast model (SPEC §0)
PRIMARY_RISK = "COMBO+FHS"  # primary risk model (SPEC §0)
REF_MODEL = "HAR"
HORIZON_DAYS = {"1d": 1, "1w": 7, "1m": 30}
DEFAULT_FORWARD_MODELS = ("HAR", "COMBO")
REFRESH_MS = 5 * 60 * 1000  # the daily run rewrites the files under a running app
MAX_IV_AGE_DAYS = 7  # an implied-vol quote older than this (calendar days before the cutoff) is not "current"
GREEN, RED = P.ZONE_COLORS["green"], P.ZONE_COLORS["red"]
STATUS_COLORS = {"PASS": GREEN, "FAIL": RED, "SKIPPED": A.NA_COLOR}
RISK_MEASURES = (("var99", "VaR 99%", "diamond"), ("var975", "VaR 97.5%", "circle"), ("es975", "ES 97.5%", "square"))
NO_RUN = "No live forecast recorded yet — run: uv run python -m volrisk_live daily"
VOL_NOTE = (
    "Forecasts F are cumulative variances over the horizon window (%²); the chart shows √(F/n·ann) for readability "
    "only. Realised = the same transform of the realised variance over the trailing window of the same length "
    "ending at the data cutoff; IV = the 30-day implied volatility (VIX / DVOL) at the cutoff, as quoted."
)
RISK_NOTE = (
    "Hypothetical 1-unit long position, μ = 0: VaR and ES are positive losses in % of the position for the next "
    "session. COMBO+FHS is the pre-registered primary risk model; HS-250 is the reference."
)
FORWARD_SUB = (
    "Every run writes its forecasts to the hash-chained ledger and submits them for an OpenTimestamps proof; they are "
    "scored once the realised data arrive (LIVE_SPEC §6), and a score is never revised. A forecast counts only if it "
    "was written before its first target session closed — by the payload's write time (the latest of its start time "
    "and its own clock readings) against the frozen session calendar; later ones stay in the score files but are left "
    "out of every table. The Bitcoin-attested time of each proof is checked by verification 8b."
)
# Timestamp chip per ots status (ots.STATUS_TEXT): never green from the proof file alone (a Bitcoin attestation in
# a proof is a claim anyone can write); green only when verification 8b checked it against the block chain.
STAMP_LEVEL: dict[str, bool | None] = {
    "verified": None, "upgraded": None, "unverified": None, "pending": None, "partial": None, "unstamped": None,
    "invalid": False, "failed": False,
}
TIMING_LEVEL = {"before_open": True, "in_session": None, "after_close": False, "unknown": None}
EXTRA_CSS = """
.kv { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 5px 18px; margin: 4px 0 0;
      font-size: 13px; }
.kv dt { color: var(--ink2); font-weight: 600; }
.kv dd { margin: 0; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
.kv ul.plain li { border-bottom: none; padding: 1px 0; }
.badge { display: inline-flex; align-items: center; gap: 6px; padding: 2px 10px; border-radius: 999px;
         border: 1px solid var(--axis); background: var(--surface); font-weight: 650; font-size: 12px; }
.vrow { display: grid; grid-template-columns: 120px minmax(0, 1fr); gap: 12px; padding: 10px 0;
        border-bottom: 1px solid var(--grid); align-items: start; }
.vrow h4 { margin: 0; font-size: 13.5px; font-weight: 600; }
.vrow .method { color: var(--ink2); font-size: 12.5px; margin: 3px 0 6px; max-width: 980px; }
.vrow .reason { color: var(--ink2); font-size: 12.5px; margin: 3px 0; }
.empty-state { padding: 14px 16px; border: 1px solid var(--border); border-left: 3px solid var(--axis);
               border-radius: 6px; color: var(--ink2); background: var(--ref); font-size: 13px; margin-top: 8px; }
code { font-family: ui-monospace, "Cascadia Mono", Consolas, monospace; font-size: 12px; }
"""


# --------------------------------------------------------------------------------------------- small helpers
def _num(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def _date(x: Any) -> str:
    if x is None or x == "" or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    try:
        t = pd.Timestamp(x)
    except (TypeError, ValueError):
        return str(x)
    return "—" if pd.isna(t) else t.strftime("%Y-%m-%d")


def _truthy(s: pd.Series) -> pd.Series:
    """Boolean column as written to CSV (True / true / 1 / 1.0; anything else, NaN included, is False)."""
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))


def _short(sha: Any, n: int = 12) -> str:
    return str(sha)[:n] if sha else "—"


def _vol(F: Any, n: Any, asset: str) -> float:
    """Annualised % volatility of a cumulative variance ``F`` over ``n`` sessions (display only)."""
    f, k = _num(F), _num(n)
    if not (f > 0 and k > 0):
        return math.nan
    return math.sqrt(f / k * annualisation(asset))


def _fmt_vol(v: Any) -> str:
    return A.fmt_num(v, 1) + "%" if A._finite(v) else "—"


def badge(status: str, title: str | None = None) -> html.Span:
    """PASS / FAIL / SKIPPED pill: coloured dot + the word (state is never shown by colour alone)."""
    color = STATUS_COLORS.get(status, A.NA_COLOR)
    return html.Span([html.Span(className="dot", style={"background": color}), html.Span(status)],
                     className=f"badge badge-{status.lower()}", style={"borderColor": color}, title=title)


def _state_chip(ok: bool | None, text: str) -> html.Span:
    """Green / red / grey dot + text; grey = not yet known (never a pass by default)."""
    return A.status_chip(text, A.NA_COLOR if ok is None else (GREEN if ok else RED))


def _empty_state(text: Any) -> html.Div:
    return html.Div(text, className="empty-state")


def _kv(pairs: Sequence[tuple[str, Any]]) -> html.Dl:
    items: list = []
    for k, v in pairs:
        items += [html.Dt(k), html.Dd("—" if v is None or (isinstance(v, str) and not v) else v)]
    return html.Dl(items, className="kv")


def _interpret_chain(res: Any) -> tuple[bool | None, str]:
    """``ledger.verify_chain()`` result ``{ok, n, first_bad, reason, ...}`` (or a bool / list) → (intact?, text)."""
    if isinstance(res, bool):
        return res, "intact" if res else "broken"
    if res is None:
        return True, "intact"
    if isinstance(res, dict):
        ok = next((res[k] for k in ("ok", "intact", "valid", "passed") if k in res), None)
        why = next((res[k] for k in ("reason", "message", "error", "first_broken") if res.get(k)), None)
        n = res.get("n", res.get("n_entries"))
        n_txt = f" · {n} entries" if isinstance(n, int) and not isinstance(n, bool) else ""
        if ok is None:
            return None, (str(why) if why else "unknown") + n_txt
        return bool(ok), ("intact" if ok else f"BROKEN: {why or 'see python -m volrisk_live verify'}") + n_txt
    if isinstance(res, (list, tuple)):
        return (not res), "intact" if not res else f"BROKEN: {res[0]}"
    return None, str(res)[:160]


# --------------------------------------------------------------------------------------------- data layer
class LiveData:
    """Read-only access to the live files. Every read is fresh: the daily run rewrites them under a running app."""

    def __init__(self, forecasts_dir: str | Path | None = None, reports_dir: str | Path | None = None,
                 live_dir: str | Path | None = None):
        self.forecasts_dir = Path(forecasts_dir) if forecasts_dir is not None else paths.FORECASTS
        self.reports_dir = Path(reports_dir) if reports_dir is not None else paths.LIVE_REPORTS
        self.live_dir = Path(live_dir) if live_dir is not None else paths.LIVE

    # ------------------------------------------------------------------ files
    @property
    def ledger_path(self) -> Path:
        return self.forecasts_dir / paths.LEDGER.name

    @property
    def runs_dir(self) -> Path:
        return self.forecasts_dir / paths.RUNS.name

    @property
    def scores_path(self) -> Path:
        return self.forecasts_dir / paths.SCORES.name

    @property
    def risk_scores_path(self) -> Path:
        return self.forecasts_dir / paths.RISK_SCORES.name

    @property
    def verification_path(self) -> Path:
        return self.reports_dir / paths.VERIFY_JSON.name

    # ------------------------------------------------------------------ ledger and payloads
    def entries(self) -> list[dict]:
        """Ledger entries in file order (unparsable lines are skipped here; the chain check reports them)."""
        if not self.ledger_path.exists():
            return []
        out = []
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(e, dict):
                    out.append(e)
        return out

    def payload_file(self, entry: dict) -> Path:
        """``runs/<run_id>.json`` of a ledger entry (its ``payload`` field is that path relative to forecasts/)."""
        ref = entry.get("payload")
        if isinstance(ref, str) and ref and (self.forecasts_dir / ref).exists():
            return self.forecasts_dir / ref
        return self.runs_dir / f"{entry.get('run_id')}.json"

    def load(self, entry: dict) -> dict | None:
        try:
            return json.loads(self.payload_file(entry).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def payload_hash_ok(self, entry: dict) -> bool | None:
        """SHA-256 of the payload file == the hash recorded in its ledger entry (None without an entry hash)."""
        want, f = entry.get("payload_sha256"), self.payload_file(entry)
        if not want or not f.exists():
            return None if not want else False
        return hashlib.sha256(f.read_bytes()).hexdigest() == want

    def runs(self) -> list[dict]:
        """Ledger entries, or — before any ledger exists — one pseudo-entry per payload file (by run id)."""
        entries = self.entries()
        if entries or not self.runs_dir.exists():
            return entries
        return [{"run_id": f.stem, "unledgered": True} for f in sorted(self.runs_dir.glob("*.json"))]

    def latest(self) -> tuple[dict | None, dict | None]:
        """(entry, payload) of the latest run."""
        runs = self.runs()
        return (runs[-1], self.load(runs[-1])) if runs else (None, None)

    def payloads(self) -> list[dict]:
        return [p for e in self.runs() if (p := self.load(e)) is not None]

    # ------------------------------------------------------------------ integrity (local, no network)
    def chain_status(self) -> tuple[bool | None, str]:
        """Hash-chain check by ``volrisk_live.ledger.verify_chain`` (which reads the default ledger)."""
        if not self.ledger_path.exists():
            return None, "no ledger yet"
        try:
            from volrisk_live import ledger

            fn = ledger.verify_chain
            takes_path = any(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
                             for p in inspect.signature(fn).parameters.values())
        except (ImportError, AttributeError, TypeError, ValueError):
            return None, "ledger module not available"
        if not takes_path and self.ledger_path.resolve() != paths.LEDGER.resolve():
            return None, f"not checked (verify_chain reads {paths.LEDGER.name} in the project folder only)"
        try:
            res = fn(self.ledger_path) if takes_path else fn()
        except Exception as e:  # noqa: BLE001 - shown as a failed check, never crashes the page
            return False, f"check failed ({type(e).__name__}: {e})"
        return _interpret_chain(res)

    def bitcoin_check(self, name: str) -> dict | None:
        """Verification 8b's record of the payload file ``name`` (``verification.json``: ``ots.verify`` of its proof
        against the Bitcoin block header, and the attested time vs the payload's first outcome), or None."""
        for c in normalise_checks(self.verification()):
            if c["id"] == "8b" and isinstance(c["evidence"], dict):
                for r in c["evidence"].get("payloads") or []:
                    if isinstance(r, dict) and r.get("payload") == name:
                        return r
        return None

    def stamp_status(self, path: Path) -> tuple[bool | None, str]:
        """OpenTimestamps state of a payload: ``ots.status`` (offline, the proof file only; wording from
        ``ots.STATUS_TEXT``) and, for a proof that claims a Bitcoin attestation, verification 8b's check of it.

        Green only when 8b verified the claimed block (header, proof of work, merkle root) and the attested time
        precedes the payload's first outcome; red for an invalid proof or a failed check; grey otherwise."""
        try:
            from volrisk_live import ots

            st, words = ots.status(path), ots.STATUS_TEXT
        except Exception as e:  # noqa: BLE001 - shown as unknown, never crashes the page
            exists = ots_path(path).exists()
            return None, f"{'proof present' if exists else 'no proof'}; status not readable ({type(e).__name__})"
        s = str(st.get("status"))
        word = words.get(s, "unknown proof status")
        heights = sorted({int(h) for h in st.get("block_heights") or []})
        if s == "upgraded" and heights:
            return self._bitcoin_chip(path.name, heights, word)
        if s in ("pending", "partial"):
            n = len(st.get("pending_calendars") or [])
            return STAMP_LEVEL[s], f"{s}: {word} ({n} calendar(s); the Bitcoin attestation follows a few hours later)"
        if s == "unstamped":
            return STAMP_LEVEL[s], f"not stamped yet: {word} (retried by the next daily run)"
        return STAMP_LEVEL.get(s), f"{s}: {st.get('reason') or word}"

    def _bitcoin_chip(self, name: str, heights: list[int], word: str) -> tuple[bool | None, str]:
        claim = f"claims Bitcoin block {heights[0]:,}"
        r = self.bitcoin_check(name)
        status = str(r.get("bitcoin") or "") if r else ""
        if r is None or not status:
            return None, f"{claim}: {word} - checked by verification 8b (python -m volrisk_live verify)"
        if status in ("failed", "invalid"):
            return False, f"{claim}: Bitcoin check {status.upper()} in verification 8b"
        if status != "verified":
            return None, f"{claim}: verification 8b says {status}"
        if sorted({int(h) for h in r.get("bitcoin_heights") or []}) != heights or r.get("digest_ok") is False:
            return None, f"{claim}: the proof changed after verification 8b - run it again"
        att, outcome = CLI.utc_time(r.get("attested_utc")), CLI.utc_time(r.get("first_outcome_utc"))
        if att is None:
            return None, f"{claim}: verification 8b gives no attested time"
        if outcome is not None and att >= outcome:
            return False, (f"Bitcoin block {heights[0]:,} mined {att:%Y-%m-%d %H:%M} UTC, NOT before the first "
                           f"outcome ({outcome:%Y-%m-%d %H:%M} UTC)")
        meta = self.verification()
        gen = meta.get("generated_utc") if isinstance(meta, dict) else None
        return True, (f"Bitcoin block {heights[0]:,} mined {att:%Y-%m-%d %H:%M} UTC"
                      + (f", before the first outcome ({outcome:%Y-%m-%d %H:%M} UTC)" if outcome else "")
                      + f" - verified by verification 8b{f' ({gen})' if gen else ''}")

    # ------------------------------------------------------------------ scores, verification, data
    def scores(self) -> pd.DataFrame:
        try:
            return SC.load_scores(self.scores_path)
        except (OSError, ValueError):
            return pd.DataFrame(columns=SC.SCORE_COLUMNS)

    def risk_scores(self) -> pd.DataFrame:
        try:
            return SC.load_risk_scores(self.risk_scores_path)
        except (OSError, ValueError):
            return pd.DataFrame(columns=SC.RISK_SCORE_COLUMNS)

    def verification(self) -> dict | list | None:
        try:
            return json.loads(self.verification_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def daily(self, asset: str) -> pd.DataFrame:
        """``session_date, tv`` of one asset from the live gold rebuild (holdout-period file, else dev file)."""
        for f in (self.live_dir / "holdout" / "daily.parquet", self.live_dir / "gold" / "daily.parquet"):
            if not f.exists():
                continue
            try:
                d = pd.read_parquet(f, columns=["asset", "session_date", "tv"])
            except (OSError, ValueError, KeyError):
                continue
            d = d[d["asset"] == asset]
            if not d.empty:
                return d.assign(session_date=pd.to_datetime(d["session_date"])).sort_values("session_date")
        return pd.DataFrame(columns=["asset", "session_date", "tv"])


    def recent_forecasts(self, asset: str, models: Sequence[str] = ("COMBO", "HAR", "GJR")) -> pd.DataFrame:
        """1d forecasts ``origin, model, F`` of one asset from the latest live walk-forward (data/live/forecasts)."""
        f = self.live_dir / "forecasts" / "forecasts.parquet"
        cols = ["origin", "model", "F"]
        if not f.exists():
            return pd.DataFrame(columns=cols)
        try:
            d = pd.read_parquet(f, columns=["asset", "horizon", *cols],
                                filters=[("asset", "==", asset), ("horizon", "==", "1d"), ("model", "in", list(models))])
        except (OSError, ValueError, KeyError):
            return pd.DataFrame(columns=cols)
        return d[cols].assign(origin=pd.to_datetime(d["origin"]))


# --------------------------------------------------------------------------------------------- tomorrow
def payload_assets(payload: dict | None) -> list[str]:
    if not payload:
        return []
    seen = list(payload.get("data") or {}) + [r.get("asset") for r in payload.get("forecasts") or []]
    return P.order_assets(a for a in seen if a)


def _pick_asset(asset: str | None, assets: Sequence[str]) -> str | None:
    return asset if asset in assets else (assets[0] if assets else None)


def tomorrow_frames(payload: dict, asset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(forecast rows, risk rows) of one asset; ``vol_ann`` recomputed only where the payload has none."""
    fc = pd.DataFrame([r for r in payload.get("forecasts") or [] if r.get("asset") == asset])
    if not fc.empty:
        given = fc["vol_ann"] if "vol_ann" in fc else pd.Series([None] * len(fc))
        fc["vol_ann"] = [v if A._finite(v) else _vol(F, n, asset)
                         for v, F, n in zip(given, fc["F"], fc["n_t"], strict=True)]
    rk = pd.DataFrame([r for r in payload.get("risk") or [] if r.get("asset") == asset])
    return fc, rk


def trailing_realised(daily: pd.DataFrame, asset: str, end: Any, horizon: str) -> tuple[float, int]:
    """Annualised realised vol over the sessions in ``(end − days, end]`` (1d: the session ``end``) and their n."""
    if daily.empty or end is None:
        return math.nan, 0
    t = pd.Timestamp(end)
    w = daily[(daily["session_date"] > t - pd.Timedelta(days=HORIZON_DAYS[horizon])) & (daily["session_date"] <= t)]
    tv = pd.to_numeric(w["tv"], errors="coerce").dropna()
    if tv.empty:
        return math.nan, 0
    return math.sqrt(tv.mean() * annualisation(asset)), int(len(tv))


def implied_now(payload: dict, asset: str, end: Any = None) -> tuple[float, str, str]:
    """(iv, source, origin) of the latest implied-vol row of ``asset`` in the payload. ``iv`` is NaN when there is
    none, or when the quote is more than ``MAX_IV_AGE_DAYS`` older than the data cutoff ``end`` (EVZ ended in 2023:
    its last quote is never shown as today's implied vol); source and origin are kept to say so."""
    rows = [r for r in payload.get("implied") or [] if r.get("asset") == asset and A._finite(r.get("iv"))]
    if not rows:
        return math.nan, "", ""
    last = max(rows, key=lambda row: str(row.get("origin")))
    src, origin = str(last.get("source") or "IV"), _date(last.get("origin"))
    if end is not None and (pd.Timestamp(end) - pd.Timestamp(last.get("origin"))).days > MAX_IV_AGE_DAYS:
        return math.nan, src, origin
    return float(last["iv"]), src, origin


def _window_label(fc: pd.DataFrame, h: str) -> str:
    g = fc[fc["horizon"] == h]
    if g.empty:
        return A.HORIZON_LABELS.get(h, h)
    r = g.iloc[0]
    first, last, n = _date(r.get("window_first")), _date(r.get("window_last")), r.get("n_t")
    if h == "1d":
        return f"1 day · next session {first}"
    n_txt = f"{int(n)} sessions" if A._finite(n) else "? sessions"
    return f"{A.HORIZON_LABELS.get(h, h)} · {first} – {last} ({n_txt})"


def tomorrow_vol_figure(fc: pd.DataFrame, asset: str, realised: dict[str, tuple[float, int]],
                        iv: tuple[float, str, str]) -> go.Figure:
    """Dot plot of the annualised forecast vol per model, one panel per horizon on one shared scale; COMBO is
    larger and value-labelled; trailing realised vol and the implied vol are reference lines."""
    if fc.empty:
        return A.empty_figure(f"No forecasts for {asset} in the latest payload.")
    horizons = A.order_horizons(fc["horizon"].unique())
    models = A.order_models(fc["model"].unique())
    label = {m: f"{m} (primary)" if m == PRIMARY else m for m in models}
    fig = make_subplots(rows=1, cols=len(horizons), shared_yaxes=True, horizontal_spacing=0.035,
                        subplot_titles=[_window_label(fc, h) for h in horizons])
    xmax = 0.0
    for k, h in enumerate(horizons, start=1):
        g = fc[fc["horizon"] == h].drop_duplicates("model").set_index("model")
        g = g.reindex([m for m in models if m in g.index])
        v = g["vol_ann"].astype(float)
        xmax = max(xmax, float(np.nanmax(v.to_numpy())) if v.notna().any() else 0.0)
        primary = [m == PRIMARY for m in g.index]
        fig.add_trace(go.Scatter(
            x=v, y=[label[m] for m in g.index], mode="markers+text", showlegend=False, name=h,
            text=[f"{x:.1f}%" if p and A._finite(x) else "" for x, p in zip(v, primary, strict=True)],
            textposition="middle right", textfont=dict(color=A.INK, size=12),
            marker=dict(color=[A.model_color(m) for m in g.index], size=[14 if p else 10 for p in primary],
                        line=dict(color=A.SURFACE, width=2)),
            customdata=np.column_stack([g["F"].astype(float), g["n_t"].astype(float)]),
            hovertemplate="%{y}: %{x:.1f}% p.a.<br>F = %{customdata[0]:.4f} %² over %{customdata[1]:.0f} "
                          "sessions<extra>" + h + "</extra>",
        ), row=1, col=k)
        r = realised.get(h, (math.nan, 0))[0]
        if A._finite(r):
            xmax = max(xmax, r)
            fig.add_vline(x=r, line=dict(color=A.INK2, width=1.5, dash="dot"), row=1, col=k)
        if A._finite(iv[0]):
            xmax = max(xmax, iv[0])
            fig.add_vline(x=iv[0], line=dict(color=P.MODEL_COLORS["IV"], width=1.5, dash="dash"), row=1, col=k)
    real_txt = ", ".join(f"{h} {realised[h][0]:.1f}%" for h in horizons if A._finite(realised.get(h, (math.nan,))[0]))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=A.INK2, width=1.5, dash="dot"),
                             name=f"Realised, trailing window ({real_txt or 'n/a'})", hoverinfo="skip"))
    if A._finite(iv[0]):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                                 line=dict(color=P.MODEL_COLORS["IV"], width=1.5, dash="dash"),
                                 name=f"Implied vol {iv[1]}, 30-day ({iv[0]:.1f}% on {iv[2]})", hoverinfo="skip"))
    A._style(fig, f"{A.asset_label(asset)} — volatility forecasts from the latest data cutoff",
             "Annualised for display only (√(F/n·ann)), % per year · one shared scale for the three horizons",
             height=190 + 26 * len(models))
    fig.update_layout(hovermode="closest", margin=dict(l=124, r=24, t=120, b=40),
                      legend=dict(y=0, yref="container", yanchor="bottom", x=0))
    fig.update_annotations(font=dict(size=12, color=A.INK2))
    fig.update_xaxes(range=[0, max(xmax, 1e-9) * 1.18], ticksuffix="%", showgrid=True, gridcolor=A.GRID)
    fig.update_yaxes(categoryorder="array", categoryarray=[label[m] for m in reversed(models)], showgrid=False,
                     tickfont=dict(color=A.INK2, size=12))
    return fig


VOL_COLUMNS: list[A.Column] = [("model", "Model", None, False)] + [
    (f"v_{h}", f"{A.HORIZON_LABELS[h]} (% p.a.)", _fmt_vol, True) for h in ("1d", "1w", "1m")
]


def tomorrow_vol_table(fc: pd.DataFrame, realised: dict[str, tuple[float, int]],
                       iv: tuple[float, str, str]) -> html.Div:
    """Table twin of the dot plot (+ the realised and implied reference rows, in italics)."""
    if fc.empty:
        return html.Div()
    wide = fc.pivot_table(index="model", columns="horizon", values="vol_ann", aggfunc="first")
    wide = wide.reindex(A.order_models(wide.index))
    rows = pd.DataFrame({"model": wide.index,
                         **{f"v_{h}": (wide[h] if h in wide else pd.Series(np.nan, index=wide.index)).to_numpy()
                            for h in ("1d", "1w", "1m")}})
    extra = [{"model": "Realised (trailing window)", **{f"v_{h}": realised.get(h, (math.nan, 0))[0]
                                                        for h in ("1d", "1w", "1m")}}]
    if A._finite(iv[0]):
        extra.append({"model": f"Implied vol ({iv[1]}, 30-day)", "v_1d": iv[0], "v_1w": iv[0], "v_1m": iv[0]})
    elif iv[1]:
        extra.append({"model": f"Implied vol: no current quote (last {iv[1]} {iv[2]})"})
    rows = pd.concat([rows, pd.DataFrame(extra)], ignore_index=True)
    models = set(fc["model"])
    return A.html_table(rows, VOL_COLUMNS, row_class=lambda r: "ref" if r["model"] == PRIMARY else (
        "descr" if r["model"] not in models else ""))


def tomorrow_risk_figure(rk: pd.DataFrame, asset: str) -> go.Figure:
    """Dot plot of next-session VaR99 / VaR97.5 / ES97.5 per risk model (colour = model, symbol = measure)."""
    if rk.empty:
        return A.empty_figure(f"No next-session VaR/ES for {asset} in the latest payload.")
    models = A.order_risk_models(rk["model"].unique())
    rk = rk.drop_duplicates("model").set_index("model").reindex(models)
    label = {m: f"{m} (primary)" if m == PRIMARY_RISK else m for m in models}
    fig = go.Figure()
    xmax = 0.0
    for col, name, symbol in RISK_MEASURES:
        v = pd.to_numeric(rk[col], errors="coerce") if col in rk else pd.Series(np.nan, index=rk.index)
        xmax = max(xmax, float(np.nanmax(v.to_numpy())) if v.notna().any() else 0.0)
        fig.add_trace(go.Scatter(
            x=v, y=[label[m] for m in models], mode="markers", showlegend=False, name=name,
            marker=dict(color=[A.model_color(m) for m in models], size=11, symbol=symbol,
                        line=dict(color=A.SURFACE, width=2)),
            hovertemplate="%{y}: " + name + " %{x:.2f}%<extra></extra>",
        ))
    for _, name, symbol in RISK_MEASURES:  # legend keys for the symbols, in neutral ink
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name, hoverinfo="skip",
                                 marker=dict(color=A.INK2, size=10, symbol=symbol)))
    date = _date(rk["date"].dropna().iloc[0]) if "date" in rk and rk["date"].notna().any() else "—"
    A._style(fig, f"{A.asset_label(asset)} — next-session VaR and ES ({date})",
             "Loss in % of a 1-unit long position, μ = 0 · colour = risk model, symbol = measure",
             height=170 + 34 * len(models))
    fig.update_layout(hovermode="closest", margin=dict(l=130, r=24, t=96, b=40),
                      legend=dict(y=0, yref="container", yanchor="bottom", x=0))
    fig.update_xaxes(range=[0, max(xmax, 1e-9) * 1.12], ticksuffix="%", showgrid=True, gridcolor=A.GRID)
    fig.update_yaxes(categoryorder="array", categoryarray=[label[m] for m in reversed(models)], showgrid=False,
                     tickfont=dict(color=A.INK2, size=12))
    return fig


RISK_COLUMNS: list[A.Column] = [
    ("model", "Risk model", None, False),
    ("date", "Session", _date, False),
    ("sigma", "σ (% per session)", lambda v: A.fmt_num(v, 3), True),
    ("var99", "VaR 99% (%)", lambda v: A.fmt_num(v, 2), True),
    ("var975", "VaR 97.5% (%)", lambda v: A.fmt_num(v, 2), True),
    ("es975", "ES 97.5% (%)", lambda v: A.fmt_num(v, 2), True),
]


def tomorrow_risk_table(rk: pd.DataFrame) -> html.Div:
    if rk.empty:
        return html.Div()
    order = {m: i for i, m in enumerate(A.order_risk_models(rk["model"]))}
    rk = rk.sort_values("model", key=lambda s: s.map(order))
    return html.Div([A.html_table(rk, RISK_COLUMNS, row_class=lambda r: "ref" if r["model"] == PRIMARY_RISK else ""),
                     html.P(RISK_NOTE, className="note")])


def _check_item(name: str, v: Any) -> html.Li:
    """One pre-recording check of the payload (reproduction, data consistency, ...) as a chip with its evidence."""
    if isinstance(v, bool):
        return html.Li(_state_chip(v, f"{name}: {'yes' if v else 'no'}"))
    if isinstance(v, dict):
        if name == "data_consistency" and isinstance(v.get("check_against_sealed"), dict):
            c = v["check_against_sealed"]
            rows = {k: (c.get(k) or {}).get("rows_compared") for k in ("dev", "holdout", "implied")}
            txt = ", ".join(f"{k} {n:,} rows" for k, n in rows.items() if isinstance(n, int))
            return html.Li(_state_chip(c.get("ok"), f"live data == sealed data ({txt or 'see verification'})"))
        ok = next((v[k] for k in ("passed", "ok") if isinstance(v.get(k), bool)), None)
        if "rows" in v:
            txt = (f"{name}: {v['rows']:,} sealed rows reproduced, max rel. diff {_num(v.get('max_rel_diff')):.1e}"
                   if isinstance(v["rows"], int) else f"{name}: {v['rows']}")
        else:
            txt = f"{name}: " + ", ".join(f"{k}={x}" for k, x in v.items() if not isinstance(x, (dict, list)))[:160]
        return html.Li(_state_chip(ok, txt))
    if isinstance(v, list):
        return html.Li(f"{name}: {', '.join(map(str, v)) if v else 'none'}")
    return html.Li(f"{name}: {v}")


def _hm(t: Any) -> str:
    return f"{t:%Y-%m-%d %H:%M}" if t is not None else "?"


def timing_chip(t: dict | None) -> html.Span:
    """When the payload was written relative to the asset's next session (``cli.payload_timing``): green before it
    opened, grey during it (data end at the previous session), red after it closed (its outcome already existed)."""
    st = (t or {}).get("state", "unknown")
    o, c = (t or {}).get("open_utc"), (t or {}).get("close_utc")
    if st == "after_close" and not (t or {}).get("rows"):
        return _state_chip(None, f"no forecast: the session had closed ({_hm(c)} UTC) when the run was written")
    text = {
        "before_open": f"recorded before it opened ({_hm(o)} UTC)",
        "in_session": f"recorded during the session (opened {_hm(o)}, closes {_hm(c)} UTC); data end at the previous "
                      "session",
        "after_close": f"recorded AFTER the session closed ({_hm(c)} UTC): its outcome already existed - not "
                       "forward-test evidence",
    }.get(st, "recording time or session hours n/a")
    return _state_chip(TIMING_LEVEL.get(st), text)


def run_times(payload: dict, entry: dict) -> list:
    """Start time (``run_utc``) and write time (``checks.computed_utc``, the machine clock) of a run; the later one
    is the recording time every timing statement uses. A start time after the write time is flagged."""
    start = CLI.utc_time(payload.get("run_utc") or entry.get("run_utc"))
    wrote = CLI.utc_time((payload.get("checks") or {}).get("computed_utc"))
    out: list = [html.Code(str(payload.get("run_id") or entry.get("run_id") or "—")),
                 f" · started {_hm(start)} UTC" if start else " · start time n/a"]
    if wrote is None:
        out += [" ", _state_chip(None, "write time not in the payload: timing judged by the start time")]
    elif start is not None and wrote < start - CLI.CLOCK_SKEW:
        out += [f" · written {_hm(wrote)} UTC ", _state_chip(False, "start time LATER than the machine clock at "
                                                                     "writing: post-dated")]
    else:
        gap = (wrote - start).total_seconds() / 60 if start else math.nan
        out.append(f" · written {_hm(wrote)} UTC (machine clock"
                   + (f", {gap:.0f} min after the start)" if math.isfinite(gap) else ")"))
    if (payload.get("checks") or {}).get("replayed_now"):
        out += [" ", _state_chip(False, "replayed run time (test sandbox): not a forward-test record")]
    return out


def tomorrow_meta(live: LiveData, entry: dict, payload: dict, asset: str, fc: pd.DataFrame) -> html.Dl:
    frozen = payload.get("frozen") or {}
    data = payload.get("data") or {}
    d = data.get(asset) or {}
    timing = CLI.payload_timing(payload).get(asset)
    nxt = d.get("next_session") or (timing or {}).get("session")
    if not nxt and "window_first" in fc:
        w = fc.loc[fc["horizon"] == "1d", "window_first"].dropna()
        nxt = w.iloc[0] if len(w) else None
    seal = frozen.get("seal_ok")
    chain_ok, chain_msg = live.chain_status()
    stamp_ok, stamp_msg = live.stamp_status(live.payload_file(entry))
    hash_ok = live.payload_hash_ok(entry)
    cutoffs = ", ".join(f"{a} {_date((data.get(a) or {}).get('last_session'))}" for a in payload_assets(payload))
    checks = payload.get("checks") or {}
    pairs = [
        ("Next session", [_date(nxt), " ", timing_chip(timing)] if nxt else "—"),
        ("Data cutoff", f"{_date(d.get('last_session'))} · all assets: {cutoffs}"),
        ("Stale data", _state_chip(False, "stale — no forecast for this asset") if d.get("stale") else "no"),
        ("Run", run_times(payload, entry)),
        ("Frozen code", [html.Code(_short(frozen.get("code_sha"))), " ",
                         _state_chip(None if seal is None else bool(seal),
                                     "seal intact" if seal else ("SEAL BROKEN" if seal is False else "seal n/a")),
                         f" · holdout opened {frozen.get('holdout_opened_utc') or '—'}"]),
        ("Live code", html.Code(_short(payload.get("live_code_sha")))),
        ("Payload SHA-256", [html.Code(_short(entry.get("payload_sha256"), 16)), " ",
                             _state_chip(hash_ok, "file matches the ledger entry" if hash_ok else (
                                 "FILE DIFFERS from the ledger entry" if hash_ok is False else "no ledger entry"))]),
        ("Ledger chain", _state_chip(chain_ok, chain_msg)),
        ("Timestamp", _state_chip(stamp_ok, stamp_msg)),
        ("Pre-recording checks", html.Ul([_check_item(k, v) for k, v in checks.items()], className="plain")
         if isinstance(checks, dict) and checks else "—"),
    ]
    return _kv(pairs)


def tomorrow_view(live: LiveData, asset: str | None) -> tuple:
    """Tomorrow-tab callback: (vol figure, vol table, risk figure, risk table, meta panel)."""
    entry, payload = live.latest()
    if entry is None or payload is None:
        msg = (NO_RUN if entry is None
               else f"The payload of the latest ledger entry is missing: {live.payload_file(entry)}")
        return A.empty_figure(msg), html.Div(), A.empty_figure(msg), html.Div(), _empty_state(msg)
    asset = _pick_asset(asset, payload_assets(payload))
    if asset is None:
        msg = "The latest payload holds no assets."
        return A.empty_figure(msg), html.Div(), A.empty_figure(msg), html.Div(), _empty_state(msg)
    fc, rk = tomorrow_frames(payload, asset)
    end = ((payload.get("data") or {}).get(asset) or {}).get("last_session")
    daily = live.daily(asset)
    realised = {h: trailing_realised(daily, asset, end, h) for h in HORIZON_DAYS}
    iv = implied_now(payload, asset, end)
    return (tomorrow_vol_figure(fc, asset, realised, iv), tomorrow_vol_table(fc, realised, iv),
            tomorrow_risk_figure(rk, asset), tomorrow_risk_table(rk), tomorrow_meta(live, entry, payload, asset, fc))


TIMELINE_SESSIONS = 60  # realised history shown before the data cutoff
TIMELINE_MODELS = (PRIMARY, "HAR", "GJR")


def timeline_figure(daily: pd.DataFrame, recent: pd.DataFrame, fc: pd.DataFrame, asset: str, end: Any,
                    iv: tuple[float, str, str]) -> go.Figure:
    """Recent days and the next session on one time axis: realised daily vol up to the data cutoff, the 1d forecast
    each model made the session before, then the next-session forecasts and the 1-week / 1-month COMBO averages
    drawn over their future windows (annualised for display: sqrt(variance per session * ann))."""
    if daily.empty or end is None:
        return A.empty_figure(f"No live data for {asset} yet.")
    ann = annualisation(asset)
    t_end = pd.Timestamp(end)
    hist = daily[daily["session_date"] <= t_end].tail(TIMELINE_SESSIONS)
    shown = set(hist["session_date"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["session_date"], y=np.sqrt(pd.to_numeric(hist["tv"], errors="coerce") * ann), mode="lines+markers",
        name="Realised (what really happened)", line=dict(color=A.INK2, width=1.5), marker=dict(size=4),
        hovertemplate="%{x|%a %d %b %Y}: realised %{y:.1f}% p.a.<extra></extra>"))
    sessions = daily["session_date"].reset_index(drop=True)
    next_of = dict(zip(sessions.iloc[:-1], sessions.iloc[1:], strict=True))
    for m in TIMELINE_MODELS:
        g = recent[recent["model"] == m].assign(target=lambda x: x["origin"].map(next_of)).dropna(subset=["target"])
        g = g[g["target"].isin(shown)].sort_values("target")
        if g.empty:
            continue
        primary = m == PRIMARY
        fig.add_trace(go.Scatter(
            x=g["target"], y=np.sqrt(g["F"].astype(float) * ann), mode="lines",
            name=f"{m}: forecast made the day before",
            line=dict(color=A.model_color(m), width=2.6 if primary else 1.3, dash=None if primary else "dot"),
            hovertemplate="%{x|%a %d %b}: " + m + " forecast %{y:.1f}% p.a.<extra></extra>"))
    future_end = t_end
    star = None
    if not fc.empty:
        for m in reversed(TIMELINE_MODELS):  # the primary model is drawn last, on top
            r = fc[(fc["model"] == m) & (fc["horizon"] == "1d")]
            if r.empty or not A._finite(r["vol_ann"].iloc[0]):
                continue
            x, v = pd.Timestamp(r["window_first"].iloc[0]), float(r["vol_ann"].iloc[0])
            future_end = max(future_end, x)
            primary = m == PRIMARY
            if primary:
                star = (x, v)
            fig.add_trace(go.Scatter(
                x=[x], y=[v], mode="markers", name=f"{m}: next session", showlegend=primary,
                text=[f"next session {x:%a %d %b}: {v:.1f}%"] if primary else None,
                marker=dict(color=A.model_color(m), size=18 if primary else 9, symbol="star" if primary else "circle",
                            line=dict(color=A.SURFACE, width=1.5)),
                hovertemplate="%{x|%a %d %b}: " + m + " next-session forecast %{y:.1f}% p.a.<extra></extra>"))
        for h, dash_ in (("1w", "dash"), ("1m", "dashdot")):
            r = fc[(fc["model"] == PRIMARY) & (fc["horizon"] == h)]
            if r.empty or not A._finite(r["vol_ann"].iloc[0]):
                continue
            x0, x1 = pd.Timestamp(r["window_first"].iloc[0]), pd.Timestamp(r["window_last"].iloc[0])
            v = float(r["vol_ann"].iloc[0])
            future_end = max(future_end, x1)
            word = A.HORIZON_LABELS[h].split()[-1]
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[v, v], mode="lines", line=dict(color=A.model_color(PRIMARY), width=2, dash=dash_),
                name=f"{PRIMARY}: next {word} ({x0:%d %b} - {x1:%d %b}): {v:.1f}%",
                hovertemplate=PRIMARY + " " + A.HORIZON_LABELS[h] + " average %{y:.1f}% p.a.<extra></extra>"))
    if A._finite(iv[0]) and future_end > t_end:
        fig.add_trace(go.Scatter(
            x=[t_end, future_end], y=[iv[0], iv[0]], mode="lines",
            line=dict(color=P.MODEL_COLORS["IV"], width=1.2, dash="dot"),
            name=f"Implied vol {iv[1]} on {iv[2]}: {iv[0]:.1f}% (the market's 30-day forecast)", hoverinfo="skip"))
    if future_end > t_end:
        fig.add_shape(type="rect", xref="x", yref="paper", x0=t_end, x1=future_end + pd.Timedelta(days=1), y0=0, y1=1,
                      fillcolor=A.GRID, opacity=0.35, line_width=0, layer="below")
    fig.add_shape(type="line", xref="x", yref="paper", x0=t_end, x1=t_end, y0=0, y1=1,
                  line=dict(color=A.MUTED, width=1, dash="dot"))
    fig.add_annotation(x=t_end, y=1, xref="x", yref="paper", text=f"data to {t_end:%d %b %Y}", showarrow=False,
                       xanchor="right", yanchor="bottom", font=dict(color=A.INK2, size=11))
    if star is not None:
        fig.add_annotation(x=star[0], y=star[1], text=f"<b>next session {star[0]:%a %d %b}: {star[1]:.1f}%</b>",
                           showarrow=True, arrowhead=0, arrowcolor=A.INK2, ax=40, ay=-48, xanchor="left",
                           font=dict(color=A.INK, size=12), bgcolor=A.SURFACE)
    A._style(fig, f"{A.asset_label(asset)} - recent days and the forecast ahead",
             "Left of the dotted line: realised volatility and the forecast made the day before. Shaded: the future "
             "(next session = star, next week and month). % per year, annualised for display", height=560)
    fig.update_layout(hovermode="x unified", margin=dict(b=150),
                      legend=dict(orientation="h", yref="paper", yanchor="top", y=-0.16, x=0))
    fig.update_yaxes(ticksuffix="%", rangemode="tozero")
    return fig


def timeline_view(live: LiveData, asset: str | None) -> go.Figure:
    """Tomorrow-tab timeline callback."""
    entry, payload = live.latest()
    if entry is None or payload is None:
        return A.empty_figure(NO_RUN if entry is None else "The payload of the latest ledger entry is missing.")
    asset = _pick_asset(asset, payload_assets(payload))
    if asset is None:
        return A.empty_figure("The latest payload holds no assets.")
    fc, _ = tomorrow_frames(payload, asset)
    end = ((payload.get("data") or {}).get(asset) or {}).get("last_session")
    return timeline_figure(live.daily(asset), live.recent_forecasts(asset), fc, asset, end,
                           implied_now(payload, asset, end))


# --------------------------------------------------------------------------------------------- forward test
def next_due(live: LiveData, scores: pd.DataFrame, risk: pd.DataFrame, asset: str | None) -> str | None:
    """Earliest day a still unscored forecast window of ``asset`` closes (``score.pending_rows``)."""
    try:
        pend = SC.pending_rows(live.payloads(), scores, risk)
    except (KeyError, TypeError, ValueError):
        return None
    pend = pend[(pend["kind"] == "forecast") & ((pend["asset"] == asset) if asset else True)]
    due = pend["due"].min() if len(pend) else None
    return None if due is None or pd.isna(due) else pd.Timestamp(due).strftime("%Y-%m-%d")


def forward_frame(scores: pd.DataFrame) -> pd.DataFrame:
    """Headline scores (``score.headline_scores``: status ok, recorded ex ante, earliest run per origin) with the
    annualised forecast and realised vol for display (per-session means ``Fbar`` / ``ybar``)."""
    h = SC.headline_scores(scores) if len(scores) else scores.iloc[0:0]
    if h.empty:
        return h.assign(vol_fc=pd.Series(dtype=float), vol_real=pd.Series(dtype=float))
    ann = h["asset"].map(lambda a: annualisation(str(a))).astype(float)
    return h.assign(origin=pd.to_datetime(h["origin"]),
                    vol_fc=np.sqrt(h["Fbar"].astype(float) * ann), vol_real=np.sqrt(h["ybar"].astype(float) * ann))


def forward_figure(head: pd.DataFrame, asset: str, horizon: str, models: Sequence[str]) -> go.Figure:
    """Forecast vs realised volatility per scored origin (annualised for display only)."""
    s = head[(head["asset"] == asset) & (head["horizon"] == horizon)]
    if s.empty:
        return A.empty_figure(f"No scored {A.HORIZON_LABELS.get(horizon, horizon)} forecasts for {asset} yet.")
    ms = [m for m in A.order_models(models) if (s["model"] == m).any()] or A.order_models(s["model"].unique())[:2]
    real = s.drop_duplicates("origin").sort_values("origin")
    fig = go.Figure(go.Bar(
        x=real["origin"], y=real["vol_real"], name="Realised", marker=dict(color=A._hex_rgba(P.REALIZED_COLOR, 0.85),
                                                                        line=dict(width=0)),
        hovertemplate="realised %{y:.1f}%<extra></extra>"))
    for m in ms:
        g = s[s["model"] == m].sort_values("origin")
        fig.add_trace(go.Scatter(x=g["origin"], y=g["vol_fc"], name=m, mode="lines+markers",
                                 line=dict(color=A.model_color(m), width=2.4 if m == PRIMARY else 1.6),
                                 marker=dict(size=8, line=dict(color=A.SURFACE, width=2)),
                                 hovertemplate=f"{m} %{{y:.1f}}%<extra></extra>"))
    A._style(fig, f"{A.asset_label(asset)} · {A.HORIZON_LABELS.get(horizon, horizon)} ahead — forward test",
             f"Forecast recorded at origin t vs realised over its target window, annualised % per year (display "
             f"only) · {real['origin'].nunique()} scored origins", height=460)
    fig.update_layout(hovermode="x unified", bargap=0.35)
    fig.update_yaxes(title_text="Annualised volatility (%)", rangemode="tozero", ticksuffix="%")
    fig.update_xaxes(hoverformat="%Y-%m-%d")
    return fig


def ratio_rows(scores: pd.DataFrame, asset: str) -> pd.DataFrame:
    """``score.ratio_table`` (the frozen leaderboard on common origins of the headline scores) for one asset."""
    try:
        rt = SC.ratio_table(scores)
    except (KeyError, ValueError):
        return pd.DataFrame()
    if rt.empty:
        return rt
    rt = rt[rt["asset"] == asset].copy()
    rt["_h"] = rt["horizon"].map(A._horizon_rank)
    return rt.sort_values(["_h", "qlike_ratio", "model"]).drop(columns="_h").reset_index(drop=True)


FORWARD_COLUMNS: list[A.Column] = [
    ("horizon", "Horizon", None, False),
    ("model", "Model", None, False),
    ("qlike_ratio", "QLIKE ratio vs HAR", A.fmt_num, True),
    ("qlike", "Mean QLIKE", lambda v: A.fmt_num(v, 4), True),
    ("n", "Common scored origins", A.fmt_int, True),
]

VAR_COLUMNS: list[A.Column] = [
    ("model", "Risk model", None, False),
    ("N", "Scored sessions", A.fmt_int, True),
    ("b99", "Breaches VaR 99%", A.fmt_int, True),
    ("exp99", "Expected", lambda v: A.fmt_num(v, 2), True),
    ("p99", "Kupiec p (99%)", A.fmt_p, True),
    ("b975", "Breaches VaR 97.5%", A.fmt_int, True),
    ("exp975", "Expected", lambda v: A.fmt_num(v, 2), True),
    ("p975", "Kupiec p (97.5%)", A.fmt_p, True),
]


def var_rows(risk: pd.DataFrame, asset: str) -> pd.DataFrame:
    """``score.risk_table`` (breaches vs expected, Kupiec p) for one asset, risk models in display order."""
    if risk.empty:
        return pd.DataFrame(columns=[c[0] for c in VAR_COLUMNS])
    rt = SC.risk_table(risk)
    rt = rt[rt["asset"] == asset]
    order = {m: i for i, m in enumerate(A.order_risk_models(rt["model"]))}
    return rt.sort_values("model", key=lambda s: s.map(order)).reset_index(drop=True)


def late_scores(scores: pd.DataFrame, risk: pd.DataFrame) -> tuple[int, int]:
    """(forecast rows, VaR rows) scored ``ok`` but recorded after their first target session had closed, or at an
    unknown time (``score.py``: ``before_close`` from ``recorded_utc`` = the latest of the payload's clocks, as
    :func:`volrisk_live.cli.recorded_utc`). They stay in the score files; ``score.headline_scores`` / ``risk_table``
    leave them out, as ``reports/live/forward_test.md`` does."""
    def n(df: pd.DataFrame, need_ex_ante: bool) -> int:
        if df.empty or "before_close" not in df:
            return 0
        ok = df[df["status"] == SC.OK]
        mask = ~_truthy(ok["before_close"])
        if need_ex_ante and "ex_ante" in ok:
            mask &= _truthy(ok["ex_ante"])
        return int(mask.sum())

    return n(scores, True), n(risk, False)


def forward_status(live: LiveData, head: pd.DataFrame, asset: str | None, due: str | None,
                   late: tuple[int, int] = (0, 0)) -> html.Div:
    """Status line of the forward test. ``head`` holds only forecasts written before their first target session
    closed (``score.headline_scores``); how many of them before it opened, and how many scored rows were left out
    because they were written after the close (``late``), are stated next to it."""
    runs = live.runs()
    if not runs:
        return _empty_state(NO_RUN)
    first = runs[0].get("run_utc") or runs[0].get("run_id")
    chain_ok, chain_msg = live.chain_status()
    n_asset = int((head["asset"] == asset).sum()) if asset and len(head) else 0
    timing = ""
    if len(head) and "before_open" in head:
        n_open = int(_truthy(head["before_open"]).sum())
        timing = (f" · written before their first target session opened: {n_open:,}, during it (from data up to the "
                  f"previous session): {len(head) - n_open:,}")
    line = html.P([html.Strong("Forward test — forecasts recorded before their outcomes, "), f"since {first} (UTC). ",
                   f"{len(runs)} run(s) in the ledger · {len(head):,} forecasts scored",
                   f" ({n_asset:,} for {asset})" if asset else "", timing, " · ledger chain: ",
                   _state_chip(chain_ok, chain_msg)])
    parts: list = [line]
    if late[0] or late[1]:
        parts.append(html.P(_state_chip(False, (
            f"{late[0]:,} forecast and {late[1]:,} VaR row(s) were written after their first target session had "
            "closed, or at an unknown time: their outcome already existed, so they are not forward-test evidence "
            "and are left out of every table below (they stay in the score files)")), className="note"))
    if n_asset == 0:
        when = f"first scores after {due}" if due else "first scores once a recorded forecast window has passed"
        return html.Div([*parts, _empty_state([html.Strong("No scored forecasts yet"), f" — {when}. A window is "
                                               "scored by the first daily run after it closes and its data exist."])])
    return html.Div([*parts, html.P(f"Next window to close: {due}." if due else "Every recorded window is scored.",
                                    className="note"),
                     html.P(f"Full report: {(live.reports_dir / paths.FORWARD_MD.name).as_posix()}", className="note")])


def forward_view(live: LiveData, asset: str | None, horizon: str | None, models: Sequence[str] | None) -> tuple:
    """Forward-test callback: (status, forecast-vs-realised figure, ratio heatmap, ratio table, VaR table)."""
    scores, risk = live.scores(), live.risk_scores()
    head = forward_frame(scores)
    if asset is None:
        asset = _pick_asset(None, payload_assets(live.latest()[1]))
    horizon = horizon or "1d"
    due = next_due(live, scores, risk, asset)
    status = forward_status(live, head, asset, due, late_scores(scores, risk))
    var = var_rows(risk, asset) if asset else pd.DataFrame()
    var_table = (A.html_table(var, VAR_COLUMNS, row_class=lambda r: "ref" if r["model"] == PRIMARY_RISK else "")
                 if len(var) else html.P("No scored VaR/ES yet.", className="note"))
    if not asset or not (head["asset"] == asset).any():
        msg = f"No scored forecasts yet — first scores after {due}" if due else "No scored forecasts yet"
        return status, A.empty_figure(msg), A.empty_figure(msg), html.P("Nothing scored yet.", className="note"), \
            var_table
    rows = ratio_rows(scores, asset)
    if rows.empty or rows["qlike_ratio"].isna().all():
        heat = A.empty_figure("No scored HAR forecasts on common origins yet.")
        table = html.P("—", className="note")
    else:
        heat = A.leaderboard_figure(None, asset, "forward", rows)  # the frozen heatmap on the forward rows
        heat.update_layout(title_text=f"{A.asset_label(asset)} — forward-test accuracy by horizon",
                           title_subtitle_text="Mean QLIKE ÷ HAR's on common scored origins (<1 = better than "
                                               "HAR, blue) · few scores = noisy")
        table = A.html_table(rows, FORWARD_COLUMNS, row_class=lambda r: "ref" if r["model"] == REF_MODEL else "")
    models = list(models or []) or list(DEFAULT_FORWARD_MODELS)
    return status, forward_figure(head, asset, horizon, models), heat, table, var_table


# --------------------------------------------------------------------------------------------- verification
def _scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return A.fmt_num(v, 6 if v != 0 and abs(v) < 1e-2 else 4)
    if isinstance(v, (dict, list)):
        s = json.dumps(v, default=str)
        return s if len(s) <= 200 else s[:199] + "…"
    return "—" if v is None else str(v)


def _evidence(ev: Any, depth: int = 0) -> Any:
    """Evidence of a check: dict → key/value list (one nested level), list of dicts → table, else text."""
    if ev is None or ev in ("", {}, []):
        return None
    if isinstance(ev, dict):
        return _kv([(str(k), _evidence(v, depth + 1) if isinstance(v, (dict, list)) and depth < 1 else _scalar(v))
                    for k, v in ev.items()])
    if isinstance(ev, list):
        if all(isinstance(x, dict) for x in ev):
            df = pd.DataFrame(ev[:50])
            more = html.P(f"… {len(ev) - 50} more rows in verification.json", className="note") if len(ev) > 50 \
                else None
            return html.Div([A.html_table(df, [(c, str(c), _scalar, False) for c in df.columns]), more])
        return ", ".join(_scalar(x) for x in ev[:50]) + (" …" if len(ev) > 50 else "")
    return html.P(str(ev), className="method")


def verification_view(live: LiveData) -> list:
    """Verification-tab callback: one row per LIVE_SPEC §7 check with its PASS / FAIL / SKIPPED badge, the method
    and the evidence numbers, from ``reports/live/verification.json``."""
    data = live.verification()
    checks = normalise_checks(data)
    if not checks:
        return [_empty_state(["No verification report yet — run ", html.Code("uv run python -m volrisk_live verify"),
                              f" (writes {live.verification_path.name} under {live.reports_dir.as_posix()})."])]
    counts = {s: sum(c["status"] == s for c in checks) for s in ("PASS", "FAIL", "SKIPPED")}
    other = len(checks) - sum(counts.values())
    meta = data if isinstance(data, dict) else {}
    when = next((meta[k] for k in ("generated_utc", "run_utc", "utc", "generated", "created_utc") if meta.get(k)),
                None)
    args = meta.get("args") if isinstance(meta.get("args"), dict) else {}
    args_txt = ", ".join(f"{k}={v}" for k, v in args.items())
    head = html.P([html.Strong(f"{len(checks)} checks: "),
                   badge("PASS"), f" {counts['PASS']}   ", badge("FAIL"), f" {counts['FAIL']}   ",
                   badge("SKIPPED"), f" {counts['SKIPPED']}" + (f" · {other} other" if other else ""),
                   html.Span(f" · generated {when}" if when else "", className="muted"),
                   html.Span(f" ({args_txt})" if args_txt else "", className="muted"),
                   html.Span(" · re-run: ", className="muted"), html.Code("uv run python -m volrisk_live verify")])
    rows: list = [head]
    for c in checks:
        title = c["title"] if c["id"] == c["title"] else f"{c['id']}. {c['title']}"
        rows.append(html.Div([
            html.Div(badge(c["status"], title=c["reason"] or None)),
            html.Div([
                html.H4(title),
                html.P(str(c["method"]), className="method") if c["method"] else None,
                html.P(f"Reason: {c['reason']}", className="reason") if c["reason"] else None,
                _evidence(c["evidence"]),
            ]),
        ], className=f"vrow v-{c['status'].lower()}"))
    return rows


# --------------------------------------------------------------------------------------------- layout and app
def live_meta_line(live: LiveData) -> str:
    runs = live.runs()
    if not runs:
        return "Live ledger: no forecast recorded yet (python -m volrisk_live daily)."
    last = runs[-1]
    return (f"Live ledger: {len(runs)} run(s), latest {last.get('run_id', '—')} "
            f"(recorded {last.get('run_utc') or '—'} UTC) · {project_relative(live.forecasts_dir)}")


def project_relative(path: Path | str) -> str:
    """``path`` relative to the project root when it lies inside it (no machine-specific prefix on screen)."""
    p = Path(path)
    try:
        return p.resolve().relative_to(paths.ROOT.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def relativize_layout(component, root: Path = paths.ROOT):
    """Replace the absolute project-root prefix in every text child of the (frozen) layout, in place."""
    prefixes = sorted({root.as_posix() + "/", str(root) + "\\", root.resolve().as_posix() + "/"}, key=len,
                      reverse=True)

    def fix(text: str) -> str:
        for pre in prefixes:
            text = text.replace(pre, "")
        return text

    kids = getattr(component, "children", None)
    if isinstance(kids, str):
        component.children = fix(kids)
    elif isinstance(kids, (list, tuple)):
        component.children = [fix(k) if isinstance(k, str) else relativize_layout(k, root) for k in kids]
    elif kids is not None and hasattr(kids, "to_plotly_json"):
        relativize_layout(kids, root)
    return component


def _graph(i: str) -> dcc.Graph:
    return dcc.Graph(id=i, config=A.GRAPH_CONFIG, figure=A.empty_figure("Loading…"))


def live_tabs(style: dict, selected: dict) -> tuple[dcc.Tab, dcc.Tab, dcc.Tab]:
    tomorrow = dcc.Tab(label="Tomorrow", value="tomorrow", style=style, selected_style=selected, children=[
        A._card("Recent days and the next session",
                "What really happened (realised volatility), what the models had forecast the day before, and the "
                "forecasts for the next session, week and month. Choose the asset above (the other controls apply "
                "to the historical tabs).", _graph("tomorrow-timeline")),
        A._card(None, None, _graph("tomorrow-vol"), html.H3("Forecast table"), html.Div(id="tomorrow-vol-table"),
                html.P(VOL_NOTE, className="note")),
        A._card(None, None, _graph("tomorrow-risk"), html.Div(id="tomorrow-risk-table")),
        A._card("How this forecast was recorded",
                "Latest payload of the forward-test ledger: the frozen models re-run on data up to the last complete "
                "UTC day, with the checks that ran before it was written.", html.Div(id="tomorrow-meta")),
    ])
    forward = dcc.Tab(label="Forward test", value="forward", style=style, selected_style=selected, children=[
        A._card("Forward test", FORWARD_SUB, html.Div(id="fwd-status")),
        A._card(None, None, _graph("fwd-graph"),
                html.P("Uses the asset, horizon and forecast-model controls above.", className="note")),
        A._card("Accuracy so far", "QLIKE ratio vs HAR per horizon and model on the scores so far (same table as "
                "reports/live/forward_test.md; no tests until enough forecasts are scored).",
                _graph("fwd-heatmap"), html.Div(id="fwd-table")),
        A._card("VaR breaches so far", "Next-session VaR from the ledger vs the realised return (when each run was "
                "written relative to its session: see the forward-test line above).",
                html.Div(id="fwd-var")),
    ])
    verify = dcc.Tab(label="Verification", value="verification", style=style, selected_style=selected, children=[
        A._card("Verification report",
                "Re-runnable checks that the results are real (LIVE_SPEC §7): frozen code, single holdout opening, raw "
                "data authenticity, independent sources, reproducibility, negative controls, no tuning after "
                "results, forward-test ledger.", html.Div(id="verify-panel")),
    ])
    return tomorrow, forward, verify


def add_live_tabs(layout: html.Div, live: LiveData) -> html.Div:
    """Insert the live tabs into the frozen layout (in memory): *Tomorrow* first and selected, *Forward test* and
    *Verification* after the frozen tabs; a live line in the header, a footer naming both sources and a refresh
    timer."""
    comps = list(layout._traverse())
    tabs = next((c for c in comps if isinstance(c, dcc.Tabs) and getattr(c, "id", None) == "tabs"), None)
    if tabs is None:
        raise RuntimeError("the frozen dashboard layout has no dcc.Tabs(id='tabs')")
    first = tabs.children[0]
    tomorrow, forward, verify = live_tabs(first.style, first.selected_style)
    tabs.children = [tomorrow, *tabs.children, forward, verify]
    tabs.value = "tomorrow"
    meta = html.P(live_meta_line(live), className="meta", id="live-meta")
    header = next((c for c in comps if isinstance(c, html.Header)), None)
    if header is not None:
        header.children = [*header.children, meta]
    footer = next((c for c in comps if isinstance(c, html.Footer)), None)
    if footer is not None:
        footer.children = ("Read-only viewer · historical tabs: the frozen pipeline results · Tomorrow, Forward test, "
                           "Verification: the live ledger and reports written by python -m volrisk_live daily / "
                           "verify · annualisation is for display only.")
    layout.children = [*([] if header is not None else [meta]), *layout.children,
                       dcc.Interval(id="live-refresh", interval=REFRESH_MS, n_intervals=0)]
    return layout


def create_app(results_dir: str | Path | None = None, reports_dir: str | Path | None = None,
               holdout_dir: str | Path | None = None, forecasts_dir: str | Path | None = None,
               live_reports_dir: str | Path | None = None, live_dir: str | Path | None = None) -> dash.Dash:
    """The frozen dashboard over ``results_dir`` / ``reports_dir`` plus the live tabs over ``forecasts_dir``
    (default ``forecasts/``), ``live_reports_dir`` (default ``reports/live``) and ``live_dir`` (``data/live``)."""
    app = A.create_app(results_dir=results_dir, reports_dir=reports_dir, holdout_dir=holdout_dir)
    live = LiveData(forecasts_dir, live_reports_dir, live_dir)
    app.title = TITLE
    app.index_string = A.INDEX_STRING.replace("</style>", EXTRA_CSS + "</style>", 1)
    add_live_tabs(app.layout, live)
    relativize_layout(app.layout)
    app.server.config["VOLRISK_LIVE"] = live

    @app.callback(Output("live-meta", "children"), Input("live-refresh", "n_intervals"))
    def _meta(_n):
        return live_meta_line(live)

    @app.callback(Output("tomorrow-vol", "figure"), Output("tomorrow-vol-table", "children"),
                  Output("tomorrow-risk", "figure"), Output("tomorrow-risk-table", "children"),
                  Output("tomorrow-meta", "children"),
                  Input("asset", "value"), Input("live-refresh", "n_intervals"))
    def _tomorrow(asset, _n):
        return tomorrow_view(live, asset)

    @app.callback(Output("tomorrow-timeline", "figure"), Input("asset", "value"),
                  Input("live-refresh", "n_intervals"))
    def _timeline(asset, _n):
        return timeline_view(live, asset)

    @app.callback(Output("fwd-status", "children"), Output("fwd-graph", "figure"), Output("fwd-heatmap", "figure"),
                  Output("fwd-table", "children"), Output("fwd-var", "children"),
                  Input("asset", "value"), Input("horizon", "value"), Input("models", "value"),
                  Input("live-refresh", "n_intervals"))
    def _forward(asset, horizon, models, _n):
        return forward_view(live, asset, horizon, models)

    @app.callback(Output("verify-panel", "children"), Input("live-refresh", "n_intervals"))
    def _verify(_n):
        return verification_view(live)

    return app


def resolve_dirs(results_dir: Path | None, live_dir: Path | None, demo: bool) -> tuple[Path | None, Path | None, str]:
    """(results_dir, live_dir, note): the demo bundle is used with ``--demo``, or automatically when the pipeline's
    results are missing (a fresh clone) and the bundle exists; explicit directories always win."""
    from volrisk import config as C

    fresh = results_dir is None and not (C.RESULTS / "forecasts.parquet").exists()
    if not (demo or fresh) or not (paths.DEMO_RESULTS / "forecasts.parquet").exists():
        return results_dir, live_dir, ""
    note = ("\n  showing the demo bundle in demo/ (snapshot of the real results; "
            "run the pipeline to rebuild data/ - see README)")
    return results_dir or paths.DEMO_RESULTS, live_dir or paths.DEMO_LIVE, note


def main(argv: list[str] | None = None, open_browser: bool = False) -> None:
    """Serve the live dashboard; the browser is opened only by the command line entry point (never in tests)."""
    p = argparse.ArgumentParser(prog="python -m volrisk_live.dashboard",
                                description="Frozen volrisk dashboard plus the live tabs (local only, read-only).")
    p.add_argument("--port", type=int, default=8050, help="port on 127.0.0.1 (default 8050)")
    p.add_argument("--debug", action="store_true", help="Dash debug mode (dev tools, auto-reload)")
    p.add_argument("--results-dir", type=Path, default=None, help="frozen results directory (default data/results)")
    p.add_argument("--reports-dir", type=Path, default=None, help="frozen reports directory (default reports)")
    p.add_argument("--forecasts-dir", type=Path, default=None, help="live ledger directory (default forecasts)")
    p.add_argument("--live-reports-dir", type=Path, default=None, help="live reports (default reports/live)")
    p.add_argument("--live-dir", type=Path, default=None, help="live data directory (default data/live)")
    p.add_argument("--demo", action="store_true",
                   help="read the committed demo bundle in demo/ (used automatically when data/results is missing)")
    p.add_argument("--no-browser", action="store_true", help="do not open the dashboard in the web browser")
    args = p.parse_args(argv)

    results_dir, live_dir, note = resolve_dirs(args.results_dir, args.live_dir, args.demo)
    app = create_app(results_dir=results_dir, reports_dir=args.reports_dir, forecasts_dir=args.forecasts_dir,
                     live_reports_dir=args.live_reports_dir, live_dir=live_dir)
    url = f"http://{HOST}:{args.port}/"
    print(f"\n  volrisk live dashboard running at {url}\n  (press Ctrl+C here to stop){note}\n", flush=True)
    if open_browser and not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    app.run(host=HOST, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main(open_browser=True)
