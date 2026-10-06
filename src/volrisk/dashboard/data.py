"""Read-only data access for the dashboard (SPEC §12): DuckDB views over the results parquet files.

One in-memory DuckDB connection holds a view per results file (``CREATE VIEW ... AS SELECT * FROM
read_parquet(...)``); nothing is ever written. Development results come from ``data/results/*.parquet``;
holdout results, once the one-time holdout run has produced them, from ``data/results/holdout/*.parquet``
(same file names). The sealed holdout *inputs* (``data/holdout/``) are never touched: the realized target
shown next to the forecasts is the ``y`` column of the same results directory's ``targets.parquet``.

Every query returns a pandas frame; a missing file gives an empty frame instead of an error, so the viewer
works with any subset of the four assets and with partially written result directories.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from volrisk import config as C

SPLITS = ("dev", "holdout")
# display-only annualisation (SPEC notation): crypto x365, SPX x252, EURUSD x260
ANNUALISATION = {"crypto": 365, "fx": 260, "xnys": 252}
# a holdout directory counts as "available" once any of these result files exists
_HOLDOUT_MARKERS = ("forecasts", "eval_leaderboard", "risk", "risk_risk_leaderboard")


def annualisation(asset: str) -> int:
    """Sessions per year used to annualise variances for display (never in a comparison)."""
    try:
        return ANNUALISATION[C.clock(asset)]
    except (KeyError, TypeError):
        return 365 if asset in C.CRYPTO else 252


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _sql_path(p: Path) -> str:
    return p.resolve().as_posix().replace("'", "''")


def _in_list(values: Iterable[str] | str | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        return [values]
    return [str(v) for v in values]


class ResultsData:
    """DuckDB views over one development results directory and (optionally) its holdout sibling.

    ``results_dir`` defaults to ``data/results``; ``holdout_dir`` to ``<results_dir>/holdout``;
    ``reports_dir`` (for the hypothesis tables) to ``reports``.
    """

    def __init__(
        self,
        results_dir: str | Path | None = None,
        reports_dir: str | Path | None = None,
        holdout_dir: str | Path | None = None,
    ) -> None:
        self.results_dir = Path(results_dir) if results_dir is not None else C.RESULTS
        self.holdout_dir = Path(holdout_dir) if holdout_dir is not None else self.results_dir / "holdout"
        self.reports_dir = Path(reports_dir) if reports_dir is not None else C.REPORTS
        for d in (self.results_dir, self.holdout_dir):
            if _inside(d, C.HOLDOUT):
                raise PermissionError(f"{d} is inside the sealed holdout directory; the dashboard never reads it")
        self._con = duckdb.connect(":memory:")
        self._lock = threading.Lock()
        self._views: dict[tuple[str, str], str] = {}
        self.skipped: dict[tuple[str, str], str] = {}  # files DuckDB cannot read -> reason (treated as missing)
        self._register()

    # ----------------------------------------------------------------------------------------- plumbing
    def dirs(self) -> dict[str, Path]:
        return {"dev": self.results_dir, "holdout": self.holdout_dir}

    def _register(self) -> None:
        """One view per results file. A file DuckDB cannot read is treated as missing: the pipeline writes an
        empty table as a zero-column parquet (e.g. ``eval_iv_mcs`` of the holdout run, where the IV MCS is not
        computed), which DuckDB refuses ("Need at least one non-root column")."""
        for split, d in self.dirs().items():
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.parquet")):
                name = f"{split}__{f.stem}"
                try:
                    self._con.execute(
                        f'CREATE OR REPLACE VIEW "{name}" AS SELECT * FROM read_parquet(\'{_sql_path(f)}\')'
                    )
                    self._con.execute(f'SELECT * FROM "{name}" LIMIT 0')
                except duckdb.Error as e:
                    self._con.execute(f'DROP VIEW IF EXISTS "{name}"')
                    self.skipped[(split, f.stem)] = str(e).splitlines()[0]
                    continue
                self._views[(split, f.stem)] = name

    def has(self, split: str, table: str) -> bool:
        return (split, table) in self._views

    def view(self, split: str, table: str) -> str:
        return f'"{self._views[(split, table)]}"'

    def columns(self, split: str, table: str) -> list[str]:
        if not self.has(split, table):
            return []
        return self.query(f"SELECT * FROM {self.view(split, table)} LIMIT 0").columns.tolist()

    def query(self, sql: str, params: Sequence | None = None) -> pd.DataFrame:
        with self._lock:
            cur = self._con.cursor()
            try:
                return cur.execute(sql, list(params or [])).df()
            finally:
                cur.close()

    def table(self, split: str, table: str, where: str = "", params: Sequence | None = None) -> pd.DataFrame:
        """``SELECT * FROM <split>/<table> [WHERE ...]``; empty frame when the file does not exist."""
        if not self.has(split, table):
            return pd.DataFrame()
        sql = f"SELECT * FROM {self.view(split, table)}" + (f" WHERE {where}" if where else "")
        return self.query(sql, params)

    def mtime(self, split: str, table: str = "forecasts") -> pd.Timestamp | None:
        p = self.dirs()[split] / f"{table}.parquet"
        return pd.Timestamp(p.stat().st_mtime, unit="s") if p.exists() else None

    # ----------------------------------------------------------------------------------------- catalogue
    def available_splits(self) -> list[str]:
        out = []
        if any(self.has("dev", t) for t in _HOLDOUT_MARKERS):
            out.append("dev")
        if any(self.has("holdout", t) for t in _HOLDOUT_MARKERS):
            out.append("holdout")
        return out

    def available_assets(self, split: str | None = None) -> list[str]:
        splits = [split] if split else list(SPLITS)
        found: set[str] = set()
        for s in splits:
            for t in ("forecasts", "risk", "eval_leaderboard"):
                if self.has(s, t):
                    found |= set(self.query(f"SELECT DISTINCT asset FROM {self.view(s, t)}")["asset"].dropna())
        known = [a for a in C.ASSETS if a in found]
        return known + sorted(found - set(known))

    def forecast_models(self, asset: str, horizon: str, split: str = "dev") -> list[str]:
        if not self.has(split, "forecasts"):
            return []
        df = self.query(
            f"SELECT DISTINCT model FROM {self.view(split, 'forecasts')} WHERE asset = ? AND horizon = ? AND split = ?",
            [asset, horizon, split],
        )
        return df["model"].tolist()

    def risk_models(self, asset: str, split: str = "dev") -> list[str]:
        if not self.has(split, "risk"):
            return []
        df = self.query(
            f"SELECT DISTINCT model FROM {self.view(split, 'risk')} WHERE asset = ? AND split = ?", [asset, split]
        )
        return df["model"].tolist()

    # ----------------------------------------------------------------------------------------- forecasts
    def forecasts_vs_realized(
        self, asset: str, horizon: str, models: Iterable[str] | str | None, split: str = "dev"
    ) -> pd.DataFrame:
        """Forecasts of ``models`` joined to the realized target, with display-only annualised vols.

        Columns: origin, model, n_t, F, y, vol_forecast, vol_realized (vols in % per year:
        ``sqrt(F/n_t·ann)`` and ``sqrt(y/n_t·ann)``).
        """
        cols = ["origin", "model", "n_t", "F", "y", "vol_forecast", "vol_realized"]
        ms = _in_list(models)
        if not ms or not self.has(split, "forecasts"):
            return pd.DataFrame(columns=cols)
        if self.has(split, "targets"):
            sql = (
                f"SELECT f.origin, f.model, f.n_t, f.F, t.y FROM {self.view(split, 'forecasts')} f "
                f"LEFT JOIN {self.view(split, 'targets')} t "
                "ON f.asset = t.asset AND f.horizon = t.horizon AND f.origin = t.origin "
                "WHERE f.asset = ? AND f.horizon = ? AND f.split = ? AND list_contains(?, f.model) "
                "ORDER BY f.model, f.origin"
            )
        else:
            sql = (
                f"SELECT origin, model, n_t, F, CAST(NULL AS DOUBLE) AS y FROM {self.view(split, 'forecasts')} "
                "WHERE asset = ? AND horizon = ? AND split = ? AND list_contains(?, model) ORDER BY model, origin"
            )
        df = self.query(sql, [asset, horizon, split, ms])
        ann = annualisation(asset)
        df["vol_forecast"] = np.sqrt(df["F"] / df["n_t"] * ann)
        df["vol_realized"] = np.sqrt(df["y"] / df["n_t"] * ann)
        return df[cols]

    def realized(self, asset: str, horizon: str, split: str = "dev") -> pd.DataFrame:
        """Realized target over the forecast period of (asset, horizon, split): origin, n_t, y, vol_realized."""
        cols = ["origin", "n_t", "y", "vol_realized"]
        if not (self.has(split, "targets") and self.has(split, "forecasts")):
            return pd.DataFrame(columns=cols)
        sql = (
            f"SELECT origin, n_t, y FROM {self.view(split, 'targets')} "
            "WHERE asset = ? AND horizon = ? AND split = ? AND y IS NOT NULL AND origin >= "
            f"(SELECT min(origin) FROM {self.view(split, 'forecasts')} WHERE asset = ? AND horizon = ? AND split = ?) "
            "ORDER BY origin"
        )
        df = self.query(sql, [asset, horizon, split, asset, horizon, split])
        df["vol_realized"] = np.sqrt(df["y"] / df["n_t"] * annualisation(asset))
        return df[cols]

    # ----------------------------------------------------------------------------------------- evaluation
    def leaderboard(self, split: str = "dev") -> pd.DataFrame:
        """QLIKE leaderboard of the split joined with DM-HLN vs HAR, the MCS and (holdout) the bootstrap CI.

        Columns: asset, horizon, split, model, qlike, mse, qlike_ratio, n, dm_hln, dm_p, dm_p_2n, dm_kernel,
        mcs_p, in_90, in_75 and, when ``eval_ratio_ci`` exists, ratio_lo, ratio_hi.
        """
        if not self.has(split, "eval_leaderboard"):
            return pd.DataFrame()
        sel = ["l.asset", "l.horizon", "l.split", "l.model", "l.qlike", "l.mse", "l.qlike_ratio", "l.n"]
        joins = []
        if self.has(split, "eval_dm_har"):
            sel += ["d.dm_hln", "d.pvalue AS dm_p", "d.pvalue_2n AS dm_p_2n", "d.kernel AS dm_kernel"]
            joins.append(f"LEFT JOIN {self.view(split, 'eval_dm_har')} d USING (asset, horizon, model)")
        if self.has(split, "eval_mcs"):
            sel += ["m.pvalue AS mcs_p", "m.in_90", "m.in_75"]
            joins.append(f"LEFT JOIN {self.view(split, 'eval_mcs')} m USING (asset, horizon, model)")
        if self.has(split, "eval_ratio_ci"):
            sel += ["c.lo AS ratio_lo", "c.hi AS ratio_hi"]
            joins.append(f"LEFT JOIN {self.view(split, 'eval_ratio_ci')} c USING (asset, horizon, model)")
        sql = f"SELECT {', '.join(sel)} FROM {self.view(split, 'eval_leaderboard')} l " + " ".join(joins)
        df = self.query(sql)
        for col in ("dm_hln", "dm_p", "dm_p_2n", "dm_kernel", "mcs_p", "in_90", "in_75"):
            if col not in df:
                df[col] = np.nan
        return df

    def iv_comparison(self, split: str = "dev") -> pd.DataFrame:
        """1m implied-vol comparison on each asset's IV subsample.

        Columns: asset, model, qlike, qlike_ratio (vs HAR on the IV subsample), n, ratio_vs_ivcal, dm_p_ivcal,
        mcs_p, in_90 (IV-inclusive MCS; dev only), enc_c, enc_p (encompassing c of the model beyond IV).
        """
        if not self.has(split, "eval_iv_leaderboard"):
            return pd.DataFrame()
        sel = ["l.asset", "l.model", "l.qlike", "l.qlike_ratio", "l.n"]
        joins = []
        if self.has(split, "eval_iv_dm"):
            sel += ["d.ratio AS ratio_vs_ivcal", "d.pvalue AS dm_p_ivcal"]
            joins.append(f"LEFT JOIN {self.view(split, 'eval_iv_dm')} d USING (asset, horizon, model)")
        if self.has(split, "eval_iv_mcs"):
            sel += ["m.pvalue AS mcs_p", "m.in_90"]
            joins.append(f"LEFT JOIN {self.view(split, 'eval_iv_mcs')} m USING (asset, horizon, model)")
        if self.has(split, "eval_encompassing"):
            sel += ["e.c AS enc_c", "e.p_c AS enc_p"]
            joins.append(f"LEFT JOIN {self.view(split, 'eval_encompassing')} e USING (asset, model)")
        sql = (
            f"SELECT {', '.join(sel)} FROM {self.view(split, 'eval_iv_leaderboard')} l "
            + " ".join(joins)
            + " WHERE l.horizon = '1m'"
        )
        df = self.query(sql)
        for col in ("ratio_vs_ivcal", "dm_p_ivcal", "mcs_p", "in_90", "enc_c", "enc_p"):
            if col not in df:
                df[col] = np.nan
        return df

    # ----------------------------------------------------------------------------------------- risk
    def risk_series(self, asset: str, models: Iterable[str] | str | None, split: str = "dev") -> pd.DataFrame:
        """Daily VaR/ES rows of ``models`` for ``asset`` in ``split``:
        date, model, r_cc, var99, var975, es975, sigma."""
        cols = ["date", "model", "r_cc", "var99", "var975", "es975", "sigma"]
        ms = _in_list(models)
        if not ms or not self.has(split, "risk"):
            return pd.DataFrame(columns=cols)
        sql = (
            f"SELECT {', '.join(cols)} FROM {self.view(split, 'risk')} "
            "WHERE asset = ? AND split = ? AND list_contains(?, model) ORDER BY model, date"
        )
        return self.query(sql, [asset, split, ms])

    def rolling_zones(self, asset: str, model: Iterable[str] | str, split: str = "dev") -> pd.DataFrame:
        """Rolling 250-observation Basel traffic light at 99% (date, model, exceptions, zone, plus_factor),
        restricted to the dates of ``split`` (the holdout file holds the whole series)."""
        cols = ["date", "model", "exceptions", "zone", "plus_factor"]
        ms = _in_list(model)
        if not ms or not self.has(split, "risk_rolling_zones"):
            return pd.DataFrame(columns=cols)
        where = "asset = ? AND list_contains(?, model)"
        params: list = [asset, ms]
        if self.has(split, "risk"):
            where += (
                f" AND date >= (SELECT min(date) FROM {self.view(split, 'risk')} WHERE asset = ? AND split = ?)"
                f" AND date <= (SELECT max(date) FROM {self.view(split, 'risk')} WHERE asset = ? AND split = ?)"
            )
            params += [asset, split, asset, split]
        view = self.view(split, "risk_rolling_zones")
        return self.query(f"SELECT {', '.join(cols)} FROM {view} WHERE {where} ORDER BY model, date", params)

    def risk_leaderboard(self, split: str = "dev") -> pd.DataFrame:
        return self.table(split, "risk_risk_leaderboard")

    def backtests(self, split: str = "dev") -> pd.DataFrame:
        return self.table(split, "risk_backtests")

    def time_in_zone(self, split: str = "dev") -> pd.DataFrame:
        return self.table(split, "risk_time_in_zone")

    def green_shares(self, split: str = "dev") -> tuple[pd.DataFrame, str]:
        """(asset, model, green) — the rolling-250 green-zone share H4 is decided on — and its source.

        Same precedence as ``volrisk.hypotheses._green_shares`` (and the report): ``risk_time_in_zone`` (rolling
        250-observation windows *ending* in the split, taken over the whole dev + holdout series), else the
        risk leaderboard's ``green_share``, else the 99% backtest ``green_share``. The latter two are computed
        over windows lying entirely *inside* the split, which differs from the H4 measure on the holdout."""
        cols = ["asset", "model", "green"]
        tz = self.time_in_zone(split)
        if not tz.empty and {"asset", "model", "green"} <= set(tz.columns):
            return tz[cols].reset_index(drop=True), "time_in_zone"
        rl = self.risk_leaderboard(split)
        if not rl.empty and {"asset", "model", "green_share"} <= set(rl.columns):
            return rl[["asset", "model", "green_share"]].rename(columns={"green_share": "green"}), "risk_leaderboard"
        bt = self.backtests(split)
        if not bt.empty and {"asset", "model", "level", "green_share"} <= set(bt.columns):
            bt = bt[bt["level"].astype(str) == "99"]
            return bt[["asset", "model", "green_share"]].rename(columns={"green_share": "green"}), "backtests_99"
        return pd.DataFrame(columns=cols), ""

    def latest_zones(self, asset: str, models: Iterable[str] | str, split: str = "dev") -> pd.DataFrame:
        """Per model: the Basel zone of the last rolling 250-observation window of ``split`` (the window ending on
        the split's last date), with that date and the number of windows ending in the split.
        Columns: model, zone, exceptions, date, windows."""
        cols = ["model", "zone", "exceptions", "date", "windows"]
        z = self.rolling_zones(asset, models, split)
        if z.empty:
            return pd.DataFrame(columns=cols)
        z = z.sort_values(["model", "date"])
        last = z.groupby("model", sort=False).tail(1).set_index("model")
        last["windows"] = z.groupby("model", sort=False).size()
        return last.reset_index()[cols]

    def har_star(self, split: str = "dev") -> dict[str, str]:
        p = self.dirs()[split] / "har_star.json"
        if not p.exists():
            p = self.results_dir / "har_star.json"  # the holdout run uses the frozen (dev) choice
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    # ----------------------------------------------------------------------------------------- hypotheses
    def hypothesis_files(self) -> list[Path]:
        tables = self.reports_dir / "tables"
        return sorted(tables.rglob("hypotheses*.csv")) if tables.is_dir() else []

    def hypotheses(self, split: str = "dev") -> pd.DataFrame:
        """H1–H5 verdict rows of ``split`` from ``reports/tables/hypotheses*.csv`` (empty if not written yet).

        The report writes a long table (``mode, hypothesis, unit, asset, horizon, required, verdict, evidence,
        note``; one ``unit == 'overall'`` row per hypothesis). A file's rows are matched by its ``mode`` (or
        ``split``) column when it has one, otherwise by its path (a path mentioning 'holdout' belongs to the
        holdout split, anything else to dev)."""
        frames = []
        for f in self.hypothesis_files():
            try:
                df = pd.read_csv(f)
            except (OSError, ValueError, pd.errors.ParserError):
                continue
            key = next((c for c in ("mode", "split") if c in df.columns), None)
            if key is not None:
                df = df[df[key].astype(str).str.lower() == split]
            elif ("holdout" in f.relative_to(self.reports_dir).as_posix().lower()) != (split == "holdout"):
                continue
            if len(df):
                frames.append(df)
        return pd.concat(frames, ignore_index=True).drop_duplicates(ignore_index=True) if frames else pd.DataFrame()
