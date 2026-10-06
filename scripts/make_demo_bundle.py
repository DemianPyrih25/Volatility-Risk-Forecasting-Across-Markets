"""Build the demo bundle in demo/ so the dashboard runs on a fresh clone without the 1.5 GB data download.

The bundle is a snapshot of the real pipeline outputs (no raw data): the frozen result tables of the development
and holdout runs, plus a small recent extract of the live data for the Tomorrow timeline. Every copied file is
listed with its source and SHA-256 in demo/MANIFEST.json.

    uv run python scripts/make_demo_bundle.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEMO = ROOT / "demo"
SKIP = {"eval_losses.parquet"}  # per-origin losses: large and not read by the dashboards
TIMELINE_MODELS = ["COMBO", "HAR", "GJR"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_results(src: Path, dst: Path, files: list[dict]) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.is_file() and f.suffix in (".parquet", ".json") and f.name not in SKIP:
            shutil.copy2(f, dst / f.name)
            files.append({"file": (dst / f.name).relative_to(ROOT).as_posix(),
                          "source": f.relative_to(ROOT).as_posix(), "sha256": sha256(dst / f.name)})


def main() -> None:
    results, live = DATA / "results", DATA / "live"
    for need in (results / "forecasts.parquet", results / "holdout" / "forecasts.parquet",
                 live / "holdout" / "daily.parquet", live / "forecasts" / "forecasts.parquet"):
        if not need.exists():
            raise SystemExit(f"missing {need.relative_to(ROOT)}: run the pipeline and `volrisk_live daily` first")
    if DEMO.exists():
        shutil.rmtree(DEMO)
    files: list[dict] = []
    copy_results(results, DEMO / "results", files)
    copy_results(results / "holdout", DEMO / "results" / "holdout", files)

    # Tomorrow-tab timeline: realised total variance and the 1d forecasts of the timeline models since the holdout
    daily = pd.read_parquet(live / "holdout" / "daily.parquet", columns=["asset", "session_date", "tv"])
    out = DEMO / "live" / "holdout" / "daily.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out, index=False)
    files.append({"file": out.relative_to(ROOT).as_posix(), "source": "data/live/holdout/daily.parquet "
                  "(columns asset, session_date, tv)", "sha256": sha256(out)})
    fc = pd.read_parquet(live / "forecasts" / "forecasts.parquet")
    fc = fc[(fc["horizon"] == "1d") & fc["model"].isin(TIMELINE_MODELS)
            & (pd.to_datetime(fc["origin"]) >= pd.Timestamp("2025-09-01"))]
    out = DEMO / "live" / "forecasts" / "forecasts.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    fc.to_parquet(out, index=False)
    files.append({"file": out.relative_to(ROOT).as_posix(), "source": "data/live/forecasts/forecasts.parquet "
                  f"(horizon 1d, models {', '.join(TIMELINE_MODELS)}, origins from 2025-09-01)", "sha256": sha256(out)})

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "purpose": "snapshot of the real pipeline outputs so the dashboard runs on a fresh clone; no raw data",
        "last_live_session": str(pd.to_datetime(daily["session_date"]).max().date()),
        "files": files,
    }
    (DEMO / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    size = sum((ROOT / f["file"]).stat().st_size for f in files)
    print(f"demo bundle: {len(files)} files, {size / 1e6:.1f} MB, live data to {manifest['last_live_session']}")


if __name__ == "__main__":
    main()
