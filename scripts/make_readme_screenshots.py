"""README screenshots of the running dashboard, taken with headless Microsoft Edge.

Starts the live dashboard on the committed demo bundle (what a fresh clone shows), one tab at a time, and writes
docs/img/<name>.png. The dashboard has a light theme only (a browser's forced dark mode garbles the charts), so the
screenshots are light; the README results chart has real light and dark versions (scripts/make_readme_figures.py).
Windows + Edge only; not needed to run the project.

    uv run python scripts/make_readme_screenshots.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
EDGE = next((p for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                          r"C:\Program Files\Microsoft\Edge\Application\msedge.exe") if Path(p).exists()), None)
WIDTH = 1280
# name, tab value, asset, viewport height, crop (top, bottom) in CSS px
SHOTS = [
    ("tomorrow", "tomorrow", "SPX", 1300, (0, 935)),
    ("leaderboard", "leaderboard", "BTC", 1500, (0, 1100)),
    ("var", "var", "SPX", 1500, (0, 1100)),
]
SERVER = """
import sys
from dash import dcc
from volrisk_live import dashboard as D, paths
tab, asset, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
app = D.create_app(results_dir=paths.DEMO_RESULTS, live_dir=paths.DEMO_LIVE)
def walk(c):
    yield c
    kids = getattr(c, "children", None)
    for k in (kids if isinstance(kids, (list, tuple)) else [kids] if kids is not None else []):
        if hasattr(k, "to_plotly_json"):
            yield from walk(k)
for c in walk(app.layout):
    if isinstance(c, dcc.Tabs) and getattr(c, "id", None) == "tabs":
        c.value = tab
    if isinstance(c, dcc.Dropdown) and getattr(c, "id", None) == "asset":
        c.value = asset
app.run(host="127.0.0.1", port=port, debug=False)
"""


def shoot(name: str, tab: str, asset: str, height: int, crop: tuple[int, int], port: int, tmp: Path) -> None:
    server = tmp / "server.py"
    server.write_text(SERVER, encoding="utf-8")
    proc = subprocess.Popen([sys.executable, str(server), tab, asset, str(port)], cwd=ROOT,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        url = f"http://127.0.0.1:{port}/"
        for _ in range(60):
            try:
                if requests.get(url, timeout=2).ok:
                    break
            except requests.RequestException:
                time.sleep(1)
        raw = tmp / f"{name}-raw.png"
        subprocess.run([EDGE, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--window-size={WIDTH},{height}",
                        "--virtual-time-budget=15000", f"--user-data-dir={tmp / 'profile'}", f"--screenshot={raw}", url],
                       check=False, timeout=120, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        img = Image.open(raw).convert("RGB")
        img = img.crop((0, crop[0], img.width, min(crop[1], img.height)))
        out = OUT / f"{name}.png"
        img.quantize(colors=192, method=Image.Quantize.MEDIANCUT).save(out, optimize=True)
        print(f"{out.relative_to(ROOT)}: {img.width}x{img.height}, {out.stat().st_size / 1e3:.0f} KB")
    finally:
        proc.terminate()
        proc.wait(timeout=20)


def main() -> None:
    if EDGE is None:
        raise SystemExit("Microsoft Edge not found")
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="readme_shots_"))
    try:
        for i, (name, tab, asset, height, crop) in enumerate(SHOTS):
            shoot(name, tab, asset, height, crop, 8090 + i, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
