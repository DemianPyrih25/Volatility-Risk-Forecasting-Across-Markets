"""``uv run python -m volrisk.dashboard [--port 8050] [--debug]`` — serve the dashboard on 127.0.0.1."""

from __future__ import annotations

import argparse
import threading
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"


def main(argv: list[str] | None = None, open_browser: bool = False) -> None:
    """Serve the dashboard; the browser is opened only by the command line entry point (never in tests)."""
    p = argparse.ArgumentParser(prog="python -m volrisk.dashboard",
                                description="Read-only dashboard over the volrisk results (local only).")
    p.add_argument("--port", type=int, default=8050, help="port on 127.0.0.1 (default 8050)")
    p.add_argument("--debug", action="store_true", help="Dash debug mode (dev tools, auto-reload)")
    p.add_argument("--results-dir", type=Path, default=None,
                   help="development results directory (default data/results; holdout results are read from "
                        "<results-dir>/holdout when present)")
    p.add_argument("--reports-dir", type=Path, default=None, help="reports directory (default reports)")
    p.add_argument("--no-browser", action="store_true", help="do not open the dashboard in the web browser")
    args = p.parse_args(argv)

    from volrisk.dashboard.app import create_app

    app = create_app(results_dir=args.results_dir, reports_dir=args.reports_dir)
    url = f"http://{HOST}:{args.port}/"
    print(f"\n  volrisk dashboard running at {url}\n  (opening it in your browser; press Ctrl+C here to stop)\n", flush=True)
    if open_browser and not args.no_browser:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    app.run(host=HOST, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main(open_browser=True)
