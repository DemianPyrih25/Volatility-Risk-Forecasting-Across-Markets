"""README results chart (light and dark): QLIKE loss of the model combination relative to the HAR baseline.

Reads the evaluation tables of the development run and the one-time holdout run (data/results, or the committed
demo/ bundle on a fresh clone) and writes docs/img/results-light.png and docs/img/results-dark.png.

    uv run python scripts/make_readme_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "img"
ASSETS = [("BTC", "BTC"), ("ETH", "ETH"), ("EURUSD", "EUR/USD"), ("SPX", "S&P 500")]
HORIZONS = [("1d", "1 day"), ("1w", "1 week"), ("1m", "1 month")]
MODEL = "COMBO"

# validated with the dataviz palette checker (both modes: all checks pass)
THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
              "holdout": "#2a78d6", "dev": "#eb6834", "band": "#eef4fc"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#8f8e86", "grid": "#33332f",
             "holdout": "#3987e5", "dev": "#d95926", "band": "#1f2a38"},
}


def results_dir() -> Path:
    real = ROOT / "data" / "results"
    return real if (real / "holdout" / "eval_ratio_ci.parquet").exists() else ROOT / "demo" / "results"


def load() -> pd.DataFrame:
    res = results_dir()
    dev = pd.read_parquet(res / "eval_leaderboard.parquet")
    dev = dev[(dev["model"] == MODEL) & (dev["split"] == "dev")][["asset", "horizon", "qlike_ratio"]]
    hold = pd.read_parquet(res / "holdout" / "eval_ratio_ci.parquet")
    hold = hold[hold["model"] == MODEL][["asset", "horizon", "ratio", "lo", "hi"]]
    return dev.rename(columns={"qlike_ratio": "dev"}).merge(hold, on=["asset", "horizon"], how="inner")


def draw(df: pd.DataFrame, theme: str) -> Path:
    t = THEMES[theme]
    rows, labels, y = [], [], 0.0
    for a, a_lab in ASSETS:
        for h, h_lab in HORIZONS:
            r = df[(df["asset"] == a) & (df["horizon"] == h)]
            if len(r):
                rows.append((y, r.iloc[0]))
                labels.append((y, f"{a_lab} · {h_lab}"))
                y += 1.0
        y += 0.6  # gap between markets
    n_better = int((df["ratio"] < 1).sum())
    n_sig = int((df["hi"] < 1).sum())

    fig, ax = plt.subplots(figsize=(8.2, 5.7), dpi=190)
    fig.patch.set_facecolor(t["surface"])
    ax.set_facecolor(t["surface"])
    lo = min(df["lo"].min(), df["dev"].min()) - 0.05
    hi = max(df["hi"].max(), df["dev"].max()) + 0.05
    ax.axvspan(lo, 1.0, color=t["band"], zorder=0, lw=0)
    ax.axvline(1.0, color=t["ink2"], lw=1.2, zorder=1)
    for yy, r in rows:
        ax.plot([r["lo"], r["hi"]], [yy, yy], color=t["holdout"], lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([r["dev"]], [yy + 0.28], s=46, facecolor=t["surface"], edgecolor=t["dev"], linewidth=2, zorder=3)
        ax.scatter([r["ratio"]], [yy], s=62, color=t["holdout"], edgecolor=t["surface"], linewidth=1.6, zorder=4)
    ax.set_yticks([yy for yy, _ in labels], [lab for _, lab in labels])
    ax.invert_yaxis()
    ax.set_xlim(lo, hi)
    ax.tick_params(colors=t["ink2"], labelsize=9, length=0)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.1f"))
    ax.grid(axis="x", color=t["grid"], lw=0.8, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("QLIKE loss of the combination ÷ loss of HAR   (below 1 = better than HAR)", color=t["ink2"],
                  fontsize=9, labelpad=8)
    ax.text(1.0, -0.9, "HAR baseline = 1", color=t["ink2"], fontsize=8.5, ha="center", va="bottom")
    ax.text(lo + 0.01, -0.9, "← better", color=t["ink2"], fontsize=8.5, ha="left", va="bottom")
    fig.suptitle(f"The combination had lower forecast loss than HAR in {n_better} of {len(df)} holdout cells",
                 x=0.03, y=0.985, ha="left", fontsize=12, fontweight="bold", color=t["ink"])
    fig.text(0.03, 0.925, f"In {n_sig} of them the whole 90% bootstrap interval is below 1. Holdout: Oct 2025 – Sep 2026, "
             "sealed and opened once;\ndevelopment: 2021 – Sep 2025. Four markets × three horizons. Source: "
             "results/holdout/eval_ratio_ci, results/eval_leaderboard.", ha="left", va="top", fontsize=8.3,
             color=t["ink2"])
    handles = [Line2D([0], [0], color=t["holdout"], lw=2, marker="o", markersize=7, markeredgecolor=t["surface"],
                      label="Holdout (90% interval)"),
               Line2D([0], [0], color="none", marker="o", markersize=6.5, markerfacecolor=t["surface"],
                      markeredgecolor=t["dev"], markeredgewidth=2, label="Development")]
    leg = ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=8.8, labelcolor=t["ink2"])
    leg.set_zorder(5)
    fig.subplots_adjust(left=0.185, right=0.97, top=0.84, bottom=0.11)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"results-{theme}.png"
    fig.savefig(path, facecolor=t["surface"])
    plt.close(fig)
    return path


def main() -> None:
    df = load()
    for theme in THEMES:
        p = draw(df, theme)
        print(f"{p.relative_to(ROOT)}: {p.stat().st_size / 1e3:.0f} KB")


if __name__ == "__main__":
    main()
