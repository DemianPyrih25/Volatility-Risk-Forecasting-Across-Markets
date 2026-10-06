"""Consistency of docs/DEVIATIONS.md with the behaviour of the code it describes (the doc check is skipped
when docs/ is not part of the checkout; the code check always runs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.stats import chi2

from volrisk.risk import backtests as bt

DEVIATIONS = Path(__file__).resolve().parents[1] / "docs" / "DEVIATIONS.md"


def _entries() -> list[str]:
    """Top-level bullet entries of the deviations log, continuation lines joined with a space."""
    entries: list[str] = []
    cur: str | None = None
    for line in DEVIATIONS.read_text(encoding="utf-8").splitlines():
        if line.startswith("- "):
            if cur is not None:
                entries.append(cur)
            cur = line[2:].strip()
        elif cur is not None and line.startswith("  ") and line.strip():
            cur += " " + line.strip()
        elif cur is not None:
            entries.append(cur)
            cur = None
    if cur is not None:
        entries.append(cur)
    return entries


@pytest.mark.skipif(not DEVIATIONS.exists(), reason="docs/DEVIATIONS.md is not in this checkout")
def test_dq_entry_documents_fixed_six_df():
    """The DQ entry must document what ``dq_test`` does (SPEC §9): χ²(6) even when X is rank-deficient."""
    dq_entries = [e for e in _entries() if "DQ test" in e]
    assert len(dq_entries) == 1
    entry = dq_entries[0]
    assert "χ²(6)" in entry
    assert "rank of the regressor matrix" not in entry  # the stale df = rank(X) rule


def test_dq_code_uses_fixed_six_df_on_singular_x():
    """``dq_test`` keeps χ²(6) when X is rank-deficient (the behaviour the deviations entry documents)."""
    rng = np.random.default_rng(20261002)
    T, p = 600, 0.01
    var = 2.0 + 0.1 * rng.standard_normal(T)
    h = np.zeros(T, dtype=int)
    h[T - 2] = 1  # breach only in the last 4 observations: lag columns 2..4 are constant
    hit = h - p
    X = np.column_stack([np.ones(T - 4)] + [hit[4 - k : T - k] for k in range(1, 5)] + [var[4:]])
    rank = int(np.linalg.matrix_rank(X))
    assert rank < 6

    res = bt.dq_test(h, p, var)
    assert np.isfinite(res["dq"])
    assert res["p"] == pytest.approx(chi2.sf(res["dq"], 6))
    assert res["p"] != pytest.approx(chi2.sf(res["dq"], rank), rel=1e-3)
