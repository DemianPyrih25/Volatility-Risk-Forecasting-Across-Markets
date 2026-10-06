"""Tests for the forecast combination COMBO (SPEC §7 model 11)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volrisk import config as C
from volrisk.models.base import FORECAST_COLUMNS
from volrisk.models.combo import combine, combo_members

D = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]).astype("datetime64[ms]")


def _rows(model: str, F: list[float], asset: str = "SPX", horizon: str = "1w", n_t: int = 5, dates=D):
    return pd.DataFrame(
        {"asset": asset, "horizon": horizon, "model": model, "origin": dates[: len(F)], "n_t": n_t, "F": F}
    )[FORECAST_COLUMNS]


@pytest.fixture
def fc() -> pd.DataFrame:
    return pd.concat(
        [
            _rows("GJR", [1.0, 2.0, 3.0]),
            _rows("HARQ", [2.0, 4.0, 6.0]),
            _rows("LGBM", [3.0, 6.0, 12.0]),
            _rows("MLP", [100.0, 100.0, 100.0]),  # not a member
            _rows("HAR", [50.0, 50.0, 50.0]),
        ],
        ignore_index=True,
    )


def test_members_from_config():
    assert combo_members() == ["GJR", "HARQ", "LGBM"] == C.load()["models"]["combo_members"]
    assert "MLP" not in combo_members()


def test_equal_weight_mean(fc):
    out = combine(fc)
    assert list(out.columns) == FORECAST_COLUMNS
    assert (out["model"] == "COMBO").all()
    np.testing.assert_allclose(out["F"], [2.0, 4.0, 7.0], rtol=0, atol=1e-15)
    assert out["n_t"].tolist() == [5, 5, 5] and out["n_t"].dtype == np.int64
    assert out["origin"].tolist() == list(D)
    assert out["origin"].dtype == fc["origin"].dtype


def test_missing_member_excludes_origin(fc):
    fc = fc.drop(fc.index[(fc["model"] == "HARQ") & (fc["origin"] == D[1])])
    fc = fc.drop(fc.index[(fc["model"] == "GJR") & (fc["origin"] == D[2])])
    out = combine(fc)
    assert out["origin"].tolist() == [D[0]]
    assert out["F"].tolist() == [2.0]


def test_nan_forecast_counts_as_missing(fc):
    fc.loc[(fc["model"] == "LGBM") & (fc["origin"] == D[0]), "F"] = np.nan
    assert combine(fc)["origin"].tolist() == [D[1], D[2]]


def test_custom_members_and_name(fc):
    out = combine(fc, members=["GJR", "MLP"], name="C2")
    assert (out["model"] == "C2").all()
    np.testing.assert_allclose(out["F"], [50.5, 51.0, 51.5])


def test_cells_are_combined_separately(fc):
    other = pd.concat(
        [_rows(m, [10.0 * (i + 1)], asset="BTC", horizon="1m", n_t=30) for i, m in enumerate(combo_members())],
        ignore_index=True,
    )
    out = combine(pd.concat([other, fc], ignore_index=True))
    assert out[["asset", "horizon"]].drop_duplicates().values.tolist() == [["BTC", "1m"], ["SPX", "1w"]]
    btc = out[out["asset"] == "BTC"]
    assert btc["F"].tolist() == [20.0] and btc["n_t"].tolist() == [30]
    assert len(out) == 4


def test_n_t_disagreement_raises(fc):
    fc.loc[(fc["model"] == "LGBM") & (fc["origin"] == D[1]), "n_t"] = 4
    with pytest.raises(ValueError, match="n_t"):
        combine(fc)


def test_n_t_disagreement_on_excluded_origin_is_ignored(fc):
    # the origin is dropped anyway because GJR is missing, so its n_t mismatch cannot reach the output
    fc.loc[(fc["model"] == "LGBM") & (fc["origin"] == D[1]), "n_t"] = 4
    fc = fc.drop(fc.index[(fc["model"] == "GJR") & (fc["origin"] == D[1])])
    assert combine(fc)["origin"].tolist() == [D[0], D[2]]


def test_duplicates_and_bad_inputs_raise(fc):
    with pytest.raises(ValueError, match="duplicate"):
        combine(pd.concat([fc, fc.iloc[:1]], ignore_index=True))
    bad = fc.copy()
    bad.loc[0, "F"] = 0.0
    with pytest.raises(ValueError, match="> 0"):
        combine(bad)
    with pytest.raises(ValueError, match="unique"):
        combine(fc, members=["GJR", "GJR"])
    with pytest.raises(ValueError, match="non-empty"):
        combine(fc, members=[])


def test_split_is_carried_and_checked(fc):
    fc["split"] = np.where(fc["origin"] == D[2], "holdout", "dev")
    out = combine(fc)
    assert list(out.columns) == FORECAST_COLUMNS + ["split"]
    assert out["split"].tolist() == ["dev", "dev", "holdout"]
    fc.loc[(fc["model"] == "GJR") & (fc["origin"] == D[0]), "split"] = "holdout"
    with pytest.raises(ValueError, match="split"):
        combine(fc)


def test_empty_when_no_member_present():
    out = combine(_rows("MLP", [1.0, 2.0]))
    assert out.empty and list(out.columns) == FORECAST_COLUMNS


def test_combo_of_identical_members_is_identity():
    rng = np.random.default_rng(0)
    F = rng.gamma(2.0, 1.0, 3).tolist()
    fc = pd.concat([_rows(m, F) for m in combo_members()], ignore_index=True)
    np.testing.assert_allclose(combine(fc)["F"], F, rtol=1e-15)
