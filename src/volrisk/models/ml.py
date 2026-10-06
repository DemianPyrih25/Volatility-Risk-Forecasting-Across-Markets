"""Machine-learning forecasters: LightGBM and an MLP trained on QLIKE (SPEC §7 models 9–10, walk-forward §6).

Both are direct models of the per-session mean target ``ybar``. LightGBM uses its ``gamma`` objective, which on
the log link ``f = log ŷbar`` has gradient ``1 − y·e^{−f}`` and Hessian ``y·e^{−f}`` — exactly the derivatives
of the QLIKE-equivalent loss ``y·e^{−f} + f`` (whose optimal constant is ``mean(y)``). The MLP minimises that
loss directly in PyTorch. The forecast is ``F = n_t · ŷbar``.

Walk-forward (``anchored_walk_forward``) with ``W = config.window()`` and ``R = config.refit_every('ml', asset)``:

* OOS start per asset (``oos_start_row``): the first session with ``W`` eligible training rows for every
  horizon, so 1d/1w/1m start on the same origin whichever horizon is forecast;
* refits sit at the session rows ``j = start + m·R`` (rows = sessions; calendar days for crypto). The schedule
  counts sessions, not forecast origins, so the ``dropped`` origins between dev and holdout still advance it
  and every horizon re-estimates on the same dates;
* the fit at ``j`` uses the last ``W`` rows eligible at origin ``j`` (purged: ``window_end ≤ j``, finite target
  and features) and serves the origins in ``[j, j+R)``.

LightGBM is driven through ``lightgbm.train`` with the SPEC hyperparameters under their scikit-learn alias
names (``LGBMRegressor`` is a thin wrapper around it but needs scikit-learn, which is not a dependency).
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from typing import Callable, Iterator

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn

from volrisk import config as C
from volrisk import targets as tgt
from volrisk.models.base import align_targets, eligible_end, predict_mask_for, to_forecast_frame
from volrisk.models.features import VARIANCE_FEATURES, ml_features

LOG_EPS = 1e-8
N_DOW = 7  # one-hot levels Mon..Sun; levels absent for an asset are constant and standardise to 0


def _days(s: pd.Series) -> np.ndarray:
    return pd.to_datetime(s).to_numpy().astype("datetime64[D]")


def _check_label(y: np.ndarray, model: str) -> None:
    if not (np.all(np.isfinite(y)) and np.all(y > 0)):
        raise ValueError(f"{model}: training label ybar must be finite and > 0")


# ---------------------------------------------------------------------------------------------- walk-forward


def eligible_counts(ok: np.ndarray, window_end: np.ndarray, origins: np.ndarray) -> np.ndarray:
    """Per origin row, the number of usable training rows (``ok``) whose target window has ended by it.

    Incomplete windows (NaT) are stamped with their own origin first: such rows never train (``ok`` is False
    for them), but a NaT inside the sample would otherwise lift ``base.eligible_end``'s monotone envelope to
    +∞ and freeze the training window from that row on.
    """
    we = np.where(np.isnat(window_end), origins, window_end)
    return np.searchsorted(np.flatnonzero(ok), eligible_end(we, origins), side="left")


@contextmanager
def _quiet(logger: logging.Logger) -> Iterator[None]:
    level = logger.level
    logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logger.setLevel(level)


def oos_start_row(
    daily: pd.DataFrame, asset: str, x_ok: np.ndarray, window: int, last_date: date | None = None
) -> int | None:
    """SPEC §6 OOS start of one asset: the first row with ``window`` eligible training rows for every horizon.

    The targets of all horizons are rebuilt from ``daily`` (``targets.build_targets``; ``last_date`` defaults to
    the last session) so the start does not depend on the horizon being forecast. ``x_ok`` flags rows with
    finite features. ``None`` if some horizon never reaches ``window`` eligible rows.
    """
    if len(daily) == 0:
        return None
    origins = _days(daily["session_date"])
    last = pd.Timestamp(origins[-1]).date() if last_date is None else last_date
    start = 0
    for h in C.HORIZONS:
        with _quiet(tgt.log):
            tg = tgt.build_targets(daily, asset, h, last)  # one row per session, in row order
        ok = x_ok & np.isfinite(tg["ybar"].to_numpy(dtype=float))
        hit = np.flatnonzero(eligible_counts(ok, _days(tg["window_end"]), origins) >= window)
        if hit.size == 0:
            return None
        start = max(start, int(hit[0]))
    return start


def anchored_walk_forward(
    X: np.ndarray,
    ybar: np.ndarray,
    window_end: np.ndarray,
    origins: np.ndarray,
    predict_mask: np.ndarray,
    window: int,
    refit_every: int,
    start: int | None,
    fit: Callable[[np.ndarray, np.ndarray], object],
    predict: Callable[[object, np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Purged rolling walk-forward with refits at the fixed session rows ``start + m·refit_every``.

    X: (T, k) features per row; ybar: (T,) target mean (NaN if unobserved); window_end/origins: datetime64[D]
    per row; predict_mask: rows for which a forecast is wanted. Origin ``i ≥ start`` (wanted, finite ``X[i]``)
    is forecast by the fit at ``j = start + ⌊(i − start)/R⌋·R`` on the last ``window`` rows eligible at
    ``origins[j]``. Fits are stateless, so a block without wanted origins is not fitted (no forecast changes);
    a block whose fit row has fewer than ``window`` eligible rows gets no forecasts. Predictions are made one
    row at a time, so a forecast never depends on which other origins are wanted (holdout run reproduces dev).
    Returns ``ybar_hat`` (T,) (NaN where no forecast was made) and the fitted rows.
    """
    if window < 1 or refit_every < 1:
        raise ValueError(f"bad walk-forward settings window={window} refit_every={refit_every}")
    T = len(ybar)
    out = np.full(T, np.nan)
    fitted: list[int] = []
    if start is None:
        return out, np.asarray(fitted, dtype=np.int64)
    x_ok = np.all(np.isfinite(X), axis=1)
    ok = np.isfinite(ybar) & x_ok
    obs_idx = np.flatnonzero(ok)
    n_elig = eligible_counts(ok, window_end, origins)
    want = np.asarray(predict_mask, dtype=bool) & x_ok
    want[:start] = False
    for j in range(start, T, refit_every):
        rows = np.flatnonzero(want[j : j + refit_every]) + j
        if rows.size == 0 or n_elig[j] < window:
            continue
        train = obs_idx[n_elig[j] - window : n_elig[j]]
        model = fit(X[train], ybar[train])
        fitted.append(j)
        for i in rows:
            out[i] = predict(model, X[i : i + 1])[0]
    return out, np.asarray(fitted, dtype=np.int64)


class _DirectML:
    """Walk-forward driver shared by the ML forecasters; subclasses supply ``design``, ``fit``, ``predict``.

    ``window`` / ``refit_every`` default to ``config.window()`` / ``config.refit_every('ml', asset)`` and can be
    injected (tests). After a ``forecast`` call, ``n_fits`` counts its re-estimations, ``fit_origins`` holds
    their origins and ``oos_start`` the asset's OOS start (``None`` without enough history).
    """

    name: str

    def __init__(self, window: int | None = None, refit_every: int | None = None) -> None:
        self.window = window
        self.refit_every = refit_every
        self.n_fits = 0
        self.fit_origins = np.array([], dtype="datetime64[ms]")
        self.oos_start: pd.Timestamp | None = None

    def design(self, daily: pd.DataFrame, asset: str) -> pd.DataFrame:
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        raise NotImplementedError

    def predict(self, model: object, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def forecast(self, daily: pd.DataFrame, targets: pd.DataFrame, asset: str, horizon: str) -> pd.DataFrame:
        X = self.design(daily, asset).to_numpy(dtype=float)
        a = align_targets(daily, targets)
        window = C.window() if self.window is None else int(self.window)
        refit = C.refit_every("ml", asset) if self.refit_every is None else int(self.refit_every)
        start = oos_start_row(daily, asset, np.all(np.isfinite(X), axis=1), window)
        self.n_fits = 0
        ybar_hat, fit_rows = anchored_walk_forward(
            X,
            a["ybar"].to_numpy(dtype=float),
            _days(a["window_end"]),
            _days(daily["session_date"]),
            predict_mask_for(a),
            window,
            refit,
            start,
            self.fit,
            self.predict,
        )
        self.fit_origins = daily["session_date"].to_numpy()[fit_rows]
        self.oos_start = None if start is None else pd.Timestamp(daily["session_date"].iloc[start])
        F = a["n_t"].to_numpy(dtype=float) * ybar_hat
        return to_forecast_frame(asset, horizon, self.name, daily["session_date"], a["n_t"], F)


# ---------------------------------------------------------------------------------------------- LightGBM


class LGBM(_DirectML):
    """LightGBM on the ``ml_features`` levels (``dow`` as an integer), gamma objective, label ``ybar``."""

    name = "LGBM"

    def __init__(
        self,
        window: int | None = None,
        refit_every: int | None = None,
        params: dict | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(window, refit_every)
        self.params = dict(params or {})
        self.seed = seed

    def lgb_params(self) -> tuple[dict, int]:
        """Native ``lightgbm.train`` parameters and the number of boosting rounds (SPEC §7)."""
        p = {
            **C.load()["models"]["lgbm"],
            "random_state": C.seed() if self.seed is None else int(self.seed),
            "deterministic": True,
            "force_row_wise": True,
            "verbose": -1,
            **self.params,
        }
        n_rounds = int(p.pop("n_estimators"))
        return p, n_rounds

    def design(self, daily: pd.DataFrame, asset: str) -> pd.DataFrame:
        return ml_features(daily, asset)

    def fit(self, X: np.ndarray, y: np.ndarray, init_score: np.ndarray | None = None) -> lgb.Booster:
        _check_label(y, self.name)
        self.n_fits += 1
        p, n_rounds = self.lgb_params()
        ds = lgb.Dataset(X, label=y, init_score=init_score, params={"verbose": -1})
        return lgb.train(p, ds, num_boost_round=n_rounds)

    def predict(self, model: lgb.Booster, X: np.ndarray) -> np.ndarray:
        return model.predict(X)  # gamma objective: exp of the raw score, i.e. ŷbar on the variance scale


# ---------------------------------------------------------------------------------------------- MLP


def mlp_design(features: pd.DataFrame) -> pd.DataFrame:
    """MLP inputs from ``ml_features``: log(x+1e-8) for variance-type columns, raw ``r_cc`` terms and
    ``j_share``, day-of-week one-hot (``dow_0``..``dow_6``) replacing the integer ``dow``."""
    out = features.drop(columns="dow").astype(float)
    for c in VARIANCE_FEATURES:
        if c in out.columns:
            out[c] = np.log(out[c] + LOG_EPS)
    dow = features["dow"].to_numpy()
    for d in range(N_DOW):
        out[f"dow_{d}"] = (dow == d).astype(float)
    return out


@dataclass
class MLPFit:
    """One re-estimation: the seed ensemble, the training-window standardisation and per-epoch losses."""

    nets: list[nn.Sequential]
    mu: np.ndarray
    sd: np.ndarray
    losses: np.ndarray  # (n_seeds, epochs) full-batch training loss before each update


@contextmanager
def _torch_deterministic() -> Iterator[None]:
    """Single-threaded, deterministic torch with the global RNG state restored afterwards."""
    n_threads = torch.get_num_threads()
    det = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.use_deterministic_algorithms(det, warn_only=warn_only)
        torch.set_num_threads(n_threads)


class MLP(_DirectML):
    """``in → 32 → 32 → 1`` SiLU network with output ``f = log ŷbar``, loss ``mean(ybar·e^{−f} + f)``,
    trained full-batch with Adam for each seed ``SEED+k`` and averaged in variance space (mean of ``e^f``)."""

    name = "MLP"

    def __init__(
        self,
        window: int | None = None,
        refit_every: int | None = None,
        hidden: tuple[int, ...] | None = None,
        epochs: int | None = None,
        lr: float | None = None,
        weight_decay: float | None = None,
        n_seeds: int | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(window, refit_every)
        cfg = C.load()["models"]["mlp"]
        self.hidden = tuple(int(h) for h in (cfg["hidden"] if hidden is None else hidden))
        self.epochs = int(cfg["epochs"] if epochs is None else epochs)
        self.lr = float(cfg["lr"] if lr is None else lr)
        self.weight_decay = float(cfg["weight_decay"] if weight_decay is None else weight_decay)
        self.n_seeds = int(cfg["n_seeds"] if n_seeds is None else n_seeds)
        self.seed = C.seed() if seed is None else int(seed)

    def design(self, daily: pd.DataFrame, asset: str) -> pd.DataFrame:
        return mlp_design(ml_features(daily, asset))

    def _net(self, n_in: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        for h in self.hidden:
            layers += [nn.Linear(n_in, h), nn.SiLU()]
            n_in = h
        layers.append(nn.Linear(n_in, 1))
        return nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> MLPFit:
        _check_label(y, self.name)
        self.n_fits += 1
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        xt = torch.from_numpy(((X - mu) / sd).astype(np.float32))
        yt = torch.from_numpy(np.asarray(y, dtype=np.float32))
        bias0 = float(np.log(np.mean(y)))
        nets, losses = [], np.empty((self.n_seeds, self.epochs))
        with _torch_deterministic():
            for k in range(self.n_seeds):
                torch.manual_seed(self.seed + k)
                net = self._net(xt.shape[1])
                with torch.no_grad():
                    net[-1].bias.fill_(bias0)
                opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay, foreach=True)
                for e in range(self.epochs):
                    opt.zero_grad(set_to_none=True)
                    f = net(xt).squeeze(1)
                    loss = (yt * torch.exp(-f) + f).mean()
                    loss.backward()
                    opt.step()
                    losses[k, e] = loss.item()
                net.eval()
                nets.append(net)
        return MLPFit(nets=nets, mu=mu, sd=sd, losses=losses)

    def predict(self, model: MLPFit, X: np.ndarray) -> np.ndarray:
        xt = torch.from_numpy(((X - model.mu) / model.sd).astype(np.float32))
        with _torch_deterministic(), torch.no_grad():
            f = torch.stack([net(xt).squeeze(1) for net in model.nets])
        return np.exp(f.numpy().astype(np.float64)).mean(axis=0)
