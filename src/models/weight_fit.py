"""WeightFitBlend (Cycle 12): a blend whose weights are FIT, leak-free,
inside each origin's training slice — the pre-registered "principled-weight"
protocol (memo 2026-08-31-cycle12-principled-weight-blend.md).

Protocol, per fit_predict call:
  1. Hold out the LAST `holdout_weeks` weeks of `train` (BiasScaled pattern —
     the holdout is inside the training slice; nothing at/after the origin is
     ever seen).
  2. Fit every component on train-minus-holdout; forecast the holdout.
  3. ONE deterministic SLSQP solve for simplex weights (w>=0, sum=1)
     minimizing holdout VN1 = (sum|A - Fw| + |sum(A - Fw)|) / sumA — convex
     in w. One objective, one holdout, one solve; no grids, no repeats.
  4. Refit all components on the FULL train; return the weighted blend.

Weights may differ per origin (each origin fits within its own slice) — by
design, not a leak. Fitted weights are printed for the run log.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import SERIES_ID


def _build(config: dict):
    from . import build_model  # deferred: avoids circular import
    return build_model(config)


class WeightFitBlend:
    """params: models=[{name,...}, ...], holdout_weeks=13"""
    name = "weight_fit_blend"

    def __init__(self, models: list[dict], holdout_weeks: int = 13):
        if len(models) < 2:
            raise ValueError("weight_fit_blend needs >=2 component models")
        self.models = models
        self.holdout_weeks = holdout_weeks

    # -- step 1-3: fit weights on the internal holdout -----------------
    def _fit_weights(self, train: pd.DataFrame) -> np.ndarray:
        from scipy.optimize import minimize

        weeks = sorted(train["ds"].unique())
        if len(weeks) <= self.holdout_weeks + 26:
            # too little history to fit honestly: fall back to equal weights
            print("[weight_fit_blend] short history - equal weights fallback",
                  flush=True)
            return np.full(len(self.models), 1.0 / len(self.models))
        holdout_ds = [pd.Timestamp(w) for w in weeks[-self.holdout_weeks:]]
        inner_train = train[train["ds"] < holdout_ds[0]]

        actual = (train[train["ds"].isin(holdout_ds)]
                  .set_index([SERIES_ID, "ds"])["y"].sort_index())
        cols = []
        for cfg in self.models:
            f = (_build(cfg).fit_predict(inner_train, holdout_ds)
                 .set_index([SERIES_ID, "ds"])["yhat"])
            f = f.reindex(actual.index)
            if f.isna().any():
                raise ValueError(
                    f"component {cfg['name']} left holdout cells unforecast")
            cols.append(f.to_numpy())
        F = np.column_stack(cols)                      # (cells, k)
        a = actual.to_numpy()
        sum_a = float(a.sum())
        if sum_a <= 0:
            print("[weight_fit_blend] zero holdout volume - equal weights",
                  flush=True)
            return np.full(len(self.models), 1.0 / len(self.models))

        def vn1(w):
            err = a - F @ w
            return (np.abs(err).sum() + abs(err.sum())) / sum_a

        k = len(self.models)
        w0 = np.full(k, 1.0 / k)
        res = minimize(vn1, w0, method="SLSQP",
                       bounds=[(0.0, 1.0)] * k,
                       constraints=[{"type": "eq",
                                     "fun": lambda w: w.sum() - 1.0}])
        w = np.clip(res.x, 0.0, 1.0)
        w = w / w.sum()
        print(f"[weight_fit_blend] holdout {holdout_ds[0].date()}.."
              f"{holdout_ds[-1].date()} vn1 {vn1(w0):.4f}->{vn1(w):.4f} "
              f"weights {np.round(w, 4).tolist()}", flush=True)
        return w

    # -- step 4: full refit + weighted blend ---------------------------
    def fit_predict(self, train, future_ds):
        w = self._fit_weights(train)
        acc = None
        for cfg, wi in zip(self.models, w):
            f = (_build(cfg).fit_predict(train, future_ds)
                 .set_index([SERIES_ID, "ds"])["yhat"] * wi)
            acc = f if acc is None else acc.add(f, fill_value=0.0)
        return acc.rename("yhat").reset_index()
