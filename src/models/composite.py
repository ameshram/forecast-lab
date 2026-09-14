"""Composite model family (U3): ensembles as one-config hypotheses.

SegmentRouter -- route each series to a sub-model by its Syntetos-Boylan
class (computed on training data only). Any registry model can be a route.

Blend -- weighted average of sub-model forecasts.

Sub-model configs use the same {name, params} shape as top-level configs, so
any future combination is a config diff, not new code.
"""
from __future__ import annotations

import pandas as pd

from ..data import SERIES_ID, syntetos_boylan_class


def _build(config: dict):
    from . import build_model  # deferred: avoids circular import
    return build_model(config)


class SegmentRouter:
    """params: default={name,...}, routes={<sb_class>: {name,...}, ...}"""
    name = "segment_router"

    def __init__(self, default: dict, routes: dict):
        self.default = default
        self.routes = routes

    def fit_predict(self, train, future_ds):
        out = (_build(self.default).fit_predict(train, future_ds)
               .set_index([SERIES_ID, "ds"]))
        classes = syntetos_boylan_class(train)
        for seg, sub_cfg in self.routes.items():
            members = classes[classes == seg].index
            if len(members) == 0:
                continue
            sub_train = train[train[SERIES_ID].isin(members)]
            sub = (_build(sub_cfg).fit_predict(sub_train, future_ds)
                   .set_index([SERIES_ID, "ds"]))
            out.loc[sub.index, "yhat"] = sub["yhat"]
        return out.reset_index()


class BiasScaled:
    """Trailing-ratio debias wrapper (Cycle 2, Option A).

    Fits the base model on train minus the last `holdout_weeks`, forecasts
    that internal holdout, and computes ratio = actual/forecast totals.
    The refit base's forecasts are scaled by the damped, clipped ratio.
    Train-only by construction: the holdout is inside the training slice.
    """
    name = "bias_scaled"

    def __init__(self, base: dict, holdout_weeks: int = 13,
                 alpha: float = 0.5, clip_low: float = 0.85,
                 clip_high: float = 1.2):
        self.base = base
        self.holdout_weeks = holdout_weeks
        self.alpha = alpha
        self.clip_low = clip_low
        self.clip_high = clip_high

    def fit_predict(self, train, future_ds):
        weeks = sorted(train["ds"].unique())
        ratio = 1.0
        if len(weeks) > self.holdout_weeks + 26:
            inner_future = [pd.Timestamp(w) for w in weeks[-self.holdout_weeks:]]
            inner_train = train[train["ds"] < inner_future[0]]
            inner_fcst = _build(self.base).fit_predict(inner_train, inner_future)
            denom = inner_fcst["yhat"].sum()
            actual = train.loc[train["ds"].isin(inner_future), "y"].sum()
            if denom > 0:
                ratio = actual / denom
        scale = 1.0 + self.alpha * (ratio - 1.0)
        scale = float(min(max(scale, self.clip_low), self.clip_high))
        out = _build(self.base).fit_predict(train, future_ds)
        out["yhat"] = out["yhat"] * scale
        return out


class Blend:
    """params: models=[{name,...}, ...], weights=[...] (normalized)."""
    name = "blend"

    def __init__(self, models: list[dict], weights: list[float] | None = None):
        self.models = models
        w = weights or [1.0] * len(models)
        total = sum(w)
        self.weights = [x / total for x in w]

    def fit_predict(self, train, future_ds):
        acc = None
        for cfg, w in zip(self.models, self.weights):
            f = (_build(cfg).fit_predict(train, future_ds)
                 .set_index([SERIES_ID, "ds"])["yhat"] * w)
            acc = f if acc is None else acc.add(f, fill_value=0.0)
        return acc.rename("yhat").reset_index()
