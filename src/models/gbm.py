"""Global LightGBM forecaster (Cycle 2, Option C).

One model over all series, direct multi-horizon with horizon-as-feature,
Tweedie objective for zero-inflated demand. Trains from scratch inside
fit_predict on the harness-provided train slice, so every backtest origin
gets its own leak-free fit.
"""
from __future__ import annotations

import numpy as np

from ..features import (CATEGORICALS, FEATURE_NAMES, build_wide,
                        prediction_matrix, training_matrix)

DEFAULT_LGB_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.2,
    "n_estimators": 300,
    "learning_rate": 0.08,
    "num_leaves": 127,
    "min_child_samples": 50,
    "colsample_bytree": 0.9,
    "subsample": 0.8,
    "subsample_freq": 1,
    "verbose": -1,
}


class GlobalLGBM:
    name = "global_lgbm"

    def __init__(self, n_anchors: int = 24, lgb_params: dict | None = None):
        self.n_anchors = n_anchors
        self.lgb_params = {**DEFAULT_LGB_PARAMS, **(lgb_params or {})}

    def fit_predict(self, train, future_ds):
        import lightgbm as lgb
        import pandas as pd

        wide_y, wide_p, client, wh = build_wide(train)
        X, y = training_matrix(wide_y, wide_p, client, wh,
                               self.n_anchors, len(future_ds))
        Xf = pd.DataFrame(X, columns=FEATURE_NAMES)
        for c in CATEGORICALS:
            Xf[c] = Xf[c].astype("int32").astype("category")

        model = lgb.LGBMRegressor(**self.lgb_params)
        model.fit(Xf, y, categorical_feature=CATEGORICALS)

        Xp, ids = prediction_matrix(wide_y, wide_p, client, wh, future_ds)
        Xpf = pd.DataFrame(Xp, columns=FEATURE_NAMES)
        for c in CATEGORICALS:
            Xpf[c] = Xpf[c].astype("int32").astype("category")
        ids["yhat"] = np.clip(model.predict(Xpf), 0.0, None)
        return ids
