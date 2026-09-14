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


DEFAULT_XGB_PARAMS = {
    "objective": "reg:tweedie",
    "tweedie_variance_power": 1.2,
    "n_estimators": 300,
    "learning_rate": 0.08,
    "grow_policy": "lossguide",
    "max_leaves": 127,
    "max_depth": 0,
    "min_child_weight": 5,
    "colsample_bytree": 0.9,
    "subsample": 0.8,
    "tree_method": "hist",
    "enable_categorical": True,
    "verbosity": 0,
}


class GlobalXGB:
    """XGBoost twin of GlobalLGBM: same feature pipeline, Tweedie objective,
    leaf-wise growth (lossguide, 127 leaves). Categoricals via pandas
    category dtype + enable_categorical; NaN lags handled natively."""
    name = "global_xgb"

    def __init__(self, n_anchors: int = 56, xgb_params: dict | None = None):
        self.n_anchors = n_anchors
        self.xgb_params = {**DEFAULT_XGB_PARAMS, **(xgb_params or {})}

    def fit_predict(self, train, future_ds):
        import pandas as pd
        import xgboost as xgb

        wide_y, wide_p, client, wh = build_wide(train)
        X, y = training_matrix(wide_y, wide_p, client, wh,
                               self.n_anchors, len(future_ds))
        # price_chg4 can be +/-inf (undefined price ratio); XGBoost's hist
        # rejects non-NaN inf. Treat it as missing (LightGBM bins it, so
        # features.py is left byte-identical to preserve global_lgbm lineage).
        Xf = pd.DataFrame(X, columns=FEATURE_NAMES).replace(
            [np.inf, -np.inf], np.nan)
        for c in CATEGORICALS:
            Xf[c] = Xf[c].astype("int32").astype("category")

        model = xgb.XGBRegressor(**self.xgb_params)
        model.fit(Xf, y)

        Xp, ids = prediction_matrix(wide_y, wide_p, client, wh, future_ds)
        Xpf = pd.DataFrame(Xp, columns=FEATURE_NAMES).replace(
            [np.inf, -np.inf], np.nan)
        for c in CATEGORICALS:
            Xpf[c] = Xpf[c].astype("int32").astype("category")
        ids["yhat"] = np.clip(model.predict(Xpf), 0.0, None)
        return ids


DEFAULT_CAT_PARAMS = {
    "loss_function": "Tweedie:variance_power=1.2",
    "iterations": 300,
    "learning_rate": 0.08,
    "depth": 7,
    "l2_leaf_reg": 3.0,
    "subsample": 0.8,
    "rsm": 0.9,
    "verbose": False,
    "allow_writing_files": False,
}


class GlobalCatBoost:
    """CatBoost twin of GlobalLGBM: same feature pipeline, Tweedie loss,
    symmetric trees (depth 7 ~ 128 leaves). client/warehouse passed as
    integer cat_features (CatBoost rejects float/NaN categoricals; they are
    never NaN here); NaN numeric lags handled natively."""
    name = "global_cat"

    def __init__(self, n_anchors: int = 56, cat_params: dict | None = None):
        self.n_anchors = n_anchors
        self.cat_params = {**DEFAULT_CAT_PARAMS, **(cat_params or {})}

    def fit_predict(self, train, future_ds):
        import pandas as pd
        from catboost import CatBoostRegressor, Pool

        wide_y, wide_p, client, wh = build_wide(train)
        X, y = training_matrix(wide_y, wide_p, client, wh,
                               self.n_anchors, len(future_ds))
        # inf price ratios -> missing (see GlobalXGB); features.py untouched.
        Xf = pd.DataFrame(X, columns=FEATURE_NAMES).replace(
            [np.inf, -np.inf], np.nan)
        for c in CATEGORICALS:
            Xf[c] = Xf[c].astype("int32")

        model = CatBoostRegressor(**self.cat_params)
        model.fit(Pool(Xf, y, cat_features=CATEGORICALS))

        Xp, ids = prediction_matrix(wide_y, wide_p, client, wh, future_ds)
        Xpf = pd.DataFrame(Xp, columns=FEATURE_NAMES).replace(
            [np.inf, -np.inf], np.nan)
        for c in CATEGORICALS:
            Xpf[c] = Xpf[c].astype("int32")
        ids["yhat"] = np.clip(model.predict(Pool(Xpf, cat_features=CATEGORICALS)),
                              0.0, None)
        return ids
