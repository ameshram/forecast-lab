"""Model registry. Every model is constructed from a config dict so the
self-improvement loop can propose experiments as pure config diffs."""
from __future__ import annotations

from .baselines import (
    MovingAverage,
    NaiveLast,
    SeasonalNaive,
    ZeroForecast,
)
from .chronos import ChronosZeroShot
from .composite import BiasScaled, Blend, SegmentRouter
from .deep import GlobalDeepAR, GlobalNHITS, GlobalPatchTST, GlobalTFT
from .gbm import GlobalCatBoost, GlobalLGBM, GlobalXGB
from .moirai_fm import MoiraiZeroShot
from .seasonal import SeasonalUpliftNaive
from .statsforecast_models import StatsForecastModel
from .timesfm_fm import TimesFMZeroShot
from .ttm_fm import TTMZeroShot
from .weight_fit import WeightFitBlend

REGISTRY = {
    "zero": ZeroForecast,
    "naive_last": NaiveLast,
    "seasonal_naive": SeasonalNaive,
    "moving_average": MovingAverage,
    "segment_router": SegmentRouter,
    "blend": Blend,
    "weight_fit_blend": WeightFitBlend,
    "bias_scaled": BiasScaled,
    "seasonal_uplift_naive": SeasonalUpliftNaive,
    "global_lgbm": GlobalLGBM,
    "global_xgb": GlobalXGB,
    "global_cat": GlobalCatBoost,
    "chronos": ChronosZeroShot,
    "timesfm": TimesFMZeroShot,
    "moirai": MoiraiZeroShot,
    "ttm": TTMZeroShot,
    "patchtst": GlobalPatchTST,
    "nhits": GlobalNHITS,
    "deepar": GlobalDeepAR,
    "tft": GlobalTFT,
    "statsforecast": StatsForecastModel,
}


def build_model(config: dict):
    """Instantiate a model from {'name': ..., 'params': {...}}."""
    name = config["name"]
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](**config.get("params", {}))
