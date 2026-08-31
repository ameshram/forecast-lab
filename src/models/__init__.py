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
from .deep import GlobalNHITS, GlobalPatchTST
from .gbm import GlobalLGBM
from .seasonal import SeasonalUpliftNaive
from .timesfm_fm import TimesFMZeroShot

REGISTRY = {
    "zero": ZeroForecast,
    "naive_last": NaiveLast,
    "seasonal_naive": SeasonalNaive,
    "moving_average": MovingAverage,
    "segment_router": SegmentRouter,
    "blend": Blend,
    "bias_scaled": BiasScaled,
    "seasonal_uplift_naive": SeasonalUpliftNaive,
    "global_lgbm": GlobalLGBM,
    "chronos": ChronosZeroShot,
    "timesfm": TimesFMZeroShot,
    "patchtst": GlobalPatchTST,
    "nhits": GlobalNHITS,
}


def build_model(config: dict):
    """Instantiate a model from {'name': ..., 'params': {...}}."""
    name = config["name"]
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](**config.get("params", {}))
