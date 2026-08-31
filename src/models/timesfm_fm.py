"""TimesFM zero-shot forecaster (Stage 1).

Wraps the pre-trained TimesFM 2.5 torch checkpoint. Nothing is fitted:
fit_predict feeds each series' history and takes the model's native point
forecast. Heavy imports live inside methods so the registry stays
importable without the DL stack installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import SERIES_ID
from .chronos import history_matrix


class TimesFMZeroShot:
    name = "timesfm"

    def __init__(self, checkpoint: str = "google/timesfm-2.5-200m-pytorch",
                 batch_size: int = 128, max_context: int = 512,
                 max_horizon: int = 64, context_weeks: int | None = None,
                 clip_negative: bool = True):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.context_weeks = context_weeks
        self.clip_negative = clip_negative
        self._model = None

    def _compiled(self):
        if self._model is None:
            import timesfm
            model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
                self.checkpoint)
            model.compile(timesfm.ForecastConfig(
                max_context=self.max_context,
                max_horizon=self.max_horizon,
                normalize_inputs=True,
                per_core_batch_size=self.batch_size,
                use_continuous_quantile_head=False,
                infer_is_positive=True,
            ))
            self._model = model
        return self._model

    def fit_predict(self, train, future_ds):
        model = self._compiled()
        wide = history_matrix(train, self.context_weeks)
        ids = wide.index.values
        h = len(future_ds)
        inputs = [row.astype(np.float32) for row in wide.values]
        point, _quantiles = model.forecast(horizon=h, inputs=inputs)
        yhat = np.asarray(point, dtype=float)[:, :h]
        if self.clip_negative:
            yhat = np.maximum(yhat, 0.0)
        return pd.DataFrame({
            SERIES_ID: np.repeat(ids, h),
            "ds": np.tile(np.array(future_ds, dtype="datetime64[ns]"),
                          len(ids)),
            "yhat": yhat.ravel(),
        })
