"""Chronos zero-shot forecaster (Stage 1).

Wraps a pre-trained Chronos checkpoint (Bolt or T5 family). Nothing is
fitted: fit_predict builds one context vector per series from `train` and
decodes quantile forecasts. Heavy imports live inside methods so the
registry stays importable without the DL stack installed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import SERIES_ID


def history_matrix(train: pd.DataFrame,
                   context_weeks: int | None = None) -> pd.DataFrame:
    """Series x week matrix of sales, oldest week first, NaN as 0."""
    wide = train.pivot_table(index=SERIES_ID, columns="ds", values="y",
                             aggfunc="sum")
    wide = wide.sort_index(axis=1).fillna(0.0)
    if context_weeks:
        wide = wide.iloc[:, -context_weeks:]
    return wide


class ChronosZeroShot:
    """Zero-shot Chronos. point='quantile' takes the configured quantile
    (0.5 = median); point='mean' takes the pipeline's mean forecast (for
    Bolt checkpoints the API returns the median as the mean — see the NOTE
    in chronos_bolt.predict_quantiles); point='qmean' averages the nine
    trained quantile heads (0.1..0.9) — a trimmed-mean readout."""
    name = "chronos"

    QMEAN_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self, checkpoint: str = "amazon/chronos-bolt-base",
                 batch_size: int = 256, quantile: float = 0.5,
                 point: str = "quantile", context_weeks: int | None = None,
                 device: str = "cpu", clip_negative: bool = True):
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.quantile = quantile
        self.point = point
        self.context_weeks = context_weeks
        self.device = device
        self.clip_negative = clip_negative
        self._pipe = None

    def _pipeline(self):
        if self._pipe is None:
            import torch
            from chronos import BaseChronosPipeline
            self._pipe = BaseChronosPipeline.from_pretrained(
                self.checkpoint, device_map=self.device,
                torch_dtype=torch.float32)
        return self._pipe

    def fit_predict(self, train, future_ds):
        import torch
        pipe = self._pipeline()
        wide = history_matrix(train, self.context_weeks)
        ids = wide.index.values
        h = len(future_ds)
        ctx = torch.tensor(wide.values, dtype=torch.float32)
        levels = (self.QMEAN_LEVELS if self.point == "qmean"
                  else [self.quantile])
        chunks = []
        for start in range(0, len(ids), self.batch_size):
            quantiles, mean = pipe.predict_quantiles(
                ctx[start:start + self.batch_size], prediction_length=h,
                quantile_levels=levels)
            if self.point == "mean":
                block = mean
            elif self.point == "qmean":
                block = quantiles.mean(dim=-1)
            else:
                block = quantiles[:, :, 0]
            chunks.append(block.cpu().numpy())
        yhat = np.vstack(chunks).astype(float)
        if self.clip_negative:
            yhat = np.maximum(yhat, 0.0)
        return pd.DataFrame({
            SERIES_ID: np.repeat(ids, h),
            "ds": np.tile(np.array(future_ds, dtype="datetime64[ns]"),
                          len(ids)),
            "yhat": yhat.ravel(),
        })
