"""Moirai 2.0 zero-shot forecaster (Cycle 10 zoo, Tier 4).

Wraps a pre-trained Salesforce Moirai-2 checkpoint via uni2ts. Nothing is
fitted: fit_predict builds one gluonts entry per series from `train` and
decodes quantile forecasts. Heavy imports live inside methods so the
registry stays importable without uni2ts installed (the uni2ts stack is
RED against the main venv - runs happen from a throwaway isolated venv).

Point-readout semantics (verified 2026-08-31 on the real checkpoint): the
predictor emits gluonts QuantileForecast objects with the nine heads
0.1..0.9; QuantileForecast.mean RETURNS THE MEDIAN with a warning ("mean
prediction is not stored") - the same median-as-mean trap as Chronos-Bolt
(524453e7d8) and Chronos-2. So this wrapper offers no 'mean' readout at
all: point='quantile' (default, the median at quantile=0.5) or
point='qmean' (average of the nine heads - CAUTION: the qmean readout
exploded on lumpy for Chronos-2, run 92906cd09e; check bias on a real
subsample before trusting it here).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import SERIES_ID
from .chronos import history_matrix


class MoiraiZeroShot:
    name = "moirai"

    QMEAN_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def __init__(self, checkpoint: str = "Salesforce/moirai-2.0-R-small",
                 batch_size: int = 64, quantile: float = 0.5,
                 point: str = "quantile", context_weeks: int | None = None,
                 max_context: int = 512, device: str = "cpu",
                 clip_negative: bool = True):
        if point not in ("quantile", "qmean"):
            raise ValueError(
                "moirai point must be 'quantile' or 'qmean' - there is no "
                "genuine mean head (QuantileForecast.mean is the median)")
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.quantile = quantile
        self.point = point
        self.context_weeks = context_weeks
        self.max_context = max_context
        self.device = device
        self.clip_negative = clip_negative

    def fit_predict(self, train, future_ds):
        import torch
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

        wide = history_matrix(train, self.context_weeks)
        if wide.shape[1] > self.max_context:
            wide = wide.iloc[:, -self.max_context:]
        ids = wide.index.values
        h = len(future_ds)

        model = Moirai2Forecast(
            module=Moirai2Module.from_pretrained(self.checkpoint),
            prediction_length=h, context_length=wide.shape[1],
            target_dim=1, feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0)
        predictor = model.create_predictor(batch_size=self.batch_size,
                                           device=self.device)

        start = pd.Period(wide.columns[0], freq="W-MON")
        values = wide.values.astype(np.float32)
        entries = [{"item_id": str(ids[i]), "start": start,
                    "target": values[i]} for i in range(len(ids))]

        yhat = np.empty((len(ids), h), dtype=float)
        with torch.no_grad():
            for i, fc in enumerate(predictor.predict(entries)):
                # predictor preserves input order; assert to be safe
                assert fc.item_id == str(ids[i]), (
                    f"forecast order mismatch at {i}: {fc.item_id}")
                if self.point == "qmean":
                    yhat[i] = np.stack(
                        [fc.quantile(q) for q in self.QMEAN_LEVELS]
                    ).mean(axis=0)
                else:
                    yhat[i] = fc.quantile(self.quantile)
        if self.clip_negative:
            yhat = np.maximum(yhat, 0.0)
        return pd.DataFrame({
            SERIES_ID: np.repeat(ids, h),
            "ds": np.tile(np.array(future_ds, dtype="datetime64[ns]"),
                          len(ids)),
            "yhat": yhat.ravel(),
        })
