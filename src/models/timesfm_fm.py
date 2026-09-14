"""TimesFM zero-shot forecaster (Stage 1; v3 branch added Cycle 13-b).

Wraps a pre-trained TimesFM torch checkpoint. Nothing is fitted:
fit_predict feeds each series' history and reads a point/quantile forecast.
Heavy imports live inside methods so the registry stays importable without
the DL stack installed.

v3 (timesfm-3.0, `google/timesfm-3.0-pytorch`, timesfm pip >=3.0):
TimesFM3Forecaster.predict_batch returns ForecastOutput objects with
`.forecast` (h,) — the native point head — and `.quantiles` (h, 9) — the
nine quantile heads 0.1..0.9. Readouts: point='native' (the point head),
'quantile' (the configured quantile, 0.5 = median), 'qmean' (mean of the
nine heads — the trimmed-mean readout that fixed Bolt's bias). Per the
Cycle-10 lesson, NO readout calibration is assumed from other families —
the semantics test + bias smoke run before any full run. Univariate only:
covariates deliberately not wired (memo 2026-09-01-cycle13b).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import SERIES_ID
from .chronos import history_matrix


class TimesFMZeroShot:
    name = "timesfm"

    QMEAN_LEVELS = 9  # v3 quantile heads are 0.1..0.9 in order

    def __init__(self, checkpoint: str = "google/timesfm-2.5-200m-pytorch",
                 batch_size: int = 128, max_context: int = 512,
                 max_horizon: int = 64, context_weeks: int | None = None,
                 point: str = "native", quantile: float = 0.5,
                 clip_negative: bool = True):
        if point not in ("native", "quantile", "qmean"):
            raise ValueError("timesfm point must be native|quantile|qmean")
        self.checkpoint = checkpoint
        self.batch_size = batch_size
        self.max_context = max_context
        self.max_horizon = max_horizon
        self.context_weeks = context_weeks
        self.point = point
        self.quantile = quantile
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

    def _is_v3(self) -> bool:
        return "timesfm-3" in self.checkpoint

    def fit_predict(self, train, future_ds):
        wide = history_matrix(train, self.context_weeks)
        ids = wide.index.values
        h = len(future_ds)
        if self._is_v3():
            yhat = self._forecast_v3(wide.values, h)
        else:
            if self.point != "native":
                raise ValueError(
                    "point readouts other than 'native' are v3-only")
            model = self._compiled()
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

    def _forecast_v3(self, values: np.ndarray, h: int) -> np.ndarray:
        """TimesFM-3 path: predict_batch -> ForecastOutput(.forecast (h,),
        .quantiles (h, 9) = levels 0.1..0.9)."""
        import timesfm
        if self._model is None:
            self._model = timesfm.TimesFM3Forecaster.from_pretrained(
                self.checkpoint, device="cpu")
        model = self._model
        ctx = values[:, -self.max_context:].astype(np.float32)
        n = len(ctx)
        yhat = np.empty((n, h), dtype=float)
        for start in range(0, n, self.batch_size):
            chunk = [row for row in ctx[start:start + self.batch_size]]
            outs = list(model.predict_batch(
                chunk, horizon=h, return_quantiles=True))
            for i, o in enumerate(outs):
                q = np.asarray(o.quantiles, dtype=float)   # (h, 9)
                if q.shape != (h, self.QMEAN_LEVELS):
                    raise ValueError(
                        f"unexpected quantile shape {q.shape}")
                if self.point == "native":
                    yhat[start + i] = np.asarray(o.forecast, dtype=float)[:h]
                elif self.point == "qmean":
                    yhat[start + i] = q.mean(axis=1)
                else:
                    # heads are 0.1..0.9: index = round(10*q) - 1
                    idx = int(round(self.quantile * 10)) - 1
                    if not 0 <= idx < self.QMEAN_LEVELS:
                        raise ValueError(
                            f"quantile {self.quantile} outside 0.1..0.9")
                    yhat[start + i] = q[:, idx]
        return yhat
