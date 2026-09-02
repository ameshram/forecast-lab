"""IBM Granite TinyTimeMixer (TTM) zero-shot forecaster (Cycle 10 zoo, Tier 4).

Wraps a pre-trained granite-timeseries-ttm checkpoint via tsfm_public
(granite-tsfm). Nothing is fitted: fit_predict batches one fixed-length
context window per series and reads the point forecast head (TTM is
MSE-trained - a single point output, no quantile/mean readout ambiguity).
Heavy imports live inside methods so the registry stays importable without
granite-tsfm installed (its stack is RED against the main venv - runs
happen from a throwaway isolated venv).

Checkpoint selection is delegated to tsfm_public.get_model, which picks the
revision matching (context_length, prediction_length) - for weekly VN1 with
context 90 and h=13 that is revision 90-30-ft-r2.1 with
prediction_filter_length=13 (verified 2026-08-31). The r2.1 revisions are
frequency-prefix-tuned and REQUIRE a freq_token in forward (weekly = 9 in
DEFAULT_FREQUENCY_MAPPING); forward raises without it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import SERIES_ID
from .chronos import history_matrix


class TTMZeroShot:
    name = "ttm"

    def __init__(self, checkpoint: str = "ibm-granite/granite-timeseries-ttm-r2",
                 context_length: int = 90, batch_size: int = 1024,
                 freq: str = "W", clip_negative: bool = True):
        self.checkpoint = checkpoint
        self.context_length = context_length
        self.batch_size = batch_size
        self.freq = freq
        self.clip_negative = clip_negative

    def fit_predict(self, train, future_ds):
        import torch
        from tsfm_public.toolkit.get_model import get_model
        from tsfm_public.toolkit.time_series_preprocessor import (
            DEFAULT_FREQUENCY_MAPPING,
        )

        h = len(future_ds)
        wide = history_matrix(train, self.context_length)
        if wide.shape[1] < self.context_length:
            raise ValueError(
                f"history ({wide.shape[1]}w) shorter than TTM context "
                f"({self.context_length}w) - pick a smaller context_length")
        ids = wide.index.values
        ctx = wide.values.astype(np.float32)[:, :, None]  # (n, ctx, 1)

        model = get_model(self.checkpoint,
                          context_length=self.context_length,
                          prediction_length=h)
        model.eval()
        tok = DEFAULT_FREQUENCY_MAPPING[self.freq]

        chunks = []
        with torch.no_grad():
            for start in range(0, len(ids), self.batch_size):
                past = torch.tensor(ctx[start:start + self.batch_size])
                out = model(
                    past_values=past,
                    freq_token=torch.full((past.shape[0],), tok,
                                          dtype=torch.long))
                chunks.append(out.prediction_outputs[:, :h, 0].cpu().numpy())
        yhat = np.vstack(chunks).astype(float)
        if self.clip_negative:
            yhat = np.maximum(yhat, 0.0)
        return pd.DataFrame({
            SERIES_ID: np.repeat(ids, h),
            "ds": np.tile(np.array(future_ds, dtype="datetime64[ns]"),
                          len(ids)),
            "yhat": yhat.ravel(),
        })
