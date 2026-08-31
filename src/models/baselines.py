"""Baseline forecasters.

Contract (all models): fit_predict(train, future_ds) -> DataFrame with columns
[client_warehouse_product_id, ds, yhat], one row per series per future week,
covering every series present in `train`. Models must only look at `train`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import SERIES_ID


def _wide(train: pd.DataFrame) -> pd.DataFrame:
    """Pivot train to series x week matrix of sales."""
    return train.pivot_table(index=SERIES_ID, columns="ds", values="y", aggfunc="sum")


def _expand(per_series: pd.Series, future_ds: list[pd.Timestamp]) -> pd.DataFrame:
    """Repeat one value per series across all future weeks."""
    out = pd.DataFrame(
        {SERIES_ID: np.repeat(per_series.index.values, len(future_ds)),
         "ds": np.tile(np.array(future_ds, dtype="datetime64[ns]"), len(per_series)),
         "yhat": np.repeat(per_series.values, len(future_ds))}
    )
    return out


def _dead_mask(wide: pd.DataFrame, weeks: int) -> pd.Series:
    """True for series with zero total sales over the trailing `weeks` weeks."""
    return wide.iloc[:, -weeks:].fillna(0).sum(axis=1) == 0


class ZeroForecast:
    """Predict zero everywhere. Floor of the leaderboard, sanity check."""
    name = "zero"

    def fit_predict(self, train, future_ds):
        series = train[SERIES_ID].unique()
        return _expand(pd.Series(0.0, index=series), future_ds)


class NaiveLast:
    """Repeat each series' last observed week."""
    name = "naive_last"

    def fit_predict(self, train, future_ds):
        wide = _wide(train)
        return _expand(wide.ffill(axis=1).iloc[:, -1].fillna(0.0), future_ds)


class MovingAverage:
    """Mean of the trailing `window` weeks; optionally force dead series to 0.

    zero_dead_weeks: if set, series with no sales in that many trailing weeks
    are forecast as 0 regardless of the moving average.
    """
    name = "moving_average"

    def __init__(self, window: int = 13, zero_dead_weeks: int | None = None):
        self.window = window
        self.zero_dead_weeks = zero_dead_weeks

    def fit_predict(self, train, future_ds):
        wide = _wide(train)
        avg = wide.iloc[:, -self.window:].fillna(0).mean(axis=1)
        if self.zero_dead_weeks:
            avg[_dead_mask(wide, self.zero_dead_weeks)] = 0.0
        return _expand(avg, future_ds)


class SeasonalNaive:
    """Use the value from `season` weeks before each target week.

    Falls back to the series' last observed value when the seasonal week
    predates the series history (young series).
    """
    name = "seasonal_naive"

    def __init__(self, season: int = 52):
        self.season = season

    def fit_predict(self, train, future_ds):
        wide = _wide(train).fillna(0)
        fallback = wide.iloc[:, -1]
        frames = []
        for ds in future_ds:
            ref = ds - pd.Timedelta(weeks=self.season)
            vals = wide[ref] if ref in wide.columns else fallback
            frames.append(pd.DataFrame({SERIES_ID: wide.index, "ds": ds,
                                        "yhat": vals.values}))
        return pd.concat(frames, ignore_index=True)
