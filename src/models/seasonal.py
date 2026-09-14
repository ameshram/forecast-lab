"""Seasonally re-anchored naive (Cycle 1, Option C).

Hypothesis: last-week naive is right about level but blind to calendar
uplift (diagnosed -22% bias on the Oct-2022 origin, and Phase 1 is an
Oct->Jan window). Forecast = last observed value x a damped ratio of
week-of-year demand indices:

    yhat(series, t) = base(series) * clip(1 + alpha * (idx_c(woy_t)/idx_c(woy_anchor) - 1))

where idx_c is the client's mean weekly total for that ISO week divided by
the client's overall mean weekly total (portfolio index as fallback), and
woy_anchor is the last training week. alpha damps the ratio; clip bounds it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import SERIES_ID
from .baselines import _wide


def _woy(ds: pd.Series) -> pd.Series:
    return ds.dt.isocalendar().week.astype(int)


class SeasonalUpliftNaive:
    name = "seasonal_uplift_naive"

    def __init__(self, alpha: float = 0.5, clip_low: float = 0.5,
                 clip_high: float = 2.0):
        self.alpha = alpha
        self.clip_low = clip_low
        self.clip_high = clip_high

    def fit_predict(self, train, future_ds):
        wide = _wide(train)
        base = wide.ffill(axis=1).iloc[:, -1].fillna(0.0)
        client_of = train.groupby(SERIES_ID)["Client"].first()

        ct = train.groupby(["Client", "ds"])["y"].sum().reset_index()
        ct["woy"] = _woy(ct["ds"])
        cli_idx = ct.groupby(["Client", "woy"])["y"].mean().div(
            ct.groupby("Client")["y"].mean(), level=0)

        pt = train.groupby("ds")["y"].sum().reset_index()
        pt["woy"] = _woy(pt["ds"])
        port_idx = pt.groupby("woy")["y"].mean() / pt["y"].mean()

        def idx(client, woy):
            v = cli_idx.get((client, woy), np.nan)
            if pd.isna(v):
                v = port_idx.get(woy, np.nan)
            return v

        anchor_woy = int(train["ds"].max().isocalendar().week)
        clients = client_of.unique()
        frames = []
        for ds in future_ds:
            woy_t = int(pd.Timestamp(ds).isocalendar().week)
            ratio = {}
            for c in clients:
                num, den = idx(c, woy_t), idx(c, anchor_woy)
                r = num / den if (num and den and not pd.isna(num)
                                  and not pd.isna(den) and den > 0) else 1.0
                r = 1.0 + self.alpha * (r - 1.0)
                ratio[c] = float(np.clip(r, self.clip_low, self.clip_high))
            mult = client_of.map(ratio)
            frames.append(pd.DataFrame({
                SERIES_ID: base.index, "ds": ds,
                "yhat": (base * mult.reindex(base.index).fillna(1.0)).values,
            }))
        return pd.concat(frames, ignore_index=True)
