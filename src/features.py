"""Feature construction for the global GBM (Cycle 2, Option C).

Leakage discipline: every feature for a sample anchored at week `ai` is
computed exclusively from columns <= ai of the training slice's wide
matrix, and targets are the anchor's future weeks *inside the same training
slice*. The harness hands models a train slice that already excludes the
backtest window, so nothing here can see evaluation data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SERIES_ID

LAGS = [1, 2, 3, 4, 8, 13, 26, 52]
ROLLS = [4, 13, 26]

FEATURE_NAMES = (
    [f"lag{k}" for k in LAGS]
    + [x for w in ROLLS for x in (f"rmean{w}", f"rstd{w}", f"nzrate{w}")]
    + ["wsls", "age", "price", "price_chg4",
       "h", "woy", "month", "client", "warehouse"]
)
CATEGORICALS = ["client", "warehouse"]


def build_wide(train: pd.DataFrame):
    wide_y = train.pivot_table(index=SERIES_ID, columns="ds", values="y",
                               aggfunc="sum")
    wide_p = (train.pivot_table(index=SERIES_ID, columns="ds", values="Price",
                                aggfunc="mean")
              .reindex(index=wide_y.index, columns=wide_y.columns)
              .ffill(axis=1))
    client = train.groupby(SERIES_ID)["Client"].first().reindex(wide_y.index)
    wh = train.groupby(SERIES_ID)["Warehouse"].first().reindex(wide_y.index)
    return wide_y, wide_p, client.values.astype("int32"), wh.values.astype("int32")


def _nonzero_indices(y: np.ndarray):
    """last_nz[:, j] = last column index <= j with y>0 (-1 if none);
    first_nz[i] = first column index with y>0 (n_cols if never)."""
    nz = y > 0
    cols = np.arange(y.shape[1])
    last_nz = np.maximum.accumulate(np.where(nz, cols, -1), axis=1)
    first_nz = np.where(nz.any(axis=1), nz.argmax(axis=1), y.shape[1])
    return last_nz, first_nz


def _anchor_block(y, p, last_nz, first_nz, ai: int) -> np.ndarray:
    """Feature block (n_series x n_base_features) for anchor column ai."""
    n = y.shape[0]
    cols = []
    for k in LAGS:
        j = ai - (k - 1)
        cols.append(y[:, j] if j >= 0 else np.full(n, np.nan, dtype="float32"))
    for w in ROLLS:
        lo = max(0, ai - w + 1)
        sl = y[:, lo:ai + 1]
        cols.append(sl.mean(axis=1))
        cols.append(sl.std(axis=1))
        cols.append((sl > 0).mean(axis=1))
    wsls = np.where(last_nz[:, ai] >= 0, ai - last_nz[:, ai], 999).astype("float32")
    age = np.where(first_nz <= ai, ai - first_nz, -1).astype("float32")
    price = p[:, ai]
    j4 = ai - 4
    with np.errstate(divide="ignore", invalid="ignore"):
        chg4 = (price / p[:, j4]) if j4 >= 0 else np.full(n, np.nan, dtype="float32")
    cols += [wsls, age, price, np.asarray(chg4, dtype="float32")]
    return np.column_stack(cols).astype("float32")


def training_matrix(wide_y: pd.DataFrame, wide_p: pd.DataFrame,
                    client: np.ndarray, wh: np.ndarray,
                    n_anchors: int, horizons: int):
    """Stack (anchor x horizon) samples: X (float32), target y."""
    y = wide_y.fillna(0.0).to_numpy(dtype="float32")
    p = wide_p.to_numpy(dtype="float32")
    dates = list(wide_y.columns)
    n_cols = len(dates)
    last_nz, first_nz = _nonzero_indices(y)

    X_parts, y_parts = [], []
    first_anchor = max(0, n_cols - 1 - n_anchors)
    for ai in range(first_anchor, n_cols - 1):
        base = _anchor_block(y, p, last_nz, first_nz, ai)
        for h in range(1, horizons + 1):
            tj = ai + h
            if tj >= n_cols:
                break
            t = pd.Timestamp(dates[tj])
            iso = t.isocalendar()
            extra = np.column_stack([
                np.full(len(base), h, dtype="float32"),
                np.full(len(base), int(iso.week), dtype="float32"),
                np.full(len(base), t.month, dtype="float32"),
                client.astype("float32"), wh.astype("float32"),
            ])
            X_parts.append(np.hstack([base, extra]))
            y_parts.append(y[:, tj])
    return np.vstack(X_parts), np.concatenate(y_parts)


def prediction_matrix(wide_y: pd.DataFrame, wide_p: pd.DataFrame,
                      client: np.ndarray, wh: np.ndarray,
                      future_ds: list) -> tuple[np.ndarray, pd.DataFrame]:
    """Feature rows anchored at the last training week, one per
    (series, future week); returns (X, id-frame aligned row-for-row)."""
    y = wide_y.fillna(0.0).to_numpy(dtype="float32")
    p = wide_p.to_numpy(dtype="float32")
    last_nz, first_nz = _nonzero_indices(y)
    ai = y.shape[1] - 1
    base = _anchor_block(y, p, last_nz, first_nz, ai)

    X_parts, ids = [], []
    for h, ds in enumerate(future_ds, start=1):
        t = pd.Timestamp(ds)
        iso = t.isocalendar()
        extra = np.column_stack([
            np.full(len(base), h, dtype="float32"),
            np.full(len(base), int(iso.week), dtype="float32"),
            np.full(len(base), t.month, dtype="float32"),
            client.astype("float32"), wh.astype("float32"),
        ])
        X_parts.append(np.hstack([base, extra]))
        ids.append(pd.DataFrame({SERIES_ID: wide_y.index, "ds": t}))
    return np.vstack(X_parts), pd.concat(ids, ignore_index=True)
