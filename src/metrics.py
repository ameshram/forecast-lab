"""Evaluation metrics. Frozen: changes here invalidate all ledger history.

Primary metric is the official VN1 competition score:

    score = ( sum|F - A|  +  | sum(F) - sum(A) | ) / sum(A)

i.e. WAPE plus an absolute portfolio-bias term, computed over all
series-week cells in the evaluation window. Lower is better.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def vn1_score(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype="float64")
    yhat = np.asarray(yhat, dtype="float64")
    denom = y.sum()
    if denom == 0:
        return np.nan
    abs_err = np.abs(yhat - y).sum()
    bias = abs(yhat.sum() - y.sum())
    return (abs_err + bias) / denom


def wape(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype="float64")
    yhat = np.asarray(yhat, dtype="float64")
    denom = y.sum()
    return np.abs(yhat - y).sum() / denom if denom else np.nan


def bias_pct(y: np.ndarray, yhat: np.ndarray) -> float:
    """Signed portfolio bias: positive = over-forecasting."""
    y = np.asarray(y, dtype="float64")
    yhat = np.asarray(yhat, dtype="float64")
    denom = y.sum()
    return (yhat.sum() - y.sum()) / denom if denom else np.nan


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype="float64")
    yhat = np.asarray(yhat, dtype="float64")
    return float(np.abs(yhat - y).mean())


ALL_METRICS = {"vn1": vn1_score, "wape": wape, "bias_pct": bias_pct, "mae": mae}


def score_frame(df: pd.DataFrame, y_col: str = "y", yhat_col: str = "yhat") -> dict:
    """All metrics over the rows of a merged actuals/forecast frame."""
    return {name: fn(df[y_col].values, df[yhat_col].values)
            for name, fn in ALL_METRICS.items()}


def score_by_segment(df: pd.DataFrame, segment_col: str,
                     y_col: str = "y", yhat_col: str = "yhat") -> pd.DataFrame:
    """Metrics per segment, plus each segment's share of actual volume."""
    rows = []
    total = df[y_col].sum()
    for seg, g in df.groupby(segment_col, observed=True):
        row = {"segment": seg, "n_series": g["client_warehouse_product_id"].nunique(),
               "volume_share": g[y_col].sum() / total if total else np.nan}
        row.update(score_frame(g, y_col, yhat_col))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("volume_share", ascending=False)
