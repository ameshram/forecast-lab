"""Rolling-origin backtest: the single evaluation path for every model.

For each origin, the model sees only data strictly before the origin week and
forecasts the next `horizon` weeks. Results are scored overall and by segment
so the self-improvement loop's DIAGNOSE stage has per-segment evidence.
"""
from __future__ import annotations

import pandas as pd

from .data import SERIES_ID, syntetos_boylan_class, volume_decile
from .metrics import score_by_segment, score_frame


def run_backtest(df: pd.DataFrame, model, origins: list[pd.Timestamp],
                 horizon: int) -> dict:
    """Backtest `model` over `origins`.

    Returns {'per_origin': DataFrame, 'pooled': dict, 'segments': DataFrame,
    'forecasts': DataFrame} where segments pools all origins.
    """
    merged_all = []
    origin_rows = []
    for origin in origins:
        train = df[df.ds < origin]
        future_ds = [origin + pd.Timedelta(weeks=i) for i in range(horizon)]
        actuals = df[df.ds.isin(future_ds)][[SERIES_ID, "ds", "y"]]

        fcst = model.fit_predict(train, future_ds)
        merged = actuals.merge(fcst, on=[SERIES_ID, "ds"], how="left")
        n_missing = merged["yhat"].isna().sum()
        if n_missing:
            raise ValueError(f"{model.name}: {n_missing} unforecast cells at "
                             f"origin {origin.date()}")

        # attach segments computed on this origin's training data only
        sb = syntetos_boylan_class(train)
        vd = volume_decile(train)
        merged = merged.join(sb, on=SERIES_ID).join(vd, on=SERIES_ID)
        merged["origin"] = origin

        origin_rows.append({"origin": origin, **score_frame(merged)})
        merged_all.append(merged)

    merged_all = pd.concat(merged_all, ignore_index=True)
    return {
        "per_origin": pd.DataFrame(origin_rows),
        "pooled": score_frame(merged_all),
        "segments": score_by_segment(merged_all, "sb_class"),
        "volume_segments": score_by_segment(merged_all, "volume_decile"),
        "forecasts": merged_all,
    }
