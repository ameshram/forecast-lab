"""Data loading, phase boundaries, and series segmentation.

The dataset is the VN1 Forecasting Accuracy Challenge data: weekly sales (y)
and price for Client-Warehouse-Product series. `raw_data.parquet` is a
complete panel: every series has a row for every week in both phases.

Phase 1 (2023-10-09 onward) is the FINAL TEST SET. It must never be used
for model selection -- only for the one-shot final evaluation.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SERIES_ID = "client_warehouse_product_id"
PHASE1_START = pd.Timestamp("2023-10-09")
HORIZON = 13  # weeks, matching the Phase 1 evaluation window

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw_data.parquet"


def load_panel(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Load the full long-format panel, sorted by series and week."""
    df = pd.read_parquet(path)
    df = df.sort_values([SERIES_ID, "ds"]).reset_index(drop=True)
    df["y"] = df["y"].astype("float64")
    return df


def split_phases(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (phase0, phase1). Phase 1 is the untouchable final test."""
    return df[df.ds < PHASE1_START].copy(), df[df.ds >= PHASE1_START].copy()


def validation_origins(df: pd.DataFrame, n_origins: int, horizon: int = HORIZON,
                       step: int | None = None) -> list[pd.Timestamp]:
    """Rolling forecast origins inside Phase 0, latest first.

    An origin is the FIRST week being forecast. The latest origin is chosen so
    the forecast window ends exactly at the last Phase 0 week; earlier origins
    step back by `step` weeks (default: one full horizon, non-overlapping).
    """
    step = step or horizon
    phase0_end = df.loc[df.ds < PHASE1_START, "ds"].max()
    last_origin = phase0_end - pd.Timedelta(weeks=horizon - 1)
    origins = [last_origin - pd.Timedelta(weeks=step * i) for i in range(n_origins)]
    first_train_weeks = ((min(origins) - df.ds.min()).days // 7)
    if first_train_weeks < 60:
        raise ValueError(
            f"Earliest origin {min(origins).date()} leaves only "
            f"{first_train_weeks} training weeks; reduce n_origins/step."
        )
    return sorted(origins)


def syntetos_boylan_class(train: pd.DataFrame) -> pd.Series:
    """Classify each series (smooth/intermittent/erratic/lumpy) on train data.

    Cut-offs: ADI 1.32, CV^2 0.49 (Syntetos & Boylan 2005). Series with fewer
    than two non-zero observations are 'insufficient'.
    """
    def classify(s: pd.Series) -> str:
        s = s.dropna()
        nonzero = s[s > 0]
        if len(nonzero) < 2:
            return "insufficient"
        adi = len(s) / len(nonzero)
        cv2 = (nonzero.std() / nonzero.mean()) ** 2
        if adi < 1.32:
            return "smooth" if cv2 < 0.49 else "erratic"
        return "intermittent" if cv2 < 0.49 else "lumpy"

    return train.groupby(SERIES_ID)["y"].apply(classify).rename("sb_class")


def volume_decile(train: pd.DataFrame) -> pd.Series:
    """Decile (1=smallest .. 10=largest) of total training volume per series."""
    totals = train.groupby(SERIES_ID)["y"].sum()
    return (pd.qcut(totals.rank(method="first"), 10, labels=False) + 1).rename("volume_decile")
