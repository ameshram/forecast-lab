"""Contract + point-readout tests for the Stage 2 PatchTST-NB wrapper.

Skipped if neuralforecast is not installed. Torch is forced single-threaded
(two OpenMP runtimes deadlock on macOS — see tests/test_stage1_foundation.py).
Trains a tiny model on a small panel; the point is contract/semantics, not fit
quality, so max_steps is small.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import SERIES_ID
from src.models import build_model


def _panel(n_series=24, n_weeks=110, seed=5):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")
    rows = []
    for i in range(n_series):
        if i < 4:                       # all-zero series (scaler NaN stress)
            y = np.zeros(n_weeks)
        elif i < 14:                    # intermittent
            y = rng.poisson(0.3, n_weeks).astype(float)
        else:                           # right-skewed counts
            y = rng.poisson(6.0, n_weeks).astype(float)
        rows.append(pd.DataFrame({SERIES_ID: f"0_{i % 3}_{i}", "Client": 0,
                                  "Warehouse": i % 3, "Product": i, "ds": ds,
                                  "y": y}))
    return pd.concat(rows, ignore_index=True)


def _future(train, n=13):
    last = train.ds.max()
    return [last + pd.Timedelta(weeks=i + 1) for i in range(n)]


def _tiny_model():
    pytest.importorskip("neuralforecast")
    import torch
    torch.set_num_threads(1)
    return build_model({"name": "patchtst", "params": {
        "accelerator": "cpu",
        "model_params": {"input_size": 32, "patch_len": 8, "stride": 8,
                         "max_steps": 40, "hidden_size": 32, "n_heads": 4,
                         "batch_size": 64, "windows_batch_size": 128,
                         "random_seed": 7}}})


def test_patchtst_contract_full_coverage_nonneg():
    train = _panel()
    future = _future(train, 13)
    out = _tiny_model().fit_predict(train, future)
    # full coverage: one row per series per future week
    assert len(out) == train[SERIES_ID].nunique() * 13
    assert set(out.columns) == {SERIES_ID, "ds", "yhat"}
    assert out["yhat"].notna().all(), "NB point forecast must never be NaN"
    assert (out["yhat"] >= 0).all(), "count forecasts must be non-negative"
    # every future week and every series present exactly once
    assert set(out["ds"]) == set(pd.to_datetime(future))
    assert out.groupby(SERIES_ID).size().eq(13).all()


def test_patchtst_poisson_contract():
    """The Poisson likelihood (cycle 6-B) must also satisfy the contract:
    full coverage, non-negative, no NaN — the failure mode that killed NB."""
    pytest.importorskip("neuralforecast")
    import torch
    torch.set_num_threads(1)
    train = _panel()
    model = build_model({"name": "patchtst", "params": {
        "accelerator": "cpu", "distribution": "Poisson",
        "model_params": {"input_size": 32, "patch_len": 8, "stride": 8,
                         "max_steps": 40, "hidden_size": 32, "n_heads": 4,
                         "batch_size": 64, "windows_batch_size": 128,
                         "random_seed": 7}}})
    out = model.fit_predict(train, _future(train, 13))
    assert len(out) == train[SERIES_ID].nunique() * 13
    assert out["yhat"].notna().all() and (out["yhat"] >= 0).all()


def test_patchtst_point_column_is_mean_not_median():
    """Guards the cycle-4 lesson: the base column must be the distribution MEAN,
    a distinct value from the median, so we never silently forecast the median."""
    pytest.importorskip("neuralforecast")
    import torch
    torch.set_num_threads(1)
    from neuralforecast import NeuralForecast
    from neuralforecast.losses.pytorch import DistributionLoss
    from neuralforecast.models import PatchTST

    train = _panel()
    df = train[[SERIES_ID, "ds", "y"]].rename(columns={SERIES_ID: "unique_id"})
    m = PatchTST(h=13, input_size=32, patch_len=8, stride=8, max_steps=40,
                 hidden_size=32, n_heads=4, batch_size=64, windows_batch_size=128,
                 loss=DistributionLoss("NegativeBinomial"), scaler_type="robust",
                 start_padding_enabled=True, random_seed=7,
                 enable_progress_bar=False, logger=False, accelerator="cpu")
    nf = NeuralForecast(models=[m], freq="W-MON")
    nf.fit(df)
    fc = nf.predict()
    assert "PatchTST" in fc.columns and "PatchTST-median" in fc.columns
    # on right-skewed count data the mean sits at or above the median overall
    assert fc["PatchTST"].mean() >= fc["PatchTST-median"].mean()
    assert not np.allclose(fc["PatchTST"], fc["PatchTST-median"])
