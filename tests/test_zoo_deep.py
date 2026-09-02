"""Contract tests for the Tier-2 deep-net zoo additions (Cycle 10): DeepAR, TFT.

Tiny/fast configs just to prove the wrapper -> neuralforecast -> contract path
(coverage, non-negativity, no dup rows, distribution-mean point column). Real
runs use the full budget and are gated by a separate bias-ratio smoke. Skipped
if neuralforecast is absent. Torch single-threaded (libomp/torch deadlock).
"""
import numpy as np
import pandas as pd
import pytest

from src.data import SERIES_ID
from src.models import build_model


def _panel(n_series=9, n_weeks=90, seed=5):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")
    rows = []
    for i in range(n_series):
        if i == 0:
            y = np.zeros(n_weeks)                         # all-zero (scaler stress)
        elif i % 2:
            y = rng.poisson(rng.uniform(0.2, 3.0), n_weeks).astype(float)
            y[rng.random(n_weeks) < 0.5] = 0.0           # intermittent
        else:
            y = rng.poisson(rng.uniform(3.0, 12.0), n_weeks).astype(float)  # skewed
        rows.append(pd.DataFrame({
            SERIES_ID: f"0_{i % 3}_{i}", "Client": 0, "Warehouse": i % 3,
            "Product": i, "ds": ds, "y": y}))
    return pd.concat(rows, ignore_index=True)


def _future(train, n=4):
    last = train.ds.max()
    return [last + pd.Timedelta(weeks=i + 1) for i in range(n)]


def _assert_contract(out, train, horizon):
    assert set(out.columns) == {SERIES_ID, "ds", "yhat"}
    assert len(out) == train[SERIES_ID].nunique() * horizon
    assert out["yhat"].notna().all() and (out["yhat"] >= 0).all()
    assert not out.duplicated([SERIES_ID, "ds"]).any()
    assert set(out["ds"]) == set(pd.to_datetime(_future(train, horizon)))


def _serial_torch():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)


FAST = {"input_size": 16, "max_steps": 20}


def test_deepar_contract():
    pytest.importorskip("neuralforecast")
    _serial_torch()
    train = _panel()
    model = build_model({"name": "deepar", "params": {
        "num_samples": 100, "model_params": FAST}})
    out = model.fit_predict(train, _future(train, 4))
    _assert_contract(out, train, 4)


def test_tft_contract():
    pytest.importorskip("neuralforecast")
    _serial_torch()
    train = _panel()
    model = build_model({"name": "tft", "params": {
        "num_samples": 100, "model_params": {**FAST, "windows_batch_size": 64}}})
    out = model.fit_predict(train, _future(train, 4))
    _assert_contract(out, train, 4)
