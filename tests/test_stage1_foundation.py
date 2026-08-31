"""Contract tests for the Stage 1 zero-shot foundation-model wrappers.

The chronos contract test uses the tiny checkpoint (~9M params) to keep the
download small; the timesfm test uses the same 200M checkpoint as the real
runs (cached under ~/.cache/huggingface after first use). Both are skipped
if the DL stack is not installed.

Torch runs single-threaded here: the LightGBM tests load Homebrew's libomp
and torch bundles its own — two OpenMP runtimes in one process deadlock in
the join barrier on macOS. Real experiment runs never import both.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import SERIES_ID
from src.models import build_model
from src.models.chronos import history_matrix


def _panel(n_series=12, n_weeks=60, seed=7):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")
    rows = []
    for i in range(n_series):
        y = rng.poisson(rng.uniform(0.2, 6.0), n_weeks).astype(float)
        rows.append(pd.DataFrame({
            SERIES_ID: f"0_{i % 3}_{i}", "Client": 0, "Warehouse": i % 3,
            "Product": i, "ds": ds, "y": y}))
    return pd.concat(rows, ignore_index=True)


def _future(train, n=4):
    last = train.ds.max()
    return [last + pd.Timedelta(weeks=i + 1) for i in range(n)]


def test_history_matrix_shape_order_and_trim():
    train = _panel(n_series=5, n_weeks=20)
    wide = history_matrix(train)
    assert wide.shape == (5, 20)
    assert list(wide.columns) == sorted(wide.columns)
    trimmed = history_matrix(train, context_weeks=8)
    assert trimmed.shape == (5, 8)
    assert list(trimmed.columns) == list(wide.columns[-8:])


def _assert_contract(out, train, horizon):
    assert len(out) == train[SERIES_ID].nunique() * horizon
    assert out["yhat"].notna().all() and (out["yhat"] >= 0).all()
    assert set(out.columns) == {SERIES_ID, "ds", "yhat"}
    # portfolio sanity: weekly forecast total within 5x of recent actual level
    recent = train[train.ds >= train.ds.max() - pd.Timedelta(weeks=7)]
    weekly = recent.groupby("ds").y.sum().mean()
    fweekly = out.groupby("ds").yhat.sum().mean()
    assert 0.2 * weekly < fweekly < 5 * weekly


def _serial_torch():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)


def test_chronos_contract_and_sanity():
    pytest.importorskip("chronos")
    _serial_torch()
    train = _panel()
    model = build_model({"name": "chronos", "params": {
        "checkpoint": "amazon/chronos-bolt-tiny", "batch_size": 8}})
    out = model.fit_predict(train, _future(train, 4))
    _assert_contract(out, train, 4)


def test_chronos_qmean_readout_differs_from_median():
    """qmean averages the nine trained quantile heads; on skewed count data
    it must sit above the median overall, and still satisfy the contract."""
    pytest.importorskip("chronos")
    _serial_torch()
    train = _panel()
    params = {"checkpoint": "amazon/chronos-bolt-tiny", "batch_size": 8}
    median = build_model({"name": "chronos", "params": params})
    qmean = build_model({"name": "chronos",
                         "params": {**params, "point": "qmean"}})
    future = _future(train, 4)
    out_m = median.fit_predict(train, future)
    out_q = qmean.fit_predict(train, future)
    _assert_contract(out_q, train, 4)
    assert not np.allclose(out_m["yhat"], out_q["yhat"])
    assert out_q["yhat"].sum() > out_m["yhat"].sum()


def test_timesfm_contract_and_sanity():
    pytest.importorskip("timesfm")
    _serial_torch()
    train = _panel()
    model = build_model({"name": "timesfm", "params": {
        "batch_size": 8, "max_context": 128}})
    out = model.fit_predict(train, _future(train, 4))
    _assert_contract(out, train, 4)
