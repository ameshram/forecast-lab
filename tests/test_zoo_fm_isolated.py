"""Contract tests for the isolated-venv foundation models (Cycle 10 zoo):
Moirai 2.0 (uni2ts) and IBM TTM (granite-tsfm).

These libraries are RED installs against the main venv, so both tests
importorskip and effectively run only from the throwaway isolated venvs
(scratchpad venv_moirai / venv_ttm). In the main venv the whole module
skips. Panels include an all-zero series (scaler stress) and intermittent
series; torch single-threaded (libomp deadlock).
"""
import numpy as np
import pandas as pd
import pytest

from src.data import SERIES_ID
from src.models import build_model


def _panel(n_series=8, n_weeks=120, seed=11):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")
    rows = []
    for i in range(n_series):
        if i == 0:
            y = np.zeros(n_weeks)                        # all-zero
        elif i % 2:
            y = rng.poisson(rng.uniform(0.3, 2.0), n_weeks).astype(float)
            y[rng.random(n_weeks) < 0.6] = 0.0           # intermittent/lumpy
        else:
            y = rng.poisson(rng.uniform(4.0, 10.0), n_weeks).astype(float)
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


def test_moirai_contract_and_readouts():
    pytest.importorskip("uni2ts")
    _serial_torch()
    train = _panel()
    future = _future(train, 4)
    med = build_model({"name": "moirai", "params": {
        "batch_size": 8}}).fit_predict(train, future)
    _assert_contract(med, train, 4)
    qm = build_model({"name": "moirai", "params": {
        "batch_size": 8, "point": "qmean"}}).fit_predict(train, future)
    _assert_contract(qm, train, 4)
    a = med.sort_values([SERIES_ID, "ds"])["yhat"].to_numpy()
    b = qm.sort_values([SERIES_ID, "ds"])["yhat"].to_numpy()
    assert not np.allclose(a, b), "qmean must differ from the median"


def test_moirai_rejects_fake_mean_readout():
    pytest.importorskip("uni2ts")
    with pytest.raises(ValueError):
        build_model({"name": "moirai", "params": {"point": "mean"}})


def test_ttm_contract():
    pytest.importorskip("tsfm_public")
    _serial_torch()
    train = _panel()
    out = build_model({"name": "ttm", "params": {
        "batch_size": 8}}).fit_predict(train, _future(train, 4))
    _assert_contract(out, train, 4)


def test_ttm_short_history_raises():
    pytest.importorskip("tsfm_public")
    train = _panel(n_weeks=60)   # < 90-week TTM context
    with pytest.raises(ValueError):
        build_model({"name": "ttm"}).fit_predict(train, _future(train, 4))
