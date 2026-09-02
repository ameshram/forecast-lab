"""TimesFM-3 branch of the timesfm wrapper (Cycle 13-b zoo addendum).

Contract + readout-distinctness on a tiny panel. Guarded on the v3 weights
being cached so CI never forces the ~1.3GB download. The full readout
semantics verdict (is the native point head the median?) comes from the
real-data semantics check recorded in the run notes, not from this panel.
"""
import pathlib

import numpy as np
import pandas as pd
import pytest

from src.data import SERIES_ID
from src.models import build_model


def _cached():
    return (pathlib.Path.home() / ".cache/huggingface/hub"
            / "models--google--timesfm-3.0-pytorch").exists()


pytestmark = pytest.mark.skipif(
    not _cached(), reason="google/timesfm-3.0-pytorch weights not cached")


def _panel(n_series=6, n_weeks=120, seed=9):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")
    rows = []
    for i in range(n_series):
        if i == 0:
            y = np.zeros(n_weeks)
        elif i % 2:
            y = rng.poisson(1.0, n_weeks).astype(float)
            y[rng.random(n_weeks) < 0.6] = 0.0
        else:
            y = rng.poisson(7.0, n_weeks).astype(float)
        rows.append(pd.DataFrame({
            SERIES_ID: f"0_{i % 2}_{i}", "Client": 0, "Warehouse": i % 2,
            "Product": i, "ds": ds, "y": y}))
    return pd.concat(rows, ignore_index=True)


def _future(train, n=4):
    last = train.ds.max()
    return [last + pd.Timedelta(weeks=i + 1) for i in range(n)]


def test_v3_contract_and_readout_distinctness():
    pytest.importorskip("timesfm")
    import torch
    torch.set_num_threads(1)
    train = _panel()
    future = _future(train, 4)

    def run(point):
        m = build_model({"name": "timesfm", "params": {
            "checkpoint": "google/timesfm-3.0-pytorch", "batch_size": 8,
            "point": point}})
        return m.fit_predict(train, future)

    native = run("native")
    med = run("quantile")
    qm = run("qmean")
    for out in (native, med, qm):
        assert set(out.columns) == {SERIES_ID, "ds", "yhat"}
        assert len(out) == train[SERIES_ID].nunique() * 4
        assert out["yhat"].notna().all() and (out["yhat"] >= 0).all()
        assert not out.duplicated([SERIES_ID, "ds"]).any()
    b = med.sort_values([SERIES_ID, "ds"])["yhat"].to_numpy()
    c = qm.sort_values([SERIES_ID, "ds"])["yhat"].to_numpy()
    assert not np.allclose(c, b), "qmean must differ from the median"


def test_v3_rejects_bad_readout():
    with pytest.raises(ValueError):
        build_model({"name": "timesfm", "params": {"point": "mean"}})


def test_v2_rejects_v3_readouts():
    pytest.importorskip("timesfm")
    train = _panel(n_series=2, n_weeks=60)
    m = build_model({"name": "timesfm", "params": {
        "checkpoint": "google/timesfm-2.5-200m-pytorch", "point": "qmean"}})
    with pytest.raises(ValueError):
        m.fit_predict(train, _future(train, 2))
