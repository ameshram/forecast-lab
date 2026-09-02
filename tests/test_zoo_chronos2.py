"""Chronos-2 branch of the chronos wrapper (Cycle 10 zoo).

Pins the empirically-verified point-readout semantics: like Chronos-Bolt,
Chronos-2's predict_quantiles returns the MEDIAN in the 'mean' slot, so
point='mean' is a no-op duplicate of the median and the real runs must use
point='qmean' (the trimmed mean over the nine quantile heads). Guarded on the
weights being cached (no tiny Chronos-2 checkpoint exists) so CI never forces
the ~large download.
"""
import pathlib

import numpy as np
import pandas as pd
import pytest

from src.data import SERIES_ID
from src.models import build_model


def _chronos2_cached():
    p = (pathlib.Path.home() / ".cache/huggingface/hub"
         / "models--amazon--chronos-2")
    return p.exists()


pytestmark = pytest.mark.skipif(
    not _chronos2_cached(), reason="amazon/chronos-2 weights not cached")


def _panel(n_series=8, n_weeks=60, seed=7):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")
    rows = []
    for i in range(n_series):
        y = rng.poisson(rng.uniform(0.3, 6.0), n_weeks).astype(float)
        y[rng.random(n_weeks) < 0.4] = 0.0
        rows.append(pd.DataFrame({
            SERIES_ID: f"0_{i % 3}_{i}", "Client": 0, "Warehouse": i % 3,
            "Product": i, "ds": ds, "y": y}))
    return pd.concat(rows, ignore_index=True)


def _future(train, n=4):
    last = train.ds.max()
    return [last + pd.Timedelta(weeks=i + 1) for i in range(n)]


def test_chronos2_contract_and_readout_semantics():
    pytest.importorskip("chronos")
    import torch
    torch.set_num_threads(1)
    train = _panel()
    future = _future(train, 4)

    def run(point):
        m = build_model({"name": "chronos", "params": {
            "checkpoint": "amazon/chronos-2", "point": point, "batch_size": 8}})
        return m.fit_predict(train, future)

    out_mean = run("mean")
    out_med = run("quantile")
    out_qmean = run("qmean")

    # contract on each
    for out in (out_mean, out_med, out_qmean):
        assert set(out.columns) == {SERIES_ID, "ds", "yhat"}
        assert len(out) == train[SERIES_ID].nunique() * 4
        assert out["yhat"].notna().all() and (out["yhat"] >= 0).all()
        assert not out.duplicated([SERIES_ID, "ds"]).any()

    a = out_mean.sort_values([SERIES_ID, "ds"])["yhat"].to_numpy()
    b = out_med.sort_values([SERIES_ID, "ds"])["yhat"].to_numpy()
    c = out_qmean.sort_values([SERIES_ID, "ds"])["yhat"].to_numpy()
    # the Bolt/Chronos-2 trap: 'mean' == median; qmean must differ
    assert np.allclose(a, b), "expected chronos-2 'mean' to equal the median"
    assert not np.allclose(c, b), "qmean must differ from the median"
