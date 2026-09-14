"""Contract + covariate leak-safety tests for the Cycle 7 N-HiTS wrapper.

Skipped if neuralforecast is absent; torch forced single-threaded (two
OpenMP runtimes deadlock on macOS). The leakage test is the important one:
covariates are built inside the wrapper, so the frozen harness cannot catch
a leak — these tests must.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import SERIES_ID
from src.models import build_model
from src.models.deep import (_add_futr_covars, _add_hist_covars,
                             _first_sale_ds)


def _panel(n_series=18, n_weeks=100, seed=4):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")
    rows = []
    for i in range(n_series):
        y = rng.poisson(0.4 if i % 2 else 4.0, n_weeks).astype(float)
        price = np.where(y > 0, rng.uniform(5, 40), np.nan)
        rows.append(pd.DataFrame({SERIES_ID: f"0_{i % 3}_{i}", "Client": 0,
                                  "Warehouse": i % 3, "Product": i, "ds": ds,
                                  "y": y, "Price": price}))
    return pd.concat(rows, ignore_index=True)


def _future(train, n=13):
    last = train.ds.max()
    return [last + pd.Timedelta(weeks=i + 1) for i in range(n)]


def _model(cov):
    pytest.importorskip("neuralforecast")
    import torch
    torch.set_num_threads(1)
    p = {"accelerator": "cpu", "distribution": "Poisson",
         "model_params": {"input_size": 32, "max_steps": 30, "batch_size": 64,
                          "windows_batch_size": 128, "random_seed": 7}}
    if cov:
        p |= {"futr_exog": ["woy", "month", "age"],
              "hist_exog": ["price", "price_chg4", "wsls"],
              "stat_exog": ["client", "warehouse"]}
    return build_model({"name": "nhits", "params": p})


def _assert_contract(out, train):
    assert len(out) == train[SERIES_ID].nunique() * 13
    assert out["yhat"].notna().all() and (out["yhat"] >= 0).all()
    assert set(out.columns) == {SERIES_ID, "ds", "yhat"}


def test_nhits_nocov_contract():
    train = _panel()
    _assert_contract(_model(False).fit_predict(train, _future(train)), train)


def test_nhits_cov_contract():
    train = _panel()
    _assert_contract(_model(True).fit_predict(train, _future(train)), train)


def test_wsls_is_past_only_no_leak():
    """wsls at week t must depend only on y at weeks <= t: poisoning a LATER
    week must not change wsls at earlier weeks."""
    ds = pd.date_range("2021-01-04", periods=20, freq="W-MON")
    y = [0, 3, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 1.0]
    base = pd.DataFrame({SERIES_ID: "a", "ds": ds, "y": y, "Price": np.nan})
    pois = base.copy(); pois.loc[pois.ds == ds[-1], "y"] = 999.0  # poison last week
    w1 = _add_hist_covars(base.rename(columns={}), ["wsls"])["wsls"].to_numpy()
    w2 = _add_hist_covars(pois.rename(columns={}), ["wsls"])["wsls"].to_numpy()
    assert np.array_equal(w1[:-1], w2[:-1])          # earlier weeks unchanged
    # wsls resets to 0 on a sale week and increments on zeros
    assert w1[0] == -1 and w1[1] == 0 and w1[2] == 1 and w1[3] == 2 and w1[4] == 0


def test_age_uses_first_sale_and_is_deterministic_future():
    ds = pd.date_range("2021-01-04", periods=12, freq="W-MON")
    y = [0, 0, 5, 0, 1, 0, 0, 3, 0, 0, 2, 0.0]     # first sale at index 2
    train = pd.DataFrame({SERIES_ID: "a", "ds": ds, "y": y})
    fs = _first_sale_ds(train)
    assert fs["a"] == ds[2]
    # future weeks: age keeps incrementing deterministically from first sale
    fut = pd.DataFrame({SERIES_ID: "a", "ds": [ds[-1] + pd.Timedelta(weeks=k)
                                               for k in (1, 2)]})
    fut = _add_futr_covars(fut, ["age"], fs)
    assert fut["age"].tolist() == [10.0, 11.0]      # weeks since first-sale week
