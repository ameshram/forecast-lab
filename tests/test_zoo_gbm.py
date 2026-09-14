"""Contract tests for the Tier-1 GBM twins (Cycle 10 zoo): XGBoost + CatBoost.

Both reuse the exact GlobalLGBM feature pipeline (src/features.py) with a
different estimator and a matching Tweedie objective. Skipped if the estimator
is not installed. The GlobalLGBM class itself is untouched (run_id lineage).
"""
import numpy as np
import pandas as pd
import pytest

from src.data import SERIES_ID
from src.models import build_model


def _panel(n_series=40, n_weeks=90, seed=3):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")
    rows = []
    for i in range(n_series):
        lam = rng.uniform(0.2, 6.0)
        y = rng.poisson(lam, n_weeks).astype(float)
        price = np.where(y > 0, rng.uniform(5, 50), np.nan)
        rows.append(pd.DataFrame({
            SERIES_ID: f"0_{i % 4}_{i}", "Client": 0, "Warehouse": i % 4,
            "Product": i, "ds": ds, "y": y, "Price": price}))
    return pd.concat(rows, ignore_index=True)


def _future(train, n=4):
    last = train.ds.max()
    return [last + pd.Timedelta(weeks=i + 1) for i in range(n)]


def _assert_contract(out, train, horizon):
    assert set(out.columns) == {SERIES_ID, "ds", "yhat"}
    assert len(out) == train[SERIES_ID].nunique() * horizon
    assert out["yhat"].notna().all() and (out["yhat"] >= 0).all()
    assert not out.duplicated([SERIES_ID, "ds"]).any()
    recent = train[train.ds >= train.ds.max() - pd.Timedelta(weeks=7)]
    weekly = recent.groupby("ds").y.sum().mean()
    fweekly = out.groupby("ds").yhat.sum().mean()
    assert 0.33 * weekly < fweekly < 3 * weekly


def test_global_xgb_contract_and_sanity():
    pytest.importorskip("xgboost")
    train = _panel()
    model = build_model({"name": "global_xgb", "params": {
        "n_anchors": 8,
        "xgb_params": {"n_estimators": 40, "max_leaves": 15}}})
    out = model.fit_predict(train, _future(train, 4))
    _assert_contract(out, train, 4)


def test_global_cat_contract_and_sanity():
    pytest.importorskip("catboost")
    train = _panel()
    model = build_model({"name": "global_cat", "params": {
        "n_anchors": 8,
        "cat_params": {"iterations": 40, "depth": 4}}})
    out = model.fit_predict(train, _future(train, 4))
    _assert_contract(out, train, 4)
