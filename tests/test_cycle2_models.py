import numpy as np
import pandas as pd

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
            SERIES_ID: f"0_{i%4}_{i}", "Client": 0, "Warehouse": i % 4,
            "Product": i, "ds": ds, "y": y, "Price": price}))
    return pd.concat(rows, ignore_index=True)


def _future(train, n=13):
    last = train.ds.max()
    return [last + pd.Timedelta(weeks=i + 1) for i in range(n)]


def test_bias_scaled_corrects_downward_shift():
    """Series drop from 10 to 5 in the last 13 weeks: inner naive forecasts
    10 vs actual 5 -> ratio 0.5, damped (alpha .5) to 0.75, clipped to 0.85."""
    ds = pd.date_range("2021-01-04", periods=60, freq="W-MON")
    y = [10.0] * 47 + [5.0] * 13
    train = pd.DataFrame({SERIES_ID: ["a"] * 60, "Client": 0, "ds": ds, "y": y})
    model = build_model({"name": "bias_scaled", "params": {
        "base": {"name": "naive_last"}, "holdout_weeks": 13,
        "alpha": 0.5, "clip_low": 0.85, "clip_high": 1.2}})
    out = model.fit_predict(train, _future(train, 2))
    assert np.allclose(out["yhat"], 5.0 * 0.85)


def test_bias_scaled_neutral_when_unbiased():
    ds = pd.date_range("2021-01-04", periods=60, freq="W-MON")
    train = pd.DataFrame({SERIES_ID: ["a"] * 60, "Client": 0, "ds": ds,
                          "y": [7.0] * 60})
    model = build_model({"name": "bias_scaled",
                         "params": {"base": {"name": "naive_last"}}})
    out = model.fit_predict(train, _future(train, 2))
    assert np.allclose(out["yhat"], 7.0)


def test_global_lgbm_contract_and_sanity():
    train = _panel()
    model = build_model({"name": "global_lgbm", "params": {
        "n_anchors": 8,
        "lgb_params": {"n_estimators": 30, "num_leaves": 15,
                       "min_child_samples": 5}}})
    future = _future(train, 4)
    out = model.fit_predict(train, future)
    assert len(out) == train[SERIES_ID].nunique() * 4      # full coverage
    assert out["yhat"].notna().all() and (out["yhat"] >= 0).all()
    # global sanity: portfolio-level forecast within 3x of recent level
    recent = train[train.ds >= train.ds.max() - pd.Timedelta(weeks=7)]
    weekly = recent.groupby("ds").y.sum().mean()
    fweekly = out.groupby("ds").yhat.sum().mean()
    assert 0.33 * weekly < fweekly < 3 * weekly


def test_feature_builder_uses_only_past_columns():
    """Poisoning the last training week must not change features anchored
    at earlier weeks (anti-leakage check on the training matrix)."""
    from src.features import build_wide, training_matrix
    train = _panel(n_series=10, n_weeks=40)
    poisoned = train.copy()
    poisoned.loc[poisoned.ds == poisoned.ds.max(), "y"] = 9999.0

    def matrix(t):
        wy, wp, c, w = build_wide(t)
        return training_matrix(wy, wp, c, w, n_anchors=5, horizons=2)

    X1, y1 = matrix(train)
    X2, y2 = matrix(poisoned)
    # targets may differ (they include the last week) but feature rows for
    # anchors strictly before the poisoned week must be identical
    n = 10  # series count -> first anchor block rows
    assert np.allclose(X1[:n], X2[:n], equal_nan=True)
