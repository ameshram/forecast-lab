"""Contract tests for the Tier-0 statsforecast wrapper (Cycle 10 zoo).

Covers the intermittent-demand family (Croston/SBA/TSB) plus classical
AutoETS/AutoTheta. The panel deliberately includes an ALL-ZERO series (the
Croston degenerate case) alongside intermittent ones, so the wrapper's
full-coverage fallback is exercised. Skipped if statsforecast is absent.
"""
import numpy as np
import pandas as pd
import pytest

from src.data import SERIES_ID
from src.models import build_model


def _panel(n_series=8, n_weeks=60, seed=7):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")
    rows = []
    for i in range(n_series):
        if i == 0:
            y = np.zeros(n_weeks)                       # all-zero (dead) series
        else:
            y = rng.poisson(rng.uniform(0.2, 5.0), n_weeks).astype(float)
            y[rng.random(n_weeks) < 0.5] = 0.0          # intermittency
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
    # no duplicate (series, ds) — would fan out the scoring join
    assert not out.duplicated([SERIES_ID, "ds"]).any()
    assert set(out["ds"]) == set(pd.to_datetime(_future(train, horizon)))
    # portfolio sanity: weekly forecast total within 5x of recent actual level
    recent = train[train.ds >= train.ds.max() - pd.Timedelta(weeks=7)]
    weekly = recent.groupby("ds").y.sum().mean()
    fweekly = out.groupby("ds").yhat.sum().mean()
    assert 0.2 * weekly < fweekly < 5 * weekly


CASES = [
    ("CrostonClassic", {}),
    ("CrostonSBA", {}),
    ("TSB", {"alpha_d": 0.2, "alpha_p": 0.2}),
    ("AutoETS", {"season_length": 1}),
    ("AutoTheta", {"season_length": 1}),
]


@pytest.mark.parametrize("model,params", CASES)
def test_statsforecast_contract(model, params):
    pytest.importorskip("statsforecast")
    train = _panel()
    m = build_model({"name": "statsforecast",
                     "params": {"model": model, "params": params}})
    out = m.fit_predict(train, _future(train, 4))
    _assert_contract(out, train, 4)


def test_dead_series_gets_full_coverage_zero():
    """The all-zero series must still receive h finite, non-negative rows."""
    pytest.importorskip("statsforecast")
    train = _panel()
    m = build_model({"name": "statsforecast",
                     "params": {"model": "CrostonClassic"}})
    out = m.fit_predict(train, _future(train, 4))
    dead = out[out[SERIES_ID] == "0_0_0"]
    assert len(dead) == 4
    assert dead["yhat"].notna().all() and (dead["yhat"] >= 0).all()


def test_unknown_model_rejected():
    with pytest.raises(ValueError):
        build_model({"name": "statsforecast", "params": {"model": "NopeModel"}})
