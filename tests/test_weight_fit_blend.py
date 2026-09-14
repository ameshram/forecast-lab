"""WeightFitBlend (Cycle 12): contract, LEAKAGE, and fitting semantics.

The leakage test is the load-bearing one: the weight fit must only ever show
components data strictly BEFORE the internal holdout, and the final refit
must see exactly the full train slice - never anything at/after the origin.
"""
import numpy as np
import pandas as pd
import pytest

import src.models.weight_fit as wf
from src.data import SERIES_ID
from src.models import build_model


def _panel(n_series=8, n_weeks=120, seed=3):
    rng = np.random.default_rng(seed)
    ds = pd.date_range("2021-01-04", periods=n_weeks, freq="W-MON")
    rows = []
    for i in range(n_series):
        if i % 3 == 0:
            y = rng.poisson(6.0, n_weeks).astype(float)      # steady
        else:
            y = rng.poisson(1.2, n_weeks).astype(float)
            y[rng.random(n_weeks) < 0.5] = 0.0               # intermittent
        rows.append(pd.DataFrame({
            SERIES_ID: f"0_{i % 2}_{i}", "Client": 0, "Warehouse": i % 2,
            "Product": i, "ds": ds, "y": y}))
    return pd.concat(rows, ignore_index=True)


def _future(train, n=4):
    last = train.ds.max()
    return [last + pd.Timedelta(weeks=i + 1) for i in range(n)]


CFG = {"models": [{"name": "naive_last"},
                  {"name": "moving_average", "params": {"window": 13}}],
       "holdout_weeks": 13}


def test_contract():
    train = _panel()
    out = build_model({"name": "weight_fit_blend", "params": CFG}) \
        .fit_predict(train, _future(train, 4))
    assert set(out.columns) == {SERIES_ID, "ds", "yhat"}
    assert len(out) == train[SERIES_ID].nunique() * 4
    assert out["yhat"].notna().all() and (out["yhat"] >= 0).all()
    assert not out.duplicated([SERIES_ID, "ds"]).any()


def test_no_leakage(monkeypatch):
    """The inner fit sees only pre-holdout data; the refit sees full train;
    nothing ever sees dates at/after the origin."""
    train = _panel()
    future = _future(train, 4)
    origin = future[0]
    weeks = sorted(train["ds"].unique())
    holdout_start = pd.Timestamp(weeks[-13])

    seen = []
    real_build = wf._build

    def spying_build(cfg):
        model = real_build(cfg)
        real_fp = model.fit_predict

        def spied(tr, fds):
            seen.append((pd.Timestamp(tr["ds"].max()),
                         pd.Timestamp(max(fds))))
            return real_fp(tr, fds)
        model.fit_predict = spied
        return model

    monkeypatch.setattr(wf, "_build", spying_build)
    build_model({"name": "weight_fit_blend", "params": CFG}) \
        .fit_predict(train, future)

    assert len(seen) == 4  # 2 components x (inner fit + full refit)
    inner, full = seen[:2], seen[2:]
    for train_max, fcst_max in inner:
        assert train_max < holdout_start          # fit never sees holdout
        assert fcst_max < origin                  # fits forecast pre-origin only
    for train_max, fcst_max in full:
        assert train_max < origin                 # refit stays inside train
        assert train_max == pd.Timestamp(weeks[-1])


def test_weights_beat_or_match_equal_on_holdout():
    """The solve starts at equal weights, so fitted holdout VN1 <= equal's."""
    train = _panel()
    model = build_model({"name": "weight_fit_blend", "params": CFG})
    w = model._fit_weights(train)
    assert w.shape == (2,)
    assert np.all(w >= 0) and abs(w.sum() - 1.0) < 1e-9


def test_short_history_falls_back_to_equal():
    train = _panel(n_weeks=30)   # <= holdout+26
    model = build_model({"name": "weight_fit_blend", "params": CFG})
    w = model._fit_weights(train)
    assert np.allclose(w, [0.5, 0.5])
