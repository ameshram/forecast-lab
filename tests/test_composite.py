import numpy as np
import pandas as pd

from src.data import SERIES_ID
from src.models import build_model


def _panel():
    """Two series, one client: 'a' smooth (constant 6), 'b' intermittent."""
    ds = pd.date_range("2022-01-03", periods=60, freq="W-MON")
    b_vals = [0.0] * 60
    for i in range(0, 60, 6):
        b_vals[i] = 12.0
    return pd.DataFrame({
        SERIES_ID: ["a"] * 60 + ["b"] * 60,
        "Client": [0] * 120,
        "ds": list(ds) * 2,
        "y": [6.0] * 60 + b_vals,
    })


def _future(train, n=3):
    last = train.ds.max()
    return [last + pd.Timedelta(weeks=i + 1) for i in range(n)]


def test_segment_router_routes_smooth_only():
    train = _panel()
    model = build_model({"name": "segment_router", "params": {
        "default": {"name": "zero"},
        "routes": {"smooth": {"name": "moving_average",
                              "params": {"window": 4}}},
    }})
    out = model.fit_predict(train, _future(train))
    a = out[out[SERIES_ID] == "a"]["yhat"]
    b = out[out[SERIES_ID] == "b"]["yhat"]
    assert np.allclose(a, 6.0)   # smooth routed to MA4
    assert np.allclose(b, 0.0)   # intermittent stays on default (zero)


def test_blend_weighted_average():
    train = _panel()
    model = build_model({"name": "blend", "params": {
        "models": [{"name": "naive_last"}, {"name": "zero"}],
        "weights": [1, 1],
    }})
    out = model.fit_predict(train, _future(train))
    a = out[out[SERIES_ID] == "a"]["yhat"]
    assert np.allclose(a, 3.0)   # (6 + 0) / 2


def test_seasonal_uplift_flat_series_stays_flat():
    train = _panel()
    model = build_model({"name": "seasonal_uplift_naive",
                         "params": {"alpha": 1.0}})
    out = model.fit_predict(train, _future(train))
    a = out[out[SERIES_ID] == "a"]["yhat"]
    # totals vary only via series b's spikes; uplift must stay clipped/sane
    assert (a >= 3.0).all() and (a <= 12.0).all()


def test_seasonal_uplift_scales_toward_high_weeks():
    ds = pd.date_range("2021-01-04", periods=156, freq="W-MON")  # 3 years
    woy = ds.isocalendar().week.astype(int)
    y = np.where((woy >= 47) & (woy <= 51), 30.0, 10.0)  # holiday uplift x3
    train = pd.DataFrame({SERIES_ID: ["a"] * 156, "Client": [0] * 156,
                          "ds": ds, "y": y})
    train = train[train.ds <= "2023-10-30"]  # anchor in a normal week
    model = build_model({"name": "seasonal_uplift_naive",
                         "params": {"alpha": 1.0}})
    future = [train.ds.max() + pd.Timedelta(weeks=i + 1) for i in range(6)]
    out = model.fit_predict(train, future).set_index("ds")
    out_woy = out.index.isocalendar().week.astype(int)
    high = out.loc[(out_woy >= 47) & (out_woy <= 51), "yhat"]
    low = out.loc[out_woy < 47, "yhat"]
    assert (high > low.max()).all()   # holiday weeks forecast above normal
    assert (high <= 20.0).all()       # clip_high=2.0 bounds the uplift
