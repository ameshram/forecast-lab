import numpy as np
import pandas as pd

from src.metrics import bias_pct, vn1_score, wape


def test_perfect_forecast_scores_zero():
    y = np.array([1.0, 2.0, 0.0, 5.0])
    assert vn1_score(y, y) == 0.0
    assert wape(y, y) == 0.0
    assert bias_pct(y, y) == 0.0


def test_vn1_adds_abs_error_and_bias():
    y = np.array([10.0, 10.0])
    yhat = np.array([12.0, 12.0])  # abs err 4, bias 4, denom 20
    assert np.isclose(vn1_score(y, yhat), 8 / 20)


def test_vn1_offsetting_errors_still_penalized_via_wape_term():
    y = np.array([10.0, 10.0])
    yhat = np.array([12.0, 8.0])  # abs err 4, bias 0
    assert np.isclose(vn1_score(y, yhat), 4 / 20)
    assert np.isclose(bias_pct(y, yhat), 0.0)


def test_bias_sign_convention():
    y = np.array([10.0])
    assert bias_pct(y, np.array([15.0])) > 0  # over-forecast positive


def test_zero_denominator_is_nan():
    assert np.isnan(vn1_score(np.zeros(3), np.ones(3)))


def test_backtest_leakage_guard():
    """Model contract: forecasts must cover exactly the training universe."""
    from src.data import SERIES_ID
    from src.models.baselines import MovingAverage

    ds = pd.date_range("2022-01-03", periods=20, freq="W-MON")
    train = pd.DataFrame({
        SERIES_ID: ["a"] * 20 + ["b"] * 20,
        "ds": list(ds) * 2,
        "y": [1.0] * 20 + [0.0] * 20,
    })
    future = [ds[-1] + pd.Timedelta(weeks=i + 1) for i in range(3)]
    out = MovingAverage(window=4).fit_predict(train, future)
    assert set(out[SERIES_ID]) == {"a", "b"}
    assert len(out) == 6
    assert np.isclose(out.loc[out[SERIES_ID] == "a", "yhat"], 1.0).all()
    assert np.isclose(out.loc[out[SERIES_ID] == "b", "yhat"], 0.0).all()


def test_dead_zeroing():
    from src.data import SERIES_ID
    from src.models.baselines import MovingAverage

    ds = pd.date_range("2022-01-03", periods=30, freq="W-MON")
    y_a = [5.0] * 10 + [0.0] * 20  # dead for the last 20 weeks
    train = pd.DataFrame({SERIES_ID: ["a"] * 30, "ds": ds, "y": y_a})
    future = [ds[-1] + pd.Timedelta(weeks=1)]
    out = MovingAverage(window=30, zero_dead_weeks=15).fit_predict(train, future)
    assert out["yhat"].iloc[0] == 0.0
