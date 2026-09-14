"""Tests for the run ledger (src/ledger.py).

Focus: record_run must emit series_scores.parquet in exactly the schema the
promotion gate (src/gates.py) consumes, with values that reproduce the pooled
VN1. This is the contract that was previously missing -- the gate read a file
the writer never produced.
"""
import numpy as np
import pandas as pd
import pytest

from src import gates, ledger, metrics
from src.data import SERIES_ID


@pytest.fixture
def exp(tmp_path, monkeypatch):
    e = tmp_path / "experiments"
    (e / "runs").mkdir(parents=True)
    monkeypatch.setattr(ledger, "EXPERIMENTS_DIR", e)
    monkeypatch.setattr(ledger, "LEDGER_PATH", e / "ledger.csv")
    return e


# series -> (y per cell, yhat per cell) over 2 origins x 2 weeks = 4 cells
_CELLS = {
    "a": ([10, 10, 10, 10], [12, 8, 10, 10]),   # abs_err 4,  y 40, yhat 40
    "b": ([5, 5, 5, 5], [5, 5, 5, 0]),           # abs_err 5,  y 20, yhat 15
    "c": ([0, 0, 2, 3], [1, 0, 2, 3]),           # abs_err 1,  y  5, yhat  6
}


def _results():
    rows = []
    for s, (ys, fs) in _CELLS.items():
        for i, (y, f) in enumerate(zip(ys, fs)):
            rows.append({SERIES_ID: s, "ds": pd.Timestamp("2023-01-02")
                         + pd.Timedelta(weeks=i), "y": float(y), "yhat": float(f),
                         "origin": pd.Timestamp("2023-01-02")})
    forecasts = pd.DataFrame(rows)
    pooled = metrics.score_frame(forecasts)
    # per_origin / segments content is irrelevant here; only shape is persisted
    per_origin = pd.DataFrame({"origin": [0, 1], "vn1": [0.2, 0.2]})
    seg = pd.DataFrame({"segment": ["lumpy"], "volume_share": [1.0], "vn1": [0.2]})
    return {"per_origin": per_origin, "segments": seg, "volume_segments": seg,
            "forecasts": forecasts, "pooled": pooled}


CONFIG = {"model": {"name": "test", "params": {}}, "backtest": {"horizon": 13}}


def test_record_run_writes_series_scores_schema(exp):
    ledger.record_run(CONFIG, _results())
    run_id = ledger.run_id_for(CONFIG)
    path = exp / "runs" / run_id / "series_scores.parquet"
    assert path.exists()

    ss = pd.read_parquet(path).set_index("series")
    assert list(pd.read_parquet(path).columns) == ["series", "abs_err_sum", "y_sum", "yhat_sum"]
    assert (ss.loc["a"] == [4.0, 40.0, 40.0]).all()
    assert (ss.loc["b"] == [5.0, 20.0, 15.0]).all()
    assert (ss.loc["c"] == [1.0, 5.0, 6.0]).all()


def test_series_scores_reproduce_pooled_vn1(exp):
    """The gate's pooled VN1 (from series_scores) must equal the frozen metric
    over the raw cells -- otherwise the gate scores something else than the ledger."""
    res = _results()
    ledger.record_run(CONFIG, res)
    ss = pd.read_parquet(exp / "runs" / ledger.run_id_for(CONFIG) / "series_scores.parquet")

    gate_pooled = gates._vn1_of(ss.abs_err_sum, ss.y_sum, ss.yhat_sum)
    cell_vn1 = metrics.vn1_score(res["forecasts"].y.values, res["forecasts"].yhat.values)
    assert np.isclose(gate_pooled, cell_vn1)
    assert np.isclose(gate_pooled, res["pooled"]["vn1"])
