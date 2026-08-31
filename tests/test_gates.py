"""Tests for the promotion gate (src/gates.py).

The gate is the project's core "decisions are code, not judgment" claim, so its
rules and refusal paths are exercised directly. Fixtures synthesise run
directories on disk (per_origin.csv, series_scores.parquet, segments.csv) and
gates.EXPERIMENTS_DIR / gates.GATE_LOG are redirected into a tmp dir, so nothing
here touches the real experiments/ ledger.
"""
import sys

import numpy as np
import pandas as pd
import pytest

from src import gates, metrics


# --------------------------------------------------------------------------- #
# fixtures / builders
# --------------------------------------------------------------------------- #
def _series(n, abs_err, y=100.0, yhat=None):
    """Per-series score frame. Series ids are shared across runs so the gate's
    paired bootstrap (which merges challenger and champion on 'series') lines up."""
    yhat = y if yhat is None else yhat
    return pd.DataFrame({
        "series": [f"s{i}" for i in range(n)],
        "abs_err_sum": np.full(n, float(abs_err)),
        "y_sum": np.full(n, float(y)),
        "yhat_sum": np.full(n, float(yhat)),
    })


def _write_run(exp, run_id, origins, vn1s, series_df, segments):
    d = exp / "runs" / run_id
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"origin": origins, "vn1": vn1s}).to_csv(d / "per_origin.csv", index=False)
    series_df.to_parquet(d / "series_scores.parquet")
    pd.DataFrame(segments).to_csv(d / "segments.csv", index=False)


@pytest.fixture
def exp(tmp_path, monkeypatch):
    """A throwaway experiments/ dir wired into the gate module."""
    e = tmp_path / "experiments"
    (e / "runs").mkdir(parents=True)
    monkeypatch.setattr(gates, "EXPERIMENTS_DIR", e)
    monkeypatch.setattr(gates, "GATE_LOG", e / "gate_log.jsonl")
    return e


ORIGINS = [0, 1, 2, 3]
# a champion that a clearly-better challenger should beat on every rule
_CHAMP_SEG = [{"segment": "lumpy", "volume_share": 0.6, "vn1": 0.10},
              {"segment": "smooth", "volume_share": 0.4, "vn1": 0.10}]
_CHAL_SEG_OK = [{"segment": "lumpy", "volume_share": 0.6, "vn1": 0.04},
                {"segment": "smooth", "volume_share": 0.4, "vn1": 0.02}]


def _champion(exp):
    _write_run(exp, "champ", ORIGINS, [0.10] * 4, _series(20, abs_err=10), _CHAMP_SEG)


# --------------------------------------------------------------------------- #
# _vn1_of: the div-by-zero fix
# --------------------------------------------------------------------------- #
def test_vn1_of_matches_frozen_metric_on_normal_data():
    # aggregates for y=[10,10], yhat=[12,12]: abs_err=4, y_sum=20, yhat_sum=24
    agg = gates._vn1_of(np.array([4.0]), np.array([20.0]), np.array([24.0]))
    cell = metrics.vn1_score(np.array([10.0, 10.0]), np.array([12.0, 12.0]))
    assert np.isclose(agg, cell) and np.isclose(agg, 0.4)


def test_vn1_of_zero_demand_is_nan_not_inf():
    # non-zero error over zero demand used to be 5/0 -> inf (+ RuntimeWarning).
    with np.errstate(divide="raise", invalid="raise"):
        out = gates._vn1_of(np.array([5.0]), np.array([0.0]), np.array([5.0]))
    assert np.isnan(out)
    assert not np.isinf(out)
    # matches the frozen metric's zero-denominator convention
    assert np.isnan(metrics.vn1_score(np.array([0.0]), np.array([5.0])))


def test_vn1_of_accepts_pandas_series():
    s = _series(3, abs_err=2, y=0.0, yhat=1.0)  # y_sum == 0
    assert np.isnan(gates._vn1_of(s.abs_err_sum, s.y_sum, s.yhat_sum))


# --------------------------------------------------------------------------- #
# paired bootstrap
# --------------------------------------------------------------------------- #
def test_bootstrap_clear_winner_is_significant():
    ch = _series(30, abs_err=1)
    ck = _series(30, abs_err=20)
    assert gates.paired_bootstrap_p(ch, ck, n_boot=300) < 0.05


def test_bootstrap_tie_is_not_significant():
    s = _series(30, abs_err=5)
    # identical challenger/champion -> every resample diff is 0, and a tie is
    # "not better", so p == 1.0
    assert gates.paired_bootstrap_p(s, s.copy(), n_boot=300) == 1.0


def test_bootstrap_all_zero_demand_no_inf_no_crash():
    s = _series(10, abs_err=0, y=0.0, yhat=0.0)
    with np.errstate(all="raise"):  # no divide/invalid FP errors escape
        p = gates.paired_bootstrap_p(s, s.copy(), n_boot=200)
    assert np.isfinite(p) and p == 1.0  # undefined comparison never PASSes


# --------------------------------------------------------------------------- #
# evaluate: rules R1-R4, verdict, origin-set guard
# --------------------------------------------------------------------------- #
def test_evaluate_pass(exp):
    _champion(exp)
    _write_run(exp, "chal", ORIGINS, [0.04] * 4, _series(20, abs_err=4), _CHAL_SEG_OK)
    rep = gates.evaluate("chal", "champ")
    assert rep["verdict"] == "PASS"
    assert all(rep["rules"].values())
    assert rep["origin_wins"] == "4/4"
    assert rep["segment_regressions"] == []


def test_evaluate_fail_when_pooled_worse(exp):
    _champion(exp)
    _write_run(exp, "chal", ORIGINS, [0.04] * 4, _series(20, abs_err=30), _CHAL_SEG_OK)
    rep = gates.evaluate("chal", "champ")
    assert rep["rules"]["R1_pooled_improves"] is False
    assert rep["verdict"] == "FAIL"


def test_evaluate_fail_on_origin_minority(exp):
    _champion(exp)
    # pooled/bootstrap favour the challenger, but it wins only 1 of 4 origins
    _write_run(exp, "chal", ORIGINS, [0.04, 0.20, 0.20, 0.20],
               _series(20, abs_err=4), _CHAL_SEG_OK)
    rep = gates.evaluate("chal", "champ")
    assert rep["rules"]["R1_pooled_improves"] is True
    assert rep["rules"]["R2_origin_majority"] is False
    assert rep["origin_wins"] == "1/4"
    assert rep["verdict"] == "FAIL"


def test_evaluate_fail_on_segment_catastrophe(exp):
    _champion(exp)
    # better pooled, origins and p, but a 60%-volume segment regresses > 5%
    bad_seg = [{"segment": "lumpy", "volume_share": 0.6, "vn1": 0.20},
               {"segment": "smooth", "volume_share": 0.4, "vn1": 0.02}]
    _write_run(exp, "chal", ORIGINS, [0.04] * 4, _series(20, abs_err=4), bad_seg)
    rep = gates.evaluate("chal", "champ")
    assert rep["rules"]["R1_pooled_improves"] is True
    assert rep["rules"]["R2_origin_majority"] is True
    assert rep["rules"]["R3_significant_p05"] is True
    assert rep["rules"]["R4_no_segment_catastrophe"] is False
    assert rep["segment_regressions"] and rep["segment_regressions"][0]["segment"] == "lumpy"
    assert rep["verdict"] == "FAIL"


def test_evaluate_rejects_mismatched_origins(exp):
    _champion(exp)
    _write_run(exp, "chal", [0, 1, 2, 9], [0.04] * 4, _series(20, abs_err=4), _CHAL_SEG_OK)
    with pytest.raises(SystemExit):
        gates.evaluate("chal", "champ")


# --------------------------------------------------------------------------- #
# main(): promotion refusal / success paths (the human-approval wall)
# --------------------------------------------------------------------------- #
def _run_main(monkeypatch, argv):
    calls = []
    monkeypatch.setattr(gates, "set_champion",
                        lambda run_id, reason: calls.append((run_id, reason)))
    monkeypatch.setattr(sys, "argv", ["gates", *argv])
    with pytest.raises(SystemExit) as ei:
        gates.main()
    return ei.value, calls


def test_promote_refused_on_gate_fail(exp, monkeypatch):
    _champion(exp)
    _write_run(exp, "chal", ORIGINS, [0.04] * 4, _series(20, abs_err=30), _CHAL_SEG_OK)
    code, calls = _run_main(
        monkeypatch, ["chal", "--champion", "champ", "--promote", "--confirm"])
    assert "gate FAIL" in str(code)
    assert calls == []  # champion never written


def test_promote_refused_without_confirm(exp, monkeypatch):
    _champion(exp)
    _write_run(exp, "chal", ORIGINS, [0.04] * 4, _series(20, abs_err=4), _CHAL_SEG_OK)
    code, calls = _run_main(monkeypatch, ["chal", "--champion", "champ", "--promote"])
    assert "confirm" in str(code).lower()
    assert calls == []  # gate PASSED, but no human confirmation -> no write


def test_promote_success_writes_champion(exp, monkeypatch):
    _champion(exp)
    _write_run(exp, "chal", ORIGINS, [0.04] * 4, _series(20, abs_err=4), _CHAL_SEG_OK)
    code, calls = _run_main(
        monkeypatch, ["chal", "--champion", "champ", "--promote", "--confirm"])
    assert code.code == 0
    assert len(calls) == 1 and calls[0][0] == "chal"
