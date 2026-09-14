"""Deterministic promotion gate: the loop's BETTER? decision is code, not judgment.

A challenger is promotable ONLY if every rule passes:

  R1  pooled VN1 strictly better than the champion's
  R2  better VN1 on a strict majority of forecast origins
  R3  paired bootstrap over series: P(challenger not better) < 0.05
  R4  no catastrophic segment regression: on every Syntetos-Boylan class
      carrying >= 10% of volume, challenger VN1 <= champion VN1 * 1.05

The gate certifies; a human approves; only then does --promote --confirm
write the champion registry. Every evaluation is appended to
experiments/gate_log.jsonl -- repeated gating against the same validation
windows is how loops overfit, so the log warns past EVAL_BUDGET evaluations.

Usage:
    python -m src.gates <challenger_run_id> [--champion <run_id>]
    python -m src.gates <challenger_run_id> --promote --confirm   # after human yes
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .ledger import EXPERIMENTS_DIR, get_champion, set_champion

EVAL_BUDGET = 25
GATE_LOG = EXPERIMENTS_DIR / "gate_log.jsonl"


def _run_dir(run_id: str):
    d = EXPERIMENTS_DIR / "runs" / run_id
    if not d.exists():
        sys.exit(f"unknown run_id {run_id}")
    return d


def _vn1_of(abs_err, y, yhat) -> float:
    """Pooled VN1 from per-series aggregates (abs_err = Sigma|F-A| per series,
    y = Sigma A per series, yhat = Sigma F per series).

    Same formula and same zero-denominator convention as metrics.vn1_score:
    an all-zero-demand slice has an undefined VN1 and scores nan, rather than
    dividing by zero (which yields inf/nan plus a RuntimeWarning and would
    silently corrupt the pooled score and the bootstrap difference).
    """
    denom = y.sum()
    if denom == 0:
        return np.nan
    return (abs_err.sum() + abs(yhat.sum() - y.sum())) / denom


def paired_bootstrap_p(ch: pd.DataFrame, ck: pd.DataFrame,
                       n_boot: int = 2000, seed: int = 7) -> float:
    """P(challenger is not better), resampling series with replacement."""
    m = ch.merge(ck, on="series", suffixes=("_c", "_k"))
    ae_c, y_c, f_c = (m["abs_err_sum_c"].values, m["y_sum_c"].values,
                      m["yhat_sum_c"].values)
    ae_k, y_k, f_k = (m["abs_err_sum_k"].values, m["y_sum_k"].values,
                      m["yhat_sum_k"].values)
    rng = np.random.default_rng(seed)
    n = len(m)
    not_better = 0
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d = (_vn1_of(ae_c[idx], y_c[idx], f_c[idx])
             - _vn1_of(ae_k[idx], y_k[idx], f_k[idx]))
        # A degenerate resample (all-zero demand) makes VN1 undefined, so d is
        # nan; count it as "not better" so an undefined comparison can never
        # push the gate toward a PASS. For finite d this is identical to d >= 0.
        if not (d < 0):
            not_better += 1
    return not_better / n_boot


def evaluate(challenger: str, champion: str) -> dict:
    cd, kd = _run_dir(challenger), _run_dir(champion)
    ch_o = pd.read_csv(cd / "per_origin.csv")
    ck_o = pd.read_csv(kd / "per_origin.csv")
    if set(ch_o.origin) != set(ck_o.origin):
        sys.exit("challenger and champion were backtested on different origins "
                 "-- rerun the challenger on the current origin set first")

    ch_s = pd.read_parquet(cd / "series_scores.parquet")
    ck_s = pd.read_parquet(kd / "series_scores.parquet")
    ch_seg = pd.read_csv(cd / "segments.csv").set_index("segment")
    ck_seg = pd.read_csv(kd / "segments.csv").set_index("segment")

    pooled_c = _vn1_of(ch_s.abs_err_sum, ch_s.y_sum, ch_s.yhat_sum)
    pooled_k = _vn1_of(ck_s.abs_err_sum, ck_s.y_sum, ck_s.yhat_sum)

    merged_o = ch_o.merge(ck_o, on="origin", suffixes=("_c", "_k"))
    origin_wins = int((merged_o.vn1_c < merged_o.vn1_k).sum())
    n_origins = len(merged_o)

    p = paired_bootstrap_p(ch_s, ck_s)

    seg_fail = []
    for seg in ck_seg.index:
        if ck_seg.loc[seg, "volume_share"] >= 0.10 and seg in ch_seg.index:
            if ch_seg.loc[seg, "vn1"] > ck_seg.loc[seg, "vn1"] * 1.05:
                seg_fail.append({"segment": seg,
                                 "challenger": round(float(ch_seg.loc[seg, "vn1"]), 4),
                                 "champion": round(float(ck_seg.loc[seg, "vn1"]), 4)})

    rules = {
        "R1_pooled_improves": bool(pooled_c < pooled_k),
        "R2_origin_majority": bool(origin_wins * 2 > n_origins),
        "R3_significant_p05": bool(p < 0.05),
        "R4_no_segment_catastrophe": len(seg_fail) == 0,
    }
    return {
        "challenger": challenger, "champion": champion,
        "pooled_vn1": {"challenger": round(float(pooled_c), 4),
                       "champion": round(float(pooled_k), 4)},
        "origin_wins": f"{origin_wins}/{n_origins}",
        "bootstrap_p_not_better": round(p, 4),
        "segment_regressions": seg_fail,
        "rules": rules,
        "verdict": "PASS" if all(rules.values()) else "FAIL",
    }


def _log_evaluation(report: dict) -> int:
    entry = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "challenger": report["challenger"], "champion": report["champion"],
             "verdict": report["verdict"]}
    with open(GATE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return sum(1 for _ in open(GATE_LOG))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("challenger")
    ap.add_argument("--champion", default=None)
    ap.add_argument("--promote", action="store_true",
                    help="write champion registry if the gate passes")
    ap.add_argument("--confirm", action="store_true",
                    help="required with --promote: records that a human approved")
    args = ap.parse_args()

    champ = args.champion or (get_champion() or {}).get("current")
    if not champ:
        sys.exit("no champion registered")

    report = evaluate(args.challenger, champ)
    n_evals = _log_evaluation(report)
    print(json.dumps(report, indent=2))
    if n_evals >= EVAL_BUDGET:
        print(f"\nWARNING: {n_evals} gate evaluations against this validation "
              f"scheme (budget {EVAL_BUDGET}). Risk of validation overfitting -- "
              f"consider adding origins or freezing further changes.",
              file=sys.stderr)

    if args.promote:
        if report["verdict"] != "PASS":
            sys.exit("refusing to promote: gate FAIL")
        if not args.confirm:
            sys.exit("refusing to promote: --confirm missing (human approval "
                     "is required before promotion)")
        set_champion(args.challenger,
                     f"Gate PASS vs {champ}: pooled "
                     f"{report['pooled_vn1']['challenger']} < "
                     f"{report['pooled_vn1']['champion']}, origins "
                     f"{report['origin_wins']}, p={report['bootstrap_p_not_better']}")
        print(f"\nPROMOTED {args.challenger} (human-confirmed)")

    sys.exit(0 if report["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
