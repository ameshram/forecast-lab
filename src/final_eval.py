"""The ONLY blessed entry point to Phase 1 -- the locked final test set.

Phase 1 (2023-10-09 onward) must never inform model selection. This script
refuses to run unless the human has created the sign-off file:

    experiments/PHASE1_SIGNOFF

(create it manually to authorize the one-shot final evaluation; the
.claude/hooks/protect_frozen.py hook blocks running this file without it).

It evaluates the CURRENT CHAMPION, trained on all of Phase 0, over the full
Phase 1 window, and writes experiments/final_eval_report.json.

Usage:
    python -m src.final_eval
"""
from __future__ import annotations

import json
import sys

import yaml

from .data import PHASE1_START, SERIES_ID, load_panel, split_phases, \
    syntetos_boylan_class
from .ledger import EXPERIMENTS_DIR, get_champion
from .metrics import score_by_segment, score_frame
from .models import build_model

SIGNOFF = EXPERIMENTS_DIR / "PHASE1_SIGNOFF"


def main() -> None:
    if not SIGNOFF.exists():
        sys.exit("REFUSED: experiments/PHASE1_SIGNOFF not found. Phase 1 is "
                 "locked; the human must create the sign-off file to authorize "
                 "the one-shot final evaluation.")
    champ = get_champion()
    if not champ or not champ.get("current"):
        sys.exit("no champion registered")
    run_id = champ["current"]
    config = yaml.safe_load(
        (EXPERIMENTS_DIR / "runs" / run_id / "config.yaml").read_text())

    df = load_panel()
    phase0, phase1 = split_phases(df)
    future_ds = sorted(phase1.ds.unique())

    model = build_model(config["model"])
    fcst = model.fit_predict(phase0, list(future_ds))
    merged = phase1[[SERIES_ID, "ds", "y"]].merge(
        fcst, on=[SERIES_ID, "ds"], how="left")
    if merged["yhat"].isna().any():
        sys.exit("champion left unforecast cells on Phase 1")

    merged = merged.join(syntetos_boylan_class(phase0), on=SERIES_ID)
    report = {
        "champion_run": run_id,
        "model": config["model"],
        "phase1_start": str(PHASE1_START.date()),
        "n_weeks": len(future_ds),
        "pooled": {k: round(float(v), 4)
                   for k, v in score_frame(merged).items()},
        "segments": score_by_segment(merged, "sb_class").round(4)
                    .to_dict(orient="records"),
    }
    out = EXPERIMENTS_DIR / "final_eval_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report["pooled"], indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
