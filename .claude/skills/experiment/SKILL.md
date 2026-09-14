---
name: experiment
description: EXPERIMENT stage — implement and run the human-approved hypothesis through the frozen harness and report results vs the champion. Use only after a /propose memo was approved.
---

# /experiment

Run ONLY what the human approved — if they approved a beam, run every
approved option; gate each challenger independently afterwards. If the approved option needs code (a new
model class), write it under `src/models/`, register it in
`src/models/__init__.py`, and add a contract test in `tests/` — never touch
the frozen files (`src/metrics.py`, `src/backtest.py`, `src/final_eval.py`).

1. Create the config in `configs/` exactly as the approved memo specified
   (backtest block stays `{n_origins: 4, horizon: 13, step: 13}` unless the
   human approved a scheme change — changing it makes runs incomparable).
2. `./forecasting/bin/python -m pytest tests/` must pass first.
3. Run: `./forecasting/bin/python -m src.run_experiment configs/<name>.yaml`
4. Verify the ledger row and `experiments/runs/<run_id>/` artifacts exist.
5. Report vs the champion: pooled VN1/WAPE/bias, per-origin wins, and the
   segment deltas relevant to the memo's hypothesis. State plainly whether
   the hypothesis was supported, refuted, or inconclusive — a refuted
   hypothesis is a finding, not a failure; append it to the memo.
6. If it looks promotable, suggest running `/promote`. Do not run the gate's
   `--promote` here.
