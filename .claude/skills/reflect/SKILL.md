---
name: reflect
description: REFLECT stage (after every gate verdict) — score the memo's prediction against the realized result, append to the learnings file, and update CLAUDE.md standing priors when ledger evidence contradicts them. This is the harness's self-improvement mechanism.
---

# /reflect

Runs after every gate verdict (PASS or FAIL), before the next /diagnose.
The frozen core (metrics, backtest, gates, hooks) is NEVER touched here —
reflection only updates the soft layer: learnings, priors, skill heuristics.

1. Append one entry to `experiments/learnings.md`:
   `cycle · run_id · hypothesis (one line) · predicted (from the memo) ·
   realized (from the ledger/gate report) · prediction error · verdict ·
   transferable lesson (one line, only if there is one)`.
   Keep entries terse — a learnings file that rambles loses its signal.
2. Check the memo's stated prediction against the realized numbers. If the
   prediction was off by more than ~50% of the predicted effect, say so in
   the entry and diagnose why (wrong mechanism? interaction? overfit
   segment?). Calibration of predictions is tracked, not just outcomes.
3. If a result contradicts a standing prior in CLAUDE.md, update that prior
   in place with the new run-id as citation (or delete it). If a result
   creates a new durable rule ("X never helps on this data because Y"), add
   it as a prior — with citation. Never add a prior from a single ambiguous
   result.
4. If the same *kind* of hypothesis has now failed twice, note it in the
   propose skill's "do not re-propose" awareness via a learnings entry
   tagged `DEAD-END`.
