---
name: diagnose
description: OBSERVE+DIAGNOSE stage of the improvement loop — read the ledger and produce an evidence report on where the champion is weak. Use at the start of a cycle or when asked "where are we / what's weak".
---

# /diagnose

Read-only. Produce the evidence a decision memo will be built on. Never
change models, configs, or the ledger in this step.

1. Load state: `experiments/ledger.csv`, `experiments/champion.json`,
   and the champion's `per_origin.csv`, `segments.csv`,
   `volume_segments.csv` under `experiments/runs/<run_id>/`.
2. Compare the champion against the best non-champion runs per segment:
   where is the champion beaten even by a non-promoted run?
3. Report, in this order:
   - Champion + pooled VN1/WAPE/bias, and per-origin stability (flag any
     origin where it is >20% worse than its own pooled score).
   - **Weakness table**: segments ranked by `(segment VN1 − pooled VN1) ×
     volume_share` — the volume-weighted improvement headroom.
   - Bias check: over- vs under-forecasting overall and per segment.
   - Learned-results recap: hypotheses the ledger has already refuted (do
     not re-propose these without new reasoning).
4. End with 2–4 candidate weakness statements phrased as testable
   hypotheses ("lumpy series are under-forecast by X% — a method that
   models demand size separately from demand timing should cut this").

Output goes in the conversation; if the human asks, also save to
`experiments/memos/<date>-diagnosis.md`.
