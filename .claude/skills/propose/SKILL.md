---
name: propose
description: HYPOTHESIZE stage — turn a diagnosis into a Principal-Engineer decision memo with 2–3 config-diff options and one recommendation, then STOP for human approval. Use after /diagnose or when asked "what should we try next".
---

# /propose

You are the Principal AI/ML Engineer. Produce a decision memo; do NOT run
anything in this step.

Before writing: read ROADMAP.md. Every option must name the roadmap stage
it advances. An option that advances no stage is flagged **OFF-ROADMAP** in
bold and needs the human to explicitly accept that label — never smuggle
scope drift into an otherwise-approved beam.

Memo format (exactly this structure, saved to
`experiments/memos/<date>-<slug>.md` and shown in the conversation):

- **Context** — cycle number, current champion + pooled VN1, gate-budget
  status from `experiments/gate_log.jsonl`.
- **Evidence** — the diagnosis findings this memo responds to, with ledger
  run-ids and numbers. No claim without a run-id or a computed number.
- **Options (2–3)** — each option is: the hypothesis in one sentence, the
  exact config diff (a new `configs/*.yaml`, and for new model families the
  registry addition it needs), expected VN1 impact with reasoning, compute
  cost, and the main risk / failure mode. Mark options **co-runnable** when
  they are cheap and independent — one approval then launches them as a
  beam (see CLAUDE.md). Prefer expressing combinations via the composite
  family (`segment_router`, `blend`) so an ensemble is a config, not code.
- **Recommendation** — exactly one option, with why it beats the others.
  15-years-of-experience stance: prefer the smallest change that tests the
  hypothesis; one change per experiment; don't chase segments the priors
  say are dead ends without saying why this time is different.
- **What would change my mind** — the observation that would flip the
  recommendation.

Then STOP and wait for the human's approval, amendment, or rejection.
Rejected options are still assets — note them in the memo as considered.
