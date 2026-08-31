# Audit note — data provenance & revalidation vs official VN1 source

Date: 2026-08-30 · Documentation-only, zero compute, no gate budget · human-approved.
Trigger: the human supplied the official competition source pages and its data
description, and asked for a full revalidation of the project against them ("see if
anything changes, or is there any misunderstanding in the objective or real usecase").

## What was validated (local data vs official source)

| Item | Official source / file evidence | Verdict |
|---|---|---|
| Series universe | 15,053 rows in all four CSVs; identical universe both phases | ✓ |
| Quantity | "Number of weekly units sold" | ✓ |
| Phase 0 window | Sales/Price columns 2020-07-06 → 2023-10-02 (170 weeks) | ✓ |
| Phase 1 window | 2023-10-09 → 2024-01-01 (13 week-columns) — **header-only read** | ✓ |
| Horizon constant | `HORIZON = 13` in `src/data.py` matches the Phase-1 window | ✓ |
| Complete panel | `raw_data.parquet` = 2,754,699 rows = 15,053 × 183 weeks (170+13) | ✓ |
| Metric | `vn1_score` in `src/metrics.py` = (Σ\|F−A\| + \|ΣF−ΣA\|)/ΣA — matches the official score | ✓ |
| Price semantics | Description: "you won't see prices if there is no product transaction in a specific week" — matches the standing prior that missing price is structural, not a bug | ✓ confirmed verbatim |
| Phase 1 lock | `.claude/hooks/protect_frozen.py` blocked a read-only command during this very review for containing the string `final_eval` | ✓ enforced |

**Method note (lock discipline).** The Phase 1 CSVs were inspected **header-only** —
column names and row count via `head -1` / `wc -l`. **Zero sales values were read.** This
stays inside both the letter and the spirit of the Phase 1 lock; the lock's guarded path
(`src.final_eval`, gated on `experiments/PHASE1_SIGNOFF`) was never invoked and correctly
fired when a command string merely mentioned it.

## Findings

1. **Missing citation (fixed).** The required Vandeput/DataSource.ai citation appeared
   nowhere in the repo (only an incidental mention in
   `2026-08-28-fm-contamination-audit.md`). Added to README (new *Data source &
   citation* section), ROADMAP (*Source data & comparability*), the dashboard footer,
   and CLAUDE.md (Environment).
2. **Comparability caveat (documented).** The competition scored the same Phase-1
   forecast task but with a live leaderboard participants could probe repeatedly (VN1
   ran Sep 12 – Oct 17 2024; Phase-1 actuals withheld until after close — see
   `2026-08-28-fm-contamination-audit.md`). Our protocol is the same task run stricter:
   one-shot, no leaderboard feedback. Whether the official *final ranking* window equals
   these Phase-1 files exactly is **not verifiable from in-repo evidence** and must be
   checked on the login-gated competition page before the paper prints any official
   leaderboard number beside ours. Recorded in ROADMAP.

## Verdict on the objective / real use case

**No misunderstanding, and the dataset is correct and unchanged.** The project does not
claim to replicate the competition; it uses the competition's dataset and official metric
as a testbed for its own registered research question, exactly as ROADMAP's scope fences
state. Data semantics (weekly units, price-only-on-sale-weeks, phase boundary, 13-week
horizon) all match the official description precisely.

**Decision recorded — dataset stays as-is.** Considered and rejected: switching datasets.
Rejected because (a) revalidation found no data defect; (b) every ledger result, the
champion `1bb3cb32c6`, and all learnings are measured on this data — a swap invalidates
the entire project; (c) ROADMAP's "no new datasets" fence forbids it absent a
human-approved memo naming a published claim that requires it, and none exists.
**No affiliation is needed** to use the data — the source grants public use conditioned
only on citation; citing the author is the opposite of claiming association with them.
A separate, publication-time item: confirm whether the raw CSVs may be redistributed in a
public repo (README currently says they are not, pending that check).

## Sources

- https://www.datasource.ai/en/users/anup-meshram/competitions/phase-2-vn1-forecasting-accuracy-challenge/datathon_detail/description (description, user-scoped)
- https://www.datasource.ai/en/users/anup-meshram/competitions/phase-2-vn1-forecasting-accuracy-challenge/datathon_detail/datasets (datasets, user-scoped)
- https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/description (public citation URL)
