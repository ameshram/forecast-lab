# Decision memo — Cycle 13-b: TimesFM-3 zoo addendum

Date 2026-09-01 · Cycle 13-b · **OFF-ROADMAP** (benchmark-zoo addendum, same
precedent and published-claim justification as the human-approved Cycle-10
zoo: comparison breadth for the paper). Human prompt: Google released
TimesFM-3 yesterday (research.google blog, 2026-08-31) — "should we include
it?"

**Standing of the stopping rule:** Cycle 13 was declared quiet (2 of 2,
human-confirmed 2026-09-01); the final-exam condition is MET. This addendum
does NOT re-open the search: it is a comparison run. Only if it computes a
pooled win over the champion (priors: very unlikely) would we face a
promotion decision — and knowing that BEFORE the one-shot Phase 1 exam is
strictly better than after. The dashboard's "search complete" banner is held
until this resolves.

## Context

- Champion: `4f9b475ead` (3-way blend), pooled VN1 **0.5137**. Budget 16/25;
  zoo runs cost 0 evals.
- TimesFM-3 (announced 2026-08-31): 330M params, 1T+ training points,
  checkpoint `google/timesfm-3.0-pytorch`, pip `timesfm==3.0.0`. Headline
  changes vs 2.x: native multivariate + covariates, and **nine quantile
  heads (0.1–0.9) replacing the point-only output**.

## Evidence (why this one model, days before freezing)

- **It tests our own published mechanism with the family's new instrument.**
  Our Cycle-3 finding: TimesFM-2.5's failure here is its point head's
  under-forecast on zero-inflated series (`e8a11552ec`: pooled 0.6512, best
  WAPE in the ledger 0.4612, intermittent bias −23%, lumpy −31%) — the
  readout, not the backbone. TimesFM-3 now has exactly the quantile-head
  structure that made Bolt-qmean (`828187da8a`, bias −0.3%) the only
  well-calibrated zero-shot readout we ever found. Either outcome is a
  paper sentence: the fix works for the family too, or the FM under-forecast
  signature survives even quantile heads.
- **Reviewer-proofing:** a same-family FM released the day before freeze,
  absent from the table, is the first review question. Cost: one ledger row.
- **Priors say it will NOT beat the champion**: every FM lost pooled (best:
  Bolt-qmean 0.5396); "bigger/newer checkpoints don't help" is
  twice-confirmed (Chronos-2 `461e681f0d`/`92906cd09e` both lose to Bolt);
  the FM ~−20% under-forecast signature holds in 3 of 4 families.
- **Feasibility (verified 2026-09-01, resolver dry-run):** `timesfm 3.0.0`
  is a GREEN in-place upgrade — "Would install timesfm-3.0.0" ONLY, zero
  dependency changes. Reproducibility caveat: the upgrade replaces the
  2.0.2 package behind `e8a11552ec`; re-running that config afterwards
  requires `pip install timesfm==2.0.2` (trivial, no dep churn) — recorded
  here as the rollback path.

## Options

**Option A — TimesFM-3 univariate zero-shot, full Cycle-10 discipline.
RECOMMENDED.** Hypothesis: quantile-head readouts fix the TimesFM family's
intermittent under-forecast, moving it from 0.6512 toward (not past)
Bolt-qmean's 0.5396. Scope and order:
  1. Upgrade `timesfm` 2.0.2→3.0.0 in the main venv (GREEN); extend
     `src/models/timesfm_fm.py` with a v3 branch (heavy imports stay inside
     methods) + contract test.
  2. **Mandatory per-family readout-semantics test** (does the point/"mean"
     output equal the median? — assumed for NO family, the `524453e7d8`
     lesson, three occurrences) and **bias-vs-actuals smoke**
     (smoke_fm.py, 2000 real series, band [0.85, 1.20]) before any full run.
  3. Full 4-origin runs for the readouts the semantics test distinguishes —
     median and qmean, matching the Bolt and Chronos-2 treatment (two
     configs, both zoo rows, zero gate spend). Univariate ONLY — the new
     covariate path is a different question (deliberately excluded: one
     change per experiment; our price covariate is structurally missing on
     non-sale weeks and untested in any zero-shot FM).
Expected: pooled **0.52–0.62** (family-best would be ≈0.54; must beat
0.5137 to matter — not predicted); smooth ≤ 0.36 (family strength);
intermittent bias improved vs −23% under qmean readout. Cost: ~1.3 GB
download + ~30–60 min CPU per readout config + smokes. Risk: v3 API
unknowns (mitigated by contract test + smoke before full runs); if the
package upgrade breaks the 2.5 API, rollback pin after the run.

**Option B — TimesFM-3 with covariates (price/promo).** Rejected now:
second change, structurally-missing price, no zero-shot-covariate prior;
would need its own memo if ever.

**Option C — don't run it; freeze the table as-is.** Defensible (the zoo
already answers the question) but leaves the obvious review question open
for the cost of one afternoon ledger row.

## Recommendation

**Option A.** Smallest change that closes the table's currency gap and
tests our own readout mechanism on its home family; every fence from
Cycle 10 applies; the stopping-rule declaration stands unless the ledger
itself says otherwise.

## Registered predictions (score at /reflect)

1. TimesFM-3 does NOT beat the champion 0.5137 pooled; best readout lands
   **0.52–0.62**.
2. The qmean-style readout is better-calibrated than the median on
   intermittent/lumpy (family-fix hypothesis): intermittent bias improves
   from 2.5's −23% to within **−10..+5%** under the quantile-mean readout.
3. Smooth stays a family strength: **≤ 0.36** on the best readout.
4. Semantics test finds the point/"mean" output is NOT a genuine
   distribution mean (coin-flip lean: it equals the median, as in all
   three prior families).
5. Zero gate evals spent; the quiet-2-of-2 declaration stands.

## What would change my mind

- A computed pooled win < 0.5137 → gate it (and the stopping-rule state is
  re-examined honestly — that is the rule working, not drift).
- v3 API requires a RED dependency change after all → isolated-venv path
  (Moirai/TTM pattern) instead of the in-place upgrade.

**STOP — awaiting human approval, amendment, or rejection.**

---

## ADDENDUM — realized results (appended at /reflect, 2026-09-01)

Human approved Option A; both readouts ran. **No champion threat; the quiet
2-of-2 stopping-rule declaration STANDS.** 0 gate evals; budget 16/25.

| | predicted | realized |
|---|---|---|
| Best readout pooled | 0.52–0.62, no threat | median `9c198e246f` **0.7304** ✗ above range (worse than 2.5's 0.6512); qmean `8582538c28` 1.0256 |
| qmean intermittent bias | −10..+5% | **−6.2% ✓** (from 2.5's −23% — mechanism held) |
| Smooth | ≤0.36 | **0.346 ✓** (median readout) |
| Native head semantics | not a genuine mean (lean: median) | **== median, byte-identical ✓** (4/4 families) |
| Gate spend | 0 | **0 ✓** |

The registered family-fix mechanism WORKED where registered (intermittent)
and was swamped by an unregistered failure: lumpy over-forecast +76%
(median) / +140% (qmean) concentrated at the Jan origin (1.98 / 2.73).
Combined with Chronos-2 (`92906cd09e`), this establishes the
generation-shift finding: newest FMs hallucinate lumpy demand at seasonal
turns where old FMs under-forecast — priors updated in CLAUDE.md. Process
lesson recorded: single-origin smokes miss origin-specific blowups; smoke
the January turn too for zero-shot models. Table current to 2026-08-31;
dashboard republished (38 runs).
