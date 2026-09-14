# Decision memo — Cycle 11: the CatBoost blend-swap

Date 2026-08-31 · Cycle 11 · **Stage 3 (continuation)** — ROADMAP Stage 3 says
"blend the best deep model with the best non-deep model"; Cycle 10's zoo made
CatBoost (`523f9c23da`, 0.5289) the best non-deep single model in the ledger,
so upgrading the blend's non-deep ingredient advances Stage 3 directly. (The
zoo that produced CatBoost was OFF-ROADMAP and human-approved; this follow-up
is on-roadmap.)

## Context

- Champion: `1bb3cb32c6` — 50/50 blend of GBM-56 (`d047d88915`) +
  Chronos-Bolt-qmean (`828187da8a`). Pooled VN1 **0.5192** / WAPE 0.4933 /
  bias −2.6%; 4/4 origins; lumpy 0.678 (bias +2.3%).
- Gate budget: **14/25** (gate_log.jsonl, 14 entries; 0 spent in Cycle 10).
- Stopping rule: Cycle 10 was quiet cycle 1 of 2. If Cycle 11 produces no
  new champion, the final-exam condition is met.

## Evidence

All numbers from `experiments/runs/<id>/` artifacts.

| Component | run_id | pooled VN1 | pooled WAPE | pooled bias | lumpy VN1 | lumpy bias |
|---|---|---|---|---|---|---|
| GBM-56 (in champion) | `d047d88915` | 0.5429 | 0.4939 | −4.9% | 0.7232 | −10.5% |
| CatBoost-56 | `523f9c23da` | 0.5289 | 0.5076 | −2.1% | 0.6671 | −1.3% |
| Chronos-qmean (in champion) | `828187da8a` | 0.5396 | 0.5368 | −0.3% | 0.925 | +15.2% |
| Champion blend | `1bb3cb32c6` | 0.5192 | 0.4933 | −2.6% | 0.678 | +2.3% |

- CatBoost beats GBM-56 pooled (0.5289 vs 0.5429), on lumpy (0.667, −1.3% —
  below even the champion blend's 0.678), and at the champion's two weak
  origins (Oct 0.677 vs 0.716; Jan 0.534 vs 0.564). Same features, same
  Tweedie, same 56 anchors — the booster alone is the change.
- **The trap check (bias is exactly linear in a blend — validated in Cycle 9,
  predicted +2% lumpy realized +2.34%):**
  - Straight swap cat+qmean 50/50 → lumpy bias (−1.3+15.2)/2 = **+6.9%**.
    CatBoost's *neutral* lumpy bias deletes the equal-and-opposite
    cancellation that is the champion's whole mechanism. Bounded WAPE
    reasoning (blend WAPE ≤ weighted avg, minus a cancellation gain that ran
    ≈2–4 pts in Cycle 9): lumpy WAPE ≈ 0.67 + |bias| 0.069 → lumpy VN1
    ≈ **0.74 vs the R4 limit 0.678×1.05 = 0.712 → likely R4-BLOCKED**, even
    though pooled bias (−1.2%) and pooled VN1 (est. ~0.51) likely improve.
    This is the qmean/gbm56 R4 story of Cycles 4–5 all over again.
  - Three-way gbm56+cat+qmean ⅓ each → lumpy bias (−10.5−1.3+15.2)/3 =
    **+1.1%** (better than champion's +2.3%); pooled bias −2.4% (≈ champion);
    pooled WAPE avg 0.5128 minus a smaller cancellation gain (CatBoost is
    error-correlated with GBM-56 — same feature matrix — so it adds less
    diversity than a new family) → est. 0.491–0.498; pooled VN1 est.
    **0.515–0.522, straddling the champion 0.5192** — a genuine coin flip
    with a better lumpy and no R4 exposure.

## Options

**Option A — straight swap: blend(CatBoost-56, Chronos-qmean, 50/50).**
Hypothesis: the better single model upgrades the blend. Config diff: new
`configs/blend_cat_qmean.yaml` (pure composite, zero new code). Expected:
pooled 0.505–0.525 (bias term improves +1.4 pts, WAPE worsens ~0.7), BUT
computed lumpy bias +6.9% → lumpy VN1 ≈ 0.74 → **R4-blocked, not
promotable**. Cost ~25 min CPU. Risk: none beyond compute; value is
falsification — it directly tests the offsetting-bias prior with a fresh
pair. **Co-runnable.**

**Option B — three-way blend: blend(GBM-56, CatBoost-56, Chronos-qmean,
⅓ each). RECOMMENDED.** Hypothesis: adding CatBoost to the champion's pair
keeps the bias cancellation (computed +1.1% lumpy) while its pooled strength
nudges WAPE down. Config diff: new `configs/blend_gbm_cat_qmean.yaml`
(pure composite). Expected pooled 0.512–0.524 (coin flip vs champion, tight
range per the composites-predict-tightly prior); lumpy ≤0.68, clears R4.
Cost ~40 min CPU. Risk: CatBoost-GBM error correlation makes the WAPE gain
smaller than the diversity story suggests → an honest no-improvement result.
**Co-runnable.**

**Option C — run nothing; declare Cycle 11 quiet and trigger the final
exam.** Fastest to paper. Rejected-for-now: two cheap, computed, in-scope
config diffs remain untested against a champion whose margin over CatBoost
alone is 1 point; the stopping rule loses no integrity from one more
evidence-based cycle, and the gate budget is untouched at 14/25.

## Recommendation

**Option B, with A co-run as the beam's falsification arm** (one approval
launches both; ~65 min total; ZERO gate spend unless a computed pooled win
appears, and at most ONE eval — on the pooled winner only — would be spent).
B is the only option whose computed lumpy bias *improves* on the champion
while staying a pooled contender; A alone would repeat the Cycle-4/5 R4
pattern the ledger already predicts. One change per experiment holds: each
option is a single blend-composition config diff.

If neither computes a pooled win: no gate spend, Cycle 11 is quiet cycle
2 of 2, and the next step is the human's PHASE1_SIGNOFF decision.

## Registered predictions (score at /reflect)

1. Both runs NaN-free, full coverage (OMP mitigation env vars).
2. Option A pooled **0.505–0.525**; lumpy bias **+5.5..+8.5%**; lumpy VN1
   **>0.712** (computed-R4-fail) — the offsetting-bias-trap prediction.
3. Option B pooled **0.512–0.524**; lumpy bias **0..+2.5%**; lumpy VN1
   **≤0.68**.
4. Gate spend this cycle ≤1 (0 if neither computes a pooled win).
5. Mechanism: the WAPE cancellation gain from adding CatBoost to the
   GBM-family side is <1 pt (error-correlated with GBM-56), i.e. B's pooled
   WAPE lands ≥0.489.

## What would change my mind

- If A's realized lumpy bias lands well under +5.5%, the linear-bias
  arithmetic misses a cat×qmean interaction — weight-structured blends
  (principled holdout-fit, not a sweep) become worth a memo.
- If B pooled lands ≥0.525, the CatBoost lead is exhausted: stop, quiet
  cycle 2, final exam.

**STOP — awaiting human approval, amendment, or rejection.**

---

## ADDENDUM — realized results (appended at /reflect, 2026-08-31)

Human approved the A+B beam; both ran; **B promoted** with human confirm.

| | predicted | realized |
|---|---|---|
| A `3330b31053` pooled | 0.505–0.525 | 0.5159 ✓ (pooled win) |
| A lumpy bias | +5.5..+8.5% (arith +6.9%) | **+6.94%** ✓ |
| A lumpy VN1 | >0.712 (computed R4 fail) | **0.7489** ✓ — unpromotable |
| B `4f9b475ead` pooled | 0.512–0.524 | **0.5137** ✓ |
| B lumpy bias | 0..+2.5% (arith +1.1%) | **+1.12%** ✓ |
| B lumpy VN1 | ≤0.68 | 0.6509 ✓ |
| B pooled WAPE | ≥0.489 | 0.4893 ✓ |
| Gate spend | ≤1 | **2 ✗** (promote-confirm re-runs the gate; count per invocation) |

Gate on B: PASS all four (3/4 origins, p=0.000, zero segment regressions) →
**promoted 2026-08-31, new champion 0.5137**. Budget 16/25. Quiet-cycle
counter reset to 0. The offsetting-bias trap is now a forward-predictive
tool (predicted before running, confirmed on a fresh pair). Engineering:
the two-config beam segfaulted (libomp, exit 139) when B's LightGBM loaded
after A's CatBoost — one run_experiment invocation per booster-mixing
config; both runs themselves clean.
