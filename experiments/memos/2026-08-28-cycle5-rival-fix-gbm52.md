# Decision memo — Cycle 5: rival fix (global_lgbm, full seasonal cycle)

Date: 2026-08-28 · Cycle 5 · Stage: **OFF-ROADMAP (Stage 0 revisit)** —
this option advances no open stage and runs only if the human explicitly
accepts that label. Justification for asking anyway: every Stage 1–3
conclusion is denominated in the champion, and the champion has a known,
registered training-window artifact. ROADMAP itself demands deep learning
"face the strongest rival at full strength."

## Context

- Champion: `7548b6e864` (global_lgbm, n_anchors=24) — pooled VN1 0.5622,
  bias −0.8%. Gate budget: 11/25 used.
- Strongest challenger: `828187da8a` (Chronos qmean, 0.5396, 4/4 origins,
  gate FAIL on R4 lumpy only).
- Retrospective Q1: is the champion's seasonal-turn weakness an artifact
  of its 24-anchor window? If yes, current Stage-1 comparisons flatter the
  foundation models, and ROADMAP prediction 2 could later confirm for the
  wrong reason (strawman rival).

## Evidence

- `7548b6e864` January origin: VN1 0.8916, bias **+25.0%** — worst in the
  ledger at that origin (naive 0.6955, router 0.6510). With 24 anchors the
  training rows span ~6 months and contain no post-holiday decline
  (cycle-2 learnings; registered lever "anchors ≥ 52").
- Independent confirmation the turn is learnable: both zero-shot FMs,
  whose pretraining saw seasonal turns, beat the champion at January by
  0.29+ (chronos 0.5994 `4248b74828`, timesfm 0.5735 `e8a11552ec`).
- October origin (0.8309, bias −19.4%) is the mirror case: no prior
  holiday ramp in the window either.

## Options

### Option A — global_lgbm, n_anchors: 56 (recommended)

**Hypothesis:** a training matrix spanning a full seasonal cycle (56
anchor weeks > 52, margin for the 13-week direct-horizon offset) teaches
the ramp and the decline; January bias +25% falls to single digits and
pooled VN1 lands ≤ 0.55.

Config diff: new `configs/gbm_v2_anchors56.yaml` — identical to
`gbm_v1.yaml` with `n_anchors: 56`. Zero new code; frozen harness.
Earliest origin (2022-10-10) has 118 training weeks, so 56 anchors + 13
horizon + lags fit without touching the 60-week guard.
Expected impact: pooled **0.53–0.56**; January 0.89 → 0.65–0.75; smooth
over-forecast (+10.2%) may shrink as the model no longer extrapolates a
recent-months trend into the turn.
Cost: one run (~10–30 min; 2.3× the rows of the 24-anchor fit); **1 gate
eval only if pooled beats 0.5622**.
Risk / failure mode: older anchors dilute recency — the very thing this
portfolio rewards (naive 0.583 vs seasonal-naive 1.021) — degrading the
Apr/Jul origins faster than January improves.

### Option B — n_anchors: 104 (two cycles) — considered, not recommended

Same mechanism, more dilution risk, slower; adds nothing A cannot tell
us. Run only if A improves January but under-shoots pooled, and only
under a fresh approval.

### Option C — do nothing until Stage 2

Keeps the roadmap clean but knowingly measures Stage 2 against a rival
with a documented artifact; rejected by the retrospective on validity
grounds — offered here so the choice is explicit.

## Recommendation

**Option A.** One config knob, directly tests Q1, and re-baselines the
whole program if it passes. Gate discipline as in cycle 4: gate only on a
pooled win (12/25 worst case). If it PASSES the gate, promote (with your
confirm) — and note the side effect: the qmean challenger's 0.5396 must
then be re-read against the new champion before any Stage-1 claims are
written.

## What would change my mind

- If January improves but Apr/Jul degrade enough to lose pooled: the
  recency/seasonality trade-off is real and 24 anchors was NOT a bug but
  a choice this data rewards — a strong paper claim; Option B is then
  pointless and Stage 2 proceeds against the current champion.
- If January barely moves at 56 anchors: the GBM's turn-blindness is
  feature-set-level (no year-over-year features), not window-level —
  which re-opens "woy as an in-model feature" (the churn-robust half of
  the cycle-1 holiday lesson) as a legitimate Stage-0 lever, again only
  under explicit approval.

## Registered predictions (to score at /reflect)

- Pooled VN1 0.53–0.56 (beats champion).
- January origin VN1 0.65–0.75, bias +25% → below +12%.
- Smooth bias shrinks from +10.2% toward +5–8%.
- No R4 catastrophe (all segments within 1.05× of champion).

**STOP — awaiting human approval, amendment, or rejection.**

---

## Outcome (appended 2026-08-29; Option A approved and run)

- `d047d88915` (n_anchors=56): pooled **0.5429** (champion 0.5622), WAPE
  0.4939, bias −4.9%, **4/4 origin wins**. January 0.892→**0.497**, bias
  +25.0%→**−1.6%**; October 0.831→0.685. **Q1 answered: the seasonal-turn
  hole was a training-window artifact.**
- Prediction scoring: pooled ✓ (0.5429 in 0.53–0.56); January bias ✓✓
  (−1.6% vs predicted <+12%); January VN1 overshot the predicted 0.65–0.75
  — the mechanism delivered ~2× the predicted improvement (same
  direction of miss as cycle 3's smooth under-call). Smooth bias +8.8% vs
  predicted +5–8% (just outside). **Missed: "no R4 catastrophe"** —
  lumpy 0.7232 > 0.6818×1.05 = 0.7159 (ratio 1.061).
- **Formal gate skipped by human decision** (verdict arithmetically
  determined; budget preserved at 11/25). No promotion; champion remains
  `7548b6e864`. `d047d88915` stands as the strongest GBM configuration,
  un-promoted, computed-FAIL on R4 only.
- Mechanism: the full-cycle window traded recency-driven lumpy
  over-forecast for seasonal knowledge — lumpy WAPE *improved* (0.656→
  0.618) while lumpy bias deepened (−2.6%→−10.5%), and the segment VN1
  bias term charges it. Lumpy has now blocked two mechanistically
  opposite challengers (qmean over-forecasts it 1.36×; gbm56
  under-forecasts it 1.061×) against the champion's best-in-ledger 0.6818.
