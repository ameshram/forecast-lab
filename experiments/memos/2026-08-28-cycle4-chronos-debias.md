# Decision memo — Cycle 4: attack the foundation-model bias term

Date: 2026-08-28 · Cycle 4 · Stage: **Stage 1 — foundation models, zero
training** (continuation; the lever was registered in cycle 3's reflection)

## Context

- Champion: `7548b6e864` (global_lgbm) — pooled VN1 **0.5622**, WAPE
  0.5542, bias −0.8%.
- Gate budget: **10 of 25 used**. Cycle 3 ran two experiments and gated
  neither (obvious pooled losses).
- Cycle 3 finding: zero-shot Chronos-Bolt (`4248b74828`) loses VN1 0.5878
  vs 0.5622 while *winning* WAPE 0.5134 vs 0.5542 — the entire loss is the
  −7.4% under-forecast bias term (VN1 = WAPE + |portfolio bias|, floor run
  `09fdb04e06`). The question this cycle: is that bias removable by
  changing which point of the predictive distribution we read out, with no
  training and no new mechanism?

## Evidence

- `4248b74828` (chronos, quantile 0.5): pooled 0.5878 / WAPE 0.5134 /
  bias −7.4%. Bias by segment: erratic −16.0%, intermittent −6.6%, lumpy
  −1.8%, smooth +2.5%. Per origin: Oct 0.836 (bias −21.7%), Jan 0.599
  (+6.4%), Apr 0.506 (−5.6%), Jul 0.529 (−7.9%).
- Arithmetic headroom: if bias went to ~0 with WAPE unchanged, VN1 ≈ 0.513
  — a clear champion beat. The median is the WAPE-optimal readout, so some
  WAPE give-back is expected; the bet is that on right-skewed demand the
  mean sits above the median and buys back more bias than it costs WAPE.
- `e8a11552ec` (timesfm): worse substrate for this lever — bias −19.0%
  and its point head already refuted as mean-calibrated (cycle 3
  learnings). Not iterated this cycle.
- Standing-prior check: trailing-ratio debiasing (`2c050a9d24`) is NOT
  being re-proposed — that mechanism scales by recent history and breaks
  at seasonal turns. This lever reads a different point of the model's own
  per-week predictive distribution: no trailing window, turn-aware by
  construction (the Jan origin evidence above shows the distribution
  handles the turn).

## Options

### Option A — Chronos-Bolt, mean readout (Stage 1) — co-runnable

**Hypothesis:** reading the predictive mean instead of the median removes
most of the −7.4% under-bias at a small WAPE cost, putting pooled VN1 at or
below the champion.

Config diff: new `configs/chronos_bolt_base_mean.yaml` — identical to
`chronos_bolt_base.yaml` with `point: mean` (parameter already exists in
`src/models/chronos.py`; zero new code). Note: for Bolt checkpoints the
API's mean is the average of the 9 trained quantiles — a trimmed mean, so
it may under-correct extreme tails.

Expected impact: pooled VN1 **0.53–0.58**, bias −2% to +4%. Genuine gate
chance (needs pooled < 0.5622 + 3/4 origins + p<0.05); call it ~40%.
Cost: ~5 min runtime; 1 gate eval *only if* it beats the champion pooled.
Risk / failure mode: mean overshoots on lumpy/erratic heavy tails —
WAPE inflates faster than bias deflates; January flips to +10%-ish bias
(median already sits at +6.4% there).

### Option B — Chronos-Bolt, quantile 0.65 readout (Stage 1) — co-runnable

**Hypothesis:** same aim through the other knob — a fixed above-median
quantile interpolated from the trained quantile heads shifts every series
up smoothly; it corrects bias with a different tail behavior than the
trimmed mean (bounded shift, no tail-mean overshoot).

Config diff: new `configs/chronos_bolt_base_q65.yaml` — `quantile: 0.65`
(parameter exists; zero new code).
Expected impact: pooled VN1 **0.53–0.59**, bias −3% to +3%.
Cost: ~5 min; gate eval only if promotable.
Risk: a uniform quantile shift over-corrects the segments that aren't
biased (smooth +2.5% already) — the router lesson (`7a52b16fe3`) says the
portfolio bias term, not segment wins, decides.

### Option C — TimesFM mean-of-quantiles readout — considered, not offered

Same lever on the weaker substrate (−19% bias, refuted point head). If the
lever works on Chronos it can be replicated on TimesFM later for the
paper's completeness table; spending a cycle on it now duplicates the test
of one mechanism. (Stage 1, but dominated by A/B.)

**Anti-shopping fence, declared now:** A and B bracket the readout family
(trimmed mean vs fixed quantile). Whatever happens, no third readout
variant gets proposed — one beam settles this mechanism, win or lose.
Stage 3 blending (champion + FM, the cycle-3 complementarity signal) stays
parked until Stage 2 per roadmap order.

## Recommendation

**Option A, with B co-run as one beam.** Both are pure config diffs on the
already-validated model family, ~10 minutes of compute total, and they
bracket the mechanism from two sides — the pair tells us *why* if the
lever fails, not just that it failed. Gate discipline: gate only a variant
that beats the champion pooled (expected ≤1 gate eval), and if both do,
gate only the better one.

## What would change my mind

- If both A and B land with pooled bias still < −5%, the under-forecast is
  not a readout artifact but a distributional miss on zero-inflated series
  — the readout family is dead here (fence above), and Stage 1 closes with
  cycle 3's answer; next memo weighs Stage 2.
- If either flips pooled bias above +5% with WAPE worse than the champion,
  same conclusion from the other side: the distribution is too coarse to
  tune by readout.
- If one passes the gate: promote it, and the paper's Stage 1 section gains
  a twist — "zero-shot beats the trained GBM once you read the right point
  of its distribution."

**STOP — awaiting human approval, amendment, or rejection.**

---

## Outcome (appended 2026-08-28 after the approved beam ran)

- `524453e7d8` (point=mean): **no-op** — for Bolt checkpoints the chronos
  package returns the median as the "mean" (source NOTE in
  `chronos_bolt.predict_quantiles`), so this duplicated `4248b74828`
  exactly. Option A was then executed correctly as `point: qmean`
  (average of the 9 trained quantile heads; code + contract test added).
- `46df0bba04` (Option B, quantile 0.65): pooled 0.7223, bias **+13.2%**,
  WAPE 0.5903 — pre-registered kill condition hit; uniform quantile shift
  over-corrects portfolio-wide (lumpy +29%). Mechanism refuted, no gate.
- `828187da8a` (Option A corrected, qmean): pooled **0.5396** (champion
  0.5622), WAPE 0.5368, bias **−0.3%**, origin wins **4/4** (Jan 0.730 vs
  0.892; Oct 0.789 vs 0.831). Prediction 0.53–0.58 / bias −2..+4% —
  calibration excellent. **Gate FAIL on R4 only**: lumpy 0.9254 vs 0.6818
  (lumpy bias +15.2%); R1/R2/R3 all passed (p=0.04). Budget 11/25.
- Mechanism read: the trimmed mean buys its neutral pooled bias by
  over-forecasting lumpy to offset residual under-forecast elsewhere — the
  offsetting-bias structure again, now inside one model's readout.
  Arithmetic check (not a run): routing lumpy back to the median readout
  would land ≈ WAPE 0.524 + |bias| 0.038 ≈ **0.562 — champion-level, no
  win**. A lumpy-specific fix must cut the over-forecast without deleting
  the offset it provides.
