# Decision memo — Cycle 9: Stage 3, the blend

Date: 2026-08-30 · Cycle 9 · Stage: **Stage 3 — the team-up** (blend the
best deep with the best non-deep; gate the blend)

## Context

- Champion `7548b6e864` (0.5622). Budget 11/25.
- Stage 2 concluded: best trained deep net is N-HiTS-no-cov `eb69d1a3bc`
  0.5677 (competitive, just shy); deep-net covariates don't integrate
  stably; Poisson is the stable count loss, N-HiTS ≫ PatchTST.
- ROADMAP registered prediction 3: "the blend beats every single model."
  Stage 3 tests it. This is where different-family, different-failure-mode
  combination is expected to win.

## Ingredients (ledger, pooled VN1 / bias / key segment)

| run | model | VN1 | bias | lumpy VN1 (bias) | January |
|---|---|---|---|---|---|
| `7548b6e864` | GBM-24 (champion) | 0.5622 | −0.8% | 0.682 (−2.6%) | 0.892 |
| `d047d88915` | GBM-56 | 0.5429 | −4.9% | 0.723 (**−10.5%**) | **0.497** |
| `828187da8a` | Chronos-qmean | 0.5396 | −0.3% | 0.925 (**+15.2%**) | 0.730 |
| `eb69d1a3bc` | N-HiTS-no-cov | 0.5677 | +3.6% | 0.777 (+8.8%) | 0.700 |

## Evidence / rationale

- **Lumpy is the blocking segment** (R4 killed gbm56 and qmean). The two
  best non-champion models fail lumpy in OPPOSITE directions: gbm56
  **−10.5%**, qmean **+15.2%**. A weighted average is the natural mechanism
  to CANCEL that bias where every single-model fix failed — and both are
  strong at the January turn (0.497, 0.730) where the champion is weak
  (0.892). Diversification also typically lowers pooled WAPE.
- Standing-prior guardrails: VN1 is NOT segment-separable and offsetting
  bias is portfolio-global (`7a52b16fe3`) — so I register the mechanism but
  the GATE decides; never predict the blend's pooled score by summing
  segment gains. A blend differs from the failed router: it averages
  portfolio-wide, it does not segment-swap.
- **Feasibility verified**: the `blend` composite (exists, leak-free —
  re-fits each component per origin on train only) runs a mixed GBM+Chronos
  blend ONLY with `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` (else the
  libomp/torch OpenMP deadlock from early in the project). Forecasts were
  never persisted, so the blend must re-run components (no cached path).

## Options

### Option A — blend GBM-56 + Chronos-qmean, 50/50 (recommended)

**Hypothesis:** averaging the two best-by-score models — a tuned GBM and a
zero-shot foundation model whose lumpy biases are equal-and-opposite —
cancels the lumpy bias that R4-blocked both, keeps their shared January
strength, and lowers WAPE by diversification, beating the champion pooled
AND clearing R4 (the first challenger to do so on lumpy).

Config: new `configs/blend_gbm56_qmean.yaml` —
`model: {name: blend, params: {models: [<gbm56 cfg>, <chronos qmean cfg>],
weights: [0.5, 0.5]}}`. Weights PINNED 50/50 (equal, untuned — a weight
sweep is shopping; a principled inverse-variance/holdout weight is a
separate follow-up memo only if 50/50 shows promise). Zero new model code.

Run: `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python -m
src.run_experiment configs/blend_gbm56_qmean.yaml`. A blend contract test +
a pre-run smoke (full-scale, first origin) confirming no deadlock/NaN before
the full 4-origin run.

Expected impact: pooled **0.50–0.55** (below both components via
diversification + bias cancellation); lumpy bias → roughly +2% (from
−10.5%/+15.2% at 50/50), lumpy VN1 improved vs both. P(beats champion) ≈
50%; **P(clears R4) markedly higher than any single model** — the point of
the blend.
Cost: re-fits gbm56 (single-threaded, slower) + chronos per origin ×4 ≈
1–1.5h; ≤1 gate eval (gate only on a pooled win).
Risk: 50/50 isn't the bias-cancelling optimum (lumpy volume-weighted
cancellation may need ~55/45) → residual lumpy bias; or diversification
gains are smaller than the offsetting-bias prior fears. Both are gate-caught.

### Option B — blend champion (GBM-24) + N-HiTS-no-cov, 50/50 — considered

The literal "promoted GBM + best TRAINED deep net." Cleaner provenance (both
are/near champions), and it showcases the Stage-2 trained net. But neither
has the equal-and-opposite lumpy property, so the lumpy-cancellation
mechanism — the strongest reason to expect an R4 pass — is weaker. Park as
the alternative if A's un-promoted components are a concern.

### Option C — 3-way champion + qmean + N-HiTS — considered, deferred

More diverse but adds weight knobs and compute; run only if a 2-way wins
and we want to push further. Not first.

## Recommendation

**Option A.** It is the mechanism-targeted blend: it attacks lumpy — the
one segment that has blocked every promotion — with the one tool
(averaging equal-and-opposite biases) no single model could use, while
keeping the January strength and gaining WAPE diversification. Best-by-score
ingredients, one pinned untuned weight, ≤1 gate eval.

## What would change my mind / forks

- A beats champion + clears R4: gate, promote on your confirm — the blend
  is the new champion; ROADMAP prediction 3 supported; this is the paper's
  climax. A principled weight-optimization follow-up may then squeeze more.
- A beats champion pooled but STILL fails R4 on lumpy: the offsetting-bias
  trap bites even a blend (lumpy cancellation incomplete at 50/50) → one
  principled-weight retry (holdout-fit weight, not a sweep) as a focused
  follow-up; if that also fails, lumpy is structurally unwinnable and the
  champion stands.
- A doesn't beat champion pooled: diversification didn't overcome the
  components' individual deficits → try Option B (champion-anchored blend)
  before concluding the blend thesis is refuted.

## Registered predictions (to score at /reflect)

1. Blend runs NaN-free with the OMP mitigation (no deadlock).
2. Pooled VN1 0.50–0.55 (beats champion 0.5622).
3. Lumpy bias magnitude < both components (|+2%| ≪ 10.5%/15.2%); lumpy VN1
   below both — the cancellation mechanism.
4. Clears R4 (no segment catastrophe) — the first challenger to do so.
5. Overall this is the likeliest promotion of the project; gate decides.

**STOP — awaiting human approval, amendment, or rejection.**

---

## Results — Option A executed (2026-08-30), run `1bb3cb32c6`

Approved and run (`configs/blend_gbm56_qmean.yaml`, OMP mitigation, 42/42 tests
passed, full-scale first-origin smoke clean: 195,689 rows, 0 NaN).

| metric | champion `7548b6e864` | blend `1bb3cb32c6` | Δ |
|---|---|---|---|
| pooled VN1 | 0.5622 | **0.5192** | −0.043 (better) |
| pooled WAPE | 0.5542 | 0.4933 | better |
| pooled bias | −0.8% | −2.6% | (already inside VN1) |

Per-origin VN1 (blend vs champion): Oct-22 0.716<0.831 · **Jan-23 0.564<0.892**
· Apr-23 0.459<0.501 · Jul-23 0.467<0.513 → **4/4 origin wins**. The January
weakness (the memo's second motivation) is fixed outright.

Segment VN1 (all four >10%-volume, gated, R4 limit = champ×1.05):
smooth 0.410 (limit 0.549) · intermittent 0.511 (0.585) · **lumpy 0.678
(limit 0.716) — below the champion's own 0.682** · erratic 0.521 (0.646).
**Better than the champion on every gated segment.** Lumpy bias landed at
**+2.34%** (memo predicted ~+2%) — the equal-and-opposite cancellation
(gbm56 −10.5% / qmean +15.2%) worked at the pooled level, and lumpy VN1
0.678 is below BOTH components (0.723 / 0.925) and below the champion.

Computed gate rules from artifacts: **R1 PASS · R2 PASS (4/4) · R4 PASS**
(first challenger ever to clear R4 on lumpy). R3 (paired bootstrap p<0.05)
not run here — left for the certified `/promote` gate to spend the single
budgeted evaluation.

Registered predictions, scored: (1) NaN-free with OMP — **supported**;
(2) pooled 0.50–0.55 beating champion — **supported** (0.5192); (3) lumpy
bias ≪ both components, lumpy VN1 below both — **supported** (+2.34%; 0.678
< 0.723 & 0.925); (4) clears R4 — **supported** (computes clear, gate to
certify); (5) likeliest promotion of the project — computes to PASS on
R1/R2/R4, gate decides R3. Hypothesis **SUPPORTED**. Next: `/promote`
(gate certifies R3, human confirms).
