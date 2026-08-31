# Decision memo — Cycle 6-B: Stage 2 retry, numerically stable count loss

Date: 2026-08-29 · Cycle 6-B · Stage: **Stage 2 — one trained deep network,
done properly** (continuation; Cycle 6 crashed as an engineering failure,
Stage 2's scientific question is still OPEN)

## Context

- Champion `7548b6e864` (global_lgbm-24, VN1 0.5622). Budget 11/25.
- Reference points (un-promoted, both R4-blocked on lumpy): gbm56
  `d047d88915` 0.5429, Chronos-qmean `828187da8a` 0.5396.
- Cycle 6 (`configs/patchtst_nb.yaml`, PatchTST + NegativeBinomial)
  **crashed ~15min into training** — NB `probs` forced to 1.0/NaN on the
  first forward/loss eval (numerical overflow from large count inputs, y up
  to 13,669). No ledger row; predictions not evaluable. The NB *mechanism*
  is neither supported nor refuted — we could not train it stably.
- Promotion bar unchanged: beat 0.5622 pooled while holding lumpy segment
  VN1 ≤ 0.7159. The open mechanism is a conditional distribution whose mean
  readout is accurate AND segment-bias-neutral on lumpy.

## Evidence (all from diagnostics, scratchpad — the Cycle 6 lesson applied)

Stability, 3723-series spiky real subsample (max y=13,669), PatchTST
input_size=64, lr=1e-3, 60 steps, distribution-mean readout:

| loss | robust scaler | identity scaler (deterministic NB-crash case) |
|---|---|---|
| NegativeBinomial | OK on subsample (but **crashed at full 15k scale**) | **CRASH** (probs overflow) |
| Poisson | **OK**, mean readout non-NaN, base_mean≈71 | **OK** — survives where NB crashes |
| Tweedie | **CRASH** `KeyError: 'rho'` | **CRASH** `KeyError: 'rho'` |

- Poisson is the only count loss that survives the identity condition that
  deterministically overflows NB → strong evidence it lacks the
  probs-saturation failure mode (single softplus rate parameter, no bounded
  `probs` to saturate).
- Poisson base column is the distribution MEAN (mean≥median share 0.59;
  Poisson is near-symmetric at these rates, so mean≈median — the readout
  distinction matters less than for NB, but the base column is still the
  mean, verified non-NaN).
- Tweedie via neuralforecast's DistributionLoss needs a `rho` kwarg and
  errored in the readout path — not a clean one-change drop-in.

## Options

### Option A — PatchTST + Poisson DistributionLoss (recommended)

**Hypothesis:** a trained global transformer with a *stable* count
likelihood (Poisson) produces a mean readout that beats the champion pooled
and, unlike gbm56/qmean, keeps lumpy segment bias small enough to clear R4.

Config diff: new `configs/patchtst_poisson.yaml` — **identical** to the
crashed `patchtst_nb.yaml` except `distribution: Poisson` (robust scaler,
input_size 64, patch 8/stride 8, max_steps 1500, hidden 128, n_heads 16,
batch 256, lr 1e-3, seed 7, num_samples 1000, cpu). One change only: the
likelihood family. Wrapper already parameterizes `distribution`; zero code
change. A `Poisson` case is added to the contract test.

**Pre-run stability gate (now a blocking pre-req per the new prior):** the
exact config is smoke-fit on a spiky real subsample and must train NaN-free
before the full 4-origin run launches. Poisson already passed this at 3723
series incl. identity; I will re-confirm on the exact final config.

Expected impact: pooled **0.52–0.62**. Reasoning: Poisson's mean is a valid,
stable point; but Poisson assumes equidispersion (var=mean) while retail
demand is over-dispersed — so its point may be slightly less well-calibrated
on lumpy tails than a working NB would be. That trade (robustness now vs
theoretical fit) mainly affects intervals/quantiles, which VN1 does not use;
the point readout is what we score. P(beats champion pooled) ≈ 35%; P(also
clears R4 on lumpy) ≈ 20%.
Cost: ~30–90 min for 4 origins on CPU; ≤1 gate eval (gate only on a pooled
win). Compute risk is now low (stability verified).
Risk / failure mode: over-dispersion mis-fit inflates lumpy error; or the
1500-step/1e-3 budget under/over-fits (untuned, not shopped — a follow-up
memo raises it as a one-change experiment if the fit is clearly starved).

### Option B — Tweedie with an explicit `rho` — considered, rejected

Tweedie is the GBM's winning objective and would be the ideal like-for-like,
but neuralforecast's DistributionLoss('Tweedie') requires a `rho` variance-
power kwarg and errored in the sample/readout path (KeyError 'rho'). Making
it work is >1 change (add rho, verify sampling) and introduces a tuning knob
(rho) — against one-change discipline. Park it; revisit only if Poisson is
stable-but-weak and we want the over-dispersion the GBM benefited from.

### Option C — N-HiTS + NB (Cycle 6 memo's original fallback) — rejected

Keeps the numerically fragile NB loss AND changes the architecture — two
changes, and it re-exposes the exact crash we just diagnosed. Worse on both
stability and experimental cleanliness than A.

## Recommendation

**Option A (Poisson).** It is the smallest change that keeps Stage 2's
question intact (trained net with a zero-inflation-aware count likelihood vs
the GBM), it is the only candidate with *verified* full-condition numerical
stability, and it costs ≤1 gate eval. The equidispersion caveat is real but
lands on intervals, not the VN1 point score.

## What would change my mind

- If the pre-run stability gate fails on the exact config (unexpected, given
  identity-scaler survival), do NOT launch — return here; the count-
  likelihood-deep-net path may be unviable and Stage 2 pivots to a
  continuous loss on log1p(y) or reconsiders whether the GBM is the ceiling.
- If Poisson trains but pooled > naive (0.583): a trained deep net loses to
  simple methods here — a headline paper result (ROADMAP prediction 2
  refuted-so-far); bring the Stage 3 blend memo forward.
- If pooled beats the champion but lumpy fails R4 like gbm56/qmean: the
  bottleneck is deeper than the loss family — Stage 3 blend becomes the main
  event with three complementary ingredients.
- If it passes gate + R4: gate, promote on your confirm, then the covariate
  run (price/calendar/age) measures information value against the new
  champion.

## Registered predictions (to score at /reflect)

1. Trains NaN-free to completion on all 4 origins (stability, the thing that
   failed in Cycle 6).
2. Pooled VN1 0.52–0.62.
3. Lumpy bias magnitude < gbm56's |−10.5%| (the count-likelihood claim).
4. R4 outcome: explicitly uncertain (my R4 record is 0/2).
5. ROADMAP P2 direction: beats champion on smooth; not necessarily on lumpy.

**STOP — awaiting human approval, amendment, or rejection.**

---

## Outcome (2026-08-29): clean NEGATIVE result — the strongest paper finding

Approved + run `9ad032350c` (trained NaN-free to completion in ~75min — the
Cycle-6 crash retired). Pooled **0.6516** (champion 0.5622, naive 0.583) —
LOST decisively; no gate run (fails R1, budget stays 11/25). smooth 0.584
(+16.4%), lumpy 0.933 (+15.9%), Jan origin 0.897 (+25%). Predictions: (1) ✓
(2) MISSED high (3) REFUTED (4) moot (5) REFUTED. Mechanism (diagnostics):
NOT underfit (Spearman 0.867, matched dispersion) — the Poisson mean cannot
represent zero-inflation (never forecasts 0; 47.6 phantom units on
actual-zero series) → +7.1% over-forecast. Re-validates the champion's
Tweedie point-mass-at-zero. ROADMAP prediction 2 refuted-so-far. See
learnings.md Cycle 6-B. Next /propose: (a) Stage 3 blend (pre-registered
branch) or (b) Tweedie-in-the-net to isolate the zero-mass mechanism.
