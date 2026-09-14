# Decision memo — Cycle 8: corrected covariate test, recency/price signal

Date: 2026-08-30 · Cycle 8 · Stage: **Stage 2 — one trained deep network**
(completes the info-value question Cycle 7's confounded arm left open)

## Context

- Champion `7548b6e864` (0.5622). Budget 11/25.
- Cycle 7: N-HiTS-Poisson **no-cov** `eb69d1a3bc` **0.5677** (bias +3.6%) —
  competitive with the champion, ≫ PatchTST. Its **with-8-covariates** arm
  `e8987a9d2b` (0.9956) was CONFOUNDED — an optimization instability from
  the full covariate set, NOT evidence information hurts (ablation: each
  group ≤1.13, all-8 →1.262; the HIST recency/price group alone →1.000).
- The ROADMAP's registered "what is the information worth" question is
  therefore still unanswered cleanly.

## Evidence

- Cycle 6-B diagnosis: the deep net's over-forecast is phantom demand on
  DEAD series (predicting historical rate after a series goes dead) →
  missing RECENCY/lifecycle signal. The recency signal is `wsls`
  (weeks-since-last-sale) + `price` (a sale only prints a price). This is
  ex-ante from the diagnosis, not from the ablation.
- Subsample ablation (Cycle 7): +hist(price, price_chg4, wsls) ALONE gives
  bias-ratio **1.000** (perfectly de-biasing) and is stable; calendar
  (futr) and client/warehouse (stat) each slightly worsen and, combined
  with hist, destabilize. So the mechanism-targeted set is HIST-only.
- Standing prior: recency ≫ seasonality here — so dropping the calendar
  (futr) covariates is principled, not just convenient.

## Options

### Option A — N-HiTS + Poisson + HIST recency/price covariates only (recommended)

**Hypothesis:** giving the net the recency/price signal (wsls, price,
price_chg4) — and ONLY that — removes the phantom-demand over-forecast and
improves pooled VN1 below the no-cov 0.5677, plausibly past the champion.

Config diff: new `configs/nhits_poisson_recency.yaml` — identical to
`nhits_poisson.yaml` (the no-cov baseline) except
`hist_exog: [price, price_chg4, wsls]` (futr_exog and stat_exog stay empty).
One change vs `eb69d1a3bc`: the recency/price information ON. Zero new code
(wrapper already supports hist_exog).

Comparison: vs `eb69d1a3bc` (0.5677) = clean covariate-value delta
(everything else identical); vs champion 0.5622 = can a recency-fed deep net
win. Mechanism check: phantom demand on actual-zero series drops from the
PatchTST 47.6 baseline / no-cov level; pooled bias moves toward the
ablation's ~1.0.

**Enhanced pre-run stability gate (Cycle-7 lesson):** the exact config is
smoke-fit on a spiky real subsample at ≥300 steps and must be (i) NaN-free,
(ii) full coverage, AND (iii) bias-ratio ΣF/ΣA in [0.85, 1.20]. The Cycle-7
gate checked only (i)/(ii) and missed the calibration blowup — (iii) is the
new guard. Do NOT launch the full run if (iii) fails.

Expected impact: pooled **0.53–0.57** (at/below no-cov, possibly past the
champion), bias within ±8%, phantom-demand-on-dead-series materially below
no-cov. P(beats champion pooled) ≈ 35%; P(also clears R4) ≈ 20%.
Cost: 1 N-HiTS run (~7–10 min); ≤1 gate eval (gate only on a pooled win).
Risk: hist covariates destabilize at 1500 steps / full scale despite the
150-step ablation (the enhanced gate catches this before spend); or they
de-bias but don't beat the tuned GBM (still a clean, publishable
info-value result).

Reproducibility note: the delta is vs the existing `eb69d1a3bc`, so it
carries torch run-to-run jitter. If the delta lands within jitter-scale
(|Δ| < ~0.01) and is decision-relevant, follow with a jitter-controlled
fresh pair; otherwise one run settles it.

### Option B — jitter-controlled fresh pair (no-cov + recency) — considered

Re-runs no-cov alongside recency in one session for a jitter-free delta.
Cleaner but re-runs a known result (near-duplicate ledger row) at 2× cost.
Reserve for the case above (ambiguous delta). Not the default.

### Option C — Stage 3 blend now — considered, deferred

N-HiTS-no-cov is a strong complementary ingredient, but leaving Stage 2's
info-value question confounded is a paper hole; answer it first (cheap),
then blend.

## Recommendation

**Option A.** One clean, mechanism-driven change (the diagnosed recency
signal), one fast run, the enhanced stability gate as the safeguard, and it
closes the ROADMAP's registered Stage-2 question that Cycle 7 left open.

## What would change my mind / forks

- Enhanced gate fails (iii): do NOT launch; the deep net can't use these
  covariates stably at this config → report the info-value question as
  "not cleanly answerable without a stabilized config" and go to Stage 3.
- Beats champion + clears R4: gate, promote on your confirm — a trained
  deep net wins Stage 2; ROADMAP prediction 2 supported.
- De-biases (bias→~0, phantom demand drops) but pooled ≥ champion: the
  information is worth X, GBM still wins — clean paper result; go to Stage 3
  blend from strength.
- No improvement over no-cov: the recency signal doesn't help the net (it's
  already implicit in the 64-week window) → info-value ≈ 0 for this net;
  go to Stage 3 blend.

## Registered predictions (to score at /reflect)

1. Trains NaN-free AND passes the enhanced gate (bias-ratio in range).
2. Pooled 0.53–0.57 (≤ no-cov 0.5677).
3. Phantom demand on actual-zero series drops materially vs no-cov.
4. Covariate delta (no-cov − recency VN1) in [−0.01, +0.06] (recency helps,
   modestly), reported relative to torch jitter.
5. R4 / beats-champion genuinely uncertain.

**STOP — awaiting human approval, amendment, or rejection.**

---

## Outcome (2026-08-30): pre-run gate FAILED — not launched (pre-registered)

Approved. The enhanced pre-run stability gate (bias-ratio guard) FAILED:
recency config bias-ratio 1.378 at 300 steps (need 0.85–1.20). Trajectory
confirms real destabilization, not noise:

| steps | no-cov | recency |
|---|---|---|
| 150 | 1.075 | 1.000 |
| 300 | 1.066 | 1.324 |
| 600 | 1.052 | 1.301 |
| 1000 | 1.055 | **1.522** |

no-cov is stable across budgets; recency destabilizes as training proceeds
(the 150-step 1.000 was an undertrained transient). Per the pre-registered
fork, the full run was NOT launched — no ledger row, no gate eval, budget
11/25. Prediction #1 (passes enhanced gate) REFUTED at the gate; the info-
value question is NOT cleanly answerable for this deep net without a
stabilization sub-project (exog scaling / lower lr / gradient clip = config
tuning = out of scope / drift). **The enhanced bias-ratio gate — added from
the Cycle-7 lesson — just prevented a wasted run and gave a decisive
answer.** Recommendation: go to Stage 3 blend (N-HiTS-no-cov `eb69d1a3bc`
0.5677 is a strong, stable, complementary ingredient).
