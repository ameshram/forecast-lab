# Decision memo — Cycle 6: Stage 2 opener, one trained deep network

Date: 2026-08-29 · Cycle 6 · Stage: **Stage 2 — one trained deep network,
done properly** (ROADMAP; direction approved by human 2026-08-29)

## Context

- Champion: `7548b6e864` (global_lgbm-24, VN1 0.5622). Gate budget 11/25.
- Knowledge reference points Stage 2 must be reported against (all
  un-promoted, both R4-blocked on lumpy): gbm56 `d047d88915` 0.5429,
  Chronos-qmean `828187da8a` 0.5396.
- The promotion bar, sharpened by cycles 4–5: **beat 0.5622 pooled while
  holding lumpy segment VN1 ≤ 0.7159** (champion 0.6818 × 1.05). Lumpy has
  blocked over-forecast (+15.2%) and under-forecast (−10.5%) challengers
  alike; the open mechanism is a conditional distribution wide enough for
  lumpy demand that its point readout stays segment-bias-neutral.
- ROADMAP registered prediction 2 to be tested: "the trained network beats
  the GBM on busy/seasonal products, not on erratic ones."

## Evidence

- 71.8% of Phase-0 cells are zero (data profile, retrospective §1) — the
  loss function, not capacity, is the registered bet.
- Chronos' quantile distribution + trimmed-mean readout nearly promoted
  (0.5396, 4/4 origins) — distributional point readouts work here when the
  distribution is right; Chronos' one global distribution family was too
  coarse for lumpy (`828187da8a`, `46df0bba04`).
- Full-cycle training window is confirmed necessary (`d047d88915`): the
  deep model gets input windows ≥ 52 weeks from day one.
- Calibration self-note: my memo predictions under-call seasonal-exposure
  effects (cycles 3 & 5) and were wrong on both R4 calls — treat the R4
  prediction below as a coin flip, not a forecast.

## Options

### Option A — PatchTST, global, negative-binomial loss, NO covariates (recommended run 1)

**Hypothesis:** a trained global transformer with an explicitly
zero-inflation-appropriate likelihood (negative binomial) learns per-series
conditional distributions whose mean readout is accurate AND
segment-bias-neutral on lumpy — the combination no zero-shot readout or
GBM window achieved.

Implementation: wrap **neuralforecast** (Nixtla) `PatchTST` with
`DistributionLoss("NegativeBinomial")` behind the standard
`fit_predict(train, future_ds)` contract (`src/models/deep.py`, registry
key `patchtst`), retraining from scratch inside fit_predict per origin —
same leak-free pattern as `global_lgbm`. Torch already installed;
neuralforecast added to requirements (version-pinned). Contract test on a
small panel. Point forecast = distribution mean. Seeded; config records
the seed (residual torch nondeterminism noted in the run's notes).

Config (`configs/patchtst_nb.yaml`) — all knobs fixed NOW to prevent
shopping: input_size 64, horizon 13 direct, patch_len 8, stride 8,
max_steps 1500, batch 256, lr default, scaler robust, seed 7. No tuning
sweep; any hyperparameter change is a new memo.

Expected impact (wide on purpose): pooled **0.52–0.60**; lumpy VN1
0.64–0.75 with lumpy bias in ±8% if the NB mechanism works. P(beats
champion pooled) ≈ 40%; P(also clears R4) ≈ 20–25%.
Cost: ~20–60 min/origin on the M1 Max (MPS, CPU fallback) → 1.5–4 h; ≤1
gate eval (gate only on a pooled win).
Risk / failure modes: MPS training instability; NB dispersion mis-fit on
lumpy's heavy tail; library API quirks. Timebox: if training won't
converge after ~2 h of setup effort, fall back to Option C.

### Option B — Option A + covariates (price, calendar, product age) — pre-authorized run 2, conditional

Same architecture and loss with exogenous features — isolates ROADMAP's
"what is that information worth" question as a one-change diff.
**Runs without a new approval ONLY if run 1 pooled ≤ 0.58**; otherwise the
family needs rethinking before more spend. Expected: −0.005 to −0.03 vs
run 1 if price/calendar carry signal. ≤1 gate eval.

### Option C — N-HiTS, same NB loss — pre-registered fallback only

Runs instead of (not in addition to) PatchTST if Option A cannot train
stably within its timebox. Same config discipline. Not a third parallel
variant — this is the roadmap's named backup.

## Recommendation

**Option A now, B conditionally pre-authorized, C as fallback.** One new
mechanism (the likelihood) enters first with everything else held minimal;
covariates enter second as their own one-change experiment. This caps
Stage 2's opener at ≤2 runs and ≤2 gate evals (worst case 13/25).

Ride-along engineering (approved direction, no model claims, no budget):
persist per-run forecast frames in `record_run` going forward; pin HF
checkpoint revisions in FM configs; regenerate the stale dashboard.

## What would change my mind

- Run 1 pooled worse than naive (>0.583): the family is failing on this
  data — stop Stage 2 spending, score ROADMAP prediction 2 as refuted-so-
  far, and bring the Stage 3 blend memo forward with existing ingredients.
- Run 1 beats pooled but fails R4 on lumpy like everyone else: the
  bottleneck is beyond the loss family — same consequence: Stage 3 blend
  becomes the main event, now with three complementary ingredients.
- Run 1 passes everything: gate, promote on your confirm, then Option B
  measures information value against the NEW champion.

## Registered predictions (to score at /reflect)

1. Pooled 0.52–0.60.
2. ROADMAP P2 direction: beats champion on smooth; does NOT beat qmean's
   0.3717 on smooth zero-shot bar.
3. Lumpy bias magnitude < gbm56's |−10.5%| (the NB mechanism's core claim).
4. R4 outcome: explicitly uncertain (my R4 record is 0/2).

**STOP — awaiting human approval, amendment, or rejection.**

---

## Pre-run reconciliation (2026-08-29, after approval + adversarial review)

Approved and built. A 5-lens read-only pre-flight review (leakage /
contract / API-semantics / config / repro) returned **safe_to_run, zero
blockers**; two LOW traceability findings addressed before launch:

- The "lr default" wording above resolves to **learning_rate = 1e-3** in
  the ledgered config (neuralforecast's tutorial LR paired with the
  reduced max_steps=1500; the library constructor default is 1e-4, tuned
  for its 5000-step default). Untuned, not shopped; recorded here so the
  ledgered LR traces unambiguously. hidden_size=128, n_heads=16,
  windows_batch_size=1024 equal library defaults (no divergence).
- **num_samples=1000** (MC draws behind the mean readout) and the env
  (**neuralforecast==3.2.1, torch==2.13.0**) are now pinned in the config
  and run notes, so the point readout is reproducible from config alone.

Point-column semantics verified empirically (mean strictly > median on
92-99% of series across 40/400/1000 steps) — not assumed.

## Outcome (2026-08-29): ENGINEERING FAIL — no ledger row, question OPEN

Run crashed ~15min into training: `ValueError: NegativeBinomial probs ...
HalfOpenInterval(0,1)`. No score produced; registered predictions NOT
evaluable; ledger untouched (last row `d047d88915`), budget still 11/25.
Root cause (diagnostics, scratchpad): NB **forward-pass numerical overflow**
from large-magnitude count inputs (y up to 13,669) → `probs`→1.0/NaN on the
first loss eval. identity scaler reproduces deterministically; robust/
standard/lr=1e-4 survive a spiky 789-series stress subsample but robust
crashed at full 15k scale; gradient clipping does NOT help (pre-gradient
failure). My initial IQR=0 guess was falsified by the scaler source. See
learnings.md Cycle 6. Next /propose decides the fix (lean: Poisson/Tweedie
loss — one clean change, keeps the trained-net-with-count-likelihood test).
