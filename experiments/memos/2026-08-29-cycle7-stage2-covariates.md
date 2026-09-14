# Decision memo — Cycle 7: Stage 2 completion, what is the information worth?

Date: 2026-08-29 · Cycle 7 · Stage: **Stage 2 — one trained deep network,
done properly** (completes the stage; direction approved 2026-08-29)

## Context

- Champion `7548b6e864` (global_lgbm-24, VN1 0.5622). Budget 11/25.
- Reference points: gbm56 `d047d88915` 0.5429, Chronos-qmean `828187da8a`
  0.5396 (both un-promoted, R4-blocked on lumpy).
- Cycle 6-B (`9ad032350c`, PatchTST + Poisson, NO covariates) LOST: pooled
  0.6516, +7.1% over-forecast, NOT underfit (Spearman 0.87). The
  over-forecast is concentrated as **phantom demand on dead series** (47.6
  forecast units on 1497 series with zero horizon demand) → diagnosed
  driver is MISSING lifecycle/recency signal, which the champion GBM has as
  features (wsls, age, price). Concluding "deep nets lose" from a
  no-covariate net would be an unfair, weak claim — this cycle runs the
  fair test.

## Evidence (verification done pre-draft — the standing discipline)

- **PatchTST cannot take covariates**: EXOGENOUS_FUTR/HIST/STAT all False in
  neuralforecast 3.2.1 (univariate patching transformer). The ROADMAP's
  literal Stage-2 spec ("PatchTST ... price, calendar, product-age inputs")
  is infeasible as written → use **N-HiTS**, the ROADMAP-registered backup,
  which supports FUTR/HIST/STAT exog (verified).
- **N-HiTS + Poisson is numerically stable** (verified on a 3000-series
  spiky real subsample, no-cov AND with futr/hist exog: non-NaN, sane mean,
  full coverage). Poisson is the verified-stable count loss (NB overflowed
  cyc6; Tweedie's exp-link overflows to ~1e13, cyc6-B verification).
- Covariates already exist leak-safe in `src/features.py`: wsls
  (weeks-since-last-sale = the recency/lifecycle signal), age, price
  (ffill), price_chg4, woy, month, client, warehouse.

## Options

### Option A — N-HiTS + Poisson matched pair, covariates OFF vs ON (recommended)

**Hypothesis:** giving the trained net the lifecycle/price/calendar signal
the GBM has removes the phantom-demand over-forecast and materially closes
(or closes) the gap to the champion; the OFF→ON delta measures what that
information is worth (ROADMAP's registered Stage-2 question).

Two configs, run as ONE beam (`run_experiment cfg1 cfg2` — sequential in a
single process, no ledger race):
- `configs/nhits_poisson.yaml` — N-HiTS + Poisson, **no covariates**
  (architecture-matched baseline so the covariate delta is clean).
- `configs/nhits_poisson_cov.yaml` — identical + covariates:
  - futr_exog (deterministic-future): `woy`, `month`, `age`
  - hist_exog (past-only, from the input window): `price` (ffill),
    `price_chg4`, `wsls`
  - stat_exog (static): `client`, `warehouse`

Shared knobs pinned: input_size 64, max_steps 1500, scaler robust, lr 1e-3,
seed 7, Poisson, distribution-mean readout, cpu. One conceptual change
between the two runs: information OFF vs ON.

Implementation (experiment stage): a `GlobalNHITS` wrapper (registry
`nhits`) building the exog panel + `futr_df` under the frozen fit_predict
contract; contract test; **adversarial pre-flight review of the covariate
wrapper** (as PatchTST got) — covariate leak-safety (hist/futr correctness,
point-in-time construction) is the key risk the frozen harness cannot catch.
Plus the blocking pre-run stability gate on a spiky real subsample.

Expected impact: no-cov run ≈ 0.63–0.67 (architecture-matched to the
PatchTST-Poisson 0.6516); with-cov run **0.55–0.62** if lifecycle signal
lands, with phantom-demand-on-dead-series dropping well below 47.6.
P(with-cov beats champion 0.5622) ≈ 30%.
Cost: 2 deep runs, ~2.5h CPU total; ≤1 gate eval (gate only a pooled win).
Risk: covariates help but not enough to beat the tuned GBM (still a clean,
publishable info-value result); or N-HiTS underperforms PatchTST as a
backbone (the OFF baseline controls for this).

### Option B — single N-HiTS-with-cov vs champion — considered, rejected

Cheaper (1 run) but cannot answer "what is the information worth" (no
matched no-cov baseline) and confounds covariates with the PatchTST→N-HiTS
switch. The registered question needs the matched pair.

### Option C — TiDE/TSMixerx instead of N-HiTS — considered, rejected

Both support exog, but N-HiTS is the ROADMAP-registered backup; no evidence
to deviate. Park as an alternative only if N-HiTS is unstable at full scale.

## Recommendation

**Option A.** It completes Stage 2 honestly (the fair, covariate-fed test of
a trained net), answers the ROADMAP's registered information-value question
with a clean OFF/ON isolation, directly attacks the diagnosed phantom-demand
driver, and uses only verified-stable components. ≤1 gate eval.

## What would change my mind / forks

- with-cov beats champion + clears R4: gate, promote on your confirm — a
  trained deep net wins Stage 2; ROADMAP prediction 2 supported.
- with-cov beats no-cov substantially but still loses to champion: the
  information is worth X but the GBM's tabular handling still wins — a clean
  paper result; proceed to Stage 3 blend from strength.
- with-cov ≈ no-cov (covariates don't help): refutes the lifecycle-signal
  hypothesis; the deep net's loss to simple methods is architectural/loss-
  deep, not information — strong "deep learning underperforms here" finding;
  go to Stage 3 blend.
- phantom-demand-on-dead-series does NOT drop with covariates: revise the
  mechanism again (verify, don't guess).

## Registered predictions (to score at /reflect)

1. Both runs train NaN-free to completion (stability).
2. no-cov pooled 0.63–0.67 (architecture-matched to PatchTST-Poisson).
3. with-cov pooled 0.55–0.62; phantom demand on actual-zero series ≪ 47.6.
4. covariate delta (no-cov − with-cov VN1) ≥ 0.03 (information has value).
5. R4 outcome uncertain (my R4 record is 0/2); whether with-cov beats the
   champion is genuinely open.

**STOP — awaiting human approval, amendment, or rejection.**

---

## Outcome (2026-08-29): one big genuine result + one confounded arm

Approved + run as a beam (both NaN-free; ~7min each — N-HiTS is fast).
No gate (both fail R1), budget 11/25.

- `eb69d1a3bc` no-cov: **0.5677** (bias +3.6%) — nearly matches champion
  0.5622 and CRUSHES PatchTST-Poisson 0.6516 (same loss). **Big finding:
  architecture dominated the Cycle-6-B "deep nets lose" story — that was a
  PatchTST artifact, now RETRACTED; N-HiTS is competitive; ROADMAP
  prediction 2 back to OPEN.**
- `e8987a9d2b` with 8 covariates: 0.9956 (bias +26.5%), uniform
  over-forecast — **CONFOUNDED, not a valid info-value result.** Subsample
  ablation: each covariate group alone is fine (≤1.13; hist recency/price
  group best at 1.000), only the full set destabilizes the optimization
  (all→1.262, reproducing the run). The matched-pair's treatment arm was
  invalidated by an optimization instability, not by information having
  negative value.
- Predictions: #1 ✓; #2 MISSED (better); #3/#4 confounded; #5 no promotion.
- Methodological debt: run_id doesn't hash model code (mitigated via deep.py
  sha1 in notes). Pre-run stability gate needs a bias-ratio check (it caught
  NaN/coverage but not the calibration blowup).
- Next /propose fork: (a) corrected covariate run — HIST recency/price
  subset only (mechanism-driven by the diagnosis, ablation-stable); or
  (b) Stage 3 blend now (N-HiTS-no-cov is a strong complementary
  ingredient). See learnings.md Cycle 7.
