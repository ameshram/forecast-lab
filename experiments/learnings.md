# Learnings ledger

Terse, append-only. One line of durable signal per entry; calibration
(predicted vs realized) tracked for every memo-backed run.

## Cycle 0 (baselines — no predictions to score)

- `e663fe200e` seasonal_naive: VN1 1.021 vs naive 0.583. **Lesson:** recency
  beats raw seasonality here; lifecycle churn makes the year-ago week a bad
  reference. (Prior recorded in CLAUDE.md.)
- `13811e908e` MA13+dead26 == MA13 exactly. **Lesson:** dead-series zeroing
  is a no-op whenever `zero_dead_weeks >= window` — the window already does
  it. DEAD-END for that config family.
- All baselines ≈2.0 on "insufficient" (3.9% of volume). **Lesson:** sparse
  tail is structurally hard and low-value; do not chase without a
  cold-start-specific mechanism. (Prior recorded.)

## Cycle 1 (beam: router + seasonal uplift) — both FAIL, both informative

- `7a52b16fe3` router smooth→MA13 · **predicted** pooled ~0.545 ·
  **realized** 0.5853 (champion 0.5830) · prediction error LARGE (predicted
  −0.038, realized +0.002). The segment mechanism worked exactly as
  predicted (smooth 0.511→0.388, pooled WAPE 0.531→0.513) but pooled VN1
  worsened: smooth's +9.1% over-forecast had been *offsetting* the
  portfolio's net under-forecast; removing it deepened net bias −5.2%→−7.3%
  and VN1's global bias term consumed the WAPE gain. **Lesson (new prior):
  VN1 is not segment-separable — score a challenger on WAPE and net-bias
  movement jointly; never sum segment improvements.** Router idea stays
  alive but needs a bias-rebalancing companion.
- `accafab14a` seasonal uplift naive · **predicted** pooled 0.56–0.57 with
  Oct-origin bias fix · **realized** 0.6956, bias −5.2%→−15.3%, and the
  target Oct-2022 origin barely moved (0.852→0.840, bias −21.9%→−20.7%).
  Mechanism refuted: client-level week-of-year totals conflate lifecycle
  churn/growth with calendar seasonality, so the re-anchoring ratios ran
  systematically below 1. `DEAD-END`: portfolio/client-level multiplicative
  woy re-anchoring. The holiday blind spot remains OPEN — next attempt
  should estimate seasonality churn-robustly (same-series y-o-y on
  surviving series, or woy as a feature inside a demand model, e.g. GBM).


## Cycle 2 (beam: bias-scaled router + global LightGBM)

- `2c050a9d24` bias_scaled(router) · **predicted** 0.555–0.575 · **realized**
  0.5642 — calibration GOOD. Pooled bias −7.3%→−2.6%; but the trailing ratio
  scaled UP into the post-holiday January origin (bias +29%, VN1 0.94 vs
  champion 0.70) — the predicted lag-at-the-turns failure, confirmed. Gate
  FAIL on R2 (2/4 origins). **Lesson:** trailing debias helps steady
  regimes, is actively harmful across seasonal turns.
- `7548b6e864` global_lgbm v1 · **predicted** 0.48–0.54 · **realized**
  0.5622 — over-optimistic (no leakage flag; error on the high side). Gate
  **PASS** (R1–R4: 3/4 origins, p=0.03, no segment regressions, bias −0.8%).
  Weakness shared with every model so far: the January origin (+25% bias) —
  with n_anchors=24 the training rows span only ~6 recent months, so the
  model has NEVER SEEN a post-holiday decline in training. **Next lever:
  anchors ≥ 52 weeks so a full seasonal cycle is in the training matrix.**

## Cycle 3 (Stage 1 beam: zero-shot foundation models) — no gates run
(pooled losses obvious from ledger; budget preserved at 10/25)

- `4248b74828` chronos-bolt-base zero-shot · **predicted** pooled
  0.58–0.68, smooth 0.50–0.56, bias −5..−18% · **realized** 0.5878 /
  0.3717 / −7.4% — pooled and bias calibration GOOD; smooth beat the
  prediction by ~30% (pretraining's fit to steady series under-called).
  Won the January origin outright (0.599 vs champion 0.892). **Lesson:
  zero-shot FM weakness is bias, not accuracy — its WAPE 0.5134 beats the
  champion's 0.5542.**
- `e8a11552ec` timesfm-2.5-200m zero-shot · **predicted** pooled 0.58–0.70
  AND less intermittent bias than chronos (point ≈ mean) · **realized**
  0.6512 ✓ but intermittent bias −23.2% vs chronos −6.6% — mechanism
  prediction **REFUTED**: TimesFM's point head is not mean-calibrated on
  zero-inflated series; Chronos-Bolt's median is the better-calibrated
  zero-shot output here. Best pooled WAPE in the ledger (0.4612) sunk by
  −19% bias (lumpy −31%). Best smooth ever (0.3381), best January origin
  (0.5735).
- ROADMAP registered prediction 1 ("FMs worse pooled, surprisingly close
  on smooth"): direction confirmed — but under-called; both FMs are
  *outright best* on smooth, beating MA13's 0.3884. Both demolish the
  champion at the seasonal turn its 24-anchor window never saw —
  independent confirmation of the anchors≥52 diagnosis. Stage 3 signal:
  champion and FMs fail in disjoint places (turns vs intermittent bias) —
  the complementarity a blend needs. Registered next lever for FMs:
  point='mean' / higher quantile is a pure config diff (`configs/`
  chronos `point` param) targeting the bias term only.

## Cycle 4 (beam: Chronos readout family — mean vs q65) · gate 11/25

- `524453e7d8` point=mean · **no-op duplicate** of `4248b74828`: the
  chronos package returns the *median* as "mean" for Bolt checkpoints
  (source NOTE in `chronos_bolt.predict_quantiles`). **Lesson: verify a
  third-party readout's semantics in source before betting a run on it;
  results identical to 4 decimals across 15k series = plumbing, not
  statistics.** Corrected as `point=qmean` (average of 9 trained
  quantiles) with a differs-from-median contract test.
- `46df0bba04` quantile 0.65 · **predicted** 0.53–0.59, bias −3..+3 ·
  **realized** 0.7223, bias +13.2% — prediction error LARGE, mechanism
  refuted: a uniform above-median quantile shift over-corrects everywhere
  (lumpy +29%). `DEAD-END`: fixed above-median quantile readout on this
  portfolio (anti-shopping fence from the memo also bars further variants).
- `828187da8a` qmean · **predicted** 0.53–0.58, bias −2..+4% · **realized**
  0.5396, bias −0.3% — calibration EXCELLENT. First run to beat the
  champion pooled AND at all 4 origins (Jan 0.730 vs 0.892). Gate **FAIL
  on R4 only** (lumpy 0.9254 vs 0.6818; lumpy bias +15.2%; R1–R3 passed,
  p=0.04). **Lesson: the trimmed mean funds its neutral pooled bias by
  over-forecasting lumpy against under-forecast elsewhere — offsetting
  bias inside a single readout. Router arithmetic (not a run): lumpy→
  median lands ≈0.562 = champion level; the naive fix deletes the offset
  it needs. Lumpy is now THE blocking segment for a zero-shot champion.**

## Cycle 5 (rival fix, OFF-ROADMAP approved: gbm anchors 24→56) · 11/25

- `d047d88915` gbm56 · **predicted** pooled 0.53–0.56, Jan 0.65–0.75 with
  bias <+12%, no R4 catastrophe · **realized** 0.5429, Jan **0.497** /
  bias **−1.6%**, 4/4 origins — pooled ✓, Jan mechanism delivered ~2× the
  predicted improvement (repeat of the cycle-3 smooth miss: I under-call
  how much seasonal exposure is worth), but **"no R4 catastrophe" was
  WRONG**: lumpy 0.7232 > 0.7159 (champion×1.05, ratio 1.061). Human
  skipped the formal gate (verdict arithmetically determined; budget
  stays 11/25); champion remains `7548b6e864`; `d047d88915` is the
  strongest GBM config, un-promoted. **Lessons:** (1) Q1 CONFIRMED — the
  champion's January hole was a training-window artifact, not a GBM
  limit; anchors≥52 lever validated and closed. (2) The full-cycle window
  traded lumpy over-forecast for lumpy under-forecast: lumpy WAPE
  improved 0.656→0.618 while lumpy bias went −2.6%→−10.5%, and segment
  VN1 charges it. (3) **Lumpy has now R4-blocked two mechanistically
  opposite challengers** (qmean +15% over, gbm56 −11% under) against the
  champion's best-in-ledger 0.6818 — segment-VN1-neutral lumpy forecasts,
  not pooled wins, are the promotion bottleneck. Do NOT loosen R4 in
  response (post-hoc rule-shopping); make Stage 2's loss target it.

## Cycle 6 (Stage 2 opener: PatchTST + NegativeBinomial) — ENGINEERING FAIL,
no ledger row, budget untouched at 11/25, scientific question still OPEN

- `configs/patchtst_nb.yaml` (global PatchTST, NB DistributionLoss,
  distribution-mean readout) **crashed ~15min in during training**:
  `ValueError: NegativeBinomial probs ... constraint HalfOpenInterval(0,1)`
  — no run recorded. Predictions (pooled 0.52-0.60, lumpy bias <|10.5%|)
  **NOT EVALUABLE**: the run produced no score. ROADMAP prediction 2 stays
  untested; Stage 2's question (does a trained NB net beat the champion,
  esp. on lumpy) is OPEN.
- Root cause CONFIRMED by diagnostics (scratchpad), not guessed — and my
  first guess (robust-scaler IQR=0 blowup) was FALSIFIED by reading the
  scaler source (MAD with std fallback + zero-protection). Actual cause:
  **NB forward-pass numerical overflow** — large-magnitude count inputs
  (real y up to 13,669) drive the network's `probs` output to 1.0/NaN on
  the FIRST loss eval, violating NB's half-open support. Evidence: identity
  scaler reproduces the crash DETERMINISTICALLY at step 0; robust/standard
  scalers and lr=1e-4 survive a 789-series spiky stress subsample;
  **gradient clipping does NOT help** (crash is pre-gradient, in the
  forward/loss). robust+lr=1e-3 (the real config) survives the subsample
  but crashed at full 15k scale on a pathological batch → robust scaling
  mitigates but is NOT reliably stable; lr is a red herring for a
  forward-pass failure.
- **Lessons:** (1) A deep count-likelihood (NB) on this heavy-tailed,
  72%-zero data needs forward-pass numerical-stability handling; input
  scaling alone is insufficient at full scale. (2) Verify a hypothesized
  failure mechanism by reproduction before recording it — the IQR guess was
  wrong. (3) The adversarial pre-flight review (config lens) FLAGGED
  divergence/NaN risk; it was noted non-blocking and hit — future deep-model
  memos must treat NB numerical stability as a blocking pre-req, smoke-tested
  on a spiky REAL subsample (the small synthetic panel used small Poisson
  values and never triggered it). (4) Engineering asset, not a scientific
  result: the NB hypothesis is neither supported nor refuted.
- Decision fork (for next /propose, NOT auto-run): (i) safer count loss —
  Poisson (single-param, no probs-saturation failure) or Tweedie (what the
  winning GBM used), same zero-inflation spirit, one clean change; (ii)
  memo's registered fallback N-HiTS + NB; (iii) standard-scaler retry (weak
  — robust already failed at scale). Lean (i).

## Cycle 6-B (Stage 2: PatchTST + Poisson) `9ad032350c` — clean NEGATIVE
result, no gate (fails R1), budget 11/25 — the strongest paper finding yet

- **predicted**: (1) trains NaN-free (2) pooled 0.52-0.62 (3) lumpy bias
  <|10.5%| (4) R4 uncertain (5) beats champion on smooth. **realized**:
  pooled **0.6516** / WAPE 0.5808 / bias **+7.1%**; smooth 0.584 (+16.4%),
  lumpy **0.933** (+15.9%), erratic 0.760 (+14.6%), intermittent 0.547;
  origins Oct 0.753 / Jan 0.897(+25%) / Apr 0.589 / Jul 0.571.
- Score: (1) **CONFIRMED** — Poisson stability fix held, trained NaN-free
  to completion (the Cycle-6 failure retired). (2) **MISSED** — 0.6516
  above the range; I was over-optimistic AGAIN on the fancy model (GBM
  cyc2 predicted 0.48-0.54→0.5622; here 0.52-0.62→0.6516) — recurring
  calibration error: I under-price how hard this data is for the trained
  net. (3) **REFUTED** — lumpy bias +15.9% (over, larger than gbm56's
  −10.5%). (4) moot — fails R1 (0.6516 ≫ 0.5622), no gate spent.
  (5) **REFUTED** — smooth 0.584 > champion 0.522; the WORST serious model
  on smooth (FMs were best at 0.37).
- Decisive mechanism (diagnostics, not guessed): the net is **NOT
  underfit** — Spearman(yhat_sum,y_sum)=0.867, forecast CV 6.02 ≈ actual
  CV 6.15, so it tracks per-series level with matched dispersion. It
  **never forecasts zero** (0% of series yhat=0 vs 9.9% actual-zero) and
  puts mean 47.6 phantom units on truly-zero series → +7.1% global
  over-forecast. **The mean of a plain count likelihood (Poisson) cannot
  represent the zero-inflation dominating 72% of the data; more training
  steps will NOT fix a structural readout bias.**
- **BIG durable finding (prior added):** this re-validates the champion's
  **Tweedie** objective — Tweedie (1<p<2) has a point mass at EXACT zero
  that a Poisson/NB mean lacks. The winning lever is the loss family's
  zero-representation, NOT the architecture (PatchTST) or training budget.
  ROADMAP prediction 2 **refuted-so-far**: a trained deep count-net loses
  decisively to both the tuned GBM and naive — and not from underfitting.
  A genuine paper section answering the North Star with mechanism.
- Decision fork (next /propose, NOT auto-run): (a) accept Stage 2's answer
  and bring the **Stage 3 blend** forward (the pre-registered "pooled >
  naive" branch); OR (b) ONE more Stage-2 test isolating the mechanism —
  **Tweedie-in-the-net** (needs the `rho` kwarg; tests "is it Tweedie's
  zero-mass that wins, independent of GBM-vs-transformer?"). Do NOT tune
  steps/lr (evidence says not underfit — that would be drift/shopping).
- **UPDATE (verification killed option (b)):** pre-draft checks on
  Tweedie-in-the-net found (1) its exp(log_mu) MEAN OVERFLOWS to ~1e13
  under the robust scaler; only standard-scaler + lr=1e-4 + gradient-clip
  was stable on the 3723-series stress test — a 3-change, fragile,
  confounded config, not the clean one-change I pitched; and (2) a
  reasoning error in my own prior: reading the MEAN, Tweedie ≈ Poisson
  (both give the positive conditional mean ≈ sample mean); a zero-inflated
  loss does not pull the MEAN readout to zero. So (b) is unlikely to fix
  the over-forecast and I revised the CLAUDE.md prior. The +7.1%
  over-forecast is concentrated as phantom demand on DEAD series (47.6
  units on 1497 actual-zero series) → most likely MISSING lifecycle/recency
  signal (no covariates), not the loss family. **Revised recommendation:
  (c) PatchTST-Poisson + covariates (price, calendar, product-age)** — the
  ROADMAP-registered Stage 2 "what is that information worth" run, which
  directly attacks the diagnosed driver and is numerically stable (Poisson
  was clean). Then (a) the Stage 3 blend. (b) parked as fragile+low-value.

## Cycle 7 (Stage 2: N-HiTS+Poisson matched pair, covariates OFF/ON) —
one BIG genuine result + one confounded arm. No gate (both fail R1), 11/25.
(PatchTST can't take exog → switched to N-HiTS, the ROADMAP backup.)

- `eb69d1a3bc` N-HiTS-Poisson **no-cov**: pooled **0.5677** (bias +3.6%),
  origins 0.70/0.70/0.53/0.54, smooth 0.483 / intermittent 0.510 / lumpy
  0.777. **predicted 0.63–0.67 → MISSED (better).** BIG finding: N-HiTS
  no-cov nearly MATCHES the champion (0.5622) and CRUSHES PatchTST-Poisson
  (0.6516) — SAME loss, same data, just a better backbone. **Cycle 6-B's
  "trained deep nets lose here" was substantially a PatchTST artifact →
  RETRACTED; N-HiTS is competitive; ROADMAP prediction 2 back to OPEN.**
- `e8987a9d2b` N-HiTS-Poisson **with 8 covariates**: pooled **0.9956**
  (bias +26.5%), uniform over-forecast across ALL segments (smooth +23%,
  intermittent +31%, lumpy +47%). predicted 0.55–0.62 → far MISSED, BUT
  **CONFOUNDED — not a valid info-value result.** Subsample ablation (150
  steps): no-cov bias-ratio 1.075, +futr 1.106, **+hist(price,chg4,wsls)
  1.000 (best)**, +stat 1.127, **+ALL 1.262** (reproduces the full run).
  Each covariate group alone is fine; the FULL 8-feature set destabilizes
  the optimization at lr=1e-3/1500 → a systematic level shift, not
  "information hurts." The recency/price HIST group ALONE is perfectly
  de-biasing (1.000).
- Prediction scoring: #1 both train NaN-free ✓; #2 no-cov MISSED (better);
  #3 with-cov MISSED (confounded); #4 covariate delta ≥0.03 — realized
  −0.428 but INVALID (optimization artifact); #5 no promotion (no-cov
  0.5677 > champion 0.5622; with-cov terrible). No gate spent.
- **Lessons:** (1) architecture ≫ expected on short intermittent series —
  never conclude "deep nets lose" from one backbone. (2) Many covariates at
  an untuned config can destabilize a deep net UNIFORMLY; ablate before
  concluding info-value; the stability gate (NaN/coverage) does NOT catch a
  calibration blowup — add a bias-ratio check to future deep pre-run gates.
  (3) The diagnosed lifecycle/recency signal (hist: price, price_chg4,
  wsls) is real and de-biasing on its own.
- Methodological debt logged: run_id hashes the config, NOT the model code
  in deep.py; mitigated for cyc7 by deep.py sha1 in notes; proper fix
  (record code/git hash in record_run) is a future research-eng task.
- Decision fork (next /propose, NOT auto-run): (a) **corrected covariate
  run** — N-HiTS-Poisson with the HIST recency/price subset only
  (price, price_chg4, wsls), mechanism-driven by the Cycle-6-B diagnosis
  (ablation-confirmed stable + de-biasing), one clean change from no-cov;
  OR (b) **Stage 3 blend** now (champion GBM + N-HiTS-no-cov / qmean),
  since N-HiTS-no-cov is a strong, complementary ingredient. Do NOT re-run
  the full 8-covariate set as-is (confounded), and do NOT tune lr/steps to
  rescue it (drift).

## Cycle 8 (Stage 2: N-HiTS+Poisson + recency/price covariates only) —
pre-run gate FAILED, NOT launched, no ledger row, budget 11/25

- `configs/nhits_poisson_recency.yaml` (hist_exog: price, price_chg4, wsls).
  Enhanced pre-run gate (NaN + coverage + BIAS-RATIO in [0.85,1.20], the new
  Cycle-7-lesson guard) **FAILED**: bias-ratio 1.378 @ 300 steps. Trajectory
  (no-cov vs recency): 150→(1.075,1.000), 300→(1.066,1.324), 600→(1.052,
  1.301), 1000→(1.055,**1.522**). no-cov STABLE across budgets; recency
  DESTABILIZES with training — **the Cycle-7 ablation's 150-step 1.000 was
  an undertrained transient (this RETRACTS Cycle-7 lesson (3) that the hist
  recency signal is "de-biasing on its own").** Pred #1 REFUTED at the gate;
  not launched per the pre-registered fork.
- **Lessons:** (1) even the mechanism-targeted recency subset can't be
  stably integrated into this deep net at the principled untuned config →
  the deep-net COVARIATE PATH is unstable here, not just the full 8-set;
  info-value NOT cleanly answerable without a stabilization sub-project
  (exog scaling / lower lr / grad-clip = tuning = drift, out of scope).
  (2) Low-step ablations MISLEAD — measure calibration at a realistic
  budget (150-step said 1.000; truth is 1.5+). (3) The enhanced bias-ratio
  pre-run gate (added from Cycle 7) prevented a wasted run and gave a
  decisive answer — keep it for all deep runs.
- **Stage 2 conclusion:** the best trained deep net here is N-HiTS-no-cov
  (`eb69d1a3bc` 0.5677, competitive with the champion 0.5622); covariates
  don't integrate stably; the loss (Poisson stable, NB/Tweedie fragile) and
  architecture (N-HiTS ≫ PatchTST) story is understood. → Stage 3 blend
  (pre-registered), with N-HiTS-no-cov as a strong, stable ingredient.

## Cycle 9 (Stage 3: the blend) `1bb3cb32c6` — gate PASS, PROMOTED, budget 14/25
(2 evals used this cycle: informational + promote; a 3rd log line is a benign
double-execution of the promote command comparing the new champion to itself.)

- `1bb3cb32c6` blend = 50/50 GBM-56 (`d047d88915`) + Chronos-qmean
  (`828187da8a`) · **predicted** (1) NaN-free w/ OMP mitigation (2) pooled
  0.50–0.55 (3) lumpy bias ≪ both components & lumpy VN1 below both
  (4) clears R4 (5) likeliest promotion · **realized** pooled **0.5192** /
  WAPE 0.4933 / bias −2.6%; origins 0.716/**0.564**/0.459/0.467 = **4/4**;
  segments smooth 0.410 / interm 0.511 / **lumpy 0.678 (bias +2.3%)** /
  erratic 0.521; gate **PASS all four** (R3 bootstrap p=**0.000**).
- Score: **ALL FIVE SUPPORTED.** Calibration EXCELLENT — pooled dead-centre
  of the range, and the bias-cancellation sub-claim landed almost exactly
  (predicted ~+2% lumpy bias, realized +2.34%). This is the first well-
  calibrated *mechanism* prediction of the project.
- **Calibration lesson (refines the standing "I run optimistic on fancy
  models" prior):** that optimism bias is specific to FRESHLY-TRAINED nets
  (training stochasticity + hard-to-price data — GBM 0.48-0.54→0.5622,
  PatchTST 0.52-0.62→0.6516). A blend is a DETERMINISTIC average of
  already-measured components, so its behaviour is arithmetic and
  predictable — predict blends/composites tightly, keep widening ranges only
  for trained nets.
- **BIG durable finding (new prior added):** the offsetting-bias trap that
  makes VN1 non-segment-separable — which killed the Cycle-1 router
  (`7a52b16fe3`) and R4-blocked both qmean (+15.2%) and gbm56 (−10.5%) on
  lumpy — is CANCELLED, not fought, by a portfolio-wide blend of two models
  with equal-and-opposite segment biases. The blend averages forecasts
  across the whole portfolio (it does NOT segment-swap like the router), so
  the equal-and-opposite lumpy biases annihilate globally: lumpy bias
  −10.5%/+15.2% → +2.3%, lumpy VN1 0.678 below BOTH components (0.723/0.925)
  AND below the champion (0.682). Lumpy — THE promotion bottleneck since
  Cycle 4 — is cleared, and on every ≥10%-volume segment the blend beats the
  old champion. ROADMAP prediction 3 CONFIRMED; different families fail in
  disjoint places (GBM at Jan turns, FM on intermittent bias), so their
  average wins where neither single model could.
- **Feasibility lesson:** the full-scale first-origin smoke (195,689 rows,
  0 NaN, 1.7 min) is a cheap, high-value pre-flight for a mixed GBM+torch
  blend — confirmed the OMP mitigation and the true runtime (~7 min total,
  not the 1–1.5 h estimated) before committing the full run.
