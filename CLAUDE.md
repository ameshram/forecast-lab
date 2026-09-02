# Forecast Lab — Constitution

## North Star (read this before anything else)

**The one goal:** answer, rigorously and reproducibly, *"Do today's best
deep learning forecasters — including pre-trained foundation models —
actually beat simpler methods on messy real-world retail demand? Where
exactly, and by how much?"* — on the VN1 dataset (15,053 weekly
Client·Warehouse·Product series, ~80% intermittent/lumpy), delivered as a
repo + dashboard + paper.

The full path, registered predictions, and scope fences live in
**ROADMAP.md** — read it at the start of every session. Current stage:
**PROJECT COMPLETE — the one-shot Phase 1 final exam ran 2026-09-01
(human sign-off + human-run): headline VN1 0.5346, WAPE 0.5141, bias
+2.05% (`experiments/final_eval_report.json`). Lumpy bias −0.6% —
the blend's cancellation mechanism generalized out-of-sample. All that
remains is writing the paper and repo polish; NO further experiments,
promotions, or Phase-1 reads, ever.** Champion is
`4f9b475ead` (blend: ⅓ GBM-56 `d047d88915` + ⅓ CatBoost-56 `523f9c23da` +
⅓ Chronos-Bolt qmean `828187da8a`, VN1 0.5137, gate PASS all four rules —
3/4 origins, p=0.0, zero segment regressions, lumpy 0.651 bias +1.1% —
promoted 2026-08-31 with human confirm; prior champions `1bb3cb32c6` 2-way
blend 0.5192, `7548b6e864` GBM-24 0.5622). Cycle 11's promotion RESET the
quiet-cycle counter to 0 (Cycle 10 had been quiet 1 of 2). Engineering
caution: never chain a LightGBM-containing config after a CatBoost-resident
process — libomp segfault (Cycle 11 beam, exit 139); one run_experiment
invocation per booster-mixing config.
Stages 0–3 done (baselines → global LightGBM → zero-shot foundation models →
trained deep nets → the blend). Cycle 10 (OFF-ROADMAP benchmark zoo,
human-approved) ran 12 challengers across Tiers 0–4: none beat the champion,
zero gate evals spent. Cycle 11 executed the CatBoost blend-swap memo: the
straight swap `3330b31053` (0.5159, pooled win) was computed-R4-blocked as
predicted (lumpy bias +6.94% vs +6.9% predicted — the offsetting-bias trap
confirmed with a fresh pair); the three-way blend `4f9b475ead` (0.5137) was
promoted. Cycle 12 spent the last diagnosed lever: the principled-weight
blend `b12f62f304` (0.5383) REFUTED weight-fitting — Stage 3 is closed with
evidence (equal weights ≈ optimal, trailing fits don't generalize, no
component left to add). Cycle 13's diagnosis found ZERO proposable
hypotheses surviving the priors and was declared quiet with human
confirmation (2026-09-01). **Quiet-cycle counter: 2 of 2 — the stopping
rule is MET.** The search is over; champion `4f9b475ead` is the final
pre-exam champion. The ONLY remaining step is human-only: create
`experiments/PHASE1_SIGNOFF`, then `python -m src.final_eval` runs the
champion ONCE on the untouched Phase 1 (the paper's headline). Do not
propose further validation-phase experiments. Gate budget final: 16/25.

Anti-drift rules: every memo option names the ROADMAP stage it advances
(else it is flagged OFF-ROADMAP and needs explicit human approval); every
number quoted anywhere must trace to a run-id or file in `experiments/` —
if it can't, it is unknown and must be stated as unknown, never estimated
from memory.

## Role

Act as the **Principal AI/ML Engineer** on this project: opinionated,
evidence-first, 15+ years of applied forecasting instincts. The human is the
approver. You recommend and execute; you never self-approve a promotion,
never touch Phase 1, and never present a recommendation without its ledger
evidence. Rigor over speed; one change per experiment.

## Non-negotiables (hooks enforce the starred ones)

1. ★ **Phase 1 (2023-10-09 →) is LOCKED.** Only `python -m src.final_eval`
   may read it for evaluation, and only after the human creates
   `experiments/PHASE1_SIGNOFF`. Never use Phase 1 for selection, tuning,
   feature scaling, or "just checking".
2. ★ **The harness is frozen**: `src/metrics.py`, `src/backtest.py`,
   `src/final_eval.py`. A changed metric invalidates every ledger row. If a
   harness change is truly needed, the human lifts the freeze by editing
   `.claude/hooks/protect_frozen.py` themselves.
3. ★ **The ledger is append-only** (`experiments/ledger.csv`,
   `champion.json`, `gate_log.jsonl`) — written only via `src/ledger.py` /
   `src/gates.py`.
4. **Every experiment is a config file** in `configs/`, run via
   `python -m src.run_experiment configs/<name>.yaml`. No ad-hoc evaluation
   paths. New models register in `src/models/__init__.py` with the
   `fit_predict(train, future_ds)` contract and enter as challengers like
   any other config.
5. **Promotion is mechanical**: `python -m src.gates <run_id>` decides
   PASS/FAIL (pooled win + origin majority + bootstrap p<0.05 + no segment
   catastrophe). You recommend, the gate certifies, the human confirms; only
   then `--promote --confirm`.
6. **Failed hypotheses are assets**: record what was tried and why it lost
   (decision memos in `experiments/memos/`, learned-results list in the
   dashboard). Never re-run a hypothesis the ledger already refuted without
   new reasoning.

## The cycle (human-in-the-loop)

`/diagnose` → `/propose` (decision memo, wait for approval) → `/experiment`
(approved configs only) → gate → `/promote` (human confirms) → `/reflect`
(score prediction vs outcome into `experiments/learnings.md`; update priors
with citations) → repeat. Skills in `.claude/skills/` define each step.

**Beam, not chain**: a memo may mark cheap, independent options as
co-runnable; one human approval launches the whole beam, and each challenger
is gated independently. Spend the human's approvals on decisions, not on
serial waiting. (Research basis: tree/parallel search beats linear agents on
MLE-bench; validation overfitting is contained by the gate budget, which
counts every gate evaluation regardless of beam width.)

## Decision memo format (every recommendation)

**Context** (cycle #, current champion + score) · **Evidence** (ledger
run-ids, segment/origin numbers) · **Options** (2–3 hypotheses as config
diffs, each with expected impact, cost, risk) · **Recommendation** (one, with
reasoning) · **What would change my mind**. Save to
`experiments/memos/<date>-<slug>.md`.

## Standing priors (evidence-backed; revise only with new ledger evidence)

- Recency beats seasonality here: heavy lifecycle churn, seasonal-naive VN1
  1.02 vs naive 0.58 (runs `e663fe200e` / `2a69e4f986`).
- Price is only recorded on sale weeks — "missing price" is structural, not
  a data bug; forward-fill is the feature, imputation models are not worth it.
- The sparse tail ("insufficient", 3.9% of volume) is not worth chasing;
  lumpy (20%) and erratic (13%) are the open battlegrounds.
- Bias costs as much as error under VN1 — check `bias_pct` before
  celebrating a WAPE win.
- **VN1 is not segment-separable**: the bias term is portfolio-global, so a
  segment-local improvement that removes *offsetting* bias can worsen the
  total score (router run `7a52b16fe3`: WAPE improved, VN1 got worse).
  Never predict a challenger's pooled score by summing segment gains.
  The FLIP SIDE (Cycle 9): the same portfolio-global bias term can be
  *exploited* — a portfolio-wide BLEND of two models with equal-and-opposite
  segment biases cancels that bias globally and wins where each single model
  was R4-blocked (`1bb3cb32c6`: lumpy −10.5%/+15.2% → +2.3%). A blend
  averages across the whole portfolio (cancels); the router segment-swaps
  (deletes an offset) — opposite effects, so blend to cancel offsetting bias,
  never route. FORWARD-CONFIRMED in Cycle 11: blend bias is exactly linear
  in component biases, so compute it BEFORE running — the straight
  CatBoost-for-GBM swap was predicted R4-blocked at lumpy +6.9% and landed
  +6.94% (`3330b31053`, pooled win, unpromotable), while the three-way
  ⅓-each blend was predicted at +1.1% and landed +1.12% (`4f9b475ead`,
  promoted 0.5137). A better single model does NOT make a better blend if
  its bias is neutral where the cancellation lives — check the arithmetic
  first; it costs nothing and has been dead-on five times.
- Client/portfolio-level multiplicative week-of-year re-anchoring is a
  DEAD-END (`accafab14a`): lifecycle churn contaminates calendar indices.
  The holiday blind spot needs churn-robust seasonality (in-model features
  or same-series y-o-y), not post-hoc scaling.
- Trailing-ratio debiasing helps steady regimes but is actively harmful
  across seasonal turns — it scales the wrong way into the January drop
  (`2c050a9d24`, +29% bias at that origin). EXTENDED (Cycle 12,
  `b12f62f304`): trailing-holdout weight FITTING is a milder failure of the
  same kind — a leak-free per-origin simplex fit improved the holdout at
  all 4 origins but the gains did not survive the 13-week jump to the
  forecast window (pooled 0.5383 vs equal-weight champion 0.5137, 1/4
  origins). DEAD-END: fit-anything-trailing on this portfolio; equal
  weights are the robust blend choice. Side-finding worth keeping: the fit
  also measured the dry well (Chronos-2-median earned ~0 weight at 3/4
  origins even from a free optimizer) and an all-GBM January mix scored
  0.4931 — the ledger's best January — confirming the 56-anchor GBMs now
  own the seasonal turn (no promotion path: per-origin weight-switching is
  the router trap in a new hat).
- A learned model's training window must span a full seasonal cycle —
  CONFIRMED and closed: 56 anchors (`d047d88915`) fixed the champion's
  January origin outright (0.892→0.497, bias +25%→−1.6%), pooled 0.5429,
  4/4 origins, but computed-FAIL on R4 (lumpy 0.7232 vs limit 0.7159);
  formal gate skipped by human, champion unchanged. `d047d88915` is the
  strongest GBM config for future blends.
- Lumpy WAS the promotion bottleneck: it R4-blocked two mechanistically
  opposite challengers — qmean over-forecasts it (+15.2%, `828187da8a`)
  and gbm56 under-forecasts it (−10.5%, `d047d88915`) — against the
  champion's best-in-ledger lumpy 0.6818. **SOLVED in Cycle 9:** a 50/50
  blend of those SAME two opposite-biased models (`1bb3cb32c6`) cancelled
  lumpy bias to +2.3%, took lumpy VN1 to 0.678 (below the champion's 0.682),
  cleared R4, and was promoted — the bottleneck yielded to averaging, not to
  loosening the rule (R4 was never touched). The instinct held: never loosen
  R4 to get past a segment (post-hoc rule-shopping); build a forecast that
  clears it.
- Zero-shot foundation models lose pooled VN1 to the champion but are the
  best models in the ledger on smooth series AND at the January seasonal
  turn (chronos `4248b74828` 0.5878, smooth 0.372, Jan 0.599; timesfm
  `e8a11552ec` 0.6512, smooth 0.338, Jan 0.574 — champion Jan 0.892;
  chronos-2 median `461e681f0d` 0.6040 has the best January in the ledger,
  0.569). Their weakness is under-forecast **bias on intermittent/lumpy**,
  not accuracy — CONFIRMED as the family signature in Cycle 10: TimesFM
  −19.0%, Moirai-2 `7c77d54b6f` −20.3%, TTM `451ecc4919` −19.9%, all with
  top-tier WAPE (Moirai 0.478 beats even the champion's 0.493) sunk by
  ~−20% bias. Debias or blend them (BLENDING CONFIRMED — Chronos-qmean
  `828187da8a` is one half of champion `1bb3cb32c6`); don't reach for
  bigger checkpoints — THREE-times confirmed and UPGRADED (Cycle 13-b):
  Chronos-2 loses to Bolt on both readouts (0.6040/1.0713 vs 0.5878/0.5396)
  and TimesFM-3 loses to TimesFM-2.5 on both readouts (`9c198e246f` 0.7304,
  `8582538c28` 1.0256 vs 0.6512). **Generation shift in the failure mode:**
  the OLD generation (Bolt, TimesFM-2.5, Moirai, TTM) under-forecasts
  sparse demand (−19..−26%); the NEW generation (Chronos-2, TimesFM-3 —
  different labs, months apart) HALLUCINATES demand on lumpy at the
  seasonal turn (lumpy bias +140..+157%, Jan origins 1.9–2.7, +71..+114%).
  On zero-inflated retail data, newer is actively worse with an inverted
  bias sign. TimesFM's point head is worse calibrated than Bolt's median
  on zero-inflated series (−23% vs −7% intermittent bias) — cycle-3
  mechanism refuted; the v3 qmean readout DID fix intermittent
  (−23%→−6.2%, the registered mechanism) but was swamped by the lumpy
  hallucination.
- **FM point-readout calibration does NOT transfer across checkpoint
  families** (Cycle 10): Bolt's winning trimmed-mean (qmean) EXPLODED on
  Chronos-2 — its upper quantile heads are far heavier on zero-inflated
  series (lumpy bias +157%, segment VN1 3.76, pooled 1.0713 `92906cd09e`).
  And every FM "mean"/point accessor so far IS the median — now **4 of 4
  families** (Bolt `524453e7d8`, Chronos-2, gluonts/Moirai, TimesFM-3
  `9c198e246f` byte-identical). Any new FM needs BOTH a readout-semantics
  test and a bias-vs-REAL-actuals smoke (scratchpad smoke_fm.py pattern;
  band [0.85,1.20]) before a full run — a contract-only smoke missed the
  +157% segment bias. AND (Cycle 13-b): a SINGLE-origin smoke cannot catch
  an origin-specific blowup — TimesFM-3 passed the Jul smoke (0.92/0.965)
  and detonated at Jan (bias +74/+114%). For zero-shot models, where an
  extra origin costs only inference, smoke BOTH a steady origin and the
  January turn.
- Benchmark-zoo ordering on this data (Cycle 10, 12 runs, none beat the
  champion): **blend > GBM ≈ classical-Theta > trained-deep > zero-shot-FM
  > intermittent-classical.** AutoTheta `88447a52f4` 0.5355 beats every
  deep net and FM; Croston/SBA (`57bee33498`/`771488878b`, −22/−26% bias)
  are the worst serious models — lifecycle churn breaks their stationary
  assumption, TSB's decay (`43f080e588` 0.5846) fixes most but never beats
  naive. **CatBoost `523f9c23da` 0.5289 is the best single model in the
  ledger** (same features/Tweedie/anchors as LGBM-56 0.5429 — booster
  implementation alone is worth ~1.4 points; lumpy 0.667 bias −1.3%, below
  the champion's 0.678; beats the champion at Oct AND Jan origins) → the
  registered next lever is the CatBoost blend-swap. Deep-net architecture
  ordering (same Poisson loss, no covariates): TFT `f7d0b97ce8` 0.5494 >
  N-HiTS 0.5677 > DeepAR `bf6eab60fb` 0.5852 > PatchTST 0.6516.
- The readout of Chronos' predictive distribution moves the bias term
  wholesale: median −7.4% (`4248b74828`), trimmed mean −0.3% and pooled
  0.5396 with 4/4 origin wins (`828187da8a`), quantile-0.65 +13.2%
  (`46df0bba04`, DEAD-END: uniform above-median readout). The trimmed
  mean gate-FAILED on R4 only: it funds neutral pooled bias by
  over-forecasting lumpy (+15.2%, segment VN1 0.925 vs 0.682) — lumpy is
  the blocking segment for a zero-shot champion, and a naive lumpy→median
  router computes to ≈0.562, champion-level (offsetting-bias trap again).
  For Bolt checkpoints the chronos API returns the median as "mean" —
  use `point: qmean` for the real trimmed mean.
- A deep count-likelihood (NegativeBinomial DistributionLoss + PatchTST via
  neuralforecast) is numerically fragile on this heavy-tailed 72%-zero data:
  large-magnitude inputs (phase0 y up to 13,681) drive the NB `probs` param to
  1.0/NaN on the FIRST forward/loss eval (crash, Cycle 6, no ledger row).
  It is a forward-pass overflow, not gradient explosion — gradient clipping
  does NOT help; input scaling (robust/standard) mitigates but is not
  reliably stable at full 15k scale; lr is irrelevant to it. Any trained
  count-distribution model must be smoke-tested on a spiky REAL subsample
  (not just the small synthetic panel) and clear numerical stability as a
  blocking pre-req before an expensive run. Poisson/Tweedie losses avoid the
  probs-saturation failure mode. Stage 2's NB hypothesis is still OPEN
  (neither supported nor refuted).
- ARCHITECTURE matters more than the deep-net-loss story first suggested.
  PatchTST + Poisson no-cov lost badly (`9ad032350c`, 0.6516), but N-HiTS +
  Poisson no-cov (`eb69d1a3bc`, **0.5677**, bias +3.6%) nearly MATCHES the
  champion 0.5622 and crushes PatchTST — same loss, same data, just a
  better backbone for these short intermittent series. So "trained deep
  nets lose here" (Cycle 6-B) was substantially a PatchTST artifact and is
  RETRACTED; N-HiTS is competitive. ROADMAP prediction 2 is back to OPEN.
  (The PatchTST +7.1% phantom-demand-on-dead-series over-forecast still
  stands as a PatchTST-specific finding.)
- Covariates DON'T integrate stably into this deep net at the principled
  untuned config (Cycles 7-8): the full 8-covariate N-HiTS run blew up
  (`e8987a9d2b`, 0.9956, +26.5%, confounded — an optimization artifact, not
  an info-value result), and even the mechanism-targeted recency subset
  (price, price_chg4, wsls) DESTABILIZES as training proceeds — bias-ratio
  trajectory 1.000(150 steps)→1.324(300)→1.522(1000) while no-cov stays
  ~1.05. Cycle 8's recency run was NOT launched (pre-run bias-ratio gate
  failed, budget 11/25). So the deep-net info-value question is not cleanly
  answerable here without a stabilization sub-project (exog scaling / lower
  lr / grad-clip = tuning = drift, out of scope). LESSON: low-step ablations
  mislead on calibration — always check bias-ratio at a realistic budget,
  and keep the bias-ratio pre-run gate for all deep runs.
- CAUTION on the loss-family lever (revised after verification, do not
  overclaim): reading the distribution MEAN, every count likelihood
  (Poisson/NB/Tweedie) gives a strictly-positive conditional-mean
  forecast ≈ the sample mean — so a zero-inflated loss does NOT lower the
  MEAN readout to zero; its effect is only via the variance function's
  reweighting of skewed observations. Tweedie-in-the-net is also
  numerically fragile here: its exp(log_mu) mean OVERFLOWS to ~1e13 under
  the robust scaler; only standard-scaler + lr=1e-4 + gradient-clip was
  stable in a stress test — and even then it still over-forecasts. So
  Tweedie-in-the-net is NOT a clean one-change experiment and is unlikely
  to fix a recency-driven over-forecast; prefer covariates or a blend.
- My memo VN1 predictions on TRAINED nets are NOISY IN BOTH DIRECTIONS —
  optimistic misses (GBM cyc2 0.48-0.54→0.5622; PatchTST-Poisson
  0.52-0.62→0.6516) AND a pessimistic one (TFT predicted 0.56-0.68→0.5494,
  Cycle 10) — so keep ranges WIDE on trained nets and treat their mechanism
  sub-claims as coin-flips, but don't assume the miss direction. The worst
  calibration failure to date was a ZERO-SHOT transfer assumption, not a
  trained net: Chronos-2 predicted 0.50-0.56 "likeliest gate candidate" →
  realized 1.0713/0.6040 (Cycle 10, readout non-transfer) — never carry a
  readout/calibration result across checkpoint families without a smoke.
  A BLEND is a deterministic average of already-measured components, so it
  predicts TIGHTLY: Cycle 9's blend `1bb3cb32c6` hit 0.50-0.55→0.5192
  dead-centre with its +2% lumpy mechanism landing at +2.34%, and Cycle 11
  went 7-of-8 sub-predictions dead-centre across both beam arms
  (`3330b31053` +6.9→+6.94%, `4f9b475ead` +1.1→+1.12%). The caveats apply
  to trained nets and cross-family transfers, NOT to composites of runs
  already in the ledger. Bookkeeping rule: gate-budget predictions count
  evals per gate INVOCATION (promote-confirm re-runs the gate = 2 per
  promotion), the one Cycle-11 miss.
- Validation is finite: gate evaluations are budgeted (25); prefer fewer,
  better-reasoned experiments.

## Environment

- Python venv: `./forecasting/bin/python` (moved venv — `pip` shebang broken,
  always use `python -m pip`).
- Tests: `./forecasting/bin/python -m pytest tests/`.
- Dashboard: `dashboard/index.html`, published as the "Forecast Lab"
  artifact — regenerate its embedded JSON from the ledger after promotions.
- `data/*DCPD*.csv` are unrelated to this project; ignore them.
- Data: official VN1 Phase 2 competition files (datasource.ai) — citation
  required in all public deliverables, no affiliation implied; see README
  §Data source & citation and memo
  `experiments/memos/2026-08-30-data-provenance-revalidation.md`.
- AWS is available via `--profile claude-code` (verified 2026-08-28; IAM
  user `claude-code`, account 913410646214). Optional — local M1 Max is the
  default compute; consider SageMaker only for heavy Stage 2+ sweeps, via a
  human-approved memo.
