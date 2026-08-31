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
**Stage 3 complete — the blend is the new champion (ROADMAP prediction 3
confirmed).** Champion is `1bb3cb32c6` (blend: 50/50 GBM-56 `d047d88915` +
Chronos-Bolt qmean `828187da8a`, VN1 0.5192, gate PASS on all four rules —
4/4 origins, bootstrap p=0.0, first challenger to clear R4 on lumpy — promoted
2026-08-30 with human confirm; prior champion `7548b6e864` GBM-24 0.5622).
Stages 0–3 done (baselines → global LightGBM → zero-shot foundation models →
trained deep nets → the blend). Per the ROADMAP stopping rule, the one-shot
Phase 1 final exam waits until two consecutive cycles produce no new champion;
the open lever is a principled-weight (holdout-fit, not a sweep) follow-up on
the blend. Gate budget: 14/25.

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
  never route.
- Client/portfolio-level multiplicative week-of-year re-anchoring is a
  DEAD-END (`accafab14a`): lifecycle churn contaminates calendar indices.
  The holiday blind spot needs churn-robust seasonality (in-model features
  or same-series y-o-y), not post-hoc scaling.
- Trailing-ratio debiasing helps steady regimes but is actively harmful
  across seasonal turns — it scales the wrong way into the January drop
  (`2c050a9d24`, +29% bias at that origin).
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
  `e8a11552ec` 0.6512, smooth 0.338, Jan 0.574 — champion Jan 0.892).
  Their weakness is under-forecast **bias on intermittent/lumpy**, not
  accuracy: both beat the champion on pooled WAPE. Debias or blend them
  (BLENDING CONFIRMED — Chronos-qmean `828187da8a` is one half of champion
  `1bb3cb32c6`); don't reach for bigger checkpoints. TimesFM's point head is worse
  calibrated than Chronos-Bolt's median on zero-inflated series (−23% vs
  −7% intermittent bias) — mechanism prediction refuted in cycle 3.
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
  large-magnitude inputs (y up to 13,669) drive the NB `probs` param to
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
- My memo VN1 predictions run OPTIMISTIC on the fancier/TRAINED model (GBM
  cyc2 0.48-0.54→0.5622; PatchTST-Poisson 0.52-0.62→0.6516) — widen ranges
  upward there, and treat a trained net's mechanism sub-claims as coin-flips.
  But a BLEND is a deterministic average of already-measured components, so
  it predicts TIGHTLY: Cycle 9's blend `1bb3cb32c6` hit 0.50-0.55→0.5192
  dead-centre and its +2% lumpy-bias-cancellation mechanism landed at +2.34%.
  The optimism caveat and coin-flip caveat apply to trained nets, NOT to
  composites of runs already in the ledger.
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
