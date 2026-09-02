# Decision memo — Cycle 10: the benchmark zoo

Date: 2026-08-31 · Cycle 10 · Stage: **OFF-ROADMAP** (Stages 0–3 are complete;
this round adds no new stage, dataset, or metric — it broadens the *comparison*
that is the project's North Star). **Requires explicit human approval** per the
anti-drift rules.

**Why OFF-ROADMAP is justified here (the published claim it serves):** the paper's
headline is "do today's best forecasters beat simple methods on messy retail
demand — *where and by how much*." Answering "by how much" credibly needs the
comparison table to cover the model families a 2026 practitioner would actually
reach for. Today the ledger is missing the canonical intermittent-demand
classical methods (Croston/TSB), the other gradient-boosters, two mainstream
deep architectures (DeepAR, TFT), and every *current* foundation model
(Chronos-2, Moirai 2.0, IBM TTM). Precedent for an approved OFF-ROADMAP
strengthening run: `d047d88915` ("Cycle 5A OFF-ROADMAP rival fix,
human-approved"). **Human direction on record (2026-08-31):** run the feasible
zoo; **TimeGPT excluded** (commercial API + would send the dataset to a third
party — 2026-08-30 decision); **AWS SageMaker permitted** as an ephemeral GPU
escape hatch.

## Context

- **Champion:** `1bb3cb32c6` — blend (50/50 GBM-56 `d047d88915` + Chronos-Bolt
  qmean `828187da8a`), pooled VN1 **0.5192**. Unaffected by this round; zoo
  models enter as comparison challengers only.
- **Gate budget: 14/25.** Crucial framing: **running a model is free** — it
  records a ledger row via `run_experiment` and does *not* call the gate. A gate
  evaluation (the 25-budget) is spent only when we certify a *promotion*
  candidate. We ran Chronos/TimesFM/PatchTST/N-HiTS with zero gate spend
  (learnings: "no gates run… budget preserved"). So a ~10-model zoo is affordable
  under a 11-eval remaining budget: **expected gate spend ≤ 2** (only a computed
  pooled win gets gated).
- **Stopping-rule note:** a Cycle-10 that produces no new champion counts as one
  of the two consecutive no-promotion cycles that trigger the one-shot Phase 1
  final exam. So the zoo either finds a new champion or advances us toward the
  finish — both are progress.

## Evidence

### Step-0 data re-validation (read-only, 2026-08-31 — human directive)

Re-ran the data contract before proposing any build. **All structural checks
green**, matching `2026-08-30-data-provenance-revalidation.md`:

| check | result |
|---|---|
| rows = 15,053 series × 183 weeks | 2,754,699 ✓ |
| duplicate (series, ds) pairs | **0** ✓ (the silent join-fanout hazard is absent) |
| week grid | all Mondays, continuous 7-day steps ✓ |
| phase boundary | phase0 ends 2023-10-02, Phase 1 starts 2023-10-09 ✓ |
| validation origins (4×13, step 13) | 2022-10-10 / 2023-01-09 / 2023-04-10 / 2023-07-10 ✓ (reproduces ledger) |
| phase0 zero-week share | 71.8% ✓ |
| price NaN | 0% on sale weeks, 100% on zero weeks ✓ (structural, matches prior) |
| phase0 y max | **13,681** — corrects CLAUDE.md's illustrative "13,669" (that is the 2nd-largest value; substance of the NB-overflow prior unaffected) |

Verdict: **data verified sound and unchanged.** Safe to build on.

### Current coverage vs. the enterprise-forecasting stack

| Tier | In ledger (run-id) | Missing (this memo) |
|---|---|---|
| 0 Baselines | naive `2a69e4f986`, seasonal-naive `e663fe200e`, MA `3891f3c0fc` | **Croston/SBA, TSB**, ETS, Theta, ARIMA |
| 1 ML (GBM) | LightGBM `7548b6e864` / `d047d88915` | XGBoost, CatBoost |
| 2 Global deep | N-HiTS `eb69d1a3bc`, PatchTST `9ad032350c` | **DeepAR, TFT** |
| 3 Foundation | Chronos-Bolt `4248b74828`/`828187da8a`, TimesFM 2.5 `e8a11552ec` | **Chronos-2**, Moirai 2.0, IBM TTM |
| 4 Ensemble | weighted blend `1bb3cb32c6` (champion) | per-series / per-horizon selection (parked, see below) |

### Feasibility (verified read-only against the venv, 2026-08-31)

| Family | Install status | Risk |
|---|---|---|
| statsforecast (Croston/SBA/TSB/AutoETS/AutoTheta/AutoARIMA) | not installed; deps all satisfied | **green** — trivial `pip install statsforecast` |
| DeepAR, TFT | **already importable** (neuralforecast 3.2.1) | **green** — zero install |
| XGBoost, CatBoost | not installed | **green** — self-contained wheels (libomp already works) |
| Chronos-2 | **no pip change** — chronos-forecasting 2.3.1 already ships `Chronos2Pipeline`; weights not cached | **yellow** — weights download + a wrapper branch (Chronos-2 output shape ≠ Bolt) |
| Moirai 2.0 (uni2ts) | not installed | **RED** — pins would downgrade torch → breaks neuralforecast; needs isolated venv |
| IBM TTM (granite-tsfm) | not installed | **RED** — pins transformers 4.x vs installed 5.16.1 → breaks chronos/timesfm; needs isolated venv |
| TimeGPT | — | **EXCLUDED** by human (commercial API + data egress) |
| FlowState, PatchTST-FM | availability unverified | investigate in Phase D, no commitment |

Hardware: 10-core M1 Max, 32 GiB, MPS (no CUDA). SageMaker available via
`--profile claude-code` for any model measured too slow locally.

### Standing constraints this plan honors

- **Contract:** every wrapper returns `[client_warehouse_product_id, ds, yhat]`,
  full coverage, and the frozen harness *raises* on any missing cell
  (`backtest.py`). Duplicate (series, ds) rows would silently fan out the scoring
  join → each new wrapper ships a **dedupe guard + a contract test** before its
  first real run (non-negotiable #4, `/experiment` rule).
- **Frozen harness untouched** (`metrics.py`, `backtest.py`, `final_eval.py`);
  ledger append-only via `src/ledger.py`; **Phase 1 never read for selection**.
- **Calibration prior:** my VN1 predictions run optimistic on trained/fancy
  models — ranges below are widened upward and mechanism sub-claims treated as
  coin-flips.
- **Composite-first:** any ensemble of a zoo winner is a later `blend` config, not
  new code.

## Options

### Option A — the full feasible zoo, as a phased beam (RECOMMENDED)

One approval launches four phases; within a phase the models are co-runnable
(independent, cheap). Each model = **one config + one ledger row**; gate only a
computed pooled winner.

**Phase A — intermittent-demand classical** (`pip install statsforecast`).
One new wrapper `StatsForecastModel` (lazy import inside `fit_predict`, `name`
attr, reuse the `_expand` cross-join for guaranteed coverage; NaN/all-zero
series fall back to 0). New registry key `statsforecast`; configs select the
model: **CrostonClassic, CrostonSBA, TSB, AutoETS, AutoTheta**. *AutoARIMA is
cost-gated* — run a timed 500-series subsample first; only launch full if it
extrapolates to < ~30 min. Compute: minutes. Contract test + all-zero-series
test required.
Expected: TSB ≥ Croston variants (obsolescence decay suits lifecycle churn);
**none beat naive 0.583 pooled**; value is the intermittent-segment story and a
proper classical yardstick for the paper. Gate spend: 0.

**Phase B — the other gradient-boosters** (`pip install xgboost catboost`).
Refactor `GlobalLGBM` into `GlobalGBM(estimator=...)` over the *existing*
decoupled feature pipeline (`src/features.py` — anchors, lags/rolls, price,
calendar). **Keep `global_lgbm` byte-identical** (run_id lineage preserved); add
registry keys `global_xgb`, `global_cat` at `n_anchors: 56` (strongest known GBM
window, `d047d88915`), Tweedie objective mapped per library (`reg:tweedie` /
`Tweedie:variance_power=1.2`). Risk: CatBoost dislikes NaN in categoricals
(the feature builder emits NaN lags) → explicit handling + test. Compute: minutes
each.
Expected: within ±0.02 of LightGBM's 0.5622 — completes the ML tier; unlikely to
beat the blend. Gate spend: 0 unless one lands < 0.5192.

**Phase C — the two mainstream deep nets** (zero install).
Extract the duplicated contract-mapping tail in `src/models/deep.py` into a
shared helper, then add thin `GlobalDeepAR` / `GlobalTFT` (per-arch DEFAULTS —
note TFT uses `n_head`, and DeepAR is recurrent with a restricted loss set and a
different output-column name that the test must assert). Poisson loss, **no
covariates** (covariates don't integrate stably here — Cycles 7–8).
**Blocking pre-req (mandatory, no exceptions):** the Cycle-8 bias-ratio smoke —
each config is smoke-fit on a spiky *real* subsample at ≥300 steps and must be
NaN-free, full-coverage, and ΣF/ΣA ∈ [0.85, 1.20] *before* the full run. If it
fails, park that architecture — **do not tune** (that is drift). Compute: ~15–40
min CPU each; TFT is the likeliest SageMaker candidate if too slow.
Expected (widened up per calibration prior): DeepAR/TFT land **0.56–0.68**
(N-HiTS 0.5677 is the deep ceiling so far). Gate spend: 0 unless one beats 0.5192.

**Phase D — the current foundation models.**
1. **Chronos-2** — no pip change; add a `Chronos2Pipeline` branch in
   `src/models/chronos.py` (its output shape and covariate interface differ from
   Bolt), download `amazon/chronos-2` weights, and — non-negotiable — a
   **point-readout semantics test** before any real run (the `524453e7d8`
   wasted-run lesson: verify what the library actually returns). Full-scale
   first-origin feasibility smoke (Cycle-9 pattern) before the 4-origin run.
   Expected **0.50–0.56**; the single likeliest gate candidate of the zoo.
2. **Moirai 2.0** and **IBM TTM** — RED installs. Each runs from a **throwaway
   isolated venv** (only pandas/numpy/pyyaml/pyarrow + the one FM lib), invoking
   `python -m src.run_experiment` from that venv against this repo (registry
   imports are lazy, so the harness loads without the main venv's DL stack). A
   `pip install --dry-run` resolver check first; **tear the venv down after**.
   The main `./forecasting` venv is never mutated. Expected **0.55–0.70**.
3. **FlowState / PatchTST-FM** — availability unverified; investigate during
   Phase D, promise nothing.
Gate spend: ≤ 2 total, only for a computed pooled win.

**SageMaker policy (approving this memo satisfies the CLAUDE.md SageMaker-via-memo
rule):** local M1 Max is default. SageMaker only for a model *measured* too slow
locally. Ephemeral discipline — spin up (e.g. `ml.g5.xlarge` ≈ $1.4/hr), run via
the same frozen harness, sync `experiments/runs/<id>/` artifacts back, write the
ledger locally, **tear the instance down immediately**. Each such use states its
instance type + rough \$ estimate.

### Option B — minimal high-value subset (considered)

Croston/TSB + DeepAR/TFT + Chronos-2 only — the highest-signal gaps, cheapest.
Rejected as the primary because it leaves the ML tier (XGB/CatBoost) and the
newest FMs (Moirai/TTM) — gaps the human explicitly called out — unfilled, and
those are cheap/isolated enough to include.

### Option C — skip the zoo, head to the final exam (considered)

Fastest path to the paper; weakest comparison table. Rejected because the
comparison *is* the deliverable, and the zoo is budget-cheap.

## Recommendation

**Option A**, phases A→D in order (A and B are near-instant and can run first as
one beam; C and D each behind their mandatory smoke/feasibility gate). It
directly serves the stated objective, spends ~0 gate budget, cannot harm the
champion, and turns "we compared a few families" into "we benchmarked the whole
stack." Each model is a config diff plus (where a new family) a registered
wrapper + contract test, exactly as the harness intends.

## What would change my mind

- A Phase-D `--dry-run` shows uni2ts/granite-tsfm would damage even an isolated
  resolver → drop that model, note it "not evaluated (dependency conflict)."
- AutoARIMA subsample extrapolates to hours → drop ARIMA, keep the rest.
- A Phase-C bias-ratio smoke fails → park that architecture (no tuning).
- Chronos-2 point-readout test shows an unexpected output semantics → fix the
  wrapper before spending compute (don't repeat `524453e7d8`).

## Registered predictions (to score at /reflect)

1. **No zoo model beats the blend `1bb3cb32c6` (0.5192) pooled.**
2. TSB ≥ Croston variants on the intermittent segment; **neither beats naive
   0.583** pooled.
3. DeepAR and TFT land **0.56–0.68** pooled (N-HiTS 0.5677 is the deep ceiling
   to date); at least one clears its bias-ratio smoke, at least a coin-flip that
   the other does not.
4. **Chronos-2 beats Bolt-qmean (0.5396)** and is the single likeliest gate
   candidate of the zoo (**0.50–0.56**).
5. Moirai 2.0 and IBM TTM land **0.55–0.70**.
6. **Total gate spend ≤ 2** for the whole cycle (budget ends ≤ 16/25).
7. XGBoost/CatBoost land within ±0.02 of LightGBM's 0.5622.

**STOP — awaiting human approval, amendment, or rejection.** Rejected options B
and C are recorded above as assets. On approval, /experiment runs the phased beam
(installs + wrappers + contract tests first; each model gated by its smoke where
noted); nothing is installed or run until then.

---

## ADDENDUM — realized results (appended at /reflect, 2026-08-31)

All 12 zoo runs completed; champion `1bb3cb32c6` (0.5192) unbeaten; **0 gate
evals spent** (budget stays 14/25); first quiet cycle (1 of 2) toward the
stopping rule. Full scoring in `experiments/learnings.md` (Cycle 10 entry).

| Model | run_id | VN1 | bias |
|---|---|---|---|
| CatBoost-56 | `523f9c23da` | **0.5289** | −2.1% |
| AutoTheta | `88447a52f4` | 0.5355 | −0.9% |
| TFT | `f7d0b97ce8` | 0.5494 | +0.3% |
| XGBoost-56 | `d1083e2be5` | 0.5509 | −5.6% |
| AutoETS | `e8dab74d6a` | 0.5715 | +1.2% |
| TSB | `43f080e588` | 0.5846 | −5.7% |
| DeepAR | `bf6eab60fb` | 0.5852 | −3.1% |
| Chronos-2 median | `461e681f0d` | 0.6040 | −9.4% |
| Moirai-2 R-small median | `7c77d54b6f` | 0.6814 | −20.3% |
| IBM TTM r2 | `451ecc4919` | 0.7595 | −19.9% |
| Croston Classic | `57bee33498` | 0.8495 | −22.5% |
| Croston SBA | `771488878b` | 0.8824 | −26.4% |
| Chronos-2 qmean | `92906cd09e` | 1.0713 | +26.2% |

Prediction scorecard: #1 ✓, #2 ✓, #3 half (TFT better than range), #4
**REFUTED** (Chronos-2 loses to Bolt on both readouts; qmean lumpy bias
+157% — readout calibration does not transfer across FM families), #5 half
(TTM worse than range), #6 ✓ (0 spent), #7 half (CatBoost better than ±0.02).

FlowState / PatchTST-FM (memo item 3, promise-nothing): not pursued — no
established zero-shot packaging found worth a RED-install risk this cycle.

Registered next lever (needs a new memo, NOT auto-run): CatBoost-for-LGBM
blend-swap inside the champion blend.
