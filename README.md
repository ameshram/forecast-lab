# Forecast Lab — Enterprise Demand Forecasting

[![tests](https://github.com/ameshram/forecast-lab/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ameshram/forecast-lab/actions/workflows/tests.yml)

**The question this project answers** (see [ROADMAP.md](ROADMAP.md), the
canonical goal document): *do today's best deep learning forecasters —
including pre-trained foundation models — actually beat simpler methods on
messy, real-world retail demand? Where exactly, and by how much?*

Testbed: the VN1 Forecasting Accuracy Challenge dataset — 15,053
Client·Warehouse·Product series of weekly retail sales (2020–2024), ~80%
intermittent or lumpy demand. Method: a human-in-the-loop
champion/challenger lab where every experiment is a config file, every
result lands in a config-hashed ledger with per-segment evidence, promotion
is decided by a deterministic gate, and each cycle's lessons feed the next
([experiments/learnings.md](experiments/learnings.md)).

**Status: complete.** Thirteen cycles, 38 runs across classical, ML, deep
and foundation-model tiers, two quiet cycles, then the one-shot final exam
on the locked Phase 1 window (2026-09-01). An independent audit of the
evidence was run before writing the paper
([experiments/memos/2026-09-01-independent-audit.md](experiments/memos/2026-09-01-independent-audit.md));
its blockers and caveats are summarised under *Status & caveats* below.

## Data source & citation

The dataset is the **VN1 Forecasting Accuracy Challenge (Phase 2)**, published on
DataSource.ai. Any public use of it must be cited:

> Vandeput, Nicolas. "VN1 Forecasting - Accuracy Challenge Phase 2." DataSource.ai,
> 3 Oct. 2024,
> https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/description

The data was downloaded from the competition's Datasets page (a datasource.ai account is
required for access) and is used verbatim: `data/Phase 0 - Sales.csv`,
`Phase 0 - Price.csv`, `Phase 1 - Sales.csv`, `Phase 1 - Price.csv` → tidied into
`data/raw_data.parquet` by `src/data.py`. Citing this source credits its author; it
implies **no affiliation** with the author or organizers.

The raw CSVs are **not redistributed** with this repository pending a license check at
publication time — download them from the source above and place them in `data/`. A
comparability caveat for the paper's headline number is recorded in
[ROADMAP.md](ROADMAP.md) under *Source data & comparability*.

## Headline: the one-shot final exam (Phase 1, 2023-10-09 → 2024-01-01)

The reigning champion — an equal-weight blend of a 56-anchor LightGBM, a
56-anchor CatBoost and Chronos-Bolt with a trimmed-mean readout, run
`4f9b475ead` — was trained on all of Phase 0 and scored **once** on the
untouched 13-week Phase 1 window after the human sign-off
(`experiments/PHASE1_SIGNOFF`). Report: `experiments/final_eval_report.json`.

| | VN1 | WAPE | Bias |
|---|---|---|---|
| **Phase 1, pooled (15,053 series × 13 weeks)** | **0.5346** | 0.5141 | +2.05% |
| Validation pool, same model (4 origins inside Phase 0) | 0.5137 | 0.4893 | −2.4% |

| Demand class (assigned on Phase 0) | Share of Phase 1 volume | VN1 | Bias |
|---|---|---|---|
| intermittent | 37.0% | 0.5314 | +7.3% |
| smooth | 26.0% | 0.4170 | +3.6% |
| lumpy | 23.0% | 0.7108 | −0.6% |
| erratic | 13.9% | 0.6878 | −11.5% |
| insufficient | 0.2% | 3.0315 | +87.9% |

The blend's lumpy bias-cancellation mechanism (−10.5% / −1.3% / +15.2% in
its components) held out-of-sample at −0.6%. The Phase 1 number is reported
on its own terms, not as a competition placement; it is a single window with
no confidence interval (per-series Phase 1 forecasts were not retained).

## Leaderboard (validation: 4 rolling origins × 13-week horizon inside Phase 0, pooled VN1, lower = better)

| Model | Run | VN1 | WAPE | Bias | Status |
|---|---|---|---|---|---|
| blend — LightGBM-56 + CatBoost-56 + Chronos-Bolt qmean (⅓ each) | `4f9b475ead` | **0.5137** | 0.4893 | −2.4% | 🏆 champion (promoted 2026-08-31); Phase 1 headline above |
| blend — CatBoost-56 + Chronos-Bolt qmean (50/50) | `3330b31053` | 0.5159 | 0.5038 | −1.2% | pooled win, unpromotable: lumpy bias +6.9% fails the segment rule (also 2/4 origins, p=0.14) |
| blend — LightGBM-56 + Chronos-Bolt qmean (50/50) | `1bb3cb32c6` | 0.5192 | 0.4933 | −2.6% | former champion (promoted 2026-08-30) |
| global CatBoost 56-anchor (Tweedie) | `523f9c23da` | 0.5289 | 0.5076 | −2.1% | best single model; blend component |
| AutoTheta (statsforecast, season 52) | `88447a52f4` | 0.5355 | 0.5269 | −0.9% | best classical model; beats every deep net and foundation model |
| weight-fit blend (4-way, holdout-fitted weights) | `b12f62f304` | 0.5383 | 0.4908 | −4.7% | fitted weights improve the holdout but not the forecast window; equal weights win |
| Chronos-Bolt (zero-shot, trimmed-mean readout) | `828187da8a` | 0.5396 | 0.5368 | −0.3% | best on smooth + January; blend component; alone fails the lumpy segment rule (+15.2%) |
| global LightGBM 56-anchor (Tweedie) | `d047d88915` | 0.5429 | 0.4939 | −4.9% | full-cycle training window; blend component; alone fails the lumpy segment rule (−10.5%) |
| TFT (Poisson, no covariates) | `f7d0b97ce8` | 0.5494 | 0.5465 | +0.3% | best trained deep net |
| global XGBoost 56-anchor (Tweedie) | `d1083e2be5` | 0.5509 | 0.4951 | −5.6% | |
| global LightGBM 24-anchor (Tweedie) | `7548b6e864` | 0.5622 | 0.5542 | −0.8% | former champion (promoted 2026-08-28) |
| bias-scaled router | `2c050a9d24` | 0.5642 | 0.5387 | −2.6% | gate fail (origin majority); trailing debias breaks at seasonal turns |
| N-HiTS (Poisson, no covariates) | `eb69d1a3bc` | 0.5677 | 0.5320 | +3.6% | trained deep net |
| AutoETS (statsforecast, season 52) | `e8dab74d6a` | 0.5715 | 0.5593 | +1.2% | |
| naive (last week) | `2a69e4f986` | 0.5830 | 0.5313 | −5.2% | initial champion; Stage 0 yardstick |
| TSB (statsforecast, α=0.2/0.2) | `43f080e588` | 0.5846 | 0.5273 | −5.7% | intermittent-demand classic; does not beat naive |
| DeepAR (Poisson) | `bf6eab60fb` | 0.5852 | 0.5537 | −3.1% | |
| router smooth→MA13 | `7a52b16fe3` | 0.5853 | 0.5127 | −7.3% | gate fail (bias term); VN1 is not segment-separable |
| Chronos-Bolt (zero-shot, median) | `4248b74828` | 0.5878 | 0.5134 | −7.4% | under-forecasts intermittent/lumpy; the "mean" readout run `524453e7d8` is byte-identical (Bolt returns the median as mean) |
| moving average 8w | `c87a16ebf7` | 0.5880 | 0.5486 | −3.9% | |
| Chronos-2 (zero-shot, median) | `461e681f0d` | 0.6040 | 0.5102 | −9.4% | best January origin in the ledger (0.569) |
| moving average 13w | `3891f3c0fc` | 0.6081 | 0.5399 | −6.8% | dead-zeroing variant `13811e908e` scored identically |
| TimesFM 2.5 (zero-shot) | `e8a11552ec` | 0.6512 | 0.4612 | −19.0% | best WAPE in the ledger; point head under-forecasts zero-inflated series |
| PatchTST (Poisson) | `9ad032350c` | 0.6516 | 0.5808 | +7.1% | trained deep net; phantom demand on dead series |
| Moirai-2 R-small (zero-shot, median) | `7c77d54b6f` | 0.6814 | 0.4781 | −20.3% | run from an isolated venv |
| seasonal uplift naive | `accafab14a` | 0.6956 | 0.5425 | −15.3% | refuted mechanism |
| Chronos-Bolt (zero-shot, q0.65) | `46df0bba04` | 0.7223 | 0.5903 | +13.2% | dead-end readout (over-forecasts) |
| TimesFM 3.0 (zero-shot, median) | `9c198e246f` | 0.7304 | 0.6607 | +7.0% | lumpy over-forecast concentrated at the January origin |
| IBM TTM r2 (zero-shot) | `451ecc4919` | 0.7595 | 0.5602 | −19.9% | run from an isolated venv |
| Croston classic | `57bee33498` | 0.8495 | 0.6246 | −22.5% | stationary-demand assumption broken by lifecycle churn |
| Croston SBA | `771488878b` | 0.8824 | 0.6188 | −26.4% | |
| N-HiTS (Poisson + 8 covariates) | `e8987a9d2b` | 0.9956 | 0.7305 | +26.5% | covariates destabilize training (untuned) |
| seasonal naive 52w | `e663fe200e` | 1.0211 | 0.7650 | −25.6% | |
| TimesFM 3.0 (zero-shot, trimmed-mean) | `8582538c28` | 1.0256 | 0.7827 | +24.3% | readout calibration does not transfer across families |
| Chronos-2 (zero-shot, trimmed-mean) | `92906cd09e` | 1.0713 | 0.8093 | +26.2% | readout calibration does not transfer across families |
| zero forecast | `09fdb04e06` | 2.0000 | 1.0000 | −100% | floor |

VN1 score = (Σ|F−A| + |ΣF−ΣA|) / ΣA — WAPE plus an absolute portfolio-bias
term (the official competition metric); Bias is signed (+ = over-forecast).
The ledger holds 38 runs; two exact no-op duplicates (MA13 dead-series
zeroing, and the Chronos "mean" readout that returns the Bolt median) are
folded into their twins above. Full evidence per run under
`experiments/runs/<run_id>/` (config, per-origin, per-segment and per-series
aggregates). Live results UI: `dashboard/index.html` (published as the
"Forecast Lab" artifact).

## Evaluation discipline

- **Phase 1 (2023-10-09 → 2024-01-01) was the locked final test set** —
  opened only via `src/final_eval.py` after a human created
  `experiments/PHASE1_SIGNOFF`, once, at the end (stopping rule in
  ROADMAP.md: two consecutive cycles without a new champion).
- All selection happened on rolling-origin backtests inside Phase 0 through
  the frozen harness (`src/metrics.py`, `src/backtest.py`); the last
  validation window ends 2023-10-02, one week before Phase 1.
- Promotion is mechanical (`src/gates.py`): pooled win + origin majority +
  paired series-bootstrap p<0.05 + no big-segment regression; gate
  evaluations are budgeted (25; 16 used). The bootstrap resamples series and
  ignores client/warehouse and calendar dependence — see the audit memo for
  cluster-bootstrap and leave-one-origin-out results.
- Governance hooks (`.claude/hooks/protect_frozen.py`) block edits to the
  frozen harness, direct edits to the ledger files, and running the final
  evaluation without the sign-off file. The Phase 1 lock is procedural: the
  data loader reads both phases and `src/run_experiment.py` discards Phase 1
  before any model sees data.
- **Ledger semantics.** `run_id` is a hash of the config. Re-running an
  identical config *replaces* that run's ledger row and artifacts rather
  than appending a second row, so copy `experiments/runs/<run_id>/` before
  re-running anything whose original you need to keep. Torch-based runs are
  not bit-reproducible; the deterministic runs (baselines, statsforecast)
  reproduce exactly.
- The working protocol lives in [CLAUDE.md](CLAUDE.md); cycle procedures in
  `.claude/skills/`; every registered prediction is scored, right or wrong,
  in `experiments/learnings.md`.

## Layout

```
data/                  raw wide CSVs + tidy raw_data.parquet (not redistributed)
src/                   data, frozen metrics/backtest, features, models/,
                       ledger, gates, final_eval, run_experiment CLI
configs/               one YAML per experiment (the only way models run)
experiments/           ledger.csv, runs/<id>/, champion.json, gate_log,
                       learnings.md, memos/, PHASE1_SIGNOFF, final_eval_report.json
dashboard/index.html   self-contained results dashboard
tests/                 metric, contract, and anti-leakage tests
scripts/               original exploratory notebooks
```

## Reproduce

```bash
python -m pip install -r requirements.txt   # macOS: brew install libomp first (LightGBM)
python -m pytest tests/
OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE python -m src.run_experiment configs/naive_last.yaml
python -m src.gates <run_id>
```

The two environment variables are required for any config that mixes a
gradient booster with a torch model in one process (every blend, and the
champion); never run a LightGBM config after CatBoost has loaded in the same
process (libomp segfault) — one `run_experiment` invocation per
booster-mixing config. Moirai-2 and IBM TTM are RED installs against this
stack and were run from throw-away isolated venvs (see their config notes).
Versions used for the recorded runs: Python 3.11.4, pandas 2.2.3, numpy
2.1.2, lightgbm 4.7.0, catboost 1.2.10, xgboost 3.2.0, torch 2.13.0,
chronos-forecasting 2.3.1, neuralforecast 3.2.1, statsforecast 2.1.1,
timesfm 3.0.0 (2.0.2 for run `e8a11552ec`). Hugging Face checkpoint
revisions in use at the time of the final exam: chronos-bolt-base
`5d9f166d`, chronos-2 `29ec3766`, timesfm-2.5-200m-pytorch `1d952420`,
timesfm-3.0-pytorch `c7190707`, moirai-2.0-R-small `30f43ff0`.

## Status & caveats

Stages 0–3 of the roadmap are complete (baselines → global LightGBM →
zero-shot foundation models → trained deep nets → the blend), the stopping
rule was met after two quiet cycles, and the final exam has run. No deep net
or foundation model alone beat the simpler methods anywhere in 38 runs; the
winner is a bias-cancelling equal-weight blend of two gradient boosters and
one zero-shot foundation-model readout, scored at VN1 0.5346 on the locked
test window. Remaining work is writing the paper.

Caveats the paper states (from the independent audit): the headline is one
Q4 window with no confidence interval; all selection used the same four
Phase 0 windows across ~38 configurations, so validation scores are
optimistic; gate p-values are series-bootstrap decision rules (the final
promotion's client-cluster p is 0.03); the zero-shot models' pretraining
corpora are undocumented or post-date the public data and no memorization
probe was run; the "newer foundation models hallucinate at seasonal turns"
and "median-as-mean in four families" findings rest on two checkpoints and
on library packaging respectively; trained deep nets were run at one untuned
budget without covariates; the three boosters share a pipeline but not
byte-identical inputs; comparability with the competition's official
ranking is unverified.
