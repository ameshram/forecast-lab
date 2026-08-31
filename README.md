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
result lands in an append-only ledger, promotion is decided by a
deterministic gate, and each cycle's lessons feed the next
([experiments/learnings.md](experiments/learnings.md)).

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

## Leaderboard (4 rolling origins × 13-week horizon, pooled VN1, lower = better)

| Model | Run | VN1 | WAPE | Bias | Status |
|---|---|---|---|---|---|
| blend — LightGBM-56 + Chronos-Bolt (50/50) | `1bb3cb32c6` | **0.5192** | 0.4933 | −2.6% | 🏆 champion (promoted 2026-08-30) |
| Chronos-Bolt (zero-shot, trimmed-mean) | `828187da8a` | 0.5396 | 0.5368 | −0.3% | best on smooth + January; blend component |
| global LightGBM 56-anchor (Tweedie) | `d047d88915` | 0.5429 | 0.4939 | −4.9% | full-cycle training; blend component; R4-blocked on lumpy alone |
| global LightGBM 24-anchor (Tweedie) | `7548b6e864` | 0.5622 | 0.5542 | −0.8% | former champion (promoted 2026-08-28) |
| bias-scaled router | `2c050a9d24` | 0.5642 | 0.5387 | −2.6% | gate fail (origin majority) |
| N-HiTS (Poisson, no covariates) | `eb69d1a3bc` | 0.5677 | 0.5320 | +3.6% | trained deep net; ≈ GBM-24 |
| naive (last week) | `2a69e4f986` | 0.5830 | 0.5313 | −5.2% | initial champion; Stage 0 yardstick |
| router smooth→MA13 | `7a52b16fe3` | 0.5853 | 0.5127 | −7.3% | gate fail (bias term) |
| Chronos-Bolt (zero-shot, median) | `4248b74828` | 0.5878 | 0.5134 | −7.4% | under-forecasts intermittent/lumpy |
| moving average 8w | `c87a16ebf7` | 0.5880 | 0.5486 | −3.9% | |
| moving average 13w | `3891f3c0fc` | 0.6081 | 0.5399 | −6.8% | dead-zeroing variant scored identically |
| TimesFM 2.5 (zero-shot) | `e8a11552ec` | 0.6512 | 0.4612 | −19.0% | point head mis-calibrated on zeros |
| PatchTST (Poisson) | `9ad032350c` | 0.6516 | 0.5808 | +7.1% | trained deep net; phantom demand on dead series |
| seasonal uplift naive | `accafab14a` | 0.6956 | 0.5425 | −15.3% | refuted mechanism |
| Chronos-Bolt (zero-shot, q0.65) | `46df0bba04` | 0.7223 | 0.5903 | +13.2% | dead-end readout (over-forecasts) |
| N-HiTS (Poisson + covariates) | `e8987a9d2b` | 0.9956 | 0.7305 | +26.5% | covariates destabilize training (untuned) |
| seasonal naive 52w | `e663fe200e` | 1.0211 | 0.7650 | −25.6% | |
| zero forecast | `09fdb04e06` | 2.0000 | 1.0000 | −100% | floor |

VN1 score = (Σ|F−A| + |ΣF−ΣA|) / ΣA — WAPE plus an absolute portfolio-bias
term (the official competition metric); Bias is signed (+ = over-forecast).
The ledger holds 20 runs; two exact no-op duplicates (MA13 dead-series
zeroing, and the Chronos "mean" readout that returns the Bolt median) are
folded into their twins above. Full evidence per run under
`experiments/runs/<run_id>/`. Live results UI: `dashboard/index.html`
(published as the "Forecast Lab" artifact).

## Evaluation discipline

- **Phase 1 (2023-10-09 → 2024-01-01) is the locked final test set** —
  openable only via `src/final_eval.py` after a human creates
  `experiments/PHASE1_SIGNOFF`, once, at the end (stopping rule in
  ROADMAP.md).
- All selection happens on rolling-origin backtests inside Phase 0 through
  the frozen harness (`src/metrics.py`, `src/backtest.py`).
- Promotion is mechanical (`src/gates.py`): pooled win + origin majority +
  paired bootstrap p<0.05 + no big-segment regression; gate evaluations are
  budgeted (25) to prevent validation overfitting.
- Governance is enforced, not aspirational: `.claude/hooks/protect_frozen.py`
  hard-blocks edits to the frozen harness, direct ledger writes, and any
  Phase-1 access without sign-off. The working protocol lives in
  [CLAUDE.md](CLAUDE.md); cycle procedures in `.claude/skills/`.

## Layout

```
data/                  raw wide CSVs + tidy raw_data.parquet
src/                   data, frozen metrics/backtest, features, models/,
                       ledger, gates, final_eval, run_experiment CLI
configs/               one YAML per experiment (the only way models run)
experiments/           ledger.csv, runs/<id>/, champion.json, gate_log,
                       learnings.md, memos/
dashboard/index.html   self-contained results dashboard
tests/                 metric, contract, and anti-leakage tests
scripts/               original exploratory notebooks
```

## Reproduce

```bash
python -m pip install -r requirements.txt   # macOS: brew install libomp first (LightGBM)
python -m pytest tests/
python -m src.run_experiment configs/naive_last.yaml configs/gbm_v1.yaml
python -m src.gates <run_id>
```

## Status & next

Stages 0–3 of the roadmap are complete (baselines → global LightGBM →
zero-shot foundation models → trained deep nets → the blend). The champion
is the **50/50 blend** of the 56-anchor LightGBM (`d047d88915`) and
Chronos-Bolt with a trimmed-mean readout (`828187da8a`) — run `1bb3cb32c6`,
VN1 0.5192, promoted 2026-08-30 after a full gate PASS (4/4 origins,
bootstrap p≈0, first challenger to clear the lumpy-segment rule). Next: a
principled-weight (holdout-fit, not a sweep) follow-up on the blend; per the
ROADMAP stopping rule, the one-shot Phase 1 final exam waits until two
consecutive cycles produce no new champion. Every registered prediction is
scored, right or wrong, in `experiments/learnings.md`.
