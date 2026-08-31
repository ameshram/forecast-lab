# Decision memo — Stage 1 kickoff: zero-shot foundation models

Date: 2026-08-28 · Cycle 3 · Stage: **Stage 1 — foundation models, zero
training** (ROADMAP.md)

## Context

- Champion: `7548b6e864` (global_lgbm, Tweedie, 24 anchors) — pooled VN1
  **0.5622**, WAPE 0.5542, bias −0.8%. Promoted 2026-08-28 over naive
  `2a69e4f986` (0.5830).
- Gate budget: **10 of 25 evaluations used** (`experiments/gate_log.jsonl`,
  10 rows incl. duplicate re-runs). 15 remain.
- Stage 0 is closed. Stage 1 asks: *can a model that has never seen this
  data beat models trained on it?* Either answer is a paper section, so a
  gate FAIL here is still a deliverable, not a loss.

## Evidence (diagnosis, cycle 3)

Champion `7548b6e864`:

- Per-origin: 2022-10-10 **0.8309**, 2023-01-09 **0.8916**, 2023-04-10
  0.5006, 2023-07-10 0.5126. Both holiday-adjacent origins are >20% worse
  than pooled (flagged). At the January origin the champion is the *worst*
  model in the ledger — naive 0.6955 (`2a69e4f986`), router 0.6510
  (`7a52b16fe3`) — with +25.0% bias: the 24-anchor training window never
  contains a post-holiday decline (learnings.md, cycle 2).
- Weakness table, (segment VN1 − pooled) × volume_share:

  | segment | VN1 | vol share | headroom |
  |---|---|---|---|
  | insufficient | 1.8072 | 3.9% | +0.0488 (out of scope, ROADMAP fence) |
  | lumpy | 0.6818 | 20.4% | **+0.0244** |
  | erratic | 0.6148 | 13.5% | +0.0071 |
  | intermittent | 0.5567 | 30.1% | −0.0017 |
  | smooth | 0.5224 | 32.1% | −0.0128 |

- Beaten-by-non-champion check: on smooth the champion (0.5224) is beaten
  badly by the plain MA13/router (0.3884, `3891f3c0fc`/`7a52b16fe3`); on
  intermittent slightly by bias_scaled (0.5453, `2c050a9d24`). On lumpy and
  erratic the champion is the best run in the ledger.
- Bias: pooled −0.8% (excellent), but smooth is +10.2% over-forecast while
  the rest sit slightly under — the offsetting-bias structure that made the
  router fail (VN1 not segment-separable, `7a52b16fe3`).
- Registered ROADMAP prediction to test this stage: *"Foundation models land
  worse than the GBM overall, but surprisingly close on steady (smooth)
  products."*

## Options

### Option A — Chronos-Bolt zero-shot (Stage 1) — co-runnable

**Hypothesis:** a pre-trained foundation model, fed each series' raw weekly
history as context, gets within striking distance of the trained GBM on
smooth series with zero training, but loses pooled VN1 because its median
forecasts under-call the intermittent/lumpy half of the portfolio.

Config diff:
- New `configs/chronos_bolt_base.yaml` — `model: {name: chronos, params:
  {checkpoint: amazon/chronos-bolt-base, batch_size: 256, quantile: 0.5,
  clip_negative: true}}`.
- New family `src/models/chronos.py` + `REGISTRY["chronos"]` (standard
  `fit_predict(train, future_ds)`; harness untouched).
- Env: add `torch` + `chronos-forecasting` to requirements (~2.5 GB install,
  ~800 MB checkpoint). No frozen file changes.

Expected impact: pooled VN1 **0.58–0.68** (worse than champion 0.5622,
near/above naive 0.5830); smooth **0.50–0.56** vs champion's 0.5224.
Reasoning: zero-shot models have no way to learn this portfolio's lifecycle
churn, but smooth series are exactly the regime pre-training covers; and VN1
charges |bias| in full (floor run `09fdb04e06`: VN1 = WAPE + |bias|), which
punishes median-quantile forecasts on 53% intermittent series.
Cost: ~20–60 min for 4 origins × 15,053 series on the M1 Max; 1 gate eval.
Risk / failure mode: median collapses to 0 on intermittent series → pooled
under-bias in the −15% range (the `accafab14a` failure magnitude). Forecast
aggregation (mean-of-quantiles vs median) is deliberately NOT tuned this
cycle — one change per experiment; register it as the follow-up lever.

### Option B — TimesFM 2.x zero-shot (Stage 1) — co-runnable

**Hypothesis:** same question with the other flagship architecture;
TimesFM's point output is mean-flavored rather than a median quantile, so it
should carry less zero-collapse under-bias on intermittent series, at the
cost of more WAPE on spiky ones. Two independent zero-shot answers make the
paper's Stage 1 claim robust to "you picked the weak checkpoint".

Config diff:
- New `configs/timesfm_v2.yaml` — `model: {name: timesfm, params:
  {checkpoint: google/timesfm-2.0-500m-pytorch, batch_size: 256,
  clip_negative: true}}`.
- New family `src/models/timesfm.py` + `REGISTRY["timesfm"]`.
- Env: `timesfm` (torch backend) — install friction on macOS arm64 is the
  known unknown.

Expected impact: pooled VN1 **0.58–0.70**, same reasoning as A.
Cost: similar runtime; 1 gate eval.
Risk / failure mode: arm64 packaging pain (timebox it — see "change my
mind"); 500M params may be slow on CPU if MPS falls over.

### Option C — global_lgbm with n_anchors ≥ 52 — **OFF-ROADMAP**

The registered "next lever" from cycle 2 (`7548b6e864` January failure:
training window must span a full seasonal cycle). It advances no open stage
— Stage 0 is DONE — so per the constitution it is flagged **OFF-ROADMAP**
and runs only if the human explicitly accepts that label. Cheapest option
on the table (pure config diff, no new code) and it strengthens the rival
deep learning must beat in Stage 2. Not recommended *this* cycle; noted so
the beam doesn't smuggle it in.

## Recommendation

**Option A, with Option B co-run as one beam under this single approval.**
Stage 1 names both models — both must run eventually for the paper, they are
independent, and neither informs the other's config, so serializing them
buys nothing and spends an extra human approval. Each is gated separately
(2 gate evals → 12/25). Option A is the primary: better-documented retail
zero-shot behavior, simpler install; if only one is approved, run A.

Option C stays parked: crowning a stronger GBM mid-stage would move the
yardstick while Stage 1 is being measured against it. Run it, if at all, as
its own explicitly-approved cycle before Stage 2 comparisons begin.

## What would change my mind

- If either package cannot produce a forecast on arm64 within ~1 hour of
  setup effort, swap that option's checkpoint family (e.g. chronos-t5-small)
  rather than debugging an install — the hypothesis is about zero-shot
  capability, not packaging.
- If a smoke run (first origin) shows pooled bias < −20% from median
  zero-collapse, stop before burning the full backtest and bring the
  mean-vs-median aggregation choice back to the human as an amended memo.
- If runtime exceeds ~2 h per origin, downgrade to the small checkpoint as a
  new config/run-id — never subsample series (harness fence).

## Registered predictions (to score at /reflect)

- Chronos-Bolt pooled VN1 0.58–0.68; smooth 0.50–0.56; pooled bias negative,
  −5% to −18%.
- TimesFM pooled VN1 0.58–0.70; less negative bias than Chronos on
  intermittent.
- Neither passes the gate vs `7548b6e864`; both beat seasonal_naive
  (1.0211, `e663fe200e`); at least one beats naive (0.5830) on smooth.

**STOP — awaiting human approval, amendment, or rejection.**

---

## Outcome (appended 2026-08-28 after the approved A+B beam ran)

Human approved the A+B beam. Runs: Chronos-Bolt `4248b74828`, TimesFM 2.5
`e8a11552ec` (checkpoint swapped from 2.0-500m to 2.5-200m under this memo's
packaging clause — installed timesfm 2.0.2 only exposes the 2.5 API).
Runtime: ~4.5 min (Chronos) / ~17 min (TimesFM) for the full 4-origin
backtest on the M1 Max — far inside the guardrails.

| | champion `7548b6e864` | chronos `4248b74828` | timesfm `e8a11552ec` |
|---|---|---|---|
| pooled VN1 | **0.5622** | 0.5878 | 0.6512 |
| pooled WAPE | 0.5542 | 0.5134 | **0.4612** |
| pooled bias | **−0.8%** | −7.4% | −19.0% |
| smooth VN1 | 0.5224 | 0.3717 | **0.3381** |
| origin 2023-01-09 | 0.8916 | **0.5994** | 0.5735* |

*best January origin in the ledger.

- Neither challenger beats the champion pooled → ROADMAP registered
  prediction 1 confirmed in direction. Neither is promotable; gate runs
  were skipped to preserve budget (still 10/25) — pooled VN1 losses of
  +0.026 / +0.089 cannot pass R1, no verdict needed to see it.
- BUT: both models beat every ledger model on smooth (0.3717 / 0.3381 vs
  prior best 0.3884 MA13), and both demolish the champion at the January
  origin (0.599 / 0.574 vs 0.892) — pre-training covers seasonal turns the
  GBM's 24-anchor window never saw. "Surprisingly close on smooth"
  under-called it: they are outright best.
- Both out-WAPE the champion; both lose VN1 entirely on the bias term.
  TimesFM posts the best pooled WAPE ever recorded (0.4612) and is sunk by
  −19% under-forecast concentrated in intermittent (−23%) and lumpy (−31%).
- Prediction scoring: Chronos pooled ✓ (0.5878 in 0.58–0.68), bias ✓
  (−7.4% in −5..−18); smooth beat the predicted 0.50–0.56 by a wide margin.
  TimesFM pooled ✓ (0.6512 in 0.58–0.70). REFUTED: "TimesFM's point output
  carries less zero-collapse bias than Chronos' median on intermittent" —
  it carries far MORE (−23.2% vs −6.6%).
- Stage 3 signal: champion and foundation models fail in different places
  (champion: seasonal turns; FMs: intermittent/lumpy bias) — exactly the
  complementarity a blend needs.
