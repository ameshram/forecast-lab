# Decision memo — Cycle 2: attack the bias at its root

## Context
Champion: `naive_last` (`2a69e4f986`), pooled VN1 0.583. Gate budget: 6/25.
Cycle 1: both challengers failed; priors updated (VN1 not
segment-separable; client-level woy re-anchoring DEAD-END).

## Evidence
1. **The champion's bias is seasonal, not constant.** Per-origin net bias:
   −21.9% (Oct origin, window contains Nov–Dec), **+12.9%** (Jan origin,
   naive anchored on inflated December weeks), −6.7% / −3.9% (mid-year).
   Same mechanism as the holiday blind spot: it under-forecasts into the
   peak and over-forecasts out of it. A static or trailing-window global
   multiplier cannot fix a sign-flipping bias — it lags exactly at the
   turns.
2. **The router's WAPE gain is real and stable**: better WAPE at 3 of 4
   origins (pooled 0.513 vs 0.531), lost only through the bias term
   (`7a52b16fe3`). Whatever fixes bias should be built ON the router.
3. Lumpy remains open (VN1 0.70, 20% of volume); nothing yet models demand
   size separately from timing.

## Options
**A. BiasScaled(router) — trailing-ratio debias wrapper** *(cheap probe)*
Wrapper model: run the base (cycle-1 router) on an internal train-only
holdout (last 13 train weeks), compute damped ratio Σactual/Σforecast, clip
[0.85, 1.2], scale the refit base's forecasts. Registry: `bias_scaled`.
Expected: fixes the −4…−7% mid-year bias, but WILL lag the seasonal flips
(Evidence #1) — predicted pooled **0.555–0.575**, i.e. beats 0.583 modestly
if mid-year gains outweigh the Oct/Jan lag. Cost: ~40 lines + config. Risk:
ratio chases noise; damping/clip mitigate. Worth running mostly because the
same wrapper is reusable for any future base (incl. GBM) and this cleanly
tests whether trailing debiasing works at all here. Co-runnable.

**B. TSB via statsforecast for lumpy/intermittent** *(queue for cycle 3)*
Proper intermittent-demand method as a router route. Better tried after C,
whose per-segment behavior will tell us where TSB is still needed.

**C. Global LightGBM, Tweedie objective** *(the main event)*
One model over all series; features: lags (1–4, 8, 13, 26, 52), rolling
means/stds, zero-run length, weeks-since-last-sale, weeks-since-launch,
forward-filled price + price change, **week-of-year (holiday inside the
model — the churn-robust seasonality the DEAD-END entry calls for)**,
Client/Warehouse categoricals. Direct 13-step forecasting with horizon as a
feature; trains inside `fit_predict` on each origin's train slice only.
Expected: attacks Evidence #1 (calendar features), #3 (Tweedie handles
zero-inflation), and the level errors at once — predicted pooled
**0.48–0.54** (VN1-winner recipes and M5 evidence; naive-beating margins of
8–15% are typical for well-featured global GBMs on this data shape).
Cost: `lightgbm` dependency + a features module + model class (~150 lines);
~4 origin-fits of a few minutes each on the M1 Max. Risk: most moving
parts — feature leakage is the classic failure; mitigated because features
are computed inside the harness's train slice per origin, and the
leakage-guard test pattern extends to the feature builder. Co-runnable.

## Recommendation
**Beam A + C.** C is the priority (all three evidence lines converge on
in-model seasonality + a stronger learner); A is a near-free probe that
banks the router's WAPE gain if trailing debiasing works, and its wrapper
is reusable regardless. B waits for C's segment breakdown.

## Predictions (for /reflect calibration)
- A: pooled 0.555–0.575; mid-year origin bias → within ±3%; Oct/Jan bias
  barely improved.
- C: pooled 0.48–0.54; Oct-origin VN1 < 0.75; Jan-origin bias inside ±8%;
  lumpy < 0.68.

## What would change my mind
If you want zero new dependencies this cycle, A alone still moves us — but
the evidence says the ceiling without in-model seasonality is low. If C's
first fit shows train-time leakage red flags (implausibly low backtest
error), I stop and audit features before trusting any number.

---
## Outcome addendum (post-gate)
Beam ran (A + C, human-approved).
- A `2c050a9d24` bias_scaled(router): **0.5642** — inside the predicted
  0.555–0.575 band (calibration good). Gate **FAIL** on R2 (2/4 origins):
  the trailing ratio scaled up into the post-holiday January origin
  (+29% bias there), the lag-at-the-turns failure predicted in this memo.
- C `7548b6e864` global_lgbm v1: **0.5622** — below the predicted 0.48–0.54
  (over-optimistic; error on the safe side, no leakage flag). Gate **PASS**
  (3/4 origins, p=0.0305, no segment regressions, bias −0.8%). Promotion
  awaiting human confirm. Key finding: with n_anchors=24 the training rows
  never contain a post-holiday decline — extending anchors ≥52 weeks is the
  registered Cycle-3 lever.
Gate budget after cycle: 9/25. Reflect entries: experiments/learnings.md.
