# Decision memo — Cycle 1: first challenger to naive_last

## Context
Champion: `naive_last` (`2a69e4f986`), pooled VN1 0.583, bias −5.2%.
Gate budget: 4/25 evaluations used. Ledger: 6 baseline runs.

## Evidence (from /diagnose, all pooled over 4 origins)
1. **The champion is already beaten on the biggest segment.** On *smooth*
   series (32.1% of volume) MA13 scores **0.3884** vs the champion's 0.5106
   (`3891f3c0fc` vs `2a69e4f986`). The champion wins every other segment.
   That is a confirmed, un-captured win sitting in the ledger.
2. **Holiday-window instability.** The champion's worst origin by far is
   2022-10-10 (VN1 0.852 vs 0.583 pooled) with bias **−21.9%**: last-week
   naive misses the Nov–Dec uplift. This matters beyond validation —
   **Phase 1 is exactly an Oct→Jan window**, so the final eval sits in the
   champion's known blind spot.
3. **Lumpy remains the largest open segment** (20.4% of volume, VN1 0.703,
   headroom 0.025): no baseline models demand-size separately from
   demand-timing, which is the textbook gap Croston-family methods target.
4. Sparse tail ("insufficient", VN1 1.98) confirmed as low value: −97% bias
   because these series barely sell; 3.9% of volume. Standing prior holds.

## Options
**A. Segment router (recommended)** — new composite model: route *smooth*
series to MA13, everything else to naive_last, segment computed on training
data only. Config: `configs/router_smooth_ma13.yaml` + a ~30-line
`SegmentRouter` model class (registry addition, no frozen files).
Expected: smooth improves 0.51→0.39, pooled ≈ **0.545** (−0.038).
Cost: minutes. Risk: low — both components already validated; worst case the
gate fails and we learn the segment boundary doesn't transfer across origins.

**B. Croston / SBA / TSB via statsforecast** — proper intermittent-demand
methods aimed at lumpy/intermittent (50% of volume combined). New dependency
+ wrapper class. Expected: lumpy 0.70 → 0.63–0.67 (uncertain); pooled effect
likely +0.01–0.03. Cost: an hour incl. dependency. Risk: Croston's known
positive bias can hurt the VN1 bias term; TSB mitigates.

**C. Holiday-aware naive** — naive_last scaled by a week-of-year uplift
index estimated per client from prior years. Directly targets Evidence #2
and de-risks the Phase-1 window. Expected: big gain on the Oct origin
(−22% bias → near 0), pooled +0.01–0.02. Cost: small. Risk: only two prior
holiday seasons to estimate from; index must be damped.

## Recommendation
**A**, alone (one change per experiment). It converts already-proven ledger
evidence into pooled improvement with near-zero risk, and exercises the full
propose→experiment→gate→promote cycle end-to-end cheaply. Queue C as cycle 2
(strategically important because of Phase 1's window), B as cycle 3.

## What would change my mind
If MA13's smooth-segment win were driven by a single origin I would drop A —
checked: it is not. Per-origin smooth-only VN1 (recomputed): MA13 wins 3 of 4
origins, decisively where the champion is weakest (0.416 vs 0.640 on
2022-10-10, 0.413 vs 0.549 on 2023-01-09) and is a statistical tie on the
one it loses (0.342 vs 0.336 on 2023-04-10). If you want to prioritize final-eval robustness over leaderboard
position, C jumps ahead of A.

---
## Outcome addendum (post-gate)
Beam ran (A + C, human-approved). **Both FAIL the gate.** Champion unchanged.
- A `7a52b16fe3`: 0.5853 vs 0.5830 — WAPE gain real, eaten by the global
  bias term. Prediction (0.545) badly miscalibrated: assumed segment
  additivity. See learnings entry; new prior recorded.
- C `accafab14a`: 0.6956 — mechanism refuted (churn-contaminated indices);
  DEAD-END recorded. Holiday weakness remains open.
Gate budget after cycle: 6/25.
