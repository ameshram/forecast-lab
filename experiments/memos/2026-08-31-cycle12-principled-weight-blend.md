# Decision memo — Cycle 12: the principled-weight blend

Date 2026-08-31 · Cycle 12 · **Stage 3 (continuation)** — the
"principled-weight (holdout-fit, not a sweep) follow-up on the blend" that the
North Star has carried as the registered open lever since Cycle 9. Advances
Stage 3's blend question to its natural endpoint: are equal weights leaving
anything on the table, and can a component that equal weights *arithmetically
cannot use* contribute under fitted weights?

## Context

- Champion: `4f9b475ead` — three-way equal blend (GBM-56 + CatBoost-56 +
  Bolt-qmean), pooled VN1 **0.5137** / WAPE 0.4893 / bias −2.4%; lumpy 0.651
  (+1.1%); origins 0.692/0.545/0.461/0.466.
- Gate budget: **16/25** (gate_log.jsonl, 16 entries after the Cycle-11
  promotion). A promotion here costs 2 more (count per gate invocation).
- Stopping rule: quiet-cycle counter is 0 (Cycle 11 promoted). This memo is
  the last diagnosed live lever; if it fails or is declined, the expected
  path is two deliberate quiet cycles → PHASE1_SIGNOFF → final exam.

## Evidence (Cycle-12 diagnosis, all from `experiments/runs/<id>/`)

- **At equal weights there is no model left to borrow from**: the champion or
  its own ingredients are best-in-ledger on lumpy (0.651), intermittent
  (0.503), erratic (0.519), and the January origin (0.545). The one segment
  where outsiders win is smooth: TimesFM `e8a11552ec` 0.338, Moirai
  `7c77d54b6f` 0.345, **Chronos-2-median `461e681f0d` 0.351** vs champion
  0.407 — worth up to +0.022 pooled if transferable.
- **Linear bias arithmetic (5-for-5 dead-centre across Cycles 9/11) bars the
  naive fix**: adding TimesFM at ¼ weight computes pooled bias −6.6% and
  lumpy −6.9% (breaks cancellation); adding Chronos-2-median at ¼ computes
  lumpy **+0.8% (cancellation survives)** but pooled bias −4.2% (worse than
  −2.4%). Chronos-2-median is the uniquely interesting candidate — neutral
  lumpy bias (−0.3%), elite smooth (0.351), strong January heritage (0.569
  as a standalone) — but its −9.4% pooled under-forecast means **only a
  weight below ¼ can use it. Equal-weight configs cannot express that;
  fitted weights can.** This is the precise gap between "the well is dry"
  and "one lever remains."
- Champion origin flag: Oct-2022 is 34.7% worse than pooled (0.692) —
  structural (earliest origin), thin actionability, not this memo's target.
- Prior AGAINST trailing fits: `2c050a9d24` — a trailing level-ratio scaled
  the wrong way into the January turn (+29% bias there). A weight fit is
  exposed to a milder version (it weights *relative model quality* measured
  on a trailing window, not a level), but the risk is real and is
  pre-registered below as the named failure mode with a detection rule.

## The fitting protocol (pre-registered, leak-free, not a sweep)

New composite `WeightFitBlend` (`src/models/weight_fit.py`, registry name
`weight_fit_blend`; contract test + an explicit leakage test in
`tests/`). Inside each origin's `fit_predict(train, future_ds)`:

1. Inner split: hold out the LAST 13 weeks of `train` (the BiasScaled
   pattern — the holdout is inside the training slice; Phase-boundary
   untouched; nothing after the origin is ever seen).
2. Fit every component on `train − holdout`; forecast the holdout.
3. Solve for weights on the simplex (w ≥ 0, Σw = 1) minimizing **holdout
   VN1** — the target metric, convex in w (sum of absolute values of linear
   functions) — with one deterministic scipy SLSQP call from the
   equal-weight start. ONE objective, ONE holdout, ONE solve. No grids, no
   repeats, no alternative objectives. Fitted weights are logged.
4. Refit all components on the FULL `train`; blend with the fitted weights.

Weights may differ per origin (each origin fits within its own training
slice) — that is the design, not a leak. Anti-shopping fence: if this run
fails its gate, the protocol is NOT re-run with a different holdout length,
objective, or component set without a new memo naming why.

## Options

**Option A — 4-way weight-fit: {GBM-56, CatBoost-56, Bolt-qmean,
Chronos-2-median}. RECOMMENDED.** Hypothesis: fitted weights admit
Chronos-2-median at the sub-¼ weight its bias profile demands, buying smooth
and turn strength while preserving the lumpy cancellation. Config:
`configs/blend_weightfit4.yaml`. Expected pooled **0.503–0.520** (wider than
a pure composite range: the fit adds holdout-generalization noise; must beat
0.5137 to matter). Cost: each component fitted twice per origin → ≈2× the
champion's runtime, ~1.5–2 h CPU. Main risk: the trailing-window fit
mis-weights at the January origin (the `2c050a9d24` mode) — detection:
Jan-origin VN1 worse than the champion's 0.545 by >5%.

**Option B — 3-way weight-fit on the champion's own components.** Hypothesis:
equal weights are already near-optimal for the current trio (the Cycle-9
diagnosis found the 2-way weight lever flat), so this mostly measures
fit-noise. Cleaner one-change control (no new component), but the diagnosis
already tells us the interesting information lives in whether a NEW component
can enter — B answers a question we largely know. Config:
`configs/blend_weightfit3.yaml`. Cost ~1.5 h. NOT co-run by default: it
doubles compute to confirm an expected null; run only if A's fitted weights
come back ≈ equal (which would make A itself the control).

**Option C — decline the lever; bank the champion.** Start the two quiet
cycles toward PHASE1_SIGNOFF now. Zero cost, zero risk, and defensible: the
champion is best-in-ledger nearly everywhere. Considered and not recommended
only because A is cheap, pre-registered, ROADMAP-anticipated, and the last
diagnosed source of genuine headroom (+0.02 smooth transfer upside).

## Recommendation

**Option A alone** (no beam — B is contingent, not independent). It is the
smallest change that tests the one remaining hypothesis; the composite family
carries it as mostly-config (one new, well-fenced composite class); and its
failure is informative either way: fitted-weights≈equal ⇒ the weight lever is
flat and the dry-well conclusion is evidence, not assumption; chronos2 weight
→0 ⇒ same conclusion for the component story.

## Registered predictions (score at /reflect)

1. Run completes NaN-free, full coverage; fitted weights lie on the simplex
   and are logged per origin.
2. Chronos-2-median's fitted weight lands **0.10–0.30** at ≥3 of 4 origins
   (its value prices in below ¼); if it fits →0 everywhere, the dry-well
   conclusion is confirmed.
3. Pooled lands **0.503–0.520**; probability of beating 0.5137 is a genuine
   coin flip — this is NOT a tight-composite prediction because the fit adds
   a new noise source (holdout generalization).
4. The named failure mode stays contained: Jan-origin VN1 ≤ 0.573
   (champion 0.545 + 5%).
5. Gate spend ≤2, counted per gate invocation (0 if pooled ≥ 0.5137).

## What would change my mind

- Fitted weights ≈ equal (all in 0.20–0.30) AND pooled within ±0.003 of the
  champion → the weight lever is flat; recommend Option C immediately, no
  variants.
- Jan-origin blowup past 0.573 → the trailing-fit prior generalizes to
  weight fitting; record as DEAD-END for trailing-fit composites and stop.
- A pooled win < 0.5137 with sane per-origin behaviour → gate it.

**STOP — awaiting human approval, amendment, or rejection.**

---

## ADDENDUM — realized results (appended at /reflect, 2026-09-01)

Human approved Option A; run `b12f62f304`. **Hypothesis REFUTED** — pooled
**0.5383** vs champion 0.5137 (1/4 origins). No gate case (R1 computed-fail),
0 evals spent, budget stays 16/25. **Cycle 12 = quiet cycle 1 of 2.**

Fitted weights [gbm56, cat56, qmean, chronos2-med] and realized origins:

| Origin | weights | holdout (eq→fit) | realized | champion |
|---|---|---|---|---|
| Oct-2022 | [.13,.23,.65,.00] | .503→.488 | 0.730 | 0.692 |
| Jan-2023 | [.42,.58,.00,.00] | .712→.667 | **0.493** | 0.545 |
| Apr-2023 | [.67,.18,.00,.14] | .537→.470 | 0.485 | 0.461 |
| Jul-2023 | [.11,.22,.68,.00] | .468→.454 | 0.468 | 0.466 |

Predictions: #1 ✓ clean · #2 alternative fired (chronos2 ~0 at 3/4 —
dry well MEASURED) · #3 missed above range · #4 ✓ Jan contained (0.493,
best January in the ledger — the all-GBM fit was right, not the 2c050a9d24
mode) · #5 ✓ 0 spent.

Lesson recorded (CLAUDE.md + learnings): DEAD-END for trailing-holdout
weight fitting — improves the holdout every time, doesn't survive the
13-week jump. Equal weights are the robust choice; Stage 3 closed with
evidence. Per this memo's own fence: no protocol variants.
