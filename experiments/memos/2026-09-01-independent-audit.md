# Independent audit — Forecast Lab (pre-paper)

Date 2026-09-01 · Auditor: independent (did not build the project) · Scope:
review / validation only. **No ledger writes, no new experiments, no Phase 1
values read, no edits to any project file except this memo.** Every
reproduction below was run in-process through the frozen harness
(`run_backtest`) *without* calling `record_run`, so no ledger row was added
and no run directory was touched (see finding 4-G for why that mattered).
Scratch scripts and logs live in the session scratchpad
(`audit_a.py`, `audit_b.py`, `repro.py`, `claims.py`, `ledger_history.py`,
`csv_vs_parquet.py`, `verify_numbers.py`, `timesfm3_check.py`).

Verdict legend: **VERIFIED** · **ISSUE (blocker | major | minor)** ·
**NOT VERIFIABLE**.

---

## 1. Claims tracing — VERIFIED, with ISSUES (1 major, several minor)

**Method.** Automated pass over CLAUDE.md, README.md, ROADMAP.md,
`experiments/learnings.md`, all 16 memos and the dashboard: every score-like
number or percentage appearing in the same paragraph as a run-id was checked
against that run's ledger row, `per_origin.csv`, `segments.csv` and
`volume_segments.csv` (752 numbers checked; 522 matched automatically; the
230 flagged were reviewed by hand). The dashboard's embedded JSON (38 runs,
per-origin and per-segment) was compared field-by-field to the artifacts:
**0 mismatches**. The README leaderboard (18 rows) matches the ledger to
every printed digit. `final_eval_report.json` numbers quoted in CLAUDE.md,
learnings and the dashboard (0.5346 / 0.5141 / +2.05% / lumpy −0.6%) match
the report.

**Every hand-reviewed flag resolved to one of:** (i) a number belonging to a
run named elsewhere in the same document (all verified, e.g. gbm56 lumpy
−10.5% = `d047d88915` −0.1054; qmean +15.2% = `828187da8a` +0.1522; CatBoost
Oct 0.677 / Jan 0.534; TSB intermittent 0.589 vs Croston 1.150; Chronos-2
qmean lumpy +157% = +1.568; TimesFM-3 qmean Jan 2.73 / +114%); (ii) a
registered prediction range; (iii) an arithmetic derivation (0.7159 =
0.6818 × 1.05; 0.712 = 0.678 × 1.05; router "≈0.562" is labelled as
arithmetic); or (iv) a **scratchpad-only diagnostic** (below).

**ISSUE (major) — README.md is stale and contradicts the rest of the repo.**
Its leaderboard names `1bb3cb32c6` (0.5192) as champion with "20 runs", and
"Status & next" describes the pre-Cycle-10 state ("Next: a principled-weight
follow-up…"). CLAUDE.md, the dashboard and learnings say 38 runs, champion
`4f9b475ead` 0.5137, final exam done. A reader landing on the README gets a
different champion and a different project status than the paper will cite.

**ISSUE (minor) — numbers that do NOT trace to any artifact under
`experiments/`** (they violate the project's own anti-drift rule; each must
either be reproduced into an artifact or be dropped from the paper):
- Cycle 7/8 bias-ratio ablation trajectories (1.075 / 1.106 / 1.000 / 1.127 /
  1.262; 1.324 / 1.301 / 1.522; 1.378). These are the sole evidence for the
  "covariates destabilise the deep net" prior and for the Cycle-8 no-launch
  decision. Scratchpad only.
- Cycle 6-B mechanism diagnostics (Spearman 0.867, forecast CV 6.02 vs 6.15,
  "47.6 phantom units on 1,497 actual-zero series", "0% vs 9.9% zero
  forecasts"). Per-cell forecasts are not persisted, so these cannot be
  recomputed from artifacts.
- Cycle 12 fitted weights per origin ([.13,.23,.65,.00] etc.) and holdout
  VN1s (0.503→0.488 …): printed to stdout only. The "all-GBM January mix
  0.42/0.58" story is therefore untraceable; the January VN1 0.4931 itself
  is in `b12f62f304/per_origin.csv` and is fine.
- FM readout-semantics sums in run notes (Chronos-2 "127,003 vs 133,687";
  TimesFM-3 "1,291,845 byte-identical") and the TimesFM-3 smoke ratios
  0.920 / 0.965; the smoke script is not in the repo. I re-verified the
  two semantics claims independently (see §7).
- Cycle-9 feasibility smoke figures ("195,689 rows, 1.7 min", "~7 min").
- Text slips: learnings says the champion lost April "by 0.002" — actual
  0.4608 vs 0.4595 = 0.0013. CLAUDE.md quotes "Moirai 0.478 beats even the
  champion's 0.493" — 0.493 was the *previous* champion's WAPE (current
  champion 0.4893); still true, but stale wording. Learnings/memos quote
  the max-y as 13,669 in places; the data value is 13,681 (already noted in
  the Cycle-10 memo).

**Minor precision note.** CLAUDE.md calls `3330b31053` "computed-R4-blocked";
the recomputed gate shows it also fails R2 (2/4 origins) and R3 (p = 0.14).
The R4 story is true but incomplete.

## 2. Leakage audit — VERIFIED (no look-ahead found), with CAVEATS

Read in full: `src/data.py`, `src/backtest.py`, `src/features.py`,
`src/models/composite.py`, `src/models/weight_fit.py`, `src/models/deep.py`
(covariate builders), all FM wrappers, `src/run_experiment.py`,
`src/final_eval.py`.

- **Phase split.** `split_phases` cuts at 2023-10-09; `run_experiment`
  passes only Phase 0 into `validation_origins` and `run_backtest`.
- **Origins (recomputed from the data):** 2022-10-10, 2023-01-09,
  2023-04-10, 2023-07-10; 13-week windows are non-overlapping; the last
  window ends **2023-10-02 = the last Phase-0 week**, one week before Phase 1;
  118 training weeks at the earliest origin. Identical origin set in all 38
  runs' `per_origin.csv`.
- **Backtest.** `train = df[df.ds < origin]` (strict); actuals come from the
  Phase-0 window; Syntetos–Boylan classes and volume deciles are computed on
  each origin's training slice only.
- **Features (GBM family).** For anchor column `ai`, lags use columns
  `ai−(k−1)` (so "lag1" is the anchor week itself — a naming quirk, not a
  leak), rolling windows `[ai−w+1, ai]`, price forward-filled left-to-right
  only, weeks-since-last-sale and age from cumulative past columns; targets
  are `ai+h` and are required to lie inside the training matrix. The
  prediction row anchors at the last training week, so train/predict
  alignment is consistent. The existing poison test
  (`test_feature_builder_uses_only_past_columns`) passes.
- **Composites.** `Blend`, `SegmentRouter`, `BiasScaled` and
  `WeightFitBlend` re-fit every component inside each origin on that
  origin's train slice; the internal holdouts (BiasScaled, WeightFitBlend)
  are the last 13 weeks *of the train slice*. `test_no_leakage` (spy on
  every component call) passes.
- **Deep-net covariates.** Future covariates are calendar/age (deterministic
  from train); historical covariates are causal (ffill, shift(4),
  cumulative wsls). No look-ahead.
- **Final exam.** Fits on all of Phase 0, scores all 13 Phase-1 weeks once,
  writes aggregates only; no feedback path.

**CAVEAT (major) — zero-shot pretraining contamination is the one leakage
class the harness cannot see, and the project's own rule was not followed.**
The champion contains `amazon/chronos-bolt-base` (released 2024-11-26,
corpus undocumented) — released *after* VN1 Phase-1 actuals became public
(2024-10-17). The project's contamination memo (2026-08-28) judged the risk
low on timing grounds and adopted a standing rule: *"if any FM component is
part of the model that runs the one-shot Phase 1 evaluation, run an
empirical memorization probe first."* No such probe exists in `experiments/`
and the exam ran without it. The five newer FMs in the zoo (Chronos-2,
Moirai-2, TTM r2, TimesFM-3) all post-date VN1's public release as well; they
are not in the champion, but zoo statements about them carry the same
unquantified risk.

**CAVEAT (minor, governance).** Nothing technically prevents reading Phase 1
during experimentation: `load_panel()` loads both phases into every process,
and the hook only blocks a *command string* containing `final_eval` (and
only until the sign-off file exists). The lock is upheld by `split_phases`
in `run_experiment` and by discipline. I found no evidence of misuse (all 38
runs' artifacts are consistent with Phase-0-only windows), but the paper
should describe the lock as procedural, not enforced.

## 3. Metric correctness — VERIFIED

- `vn1_score = (Σ|F−A| + |ΣF−ΣA|) / ΣA`, pooled over all series-week cells —
  the official VN1 competition score. `bias_pct` sign convention (+ = over)
  and the zero-denominator → NaN convention are consistent between
  `metrics.py` and `gates._vn1_of`.
- `score_by_segment`: volume share = segment ΣA / total ΣA; sums to 1.000000
  in all 38 runs.
- **Independent recomputation** of pooled VN1 / WAPE / bias from
  `series_scores.parquet` for all 38 runs: max |ledger − recomputed| =
  3.9e-8 (XGBoost, float32 rounding); all others ≤ 8e-16.
- Pooled scores reconstructed from `per_origin.csv` (WAPE and bias weighted
  by the recomputed per-window actual volumes 3,451,759 / 3,213,544 /
  3,331,236 / 3,508,827) match the ledger to 4 dp for every run tested.
- Final report: `vn1 = wape + |bias|` holds for the pooled row and all five
  segment rows (as it must for this metric).

## 4. Reproducibility — VERIFIED for deterministic configs; ISSUES (1 major, several caveats)

Reproductions were run from each run's *stored* `config.yaml` via
`run_backtest`, in fresh processes with `OMP_NUM_THREADS=1
KMP_DUPLICATE_LIB_OK=TRUE`, comparing pooled, per-origin, per-segment and
per-series aggregates to the stored artifacts:

| run | model | pooled diff | per-series yhat_sum max diff |
|---|---|---|---|
| `2a69e4f986` | naive_last | 0.0 | 0.0 |
| `e663fe200e` | seasonal_naive 52 | 0.0 | 0.0 |
| `771488878b` | Croston-SBA (statsforecast) | 0.0 | 0.0 |
| `43f080e588` | TSB (statsforecast) | 0.0 | 0.0 |
| `88447a52f4` | AutoTheta (statsforecast, season 52, n_jobs=-1) | 0.0 | 0.0 |

All five reproduce exactly (782,756 forecast cells, 15,053 series, no NaN /
negative / inf; per-origin and per-segment diffs ≤ 2e-16). **Nothing was
appended to the ledger** — see 4-G.

**Torch / GBM models (documented non-reproducible) verified internally:**
- The champion blend `4f9b475ead` per-series `yhat_sum` equals the ⅓-mean of
  its three component runs' `yhat_sum` (`d047d88915`, `523f9c23da`,
  `828187da8a`) to max 2.4e-6 (portfolio-level relative 2e-12). LightGBM-56,
  CatBoost-56 and Chronos-Bolt-qmean therefore reproduced *identically*
  inside the blend run three days after their standalone runs. Blend segment
  biases equal the component means exactly (lumpy +0.0112 = mean of −0.1054,
  −0.0133, +0.1522). Same for `1bb3cb32c6` (lumpy +0.0234 = mean of the two).
- `524453e7d8` (Bolt point=mean) and `4248b74828` (median): series_scores
  byte-identical, confirming the "mean == median" no-op. `13811e908e` vs
  `3891f3c0fc` identical likewise.

**ISSUE 4-G (major) — the ledger is NOT append-only.** `ledger.record_run`
computes `run_id = sha1(config)`, then *drops any existing row with that
run_id* and *overwrites* `experiments/runs/<run_id>/`. Re-running any config
silently replaces its history. This is not hypothetical: the six baseline
rows carry ledger timestamps 20:49:34–20:50:08 UTC on 2026-08-28, while
`champion.json` (initial champion at 20:33:49) and the
`dashboard_data.json` snapshot (20:34 UTC) already referenced those runs —
the original rows were overwritten by a re-run 16 minutes later (values
identical because the baselines are deterministic, so no scientific damage,
but the "append-only ledger" claim in CLAUDE.md, README and the hook
docstring is false in code and in practice). Corollary: the instruction that
"reruns append ledger rows" is wrong — a rerun of the champion config would
destroy the champion's artifacts. Also `run_id` hashes the `notes` field,
so editing a config's notes forks a new run_id for the same experiment.

**Further reproducibility caveats:**
- `run_id` hashes config only, not code (acknowledged debt in learnings);
  wrappers changed across cycles.
- HF checkpoints are unpinned in configs (the contamination memo's "pin
  revisions" action was never done). Local cache revisions at audit time:
  chronos-bolt-base `5d9f166d…`, chronos-2 `29ec3766…`, timesfm-2.5
  `1d952420…`, timesfm-3.0 `c7190707…`, moirai-2.0-R-small `30f43ff0…`.
  Record these in the paper's appendix.
- `timesfm` was upgraded 2.0.2 → 3.0.0 after `e8a11552ec`; Moirai and TTM
  ran from throw-away venvs that no longer exist and have no lock file.
- Per-cell forecasts (`forecasts.parquet`) are never persisted, so nothing
  finer than per-series/per-origin aggregates can be re-analysed without
  re-running.
- **Version control:** git HEAD (2026-08-31 18:43 local) holds a 20-row
  ledger. The 18 Cycle-10–13 runs, four memos, four model wrappers, seven
  test files, the `4f9b475ead` promotion, `PHASE1_SIGNOFF` and
  `final_eval_report.json` are all **uncommitted** (65 changed/untracked
  paths). As it stands, the repository the paper will cite does not contain
  the paper's evidence.

## 5. Gate & promotion integrity — VERIFIED, with notes

- `gate_log.jsonl` has 16 lines = the claimed 16/25.
- Every promotion in `champion.json` has a PASS line at the exact
  `promoted_at` second (`7548b6e864` 21:53:44; `1bb3cb32c6` 17:33:56;
  `4f9b475ead` 23:33:22), each preceded by an informational PASS. The initial
  champion `2a69e4f986` was written directly via `set_champion` with a custom
  reason (not through a gate) — acceptable bootstrap, but not "mechanical".
- **Recomputed `gates.evaluate()` (no log write) for the three promotions**:
  identical rules, origin wins and p-values to the recorded reasons
  (3/4 p=0.0305; 4/4 p=0.0; 3/4 p=0.0). Recomputed FAILs match the memos
  (`828187da8a` R4 lumpy 0.9254 vs 0.6818; `d047d88915` R4 0.7232; `3330b31053`
  R2/R3/R4; `2c050a9d24` R2 only; `c87a16ebf7`, `7a52b16fe3`, `accafab14a`).
- Composition of the 16: two same-second duplicates (`c87a16ebf7` ×2;
  `2a69e4f986` vs the zero floor ×2), one self-evaluation (`1bb3cb32c6` vs
  itself), three promote-confirm re-runs. Counting per invocation is
  conservative and is documented.
- R4 uses the champion's volume shares; segment membership is identical
  across runs (classification depends only on the origin's training data),
  so segment comparisons are like-for-like.
- Code note: `evaluate()` inner-joins on series and computes the pooled
  scores on the *unjoined* frames; harmless today (15,053 = 15,053 in every
  pair) but would silently diverge if a run ever dropped series.

## 6. Final-exam consistency — VERIFIED (code inspection + artifacts only)

- `champion_run` = `4f9b475ead`; the report's `model` block is
  JSON-canonically identical to `experiments/runs/4f9b475ead/config.yaml`
  and to `configs/blend_gbm_cat_qmean.yaml`.
- Per-class `n_series` (7,664 / 937 / 4,397 / 782 / 1,273) sum to 15,053 and
  equal my own Syntetos–Boylan classification of the full Phase 0 exactly;
  volume shares sum to 1.0; the metric identity holds in every row.
- Timeline: last `champion.json` write 2026-08-31 23:33:22 UTC → last
  validation run 2026-09-01 20:52:16 UTC (`8582538c28`, TimesFM-3 qmean,
  approved as a comparison-only addendum after the quiet declaration) →
  `PHASE1_SIGNOFF` 21:16:56 UTC → report 22:02:19 UTC. Ordered correctly; the
  sign-off post-dates the last promotion by ~22 h.
- `final_eval.py` contains no selection or feedback path; it reads the
  registered champion, fits once on Phase 0, scores Phase 1 once, writes
  aggregates. **Caveats:** (a) the "one shot" is procedural — once the
  sign-off file exists the hook permits unlimited re-runs and the report
  would be silently overwritten; (b) the learnings entry about a first
  attempt that deadlocked and was killed before producing a readout cannot
  be verified from artifacts (only one report mtime exists); (c) Phase-1
  forecasts are not persisted, so the headline has no per-series backing
  and no confidence interval can be computed after the fact.
- Volume shares (intermittent 37%, smooth 26%, lumpy 23%, erratic 14%,
  insufficient 0.2%) are plausible: classes are assigned on the full Phase 0
  (vs per-origin slices in validation, where insufficient carried 3.9%), and
  the window is Q4.

## 7. Statistical rigor — ISSUE (major: framing), claims partly overreach

**Bootstrap (`paired_bootstrap_p`).** Resamples *series* with replacement
(n = 15,053), recomputes the pooled VN1 difference, and reports the share of
resamples with d ≥ 0 (NaN counted as "not better"). Seed-stable (seeds 1, 2,
3, 7 all give 0.000 for the final promotion). This is a one-sided
percentile-bootstrap interval check, not a null-calibrated hypothesis test;
usable as a decision rule, but the paper must describe it as such.

**Dependence is ignored.** Series within a client/warehouse share demand
shocks, and every series is scored on the same four calendar windows.
Cluster bootstrap (my recomputation, 2,000 draws):

| comparison | series p | client-cluster p (n=46) | warehouse-cluster p (n=328) |
|---|---|---|---|
| `4f9b475ead` vs `1bb3cb32c6` (final promotion) | 0.000 | **0.028** | 0.009 |
| `1bb3cb32c6` vs `7548b6e864` | 0.000 | 0.0005 | 0.000 |
| `7548b6e864` vs `2a69e4f986` (first promotion) | 0.031 | **0.30** | 0.25 |
| `4f9b475ead` vs CatBoost `523f9c23da` | — | **0.054** | 0.035 |
| `4f9b475ead` vs AutoTheta `88447a52f4` | — | 0.004 | 0.002 |

The first promotion (GBM-24 over naive) would not have passed R3 under
client-level resampling, and the champion's edge over CatBoost alone is
borderline at the client level.

**Temporal dependence: only 4 origins.** A sign test over origins cannot
reach p < 0.05 (min 1/16). Leave-one-origin-out pooled scores: the champion
beats `1bb3cb32c6`, CatBoost, AutoTheta and naive under every drop (robust);
GBM-24 vs naive **loses** when the Oct-2022 origin is dropped (0.5818 vs
0.5016) — the Stage-0 promotion hinged on one window; TFT vs N-HiTS flips
under two of four drops, so the published deep-net architecture ordering
(TFT > N-HiTS > DeepAR > PatchTST) is not origin-robust in its middle.

**Multiplicity / validation overfitting.** 16 gate invocations understate
the search: 38 configurations were scored on the same four windows and
dozens of memo-level decisions read those same numbers. The champion's
0.5137 is the best-of-38 on the selection set and is therefore optimistic;
the Phase-1 0.5346 is the only unbiased number and must be the sole
headline. The +0.021 gap is consistent with selection bias plus the Q4
window; it cannot be decomposed with the stored artifacts.

**"Generation shift" (newer FMs hallucinate lumpy demand at the turn).** The
artifact numbers are real: Chronos-2 qmean lumpy +157%, Jan VN1 1.876
(+71%); TimesFM-3 median lumpy +76% / Jan 1.978 (+74%), qmean lumpy +140% /
Jan 2.729 (+114%). But: (i) n = 2 models, one checkpoint each, one dataset;
(ii) it is **readout-dependent for Chronos-2** — its median readout has
lumpy bias −0.3% and the best January in the ledger (0.569), so Chronos-2
only "hallucinates" under the qmean readout, whereas TimesFM-3 does under
both; (iii) the older-generation "under-forecast signature" is itself
readout-dependent (Bolt qmean lumpy +15%). "Newer is actively worse with an
inverted bias sign" should be downgraded to an observation about two
checkpoints under stated readouts.

**"Median-as-point in 4/4 families."** Verified per family: Bolt — source
comment `# NOTE: the median is returned as the mean here`
(`chronos_bolt.py` line 642) and byte-identical runs; Chronos-2 — the repo
test `test_chronos2_contract_and_readout_semantics` ran and passed in my
suite run against the cached weights; gluonts/Moirai — `QuantileForecast.mean`
in gluonts source returns `self.quantile("p50")` with a warning when no mean
is stored (this is a *gluonts packaging* property, phrase it that way);
TimesFM-3 — my independent check on a 12-series panel: native head vs
q0.5 max |diff| = 0.0, quantile heads monotone (q0.1 ≤ q0.5 ≤ q0.9 on 100% of
cells). Holds, but "4 of 4" counts only families with quantile heads (TTM
has none; TimesFM-2.5's point head is a separate head).

**Family ordering claim** (blend > GBM ≈ Theta > deep > FM >
intermittent-classical): the middle of the ordering (AutoTheta 0.5355, LGBM-56
0.5429, TFT 0.5494, XGB 0.5509, N-HiTS 0.5677) spans 0.03 and is
origin-sensitive; present with per-origin ranges, not as a strict order.

## 8. Data audit — VERIFIED (Phase 0 values; Phase 1 dates only)

- `raw_data.parquet` Phase 0: 2,559,010 rows = 15,053 series × 170 weeks;
  0 duplicate (series, week) pairs; all Mondays; continuous 7-day grid
  2020-07-06 → 2023-10-02; every series has all 170 weeks; `y` has 0 NaN,
  0 negatives, max 13,681, 71.76% zeros.
- Price is NaN on exactly the zero-sale cells (1,836,437) and never on a
  sale cell — the "structural missingness" prior holds. **Note:** 4,755
  sale cells have `Price == 0`; these create ±inf price ratios that LightGBM
  bins natively while CatBoost/XGBoost receive NaN — so the three boosters
  do not see byte-identical features (documented in `gbm.py`, but the
  "same features" wording should say "same pipeline").
- Phase 1, dates only (no sales or price values were loaded): 195,689 rows =
  15,053 × 13 weeks, 2023-10-09 → 2024-01-01, 0 duplicates; the boundary gap
  is exactly 7 days.
- Provenance: Phase-0 parquet equals the official `Phase 0 - Sales.csv` /
  `Price.csv` cell-for-cell (y exact; price within float32 rounding 2e-4;
  identical NaN pattern); series id = `Client_Warehouse_Product` for all
  rows; 46 clients, 328 warehouses, 11,171 products.
- `data/*DCPD*.csv`: no such files exist; the only mention in the repo is the
  CLAUDE.md warning; no code, config, test or notebook references them.
  `data/` is git-ignored, so past presence cannot be checked, but no code
  path could have consumed them.

## 9. Tests + code review — VERIFIED (suite green) with ISSUES

- Full suite with `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE`: **61 passed,
  4 skipped** (Moirai/TTM isolated-venv tests), 26 s. LightGBM, CatBoost,
  XGBoost, torch, Chronos-2 and TimesFM-3 all ran in one process.
- **ISSUE (major)** `record_run` replace-not-append (see 4-G). No test covers
  it.
- **ISSUE (minor)** `Blend` / `WeightFitBlend` accumulate with
  `acc.add(f, fill_value=0.0)`: a component that leaves a cell unforecast is
  silently blended as 0 instead of raising; the harness's coverage check
  only sees the summed frame. Verified not triggered in either champion
  (linearity check), but it is a latent contract hole — add an assertion.
- **ISSUE (minor)** `StatsForecastModel` fills any series the library drops
  with 0 (`fillna(0.0)` + left-merge). Intended for all-zero series, but it
  also masks library failures. `n_jobs=-1` in the AutoTheta/AutoETS configs
  contradicts the docstring's "1 = deterministic"; Theta is deterministic
  regardless (AutoTheta reproduced exactly, §4).
- `evaluate()` inner-join silently drops non-overlapping series (no effect
  today).
- Readout indexing reviewed and correct: `chronos.py` (single-level list →
  index 0; qmean = mean over the last dim; Chronos-2 path indexes variate 0);
  `timesfm_fm.py` v3 (`idx = round(10q) − 1` relies on heads ordered
  0.1..0.9 — confirmed monotone empirically; `(h, 9)` shape guard would
  catch a 10-column layout); TimesFM-2.5 uses the point head `[:, :h]`;
  `moirai_fm.py` asserts item order; `ttm_fm.py` slices `[:, :h, 0]`.
- `features.py`: "lag1" is the anchor week (lag-0 relative to the anchor);
  cosmetic but confusing in a paper's feature table.
- Determinism: LightGBM has no explicit seed in config (library default is
  fixed; reproduced exactly under `OMP_NUM_THREADS=1`); CatBoost default
  seed 0; torch models seeded but documented non-reproducible.
- `run_experiment` runs multiple configs in one process (libomp hazard,
  documented). CI runs a light subset on Python 3.12 while the local venv is
  3.11.4.
- Hook (`protect_frozen.py`): blocks writes to frozen/protected files and
  the `final_eval` command string; it does not (and cannot) prevent Phase-1
  reads through `load_panel`, and it treats any `>` in a command that
  mentions a protected path as a write (harmless false positives).

---

## Blockers before paper submission

1. **Commit and tag the evidence.** Cycles 10–13 (18 runs, 4 memos, 4
   wrappers, 7 test files), the `4f9b475ead` promotion, the sign-off file and
   the final report are uncommitted. A paper cannot cite a repository state
   that does not exist.
2. **Fix the README** (stale champion, run count, status) so all public
   surfaces agree.
3. **Correct or fix the "append-only ledger" claim.** Either make
   `record_run` refuse to overwrite an existing run_id (or version it), or
   remove the append-only language from CLAUDE.md, README, hook and paper.
   Never re-run the champion config as-is: it would overwrite the champion's
   artifacts.
4. **State the FM-contamination limitation explicitly** (Chronos-Bolt
   corpus undocumented; checkpoint post-dates the public Phase-1 actuals; the
   project's own memorization-probe rule was not executed). If a probe is
   wanted it can be designed on Phase-0 windows only; it must not touch
   Phase 1.
5. **Re-frame the statistics.** Report the gate p-values as
   series-percentile-bootstrap decision rules; add the cluster-bootstrap and
   leave-one-origin-out results (this memo) and the 4-origin limitation; make
   0.5346 the only headline and label 0.5137 as a selection-set score.
6. **Pin what can still be pinned**: record the HF checkpoint revisions
   listed in §4 and the package versions (lightgbm 4.7.0, catboost 1.2.10,
   xgboost 3.2.0, torch 2.13.0, chronos-forecasting 2.3.1, timesfm 3.0.0 —
   note 2.0.2 for `e8a11552ec` — neuralforecast 3.2.1, statsforecast 2.1.1,
   pandas 2.2.3, numpy 2.1.2, Python 3.11.4).
7. Either reproduce the scratchpad-only diagnostics into `experiments/`
   artifacts (ablation trajectories, Cycle-12 weights, phantom-demand
   diagnostics) or drop those numbers from the paper.

## Caveats the paper must state

- The headline (VN1 0.5346) is one 13-week Q4 window on 15,053 series with
  no confidence interval; per-series Phase-1 forecasts were not retained.
- All model selection used the same four Phase-0 windows; ~38
  configurations were compared; validation scores are optimistic.
- Gate significance is a series-resampling bootstrap that ignores
  client/warehouse and calendar dependence; the final promotion's
  client-cluster p is 0.03 and the champion-vs-CatBoost edge is borderline
  (0.05); the first promotion (GBM-24 over naive) is not significant under
  cluster resampling and depends on the Oct-2022 origin.
- Zero-shot models: pretraining corpora undocumented/post-date the public
  data; no covariates or prices were given to any FM; results are
  checkpoint- and readout-specific (Chronos-2 median vs qmean differ by
  0.47 VN1).
- "Generation shift" rests on two checkpoints on one dataset and is
  readout-dependent; "median-as-mean" is a packaging property of four
  libraries' accessors.
- Trained deep nets were run at one untuned budget (1,500 steps, lr 1e-3)
  with no covariates; the covariate information-value question is
  unresolved (the destabilisation evidence is scratchpad-only); N-HiTS/TFT vs
  GBM differences are within one origin's swing.
- The three boosters share a pipeline but not byte-identical inputs (inf
  price ratios handled differently).
- The Phase-1 lock and the one-shot exam are procedural (hook + discipline),
  not technically enforced; `load_panel` loads Phase 1 in every process.
- Comparability with the competition's official ranking is unverified
  (already in ROADMAP); Phase-1 segment shares differ from validation shares
  because classes are assigned on the full Phase 0.
- Run ids hash configs, not code; three FM environments (timesfm 2.0.2,
  Moirai venv, TTM venv) no longer exist as run.

## Addendum — AutoTheta reproduction (`88447a52f4`)

Reproduced exactly on the third attempt (59.5 s, 4 origins): pooled VN1
0.535538 / WAPE 0.526908 / bias −0.863% identical to the ledger; per-series
`yhat_sum` max diff 0.0. The two failed attempts were a multiprocessing
main-guard defect in the *audit* script (statsforecast `n_jobs=-1` spawns
workers on macOS), not a project defect; the AutoTheta claim (best classical
model, 0.5355) stands.

## Audit hygiene check (post-audit)

Verified after all runs: `experiments/ledger.csv`, `gate_log.jsonl`,
`champion.json` and every `experiments/runs/<id>/` directory carry their
pre-audit modification times; no Phase-1 sales or price values were loaded
(only the `ds` column for the boundary/grid check); no experiment config was
run through `run_experiment` or `record_run`.
