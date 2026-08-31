# North Star — the one goal of this project

> **Research question: Do today's best deep learning forecasters — including
> pre-trained foundation models — actually beat simpler methods on messy,
> real-world retail demand? Where exactly, and by how much?**

Everything in this repo serves that question. The simple baselines are the
yardstick, the LightGBM is the strong conventional rival deep learning must
overcome, and the champion/challenger harness is the methodology that makes
the comparison trustworthy. The deliverables are: (1) this repo, reproducible
end-to-end; (2) the Forecast Lab dashboard; (3) a paper answering the
question, with the untouched Phase 1 as its headline out-of-sample result.

## The path (stages advance in order; a memo must name its stage)

- **Stage 0 — Yardstick + strong rival. DONE.**
  Baselines (best: naive 0.583), classical challengers, global LightGBM
  (`7548b6e864`, 0.5622, gate PASS — crown it before Stage 1 so deep
  learning faces the strongest rival at full strength).
- **Stage 1 — Foundation models, zero training.**
  Chronos and TimesFM forecast our series with no training. Question
  answered: can a model that has never seen this data beat models trained
  on it? Either answer is a paper section.
- **Stage 2 — One trained deep network, done properly.**
  PatchTST (backup: N-HiTS) as a single global model over all series, with
  a zero-inflation-aware loss (negative binomial). It also gets price,
  calendar, and product-age inputs — so we learn what that information is
  worth. Trains locally on the M1 Max.
- **Stage 3 — The team-up.**
  Blend the best deep model with the best non-deep model; gate the blend.
  Different families fail in different places (see the January problem in
  `experiments/learnings.md`) — combination is where the field's wins come
  from.
- **Final exam — one shot, then stop.**
  Stopping rule: when two consecutive cycles produce no new champion, the
  human signs off (`experiments/PHASE1_SIGNOFF`) and the reigning champion
  runs ONCE on Phase 1. That number is the paper's headline; never touching
  Phase 1 until then is the paper's credibility.

## Registered predictions (scored at each stage's /reflect)

1. Foundation models land worse than the GBM overall, but surprisingly
   close on steady (smooth) products.
2. The trained network beats the GBM on busy/seasonal products, not on
   erratic ones.
3. The blend beats every single model.

## Scope fences (drift and hallucination guards)

- The **metric, backtest scheme, and Phase 1 lock never change** for the
  paper's duration. If any of them changes, every prior result is invalid.
- **No new datasets, no new metrics, no new phases** without a human-approved
  memo that names what published claim requires them.
- **Numbers come from the ledger, never from memory.** Any score quoted in a
  memo, commit, dashboard, or draft must trace to a run-id or a file in
  `experiments/`. If it can't, it is unknown — say so.
- An option that does not advance the current stage must be flagged
  **OFF-ROADMAP** in its memo and needs explicit human approval.
- The sparse tail ("insufficient" class) stays out of scope; the ceiling on
  cycles stays at the gate budget (25 evaluations).

## Source data & comparability

- **Provenance.** Official VN1 Forecasting Accuracy Challenge (Phase 2) data
  from DataSource.ai. Citation (required on any public use; see
  [README.md](README.md) §Data source & citation): Vandeput, Nicolas,
  "VN1 Forecasting - Accuracy Challenge Phase 2," DataSource.ai, 3 Oct. 2024.
  Data revalidated against the official description 2026-08-30 — series count,
  phase windows, horizon, metric, and price semantics all match (memo
  `experiments/memos/2026-08-30-data-provenance-revalidation.md`).
- **Comparability caveat (for the paper).** Our task mirrors the competition's
  graded phase — train on Phase 0, forecast the 13-week Phase-1 window — but
  our protocol is *stricter*: the reigning champion runs on Phase 1 **once**,
  with no leaderboard feedback of any kind. The competition let participants
  probe a live leaderboard on that window repeatedly. This is a credibility
  point in our favour and should be stated as such.
- **Unverified, do not assume.** Whether the competition's *official final
  ranking* was scored on exactly these Phase-1 files (versus a later window
  with more training data) is **not verifiable from in-repo evidence**. Before
  the paper ever prints an official leaderboard score beside ours, that must be
  confirmed on the (login-gated) competition page and the protocol difference
  stated. Until then, our Phase-1 number is reported on its own terms, not as a
  competition placement.
