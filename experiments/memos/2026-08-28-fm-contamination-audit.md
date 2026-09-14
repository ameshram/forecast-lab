# Audit note — foundation-model pretraining contamination (VN1)

Date: 2026-08-28 · Documentation-only, zero compute, no gate budget.
Question (retrospective Q2): could Chronos-Bolt or TimesFM 2.5 have seen
VN1 data in pretraining, undermining the "zero-shot" claim?

## Timeline facts (sources linked below)

- VN1 competition ran **Sep 12 – Oct 17, 2024** (DataSource.ai; SupChains/
  Syrup/Flieber). Phase 0 data public from ~Sep 12, 2024; **Phase 1
  actuals revealed only after Oct 17, 2024**.
- **Chronos corpus** (paper arXiv:2403.07815) assembled by **March 2024**
  — predates VN1's existence entirely.
- **Chronos-Bolt** released **Nov 26, 2024**. Its model card does NOT
  disclose the corpus ("nearly 100B observations"); maintainers have not
  published pretraining data details. Community accounts describe the
  public-data portion as a filtered version of the paper corpus plus more
  synthetic (GP-kernel) data — synthetic expansion cannot add VN1.
- **TimesFM 2.5** (checkpoint line released 2025) DOES disclose:
  GiftEvalPretrain (Salesforce) + Wikimedia pageviews (**cutoff Nov
  2023**) + Google Trends (**cutoff EoY 2022**) + synthetic.
- **GiftEvalPretrain** was assembled for the GIFT-Eval paper
  (arXiv:2410.10393, submitted **Oct 14, 2024**) from established public
  archives, expressly curated to be leakage-free vs its benchmark. Its
  assembly predates the VN1 Phase-1 reveal (Oct 17), and Phase-0
  inclusion within a one-month window from an archive-derived corpus is
  implausible. Individual datasets are not enumerated on the card.

## Verdict

- **Phase 1 (the paper's headline test set): contamination effectively
  ruled out for TimesFM 2.5's disclosed sources** (all predate the
  Phase-1 reveal) and **low-risk for Chronos-Bolt** (a Sep–Nov 2024
  window exists in principle, but no evidence of corpus re-collection,
  and the described pipeline reuses the pre-VN1 paper corpus).
- **Phase 0: low risk for both** (Chronos corpus predates VN1; TimesFM's
  dated sources predate it; GiftEvalPretrain timing argues against).
- Residual weakness: Chronos-Bolt's corpus is **undocumented** — the
  clean verdict rests on timing and secondary accounts, not a disclosure.

## Actions adopted

1. Paper carries one limitation sentence: FM pretraining corpora are not
   fully enumerable; timing evidence (above) argues against VN1 inclusion.
2. **Pin HF checkpoint revisions** (repo names are mutable pointers) —
   research-engineering item, pending approval.
3. **Standing governance rule proposed:** if any FM component is part of
   the model that ultimately runs the one-shot Phase 1 evaluation, run an
   empirical memorization probe first (e.g., zero-shot error on VN1 vs a
   value-perturbed copy) under its own approved memo. Zero-shot scores on
   Phase 0 selection remain valid either way for model *selection*.

Sources: datasource.ai VN1 pages; huggingface.co/amazon/chronos-bolt-base;
huggingface.co/google/timesfm-2.5-200m-pytorch;
huggingface.co/datasets/Salesforce/GiftEvalPretrain; arXiv:2403.07815;
arXiv:2410.10393; github.com/amazon-science/chronos-forecasting (#306).
