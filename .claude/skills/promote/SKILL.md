---
name: promote
description: EVALUATE/PROMOTE stage — run the deterministic promotion gate on a challenger, recommend, and (only after an explicit human yes) write the champion registry and refresh the dashboard.
---

# /promote

The gate decides; you recommend; the human confirms. Never promote on your
own judgment, and never argue a FAIL into a promotion.

1. Run: `./forecasting/bin/python -m src.gates <challenger_run_id>`
   (R1 pooled win, R2 origin majority, R3 paired-bootstrap p<0.05,
   R4 no ≥10%-volume segment worse by >5%).
2. Present the gate report with your recommendation:
   - **PASS** → recommend promotion; summarize what the new champion changes
     and any segment trade-offs. Heed the gate-budget warning if shown.
   - **FAIL** → name the failed rule(s) and what result would pass; record
     the learned result in the memo. Stop here.
3. Only after the human explicitly confirms, run:
   `./forecasting/bin/python -m src.gates <run_id> --promote --confirm`
4. After promotion: regenerate the dashboard's embedded JSON from the
   ledger (leaderboard, promotion log, learned-results list) in
   `dashboard/index.html`, republish the "Forecast Lab" artifact, and state
   the new champion + score in the conversation.
5. If the promotion completes a ROADMAP.md stage, update the "Current
   stage" line in CLAUDE.md's North Star section — that line is how every
   future session knows where the project stands.
