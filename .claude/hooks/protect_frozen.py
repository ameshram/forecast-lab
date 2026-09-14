#!/usr/bin/env python3
"""PreToolUse hard gate for the Forecast Lab.

Blocks (exit code 2 = deny, stderr shown to the agent):
  1. Edits to FROZEN files -- the evaluation harness and this governance
     layer itself. A changed metric or backtest invalidates every ledger row.
  2. Direct edits to PROTECTED ledger artifacts -- they are append-only and
     written exclusively through src/ledger.py.
  3. Shell writes that target any of the above.
  4. Running the Phase 1 final evaluation without the human sign-off file
     (experiments/PHASE1_SIGNOFF) -- Phase 1 is the locked test set.

Everything else is allowed (exit 0). Reads are never blocked.
"""
import json
import os
import re
import sys

FROZEN = [
    "src/metrics.py",
    "src/backtest.py",
    "src/final_eval.py",
    ".claude/hooks/protect_frozen.py",
    ".claude/settings.json",
]
PROTECTED = [
    "experiments/ledger.csv",
    "experiments/champion.json",
    "experiments/gate_log.jsonl",
]
WRITE_TOOLS = {"Edit", "Write", "MultiEdit", "NotebookEdit"}
SHELL_WRITE_RE = re.compile(r"(>>?|\btee\b|\bsed\s+-i|\brm\b|\bmv\b|\btruncate\b)")


def deny(msg: str) -> None:
    print(f"BLOCKED by protect_frozen hook: {msg}", file=sys.stderr)
    sys.exit(2)


def main() -> None:
    try:
        data = json.load(sys.stdin)
    except Exception:
        sys.exit(0)  # malformed input: do not break the session
    tool = data.get("tool_name", "")
    ti = data.get("tool_input", {}) or {}

    if tool in WRITE_TOOLS:
        path = str(ti.get("file_path") or ti.get("notebook_path") or "")
        norm = path.replace("\\", "/")
        for f in FROZEN:
            if norm.endswith(f):
                deny(f"{f} is FROZEN (evaluation harness / governance). "
                     f"Changing it invalidates the ledger. If the human has "
                     f"explicitly ordered a harness change, they must lift the "
                     f"freeze by editing this hook themselves.")
        for f in PROTECTED:
            if norm.endswith(f):
                deny(f"{f} is append-only and written exclusively through "
                     f"src/ledger.py or src/gates.py. Use those entry points.")

    if tool == "Bash":
        cmd = str(ti.get("command") or "")
        for f in FROZEN + PROTECTED:
            if f in cmd and SHELL_WRITE_RE.search(cmd):
                deny(f"shell command touches protected file {f} with a "
                     f"write-capable operation. Reads are fine via the Read "
                     f"tool; writes must go through the blessed entry points.")
        if "final_eval" in cmd:
            root = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())
            if not os.path.exists(os.path.join(root, "experiments",
                                               "PHASE1_SIGNOFF")):
                deny("Phase 1 is LOCKED. src.final_eval may only run after "
                     "the human creates experiments/PHASE1_SIGNOFF to "
                     "authorize the one-shot final evaluation.")

    sys.exit(0)


if __name__ == "__main__":
    main()
