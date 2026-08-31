"""Experiment ledger: the loop's memory.

Every backtest run is recorded with full lineage so the self-improvement loop
can OBSERVE past results, DIAGNOSE weak segments, and track champion history.

Layout (under experiments/):
    ledger.csv                     -- one row per run (pooled metrics + lineage)
    runs/<run_id>/config.yaml      -- exact config that produced the run
    runs/<run_id>/per_origin.csv   -- metrics per forecast origin
    runs/<run_id>/segments.csv     -- metrics per Syntetos-Boylan class
    runs/<run_id>/volume_segments.csv
    runs/<run_id>/series_scores.parquet -- per-series pooled aggregates (gate input)
    champion.json                  -- current champion run_id + lineage
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from .data import PROJECT_ROOT, SERIES_ID

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
LEDGER_PATH = EXPERIMENTS_DIR / "ledger.csv"
CHAMPION_PATH = EXPERIMENTS_DIR / "champion.json"


def run_id_for(config: dict) -> str:
    """Deterministic id from the config content (same config -> same id)."""
    blob = json.dumps(config, sort_keys=True, default=str).encode()
    return hashlib.sha1(blob).hexdigest()[:10]


def record_run(config: dict, results: dict, notes: str = "",
               parent: str | None = None) -> str:
    """Persist a backtest run; returns its run_id."""
    run_id = run_id_for(config)
    run_dir = EXPERIMENTS_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    results["per_origin"].to_csv(run_dir / "per_origin.csv", index=False)
    results["segments"].to_csv(run_dir / "segments.csv", index=False)
    results["volume_segments"].to_csv(run_dir / "volume_segments.csv", index=False)

    # per-series evidence (pooled over origins) -- required by the promotion
    # gate's paired bootstrap in src/gates.py
    fc = results["forecasts"]
    series_scores = (
        fc.assign(abs_err=(fc["yhat"] - fc["y"]).abs())
        .groupby(SERIES_ID)
        .agg(abs_err_sum=("abs_err", "sum"), y_sum=("y", "sum"),
             yhat_sum=("yhat", "sum"))
        .reset_index()
        .rename(columns={SERIES_ID: "series"})
    )
    series_scores.to_parquet(run_dir / "series_scores.parquet", index=False)

    row = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config["model"]["name"],
        "params": json.dumps(config["model"].get("params", {})),
        "n_origins": len(results["per_origin"]),
        "horizon": config["backtest"]["horizon"],
        "parent": parent or "",
        "notes": notes,
        **results["pooled"],
    }
    ledger = pd.read_csv(LEDGER_PATH) if LEDGER_PATH.exists() else pd.DataFrame()
    ledger = ledger[ledger.get("run_id") != run_id] if len(ledger) else ledger
    ledger = pd.concat([ledger, pd.DataFrame([row])], ignore_index=True)
    ledger.to_csv(LEDGER_PATH, index=False)
    return run_id


def load_ledger() -> pd.DataFrame:
    if not LEDGER_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(LEDGER_PATH)


def get_champion() -> dict | None:
    if not CHAMPION_PATH.exists():
        return None
    return json.loads(CHAMPION_PATH.read_text())


def set_champion(run_id: str, reason: str) -> None:
    """Promote a run to champion, keeping promotion history."""
    history = get_champion() or {"current": None, "history": []}
    history["history"].append({
        "run_id": run_id, "reason": reason,
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })
    history["current"] = run_id
    CHAMPION_PATH.write_text(json.dumps(history, indent=2))
