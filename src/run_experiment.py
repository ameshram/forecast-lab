"""Run one experiment from a YAML config and record it in the ledger.

Usage:
    python -m src.run_experiment configs/ma13.yaml
    python -m src.run_experiment configs/*.yaml      # several in sequence

Config schema:
    model:    {name: <registry key>, params: {...}}
    backtest: {n_origins: int, horizon: int, step: int|null}
    notes:    optional free text
"""
from __future__ import annotations

import sys

import yaml

from .backtest import run_backtest
from .data import load_panel, split_phases, validation_origins
from .ledger import record_run
from .models import build_model


def run_config(path: str, df=None) -> str:
    config = yaml.safe_load(open(path))
    if df is None:
        df = load_panel()
    phase0, _ = split_phases(df)  # Phase 1 never enters model selection

    bt = config["backtest"]
    origins = validation_origins(phase0, bt["n_origins"], bt["horizon"],
                                 bt.get("step"))
    model = build_model(config["model"])
    results = run_backtest(phase0, model, origins, bt["horizon"])
    run_id = record_run(config, results, notes=config.get("notes", ""))

    print(f"\n=== {config['model']['name']} "
          f"{config['model'].get('params', {})} -> run {run_id} ===")
    print("pooled:", {k: round(v, 4) for k, v in results["pooled"].items()})
    print(results["per_origin"].assign(
        origin=lambda d: d.origin.dt.date).round(4).to_string(index=False))
    print("\nby Syntetos-Boylan class:")
    print(results["segments"].round(4).to_string(index=False))
    return run_id


def main() -> None:
    paths = sys.argv[1:]
    if not paths:
        sys.exit("usage: python -m src.run_experiment <config.yaml> [...]")
    df = load_panel()
    for path in paths:
        run_config(path, df=df)


if __name__ == "__main__":
    main()
