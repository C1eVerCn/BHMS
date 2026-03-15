#!/usr/bin/env python3
"""Run multi-seed experiments for a BHMS source/model pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.experiment_runner import (
    DEFAULT_SEEDS,
    create_multi_seed_summary,
    default_config_for,
    generate_source_plot_bundle,
    resolve_path,
    run_training_experiment,
)


def parse_seeds(raw: str | None) -> list[int]:
    if not raw:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BHMS multi-seed experiment")
    parser.add_argument("--source", choices=["nasa", "calce", "kaggle"], required=True)
    parser.add_argument("--model", choices=["bilstm", "hybrid"], required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, default: 7,21,42")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source.lower()
    model = args.model.lower()
    config_path = args.config or str(default_config_for(source, model))
    summary_path = resolve_path(f"data/models/{source}/{model}/{model}_multi_seed_summary.json")
    if summary_path.exists() and not args.force:
        print(summary_path.read_text(encoding="utf-8"))
        return

    seeds = parse_seeds(args.seeds)
    per_seed = []
    for seed in seeds:
        per_seed.append(
            run_training_experiment(
                source,
                model,
                config_path=config_path,
                training_overrides={"seed": seed, "model_version": f"{source}-{model}-seed-{seed}"},
                artifact_subdir=f"runs/seed-{seed}",
                suite_kind="multi_seed",
                variant_key=model,
                persist_training_run=True,
            )
        )

    summary = create_multi_seed_summary(
        source,
        model,
        seeds=seeds,
        per_seed_summaries=per_seed,
        config_path=config_path,
    )
    generate_source_plot_bundle(source)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
