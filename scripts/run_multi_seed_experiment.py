#!/usr/bin/env python3
"""Run multi-seed lifecycle experiments for a BHMS source/model pair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.source_registry import get_dataset_card, list_supported_sources
from ml.training.experiment_runner import (
    DEFAULT_SEEDS,
    create_multi_seed_summary,
    default_config_for,
    generate_source_plot_bundle,
    resolve_path,
    run_training_experiment,
)
from ml.training.lifecycle_experiment_runner import MODEL_ARTIFACT_ALIASES, run_lifecycle_experiment


MODEL_CHOICES = ["bilstm", "hybrid", "lifecycle_bilstm", "lifecycle_hybrid"]


def parse_seeds(raw: str | None) -> list[int]:
    if not raw:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def artifact_model_type(model_type: str) -> str:
    return MODEL_ARTIFACT_ALIASES.get(model_type.lower(), model_type.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BHMS multi-seed lifecycle experiment")
    parser.add_argument("--source", choices=list_supported_sources(), required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--task", choices=["lifecycle", "rul"], default="lifecycle")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, default: 7,21,42")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source.lower()
    card = get_dataset_card(source)
    if not card.training_ready:
        raise SystemExit(f"{source} is marked as {card.ingestion_mode} and is excluded from lifecycle multi-seed experiments.")
    requested_model = args.model.lower()
    artifact_model = artifact_model_type(requested_model)
    config_path = args.config or str(default_config_for(source, artifact_model))
    summary_path = resolve_path(f"data/models/{source}/{artifact_model}/{artifact_model}_multi_seed_summary.json")
    if summary_path.exists() and not args.force:
        print(summary_path.read_text(encoding="utf-8"))
        return

    seeds = parse_seeds(args.seeds)
    per_seed = []
    for seed in seeds:
        runner = run_lifecycle_experiment if args.task == "lifecycle" else run_training_experiment
        run_model = requested_model if args.task == "lifecycle" else artifact_model
        per_seed.append(
            runner(
                source,
                run_model,
                config_path=config_path,
                training_overrides={"seed": seed, "model_version": f"{source}-{artifact_model}-seed-{seed}"},
                artifact_subdir=f"runs/seed-{seed}",
                suite_kind="multi_seed",
                variant_key=artifact_model,
                persist_training_run=True,
            )
        )

    summary = create_multi_seed_summary(
        source,
        artifact_model,
        seeds=seeds,
        per_seed_summaries=per_seed,
        config_path=config_path,
    )
    generate_source_plot_bundle(source)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
