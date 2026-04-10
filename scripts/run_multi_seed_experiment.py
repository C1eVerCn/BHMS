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
    load_json,
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


def reusable_root_summary(
    source: str,
    model_type: str,
    seed: int,
    task: str,
    *,
    model_dir: str | Path = "data/models",
    requested_config_path: str | Path | None = None,
) -> dict[str, object] | None:
    summary_path = resolve_path(Path(model_dir) / source / model_type / f"{model_type}_experiment_summary.json")
    summary = load_json(summary_path)
    if not summary:
        return None
    summary_config_path = summary.get("config_path")
    if requested_config_path and not isinstance(summary_config_path, str):
        return None
    if requested_config_path and resolve_path(summary_config_path) != resolve_path(requested_config_path):
        return None
    resolved_seed = summary.get("seed")
    if resolved_seed is None:
        resolved_seed = ((summary.get("training_config") or {}).get("seed"))
    if int(resolved_seed or -1) != int(seed):
        return None
    if task == "lifecycle" and summary.get("task_kind") not in {None, "lifecycle"}:
        return None
    if not summary.get("best_checkpoint") and not summary.get("final_checkpoint"):
        return None
    return summary


def checkpoint_matches_task(checkpoint_path: str | Path, task: str) -> bool:
    if task != "lifecycle":
        return True
    candidate = Path(checkpoint_path)
    if not candidate.exists():
        return False
    import torch

    payload = torch.load(candidate, map_location="cpu")
    return payload.get("task_kind") == "lifecycle"


def existing_multi_seed_summary(
    summary_path: Path,
    task: str,
    *,
    requested_config_path: str | Path | None = None,
    requested_seeds: list[int] | None = None,
) -> dict[str, object] | None:
    payload = load_json(summary_path)
    if not payload:
        return None
    summary_config_path = payload.get("config_path")
    if requested_config_path and not isinstance(summary_config_path, str):
        return None
    if requested_config_path and resolve_path(summary_config_path) != resolve_path(requested_config_path):
        return None
    if requested_seeds is not None and list(payload.get("seeds") or []) != list(requested_seeds):
        return None
    if task != "lifecycle":
        return payload
    if payload.get("task_kind") == "lifecycle":
        return payload
    best_checkpoint = (payload.get("best_checkpoint") or {}).get("path")
    if isinstance(best_checkpoint, str):
        resolved = resolve_path(best_checkpoint)
        if checkpoint_matches_task(resolved, task):
            return payload
    return None


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
    seeds = parse_seeds(args.seeds)
    summary_path = resolve_path(f"data/models/{source}/{artifact_model}/{artifact_model}_multi_seed_summary.json")
    cached_summary = None
    if not args.force:
        cached_summary = existing_multi_seed_summary(
            summary_path,
            args.task,
            requested_config_path=config_path,
            requested_seeds=seeds,
        )
    if cached_summary is not None and not args.force:
        print(json.dumps(cached_summary, ensure_ascii=False, indent=2))
        return

    per_seed = []
    for seed in seeds:
        cached = None
        if not args.force:
            cached = reusable_root_summary(
                source,
                artifact_model,
                seed,
                args.task,
                requested_config_path=config_path,
            )
        if cached is not None:
            per_seed.append(cached)
            continue
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
