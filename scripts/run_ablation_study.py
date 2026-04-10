#!/usr/bin/env python3
"""Run lifecycle hybrid ablation studies for a BHMS source."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data.source_registry import get_dataset_card, list_supported_sources
from ml.training.experiment_artifacts import (
    aggregate_metrics,
    plot_error_distribution,
    plot_metric_summary,
    plot_training_curves,
    write_plot_manifest,
)
from ml.training.experiment_constants import ABLATION_VARIANTS, DEFAULT_SEEDS
from ml.training.experiment_runner import (
    create_ablation_summary,
    create_multi_seed_summary,
    default_config_for,
    generate_source_plot_bundle,
    load_json,
    resolve_path,
    run_training_experiment,
)
from ml.training.lifecycle_experiment_runner import run_lifecycle_experiment


def parse_seeds(raw: str | None) -> list[int]:
    if not raw:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def build_variant_summary(
    source: str,
    variant: dict[str, object],
    per_seed_summaries: list[dict[str, object]],
    *,
    model_dir: str | Path = "data/models",
) -> dict[str, object]:
    key = str(variant["key"])
    artifact_root = resolve_path(model_dir) / source / "hybrid" / "ablation" / key
    plots_dir = artifact_root / "plots"
    per_seed_runs = []
    for summary in per_seed_summaries:
        per_seed_runs.append(
            {
                "seed": summary.get("seed"),
                "metrics": summary.get("test_metrics", {}),
            }
        )
    summary = {
        "key": key,
        "label": variant["label"],
        "description": variant["description"],
        "status": "available",
        "seeds": [int(item.get("seed", 0)) for item in per_seed_runs],
        "config_overrides": {
            "model_overrides": variant.get("model_overrides", {}),
            "feature_columns": variant.get("feature_columns"),
        },
        "per_seed_runs": per_seed_runs,
        "aggregate_metrics": aggregate_metrics([item.get("metrics", {}) for item in per_seed_runs]),
    }
    plot_metric_summary(
        summary,
        plots_dir / "metrics_overview.png",
        title=f"{source.upper()} {key} metrics",
        description="Aggregate mean/std metrics for the ablation variant.",
    )
    plot_error_distribution(
        per_seed_runs,
        plots_dir / "error_distribution.png",
        title=f"{source.upper()} {key} error distribution",
        description="Prediction error histogram for each seed in the ablation variant.",
    )
    plot_training_curves(
        per_seed_runs,
        plots_dir / "training_curves.png",
        title=f"{source.upper()} {key} training curves",
        description="Train/validation loss for the ablation variant.",
    )
    write_plot_manifest(plots_dir)
    return summary


def full_summary_matches_request(
    summary: dict[str, object] | None,
    *,
    config_path: str | Path,
    seeds: list[int],
    task: str,
) -> bool:
    if not summary:
        return False
    if task == "lifecycle" and summary.get("task_kind") != "lifecycle":
        return False
    summary_config_path = summary.get("config_path")
    if not isinstance(summary_config_path, str):
        return False
    if resolve_path(summary_config_path) != resolve_path(config_path):
        return False
    return list(summary.get("seeds") or []) == list(seeds)


def ensure_full_hybrid_summary(
    source: str,
    *,
    task: str,
    config_path: str | Path,
    seeds: list[int],
    refresh_full_hybrid: bool = False,
) -> tuple[dict[str, object], bool]:
    full_summary = load_json(resolve_path(f"data/models/{source}/hybrid/hybrid_multi_seed_summary.json"))
    runner = run_lifecycle_experiment if task == "lifecycle" else run_training_experiment
    refreshed = refresh_full_hybrid or not full_summary_matches_request(full_summary, config_path=config_path, seeds=seeds, task=task)
    if not refreshed and full_summary is not None:
        return full_summary, False

    seed_runs = []
    for seed in seeds:
        seed_runs.append(
            runner(
                source,
                "hybrid",
                config_path=config_path,
                training_overrides={"seed": seed, "model_version": f"{source}-hybrid-seed-{seed}"},
                artifact_subdir=f"runs/seed-{seed}",
                suite_kind="multi_seed",
                variant_key="hybrid",
                persist_training_run=True,
            )
        )
    summary = create_multi_seed_summary(
        source,
        "hybrid",
        seeds=seeds,
        per_seed_summaries=seed_runs,
        config_path=config_path,
    )
    return summary, True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BHMS lifecycle hybrid ablation study")
    parser.add_argument("--source", choices=list_supported_sources(), required=True)
    parser.add_argument("--task", choices=["lifecycle", "rul"], default="lifecycle")
    parser.add_argument("--config", default=None)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, default: 7,21,42")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--refresh-full-hybrid",
        action="store_true",
        help="Force-refresh the Hybrid multi-seed truth summary before running ablations.",
    )
    args = parser.parse_args()

    source = args.source.lower()
    card = get_dataset_card(source)
    if not card.training_ready:
        raise SystemExit(f"{source} is marked as {card.ingestion_mode} and is excluded from lifecycle ablation studies.")
    config_path = args.config or str(default_config_for(source, "hybrid"))
    summary_path = resolve_path(f"data/models/{source}/ablation_summary.json")
    if summary_path.exists() and not args.force:
        print(summary_path.read_text(encoding="utf-8"))
        return

    seeds = parse_seeds(args.seeds)
    full_summary, _ = ensure_full_hybrid_summary(
        source,
        task=args.task,
        config_path=config_path,
        seeds=seeds,
        refresh_full_hybrid=args.refresh_full_hybrid,
    )
    runner = run_lifecycle_experiment if args.task == "lifecycle" else run_training_experiment

    variants: list[dict[str, object]] = []
    variants.append(
        {
            "key": "full_hybrid",
            "label": "完整 Hybrid",
            "description": "复用主实验的 lifecycle hybrid 多随机种子结果。",
            "status": "available",
            "seeds": full_summary.get("seeds", seeds),
            "aggregate_metrics": full_summary.get("aggregate_metrics", {}),
        }
    )

    for variant in ABLATION_VARIANTS:
        if variant["key"] == "full_hybrid":
            continue
        per_seed = []
        for seed in seeds:
            config_overrides = {"model": dict(variant.get("model_overrides", {}))}
            if variant.get("feature_columns"):
                config_overrides["data"] = {"feature_columns": list(variant["feature_columns"])}
            per_seed.append(
                runner(
                    source,
                    "hybrid",
                    config_path=config_path,
                    config_overrides=config_overrides,
                    training_overrides={"seed": seed, "model_version": f"{source}-{variant['key']}-seed-{seed}"},
                    artifact_subdir=f"ablation/{variant['key']}/seed-{seed}",
                    suite_kind="ablation",
                    variant_key=str(variant["key"]),
                    persist_training_run=True,
                )
            )
        variants.append(build_variant_summary(source, variant, per_seed))

    summary = create_ablation_summary(source, variants=variants)
    generate_source_plot_bundle(source)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
