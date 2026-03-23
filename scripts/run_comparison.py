#!/usr/bin/env python3
"""运行或汇总单来源 lifecycle Bi-LSTM 与 Hybrid 对比实验。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse

from ml.data.source_registry import get_dataset_card, list_supported_sources
from ml.training.experiment_runner import default_config_for, generate_source_plot_bundle, run_training_experiment
from ml.training.lifecycle_experiment_runner import run_lifecycle_experiment


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def preferred_metrics(experiment_summary: dict | None, multi_seed_summary: dict | None) -> dict:
    aggregate_mean = ((multi_seed_summary or {}).get("aggregate_metrics") or {}).get("mean") or {}
    return aggregate_mean or (experiment_summary or {}).get("test_metrics", {}) or {}


def preferred_checkpoint(experiment_summary: dict | None, multi_seed_summary: dict | None) -> str | None:
    best_checkpoint = (multi_seed_summary or {}).get("best_checkpoint") or {}
    if isinstance(best_checkpoint, dict) and best_checkpoint.get("path"):
        return best_checkpoint["path"]
    return (experiment_summary or {}).get("best_checkpoint")


def load_or_train(source: str, model_type: str, force: bool, task: str) -> tuple[dict | None, dict | None]:
    summary_path = Path("data/models") / source / model_type / f"{model_type}_experiment_summary.json"
    multi_seed_summary = load_json(Path("data/models") / source / model_type / f"{model_type}_multi_seed_summary.json")
    if summary_path.exists() and not force:
        experiment_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    elif multi_seed_summary is not None and not force:
        experiment_summary = None
    else:
        runner = run_lifecycle_experiment if task == "lifecycle" else run_training_experiment
        experiment_summary = runner(
            source,
            model_type,
            config_path=default_config_for(source, model_type),
            persist_training_run=True,
        )
    return experiment_summary, multi_seed_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BHMS source comparison experiment")
    parser.add_argument("--source", choices=list_supported_sources(), required=True)
    parser.add_argument("--task", choices=["lifecycle", "rul"], default="lifecycle")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    card = get_dataset_card(args.source)
    if not card.training_ready:
        raise SystemExit(f"{args.source} is marked as {card.ingestion_mode} and is excluded from lifecycle comparison benchmarks.")

    results = {model_type: load_or_train(args.source, model_type, force=args.force, task=args.task) for model_type in ("bilstm", "hybrid")}
    comparison = {
        "source": args.source,
        "task_kind": args.task,
        "models": {
            model: {
                "best_val_loss": (experiment_summary or {}).get("best_val_loss"),
                "test_metrics": preferred_metrics(experiment_summary, multi_seed_summary),
                "single_run_test_metrics": (experiment_summary or {}).get("test_metrics", {}),
                "best_checkpoint": preferred_checkpoint(experiment_summary, multi_seed_summary),
                "final_checkpoint": (experiment_summary or {}).get("final_checkpoint"),
                "requested_model_type": (experiment_summary or {}).get("requested_model_type", model),
                "multi_seed_available": bool(multi_seed_summary),
            }
            for model, (experiment_summary, multi_seed_summary) in results.items()
        },
    }
    output_path = Path("data/models") / args.source / "comparison_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    generate_source_plot_bundle(args.source)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
