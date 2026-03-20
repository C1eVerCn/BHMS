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
from ml.training.experiment_runner import default_config_for, run_training_experiment
from ml.training.lifecycle_experiment_runner import run_lifecycle_experiment


def load_or_train(source: str, model_type: str, force: bool, task: str) -> dict:
    summary_path = Path("data/models") / source / model_type / f"{model_type}_experiment_summary.json"
    if summary_path.exists() and not force:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    runner = run_lifecycle_experiment if task == "lifecycle" else run_training_experiment
    return runner(
        source,
        model_type,
        config_path=default_config_for(source, model_type),
        persist_training_run=True,
    )


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
                "best_val_loss": result.get("best_val_loss"),
                "test_metrics": result.get("test_metrics", {}),
                "best_checkpoint": result.get("best_checkpoint"),
                "final_checkpoint": result.get("final_checkpoint"),
                "requested_model_type": result.get("requested_model_type", model),
            }
            for model, result in results.items()
        },
    }
    output_path = Path("data/models") / args.source / "comparison_summary.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
