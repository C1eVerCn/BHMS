"""Lifecycle transfer benchmark orchestration for multisource pretrain -> target fine-tune."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from ml.training.experiment_artifacts import (
    aggregate_metrics,
    plot_error_distribution,
    plot_metric_summary,
    plot_training_curves,
    select_best_run,
    serialize_path,
    write_json,
    write_plot_manifest,
)
from ml.training.experiment_runner import resolve_path
from ml.training.lifecycle_experiment_runner import run_lifecycle_experiment


def _transfer_run_record(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": summary.get("seed"),
        "metrics": summary.get("test_metrics", {}),
        "best_checkpoint": summary.get("best_checkpoint"),
        "final_checkpoint": summary.get("final_checkpoint"),
        "artifact_dir": summary.get("artifact_dir"),
        "summary_path": summary.get("summary_path"),
        "history": summary.get("history", {}),
        "history_summary": summary.get("history_summary", {}),
        "test_details": summary.get("test_details", {}),
        "test_details_path": summary.get("test_details_path"),
        "stage_kind": summary.get("stage_kind"),
    }


def create_transfer_summary(
    *,
    target_source: str,
    model_type: str,
    transfer_key: str,
    pretrain_config_path: str | Path,
    fine_tune_config_path: str | Path,
    seeds: list[int],
    pretrain_runs: list[dict[str, Any]],
    fine_tune_runs: list[dict[str, Any]],
    model_dir: str | Path = "data/models",
) -> dict[str, Any]:
    artifact_root = resolve_path(model_dir) / target_source / model_type / "transfer" / transfer_key
    plots_dir = artifact_root / "plots"
    pretrain_records = [_transfer_run_record(item) for item in pretrain_runs]
    fine_tune_records = [_transfer_run_record(item) for item in fine_tune_runs]
    aggregate = aggregate_metrics([item.get("metrics", {}) for item in fine_tune_records])
    best_run = select_best_run(fine_tune_records)
    summary = {
        "target_source": target_source,
        "model_type": model_type,
        "task_kind": "lifecycle",
        "suite_kind": "transfer",
        "transfer_key": transfer_key,
        "available": True,
        "seeds": seeds,
        "pretrain_config_path": serialize_path(resolve_path(pretrain_config_path)),
        "fine_tune_config_path": serialize_path(resolve_path(fine_tune_config_path)),
        "pretrain_source_scope": pretrain_runs[0].get("source_scope", []) if pretrain_runs else [],
        "fine_tune_source_scope": fine_tune_runs[0].get("source_scope", []) if fine_tune_runs else [target_source],
        "pretrain_runs": pretrain_records,
        "fine_tune_runs": fine_tune_records,
        "aggregate_metrics": aggregate,
        "best_checkpoint": {
            "seed": best_run.get("seed") if best_run else None,
            "path": best_run.get("best_checkpoint") if best_run else None,
        },
        "artifact_paths": {
            "summary": serialize_path(artifact_root / f"{model_type}_transfer_summary.json"),
            "plots_dir": serialize_path(plots_dir),
        },
        "generated_at": datetime.utcnow().isoformat(),
    }
    summary["plots"] = [
        plot_metric_summary(
            summary,
            plots_dir / "metrics_overview.png",
            title=f"{target_source.upper()} {model_type} transfer metrics",
            description="Mean and std across fine-tune seeds for the transfer benchmark.",
        ),
        plot_error_distribution(
            fine_tune_records,
            plots_dir / "error_distribution.png",
            title=f"{target_source.upper()} {model_type} transfer error distribution",
            description="Prediction error histogram for the fine-tuned transfer runs.",
        ),
        plot_training_curves(
            fine_tune_records,
            plots_dir / "training_curves.png",
            title=f"{target_source.upper()} {model_type} transfer curves",
            description="Train/validation curves for the fine-tuned transfer runs.",
        ),
    ]
    write_plot_manifest(plots_dir)
    write_json(artifact_root / f"{model_type}_transfer_summary.json", summary)
    return summary


def run_transfer_benchmark(
    target_source: str,
    model_type: str,
    *,
    pretrain_config_path: str | Path,
    fine_tune_config_path: str | Path,
    seeds: list[int],
    model_dir: str | Path = "data/models",
    persist_training_run: bool = True,
    pretrain_training_overrides: Optional[dict[str, Any]] = None,
    fine_tune_training_overrides: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    target_source = target_source.lower()
    model_type = model_type.lower()
    transfer_key = f"multisource_to_{target_source}"
    pretrain_runs: list[dict[str, Any]] = []
    fine_tune_runs: list[dict[str, Any]] = []

    for seed in seeds:
        pretrain_overrides = {
            "seed": seed,
            "model_version": f"{target_source}-multisource-pretrain-{model_type}-seed-{seed}",
            **(pretrain_training_overrides or {}),
        }
        pretrain_summary = run_lifecycle_experiment(
            target_source,
            model_type,
            config_path=pretrain_config_path,
            training_overrides=pretrain_overrides,
            artifact_subdir=f"transfer/{transfer_key}/pretrain/seed-{seed}",
            suite_kind="transfer",
            variant_key=transfer_key,
            stage_kind="pretrain",
            persist_training_run=persist_training_run,
        )
        pretrain_runs.append(pretrain_summary)

        init_checkpoint = pretrain_summary.get("best_checkpoint") or pretrain_summary.get("final_checkpoint")
        fine_tune_overrides = {
            "seed": seed,
            "model_version": f"{target_source}-transfer-finetune-{model_type}-seed-{seed}",
            "init_from_checkpoint": init_checkpoint,
            **(fine_tune_training_overrides or {}),
        }
        fine_tune_summary = run_lifecycle_experiment(
            target_source,
            model_type,
            config_path=fine_tune_config_path,
            training_overrides=fine_tune_overrides,
            artifact_subdir=f"transfer/{transfer_key}/fine_tune/seed-{seed}",
            suite_kind="transfer",
            variant_key=transfer_key,
            stage_kind="fine_tune",
            persist_training_run=persist_training_run,
        )
        fine_tune_runs.append(fine_tune_summary)

    return create_transfer_summary(
        target_source=target_source,
        model_type=model_type,
        transfer_key=transfer_key,
        pretrain_config_path=pretrain_config_path,
        fine_tune_config_path=fine_tune_config_path,
        seeds=seeds,
        pretrain_runs=pretrain_runs,
        fine_tune_runs=fine_tune_runs,
        model_dir=model_dir,
    )


__all__ = ["create_transfer_summary", "run_transfer_benchmark"]
