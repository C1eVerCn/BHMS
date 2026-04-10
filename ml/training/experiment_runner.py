"""Shared training runners for baseline, multi-seed, and ablation experiments."""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml

from backend.app.core.database import get_database
from backend.app.services.repository import BHMSRepository
from ml.data import RULDataModule
from ml.data.processed_paths import resolve_cycle_summary_path
from ml.training.experiment_constants import (
    ABLATION_VARIANTS,
    DEFAULT_CONFIGS,
    DEFAULT_SEEDS,
    PROJECT_ROOT,
)
from ml.training.experiment_artifacts import (
    aggregate_metrics,
    plot_ablation_overview,
    plot_error_distribution,
    plot_metric_summary,
    plot_source_comparison,
    plot_split_overview,
    plot_training_curves,
    serialize_path,
    write_json,
    write_plot_manifest,
)
from ml.training.trainer import RULTrainer, TrainingConfig, build_model


def load_yaml(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    return yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def default_config_for(source: str, model_type: str) -> Path:
    return DEFAULT_CONFIGS[(source.lower(), model_type.lower())]


def _history_summary(history: dict[str, Any]) -> dict[str, Any]:
    train_history = history.get("train", []) or []
    val_history = history.get("val", []) or []
    return {
        "epochs_ran": len(train_history),
        "last_train_loss": train_history[-1].get("loss") if train_history else None,
        "last_val_loss": val_history[-1].get("loss") if val_history else None,
    }


def _run_record(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "seed": summary.get("seed"),
        "suite_kind": summary.get("suite_kind"),
        "variant_key": summary.get("variant_key"),
        "metrics": summary.get("test_metrics", {}),
        "best_checkpoint": summary.get("best_checkpoint"),
    }


def _best_checkpoint_payload(per_seed_summaries: list[dict[str, Any]], metric_key: str = "rmse") -> dict[str, Any]:
    ranked = [
        item
        for item in per_seed_summaries
        if isinstance((item.get("test_metrics") or {}).get(metric_key), (int, float))
    ]
    if not ranked:
        return {"seed": None, "path": None}
    best = min(ranked, key=lambda item: float(item["test_metrics"][metric_key]))
    return {
        "seed": best.get("seed"),
        "path": best.get("best_checkpoint") or best.get("final_checkpoint"),
    }


def run_training_experiment(
    source: str,
    model_type: str,
    *,
    config_path: str | Path | None = None,
    config_overrides: Optional[dict[str, Any]] = None,
    training_overrides: Optional[dict[str, Any]] = None,
    artifact_subdir: str | None = None,
    suite_kind: str = "baseline",
    variant_key: str = "baseline",
    repository: Optional[BHMSRepository] = None,
    persist_training_run: bool = True,
) -> dict[str, Any]:
    source = source.lower()
    model_type = model_type.lower()
    resolved_config = resolve_path(config_path or default_config_for(source, model_type))
    config = load_yaml(resolved_config)
    merged = merge_configs(config, config_overrides or {})
    data_cfg = dict(merged.get("data", {}))
    model_cfg = dict(merged.get("model", {}))
    training_cfg = merge_configs(dict(merged.get("training", {})), training_overrides or {})
    seed = int(training_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    csv_path = resolve_cycle_summary_path(resolve_path(data_cfg["csv_path"]), source=source)
    merged.setdefault("data", {})["csv_path"] = serialize_path(csv_path)
    data_cfg["csv_path"] = merged["data"]["csv_path"]
    checkpoint_dir = resolve_path(training_cfg.get("checkpoint_dir", "data/models"))
    log_dir = resolve_path(training_cfg.get("log_dir", "data/models/logs"))
    training_cfg.pop("checkpoint_dir", None)
    training_cfg.pop("log_dir", None)

    if persist_training_run and repository is None:
        get_database().initialize()
    repo = repository or BHMSRepository()

    data_module = RULDataModule(
        csv_path=csv_path,
        source=source,
        seq_len=int(data_cfg.get("seq_len", 30)),
        batch_size=int(training_cfg.get("batch_size", 16)),
        feature_cols=data_cfg.get("feature_columns"),
        output_dir=csv_path.parent,
        seed=seed,
    )
    data_summary = data_module.summary()
    if artifact_subdir:
        root_summary_path = csv_path.parent / f"{source}_dataset_summary.json"
        if not root_summary_path.exists():
            data_module.export_metadata()
        profile_prefix = variant_key if suite_kind == "ablation" else source
        data_module.export_metadata(
            output_dir=checkpoint_dir / source / model_type / artifact_subdir / "data_profile",
            file_prefix=profile_prefix,
        )
    else:
        data_module.export_metadata()

    model, model_config = build_model(model_type, input_dim=len(data_summary["feature_columns"]), overrides=model_cfg)
    trainer = RULTrainer(
        model=model,
        model_config=model_config,
        training_config=TrainingConfig(
            source=source,
            model_type=model_type,
            checkpoint_dir=str(checkpoint_dir),
            log_dir=str(log_dir),
            artifact_subdir=artifact_subdir,
            **training_cfg,
        ),
        train_loader=data_module.train_loader(),
        val_loader=data_module.val_loader(),
        test_loader=data_module.test_loader(),
        data_summary=data_summary,
    )
    result = trainer.train()
    test_details = result.get("test_details", trainer.test_outputs)
    test_details_path = trainer.checkpoint_dir / "test_details.json"
    write_json(test_details_path, test_details)
    summary = {
        **result,
        "source": source,
        "model_type": model_type,
        "suite_kind": suite_kind,
        "variant_key": variant_key,
        "seed": trainer.config.seed,
        "artifact_dir": serialize_path(trainer.checkpoint_dir),
        "log_dir": serialize_path(trainer.log_dir),
        "config_path": serialize_path(resolved_config),
        "config_snapshot": merged,
        "split_snapshot": data_summary.get("split", {}),
        "feature_columns": data_summary.get("feature_columns", []),
        "model_config": model_config,
        "training_config": asdict(trainer.config),
        "history_summary": _history_summary(result.get("history", {})),
        "test_details": test_details,
        "test_details_path": serialize_path(test_details_path),
    }
    summary_path = trainer.checkpoint_dir / f"{model_type}_experiment_summary.json"
    write_json(summary_path, summary)
    summary["summary_path"] = serialize_path(summary_path)

    if persist_training_run:
        repo.insert_training_run(
            {
                "source": source,
                "model_type": model_type,
                "model_version": trainer.config.model_version,
                "best_checkpoint_path": summary.get("best_checkpoint"),
                "final_checkpoint_path": summary.get("final_checkpoint"),
                "metrics": summary.get("test_metrics", {}),
                "metadata": {
                    "seed": trainer.config.seed,
                    "suite_kind": suite_kind,
                    "variant_key": variant_key,
                    "artifact_dir": serialize_path(trainer.checkpoint_dir),
                    "config_path": serialize_path(resolved_config),
                    "config_snapshot": merged,
                    "split_snapshot": data_summary.get("split", {}),
                    "feature_columns": data_summary.get("feature_columns", []),
                    "summary_path": serialize_path(summary_path),
                    "test_details_path": serialize_path(test_details_path),
                },
            }
        )
    return summary


def create_multi_seed_summary(
    source: str,
    model_type: str,
    *,
    seeds: list[int],
    per_seed_summaries: list[dict[str, Any]],
    config_path: str | Path,
    model_dir: str | Path = "data/models",
) -> dict[str, Any]:
    artifact_root = resolve_path(model_dir) / source / model_type
    plots_dir = artifact_root / "plots"
    per_seed_runs = [_run_record(summary) for summary in per_seed_summaries]
    aggregate = aggregate_metrics([run.get("metrics", {}) for run in per_seed_runs])
    best_checkpoint = _best_checkpoint_payload(per_seed_summaries)
    summary = {
        "source": source,
        "model_type": model_type,
        "task_kind": per_seed_summaries[0].get("task_kind", "rul") if per_seed_summaries else "rul",
        "suite_kind": "multi_seed",
        "available": True,
        "seeds": seeds,
        "config_path": serialize_path(resolve_path(config_path)),
        "config_snapshot": per_seed_summaries[0].get("config_snapshot", {}) if per_seed_summaries else {},
        "split_snapshot": per_seed_summaries[0].get("split_snapshot", {}) if per_seed_summaries else {},
        "feature_columns": per_seed_summaries[0].get("feature_columns", []) if per_seed_summaries else [],
        "per_seed_runs": per_seed_runs,
        "aggregate_metrics": aggregate,
        "best_checkpoint": best_checkpoint,
        "artifact_paths": {
            "summary": serialize_path(artifact_root / f"{model_type}_multi_seed_summary.json"),
            "plots_dir": serialize_path(plots_dir),
        },
        "generated_at": datetime.utcnow().isoformat(),
    }
    plots = [
        plot_metric_summary(
            summary,
            plots_dir / "metrics_overview.png",
            title=f"{source.upper()} {model_type} multi-seed metrics",
            description="Mean and std across the configured random seeds.",
        ),
        plot_error_distribution(
            per_seed_runs,
            plots_dir / "error_distribution.png",
            title=f"{source.upper()} {model_type} error distribution",
            description="Prediction error histogram for every configured seed.",
        ),
        plot_training_curves(
            per_seed_runs,
            plots_dir / "training_curves.png",
            title=f"{source.upper()} {model_type} training curves",
            description="Train/validation loss curves for the configured seed runs.",
        ),
    ]
    summary["plots"] = plots
    write_plot_manifest(plots_dir)
    write_json(artifact_root / f"{model_type}_multi_seed_summary.json", summary)
    return summary


def create_ablation_summary(
    source: str,
    *,
    variants: list[dict[str, Any]],
    model_dir: str | Path = "data/models",
) -> dict[str, Any]:
    from ml.training.benchmark_truth import SUMMARY_VERSION, TASK_KIND, build_ablation_guard

    artifact_root = resolve_path(model_dir) / source
    plots_dir = artifact_root / "plots"
    full_variant = next((item for item in variants if item.get("key") == "full_hybrid"), None)
    full_rmse = (((full_variant or {}).get("aggregate_metrics") or {}).get("mean") or {}).get("rmse")
    full_r2 = (((full_variant or {}).get("aggregate_metrics") or {}).get("mean") or {}).get("r2")
    for variant in variants:
        metrics = (((variant.get("aggregate_metrics") or {}).get("mean")) or {})
        current_rmse = metrics.get("rmse")
        current_r2 = metrics.get("r2")
        delta_rmse = None
        delta_r2 = None
        if isinstance(current_rmse, (int, float)) and isinstance(full_rmse, (int, float)):
            delta_rmse = round(float(current_rmse) - float(full_rmse), 6)
        if isinstance(current_r2, (int, float)) and isinstance(full_r2, (int, float)):
            delta_r2 = round(float(current_r2) - float(full_r2), 6)
        variant["delta_vs_full"] = {"rmse": delta_rmse, "r2": delta_r2}
    summary = {
        "source": source,
        "task_kind": (full_variant or {}).get("task_kind", TASK_KIND),
        "summary_version": SUMMARY_VERSION,
        "available": True,
        "variants": variants,
        "notes": [
            "Ablation summaries reuse the same fixed battery split and the same seed set as the main multi-seed experiment.",
            "Delta is computed against full_hybrid using aggregate RMSE.",
        ],
        "artifact_paths": {
            "summary": serialize_path(artifact_root / "ablation_summary.json"),
            "plots_dir": serialize_path(plots_dir),
        },
        "generated_at": datetime.utcnow().isoformat(),
    }
    summary["guardrail"] = build_ablation_guard(summary)
    summary["plots"] = [
        plot_ablation_overview(
            variants,
            plots_dir / "ablation_overview.png",
            title=f"{source.upper()} ablation overview",
            description="Aggregate RMSE for each hybrid ablation variant.",
        )
    ]
    write_plot_manifest(plots_dir)
    write_json(artifact_root / "ablation_summary.json", summary)
    return summary


def load_json(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def _comparison_ready_summary(
    multi_seed_summary: dict[str, Any] | None,
    experiment_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if multi_seed_summary:
        return multi_seed_summary
    metrics = (experiment_summary or {}).get("test_metrics") or {}
    if not metrics:
        return None
    std = {key: 0.0 for key in metrics if isinstance(metrics.get(key), (int, float))}
    return {
        "aggregate_metrics": {
            "mean": metrics,
            "std": std,
        }
    }


def generate_source_plot_bundle(
    source: str,
    *,
    model_dir: str | Path = "data/models",
    processed_dir: str | Path = "data/processed",
) -> list[dict[str, Any]]:
    source = source.lower()
    model_root = resolve_path(model_dir) / source
    processed_root = resolve_path(processed_dir) / source
    plots_dir = model_root / "plots"
    plots: list[dict[str, Any]] = []
    bilstm_summary = _comparison_ready_summary(
        load_json(model_root / "bilstm" / "bilstm_multi_seed_summary.json"),
        load_json(model_root / "bilstm" / "bilstm_experiment_summary.json"),
    )
    hybrid_summary = _comparison_ready_summary(
        load_json(model_root / "hybrid" / "hybrid_multi_seed_summary.json"),
        load_json(model_root / "hybrid" / "hybrid_experiment_summary.json"),
    )
    if bilstm_summary and hybrid_summary:
        plots.append(
            plot_source_comparison(
                source,
                {"bilstm": bilstm_summary, "hybrid": hybrid_summary},
                plots_dir / "experiment_summary.png",
                title=f"{source.upper()} model comparison",
                description="Aggregate multi-seed metric comparison between Bi-LSTM and Hybrid.",
            )
        )
    split_payload = load_json(processed_root / f"{source}_split.json")
    if split_payload:
        plots.append(
            plot_split_overview(
                split_payload,
                plots_dir / "dataset_split.png",
                title=f"{source.upper()} dataset split",
                description="Battery counts in the fixed train/validation/test split.",
            )
        )
    ablation_summary = load_json(model_root / "ablation_summary.json")
    if ablation_summary:
        plots.append(
            plot_ablation_overview(
                ablation_summary.get("variants", []),
                plots_dir / "experiment_ablations.png",
                title=f"{source.upper()} ablation summary",
                description="Aggregate RMSE comparison for the configured ablation variants.",
            )
        )
    write_plot_manifest(plots_dir)
    return plots


__all__ = [
    "ABLATION_VARIANTS",
    "DEFAULT_SEEDS",
    "create_ablation_summary",
    "create_multi_seed_summary",
    "default_config_for",
    "generate_source_plot_bundle",
    "load_json",
    "load_yaml",
    "merge_configs",
    "resolve_path",
    "run_training_experiment",
]
