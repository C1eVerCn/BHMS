"""Lifecycle experiment runner built on the xLSTM-centered forecasting stack."""

from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch

from backend.app.core.database import get_database
from backend.app.services.repository import BHMSRepository
from ml.data import LifecycleDataModule, LifecycleTargetConfig
from ml.data.processed_paths import resolve_cycle_summary_path
from ml.training.experiment_artifacts import serialize_path, write_json
from ml.training.experiment_runner import (
    create_ablation_summary,
    create_multi_seed_summary,
    default_config_for,
    generate_source_plot_bundle,
    load_yaml,
    merge_configs,
    resolve_path,
)
from ml.training.lifecycle_trainer import LifecycleTrainer, LifecycleTrainingConfig, build_lifecycle_model


MODEL_ARTIFACT_ALIASES = {
    "hybrid": "hybrid",
    "bilstm": "bilstm",
    "lifecycle_hybrid": "hybrid",
    "lifecycle_bilstm": "bilstm",
}


def _history_summary(history: dict[str, Any]) -> dict[str, Any]:
    train_history = history.get("train", []) or []
    val_history = history.get("val", []) or []
    return {
        "epochs_ran": len(train_history),
        "last_train_loss": train_history[-1].get("loss") if train_history else None,
        "last_val_loss": val_history[-1].get("loss") if val_history else None,
    }


def _target_config_from_payload(payload: dict[str, Any]) -> LifecycleTargetConfig:
    config = dict(payload or {})
    if "observation_ratios" in config:
        config["observation_ratios"] = tuple(float(item) for item in config["observation_ratios"])
    return LifecycleTargetConfig(**config)


def run_lifecycle_experiment(
    source: str,
    model_type: str,
    *,
    config_path: str | Path | None = None,
    config_overrides: Optional[dict[str, Any]] = None,
    training_overrides: Optional[dict[str, Any]] = None,
    artifact_subdir: str | None = None,
    suite_kind: str = "baseline",
    variant_key: str = "baseline",
    stage_kind: str = "baseline",
    repository: Optional[BHMSRepository] = None,
    persist_training_run: bool = True,
) -> dict[str, Any]:
    source = source.lower()
    requested_model_type = model_type.lower()
    artifact_model_type = MODEL_ARTIFACT_ALIASES.get(requested_model_type, requested_model_type)
    resolved_config = resolve_path(config_path or default_config_for(source, artifact_model_type))
    config = load_yaml(resolved_config)
    merged = merge_configs(config, config_overrides or {})
    data_cfg = dict(merged.get("data", {}))
    model_cfg = dict(merged.get("model", {}))
    training_cfg = merge_configs(dict(merged.get("training", {})), training_overrides or {})
    target_cfg = _target_config_from_payload(data_cfg.get("target_config", {}))
    model_cfg.setdefault("future_len", target_cfg.future_len)
    seed = int(training_cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if persist_training_run and repository is None:
        get_database().initialize()
    repo = repository or BHMSRepository()

    csv_paths_cfg = data_cfg.get("csv_paths")
    if csv_paths_cfg:
        csv_paths = [resolve_cycle_summary_path(resolve_path(path)) for path in csv_paths_cfg]
        csv_path = csv_paths[0]
        merged.setdefault("data", {})["csv_paths"] = [serialize_path(path) for path in csv_paths]
        data_cfg["csv_paths"] = merged["data"]["csv_paths"]
        if "csv_path" in merged.get("data", {}):
            merged["data"].pop("csv_path", None)
            data_cfg.pop("csv_path", None)
    else:
        csv_path = resolve_cycle_summary_path(resolve_path(data_cfg["csv_path"]), source=source)
        csv_paths = [csv_path]
        merged.setdefault("data", {})["csv_path"] = serialize_path(csv_path)
        data_cfg["csv_path"] = merged["data"]["csv_path"]
    source_scope = data_cfg.get("sources") or source
    checkpoint_root = resolve_path(training_cfg.get("checkpoint_dir", "data/models"))
    log_root = resolve_path(training_cfg.get("log_dir", "data/models/logs"))
    if len(csv_paths) > 1:
        data_output_dir = checkpoint_root / source / artifact_model_type
        if artifact_subdir:
            data_output_dir = data_output_dir / artifact_subdir
        data_output_dir = data_output_dir / "data_profile"
    else:
        data_output_dir = csv_path.parent
    data_module = LifecycleDataModule(
        csv_path=csv_paths if len(csv_paths) > 1 else csv_path,
        source=source_scope,
        batch_size=int(training_cfg.get("batch_size", 16)),
        feature_cols=data_cfg.get("feature_columns"),
        output_dir=data_output_dir,
        seed=seed,
        target_config=target_cfg,
    )
    data_summary = data_module.summary()
    if artifact_subdir:
        root_summary_path = data_output_dir / f"{source}_lifecycle_dataset_summary.json"
        if not root_summary_path.exists():
            data_module.export_metadata()
        profile_prefix = variant_key if suite_kind == "ablation" else source
        target_dir = (
            checkpoint_root
            / source
            / artifact_model_type
            / artifact_subdir
            / "data_profile"
        )
        data_module.export_metadata(target_dir, file_prefix=profile_prefix)
    else:
        data_module.export_metadata()

    model, model_config = build_lifecycle_model(
        requested_model_type,
        input_dim=len(data_summary["feature_columns"]),
        vocab_sizes={
            "source": len(data_summary["domain_vocab"]["source_to_id"]),
            "chemistry": len(data_summary["domain_vocab"]["chemistry_to_id"]),
            "protocol": len(data_summary["domain_vocab"]["protocol_to_id"]),
        },
        overrides=model_cfg,
    )
    trainer = LifecycleTrainer(
        model=model,
        model_config=model_config,
        training_config=LifecycleTrainingConfig(
            source=source,
            model_type=artifact_model_type,
            checkpoint_dir=str(checkpoint_root),
            log_dir=str(log_root),
            artifact_subdir=artifact_subdir,
            **{key: value for key, value in training_cfg.items() if key not in {"checkpoint_dir", "log_dir", "batch_size"}},
        ),
        train_loader=data_module.train_loader(),
        val_loader=data_module.val_loader(),
        test_loader=data_module.test_loader(),
        data_summary=data_summary,
    )
    result = trainer.fit()
    summary = {
        **result,
        "source": source,
        "task_kind": "lifecycle",
        "model_type": artifact_model_type,
        "requested_model_type": requested_model_type,
        "suite_kind": suite_kind,
        "variant_key": variant_key,
        "stage_kind": stage_kind,
        "source_scope": source_scope if isinstance(source_scope, list) else [source_scope],
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
        "test_details_path": serialize_path(trainer.checkpoint_dir / "test_details.json"),
    }
    write_json(trainer.checkpoint_dir / "test_details.json", result.get("test_details", {}))
    summary_path = trainer.checkpoint_dir / f"{artifact_model_type}_experiment_summary.json"
    write_json(summary_path, summary)
    summary["summary_path"] = serialize_path(summary_path)
    if not (trainer.checkpoint_dir / "test_details.json").exists() or not summary_path.exists():
        raise RuntimeError(f"Lifecycle experiment artifacts are incomplete for {source}/{artifact_model_type}.")

    if persist_training_run:
        repo.insert_training_run(
            {
                "source": source,
                "model_type": artifact_model_type,
                "model_version": trainer.config.model_version,
                "best_checkpoint_path": summary.get("best_checkpoint"),
                "final_checkpoint_path": summary.get("final_checkpoint"),
                "metrics": summary.get("test_metrics", {}),
                "metadata": {
                    "task_kind": "lifecycle",
                    "requested_model_type": requested_model_type,
                    "seed": trainer.config.seed,
                    "suite_kind": suite_kind,
                    "variant_key": variant_key,
                    "stage_kind": stage_kind,
                    "source_scope": source_scope if isinstance(source_scope, list) else [source_scope],
                    "artifact_dir": serialize_path(trainer.checkpoint_dir),
                    "config_path": serialize_path(resolved_config),
                    "config_snapshot": merged,
                    "split_snapshot": data_summary.get("split", {}),
                    "feature_columns": data_summary.get("feature_columns", []),
                    "summary_path": serialize_path(summary_path),
                    "test_details_path": serialize_path(trainer.checkpoint_dir / "test_details.json"),
                },
            }
        )
    return summary


__all__ = [
    "create_ablation_summary",
    "create_multi_seed_summary",
    "default_config_for",
    "generate_source_plot_bundle",
    "run_lifecycle_experiment",
]
