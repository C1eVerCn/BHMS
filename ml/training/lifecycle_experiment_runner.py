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

    csv_path = resolve_path(data_cfg["csv_path"])
    source_scope = data_cfg.get("sources") or source
    data_module = LifecycleDataModule(
        csv_path=csv_path,
        source=source_scope,
        batch_size=int(training_cfg.get("batch_size", 16)),
        feature_cols=data_cfg.get("feature_columns"),
        output_dir=csv_path.parent,
        seed=seed,
        target_config=target_cfg,
    )
    data_summary = data_module.summary()
    if artifact_subdir:
        root_summary_path = csv_path.parent / f"{source}_lifecycle_dataset_summary.json"
        if not root_summary_path.exists():
            data_module.export_metadata()
        profile_prefix = variant_key if suite_kind == "ablation" else source
        target_dir = (
            resolve_path(training_cfg.get("checkpoint_dir", "data/models"))
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
            checkpoint_dir=str(resolve_path(training_cfg.get("checkpoint_dir", "data/models"))),
            log_dir=str(resolve_path(training_cfg.get("log_dir", "data/models/logs"))),
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
        "model_type": artifact_model_type,
        "requested_model_type": requested_model_type,
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
        "history_summary": {
            "epochs_ran": len(result.get("history", {}).get("train", []) or []),
            "last_train_loss": (result.get("history", {}).get("train", []) or [{}])[-1].get("loss"),
            "last_val_loss": (result.get("history", {}).get("val", []) or [{}])[-1].get("loss"),
        },
        "test_details_path": serialize_path(trainer.checkpoint_dir / "test_details.json"),
    }
    write_json(trainer.checkpoint_dir / "test_details.json", result.get("test_details", {}))
    summary_path = trainer.checkpoint_dir / f"{artifact_model_type}_experiment_summary.json"
    write_json(summary_path, summary)
    summary["summary_path"] = serialize_path(summary_path)

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
