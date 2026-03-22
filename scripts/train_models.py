#!/usr/bin/env python3
"""按来源训练 BHMS 模型并输出生命周期实验摘要。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from typing import Any

import yaml

from backend.app.core.database import get_database
from backend.app.services.repository import BHMSRepository
from ml.data import LifecycleDataModule, LifecycleTargetConfig, RULDataModule
from ml.data.processed_paths import resolve_cycle_summary_path
from ml.data.source_registry import get_dataset_card, list_supported_sources
from ml.training import LifecycleTrainer, LifecycleTrainingConfig, build_lifecycle_model, run_lifecycle_experiment
from ml.training.experiment_runner import merge_configs, run_training_experiment
from ml.training.lifecycle_experiment_runner import MODEL_ARTIFACT_ALIASES
from ml.training.trainer import RULTrainer, TrainingConfig, build_model


MODEL_CHOICES = ["bilstm", "hybrid", "lifecycle_bilstm", "lifecycle_hybrid"]


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def artifact_model_type(model_type: str) -> str:
    return MODEL_ARTIFACT_ALIASES.get(model_type.lower(), model_type.lower())


def target_config_from_payload(payload: dict[str, Any]) -> LifecycleTargetConfig:
    config = dict(payload or {})
    if "observation_ratios" in config:
        config["observation_ratios"] = tuple(float(item) for item in config["observation_ratios"])
    return LifecycleTargetConfig(**config)


def build_eval_only_summary(
    *,
    source: str,
    task_kind: str,
    requested_model_type: str,
    model_type: str,
    mode: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    history = {
        "train": [],
        "val": [metrics] if mode == "validate_only" else [],
        "test": [metrics] if mode == "test_only" else [],
    }
    return {
        "source": source,
        "task_kind": task_kind,
        "requested_model_type": requested_model_type,
        "model_type": model_type,
        "mode": mode,
        "status": "completed",
        "best_val_loss": metrics.get("loss") if mode == "validate_only" else None,
        "best_checkpoint": None,
        "final_checkpoint": None,
        "history": history,
        "history_summary": {
            "epochs_ran": 0,
            "last_train_loss": None,
            "last_val_loss": metrics.get("loss") if mode == "validate_only" else None,
        },
        "validation_metrics": metrics if mode == "validate_only" else {},
        "test_metrics": metrics if mode == "test_only" else {},
    }


def run_lifecycle_validation_or_test(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))
    training_cfg = dict(config.get("training", {}))
    target_cfg = target_config_from_payload(data_cfg.get("target_config", {}))
    model_cfg.setdefault("future_len", target_cfg.future_len)
    artifact_model = artifact_model_type(args.model)
    csv_path = resolve_cycle_summary_path(data_cfg["csv_path"], source=args.source)
    data_module = LifecycleDataModule(
        csv_path=csv_path,
        source=data_cfg.get("sources") or args.source,
        batch_size=int(training_cfg.get("batch_size", 16)),
        feature_cols=data_cfg.get("feature_columns"),
        output_dir=csv_path.parent,
        target_config=target_cfg,
    )
    data_summary = data_module.summary(path_root=PROJECT_ROOT)
    data_module.export_metadata(path_root=PROJECT_ROOT)
    model, model_config = build_lifecycle_model(
        args.model,
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
            source=args.source,
            model_type=artifact_model,
            **{
                **{key: value for key, value in training_cfg.items() if key != "batch_size"},
                **({"resume_from": args.resume} if args.resume else {}),
            },
        ),
        train_loader=data_module.train_loader(),
        val_loader=data_module.val_loader(),
        test_loader=data_module.test_loader(),
        data_summary=data_summary,
    )
    if args.validate_only:
        return {"val": trainer._run_epoch(trainer.val_loader, training=False)}
    return {"test": trainer.test()}


def run_rul_validation_or_test(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = dict(config.get("data", {}))
    model_cfg = dict(config.get("model", {}))
    training_cfg = dict(config.get("training", {}))
    artifact_model = artifact_model_type(args.model)
    csv_path = resolve_cycle_summary_path(data_cfg["csv_path"], source=args.source)
    data_module = RULDataModule(
        csv_path=csv_path,
        source=args.source,
        seq_len=int(data_cfg.get("seq_len", 30)),
        batch_size=int(training_cfg.get("batch_size", 16)),
        feature_cols=data_cfg.get("feature_columns"),
        output_dir=csv_path.parent,
    )
    data_summary = data_module.summary(path_root=PROJECT_ROOT)
    data_module.export_metadata(path_root=PROJECT_ROOT)

    model, model_config = build_model(artifact_model, input_dim=len(data_summary["feature_columns"]), overrides=model_cfg)
    trainer = RULTrainer(
        model=model,
        model_config=model_config,
        training_config=TrainingConfig(
            source=args.source,
            model_type=artifact_model,
            resume_from=args.resume,
            **training_cfg,
        ),
        train_loader=data_module.train_loader(),
        val_loader=data_module.val_loader(),
        test_loader=data_module.test_loader(),
        data_summary=data_summary,
    )
    if args.validate_only:
        return {"val": trainer._run_epoch(trainer.val_loader, training=False)}
    return {"test": trainer.test()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BHMS source-specific models")
    parser.add_argument("--source", choices=list_supported_sources(), required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--task", choices=["lifecycle", "rul"], default="lifecycle")
    parser.add_argument("--stage-kind", choices=["baseline", "pretrain", "fine_tune", "final_evaluation"], default="baseline")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--init-from-checkpoint", default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--override-json", default=None, help="JSON string merged into the YAML config")
    args = parser.parse_args()
    card = get_dataset_card(args.source)
    if not card.training_ready:
        raise SystemExit(f"{args.source} is marked as {card.ingestion_mode} and does not support lifecycle benchmark training.")

    get_database().initialize()
    config = load_yaml(args.config)
    config_overrides = json.loads(args.override_json) if args.override_json else None
    if args.override_json:
        config = merge_configs(config, config_overrides or {})

    if not args.validate_only and not args.test_only:
        training_overrides = {}
        if args.resume:
            training_overrides["resume_from"] = args.resume
        if args.init_from_checkpoint:
            training_overrides["init_from_checkpoint"] = args.init_from_checkpoint
        if not training_overrides:
            training_overrides = None
        if args.task == "lifecycle":
            result = run_lifecycle_experiment(
                args.source,
                args.model,
                config_path=args.config,
                config_overrides=config_overrides,
                training_overrides=training_overrides,
                stage_kind=args.stage_kind,
                repository=BHMSRepository(),
                persist_training_run=True,
            )
        else:
            result = run_training_experiment(
                args.source,
                artifact_model_type(args.model),
                config_path=args.config,
                config_overrides=config_overrides,
                training_overrides=training_overrides,
                repository=BHMSRepository(),
                persist_training_run=True,
            )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Saved summary to: {result['summary_path']}")
        return

    artifact_model = artifact_model_type(args.model)
    raw_result = run_lifecycle_validation_or_test(args, config) if args.task == "lifecycle" else run_rul_validation_or_test(args, config)
    mode = "validate_only" if args.validate_only else "test_only"
    metrics_key = "val" if args.validate_only else "test"
    metrics = raw_result.get(metrics_key, {})
    result = build_eval_only_summary(
        source=args.source,
        task_kind=args.task,
        requested_model_type=args.model,
        model_type=artifact_model,
        mode=mode,
        metrics=metrics,
    )
    result[metrics_key] = metrics
    summary_root = resolve_cycle_summary_path(config["data"]["csv_path"], source=args.source).parent
    summary_path = summary_root / f"{artifact_model}_experiment_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
