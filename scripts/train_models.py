#!/usr/bin/env python3
"""按来源训练 BHMS 模型并输出对比实验摘要。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from backend.app.core.database import get_database
from backend.app.services.repository import BHMSRepository
from ml.data import RULDataModule
from ml.training.experiment_runner import run_training_experiment
from ml.training.trainer import RULTrainer, TrainingConfig, build_model


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BHMS source-specific models")
    parser.add_argument("--source", choices=["nasa", "calce", "kaggle"], required=True)
    parser.add_argument("--model", choices=["bilstm", "hybrid"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--test-only", action="store_true")
    args = parser.parse_args()

    get_database().initialize()
    if not args.validate_only and not args.test_only:
        result = run_training_experiment(
            args.source,
            args.model,
            config_path=args.config,
            repository=BHMSRepository(),
            persist_training_run=True,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"Saved summary to: {result['summary_path']}")
        return

    config = load_yaml(args.config)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    training_cfg = config.get("training", {})

    data_module = RULDataModule(
        csv_path=data_cfg["csv_path"],
        source=args.source,
        seq_len=int(data_cfg.get("seq_len", 30)),
        batch_size=int(training_cfg.get("batch_size", 16)),
        feature_cols=data_cfg.get("feature_columns"),
        output_dir=Path(data_cfg["csv_path"]).parent,
    )
    data_summary = data_module.summary()
    data_module.export_metadata()

    model, model_config = build_model(args.model, input_dim=len(data_summary["feature_columns"]), overrides=model_cfg)
    trainer = RULTrainer(
        model=model,
        model_config=model_config,
        training_config=TrainingConfig(
            source=args.source,
            model_type=args.model,
            resume_from=args.resume,
            **training_cfg,
        ),
        train_loader=data_module.train_loader(),
        val_loader=data_module.val_loader(),
        test_loader=data_module.test_loader(),
        data_summary=data_summary,
    )

    if args.validate_only:
        result = {"val": trainer._run_epoch(trainer.val_loader, training=False)}
    elif args.test_only:
        result = {"test": trainer.test()}

    summary_path = Path(Path(data_cfg["csv_path"]).parent) / f"{args.model}_experiment_summary.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
