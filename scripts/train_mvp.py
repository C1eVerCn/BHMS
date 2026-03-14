#!/usr/bin/env python3
"""训练 BHMS MVP 的 Bi-LSTM 或 xLSTM-Transformer 模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.data import RULDataModule
from ml.training.trainer import RULTrainer, TrainingConfig, build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BHMS MVP models")
    parser.add_argument("--data", default="data/processed/nasa_cycle_summary.csv", help="Prepared CSV path")
    parser.add_argument("--model", choices=["bilstm", "hybrid"], default="hybrid")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--checkpoint-dir", default="data/models")
    args = parser.parse_args()

    data_module = RULDataModule(args.data, seq_len=args.seq_len, batch_size=args.batch_size)
    model, model_config = build_model(args.model)
    trainer = RULTrainer(
        model=model,
        model_config=model_config,
        training_config=TrainingConfig(
            model_type=args.model,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir,
            log_dir=str(Path(args.checkpoint_dir) / "logs"),
        ),
        train_loader=data_module.train_loader(),
        val_loader=data_module.val_loader(),
        test_loader=data_module.test_loader(),
    )
    train_result = trainer.train()
    test_result = trainer.test()
    summary = {
        "data": data_module.summary(),
        "train_result": train_result,
        "test_result": test_result,
    }
    summary_path = Path(args.checkpoint_dir) / f"{args.model}_training_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
