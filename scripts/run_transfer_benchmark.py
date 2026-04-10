#!/usr/bin/env python3
"""Run lifecycle transfer benchmarks: multisource pretrain -> target fine-tune."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training import run_transfer_benchmark


DEFAULT_SEEDS = [7, 21, 42]


def parse_seeds(raw: str | None) -> list[int]:
    if not raw:
        return list(DEFAULT_SEEDS)
    return [int(item.strip()) for item in raw.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BHMS lifecycle transfer benchmark")
    parser.add_argument("--target", choices=["calce", "nasa"], required=True)
    parser.add_argument("--model", choices=["bilstm", "hybrid"], default="hybrid")
    parser.add_argument("--pretrain-config", required=True)
    parser.add_argument("--finetune-config", required=True)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds, default: 7,21,42")
    parser.add_argument("--model-dir", default="data/models")
    args = parser.parse_args()

    summary = run_transfer_benchmark(
        args.target,
        args.model,
        pretrain_config_path=args.pretrain_config,
        fine_tune_config_path=args.finetune_config,
        seeds=parse_seeds(args.seeds),
        model_dir=args.model_dir,
        persist_training_run=True,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
