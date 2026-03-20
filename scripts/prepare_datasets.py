#!/usr/bin/env python3
"""从当前训练池导出 BHMS 本地训练数据。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json

from backend.app.core.database import get_database
from backend.app.services.battery_service import BatteryService
from ml.data.source_registry import get_dataset_card, list_supported_sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BHMS local training-pool datasets")
    parser.add_argument("--source", choices=list_supported_sources(), required=True)
    parser.add_argument("--include-in-training", action="store_true")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    card = get_dataset_card(args.source)
    if not card.training_ready:
        raise SystemExit(f"{args.source} is marked as {card.ingestion_mode} and cannot prepare lifecycle training datasets.")

    get_database().initialize()
    service = BatteryService()
    if args.include_in_training:
        service.import_builtin_source(args.source, include_in_training=True)
    payload = service.prepare_training_dataset(args.source, seq_len=args.seq_len, batch_size=args.batch_size)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
