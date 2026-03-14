#!/usr/bin/env python3
"""按来源准备 BHMS 训练数据。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from pathlib import Path

from backend.app.core.config import get_settings
from backend.app.core.database import get_database
from backend.app.services.battery_service import BatteryService
from ml.data.dataset import RULDataModule


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BHMS source datasets")
    parser.add_argument("--source", choices=["nasa", "calce", "kaggle"], required=True)
    parser.add_argument("--include-in-training", action="store_true")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    get_database().initialize()
    settings = get_settings()
    service = BatteryService(settings=settings)
    summary = service.import_builtin_source(args.source, include_in_training=args.include_in_training)
    source_csv = settings.processed_dir / args.source / f"{args.source}_cycle_summary.csv"
    data_module = RULDataModule(
        csv_path=source_csv,
        source=args.source,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        output_dir=settings.processed_dir / args.source,
    )
    metadata_paths = data_module.export_metadata()
    payload = {
        "import_summary": summary,
        "data_summary": data_module.summary(),
        "metadata_paths": metadata_paths,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
