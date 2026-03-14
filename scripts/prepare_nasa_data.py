#!/usr/bin/env python3
"""将 NASA 原始 MAT 文件转换为训练用周期级 CSV。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.data import NASABatteryPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare NASA battery dataset for BHMS MVP")
    parser.add_argument("--input-dir", default="data/raw/nasa", help="NASA MAT files directory")
    parser.add_argument("--output", default="data/processed/nasa_cycle_summary.csv", help="Output CSV path")
    parser.add_argument("--battery-id", action="append", dest="battery_ids", help="Optional battery ID filter")
    args = parser.parse_args()

    processor = NASABatteryPreprocessor()
    frame = processor.process_directory(args.input_dir, output_path=args.output, battery_ids=args.battery_ids)
    metadata_path = Path(args.output).with_suffix(".split.json")
    battery_ids = list(frame["battery_id"].unique())
    if len(battery_ids) >= 3:
        split = processor.split_batteries(battery_ids)
        split_payload = split.to_dict()
    else:
        split_payload = {
            "train_batteries": battery_ids,
            "val_batteries": [],
            "test_batteries": [],
        }
    metadata_path.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Prepared rows: {len(frame)}")
    print(f"Saved CSV: {args.output}")
    print(f"Saved split metadata: {metadata_path}")
    print(json.dumps(split_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
