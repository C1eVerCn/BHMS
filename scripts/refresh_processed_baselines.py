#!/usr/bin/env python3
"""从 data/raw 重建仓库级 processed 基线。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import Settings, get_settings
from ml.data.adapters import CALCEAdapter, KaggleAdapter, NASAAdapter
from ml.data.dataset import RULDataModule
from ml.data.schema import TRAINING_FEATURE_COLUMNS

SUPPORTED_SOURCES = ("nasa", "calce", "kaggle")


def refresh_processed_baselines(
    *,
    sources: list[str] | None = None,
    settings: Settings | None = None,
    seq_len: int = 30,
    batch_size: int = 16,
    force: bool = False,
) -> dict[str, dict[str, Any]]:
    del force  # 基线刷新默认就是覆盖写入，保留该参数用于 CLI 兼容性。
    runtime_settings = settings or get_settings()
    selected_sources = [item.lower() for item in (sources or list(SUPPORTED_SOURCES))]
    results: dict[str, dict[str, Any]] = {}
    for source in selected_sources:
        if source not in SUPPORTED_SOURCES:
            raise ValueError(f"Unsupported source: {source}")
        results[source] = refresh_processed_source(
            source=source,
            settings=runtime_settings,
            seq_len=seq_len,
            batch_size=batch_size,
        )
    return results


def refresh_processed_source(
    *,
    source: str,
    settings: Settings,
    seq_len: int,
    batch_size: int,
) -> dict[str, Any]:
    adapter = _adapter_for(source, settings)
    raw_dir = _raw_dir_for(source, settings)
    output_dir = settings.processed_dir / source
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_csv = output_dir / f"{source}_cycle_summary.csv"

    frame = adapter.process_directory(raw_dir, output_path=processed_csv)
    data_module = RULDataModule(
        csv_path=processed_csv,
        source=source,
        seq_len=seq_len,
        batch_size=batch_size,
        feature_cols=TRAINING_FEATURE_COLUMNS,
        output_dir=output_dir,
        reuse_existing_split=False,
    )
    provenance = {
        "kind": "repo_baseline",
        "source_root": f"data/raw/{source}",
        "generated_by": "scripts/refresh_processed_baselines.py",
    }
    metadata_paths = data_module.export_metadata(
        path_root=settings.project_root,
        provenance=provenance,
    )
    return {
        "source": source,
        "csv_path": str(processed_csv),
        "row_count": int(len(frame)),
        "battery_count": int(frame["canonical_battery_id"].nunique()),
        "metadata_paths": metadata_paths,
        "data_summary": data_module.summary(path_root=settings.project_root, provenance=provenance),
    }


def _adapter_for(source: str, settings: Settings):
    mapping = {
        "nasa": NASAAdapter(eol_capacity_ratio=settings.battery_eol_ratio),
        "calce": CALCEAdapter(eol_capacity_ratio=settings.battery_eol_ratio),
        "kaggle": KaggleAdapter(eol_capacity_ratio=settings.battery_eol_ratio),
    }
    return mapping[source]


def _raw_dir_for(source: str, settings: Settings) -> Path:
    mapping = {
        "nasa": settings.raw_nasa_dir,
        "calce": settings.raw_calce_dir,
        "kaggle": settings.raw_kaggle_dir,
    }
    return mapping[source]


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh committed processed baselines from data/raw")
    parser.add_argument("--sources", nargs="+", choices=list(SUPPORTED_SOURCES), default=list(SUPPORTED_SOURCES))
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    payload = refresh_processed_baselines(
        sources=args.sources,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        force=args.force,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
