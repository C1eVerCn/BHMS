"""数据链路测试。"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings  # noqa: E402
from ml.data import NASABatteryPreprocessor, RULDataModule, create_synthetic_data  # noqa: E402
from ml.data.adapters import CALCEAdapter, KaggleAdapter  # noqa: E402
from scripts.refresh_processed_baselines import refresh_processed_baselines  # noqa: E402


def _write_source_csv(path: Path, source_prefix: str) -> None:
    rows = []
    for battery_index in range(4):
        battery_id = f"{source_prefix}_{battery_index:03d}"
        for cycle in range(1, 26):
            rows.append(
                {
                    "source_battery_id": battery_id,
                    "cycle_number": cycle,
                    "voltage_mean": 3.7,
                    "current_mean": -1.8,
                    "temperature_mean": 25.0,
                    "capacity": max(1.0, 2.0 - cycle * 0.015),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _make_settings(tmp_path: Path):
    base = get_settings()
    data_dir = tmp_path / "data"
    return replace(
        base,
        project_root=tmp_path,
        data_dir=data_dir,
        raw_nasa_dir=data_dir / "raw" / "nasa",
        raw_calce_dir=data_dir / "raw" / "calce",
        raw_kaggle_dir=data_dir / "raw" / "kaggle",
        processed_dir=data_dir / "processed",
        knowledge_path=PROJECT_ROOT / "data" / "knowledge" / "battery_fault_knowledge.json",
        model_dir=data_dir / "models",
        upload_dir=data_dir / "uploads",
        demo_upload_dir=data_dir / "demo_uploads",
        database_path=data_dir / "bhms.db",
        graph_backend="memory",
    )


def test_synthetic_data_and_datamodule(tmp_path: Path):
    csv_path = tmp_path / "synthetic.csv"
    create_synthetic_data(csv_path, num_batteries=6, num_cycles=60)
    data_module = RULDataModule(csv_path, source="synthetic", seq_len=20, batch_size=4)
    summary = data_module.summary()
    assert summary["num_samples"]["train"] > 0
    assert summary["num_samples"]["val"] > 0
    assert summary["num_samples"]["test"] > 0
    metadata = data_module.export_metadata()
    assert Path(metadata["summary"]).exists()


def test_nasa_preprocessor_parses_one_file():
    nasa_file = PROJECT_ROOT / "data" / "raw" / "nasa" / "B0005.mat"
    processor = NASABatteryPreprocessor()
    frame = processor.parse_battery_file(nasa_file)
    assert not frame.empty
    assert {"battery_id", "canonical_battery_id", "capacity", "RUL", "source"}.issubset(frame.columns)


def test_calce_and_kaggle_csv_adapters(tmp_path: Path):
    calce_csv = tmp_path / "demo_calce.csv"
    kaggle_csv = tmp_path / "demo_kaggle.csv"
    _write_source_csv(calce_csv, "CALCE")
    _write_source_csv(kaggle_csv, "KAGGLE")

    calce_frame = CALCEAdapter().process_file(calce_csv)
    kaggle_frame = KaggleAdapter().process_file(kaggle_csv)

    assert set(calce_frame["source"].unique()) == {"calce"}
    assert set(kaggle_frame["source"].unique()) == {"kaggle"}
    assert calce_frame["canonical_battery_id"].nunique() == 4
    assert kaggle_frame["canonical_battery_id"].nunique() == 4


def test_refresh_processed_baselines_rebuilds_relative_metadata_without_demo_uploads(tmp_path: Path):
    settings = _make_settings(tmp_path)
    raw_calce = settings.raw_calce_dir / "calce_source.csv"
    raw_kaggle = settings.raw_kaggle_dir / "kaggle_source.csv"
    _write_source_csv(raw_calce, "CALCE")
    _write_source_csv(raw_kaggle, "KAGGLE")

    demo_calce = settings.demo_upload_dir / "calce" / "calce_unseen_demo.csv"
    _write_source_csv(demo_calce, "CALCE_UNSEEN")

    stale_dir = settings.processed_dir / "calce"
    stale_dir.mkdir(parents=True, exist_ok=True)
    (stale_dir / "calce_split.json").write_text(
        '{"train_batteries":["calce::calce_demo::CALCE_UNSEEN_000"],"val_batteries":["calce::calce_demo::CALCE_UNSEEN_001"],"test_batteries":["calce::calce_demo::CALCE_UNSEEN_002"]}',
        encoding="utf-8",
    )
    (stale_dir / "calce_feature_config.json").write_text(
        '{"source":"calce","feature_columns":["capacity","cycle_number"],"target_column":"RUL"}',
        encoding="utf-8",
    )

    first_pass = refresh_processed_baselines(
        sources=["calce", "kaggle"],
        settings=settings,
        seq_len=30,
        batch_size=16,
    )
    snapshot = {
        path.name: path.read_bytes()
        for path in sorted((settings.processed_dir / "calce").glob("calce_*"))
    }
    second_pass = refresh_processed_baselines(
        sources=["calce", "kaggle"],
        settings=settings,
        seq_len=30,
        batch_size=16,
    )

    calce_summary = first_pass["calce"]["data_summary"]
    assert calce_summary["csv_path"] == "data/processed/calce/calce_cycle_summary.csv"
    assert calce_summary["provenance"]["kind"] == "repo_baseline"
    assert calce_summary["seq_len"] == 30
    assert calce_summary["batch_size"] == 16

    calce_frame = CALCEAdapter().process_file(raw_calce)
    committed_frame = pd.read_csv(settings.processed_dir / "calce" / "calce_cycle_summary.csv")
    assert set(committed_frame["canonical_battery_id"]) == set(calce_frame["canonical_battery_id"])
    assert not committed_frame["source_battery_id"].str.contains("UNSEEN").any()

    feature_config = json.loads((settings.processed_dir / "calce" / "calce_feature_config.json").read_text(encoding="utf-8"))
    assert "voltage_mean" in feature_config["feature_columns"]
    assert feature_config["feature_columns"][-2:] == ["capacity", "cycle_number"]

    summary_json = json.loads((settings.processed_dir / "calce" / "calce_dataset_summary.json").read_text(encoding="utf-8"))
    assert summary_json["csv_path"] == "data/processed/calce/calce_cycle_summary.csv"
    assert "/Users/chris/" not in (settings.processed_dir / "calce" / "calce_dataset_summary.json").read_text(encoding="utf-8")

    assert first_pass["kaggle"]["data_summary"]["csv_path"] == "data/processed/kaggle/kaggle_cycle_summary.csv"
    assert snapshot == {
        path.name: path.read_bytes()
        for path in sorted((settings.processed_dir / "calce").glob("calce_*"))
    }
    assert second_pass["calce"]["data_summary"]["provenance"]["generated_by"] == "scripts/refresh_processed_baselines.py"
