"""数据链路测试。"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data import NASABatteryPreprocessor, RULDataModule, create_synthetic_data  # noqa: E402
from ml.data.adapters import CALCEAdapter, KaggleAdapter  # noqa: E402


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
