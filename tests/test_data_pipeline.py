"""数据链路测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data import NASABatteryPreprocessor, RULDataModule, create_synthetic_data  # noqa: E402


def test_synthetic_data_and_datamodule(tmp_path: Path):
    csv_path = tmp_path / "synthetic.csv"
    create_synthetic_data(csv_path, num_batteries=6, num_cycles=60)
    data_module = RULDataModule(csv_path, seq_len=20, batch_size=4)
    summary = data_module.summary()
    assert summary["num_samples"]["train"] > 0
    assert summary["num_samples"]["val"] > 0
    assert summary["num_samples"]["test"] > 0


def test_nasa_preprocessor_parses_one_file():
    nasa_file = PROJECT_ROOT / "data" / "raw" / "nasa" / "B0005.mat"
    processor = NASABatteryPreprocessor()
    frame = processor.parse_battery_file(nasa_file)
    assert not frame.empty
    assert {"battery_id", "cycle_number", "capacity", "RUL"}.issubset(frame.columns)
