"""Lifecycle data and adapter tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.data import LifecycleDataModule, LifecycleTargetConfig, create_synthetic_data  # noqa: E402
from ml.data.adapters import HUSTAdapter  # noqa: E402


def test_hust_adapter_adds_required_metadata(tmp_path: Path):
    csv_path = tmp_path / "hust.csv"
    pd.DataFrame(
        {
            "cell_id": ["HUST_001"] * 8,
            "cycle": list(range(1, 9)),
            "voltage": [3.72, 3.71, 3.70, 3.69, 3.68, 3.67, 3.66, 3.65],
            "current": [-1.0] * 8,
            "temperature": [25.0, 25.1, 25.2, 25.2, 25.3, 25.4, 25.5, 25.6],
            "capacity_ah": [2.05, 2.03, 2.01, 1.99, 1.97, 1.95, 1.93, 1.91],
        }
    ).to_csv(csv_path, index=False)

    frame = HUSTAdapter().process_file(csv_path)

    for column in (
        "chemistry",
        "form_factor",
        "protocol_id",
        "charge_c_rate",
        "discharge_c_rate",
        "ambient_temp",
        "nominal_capacity",
        "eol_ratio",
        "dataset_license",
    ):
        assert column in frame.columns
    assert frame["source"].iloc[0] == "hust"
    assert frame["protocol_id"].iloc[0] == "hust_77cell_cycle_aging"


def test_lifecycle_data_module_builds_multisource_targets(tmp_path: Path):
    calce_frame = create_synthetic_data(tmp_path / "calce.csv", num_batteries=4, num_cycles=48, source="calce", dataset_name="calce_lifecycle")
    hust_frame = create_synthetic_data(tmp_path / "hust.csv", num_batteries=4, num_cycles=48, source="hust", dataset_name="hust_lifecycle")
    combined = pd.concat([calce_frame, hust_frame], ignore_index=True)
    csv_path = tmp_path / "combined.csv"
    combined.to_csv(csv_path, index=False)

    module = LifecycleDataModule(
        csv_path,
        source=["calce", "hust"],
        batch_size=2,
        target_config=LifecycleTargetConfig(future_len=32, encoder_len=24),
    )
    batch = next(iter(module.train_loader()))
    paths = module.export_metadata(tmp_path)

    assert batch["sequence"].shape[-2:] == (24, len(module.feature_cols))
    assert batch["trajectory_target"].shape[-1] == 32
    assert batch["rul_target"].shape[-1] == 1
    assert batch["source_id"].dtype == torch.int64
    assert Path(paths["summary"]).exists()
    assert Path(paths["target_config"]).exists()
