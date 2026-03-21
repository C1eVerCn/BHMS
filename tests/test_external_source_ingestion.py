"""External raw-source ingestion tests."""

from __future__ import annotations

import pickle
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
from scipy.io import savemat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.core.database import DatabaseManager  # noqa: E402
from backend.app.core.exceptions import BHMSException  # noqa: E402
from backend.app.services.battery_service import BatteryService  # noqa: E402
from backend.app.services.repository import BHMSRepository  # noqa: E402
from ml.data.adapters import HUSTAdapter, MATRAdapter, OxfordAdapter, PulseBatAdapter  # noqa: E402
from ml.data.source_registry import (  # noqa: E402
    list_auxiliary_sources,
    list_enhancement_only_sources,
    list_training_ready_sources,
)
from scripts.refresh_processed_baselines import refresh_processed_baselines  # noqa: E402


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
        raw_hust_dir=data_dir / "raw" / "hust",
        raw_matr_dir=data_dir / "raw" / "matr",
        raw_oxford_dir=data_dir / "raw" / "oxford",
        raw_pulsebat_dir=data_dir / "raw" / "pulsebat",
        processed_dir=data_dir / "processed",
        knowledge_path=PROJECT_ROOT / "data" / "knowledge" / "battery_fault_knowledge.json",
        model_dir=data_dir / "models",
        upload_dir=data_dir / "uploads",
        demo_upload_dir=data_dir / "demo_uploads",
        database_path=data_dir / "bhms.db",
        graph_backend="memory",
    )


def _write_hust_zip(raw_dir: Path, battery_count: int = 3, cycle_count: int = 12) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "hust_data.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("our_data/", b"")
        for battery_index in range(battery_count):
            battery_id = f"{battery_index + 1}-{battery_index + 1}"
            cycle_map: dict[int, pd.DataFrame] = {}
            for cycle in range(1, cycle_count + 1):
                cycle_capacity = 1200 - cycle * 20 - battery_index * 15
                cycle_map[cycle] = pd.DataFrame(
                    {
                        "Status": [
                            "Constant current charge",
                            "Constant current-constant voltage charge",
                            "Constant current discharge_0",
                            "Constant current discharge_1",
                            "Constant current discharge_2",
                            "Constant current discharge_3",
                        ],
                        "Cycle number": [cycle] * 6,
                        "Current (mA)": [1500, 1500, -1000, -1000, -1000, -1000],
                        "Voltage (V)": [3.45, 3.85, 4.10, 3.95, 3.75, 3.55],
                        "Capacity (mAh)": [0.0, cycle_capacity * 0.72, cycle_capacity, cycle_capacity * 0.68, cycle_capacity * 0.32, 1.0],
                        "Time (s)": [0, 10, 20, 30, 40, 50],
                    }
                )
            payload = {
                battery_id: {
                    "rul": {cycle: cycle_count + 100 - cycle for cycle in range(1, cycle_count + 1)},
                    "dq": {cycle: 0.0 for cycle in range(1, cycle_count + 1)},
                    "data": cycle_map,
                }
            }
            archive.writestr(f"our_data/{battery_id}.pkl", pickle.dumps(payload))
    return zip_path


def _write_ref_matrix(handle: h5py.File, group: h5py.Group, key: str, arrays: list[np.ndarray], prefix: str) -> None:
    ref_dataset = group.create_dataset(key, shape=(len(arrays), 1), dtype=h5py.ref_dtype)
    for index, values in enumerate(arrays):
        dataset = handle.create_dataset(f"{prefix}_{key}_{index}", data=np.asarray(values, dtype=float))
        ref_dataset[index, 0] = dataset.ref


def _write_matr_batch(raw_dir: Path, file_name: str = "MATR_batch_20170512.mat", cell_count: int = 3, cycle_count: int = 12) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    mat_path = raw_dir / file_name
    with h5py.File(mat_path, "w") as handle:
        batch = handle.create_group("batch")
        summary_refs = batch.create_dataset("summary", shape=(cell_count, 1), dtype=h5py.ref_dtype)
        cycles_refs = batch.create_dataset("cycles", shape=(cell_count, 1), dtype=h5py.ref_dtype)
        policy_refs = batch.create_dataset("policy_readable", shape=(cell_count, 1), dtype=h5py.ref_dtype)
        for cell_index in range(cell_count):
            summary_group = handle.create_group(f"summary_cell_{cell_index}")
            cycles = np.arange(1, cycle_count + 1, dtype=float)
            q_discharge = 1.08 - cell_index * 0.02 - cycles * 0.008
            summary_group.create_dataset("cycle", data=cycles.reshape(1, -1))
            summary_group.create_dataset("QDischarge", data=q_discharge.reshape(1, -1))
            summary_group.create_dataset("IR", data=(0.015 + cycles * 0.0002).reshape(1, -1))
            summary_group.create_dataset("Tavg", data=(30.0 + cycles * 0.08).reshape(1, -1))
            summary_group.create_dataset("Tmin", data=(28.5 + cycles * 0.05).reshape(1, -1))
            summary_group.create_dataset("Tmax", data=(31.5 + cycles * 0.10).reshape(1, -1))

            cycles_group = handle.create_group(f"cycles_cell_{cell_index}")
            voltage_arrays = [np.linspace(2.8, 4.1, 6) - cell_index * 0.01 - cycle * 0.002 for cycle in range(cycle_count)]
            current_arrays = [np.array([1.5, 1.5, -1.0, -1.0, -1.0, -1.0], dtype=float) for _ in range(cycle_count)]
            temperature_arrays = [np.linspace(29.0, 33.0, 6) + cycle * 0.04 for cycle in range(cycle_count)]
            _write_ref_matrix(handle, cycles_group, "V", voltage_arrays, f"cell_{cell_index}")
            _write_ref_matrix(handle, cycles_group, "I", current_arrays, f"cell_{cell_index}")
            _write_ref_matrix(handle, cycles_group, "T", temperature_arrays, f"cell_{cell_index}")

            policy = f"4C-POLICY-{cell_index + 1}"
            policy_dataset = handle.create_dataset(
                f"policy_{cell_index}",
                data=np.asarray([[ord(char)] for char in policy], dtype=np.uint16),
            )
            summary_refs[cell_index, 0] = summary_group.ref
            cycles_refs[cell_index, 0] = cycles_group.ref
            policy_refs[cell_index, 0] = policy_dataset.ref
    return mat_path


def _write_oxford_mat(raw_dir: Path, cell_count: int = 3, cycle_numbers: list[int] | None = None) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    mat_path = raw_dir / "Oxford_Battery_Degradation_Dataset_1.mat"
    cycle_numbers = cycle_numbers or [0, 100, 200, 300, 400, 500]
    payload: dict[str, object] = {}
    for cell_index in range(cell_count):
        cell_payload: dict[str, object] = {}
        for offset, cycle_number in enumerate(cycle_numbers):
            discharge_capacity = 740 - cell_index * 10 - offset * 12
            time_axis = np.array([0.0, 1.0 / 3600.0, 2.0 / 3600.0, 3.0 / 3600.0])
            cell_payload[f"cyc{cycle_number:04d}"] = {
                "C1dc": {
                    "t": time_axis,
                    "v": np.array([4.18, 4.02, 3.84, 2.95]) - offset * 0.002,
                    "q": np.array([0.0, -discharge_capacity * 0.35, -discharge_capacity * 0.7, -discharge_capacity]),
                    "T": np.array([30.0, 30.4, 30.8, 31.1]) + cell_index * 0.2,
                },
                "C1ch": {
                    "t": time_axis + 1.0,
                    "v": np.array([3.0, 3.5, 3.9, 4.2]) - offset * 0.002,
                    "q": np.array([0.0, discharge_capacity * 0.3, discharge_capacity * 0.6, discharge_capacity * 0.95]),
                    "T": np.array([29.8, 30.1, 30.5, 30.9]) + cell_index * 0.2,
                },
            }
        payload[f"Cell{cell_index + 1}"] = cell_payload
    savemat(mat_path, payload)
    return mat_path


def _write_pulsebat_repo(raw_dir: Path) -> Path:
    repo_root = raw_dir / "Pulse-Voltage-Response-Generation-main"
    (repo_root / "Resources").mkdir(parents=True, exist_ok=True)
    (repo_root / "Unexpected Situations Handling").mkdir(parents=True, exist_ok=True)
    (repo_root / "README.md").write_text("# PulseBat\nSynthetic asset manifest", encoding="utf-8")
    (repo_root / "Resources" / "SOH Distribution.png").write_bytes(b"png")
    (repo_root / "Resources" / "NMC 2.1Ah Feature U1-U21 Description.png").write_bytes(b"png")
    (repo_root / "Unexpected Situations Handling" / "sample_issue.xlsx").write_bytes(b"xlsx")
    for script_name in (
        "step_1_extract workstep sheet.py",
        "step_2_feature extraction_adjustable.py",
        "step_3_feature collection_adjustable.py",
    ):
        (repo_root / script_name).write_text("print('pulsebat')\n", encoding="utf-8")
    return repo_root


def test_source_registry_exposes_training_auxiliary_and_enhancement_roles():
    assert {"nasa", "calce", "kaggle", "hust", "matr"} <= set(list_training_ready_sources())
    assert "oxford" in list_auxiliary_sources()
    assert list_enhancement_only_sources() == ["pulsebat"]


def test_hust_adapter_processes_raw_zip(tmp_path: Path):
    raw_dir = tmp_path / "hust"
    _write_hust_zip(raw_dir)
    frame = HUSTAdapter().process_directory(raw_dir)

    assert set(frame["source"].unique()) == {"hust"}
    assert frame["source_battery_id"].nunique() == 3
    assert frame["capacity"].min() > 0.8
    assert {"eol_cycle", "RUL", "protocol_id"} <= set(frame.columns)


def test_matr_adapter_processes_raw_mat(tmp_path: Path):
    raw_dir = tmp_path / "matr"
    _write_matr_batch(raw_dir)
    frame = MATRAdapter().process_directory(raw_dir)

    assert set(frame["source"].unique()) == {"matr"}
    assert frame["source_battery_id"].nunique() == 3
    assert frame["protocol_id"].str.contains("4C-POLICY").all()
    assert frame["internal_resistance"].max() > 0.0
    assert frame["capacity"].min() > 0.0


def test_oxford_adapter_processes_raw_mat(tmp_path: Path):
    raw_dir = tmp_path / "oxford"
    _write_oxford_mat(raw_dir)
    frame = OxfordAdapter().process_directory(raw_dir)

    assert set(frame["source"].unique()) == {"oxford"}
    assert frame["source_battery_id"].nunique() == 3
    assert frame["current_mean"].abs().max() > 0.0
    assert frame["capacity"].min() > 0.5


def test_pulsebat_asset_manifest_stays_out_of_training_pool(tmp_path: Path):
    raw_dir = tmp_path / "pulsebat"
    _write_pulsebat_repo(raw_dir)
    output_dir = tmp_path / "processed" / "pulsebat"
    payload = PulseBatAdapter().build_enhancement_assets(raw_dir, output_dir)

    assert Path(payload["asset_manifest_path"]).exists()
    assert Path(payload["dataset_summary_path"]).exists()
    assert Path(payload["feature_index_path"]).exists()
    assert payload["asset_count"] >= 6
    assert payload["dataset_summary"]["training_ready"] is False


def test_refresh_processed_baselines_compresses_hust_and_matr_cycle_summaries(tmp_path: Path):
    settings = _make_settings(tmp_path)
    _write_hust_zip(settings.raw_hust_dir, battery_count=3, cycle_count=12)
    _write_matr_batch(settings.raw_matr_dir, cell_count=3, cycle_count=12)

    summary = refresh_processed_baselines(
        sources=["hust", "matr"],
        settings=settings,
        seq_len=6,
        batch_size=2,
    )

    hust_csv = settings.processed_dir / "hust" / "hust_cycle_summary.csv.gz"
    matr_csv = settings.processed_dir / "matr" / "matr_cycle_summary.csv.gz"
    assert summary["hust"]["csv_path"].endswith(".csv.gz")
    assert summary["matr"]["csv_path"].endswith(".csv.gz")
    assert hust_csv.exists()
    assert matr_csv.exists()
    assert not (settings.processed_dir / "hust" / "hust_cycle_summary.csv").exists()
    assert not (settings.processed_dir / "matr" / "matr_cycle_summary.csv").exists()
    assert not pd.read_csv(hust_csv).empty
    assert not pd.read_csv(matr_csv).empty


def test_battery_service_import_builtin_source_supports_external_modes(tmp_path: Path):
    settings = _make_settings(tmp_path)
    _write_hust_zip(settings.raw_hust_dir)
    _write_matr_batch(settings.raw_matr_dir)
    _write_oxford_mat(settings.raw_oxford_dir)
    _write_pulsebat_repo(settings.raw_pulsebat_dir)

    database = DatabaseManager(settings.database_path)
    database.initialize()
    repository = BHMSRepository(database)
    service = BatteryService(repository=repository, settings=settings)

    hust_summary = service.import_builtin_source("hust", include_in_training=True)
    matr_summary = service.import_builtin_source("matr", include_in_training=True)
    oxford_summary = service.import_builtin_source("oxford", include_in_training=True)
    pulsebat_summary = service.import_builtin_source("pulsebat", include_in_training=True)

    assert hust_summary["battery_ids"]
    assert matr_summary["battery_ids"]
    assert oxford_summary["battery_ids"]
    assert hust_summary["validation_summary"]["ingestion_mode"] == "builtin_source/raw_converter"
    assert matr_summary["validation_summary"]["ingestion_mode"] == "builtin_source/raw_converter"
    assert oxford_summary["validation_summary"]["ingestion_mode"] == "builtin_source/raw_converter"
    assert oxford_summary["include_in_training"] is False

    hust_prepared = service.prepare_training_dataset("hust", seq_len=6, batch_size=2)
    matr_prepared = service.prepare_training_dataset("matr", seq_len=6, batch_size=2)
    assert Path(hust_prepared["csv_path"]).exists()
    assert Path(matr_prepared["csv_path"]).exists()

    with pytest.raises(BHMSException):
        service.prepare_training_dataset("oxford", seq_len=6, batch_size=2)

    assert pulsebat_summary["battery_ids"] == []
    assert pulsebat_summary["result_type"] == "enhancement_asset_imported"
    assert pulsebat_summary["validation_summary"]["asset_count"] >= 6
    assert Path(pulsebat_summary["asset_manifest_path"]).exists()
