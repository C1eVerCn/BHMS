"""Raw-source adapters for external battery datasets."""

from __future__ import annotations

import pickle
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat, whosmat

from ml.data.schema import finalize_cycle_frame, write_json
from ml.data.source_registry import get_dataset_card
from .base import BaseBatteryAdapter
from .csv_adapter import GenericCSVAdapter

try:  # pragma: no cover - exercised in integration tests when h5py is installed.
    import h5py
except ModuleNotFoundError:  # pragma: no cover
    h5py = None  # type: ignore[assignment]


def _selection_set(battery_ids: Optional[Iterable[str]]) -> set[str] | None:
    if battery_ids is None:
        return None
    return {str(item).strip().upper() for item in battery_ids if str(item).strip()}


def _matches_selection(raw_id: str, selected: set[str] | None) -> bool:
    if selected is None:
        return True
    upper = raw_id.upper()
    return any(candidate == upper or candidate.endswith(f"::{upper}") for candidate in selected)


def _series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_numeric(values, errors="coerce").dropna()
    return pd.to_numeric(pd.Series(values), errors="coerce").dropna()


def _mean(values: pd.Series) -> float:
    return float(values.mean()) if not values.empty else 0.0


def _std(values: pd.Series) -> float:
    return float(values.std(ddof=0)) if len(values) > 1 else 0.0


def _min(values: pd.Series) -> float:
    return float(values.min()) if not values.empty else 0.0


def _max(values: pd.Series) -> float:
    return float(values.max()) if not values.empty else 0.0


class HUSTAdapter(BaseBatteryAdapter):
    source_name = "hust"

    def __init__(self, eol_capacity_ratio: float = 0.8):
        super().__init__(eol_capacity_ratio=eol_capacity_ratio)
        self.card = get_dataset_card("hust")
        self.csv_fallback = GenericCSVAdapter(
            source_name=self.card.source,
            dataset_name=self.card.dataset_name,
            eol_capacity_ratio=eol_capacity_ratio,
            metadata_defaults=self.card.metadata_defaults,
        )

    def process_directory(
        self,
        input_dir: str | Path,
        output_path: str | Path | None = None,
        battery_ids: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        directory = Path(input_dir)
        selected = _selection_set(battery_ids)
        raw_zip = directory / "hust_data.zip"
        if raw_zip.exists():
            frame = self._process_zip(raw_zip, selected)
        else:
            pkl_files = sorted(path for path in directory.rglob("*.pkl") if path.is_file())
            if pkl_files:
                frame = self._process_pkl_files(pkl_files, selected)
            else:
                frame = self.csv_fallback.process_directory(directory, output_path=None, battery_ids=battery_ids)
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output_path, index=False)
        return frame

    def process_file(self, file_path: str | Path, battery_id_hint: str | None = None) -> pd.DataFrame:
        path = Path(file_path)
        selected = _selection_set([battery_id_hint] if battery_id_hint else None)
        if path.suffix.lower() == ".csv":
            return self.csv_fallback.process_file(path, battery_id_hint=battery_id_hint)
        if path.suffix.lower() == ".zip":
            return self._process_zip(path, selected)
        if path.suffix.lower() == ".pkl":
            return self._process_pkl_files([path], selected)
        raise ValueError(f"HUST 原始导入暂不支持文件类型: {path.suffix}")

    def _process_zip(self, zip_path: Path, selected: set[str] | None) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        with zipfile.ZipFile(zip_path) as archive:
            names = sorted(name for name in archive.namelist() if name.startswith("our_data/") and name.endswith(".pkl"))
            for name in names:
                raw_id = Path(name).stem
                if not _matches_selection(raw_id, selected):
                    continue
                with archive.open(name) as handle:
                    payload = pickle.load(handle)
                records.extend(self._records_from_payload(payload, fallback_id=raw_id))
        if not records:
            raise ValueError("HUST 原始资产中未找到可转换的电池循环数据")
        return finalize_cycle_frame(
            pd.DataFrame(records),
            source=self.card.source,
            dataset_name=self.card.dataset_name,
            eol_capacity_ratio=self.eol_capacity_ratio,
            metadata_defaults=self.card.metadata_defaults,
        )

    def _process_pkl_files(self, pkl_files: list[Path], selected: set[str] | None) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        for file_path in pkl_files:
            raw_id = file_path.stem
            if not _matches_selection(raw_id, selected):
                continue
            with file_path.open("rb") as handle:
                payload = pickle.load(handle)
            records.extend(self._records_from_payload(payload, fallback_id=raw_id))
        if not records:
            raise ValueError("HUST 原始资产中未找到可转换的 PKL 数据")
        return finalize_cycle_frame(
            pd.DataFrame(records),
            source=self.card.source,
            dataset_name=self.card.dataset_name,
            eol_capacity_ratio=self.eol_capacity_ratio,
            metadata_defaults=self.card.metadata_defaults,
        )

    def _records_from_payload(self, payload: Any, *, fallback_id: str) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            raise ValueError("HUST PKL 顶层结构不是字典")
        if "data" in payload:
            battery_id = fallback_id
            battery_payload = payload
        elif len(payload) == 1:
            battery_id, battery_payload = next(iter(payload.items()))
        else:
            battery_id = fallback_id
            battery_payload = payload.get(fallback_id) or payload.get(str(fallback_id))
        if not isinstance(battery_payload, dict):
            raise ValueError(f"HUST 电池 {battery_id} 的 PKL 结构异常")
        cycle_map = battery_payload.get("data")
        if not isinstance(cycle_map, dict):
            raise ValueError(f"HUST 电池 {battery_id} 缺少 data[cycle] 结构")

        records: list[dict[str, Any]] = []
        for cycle_key in sorted(cycle_map, key=lambda item: int(item)):
            cycle_frame = cycle_map[cycle_key]
            if not isinstance(cycle_frame, pd.DataFrame):
                cycle_frame = pd.DataFrame(cycle_frame)
            summary = self._summarize_cycle_frame(cycle_frame)
            if summary is None:
                continue
            summary["source_battery_id"] = str(battery_id)
            records.append(summary)
        return records

    def _summarize_cycle_frame(self, cycle_frame: pd.DataFrame) -> dict[str, Any] | None:
        if cycle_frame.empty:
            return None
        voltage = _series(cycle_frame.get("Voltage (V)"))
        current = _series(cycle_frame.get("Current (mA)")) / 1000.0
        capacity_m_ah = _series(cycle_frame.get("Capacity (mAh)"))
        cycle_numbers = _series(cycle_frame.get("Cycle number"))
        if voltage.empty or current.empty or capacity_m_ah.empty or cycle_numbers.empty:
            return None
        discharge_current = current[current < 0]
        capacity_ah = _max(capacity_m_ah) / 1000.0
        if capacity_ah <= 0:
            return None
        return {
            "cycle_number": int(round(float(cycle_numbers.iloc[-1]))),
            "voltage_mean": _mean(voltage),
            "voltage_std": _std(voltage),
            "voltage_min": _min(voltage),
            "voltage_max": _max(voltage),
            "current_mean": _mean(current),
            "current_std": _std(current),
            "current_load_mean": _mean(discharge_current if not discharge_current.empty else current),
            "temperature_mean": 0.0,
            "temperature_std": 0.0,
            "temperature_rise_rate": 0.0,
            "internal_resistance": 0.0,
            "capacity": capacity_ah,
        }


class MATRAdapter(BaseBatteryAdapter):
    source_name = "matr"

    def __init__(self, eol_capacity_ratio: float = 0.8):
        super().__init__(eol_capacity_ratio=eol_capacity_ratio)
        self.card = get_dataset_card("matr")
        self.csv_fallback = GenericCSVAdapter(
            source_name=self.card.source,
            dataset_name=self.card.dataset_name,
            eol_capacity_ratio=eol_capacity_ratio,
            metadata_defaults=self.card.metadata_defaults,
        )

    def process_directory(
        self,
        input_dir: str | Path,
        output_path: str | Path | None = None,
        battery_ids: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        directory = Path(input_dir)
        mat_files = sorted(directory.glob("MATR_batch_*.mat"))
        if mat_files:
            frame = self._process_mat_files(mat_files, _selection_set(battery_ids))
        else:
            frame = self.csv_fallback.process_directory(directory, output_path=None, battery_ids=battery_ids)
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output_path, index=False)
        return frame

    def process_file(self, file_path: str | Path, battery_id_hint: str | None = None) -> pd.DataFrame:
        path = Path(file_path)
        if path.suffix.lower() == ".csv":
            return self.csv_fallback.process_file(path, battery_id_hint=battery_id_hint)
        if path.suffix.lower() != ".mat":
            raise ValueError(f"MATR 原始导入暂不支持文件类型: {path.suffix}")
        return self._process_mat_files([path], _selection_set([battery_id_hint] if battery_id_hint else None))

    def _process_mat_files(self, mat_files: list[Path], selected: set[str] | None) -> pd.DataFrame:
        if h5py is None:
            raise ImportError("MATR 原始 MAT v7.3 解析依赖 h5py，请先在项目环境中安装 h5py")
        records: list[dict[str, Any]] = []
        for file_path in mat_files:
            with h5py.File(file_path, "r") as handle:
                batch = handle["batch"]
                cell_count = int(batch["summary"].shape[0])
                for index in range(cell_count):
                    raw_id = f"{file_path.stem}_cell_{index + 1:03d}"
                    if not _matches_selection(raw_id, selected):
                        continue
                    summary_group = self._group_from_ref(handle, batch["summary"][index, 0])
                    cycles_group = self._group_from_ref(handle, batch["cycles"][index, 0])
                    policy_readable = self._decode_string(handle, batch["policy_readable"][index, 0])
                    records.extend(
                        self._records_from_groups(
                            handle,
                            summary_group=summary_group,
                            cycles_group=cycles_group,
                            source_battery_id=raw_id,
                            protocol_id=policy_readable or str(self.card.metadata_defaults.get("protocol_id", "matr_fast_charge")),
                        )
                    )
        if not records:
            raise ValueError("MATR 原始资产中未找到可转换的 batch/cell 数据")
        return finalize_cycle_frame(
            pd.DataFrame(records),
            source=self.card.source,
            dataset_name=self.card.dataset_name,
            eol_capacity_ratio=self.eol_capacity_ratio,
            metadata_defaults=self.card.metadata_defaults,
        )

    @staticmethod
    def _group_from_ref(handle: Any, ref: Any):
        return handle[ref]

    @staticmethod
    def _dataset_values(dataset: Any) -> np.ndarray:
        return np.asarray(dataset[()]).reshape(-1)

    def _decode_string(self, handle: Any, ref: Any) -> str:
        dataset = handle[ref]
        array = np.asarray(dataset[()])
        if array.dtype.kind in {"i", "u"}:
            chars = [chr(int(value)) for value in array.reshape(-1) if int(value) != 0 and int(value) <= 0x10FFFF]
            return "".join(chars)
        if array.dtype.kind in {"S", "U"}:
            return "".join(item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in array.reshape(-1))
        return ""

    def _scalar_from_ref(self, handle: Any, ref: Any) -> float | None:
        values = self._dataset_values(handle[ref])
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return None
        return float(finite.reshape(-1)[0])

    def _cycle_series(self, handle: Any, dataset: Any, index: int) -> pd.Series:
        ref = dataset[index, 0]
        values = self._dataset_values(handle[ref])
        return _series(values)

    def _records_from_groups(
        self,
        handle: Any,
        *,
        summary_group: Any,
        cycles_group: Any,
        source_battery_id: str,
        protocol_id: str,
    ) -> list[dict[str, Any]]:
        cycle_numbers = _series(self._dataset_values(summary_group["cycle"]))
        capacity = _series(self._dataset_values(summary_group["QDischarge"]))
        internal_resistance = _series(self._dataset_values(summary_group["IR"]))
        temperature_avg = _series(self._dataset_values(summary_group["Tavg"]))
        temperature_min = _series(self._dataset_values(summary_group["Tmin"]))
        temperature_max = _series(self._dataset_values(summary_group["Tmax"]))
        length = min(len(cycle_numbers), len(capacity))
        records: list[dict[str, Any]] = []
        for index in range(length):
            cycle_number = int(round(float(cycle_numbers.iloc[index])))
            cycle_capacity = float(capacity.iloc[index])
            if cycle_number <= 0 or not np.isfinite(cycle_capacity) or cycle_capacity <= 0:
                continue
            voltage = self._cycle_series(handle, cycles_group["V"], index)
            current = self._cycle_series(handle, cycles_group["I"], index)
            temperature = self._cycle_series(handle, cycles_group["T"], index)
            if voltage.empty:
                continue
            ir = float(internal_resistance.iloc[index]) if index < len(internal_resistance) and np.isfinite(internal_resistance.iloc[index]) else 0.0
            tavg = float(temperature_avg.iloc[index]) if index < len(temperature_avg) and np.isfinite(temperature_avg.iloc[index]) else _mean(temperature)
            tmin = float(temperature_min.iloc[index]) if index < len(temperature_min) and np.isfinite(temperature_min.iloc[index]) else _min(temperature)
            tmax = float(temperature_max.iloc[index]) if index < len(temperature_max) and np.isfinite(temperature_max.iloc[index]) else _max(temperature)
            discharge_current = current[current < 0]
            records.append(
                {
                    "source_battery_id": source_battery_id,
                    "cycle_number": cycle_number,
                    "protocol_id": protocol_id,
                    "voltage_mean": _mean(voltage),
                    "voltage_std": _std(voltage),
                    "voltage_min": _min(voltage),
                    "voltage_max": _max(voltage),
                    "current_mean": _mean(current),
                    "current_std": _std(current),
                    "current_load_mean": _mean(discharge_current if not discharge_current.empty else current),
                    "temperature_mean": tavg,
                    "temperature_std": _std(temperature) if not temperature.empty else max(0.0, (tmax - tmin) / 2.0),
                    "temperature_rise_rate": max(0.0, tmax - tmin),
                    "internal_resistance": ir,
                    "capacity": cycle_capacity,
                }
            )
        return records


class OxfordAdapter(BaseBatteryAdapter):
    source_name = "oxford"

    def __init__(self, eol_capacity_ratio: float = 0.8):
        super().__init__(eol_capacity_ratio=eol_capacity_ratio)
        self.card = get_dataset_card("oxford")
        self.csv_fallback = GenericCSVAdapter(
            source_name=self.card.source,
            dataset_name=self.card.dataset_name,
            eol_capacity_ratio=eol_capacity_ratio,
            metadata_defaults=self.card.metadata_defaults,
        )

    def process_directory(
        self,
        input_dir: str | Path,
        output_path: str | Path | None = None,
        battery_ids: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        directory = Path(input_dir)
        mat_path = directory / "Oxford_Battery_Degradation_Dataset_1.mat"
        if mat_path.exists():
            frame = self._process_mat(mat_path, _selection_set(battery_ids))
        else:
            frame = self.csv_fallback.process_directory(directory, output_path=None, battery_ids=battery_ids)
        if output_path is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(output_path, index=False)
        return frame

    def process_file(self, file_path: str | Path, battery_id_hint: str | None = None) -> pd.DataFrame:
        path = Path(file_path)
        if path.suffix.lower() == ".csv":
            return self.csv_fallback.process_file(path, battery_id_hint=battery_id_hint)
        if path.suffix.lower() != ".mat":
            raise ValueError(f"Oxford 原始导入暂不支持文件类型: {path.suffix}")
        return self._process_mat(path, _selection_set([battery_id_hint] if battery_id_hint else None))

    def _process_mat(self, mat_path: Path, selected: set[str] | None) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        cell_names = [name for name, _, kind in whosmat(mat_path) if kind == "struct" and name.lower().startswith("cell")]
        for cell_name in cell_names:
            if not _matches_selection(cell_name, selected):
                continue
            try:
                payload = loadmat(
                    mat_path,
                    variable_names=[cell_name],
                    squeeze_me=True,
                    struct_as_record=False,
                    verify_compressed_data_integrity=False,
                )
            except OSError:
                continue
            cell_struct = payload[cell_name]
            for cycle_field in sorted(getattr(cell_struct, "_fieldnames", [])):
                summary = self._summarize_cycle(cell_name, cycle_field, getattr(cell_struct, cycle_field))
                if summary is not None:
                    records.append(summary)
        if not records:
            raise ValueError("Oxford MAT 资产中未找到可转换的 Cell* 周期数据")
        return finalize_cycle_frame(
            pd.DataFrame(records),
            source=self.card.source,
            dataset_name=self.card.dataset_name,
            eol_capacity_ratio=self.eol_capacity_ratio,
            metadata_defaults=self.card.metadata_defaults,
        )

    def _summarize_cycle(self, cell_name: str, cycle_field: str, cycle_struct: Any) -> dict[str, Any] | None:
        match = re.search(r"(\d+)$", cycle_field)
        if match is None:
            return None
        voltage_segments: list[np.ndarray] = []
        current_segments: list[np.ndarray] = []
        temperature_segments: list[np.ndarray] = []
        discharge_capacities: list[float] = []
        total_duration_hours = 0.0
        for branch_name in getattr(cycle_struct, "_fieldnames", []):
            branch = getattr(cycle_struct, branch_name)
            voltage = np.asarray(getattr(branch, "v", np.array([])), dtype=float).reshape(-1)
            charge = np.asarray(getattr(branch, "q", np.array([])), dtype=float).reshape(-1)
            timestamp = np.asarray(getattr(branch, "t", np.array([])), dtype=float).reshape(-1)
            temperature = np.asarray(getattr(branch, "T", np.array([])), dtype=float).reshape(-1)
            if voltage.size:
                voltage_segments.append(voltage[np.isfinite(voltage)])
            if temperature.size:
                temperature_segments.append(temperature[np.isfinite(temperature)])
            current = self._derive_current_from_charge(charge, timestamp)
            if current.size:
                current_segments.append(current)
            if charge.size:
                duration_hours = max(0.0, float(np.nanmax(timestamp) - np.nanmin(timestamp)) * 24.0)
                total_duration_hours += duration_hours
                if branch_name.lower().endswith("dc"):
                    discharge_capacities.append(float(np.nanmax(np.abs(charge))))
        voltage = np.concatenate([segment for segment in voltage_segments if segment.size], axis=0) if voltage_segments else np.array([])
        current = np.concatenate([segment for segment in current_segments if segment.size], axis=0) if current_segments else np.array([])
        temperature = np.concatenate([segment for segment in temperature_segments if segment.size], axis=0) if temperature_segments else np.array([])
        if voltage.size == 0:
            return None
        capacity_ah = (max(discharge_capacities) / 1000.0) if discharge_capacities else 0.0
        if capacity_ah <= 0:
            return None
        discharge_current = current[current < 0]
        rise_rate = 0.0
        if temperature.size > 0:
            rise_rate = (float(np.nanmax(temperature)) - float(np.nanmin(temperature))) / max(total_duration_hours, 1e-6)
        return {
            "source_battery_id": cell_name,
            "cycle_number": int(match.group(1)),
            "voltage_mean": float(np.nanmean(voltage)),
            "voltage_std": float(np.nanstd(voltage)) if voltage.size > 1 else 0.0,
            "voltage_min": float(np.nanmin(voltage)),
            "voltage_max": float(np.nanmax(voltage)),
            "current_mean": float(np.nanmean(current)) if current.size else 0.0,
            "current_std": float(np.nanstd(current)) if current.size > 1 else 0.0,
            "current_load_mean": float(np.nanmean(discharge_current)) if discharge_current.size else (float(np.nanmean(current)) if current.size else 0.0),
            "temperature_mean": float(np.nanmean(temperature)) if temperature.size else 0.0,
            "temperature_std": float(np.nanstd(temperature)) if temperature.size > 1 else 0.0,
            "temperature_rise_rate": rise_rate,
            "internal_resistance": 0.0,
            "capacity": capacity_ah,
        }

    @staticmethod
    def _derive_current_from_charge(charge: np.ndarray, timestamp: np.ndarray) -> np.ndarray:
        if charge.size < 2 or timestamp.size < 2:
            return np.array([], dtype=float)
        charge = charge.astype(float, copy=False)
        timestamp = timestamp.astype(float, copy=False)
        dq = np.diff(charge)
        dt_hours = np.diff(timestamp) * 24.0
        mask = np.isfinite(dq) & np.isfinite(dt_hours) & (np.abs(dt_hours) > 1e-9)
        if not np.any(mask):
            return np.array([], dtype=float)
        return dq[mask] / dt_hours[mask] / 1000.0


class PulseBatAdapter(BaseBatteryAdapter):
    source_name = "pulsebat"

    def __init__(self, eol_capacity_ratio: float = 0.8):
        super().__init__(eol_capacity_ratio=eol_capacity_ratio)
        self.card = get_dataset_card("pulsebat")

    def process_directory(
        self,
        input_dir: str | Path,
        output_path: str | Path | None = None,
        battery_ids: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        _ = input_dir, output_path, battery_ids
        raise ValueError("PulseBat 仅支持 enhancement 资产建档，不生成 lifecycle cycle summary")

    def process_file(self, file_path: str | Path, battery_id_hint: str | None = None) -> pd.DataFrame:
        _ = file_path, battery_id_hint
        raise ValueError("PulseBat 上传文件不会直接转换为 lifecycle cycle summary")

    def build_enhancement_assets(self, input_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
        raw_root = Path(input_dir)
        processed_root = Path(output_dir)
        processed_root.mkdir(parents=True, exist_ok=True)
        repo_root = raw_root / "Pulse-Voltage-Response-Generation-main"
        readme_path = repo_root / "README.md"
        resources_dir = repo_root / "Resources"
        unexpected_dir = repo_root / "Unexpected Situations Handling"
        script_paths = [
            repo_root / "step_1_extract workstep sheet.py",
            repo_root / "step_2_feature extraction_adjustable.py",
            repo_root / "step_3_feature collection_adjustable.py",
        ]
        asset_entries: list[dict[str, Any]] = []
        for path in [readme_path, *sorted(resources_dir.glob("*")), *sorted(unexpected_dir.glob("*.xlsx")), *script_paths]:
            if not path.exists():
                continue
            relative_path = str(path.relative_to(raw_root))
            if path == readme_path:
                asset_kind = "readme"
            elif path.parent == resources_dir:
                asset_kind = "resource_figure"
            elif path.parent == unexpected_dir:
                asset_kind = "unexpected_workbook"
            else:
                asset_kind = "upstream_script"
            asset_entries.append(
                {
                    "key": relative_path.replace("/", "::"),
                    "kind": asset_kind,
                    "path": relative_path,
                    "size_bytes": path.stat().st_size,
                }
            )

        manifest_path = processed_root / "pulsebat_asset_manifest.json"
        dataset_summary_path = processed_root / "pulsebat_dataset_summary.json"
        feature_index_path = processed_root / "pulsebat_feature_index.json"

        manifest = {
            "source": self.card.source,
            "dataset_name": self.card.dataset_name,
            "ingestion_mode": self.card.ingestion_mode,
            "training_ready": self.card.training_ready,
            "asset_count": len(asset_entries),
            "assets": asset_entries,
        }
        dataset_summary = {
            "source": self.card.source,
            "dataset_name": self.card.dataset_name,
            "csv_path": None,
            "feature_columns": [],
            "split": {},
            "num_samples": {},
            "num_batteries": 0,
            "ingestion_mode": self.card.ingestion_mode,
            "training_ready": self.card.training_ready,
            "source_group": self.card.group,
            "asset_count": len(asset_entries),
            "asset_categories": {
                "readme": sum(1 for item in asset_entries if item["kind"] == "readme"),
                "resource_figure": sum(1 for item in asset_entries if item["kind"] == "resource_figure"),
                "unexpected_workbook": sum(1 for item in asset_entries if item["kind"] == "unexpected_workbook"),
                "upstream_script": sum(1 for item in asset_entries if item["kind"] == "upstream_script"),
            },
            "provenance": {
                "kind": "enhancement_assets",
                "source_root": "data/raw/pulsebat",
                "generated_by": "pulsebat_asset_builder",
            },
        }
        feature_index = {
            "source": self.card.source,
            "dataset_name": self.card.dataset_name,
            "ingestion_mode": self.card.ingestion_mode,
            "training_ready": self.card.training_ready,
            "feature_guides": [
                {
                    "label": path.name,
                    "path": str(path.relative_to(raw_root)),
                }
                for path in sorted(resources_dir.glob("*Feature*"))
                if path.exists()
            ],
            "upstream_scripts": [
                {
                    "name": path.stem,
                    "path": str(path.relative_to(raw_root)),
                }
                for path in script_paths
                if path.exists()
            ],
            "unexpected_situations": [
                {
                    "file_name": path.name,
                    "path": str(path.relative_to(raw_root)),
                }
                for path in sorted(unexpected_dir.glob("*.xlsx"))
                if path.exists()
            ],
        }
        write_json(manifest_path, manifest)
        write_json(dataset_summary_path, dataset_summary)
        write_json(feature_index_path, feature_index)
        return {
            "asset_manifest_path": str(manifest_path),
            "dataset_summary_path": str(dataset_summary_path),
            "feature_index_path": str(feature_index_path),
            "asset_count": len(asset_entries),
            "asset_manifest": manifest,
            "dataset_summary": dataset_summary,
            "feature_index": feature_index,
        }


__all__ = ["HUSTAdapter", "MATRAdapter", "OxfordAdapter", "PulseBatAdapter"]
