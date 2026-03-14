"""NASA PCoE 电池数据预处理工具。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.io import loadmat


DEFAULT_FEATURE_COLUMNS = [
    "voltage_mean",
    "voltage_std",
    "voltage_min",
    "voltage_max",
    "current_mean",
    "current_std",
    "temperature_mean",
    "temperature_std",
    "capacity",
    "cycle_number",
]


@dataclass(slots=True)
class DatasetSplit:
    train_batteries: list[str]
    val_batteries: list[str]
    test_batteries: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "train_batteries": self.train_batteries,
            "val_batteries": self.val_batteries,
            "test_batteries": self.test_batteries,
        }


class NASABatteryPreprocessor:
    """将 NASA MAT 文件转换为周期级结构化数据。"""

    def __init__(self, eol_capacity_ratio: float = 0.8):
        self.eol_capacity_ratio = eol_capacity_ratio

    @staticmethod
    def _format_timestamp(values: Sequence[float] | np.ndarray | None) -> Optional[str]:
        if values is None:
            return None
        arr = np.asarray(values).astype(float).flatten()
        if arr.size < 6:
            return None
        year, month, day, hour, minute, second = arr[:6]
        second_int = int(second)
        microseconds = int(round((second - second_int) * 1_000_000))
        return (
            f"{int(year):04d}-{int(month):02d}-{int(day):02d}T"
            f"{int(hour):02d}:{int(minute):02d}:{second_int:02d}.{microseconds:06d}"
        )

    @staticmethod
    def _safe_float(value: float | np.ndarray) -> float:
        arr = np.asarray(value, dtype=float).flatten()
        if arr.size == 0:
            return 0.0
        return float(arr[0])

    @staticmethod
    def _summary(values: np.ndarray) -> tuple[float, float, float, float]:
        arr = np.asarray(values, dtype=float)
        return float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())

    def parse_battery_file(self, file_path: str | Path) -> pd.DataFrame:
        path = Path(file_path)
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
        battery_id = path.stem
        raw_battery = mat[battery_id]

        rows: list[dict[str, object]] = []
        discharge_index = 0
        for cycle in raw_battery.cycle:
            if getattr(cycle, "type", "") != "discharge":
                continue

            discharge_index += 1
            data = cycle.data
            voltage_mean, voltage_std, voltage_min, voltage_max = self._summary(data.Voltage_measured)
            current_mean, current_std, _, _ = self._summary(data.Current_measured)
            temperature_mean, temperature_std, _, _ = self._summary(data.Temperature_measured)
            current_load_mean = float(np.asarray(data.Current_load, dtype=float).mean())
            temperature_rise_rate = 0.0
            temp_series = np.asarray(data.Temperature_measured, dtype=float)
            time_series = np.asarray(data.Time, dtype=float)
            if temp_series.size > 1 and time_series.size > 1:
                delta_t = np.diff(time_series)
                delta_temp = np.diff(temp_series)
                valid = delta_t > 0
                if np.any(valid):
                    rates = delta_temp[valid] / np.maximum(delta_t[valid], 1e-6)
                    temperature_rise_rate = float(np.max(rates))

            rows.append(
                {
                    "battery_id": battery_id,
                    "cycle_number": discharge_index,
                    "timestamp": self._format_timestamp(getattr(cycle, "time", None)),
                    "ambient_temperature": float(getattr(cycle, "ambient_temperature", 0.0)),
                    "voltage_mean": voltage_mean,
                    "voltage_std": voltage_std,
                    "voltage_min": voltage_min,
                    "voltage_max": voltage_max,
                    "current_mean": current_mean,
                    "current_std": current_std,
                    "temperature_mean": temperature_mean,
                    "temperature_std": temperature_std,
                    "temperature_rise_rate": temperature_rise_rate,
                    "current_load_mean": current_load_mean,
                    "capacity": self._safe_float(data.Capacity),
                    "source_type": "nasa_discharge",
                }
            )

        if not rows:
            raise ValueError(f"未在 {path} 中解析到放电周期数据")

        frame = pd.DataFrame(rows).sort_values(["battery_id", "cycle_number"]).reset_index(drop=True)
        initial_capacity = float(frame["capacity"].iloc[0])
        frame["initial_capacity"] = initial_capacity
        frame["capacity_ratio"] = frame["capacity"] / max(initial_capacity, 1e-6)
        eol_candidates = frame.loc[frame["capacity_ratio"] <= self.eol_capacity_ratio, "cycle_number"]
        eol_cycle = int(eol_candidates.iloc[0]) if not eol_candidates.empty else int(frame["cycle_number"].max())
        frame["eol_cycle"] = eol_cycle
        frame["RUL"] = np.maximum(0, eol_cycle - frame["cycle_number"])
        frame["health_score"] = (frame["capacity_ratio"] * 100).clip(lower=0, upper=100)
        frame["status"] = frame["health_score"].map(self._health_status)
        return frame

    @staticmethod
    def _health_status(score: float) -> str:
        if score >= 85:
            return "good"
        if score >= 70:
            return "warning"
        return "critical"

    def process_directory(
        self,
        input_dir: str | Path,
        output_path: str | Path | None = None,
        battery_ids: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        directory = Path(input_dir)
        selected = {battery_id.upper() for battery_id in battery_ids} if battery_ids else None

        frames: list[pd.DataFrame] = []
        for mat_file in sorted(directory.glob("B*.mat")):
            if selected and mat_file.stem.upper() not in selected:
                continue
            frames.append(self.parse_battery_file(mat_file))

        if not frames:
            raise ValueError(f"在 {directory} 中未找到符合条件的 NASA MAT 文件")

        combined = pd.concat(frames, ignore_index=True)
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(destination, index=False)
        return combined

    @staticmethod
    def split_batteries(
        battery_ids: Sequence[str],
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
    ) -> DatasetSplit:
        unique_ids = sorted(dict.fromkeys(battery_ids))
        total = len(unique_ids)
        if total < 3:
            raise ValueError("至少需要 3 个电池样本才能划分 train/val/test")

        train_end = max(1, int(total * train_ratio))
        val_end = max(train_end + 1, int(total * (train_ratio + val_ratio)))
        val_end = min(val_end, total - 1)

        split = DatasetSplit(
            train_batteries=unique_ids[:train_end],
            val_batteries=unique_ids[train_end:val_end],
            test_batteries=unique_ids[val_end:],
        )
        if not split.val_batteries or not split.test_batteries:
            raise ValueError("数据划分失败，请检查电池数量与比例设置")
        return split


__all__ = [
    "DEFAULT_FEATURE_COLUMNS",
    "DatasetSplit",
    "NASABatteryPreprocessor",
]
