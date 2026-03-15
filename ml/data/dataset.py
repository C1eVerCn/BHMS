"""RUL 训练数据集、来源级数据模块与元数据导出。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ml.data.nasa_preprocessor import DEFAULT_FEATURE_COLUMNS, DatasetSplit, NASABatteryPreprocessor


@dataclass(slots=True)
class NormalizationStats:
    means: dict[str, float]
    stds: dict[str, float]

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"means": self.means, "stds": self.stds}


class BatterySequenceDataset(Dataset):
    """按电池维度切分后的序列样本。"""

    def __init__(
        self,
        data: pd.DataFrame,
        battery_ids: Sequence[str],
        seq_len: int = 30,
        feature_cols: Optional[Sequence[str]] = None,
        target_col: str = "RUL",
        normalization: Optional[NormalizationStats] = None,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.feature_cols = list(feature_cols or DEFAULT_FEATURE_COLUMNS)
        self.target_col = target_col
        self.normalization = normalization
        self.data = data[data["canonical_battery_id"].isin(battery_ids)].copy()
        self.data.sort_values(["canonical_battery_id", "cycle_number"], inplace=True)
        self.samples = self._build_samples()

    def _build_samples(self) -> list[tuple[np.ndarray, np.ndarray]]:
        samples: list[tuple[np.ndarray, np.ndarray]] = []
        for canonical_battery_id, group in self.data.groupby("canonical_battery_id"):
            _ = canonical_battery_id
            group = group.reset_index(drop=True)
            if len(group) < self.seq_len:
                continue
            values = group[self.feature_cols].astype(float).to_numpy(copy=True)
            targets = group[self.target_col].astype(float).to_numpy(copy=True)
            if self.normalization is not None:
                for index, column in enumerate(self.feature_cols):
                    mean = self.normalization.means[column]
                    std = self.normalization.stds[column]
                    values[:, index] = (values[:, index] - mean) / max(std, 1e-6)
            for start in range(0, len(group) - self.seq_len + 1):
                end = start + self.seq_len
                samples.append((values[start:end], np.array([targets[end - 1]], dtype=np.float32)))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        features, target = self.samples[index]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32)


class RULDataModule:
    """封装来源级数据准备、划分与 DataLoader 构建。"""

    def __init__(
        self,
        csv_path: str | Path,
        source: str,
        seq_len: int = 30,
        batch_size: int = 32,
        feature_cols: Optional[Sequence[str]] = None,
        num_workers: int = 0,
        output_dir: str | Path | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.source = source.lower()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.feature_cols = list(feature_cols or DEFAULT_FEATURE_COLUMNS)
        self.num_workers = num_workers
        self.output_dir = Path(output_dir) if output_dir is not None else self.csv_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = pd.read_csv(self.csv_path)
        self.data = self.data[self.data["source"].str.lower() == self.source].copy()
        if self.data.empty:
            raise ValueError(f"数据文件 {self.csv_path} 中不存在来源 {self.source} 的样本")
        self.data.sort_values(["canonical_battery_id", "cycle_number"], inplace=True)
        self.split = self._resolve_split()
        self.normalization = self._compute_normalization(self.split.train_batteries)
        self.train_dataset = BatterySequenceDataset(
            self.data,
            battery_ids=self.split.train_batteries,
            seq_len=self.seq_len,
            feature_cols=self.feature_cols,
            normalization=self.normalization,
        )
        self.val_dataset = BatterySequenceDataset(
            self.data,
            battery_ids=self.split.val_batteries,
            seq_len=self.seq_len,
            feature_cols=self.feature_cols,
            normalization=self.normalization,
        )
        self.test_dataset = BatterySequenceDataset(
            self.data,
            battery_ids=self.split.test_batteries,
            seq_len=self.seq_len,
            feature_cols=self.feature_cols,
            normalization=self.normalization,
        )

    def _resolve_split(self) -> DatasetSplit:
        battery_ids = list(self.data["canonical_battery_id"].unique())
        split_path = self.output_dir / f"{self.source}_split.json"
        if split_path.exists():
            payload = json.loads(split_path.read_text(encoding="utf-8"))
            split = DatasetSplit.from_dict(payload)
            all_ids = set(split.train_batteries + split.val_batteries + split.test_batteries)
            if all_ids == set(battery_ids) and split.val_batteries and split.test_batteries:
                return split
        return NASABatteryPreprocessor.split_batteries(battery_ids)

    def _compute_normalization(self, train_batteries: Iterable[str]) -> NormalizationStats:
        train_frame = self.data[self.data["canonical_battery_id"].isin(train_batteries)]
        means = {column: float(train_frame[column].mean()) for column in self.feature_cols}
        stds = {column: float(train_frame[column].std(ddof=0)) or 1.0 for column in self.feature_cols}
        return NormalizationStats(means=means, stds=stds)

    def train_loader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def val_loader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def test_loader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    def summary(self) -> dict[str, object]:
        source_frame = self.data.copy()
        return {
            "csv_path": str(self.csv_path),
            "source": self.source,
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "feature_columns": self.feature_cols,
            "split": self.split.to_dict(),
            "normalization": self.normalization.to_dict(),
            "num_batteries": int(source_frame["canonical_battery_id"].nunique()),
            "num_samples": {
                "train": len(self.train_dataset),
                "val": len(self.val_dataset),
                "test": len(self.test_dataset),
            },
        }

    def export_metadata(self, output_dir: str | Path | None = None, file_prefix: Optional[str] = None) -> dict[str, str]:
        summary = self.summary()
        target_dir = Path(output_dir) if output_dir is not None else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        prefix = file_prefix or self.source
        split_path = target_dir / f"{prefix}_split.json"
        normalization_path = target_dir / f"{prefix}_normalization.json"
        feature_path = target_dir / f"{prefix}_feature_config.json"
        summary_path = target_dir / f"{prefix}_dataset_summary.json"
        split_path.write_text(json.dumps(summary["split"], ensure_ascii=False, indent=2), encoding="utf-8")
        normalization_path.write_text(json.dumps(summary["normalization"], ensure_ascii=False, indent=2), encoding="utf-8")
        feature_path.write_text(
            json.dumps({"source": self.source, "feature_columns": self.feature_cols, "target_column": "RUL"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "split": str(split_path),
            "normalization": str(normalization_path),
            "feature_config": str(feature_path),
            "summary": str(summary_path),
        }


def create_synthetic_data(
    output_path: str | Path,
    num_batteries: int = 8,
    num_cycles: int = 120,
    noise_level: float = 0.05,
    source: str = "synthetic",
    dataset_name: str = "synthetic_demo",
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frames: list[pd.DataFrame] = []
    for battery_index in range(num_batteries):
        source_battery_id = f"{source.upper()}_{battery_index:03d}"
        cycles = np.arange(1, num_cycles + 1)
        initial_capacity = rng.uniform(1.9, 2.2)
        decay_rate = rng.uniform(0.0008, 0.0025)
        capacity = initial_capacity * np.exp(-decay_rate * cycles) + rng.normal(0, noise_level, size=num_cycles)
        capacity = np.clip(capacity, 0.5, None)
        eol_candidates = np.where(capacity <= initial_capacity * 0.8)[0]
        eol_cycle = int(eol_candidates[0] + 1) if len(eol_candidates) > 0 else int(cycles[-1])
        canonical_battery_id = f"{source.lower()}::{dataset_name.lower()}::{source_battery_id}"
        frame = pd.DataFrame(
            {
                "battery_id": canonical_battery_id,
                "canonical_battery_id": canonical_battery_id,
                "source": source.lower(),
                "dataset_name": dataset_name,
                "source_battery_id": source_battery_id,
                "cycle_number": cycles,
                "timestamp": [f"2026-01-01T00:{idx % 60:02d}:00" for idx in range(num_cycles)],
                "ambient_temperature": rng.uniform(20, 30, size=num_cycles),
                "voltage_mean": 3.7 + rng.normal(0, 0.05, size=num_cycles),
                "voltage_std": 0.06 + rng.normal(0, 0.01, size=num_cycles),
                "voltage_min": 3.45 + rng.normal(0, 0.05, size=num_cycles),
                "voltage_max": 4.12 + rng.normal(0, 0.05, size=num_cycles),
                "current_mean": -1.8 + rng.normal(0, 0.08, size=num_cycles),
                "current_std": 0.12 + rng.normal(0, 0.02, size=num_cycles),
                "current_load_mean": -1.7 + rng.normal(0, 0.05, size=num_cycles),
                "temperature_mean": 25 + rng.normal(0, 2.0, size=num_cycles),
                "temperature_std": 1.5 + rng.normal(0, 0.2, size=num_cycles),
                "temperature_rise_rate": np.abs(rng.normal(0.8, 0.3, size=num_cycles)),
                "internal_resistance": np.abs(rng.normal(0.02, 0.004, size=num_cycles)),
                "capacity": capacity,
                "source_type": f"{source.lower()}_demo",
                "initial_capacity": initial_capacity,
                "capacity_ratio": capacity / initial_capacity,
                "eol_cycle": eol_cycle,
                "RUL": np.maximum(0, eol_cycle - cycles),
                "health_score": np.clip((capacity / initial_capacity) * 100, 0, 100),
            }
        )
        frame["status"] = np.where(frame["health_score"] >= 85, "good", np.where(frame["health_score"] >= 70, "warning", "critical"))
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(destination, index=False)
    return result


__all__ = [
    "BatterySequenceDataset",
    "NormalizationStats",
    "RULDataModule",
    "create_synthetic_data",
]
