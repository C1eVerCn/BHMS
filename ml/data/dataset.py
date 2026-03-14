"""RUL 训练数据集与数据加载器。"""

from __future__ import annotations

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
        self.data = data[data["battery_id"].isin(battery_ids)].copy()
        self.data.sort_values(["battery_id", "cycle_number"], inplace=True)
        self.samples = self._build_samples()

    def _build_samples(self) -> list[tuple[np.ndarray, np.ndarray]]:
        samples: list[tuple[np.ndarray, np.ndarray]] = []
        for _, group in self.data.groupby("battery_id"):
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
    """封装数据准备、划分与 DataLoader 构建。"""

    def __init__(
        self,
        csv_path: str | Path,
        seq_len: int = 30,
        batch_size: int = 32,
        feature_cols: Optional[Sequence[str]] = None,
        num_workers: int = 0,
    ):
        self.csv_path = Path(csv_path)
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.feature_cols = list(feature_cols or DEFAULT_FEATURE_COLUMNS)
        self.num_workers = num_workers
        self.data = pd.read_csv(self.csv_path)
        self.data.sort_values(["battery_id", "cycle_number"], inplace=True)
        self.split = NASABatteryPreprocessor.split_batteries(self.data["battery_id"].unique())
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

    def _compute_normalization(self, train_batteries: Iterable[str]) -> NormalizationStats:
        train_frame = self.data[self.data["battery_id"].isin(train_batteries)]
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
        return {
            "csv_path": str(self.csv_path),
            "seq_len": self.seq_len,
            "batch_size": self.batch_size,
            "split": self.split.to_dict(),
            "normalization": self.normalization.to_dict(),
            "num_samples": {
                "train": len(self.train_dataset),
                "val": len(self.val_dataset),
                "test": len(self.test_dataset),
            },
        }


def create_synthetic_data(
    output_path: str | Path,
    num_batteries: int = 8,
    num_cycles: int = 120,
    noise_level: float = 0.05,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    frames: list[pd.DataFrame] = []
    for battery_index in range(num_batteries):
        battery_id = f"SYN_{battery_index:03d}"
        cycles = np.arange(1, num_cycles + 1)
        initial_capacity = rng.uniform(1.9, 2.2)
        decay_rate = rng.uniform(0.0008, 0.0025)
        capacity = initial_capacity * np.exp(-decay_rate * cycles) + rng.normal(0, noise_level, size=num_cycles)
        capacity = np.clip(capacity, 0.5, None)
        eol_candidates = np.where(capacity <= initial_capacity * 0.8)[0]
        eol_cycle = int(eol_candidates[0] + 1) if len(eol_candidates) > 0 else int(cycles[-1])
        frame = pd.DataFrame(
            {
                "battery_id": battery_id,
                "cycle_number": cycles,
                "timestamp": [f"2026-01-01T00:{idx:02d}:00" for idx in range(num_cycles)],
                "ambient_temperature": rng.uniform(20, 30, size=num_cycles),
                "voltage_mean": 3.7 + rng.normal(0, 0.05, size=num_cycles),
                "voltage_std": 0.06 + rng.normal(0, 0.01, size=num_cycles),
                "voltage_min": 3.45 + rng.normal(0, 0.05, size=num_cycles),
                "voltage_max": 4.12 + rng.normal(0, 0.05, size=num_cycles),
                "current_mean": -1.8 + rng.normal(0, 0.08, size=num_cycles),
                "current_std": 0.12 + rng.normal(0, 0.02, size=num_cycles),
                "temperature_mean": 25 + rng.normal(0, 2.0, size=num_cycles),
                "temperature_std": 1.5 + rng.normal(0, 0.2, size=num_cycles),
                "temperature_rise_rate": np.abs(rng.normal(0.8, 0.3, size=num_cycles)),
                "current_load_mean": -1.7 + rng.normal(0, 0.05, size=num_cycles),
                "capacity": capacity,
                "source_type": "synthetic",
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
