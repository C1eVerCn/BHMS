"""Lifecycle data module and labels for multi-source battery forecasting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ml.data.dataset import NormalizationStats, _serialize_path
from ml.data.nasa_preprocessor import DatasetSplit, NASABatteryPreprocessor
from ml.data.processed_paths import resolve_cycle_summary_path
from ml.data.schema import TRAINING_FEATURE_COLUMNS
from ml.data.source_registry import SOURCE_REGISTRY

LIFECYCLE_FEATURE_COLUMNS = [*TRAINING_FEATURE_COLUMNS, "capacity_ratio"]


@dataclass(slots=True)
class LifecycleTargetConfig:
    observation_ratios: tuple[float, ...] = (0.2, 0.3, 0.4)
    default_observation_ratio: float = 0.3
    encoder_len: int = 64
    future_len: int = 64
    target_column: str = "capacity_ratio"

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_ratios": list(self.observation_ratios),
            "default_observation_ratio": self.default_observation_ratio,
            "encoder_len": self.encoder_len,
            "future_len": self.future_len,
            "target_column": self.target_column,
        }


@dataclass(slots=True)
class DomainVocab:
    source_to_id: dict[str, int]
    chemistry_to_id: dict[str, int]
    protocol_to_id: dict[str, int]

    @classmethod
    def build(cls, frame: pd.DataFrame) -> "DomainVocab":
        def _encode(column: str) -> dict[str, int]:
            values = sorted({str(item) for item in frame.get(column, pd.Series(dtype=str)).fillna("unknown").tolist()})
            if "unknown" not in values:
                values.insert(0, "unknown")
            return {value: index for index, value in enumerate(values)}

        return cls(
            source_to_id=_encode("source"),
            chemistry_to_id=_encode("chemistry"),
            protocol_to_id=_encode("protocol_id"),
        )

    def to_dict(self) -> dict[str, dict[str, int]]:
        return {
            "source_to_id": self.source_to_id,
            "chemistry_to_id": self.chemistry_to_id,
            "protocol_to_id": self.protocol_to_id,
        }


def estimate_knee_cycle(group: pd.DataFrame) -> float | None:
    if len(group) < 12:
        return None
    cycles = group["cycle_number"].astype(float).to_numpy(copy=True)
    capacity_ratio = group["capacity_ratio"].astype(float).to_numpy(copy=True)
    slopes = np.gradient(capacity_ratio, cycles)
    curvature = np.gradient(slopes, cycles)
    search_start = max(3, len(group) // 6)
    search_end = max(search_start + 3, len(group) - 3)
    candidate_idx = int(np.argmin(curvature[search_start:search_end]) + search_start)
    if not np.isfinite(curvature[candidate_idx]):
        return None
    if curvature[candidate_idx] >= -1e-4:
        return None
    return float(cycles[candidate_idx])


def _resample_1d(values: np.ndarray, target_len: int) -> np.ndarray:
    if len(values) == target_len:
        return values.astype(np.float32, copy=True)
    if len(values) == 1:
        return np.repeat(values.astype(np.float32), target_len)
    source_grid = np.linspace(0.0, 1.0, num=len(values), dtype=np.float32)
    target_grid = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(target_grid, source_grid, values.astype(np.float32)).astype(np.float32)


def _resample_2d(values: np.ndarray, target_len: int) -> np.ndarray:
    columns = [_resample_1d(values[:, index], target_len) for index in range(values.shape[1])]
    return np.stack(columns, axis=-1).astype(np.float32)


class LifecycleSequenceDataset(Dataset):
    """Lifecycle samples built from observed-prefix windows and future trajectories."""

    def __init__(
        self,
        data: pd.DataFrame,
        battery_ids: Sequence[str],
        *,
        feature_cols: Sequence[str],
        normalization: NormalizationStats,
        target_config: LifecycleTargetConfig,
        vocab: DomainVocab,
    ):
        super().__init__()
        self.data = data[data["canonical_battery_id"].isin(battery_ids)].copy()
        self.feature_cols = list(feature_cols)
        self.normalization = normalization
        self.target_config = target_config
        self.vocab = vocab
        self.samples = self._build_samples()

    def _encode_domain(self, row: pd.Series) -> dict[str, int]:
        source = str(row.get("source") or "unknown")
        chemistry = str(row.get("chemistry") or "unknown")
        protocol = str(row.get("protocol_id") or "unknown")
        return {
            "source_id": self.vocab.source_to_id.get(source, self.vocab.source_to_id["unknown"]),
            "chemistry_id": self.vocab.chemistry_to_id.get(chemistry, self.vocab.chemistry_to_id["unknown"]),
            "protocol_id": self.vocab.protocol_to_id.get(protocol, self.vocab.protocol_to_id["unknown"]),
        }

    def _normalize_sequence(self, sequence: np.ndarray) -> np.ndarray:
        normalized = sequence.astype(np.float32, copy=True)
        for index, column in enumerate(self.feature_cols):
            mean = self.normalization.means.get(column, 0.0)
            std = self.normalization.stds.get(column, 1.0) or 1.0
            normalized[:, index] = (normalized[:, index] - mean) / max(std, 1e-6)
        return normalized

    def _build_samples(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        ratios = self.target_config.observation_ratios
        for _, group in self.data.groupby("canonical_battery_id"):
            group = group.sort_values("cycle_number").reset_index(drop=True)
            knee_cycle = estimate_knee_cycle(group)
            eol_cycle = float(group["eol_cycle"].iloc[-1])
            for ratio in ratios:
                observation_count = int(max(6, round(len(group) * ratio)))
                if observation_count >= len(group) - 2:
                    continue
                observed = group.iloc[:observation_count].copy()
                future = group.iloc[observation_count:].copy()
                if future.empty:
                    continue
                observed_features = observed[self.feature_cols].astype(float).to_numpy(copy=True)
                observed_features = _resample_2d(observed_features, self.target_config.encoder_len)
                observed_features = self._normalize_sequence(observed_features)
                future_capacity = future[self.target_config.target_column].astype(float).to_numpy(copy=True)
                trajectory_target = _resample_1d(future_capacity, self.target_config.future_len)
                last_observed_cycle = float(observed["cycle_number"].iloc[-1])
                last_capacity_ratio = float(observed["capacity_ratio"].iloc[-1])
                knee_mask = 1.0 if knee_cycle is not None and float(knee_cycle) > last_observed_cycle else 0.0
                domain = self._encode_domain(observed.iloc[-1])
                items.append(
                    {
                        "battery_id": str(observed["canonical_battery_id"].iloc[-1]),
                        "sequence": observed_features,
                        "trajectory_target": trajectory_target,
                        "soh_target": trajectory_target.copy(),
                        "rul_target": np.asarray([max(0.0, eol_cycle - last_observed_cycle)], dtype=np.float32),
                        "eol_target": np.asarray([eol_cycle], dtype=np.float32),
                        "knee_target": np.asarray([float(knee_cycle or 0.0)], dtype=np.float32),
                        "knee_mask": np.asarray([knee_mask], dtype=np.float32),
                        "last_capacity_ratio": np.asarray([last_capacity_ratio], dtype=np.float32),
                        "observed_cycle": np.asarray([last_observed_cycle], dtype=np.float32),
                        "observation_ratio": np.asarray([ratio], dtype=np.float32),
                        **{key: np.asarray([value], dtype=np.int64) for key, value in domain.items()},
                    }
                )
        return items

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        tensor_payload: dict[str, Any] = {"battery_id": sample["battery_id"]}
        for key, value in sample.items():
            if key == "battery_id":
                continue
            dtype = torch.long if key.endswith("_id") else torch.float32
            tensor_payload[key] = torch.tensor(value, dtype=dtype)
        return tensor_payload


class LifecycleDataModule:
    """Lifecycle forecasting data module supporting source-aware domain labels."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        source: str | Sequence[str] | None = None,
        batch_size: int = 16,
        feature_cols: Optional[Sequence[str]] = None,
        num_workers: int = 0,
        output_dir: str | Path | None = None,
        reuse_existing_split: bool = True,
        seed: int = 42,
        target_config: LifecycleTargetConfig | None = None,
    ):
        source_hint = source.lower() if isinstance(source, str) else None
        self.csv_path = resolve_cycle_summary_path(csv_path, source=source_hint)
        self.batch_size = batch_size
        self.feature_cols = list(feature_cols or LIFECYCLE_FEATURE_COLUMNS)
        self.num_workers = num_workers
        self.reuse_existing_split = reuse_existing_split
        self.seed = seed
        self.target_config = target_config or LifecycleTargetConfig()
        self.output_dir = Path(output_dir) if output_dir is not None else self.csv_path.parent
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data = pd.read_csv(self.csv_path)
        self.sources = self._resolve_sources(source)
        self.data = self.data[self.data["source"].str.lower().isin(self.sources)].copy()
        if self.data.empty:
            raise ValueError(f"数据文件 {self.csv_path} 中不存在来源 {self.sources} 的样本")
        self.data.sort_values(["canonical_battery_id", "cycle_number"], inplace=True)
        self.split = self._resolve_split()
        self.normalization = self._compute_normalization(self.split.train_batteries)
        self.vocab = DomainVocab.build(self.data)
        self.train_dataset = LifecycleSequenceDataset(
            self.data,
            self.split.train_batteries,
            feature_cols=self.feature_cols,
            normalization=self.normalization,
            target_config=self.target_config,
            vocab=self.vocab,
        )
        self.val_dataset = LifecycleSequenceDataset(
            self.data,
            self.split.val_batteries,
            feature_cols=self.feature_cols,
            normalization=self.normalization,
            target_config=self.target_config,
            vocab=self.vocab,
        )
        self.test_dataset = LifecycleSequenceDataset(
            self.data,
            self.split.test_batteries,
            feature_cols=self.feature_cols,
            normalization=self.normalization,
            target_config=self.target_config,
            vocab=self.vocab,
        )

    def _resolve_sources(self, source: str | Sequence[str] | None) -> list[str]:
        if source is None:
            return sorted(self.data["source"].astype(str).str.lower().unique().tolist())
        if isinstance(source, str):
            return [source.lower()]
        return sorted({str(item).lower() for item in source})

    def _resolve_split(self) -> DatasetSplit:
        battery_ids = list(self.data["canonical_battery_id"].unique())
        split_name = self._split_prefix()
        split_path = self.output_dir / f"{split_name}_split.json"
        if self.reuse_existing_split and split_path.exists():
            payload = json.loads(split_path.read_text(encoding="utf-8"))
            split = DatasetSplit.from_dict(payload)
            all_ids = set(split.train_batteries + split.val_batteries + split.test_batteries)
            if all_ids == set(battery_ids) and split.val_batteries and split.test_batteries:
                return split
        return NASABatteryPreprocessor.split_batteries(battery_ids)

    def _split_prefix(self) -> str:
        return self.sources[0] if len(self.sources) == 1 else "multisource"

    def _compute_normalization(self, train_batteries: Iterable[str]) -> NormalizationStats:
        train_frame = self.data[self.data["canonical_battery_id"].isin(train_batteries)]
        means = {column: float(train_frame[column].mean()) for column in self.feature_cols}
        stds = {column: float(train_frame[column].std(ddof=0)) or 1.0 for column in self.feature_cols}
        return NormalizationStats(means=means, stds=stds)

    def _build_loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if shuffle else None,
        )

    def train_loader(self) -> DataLoader:
        return self._build_loader(self.train_dataset, shuffle=True)

    def val_loader(self) -> DataLoader:
        return self._build_loader(self.val_dataset, shuffle=False)

    def test_loader(self) -> DataLoader:
        return self._build_loader(self.test_dataset, shuffle=False)

    def summary(self, *, path_root: str | Path | None = None) -> dict[str, Any]:
        payload = {
            "csv_path": _serialize_path(self.csv_path, path_root),
            "sources": self.sources,
            "feature_columns": self.feature_cols,
            "target_config": self.target_config.to_dict(),
            "split": self.split.to_dict(),
            "normalization": self.normalization.to_dict(),
            "domain_vocab": self.vocab.to_dict(),
            "num_batteries": int(self.data["canonical_battery_id"].nunique()),
            "num_samples": {
                "train": len(self.train_dataset),
                "val": len(self.val_dataset),
                "test": len(self.test_dataset),
            },
        }
        if len(self.sources) == 1:
            card = SOURCE_REGISTRY.get(self.sources[0])
            if card is not None:
                payload.update(
                    {
                        "ingestion_mode": card.ingestion_mode,
                        "training_ready": card.training_ready,
                        "source_group": card.group,
                    }
                )
        return payload

    def export_metadata(
        self,
        output_dir: str | Path | None = None,
        *,
        file_prefix: str | None = None,
        path_root: str | Path | None = None,
    ) -> dict[str, str]:
        summary = self.summary(path_root=path_root)
        target_dir = Path(output_dir) if output_dir is not None else self.output_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        prefix = (file_prefix or self._split_prefix()).lower()
        split_path = target_dir / f"{prefix}_split.json"
        default_feature_config_path = target_dir / f"{prefix}_feature_config.json"
        feature_config_path = target_dir / f"{prefix}_lifecycle_feature_config.json"
        summary_path = target_dir / f"{prefix}_lifecycle_dataset_summary.json"
        target_config_path = target_dir / f"{prefix}_lifecycle_target_config.json"
        split_path.write_text(json.dumps(summary["split"], ensure_ascii=False, indent=2), encoding="utf-8")
        feature_payload = {
            "sources": self.sources,
            "feature_columns": self.feature_cols,
            "target_column": self.target_config.target_column,
            "task_kind": "lifecycle",
        }
        feature_content = json.dumps(feature_payload, ensure_ascii=False, indent=2)
        feature_config_path.write_text(feature_content, encoding="utf-8")
        if not default_feature_config_path.exists():
            default_feature_config_path.write_text(feature_content, encoding="utf-8")
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        target_config_path.write_text(json.dumps(self.target_config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "split": str(split_path),
            "feature_config": str(feature_config_path),
            "summary": str(summary_path),
            "target_config": str(target_config_path),
        }


__all__ = [
    "DomainVocab",
    "LIFECYCLE_FEATURE_COLUMNS",
    "LifecycleDataModule",
    "LifecycleSequenceDataset",
    "LifecycleTargetConfig",
    "estimate_knee_cycle",
]
