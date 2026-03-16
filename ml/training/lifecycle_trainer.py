"""Lifecycle training utilities for hybrid and baseline forecasting models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from ml.models import (
    LifecycleBiLSTMConfig,
    LifecycleBiLSTMPredictor,
    LifecycleHybridConfig,
    LifecycleHybridPredictor,
    LifecycleLoss,
)
from ml.training.experiment_artifacts import serialize_path, write_json


@dataclass
class LifecycleTrainingConfig:
    model_type: str = "lifecycle_hybrid"
    num_epochs: int = 30
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    device: str = "auto"
    max_grad_norm: float = 1.0
    checkpoint_dir: str = "data/models/lifecycle"
    model_version: str = "lifecycle-v1"
    seed: int = 42


def _dataclass_from_overrides(dataclass_type, overrides: Optional[dict[str, Any]] = None):
    allowed = {field.name for field in fields(dataclass_type)}
    payload = {key: value for key, value in (overrides or {}).items() if key in allowed}
    return dataclass_type(**payload)


def build_lifecycle_model(
    model_type: str,
    *,
    input_dim: int,
    vocab_sizes: dict[str, int],
    overrides: Optional[dict[str, Any]] = None,
):
    if model_type == "lifecycle_hybrid":
        config = _dataclass_from_overrides(
            LifecycleHybridConfig,
            {
                "input_dim": input_dim,
                "source_vocab_size": vocab_sizes.get("source", 1),
                "chemistry_vocab_size": vocab_sizes.get("chemistry", 1),
                "protocol_vocab_size": vocab_sizes.get("protocol", 1),
                **(overrides or {}),
            },
        )
        return LifecycleHybridPredictor(config), asdict(config)
    if model_type == "lifecycle_bilstm":
        config = _dataclass_from_overrides(
            LifecycleBiLSTMConfig,
            {
                "input_dim": input_dim,
                "source_vocab_size": vocab_sizes.get("source", 1),
                "chemistry_vocab_size": vocab_sizes.get("chemistry", 1),
                "protocol_vocab_size": vocab_sizes.get("protocol", 1),
                **(overrides or {}),
            },
        )
        return LifecycleBiLSTMPredictor(config), asdict(config)
    raise ValueError(f"Unknown lifecycle model type: {model_type}")


class LifecycleTrainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        model_config: dict[str, Any],
        training_config: LifecycleTrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        self.model = model
        self.model_config = model_config
        self.config = training_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = self._resolve_device(training_config.device)
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.criterion = LifecycleLoss()
        self.history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}
        self.checkpoint_dir = Path(self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _forward_batch(self, batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        outputs = self.model(
            batch["sequence"].to(self.device),
            source_id=batch.get("source_id"),
            chemistry_id=batch.get("chemistry_id"),
            protocol_id=batch.get("protocol_id"),
            last_capacity_ratio=batch.get("last_capacity_ratio"),
            observed_cycle=batch.get("observed_cycle"),
        )
        losses = self.criterion(outputs, batch)
        return outputs, losses

    @staticmethod
    def _metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
        trajectory = outputs["trajectory"].detach().cpu()
        trajectory_target = batch["trajectory_target"].detach().cpu()
        rul = outputs["rul"].detach().cpu()
        rul_target = batch["rul_target"].detach().cpu()
        eol = outputs["eol_cycle"].detach().cpu()
        eol_target = batch["eol_target"].detach().cpu()
        knee = outputs["knee_cycle"].detach().cpu()
        knee_target = batch["knee_target"].detach().cpu()
        knee_mask = batch["knee_mask"].detach().cpu()
        knee_error = torch.abs(knee - knee_target)
        masked_knee = (knee_error * knee_mask).sum() / torch.clamp(knee_mask.sum(), min=1.0)
        return {
            "trajectory_mae": float(torch.mean(torch.abs(trajectory - trajectory_target)).item()),
            "rul_mae": float(torch.mean(torch.abs(rul - rul_target)).item()),
            "eol_mae": float(torch.mean(torch.abs(eol - eol_target)).item()),
            "knee_mae": float(masked_knee.item()),
        }

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(mode=training)
        aggregates: dict[str, float] = {}
        batches = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for batch in loader:
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                outputs, losses = self._forward_batch(batch)
                if training:
                    losses["loss"].backward()
                    if self.config.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                metrics = {key: float(value.detach().cpu().item()) for key, value in losses.items()}
                metrics.update(self._metrics(outputs, batch))
                for key, value in metrics.items():
                    aggregates[key] = aggregates.get(key, 0.0) + value
                batches += 1
        return {key: value / max(1, batches) for key, value in aggregates.items()}

    def save_checkpoint(self, filename: str) -> Path:
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": self.model_config,
                "training_config": asdict(self.config),
                "history": self.history,
            },
            checkpoint_path,
        )
        write_json(
            checkpoint_path.with_suffix(".json"),
            {
                "checkpoint_path": serialize_path(checkpoint_path),
                "model_config": self.model_config,
                "training_config": asdict(self.config),
                "history": self.history,
            },
        )
        return checkpoint_path

    def fit(self) -> dict[str, Any]:
        best_val = float("inf")
        best_path: Optional[Path] = None
        for _ in range(self.config.num_epochs):
            train_metrics = self._run_epoch(self.train_loader, training=True)
            self.history["train"].append(train_metrics)
            if self.val_loader is not None:
                val_metrics = self._run_epoch(self.val_loader, training=False)
            else:
                val_metrics = train_metrics
            self.history["val"].append(val_metrics)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                best_path = self.save_checkpoint(f"{self.config.model_type}_best.pt")
        final_path = self.save_checkpoint(f"{self.config.model_type}_final.pt")
        summary = {
            "model_type": self.config.model_type,
            "model_version": self.config.model_version,
            "history": self.history,
            "best_checkpoint": serialize_path(best_path) if best_path else None,
            "final_checkpoint": serialize_path(final_path),
        }
        write_json(self.checkpoint_dir / f"{self.config.model_type}_summary.json", summary)
        return summary


__all__ = [
    "LifecycleTrainingConfig",
    "LifecycleTrainer",
    "build_lifecycle_model",
]
