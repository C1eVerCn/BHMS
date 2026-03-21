"""Lifecycle training utilities for hybrid and baseline forecasting models."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader

from ml.models import (
    LifecycleBiLSTMConfig,
    LifecycleBiLSTMPredictor,
    LifecycleHybridConfig,
    LifecycleHybridPredictor,
    LifecycleLoss,
)
from ml.training.experiment_artifacts import serialize_path, write_json
from ml.training.trainer import EarlyStopping


MODEL_ALIASES = {
    "hybrid": "lifecycle_hybrid",
    "bilstm": "lifecycle_bilstm",
    "lifecycle_hybrid": "lifecycle_hybrid",
    "lifecycle_bilstm": "lifecycle_bilstm",
}


@dataclass
class LifecycleTrainingConfig:
    source: str = "nasa"
    model_type: str = "hybrid"
    task_kind: str = "lifecycle"
    num_epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    lr_scheduler: str = "cosine_warm_restarts"
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    device: str = "auto"
    use_amp: bool = False
    max_grad_norm: float = 1.0
    checkpoint_dir: str = "data/models"
    log_dir: str = "data/models/logs"
    artifact_subdir: Optional[str] = None
    model_version: str = "lifecycle-v1"
    seed: int = 42
    resume_from: Optional[str] = None
    run_name: Optional[str] = None


def _dataclass_from_overrides(dataclass_type, overrides: Optional[dict[str, Any]] = None):
    allowed = {field.name for field in fields(dataclass_type)}
    payload = {key: value for key, value in (overrides or {}).items() if key in allowed}
    return dataclass_type(**payload)


def _canonical_model_type(model_type: str) -> str:
    try:
        return MODEL_ALIASES[model_type.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown lifecycle model type: {model_type}") from exc


def build_lifecycle_model(
    model_type: str,
    *,
    input_dim: int,
    vocab_sizes: dict[str, int],
    overrides: Optional[dict[str, Any]] = None,
):
    canonical = _canonical_model_type(model_type)
    if canonical == "lifecycle_hybrid":
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


class LifecycleTrainer:
    def __init__(
        self,
        *,
        model: torch.nn.Module,
        model_config: dict[str, Any],
        training_config: LifecycleTrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        data_summary: Optional[dict[str, Any]] = None,
    ):
        self.model = model
        self.model_config = model_config
        self.config = training_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.data_summary = data_summary or {}
        self.device = self._resolve_device(training_config.device)
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.scheduler = self._build_scheduler()
        self.criterion = LifecycleLoss()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.config.use_amp and torch.cuda.is_available())
        self.early_stopping = EarlyStopping(training_config.patience, training_config.min_delta) if training_config.early_stopping else None
        self.run_name = training_config.run_name or f"{training_config.source}_{training_config.model_type}"
        self.history: dict[str, list[dict[str, float]]] = {"train": [], "val": [], "test": []}
        self.test_outputs: dict[str, Any] = {
            "rul_predictions": [],
            "rul_targets": [],
            "trajectory_predictions": [],
            "trajectory_targets": [],
            "errors": [],
        }
        self.checkpoint_dir = Path(self.config.checkpoint_dir) / self.config.source / self.config.model_type
        if self.config.artifact_subdir:
            self.checkpoint_dir = self.checkpoint_dir / self.config.artifact_subdir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(self.config.log_dir) / self.config.source / self.config.model_type
        if self.config.artifact_subdir:
            self.log_dir = self.log_dir / self.config.artifact_subdir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.best_checkpoint_path: Optional[Path] = None
        self.final_checkpoint_path: Optional[Path] = None
        self.best_epoch: Optional[int] = None
        self.stopped_early = False
        self.start_epoch = 0
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if self.config.resume_from:
            self._resume(self.config.resume_from)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(device)

    def _build_scheduler(self):
        if self.config.lr_scheduler == "cosine_warm_restarts":
            return CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        if self.config.lr_scheduler == "plateau":
            return ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)
        return None

    @staticmethod
    def _ensure_finite_tensor(name: str, value: torch.Tensor) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} contains NaN/Inf and the lifecycle run is aborted.")

    @staticmethod
    def _ensure_finite_metric(name: str, value: float) -> None:
        if not math.isfinite(float(value)):
            raise FloatingPointError(f"{name} became non-finite and the lifecycle run is aborted.")

    def _forward_batch(self, batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
            outputs = self.model(
                batch["sequence"].to(self.device),
                source_id=batch.get("source_id"),
                chemistry_id=batch.get("chemistry_id"),
                protocol_id=batch.get("protocol_id"),
                last_capacity_ratio=batch.get("last_capacity_ratio"),
                observed_cycle=batch.get("observed_cycle"),
            )
            losses = self.criterion(outputs, batch)
        for name, loss in losses.items():
            self._ensure_finite_tensor(name, loss)
        return outputs, losses

    @staticmethod
    def _aggregate_metrics(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
        trajectory = outputs["trajectory"].detach().cpu()
        trajectory_target = batch["trajectory_target"].detach().cpu()
        rul = outputs["rul"].detach().cpu()
        rul_target = batch["rul_target"].detach().cpu()
        eol = outputs["eol_cycle"].detach().cpu()
        eol_target = batch["eol_target"].detach().cpu()
        knee = outputs["knee_cycle"].detach().cpu()
        knee_target = batch["knee_target"].detach().cpu()
        knee_mask = batch["knee_mask"].detach().cpu()

        flat_pred = trajectory.view(-1)
        flat_target = trajectory_target.view(-1)
        traj_mse = torch.mean((flat_pred - flat_target) ** 2)
        traj_rmse = torch.sqrt(traj_mse)
        traj_mae = torch.mean(torch.abs(flat_pred - flat_target))
        denom = torch.clamp(torch.abs(flat_target), min=1e-6)
        traj_mape = torch.mean(torch.abs((flat_pred - flat_target) / denom)) * 100.0
        target_mean = torch.mean(flat_target)
        ss_tot = torch.sum((flat_target - target_mean) ** 2)
        ss_res = torch.sum((flat_target - flat_pred) ** 2)
        traj_r2 = 1.0 - (ss_res / torch.clamp(ss_tot, min=1e-6))

        knee_error = torch.abs(knee - knee_target)
        masked_knee = (knee_error * knee_mask).sum() / torch.clamp(knee_mask.sum(), min=1.0)
        return {
            "trajectory_rmse": float(traj_rmse.item()),
            "trajectory_mae": float(traj_mae.item()),
            "trajectory_mape": float(traj_mape.item()),
            "trajectory_r2": float(traj_r2.item()),
            "rul_mae": float(torch.mean(torch.abs(rul - rul_target)).item()),
            "rul_rmse": float(torch.sqrt(torch.mean((rul - rul_target) ** 2)).item()),
            "eol_mae": float(torch.mean(torch.abs(eol - eol_target)).item()),
            "eol_rmse": float(torch.sqrt(torch.mean((eol - eol_target) ** 2)).item()),
            "knee_mae": float(masked_knee.item()),
            # Compatibility aliases for existing comparison/report utilities.
            "rmse": float(traj_rmse.item()),
            "mae": float(traj_mae.item()),
            "mape": float(traj_mape.item()),
            "r2": float(traj_r2.item()),
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
                    self.scaler.scale(losses["loss"]).backward()
                    if self.config.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                metrics = {key: float(value.detach().cpu().item()) for key, value in losses.items()}
                metrics.update(self._aggregate_metrics(outputs, batch))
                for key, value in metrics.items():
                    self._ensure_finite_metric(key, value)
                    aggregates[key] = aggregates.get(key, 0.0) + value
                batches += 1
        result = {key: value / max(1, batches) for key, value in aggregates.items()}
        result["lr"] = float(self.optimizer.param_groups[0]["lr"])
        for key, value in result.items():
            self._ensure_finite_metric(key, value)
        return result

    def _resume(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state:
            self.optimizer.load_state_dict(optimizer_state)
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state and self.scheduler is not None:
            self.scheduler.load_state_dict(scheduler_state)
        scaler_state = checkpoint.get("scaler_state_dict")
        if scaler_state and self.scaler.is_enabled():
            self.scaler.load_state_dict(scaler_state)
        self.best_val_loss = float(checkpoint.get("best_val_loss", self.best_val_loss))
        self.best_epoch = checkpoint.get("best_epoch")
        self.history = checkpoint.get("history", self.history)
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        if self.early_stopping is not None:
            self.early_stopping.best_loss = self.best_val_loss if math.isfinite(self.best_val_loss) else None

    def save_checkpoint(self, filename: str, epoch: int, val_loss: float, *, test_metrics: Optional[dict[str, float]] = None) -> Path:
        checkpoint_path = self.checkpoint_dir / filename
        metadata = {
            "source": self.config.source,
            "task_kind": self.config.task_kind,
            "model_type": self.config.model_type,
            "model_version": self.config.model_version,
            "model_config": self.model_config,
            "training_config": asdict(self.config),
            "data_summary": self.data_summary,
            "feature_columns": self.data_summary.get("feature_columns", []),
            "normalization": self.data_summary.get("normalization", {}),
            "domain_vocab": self.data_summary.get("domain_vocab", {}),
            "target_config": self.data_summary.get("target_config", {}),
            "test_metrics": test_metrics or {},
            "history": self.history,
            "best_val_loss": val_loss,
            "best_epoch": self.best_epoch,
            "device": str(self.device),
            "run_name": self.run_name,
        }
        torch.save(
            {
                "epoch": epoch,
                **metadata,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
                "scaler_state_dict": self.scaler.state_dict() if self.scaler.is_enabled() else None,
            },
            checkpoint_path,
        )
        write_json(checkpoint_path.with_suffix(".json"), metadata)
        return checkpoint_path

    def test(self) -> dict[str, float]:
        if self.test_loader is None:
            return {}
        self.model.eval()
        aggregates: dict[str, float] = {}
        batches = 0
        rul_predictions: list[float] = []
        rul_targets: list[float] = []
        trajectory_predictions: list[list[float]] = []
        trajectory_targets: list[list[float]] = []
        with torch.no_grad():
            for batch in self.test_loader:
                outputs, losses = self._forward_batch(batch)
                metrics = {key: float(value.detach().cpu().item()) for key, value in losses.items()}
                metrics.update(self._aggregate_metrics(outputs, batch))
                for key, value in metrics.items():
                    self._ensure_finite_metric(key, value)
                    aggregates[key] = aggregates.get(key, 0.0) + value
                batches += 1
                rul_predictions.extend(outputs["rul"].detach().cpu().view(-1).tolist())
                rul_targets.extend(batch["rul_target"].detach().cpu().view(-1).tolist())
                trajectory_predictions.extend(outputs["trajectory"].detach().cpu().tolist())
                trajectory_targets.extend(batch["trajectory_target"].detach().cpu().tolist())
        result = {key: value / max(1, batches) for key, value in aggregates.items()}
        result["lr"] = float(self.optimizer.param_groups[0]["lr"])
        self.test_outputs = {
            "rul_predictions": [float(item) for item in rul_predictions],
            "rul_targets": [float(item) for item in rul_targets],
            "trajectory_predictions": [[float(value) for value in row] for row in trajectory_predictions],
            "trajectory_targets": [[float(value) for value in row] for row in trajectory_targets],
            "errors": [float(pred - target) for pred, target in zip(rul_predictions, rul_targets)],
        }
        for key, value in result.items():
            self._ensure_finite_metric(key, value)
        return result

    def fit(self) -> dict[str, Any]:
        started_at = time.perf_counter()
        last_epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, self.config.num_epochs):
            last_epoch = epoch
            train_metrics = self._run_epoch(self.train_loader, training=True)
            self.history["train"].append(train_metrics)
            val_metrics = self._run_epoch(self.val_loader, training=False) if self.val_loader is not None else train_metrics
            self.history["val"].append(val_metrics)
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["loss"])
            elif self.scheduler is not None:
                self.scheduler.step(epoch + 1)
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_epoch = epoch + 1
                self.best_checkpoint_path = self.save_checkpoint(
                    f"{self.config.model_type}_best.pt",
                    epoch=epoch,
                    val_loss=self.best_val_loss,
                )
            if self.early_stopping and self.early_stopping.should_stop(val_metrics["loss"]):
                self.stopped_early = True
                break
        test_metrics = self.test()
        self.history["test"] = [test_metrics] if test_metrics else []
        self.final_checkpoint_path = self.save_checkpoint(
            f"{self.config.model_type}_final.pt",
            epoch=max(last_epoch, 0),
            val_loss=self.best_val_loss,
            test_metrics=test_metrics,
        )
        duration_sec = round(time.perf_counter() - started_at, 3)
        epochs_completed = len(self.history.get("train", []))
        summary = {
            "source": self.config.source,
            "task_kind": self.config.task_kind,
            "model_type": self.config.model_type,
            "model_version": self.config.model_version,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "epochs_completed": epochs_completed,
            "stopped_early": self.stopped_early,
            "resume_from": serialize_path(self.config.resume_from) if self.config.resume_from else None,
            "device": str(self.device),
            "duration_sec": duration_sec,
            "status": "completed",
            "history": self.history,
            "test_metrics": test_metrics,
            "test_details": self.test_outputs,
            "best_checkpoint": serialize_path(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            "final_checkpoint": serialize_path(self.final_checkpoint_path) if self.final_checkpoint_path else None,
        }
        write_json(self.checkpoint_dir / "training_summary.json", summary)
        return summary


__all__ = [
    "LifecycleTrainingConfig",
    "LifecycleTrainer",
    "build_lifecycle_model",
]
