"""RUL 模型训练器，支持按来源训练 Bi-LSTM 与 xLSTM-Transformer。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml.models import BiLSTMConfig, BiLSTMRULPredictor, RULPredictor, RULPredictorConfig, RULLoss


@dataclass
class TrainingConfig:
    source: str = "nasa"
    model_type: str = "hybrid"
    num_epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-2
    lr_scheduler: str = "cosine_warm_restarts"
    early_stopping: bool = True
    patience: int = 20
    min_delta: float = 1e-4
    loss_type: str = "huber"
    huber_delta: float = 1.0
    checkpoint_dir: str = "data/models"
    log_dir: str = "data/models/logs"
    device: str = "auto"
    use_amp: bool = False
    max_grad_norm: float = 1.0
    seed: int = 42
    model_version: str = "mvp-v2"
    resume_from: Optional[str] = None
    run_name: Optional[str] = None


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss: Optional[float] = None

    def should_stop(self, loss: float) -> bool:
        if self.best_loss is None or loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class RULTrainer:
    def __init__(
        self,
        model: nn.Module,
        model_config: dict[str, Any],
        training_config: TrainingConfig,
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
        self.criterion = self._build_criterion()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp and torch.cuda.is_available())
        self.early_stopping = EarlyStopping(training_config.patience, training_config.min_delta) if training_config.early_stopping else None
        self.run_name = training_config.run_name or f"{training_config.source}_{training_config.model_type}"
        self.checkpoint_dir = Path(training_config.checkpoint_dir) / training_config.source / training_config.model_type
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(training_config.log_dir) / training_config.source / training_config.model_type
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self.best_val_loss = float("inf")
        self.best_checkpoint_path: Optional[Path] = None
        self.final_checkpoint_path: Optional[Path] = None
        self.history: dict[str, list[dict[str, float]]] = {"train": [], "val": [], "test": []}
        self.start_epoch = 0
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if self.config.resume_from:
            self._resume(self.config.resume_from)

    def _resolve_device(self, value: str) -> torch.device:
        if value == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(value)

    def _build_scheduler(self):
        if self.config.lr_scheduler == "cosine_warm_restarts":
            return CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        if self.config.lr_scheduler == "plateau":
            return ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=3)
        return None

    def _build_criterion(self):
        if self.config.loss_type == "huber":
            return RULLoss(delta=self.config.huber_delta)
        if self.config.loss_type == "mse":
            return nn.MSELoss()
        if self.config.loss_type == "mae":
            return nn.L1Loss()
        raise ValueError(f"未知损失函数类型: {self.config.loss_type}")

    @staticmethod
    def _calculate_metrics(predictions: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
        mse = nn.functional.mse_loss(predictions, targets)
        mae = nn.functional.l1_loss(predictions, targets)
        rmse = torch.sqrt(mse)
        denom = torch.clamp(torch.abs(targets), min=1.0)
        mape = torch.mean(torch.abs((predictions - targets) / denom)) * 100.0
        target_mean = torch.mean(targets)
        ss_tot = torch.sum((targets - target_mean) ** 2)
        ss_res = torch.sum((targets - predictions) ** 2)
        r2 = 1.0 - (ss_res / torch.clamp(ss_tot, min=1e-6))
        return {
            "rmse": float(rmse.item()),
            "mae": float(mae.item()),
            "mape": float(mape.item()),
            "r2": float(r2.item()),
        }

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0.0
        aggregate = {"rmse": 0.0, "mae": 0.0, "mape": 0.0, "r2": 0.0}
        batch_count = 0
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                    predictions, _ = self.model(features)
                    loss = self.criterion(predictions, targets)
                if training:
                    self.scaler.scale(loss).backward()
                    if self.config.max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                metrics = self._calculate_metrics(predictions, targets)
                total_loss += float(loss.item())
                for key, value in metrics.items():
                    aggregate[key] += value
                batch_count += 1
        result = {"loss": total_loss / max(batch_count, 1)}
        for key, value in aggregate.items():
            result[key] = value / max(batch_count, 1)
        result["lr"] = float(self.optimizer.param_groups[0]["lr"])
        return result

    def _resume(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = float(checkpoint.get("best_val_loss", self.best_val_loss))
        self.history = checkpoint.get("history", self.history)
        self.start_epoch = int(checkpoint.get("epoch", -1)) + 1

    def save_checkpoint(self, filename: str, epoch: int, val_loss: float, test_metrics: Optional[dict[str, float]] = None) -> Path:
        checkpoint_path = self.checkpoint_dir / filename
        metadata = {
            "source": self.config.source,
            "model_type": self.config.model_type,
            "model_version": self.config.model_version,
            "model_config": self.model_config,
            "training_config": asdict(self.config),
            "data_summary": self.data_summary,
            "feature_columns": self.data_summary.get("feature_columns", []),
            "normalization": self.data_summary.get("normalization", {}),
            "best_val_loss": val_loss,
            "test_metrics": test_metrics or {},
            "history": self.history,
        }
        torch.save(
            {
                "epoch": epoch,
                **metadata,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            checkpoint_path,
        )
        checkpoint_path.with_suffix(".json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        return checkpoint_path

    def train(self) -> dict[str, Any]:
        last_epoch = self.start_epoch - 1
        for epoch in range(self.start_epoch, self.config.num_epochs):
            last_epoch = epoch
            train_metrics = self._run_epoch(self.train_loader, training=True)
            self.history["train"].append(train_metrics)
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"train/{key}", value, epoch)

            val_metrics = self._run_epoch(self.val_loader, training=False) if self.val_loader is not None else train_metrics
            self.history["val"].append(val_metrics)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"val/{key}", value, epoch)

            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics["loss"])
            elif self.scheduler is not None:
                self.scheduler.step(epoch + 1)

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_checkpoint_path = self.save_checkpoint(f"{self.config.model_type}_best.pt", epoch, self.best_val_loss)

            if self.early_stopping and self.early_stopping.should_stop(val_metrics["loss"]):
                break

        test_metrics = self.test() if self.test_loader is not None else {}
        self.history["test"] = [test_metrics] if test_metrics else []
        self.final_checkpoint_path = self.save_checkpoint(
            f"{self.config.model_type}_final.pt",
            epoch=last_epoch,
            val_loss=self.best_val_loss,
            test_metrics=test_metrics,
        )
        comparison_summary = {
            "source": self.config.source,
            "model_type": self.config.model_type,
            "model_version": self.config.model_version,
            "best_val_loss": self.best_val_loss,
            "best_checkpoint": str(self.best_checkpoint_path) if self.best_checkpoint_path else None,
            "final_checkpoint": str(self.final_checkpoint_path),
            "data_summary": self.data_summary,
            "history": self.history,
            "test_metrics": test_metrics,
        }
        (self.checkpoint_dir / "training_summary.json").write_text(
            json.dumps(comparison_summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.writer.close()
        return comparison_summary

    def test(self) -> dict[str, float]:
        if self.test_loader is None:
            return {}
        return self._run_epoch(self.test_loader, training=False)


def _dataclass_from_overrides(dataclass_type, overrides: Optional[dict[str, Any]] = None):
    allowed = {field.name for field in fields(dataclass_type)}
    payload = {key: value for key, value in (overrides or {}).items() if key in allowed}
    return dataclass_type(**payload)


def build_model(model_type: str, input_dim: int = 10, overrides: Optional[dict[str, Any]] = None):
    if model_type == "hybrid":
        config = _dataclass_from_overrides(RULPredictorConfig, {"input_dim": input_dim, **(overrides or {})})
        return RULPredictor(config), asdict(config)
    if model_type == "bilstm":
        config = _dataclass_from_overrides(BiLSTMConfig, {"input_dim": input_dim, **(overrides or {})})
        return BiLSTMRULPredictor(config), asdict(config)
    raise ValueError(f"未知模型类型: {model_type}")


__all__ = ["EarlyStopping", "RULTrainer", "TrainingConfig", "build_model"]
