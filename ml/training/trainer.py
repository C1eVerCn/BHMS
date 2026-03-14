"""RUL 模型训练器，支持 Bi-LSTM 与 xLSTM-Transformer。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ml.models import BiLSTMConfig, BiLSTMRULPredictor, RULPredictor, RULPredictorConfig, RULLoss


@dataclass
class TrainingConfig:
    model_type: str = "hybrid"
    num_epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lr_scheduler: str = "cosine"
    early_stopping: bool = True
    patience: int = 8
    min_delta: float = 1e-4
    loss_type: str = "huber"
    huber_delta: float = 1.0
    checkpoint_dir: str = "data/models"
    log_dir: str = "data/models/logs"
    device: str = "auto"
    use_amp: bool = False
    max_grad_norm: float = 1.0
    seed: int = 42
    model_version: str = "mvp-v1"


class EarlyStopping:
    def __init__(self, patience: int = 8, min_delta: float = 1e-4):
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
    ):
        self.model = model
        self.model_config = model_config
        self.config = training_config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = self._resolve_device(training_config.device)
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.scheduler = self._build_scheduler()
        self.criterion = self._build_criterion()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.use_amp and torch.cuda.is_available())
        self.early_stopping = EarlyStopping(training_config.patience, training_config.min_delta) if training_config.early_stopping else None
        self.checkpoint_dir = Path(training_config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = Path(training_config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self.best_val_loss = float("inf")
        self.history: dict[str, list[dict[str, float]]] = {"train": [], "val": []}
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

    def _resolve_device(self, value: str) -> torch.device:
        if value == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(value)

    def _build_scheduler(self):
        if self.config.lr_scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=max(1, self.config.num_epochs), eta_min=1e-6)
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

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()
        total_loss = 0.0
        total_rmse = 0.0
        total_mae = 0.0
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
                rmse = torch.sqrt(nn.functional.mse_loss(predictions, targets))
                mae = nn.functional.l1_loss(predictions, targets)
                total_loss += float(loss.item())
                total_rmse += float(rmse.item())
                total_mae += float(mae.item())
                batch_count += 1
        return {
            "loss": total_loss / max(batch_count, 1),
            "rmse": total_rmse / max(batch_count, 1),
            "mae": total_mae / max(batch_count, 1),
        }

    def save_checkpoint(self, filename: str, epoch: int, val_loss: float) -> Path:
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(
            {
                "epoch": epoch,
                "model_type": self.config.model_type,
                "model_version": self.config.model_version,
                "model_config": self.model_config,
                "training_config": asdict(self.config),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_loss": val_loss,
                "history": self.history,
            },
            checkpoint_path,
        )
        metadata_path = checkpoint_path.with_suffix(".json")
        metadata_path.write_text(
            json.dumps(
                {
                    "model_type": self.config.model_type,
                    "model_version": self.config.model_version,
                    "model_config": self.model_config,
                    "best_val_loss": val_loss,
                    "history": self.history,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return checkpoint_path

    def train(self) -> dict[str, Any]:
        for epoch in range(self.config.num_epochs):
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
                self.scheduler.step()

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.save_checkpoint(f"{self.config.model_type}_best.pt", epoch, self.best_val_loss)

            if self.early_stopping and self.early_stopping.should_stop(val_metrics["loss"]):
                break

        final_path = self.save_checkpoint(f"{self.config.model_type}_final.pt", epoch=len(self.history["train"]) - 1, val_loss=self.best_val_loss)
        self.writer.close()
        return {
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "final_checkpoint": str(final_path),
        }

    def test(self) -> dict[str, float]:
        if self.test_loader is None:
            return {}
        return self._run_epoch(self.test_loader, training=False)


def build_model(model_type: str, input_dim: int = 10):
    if model_type == "hybrid":
        config = RULPredictorConfig(input_dim=input_dim)
        return RULPredictor(config), asdict(config)
    if model_type == "bilstm":
        config = BiLSTMConfig(input_dim=input_dim)
        return BiLSTMRULPredictor(config), asdict(config)
    raise ValueError(f"未知模型类型: {model_type}")


__all__ = ["EarlyStopping", "RULTrainer", "TrainingConfig", "build_model"]
