"""Bi-LSTM 基线模型，用于 RUL 对照实验与 fallback。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BiLSTMConfig:
    input_dim: int = 10
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.1
    bidirectional: bool = True
    output_dim: int = 1


class BiLSTMRULPredictor(nn.Module):
    """基于 Bi-LSTM 的基线 RUL 预测器。"""

    def __init__(self, config: BiLSTMConfig):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.lstm = nn.LSTM(
            input_size=config.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(output_dim),
            nn.Linear(output_dim, output_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(output_dim // 2, config.output_dim),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.input_norm(x)
        sequence_out, _ = self.lstm(x)
        pooled = sequence_out[:, -1, :]
        prediction = F.relu(self.head(pooled))
        features = {"sequence_out": sequence_out, "pooled_feat": pooled} if return_features else None
        return prediction, features

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            prediction, _ = self.forward(x)
        return prediction.squeeze(-1)


__all__ = ["BiLSTMConfig", "BiLSTMRULPredictor"]
