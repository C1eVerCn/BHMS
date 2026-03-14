"""xLSTM-Transformer 混合 RUL 预测器。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.transformer import StackedTransformer
from ml.models.xlstm import StackedxLSTM


@dataclass
class RULPredictorConfig:
    input_dim: int = 10
    d_model: int = 128
    xlstm_layers: int = 2
    xlstm_heads: int = 4
    use_mlstm: bool = True
    use_slstm: bool = True
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_ff_mult: int = 4
    fusion_dim: int = 128
    output_dim: int = 1
    max_rul: float = 2000.0
    dropout: float = 0.1
    max_seq_len: int = 5000


class FeatureFusion(nn.Module):
    def __init__(self, xlstm_dim: int, trans_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.xlstm_proj = nn.Linear(xlstm_dim, output_dim)
        self.trans_proj = nn.Linear(trans_dim, output_dim)
        self.gate = nn.Sequential(nn.Linear(output_dim * 2, output_dim), nn.Sigmoid())
        self.fusion_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xlstm_feat: torch.Tensor, trans_feat: torch.Tensor) -> torch.Tensor:
        xlstm_proj = self.xlstm_proj(xlstm_feat)
        trans_proj = self.trans_proj(trans_feat)
        gate = self.gate(torch.cat([xlstm_proj, trans_proj], dim=-1))
        fused = gate * xlstm_proj + (1.0 - gate) * trans_proj
        return self.dropout(self.fusion_norm(fused))


class RULPredictor(nn.Module):
    def __init__(self, config: RULPredictorConfig):
        super().__init__()
        self.config = config
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.xlstm = StackedxLSTM(
            input_dim=config.d_model,
            hidden_dim=config.d_model,
            num_layers=config.xlstm_layers,
            num_heads=config.xlstm_heads,
            dropout=config.dropout,
            use_mlstm=config.use_mlstm,
            use_slstm=config.use_slstm,
        )
        self.transformer = StackedTransformer(
            d_model=config.d_model,
            num_layers=config.transformer_layers,
            num_heads=config.transformer_heads,
            d_ff=config.d_model * config.transformer_ff_mult,
            dropout=config.dropout,
            max_len=config.max_seq_len,
        )
        self.fusion = FeatureFusion(config.d_model, config.d_model, config.fusion_dim, config.dropout)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        self.predictor = nn.Sequential(
            nn.LayerNorm(config.fusion_dim),
            nn.Linear(config.fusion_dim, config.fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim // 2, config.fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim // 4, config.output_dim),
        )
        self.register_buffer("max_rul", torch.tensor(config.max_rul))
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, xlstm_states: Optional[list] = None, return_features: bool = False):
        x = self.input_dropout(self.input_proj(self.input_norm(x)))
        xlstm_out, new_xlstm_states = self.xlstm(x, xlstm_states)
        trans_out, attn_weights = self.transformer(xlstm_out)
        fused_feat = self.fusion(xlstm_out, trans_out)
        pooled = self.temporal_pool(fused_feat.transpose(1, 2)).squeeze(-1)
        rul_pred = F.relu(self.predictor(pooled))
        features = None
        if return_features:
            features = {
                "xlstm_out": xlstm_out,
                "trans_out": trans_out,
                "fused_feat": fused_feat,
                "pooled_feat": pooled,
                "attn_weights": attn_weights,
                "xlstm_states": new_xlstm_states,
            }
        return rul_pred, features

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            prediction, _ = self.forward(x)
        return prediction.squeeze(-1)


class RULLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        error = torch.abs(pred - target)
        quadratic = torch.minimum(error, torch.full_like(error, self.delta))
        linear = error - quadratic
        return (0.5 * quadratic.pow(2) + self.delta * linear).mean()


__all__ = ["FeatureFusion", "RULPredictor", "RULPredictorConfig", "RULLoss"]
