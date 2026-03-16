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
    use_xlstm: bool = True
    use_mlstm: bool = True
    use_slstm: bool = True
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_ff_mult: int = 4
    use_transformer: bool = True
    fusion_dim: int = 128
    output_dim: int = 1
    max_rul: float = 2000.0
    dropout: float = 0.1
    max_seq_len: int = 5000
    transformer_parallel: bool = False
    pooling_mode: str = "mean"
    pooling_hidden_dim: int = 128


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


class TemporalAttentionPooling(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.mix = nn.Sequential(nn.Linear(input_dim * 2, input_dim), nn.Sigmoid())
        self.output_norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.score(sequence)
        weights = torch.softmax(logits, dim=1)
        context = torch.sum(weights * sequence, dim=1)
        recent = sequence[:, -1, :]
        gate = self.mix(torch.cat([context, recent], dim=-1))
        pooled = gate * recent + (1.0 - gate) * context
        return self.dropout(self.output_norm(pooled)), weights.squeeze(-1)


class RULPredictor(nn.Module):
    def __init__(self, config: RULPredictorConfig):
        super().__init__()
        self.config = config
        self.use_xlstm = config.use_xlstm
        self.use_transformer = config.use_transformer
        self.use_dual_path = self.use_xlstm and self.use_transformer
        self.transformer_parallel = config.transformer_parallel
        self.pooling_mode = config.pooling_mode.lower()
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        if self.use_xlstm:
            self.xlstm = StackedxLSTM(
                input_dim=config.d_model,
                hidden_dim=config.d_model,
                num_layers=config.xlstm_layers,
                num_heads=config.xlstm_heads,
                dropout=config.dropout,
                use_mlstm=config.use_mlstm,
                use_slstm=config.use_slstm,
            )
        else:
            self.xlstm = None
        if self.use_transformer:
            self.transformer = StackedTransformer(
                d_model=config.d_model,
                num_layers=config.transformer_layers,
                num_heads=config.transformer_heads,
                d_ff=config.d_model * config.transformer_ff_mult,
                dropout=config.dropout,
                max_len=config.max_seq_len,
            )
        else:
            self.transformer = None
        if self.use_dual_path:
            self.fusion = FeatureFusion(config.d_model, config.d_model, config.fusion_dim, config.dropout)
        else:
            self.single_path_adapter = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.fusion_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
        if self.pooling_mode == "attention":
            pooling_hidden_dim = max(8, config.pooling_hidden_dim)
            self.temporal_pool = TemporalAttentionPooling(config.fusion_dim, pooling_hidden_dim, config.dropout)
        else:
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
        xlstm_out = x
        new_xlstm_states = xlstm_states
        if self.xlstm is not None:
            xlstm_out, new_xlstm_states = self.xlstm(x, xlstm_states)
        trans_input = x if self.transformer_parallel or self.xlstm is None else xlstm_out
        trans_out = trans_input
        attn_weights = []
        if self.transformer is not None:
            trans_out, attn_weights = self.transformer(trans_input)
        if self.use_dual_path:
            fused_feat = self.fusion(xlstm_out, trans_out)
        else:
            primary_path = xlstm_out if self.xlstm is not None else trans_out
            fused_feat = self.single_path_adapter(primary_path)
        pooling_weights = None
        if self.pooling_mode == "attention":
            pooled, pooling_weights = self.temporal_pool(fused_feat)
        else:
            pooled = self.temporal_pool(fused_feat.transpose(1, 2)).squeeze(-1)
        rul_pred = F.relu(self.predictor(pooled))
        features = None
        if return_features:
            features = {
                "xlstm_out": xlstm_out,
                "trans_out": trans_out,
                "fused_feat": fused_feat,
                "pooled_feat": pooled,
                "pooling_weights": pooling_weights,
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

__all__ = ["FeatureFusion", "RULPredictor", "RULPredictorConfig", "RULLoss", "TemporalAttentionPooling"]
