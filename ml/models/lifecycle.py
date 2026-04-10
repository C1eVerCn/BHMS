"""Lifecycle forecasting models centered on xLSTM + Transformer hybrid encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.hybrid.rul_predictor import FeatureFusion, TemporalAttentionPooling
from ml.models.transformer import StackedTransformer
from ml.models.xlstm import StackedxLSTM


def _compatible_heads(hidden_dim: int, preferred: int) -> int:
    for value in range(min(preferred, hidden_dim), 0, -1):
        if hidden_dim % value == 0:
            return value
    return 1


@dataclass
class LifecycleHybridConfig:
    input_dim: int = 12
    d_model: int = 128
    xlstm_layers: int = 2
    xlstm_heads: int = 4
    transformer_layers: int = 2
    transformer_heads: int = 8
    transformer_ff_mult: int = 4
    fusion_dim: int = 128
    decoder_heads: int = 4
    future_len: int = 64
    dropout: float = 0.1
    max_seq_len: int = 512
    domain_embedding_dim: int = 16
    source_vocab_size: int = 8
    chemistry_vocab_size: int = 8
    protocol_vocab_size: int = 16
    use_xlstm: bool = True
    use_transformer: bool = True
    use_domain_embeddings: bool = True
    use_trajectory_head: bool = True
    use_uncertainty_head: bool = True
    use_fluctuation_head: bool = True
    trajectory_scale: float = 64.0
    fluctuation_scale: float = 0.12
    pooling_hidden_dim: int = 128


@dataclass
class LifecycleBiLSTMConfig:
    input_dim: int = 12
    hidden_dim: int = 96
    num_layers: int = 2
    bidirectional: bool = True
    future_len: int = 64
    dropout: float = 0.1
    decoder_heads: int = 4
    domain_embedding_dim: int = 16
    source_vocab_size: int = 8
    chemistry_vocab_size: int = 8
    protocol_vocab_size: int = 16
    use_domain_embeddings: bool = True
    use_trajectory_head: bool = True
    use_uncertainty_head: bool = True
    use_fluctuation_head: bool = True
    trajectory_scale: float = 64.0
    fluctuation_scale: float = 0.12
    pooling_hidden_dim: int = 128


class DomainConditioning(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        source_vocab_size: int,
        chemistry_vocab_size: int,
        protocol_vocab_size: int,
        embedding_dim: int,
        dropout: float,
        enabled: bool,
        zero_for_degenerate_vocab: bool = False,
    ):
        super().__init__()
        self.enabled = enabled
        self.hidden_dim = hidden_dim
        # Within-source runs often collapse to a single constant domain token.
        # For Hybrid we can disable that constant bias entirely while preserving
        # domain conditioning for true multi-domain settings.
        self.degenerate_vocab = (
            enabled
            and zero_for_degenerate_vocab
            and source_vocab_size <= 2
            and chemistry_vocab_size <= 1
            and protocol_vocab_size <= 1
        )
        self.loss_enabled = enabled and not self.degenerate_vocab and source_vocab_size > 1
        self.source_embedding = nn.Embedding(max(1, source_vocab_size), embedding_dim)
        self.chemistry_embedding = nn.Embedding(max(1, chemistry_vocab_size), embedding_dim)
        self.protocol_embedding = nn.Embedding(max(1, protocol_vocab_size), embedding_dim)
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        batch_size: int,
        seq_len: int,
        *,
        source_id: Optional[torch.Tensor],
        chemistry_id: Optional[torch.Tensor],
        protocol_id: Optional[torch.Tensor],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled or self.degenerate_vocab:
            zeros = torch.zeros(batch_size, self.hidden_dim, device=device)
            return zeros[:, None, :].expand(batch_size, seq_len, self.hidden_dim), zeros
        source_id = torch.zeros(batch_size, dtype=torch.long, device=device) if source_id is None else source_id.view(-1).to(device)
        chemistry_id = torch.zeros(batch_size, dtype=torch.long, device=device) if chemistry_id is None else chemistry_id.view(-1).to(device)
        protocol_id = torch.zeros(batch_size, dtype=torch.long, device=device) if protocol_id is None else protocol_id.view(-1).to(device)
        combined = torch.cat(
            [
                self.source_embedding(source_id),
                self.chemistry_embedding(chemistry_id),
                self.protocol_embedding(protocol_id),
            ],
            dim=-1,
        )
        context = self.proj(combined)
        return context[:, None, :].expand(batch_size, seq_len, self.hidden_dim), context


class DominanceSafeFusion(nn.Module):
    def __init__(self, xlstm_dim: int, trans_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fused_path = FeatureFusion(xlstm_dim, trans_dim, output_dim, dropout)
        self.xlstm_only = nn.Sequential(
            nn.LayerNorm(xlstm_dim),
            nn.Linear(xlstm_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.transformer_only = nn.Sequential(
            nn.LayerNorm(trans_dim),
            nn.Linear(trans_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.selector = nn.Linear(output_dim * 3, 3)
        nn.init.zeros_(self.selector.weight)
        with torch.no_grad():
            self.selector.bias.copy_(torch.tensor([2.0, 0.0, 0.0]))

    def forward(self, xlstm_feat: torch.Tensor, trans_feat: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        fused_candidate = self.fused_path(xlstm_feat, trans_feat)
        xlstm_candidate = self.xlstm_only(xlstm_feat)
        transformer_candidate = self.transformer_only(trans_feat)
        selector_logits = self.selector(torch.cat([fused_candidate, xlstm_candidate, transformer_candidate], dim=-1))
        selector_weights = torch.softmax(selector_logits, dim=-1)
        combined = (
            selector_weights[..., 0:1] * fused_candidate
            + selector_weights[..., 1:2] * xlstm_candidate
            + selector_weights[..., 2:3] * transformer_candidate
        )
        return combined, {
            "fused_candidate": fused_candidate,
            "xlstm_only_candidate": xlstm_candidate,
            "transformer_only_candidate": transformer_candidate,
            "selector_logits": selector_logits,
            "selector_weights": selector_weights,
        }


class LifecycleDecoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int,
        future_len: int,
        decoder_heads: int,
        dropout: float,
        trajectory_scale: float,
        source_vocab_size: int,
        use_trajectory_head: bool,
        use_uncertainty_head: bool,
        use_fluctuation_head: bool,
        fluctuation_scale: float,
        dual_trajectory_fusion: bool = False,
    ):
        super().__init__()
        heads = _compatible_heads(hidden_dim, decoder_heads)
        self.future_len = future_len
        self.trajectory_scale = max(1.0, trajectory_scale)
        self.use_trajectory_head = use_trajectory_head
        self.use_uncertainty_head = use_uncertainty_head
        self.use_fluctuation_head = use_fluctuation_head and fluctuation_scale > 0
        self.fluctuation_scale = max(0.0, fluctuation_scale)
        self.dual_trajectory_fusion = use_trajectory_head and dual_trajectory_fusion
        self.query_embed = nn.Parameter(torch.randn(future_len, hidden_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, heads, dropout=dropout, batch_first=True)
        self.decoder_ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.trajectory_head = nn.Linear(hidden_dim, 1)
        self.trajectory_fallback = nn.Linear(hidden_dim, future_len)
        self.fluctuation_head = nn.Linear(hidden_dim, 1) if self.use_fluctuation_head else None
        self.trajectory_gate = nn.Linear(hidden_dim * 2, 1) if self.dual_trajectory_fusion else None
        if self.trajectory_gate is not None:
            nn.init.zeros_(self.trajectory_gate.weight)
            nn.init.constant_(self.trajectory_gate.bias, 1.0)
        if self.fluctuation_head is not None:
            nn.init.zeros_(self.fluctuation_head.weight)
            nn.init.zeros_(self.fluctuation_head.bias)
        envelope = 1.0 - 0.55 * torch.linspace(0.0, 1.0, steps=future_len, dtype=torch.float32).pow(1.2)
        self.register_buffer("fluctuation_envelope", envelope, persistent=False)
        self.rul_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.eol_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.knee_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))
        self.uncertainty_head = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))
        self.domain_head = nn.Linear(hidden_dim, max(1, source_vocab_size))

    @staticmethod
    def _effective_trajectory_gate(raw_gate: torch.Tensor) -> torch.Tensor:
        # Keep the step decoder as the default path and let the fallback branch
        # act as a corrective global prior instead of dominating mid-range gates.
        return 1.0 - torch.square(1.0 - raw_gate)

    def forward(
        self,
        memory: torch.Tensor,
        pooled: torch.Tensor,
        *,
        last_capacity_ratio: Optional[torch.Tensor],
        observed_cycle: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch_size = memory.size(0)
        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)
        decoded, decoder_attn = self.cross_attn(queries, memory, memory, need_weights=True)
        decoded = self.decoder_norm(decoded + self.decoder_ffn(decoded))

        fallback_steps = F.softplus(self.trajectory_fallback(pooled))
        trajectory_gate = None
        if self.use_trajectory_head:
            step_steps = F.softplus(self.trajectory_head(decoded).squeeze(-1))
            if self.dual_trajectory_fusion and self.trajectory_gate is not None:
                pooled_context = pooled[:, None, :].expand(batch_size, self.future_len, pooled.size(-1))
                raw_gate = torch.sigmoid(self.trajectory_gate(torch.cat([decoded, pooled_context], dim=-1))).squeeze(-1)
                trajectory_gate = self._effective_trajectory_gate(raw_gate)
                fade_steps = trajectory_gate * step_steps + (1.0 - trajectory_gate) * fallback_steps
            else:
                fade_steps = step_steps
        else:
            fade_steps = fallback_steps

        base_capacity = torch.ones(batch_size, 1, device=memory.device)
        if last_capacity_ratio is not None:
            base_capacity = last_capacity_ratio.view(batch_size, 1).to(memory.device)
        base_trajectory = torch.clamp(base_capacity - torch.cumsum(fade_steps, dim=1) / self.trajectory_scale, min=0.0, max=1.2)
        fluctuation = None
        trajectory = base_trajectory
        if self.fluctuation_head is not None:
            fluctuation = torch.tanh(self.fluctuation_head(decoded).squeeze(-1))
            fluctuation = fluctuation * self.fluctuation_scale * self.fluctuation_envelope.view(1, -1).to(memory.device)
            trajectory = torch.clamp(base_trajectory + fluctuation, min=0.0, max=1.2)
            if self.future_len == 1:
                trajectory = base_trajectory
            elif self.future_len == 2:
                trajectory = torch.cat(
                    [
                        base_trajectory[:, :1],
                        torch.minimum(trajectory[:, -1:], base_trajectory[:, -1:]),
                    ],
                    dim=1,
                )
            else:
                trajectory = torch.cat(
                    [
                        base_trajectory[:, :1],
                        trajectory[:, 1:-1],
                        torch.minimum(trajectory[:, -1:], base_trajectory[:, -1:]),
                    ],
                    dim=1,
                )
        soh = trajectory.clone()

        observed_cycle = torch.zeros(batch_size, 1, device=memory.device) if observed_cycle is None else observed_cycle.view(batch_size, 1).to(memory.device)
        rul = F.softplus(self.rul_head(pooled))
        eol_cycle = observed_cycle + F.softplus(self.eol_head(pooled))
        knee_cycle = observed_cycle + F.softplus(self.knee_head(pooled))
        uncertainty = F.softplus(self.uncertainty_head(pooled)) if self.use_uncertainty_head else torch.zeros_like(rul)
        return {
            "trajectory": trajectory,
            "trajectory_base": base_trajectory,
            "soh_trajectory": soh,
            "rul": rul,
            "eol_cycle": eol_cycle,
            "knee_cycle": knee_cycle,
            "uncertainty": uncertainty,
            "domain_logits": self.domain_head(pooled),
            "decoder_attention": decoder_attn,
            "trajectory_gate": trajectory_gate,
            "trajectory_fluctuation": fluctuation,
        }


class LifecycleHybridPredictor(nn.Module):
    def __init__(self, config: LifecycleHybridConfig):
        super().__init__()
        self.config = config
        self.use_xlstm = config.use_xlstm
        self.use_transformer = config.use_transformer
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.domain = DomainConditioning(
            hidden_dim=config.d_model,
            source_vocab_size=config.source_vocab_size,
            chemistry_vocab_size=config.chemistry_vocab_size,
            protocol_vocab_size=config.protocol_vocab_size,
            embedding_dim=config.domain_embedding_dim,
            dropout=config.dropout,
            enabled=config.use_domain_embeddings,
            zero_for_degenerate_vocab=True,
        )
        self.xlstm = (
            StackedxLSTM(
                input_dim=config.d_model,
                hidden_dim=config.d_model,
                num_layers=config.xlstm_layers,
                num_heads=config.xlstm_heads,
                dropout=config.dropout,
            )
            if config.use_xlstm
            else None
        )
        self.transformer = (
            StackedTransformer(
                d_model=config.d_model,
                num_layers=config.transformer_layers,
                num_heads=config.transformer_heads,
                d_ff=config.d_model * config.transformer_ff_mult,
                dropout=config.dropout,
                max_len=config.max_seq_len,
            )
            if config.use_transformer
            else None
        )
        if config.use_xlstm and config.use_transformer:
            self.fusion = DominanceSafeFusion(config.d_model, config.d_model, config.fusion_dim, config.dropout)
        else:
            self.fusion = None
            self.single_path = nn.Sequential(
                nn.LayerNorm(config.d_model),
                nn.Linear(config.d_model, config.fusion_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
            )
        self.temporal_pool = TemporalAttentionPooling(config.fusion_dim, config.pooling_hidden_dim, config.dropout)
        self.decoder = LifecycleDecoder(
            hidden_dim=config.fusion_dim,
            future_len=config.future_len,
            decoder_heads=config.decoder_heads,
            dropout=config.dropout,
            trajectory_scale=config.trajectory_scale,
            source_vocab_size=config.source_vocab_size,
            use_trajectory_head=config.use_trajectory_head,
            use_uncertainty_head=config.use_uncertainty_head,
            use_fluctuation_head=config.use_fluctuation_head,
            fluctuation_scale=config.fluctuation_scale,
            dual_trajectory_fusion=True,
        )

    def forward(
        self,
        sequence: torch.Tensor,
        *,
        source_id: Optional[torch.Tensor] = None,
        chemistry_id: Optional[torch.Tensor] = None,
        protocol_id: Optional[torch.Tensor] = None,
        last_capacity_ratio: Optional[torch.Tensor] = None,
        observed_cycle: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        tokens = self.input_dropout(self.input_proj(self.input_norm(sequence)))
        domain_sequence, domain_context = self.domain(
            tokens.size(0),
            tokens.size(1),
            source_id=source_id,
            chemistry_id=chemistry_id,
            protocol_id=protocol_id,
            device=tokens.device,
        )
        tokens = tokens + domain_sequence

        xlstm_out = tokens
        xlstm_states = None
        if self.xlstm is not None:
            xlstm_out, xlstm_states = self.xlstm(tokens)
        transformer_out = xlstm_out if self.xlstm is not None else tokens
        attn_weights = []
        if self.transformer is not None:
            transformer_out, attn_weights = self.transformer(tokens)
        fusion_details = None
        if self.fusion is not None:
            fused, fusion_details = self.fusion(xlstm_out, transformer_out)
        else:
            primary = xlstm_out if self.xlstm is not None else transformer_out
            fused = self.single_path(primary)
        pooled, pooling_weights = self.temporal_pool(fused)
        pooled = pooled + domain_context
        outputs = self.decoder(
            fused,
            pooled,
            last_capacity_ratio=last_capacity_ratio,
            observed_cycle=observed_cycle,
        )
        outputs["domain_loss_enabled"] = torch.tensor(self.domain.loss_enabled, device=fused.device)
        if return_features:
            outputs["features"] = {
                "xlstm_out": xlstm_out,
                "transformer_out": transformer_out,
                "fused": fused,
                "fusion_details": fusion_details,
                "pooled": pooled,
                "pooling_weights": pooling_weights,
                "attn_weights": attn_weights,
                "xlstm_states": xlstm_states,
            }
        return outputs


class LifecycleBiLSTMPredictor(nn.Module):
    def __init__(self, config: LifecycleBiLSTMConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        self.input_norm = nn.LayerNorm(config.input_dim)
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dim)
        self.input_dropout = nn.Dropout(config.dropout)
        self.domain = DomainConditioning(
            hidden_dim=config.hidden_dim,
            source_vocab_size=config.source_vocab_size,
            chemistry_vocab_size=config.chemistry_vocab_size,
            protocol_vocab_size=config.protocol_vocab_size,
            embedding_dim=config.domain_embedding_dim,
            dropout=config.dropout,
            enabled=config.use_domain_embeddings,
        )
        self.encoder = nn.LSTM(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=config.bidirectional,
        )
        self.domain_output_proj = nn.Linear(config.hidden_dim, self.hidden_dim)
        self.adapter = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.temporal_pool = TemporalAttentionPooling(self.hidden_dim, config.pooling_hidden_dim, config.dropout)
        self.decoder = LifecycleDecoder(
            hidden_dim=self.hidden_dim,
            future_len=config.future_len,
            decoder_heads=config.decoder_heads,
            dropout=config.dropout,
            trajectory_scale=config.trajectory_scale,
            source_vocab_size=config.source_vocab_size,
            use_trajectory_head=config.use_trajectory_head,
            use_uncertainty_head=config.use_uncertainty_head,
            use_fluctuation_head=config.use_fluctuation_head,
            fluctuation_scale=config.fluctuation_scale,
            dual_trajectory_fusion=False,
        )

    def forward(
        self,
        sequence: torch.Tensor,
        *,
        source_id: Optional[torch.Tensor] = None,
        chemistry_id: Optional[torch.Tensor] = None,
        protocol_id: Optional[torch.Tensor] = None,
        last_capacity_ratio: Optional[torch.Tensor] = None,
        observed_cycle: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        tokens = self.input_dropout(self.input_proj(self.input_norm(sequence)))
        domain_sequence, domain_context = self.domain(
            tokens.size(0),
            tokens.size(1),
            source_id=source_id,
            chemistry_id=chemistry_id,
            protocol_id=protocol_id,
            device=tokens.device,
        )
        encoded, _ = self.encoder(tokens + domain_sequence)
        encoded = self.adapter(encoded)
        pooled, pooling_weights = self.temporal_pool(encoded)
        pooled = pooled + self.domain_output_proj(domain_context)
        outputs = self.decoder(
            encoded,
            pooled,
            last_capacity_ratio=last_capacity_ratio,
            observed_cycle=observed_cycle,
        )
        outputs["domain_loss_enabled"] = torch.tensor(self.domain.loss_enabled, device=encoded.device)
        if return_features:
            outputs["features"] = {
                "encoded": encoded,
                "pooled": pooled,
                "pooling_weights": pooling_weights,
            }
        return outputs


class LifecycleLoss(nn.Module):
    def __init__(
        self,
        *,
        traj_weight: float = 1.0,
        rul_weight: float = 0.5,
        eol_weight: float = 0.4,
        knee_weight: float = 0.25,
        mono_weight: float = 0.02,
        smooth_weight: float = 0.0,
        domain_weight: float = 0.1,
    ):
        super().__init__()
        self.traj_weight = traj_weight
        self.rul_weight = rul_weight
        self.eol_weight = eol_weight
        self.knee_weight = knee_weight
        self.mono_weight = mono_weight
        self.smooth_weight = smooth_weight
        self.domain_weight = domain_weight

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        trajectory = outputs["trajectory"]
        trajectory_target = batch["trajectory_target"].to(trajectory.device)
        rul_target = batch["rul_target"].to(trajectory.device)
        eol_target = batch["eol_target"].to(trajectory.device)
        knee_target = batch["knee_target"].to(trajectory.device)
        knee_mask = batch["knee_mask"].to(trajectory.device)
        source_id = batch["source_id"].view(-1).to(trajectory.device)

        traj_loss = F.smooth_l1_loss(trajectory, trajectory_target)
        rul_loss = F.smooth_l1_loss(outputs["rul"], rul_target)
        eol_loss = F.smooth_l1_loss(outputs["eol_cycle"], eol_target)
        if float(knee_mask.sum().item()) > 0:
            knee_error = F.smooth_l1_loss(outputs["knee_cycle"], knee_target, reduction="none")
            knee_loss = (knee_error * knee_mask).sum() / knee_mask.sum()
        else:
            knee_loss = trajectory.new_tensor(0.0)
        mono_penalty = torch.relu(trajectory[:, 1:] - trajectory[:, :-1]).mean()
        smooth_penalty = torch.abs(trajectory[:, 2:] - 2 * trajectory[:, 1:-1] + trajectory[:, :-2]).mean()
        domain_loss_enabled = outputs.get("domain_loss_enabled", True)
        if isinstance(domain_loss_enabled, torch.Tensor):
            domain_loss_enabled = bool(domain_loss_enabled.detach().to("cpu").item())
        if domain_loss_enabled and outputs["domain_logits"].size(-1) > 1:
            domain_loss = F.cross_entropy(outputs["domain_logits"], source_id)
        else:
            domain_loss = trajectory.new_tensor(0.0)

        total = (
            self.traj_weight * traj_loss
            + self.rul_weight * rul_loss
            + self.eol_weight * eol_loss
            + self.knee_weight * knee_loss
            + self.mono_weight * mono_penalty
            + self.smooth_weight * smooth_penalty
            + self.domain_weight * domain_loss
        )
        return {
            "loss": total,
            "traj_loss": traj_loss,
            "rul_loss": rul_loss,
            "eol_loss": eol_loss,
            "knee_loss": knee_loss,
            "mono_loss": mono_penalty,
            "smooth_loss": smooth_penalty,
            "domain_loss": domain_loss,
        }


__all__ = [
    "DominanceSafeFusion",
    "LifecycleBiLSTMConfig",
    "LifecycleBiLSTMPredictor",
    "LifecycleHybridConfig",
    "LifecycleHybridPredictor",
    "LifecycleLoss",
]
