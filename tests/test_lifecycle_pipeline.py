"""Lifecycle data/model pipeline tests."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models import (  # noqa: E402
    LifecycleBiLSTMConfig,
    LifecycleBiLSTMPredictor,
    LifecycleHybridConfig,
    LifecycleHybridPredictor,
    LifecycleLoss,
)


def test_lifecycle_hybrid_predictor_outputs_monotonic_trajectory():
    model = LifecycleHybridPredictor(
        LifecycleHybridConfig(
            input_dim=12,
            d_model=32,
            fusion_dim=32,
            xlstm_layers=1,
            transformer_layers=1,
            transformer_heads=4,
            future_len=16,
            source_vocab_size=4,
            chemistry_vocab_size=4,
            protocol_vocab_size=4,
            pooling_hidden_dim=16,
        )
    )
    outputs = model(
        torch.randn(3, 24, 12),
        source_id=torch.tensor([[1], [2], [1]]),
        chemistry_id=torch.tensor([[1], [1], [2]]),
        protocol_id=torch.tensor([[1], [2], [3]]),
        last_capacity_ratio=torch.tensor([[0.97], [0.94], [0.91]]),
        observed_cycle=torch.tensor([[20.0], [25.0], [30.0]]),
        return_features=True,
    )

    assert outputs["trajectory"].shape == (3, 16)
    assert outputs["rul"].shape == (3, 1)
    assert outputs["eol_cycle"].shape == (3, 1)
    assert outputs["knee_cycle"].shape == (3, 1)
    assert outputs["domain_logits"].shape == (3, 4)
    assert torch.all(outputs["trajectory"][:, 1:] <= outputs["trajectory"][:, :-1] + 1e-6)
    assert "features" in outputs


def test_lifecycle_baseline_and_loss_work_together():
    batch = {
        "sequence": torch.randn(2, 20, 12),
        "source_id": torch.tensor([[1], [0]]),
        "chemistry_id": torch.tensor([[1], [1]]),
        "protocol_id": torch.tensor([[2], [3]]),
        "last_capacity_ratio": torch.tensor([[0.95], [0.9]]),
        "observed_cycle": torch.tensor([[15.0], [17.0]]),
        "trajectory_target": torch.full((2, 12), 0.8),
        "rul_target": torch.tensor([[30.0], [24.0]]),
        "eol_target": torch.tensor([[45.0], [41.0]]),
        "knee_target": torch.tensor([[28.0], [0.0]]),
        "knee_mask": torch.tensor([[1.0], [0.0]]),
    }
    model = LifecycleBiLSTMPredictor(
        LifecycleBiLSTMConfig(
            input_dim=12,
            hidden_dim=16,
            num_layers=1,
            future_len=12,
            source_vocab_size=3,
            chemistry_vocab_size=3,
            protocol_vocab_size=4,
            pooling_hidden_dim=12,
        )
    )
    outputs = model(
        batch["sequence"],
        source_id=batch["source_id"],
        chemistry_id=batch["chemistry_id"],
        protocol_id=batch["protocol_id"],
        last_capacity_ratio=batch["last_capacity_ratio"],
        observed_cycle=batch["observed_cycle"],
    )
    losses = LifecycleLoss()(outputs, batch)

    assert outputs["trajectory"].shape == (2, 12)
    assert losses["loss"].item() > 0
    assert losses["knee_loss"].item() >= 0
