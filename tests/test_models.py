"""模型单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.models import BiLSTMConfig, BiLSTMRULPredictor, RULPredictor, RULPredictorConfig  # noqa: E402
from ml.models.transformer import PositionalEncoding, TransformerBlock  # noqa: E402
from ml.models.xlstm import mLSTM, sLSTM, xLSTMBlock  # noqa: E402


@pytest.mark.parametrize("batch_size,seq_len,input_dim,hidden_dim", [(4, 24, 10, 32)])
def test_mlstm(batch_size: int, seq_len: int, input_dim: int, hidden_dim: int):
    model = mLSTM(input_dim, hidden_dim, num_heads=4)
    x = torch.randn(batch_size, seq_len, input_dim)
    output, state = model(x)
    assert output.shape == (batch_size, seq_len, hidden_dim)
    assert len(state) == 3


def test_slstm():
    model = sLSTM(10, 32, num_layers=1)
    output, states = model(torch.randn(2, 30, 10))
    assert output.shape == (2, 30, 32)
    assert len(states) == 1


def test_xlstm_block():
    block = xLSTMBlock(10, 32, num_heads=4)
    output, mlstm_state, slstm_state = block(torch.randn(2, 20, 10))
    assert output.shape == (2, 20, 10)
    assert mlstm_state is not None
    assert slstm_state is not None


def test_transformer_block_and_positional_encoding():
    pos = PositionalEncoding(32)
    block = TransformerBlock(32, num_heads=4, d_ff=64)
    x = pos(torch.randn(2, 16, 32))
    output, attn = block(x)
    assert output.shape == (2, 16, 32)
    assert attn.shape == (2, 4, 16, 16)


def test_hybrid_rul_predictor():
    config = RULPredictorConfig(input_dim=10, d_model=64, xlstm_layers=1, transformer_layers=1)
    model = RULPredictor(config)
    prediction, features = model(torch.randn(3, 30, 10), return_features=True)
    assert prediction.shape == (3, 1)
    assert torch.all(prediction >= 0)
    assert features is not None
    assert "fused_feat" in features


def test_bilstm_baseline_predictor():
    config = BiLSTMConfig(input_dim=10, hidden_dim=32, num_layers=1)
    model = BiLSTMRULPredictor(config)
    prediction, features = model(torch.randn(3, 25, 10), return_features=True)
    assert prediction.shape == (3, 1)
    assert torch.all(prediction >= 0)
    assert features is not None
    assert features["pooled_feat"].shape[-1] == 64


def test_hybrid_output_head_has_no_sigmoid():
    model = RULPredictor(RULPredictorConfig(input_dim=10, d_model=32, xlstm_layers=1, transformer_layers=1))
    assert isinstance(model.predictor[-1], torch.nn.Linear)
