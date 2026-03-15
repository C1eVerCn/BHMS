"""Shared constants for experiment orchestration."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SEEDS = [7, 21, 42]

DEFAULT_CONFIGS = {
    ("nasa", "bilstm"): PROJECT_ROOT / "configs" / "nasa_bilstm.yaml",
    ("nasa", "hybrid"): PROJECT_ROOT / "configs" / "nasa_hybrid.yaml",
    ("calce", "bilstm"): PROJECT_ROOT / "configs" / "calce_bilstm.yaml",
    ("calce", "hybrid"): PROJECT_ROOT / "configs" / "calce_hybrid.yaml",
    ("kaggle", "bilstm"): PROJECT_ROOT / "configs" / "kaggle_bilstm.yaml",
    ("kaggle", "hybrid"): PROJECT_ROOT / "configs" / "kaggle_hybrid.yaml",
}

ABLATION_VARIANTS = [
    {
        "key": "full_hybrid",
        "label": "完整 Hybrid",
        "description": "保留 xLSTM 与 Transformer 双路径融合结构。",
        "model_overrides": {},
    },
    {
        "key": "no_xlstm",
        "label": "去掉 xLSTM",
        "description": "仅保留 Transformer 路径，验证 xLSTM 的贡献。",
        "model_overrides": {"use_xlstm": False, "use_transformer": True},
    },
    {
        "key": "no_transformer",
        "label": "去掉 Transformer",
        "description": "仅保留 xLSTM 路径，验证 Transformer 的贡献。",
        "model_overrides": {"use_xlstm": True, "use_transformer": False},
    },
    {
        "key": "capacity_only",
        "label": "仅容量/循环特征",
        "description": "只保留容量与循环次数，验证多特征设计是否必要。",
        "model_overrides": {},
        "feature_columns": ["capacity", "cycle_number"],
    },
]

__all__ = ["ABLATION_VARIANTS", "DEFAULT_CONFIGS", "DEFAULT_SEEDS", "PROJECT_ROOT"]
