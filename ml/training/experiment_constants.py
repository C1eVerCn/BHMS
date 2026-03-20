"""Shared constants for experiment orchestration."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SEEDS = [7, 21, 42]

DEFAULT_CONFIGS = {
    (source, model_type): PROJECT_ROOT / "configs" / f"{source}_{model_type}.yaml"
    for source in ("nasa", "calce", "kaggle", "hust", "matr", "oxford", "pulsebat")
    for model_type in ("bilstm", "hybrid")
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
        "key": "no_domain_embedding",
        "label": "去掉 domain embedding",
        "description": "移除 source/chemistry/protocol 条件编码，验证多源域先验的贡献。",
        "model_overrides": {"use_domain_embeddings": False},
    },
    {
        "key": "no_trajectory_head",
        "label": "去掉 trajectory head",
        "description": "关闭未来 trajectory decoder，验证显式轨迹监督对 knee/EOL 的贡献。",
        "model_overrides": {"use_trajectory_head": False},
    },
]

__all__ = ["ABLATION_VARIANTS", "DEFAULT_CONFIGS", "DEFAULT_SEEDS", "PROJECT_ROOT"]
