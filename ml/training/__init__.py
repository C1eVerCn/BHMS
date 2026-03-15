"""
模型训练模块
"""

from .experiment_constants import ABLATION_VARIANTS, DEFAULT_SEEDS
from .experiment_runner import (
    create_ablation_summary,
    create_multi_seed_summary,
    generate_source_plot_bundle,
    run_training_experiment,
)
from .trainer import RULTrainer, TrainingConfig

__all__ = [
    "ABLATION_VARIANTS",
    "DEFAULT_SEEDS",
    "RULTrainer",
    "TrainingConfig",
    "create_ablation_summary",
    "create_multi_seed_summary",
    "generate_source_plot_bundle",
    "run_training_experiment",
]
