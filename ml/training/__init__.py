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
from .lifecycle_experiment_runner import run_lifecycle_experiment
from .lifecycle_transfer_runner import create_transfer_summary, run_transfer_benchmark
from .lifecycle_trainer import LifecycleTrainer, LifecycleTrainingConfig, build_lifecycle_model
from .trainer import RULTrainer, TrainingConfig

__all__ = [
    "ABLATION_VARIANTS",
    "DEFAULT_SEEDS",
    "LifecycleTrainer",
    "LifecycleTrainingConfig",
    "RULTrainer",
    "TrainingConfig",
    "build_lifecycle_model",
    "create_ablation_summary",
    "create_multi_seed_summary",
    "create_transfer_summary",
    "generate_source_plot_bundle",
    "run_lifecycle_experiment",
    "run_transfer_benchmark",
    "run_training_experiment",
]
