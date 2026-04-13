"""模型训练模块。"""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "ABLATION_VARIANTS": ("ml.training.experiment_constants", "ABLATION_VARIANTS"),
    "DEFAULT_SEEDS": ("ml.training.experiment_constants", "DEFAULT_SEEDS"),
    "create_ablation_summary": ("ml.training.experiment_runner", "create_ablation_summary"),
    "create_multi_seed_summary": ("ml.training.experiment_runner", "create_multi_seed_summary"),
    "generate_source_plot_bundle": ("ml.training.experiment_runner", "generate_source_plot_bundle"),
    "run_training_experiment": ("ml.training.experiment_runner", "run_training_experiment"),
    "run_lifecycle_experiment": ("ml.training.lifecycle_experiment_runner", "run_lifecycle_experiment"),
    "create_transfer_summary": ("ml.training.lifecycle_transfer_runner", "create_transfer_summary"),
    "run_transfer_benchmark": ("ml.training.lifecycle_transfer_runner", "run_transfer_benchmark"),
    "LifecycleTrainer": ("ml.training.lifecycle_trainer", "LifecycleTrainer"),
    "LifecycleTrainingConfig": ("ml.training.lifecycle_trainer", "LifecycleTrainingConfig"),
    "build_lifecycle_model": ("ml.training.lifecycle_trainer", "build_lifecycle_model"),
    "RULTrainer": ("ml.training.trainer", "RULTrainer"),
    "TrainingConfig": ("ml.training.trainer", "TrainingConfig"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, symbol_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value
