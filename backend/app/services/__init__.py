"""服务层导出。"""

from __future__ import annotations

from importlib import import_module

__all__ = ["BatteryService", "BHMSRepository", "PredictionService", "TrainingService"]

_SERVICE_MODULES = {
    "BatteryService": "backend.app.services.battery_service",
    "BHMSRepository": "backend.app.services.repository",
    "PredictionService": "backend.app.services.model_service",
    "TrainingService": "backend.app.services.training_service",
}


def __getattr__(name: str):
    if name not in _SERVICE_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_SERVICE_MODULES[name])
    return getattr(module, name)
