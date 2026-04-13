"""启动路径懒加载回归测试。"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.services.battery_service import BatteryService  # noqa: E402


def _purge_modules(prefix: str) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            sys.modules.pop(name, None)


def test_ml_training_package_import_stays_lazy():
    _purge_modules("ml.training")

    package = importlib.import_module("ml.training")

    assert "ml.training.experiment_runner" not in sys.modules
    assert "ml.training.lifecycle_experiment_runner" not in sys.modules
    assert "ml.training.lifecycle_transfer_runner" not in sys.modules
    assert package.DEFAULT_SEEDS == [7, 21, 42]
    assert "ml.training.experiment_constants" in sys.modules
    assert "ml.training.experiment_runner" not in sys.modules


def test_battery_service_builds_adapters_on_demand():
    service = BatteryService(repository=object(), settings=get_settings())

    assert service.adapters == {}

    adapter = service._get_adapter("calce")

    assert adapter.__class__.__name__ == "CALCEAdapter"
    assert list(service.adapters) == ["calce"]
