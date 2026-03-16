"""Lifecycle API service tests."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.core.database import DatabaseManager  # noqa: E402
from backend.app.services.battery_service import BatteryService  # noqa: E402
from backend.app.services.model_service import PredictionService  # noqa: E402
from backend.app.services.repository import BHMSRepository  # noqa: E402
from ml.data.dataset import create_synthetic_data  # noqa: E402


def _make_settings(tmp_path: Path):
    base = get_settings()
    data_dir = tmp_path / "data"
    return replace(
        base,
        project_root=tmp_path,
        data_dir=data_dir,
        raw_nasa_dir=data_dir / "raw" / "nasa",
        raw_calce_dir=data_dir / "raw" / "calce",
        raw_kaggle_dir=data_dir / "raw" / "kaggle",
        raw_hust_dir=data_dir / "raw" / "hust",
        raw_matr_dir=data_dir / "raw" / "matr",
        raw_oxford_dir=data_dir / "raw" / "oxford",
        raw_pulsebat_dir=data_dir / "raw" / "pulsebat",
        processed_dir=data_dir / "processed",
        knowledge_path=PROJECT_ROOT / "data" / "knowledge" / "battery_fault_knowledge.json",
        model_dir=data_dir / "models",
        upload_dir=data_dir / "uploads",
        demo_upload_dir=data_dir / "demo_uploads",
        database_path=data_dir / "bhms.db",
        graph_backend="memory",
    )


def test_prediction_service_supports_lifecycle_prediction_and_mechanism_explanation(tmp_path: Path):
    settings = _make_settings(tmp_path)
    database = DatabaseManager(settings.database_path)
    database.initialize()
    repo = BHMSRepository(database)
    battery_service = BatteryService(repository=repo, settings=settings)
    prediction_service = PredictionService(repository=repo, settings=settings)

    frame = create_synthetic_data(tmp_path / "calce.csv", num_batteries=1, num_cycles=36, source="calce", dataset_name="calce_service")
    battery_service.import_frame(frame, source="calce", dataset_path=tmp_path / "calce.csv", include_in_training=True)
    battery_id = str(frame["battery_id"].iloc[0])

    lifecycle = prediction_service.predict_lifecycle(battery_id=battery_id, model_name="hybrid", seq_len=20)
    explanation = prediction_service.explain_mechanism(
        battery_id=battery_id,
        anomalies=[
            {
                "code": "capacity_drop",
                "symptom": "容量骤降",
                "severity": "high",
                "description": "容量快速下降",
                "source": "statistical",
                "evidence": ["capacity_ratio below baseline"],
            }
        ],
    )

    assert lifecycle["predicted_eol_cycle"] is not None
    assert lifecycle["trajectory"]
    assert lifecycle["future_risks"]["future_capacity_fade_pattern"] in {"stable_decline", "accelerated_tail_fade"}
    assert explanation["graph_backend"] == "memory"
    assert "lifecycle_evidence" in explanation
