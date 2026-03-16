"""来源级推理服务测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.predictor import RULInferenceService  # noqa: E402


def test_inference_service_uses_heuristic_when_no_checkpoint(tmp_path: Path):
    service = RULInferenceService(tmp_path)
    sequence = np.array([
        [3.7, 0.05, 3.5, 4.1, -1.9, 0.1, 25.0, 1.1, 0.3, 2.0, 1],
        [3.68, 0.05, 3.48, 4.08, -1.9, 0.1, 25.5, 1.1, 0.35, 1.95, 2],
        [3.66, 0.05, 3.46, 4.06, -1.9, 0.1, 26.0, 1.2, 0.4, 1.9, 3],
    ], dtype=float)
    output = service.predict(sequence=sequence, source="calce", model_name="hybrid")
    assert output.fallback_used is True
    assert output.model_source == "calce"
    assert output.predicted_rul >= 0


def test_calibration_raises_implausibly_short_model_output():
    calibrated = RULInferenceService._calibrate_rul_prediction(
        raw_prediction=2.8,
        trend_proxy_rul=34.0,
        latest_capacity=1.85,
        eol_capacity=1.62,
        initial_capacity=2.02,
    )
    assert calibrated["applied"] is True
    assert calibrated["predicted_rul"] > 2.8


def test_resolve_checkpoint_prefers_best_seed_from_summary(tmp_path: Path):
    service = RULInferenceService(tmp_path)
    worse_checkpoint = tmp_path / "calce" / "hybrid" / "optimized-config" / "runs" / "seed-7" / "hybrid_best.pt"
    better_checkpoint = tmp_path / "calce" / "hybrid" / "optimized-config" / "runs" / "seed-21" / "hybrid_best.pt"
    worse_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    better_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    worse_checkpoint.write_bytes(b"worse")
    better_checkpoint.write_bytes(b"better")

    summary_path = tmp_path / "calce" / "hybrid" / "optimized-config" / "optimized_multi_seed_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "per_seed_runs": [
                    {"seed": 7, "metrics": {"rmse": 10.0}, "best_checkpoint": str(worse_checkpoint)},
                    {"seed": 21, "metrics": {"rmse": 6.0}, "best_checkpoint": str(better_checkpoint)},
                ]
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    resolved = service._resolve_checkpoint("calce", "hybrid")
    assert resolved == better_checkpoint
