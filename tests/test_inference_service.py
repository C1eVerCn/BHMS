"""来源级推理服务测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.predictor import LifecycleInferenceService, RULInferenceService  # noqa: E402


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


def test_lifecycle_inference_skips_non_lifecycle_summaries(tmp_path: Path):
    service = LifecycleInferenceService(tmp_path)
    stale_checkpoint = tmp_path / "calce" / "hybrid" / "optimized-config" / "runs" / "seed-7" / "hybrid_best.pt"
    fresh_checkpoint = tmp_path / "calce" / "hybrid" / "hybrid_best.pt"
    stale_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    fresh_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task_kind": "rul", "model_state_dict": {}}, stale_checkpoint)
    torch.save({"task_kind": "lifecycle", "model_state_dict": {}}, fresh_checkpoint)

    stale_summary = tmp_path / "calce" / "hybrid" / "optimized-config" / "optimized_multi_seed_summary.json"
    stale_summary.write_text(
        json.dumps(
            {
                "best_checkpoint": {"path": str(stale_checkpoint)},
                "per_seed_runs": [{"seed": 7, "metrics": {"rmse": 1.0}, "best_checkpoint": str(stale_checkpoint)}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    resolved = service._resolve_checkpoint("calce", "hybrid")
    assert resolved == fresh_checkpoint


def test_lifecycle_inference_prefers_final_release_manifest(tmp_path: Path):
    service = LifecycleInferenceService(tmp_path)
    stale_checkpoint = tmp_path / "calce" / "hybrid" / "hybrid_best.pt"
    release_checkpoint = tmp_path / "calce" / "hybrid" / "release" / "hybrid_release.pt"
    stale_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    release_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task_kind": "lifecycle", "model_state_dict": {}}, stale_checkpoint)
    torch.save({"task_kind": "lifecycle", "model_state_dict": {}}, release_checkpoint)

    summary_path = tmp_path / "calce" / "hybrid" / "hybrid_multi_seed_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "best_checkpoint": {"path": str(stale_checkpoint)},
                "per_seed_runs": [{"seed": 7, "metrics": {"rmse": 0.9}, "best_checkpoint": str(stale_checkpoint)}],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    release_manifest = tmp_path / "calce" / "hybrid" / "release" / "final_release.json"
    release_manifest.write_text(
        json.dumps(
            {
                "checkpoint_path": str(release_checkpoint),
                "summary_path": str(summary_path),
                "task_kind": "lifecycle",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    resolved = service._resolve_checkpoint("calce", "hybrid")
    assert resolved == release_checkpoint


def test_lifecycle_inference_can_resolve_transfer_summary(tmp_path: Path):
    service = LifecycleInferenceService(tmp_path)
    checkpoint = tmp_path / "calce" / "hybrid" / "transfer" / "multisource_to_calce" / "fine_tune" / "seed-7" / "hybrid_best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task_kind": "lifecycle", "model_state_dict": {}}, checkpoint)

    summary_path = tmp_path / "calce" / "hybrid" / "transfer" / "multisource_to_calce" / "hybrid_transfer_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "fine_tune_runs": [{"seed": 7, "metrics": {"trajectory_rmse": 0.12}, "best_checkpoint": str(checkpoint)}],
                "best_checkpoint": {"seed": 7, "path": str(checkpoint)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    resolved = service._resolve_checkpoint("calce", "hybrid")
    assert resolved == checkpoint


def test_lifecycle_inference_resolves_relative_release_manifest_paths(tmp_path: Path):
    model_dir = tmp_path / "data" / "models"
    service = LifecycleInferenceService(model_dir)
    checkpoint = model_dir / "calce" / "hybrid" / "transfer" / "multisource_to_calce" / "fine_tune" / "seed-7" / "hybrid_best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task_kind": "lifecycle", "model_state_dict": {}}, checkpoint)

    summary_path = model_dir / "calce" / "hybrid" / "transfer" / "multisource_to_calce" / "hybrid_transfer_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "suite_kind": "transfer",
                "fine_tune_runs": [
                    {
                        "seed": 7,
                        "metrics": {"trajectory_rmse": 0.08},
                        "best_checkpoint": "data/models/calce/hybrid/transfer/multisource_to_calce/fine_tune/seed-7/hybrid_best.pt",
                    }
                ],
                "best_checkpoint": {
                    "seed": 7,
                    "path": "data/models/calce/hybrid/transfer/multisource_to_calce/fine_tune/seed-7/hybrid_best.pt",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    release_manifest = model_dir / "calce" / "hybrid" / "release" / "final_release.json"
    release_manifest.parent.mkdir(parents=True, exist_ok=True)
    release_manifest.write_text(
        json.dumps(
            {
                "task_kind": "lifecycle",
                "checkpoint_path": "data/models/calce/hybrid/transfer/multisource_to_calce/fine_tune/seed-7/hybrid_best.pt",
                "summary_path": "data/models/calce/hybrid/transfer/multisource_to_calce/hybrid_transfer_summary.json",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    resolved = service._resolve_checkpoint("calce", "hybrid")
    assert resolved == checkpoint
