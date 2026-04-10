"""Release asset validation and metadata normalization tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.archive_experiment_artifacts import archive_experiment_artifacts  # noqa: E402
from scripts.normalize_repo_metadata_paths import normalize_metadata_paths  # noqa: E402
from scripts.promote_lifecycle_release import promote_release  # noqa: E402
from scripts.validate_release_assets import validate_release_assets  # noqa: E402


def test_normalize_metadata_paths_rewrites_repo_absolute_paths(tmp_path: Path):
    sample_dir = tmp_path / "data" / "models"
    sample_dir.mkdir(parents=True, exist_ok=True)
    payload_path = sample_dir / "sample.json"
    payload_path.write_text(
        json.dumps(
            {
                "summary_path": f"{PROJECT_ROOT}/data/models/calce/hybrid/hybrid_multi_seed_summary.json",
                "nested": {
                    "csv_path": f"{PROJECT_ROOT}/data/processed/calce/calce_cycle_summary.csv",
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    report = normalize_metadata_paths(root_dirs=[sample_dir], check_only=False)
    normalized = json.loads(payload_path.read_text(encoding="utf-8"))

    assert report["ok"] is True
    assert report["changed"] == 1
    assert normalized["summary_path"] == "data/models/calce/hybrid/hybrid_multi_seed_summary.json"
    assert normalized["nested"]["csv_path"] == "data/processed/calce/calce_cycle_summary.csv"


def test_validate_release_assets_matches_current_repo_state():
    report = validate_release_assets()

    assert report["ok"] is True
    assert report["absolute_path_hits"] == []
    assert len(report["release_checks"]) == 10
    assert all(item["checkpoint_exists"] for item in report["release_checks"])
    assert all(item["checkpoint_in_release_dir"] for item in report["release_checks"])
    summary_checks = [item for item in report["summary_checks"] if item.get("suite_kind") in {"transfer", "multi_seed"}]
    assert summary_checks
    assert all(item["metric_keys"] for item in summary_checks)
    assert report["failed_contract_checks"] == []
    assert report["paper_gate"]["paper_gate_passed"] is False
    assert {(item["source"], item["unit"]) for item in report["paper_gate"]["failing_units"]} == {
        ("nasa", "ablation"),
        ("calce", "within_source"),
        ("calce", "ablation"),
        ("kaggle", "within_source"),
        ("matr", "ablation"),
    }


def test_promote_release_copies_checkpoint_into_release_and_syncs_summary(tmp_path: Path):
    model_dir = tmp_path / "data" / "models"
    source_checkpoint = model_dir / "calce" / "hybrid" / "transfer" / "multisource_to_calce" / "fine_tune" / "seed-7" / "hybrid_best.pt"
    source_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"task_kind": "lifecycle", "model_type": "hybrid", "model_state_dict": {}, "model_config": {}}, source_checkpoint)

    summary_path = model_dir / "calce" / "hybrid" / "transfer" / "multisource_to_calce" / "hybrid_transfer_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "task_kind": "lifecycle",
                "suite_kind": "transfer",
                "fine_tune_runs": [
                    {"seed": 7, "metrics": {"trajectory_rmse": 0.08}, "best_checkpoint": str(source_checkpoint)}
                ],
                "best_checkpoint": {"seed": 7, "path": str(source_checkpoint)},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = promote_release(
        model_dir=model_dir,
        source="calce",
        model_type="hybrid",
        label="transfer-final-test",
    )

    release_checkpoint = model_dir / "calce" / "hybrid" / "release" / "checkpoints" / "hybrid_release.pt"
    assert release_checkpoint.exists()
    assert Path(manifest["checkpoint_path"]) == release_checkpoint

    updated_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert Path(updated_summary["best_checkpoint"]["path"]) == release_checkpoint


def test_archive_experiment_artifacts_moves_intermediate_outputs(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Codex"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "config", "user.email", "codex@example.com"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / ".gitignore").write_text("data/archive/\n", encoding="utf-8")

    release_manifest = tmp_path / "data" / "models" / "hust" / "hybrid" / "release" / "final_release.json"
    release_manifest.parent.mkdir(parents=True, exist_ok=True)
    release_manifest.write_text(json.dumps({"checkpoint_path": "data/models/hust/hybrid/release/checkpoints/hybrid_release.pt"}), encoding="utf-8")

    intermediate_file = tmp_path / "data" / "models" / "hust" / "hybrid" / "hybrid_best.pt"
    intermediate_file.parent.mkdir(parents=True, exist_ok=True)
    intermediate_file.write_bytes(b"checkpoint")

    run_file = tmp_path / "data" / "models" / "hust" / "hybrid" / "runs" / "seed-7" / "hybrid_best.pt"
    run_file.parent.mkdir(parents=True, exist_ok=True)
    run_file.write_bytes(b"seed-run")

    subprocess.run(["git", "add", ".gitignore", "data/models"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(["git", "commit", "-m", "seed artifacts"], cwd=tmp_path, check=True, capture_output=True, text=True)

    report = archive_experiment_artifacts(
        project_root=tmp_path,
        archive_label="2026-03-23",
        sources=["hust"],
        models=["hybrid"],
    )

    archived_file = tmp_path / "data" / "archive" / "experiments" / "2026-03-23" / "data" / "models" / "hust" / "hybrid" / "hybrid_best.pt"
    archived_run = tmp_path / "data" / "archive" / "experiments" / "2026-03-23" / "data" / "models" / "hust" / "hybrid" / "runs" / "seed-7" / "hybrid_best.pt"
    assert report["moved_count"] == 2
    assert archived_file.exists()
    assert archived_run.exists()
    assert release_manifest.exists()

    second = archive_experiment_artifacts(
        project_root=tmp_path,
        archive_label="2026-03-23",
        sources=["hust"],
        models=["hybrid"],
    )
    assert second["moved_count"] == 0
