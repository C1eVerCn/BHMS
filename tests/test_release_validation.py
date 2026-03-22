"""Release asset validation and metadata normalization tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.normalize_repo_metadata_paths import normalize_metadata_paths  # noqa: E402
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
    assert len(report["release_checks"]) == 8
    assert all(item["checkpoint_exists"] for item in report["release_checks"])
