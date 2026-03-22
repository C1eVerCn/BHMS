#!/usr/bin/env python3
"""Create the final lifecycle release manifest used by inference."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.predictor import LifecycleInferenceService
from ml.training.experiment_artifacts import serialize_path


def default_summary_candidates(model_dir: Path, source: str, model_type: str) -> list[Path]:
    root = model_dir / source / model_type
    return [
        root / "transfer" / f"multisource_to_{source}" / f"{model_type}_transfer_summary.json",
        root / f"{model_type}_multi_seed_summary.json",
        root / "optimized-config" / "optimized_multi_seed_summary.json",
        root / f"{model_type}_experiment_summary.json",
    ]


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_summary_and_checkpoint(
    *,
    model_dir: Path,
    source: str,
    model_type: str,
    summary_override: str | None,
    checkpoint_override: str | None,
) -> tuple[Path | None, Path]:
    service = LifecycleInferenceService(model_dir)

    if checkpoint_override:
        checkpoint_path = Path(checkpoint_override).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = (PROJECT_ROOT / checkpoint_path).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Lifecycle checkpoint does not exist: {checkpoint_path}")
        if not service._checkpoint_is_lifecycle(checkpoint_path):
            raise ValueError(f"Checkpoint is not a lifecycle model: {checkpoint_path}")
        summary_path = None
        if summary_override:
            summary_path = Path(summary_override).expanduser()
            if not summary_path.is_absolute():
                summary_path = (PROJECT_ROOT / summary_path).resolve()
        return summary_path, checkpoint_path

    summary_candidates = [Path(summary_override).expanduser()] if summary_override else default_summary_candidates(model_dir, source, model_type)
    for summary_path in summary_candidates:
        if not summary_path.is_absolute():
            summary_path = (PROJECT_ROOT / summary_path).resolve()
        checkpoint_path = service._resolve_lifecycle_checkpoint_from_summary(summary_path)
        if checkpoint_path is not None:
            return summary_path, checkpoint_path
    raise FileNotFoundError(f"Unable to resolve a lifecycle release candidate for {source}/{model_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a lifecycle checkpoint into the final release manifest")
    parser.add_argument("--source", required=True, choices=["nasa", "calce", "kaggle", "hust", "matr", "oxford", "pulsebat"])
    parser.add_argument("--model", required=True, choices=["bilstm", "hybrid"])
    parser.add_argument("--summary", default=None, help="Optional summary JSON to promote")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint override")
    parser.add_argument("--label", default="final-lifecycle-release", help="Release label stored in the manifest")
    parser.add_argument("--model-dir", default="data/models")
    args = parser.parse_args()

    model_dir = Path(args.model_dir).expanduser()
    if not model_dir.is_absolute():
        model_dir = (PROJECT_ROOT / model_dir).resolve()

    summary_path, checkpoint_path = resolve_summary_and_checkpoint(
        model_dir=model_dir,
        source=args.source,
        model_type=args.model,
        summary_override=args.summary,
        checkpoint_override=args.checkpoint,
    )

    summary_payload = load_json(summary_path) if summary_path and summary_path.exists() else {}
    manifest = {
        "source": args.source,
        "model_type": args.model,
        "task_kind": "lifecycle",
        "release_label": args.label,
        "checkpoint_path": serialize_path(checkpoint_path),
        "summary_path": serialize_path(summary_path) if summary_path else None,
        "suite_kind": summary_payload.get("suite_kind"),
        "stage_kind": summary_payload.get("stage_kind"),
        "transfer_key": summary_payload.get("transfer_key"),
        "source_scope": summary_payload.get("source_scope") or summary_payload.get("fine_tune_source_scope"),
        "generated_at": datetime.utcnow().isoformat(),
    }
    release_path = model_dir / args.source / args.model / "release" / "final_release.json"
    release_path.parent.mkdir(parents=True, exist_ok=True)
    release_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Saved release manifest to: {release_path}")


if __name__ == "__main__":
    main()
