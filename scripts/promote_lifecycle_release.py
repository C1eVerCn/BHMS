#!/usr/bin/env python3
"""Create the final lifecycle release manifest used by inference."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.predictor import LifecycleInferenceService
from ml.training.experiment_artifacts import serialize_path, write_json


def default_summary_candidates(model_dir: Path, source: str, model_type: str) -> list[Path]:
    root = model_dir / source / model_type
    return [
        root / "transfer" / f"multisource_to_{source}" / f"{model_type}_transfer_summary.json",
        root / f"{model_type}_multi_seed_summary.json",
        root / "optimized-config" / "optimized_multi_seed_summary.json",
        root / f"{model_type}_experiment_summary.json",
    ]


def load_json(path: Path) -> dict[str, Any]:
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


def release_checkpoint_path(model_dir: Path, source: str, model_type: str, checkpoint_path: Path) -> Path:
    suffix = checkpoint_path.suffix or ".pt"
    return model_dir / source / model_type / "release" / "checkpoints" / f"{model_type}_release{suffix}"


def copy_checkpoint_into_release(model_dir: Path, source: str, model_type: str, checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    destination = release_checkpoint_path(model_dir, source, model_type, checkpoint_path)
    if checkpoint_path == destination:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, destination)
    return destination


def sync_summary_best_checkpoint(summary_path: Path | None, release_checkpoint: Path) -> None:
    if summary_path is None or not summary_path.exists():
        return
    payload = load_json(summary_path)
    serialized = serialize_path(release_checkpoint)
    current = payload.get("best_checkpoint")
    if isinstance(current, dict):
        updated = dict(current)
        updated["path"] = serialized
        payload["best_checkpoint"] = updated
    elif isinstance(current, str):
        payload["best_checkpoint"] = serialized
    else:
        payload["best_checkpoint"] = {"path": serialized}
    write_json(summary_path, payload)


def promote_release(
    *,
    model_dir: Path,
    source: str,
    model_type: str,
    summary_override: str | None = None,
    checkpoint_override: str | None = None,
    label: str = "final-lifecycle-release",
) -> dict[str, Any]:
    summary_path, checkpoint_path = resolve_summary_and_checkpoint(
        model_dir=model_dir,
        source=source,
        model_type=model_type,
        summary_override=summary_override,
        checkpoint_override=checkpoint_override,
    )
    release_checkpoint = copy_checkpoint_into_release(model_dir, source, model_type, checkpoint_path)
    sync_summary_best_checkpoint(summary_path, release_checkpoint)

    summary_payload = load_json(summary_path) if summary_path and summary_path.exists() else {}
    manifest = {
        "source": source,
        "model_type": model_type,
        "task_kind": "lifecycle",
        "release_label": label,
        "checkpoint_path": serialize_path(release_checkpoint),
        "summary_path": serialize_path(summary_path) if summary_path else None,
        "suite_kind": summary_payload.get("suite_kind"),
        "stage_kind": summary_payload.get("stage_kind"),
        "transfer_key": summary_payload.get("transfer_key"),
        "source_scope": summary_payload.get("source_scope") or summary_payload.get("fine_tune_source_scope"),
        "generated_at": datetime.utcnow().isoformat(),
    }
    release_path = model_dir / source / model_type / "release" / "final_release.json"
    write_json(release_path, manifest)
    return manifest


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

    manifest = promote_release(
        model_dir=model_dir,
        source=args.source,
        model_type=args.model,
        summary_override=args.summary,
        checkpoint_override=args.checkpoint,
        label=args.label,
    )
    release_path = model_dir / args.source / args.model / "release" / "final_release.json"
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Saved release manifest to: {release_path}")


if __name__ == "__main__":
    main()
