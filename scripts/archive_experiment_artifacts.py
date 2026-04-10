#!/usr/bin/env python3
"""Archive intermediate BHMS experiment artifacts and stop tracking them in Git."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SOURCES = ("calce", "nasa", "hust", "matr")
DEFAULT_MODELS = ("hybrid", "bilstm")
TRANSFER_SOURCES = {"calce", "nasa"}
ROOT_INTERMEDIATE_NAMES = (
    "{model}_best.json",
    "{model}_best.pt",
    "{model}_final.json",
    "{model}_final.pt",
    "{model}_experiment_summary.json",
    "training_summary.json",
    "test_details.json",
)


def _existing(path: Path) -> Path | None:
    return path if path.exists() else None


def _collapse_paths(paths: Iterable[Path]) -> list[Path]:
    ordered = sorted({path.resolve() for path in paths if path is not None}, key=lambda item: (len(item.parts), str(item)))
    selected: list[Path] = []
    for path in ordered:
        if any(path == parent or parent in path.parents for parent in selected):
            continue
        selected.append(path)
    return selected


def _candidate_paths(model_root: Path, *, sources: Sequence[str], models: Sequence[str]) -> list[Path]:
    candidates: list[Path] = []
    for source in sources:
        for model in models:
            base = model_root / source / model
            candidates.append(base / "runs")
            candidates.append(base / "ablation")
            if source in TRANSFER_SOURCES:
                transfer_root = base / "transfer" / f"multisource_to_{source}"
                candidates.append(transfer_root / "pretrain")
                candidates.append(transfer_root / "fine_tune")
            for template in ROOT_INTERMEDIATE_NAMES:
                candidates.append(base / template.format(model=model))
    return _collapse_paths(filter(None, (_existing(path) for path in candidates)))


def _archive_destination(project_root: Path, archive_root: Path, source_path: Path) -> Path:
    return archive_root / source_path.resolve().relative_to(project_root.resolve())


def _git_ls_files(project_root: Path, relative_path: Path) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", str(relative_path)],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _is_tracked(project_root: Path, source_path: Path) -> bool:
    relative_path = source_path.resolve().relative_to(project_root.resolve())
    return bool(_git_ls_files(project_root, relative_path))


def _drop_from_git(project_root: Path, source_path: Path) -> None:
    relative_path = source_path.resolve().relative_to(project_root.resolve())
    command = ["git", "rm", "-r", "--cached", "--ignore-unmatch", "--", str(relative_path)]
    subprocess.run(command, cwd=project_root, check=True, capture_output=True, text=True)


def _move_path(source_path: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Archive destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source_path), str(destination))


def _prune_empty_parents(path: Path, *, stop_at: Path) -> None:
    current = path.parent
    stop = stop_at.resolve()
    while current.resolve() != stop:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def archive_experiment_artifacts(
    *,
    project_root: Path = PROJECT_ROOT,
    model_dir: str | Path = "data/models",
    archive_label: str | None = None,
    sources: Sequence[str] = DEFAULT_SOURCES,
    models: Sequence[str] = DEFAULT_MODELS,
    dry_run: bool = False,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    model_root = Path(model_dir).expanduser()
    if not model_root.is_absolute():
        model_root = (project_root / model_root).resolve()
    archive_root = project_root / "data" / "archive" / "experiments" / (archive_label or date.today().isoformat())

    selected_paths = _candidate_paths(
        model_root,
        sources=[source.lower() for source in sources],
        models=[model.lower() for model in models],
    )
    operations: list[dict[str, Any]] = []
    moved = 0
    dropped = 0

    for source_path in selected_paths:
        relative_path = source_path.relative_to(project_root)
        tracked = _is_tracked(project_root, source_path)
        destination = _archive_destination(project_root, archive_root, source_path)
        operations.append(
            {
                "source": str(source_path),
                "relative_path": str(relative_path),
                "destination": str(destination),
                "tracked": tracked,
                "kind": "directory" if source_path.is_dir() else "file",
            }
        )
        if dry_run:
            continue
        _move_path(source_path, destination)
        if tracked:
            _drop_from_git(project_root, source_path)
            dropped += 1
        moved += 1
        _prune_empty_parents(source_path, stop_at=model_root)

    report = {
        "ok": True,
        "project_root": str(project_root),
        "model_dir": str(model_root),
        "archive_root": str(archive_root),
        "dry_run": dry_run,
        "moved_count": moved,
        "git_removed_from_index_count": dropped,
        "operations": operations,
        "generated_at": datetime.utcnow().isoformat(),
    }
    if not dry_run:
        archive_root.mkdir(parents=True, exist_ok=True)
        manifest_path = archive_root / "archive_manifest.json"
        manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        report["manifest_path"] = str(manifest_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Archive intermediate BHMS experiment artifacts")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--archive-label", default=None, help="Archive bucket name, default: today's date")
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    report = archive_experiment_artifacts(
        model_dir=args.model_dir,
        archive_label=args.archive_label,
        sources=args.sources,
        models=args.models,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
