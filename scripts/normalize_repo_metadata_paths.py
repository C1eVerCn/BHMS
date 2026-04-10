#!/usr/bin/env python3
"""Rewrite repo JSON metadata so in-repo paths are stored relative to the project root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.training.experiment_artifacts import relativize_payload


DEFAULT_ROOTS = ("data/models", "data/processed")


def collect_json_files(root_dirs: Sequence[str | Path]) -> list[Path]:
    files: list[Path] = []
    for root_dir in root_dirs:
        root = Path(root_dir).expanduser()
        if not root.is_absolute():
            root = (PROJECT_ROOT / root).resolve()
        if not root.exists():
            continue
        files.extend(sorted(path for path in root.rglob("*.json") if path.is_file()))
    return files


def normalize_json_file(path: Path, *, check_only: bool = False) -> dict[str, Any]:
    original = path.read_text(encoding="utf-8")
    payload = json.loads(original)
    normalized = relativize_payload(payload)
    rendered = json.dumps(normalized, ensure_ascii=False, indent=2)
    if original.endswith("\n"):
        rendered += "\n"
    changed = rendered != original
    if changed and not check_only:
        path.write_text(rendered, encoding="utf-8")
    return {"path": str(path), "changed": changed}


def normalize_metadata_paths(
    *,
    root_dirs: Sequence[str | Path] = DEFAULT_ROOTS,
    check_only: bool = False,
) -> dict[str, Any]:
    files = collect_json_files(root_dirs)
    scanned = 0
    changed = 0
    changed_files: list[str] = []
    errors: list[dict[str, str]] = []

    for path in files:
        scanned += 1
        try:
            result = normalize_json_file(path, check_only=check_only)
        except json.JSONDecodeError as exc:
            errors.append({"path": str(path), "error": f"json_decode_error: {exc}"})
            continue
        if result["changed"]:
            changed += 1
            changed_files.append(str(path))

    return {
        "ok": not errors,
        "mode": "check" if check_only else "rewrite",
        "scanned": scanned,
        "changed": changed,
        "changed_files": changed_files,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize in-repo metadata paths to project-relative JSON values")
    parser.add_argument("--root", dest="roots", action="append", default=None, help="Root directory to scan; default: data/models and data/processed")
    parser.add_argument("--check", action="store_true", help="Only report files that would change")
    args = parser.parse_args()

    report = normalize_metadata_paths(root_dirs=args.roots or DEFAULT_ROOTS, check_only=args.check)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"] or (args.check and report["changed"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
