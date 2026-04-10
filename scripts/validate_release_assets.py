#!/usr/bin/env python3
"""Validate BHMS lifecycle release assets, summary coverage, and path portability."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.predictor import LifecycleInferenceService
from ml.training.benchmark_truth import (
    CORE_TRANSFER_SOURCES,
    CORE_WITHIN_SOURCE_SOURCES,
    build_source_comparison_summary,
    collect_paper_evidence,
)


DEFAULT_RELEASE_SOURCES = ("calce", "nasa", "kaggle", "hust", "matr")
DEFAULT_MODELS = ("hybrid", "bilstm")
TRANSFER_SOURCES = tuple(CORE_TRANSFER_SOURCES)
WITHIN_SOURCE_SUMMARY_SOURCES = tuple(CORE_WITHIN_SOURCE_SOURCES)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for item in value.values():
            yield from walk_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from walk_strings(item)
    elif isinstance(value, str):
        yield value


def find_absolute_path_hits(root: Path) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    if not root.exists():
        return hits
    project_root_text = str(PROJECT_ROOT)
    for path in sorted(root.rglob("*.json")):
        try:
            payload = load_json(path)
        except json.JSONDecodeError as exc:
            hits.append({"path": str(path), "kind": "json_decode_error", "detail": str(exc)})
            continue
        matched = [item for item in walk_strings(payload) if project_root_text in item]
        if matched:
            hits.append(
                {
                    "path": str(path),
                    "kind": "absolute_project_path",
                    "examples": matched[:5],
                    "count": len(matched),
                }
            )
    return hits


def _resolve_reference(summary_path: Path, raw_path: str | None) -> str | None:
    if not raw_path:
        return None
    resolved = LifecycleInferenceService._resolve_reference_path(summary_path, raw_path)
    return str(resolved) if resolved is not None else None


def _path_in_release_dir(candidate: Path | None, model_root: Path, source: str, model: str) -> bool:
    if candidate is None:
        return False
    release_dir = model_root / source / model / "release" / "checkpoints"
    try:
        candidate.resolve().relative_to(release_dir.resolve())
    except ValueError:
        return False
    return True


def _check_summary(path: Path, expected_suite_kind: str, *, model_root: Path, source: str, model: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "suite_kind": None,
            "available": False,
            "ok": False,
            "error": "missing_file",
        }

    payload = load_json(path)
    best_checkpoint = payload.get("best_checkpoint")
    raw_best_path: str | None = None
    if isinstance(best_checkpoint, dict):
        raw_best_path = best_checkpoint.get("path")
    elif isinstance(best_checkpoint, str):
        raw_best_path = best_checkpoint
    resolved_best = LifecycleInferenceService._resolve_reference_path(path, raw_best_path) if raw_best_path else None
    best_in_release_dir = _path_in_release_dir(resolved_best, model_root, source, model)
    metrics = _mean_metrics(payload)

    report = {
        "path": str(path),
        "suite_kind": payload.get("suite_kind"),
        "available": payload.get("available", True),
        "best_checkpoint": payload.get("best_checkpoint"),
        "resolved_best_checkpoint": str(resolved_best) if resolved_best else None,
        "best_checkpoint_exists": resolved_best.exists() if resolved_best else False,
        "best_checkpoint_in_release_dir": best_in_release_dir,
        "metric_keys": sorted(metrics.keys()),
        "ok": (
            payload.get("suite_kind") == expected_suite_kind
            and payload.get("available", True) is not False
            and bool(metrics)
        ),
    }
    if not report["ok"] and "error" not in report:
        if not metrics:
            report["error"] = "missing_metrics"
        else:
            report["error"] = "suite_kind_mismatch"
    return report


def _mean_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    aggregate = (payload or {}).get("aggregate_metrics") or {}
    mean = aggregate.get("mean")
    if isinstance(mean, dict):
        return mean
    return (payload or {}).get("test_metrics") or {}


def _comparison_contract_view(payload: dict[str, Any]) -> dict[str, Any]:
    unit_view: dict[str, Any] = {}
    for unit in payload.get("benchmark_units", []):
        key = unit.get("key")
        if not key:
            continue
        unit_view[key] = {
            "suite_kind": unit.get("suite_kind"),
            "winner_model": unit.get("winner_model"),
            "paper_gate_passed": unit.get("paper_gate_passed"),
            "paper_gate": unit.get("paper_gate"),
            "models": {
                model_name: {
                    "summary_path": model_payload.get("summary_path"),
                    "metrics": model_payload.get("metrics"),
                    "std": model_payload.get("std"),
                    "best_checkpoint": model_payload.get("best_checkpoint"),
                }
                for model_name, model_payload in ((unit.get("models") or {}).items())
            },
        }
    ablation_gate = payload.get("ablation_gate") or {}
    return {
        "summary_version": payload.get("summary_version"),
        "task_kind": payload.get("task_kind"),
        "required_units": payload.get("required_units"),
        "best_models": payload.get("best_models"),
        "paper_gate": payload.get("paper_gate"),
        "ablation_gate": {
            "available": ablation_gate.get("available"),
            "checked_variants": ablation_gate.get("checked_variants"),
            "blocking_variants": [item.get("key") for item in ablation_gate.get("blocking_variants", [])],
            "passed": ablation_gate.get("passed"),
        },
        "benchmark_units": unit_view,
    }


def _check_comparison_summary(path: Path, *, source: str, model_root: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "kind": "comparison_summary.json",
            "ok": False,
            "error": "missing_file",
        }
    payload = load_json(path)
    expected = build_source_comparison_summary(source, model_dir=model_root)
    actual_view = _comparison_contract_view(payload)
    expected_view = _comparison_contract_view(expected)
    ok = actual_view == expected_view
    return {
        "path": str(path),
        "kind": "comparison_summary.json",
        "ok": ok,
        "summary_version": payload.get("summary_version"),
        "paper_gate_passed": (payload.get("paper_gate") or {}).get("passed"),
        "error": None if ok else "comparison_summary_not_derived_from_truth",
    }


def _check_ablation_summary(path: Path, *, source: str, model_root: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path),
            "kind": "ablation_summary.json",
            "ok": False,
            "error": "missing_file",
        }
    payload = load_json(path)
    truth_path = model_root / source / "hybrid" / "hybrid_multi_seed_summary.json"
    truth_summary = load_json(truth_path)
    full_variant = next((item for item in payload.get("variants", []) if item.get("key") == "full_hybrid"), None)
    full_metrics = _mean_metrics(full_variant or {})
    truth_metrics = _mean_metrics(truth_summary)
    full_seeds = (full_variant or {}).get("seeds", [])
    truth_seeds = (truth_summary or {}).get("seeds", [])
    guard = payload.get("guardrail") or {}
    ok = (
        payload.get("task_kind") == "lifecycle"
        and full_variant is not None
        and full_metrics == truth_metrics
        and full_seeds == truth_seeds
        and isinstance(guard.get("checked_variants"), list)
    )
    return {
        "path": str(path),
        "kind": "ablation_summary.json",
        "ok": ok,
        "task_kind": payload.get("task_kind"),
        "guardrail_passed": guard.get("passed"),
        "error": None if ok else "ablation_truth_source_mismatch",
    }


def validate_release_assets(
    *,
    project_root: Path = PROJECT_ROOT,
    model_dir: str | Path = "data/models",
    processed_dir: str | Path = "data/processed",
    release_sources: Iterable[str] = DEFAULT_RELEASE_SOURCES,
    models: Iterable[str] = DEFAULT_MODELS,
    require_paper_gate: bool = False,
) -> dict[str, Any]:
    model_root = Path(model_dir).expanduser()
    if not model_root.is_absolute():
        model_root = (project_root / model_root).resolve()
    processed_root = Path(processed_dir).expanduser()
    if not processed_root.is_absolute():
        processed_root = (project_root / processed_root).resolve()

    release_sources = tuple(source.lower() for source in release_sources)
    models = tuple(model.lower() for model in models)

    summary_checks: list[dict[str, Any]] = []
    for source in TRANSFER_SOURCES:
        for model in models:
            path = model_root / source / model / "transfer" / f"multisource_to_{source}" / f"{model}_transfer_summary.json"
            summary_checks.append(_check_summary(path, "transfer", model_root=model_root, source=source, model=model))
    for source in WITHIN_SOURCE_SUMMARY_SOURCES:
        for model in models:
            path = model_root / source / model / f"{model}_multi_seed_summary.json"
            summary_checks.append(_check_summary(path, "multi_seed", model_root=model_root, source=source, model=model))

    contract_checks: list[dict[str, Any]] = []
    for source in WITHIN_SOURCE_SUMMARY_SOURCES:
        contract_checks.append(_check_ablation_summary(model_root / source / "ablation_summary.json", source=source, model_root=model_root))
        contract_checks.append(_check_comparison_summary(model_root / source / "comparison_summary.json", source=source, model_root=model_root))

    service = LifecycleInferenceService(model_root)
    release_checks: list[dict[str, Any]] = []
    for source in release_sources:
        for model in models:
            release_path = model_root / source / model / "release" / "final_release.json"
            if not release_path.exists():
                release_checks.append(
                    {
                        "source": source,
                        "model": model,
                        "release_path": str(release_path),
                        "ok": False,
                        "error": "missing_file",
                    }
                )
                continue
            payload = load_json(release_path)
            try:
                resolved_checkpoint = service._resolve_checkpoint(source, model)
            except Exception as exc:
                release_checks.append(
                    {
                        "source": source,
                        "model": model,
                        "release_path": str(release_path),
                        "release_label": payload.get("release_label"),
                        "suite_kind": payload.get("suite_kind"),
                        "summary_path": payload.get("summary_path"),
                        "checkpoint_path": payload.get("checkpoint_path"),
                        "ok": False,
                        "error": str(exc),
                    }
                )
                continue

            checkpoint_exists = resolved_checkpoint.exists()
            checkpoint_in_release_dir = _path_in_release_dir(resolved_checkpoint, model_root, source, model)
            release_checks.append(
                {
                    "source": source,
                    "model": model,
                    "release_path": str(release_path),
                    "release_label": payload.get("release_label"),
                    "suite_kind": payload.get("suite_kind"),
                    "summary_path": payload.get("summary_path"),
                    "resolved_summary_path": _resolve_reference(release_path, payload.get("summary_path")),
                    "checkpoint_path": payload.get("checkpoint_path"),
                    "resolved_checkpoint_path": str(resolved_checkpoint),
                    "checkpoint_exists": checkpoint_exists,
                    "checkpoint_in_release_dir": checkpoint_in_release_dir,
                    "ok": checkpoint_exists and checkpoint_in_release_dir,
                }
            )

    absolute_path_hits = find_absolute_path_hits(model_root) + find_absolute_path_hits(processed_root)
    failed_summary_checks = [item for item in summary_checks if not item.get("ok")]
    failed_release_checks = [item for item in release_checks if not item.get("ok")]
    failed_contract_checks = [item for item in contract_checks if not item.get("ok")]
    paper_evidence = collect_paper_evidence(model_dir=model_root)
    paper_gate = {
        "required_unit_count": len(paper_evidence.get("matrix", [])),
        "paper_gate_passed": paper_evidence.get("paper_gate_passed", False),
        "failing_units": paper_evidence.get("failing_units", []),
        "enforced": require_paper_gate,
    }

    return {
        "ok": (
            not absolute_path_hits
            and not failed_summary_checks
            and not failed_release_checks
            and not failed_contract_checks
            and (paper_gate["paper_gate_passed"] or not require_paper_gate)
        ),
        "project_root": str(project_root),
        "model_dir": str(model_root),
        "processed_dir": str(processed_root),
        "summary_checks": summary_checks,
        "contract_checks": contract_checks,
        "release_checks": release_checks,
        "paper_gate": paper_gate,
        "absolute_path_hits": absolute_path_hits,
        "failed_summary_checks": failed_summary_checks,
        "failed_contract_checks": failed_contract_checks,
        "failed_release_checks": failed_release_checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate BHMS lifecycle release assets and metadata portability")
    parser.add_argument("--model-dir", default="data/models")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--sources", nargs="+", default=list(DEFAULT_RELEASE_SOURCES))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--require-paper-gate", action="store_true")
    args = parser.parse_args()

    report = validate_release_assets(
        model_dir=args.model_dir,
        processed_dir=args.processed_dir,
        release_sources=args.sources,
        models=args.models,
        require_paper_gate=args.require_paper_gate,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
