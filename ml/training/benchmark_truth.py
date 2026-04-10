"""Unified benchmark truth-source utilities for lifecycle paper evidence."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

CORE_WITHIN_SOURCE_SOURCES = ("nasa", "calce", "kaggle", "hust", "matr")
CORE_TRANSFER_SOURCES = ("calce", "nasa")
CORE_MODELS = ("bilstm", "hybrid")
PAPER_GATE_ABLATION_KEYS = ("no_xlstm", "no_transformer")
PAPER_GATE_ABLATION_SOURCES = ("nasa", "calce", "matr")
SUMMARY_VERSION = "paper_truth_v1"
TASK_KIND = "lifecycle"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def serialize_path(path: str | Path | None) -> str | None:
    if path is None:
        return None
    candidate = Path(path)
    if not candidate.is_absolute():
        return str(candidate.as_posix())
    try:
        return str(candidate.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix())
    except ValueError:
        return str(candidate)


def load_json(path: str | Path) -> dict[str, Any] | None:
    candidate = Path(path)
    if not candidate.exists():
        return None
    return json.loads(candidate.read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _metric(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _aggregate_mean(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    aggregate = summary.get("aggregate_metrics") or {}
    mean = aggregate.get("mean")
    if isinstance(mean, dict):
        return mean
    return summary.get("test_metrics") or {}


def _aggregate_std(summary: dict[str, Any] | None) -> dict[str, Any]:
    if not summary:
        return {}
    aggregate = summary.get("aggregate_metrics") or {}
    std = aggregate.get("std")
    return std if isinstance(std, dict) else {}


def _suite_paths(source: str, model_type: str, model_dir: str | Path) -> dict[str, Path]:
    base = resolve_path(model_dir) / source / model_type
    return {
        "multi_seed": base / f"{model_type}_multi_seed_summary.json",
        "transfer": base / "transfer" / f"multisource_to_{source}" / f"{model_type}_transfer_summary.json",
        "release": base / "release" / "final_release.json",
    }


def load_suite_summary(source: str, model_type: str, suite_kind: str, *, model_dir: str | Path = "data/models") -> tuple[Path, dict[str, Any] | None]:
    paths = _suite_paths(source, model_type, model_dir)
    path = paths[suite_kind]
    return path, load_json(path)


def _best_checkpoint_payload(summary: dict[str, Any] | None) -> dict[str, Any] | None:
    best_checkpoint = (summary or {}).get("best_checkpoint")
    if isinstance(best_checkpoint, dict):
        return deepcopy(best_checkpoint)
    if isinstance(best_checkpoint, str):
        return {"path": best_checkpoint}
    return None


def _model_suite_payload(source: str, model_type: str, suite_kind: str, *, model_dir: str | Path = "data/models") -> dict[str, Any]:
    path, summary = load_suite_summary(source, model_type, suite_kind, model_dir=model_dir)
    metrics = _aggregate_mean(summary)
    payload = {
        "model_type": model_type,
        "suite_kind": suite_kind,
        "summary_path": serialize_path(path),
        "available": bool(summary),
        "metrics": metrics,
        "std": _aggregate_std(summary),
        "best_checkpoint": _best_checkpoint_payload(summary),
        "generated_at": (summary or {}).get("generated_at"),
        "summary_version": (summary or {}).get("summary_version"),
    }
    if summary:
        payload["task_kind"] = summary.get("task_kind", TASK_KIND)
        payload["seeds"] = summary.get("seeds", [])
        payload["release_path"] = serialize_path(_suite_paths(source, model_type, model_dir)["release"])
    return payload


def _dominance(hybrid_metrics: dict[str, Any], bilstm_metrics: dict[str, Any]) -> dict[str, Any]:
    hybrid_rmse = _metric(hybrid_metrics, "rmse")
    bilstm_rmse = _metric(bilstm_metrics, "rmse")
    hybrid_r2 = _metric(hybrid_metrics, "r2")
    bilstm_r2 = _metric(bilstm_metrics, "r2")
    checks = {
        "rmse": hybrid_rmse is not None and bilstm_rmse is not None and hybrid_rmse < bilstm_rmse,
        "r2": hybrid_r2 is not None and bilstm_r2 is not None and hybrid_r2 > bilstm_r2,
    }
    hybrid_beats = all(checks.values())
    bilstm_beats = (
        hybrid_rmse is not None
        and bilstm_rmse is not None
        and bilstm_rmse < hybrid_rmse
        and hybrid_r2 is not None
        and bilstm_r2 is not None
        and bilstm_r2 > hybrid_r2
    )
    winner_model: str | None
    if hybrid_beats:
        winner_model = "hybrid"
    elif bilstm_beats:
        winner_model = "bilstm"
    else:
        winner_model = None
    return {
        "winner_model": winner_model,
        "hybrid_beats_bilstm": hybrid_beats,
        "checks": checks,
        "hybrid_rmse": hybrid_rmse,
        "bilstm_rmse": bilstm_rmse,
        "hybrid_r2": hybrid_r2,
        "bilstm_r2": bilstm_r2,
    }


def build_benchmark_unit(source: str, unit_key: str, *, model_dir: str | Path = "data/models") -> dict[str, Any]:
    suite_kind = "transfer" if unit_key == "transfer" else "multi_seed"
    models = {
        model_type: _model_suite_payload(source, model_type, suite_kind, model_dir=model_dir)
        for model_type in CORE_MODELS
    }
    hybrid = models["hybrid"]
    bilstm = models["bilstm"]
    gate = _dominance(hybrid.get("metrics") or {}, bilstm.get("metrics") or {})
    available = bool(hybrid.get("available") and bilstm.get("available"))
    notes: list[str] = []
    if not available:
        missing = [model for model, payload in models.items() if not payload.get("available")]
        notes.append(f"缺少 {', '.join(missing)} 的 {suite_kind} 摘要。")
    if gate["winner_model"] is None and available:
        notes.append("当前不存在同时在 RMSE 和 R2 上严格占优的单一模型。")
    elif gate["winner_model"] == "hybrid":
        notes.append("Hybrid 在该 benchmark 单元同时满足更低 RMSE 与更高 R2。")
    elif gate["winner_model"] == "bilstm":
        notes.append("BiLSTM 在该 benchmark 单元同时满足更低 RMSE 与更高 R2。")
    return {
        "key": unit_key,
        "label": "Transfer benchmark" if unit_key == "transfer" else "Within-source multi-seed",
        "suite_kind": suite_kind,
        "required_for_paper": unit_key == "within_source" or source in CORE_TRANSFER_SOURCES,
        "available": available,
        "winner_model": gate["winner_model"],
        "paper_gate_passed": gate["hybrid_beats_bilstm"],
        "paper_gate": {
            "hybrid_beats_bilstm": gate["hybrid_beats_bilstm"],
            "checks": gate["checks"],
            "metric_snapshot": {
                "hybrid_rmse": gate["hybrid_rmse"],
                "bilstm_rmse": gate["bilstm_rmse"],
                "hybrid_r2": gate["hybrid_r2"],
                "bilstm_r2": gate["bilstm_r2"],
            },
        },
        "models": models,
        "notes": notes,
    }


def build_full_hybrid_variant(full_summary: dict[str, Any], *, source: str) -> dict[str, Any]:
    return {
        "key": "full_hybrid",
        "label": "完整 Hybrid",
        "description": "复用主实验的 lifecycle hybrid 多随机种子结果。",
        "status": "available",
        "seeds": deepcopy(full_summary.get("seeds", [])),
        "aggregate_metrics": deepcopy(full_summary.get("aggregate_metrics", {})),
        "best_checkpoint": deepcopy(full_summary.get("best_checkpoint", {})),
        "task_kind": full_summary.get("task_kind", TASK_KIND),
        "source": source,
    }


def _variant_metrics(variant: dict[str, Any]) -> dict[str, Any]:
    return ((variant.get("aggregate_metrics") or {}).get("mean") or {}) or (variant.get("test_metrics") or {})


def build_ablation_guard(summary: dict[str, Any]) -> dict[str, Any]:
    variants = {item.get("key"): item for item in summary.get("variants", []) if item.get("key")}
    full_variant = variants.get("full_hybrid")
    full_metrics = _variant_metrics(full_variant or {})
    full_rmse = _metric(full_metrics, "rmse")
    full_r2 = _metric(full_metrics, "r2")
    blocking: list[dict[str, Any]] = []
    for key in PAPER_GATE_ABLATION_KEYS:
        variant = variants.get(key)
        if not variant:
            continue
        metrics = _variant_metrics(variant)
        rmse = _metric(metrics, "rmse")
        r2 = _metric(metrics, "r2")
        if rmse is None or r2 is None or full_rmse is None or full_r2 is None:
            continue
        if rmse < full_rmse and r2 > full_r2:
            blocking.append(
                {
                    "key": key,
                    "rmse": rmse,
                    "r2": r2,
                    "delta_vs_full": variant.get("delta_vs_full", {}),
                }
            )
    return {
        "checked_variants": list(PAPER_GATE_ABLATION_KEYS),
        "blocking_variants": blocking,
        "passed": not blocking,
    }


def sync_ablation_summary(source: str, *, model_dir: str | Path = "data/models", write: bool = True) -> dict[str, Any] | None:
    source = source.lower()
    ablation_path = resolve_path(model_dir) / source / "ablation_summary.json"
    full_summary = load_json(resolve_path(model_dir) / source / "hybrid" / "hybrid_multi_seed_summary.json")
    ablation_summary = load_json(ablation_path)
    if not ablation_summary or not full_summary:
        return ablation_summary

    variants: list[dict[str, Any]] = [deepcopy(item) for item in ablation_summary.get("variants", [])]
    full_variant = build_full_hybrid_variant(full_summary, source=source)
    updated_variants: list[dict[str, Any]] = []
    replaced_full = False
    for item in variants:
        if item.get("key") == "full_hybrid":
            merged = full_variant
            if item.get("description"):
                merged["description"] = item["description"]
            updated_variants.append(merged)
            replaced_full = True
        else:
            updated_variants.append(item)
    if not replaced_full:
        updated_variants.insert(0, full_variant)

    full_rmse = _metric(_variant_metrics(full_variant), "rmse")
    full_r2 = _metric(_variant_metrics(full_variant), "r2")
    for variant in updated_variants:
        metrics = _variant_metrics(variant)
        current_rmse = _metric(metrics, "rmse")
        current_r2 = _metric(metrics, "r2")
        delta_rmse = None if current_rmse is None or full_rmse is None else round(current_rmse - full_rmse, 6)
        delta_r2 = None if current_r2 is None or full_r2 is None else round(current_r2 - full_r2, 6)
        variant["delta_vs_full"] = {"rmse": delta_rmse, "r2": delta_r2}

    ablation_summary["source"] = source
    ablation_summary["task_kind"] = TASK_KIND
    ablation_summary["summary_version"] = SUMMARY_VERSION
    ablation_summary["truth_source"] = serialize_path(resolve_path(model_dir) / source / "hybrid" / "hybrid_multi_seed_summary.json")
    ablation_summary["variants"] = updated_variants
    ablation_summary["guardrail"] = build_ablation_guard(ablation_summary)
    notes = list(ablation_summary.get("notes", []))
    truth_note = "full_hybrid 已与主实验 Hybrid 多随机种子摘要强制对齐，论文引用请以该 truth-source 为准。"
    if truth_note not in notes:
        notes.append(truth_note)
    ablation_summary["notes"] = notes
    ablation_summary["generated_at"] = datetime.utcnow().isoformat()
    if write:
        from ml.training.experiment_runner import generate_source_plot_bundle

        write_json(ablation_path, ablation_summary)
        generate_source_plot_bundle(source, model_dir=model_dir)
    return ablation_summary


def _required_units_for_source(source: str) -> list[str]:
    required = ["within_source"]
    if source in CORE_TRANSFER_SOURCES:
        required.append("transfer")
    if source in PAPER_GATE_ABLATION_SOURCES:
        required.append("ablation")
    return required


def build_source_comparison_summary(source: str, *, model_dir: str | Path = "data/models") -> dict[str, Any]:
    source = source.lower()
    required_units = _required_units_for_source(source)
    benchmark_units = [build_benchmark_unit(source, "within_source", model_dir=model_dir)]
    if source in CORE_TRANSFER_SOURCES:
        benchmark_units.append(build_benchmark_unit(source, "transfer", model_dir=model_dir))

    ablation_summary = sync_ablation_summary(source, model_dir=model_dir, write=False)
    ablation_guard = (ablation_summary or {}).get("guardrail") or {
        "checked_variants": list(PAPER_GATE_ABLATION_KEYS),
        "blocking_variants": [],
        "passed": False,
    }
    best_models = {item["key"]: item.get("winner_model") for item in benchmark_units}
    passing_units = [item["key"] for item in benchmark_units if item.get("paper_gate_passed")]
    available_units = [item["key"] for item in benchmark_units if item.get("available")]
    if ablation_summary:
        available_units.append("ablation")
    if source in PAPER_GATE_ABLATION_SOURCES and ablation_summary and ablation_guard.get("passed"):
        passing_units.append("ablation")
    failing_units = [key for key in required_units if key not in passing_units]
    warnings: list[str] = []
    for item in benchmark_units:
        if not item.get("available"):
            warnings.extend(item.get("notes", []))
        elif not item.get("paper_gate_passed"):
            warnings.append(f"{item['label']} 未达到 Hybrid 全面优于 BiLSTM 的论文门槛。")
    if source in PAPER_GATE_ABLATION_SOURCES and not ablation_summary:
        warnings.append("缺少论文要求的 Hybrid 消融摘要，当前不能作为结构优势强证据。")
    elif ablation_summary and not ablation_guard.get("passed"):
        warnings.append("Hybrid 消融存在优于 full_hybrid 的关键反例，当前不能作为结构优势强证据。")
    summary = {
        "source": source,
        "task_kind": TASK_KIND,
        "summary_version": SUMMARY_VERSION,
        "required_units": required_units,
        "benchmark_units": benchmark_units,
        "best_models": best_models,
        "best_model": best_models.get("within_source"),
        "paper_gate": {
            "required_units": required_units,
            "available_units": available_units,
            "passing_units": passing_units,
            "failing_units": failing_units,
            "passed": not failing_units,
        },
        "ablation_gate": {
            "available": bool(ablation_summary),
            **ablation_guard,
        },
        "warnings": list(dict.fromkeys(warnings)),
        "generated_at": datetime.utcnow().isoformat(),
    }
    return summary


def write_source_comparison_summary(source: str, *, model_dir: str | Path = "data/models") -> Path:
    payload = build_source_comparison_summary(source, model_dir=model_dir)
    path = resolve_path(model_dir) / source / "comparison_summary.json"
    return write_json(path, payload)


def normalize_legacy_comparison_summary(payload: dict[str, Any] | None, *, source: str) -> dict[str, Any]:
    if not payload:
        return build_source_comparison_summary(source)
    if payload.get("benchmark_units"):
        return payload
    models = payload.get("models") or {}
    unit_models = {}
    for model_name in CORE_MODELS:
        model_payload = models.get(model_name) or {}
        unit_models[model_name] = {
            "model_type": model_name,
            "suite_kind": "legacy",
            "summary_path": None,
            "available": bool(model_payload),
            "metrics": model_payload.get("test_metrics") or {},
            "std": {},
            "best_checkpoint": model_payload.get("best_checkpoint"),
            "generated_at": None,
        }
    gate = _dominance(unit_models.get("hybrid", {}).get("metrics") or {}, unit_models.get("bilstm", {}).get("metrics") or {})
    return {
        "source": source,
        "task_kind": payload.get("task_kind", TASK_KIND),
        "summary_version": "legacy_v0",
        "required_units": ["within_source"],
        "benchmark_units": [
            {
                "key": "within_source",
                "label": "Legacy comparison",
                "suite_kind": "legacy",
                "required_for_paper": False,
                "available": any(item.get("available") for item in unit_models.values()),
                "winner_model": gate["winner_model"],
                "paper_gate_passed": gate["hybrid_beats_bilstm"],
                "paper_gate": {
                    "hybrid_beats_bilstm": gate["hybrid_beats_bilstm"],
                    "checks": gate["checks"],
                    "metric_snapshot": {
                        "hybrid_rmse": gate["hybrid_rmse"],
                        "bilstm_rmse": gate["bilstm_rmse"],
                        "hybrid_r2": gate["hybrid_r2"],
                        "bilstm_r2": gate["bilstm_r2"],
                    },
                },
                "models": unit_models,
                "notes": ["当前 comparison_summary 为旧格式，建议重新执行 benchmark truth rebuild。"],
            }
        ],
        "best_models": {"within_source": gate["winner_model"]},
        "best_model": gate["winner_model"],
        "paper_gate": {
            "required_units": ["within_source"],
            "available_units": ["within_source"],
            "passing_units": ["within_source"] if gate["hybrid_beats_bilstm"] else [],
            "failing_units": [] if gate["hybrid_beats_bilstm"] else ["within_source"],
            "passed": gate["hybrid_beats_bilstm"],
        },
        "ablation_gate": {"available": False, "checked_variants": [], "blocking_variants": [], "passed": False},
        "warnings": ["comparison_summary 仍为旧格式。"],
        "generated_at": payload.get("generated_at"),
    }


def collect_paper_evidence(*, model_dir: str | Path = "data/models") -> dict[str, Any]:
    sources = list(CORE_WITHIN_SOURCE_SOURCES)
    comparisons = {source: build_source_comparison_summary(source, model_dir=model_dir) for source in sources}
    matrix: list[dict[str, Any]] = []
    for source, summary in comparisons.items():
        for unit in summary.get("benchmark_units", []):
            models = unit.get("models", {})
            hybrid_metrics = (models.get("hybrid") or {}).get("metrics") or {}
            bilstm_metrics = (models.get("bilstm") or {}).get("metrics") or {}
            matrix.append(
                {
                    "source": source,
                    "unit": unit.get("key"),
                    "suite_kind": unit.get("suite_kind"),
                    "winner_model": unit.get("winner_model"),
                    "paper_gate_passed": unit.get("paper_gate_passed"),
                    "hybrid_rmse": _metric(hybrid_metrics, "rmse"),
                    "hybrid_r2": _metric(hybrid_metrics, "r2"),
                    "bilstm_rmse": _metric(bilstm_metrics, "rmse"),
                    "bilstm_r2": _metric(bilstm_metrics, "r2"),
                }
            )
    matrix_index = {(item["source"], item["unit"]): item for item in matrix}
    failing_units: list[dict[str, Any]] = []
    for source, summary in comparisons.items():
        gate = summary.get("paper_gate") or {}
        for unit_key in gate.get("failing_units", []):
            if unit_key == "ablation":
                ablation_gate = summary.get("ablation_gate") or {}
                failing_units.append(
                    {
                        "source": source,
                        "unit": "ablation",
                        "suite_kind": "ablation",
                        "winner_model": None,
                        "paper_gate_passed": False,
                        "hybrid_rmse": None,
                        "hybrid_r2": None,
                        "bilstm_rmse": None,
                        "bilstm_r2": None,
                        "blocking_variants": ablation_gate.get("blocking_variants", []),
                    }
                )
                continue
            matrix_item = matrix_index.get((source, unit_key))
            if matrix_item is not None:
                failing_units.append(matrix_item)
    return {
        "summary_version": SUMMARY_VERSION,
        "task_kind": TASK_KIND,
        "core_sources": sources,
        "transfer_sources": list(CORE_TRANSFER_SOURCES),
        "matrix": matrix,
        "paper_gate_passed": not failing_units,
        "failing_units": failing_units,
        "generated_at": datetime.utcnow().isoformat(),
        "source_summaries": comparisons,
    }


def render_paper_evidence_markdown(evidence: dict[str, Any]) -> str:
    lines = [
        "# BHMS 论文证据包",
        "",
        f"- 生成时间：{evidence.get('generated_at', '--')}",
        f"- 总体论文门槛：{'通过' if evidence.get('paper_gate_passed') else '未通过'}",
        "- 真值来源：仅使用 `*_multi_seed_summary.json` 与 `*_transfer_summary.json` 聚合结果。",
        "",
        "## 核心 benchmark 矩阵",
        "",
        "| Source | Unit | Hybrid RMSE | Hybrid R2 | BiLSTM RMSE | BiLSTM R2 | Winner | Gate |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for item in evidence.get("matrix", []):
        lines.append(
            f"| {item['source']} | {item['unit']} | {item['hybrid_rmse']} | {item['hybrid_r2']} | {item['bilstm_rmse']} | {item['bilstm_r2']} | {item['winner_model'] or 'mixed'} | {'pass' if item['paper_gate_passed'] else 'fail'} |"
        )
    lines.extend(["", "## 结论", ""])
    if evidence.get("paper_gate_passed"):
        lines.append("- 当前核心矩阵已满足 Hybrid 在全部 benchmark 单元上同时优于 BiLSTM，可进入论文主结论撰写。")
    else:
        lines.append("- 当前核心矩阵尚未满足 Hybrid 在全部 benchmark 单元上全面优于 BiLSTM，论文主结论仍需保持“继续优化中”。")
        for item in evidence.get("failing_units", [])[:12]:
            if item.get("unit") == "ablation":
                blocking = ", ".join(variant.get("key", "--") for variant in item.get("blocking_variants", [])[:4]) or "--"
                lines.append(f"- 未通过单元：{item['source']} / ablation，存在 blocking 变体：{blocking}。")
            else:
                lines.append(
                    f"- 未通过单元：{item['source']} / {item['unit']}，Hybrid RMSE={item['hybrid_rmse']}，BiLSTM RMSE={item['bilstm_rmse']}，Hybrid R2={item['hybrid_r2']}，BiLSTM R2={item['bilstm_r2']}。"
                )
    lines.extend(["", "## 消融门槛", ""])
    for source, summary in evidence.get("source_summaries", {}).items():
        guard = summary.get("ablation_gate") or {}
        status = "pass" if guard.get("passed") else "fail"
        lines.append(f"- {source}: {status}")
        for variant in guard.get("blocking_variants", [])[:4]:
            lines.append(f"  - blocking: {variant.get('key')} rmse={variant.get('rmse')} r2={variant.get('r2')}")
    return "\n".join(lines)


def rebuild_benchmark_truth_assets(
    *,
    model_dir: str | Path = "data/models",
    sources: list[str] | None = None,
    paper_json_path: str | Path | None = None,
    paper_markdown_path: str | Path | None = None,
) -> dict[str, Any]:
    target_sources = sources or list(CORE_WITHIN_SOURCE_SOURCES)
    written: dict[str, Any] = {"comparison_summaries": [], "ablation_summaries": [], "paper_artifacts": []}
    for source in target_sources:
        synced = sync_ablation_summary(source, model_dir=model_dir, write=True)
        if synced:
            written["ablation_summaries"].append(str(resolve_path(model_dir) / source / "ablation_summary.json"))
        written["comparison_summaries"].append(str(write_source_comparison_summary(source, model_dir=model_dir)))
    evidence = collect_paper_evidence(model_dir=model_dir)
    if paper_json_path:
        written["paper_artifacts"].append(str(write_json(paper_json_path, evidence)))
    if paper_markdown_path:
        target = Path(paper_markdown_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render_paper_evidence_markdown(evidence), encoding="utf-8")
        written["paper_artifacts"].append(str(target))
    written["paper_gate_passed"] = evidence["paper_gate_passed"]
    written["failing_units"] = evidence["failing_units"]
    return written


__all__ = [
    "CORE_MODELS",
    "CORE_TRANSFER_SOURCES",
    "CORE_WITHIN_SOURCE_SOURCES",
    "PAPER_GATE_ABLATION_KEYS",
    "PAPER_GATE_ABLATION_SOURCES",
    "SUMMARY_VERSION",
    "TASK_KIND",
    "build_ablation_guard",
    "build_benchmark_unit",
    "build_source_comparison_summary",
    "collect_paper_evidence",
    "load_json",
    "normalize_legacy_comparison_summary",
    "rebuild_benchmark_truth_assets",
    "render_paper_evidence_markdown",
    "sync_ablation_summary",
    "write_source_comparison_summary",
]
