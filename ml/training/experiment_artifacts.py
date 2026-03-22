"""Helpers for experiment aggregation and static artifact generation."""

from __future__ import annotations

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager

    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ModuleNotFoundError:
    plt = None
    MATPLOTLIB_AVAILABLE = False


PLOT_METRIC_KEYS = ("rmse", "mae", "mape", "r2")
AGGREGATE_METRIC_KEYS = (
    "rmse",
    "mae",
    "mape",
    "r2",
    "trajectory_rmse",
    "trajectory_mae",
    "trajectory_r2",
    "rul_mae",
    "rul_rmse",
    "eol_mae",
    "eol_rmse",
    "knee_mae",
)
PLACEHOLDER_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9sot1mAAAAAASUVORK5CYII="
)
CJK_FONT_CANDIDATES = [
    "PingFang SC",
    "Hiragino Sans GB",
    "Heiti SC",
    "Arial Unicode MS",
    "Noto Sans CJK SC",
    "SimHei",
    "Microsoft YaHei",
]
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _configure_matplotlib_fonts() -> None:
    if not MATPLOTLIB_AVAILABLE:
        return
    available = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((font for font in CJK_FONT_CANDIDATES if font in available), None)
    if selected:
        plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


_configure_matplotlib_fonts()


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(relativize_payload(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return destination


def write_placeholder_png(path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(PLACEHOLDER_PNG)
    return destination


def aggregate_metrics(metric_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    rows = list(metric_rows)
    mean: dict[str, float | None] = {}
    std: dict[str, float | None] = {}
    metric_keys = list(AGGREGATE_METRIC_KEYS)
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in metric_keys:
                metric_keys.append(key)
    for key in metric_keys:
        values = [float(item[key]) for item in rows if isinstance(item.get(key), (int, float))]
        if not values:
            mean[key] = None
            std[key] = None
            continue
        mean[key] = round(float(np.mean(values)), 6)
        std[key] = round(float(np.std(values, ddof=0)), 6)
    return {"mean": mean, "std": std}


def select_best_run(per_seed_runs: list[dict[str, Any]], metric_key: str = "rmse") -> dict[str, Any] | None:
    ranked = [
        item
        for item in per_seed_runs
        if isinstance((item.get("metrics") or {}).get(metric_key), (int, float))
    ]
    if not ranked:
        return None
    return min(ranked, key=lambda item: float(item["metrics"][metric_key]))


def write_plot_metadata(
    image_path: str | Path,
    *,
    key: str,
    title: str,
    description: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image = Path(image_path)
    payload = {
        "key": key,
        "title": title,
        "description": description,
        "path": serialize_path(image),
        "generated_at": datetime.utcnow().isoformat(),
    }
    if extra:
        payload.update(extra)
    metadata_path = image.with_suffix(".json")
    write_json(metadata_path, payload)
    payload["metadata_path"] = serialize_path(metadata_path)
    return payload


def serialize_path(path: str | Path) -> str:
    candidate = Path(path)
    resolved = candidate.resolve() if candidate.exists() else (PROJECT_ROOT / candidate).resolve() if not candidate.is_absolute() else candidate
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(candidate)


def relativize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: relativize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relativize_payload(item) for item in value]
    if isinstance(value, str) and str(PROJECT_ROOT) in value:
        try:
            return str(Path(value).resolve().relative_to(PROJECT_ROOT))
        except Exception:
            return value
    return value


def plot_metric_summary(summary: dict[str, Any], output_path: str | Path, *, title: str, description: str) -> dict[str, Any]:
    aggregate = summary.get("aggregate_metrics") or {}
    mean = aggregate.get("mean") or {}
    std = aggregate.get("std") or {}
    labels = []
    values = []
    errors = []
    for key in PLOT_METRIC_KEYS:
        if not isinstance(mean.get(key), (int, float)):
            continue
        labels.append(key.upper())
        values.append(float(mean[key]))
        errors.append(float(std.get(key) or 0.0))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not MATPLOTLIB_AVAILABLE:
        write_placeholder_png(output)
        return write_plot_metadata(output, key=output.stem, title=title, description=description)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(len(labels))
    ax.bar(positions, values, yerr=errors, capsize=6, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][: len(labels)])
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylabel("Metric value")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return write_plot_metadata(output, key=output.stem, title=title, description=description)


def plot_error_distribution(
    per_seed_runs: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str,
    description: str,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not MATPLOTLIB_AVAILABLE:
        write_placeholder_png(output)
        return write_plot_metadata(output, key=output.stem, title=title, description=description)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    has_data = False
    for run in per_seed_runs:
        errors = (((run.get("test_details") or {}).get("errors")) or [])[:]
        if not errors:
            continue
        has_data = True
        ax.hist(errors, bins=min(20, max(6, len(errors) // 2)), alpha=0.35, label=f"seed {run.get('seed')}")
    if not has_data:
        ax.text(0.5, 0.5, "No error details", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Frequency")
    if has_data:
        ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return write_plot_metadata(output, key=output.stem, title=title, description=description)


def plot_training_curves(
    per_seed_runs: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str,
    description: str,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not MATPLOTLIB_AVAILABLE:
        write_placeholder_png(output)
        return write_plot_metadata(output, key=output.stem, title=title, description=description)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    has_data = False
    for run in per_seed_runs:
        history = run.get("history") or {}
        train_losses = [float(item.get("loss", 0.0)) for item in history.get("train", []) if isinstance(item.get("loss"), (int, float))]
        val_losses = [float(item.get("loss", 0.0)) for item in history.get("val", []) if isinstance(item.get("loss"), (int, float))]
        if not train_losses:
            continue
        has_data = True
        epochs = np.arange(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label=f"train seed {run.get('seed')}", alpha=0.6)
        if val_losses:
            ax.plot(epochs[: len(val_losses)], val_losses, linestyle="--", label=f"val seed {run.get('seed')}", alpha=0.6)
    if not has_data:
        ax.text(0.5, 0.5, "No training history", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    if has_data:
        ax.legend(ncol=2, fontsize=8)
    ax.grid(linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return write_plot_metadata(output, key=output.stem, title=title, description=description)


def plot_split_overview(split_snapshot: dict[str, Any], output_path: str | Path, *, title: str, description: str) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not MATPLOTLIB_AVAILABLE:
        write_placeholder_png(output)
        return write_plot_metadata(output, key=output.stem, title=title, description=description, extra={"split_snapshot": split_snapshot})
    counts = {
        "Train": len(split_snapshot.get("train_batteries", [])),
        "Val": len(split_snapshot.get("val_batteries", [])),
        "Test": len(split_snapshot.get("test_batteries", [])),
    }
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.bar(list(counts.keys()), list(counts.values()), color=["#2ca02c", "#ff7f0e", "#1f77b4"])
    ax.set_title(title)
    ax.set_ylabel("Battery count")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return write_plot_metadata(output, key=output.stem, title=title, description=description, extra={"split_snapshot": split_snapshot})


def plot_source_comparison(
    source: str,
    model_summaries: dict[str, dict[str, Any]],
    output_path: str | Path,
    *,
    title: str,
    description: str,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not MATPLOTLIB_AVAILABLE:
        write_placeholder_png(output)
        return write_plot_metadata(output, key=output.stem, title=title, description=description)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels = list(PLOT_METRIC_KEYS)
    positions = np.arange(len(labels))
    width = 0.34
    for offset, (model_type, summary) in zip((-width / 2, width / 2), model_summaries.items()):
        mean = ((summary.get("aggregate_metrics") or {}).get("mean")) or {}
        values = [float(mean.get(key) or 0.0) for key in labels]
        ax.bar(positions + offset, values, width=width, label=model_type)
    ax.set_xticks(positions)
    ax.set_xticklabels([item.upper() for item in labels])
    ax.set_title(title or f"{source.upper()} experiment comparison")
    ax.set_ylabel("Metric value")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return write_plot_metadata(output, key=output.stem, title=title, description=description)


def plot_ablation_overview(
    variants: list[dict[str, Any]],
    output_path: str | Path,
    *,
    title: str,
    description: str,
) -> dict[str, Any]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not MATPLOTLIB_AVAILABLE:
        write_placeholder_png(output)
        return write_plot_metadata(output, key=output.stem, title=title, description=description)
    labels = [item.get("label", item.get("key", "--")) for item in variants]
    rmse_values = [float((((item.get("aggregate_metrics") or {}).get("mean")) or {}).get("rmse") or 0.0) for item in variants]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(labels, rmse_values, color="#4c78a8")
    ax.set_title(title)
    ax.set_ylabel("RMSE")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return write_plot_metadata(output, key=output.stem, title=title, description=description)


def collect_plot_metadata(directory: str | Path) -> list[dict[str, Any]]:
    root = Path(directory)
    if not root.exists():
        return []
    items: list[dict[str, Any]] = []
    for metadata_path in sorted(root.glob("*.json")):
        if metadata_path.name == "plot_manifest.json":
            continue
        try:
            items.append(json.loads(metadata_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return items


def write_plot_manifest(directory: str | Path) -> list[dict[str, Any]]:
    root = Path(directory)
    plots = collect_plot_metadata(root)
    write_json(root / "plot_manifest.json", {"plots": plots, "generated_at": datetime.utcnow().isoformat()})
    return plots


__all__ = [
    "AGGREGATE_METRIC_KEYS",
    "PLOT_METRIC_KEYS",
    "aggregate_metrics",
    "collect_plot_metadata",
    "write_placeholder_png",
    "plot_ablation_overview",
    "plot_error_distribution",
    "plot_metric_summary",
    "plot_source_comparison",
    "plot_split_overview",
    "plot_training_curves",
    "select_best_run",
    "write_json",
    "write_plot_manifest",
]
