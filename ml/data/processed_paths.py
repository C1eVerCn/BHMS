"""Helpers for locating and cleaning processed cycle summary files."""

from __future__ import annotations

from pathlib import Path

from ml.data.source_registry import get_dataset_card


def use_compressed_cycle_summary(source: str) -> bool:
    card = get_dataset_card(source)
    return card.ingestion_mode == "raw_converter" and card.training_ready


def cycle_summary_filename(source: str) -> str:
    base_name = f"{source.lower()}_cycle_summary.csv"
    if use_compressed_cycle_summary(source):
        return f"{base_name}.gz"
    return base_name


def cycle_summary_path(source: str, output_dir: str | Path) -> Path:
    return Path(output_dir) / cycle_summary_filename(source)


def resolve_cycle_summary_path(path: str | Path, source: str | None = None) -> Path:
    candidate = Path(path)
    if candidate.exists():
        return candidate

    variants: list[Path] = []
    if source:
        variants.extend(cycle_summary_variants(source, candidate.parent))

    candidate_text = str(candidate)
    if candidate_text.endswith(".csv.gz"):
        variants.append(Path(candidate_text[:-3]))
    elif candidate_text.endswith(".csv"):
        variants.append(Path(f"{candidate_text}.gz"))

    for variant in variants:
        if variant != candidate and variant.exists():
            return variant
    return candidate


def cycle_summary_variants(source: str, output_dir: str | Path) -> tuple[Path, Path]:
    base = Path(output_dir) / f"{source.lower()}_cycle_summary.csv"
    return base, Path(f"{base}.gz")


def cleanup_cycle_summary_variants(source: str, output_dir: str | Path, keep_path: str | Path) -> None:
    keep = Path(keep_path)
    for candidate in cycle_summary_variants(source, output_dir):
        if candidate == keep or not candidate.exists():
            continue
        candidate.unlink()

