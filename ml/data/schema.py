"""Shared battery data schema definitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

CANONICAL_METADATA_COLUMNS = [
    "battery_id",
    "canonical_battery_id",
    "source",
    "dataset_name",
    "source_battery_id",
    "cycle_number",
    "timestamp",
    "ambient_temperature",
    "source_type",
]

BATTERY_FEATURE_COLUMNS = [
    "voltage_mean",
    "voltage_std",
    "voltage_min",
    "voltage_max",
    "current_mean",
    "current_std",
    "current_load_mean",
    "temperature_mean",
    "temperature_std",
    "temperature_rise_rate",
    "internal_resistance",
    "capacity",
]

BATTERY_TARGET_COLUMNS = [
    "initial_capacity",
    "capacity_ratio",
    "eol_cycle",
    "RUL",
    "health_score",
    "status",
]

BATTERY_SCHEMA_COLUMNS = CANONICAL_METADATA_COLUMNS + BATTERY_FEATURE_COLUMNS + BATTERY_TARGET_COLUMNS
TRAINING_FEATURE_COLUMNS = [
    "voltage_mean",
    "voltage_std",
    "voltage_min",
    "voltage_max",
    "current_mean",
    "current_std",
    "temperature_mean",
    "temperature_std",
    "temperature_rise_rate",
    "capacity",
    "cycle_number",
]
REQUIRED_CYCLE_COLUMNS = {"cycle_number", "voltage_mean", "current_mean", "temperature_mean", "capacity"}
NUMERIC_COLUMNS = [
    "ambient_temperature",
    "voltage_mean",
    "voltage_std",
    "voltage_min",
    "voltage_max",
    "current_mean",
    "current_std",
    "current_load_mean",
    "temperature_mean",
    "temperature_std",
    "temperature_rise_rate",
    "internal_resistance",
    "capacity",
    "initial_capacity",
    "capacity_ratio",
    "eol_cycle",
    "RUL",
    "health_score",
]


@dataclass(slots=True)
class DatasetSplit:
    train_batteries: list[str]
    val_batteries: list[str]
    test_batteries: list[str]

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "train_batteries": self.train_batteries,
            "val_batteries": self.val_batteries,
            "test_batteries": self.test_batteries,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, list[str]]) -> "DatasetSplit":
        return cls(
            train_batteries=list(payload.get("train_batteries", [])),
            val_batteries=list(payload.get("val_batteries", [])),
            test_batteries=list(payload.get("test_batteries", [])),
        )


def build_canonical_battery_id(source: str, dataset_name: str, source_battery_id: str) -> str:
    return f"{source.lower()}::{dataset_name.lower()}::{source_battery_id}"


def health_status(score: float) -> str:
    if score >= 85:
        return "good"
    if score >= 70:
        return "warning"
    return "critical"


def finalize_cycle_frame(
    frame: pd.DataFrame,
    *,
    source: str,
    dataset_name: str,
    eol_capacity_ratio: float,
) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]

    for column in REQUIRED_CYCLE_COLUMNS:
        if column not in normalized.columns:
            raise ValueError(f"缺少必需列: {column}")

    if "source_battery_id" not in normalized.columns:
        if "battery_id" in normalized.columns:
            normalized["source_battery_id"] = normalized["battery_id"].astype(str)
        else:
            raise ValueError("缺少 source_battery_id/battery_id 列")
    normalized["source_battery_id"] = normalized["source_battery_id"].astype(str)
    normalized["source"] = source.lower()
    normalized["dataset_name"] = dataset_name
    normalized["canonical_battery_id"] = normalized["source_battery_id"].map(
        lambda battery_id: build_canonical_battery_id(source, dataset_name, str(battery_id))
    )
    normalized["battery_id"] = normalized["canonical_battery_id"]

    defaults = {
        "timestamp": None,
        "ambient_temperature": 0.0,
        "voltage_std": 0.0,
        "voltage_min": 0.0,
        "voltage_max": 0.0,
        "current_std": 0.0,
        "current_load_mean": 0.0,
        "temperature_std": 0.0,
        "temperature_rise_rate": 0.0,
        "internal_resistance": 0.0,
        "source_type": f"{source.lower()}_cycle",
    }
    for column, default in defaults.items():
        if column not in normalized.columns:
            normalized[column] = default

    result = enrich_existing_cycle_frame(normalized, eol_capacity_ratio=eol_capacity_ratio)
    for column in BATTERY_SCHEMA_COLUMNS:
        if column not in result.columns:
            result[column] = 0.0 if column in NUMERIC_COLUMNS else None
    result = result[BATTERY_SCHEMA_COLUMNS]
    return result


def enrich_existing_cycle_frame(frame: pd.DataFrame, *, eol_capacity_ratio: float) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.columns = [str(column).strip() for column in normalized.columns]
    for column in NUMERIC_COLUMNS:
        if column in normalized.columns:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    normalized["cycle_number"] = normalized["cycle_number"].fillna(0).astype(int)
    normalized.sort_values(["canonical_battery_id", "cycle_number"], inplace=True)

    frames: list[pd.DataFrame] = []
    for canonical_battery_id, group in normalized.groupby("canonical_battery_id"):
        group = group.reset_index(drop=True).copy()
        initial_capacity = float(group["capacity"].dropna().iloc[0])
        group["initial_capacity"] = initial_capacity
        group["capacity_ratio"] = group["capacity"] / max(initial_capacity, 1e-6)
        eol_candidates = group.loc[group["capacity_ratio"] <= eol_capacity_ratio, "cycle_number"]
        eol_cycle = int(eol_candidates.iloc[0]) if not eol_candidates.empty else int(group["cycle_number"].max())
        group["eol_cycle"] = eol_cycle
        group["RUL"] = (eol_cycle - group["cycle_number"]).clip(lower=0)
        group["health_score"] = (group["capacity_ratio"] * 100).clip(lower=0, upper=100)
        group["status"] = group["health_score"].map(health_status)
        group["battery_id"] = canonical_battery_id
        frames.append(group)

    return pd.concat(frames, ignore_index=True)


def write_json(path: str | Path, payload: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "BATTERY_FEATURE_COLUMNS",
    "BATTERY_SCHEMA_COLUMNS",
    "CANONICAL_METADATA_COLUMNS",
    "DatasetSplit",
    "NUMERIC_COLUMNS",
    "REQUIRED_CYCLE_COLUMNS",
    "TRAINING_FEATURE_COLUMNS",
    "BATTERY_TARGET_COLUMNS",
    "build_canonical_battery_id",
    "enrich_existing_cycle_frame",
    "finalize_cycle_frame",
    "health_status",
]
