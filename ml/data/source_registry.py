"""Dataset registry and source metadata defaults for BHMS battery datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class DatasetCard:
    source: str
    dataset_name: str
    adapter_kind: str
    group: str
    description: str
    metadata_defaults: dict[str, Any] = field(default_factory=dict)


SOURCE_REGISTRY: dict[str, DatasetCard] = {
    "nasa": DatasetCard(
        source="nasa",
        dataset_name="nasa_pcoe",
        adapter_kind="nasa",
        group="main_lifecycle",
        description="NASA PCoE lithium-ion battery aging benchmark.",
        metadata_defaults={
            "chemistry": "Li-ion",
            "form_factor": "18650 cylindrical",
            "protocol_id": "nasa_pcoe_standard",
            "charge_c_rate": 1.5,
            "discharge_c_rate": 2.0,
            "ambient_temp": 24.0,
            "dataset_license": "NASA open data",
        },
    ),
    "calce": DatasetCard(
        source="calce",
        dataset_name="calce_demo",
        adapter_kind="csv",
        group="main_lifecycle",
        description="CALCE lithium-ion cycle aging dataset.",
        metadata_defaults={
            "chemistry": "Li-ion",
            "form_factor": "pouch",
            "protocol_id": "calce_cycle_aging",
            "charge_c_rate": 1.0,
            "discharge_c_rate": 1.0,
            "ambient_temp": 25.0,
            "dataset_license": "CALCE academic use",
        },
    ),
    "kaggle": DatasetCard(
        source="kaggle",
        dataset_name="kaggle_demo",
        adapter_kind="csv",
        group="demo",
        description="Lightweight CSV demo dataset used for local validation.",
        metadata_defaults={
            "chemistry": "Li-ion",
            "form_factor": "unknown",
            "protocol_id": "kaggle_demo_protocol",
            "charge_c_rate": 1.0,
            "discharge_c_rate": 1.0,
            "ambient_temp": 25.0,
            "dataset_license": "Platform dependent",
        },
    ),
    "hust": DatasetCard(
        source="hust",
        dataset_name="hust_77cell",
        adapter_kind="csv",
        group="main_lifecycle",
        description="HUST 77-cell benchmark for lithium-ion lifecycle prediction.",
        metadata_defaults={
            "chemistry": "Li-ion (NCM)",
            "form_factor": "18650 cylindrical",
            "protocol_id": "hust_77cell_cycle_aging",
            "charge_c_rate": 1.5,
            "discharge_c_rate": 1.0,
            "ambient_temp": 25.0,
            "dataset_license": "Mendeley Data CC BY 4.0",
        },
    ),
    "matr": DatasetCard(
        source="matr",
        dataset_name="matr_severson",
        adapter_kind="csv",
        group="main_lifecycle",
        description="Severson/MATR fast-charging cycle life dataset.",
        metadata_defaults={
            "chemistry": "Li-ion (LFP/graphite)",
            "form_factor": "coin/cylindrical lab cell",
            "protocol_id": "matr_fast_charge",
            "charge_c_rate": 4.0,
            "discharge_c_rate": 1.0,
            "ambient_temp": 30.0,
            "dataset_license": "Published academic dataset",
        },
    ),
    "oxford": DatasetCard(
        source="oxford",
        dataset_name="oxford_degradation_1",
        adapter_kind="csv",
        group="trajectory_enhancement",
        description="Oxford Battery Degradation Dataset 1 for trajectory supervision.",
        metadata_defaults={
            "chemistry": "Li-ion",
            "form_factor": "pouch",
            "protocol_id": "oxford_degradation_1",
            "charge_c_rate": 1.0,
            "discharge_c_rate": 1.0,
            "ambient_temp": 25.0,
            "dataset_license": "Oxford ORA repository",
        },
    ),
    "pulsebat": DatasetCard(
        source="pulsebat",
        dataset_name="pulsebat",
        adapter_kind="csv",
        group="diagnostic_enhancement",
        description="Pulse-driven battery diagnostics dataset for mechanism evidence augmentation.",
        metadata_defaults={
            "chemistry": "Li-ion",
            "form_factor": "pouch",
            "protocol_id": "pulsebat_diagnostic_protocol",
            "charge_c_rate": 1.0,
            "discharge_c_rate": 1.0,
            "ambient_temp": 25.0,
            "dataset_license": "See PulseBat release terms",
        },
    ),
}


def get_dataset_card(source: str) -> DatasetCard:
    key = source.lower()
    try:
        return SOURCE_REGISTRY[key]
    except KeyError as exc:
        raise KeyError(f"Unsupported dataset source: {source}") from exc


def list_supported_sources() -> list[str]:
    return sorted(SOURCE_REGISTRY)


__all__ = ["DatasetCard", "SOURCE_REGISTRY", "get_dataset_card", "list_supported_sources"]
