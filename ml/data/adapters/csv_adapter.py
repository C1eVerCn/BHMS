"""CSV adapters for CALCE and Kaggle style datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ml.data.schema import finalize_cycle_frame
from ml.data.source_registry import get_dataset_card
from .base import BaseBatteryAdapter

COMMON_COLUMN_ALIASES = {
    "battery": "source_battery_id",
    "battery_id": "source_battery_id",
    "cell_id": "source_battery_id",
    "cell_name": "source_battery_id",
    "cycle": "cycle_number",
    "cycle_id": "cycle_number",
    "cycle_index": "cycle_number",
    "voltage": "voltage_mean",
    "current": "current_mean",
    "temperature": "temperature_mean",
    "temp": "temperature_mean",
    "resistance": "internal_resistance",
    "ir": "internal_resistance",
    "capacity_ah": "capacity",
    "discharge_capacity": "capacity",
    "discharge_capacity_ah": "capacity",
    "q_discharge": "capacity",
    "ambient_temp": "ambient_temp",
}


class GenericCSVAdapter(BaseBatteryAdapter):
    source_name = "csv"
    dataset_name = "uploaded"

    def __init__(
        self,
        source_name: str,
        dataset_name: str,
        eol_capacity_ratio: float = 0.8,
        metadata_defaults: Optional[dict[str, object]] = None,
        column_aliases: Optional[dict[str, str]] = None,
    ):
        super().__init__(eol_capacity_ratio=eol_capacity_ratio)
        self.source_name = source_name
        self.dataset_name = dataset_name
        self.metadata_defaults = dict(metadata_defaults or {})
        self.column_aliases = {**COMMON_COLUMN_ALIASES, **(column_aliases or {})}

    def _normalize_columns(self, frame: pd.DataFrame) -> pd.DataFrame:
        renamed = frame.copy()
        rename_map: dict[str, str] = {}
        for column in renamed.columns:
            key = str(column).strip()
            alias = self.column_aliases.get(key.lower())
            if alias and alias not in renamed.columns:
                rename_map[column] = alias
        if rename_map:
            renamed = renamed.rename(columns=rename_map)
        return renamed

    def _load_csv(self, file_path: str | Path, battery_id_hint: str | None = None) -> pd.DataFrame:
        frame = self._normalize_columns(pd.read_csv(file_path))
        frame.columns = [str(column).strip() for column in frame.columns]
        if battery_id_hint and "battery_id" not in frame.columns and "source_battery_id" not in frame.columns:
            frame["source_battery_id"] = battery_id_hint
        return finalize_cycle_frame(
            frame,
            source=self.source_name,
            dataset_name=self.dataset_name,
            eol_capacity_ratio=self.eol_capacity_ratio,
            metadata_defaults=self.metadata_defaults,
        )

    def process_file(self, file_path: str | Path, battery_id_hint: str | None = None) -> pd.DataFrame:
        return self._load_csv(file_path, battery_id_hint=battery_id_hint)

    def process_directory(
        self,
        input_dir: str | Path,
        output_path: str | Path | None = None,
        battery_ids: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        directory = Path(input_dir)
        frames: list[pd.DataFrame] = []
        selected = {battery_id.upper() for battery_id in battery_ids} if battery_ids else None
        for csv_file in sorted(directory.glob("*.csv")):
            frame = self._load_csv(csv_file)
            if selected is not None:
                frame = frame[frame["source_battery_id"].str.upper().isin(selected)]
                if frame.empty:
                    continue
            frames.append(frame)
        if not frames:
            raise ValueError(f"在 {directory} 中未找到可导入的 CSV 文件")
        combined = pd.concat(frames, ignore_index=True)
        if output_path is not None:
            destination = Path(output_path)
            destination.parent.mkdir(parents=True, exist_ok=True)
            combined.to_csv(destination, index=False)
        return combined


class CALCEAdapter(GenericCSVAdapter):
    def __init__(self, eol_capacity_ratio: float = 0.8):
        card = get_dataset_card("calce")
        super().__init__(
            source_name=card.source,
            dataset_name=card.dataset_name,
            eol_capacity_ratio=eol_capacity_ratio,
            metadata_defaults=card.metadata_defaults,
        )


class KaggleAdapter(GenericCSVAdapter):
    def __init__(self, eol_capacity_ratio: float = 0.8):
        card = get_dataset_card("kaggle")
        super().__init__(
            source_name=card.source,
            dataset_name=card.dataset_name,
            eol_capacity_ratio=eol_capacity_ratio,
            metadata_defaults=card.metadata_defaults,
        )


class HUSTAdapter(GenericCSVAdapter):
    def __init__(self, eol_capacity_ratio: float = 0.8):
        card = get_dataset_card("hust")
        super().__init__(
            source_name=card.source,
            dataset_name=card.dataset_name,
            eol_capacity_ratio=eol_capacity_ratio,
            metadata_defaults=card.metadata_defaults,
        )


class MATRAdapter(GenericCSVAdapter):
    def __init__(self, eol_capacity_ratio: float = 0.8):
        card = get_dataset_card("matr")
        super().__init__(
            source_name=card.source,
            dataset_name=card.dataset_name,
            eol_capacity_ratio=eol_capacity_ratio,
            metadata_defaults=card.metadata_defaults,
        )


class OxfordAdapter(GenericCSVAdapter):
    def __init__(self, eol_capacity_ratio: float = 0.8):
        card = get_dataset_card("oxford")
        super().__init__(
            source_name=card.source,
            dataset_name=card.dataset_name,
            eol_capacity_ratio=eol_capacity_ratio,
            metadata_defaults=card.metadata_defaults,
        )


class PulseBatAdapter(GenericCSVAdapter):
    def __init__(self, eol_capacity_ratio: float = 0.8):
        card = get_dataset_card("pulsebat")
        super().__init__(
            source_name=card.source,
            dataset_name=card.dataset_name,
            eol_capacity_ratio=eol_capacity_ratio,
            metadata_defaults=card.metadata_defaults,
        )
