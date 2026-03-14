"""NASA adapter wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ml.data.nasa_preprocessor import NASABatteryPreprocessor
from .base import BaseBatteryAdapter


class NASAAdapter(BaseBatteryAdapter):
    source_name = "nasa"

    def __init__(self, eol_capacity_ratio: float = 0.8):
        super().__init__(eol_capacity_ratio=eol_capacity_ratio)
        self.preprocessor = NASABatteryPreprocessor(eol_capacity_ratio=eol_capacity_ratio)

    def process_directory(
        self,
        input_dir: str | Path,
        output_path: str | Path | None = None,
        battery_ids: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        return self.preprocessor.process_directory(input_dir, output_path=output_path, battery_ids=battery_ids)

    def process_file(self, file_path: str | Path, battery_id_hint: str | None = None) -> pd.DataFrame:
        _ = battery_id_hint
        return self.preprocessor.parse_battery_file(Path(file_path))
