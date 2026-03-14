"""Base adapter interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


class BaseBatteryAdapter(ABC):
    source_name: str

    def __init__(self, eol_capacity_ratio: float = 0.8):
        self.eol_capacity_ratio = eol_capacity_ratio

    @abstractmethod
    def process_directory(
        self,
        input_dir: str | Path,
        output_path: str | Path | None = None,
        battery_ids: Optional[Iterable[str]] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def process_file(self, file_path: str | Path, battery_id_hint: str | None = None) -> pd.DataFrame:
        raise NotImplementedError
