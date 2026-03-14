"""应用配置。"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os


@dataclass(frozen=True)
class Settings:
    app_name: str
    api_prefix: str
    project_root: Path
    data_dir: Path
    raw_nasa_dir: Path
    raw_calce_dir: Path
    raw_kaggle_dir: Path
    processed_dir: Path
    knowledge_path: Path
    model_dir: Path
    upload_dir: Path
    database_path: Path
    default_seq_len: int
    default_page_size: int
    battery_eol_ratio: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"
    return Settings(
        app_name="BHMS - 锂电池健康管理系统",
        api_prefix="/api/v1",
        project_root=project_root,
        data_dir=data_dir,
        raw_nasa_dir=data_dir / "raw" / "nasa",
        raw_calce_dir=data_dir / "raw" / "calce",
        raw_kaggle_dir=data_dir / "raw" / "kaggle",
        processed_dir=data_dir / "processed",
        knowledge_path=data_dir / "knowledge" / "battery_fault_knowledge.json",
        model_dir=data_dir / "models",
        upload_dir=data_dir / "uploads",
        database_path=project_root / os.getenv("BHMS_DB_PATH", "data/bhms.db"),
        default_seq_len=int(os.getenv("BHMS_DEFAULT_SEQ_LEN", "30")),
        default_page_size=int(os.getenv("BHMS_DEFAULT_PAGE_SIZE", "10")),
        battery_eol_ratio=float(os.getenv("BHMS_EOL_RATIO", "0.8")),
    )
