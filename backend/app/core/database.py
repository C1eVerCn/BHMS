"""SQLite 持久化管理。"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from backend.app.core.config import get_settings


BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS batteries (
    battery_id TEXT PRIMARY KEY,
    canonical_battery_id TEXT,
    source TEXT NOT NULL,
    dataset_name TEXT,
    source_battery_id TEXT,
    chemistry TEXT,
    form_factor TEXT,
    protocol_id TEXT,
    charge_c_rate REAL,
    discharge_c_rate REAL,
    ambient_temp REAL,
    nominal_capacity REAL,
    eol_ratio REAL,
    dataset_license TEXT,
    cycle_count INTEGER DEFAULT 0,
    latest_capacity REAL,
    initial_capacity REAL,
    health_score REAL,
    status TEXT,
    last_update TEXT,
    dataset_path TEXT,
    include_in_training INTEGER DEFAULT 0,
    metadata_json TEXT
);

CREATE TABLE IF NOT EXISTS cycle_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    battery_id TEXT NOT NULL,
    canonical_battery_id TEXT,
    source TEXT,
    dataset_name TEXT,
    source_battery_id TEXT,
    cycle_number INTEGER NOT NULL,
    timestamp TEXT,
    ambient_temperature REAL,
    voltage_mean REAL,
    voltage_std REAL,
    voltage_min REAL,
    voltage_max REAL,
    current_mean REAL,
    current_std REAL,
    current_load_mean REAL,
    temperature_mean REAL,
    temperature_std REAL,
    temperature_rise_rate REAL,
    internal_resistance REAL,
    capacity REAL,
    source_type TEXT,
    UNIQUE(battery_id, cycle_number)
);
CREATE INDEX IF NOT EXISTS idx_cycle_points_battery_cycle ON cycle_points(battery_id, cycle_number);

CREATE TABLE IF NOT EXISTS dataset_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    battery_id TEXT,
    source TEXT,
    dataset_name TEXT,
    file_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    row_count INTEGER DEFAULT 0,
    include_in_training INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    validation_summary_json TEXT
);

CREATE TABLE IF NOT EXISTS prediction_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    battery_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    predicted_rul REAL NOT NULL,
    confidence REAL NOT NULL,
    input_seq_len INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    payload_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_prediction_battery ON prediction_records(battery_id, created_at DESC);

CREATE TABLE IF NOT EXISTS anomaly_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    battery_id TEXT NOT NULL,
    code TEXT NOT NULL,
    symptom TEXT NOT NULL,
    severity TEXT NOT NULL,
    metric_name TEXT,
    metric_value REAL,
    threshold_value TEXT,
    description TEXT,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metadata_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_anomaly_battery ON anomaly_events(battery_id, created_at DESC);

CREATE TABLE IF NOT EXISTS diagnosis_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    battery_id TEXT NOT NULL,
    fault_type TEXT NOT NULL,
    confidence REAL NOT NULL,
    severity TEXT NOT NULL,
    description TEXT,
    root_causes_json TEXT,
    recommendations_json TEXT,
    related_symptoms_json TEXT,
    evidence_json TEXT,
    payload_json TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_diagnosis_battery ON diagnosis_records(battery_id, created_at DESC);

CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    model_type TEXT NOT NULL,
    model_version TEXT,
    best_checkpoint_path TEXT,
    final_checkpoint_path TEXT,
    metrics_json TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_training_runs_source ON training_runs(source, model_type, created_at DESC);

CREATE TABLE IF NOT EXISTS training_jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    model_scope TEXT NOT NULL,
    status TEXT NOT NULL,
    current_stage TEXT,
    force_run INTEGER DEFAULT 0,
    baseline_json TEXT,
    result_json TEXT,
    log_excerpt TEXT,
    error_message TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_training_jobs_source ON training_jobs(source, created_at DESC);
"""

MIGRATIONS = {
    "batteries": {
        "canonical_battery_id": "ALTER TABLE batteries ADD COLUMN canonical_battery_id TEXT",
        "dataset_name": "ALTER TABLE batteries ADD COLUMN dataset_name TEXT",
        "source_battery_id": "ALTER TABLE batteries ADD COLUMN source_battery_id TEXT",
        "form_factor": "ALTER TABLE batteries ADD COLUMN form_factor TEXT",
        "protocol_id": "ALTER TABLE batteries ADD COLUMN protocol_id TEXT",
        "charge_c_rate": "ALTER TABLE batteries ADD COLUMN charge_c_rate REAL",
        "discharge_c_rate": "ALTER TABLE batteries ADD COLUMN discharge_c_rate REAL",
        "ambient_temp": "ALTER TABLE batteries ADD COLUMN ambient_temp REAL",
        "eol_ratio": "ALTER TABLE batteries ADD COLUMN eol_ratio REAL",
        "dataset_license": "ALTER TABLE batteries ADD COLUMN dataset_license TEXT",
        "include_in_training": "ALTER TABLE batteries ADD COLUMN include_in_training INTEGER DEFAULT 0",
    },
    "cycle_points": {
        "canonical_battery_id": "ALTER TABLE cycle_points ADD COLUMN canonical_battery_id TEXT",
        "source": "ALTER TABLE cycle_points ADD COLUMN source TEXT",
        "dataset_name": "ALTER TABLE cycle_points ADD COLUMN dataset_name TEXT",
        "source_battery_id": "ALTER TABLE cycle_points ADD COLUMN source_battery_id TEXT",
    },
    "dataset_files": {
        "source": "ALTER TABLE dataset_files ADD COLUMN source TEXT",
        "dataset_name": "ALTER TABLE dataset_files ADD COLUMN dataset_name TEXT",
        "include_in_training": "ALTER TABLE dataset_files ADD COLUMN include_in_training INTEGER DEFAULT 0",
    },
    "diagnosis_records": {
        "payload_json": "ALTER TABLE diagnosis_records ADD COLUMN payload_json TEXT",
    },
    "training_jobs": {},
}


class DatabaseManager:
    def __init__(self, database_path: str | Path | None = None):
        settings = get_settings()
        self.database_path = Path(database_path or settings.database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connection() as connection:
            connection.executescript(BASE_SCHEMA)
            self._apply_migrations(connection)
            connection.execute("CREATE INDEX IF NOT EXISTS idx_cycle_points_source ON cycle_points(source, dataset_name)")

    @staticmethod
    def _apply_migrations(connection: sqlite3.Connection) -> None:
        for table, columns in MIGRATIONS.items():
            existing = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
            for column, statement in columns.items():
                if column not in existing:
                    connection.execute(statement)


_db_manager: DatabaseManager | None = None


def get_database() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager
