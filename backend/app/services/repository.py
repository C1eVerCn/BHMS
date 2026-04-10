"""SQLite 访问层。"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional

from backend.app.core.database import DatabaseManager, get_database


class BHMSRepository:
    def __init__(self, database: Optional[DatabaseManager] = None):
        self.database = database or get_database()

    @staticmethod
    def _now() -> str:
        return datetime.utcnow().isoformat()

    @staticmethod
    def _decode_json(value: Optional[str], default: Any) -> Any:
        if not value:
            return default
        return json.loads(value)

    def count_batteries(self) -> int:
        with self.database.connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM batteries").fetchone()
            return int(row["count"])

    def count_batteries_by_source(self, source: str) -> int:
        with self.database.connection() as connection:
            row = connection.execute("SELECT COUNT(*) AS count FROM batteries WHERE source = ?", (source,)).fetchone()
            return int(row["count"])

    def count_canonical_batteries_by_source(self, source: str) -> int:
        with self.database.connection() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM batteries WHERE LOWER(source) = ? AND battery_id LIKE ?",
                (source.lower(), f"{source.lower()}::%"),
            ).fetchone()
            return int(row["count"])

    def delete_batteries_by_source(self, source: str) -> None:
        with self.database.connection() as connection:
            battery_rows = connection.execute("SELECT battery_id FROM batteries WHERE LOWER(source) = ?", (source.lower(),)).fetchall()
            battery_ids = [row["battery_id"] for row in battery_rows]
            for battery_id in battery_ids:
                connection.execute("DELETE FROM cycle_points WHERE battery_id = ?", (battery_id,))
                connection.execute("DELETE FROM prediction_records WHERE battery_id = ?", (battery_id,))
                connection.execute("DELETE FROM anomaly_events WHERE battery_id = ?", (battery_id,))
                connection.execute("DELETE FROM diagnosis_records WHERE battery_id = ?", (battery_id,))
            connection.execute("DELETE FROM batteries WHERE LOWER(source) = ?", (source.lower(),))

    def upsert_battery(self, battery: dict[str, Any]) -> None:
        with self.database.connection() as connection:
            connection.execute(
                """
                INSERT INTO batteries (
                    battery_id, canonical_battery_id, source, dataset_name, source_battery_id,
                    chemistry, form_factor, protocol_id, charge_c_rate, discharge_c_rate,
                    ambient_temp, nominal_capacity, eol_ratio, dataset_license,
                    cycle_count, latest_capacity, initial_capacity,
                    health_score, status, last_update, dataset_path, include_in_training, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(battery_id) DO UPDATE SET
                    canonical_battery_id=excluded.canonical_battery_id,
                    source=excluded.source,
                    dataset_name=excluded.dataset_name,
                    source_battery_id=excluded.source_battery_id,
                    chemistry=excluded.chemistry,
                    form_factor=excluded.form_factor,
                    protocol_id=excluded.protocol_id,
                    charge_c_rate=excluded.charge_c_rate,
                    discharge_c_rate=excluded.discharge_c_rate,
                    ambient_temp=excluded.ambient_temp,
                    nominal_capacity=excluded.nominal_capacity,
                    eol_ratio=excluded.eol_ratio,
                    dataset_license=excluded.dataset_license,
                    cycle_count=excluded.cycle_count,
                    latest_capacity=excluded.latest_capacity,
                    initial_capacity=excluded.initial_capacity,
                    health_score=excluded.health_score,
                    status=excluded.status,
                    last_update=excluded.last_update,
                    dataset_path=excluded.dataset_path,
                    include_in_training=excluded.include_in_training,
                    metadata_json=excluded.metadata_json
                """,
                (
                    battery["battery_id"],
                    battery.get("canonical_battery_id", battery["battery_id"]),
                    battery["source"],
                    battery.get("dataset_name"),
                    battery.get("source_battery_id"),
                    battery.get("chemistry"),
                    battery.get("form_factor"),
                    battery.get("protocol_id"),
                    battery.get("charge_c_rate"),
                    battery.get("discharge_c_rate"),
                    battery.get("ambient_temp"),
                    battery.get("nominal_capacity"),
                    battery.get("eol_ratio"),
                    battery.get("dataset_license"),
                    battery.get("cycle_count", 0),
                    battery.get("latest_capacity"),
                    battery.get("initial_capacity"),
                    battery.get("health_score", 0.0),
                    battery.get("status", "unknown"),
                    battery.get("last_update"),
                    battery.get("dataset_path"),
                    int(bool(battery.get("include_in_training", False))),
                    json.dumps(battery.get("metadata", {}), ensure_ascii=False),
                ),
            )

    def set_battery_training_flag(self, battery_id: str, include_in_training: bool) -> None:
        with self.database.connection() as connection:
            connection.execute(
                "UPDATE batteries SET include_in_training = ? WHERE battery_id = ?",
                (int(bool(include_in_training)), battery_id),
            )
            connection.execute(
                "UPDATE dataset_files SET include_in_training = ? WHERE instr(COALESCE(battery_id, ''), ?) > 0",
                (int(bool(include_in_training)), battery_id),
            )

    def replace_cycle_points(self, battery_id: str, points: Iterable[dict[str, Any]]) -> int:
        points = list(points)
        with self.database.connection() as connection:
            connection.execute("DELETE FROM cycle_points WHERE battery_id = ?", (battery_id,))
            for point in points:
                connection.execute(
                    """
                    INSERT INTO cycle_points (
                        battery_id, canonical_battery_id, source, dataset_name, source_battery_id,
                        cycle_number, timestamp, ambient_temperature,
                        voltage_mean, voltage_std, voltage_min, voltage_max,
                        current_mean, current_std, current_load_mean,
                        temperature_mean, temperature_std, temperature_rise_rate,
                        internal_resistance, capacity, source_type
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        point["battery_id"],
                        point.get("canonical_battery_id", point["battery_id"]),
                        point.get("source"),
                        point.get("dataset_name"),
                        point.get("source_battery_id"),
                        point["cycle_number"],
                        point.get("timestamp"),
                        point.get("ambient_temperature"),
                        point.get("voltage_mean"),
                        point.get("voltage_std"),
                        point.get("voltage_min"),
                        point.get("voltage_max"),
                        point.get("current_mean"),
                        point.get("current_std"),
                        point.get("current_load_mean"),
                        point.get("temperature_mean"),
                        point.get("temperature_std"),
                        point.get("temperature_rise_rate"),
                        point.get("internal_resistance"),
                        point.get("capacity"),
                        point.get("source_type"),
                    ),
                )
        return len(points)

    def query_training_cycle_points(self, source: str) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT cp.battery_id, cp.canonical_battery_id, cp.source, cp.dataset_name, cp.source_battery_id,
                       cp.cycle_number, cp.timestamp, cp.ambient_temperature,
                       cp.voltage_mean, cp.voltage_std, cp.voltage_min, cp.voltage_max,
                       cp.current_mean, cp.current_std, cp.current_load_mean,
                       cp.temperature_mean, cp.temperature_std, cp.temperature_rise_rate,
                       cp.internal_resistance, cp.capacity, cp.source_type,
                       b.chemistry, b.form_factor, b.protocol_id, b.charge_c_rate,
                       b.discharge_c_rate, b.ambient_temp, b.nominal_capacity,
                       b.dataset_license
                FROM cycle_points cp
                INNER JOIN batteries b ON b.battery_id = cp.battery_id
                WHERE LOWER(b.source) = ? AND b.include_in_training = 1
                ORDER BY cp.battery_id, cp.cycle_number ASC
                """,
                (source.lower(),),
            ).fetchall()
        return [dict(row) for row in rows]

    def insert_dataset_file(self, record: dict[str, Any]) -> int:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO dataset_files (
                    battery_id, source, dataset_name, file_name, file_path,
                    file_type, row_count, include_in_training, created_at, validation_summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.get("battery_id"),
                    record.get("source"),
                    record.get("dataset_name"),
                    record["file_name"],
                    record["file_path"],
                    record["file_type"],
                    record.get("row_count", 0),
                    int(bool(record.get("include_in_training", False))),
                    record.get("created_at", self._now()),
                    json.dumps(record.get("validation_summary", {}), ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def insert_training_run(self, record: dict[str, Any]) -> int:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO training_runs (
                    source, model_type, model_version, best_checkpoint_path, final_checkpoint_path,
                    metrics_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["source"],
                    record["model_type"],
                    record.get("model_version"),
                    record.get("best_checkpoint_path"),
                    record.get("final_checkpoint_path"),
                    json.dumps(record.get("metrics", {}), ensure_ascii=False),
                    json.dumps(record.get("metadata", {}), ensure_ascii=False),
                    record.get("created_at", self._now()),
                ),
            )
            return int(cursor.lastrowid)

    def latest_completed_training_job(self, source: str) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT * FROM training_jobs
                WHERE source = ? AND status = 'completed'
                ORDER BY created_at DESC LIMIT 1
                """,
                (source,),
            ).fetchone()
        return self._decode_training_job(row)

    def list_batteries(self, page: int, page_size: int) -> tuple[list[dict[str, Any]], int]:
        offset = (page - 1) * page_size
        with self.database.connection() as connection:
            total_row = connection.execute("SELECT COUNT(*) AS count FROM batteries").fetchone()
            rows = connection.execute(
                """
                SELECT battery_id, canonical_battery_id, source, dataset_name, source_battery_id,
                       chemistry, form_factor, protocol_id, charge_c_rate, discharge_c_rate,
                       ambient_temp, nominal_capacity, eol_ratio, dataset_license, cycle_count, latest_capacity,
                       initial_capacity, health_score, status, last_update, dataset_path,
                       include_in_training
                FROM batteries
                ORDER BY include_in_training ASC, source ASC, battery_id ASC
                LIMIT ? OFFSET ?
                """,
                (page_size, offset),
            ).fetchall()
        return [dict(row) for row in rows], int(total_row["count"])

    def list_battery_options(self) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT battery_id, source, dataset_name, status, cycle_count, include_in_training
                FROM batteries
                ORDER BY include_in_training ASC, source ASC, battery_id ASC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def get_battery(self, battery_id: str) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT battery_id, canonical_battery_id, source, dataset_name, source_battery_id,
                       chemistry, form_factor, protocol_id, charge_c_rate, discharge_c_rate,
                       ambient_temp, nominal_capacity, eol_ratio, dataset_license, cycle_count, latest_capacity,
                       initial_capacity, health_score, status, last_update, dataset_path,
                       include_in_training
                FROM batteries WHERE battery_id = ?
                """,
                (battery_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_cycle_points(self, battery_id: str, limit: int = 200, descending: bool = False) -> list[dict[str, Any]]:
        order = "DESC" if descending else "ASC"
        with self.database.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT battery_id, canonical_battery_id, source, dataset_name, source_battery_id,
                       cycle_number, timestamp, ambient_temperature,
                       voltage_mean, voltage_std, voltage_min, voltage_max,
                       current_mean, current_std, current_load_mean,
                       temperature_mean, temperature_std, temperature_rise_rate,
                       internal_resistance, capacity, source_type
                FROM cycle_points
                WHERE battery_id = ?
                ORDER BY cycle_number {order}
                LIMIT ?
                """,
                (battery_id, limit),
            ).fetchall()
        items = [dict(row) for row in rows]
        return list(reversed(items)) if descending else items

    def get_latest_cycle_point(self, battery_id: str) -> Optional[dict[str, Any]]:
        items = self.get_cycle_points(battery_id, limit=1, descending=True)
        return items[0] if items else None

    def insert_prediction(self, record: dict[str, Any]) -> int:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO prediction_records (
                    battery_id, model_name, predicted_rul, confidence,
                    input_seq_len, created_at, source, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["battery_id"],
                    record["model_name"],
                    record["predicted_rul"],
                    record["confidence"],
                    record["input_seq_len"],
                    record.get("created_at", self._now()),
                    record.get("source", "api"),
                    json.dumps(record.get("payload", {}), ensure_ascii=False),
                ),
            )
            return int(cursor.lastrowid)

    def get_prediction(self, prediction_id: int) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT id, battery_id, model_name, predicted_rul, confidence, input_seq_len, created_at, source, payload_json
                FROM prediction_records
                WHERE id = ?
                """,
                (prediction_id,),
            ).fetchone()
        return self._decode_prediction(row)

    def list_predictions(self, battery_id: str, limit: int = 10, model_name: Optional[str] = None) -> list[dict[str, Any]]:
        conditions = ["battery_id = ?"]
        params: list[Any] = [battery_id]
        normalized_model = str(model_name or "").strip().lower()
        if normalized_model and normalized_model != "auto":
            conditions.append("LOWER(model_name) = ?")
            params.append(normalized_model)
        with self.database.connection() as connection:
            rows = connection.execute(
                f"""
                SELECT id, battery_id, model_name, predicted_rul, confidence, input_seq_len, created_at, source, payload_json
                FROM prediction_records
                WHERE {' AND '.join(conditions)}
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (*params, limit),
            ).fetchall()
        return [item for item in (self._decode_prediction(row) for row in rows) if item is not None]

    def insert_anomaly_events(self, battery_id: str, events: Iterable[dict[str, Any]]) -> list[int]:
        event_ids: list[int] = []
        with self.database.connection() as connection:
            for event in events:
                cursor = connection.execute(
                    """
                    INSERT INTO anomaly_events (
                        battery_id, code, symptom, severity, metric_name,
                        metric_value, threshold_value, description,
                        source, created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        battery_id,
                        event["code"],
                        event["symptom"],
                        event["severity"],
                        event.get("metric_name"),
                        event.get("metric_value"),
                        event.get("threshold_value"),
                        event.get("description"),
                        event.get("source", "statistical"),
                        self._now(),
                        json.dumps(
                            {
                                "evidence": event.get("evidence", []),
                                "evidence_source": event.get("evidence_source"),
                                "rule_id": event.get("rule_id"),
                                "confidence_basis": event.get("confidence_basis", []),
                                "source_scope": event.get("source_scope", []),
                            },
                            ensure_ascii=False,
                        ),
                    ),
                )
                event_ids.append(int(cursor.lastrowid))
        return event_ids

    def list_anomalies(self, battery_id: str, limit: int = 10) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT code, symptom, severity, metric_name, metric_value, threshold_value, description, source, metadata_json
                FROM anomaly_events
                WHERE battery_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (battery_id, limit),
            ).fetchall()
        result = []
        for row in rows:
            item = dict(row)
            metadata = self._decode_json(item.pop("metadata_json"), {})
            item["evidence"] = metadata.get("evidence", [])
            item["evidence_source"] = metadata.get("evidence_source", "statistical_rules")
            item["rule_id"] = metadata.get("rule_id")
            item["confidence_basis"] = metadata.get("confidence_basis", [])
            item["source_scope"] = metadata.get("source_scope", [])
            result.append(item)
        return result

    def insert_diagnosis(self, record: dict[str, Any]) -> int:
        with self.database.connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO diagnosis_records (
                    battery_id, fault_type, confidence, severity, description,
                    root_causes_json, recommendations_json, related_symptoms_json,
                    evidence_json, payload_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["battery_id"],
                    record["fault_type"],
                    record["confidence"],
                    record["severity"],
                    record.get("description"),
                    json.dumps(record.get("root_causes", []), ensure_ascii=False),
                    json.dumps(record.get("recommendations", []), ensure_ascii=False),
                    json.dumps(record.get("related_symptoms", []), ensure_ascii=False),
                    json.dumps(record.get("evidence", []), ensure_ascii=False),
                    json.dumps(record.get("payload", {}), ensure_ascii=False),
                    record.get("created_at", self._now()),
                ),
            )
            return int(cursor.lastrowid)

    def get_diagnosis(self, diagnosis_id: int) -> Optional[dict[str, Any]]:
        with self.database.connection() as connection:
            row = connection.execute(
                """
                SELECT id, battery_id, fault_type, confidence, severity, description,
                       root_causes_json, recommendations_json, related_symptoms_json,
                       evidence_json, payload_json, created_at
                FROM diagnosis_records
                WHERE id = ?
                """,
                (diagnosis_id,),
            ).fetchone()
        return self._decode_diagnosis(row)

    def list_diagnoses(self, battery_id: str, limit: int = 10) -> list[dict[str, Any]]:
        with self.database.connection() as connection:
            rows = connection.execute(
                """
                SELECT id, battery_id, fault_type, confidence, severity, description,
                       root_causes_json, recommendations_json, related_symptoms_json,
                       evidence_json, payload_json, created_at
                FROM diagnosis_records
                WHERE battery_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (battery_id, limit),
            ).fetchall()
        return [item for item in (self._decode_diagnosis(row) for row in rows) if item is not None]

    def dashboard_summary(self) -> dict[str, Any]:
        with self.database.connection() as connection:
            counts = connection.execute(
                """
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN status = 'good' THEN 1 ELSE 0 END) AS good_count,
                       SUM(CASE WHEN status = 'warning' THEN 1 ELSE 0 END) AS warning_count,
                       SUM(CASE WHEN status = 'critical' THEN 1 ELSE 0 END) AS critical_count,
                       AVG(health_score) AS average_health_score
                FROM batteries
                """
            ).fetchone()
            source_rows = connection.execute(
                """
                SELECT source, COUNT(*) AS battery_count
                FROM batteries
                GROUP BY source
                ORDER BY source
                """
            ).fetchall()
            alerts = connection.execute(
                """
                SELECT battery_id, symptom, severity, description
                FROM anomaly_events
                ORDER BY created_at DESC
                LIMIT 5
                """
            ).fetchall()
            trend_rows = connection.execute(
                """
                SELECT cycle_number, AVG(capacity) AS avg_capacity
                FROM cycle_points
                GROUP BY cycle_number
                ORDER BY cycle_number ASC
                LIMIT 50
                """
            ).fetchall()
        return {
            "total_batteries": int(counts["total"] or 0),
            "good_batteries": int(counts["good_count"] or 0),
            "warning_batteries": int(counts["warning_count"] or 0),
            "critical_batteries": int(counts["critical_count"] or 0),
            "average_health_score": float(counts["average_health_score"] or 0.0),
            "batteries_by_source": [dict(row) for row in source_rows],
            "recent_alerts": [dict(row) for row in alerts],
            "capacity_trend": [dict(row) for row in trend_rows],
        }

    def _decode_prediction(self, row: Any) -> Optional[dict[str, Any]]:
        if row is None:
            return None
        item = dict(row)
        payload = self._decode_json(item.pop("payload_json"), {})
        item["payload"] = payload
        for key, value in payload.items():
            if key not in item:
                item[key] = value
        return item

    def _decode_diagnosis(self, row: Any) -> Optional[dict[str, Any]]:
        if row is None:
            return None
        item = dict(row)
        item["root_causes"] = self._decode_json(item.pop("root_causes_json"), [])
        item["recommendations"] = self._decode_json(item.pop("recommendations_json"), [])
        item["related_symptoms"] = self._decode_json(item.pop("related_symptoms_json"), [])
        item["evidence"] = self._decode_json(item.pop("evidence_json"), [])
        payload = self._decode_json(item.pop("payload_json"), {})
        item["payload"] = payload
        for key, value in payload.items():
            if key not in item:
                item[key] = value
        return item

    def _decode_training_job(self, row: Any) -> Optional[dict[str, Any]]:
        if row is None:
            return None
        item = dict(row)
        item["force_run"] = bool(item.get("force_run"))
        item["baseline"] = self._decode_json(item.pop("baseline_json"), None)
        item["result"] = self._decode_json(item.pop("result_json"), None)
        item["metadata"] = self._decode_json(item.pop("metadata_json"), {})
        if "job_kind" not in item and item["metadata"].get("job_kind"):
            item["job_kind"] = item["metadata"]["job_kind"]
        if "seed_count" not in item and item["metadata"].get("seed_count") is not None:
            item["seed_count"] = item["metadata"]["seed_count"]
        return item


__all__ = ["BHMSRepository"]
