"""电池数据、导入与查询服务。"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import BHMSException
from backend.app.services.repository import BHMSRepository
from ml.data import NASABatteryPreprocessor


class BatteryService:
    def __init__(self, repository: Optional[BHMSRepository] = None, settings: Optional[Settings] = None):
        self.repository = repository or BHMSRepository()
        self.settings = settings or get_settings()
        self.preprocessor = NASABatteryPreprocessor(eol_capacity_ratio=self.settings.battery_eol_ratio)

    def bootstrap_demo_data(self) -> None:
        if self.repository.count_batteries() > 0:
            return
        raw_dir = self.settings.raw_nasa_dir
        if not raw_dir.exists():
            return
        processed_csv = self.settings.processed_dir / "nasa_cycle_summary.csv"
        frame = self.preprocessor.process_directory(raw_dir, output_path=processed_csv)
        self.import_frame(frame, source="NASA", dataset_path=processed_csv)

    def import_nasa(self, battery_ids: Optional[list[str]] = None) -> dict[str, Any]:
        processed_csv = self.settings.processed_dir / "nasa_cycle_summary.csv"
        frame = self.preprocessor.process_directory(self.settings.raw_nasa_dir, output_path=processed_csv, battery_ids=battery_ids)
        summary = self.import_frame(frame, source="NASA", dataset_path=processed_csv)
        summary["file_name"] = processed_csv.name
        summary["file_path"] = str(processed_csv)
        return summary

    def import_csv_file(self, file_path: str | Path, battery_id_hint: Optional[str] = None) -> dict[str, Any]:
        path = Path(file_path)
        frame = pd.read_csv(path)
        frame.columns = [column.strip() for column in frame.columns]
        required_columns = {
            "cycle_number",
            "voltage_mean",
            "current_mean",
            "temperature_mean",
            "capacity",
        }
        missing = sorted(required_columns - set(frame.columns))
        if missing:
            raise BHMSException(f"上传文件缺少必需列: {', '.join(missing)}", status_code=400, code="invalid_upload")
        if "battery_id" not in frame.columns:
            if not battery_id_hint:
                raise BHMSException("CSV 中缺少 battery_id 列，且未提供电池 ID", status_code=400, code="missing_battery_id")
            frame["battery_id"] = battery_id_hint
        frame["battery_id"] = frame["battery_id"].astype(str)
        for column in [
            "voltage_std",
            "voltage_min",
            "voltage_max",
            "current_std",
            "current_load_mean",
            "temperature_std",
            "temperature_rise_rate",
            "ambient_temperature",
        ]:
            if column not in frame.columns:
                frame[column] = 0.0
        if "voltage_min" not in frame.columns or (frame["voltage_min"] == 0).all():
            frame["voltage_min"] = frame["voltage_mean"] - frame["voltage_std"].fillna(0)
        if "voltage_max" not in frame.columns or (frame["voltage_max"] == 0).all():
            frame["voltage_max"] = frame["voltage_mean"] + frame["voltage_std"].fillna(0)
        if "timestamp" not in frame.columns:
            frame["timestamp"] = None
        if "source_type" not in frame.columns:
            frame["source_type"] = "uploaded_csv"

        normalized_frames: list[pd.DataFrame] = []
        for battery_id, group in frame.groupby("battery_id"):
            group = group.sort_values("cycle_number").reset_index(drop=True)
            initial_capacity = float(group["capacity"].iloc[0])
            group["initial_capacity"] = initial_capacity
            group["capacity_ratio"] = group["capacity"] / max(initial_capacity, 1e-6)
            eol_candidates = group.loc[group["capacity_ratio"] <= self.settings.battery_eol_ratio, "cycle_number"]
            eol_cycle = int(eol_candidates.iloc[0]) if not eol_candidates.empty else int(group["cycle_number"].max())
            group["eol_cycle"] = eol_cycle
            group["RUL"] = (eol_cycle - group["cycle_number"]).clip(lower=0)
            group["health_score"] = (group["capacity_ratio"] * 100).clip(lower=0, upper=100)
            group["status"] = group["health_score"].map(self._health_status)
            normalized_frames.append(group)
        normalized = pd.concat(normalized_frames, ignore_index=True)
        summary = self.import_frame(normalized, source="CSV", dataset_path=path)
        summary["file_name"] = path.name
        summary["file_path"] = str(path)
        return summary

    def import_frame(self, frame: pd.DataFrame, source: str, dataset_path: str | Path) -> dict[str, Any]:
        battery_ids: list[str] = []
        imported_cycles = 0
        for battery_id, group in frame.groupby("battery_id"):
            group = group.sort_values("cycle_number").reset_index(drop=True)
            battery_ids.append(str(battery_id))
            latest = group.iloc[-1]
            initial = group.iloc[0]
            battery_record = {
                "battery_id": str(battery_id),
                "source": source,
                "chemistry": "NASA Li-ion" if source == "NASA" else "Imported CSV",
                "nominal_capacity": float(initial.get("initial_capacity", initial["capacity"])),
                "initial_capacity": float(initial.get("initial_capacity", initial["capacity"])),
                "latest_capacity": float(latest["capacity"]),
                "cycle_count": int(group["cycle_number"].max()),
                "health_score": float(latest.get("health_score", 0.0)),
                "status": str(latest.get("status", "good")),
                "last_update": str(latest.get("timestamp") or ""),
                "dataset_path": str(dataset_path),
                "metadata": {
                    "source_type": source,
                    "eol_cycle": int(latest.get("eol_cycle", latest["cycle_number"])),
                    "processed_rows": int(len(group)),
                },
            }
            self.repository.upsert_battery(battery_record)
            points = group[
                [
                    "battery_id",
                    "cycle_number",
                    "timestamp",
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
                    "capacity",
                    "source_type",
                ]
            ].to_dict(orient="records")
            imported_cycles += self.repository.replace_cycle_points(str(battery_id), points)
        validation_summary = {
            "battery_count": len(battery_ids),
            "imported_cycles": imported_cycles,
            "source": source,
        }
        self.repository.insert_dataset_file(
            {
                "battery_id": ",".join(battery_ids),
                "file_name": Path(dataset_path).name,
                "file_path": str(dataset_path),
                "file_type": source.lower(),
                "row_count": imported_cycles,
                "validation_summary": validation_summary,
            }
        )
        return {
            "battery_ids": battery_ids,
            "imported_cycles": imported_cycles,
            "validation_summary": validation_summary,
        }

    def list_batteries(self, page: int, page_size: int) -> dict[str, Any]:
        items, total = self.repository.list_batteries(page=page, page_size=page_size)
        return {"items": items, "page": page, "page_size": page_size, "total": total}

    def get_battery(self, battery_id: str) -> dict[str, Any]:
        battery = self.repository.get_battery(battery_id)
        if battery is None:
            raise BHMSException(f"未找到电池 {battery_id}", status_code=404, code="battery_not_found")
        return battery

    def get_cycles(self, battery_id: str, limit: int = 200) -> list[dict[str, Any]]:
        battery = self.get_battery(battery_id)
        _ = battery
        return self.repository.get_cycle_points(battery_id, limit=limit)

    def get_history(self, battery_id: str) -> dict[str, Any]:
        self.get_battery(battery_id)
        return {
            "battery_id": battery_id,
            "predictions": self.repository.list_predictions(battery_id),
            "diagnoses": self.repository.list_diagnoses(battery_id),
            "anomalies": self.repository.list_anomalies(battery_id),
        }

    def get_health(self, battery_id: str) -> dict[str, Any]:
        battery = self.get_battery(battery_id)
        predictions = self.repository.list_predictions(battery_id, limit=1)
        anomalies = self.repository.list_anomalies(battery_id, limit=10)
        return {
            "battery_id": battery_id,
            "overall_health": battery["status"],
            "health_score": battery["health_score"],
            "latest_capacity": battery.get("latest_capacity"),
            "rul_prediction": predictions[0]["predicted_rul"] if predictions else None,
            "anomaly_count": len(anomalies),
            "last_update": battery.get("last_update"),
        }

    def get_dashboard(self) -> dict[str, Any]:
        summary = self.repository.dashboard_summary()
        summary["health_distribution"] = [
            {"name": "健康", "value": summary["good_batteries"]},
            {"name": "预警", "value": summary["warning_batteries"]},
            {"name": "故障", "value": summary["critical_batteries"]},
        ]
        return summary

    @staticmethod
    def _health_status(score: float) -> str:
        if score >= 85:
            return "good"
        if score >= 70:
            return "warning"
        return "critical"
