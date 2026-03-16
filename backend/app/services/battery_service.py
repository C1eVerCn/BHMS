"""电池数据、导入与查询服务。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import BHMSException
from backend.app.services.repository import BHMSRepository
from ml.data.adapters import CALCEAdapter, HUSTAdapter, KaggleAdapter, MATRAdapter, NASAAdapter, OxfordAdapter, PulseBatAdapter
from ml.data.dataset import RULDataModule
from ml.data.schema import BATTERY_SCHEMA_COLUMNS, enrich_existing_cycle_frame
from ml.data.source_registry import get_dataset_card, list_supported_sources


class BatteryService:
    def __init__(self, repository: Optional[BHMSRepository] = None, settings: Optional[Settings] = None):
        self.repository = repository or BHMSRepository()
        self.settings = settings or get_settings()
        self.adapters = {
            "nasa": NASAAdapter(eol_capacity_ratio=self.settings.battery_eol_ratio),
            "calce": CALCEAdapter(eol_capacity_ratio=self.settings.battery_eol_ratio),
            "kaggle": KaggleAdapter(eol_capacity_ratio=self.settings.battery_eol_ratio),
            "hust": HUSTAdapter(eol_capacity_ratio=self.settings.battery_eol_ratio),
            "matr": MATRAdapter(eol_capacity_ratio=self.settings.battery_eol_ratio),
            "oxford": OxfordAdapter(eol_capacity_ratio=self.settings.battery_eol_ratio),
            "pulsebat": PulseBatAdapter(eol_capacity_ratio=self.settings.battery_eol_ratio),
        }

    def bootstrap_demo_data(self) -> None:
        for source in list_supported_sources():
            if self.repository.count_canonical_batteries_by_source(source) > 0:
                continue
            raw_dir = self._source_dir(source)
            if not raw_dir.exists() or not any(raw_dir.iterdir()):
                continue
            try:
                self.import_builtin_source(source=source, include_in_training=(source == "nasa"))
            except Exception:
                continue

    def import_builtin_source(
        self,
        source: str,
        battery_ids: Optional[list[str]] = None,
        include_in_training: bool = False,
    ) -> dict[str, Any]:
        source = source.lower()
        adapter = self._get_adapter(source)
        output_dir = self.settings.processed_dir / source
        output_dir.mkdir(parents=True, exist_ok=True)
        processed_csv = output_dir / f"{source}_cycle_summary.csv"
        frame = adapter.process_directory(self._source_dir(source), output_path=processed_csv, battery_ids=battery_ids)
        summary = self.import_frame(
            frame,
            source=source,
            dataset_path=processed_csv,
            include_in_training=include_in_training,
        )
        summary["file_name"] = processed_csv.name
        summary["file_path"] = str(processed_csv)
        summary["validation_summary"]["ingestion_mode"] = "builtin_source"
        return summary

    def import_uploaded_file(
        self,
        file_path: str | Path,
        source: str | None = None,
        battery_id_hint: Optional[str] = None,
        include_in_training: bool = False,
    ) -> dict[str, Any]:
        path = Path(file_path)
        resolved_source = self._resolve_source(source, path)
        adapter = self._get_adapter(resolved_source)
        frame = adapter.process_file(path, battery_id_hint=battery_id_hint)
        summary = self.import_frame(
            frame,
            source=resolved_source,
            dataset_path=path,
            include_in_training=include_in_training,
        )
        summary["file_name"] = path.name
        summary["file_path"] = str(path)
        summary["detected_source"] = resolved_source
        summary["validation_summary"]["ingestion_mode"] = "uploaded_file"
        summary["validation_summary"]["ready_for_immediate_analysis"] = bool(summary["battery_ids"])
        return summary

    def import_frame(
        self,
        frame: pd.DataFrame,
        source: str,
        dataset_path: str | Path,
        include_in_training: bool,
    ) -> dict[str, Any]:
        battery_ids: list[str] = []
        imported_cycles = 0
        dataset_name = str(frame["dataset_name"].iloc[0]) if not frame.empty else source
        card = get_dataset_card(source)
        for battery_id, group in frame.groupby("battery_id"):
            group = group.sort_values("cycle_number").reset_index(drop=True)
            battery_ids.append(str(battery_id))
            latest = group.iloc[-1]
            initial = group.iloc[0]
            battery_record = {
                "battery_id": str(battery_id),
                "canonical_battery_id": str(latest.get("canonical_battery_id", battery_id)),
                "source": source,
                "dataset_name": str(latest.get("dataset_name", dataset_name)),
                "source_battery_id": str(latest.get("source_battery_id", battery_id)),
                "chemistry": str(latest.get("chemistry") or card.metadata_defaults.get("chemistry")),
                "form_factor": str(latest.get("form_factor") or card.metadata_defaults.get("form_factor")),
                "protocol_id": str(latest.get("protocol_id") or card.metadata_defaults.get("protocol_id")),
                "charge_c_rate": float(latest.get("charge_c_rate", card.metadata_defaults.get("charge_c_rate", 1.0))),
                "discharge_c_rate": float(latest.get("discharge_c_rate", card.metadata_defaults.get("discharge_c_rate", 1.0))),
                "ambient_temp": float(latest.get("ambient_temp", card.metadata_defaults.get("ambient_temp", 25.0))),
                "nominal_capacity": float(initial.get("nominal_capacity", initial.get("initial_capacity", initial["capacity"]))),
                "initial_capacity": float(initial.get("initial_capacity", initial["capacity"])),
                "latest_capacity": float(latest["capacity"]),
                "eol_ratio": float(latest.get("eol_ratio", self.settings.battery_eol_ratio)),
                "dataset_license": str(latest.get("dataset_license") or card.metadata_defaults.get("dataset_license", "unknown")),
                "cycle_count": int(group["cycle_number"].max()),
                "health_score": float(latest.get("health_score", 0.0)),
                "status": str(latest.get("status", "good")),
                "last_update": str(latest.get("timestamp") or ""),
                "dataset_path": str(dataset_path),
                "include_in_training": include_in_training,
                "metadata": {
                    "source_type": source,
                    "dataset_name": str(latest.get("dataset_name", dataset_name)),
                    "eol_cycle": int(latest.get("eol_cycle", latest["cycle_number"])),
                    "processed_rows": int(len(group)),
                    "group": card.group,
                    "description": card.description,
                },
            }
            self.repository.upsert_battery(battery_record)
            points = group.to_dict(orient="records")
            imported_cycles += self.repository.replace_cycle_points(str(battery_id), points)

        source_distribution = frame.groupby("source")["battery_id"].nunique().to_dict() if not frame.empty else {}
        validation_summary = {
            "battery_count": len(battery_ids),
            "imported_cycles": imported_cycles,
            "source": source,
            "dataset_name": dataset_name,
            "include_in_training": include_in_training,
            "source_distribution": source_distribution,
            "file_type": Path(dataset_path).suffix.lower() or source.lower(),
        }
        self.repository.insert_dataset_file(
            {
                "battery_id": ",".join(battery_ids),
                "source": source,
                "dataset_name": dataset_name,
                "file_name": Path(dataset_path).name,
                "file_path": str(dataset_path),
                "file_type": source.lower(),
                "row_count": imported_cycles,
                "include_in_training": include_in_training,
                "validation_summary": validation_summary,
            }
        )
        return {
            "battery_ids": battery_ids,
            "imported_cycles": imported_cycles,
            "validation_summary": validation_summary,
            "include_in_training": include_in_training,
            "source": source,
            "dataset_name": dataset_name,
        }

    def list_batteries(self, page: int, page_size: int) -> dict[str, Any]:
        items, total = self.repository.list_batteries(page=page, page_size=page_size)
        return {"items": items, "page": page, "page_size": page_size, "total": total}

    def update_training_candidate(self, battery_id: str, include_in_training: bool) -> dict[str, Any]:
        battery = self.get_battery(battery_id)
        self.repository.set_battery_training_flag(battery_id, include_in_training)
        battery["include_in_training"] = include_in_training
        return battery

    def get_battery(self, battery_id: str) -> dict[str, Any]:
        battery = self.repository.get_battery(battery_id)
        if battery is None:
            raise BHMSException(f"未找到电池 {battery_id}", status_code=404, code="battery_not_found")
        return battery

    def get_cycles(self, battery_id: str, limit: int = 200) -> list[dict[str, Any]]:
        self.get_battery(battery_id)
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
            "source": battery.get("source"),
            "dataset_name": battery.get("dataset_name"),
        }

    def get_dashboard(self) -> dict[str, Any]:
        summary = self.repository.dashboard_summary()
        summary["health_distribution"] = [
            {"name": "健康", "value": summary["good_batteries"]},
            {"name": "预警", "value": summary["warning_batteries"]},
            {"name": "故障", "value": summary["critical_batteries"]},
        ]
        return summary

    def prepare_training_dataset(self, source: str, seq_len: int = 30, batch_size: int = 16) -> dict[str, Any]:
        source = source.lower()
        rows = self.repository.query_training_cycle_points(source)
        if not rows:
            raise BHMSException(
                f"来源 {source} 当前没有被标记为训练候选的样本",
                status_code=400,
                code="missing_training_candidates",
            )
        frame = pd.DataFrame(rows)
        enriched = enrich_existing_cycle_frame(frame, eol_capacity_ratio=self.settings.battery_eol_ratio)
        enriched = enriched[BATTERY_SCHEMA_COLUMNS]
        output_dir = self.settings.processed_dir / source
        output_dir.mkdir(parents=True, exist_ok=True)
        source_csv = output_dir / f"{source}_cycle_summary.csv"
        enriched.to_csv(source_csv, index=False)
        data_module = RULDataModule(
            csv_path=source_csv,
            source=source,
            seq_len=seq_len,
            batch_size=batch_size,
            output_dir=output_dir,
        )
        metadata_paths = data_module.export_metadata()
        return {
            "import_summary": {
                "source": source,
                "battery_ids": sorted(enriched["battery_id"].unique().tolist()),
                "imported_cycles": int(len(enriched)),
                "include_in_training": True,
                "validation_summary": {
                    "battery_count": int(enriched["battery_id"].nunique()),
                    "imported_cycles": int(len(enriched)),
                    "source": source,
                    "dataset_name": "training_pool",
                    "include_in_training": True,
                    "ingestion_mode": "training_pool",
                },
            },
            "data_summary": data_module.summary(),
            "metadata_paths": metadata_paths,
            "csv_path": str(source_csv),
        }

    def _resolve_source(self, source: str | None, file_path: Path) -> str:
        if source and source.lower() != "auto":
            return source.lower()
        name = file_path.name.lower()
        for candidate in self.adapters:
            if candidate in name:
                return candidate
        return "kaggle"

    def _get_adapter(self, source: str):
        try:
            return self.adapters[source.lower()]
        except KeyError as exc:
            raise BHMSException(f"暂不支持的数据源: {source}", status_code=400, code="unsupported_source") from exc

    def _source_dir(self, source: str) -> Path:
        source = source.lower()
        mapping = {
            "nasa": self.settings.raw_nasa_dir,
            "calce": self.settings.raw_calce_dir,
            "kaggle": self.settings.raw_kaggle_dir,
            "hust": self.settings.raw_hust_dir,
            "matr": self.settings.raw_matr_dir,
            "oxford": self.settings.raw_oxford_dir,
            "pulsebat": self.settings.raw_pulsebat_dir,
        }
        return mapping[source]

    @staticmethod
    def _chemistry_for_source(source: str) -> str:
        try:
            return str(get_dataset_card(source).metadata_defaults.get("chemistry", "Imported Battery"))
        except KeyError:
            return "Imported Battery"
