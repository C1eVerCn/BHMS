"""预测、异常检测与诊断服务。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import BHMSException
from backend.app.services.repository import BHMSRepository
from kg.graphrag_engine import GraphRAGEngine
from ml.inference import AnomalyDetector, AnomalyThreshold, RULInferenceService


class PredictionService:
    def __init__(self, repository: Optional[BHMSRepository] = None, settings: Optional[Settings] = None):
        self.repository = repository or BHMSRepository()
        self.settings = settings or get_settings()
        self.inference_service = RULInferenceService(self.settings.model_dir)
        self.anomaly_detector = AnomalyDetector(use_statistical=True, use_isolation_forest=False, thresholds=AnomalyThreshold())
        self.diagnosis_engine = GraphRAGEngine(knowledge_path=self.settings.knowledge_path)

    def predict_rul(self, battery_id: str, model_name: str = "hybrid", seq_len: int = 30, historical_data: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        points = historical_data or self.repository.get_cycle_points(battery_id, limit=seq_len, descending=True)
        if len(points) < 10:
            raise BHMSException("历史数据不足，至少需要 10 个周期点", status_code=400, code="insufficient_history")
        sequence = self.inference_service.sequence_from_cycle_points(points)
        output = self.inference_service.predict(sequence, model_name=model_name)
        record_id = self.repository.insert_prediction(
            {
                "battery_id": battery_id,
                "model_name": output.model_name,
                "predicted_rul": round(output.predicted_rul, 2),
                "confidence": output.confidence,
                "input_seq_len": len(points),
                "source": "api",
                "payload": {"fallback_used": output.fallback_used},
            }
        )
        return {
            "id": record_id,
            "battery_id": battery_id,
            "model_name": output.model_name,
            "predicted_rul": round(output.predicted_rul, 2),
            "confidence": round(output.confidence, 3),
            "model_version": output.model_version,
            "fallback_used": output.fallback_used,
            "prediction_time": datetime.utcnow().isoformat(),
        }

    def detect_anomaly(self, battery_id: str, current_data: Optional[dict[str, Any]] = None, baseline_capacity: Optional[float] = None) -> dict[str, Any]:
        battery = self.repository.get_battery(battery_id)
        if battery is None:
            raise BHMSException(f"未找到电池 {battery_id}", status_code=404, code="battery_not_found")
        point = current_data or self.repository.get_latest_cycle_point(battery_id)
        if point is None:
            raise BHMSException("缺少可用于异常检测的周期数据", status_code=400, code="missing_cycle_data")
        baseline = baseline_capacity or battery.get("initial_capacity") or point.get("capacity")
        self.anomaly_detector.set_baseline(capacity=float(baseline))
        features = {
            "capacity": point.get("capacity", 0.0),
            "voltage_mean": point.get("voltage_mean", 0.0),
            "temperature_mean": point.get("temperature_mean", 0.0),
            "temperature_rise_rate": point.get("temperature_rise_rate", 0.0),
            "current_mean": point.get("current_mean", 0.0),
            "internal_resistance": point.get("internal_resistance"),
        }
        results = self.anomaly_detector.detect(features)
        events = [event.to_dict() for event in results["events"]]
        event_ids = self.repository.insert_anomaly_events(battery_id, events) if events else []
        return {
            "battery_id": battery_id,
            "is_anomaly": results["is_anomaly"],
            "max_severity": results["max_severity"],
            "summary": results["summary"],
            "event_ids": event_ids,
            "events": events,
            "detection_time": datetime.utcnow().isoformat(),
        }

    def diagnose(self, battery_id: str, anomalies: list[dict[str, Any]], battery_info: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        if not anomalies:
            anomalies = self.repository.list_anomalies(battery_id, limit=5)
        if not anomalies:
            raise BHMSException("缺少诊断所需的异常事件", status_code=400, code="missing_anomalies")
        diagnosis = self.diagnosis_engine.diagnose(anomalies=anomalies, battery_info=battery_info)
        record_id = self.repository.insert_diagnosis(
            {
                "battery_id": battery_id,
                "fault_type": diagnosis.fault_type,
                "confidence": diagnosis.confidence,
                "severity": diagnosis.severity,
                "description": diagnosis.description,
                "root_causes": diagnosis.root_causes,
                "recommendations": diagnosis.recommendations,
                "related_symptoms": diagnosis.related_symptoms,
                "evidence": diagnosis.evidence,
            }
        )
        return {"id": record_id, "battery_id": battery_id, **diagnosis.to_dict(), "diagnosis_time": datetime.utcnow().isoformat()}
