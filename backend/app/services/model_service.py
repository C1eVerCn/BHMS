"""预测、异常检测、GraphRAG 诊断与报告服务。"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import numpy as np

from backend.app.core.config import Settings, get_settings
from backend.app.core.exceptions import BHMSException
from backend.app.services.repository import BHMSRepository
from kg.graphrag_engine import GraphRAGEngine, GraphTrace
from ml.inference import AnomalyDetector, AnomalyThreshold, LifecycleInferenceService, RULInferenceService


class PredictionService:
    def __init__(self, repository: Optional[BHMSRepository] = None, settings: Optional[Settings] = None):
        self.repository = repository or BHMSRepository()
        self.settings = settings or get_settings()
        self.inference_service = RULInferenceService(self.settings.model_dir)
        self.lifecycle_inference_service = LifecycleInferenceService(self.settings.model_dir)
        self.anomaly_detector = AnomalyDetector(use_statistical=True, use_isolation_forest=True, thresholds=AnomalyThreshold())
        self.diagnosis_engine = GraphRAGEngine(
            knowledge_path=self.settings.knowledge_path,
            graph_backend=self.settings.graph_backend,
            neo4j_uri=self.settings.neo4j_uri,
            neo4j_user=self.settings.neo4j_user,
            neo4j_password=self.settings.neo4j_password,
            neo4j_database=self.settings.neo4j_database,
        )

    def predict_rul(self, battery_id: str, model_name: str = "hybrid", seq_len: int = 30, historical_data: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
        prediction = self._predict_lifecycle_payload(
            battery_id=battery_id,
            model_name=model_name,
            seq_len=seq_len,
            historical_data=historical_data,
            request_source="api_v1",
        )
        return {
            "id": prediction["id"],
            "battery_id": prediction["battery_id"],
            "model_name": prediction["model_name"],
            "predicted_rul": prediction["predicted_rul"],
            "confidence": prediction["confidence"],
            "model_version": prediction["model_version"],
            "model_source": prediction["model_source"],
            "checkpoint_id": prediction["checkpoint_id"],
            "fallback_used": prediction["fallback_used"],
            "prediction_time": prediction["prediction_time"],
            "projection": prediction["projection"],
            "explanation": prediction.get("explanation"),
            "report_markdown": prediction["report_markdown"],
        }

    def predict_lifecycle(
        self,
        battery_id: str,
        model_name: str = "hybrid",
        seq_len: int = 30,
        historical_data: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        prediction = self._predict_lifecycle_payload(
            battery_id=battery_id,
            model_name=model_name,
            seq_len=seq_len,
            historical_data=historical_data,
            request_source="api_v2",
        )
        return {
            "id": prediction["id"],
            "battery_id": battery_id,
            "model_name": prediction["model_name"],
            "model_version": prediction["model_version"],
            "model_source": prediction["model_source"],
            "prediction_time": prediction["prediction_time"],
            "predicted_rul": prediction["predicted_rul"],
            "confidence": prediction["confidence"],
            "checkpoint_id": prediction["checkpoint_id"],
            "fallback_used": prediction["fallback_used"],
            "predicted_knee_cycle": prediction.get("predicted_knee_cycle"),
            "predicted_eol_cycle": prediction.get("predicted_eol_cycle"),
            "trajectory": prediction.get("trajectory", []),
            "risk_windows": prediction.get("risk_windows", []),
            "future_risks": prediction.get("future_risks", {}),
            "model_evidence": prediction.get("model_evidence", {}),
            "uncertainty": prediction.get("uncertainty"),
            "projection": prediction["projection"],
            "explanation": prediction.get("explanation"),
            "report_markdown": prediction["report_markdown"],
        }

    def _predict_lifecycle_payload(
        self,
        *,
        battery_id: str,
        model_name: str,
        seq_len: int,
        historical_data: Optional[list[dict[str, Any]]],
        request_source: str,
    ) -> dict[str, Any]:
        battery = self.repository.get_battery(battery_id)
        if battery is None:
            raise BHMSException(f"未找到电池 {battery_id}", status_code=404, code="battery_not_found")
        all_points = historical_data or self.repository.get_cycle_points(battery_id, limit=1000)
        if len(all_points) < 10:
            raise BHMSException("历史数据不足，至少需要 10 个周期点", status_code=400, code="insufficient_history")
        input_points = all_points[-seq_len:]
        lifecycle_output = self.lifecycle_inference_service.predict(
            sequence=None,
            points=input_points,
            source=battery["source"],
            model_name=model_name,
            battery_info=battery,
        )
        prediction_time = datetime.utcnow().isoformat()
        explanation = lifecycle_output.explanation.to_dict() if lifecycle_output.explanation else None
        future_risks = self._build_lifecycle_evidence(
            battery,
            {
                "predicted_knee_cycle": lifecycle_output.predicted_knee_cycle,
                "predicted_eol_cycle": lifecycle_output.predicted_eol_cycle,
                "trajectory": lifecycle_output.trajectory,
                "projection": lifecycle_output.projection,
            },
        )
        model_evidence = self._build_model_evidence(
            {
                "explanation": explanation,
                "payload": {"explanation": explanation},
            }
        )
        risk_windows = self._build_risk_windows(future_risks)
        report_markdown = self._build_prediction_report(
            battery=battery,
            output=lifecycle_output,
            projection=lifecycle_output.projection,
            prediction_time=prediction_time,
            future_risks=future_risks,
            model_evidence=model_evidence,
        )
        payload = {
            "task_kind": "lifecycle",
            "fallback_used": lifecycle_output.fallback_used,
            "model_version": lifecycle_output.model_version,
            "model_source": lifecycle_output.model_source,
            "checkpoint_id": lifecycle_output.checkpoint_id,
            "prediction_time": prediction_time,
            "projection": lifecycle_output.projection,
            "explanation": explanation,
            "report_markdown": report_markdown,
            "predicted_knee_cycle": lifecycle_output.predicted_knee_cycle,
            "predicted_eol_cycle": lifecycle_output.predicted_eol_cycle,
            "trajectory": lifecycle_output.trajectory,
            "risk_windows": risk_windows,
            "future_risks": future_risks,
            "model_evidence": model_evidence,
            "uncertainty": lifecycle_output.uncertainty,
        }
        record_id = self.repository.insert_prediction(
            {
                "battery_id": battery_id,
                "model_name": lifecycle_output.model_name,
                "predicted_rul": round(lifecycle_output.predicted_rul, 2),
                "confidence": lifecycle_output.confidence,
                "input_seq_len": len(input_points),
                "source": request_source,
                "payload": payload,
            }
        )
        return {
            "id": record_id,
            "battery_id": battery_id,
            "model_name": lifecycle_output.model_name,
            "predicted_rul": round(lifecycle_output.predicted_rul, 2),
            "confidence": round(lifecycle_output.confidence, 3),
            "model_version": lifecycle_output.model_version,
            "model_source": lifecycle_output.model_source,
            "checkpoint_id": lifecycle_output.checkpoint_id,
            "fallback_used": lifecycle_output.fallback_used,
            "prediction_time": prediction_time,
            "projection": lifecycle_output.projection,
            "explanation": explanation,
            "report_markdown": report_markdown,
            "predicted_knee_cycle": lifecycle_output.predicted_knee_cycle,
            "predicted_eol_cycle": lifecycle_output.predicted_eol_cycle,
            "trajectory": lifecycle_output.trajectory,
            "risk_windows": risk_windows,
            "future_risks": future_risks,
            "model_evidence": model_evidence,
            "uncertainty": lifecycle_output.uncertainty,
        }

    def explain_mechanism(
        self,
        battery_id: str,
        anomalies: Optional[list[dict[str, Any]]] = None,
        battery_info: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        anomaly_list = anomalies or self.repository.list_anomalies(battery_id, limit=5)
        if not anomaly_list:
            anomaly_result = self.detect_anomaly(battery_id)
            anomaly_list = anomaly_result["events"]
        diagnosis = self.diagnose(battery_id=battery_id, anomalies=anomaly_list, battery_info=battery_info)
        battery = self.repository.get_battery(battery_id)
        latest_prediction = self.repository.list_predictions(battery_id, limit=1)
        lifecycle_evidence = self._build_lifecycle_evidence(battery or {}, latest_prediction[0] if latest_prediction else None)
        model_evidence = self._build_model_evidence(latest_prediction[0] if latest_prediction else None)
        return {
            **diagnosis,
            "lifecycle_evidence": lifecycle_evidence,
            "model_evidence": model_evidence,
            "graph_backend": self.diagnosis_engine.active_backend,
        }

    def detect_anomaly(self, battery_id: str, current_data: Optional[dict[str, Any]] = None, baseline_capacity: Optional[float] = None) -> dict[str, Any]:
        battery = self.repository.get_battery(battery_id)
        if battery is None:
            raise BHMSException(f"未找到电池 {battery_id}", status_code=404, code="battery_not_found")
        point = current_data or self.repository.get_latest_cycle_point(battery_id)
        if point is None:
            raise BHMSException("缺少可用于异常检测的周期数据", status_code=400, code="missing_cycle_data")
        baseline = baseline_capacity or battery.get("initial_capacity") or point.get("capacity")
        baseline_point = None
        if current_data is None:
            history = self.repository.get_cycle_points(battery_id, limit=1000)
            baseline_point = history[0] if history else None
        baseline_resistance = None
        if baseline_point is not None:
            baseline_resistance = baseline_point.get("internal_resistance")
        elif current_data is not None:
            baseline_resistance = current_data.get("internal_resistance")
        self.anomaly_detector.set_baseline(
            capacity=float(baseline),
            resistance=float(baseline_resistance) if baseline_resistance is not None else None,
        )
        features = {
            "capacity": point.get("capacity", 0.0),
            "voltage_mean": point.get("voltage_mean", 0.0),
            "temperature_mean": point.get("temperature_mean", 0.0),
            "temperature_rise_rate": point.get("temperature_rise_rate", 0.0),
            "current_mean": point.get("current_mean", 0.0),
            "internal_resistance": point.get("internal_resistance"),
        }
        results = self.anomaly_detector.detect(features, source_scope=str(battery.get("source", "")))
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
        battery = self.repository.get_battery(battery_id)
        if battery is None:
            raise BHMSException(f"未找到电池 {battery_id}", status_code=404, code="battery_not_found")
        if not anomalies:
            anomalies = self.repository.list_anomalies(battery_id, limit=5)
        latest_prediction = self.repository.list_predictions(battery_id, limit=1)
        lifecycle_evidence = self._build_lifecycle_evidence(battery, latest_prediction[0] if latest_prediction else None)
        model_evidence = self._build_model_evidence(latest_prediction[0] if latest_prediction else None)
        if not anomalies:
            diagnosis = {
                "fault_type": "未发现明显故障",
                "confidence": 0.92,
                "severity": "info",
                "description": "当前未检测到可支撑故障诊断的异常事件，电池整体表现处于可继续观察状态。",
                "root_causes": ["本次异常检测未发现显著异常模式"],
                "recommendations": ["继续保持常规监测", "按计划执行容量复测与维护", "若后续出现异常波动再发起深入诊断"],
                "related_symptoms": [],
                "evidence": ["异常检测结果为空，系统未识别到容量、电压或温度异常事件"],
                "candidate_faults": [],
                "graph_trace": GraphTrace(matched_symptoms=[], nodes=[], edges=[], ranking_basis=["无异常事件，因此未进入图谱检索"]).to_dict(),
                "decision_basis": ["当前无异常事件，因此系统直接返回“未发现明显故障”。"],
                "report_markdown": "# 电池故障诊断报告\n\n## 诊断结论\n- 当前未发现明显故障\n\n## 说明\n- 本次异常检测未发现可支撑故障诊断的异常事件。",
            }
            record_id = self.repository.insert_diagnosis(
                {
                    "battery_id": battery_id,
                    **diagnosis,
                    "payload": {
                        "candidate_faults": [],
                        "graph_trace": diagnosis["graph_trace"],
                        "decision_basis": diagnosis["decision_basis"],
                        "report_markdown": diagnosis["report_markdown"],
                    },
                }
            )
            return {
                "id": record_id,
                "battery_id": battery_id,
                **diagnosis,
                "diagnosis_time": datetime.utcnow().isoformat(),
            }
        diagnosis = self.diagnosis_engine.diagnose(
            anomalies=anomalies,
            battery_info=battery_info or battery,
            lifecycle_evidence=lifecycle_evidence,
            model_evidence=model_evidence,
        )
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
                "payload": {
                    "candidate_faults": [item.to_dict() for item in diagnosis.candidate_faults],
                    "graph_trace": diagnosis.graph_trace.to_dict(),
                    "decision_basis": diagnosis.decision_basis,
                    "report_markdown": diagnosis.report_markdown,
                },
            }
        )
        return {"id": record_id, "battery_id": battery_id, **diagnosis.to_dict(), "diagnosis_time": datetime.utcnow().isoformat()}

    def _build_lifecycle_evidence(self, battery: dict[str, Any], prediction: Optional[dict[str, Any]]) -> dict[str, Any]:
        payload = ((prediction or {}).get("payload")) or {}
        if payload.get("future_risks"):
            return dict(payload["future_risks"])
        projection = ((prediction or {}).get("projection")) or payload.get("projection") or {}
        trajectory = ((prediction or {}).get("trajectory")) or payload.get("trajectory") or []
        forecast_points = projection.get("forecast_points") or []
        predicted_eol_cycle = (prediction or {}).get("predicted_eol_cycle") or payload.get("predicted_eol_cycle") or projection.get("predicted_eol_cycle")
        predicted_knee_cycle = (prediction or {}).get("predicted_knee_cycle") or payload.get("predicted_knee_cycle")
        last_actual_cycle = None
        if forecast_points:
            last_actual_cycle = float(forecast_points[0].get("cycle", 0.0))
        elif trajectory:
            last_actual_cycle = float(trajectory[0].get("cycle", 0.0))
        future_capacity_fade_pattern = "stable_decline"
        accelerated_window = None
        if trajectory:
            capacities = np.asarray([float(item.get("capacity_ratio", item.get("soh", 0.0)) or 0.0) for item in trajectory], dtype=float)
            cycles = np.asarray([float(item.get("cycle", 0.0) or 0.0) for item in trajectory], dtype=float)
        else:
            capacities = np.asarray([float(item.get("capacity", 0.0) or 0.0) for item in forecast_points], dtype=float)
            cycles = np.asarray([float(item.get("cycle", 0.0) or 0.0) for item in forecast_points], dtype=float)
        if capacities.size >= 3:
            slopes = np.diff(capacities)
            if slopes.size and float(np.min(slopes)) < float(np.mean(slopes) * 1.35):
                future_capacity_fade_pattern = "accelerated_tail_fade"
                accelerated_index = int(np.argmin(slopes))
                start_cycle = cycles[max(0, accelerated_index)]
                end_cycle = cycles[min(len(cycles) - 1, accelerated_index + 2)]
                accelerated_window = f"{start_cycle:.0f}-{end_cycle:.0f}"
        health_score = float(battery.get("health_score", 0.0) or 0.0)
        temperature_risk = "high" if health_score < 65 else "medium" if health_score < 80 else "low"
        resistance_risk = "high" if health_score < 60 else "medium" if health_score < 78 else "low"
        voltage_risk = "medium" if health_score < 75 else "low"
        if predicted_knee_cycle is None and predicted_eol_cycle is not None and last_actual_cycle is not None:
            span = max(5.0, float(predicted_eol_cycle) - last_actual_cycle)
            predicted_knee_cycle = round(last_actual_cycle + span * 0.35, 2)
        return {
            "predicted_knee_cycle": predicted_knee_cycle,
            "predicted_eol_cycle": predicted_eol_cycle,
            "accelerated_degradation_window": accelerated_window,
            "future_capacity_fade_pattern": future_capacity_fade_pattern,
            "temperature_risk": temperature_risk,
            "resistance_risk": resistance_risk,
            "voltage_risk": voltage_risk,
        }

    @staticmethod
    def _build_model_evidence(prediction: Optional[dict[str, Any]]) -> dict[str, Any]:
        payload = (prediction or {}).get("payload") or {}
        explanation = payload.get("explanation") or (prediction or {}).get("explanation") or {}
        feature_contributions = explanation.get("feature_contributions") or []
        window_contributions = explanation.get("window_contributions") or []
        confidence_factors = (explanation.get("confidence_summary") or {}).get("factors") or []
        return {
            "top_features": [str(item.get("feature")) for item in feature_contributions[:3] if item.get("feature")],
            "critical_windows": [str(item.get("window_label")) for item in window_contributions[:3] if item.get("window_label")],
            "confidence_factors": [str(item) for item in confidence_factors[:3]],
        }

    @staticmethod
    def _build_lifecycle_trajectory(projection: dict[str, Any], *, initial_capacity: float) -> list[dict[str, Any]]:
        safe_initial_capacity = max(initial_capacity, 1e-6)
        items = []
        for point in projection.get("forecast_points", []):
            capacity = float(point.get("capacity", 0.0) or 0.0)
            items.append(
                {
                    "cycle": float(point.get("cycle", 0.0) or 0.0),
                    "capacity_ratio": round(capacity / safe_initial_capacity, 4),
                    "soh": round(capacity / safe_initial_capacity, 4),
                }
            )
        return items

    @staticmethod
    def _build_risk_windows(lifecycle_evidence: dict[str, Any]) -> list[dict[str, Any]]:
        window = lifecycle_evidence.get("accelerated_degradation_window")
        if not window:
            return []
        try:
            start_text, end_text = str(window).split("-", maxsplit=1)
            start_cycle = float(start_text.strip().split()[-1])
            end_cycle = float(end_text.strip().split()[0])
        except Exception:
            start_cycle, end_cycle = 0.0, 0.0
        severity = "high" if lifecycle_evidence.get("temperature_risk") == "high" else "medium"
        return [
            {
                "label": "accelerated_degradation",
                "start_cycle": start_cycle,
                "end_cycle": end_cycle,
                "severity": severity,
                "description": "预测窗口内衰退斜率上升，需优先关注温度与内阻风险。",
            }
        ]

    def get_prediction_report(self, prediction_id: int) -> str:
        prediction = self.repository.get_prediction(prediction_id)
        if prediction is None:
            raise BHMSException(f"未找到预测记录 {prediction_id}", status_code=404, code="prediction_not_found")
        report = prediction.get("report_markdown") or prediction.get("payload", {}).get("report_markdown")
        if not report:
            raise BHMSException("预测报告不存在", status_code=404, code="prediction_report_not_found")
        return str(report)

    def get_diagnosis_report(self, diagnosis_id: int) -> str:
        diagnosis = self.repository.get_diagnosis(diagnosis_id)
        if diagnosis is None:
            raise BHMSException(f"未找到诊断记录 {diagnosis_id}", status_code=404, code="diagnosis_not_found")
        report = diagnosis.get("report_markdown") or diagnosis.get("payload", {}).get("report_markdown")
        if not report:
            raise BHMSException("诊断报告不存在", status_code=404, code="diagnosis_report_not_found")
        return str(report)

    def _build_projection(self, points: list[dict[str, Any]], predicted_rul: float, confidence: float) -> dict[str, Any]:
        cycles = np.asarray([float(item.get("cycle_number", 0.0) or 0.0) for item in points], dtype=float)
        capacities = np.asarray([float(item.get("capacity", 0.0) or 0.0) for item in points], dtype=float)
        initial_capacity = float(capacities[0]) if capacities.size else 0.0
        eol_capacity = initial_capacity * self.settings.battery_eol_ratio
        last_cycle = float(cycles[-1]) if cycles.size else 0.0
        last_capacity = float(capacities[-1]) if capacities.size else 0.0
        predicted_eol_cycle = max(last_cycle, last_cycle + float(predicted_rul))
        horizon = max(predicted_eol_cycle - last_cycle, 1.0)
        method = self._choose_projection_method(cycles, capacities)
        num_points = max(12, min(80, int(horizon) + 1))
        forecast_cycles = np.linspace(last_cycle, predicted_eol_cycle, num=num_points)
        progress = (forecast_cycles - last_cycle) / max(horizon, 1e-6)
        if method == "exponential":
            rate = np.log(max(last_capacity, eol_capacity + 1e-6) / max(eol_capacity, 1e-6)) / max(horizon, 1e-6)
            forecast_capacities = last_capacity * np.exp(-rate * (forecast_cycles - last_cycle))
            forecast_capacities[-1] = eol_capacity
        else:
            forecast_capacities = last_capacity + (eol_capacity - last_capacity) * progress
        forecast_capacities = np.minimum.accumulate(forecast_capacities)
        band_width = max(initial_capacity * 0.015, (1.0 - confidence) * initial_capacity * 0.18)
        confidence_band = [
            {
                "cycle": round(float(cycle), 2),
                "lower": round(float(max(eol_capacity, capacity - band_width)), 4),
                "upper": round(float(capacity + band_width), 4),
            }
            for cycle, capacity in zip(forecast_cycles, forecast_capacities)
        ]
        return {
            "actual_points": [{"cycle": round(float(cycle), 2), "capacity": round(float(capacity), 4)} for cycle, capacity in zip(cycles, capacities)],
            "forecast_points": [{"cycle": round(float(cycle), 2), "capacity": round(float(capacity), 4)} for cycle, capacity in zip(forecast_cycles, forecast_capacities)],
            "eol_capacity": round(float(eol_capacity), 4),
            "predicted_eol_cycle": round(float(predicted_eol_cycle), 2),
            "confidence_band": confidence_band,
            "projection_method": method,
        }

    @staticmethod
    def _choose_projection_method(cycles: np.ndarray, capacities: np.ndarray) -> str:
        if cycles.size < 6:
            return "linear"
        recent_cycles = cycles[-min(12, cycles.size) :]
        recent_capacities = capacities[-min(12, capacities.size) :]
        linear = np.polyfit(recent_cycles, recent_capacities, deg=1)
        linear_pred = np.polyval(linear, recent_cycles)
        linear_mse = float(np.mean((recent_capacities - linear_pred) ** 2))
        safe_capacities = np.clip(recent_capacities, 1e-6, None)
        exp_fit = np.polyfit(recent_cycles, np.log(safe_capacities), deg=1)
        exp_pred = np.exp(np.polyval(exp_fit, recent_cycles))
        exp_mse = float(np.mean((recent_capacities - exp_pred) ** 2))
        return "exponential" if exp_mse < linear_mse * 0.92 else "linear"

    def _build_prediction_report(
        self,
        battery: dict[str, Any],
        output,
        projection: dict[str, Any],
        prediction_time: str,
        future_risks: Optional[dict[str, Any]] = None,
        model_evidence: Optional[dict[str, Any]] = None,
    ) -> str:
        explanation = output.explanation.to_dict() if output.explanation else {}
        future_risks = future_risks or {}
        model_evidence = model_evidence or {}
        lines = [
            "# 全生命周期预测报告",
            "",
            "## 一、生命周期概览",
            f"- 电池 ID：{battery['battery_id']}",
            f"- 数据来源：{battery.get('source', 'unknown')}",
            f"- 模型：{output.model_name}",
            f"- 预测 RUL：{output.predicted_rul:.2f} cycles",
            f"- 置信度：{output.confidence * 100:.1f}%",
            f"- 预测时间：{prediction_time}",
            "",
            "## 二、生命周期轨迹",
            f"- 当前历史轨迹长度：{len(projection['actual_points'])} 个 cycle",
            f"- 预测 EOL 周期：{projection['predicted_eol_cycle']}",
            f"- EOL 容量阈值：{projection['eol_capacity']}",
            f"- 投影方法：{projection.get('projection_method', 'linear')}",
            "",
            "## 三、knee / 风险窗口",
            f"- 预测 knee 周期：{future_risks.get('predicted_knee_cycle', '--')}",
            f"- 衰退模式：{future_risks.get('future_capacity_fade_pattern', '--')}",
            f"- 温度风险：{future_risks.get('temperature_risk', '--')}",
            f"- 内阻风险：{future_risks.get('resistance_risk', '--')}",
            f"- 电压风险：{future_risks.get('voltage_risk', '--')}",
            "",
            "## 四、模型证据链",
        ]
        for item in explanation.get("feature_contributions", [])[:5]:
            lines.append(f"- 特征 {item['feature']}：影响值 {item['impact']}，{item['description']}")
        if model_evidence.get("top_features"):
            lines.extend(["", "## 五、辅助证据"])
            lines.extend(f"- 关键特征：{item}" for item in model_evidence.get("top_features", []))
            lines.extend(f"- 关键窗口：{item}" for item in model_evidence.get("critical_windows", []))
        lines.extend(["", "## 六、关键时间窗口"])
        for item in explanation.get("window_contributions", [])[:4]:
            lines.append(f"- {item['window_label']}：影响值 {item['impact']}，{item['description']}")
        lines.extend(["", "## 七、置信度说明"])
        confidence_summary = explanation.get("confidence_summary", {})
        for factor in confidence_summary.get("factors", []):
            lines.append(f"- {factor}")
        lines.extend(
            [
                "",
                "## 八、限制说明",
                "- 当前系统优先输出可审计的生命周期轨迹、knee/EOL/RUL 与证据链。",
                "- 若生命周期训练权重缺失，系统会退回到启发式生命周期估计，并显式标记 fallback。",
                "- 当前解释结果强调可追踪证据，而不是模型内部隐式思维过程。",
            ]
        )
        return "\n".join(lines)


__all__ = ["PredictionService"]
