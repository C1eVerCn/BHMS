"""异常检测模块，输出标准化症状事件。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyType(str, Enum):
    CAPACITY_DROP = "capacity_drop"
    VOLTAGE_ANOMALY = "voltage_anomaly"
    TEMPERATURE_ANOMALY = "temperature_anomaly"
    CURRENT_ANOMALY = "current_anomaly"
    INTERNAL_RESISTANCE = "internal_resistance"
    UNKNOWN = "unknown"


ANOMALY_LABELS = {
    AnomalyType.CAPACITY_DROP: "容量骤降",
    AnomalyType.VOLTAGE_ANOMALY: "电压异常",
    AnomalyType.TEMPERATURE_ANOMALY: "温度异常",
    AnomalyType.CURRENT_ANOMALY: "电流异常",
    AnomalyType.INTERNAL_RESISTANCE: "内阻异常",
    AnomalyType.UNKNOWN: "未知异常",
}

SOURCE_THRESHOLD_PROFILES: dict[str, dict[str, float]] = {
    "nasa": {
        "capacity_drop_threshold": 0.18,
        "temperature_high_threshold": 48.0,
        "temperature_rise_rate": 3.5,
        "current_spike_threshold": 4.5,
        "resistance_increase_threshold": 0.25,
    },
    "calce": {
        "capacity_drop_threshold": 0.16,
        "temperature_high_threshold": 42.0,
        "temperature_rise_rate": 2.8,
        "current_spike_threshold": 4.0,
        "resistance_increase_threshold": 0.22,
    },
    "kaggle": {
        "capacity_drop_threshold": 0.17,
        "temperature_high_threshold": 43.0,
        "temperature_rise_rate": 3.0,
        "current_spike_threshold": 4.2,
        "resistance_increase_threshold": 0.24,
    },
}


@dataclass
class AnomalyThreshold:
    capacity_drop_threshold: float = 0.20
    capacity_fade_rate: float = 0.05
    voltage_jump_threshold: float = 0.1
    voltage_range_min: float = 2.5
    voltage_range_max: float = 4.5
    temperature_high_threshold: float = 45.0
    temperature_low_threshold: float = -10.0
    temperature_rise_rate: float = 5.0
    current_spike_threshold: float = 5.0
    resistance_increase_threshold: float = 0.30


@dataclass
class AnomalyEvent:
    code: str
    symptom: str
    severity: str
    metric_name: str
    metric_value: Optional[float]
    threshold_value: Optional[str]
    description: str
    source: str = "statistical"
    evidence: list[str] = field(default_factory=list)
    evidence_source: str = "statistical_rules"
    rule_id: Optional[str] = None
    confidence_basis: list[str] = field(default_factory=list)
    source_scope: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StatisticalDetector:
    def __init__(self, thresholds: Optional[AnomalyThreshold] = None):
        self.thresholds = thresholds or AnomalyThreshold()
        self.baseline_capacity: Optional[float] = None
        self.baseline_resistance: Optional[float] = None

    def set_baseline(self, capacity: float, resistance: Optional[float] = None) -> None:
        self.baseline_capacity = capacity
        self.baseline_resistance = resistance

    def detect(self, features: dict[str, float], source_scope: Optional[str] = None) -> list[AnomalyEvent]:
        events: list[AnomalyEvent] = []
        thresholds = self._thresholds_for_source(source_scope)
        source_tags = [source_scope.lower()] if source_scope else ["generic"]

        if "capacity" in features and self.baseline_capacity:
            capacity = float(features["capacity"])
            capacity_ratio = capacity / max(self.baseline_capacity, 1e-6)
            if capacity_ratio < (1 - thresholds.capacity_drop_threshold):
                events.append(
                    AnomalyEvent(
                        code=AnomalyType.CAPACITY_DROP.value,
                        symptom="容量衰减过快",
                        severity=self._calculate_descending_severity(capacity_ratio, 1 - thresholds.capacity_drop_threshold, 0.55),
                        metric_name="capacity",
                        metric_value=capacity,
                        threshold_value=f"<{self.baseline_capacity * (1 - thresholds.capacity_drop_threshold):.3f}",
                        description=f"容量衰减至 {capacity_ratio * 100:.1f}% ，低于设定阈值。",
                        evidence=[f"基线容量: {self.baseline_capacity:.3f}Ah", f"来源阈值: 容量保持率 < {(1 - thresholds.capacity_drop_threshold) * 100:.1f}%"],
                        evidence_source="source_adaptive_rules",
                        rule_id="STAT-CAP-001",
                        confidence_basis=["容量保持率阈值命中", "相对基线容量偏离", "来源自适应阈值"],
                        source_scope=source_tags,
                    )
                )

        if "voltage_mean" in features:
            voltage = float(features["voltage_mean"])
            if voltage < thresholds.voltage_range_min or voltage > thresholds.voltage_range_max:
                severity = "critical" if voltage < 2.0 or voltage > 5.0 else "high" if voltage < 2.3 or voltage > 4.8 else "medium"
                symptom = "欠压" if voltage < thresholds.voltage_range_min else "过压"
                events.append(
                    AnomalyEvent(
                        code=AnomalyType.VOLTAGE_ANOMALY.value,
                        symptom=symptom,
                        severity=severity,
                        metric_name="voltage_mean",
                        metric_value=voltage,
                        threshold_value=f"[{thresholds.voltage_range_min}, {thresholds.voltage_range_max}]",
                        description=f"平均电压 {voltage:.2f}V 超出正常工作区间。",
                        evidence_source="source_adaptive_rules",
                        rule_id="STAT-VOLT-001",
                        confidence_basis=["电压工作区间越界", "来源自适应安全区间"],
                        source_scope=source_tags,
                    )
                )

        if "temperature_mean" in features:
            temperature = float(features["temperature_mean"])
            if temperature > thresholds.temperature_high_threshold or temperature < thresholds.temperature_low_threshold:
                events.append(
                    AnomalyEvent(
                        code=AnomalyType.TEMPERATURE_ANOMALY.value,
                        symptom=ANOMALY_LABELS[AnomalyType.TEMPERATURE_ANOMALY],
                        severity=self._calculate_temperature_severity(temperature, thresholds),
                        metric_name="temperature_mean",
                        metric_value=temperature,
                        threshold_value=f"[{thresholds.temperature_low_threshold}, {thresholds.temperature_high_threshold}]",
                        description=f"平均温度 {temperature:.1f}°C 超出安全范围。",
                        evidence_source="source_adaptive_rules",
                        rule_id="STAT-TEMP-001",
                        confidence_basis=["温度安全阈值越界", "热风险等级映射"],
                        source_scope=source_tags,
                    )
                )

        if "temperature_rise_rate" in features:
            rise_rate = float(features["temperature_rise_rate"])
            if rise_rate > thresholds.temperature_rise_rate:
                events.append(
                    AnomalyEvent(
                        code=AnomalyType.TEMPERATURE_ANOMALY.value,
                        symptom="温升过快",
                        severity=self._calculate_ascending_severity(rise_rate, thresholds.temperature_rise_rate, thresholds.temperature_rise_rate * 3.5),
                        metric_name="temperature_rise_rate",
                        metric_value=rise_rate,
                        threshold_value=f">{thresholds.temperature_rise_rate}",
                        description=f"温升速率达到 {rise_rate:.2f}°C/min ，高于经验阈值。",
                        evidence_source="source_adaptive_rules",
                        rule_id="STAT-TEMP-002",
                        confidence_basis=["温升速率阈值命中", "热失控先导信号"],
                        source_scope=source_tags,
                    )
                )

        if "current_mean" in features and abs(float(features["current_mean"])) > thresholds.current_spike_threshold:
            current = float(features["current_mean"])
            events.append(
                AnomalyEvent(
                    code=AnomalyType.CURRENT_ANOMALY.value,
                    symptom="电流尖峰",
                    severity=self._calculate_ascending_severity(abs(current), thresholds.current_spike_threshold, thresholds.current_spike_threshold * 3.2),
                    metric_name="current_mean",
                    metric_value=current,
                    threshold_value=f"±{thresholds.current_spike_threshold}",
                    description=f"平均电流幅值 {current:.2f}A 过高，可能存在异常工况。",
                    evidence_source="source_adaptive_rules",
                    rule_id="STAT-CURR-001",
                    confidence_basis=["电流幅值超阈值", "工况冲击风险"],
                    source_scope=source_tags,
                )
            )

        if "internal_resistance" in features and self.baseline_resistance:
            resistance = float(features["internal_resistance"])
            resistance_ratio = resistance / max(self.baseline_resistance, 1e-6)
            if resistance_ratio > (1 + thresholds.resistance_increase_threshold):
                events.append(
                    AnomalyEvent(
                        code=AnomalyType.INTERNAL_RESISTANCE.value,
                        symptom="内阻增大",
                        severity=self._calculate_ascending_severity(
                            resistance_ratio,
                            1 + thresholds.resistance_increase_threshold,
                            2.2,
                        ),
                        metric_name="internal_resistance",
                        metric_value=resistance,
                        threshold_value=f">{self.baseline_resistance * (1 + thresholds.resistance_increase_threshold):.4f}",
                        description=f"内阻增加至基线的 {resistance_ratio * 100:.1f}% 。",
                        evidence_source="source_adaptive_rules",
                        rule_id="STAT-IR-001",
                        confidence_basis=["内阻相对基线升高", "极化/老化规则命中"],
                        source_scope=source_tags,
                    )
                )

        return events

    def _thresholds_for_source(self, source_scope: Optional[str]) -> AnomalyThreshold:
        if not source_scope:
            return self.thresholds
        overrides = SOURCE_THRESHOLD_PROFILES.get(source_scope.lower())
        if not overrides:
            return self.thresholds
        return AnomalyThreshold(**{**asdict(self.thresholds), **overrides})

    @staticmethod
    def _calculate_temperature_severity(temperature: float, thresholds: AnomalyThreshold) -> str:
        if temperature > thresholds.temperature_high_threshold + 18 or temperature < thresholds.temperature_low_threshold - 12:
            return "critical"
        if temperature > thresholds.temperature_high_threshold + 10 or temperature < thresholds.temperature_low_threshold - 8:
            return "high"
        if temperature > thresholds.temperature_high_threshold + 4 or temperature < thresholds.temperature_low_threshold - 4:
            return "medium"
        return "low"

    @staticmethod
    def _calculate_ascending_severity(value: float, threshold: float, critical_max: float) -> str:
        ratio = (value - threshold) / max(critical_max - threshold, 1e-6)
        if ratio < 0.25:
            return "low"
        if ratio < 0.55:
            return "medium"
        if ratio < 0.85:
            return "high"
        return "critical"

    @staticmethod
    def _calculate_descending_severity(value: float, threshold: float, critical_min: float) -> str:
        ratio = (threshold - value) / max(threshold - critical_min, 1e-6)
        if ratio < 0.25:
            return "low"
        if ratio < 0.55:
            return "medium"
        if ratio < 0.85:
            return "high"
        return "critical"


class IsolationForestDetector:
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100, random_state: int = 42):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names: list[str] = []

    def fit(self, X: np.ndarray, feature_names: Optional[list[str]] = None) -> None:
        self.feature_names = feature_names or []
        scaled = self.scaler.fit_transform(X)
        self.model.fit(scaled)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            raise RuntimeError("模型尚未训练，请先调用 fit()")
        scaled = self.scaler.transform(X)
        return self.model.predict(scaled), self.model.decision_function(scaled)

    def detect_anomalies(self, X: np.ndarray, feature_names: Optional[list[str]] = None) -> list[AnomalyEvent]:
        predictions, scores = self.predict(X)
        names = feature_names or self.feature_names
        events: list[AnomalyEvent] = []
        for index, (prediction, score) in enumerate(zip(predictions, scores)):
            if prediction != -1:
                continue
            description = f"多维特征异常，孤立森林分数 {score:.3f}。"
            evidence: list[str] = []
            if names and len(names) == X.shape[1]:
                z_scores = np.abs((X[index] - self.scaler.mean_) / np.sqrt(self.scaler.var_ + 1e-8))
                top_index = int(np.argmax(z_scores))
                description += f" 主要异常特征为 {names[top_index]}。"
                evidence.append(f"contributing_feature={names[top_index]}")
            events.append(
                AnomalyEvent(
                    code=AnomalyType.UNKNOWN.value,
                    symptom=ANOMALY_LABELS[AnomalyType.UNKNOWN],
                    severity=self._score_to_severity(float(score)),
                    metric_name="if_score",
                    metric_value=float(score),
                    threshold_value="<0",
                    description=description,
                    source="isolation_forest",
                    evidence=evidence,
                    evidence_source="isolation_forest_model",
                    rule_id="IF-UNKNOWN-001",
                    confidence_basis=["孤立森林异常分数", "多维特征离群程度"],
                    source_scope=["generic"],
                )
            )
        return events

    @staticmethod
    def _score_to_severity(score: float) -> str:
        if score < -0.3:
            return "high"
        if score < -0.1:
            return "medium"
        return "low"


class AnomalyDetector:
    def __init__(
        self,
        use_statistical: bool = True,
        use_isolation_forest: bool = True,
        thresholds: Optional[AnomalyThreshold] = None,
    ):
        self.statistical_detector = StatisticalDetector(thresholds) if use_statistical else None
        self.if_detector = IsolationForestDetector() if use_isolation_forest else None

    def set_baseline(self, capacity: float, resistance: Optional[float] = None) -> None:
        if self.statistical_detector:
            self.statistical_detector.set_baseline(capacity, resistance)

    def fit_isolation_forest(self, X: np.ndarray, feature_names: Optional[list[str]] = None) -> None:
        if self.if_detector:
            self.if_detector.fit(X, feature_names)

    def detect(
        self,
        features: dict[str, float],
        X_multivariate: Optional[np.ndarray] = None,
        source_scope: Optional[str] = None,
    ) -> dict[str, Any]:
        statistical_events = (
            self.statistical_detector.detect(features, source_scope=source_scope)
            if self.statistical_detector and features
            else []
        )
        if_events = self.if_detector.detect_anomalies(X_multivariate) if self.if_detector and X_multivariate is not None else []
        all_events = statistical_events + if_events
        severities = [event.severity for event in all_events]
        max_severity = None
        if severities:
            if "critical" in severities:
                max_severity = "critical"
            elif "high" in severities:
                max_severity = "high"
            elif "medium" in severities:
                max_severity = "medium"
            else:
                max_severity = "low"
        return {
            "events": all_events,
            "statistical_events": statistical_events,
            "isolation_forest_events": if_events,
            "is_anomaly": bool(all_events),
            "max_severity": max_severity,
            "summary": self.get_summary(all_events),
        }

    def get_summary(self, events: list[AnomalyEvent]) -> str:
        if not events:
            return "未检测到异常"
        lines = [f"检测到 {len(events)} 个异常:"]
        for index, event in enumerate(events[:5], start=1):
            lines.append(f"  {index}. [{event.severity.upper()}] {event.symptom}: {event.description}")
        if len(events) > 5:
            lines.append(f"  ... 还有 {len(events) - 5} 个异常")
        return "\n".join(lines)


__all__ = [
    "ANOMALY_LABELS",
    "AnomalyDetector",
    "AnomalyEvent",
    "AnomalyThreshold",
    "AnomalyType",
    "IsolationForestDetector",
    "StatisticalDetector",
]
