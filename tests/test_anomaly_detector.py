"""异常检测模块单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.inference.anomaly_detector import (  # noqa: E402
    ANOMALY_LABELS,
    AnomalyDetector,
    AnomalyThreshold,
    AnomalyType,
    IsolationForestDetector,
    StatisticalDetector,
)


def test_statistical_detector_detects_capacity_and_temperature_events():
    detector = StatisticalDetector()
    detector.set_baseline(capacity=2.0)
    events = detector.detect({"capacity": 1.45, "voltage_mean": 3.7, "temperature_mean": 52.0}, source_scope="nasa")
    assert len(events) >= 2
    symptoms = {event.symptom for event in events}
    assert "容量衰减过快" in symptoms
    assert ANOMALY_LABELS[AnomalyType.TEMPERATURE_ANOMALY] in symptoms
    assert all(event.rule_id for event in events)
    assert all(event.source_scope for event in events)


def test_isolation_forest_detector_reports_unknown_event():
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(200, 4))
    detector = IsolationForestDetector(contamination=0.05)
    detector.fit(X_train, feature_names=["v", "i", "t", "c"])
    X_test = rng.normal(size=(10, 4))
    X_test[0] = [8.0, 8.0, 8.0, 8.0]
    events = detector.detect_anomalies(X_test)
    assert events
    assert events[0].code == AnomalyType.UNKNOWN.value


def test_anomaly_detector_summary():
    detector = AnomalyDetector(use_isolation_forest=False, thresholds=AnomalyThreshold())
    detector.set_baseline(capacity=2.0)
    result = detector.detect({"capacity": 1.4, "temperature_mean": 50.0, "current_mean": 1.0}, source_scope="calce")
    assert result["is_anomaly"] is True
    assert result["max_severity"] in {"low", "medium", "high", "critical"}
    assert "检测到" in result["summary"]


def test_statistical_detector_detects_internal_resistance_event():
    detector = StatisticalDetector()
    detector.set_baseline(capacity=2.0, resistance=0.02)
    events = detector.detect({"capacity": 1.9, "internal_resistance": 0.029}, source_scope="kaggle")
    assert any(event.code == AnomalyType.INTERNAL_RESISTANCE.value for event in events)


def test_source_profiles_adjust_temperature_thresholds():
    detector = StatisticalDetector()
    detector.set_baseline(capacity=2.0)
    calce_events = detector.detect({"temperature_mean": 44.0}, source_scope="calce")
    nasa_events = detector.detect({"temperature_mean": 44.0}, source_scope="nasa")

    assert any(event.code == AnomalyType.TEMPERATURE_ANOMALY.value for event in calce_events)
    assert not nasa_events
