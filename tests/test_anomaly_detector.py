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
    events = detector.detect({"capacity": 1.45, "voltage_mean": 3.7, "temperature_mean": 52.0})
    assert len(events) >= 2
    symptoms = {event.symptom for event in events}
    assert ANOMALY_LABELS[AnomalyType.CAPACITY_DROP] in symptoms
    assert ANOMALY_LABELS[AnomalyType.TEMPERATURE_ANOMALY] in symptoms


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
    result = detector.detect({"capacity": 1.4, "temperature_mean": 50.0, "current_mean": 1.0})
    assert result["is_anomaly"] is True
    assert result["max_severity"] in {"low", "medium", "high"}
    assert "检测到" in result["summary"]
