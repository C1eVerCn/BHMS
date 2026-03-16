"""GraphRAG lifecycle reasoning tests."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kg.graphrag_engine import GraphRAGEngine  # noqa: E402


def test_graphrag_consumes_lifecycle_and_model_evidence():
    engine = GraphRAGEngine(
        graph_backend="memory",
        knowledge_path=PROJECT_ROOT / "data" / "knowledge" / "battery_fault_knowledge.json",
    )
    result = engine.diagnose(
        anomalies=[
            {"symptom": "容量骤降", "severity": "high", "description": "容量在最近窗口快速下降"},
            {"symptom": "温度异常", "severity": "high", "description": "温升明显偏高"},
        ],
        battery_info={"source": "calce", "chemistry": "Li-ion", "protocol_id": "fast_charge"},
        lifecycle_evidence={
            "predicted_knee_cycle": 118,
            "predicted_eol_cycle": 176,
            "accelerated_degradation_window": "late-stage 120-150",
            "future_capacity_fade_pattern": "accelerated_tail_fade",
            "temperature_risk": "high",
            "resistance_risk": "medium",
            "voltage_risk": "low",
        },
        model_evidence={
            "top_features": ["capacity", "temperature_mean"],
            "critical_windows": ["cycles 90-120"],
            "confidence_factors": ["recent degradation slope steepens"],
        },
    )

    assert result.candidate_faults
    breakdown = result.candidate_faults[0].score_breakdown
    assert {"symptom_match", "future_risk_match", "stage_consistency", "source_scope_match", "threshold_hint_match"} <= set(breakdown)
    assert any(node.node_type == "RiskWindow" for node in result.graph_trace.nodes)
    assert any(node.node_type == "Mechanism" for node in result.graph_trace.nodes)
    assert any(edge.relation == "FUTURE_RISK" for edge in result.graph_trace.edges)
    assert "生命周期预测证据" in result.report_markdown
