"""GraphRAG 诊断引擎单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kg.graphrag_engine import DiagnosisResult, GraphRAGEngine, KnowledgeGraph, LLMInterface  # noqa: E402


def test_knowledge_graph_can_query_capacity_faults():
    graph = KnowledgeGraph()
    results = graph.query_fault_by_symptom("容量骤降")
    assert results
    assert results[0]["fault"] == "容量衰减异常"


def test_knowledge_graph_returns_fault_details():
    graph = KnowledgeGraph()
    details = graph.get_fault_details("热失控风险")
    assert details["name"] == "热失控风险"
    assert "温度异常" in details["symptoms"]


def test_llm_interface_returns_structured_diagnosis():
    llm = LLMInterface()
    diagnosis = llm.generate_diagnosis(
        anomalies=[{"symptom": "容量骤降", "severity": "high", "description": "容量快速下降"}],
        fault_candidates=[
            {
                "name": "容量衰减异常",
                "severity": "high",
                "score": 0.82,
                "description": "电池容量衰减速度超出正常范围。",
                "causes": ["过充"],
                "recommendations": ["降低充电倍率"],
                "matched_symptoms": ["容量骤降"],
                "all_symptoms": ["容量骤降", "温度异常", "电流异常"],
                "rule_id": "RULE-AGE-001",
                "evidence_source": ["公开退化机理规则"],
                "confidence_basis": ["症状覆盖率", "严重度加权"],
                "source_scope": ["generic"],
                "threshold_hints": ["capacity_ratio < 0.85"],
                "symptom_coverage": 0.667,
                "matched_symptom_count": 1,
                "score_breakdown": {"coverage_score": 0.2, "severity_score": 0.1},
                "evidence_templates": ["检测到{symptom}。"],
            }
        ],
        context={"battery_info": {"model": "NASA"}},
    )
    assert isinstance(diagnosis, DiagnosisResult)
    assert diagnosis.fault_type == "容量衰减异常"
    assert diagnosis.confidence > 0
    assert diagnosis.decision_basis
    assert diagnosis.candidate_faults[0].rule_id == "RULE-AGE-001"


def test_graphrag_engine_end_to_end():
    engine = GraphRAGEngine()
    diagnosis = engine.diagnose(
        anomalies=[
            {"symptom": "温度异常", "severity": "high", "description": "平均温度达到 56C"},
            {"symptom": "电压异常", "severity": "medium", "description": "存在欠压波动"},
        ],
        battery_info={"battery_id": "B0005", "source": "nasa"},
    )
    assert diagnosis.fault_type in {"热失控风险", "工况波动异常", "电压采样异常"}
    assert diagnosis.decision_basis
    assert diagnosis.candidate_faults[0].rule_id is not None
    report = engine.generate_report(diagnosis)
    assert "电池故障诊断报告" in report


def test_graphrag_keeps_sensor_drift_when_sampling_evidence_is_explicit():
    engine = GraphRAGEngine()
    diagnosis = engine.diagnose(
        anomalies=[
            {"symptom": "电压异常", "severity": "medium", "description": "采样值整体偏移，怀疑传感器漂移"},
            {"symptom": "温度异常", "severity": "low", "description": "标定后仍存在持续偏移"},
        ],
        battery_info={"battery_id": "B-SENSOR", "source": "nasa"},
    )

    assert diagnosis.candidate_faults
    top_names = {item.name for item in diagnosis.candidate_faults[:2]}
    assert "传感器漂移" in top_names


def test_graphrag_engine_falls_back_to_memory_when_neo4j_is_unavailable(monkeypatch: pytest.MonkeyPatch):
    engine = GraphRAGEngine(
        graph_backend="neo4j",
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="bhmsneo4j",
    )

    def _raise_unavailable(*args, **kwargs):
        raise RuntimeError("neo4j unavailable in test")

    monkeypatch.setattr(engine.kg, "rank_faults", _raise_unavailable)
    diagnosis = engine.diagnose(
        anomalies=[{"symptom": "容量骤降", "severity": "high", "description": "容量快速下降"}],
        battery_info={"battery_id": "B-NO-NEO4J"},
    )

    assert engine.active_backend == "memory"
    assert diagnosis.fault_type == "容量衰减异常"
    assert any("Neo4j 不可用" in item for item in diagnosis.evidence)
