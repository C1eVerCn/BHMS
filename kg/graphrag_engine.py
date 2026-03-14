"""面向 MVP 的 GraphRAG 诊断引擎。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class DiagnosisResult:
    fault_type: str
    confidence: float
    description: str
    root_causes: list[str]
    recommendations: list[str]
    related_symptoms: list[str]
    severity: str
    evidence: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class KnowledgeGraph:
    """以 JSON 文件承载的轻量知识图谱。"""

    def __init__(self, knowledge_path: str | Path | None = None):
        default_path = Path(__file__).resolve().parents[1] / "data" / "knowledge" / "battery_fault_knowledge.json"
        self.knowledge_path = Path(knowledge_path) if knowledge_path else default_path
        payload = json.loads(self.knowledge_path.read_text(encoding="utf-8"))
        self.symptom_aliases: dict[str, str] = payload.get("symptom_aliases", {})
        self.faults: list[dict[str, Any]] = payload.get("faults", [])

    def normalize_symptom(self, symptom: str) -> str:
        return self.symptom_aliases.get(symptom, symptom)

    def query_fault_by_symptom(self, symptom: str) -> list[dict[str, Any]]:
        normalized = self.normalize_symptom(symptom)
        results = []
        for fault in self.faults:
            if normalized in [self.normalize_symptom(item) for item in fault.get("symptoms", [])]:
                results.append(
                    {
                        "fault": fault["name"],
                        "category": fault.get("category", "未知"),
                        "severity": fault.get("severity", "medium"),
                        "description": fault.get("description", ""),
                    }
                )
        return results

    def get_fault_details(self, fault_name: str) -> dict[str, Any]:
        for fault in self.faults:
            if fault.get("name") == fault_name:
                return dict(fault)
        return {}

    def rank_faults(self, symptoms: list[str], severity_map: Optional[dict[str, str]] = None) -> list[dict[str, Any]]:
        severity_bonus = {"low": 0.05, "medium": 0.10, "high": 0.18, "critical": 0.24}
        severity_map = severity_map or {}
        scored: list[dict[str, Any]] = []
        normalized_symptoms = [self.normalize_symptom(item) for item in symptoms]
        for fault in self.faults:
            fault_symptoms = [self.normalize_symptom(item) for item in fault.get("symptoms", [])]
            matched = [symptom for symptom in normalized_symptoms if symptom in fault_symptoms]
            if not matched:
                continue
            score = 0.35 + 0.18 * len(matched)
            score += sum(severity_bonus.get(severity_map.get(symptom, "low"), 0.0) for symptom in matched)
            scored.append({**fault, "matched_symptoms": matched, "score": min(score, 0.95)})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored


class LLMInterface:
    """MVP 使用模板化文本生成，保留 LLM 接口边界。"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.use_mock = api_key is None

    def generate_diagnosis(
        self,
        anomalies: list[dict[str, Any]],
        fault_candidates: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> DiagnosisResult:
        if not fault_candidates:
            return DiagnosisResult(
                fault_type="未识别故障",
                confidence=0.0,
                description="当前异常现象暂时无法与本地知识库建立稳定映射。",
                root_causes=["知识库缺少对应案例"],
                recommendations=["补充更多运行工况或扩充故障知识条目"],
                related_symptoms=[item.get("symptom") or item.get("type", "未知异常") for item in anomalies],
                severity="unknown",
                evidence=["未找到匹配的故障候选"],
            )

        primary = fault_candidates[0]
        matched = primary.get("matched_symptoms", [])
        related_symptoms = [item.get("symptom") or item.get("type", "未知异常") for item in anomalies]
        evidence = [item.get("description", "") for item in anomalies if item.get("description")]
        for template in primary.get("evidence_templates", []):
            symptom = matched[0] if matched else (related_symptoms[0] if related_symptoms else "异常症状")
            evidence.append(template.format(symptom=symptom))
        battery_info = context.get("battery_info", {})
        usage_hint = ""
        if battery_info:
            usage_hint = f" 当前样本电池信息为: {json.dumps(battery_info, ensure_ascii=False)}。"
        description = primary.get("description", "") + usage_hint
        return DiagnosisResult(
            fault_type=primary.get("name", "未知故障"),
            confidence=float(primary.get("score", 0.0)),
            description=description,
            root_causes=list(primary.get("causes", ["未知原因"])),
            recommendations=list(primary.get("recommendations", ["建议进一步检查"])),
            related_symptoms=related_symptoms,
            severity=primary.get("severity", "medium"),
            evidence=evidence,
        )


class GraphRAGEngine:
    def __init__(self, llm_api_key: Optional[str] = None, knowledge_path: str | Path | None = None):
        self.kg = KnowledgeGraph(knowledge_path=knowledge_path)
        self.llm = LLMInterface(llm_api_key)

    def diagnose(self, anomalies: list[dict[str, Any]], battery_info: Optional[dict[str, Any]] = None) -> DiagnosisResult:
        symptoms = []
        severity_map: dict[str, str] = {}
        for anomaly in anomalies:
            symptom = str(anomaly.get("symptom") or anomaly.get("type") or "未知异常")
            normalized = self.kg.normalize_symptom(symptom)
            symptoms.append(normalized)
            severity_map[normalized] = str(anomaly.get("severity", "low"))
        ranked_faults = self.kg.rank_faults(symptoms, severity_map=severity_map)
        return self.llm.generate_diagnosis(
            anomalies=anomalies,
            fault_candidates=ranked_faults,
            context={
                "battery_info": battery_info or {},
                "symptoms": symptoms,
                "severity_map": severity_map,
            },
        )

    def generate_report(self, diagnosis: DiagnosisResult) -> str:
        lines = [
            "# 电池故障诊断报告",
            "",
            "## 诊断结果",
            f"- 故障类型: {diagnosis.fault_type}",
            f"- 置信度: {diagnosis.confidence * 100:.1f}%",
            f"- 严重程度: {diagnosis.severity}",
            "",
            "## 故障描述",
            diagnosis.description,
            "",
            "## 相关症状",
        ]
        lines.extend(f"- {symptom}" for symptom in diagnosis.related_symptoms)
        lines.extend(["", "## 根本原因"])
        lines.extend(f"- {cause}" for cause in diagnosis.root_causes)
        lines.extend(["", "## 处理建议"])
        lines.extend(f"- {item}" for item in diagnosis.recommendations)
        lines.extend(["", "## 诊断依据"])
        lines.extend(f"- {item}" for item in diagnosis.evidence)
        return "\n".join(lines)


__all__ = ["DiagnosisResult", "GraphRAGEngine", "KnowledgeGraph", "LLMInterface"]
