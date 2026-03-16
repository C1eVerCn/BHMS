"""面向毕业设计成品的 GraphRAG 诊断引擎。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class LifecycleEvidence:
    predicted_knee_cycle: Optional[float] = None
    predicted_eol_cycle: Optional[float] = None
    accelerated_degradation_window: Optional[str] = None
    future_capacity_fade_pattern: Optional[str] = None
    temperature_risk: Optional[str] = None
    resistance_risk: Optional[str] = None
    voltage_risk: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelEvidence:
    top_features: list[str] = field(default_factory=list)
    critical_windows: list[str] = field(default_factory=list)
    confidence_factors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateFault:
    name: str
    score: float
    severity: str
    description: str
    category: str
    matched_symptoms: list[str] = field(default_factory=list)
    all_symptoms: list[str] = field(default_factory=list)
    root_causes: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    evidence_source: list[str] = field(default_factory=list)
    rule_id: Optional[str] = None
    confidence_basis: list[str] = field(default_factory=list)
    source_scope: list[str] = field(default_factory=list)
    threshold_hints: list[str] = field(default_factory=list)
    mechanisms: list[str] = field(default_factory=list)
    stage_scope: list[str] = field(default_factory=list)
    precursor_signals: list[str] = field(default_factory=list)
    future_risk_patterns: list[str] = field(default_factory=list)
    operating_condition_tags: list[str] = field(default_factory=list)
    references: list[str] = field(default_factory=list)
    symptom_coverage: float = 0.0
    matched_symptom_count: int = 0
    score_breakdown: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GraphTraceNode:
    id: str
    label: str
    node_type: str
    evidence_source: list[str] = field(default_factory=list)
    rule_id: Optional[str] = None
    confidence_basis: list[str] = field(default_factory=list)
    source_scope: list[str] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphTraceEdge:
    source: str
    target: str
    relation: str


@dataclass
class GraphTrace:
    matched_symptoms: list[str]
    nodes: list[GraphTraceNode]
    edges: list[GraphTraceEdge]
    ranking_basis: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "matched_symptoms": self.matched_symptoms,
            "nodes": [asdict(node) for node in self.nodes],
            "edges": [asdict(edge) for edge in self.edges],
            "ranking_basis": self.ranking_basis,
        }


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
    candidate_faults: list[CandidateFault]
    graph_trace: GraphTrace
    decision_basis: list[str]
    report_markdown: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "fault_type": self.fault_type,
            "confidence": self.confidence,
            "description": self.description,
            "root_causes": self.root_causes,
            "recommendations": self.recommendations,
            "related_symptoms": self.related_symptoms,
            "severity": self.severity,
            "evidence": self.evidence,
            "candidate_faults": [item.to_dict() for item in self.candidate_faults],
            "graph_trace": self.graph_trace.to_dict(),
            "decision_basis": self.decision_basis,
            "report_markdown": self.report_markdown,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class SeedKnowledge:
    def __init__(self, knowledge_path: str | Path | None = None):
        default_path = Path(__file__).resolve().parents[1] / "data" / "knowledge" / "battery_fault_knowledge.json"
        self.knowledge_path = Path(knowledge_path) if knowledge_path else default_path
        payload = json.loads(self.knowledge_path.read_text(encoding="utf-8"))
        self.symptom_aliases: dict[str, str] = payload.get("symptom_aliases", {})
        self.faults: list[dict[str, Any]] = [self._normalize_fault_entry(item) for item in payload.get("faults", [])]

    def normalize_symptom(self, symptom: str) -> str:
        return self.symptom_aliases.get(symptom, symptom)

    @staticmethod
    def _normalize_fault_entry(fault: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(fault)
        normalized.setdefault("mechanisms", list(normalized.get("causes", [])))
        normalized.setdefault("stage_scope", ["whole_lifecycle"])
        normalized.setdefault("precursor_signals", list(normalized.get("symptoms", [])))
        normalized.setdefault("future_risk_patterns", list(normalized.get("threshold_hints", [])) + list(normalized.get("symptoms", [])))
        normalized.setdefault("operating_condition_tags", [])
        normalized.setdefault("references", list(normalized.get("evidence_source", [])))
        return normalized

    def get_fault_details(self, fault_name: str) -> dict[str, Any]:
        return self.seed.get_fault_details(fault_name)


class KnowledgeGraph:
    """兼容测试与本地分析的轻量知识图谱。"""

    def __init__(self, knowledge_path: str | Path | None = None):
        self.seed = SeedKnowledge(knowledge_path)
        self.symptom_aliases = self.seed.symptom_aliases
        self.faults = self.seed.faults

    def normalize_symptom(self, symptom: str) -> str:
        return self.seed.normalize_symptom(symptom)

    def query_fault_by_symptom(self, symptom: str) -> list[dict[str, Any]]:
        normalized = self.normalize_symptom(symptom)
        results = []
        for fault in self.faults:
            fault_symptoms = [self.normalize_symptom(item) for item in fault.get("symptoms", [])]
            if normalized not in fault_symptoms:
                continue
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

    def rank_faults(
        self,
        symptoms: list[str],
        severity_map: Optional[dict[str, str]] = None,
        battery_source: Optional[str] = None,
        lifecycle_evidence: Optional[LifecycleEvidence] = None,
    ) -> list[dict[str, Any]]:
        severity_map = severity_map or {}
        normalized = [self.normalize_symptom(symptom) for symptom in symptoms]
        ranked: list[dict[str, Any]] = []
        for fault in self.faults:
            all_symptoms = [self.normalize_symptom(item) for item in fault.get("symptoms", [])]
            matched = [item for item in normalized if item in all_symptoms]
            if not matched:
                continue
            score, breakdown = _fault_score(
                matched,
                all_symptoms,
                severity_map,
                battery_source=battery_source,
                source_scope=fault.get("source_scope"),
                lifecycle_evidence=lifecycle_evidence,
                stage_scope=fault.get("stage_scope"),
                future_risk_patterns=fault.get("future_risk_patterns"),
                precursor_signals=fault.get("precursor_signals"),
                threshold_hints=fault.get("threshold_hints"),
            )
            ranked.append(
                {
                    **fault,
                    "matched_symptoms": matched,
                    "all_symptoms": all_symptoms,
                    "score": score,
                    "symptom_coverage": breakdown["coverage_ratio"],
                    "matched_symptom_count": len(set(matched)),
                    "score_breakdown": breakdown,
                }
            )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked


class Neo4jKnowledgeGraph:
    def __init__(
        self,
        knowledge_path: str | Path | None = None,
        *,
        uri: str,
        user: str,
        password: str,
        database: str = "neo4j",
    ):
        self.seed = SeedKnowledge(knowledge_path)
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self._driver = None
        self._seeded = False

    def normalize_symptom(self, symptom: str) -> str:
        return self.seed.normalize_symptom(symptom)

    def _get_driver(self):
        if self._driver is None:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def ensure_seeded(self) -> None:
        if self._seeded:
            return
        driver = self._get_driver()
        with driver.session(database=self.database) as session:
            session.run("CREATE CONSTRAINT fault_name IF NOT EXISTS FOR (n:Fault) REQUIRE n.name IS UNIQUE")
            session.run("CREATE CONSTRAINT symptom_name IF NOT EXISTS FOR (n:Symptom) REQUIRE n.name IS UNIQUE")
            session.run("CREATE CONSTRAINT cause_name IF NOT EXISTS FOR (n:Cause) REQUIRE n.name IS UNIQUE")
            session.run("CREATE CONSTRAINT recommendation_name IF NOT EXISTS FOR (n:Recommendation) REQUIRE n.name IS UNIQUE")
            count = session.run("MATCH (n:Fault) RETURN count(n) AS count").single()["count"]
            if count == 0:
                for fault in self.seed.faults:
                    session.run(
                        """
                        MERGE (f:Fault {name: $name})
                        SET f.category = $category,
                            f.severity = $severity,
                            f.description = $description,
                            f.rule_id = $rule_id,
                            f.evidence_source = $evidence_source,
                            f.source_scope = $source_scope,
                            f.confidence_basis = $confidence_basis,
                            f.threshold_hints = $threshold_hints,
                            f.symptom_weights_json = $symptom_weights_json
                        """,
                        name=fault["name"],
                        category=fault.get("category", "未知"),
                        severity=fault.get("severity", "medium"),
                        description=fault.get("description", ""),
                        rule_id=fault.get("rule_id"),
                        evidence_source=fault.get("evidence_source", []),
                        source_scope=fault.get("source_scope", ["generic"]),
                        confidence_basis=fault.get("confidence_basis", []),
                        threshold_hints=fault.get("threshold_hints", []),
                        symptom_weights_json=json.dumps(fault.get("symptom_weights", {}), ensure_ascii=False),
                    )
                    for symptom in fault.get("symptoms", []):
                        normalized = self.normalize_symptom(symptom)
                        aliases = sorted({alias for alias, target in self.seed.symptom_aliases.items() if target == normalized})
                        session.run(
                            """
                            MERGE (s:Symptom {name: $name})
                            SET s.aliases = $aliases
                            WITH s
                            MATCH (f:Fault {name: $fault_name})
                            MERGE (f)-[:HAS_SYMPTOM]->(s)
                            """,
                            name=normalized,
                            aliases=aliases,
                            fault_name=fault["name"],
                        )
                    for cause in fault.get("causes", []):
                        session.run(
                            """
                            MERGE (c:Cause {name: $name})
                            WITH c
                            MATCH (f:Fault {name: $fault_name})
                            MERGE (f)-[:HAS_CAUSE]->(c)
                            """,
                            name=cause,
                            fault_name=fault["name"],
                        )
                    for recommendation in fault.get("recommendations", []):
                        session.run(
                            """
                            MERGE (r:Recommendation {name: $name})
                            WITH r
                            MATCH (f:Fault {name: $fault_name})
                            MERGE (f)-[:HAS_RECOMMENDATION]->(r)
                            """,
                            name=recommendation,
                            fault_name=fault["name"],
                        )
        self._seeded = True

    def rank_faults(
        self,
        symptoms: list[str],
        severity_map: Optional[dict[str, str]] = None,
        battery_source: Optional[str] = None,
        lifecycle_evidence: Optional[LifecycleEvidence] = None,
    ) -> list[dict[str, Any]]:
        severity_map = severity_map or {}
        if not symptoms:
            return []
        self.ensure_seeded()
        normalized = [self.normalize_symptom(symptom) for symptom in symptoms]
        query = """
        UNWIND $symptoms AS symptom_name
        MATCH (matched:Symptom)
        WHERE toLower(matched.name) = toLower(symptom_name)
           OR any(alias IN coalesce(matched.aliases, []) WHERE toLower(alias) = toLower(symptom_name))
        WITH collect(DISTINCT matched) AS matched_nodes
        MATCH (fault:Fault)-[:HAS_SYMPTOM]->(symptom:Symptom)
        WHERE symptom IN matched_nodes
        OPTIONAL MATCH (fault)-[:HAS_SYMPTOM]->(all_symptom:Symptom)
        OPTIONAL MATCH (fault)-[:HAS_CAUSE]->(cause:Cause)
        OPTIONAL MATCH (fault)-[:HAS_RECOMMENDATION]->(recommendation:Recommendation)
        RETURN fault.name AS name,
               fault.category AS category,
               fault.severity AS severity,
               fault.description AS description,
               fault.rule_id AS rule_id,
               fault.evidence_source AS evidence_source,
               fault.source_scope AS source_scope,
               fault.confidence_basis AS confidence_basis,
               fault.threshold_hints AS threshold_hints,
               fault.symptom_weights_json AS symptom_weights_json,
               collect(DISTINCT symptom.name) AS matched_symptoms,
               collect(DISTINCT all_symptom.name) AS all_symptoms,
               collect(DISTINCT cause.name) AS causes,
               collect(DISTINCT recommendation.name) AS recommendations
        """
        driver = self._get_driver()
        ranked: list[dict[str, Any]] = []
        with driver.session(database=self.database) as session:
            for record in session.run(query, symptoms=normalized):
                all_symptoms = [item for item in record["all_symptoms"] if item]
                matched_symptoms = [item for item in record["matched_symptoms"] if item]
                symptom_weights = json.loads(record["symptom_weights_json"]) if record["symptom_weights_json"] else {}
                seed_details = self.seed.get_fault_details(record["name"])
                score, breakdown = _fault_score(
                    matched_symptoms,
                    all_symptoms,
                    severity_map,
                    battery_source=battery_source,
                    source_scope=record["source_scope"],
                    lifecycle_evidence=lifecycle_evidence,
                    stage_scope=seed_details.get("stage_scope"),
                    future_risk_patterns=seed_details.get("future_risk_patterns"),
                    precursor_signals=seed_details.get("precursor_signals"),
                    threshold_hints=record["threshold_hints"],
                )
                ranked.append(
                    {
                        "name": record["name"],
                        "category": record["category"] or "未知",
                        "severity": record["severity"] or "medium",
                        "description": record["description"] or "",
                        "rule_id": record["rule_id"],
                        "evidence_source": record["evidence_source"] or [],
                        "source_scope": record["source_scope"] or ["generic"],
                        "confidence_basis": record["confidence_basis"] or [],
                        "threshold_hints": record["threshold_hints"] or [],
                        "matched_symptoms": matched_symptoms,
                        "all_symptoms": all_symptoms,
                        "causes": [item for item in record["causes"] if item],
                        "recommendations": [item for item in record["recommendations"] if item],
                        "mechanisms": list(seed_details.get("mechanisms", [])),
                        "stage_scope": list(seed_details.get("stage_scope", [])),
                        "precursor_signals": list(seed_details.get("precursor_signals", [])),
                        "future_risk_patterns": list(seed_details.get("future_risk_patterns", [])),
                        "operating_condition_tags": list(seed_details.get("operating_condition_tags", [])),
                        "references": list(seed_details.get("references", [])),
                        "score": score,
                        "symptom_coverage": breakdown["coverage_ratio"],
                        "matched_symptom_count": len(set(matched_symptoms)),
                        "score_breakdown": breakdown,
                    }
                )
        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked


class LLMInterface:
    """使用模板化方式生成结构化诊断结果，不暴露原始内部思维链。"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.use_mock = api_key is None

    def generate_diagnosis(
        self,
        anomalies: list[dict[str, Any]],
        fault_candidates: list[dict[str, Any]],
        context: dict[str, Any],
    ) -> DiagnosisResult:
        related_symptoms = [item.get("symptom") or item.get("type", "未知异常") for item in anomalies]
        lifecycle_evidence = _coerce_lifecycle_evidence(context.get("lifecycle_evidence"))
        model_evidence = _coerce_model_evidence(context.get("model_evidence"))
        graph_trace = build_graph_trace(
            fault_candidates[:3],
            related_symptoms,
            battery_info=context.get("battery_info") or {},
            lifecycle_evidence=lifecycle_evidence,
            model_evidence=model_evidence,
        )
        backend_warning = str(context.get("backend_warning") or "").strip()
        decision_basis = build_decision_basis(
            fault_candidates[:3],
            context.get("battery_info") or {},
            lifecycle_evidence=lifecycle_evidence,
            model_evidence=model_evidence,
        )
        if not fault_candidates:
            evidence = ["未找到匹配的 Fault 节点"]
            if backend_warning:
                evidence.append(backend_warning)
            evidence.extend(decision_basis)
            report = self._render_report(
                fault_type="未识别故障",
                confidence=0.0,
                severity="unknown",
                description="当前异常现象暂时无法与图谱建立稳定映射。",
                root_causes=["图谱中缺少对应知识条目"],
                recommendations=["补充更多运行工况或扩充故障知识图谱"],
                related_symptoms=related_symptoms,
                evidence=evidence,
                candidates=[],
                graph_trace=graph_trace,
                decision_basis=decision_basis,
            )
            return DiagnosisResult(
                fault_type="未识别故障",
                confidence=0.0,
                description="当前异常现象暂时无法与图谱建立稳定映射。",
                root_causes=["图谱中缺少对应知识条目"],
                recommendations=["补充更多运行工况或扩充故障知识图谱"],
                related_symptoms=related_symptoms,
                severity="unknown",
                evidence=evidence,
                candidate_faults=[],
                graph_trace=graph_trace,
                decision_basis=decision_basis,
                report_markdown=report,
            )

        primary = fault_candidates[0]
        candidate_models = [
            CandidateFault(
                name=item["name"],
                score=float(item["score"]),
                severity=item.get("severity", "medium"),
                description=item.get("description", ""),
                category=item.get("category", "未知"),
                matched_symptoms=list(item.get("matched_symptoms", [])),
                all_symptoms=list(item.get("all_symptoms", [])),
                root_causes=list(item.get("causes", [])),
                recommendations=list(item.get("recommendations", [])),
                evidence_source=list(item.get("evidence_source", [])),
                rule_id=item.get("rule_id"),
                confidence_basis=list(item.get("confidence_basis", [])),
                source_scope=list(item.get("source_scope", [])),
                threshold_hints=list(item.get("threshold_hints", [])),
                mechanisms=list(item.get("mechanisms", [])),
                stage_scope=list(item.get("stage_scope", [])),
                precursor_signals=list(item.get("precursor_signals", [])),
                future_risk_patterns=list(item.get("future_risk_patterns", [])),
                operating_condition_tags=list(item.get("operating_condition_tags", [])),
                references=list(item.get("references", [])),
                symptom_coverage=float(item.get("symptom_coverage", 0.0) or 0.0),
                matched_symptom_count=int(item.get("matched_symptom_count", len(item.get("matched_symptoms", [])))),
                score_breakdown=dict(item.get("score_breakdown", {})),
            )
            for item in fault_candidates[:3]
        ]
        evidence = [item.get("description", "") for item in anomalies if item.get("description")]
        evidence.append(f"图谱匹配到 {primary['name']}，症状覆盖率 {len(primary.get('matched_symptoms', []))}/{max(len(primary.get('all_symptoms', [])), 1)}。")
        if backend_warning:
            evidence.append(backend_warning)
        evidence.extend(decision_basis)
        usage_hint = ""
        battery_info = context.get("battery_info") or {}
        if battery_info:
            usage_hint = f" 当前样本电池信息：{json.dumps(battery_info, ensure_ascii=False)}。"
        description = primary.get("description", "") + usage_hint
        report = self._render_report(
            fault_type=primary["name"],
            confidence=float(primary["score"]),
            severity=primary.get("severity", "medium"),
            description=description,
            root_causes=list(primary.get("causes", [])),
            recommendations=list(primary.get("recommendations", [])),
            related_symptoms=related_symptoms,
            evidence=evidence,
            candidates=candidate_models,
            graph_trace=graph_trace,
            decision_basis=decision_basis,
            lifecycle_evidence=lifecycle_evidence,
            model_evidence=model_evidence,
        )
        return DiagnosisResult(
            fault_type=primary["name"],
            confidence=float(primary["score"]),
            description=description,
            root_causes=list(primary.get("causes", [])),
            recommendations=list(primary.get("recommendations", [])),
            related_symptoms=related_symptoms,
            severity=primary.get("severity", "medium"),
            evidence=evidence,
            candidate_faults=candidate_models,
            graph_trace=graph_trace,
            decision_basis=decision_basis,
            report_markdown=report,
        )

    @staticmethod
    def _render_report(
        *,
        fault_type: str,
        confidence: float,
        severity: str,
        description: str,
        root_causes: list[str],
        recommendations: list[str],
        related_symptoms: list[str],
        evidence: list[str],
        candidates: list[CandidateFault],
        graph_trace: GraphTrace,
        decision_basis: list[str],
        lifecycle_evidence: Optional[LifecycleEvidence] = None,
        model_evidence: Optional[ModelEvidence] = None,
    ) -> str:
        lines = [
            "# 电池故障诊断报告",
            "",
            "## 一、诊断结论",
            f"- 故障类型：{fault_type}",
            f"- 置信度：{confidence * 100:.1f}%",
            f"- 严重程度：{severity}",
            "",
            "## 二、异常摘要",
        ]
        lines.extend(f"- {item}" for item in related_symptoms or ["未检测到异常症状"])
        lines.extend(["", "## 三、GraphRAG 检索说明", description, "", "## 四、候选故障排序"])
        if candidates:
            for item in candidates:
                lines.append(f"- {item.name}：得分 {item.score:.3f}，匹配症状 {', '.join(item.matched_symptoms) or '无'}")
        else:
            lines.append("- 未找到候选故障")
        lines.extend(["", "## 五、根因链"])
        lines.extend(f"- {item}" for item in root_causes or ["暂无明确根因"])
        lines.extend(["", "## 六、处理建议"])
        lines.extend(f"- {item}" for item in recommendations or ["建议继续监测"])
        lines.extend(["", "## 七、证据条目"])
        lines.extend(f"- {item}" for item in evidence or ["暂无证据"])
        lines.extend(["", "## 八、生命周期预测证据"])
        lifecycle_items = _lifecycle_evidence_lines(lifecycle_evidence)
        lines.extend(f"- {item}" for item in lifecycle_items or ["暂无生命周期预测证据"])
        lines.extend(["", "## 九、模型解释证据"])
        model_items = _model_evidence_lines(model_evidence)
        lines.extend(f"- {item}" for item in model_items or ["暂无模型解释证据"])
        lines.extend(["", "## 十、排序与决策依据"])
        lines.extend(f"- {item}" for item in decision_basis or ["暂无额外排序依据"])
        lines.extend(["", "## 十一、图谱子图摘要"])
        lines.extend(f"- 节点数：{len(graph_trace.nodes)}" for _ in [0])
        lines.extend(f"- 关系数：{len(graph_trace.edges)}" for _ in [0])
        lines.extend(["", "## 十二、限制说明", "- 当前结果展示的是可审计证据链，而不是模型内部隐式思维过程。"])
        return "\n".join(lines)


class GraphRAGEngine:
    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        knowledge_path: str | Path | None = None,
        *,
        graph_backend: str = "memory",
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        neo4j_database: str = "neo4j",
    ):
        self.seed = SeedKnowledge(knowledge_path)
        self.graph_backend = graph_backend.lower()
        self.fallback_kg = KnowledgeGraph(knowledge_path)
        self.active_backend = self.graph_backend
        if self.graph_backend == "neo4j":
            if not neo4j_uri or not neo4j_user or neo4j_password is None:
                raise ValueError("使用 Neo4j 图谱时必须提供 URI、用户名和密码")
            self.kg = Neo4jKnowledgeGraph(
                knowledge_path=knowledge_path,
                uri=neo4j_uri,
                user=neo4j_user,
                password=neo4j_password,
                database=neo4j_database,
            )
        else:
            self.kg = self.fallback_kg
        self.llm = LLMInterface(llm_api_key)

    def diagnose(
        self,
        anomalies: list[dict[str, Any]],
        battery_info: Optional[dict[str, Any]] = None,
        lifecycle_evidence: Optional[dict[str, Any] | LifecycleEvidence] = None,
        model_evidence: Optional[dict[str, Any] | ModelEvidence] = None,
    ) -> DiagnosisResult:
        symptoms = []
        severity_map: dict[str, str] = {}
        battery_source = str((battery_info or {}).get("source", "") or "").lower() or None
        lifecycle_payload = _coerce_lifecycle_evidence(lifecycle_evidence)
        model_payload = _coerce_model_evidence(model_evidence)
        for anomaly in anomalies:
            symptom = str(anomaly.get("symptom") or anomaly.get("type") or "未知异常")
            normalized = self.kg.normalize_symptom(symptom)
            symptoms.append(normalized)
            severity_map[normalized] = str(anomaly.get("severity", "low"))
        backend_warning = ""
        try:
            ranked_faults = self.kg.rank_faults(
                symptoms,
                severity_map=severity_map,
                battery_source=battery_source,
                lifecycle_evidence=lifecycle_payload,
            )
            self.active_backend = self.graph_backend
        except Exception as exc:
            if self.graph_backend != "neo4j":
                raise
            self.kg = self.fallback_kg
            self.active_backend = "memory"
            backend_warning = f"Neo4j 不可用，已自动切换到 memory 图谱后端：{exc}"
            ranked_faults = self.kg.rank_faults(
                symptoms,
                severity_map=severity_map,
                battery_source=battery_source,
                lifecycle_evidence=lifecycle_payload,
            )
        return self.llm.generate_diagnosis(
            anomalies=anomalies,
            fault_candidates=ranked_faults,
            context={
                "battery_info": battery_info or {},
                "symptoms": symptoms,
                "severity_map": severity_map,
                "lifecycle_evidence": lifecycle_payload.to_dict() if lifecycle_payload else None,
                "model_evidence": model_payload.to_dict() if model_payload else None,
                "graph_backend_used": self.active_backend,
                "backend_warning": backend_warning,
            },
        )

    def generate_report(self, diagnosis: DiagnosisResult) -> str:
        return diagnosis.report_markdown


def _coerce_lifecycle_evidence(payload: Optional[dict[str, Any] | LifecycleEvidence]) -> Optional[LifecycleEvidence]:
    if payload is None:
        return None
    if isinstance(payload, LifecycleEvidence):
        return payload
    return LifecycleEvidence(**payload)


def _coerce_model_evidence(payload: Optional[dict[str, Any] | ModelEvidence]) -> Optional[ModelEvidence]:
    if payload is None:
        return None
    if isinstance(payload, ModelEvidence):
        return payload
    return ModelEvidence(**payload)


def _tokenize_text(values: list[str]) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for token in str(value).replace("/", " ").replace(",", " ").replace("，", " ").split():
            if token:
                tokens.add(token.lower())
        compact = str(value).replace(" ", "").lower()
        if compact:
            tokens.add(compact)
    return tokens


def _infer_stage_label(lifecycle_evidence: Optional[LifecycleEvidence]) -> str:
    if lifecycle_evidence is None:
        return "mid"
    window = str(lifecycle_evidence.accelerated_degradation_window or "").lower()
    if any(keyword in window for keyword in ("late", "后期", "接近 eol", "near_eol")):
        return "late"
    if any(keyword in window for keyword in ("early", "前期")):
        return "early"
    return "mid"


def _future_risk_match(
    lifecycle_evidence: Optional[LifecycleEvidence],
    future_risk_patterns: Optional[list[str]],
    precursor_signals: Optional[list[str]],
    threshold_hints: Optional[list[str]],
) -> float:
    if lifecycle_evidence is None:
        return 0.3
    evidence_items = [
        str(lifecycle_evidence.future_capacity_fade_pattern or ""),
        str(lifecycle_evidence.temperature_risk or ""),
        str(lifecycle_evidence.resistance_risk or ""),
        str(lifecycle_evidence.voltage_risk or ""),
        str(lifecycle_evidence.accelerated_degradation_window or ""),
    ]
    evidence_tokens = _tokenize_text([item for item in evidence_items if item])
    candidate_tokens = _tokenize_text(list(future_risk_patterns or []) + list(precursor_signals or []) + list(threshold_hints or []))
    if not candidate_tokens:
        return 0.4
    overlap = len(evidence_tokens & candidate_tokens) / max(len(candidate_tokens), 1)
    if any("high" in item.lower() or "高" in item for item in evidence_items):
        overlap = min(1.0, overlap + 0.15)
    return round(min(1.0, overlap), 3)


def _stage_consistency(lifecycle_evidence: Optional[LifecycleEvidence], stage_scope: Optional[list[str]]) -> float:
    if not stage_scope:
        return 0.5
    normalized_scope = [str(item).lower() for item in stage_scope]
    if "whole_lifecycle" in normalized_scope or "any" in normalized_scope:
        return 0.9
    stage_label = _infer_stage_label(lifecycle_evidence)
    return 1.0 if stage_label in normalized_scope else 0.35


def _source_scope_match(battery_source: Optional[str], source_scope: Optional[list[str]]) -> float:
    normalized_scope = [str(item).lower() for item in (source_scope or ["generic"])]
    if battery_source and battery_source in normalized_scope:
        return 1.0
    if "generic" in normalized_scope:
        return 0.7
    return 0.25


def _threshold_hint_match(
    severity_map: dict[str, str],
    threshold_hints: Optional[list[str]],
    lifecycle_evidence: Optional[LifecycleEvidence],
) -> float:
    hints = list(threshold_hints or [])
    if not hints:
        return 0.35
    score = min(1.0, 0.2 + 0.2 * len(hints))
    if any(value in {"high", "critical"} for value in severity_map.values()):
        score = min(1.0, score + 0.2)
    if lifecycle_evidence and any("high" in str(item).lower() or "高" in str(item) for item in lifecycle_evidence.to_dict().values() if item):
        score = min(1.0, score + 0.15)
    return round(score, 3)


def _lifecycle_evidence_lines(evidence: Optional[LifecycleEvidence]) -> list[str]:
    if evidence is None:
        return []
    labels = {
        "predicted_knee_cycle": "预测 knee 周期",
        "predicted_eol_cycle": "预测 EOL 周期",
        "accelerated_degradation_window": "加速退化窗口",
        "future_capacity_fade_pattern": "未来容量衰减模式",
        "temperature_risk": "温度风险",
        "resistance_risk": "内阻风险",
        "voltage_risk": "电压风险",
    }
    lines = []
    for key, label in labels.items():
        value = getattr(evidence, key)
        if value is not None and value != "":
            lines.append(f"{label}：{value}")
    return lines


def _model_evidence_lines(evidence: Optional[ModelEvidence]) -> list[str]:
    if evidence is None:
        return []
    lines = [f"关键特征：{', '.join(evidence.top_features)}"] if evidence.top_features else []
    if evidence.critical_windows:
        lines.append(f"关键时间窗口：{', '.join(evidence.critical_windows)}")
    lines.extend(evidence.confidence_factors)
    return lines


def _fault_score(
    matched_symptoms: list[str],
    all_symptoms: list[str],
    severity_map: dict[str, str],
    *,
    battery_source: Optional[str] = None,
    source_scope: Optional[list[str]] = None,
    lifecycle_evidence: Optional[LifecycleEvidence] = None,
    stage_scope: Optional[list[str]] = None,
    future_risk_patterns: Optional[list[str]] = None,
    precursor_signals: Optional[list[str]] = None,
    threshold_hints: Optional[list[str]] = None,
) -> tuple[float, dict[str, Any]]:
    if not matched_symptoms:
        return 0.0, {
            "coverage_ratio": 0.0,
            "symptom_match": 0.0,
            "future_risk_match": 0.0,
            "stage_consistency": 0.0,
            "source_scope_match": 0.0,
            "threshold_hint_match": 0.0,
            "final_score": 0.0,
        }
    distinct_matched = sorted(set(matched_symptoms))
    distinct_all = sorted(set(all_symptoms))
    coverage_ratio = min(len(distinct_matched) / max(len(distinct_all), 1), 1.0)
    symptom_match = round(coverage_ratio, 3)
    future_risk_match = _future_risk_match(lifecycle_evidence, future_risk_patterns, precursor_signals, threshold_hints)
    stage_consistency = round(_stage_consistency(lifecycle_evidence, stage_scope), 3)
    source_scope_match = round(_source_scope_match(battery_source, source_scope), 3)
    threshold_hint_match = _threshold_hint_match(severity_map, threshold_hints, lifecycle_evidence)
    final_score = min(
        0.98,
        round(
            0.30 * symptom_match
            + 0.25 * future_risk_match
            + 0.20 * stage_consistency
            + 0.15 * source_scope_match
            + 0.10 * threshold_hint_match,
            3,
        ),
    )
    return final_score, {
        "coverage_ratio": round(coverage_ratio, 3),
        "symptom_match": symptom_match,
        "future_risk_match": future_risk_match,
        "stage_consistency": stage_consistency,
        "source_scope_match": source_scope_match,
        "threshold_hint_match": threshold_hint_match,
        "matched_symptom_count": len(distinct_matched),
        "final_score": final_score,
    }


def build_decision_basis(
    fault_candidates: list[dict[str, Any]],
    battery_info: dict[str, Any],
    *,
    lifecycle_evidence: Optional[LifecycleEvidence] = None,
    model_evidence: Optional[ModelEvidence] = None,
) -> list[str]:
    if not fault_candidates:
        return ["当前异常症状未能在知识库中形成稳定命中，因此系统返回未识别故障。"]
    primary = fault_candidates[0]
    basis = [
        f"首选规则 {primary.get('rule_id') or '--'}，匹配 {len(primary.get('matched_symptoms', []))}/{max(len(primary.get('all_symptoms', [])), 1)} 个症状。",
    ]
    coverage = float(primary.get("symptom_coverage", 0.0) or 0.0)
    basis.append(f"症状覆盖率为 {coverage:.3f}，得分 {float(primary.get('score', 0.0)):.3f}。")
    breakdown = primary.get("score_breakdown") or {}
    basis.append(
        "得分拆解："
        f"symptom {breakdown.get('symptom_match', 0.0)}, "
        f"future {breakdown.get('future_risk_match', 0.0)}, "
        f"stage {breakdown.get('stage_consistency', 0.0)}, "
        f"source {breakdown.get('source_scope_match', 0.0)}, "
        f"threshold {breakdown.get('threshold_hint_match', 0.0)}。"
    )
    battery_source = str(battery_info.get("source", "--")).lower()
    if battery_source and primary.get("source_scope"):
        basis.append(f"来源适用范围 {', '.join(primary.get('source_scope', []))}，当前样本来源 {battery_source}。")
    if lifecycle_evidence is not None:
        for item in _lifecycle_evidence_lines(lifecycle_evidence)[:3]:
            basis.append(f"生命周期证据：{item}。")
    if model_evidence is not None:
        for item in _model_evidence_lines(model_evidence)[:2]:
            basis.append(f"模型证据：{item}。")
    if len(fault_candidates) > 1:
        runner_up = fault_candidates[1]
        gap = round(float(primary.get("score", 0.0)) - float(runner_up.get("score", 0.0)), 3)
        basis.append(
            f"相较次优候选 {runner_up.get('name', '--')}，当前首选高出 {gap:.3f} 分，主要因为匹配症状更多或来源适用性更高。"
        )
    return basis


def build_graph_trace(
    fault_candidates: list[dict[str, Any]],
    matched_symptoms: list[str],
    *,
    battery_info: Optional[dict[str, Any]] = None,
    lifecycle_evidence: Optional[LifecycleEvidence] = None,
    model_evidence: Optional[ModelEvidence] = None,
) -> GraphTrace:
    nodes: list[GraphTraceNode] = []
    edges: list[GraphTraceEdge] = []
    seen_nodes: set[str] = set()

    def add_node(
        node_id: str,
        label: str,
        node_type: str,
        *,
        evidence_source: Optional[list[str]] = None,
        rule_id: Optional[str] = None,
        confidence_basis: Optional[list[str]] = None,
        source_scope: Optional[list[str]] = None,
        **properties: Any,
    ) -> None:
        if node_id in seen_nodes:
            return
        seen_nodes.add(node_id)
        nodes.append(
            GraphTraceNode(
                id=node_id,
                label=label,
                node_type=node_type,
                evidence_source=list(evidence_source or []),
                rule_id=rule_id,
                confidence_basis=list(confidence_basis or []),
                source_scope=list(source_scope or []),
                properties=properties,
            )
        )

    for symptom in matched_symptoms:
        add_node(f"symptom::{symptom}", symptom, "Symptom")

    if battery_info:
        chemistry = str(battery_info.get("chemistry") or "")
        protocol = str(battery_info.get("protocol_id") or "")
        if chemistry:
            add_node("chemistry::current", chemistry, "Chemistry")
        if protocol:
            add_node("protocol::current", protocol, "Protocol")
    if lifecycle_evidence and lifecycle_evidence.accelerated_degradation_window:
        add_node(
            "risk_window::accelerated",
            str(lifecycle_evidence.accelerated_degradation_window),
            "RiskWindow",
            **lifecycle_evidence.to_dict(),
        )

    for candidate in fault_candidates[:3]:
        fault_id = f"fault::{candidate['name']}"
        add_node(
            fault_id,
            candidate["name"],
            "Fault",
            evidence_source=candidate.get("evidence_source"),
            rule_id=candidate.get("rule_id"),
            confidence_basis=candidate.get("confidence_basis"),
            source_scope=candidate.get("source_scope"),
            score=float(candidate.get("score", 0.0)),
            severity=candidate.get("severity", "medium"),
            category=candidate.get("category", "未知"),
            score_breakdown=candidate.get("score_breakdown", {}),
        )
        if battery_info:
            if battery_info.get("chemistry"):
                edges.append(GraphTraceEdge(source=fault_id, target="chemistry::current", relation="APPLIES_TO_CHEMISTRY"))
            if battery_info.get("protocol_id"):
                edges.append(GraphTraceEdge(source=fault_id, target="protocol::current", relation="APPLIES_TO_PROTOCOL"))
        if lifecycle_evidence and lifecycle_evidence.accelerated_degradation_window:
            edges.append(GraphTraceEdge(source=fault_id, target="risk_window::accelerated", relation="FUTURE_RISK"))
        for symptom in candidate.get("matched_symptoms", []):
            symptom_id = f"symptom::{symptom}"
            add_node(symptom_id, symptom, "Symptom")
            edges.append(GraphTraceEdge(source=fault_id, target=symptom_id, relation="HAS_SYMPTOM"))
        for mechanism in candidate.get("mechanisms", []) or candidate.get("causes", []):
            mechanism_id = f"mechanism::{mechanism}"
            add_node(mechanism_id, mechanism, "Mechanism")
            edges.append(GraphTraceEdge(source=fault_id, target=mechanism_id, relation="INDICATES_MECHANISM"))
        for recommendation in candidate.get("recommendations", []):
            rec_id = f"action::{recommendation}"
            add_node(rec_id, recommendation, "Action")
            edges.append(GraphTraceEdge(source=fault_id, target=rec_id, relation="RECOMMENDS_ACTION"))
        if model_evidence:
            for feature in model_evidence.top_features[:3]:
                node_id = f"model_feature::{feature}"
                add_node(node_id, feature, "ModelEvidence")
                edges.append(GraphTraceEdge(source=fault_id, target=node_id, relation="SUPPORTED_BY_MODEL"))

    return GraphTrace(
        matched_symptoms=matched_symptoms,
        nodes=nodes,
        edges=edges,
        ranking_basis=[
            "症状匹配（0.30）",
            "未来风险匹配（0.25）",
            "阶段一致性（0.20）",
            "来源适用范围（0.15）",
            "阈值提示匹配（0.10）",
        ],
    )


__all__ = [
    "CandidateFault",
    "DiagnosisResult",
    "GraphRAGEngine",
    "GraphTrace",
    "KnowledgeGraph",
    "LifecycleEvidence",
    "LLMInterface",
    "ModelEvidence",
    "Neo4jKnowledgeGraph",
    "SeedKnowledge",
]
