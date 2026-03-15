"""API 域模型。"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class CyclePoint(BaseModel):
    battery_id: str = Field(..., description="电池 ID")
    canonical_battery_id: Optional[str] = None
    source: Optional[str] = None
    dataset_name: Optional[str] = None
    source_battery_id: Optional[str] = None
    cycle_number: int = Field(..., description="循环次数")
    timestamp: Optional[str] = Field(default=None, description="时间戳")
    ambient_temperature: float = 0.0
    voltage_mean: float
    voltage_std: float = 0.0
    voltage_min: float = 0.0
    voltage_max: float = 0.0
    current_mean: float
    current_std: float = 0.0
    current_load_mean: float = 0.0
    temperature_mean: float
    temperature_std: float = 0.0
    temperature_rise_rate: float = 0.0
    internal_resistance: Optional[float] = None
    capacity: float
    source_type: str = "uploaded"


class Battery(BaseModel):
    battery_id: str
    canonical_battery_id: Optional[str] = None
    source: str
    dataset_name: Optional[str] = None
    source_battery_id: Optional[str] = None
    chemistry: Optional[str] = None
    nominal_capacity: Optional[float] = None
    initial_capacity: Optional[float] = None
    latest_capacity: Optional[float] = None
    cycle_count: int
    health_score: float
    status: str
    last_update: Optional[str] = None
    dataset_path: Optional[str] = None
    include_in_training: bool = False


class TrainingRun(BaseModel):
    id: int
    source: str
    model_type: str
    model_version: Optional[str] = None
    best_checkpoint_path: Optional[str] = None
    final_checkpoint_path: Optional[str] = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str


class PredictionPoint(BaseModel):
    cycle: float
    capacity: float


class ConfidenceBandPoint(BaseModel):
    cycle: float
    lower: float
    upper: float


class AttentionHeatmap(BaseModel):
    x_labels: list[str] = Field(default_factory=list)
    y_labels: list[str] = Field(default_factory=list)
    values: list[list[float]] = Field(default_factory=list)
    disclaimer: str = "注意力热力图仅作为辅助参考，不直接等于因果解释。"


class FeatureContribution(BaseModel):
    feature: str
    impact: float
    direction: str
    description: str


class WindowContribution(BaseModel):
    window_label: str
    start_cycle: float
    end_cycle: float
    impact: float
    description: str


class PredictionExplanation(BaseModel):
    input_summary: dict[str, Any] = Field(default_factory=dict)
    model_info: dict[str, Any] = Field(default_factory=dict)
    feature_contributions: list[FeatureContribution] = Field(default_factory=list)
    window_contributions: list[WindowContribution] = Field(default_factory=list)
    confidence_summary: dict[str, Any] = Field(default_factory=dict)
    attention_heatmap: Optional[AttentionHeatmap] = None


class PredictionProjection(BaseModel):
    actual_points: list[PredictionPoint] = Field(default_factory=list)
    forecast_points: list[PredictionPoint] = Field(default_factory=list)
    eol_capacity: float
    predicted_eol_cycle: float
    confidence_band: list[ConfidenceBandPoint] = Field(default_factory=list)


class PredictionRecord(BaseModel):
    id: int
    battery_id: str
    model_name: str
    predicted_rul: float
    confidence: float
    input_seq_len: int
    created_at: str
    source: str
    payload: dict[str, Any] = Field(default_factory=dict)
    model_version: Optional[str] = None
    model_source: Optional[str] = None
    checkpoint_id: Optional[str] = None
    fallback_used: Optional[bool] = None
    prediction_time: Optional[str] = None
    projection: Optional[PredictionProjection] = None
    explanation: Optional[PredictionExplanation] = None
    report_markdown: Optional[str] = None


class PredictionResult(PredictionRecord):
    model_version: str
    model_source: str
    fallback_used: bool
    prediction_time: str
    projection: PredictionProjection
    explanation: PredictionExplanation
    report_markdown: str


class AnomalyEventModel(BaseModel):
    code: str
    symptom: str
    severity: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold_value: Optional[str] = None
    description: str
    source: str = "statistical"
    evidence: list[str] = Field(default_factory=list)
    evidence_source: str = "statistical_rules"
    rule_id: Optional[str] = None
    confidence_basis: list[str] = Field(default_factory=list)
    source_scope: list[str] = Field(default_factory=list)


class CandidateFault(BaseModel):
    name: str
    score: float
    severity: str
    description: str
    category: Optional[str] = None
    matched_symptoms: list[str] = Field(default_factory=list)
    all_symptoms: list[str] = Field(default_factory=list)
    root_causes: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    evidence_source: list[str] = Field(default_factory=list)
    rule_id: Optional[str] = None
    confidence_basis: list[str] = Field(default_factory=list)
    source_scope: list[str] = Field(default_factory=list)
    threshold_hints: list[str] = Field(default_factory=list)
    symptom_coverage: float = 0.0
    matched_symptom_count: int = 0
    score_breakdown: dict[str, Any] = Field(default_factory=dict)


class GraphTraceNode(BaseModel):
    id: str
    label: str
    node_type: str
    evidence_source: list[str] = Field(default_factory=list)
    rule_id: Optional[str] = None
    confidence_basis: list[str] = Field(default_factory=list)
    source_scope: list[str] = Field(default_factory=list)
    properties: dict[str, Any] = Field(default_factory=dict)


class GraphTraceEdge(BaseModel):
    source: str
    target: str
    relation: str


class GraphTrace(BaseModel):
    matched_symptoms: list[str] = Field(default_factory=list)
    nodes: list[GraphTraceNode] = Field(default_factory=list)
    edges: list[GraphTraceEdge] = Field(default_factory=list)
    ranking_basis: list[str] = Field(default_factory=list)


class DiagnosisRecord(BaseModel):
    id: int
    battery_id: str
    fault_type: str
    confidence: float
    severity: str
    description: str
    root_causes: list[str]
    recommendations: list[str]
    related_symptoms: list[str]
    evidence: list[str]
    created_at: str
    payload: dict[str, Any] = Field(default_factory=dict)
    candidate_faults: list[CandidateFault] = Field(default_factory=list)
    graph_trace: Optional[GraphTrace] = None
    decision_basis: list[str] = Field(default_factory=list)
    report_markdown: Optional[str] = None


class DiagnosisResult(BaseModel):
    id: int
    battery_id: str
    fault_type: str
    confidence: float
    severity: str
    description: str
    root_causes: list[str]
    recommendations: list[str]
    related_symptoms: list[str]
    evidence: list[str]
    diagnosis_time: str
    candidate_faults: list[CandidateFault] = Field(default_factory=list)
    graph_trace: GraphTrace
    decision_basis: list[str] = Field(default_factory=list)
    report_markdown: str


class KnowledgeEntry(BaseModel):
    name: str
    category: str
    severity: str
    description: str
    symptoms: list[str]
    causes: list[str]
    recommendations: list[str]
    evidence_source: list[str] = Field(default_factory=list)
    rule_id: Optional[str] = None
    confidence_basis: list[str] = Field(default_factory=list)
    source_scope: list[str] = Field(default_factory=list)
    threshold_hints: list[str] = Field(default_factory=list)


class BatteryHealth(BaseModel):
    battery_id: str
    overall_health: str
    health_score: float
    latest_capacity: Optional[float] = None
    rul_prediction: Optional[float] = None
    anomaly_count: int
    last_update: Optional[str] = None
    source: Optional[str] = None
    dataset_name: Optional[str] = None


class DashboardSummary(BaseModel):
    total_batteries: int
    good_batteries: int
    warning_batteries: int
    critical_batteries: int
    average_health_score: float
    recent_alerts: list[dict[str, Any]]
    health_distribution: list[dict[str, Any]]
    capacity_trend: list[dict[str, Any]]
    batteries_by_source: list[dict[str, Any]] = Field(default_factory=list)


class PaginatedBatteries(BaseModel):
    items: list[Battery]
    page: int
    page_size: int
    total: int


class BatteryHistory(BaseModel):
    battery_id: str
    predictions: list[PredictionRecord]
    diagnoses: list[DiagnosisRecord]
    anomalies: list[AnomalyEventModel]


class UploadSummary(BaseModel):
    battery_ids: list[str]
    imported_cycles: int
    file_name: str
    file_path: str
    validation_summary: dict[str, Any]
    include_in_training: bool = False
    source: str
    dataset_name: Optional[str] = None
    detected_source: Optional[str] = None


class TrainingJob(BaseModel):
    id: int
    source: str
    model_scope: Literal["bilstm", "hybrid", "all"]
    job_kind: Optional[Literal["baseline", "multi_seed", "ablation", "full_suite"]] = None
    seed_count: Optional[int] = None
    status: str
    current_stage: Optional[str] = None
    force_run: bool = False
    baseline: Optional[dict[str, Any]] = None
    result: Optional[dict[str, Any]] = None
    log_excerpt: Optional[str] = None
    error_message: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


class TrainingComparison(BaseModel):
    source: str
    previous: Optional[dict[str, Any]] = None
    current: Optional[dict[str, Any]] = None
    latest_job: Optional[TrainingJob] = None
    runs: list[TrainingRun] = Field(default_factory=list)


class RULPredictionRequest(BaseModel):
    battery_id: str
    model_name: str = "hybrid"
    seq_len: int = Field(default=30, ge=10, le=500)
    historical_data: Optional[list[CyclePoint]] = None


class AnomalyDetectionRequest(BaseModel):
    battery_id: str
    current_data: Optional[CyclePoint] = None
    baseline_capacity: Optional[float] = None
    use_latest: bool = True


class DiagnosisRequest(BaseModel):
    battery_id: str
    anomalies: list[AnomalyEventModel] = Field(default_factory=list)
    battery_info: Optional[dict[str, Any]] = None


class DataImportRequest(BaseModel):
    battery_ids: Optional[list[str]] = None
    source: str = "nasa"
    include_in_training: bool = False


class UpdateTrainingCandidateRequest(BaseModel):
    include_in_training: bool = True


class CreateTrainingJobRequest(BaseModel):
    source: Literal["nasa", "calce", "kaggle"]
    model_scope: Literal["bilstm", "hybrid", "all"] = "all"
    force_run: bool = False
    job_kind: Literal["baseline", "multi_seed", "ablation", "full_suite"] = "baseline"
    seed_count: int = Field(default=3, ge=1, le=5)


class BatteryCyclesResponse(BaseModel):
    battery_id: str
    items: list[CyclePoint]
