"""API 域模型。"""

from __future__ import annotations

from typing import Any, Optional

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


class KnowledgeEntry(BaseModel):
    name: str
    category: str
    severity: str
    description: str
    symptoms: list[str]
    causes: list[str]
    recommendations: list[str]


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


class BatteryCyclesResponse(BaseModel):
    battery_id: str
    items: list[CyclePoint]
