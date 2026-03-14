export type BatteryStatus = 'good' | 'warning' | 'critical' | 'unknown'

export interface CyclePoint {
  battery_id: string
  cycle_number: number
  timestamp?: string | null
  ambient_temperature?: number
  voltage_mean: number
  voltage_std?: number
  voltage_min?: number
  voltage_max?: number
  current_mean: number
  current_std?: number
  current_load_mean?: number
  temperature_mean: number
  temperature_std?: number
  temperature_rise_rate?: number
  internal_resistance?: number | null
  capacity: number
  source_type?: string
}

export interface Battery {
  battery_id: string
  source: string
  chemistry?: string | null
  nominal_capacity?: number | null
  initial_capacity?: number | null
  latest_capacity?: number | null
  cycle_count: number
  health_score: number
  status: BatteryStatus
  last_update?: string | null
  dataset_path?: string | null
}

export interface PredictionRecord {
  id: number
  battery_id: string
  model_name: string
  predicted_rul: number
  confidence: number
  input_seq_len: number
  created_at: string
  source: string
}

export interface PredictionResult extends PredictionRecord {
  model_version: string
  fallback_used: boolean
  prediction_time: string
}

export interface AnomalyEvent {
  code: string
  symptom: string
  severity: 'low' | 'medium' | 'high' | 'critical' | string
  metric_name?: string | null
  metric_value?: number | null
  threshold_value?: string | null
  description: string
  source: string
  evidence: string[]
}

export interface AnomalyDetectionResult {
  battery_id: string
  is_anomaly: boolean
  max_severity?: string | null
  summary: string
  event_ids: number[]
  events: AnomalyEvent[]
  detection_time: string
}

export interface DiagnosisRecord {
  id: number
  battery_id: string
  fault_type: string
  confidence: number
  severity: string
  description: string
  root_causes: string[]
  recommendations: string[]
  related_symptoms: string[]
  evidence: string[]
  created_at: string
}

export interface DiagnosisResult {
  id: number
  battery_id: string
  fault_type: string
  confidence: number
  severity: string
  description: string
  root_causes: string[]
  recommendations: string[]
  related_symptoms: string[]
  evidence: string[]
  diagnosis_time: string
}

export interface BatteryHealth {
  battery_id: string
  overall_health: BatteryStatus
  health_score: number
  latest_capacity?: number | null
  rul_prediction?: number | null
  anomaly_count: number
  last_update?: string | null
}

export interface DashboardAlert {
  battery_id: string
  symptom: string
  severity: string
  description: string
}

export interface DashboardSummary {
  total_batteries: number
  good_batteries: number
  warning_batteries: number
  critical_batteries: number
  average_health_score: number
  recent_alerts: DashboardAlert[]
  health_distribution: Array<{ name: string; value: number }>
  capacity_trend: Array<{ cycle_number: number; avg_capacity: number }>
}

export interface PaginatedBatteries {
  items: Battery[]
  page: number
  page_size: number
  total: number
}

export interface BatteryHistory {
  battery_id: string
  predictions: PredictionRecord[]
  diagnoses: DiagnosisRecord[]
  anomalies: AnomalyEvent[]
}

export interface UploadSummary {
  battery_ids: string[]
  imported_cycles: number
  file_name: string
  file_path: string
  validation_summary: Record<string, unknown>
}
