export type BatteryStatus = 'good' | 'warning' | 'critical' | 'unknown'

export interface CyclePoint {
  battery_id: string
  canonical_battery_id?: string | null
  source?: string | null
  dataset_name?: string | null
  source_battery_id?: string | null
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
  canonical_battery_id?: string | null
  source: string
  dataset_name?: string | null
  source_battery_id?: string | null
  chemistry?: string | null
  nominal_capacity?: number | null
  initial_capacity?: number | null
  latest_capacity?: number | null
  cycle_count: number
  health_score: number
  status: BatteryStatus
  last_update?: string | null
  dataset_path?: string | null
  include_in_training?: boolean
}

export interface TrainingRun {
  id: number
  source: string
  model_type: string
  model_version?: string | null
  best_checkpoint_path?: string | null
  final_checkpoint_path?: string | null
  metrics: Record<string, unknown>
  metadata: Record<string, unknown>
  created_at: string
}

export interface PredictionPoint {
  cycle: number
  capacity: number
}

export interface ConfidenceBandPoint {
  cycle: number
  lower: number
  upper: number
}

export interface PredictionProjection {
  actual_points: PredictionPoint[]
  forecast_points: PredictionPoint[]
  eol_capacity: number
  predicted_eol_cycle: number
  confidence_band: ConfidenceBandPoint[]
  projection_method?: string
}

export interface AttentionHeatmap {
  x_labels: string[]
  y_labels: string[]
  values: number[][]
  disclaimer: string
}

export interface FeatureContribution {
  feature: string
  impact: number
  direction: string
  description: string
}

export interface WindowContribution {
  window_label: string
  start_cycle: number
  end_cycle: number
  impact: number
  description: string
}

export interface PredictionExplanation {
  input_summary: Record<string, unknown>
  model_info: Record<string, unknown>
  feature_contributions: FeatureContribution[]
  window_contributions: WindowContribution[]
  confidence_summary: Record<string, unknown>
  attention_heatmap?: AttentionHeatmap | null
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
  payload?: Record<string, unknown>
  model_version?: string | null
  model_source?: string | null
  checkpoint_id?: string | null
  fallback_used?: boolean
  prediction_time?: string | null
  projection?: PredictionProjection
  explanation?: PredictionExplanation
  report_markdown?: string | null
}

export interface PredictionResult extends PredictionRecord {
  model_version: string
  model_source: string
  fallback_used: boolean
  prediction_time: string
  projection: PredictionProjection
  explanation: PredictionExplanation
  report_markdown: string
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
  evidence_source?: string
  rule_id?: string | null
  confidence_basis?: string[]
  source_scope?: string[]
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

export interface CandidateFault {
  name: string
  score: number
  severity: string
  description: string
  category?: string | null
  matched_symptoms: string[]
  all_symptoms?: string[]
  root_causes: string[]
  recommendations: string[]
  evidence_source?: string[]
  rule_id?: string | null
  confidence_basis?: string[]
  source_scope?: string[]
  threshold_hints?: string[]
  symptom_coverage?: number
  matched_symptom_count?: number
  score_breakdown?: Record<string, unknown>
}

export interface GraphTraceNode {
  id: string
  label: string
  node_type: string
  evidence_source?: string[]
  rule_id?: string | null
  confidence_basis?: string[]
  source_scope?: string[]
  properties: Record<string, unknown>
}

export interface GraphTraceEdge {
  source: string
  target: string
  relation: string
}

export interface GraphTrace {
  matched_symptoms: string[]
  nodes: GraphTraceNode[]
  edges: GraphTraceEdge[]
  ranking_basis: string[]
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
  payload?: Record<string, unknown>
  candidate_faults?: CandidateFault[]
  graph_trace?: GraphTrace
  decision_basis?: string[]
  report_markdown?: string | null
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
  candidate_faults: CandidateFault[]
  graph_trace: GraphTrace
  decision_basis?: string[]
  report_markdown: string
}

export interface BatteryHealth {
  battery_id: string
  overall_health: BatteryStatus
  health_score: number
  latest_capacity?: number | null
  rul_prediction?: number | null
  anomaly_count: number
  last_update?: string | null
  source?: string | null
  dataset_name?: string | null
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
  batteries_by_source: Array<{ source: string; battery_count: number }>
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
  include_in_training: boolean
  source: string
  dataset_name?: string | null
  detected_source?: string | null
}

export interface TrainingJob {
  id: number
  source: string
  model_scope: 'bilstm' | 'hybrid' | 'all'
  job_kind?: 'baseline' | 'multi_seed' | 'ablation' | 'full_suite'
  seed_count?: number
  status: string
  current_stage?: string | null
  force_run?: boolean
  baseline?: Record<string, unknown> | null
  result?: Record<string, unknown> | null
  log_excerpt?: string | null
  error_message?: string | null
  metadata?: Record<string, unknown>
  created_at: string
  started_at?: string | null
  finished_at?: string | null
}

export interface TrainingComparison {
  source: string
  previous?: Record<string, unknown> | null
  current?: Record<string, unknown> | null
  latest_job?: TrainingJob | null
  runs: TrainingRun[]
}

export interface TrainingSourceOverview {
  source: string
  best_model?: string | null
  dataset_batteries: number
  academic_status: string
  headline: string
  warnings: string[]
  plot_count?: number
}

export interface ExperimentOverview {
  generated_at: string
  sources: TrainingSourceOverview[]
  summary_notes: string[]
  warnings: string[]
}

export interface ModelExperimentDetail {
  model_type: string
  best_val_loss?: number | null
  test_metrics: Record<string, number | null | undefined>
  best_checkpoint?: string | null
  final_checkpoint?: string | null
  history_available: boolean
  multi_seed_summary?: Record<string, unknown> | null
  single_run_summary?: Record<string, unknown> | null
  multi_seed_available?: boolean
  aggregate_metrics?: {
    mean?: Record<string, number | null | undefined>
    std?: Record<string, number | null | undefined>
  }
  per_seed_runs?: Array<{
    seed?: number
    metrics?: Record<string, number | null | undefined>
    best_checkpoint?: string | null
    final_checkpoint?: string | null
    artifact_dir?: string | null
    summary_path?: string | null
    history_summary?: Record<string, unknown>
  }>
  plots?: PlotArtifact[]
  artifact_paths?: Record<string, unknown>
  preferred_metrics?: Record<string, number | null | undefined>
  assessment: string
}

export interface ExperimentDetail {
  source: string
  dataset_summary: Record<string, unknown>
  comparison: TrainingComparison
  models: Record<string, ModelExperimentDetail>
  best_model?: string | null
  headline: string
  academic_status: string
  warnings: string[]
  key_findings: string[]
  plots?: PlotArtifact[]
  artifact_paths?: Record<string, unknown>
  recommended_commands: Record<string, string>
}

export interface PlotArtifact {
  key: string
  title: string
  description: string
  path: string
  metadata_path?: string
  generated_at?: string
}

export interface AblationVariant {
  key: string
  label: string
  description: string
  status?: string
  seeds?: number[]
  config_overrides?: Record<string, unknown>
  feature_columns?: string[]
  metrics?: Record<string, number | null | undefined>
  aggregate_metrics?: {
    mean?: Record<string, number | null | undefined>
    std?: Record<string, number | null | undefined>
  }
  per_seed_runs?: Array<{
    seed?: number
    metrics?: Record<string, number | null | undefined>
    best_checkpoint?: string | null
    artifact_dir?: string | null
  }>
  delta_vs_full?: Record<string, number | null | undefined>
  artifact_paths?: Record<string, unknown>
  plots?: PlotArtifact[]
  best_checkpoint?: Record<string, unknown>
}

export interface AblationResult {
  source: string
  available: boolean
  notes: string[]
  recommended_command?: string
  variants: AblationVariant[]
  plots?: PlotArtifact[]
  artifact_paths?: Record<string, unknown>
  generated_at?: string
}

export interface DemoPreset {
  name: string
  path: string
  source: string
  scenario: string
  recommended: boolean
  description: string
}

export interface DatasetBreakdownItem {
  dataset_name: string
  battery_count: number
}

export interface FeatureRange {
  min?: number | null
  max?: number | null
  avg?: number | null
}

export interface DatasetFileRecord {
  file_name: string
  file_path: string
  row_count?: number | null
  include_in_training: boolean
  created_at?: string | null
}

export interface DatasetProfile {
  source: string
  battery_count: number
  training_candidate_count: number
  cycle_point_count: number
  dataset_names: string[]
  dataset_breakdown: DatasetBreakdownItem[]
  cycle_window: { min_cycle: number; max_cycle: number }
  feature_ranges: Record<string, FeatureRange>
  missing_stats: Record<string, number>
  top_batteries_by_cycles: Array<{ battery_id: string; first_cycle: number; last_cycle: number; cycle_points: number }>
  available_feature_columns: string[]
  split: Record<string, unknown>
  num_samples: Record<string, number>
  processed_summary_path: string
  comparison_summary_path: string
  comparison_available: boolean
  dataset_files: DatasetFileRecord[]
  demo_files: DemoPreset[]
  generated_at: string
}

export interface KnowledgeSummary {
  fault_count: number
  symptom_alias_count: number
  categories: Record<string, number>
  source_coverage?: Record<string, number>
  severity_distribution: Record<string, number>
  evidence_sources?: Array<[string, number]>
  rule_count?: number
  threshold_rule_count?: number
  fault_names: string[]
  top_symptoms: Array<[string, number]>
  graph_backend: string
  knowledge_path: string
  coverage_notes: string[]
  generated_at: string
}

export interface SourceStatus {
  source: string
  raw_file_count: number
  battery_count: number
  training_candidate_count: number
  processed_ready: boolean
  comparison_ready: boolean
  demo_preset_count: number
  best_model?: string | null
}

export interface SystemStatus {
  app_name: string
  api_prefix: string
  graph_backend: string
  database_path: string
  database_ready: boolean
  knowledge_ready: boolean
  demo_preset_count: number
  source_statuses: SourceStatus[]
  demo_acceptance_flow: string[]
  warnings: string[]
  generated_at: string
}

export interface CaseArtifact {
  key: string
  title: string
  available: boolean
  description: string
}

export interface CaseBundle {
  battery_id: string
  source?: string | null
  dataset_name?: string | null
  health_score?: number | null
  cycle_count?: number | null
  prediction?: PredictionRecord | null
  diagnosis?: DiagnosisRecord | null
  anomalies: AnomalyEvent[]
  dataset_profile: DatasetProfile
  dataset_position?: {
    canonical_battery_id?: string
    split_name?: string
    include_in_training?: boolean
    dataset_name?: string
    source?: string
  }
  export_ready?: boolean
  last_export?: {
    export_dir: string
    generated_at?: string
    files?: ExportedFile[]
    manifest?: Record<string, unknown>
  } | null
  chart_artifacts?: Array<Record<string, unknown>>
  artifacts: CaseArtifact[]
  recommended_story: string[]
  bundle_markdown: string
  experiment_context?: Record<string, unknown>
  generated_at: string
}

export interface ExportedFile {
  path: string
  kind: string
  key?: string
}

export interface CaseBundleExportResult {
  export_dir: string
  files: ExportedFile[]
  generated_artifacts: Record<string, boolean>
  bundle_snapshot: CaseBundle
}
