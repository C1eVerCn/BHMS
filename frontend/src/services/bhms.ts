import { request } from './api'
import type {
  AblationResult,
  AnomalyDetectionResult,
  AnomalyEvent,
  Battery,
  BatteryOptionsResponse,
  BatteryHealth,
  BatteryHistory,
  CaseBundle,
  CaseBundleExportResult,
  CyclePoint,
  DashboardSummary,
  DatasetProfile,
  DemoPreset,
  ExperimentDetail,
  ExperimentOverview,
  KnowledgeSummary,
  LifecycleModelName,
  LifecyclePredictionResult,
  MechanismExplanationResult,
  PaginatedBatteries,
  SupportedSource,
  SystemStatus,
  TrainingComparison,
  UploadSummary,
} from '../types/domain'

export function getDashboardSummary() {
  return request<DashboardSummary>({ method: 'GET', url: '/dashboard/summary' })
}

export function listBatteries(page = 1, pageSize = 10) {
  return request<PaginatedBatteries>({ method: 'GET', url: '/batteries', params: { page, page_size: pageSize } })
}

export function listBatteryOptions() {
  return request<BatteryOptionsResponse>({ method: 'GET', url: '/batteries/options' })
}

export function getBattery(batteryId: string) {
  return request<Battery>({ method: 'GET', url: `/battery/${batteryId}` })
}

export function getBatteryCycles(batteryId: string, limit = 120) {
  return request<{ battery_id: string; items: CyclePoint[] }>({ method: 'GET', url: `/battery/${batteryId}/cycles`, params: { limit } })
}

export function getBatteryHistory(batteryId: string) {
  return request<BatteryHistory>({ method: 'GET', url: `/battery/${batteryId}/history` })
}

export function getBatteryHealth(batteryId: string) {
  return request<BatteryHealth>({ method: 'GET', url: `/battery/${batteryId}/health` })
}

export function updateTrainingCandidate(batteryId: string, includeInTraining: boolean) {
  return request<Battery>({
    method: 'POST',
    url: `/battery/${batteryId}/training-candidate`,
    data: { include_in_training: includeInTraining },
  })
}

export async function uploadBatteryData(file: File, options?: { batteryId?: string; source?: string; includeInTraining?: boolean }) {
  const formData = new FormData()
  formData.append('file', file)
  if (options?.batteryId) {
    formData.append('battery_id', options.batteryId)
  }
  formData.append('source', options?.source ?? 'auto')
  formData.append('include_in_training', String(Boolean(options?.includeInTraining)))
  return request<UploadSummary>({
    method: 'POST',
    url: '/data/upload',
    data: formData,
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export function importSourceData(source: SupportedSource, batteryIds?: string[], includeInTraining = false) {
  return request<UploadSummary>({
    method: 'POST',
    url: '/data/import-source',
    data: { battery_ids: batteryIds ?? null, source, include_in_training: includeInTraining },
  })
}

export function importDemoPreset(presetName: string, includeInTraining = false) {
  return request<UploadSummary>({
    method: 'POST',
    url: '/data/import-demo-preset',
    data: { preset_name: presetName, include_in_training: includeInTraining },
  })
}

export function predictLifecycle(payload: { battery_id: string; model_name: LifecycleModelName; seq_len: number; historical_data?: CyclePoint[] }) {
  return request<LifecyclePredictionResult>({ method: 'POST', url: '/predict/lifecycle', data: payload, baseURL: '/api/v2' })
}

export function detectAnomaly(payload: { battery_id: string; current_data?: CyclePoint; baseline_capacity?: number; use_latest?: boolean }) {
  return request<AnomalyDetectionResult>({ method: 'POST', url: '/detect/anomaly', data: payload })
}

export function explainMechanism(payload: {
  battery_id: string
  anomalies?: AnomalyEvent[]
  battery_info?: Record<string, unknown>
  model_name?: LifecycleModelName
  seq_len?: number
}) {
  return request<MechanismExplanationResult>({ method: 'POST', url: '/explain/mechanism', data: payload, baseURL: '/api/v2' })
}

export function getTrainingComparison(source: string) {
  return request<TrainingComparison>({ method: 'GET', url: '/training/comparison', params: { source } })
}

export function getTrainingOverview() {
  return request<ExperimentOverview>({ method: 'GET', url: '/training/overview' })
}

export function getTrainingExperimentDetail(source: string) {
  return request<ExperimentDetail>({ method: 'GET', url: `/training/experiments/${source}` })
}

export function getTrainingAblationSummary(source: string) {
  return request<AblationResult>({ method: 'GET', url: `/training/ablations/${source}` })
}

export function getSystemStatus() {
  return request<SystemStatus>({ method: 'GET', url: '/system/status' })
}

export function getDemoPresets() {
  return request<DemoPreset[]>({ method: 'GET', url: '/demo/presets' })
}

export function getDatasetProfile(source: string) {
  return request<DatasetProfile>({ method: 'GET', url: `/data/profile/${source}` })
}

export function getKnowledgeSummary() {
  return request<KnowledgeSummary>({ method: 'GET', url: '/diagnosis/knowledge/summary' })
}

export function getCaseBundle(batteryId: string) {
  return request<CaseBundle>({ method: 'GET', url: `/reports/case-bundle/${batteryId}` })
}

export function exportCaseBundle(batteryId: string, ensureArtifacts = true) {
  return request<CaseBundleExportResult>({
    method: 'POST',
    url: `/reports/case-bundle/${batteryId}/export`,
    params: { ensure_artifacts: ensureArtifacts },
  })
}
