import { request } from './api'
import type {
  AnomalyDetectionResult,
  AnomalyEvent,
  Battery,
  BatteryHealth,
  BatteryHistory,
  CyclePoint,
  DashboardSummary,
  DiagnosisResult,
  PaginatedBatteries,
  PredictionResult,
  UploadSummary,
} from '../types/domain'

export function getDashboardSummary() {
  return request<DashboardSummary>({ method: 'GET', url: '/dashboard/summary' })
}

export function listBatteries(page = 1, pageSize = 10) {
  return request<PaginatedBatteries>({ method: 'GET', url: '/batteries', params: { page, page_size: pageSize } })
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

export function importSourceData(source: 'nasa' | 'calce' | 'kaggle', batteryIds?: string[], includeInTraining = false) {
  return request<UploadSummary>({
    method: 'POST',
    url: '/data/import-source',
    data: { battery_ids: batteryIds ?? null, source, include_in_training: includeInTraining },
  })
}

export function predictRul(payload: { battery_id: string; model_name: string; seq_len: number; historical_data?: CyclePoint[] }) {
  return request<PredictionResult>({ method: 'POST', url: '/predict/rul', data: payload })
}

export function detectAnomaly(payload: { battery_id: string; current_data?: CyclePoint; baseline_capacity?: number; use_latest?: boolean }) {
  return request<AnomalyDetectionResult>({ method: 'POST', url: '/detect/anomaly', data: payload })
}

export function diagnoseBattery(payload: { battery_id: string; anomalies: AnomalyEvent[]; battery_info?: Record<string, unknown> }) {
  return request<DiagnosisResult>({ method: 'POST', url: '/diagnose', data: payload })
}
