import { create } from 'zustand'

import {
  detectAnomaly,
  diagnoseBattery,
  getBatteryCycles,
  getBatteryHealth,
  getBatteryHistory,
  getDashboardSummary,
  importSourceData,
  listBatteries,
  predictRul,
  uploadBatteryData,
} from '../services/bhms'
import type {
  AnomalyDetectionResult,
  Battery,
  BatteryHealth,
  BatteryHistory,
  CyclePoint,
  DashboardSummary,
  DiagnosisResult,
  PredictionResult,
  UploadSummary,
} from '../types/domain'

interface UploadOptions {
  batteryId?: string
  source?: string
  includeInTraining?: boolean
}

interface BhmsState {
  dashboard: DashboardSummary | null
  batteries: Battery[]
  selectedBatteryId: string | null
  batteryHealth: Record<string, BatteryHealth>
  batteryCycles: Record<string, CyclePoint[]>
  batteryHistory: Record<string, BatteryHistory>
  latestPrediction: Record<string, PredictionResult>
  latestAnomaly: Record<string, AnomalyDetectionResult>
  latestDiagnosis: Record<string, DiagnosisResult>
  pagination: { page: number; pageSize: number; total: number }
  loading: boolean
  actionLoading: boolean
  error: string | null
  lastUpload: UploadSummary | null
  init: () => Promise<void>
  loadDashboard: () => Promise<void>
  loadBatteries: (page?: number, pageSize?: number) => Promise<void>
  selectBattery: (batteryId: string | null) => void
  loadBatteryContext: (batteryId: string) => Promise<void>
  uploadFile: (file: File, options?: UploadOptions) => Promise<UploadSummary>
  importSource: (source: 'nasa' | 'calce' | 'kaggle', includeInTraining?: boolean) => Promise<UploadSummary>
  runPrediction: (batteryId: string, modelName: string, seqLen: number) => Promise<PredictionResult>
  runDiagnosisWorkflow: (batteryId: string) => Promise<{ anomaly: AnomalyDetectionResult; diagnosis: DiagnosisResult }>
  clearError: () => void
}

export const useBhmsStore = create<BhmsState>((set, get) => ({
  dashboard: null,
  batteries: [],
  selectedBatteryId: null,
  batteryHealth: {},
  batteryCycles: {},
  batteryHistory: {},
  latestPrediction: {},
  latestAnomaly: {},
  latestDiagnosis: {},
  pagination: { page: 1, pageSize: 10, total: 0 },
  loading: false,
  actionLoading: false,
  error: null,
  lastUpload: null,
  clearError: () => set({ error: null }),
  init: async () => {
    if (get().loading) return
    set({ loading: true, error: null })
    try {
      await Promise.all([get().loadDashboard(), get().loadBatteries(1, get().pagination.pageSize)])
      const firstBattery = get().selectedBatteryId ?? get().batteries[0]?.battery_id ?? null
      if (firstBattery) {
        await get().loadBatteryContext(firstBattery)
      }
    } catch (error) {
      set({ error: error instanceof Error ? error.message : '初始化失败' })
    } finally {
      set({ loading: false })
    }
  },
  loadDashboard: async () => {
    const dashboard = await getDashboardSummary()
    set({ dashboard })
  },
  loadBatteries: async (page = 1, pageSize = 10) => {
    const response = await listBatteries(page, pageSize)
    set((state) => ({
      batteries: response.items,
      pagination: { page: response.page, pageSize: response.page_size, total: response.total },
      selectedBatteryId: state.selectedBatteryId ?? response.items[0]?.battery_id ?? null,
    }))
  },
  selectBattery: (batteryId) => set({ selectedBatteryId: batteryId }),
  loadBatteryContext: async (batteryId) => {
    set({ actionLoading: true, error: null, selectedBatteryId: batteryId })
    try {
      const [health, cycles, history] = await Promise.all([
        getBatteryHealth(batteryId),
        getBatteryCycles(batteryId),
        getBatteryHistory(batteryId),
      ])
      set((state) => ({
        batteryHealth: { ...state.batteryHealth, [batteryId]: health },
        batteryCycles: { ...state.batteryCycles, [batteryId]: cycles.items },
        batteryHistory: { ...state.batteryHistory, [batteryId]: history },
      }))
    } catch (error) {
      set({ error: error instanceof Error ? error.message : '加载电池上下文失败' })
    } finally {
      set({ actionLoading: false })
    }
  },
  uploadFile: async (file, options) => {
    set({ actionLoading: true, error: null })
    try {
      const result = await uploadBatteryData(file, options)
      set({ lastUpload: result })
      await Promise.all([get().loadDashboard(), get().loadBatteries(get().pagination.page, get().pagination.pageSize)])
      const target = result.battery_ids[0]
      if (target) {
        await get().loadBatteryContext(target)
      }
      return result
    } catch (error) {
      const message = error instanceof Error ? error.message : '上传失败'
      set({ error: message })
      throw error
    } finally {
      set({ actionLoading: false })
    }
  },
  importSource: async (source, includeInTraining = false) => {
    set({ actionLoading: true, error: null })
    try {
      const result = await importSourceData(source, undefined, includeInTraining)
      set({ lastUpload: result })
      await Promise.all([get().loadDashboard(), get().loadBatteries(1, get().pagination.pageSize)])
      const first = result.battery_ids[0]
      if (first) {
        await get().loadBatteryContext(first)
      }
      return result
    } catch (error) {
      const message = error instanceof Error ? error.message : `导入 ${source} 数据失败`
      set({ error: message })
      throw error
    } finally {
      set({ actionLoading: false })
    }
  },
  runPrediction: async (batteryId, modelName, seqLen) => {
    set({ actionLoading: true, error: null })
    try {
      const prediction = await predictRul({ battery_id: batteryId, model_name: modelName, seq_len: seqLen })
      const history = await getBatteryHistory(batteryId)
      const health = await getBatteryHealth(batteryId)
      set((state) => ({
        latestPrediction: { ...state.latestPrediction, [batteryId]: prediction },
        batteryHistory: { ...state.batteryHistory, [batteryId]: history },
        batteryHealth: { ...state.batteryHealth, [batteryId]: health },
      }))
      return prediction
    } catch (error) {
      const message = error instanceof Error ? error.message : '预测失败'
      set({ error: message })
      throw error
    } finally {
      set({ actionLoading: false })
    }
  },
  runDiagnosisWorkflow: async (batteryId) => {
    set({ actionLoading: true, error: null })
    try {
      const anomaly = await detectAnomaly({ battery_id: batteryId, use_latest: true })
      const diagnosis = await diagnoseBattery({ battery_id: batteryId, anomalies: anomaly.events })
      const history = await getBatteryHistory(batteryId)
      const health = await getBatteryHealth(batteryId)
      set((state) => ({
        latestAnomaly: { ...state.latestAnomaly, [batteryId]: anomaly },
        latestDiagnosis: { ...state.latestDiagnosis, [batteryId]: diagnosis },
        batteryHistory: { ...state.batteryHistory, [batteryId]: history },
        batteryHealth: { ...state.batteryHealth, [batteryId]: health },
      }))
      return { anomaly, diagnosis }
    } catch (error) {
      const message = error instanceof Error ? error.message : '诊断失败'
      set({ error: message })
      throw error
    } finally {
      set({ actionLoading: false })
    }
  },
}))
