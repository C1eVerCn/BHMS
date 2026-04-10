import { create } from 'zustand'

import {
  detectAnomaly,
  exportCaseBundle,
  explainMechanism,
  getBattery,
  getBatteryCycles,
  getBatteryHealth,
  getBatteryHistory,
  getCaseBundle,
  getDashboardSummary,
  getDemoPresets,
  getDatasetProfile,
  getKnowledgeSummary,
  getSystemStatus,
  getTrainingAblationSummary,
  getTrainingComparison,
  getTrainingExperimentDetail,
  getTrainingOverview,
  importDemoPreset,
  importSourceData,
  listBatteryOptions,
  listBatteries,
  predictLifecycle,
  updateTrainingCandidate,
  uploadBatteryData,
} from '../services/bhms'
import type {
  AblationResult,
  AnomalyDetectionResult,
  Battery,
  BatteryOption,
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
  PredictionRecord,
  SupportedSource,
  SystemStatus,
  TrainingComparison,
  UploadSummary,
} from '../types/domain'

interface UploadOptions {
  batteryId?: string
  source?: SupportedSource | 'auto'
  includeInTraining?: boolean
}

type StoredBatteryProfile = BatteryOption & Partial<Battery>
type SelectableLifecycleModelName = Exclude<LifecycleModelName, 'auto'>

interface LifecycleRequestConfig {
  modelName: SelectableLifecycleModelName
  seqLen: number
}

const defaultLifecycleRequestConfig: LifecycleRequestConfig = {
  modelName: 'hybrid',
  seqLen: 30,
}

interface BhmsState {
  dashboard: DashboardSummary | null
  batteryPageItems: Battery[]
  batteryOptions: BatteryOption[]
  batteryById: Record<string, StoredBatteryProfile>
  selectedBatteryId: string | null
  batteryHealth: Record<string, BatteryHealth>
  batteryCycles: Record<string, CyclePoint[]>
  batteryHistory: Record<string, BatteryHistory>
  latestLifecyclePrediction: Record<string, LifecyclePredictionResult>
  latestAnomaly: Record<string, AnomalyDetectionResult>
  latestMechanismExplanation: Record<string, MechanismExplanationResult>
  lifecycleRequestConfig: LifecycleRequestConfig
  trainingComparison: Record<string, TrainingComparison>
  trainingOverview: ExperimentOverview | null
  experimentDetails: Record<string, ExperimentDetail>
  ablationSummary: Record<string, AblationResult>
  systemStatus: SystemStatus | null
  datasetProfiles: Record<string, DatasetProfile>
  knowledgeSummary: KnowledgeSummary | null
  demoPresets: DemoPreset[]
  caseBundles: Record<string, CaseBundle>
  caseBundleExports: Record<string, CaseBundleExportResult>
  pagination: { page: number; pageSize: number; total: number }
  loading: boolean
  actionLoading: boolean
  trainingLoading: boolean
  insightLoading: boolean
  error: string | null
  lastUpload: UploadSummary | null
  init: () => Promise<void>
  loadDashboard: () => Promise<void>
  loadBatteries: (page?: number, pageSize?: number) => Promise<void>
  loadBatteryOptions: () => Promise<void>
  loadDemoPresets: () => Promise<void>
  selectBattery: (batteryId: string | null) => void
  loadBatteryContext: (batteryId: string) => Promise<void>
  uploadFile: (file: File, options?: UploadOptions) => Promise<UploadSummary>
  importSource: (source: SupportedSource, includeInTraining?: boolean) => Promise<UploadSummary>
  importPreset: (presetName: string, includeInTraining?: boolean) => Promise<UploadSummary>
  markTrainingCandidate: (batteryId: string, includeInTraining: boolean) => Promise<Battery>
  setLifecycleRequestConfig: (patch: Partial<LifecycleRequestConfig>) => void
  runLifecyclePrediction: (batteryId: string, modelName: SelectableLifecycleModelName, seqLen: number) => Promise<LifecyclePredictionResult>
  runMechanismExplanation: (batteryId: string, options?: Partial<LifecycleRequestConfig>) => Promise<MechanismExplanationResult>
  runDiagnosisWorkflow: (
    batteryId: string,
    options?: Partial<LifecycleRequestConfig>
  ) => Promise<{ prediction: LifecyclePredictionResult; anomaly: AnomalyDetectionResult; diagnosis: MechanismExplanationResult }>
  loadTrainingComparison: (source: string) => Promise<void>
  loadTrainingOverview: () => Promise<void>
  loadExperimentDetail: (source: string) => Promise<void>
  loadAblationSummary: (source: string) => Promise<void>
  loadSystemStatus: () => Promise<void>
  loadDatasetProfile: (source: string) => Promise<void>
  loadKnowledgeSummary: () => Promise<void>
  loadCaseBundle: (batteryId: string) => Promise<void>
  exportCaseBundle: (batteryId: string, ensureArtifacts?: boolean) => Promise<CaseBundleExportResult>
  clearError: () => void
}

function hydratePrediction(record?: PredictionRecord): LifecyclePredictionResult | undefined {
  if (!record || !record.projection || !record.model_version || !record.model_source) return undefined
  return {
    ...record,
    model_version: record.model_version,
    model_source: record.model_source,
    fallback_used: Boolean(record.fallback_used),
    prediction_time: record.prediction_time ?? record.created_at,
    trajectory: record.trajectory ?? [],
    risk_windows: record.risk_windows ?? [],
    future_risks: record.future_risks ?? {},
    model_evidence: record.model_evidence ?? {},
    projection: record.projection,
    explanation: record.explanation ?? null,
    report_markdown: record.report_markdown ?? '',
  }
}

function sortBatteryOptions(items: BatteryOption[]) {
  return [...items].sort((left, right) => {
    const trainingDelta = Number(Boolean(left.include_in_training)) - Number(Boolean(right.include_in_training))
    if (trainingDelta !== 0) return trainingDelta
    const sourceDelta = left.source.localeCompare(right.source)
    if (sourceDelta !== 0) return sourceDelta
    return left.battery_id.localeCompare(right.battery_id)
  })
}

function toStoredBatteryProfile(item: BatteryOption): StoredBatteryProfile {
  return { ...item }
}

export const useBhmsStore = create<BhmsState>((set, get) => ({
  dashboard: null,
  batteryPageItems: [],
  batteryOptions: [],
  batteryById: {},
  selectedBatteryId: null,
  batteryHealth: {},
  batteryCycles: {},
  batteryHistory: {},
  latestLifecyclePrediction: {},
  latestAnomaly: {},
  latestMechanismExplanation: {},
  lifecycleRequestConfig: defaultLifecycleRequestConfig,
  trainingComparison: {},
  trainingOverview: null,
  experimentDetails: {},
  ablationSummary: {},
  systemStatus: null,
  datasetProfiles: {},
  knowledgeSummary: null,
  demoPresets: [],
  caseBundles: {},
  caseBundleExports: {},
  pagination: { page: 1, pageSize: 20, total: 0 },
  loading: false,
  actionLoading: false,
  trainingLoading: false,
  insightLoading: false,
  error: null,
  lastUpload: null,
  clearError: () => set({ error: null }),
  init: async () => {
    if (get().loading) return
    set({ loading: true, error: null })
    try {
      await Promise.all([get().loadDashboard(), get().loadBatteries(1, get().pagination.pageSize), get().loadBatteryOptions(), get().loadDemoPresets()])
      const firstBattery = get().selectedBatteryId ?? get().batteryOptions[0]?.battery_id ?? null
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
  loadBatteries: async (page = 1, pageSize = 20) => {
    const response = await listBatteries(page, pageSize)
    set((state) => ({
      batteryPageItems: response.items,
      pagination: { page: response.page, pageSize: response.page_size, total: response.total },
      batteryById: {
        ...state.batteryById,
        ...Object.fromEntries(response.items.map((item) => [item.battery_id, { ...(state.batteryById[item.battery_id] ?? {}), ...item }])),
      },
      selectedBatteryId: state.selectedBatteryId ?? state.batteryOptions[0]?.battery_id ?? response.items[0]?.battery_id ?? null,
    }))
  },
  loadBatteryOptions: async () => {
    const response = await listBatteryOptions()
    const items = sortBatteryOptions(response.items)
    set((state) => ({
      batteryOptions: items,
      batteryById: {
        ...state.batteryById,
        ...Object.fromEntries(items.map((item) => [item.battery_id, { ...(state.batteryById[item.battery_id] ?? {}), ...toStoredBatteryProfile(item) }])),
      },
      selectedBatteryId: state.selectedBatteryId ?? items[0]?.battery_id ?? null,
    }))
  },
  loadDemoPresets: async () => {
    const demoPresets = await getDemoPresets()
    set({ demoPresets })
  },
  selectBattery: (batteryId) => set({ selectedBatteryId: batteryId }),
  loadBatteryContext: async (batteryId) => {
    set({ actionLoading: true, error: null, selectedBatteryId: batteryId })
    try {
      const [battery, health, cycles, history] = await Promise.all([getBattery(batteryId), getBatteryHealth(batteryId), getBatteryCycles(batteryId), getBatteryHistory(batteryId)])
      const hydratedPrediction = hydratePrediction(history.predictions[0])
      set((state) => ({
        batteryById: { ...state.batteryById, [batteryId]: { ...(state.batteryById[batteryId] ?? {}), ...battery } },
        batteryHealth: { ...state.batteryHealth, [batteryId]: health },
        batteryCycles: { ...state.batteryCycles, [batteryId]: cycles.items },
        batteryHistory: { ...state.batteryHistory, [batteryId]: history },
        latestLifecyclePrediction: hydratedPrediction
          ? { ...state.latestLifecyclePrediction, [batteryId]: hydratedPrediction }
          : state.latestLifecyclePrediction,
        latestAnomaly: {
          ...state.latestAnomaly,
          [batteryId]: {
            battery_id: batteryId,
            is_anomaly: history.anomalies.length > 0,
            max_severity: history.anomalies[0]?.severity ?? null,
            summary: history.anomalies.length ? '已从历史异常记录中恢复最近一次异常列表。' : '暂无异常记录。',
            event_ids: [],
            events: history.anomalies,
            detection_time: history.diagnoses[0]?.created_at ?? new Date().toISOString(),
          },
        },
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
      await Promise.all([get().loadDashboard(), get().loadBatteries(get().pagination.page, get().pagination.pageSize), get().loadBatteryOptions()])
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
      await Promise.all([get().loadDashboard(), get().loadBatteries(1, get().pagination.pageSize), get().loadBatteryOptions()])
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
  importPreset: async (presetName, includeInTraining = false) => {
    set({ actionLoading: true, error: null })
    try {
      const result = await importDemoPreset(presetName, includeInTraining)
      set({ lastUpload: result })
      await Promise.all([get().loadDashboard(), get().loadBatteries(1, get().pagination.pageSize), get().loadBatteryOptions(), get().loadDemoPresets()])
      const first = result.battery_ids[0]
      if (first) {
        await get().loadBatteryContext(first)
      }
      return result
    } catch (error) {
      const message = error instanceof Error ? error.message : `导入演示样本 ${presetName} 失败`
      set({ error: message })
      throw error
    } finally {
      set({ actionLoading: false })
    }
  },
  markTrainingCandidate: async (batteryId, includeInTraining) => {
    set({ actionLoading: true, error: null })
    try {
      const battery = await updateTrainingCandidate(batteryId, includeInTraining)
      set((state) => ({
        batteryPageItems: state.batteryPageItems.map((item) => (item.battery_id === batteryId ? { ...item, include_in_training: includeInTraining } : item)),
        batteryOptions: sortBatteryOptions(
          state.batteryOptions.map((item) => (item.battery_id === batteryId ? { ...item, include_in_training: includeInTraining } : item))
        ),
        batteryById: {
          ...state.batteryById,
          [batteryId]: { ...(state.batteryById[batteryId] ?? {}), ...battery, include_in_training: includeInTraining },
        },
      }))
      await get().loadBatteryContext(batteryId)
      return battery
    } catch (error) {
      const message = error instanceof Error ? error.message : '更新训练候选状态失败'
      set({ error: message })
      throw error
    } finally {
      set({ actionLoading: false })
    }
  },
  setLifecycleRequestConfig: (patch) =>
    set((state) => ({
      lifecycleRequestConfig: {
        ...state.lifecycleRequestConfig,
        ...patch,
        seqLen: patch.seqLen ?? state.lifecycleRequestConfig.seqLen,
      },
    })),
  runLifecyclePrediction: async (batteryId, modelName, seqLen) => {
    set({ actionLoading: true, error: null })
    try {
      const prediction = await predictLifecycle({ battery_id: batteryId, model_name: modelName, seq_len: seqLen })
      const history = await getBatteryHistory(batteryId)
      const health = await getBatteryHealth(batteryId)
      set((state) => ({
        lifecycleRequestConfig: { modelName, seqLen },
        latestLifecyclePrediction: { ...state.latestLifecyclePrediction, [batteryId]: prediction },
        batteryHistory: { ...state.batteryHistory, [batteryId]: history },
        batteryHealth: { ...state.batteryHealth, [batteryId]: health },
      }))
      return prediction
    } catch (error) {
      const message = error instanceof Error ? error.message : '生命周期预测失败'
      set({ error: message })
      throw error
    } finally {
      set({ actionLoading: false })
    }
  },
  runMechanismExplanation: async (batteryId, options) => {
    set({ actionLoading: true, error: null })
    try {
      const anomaly = get().latestAnomaly[batteryId]
      const lifecycleRequestConfig = { ...get().lifecycleRequestConfig, ...options }
      const explanation = await explainMechanism({
        battery_id: batteryId,
        anomalies: anomaly?.events,
        model_name: lifecycleRequestConfig.modelName,
        seq_len: lifecycleRequestConfig.seqLen,
      })
      const history = await getBatteryHistory(batteryId)
      const health = await getBatteryHealth(batteryId)
      const hydratedPrediction = hydratePrediction(history.predictions[0])
      set((state) => ({
        lifecycleRequestConfig,
        latestLifecyclePrediction: hydratedPrediction
          ? { ...state.latestLifecyclePrediction, [batteryId]: hydratedPrediction }
          : state.latestLifecyclePrediction,
        latestMechanismExplanation: { ...state.latestMechanismExplanation, [batteryId]: explanation },
        batteryHistory: { ...state.batteryHistory, [batteryId]: history },
        batteryHealth: { ...state.batteryHealth, [batteryId]: health },
      }))
      return explanation
    } catch (error) {
      const message = error instanceof Error ? error.message : '机理解释失败'
      set({ error: message })
      throw error
    } finally {
      set({ actionLoading: false })
    }
  },
  runDiagnosisWorkflow: async (batteryId, options) => {
    set({ actionLoading: true, error: null })
    try {
      const lifecycleRequestConfig = { ...get().lifecycleRequestConfig, ...options }
      const prediction = await predictLifecycle({
        battery_id: batteryId,
        model_name: lifecycleRequestConfig.modelName,
        seq_len: lifecycleRequestConfig.seqLen,
      })
      const anomaly = await detectAnomaly({ battery_id: batteryId, use_latest: true })
      set((state) => ({ latestAnomaly: { ...state.latestAnomaly, [batteryId]: anomaly } }))
      const diagnosis = await explainMechanism({
        battery_id: batteryId,
        anomalies: anomaly.events,
        model_name: lifecycleRequestConfig.modelName,
        seq_len: lifecycleRequestConfig.seqLen,
      })
      const history = await getBatteryHistory(batteryId)
      const health = await getBatteryHealth(batteryId)
      set((state) => ({
        lifecycleRequestConfig,
        latestLifecyclePrediction: { ...state.latestLifecyclePrediction, [batteryId]: prediction },
        latestAnomaly: { ...state.latestAnomaly, [batteryId]: anomaly },
        latestMechanismExplanation: { ...state.latestMechanismExplanation, [batteryId]: diagnosis },
        batteryHistory: { ...state.batteryHistory, [batteryId]: history },
        batteryHealth: { ...state.batteryHealth, [batteryId]: health },
      }))
      return { prediction, anomaly, diagnosis }
    } catch (error) {
      const message = error instanceof Error ? error.message : '机理解释失败'
      set({ error: message })
      throw error
    } finally {
      set({ actionLoading: false })
    }
  },
  loadTrainingComparison: async (source) => {
    set({ trainingLoading: true, error: null })
    try {
      const comparison = await getTrainingComparison(source)
      set((state) => ({ trainingComparison: { ...state.trainingComparison, [source]: comparison } }))
    } catch (error) {
      set({ error: error instanceof Error ? error.message : '加载训练对比失败' })
    } finally {
      set({ trainingLoading: false })
    }
  },
  loadTrainingOverview: async () => {
    set({ trainingLoading: true, error: null })
    try {
      const overview = await getTrainingOverview()
      set({ trainingOverview: overview })
    } catch (error) {
      set({ error: error instanceof Error ? error.message : '加载实验概览失败' })
    } finally {
      set({ trainingLoading: false })
    }
  },
  loadExperimentDetail: async (source) => {
    set({ trainingLoading: true, error: null })
    try {
      const detail = await getTrainingExperimentDetail(source)
      set((state) => ({ experimentDetails: { ...state.experimentDetails, [source]: detail } }))
    } catch (error) {
      set({ error: error instanceof Error ? error.message : '加载实验详情失败' })
    } finally {
      set({ trainingLoading: false })
    }
  },
  loadAblationSummary: async (source) => {
    set({ trainingLoading: true, error: null })
    try {
      const summary = await getTrainingAblationSummary(source)
      set((state) => ({ ablationSummary: { ...state.ablationSummary, [source]: summary } }))
    } catch (error) {
      set({ error: error instanceof Error ? error.message : '加载消融实验失败' })
    } finally {
      set({ trainingLoading: false })
    }
  },
  loadSystemStatus: async () => {
    set({ insightLoading: true, error: null })
    try {
      const status = await getSystemStatus()
      set({ systemStatus: status })
    } catch (error) {
      set({ error: error instanceof Error ? error.message : '加载系统状态失败' })
    } finally {
      set({ insightLoading: false })
    }
  },
  loadDatasetProfile: async (source) => {
    set({ insightLoading: true, error: null })
    try {
      const profile = await getDatasetProfile(source)
      set((state) => ({ datasetProfiles: { ...state.datasetProfiles, [source]: profile } }))
    } catch (error) {
      set({ error: error instanceof Error ? error.message : '加载数据画像失败' })
    } finally {
      set({ insightLoading: false })
    }
  },
  loadKnowledgeSummary: async () => {
    set({ insightLoading: true, error: null })
    try {
      const summary = await getKnowledgeSummary()
      set({ knowledgeSummary: summary })
    } catch (error) {
      set({ error: error instanceof Error ? error.message : '加载知识库摘要失败' })
    } finally {
      set({ insightLoading: false })
    }
  },
  loadCaseBundle: async (batteryId) => {
    set({ insightLoading: true, error: null })
    try {
      const bundle = await getCaseBundle(batteryId)
      set((state) => ({ caseBundles: { ...state.caseBundles, [batteryId]: bundle } }))
    } catch (error) {
      set({ error: error instanceof Error ? error.message : '加载案例包失败' })
    } finally {
      set({ insightLoading: false })
    }
  },
  exportCaseBundle: async (batteryId, ensureArtifacts = true) => {
    set({ insightLoading: true, error: null })
    try {
      const result = await exportCaseBundle(batteryId, ensureArtifacts)
      set((state) => ({
        caseBundleExports: { ...state.caseBundleExports, [batteryId]: result },
        caseBundles: { ...state.caseBundles, [batteryId]: result.bundle_snapshot },
      }))
      return result
    } catch (error) {
      const message = error instanceof Error ? error.message : '导出案例目录失败'
      set({ error: message })
      throw error
    } finally {
      set({ insightLoading: false })
    }
  },
}))
