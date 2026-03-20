import { create } from 'zustand'

import {
  createTrainingJob,
  detectAnomaly,
  downloadDiagnosisReport,
  downloadPredictionReport,
  exportCaseBundle,
  explainMechanism,
  getBatteryCycles,
  getBatteryHealth,
  getBatteryHistory,
  getCaseBundle,
  getDashboardSummary,
  getDatasetProfile,
  getKnowledgeSummary,
  getSystemStatus,
  getTrainingAblationSummary,
  getTrainingComparison,
  getTrainingExperimentDetail,
  getTrainingOverview,
  importSourceData,
  listBatteries,
  listTrainingJobs,
  predictLifecycle,
  updateTrainingCandidate,
  uploadBatteryData,
} from '../services/bhms'
import type {
  AblationResult,
  AnomalyDetectionResult,
  Battery,
  BatteryHealth,
  BatteryHistory,
  CaseBundle,
  CaseBundleExportResult,
  CyclePoint,
  DashboardSummary,
  DatasetProfile,
  DiagnosisRecord,
  DiagnosisResult,
  ExperimentDetail,
  ExperimentOverview,
  KnowledgeSummary,
  LifecyclePredictionResult,
  MechanismExplanationResult,
  PredictionRecord,
  PredictionResult,
  SupportedSource,
  SystemStatus,
  TrainingComparison,
  TrainingJob,
  UploadSummary,
} from '../types/domain'

interface UploadOptions {
  batteryId?: string
  source?: SupportedSource | 'auto'
  includeInTraining?: boolean
}

interface BhmsState {
  dashboard: DashboardSummary | null
  batteries: Battery[]
  selectedBatteryId: string | null
  batteryHealth: Record<string, BatteryHealth>
  batteryCycles: Record<string, CyclePoint[]>
  batteryHistory: Record<string, BatteryHistory>
  latestPrediction: Record<string, PredictionResult | LifecyclePredictionResult>
  latestLifecyclePrediction: Record<string, LifecyclePredictionResult>
  latestAnomaly: Record<string, AnomalyDetectionResult>
  latestDiagnosis: Record<string, DiagnosisResult>
  latestMechanismExplanation: Record<string, MechanismExplanationResult>
  trainingJobs: TrainingJob[]
  trainingComparison: Record<string, TrainingComparison>
  trainingOverview: ExperimentOverview | null
  experimentDetails: Record<string, ExperimentDetail>
  ablationSummary: Record<string, AblationResult>
  systemStatus: SystemStatus | null
  datasetProfiles: Record<string, DatasetProfile>
  knowledgeSummary: KnowledgeSummary | null
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
  selectBattery: (batteryId: string | null) => void
  loadBatteryContext: (batteryId: string) => Promise<void>
  uploadFile: (file: File, options?: UploadOptions) => Promise<UploadSummary>
  importSource: (source: SupportedSource, includeInTraining?: boolean) => Promise<UploadSummary>
  markTrainingCandidate: (batteryId: string, includeInTraining: boolean) => Promise<Battery>
  runLifecyclePrediction: (batteryId: string, modelName: string, seqLen: number) => Promise<LifecyclePredictionResult>
  runPrediction: (batteryId: string, modelName: string, seqLen: number) => Promise<LifecyclePredictionResult>
  runMechanismExplanation: (batteryId: string) => Promise<MechanismExplanationResult>
  runDiagnosisWorkflow: (batteryId: string) => Promise<{ anomaly: AnomalyDetectionResult; diagnosis: MechanismExplanationResult }>
  loadTrainingJobs: (source?: string) => Promise<void>
  loadTrainingComparison: (source: string) => Promise<void>
  loadTrainingOverview: () => Promise<void>
  loadExperimentDetail: (source: string) => Promise<void>
  loadAblationSummary: (source: string) => Promise<void>
  startTrainingJob: (
    source: SupportedSource,
    modelScope: 'bilstm' | 'hybrid' | 'all',
    jobKind?: 'baseline' | 'multi_seed' | 'ablation' | 'full_suite',
    forceRun?: boolean,
    seedCount?: number
  ) => Promise<TrainingJob>
  loadSystemStatus: () => Promise<void>
  loadDatasetProfile: (source: string) => Promise<void>
  loadKnowledgeSummary: () => Promise<void>
  loadCaseBundle: (batteryId: string) => Promise<void>
  exportCaseBundle: (batteryId: string, ensureArtifacts?: boolean) => Promise<CaseBundleExportResult>
  getPredictionReportText: (predictionId: number) => Promise<string>
  getDiagnosisReportText: (diagnosisId: number) => Promise<string>
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

function hydrateDiagnosis(record?: DiagnosisRecord): DiagnosisResult | undefined {
  if (!record || !record.graph_trace) return undefined
  return {
    id: record.id,
    battery_id: record.battery_id,
    fault_type: record.fault_type,
    confidence: record.confidence,
    severity: record.severity,
    description: record.description,
    root_causes: record.root_causes,
    recommendations: record.recommendations,
    related_symptoms: record.related_symptoms,
    evidence: record.evidence,
    diagnosis_time: record.created_at,
    candidate_faults: record.candidate_faults ?? [],
    graph_trace: record.graph_trace,
    decision_basis: record.decision_basis ?? [],
    report_markdown: record.report_markdown ?? '',
  }
}

export const useBhmsStore = create<BhmsState>((set, get) => ({
  dashboard: null,
  batteries: [],
  selectedBatteryId: null,
  batteryHealth: {},
  batteryCycles: {},
  batteryHistory: {},
  latestPrediction: {},
  latestLifecyclePrediction: {},
  latestAnomaly: {},
  latestDiagnosis: {},
  latestMechanismExplanation: {},
  trainingJobs: [],
  trainingComparison: {},
  trainingOverview: null,
  experimentDetails: {},
  ablationSummary: {},
  systemStatus: null,
  datasetProfiles: {},
  knowledgeSummary: null,
  caseBundles: {},
  caseBundleExports: {},
  pagination: { page: 1, pageSize: 10, total: 0 },
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
      const [health, cycles, history] = await Promise.all([getBatteryHealth(batteryId), getBatteryCycles(batteryId), getBatteryHistory(batteryId)])
      const hydratedPrediction = hydratePrediction(history.predictions[0])
      const hydratedDiagnosis = hydrateDiagnosis(history.diagnoses[0])
      set((state) => ({
        batteryHealth: { ...state.batteryHealth, [batteryId]: health },
        batteryCycles: { ...state.batteryCycles, [batteryId]: cycles.items },
        batteryHistory: { ...state.batteryHistory, [batteryId]: history },
        latestLifecyclePrediction: hydratedPrediction
          ? { ...state.latestLifecyclePrediction, [batteryId]: hydratedPrediction }
          : state.latestLifecyclePrediction,
        latestPrediction: hydratedPrediction ? { ...state.latestPrediction, [batteryId]: hydratedPrediction } : state.latestPrediction,
        latestDiagnosis: hydratedDiagnosis ? { ...state.latestDiagnosis, [batteryId]: hydratedDiagnosis } : state.latestDiagnosis,
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
  markTrainingCandidate: async (batteryId, includeInTraining) => {
    set({ actionLoading: true, error: null })
    try {
      const battery = await updateTrainingCandidate(batteryId, includeInTraining)
      set((state) => ({
        batteries: state.batteries.map((item) => (item.battery_id === batteryId ? { ...item, include_in_training: includeInTraining } : item)),
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
  runLifecyclePrediction: async (batteryId, modelName, seqLen) => {
    set({ actionLoading: true, error: null })
    try {
      const prediction = await predictLifecycle({ battery_id: batteryId, model_name: modelName, seq_len: seqLen })
      const history = await getBatteryHistory(batteryId)
      const health = await getBatteryHealth(batteryId)
      set((state) => ({
        latestLifecyclePrediction: { ...state.latestLifecyclePrediction, [batteryId]: prediction },
        latestPrediction: { ...state.latestPrediction, [batteryId]: prediction },
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
  runPrediction: async (batteryId, modelName, seqLen) => get().runLifecyclePrediction(batteryId, modelName, seqLen),
  runMechanismExplanation: async (batteryId) => {
    set({ actionLoading: true, error: null })
    try {
      const anomaly = get().latestAnomaly[batteryId]
      const explanation = await explainMechanism({ battery_id: batteryId, anomalies: anomaly?.events })
      const history = await getBatteryHistory(batteryId)
      const health = await getBatteryHealth(batteryId)
      set((state) => ({
        latestMechanismExplanation: { ...state.latestMechanismExplanation, [batteryId]: explanation },
        latestDiagnosis: { ...state.latestDiagnosis, [batteryId]: explanation },
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
  runDiagnosisWorkflow: async (batteryId) => {
    set({ actionLoading: true, error: null })
    try {
      const anomaly = await detectAnomaly({ battery_id: batteryId, use_latest: true })
      set((state) => ({ latestAnomaly: { ...state.latestAnomaly, [batteryId]: anomaly } }))
      const diagnosis = await explainMechanism({ battery_id: batteryId, anomalies: anomaly.events })
      const history = await getBatteryHistory(batteryId)
      const health = await getBatteryHealth(batteryId)
      set((state) => ({
        latestAnomaly: { ...state.latestAnomaly, [batteryId]: anomaly },
        latestMechanismExplanation: { ...state.latestMechanismExplanation, [batteryId]: diagnosis },
        latestDiagnosis: { ...state.latestDiagnosis, [batteryId]: diagnosis },
        batteryHistory: { ...state.batteryHistory, [batteryId]: history },
        batteryHealth: { ...state.batteryHealth, [batteryId]: health },
      }))
      return { anomaly, diagnosis }
    } catch (error) {
      const message = error instanceof Error ? error.message : '机理解释失败'
      set({ error: message })
      throw error
    } finally {
      set({ actionLoading: false })
    }
  },
  loadTrainingJobs: async (source) => {
    set({ trainingLoading: true, error: null })
    try {
      const trainingJobs = await listTrainingJobs(source)
      set({ trainingJobs })
    } catch (error) {
      set({ error: error instanceof Error ? error.message : '加载训练任务失败' })
    } finally {
      set({ trainingLoading: false })
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
  startTrainingJob: async (source, modelScope, jobKind = 'baseline', forceRun = false, seedCount = 3) => {
    set({ trainingLoading: true, error: null })
    try {
      const job = await createTrainingJob({
        source,
        model_scope: modelScope,
        force_run: forceRun,
        job_kind: jobKind,
        seed_count: seedCount,
      })
      await Promise.all([
        get().loadTrainingJobs(source),
        get().loadTrainingComparison(source),
        get().loadTrainingOverview(),
        get().loadExperimentDetail(source),
        get().loadAblationSummary(source),
      ])
      return job
    } catch (error) {
      const message = error instanceof Error ? error.message : '创建训练任务失败'
      set({ error: message })
      throw error
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
  getPredictionReportText: async (predictionId) => downloadPredictionReport(predictionId),
  getDiagnosisReportText: async (diagnosisId) => downloadDiagnosisReport(diagnosisId),
}))
