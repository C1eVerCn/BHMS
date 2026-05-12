export function compactText(value: string | null | undefined, max = 28, fallback = '--') {
  if (!value) return fallback
  const normalized = String(value).trim()
  if (!normalized) return fallback
  if (normalized.length <= max) return normalized
  const keep = Math.max(6, Math.floor((max - 3) / 2))
  return `${normalized.slice(0, keep)}...${normalized.slice(-keep)}`
}

export function compactBatteryId(value: string | null | undefined, max = 22) {
  if (!value) return '未选择'
  const tail = value.split('::').pop() || value
  return compactText(tail, max, '未选择')
}

const sourceLabels: Record<string, string> = {
  nasa: 'NASA',
  calce: 'CALCE',
  kaggle: 'Kaggle',
  hust: 'HUST',
  matr: 'MATR',
  oxford: 'Oxford',
  pulsebat: 'PulseBat',
}

const modelLabels: Record<string, string> = {
  hybrid: 'xLSTM-Transformer',
  lifecycle_hybrid: 'xLSTM-Transformer',
  bilstm: 'Bi-LSTM',
  lifecycle_bilstm: 'Bi-LSTM',
  heuristic: '启发式',
  auto: '自动',
}

const projectionLabels: Record<string, string> = {
  linear: '线性投影',
  exponential: '指数投影',
}

const featureLabels: Record<string, string> = {
  voltage_mean: '平均电压',
  voltage_std: '电压波动',
  voltage_min: '最低电压',
  voltage_max: '最高电压',
  current_mean: '平均电流',
  current_std: '电流波动',
  current_load_mean: '负载电流',
  temperature_mean: '平均温度',
  temperature_std: '温度波动',
  temperature_rise_rate: '温升速率',
  internal_resistance: '内阻',
  capacity: '容量',
  cycle_number: '循环次数',
  capacity_ratio: '容量比例',
}

const severityLabels: Record<string, string> = {
  critical: '高风险',
  high: '高',
  medium: '中',
  low: '低',
  info: '正常',
  success: '正常',
  warning: '预警',
  good: '良好',
  neutral: '一般',
  unknown: '未知',
}

const trainingScopeLabels: Record<string, string> = {
  all: '全部模型',
  bilstm: 'Bi-LSTM',
  hybrid: 'Hybrid',
}

const trainingJobKindLabels: Record<string, string> = {
  baseline: '基线训练',
  multi_seed: '多随机种子',
  ablation: '消融实验',
  full_suite: '完整实验套件',
}

const trainingStatusLabels: Record<string, string> = {
  queued: '排队中',
  running: '运行中',
  completed: '已完成',
  failed: '失败',
}

const trainingStageLabels: Record<string, string> = {
  queued: '等待开始',
  prepare_dataset: '准备数据',
  train_bilstm: '训练 Bi-LSTM',
  train_hybrid: '训练 Hybrid',
  baseline_bilstm: '基线 Bi-LSTM',
  baseline_hybrid: '基线 Hybrid',
  multi_seed_bilstm: 'Bi-LSTM 多随机种子',
  multi_seed_hybrid: 'Hybrid 多随机种子',
  ablation_hybrid: 'Hybrid 消融实验',
  compare_models: '生成对比',
  generate_plots: '生成图表',
  completed: '完成',
  failed: '失败',
}

export function formatSourceLabel(value: string | null | undefined) {
  if (!value) return '--'
  return sourceLabels[value.toLowerCase()] ?? value.toUpperCase()
}

export function formatModelLabel(value: string | null | undefined) {
  if (!value) return '--'
  return modelLabels[value.toLowerCase()] ?? value
}

export function formatProjectionMethod(value: string | null | undefined) {
  if (!value) return '--'
  return projectionLabels[value.toLowerCase()] ?? value
}

export function formatFeatureLabel(value: string | null | undefined) {
  if (!value) return '--'
  return featureLabels[value] ?? value
}

export function formatSeverityLabel(value: string | null | undefined, fallback = '--') {
  if (!value) return fallback
  return severityLabels[value.toLowerCase()] ?? value
}

export function formatTrainingScopeLabel(value: string | null | undefined) {
  if (!value) return '--'
  return trainingScopeLabels[value.toLowerCase()] ?? value
}

export function formatTrainingJobKindLabel(value: string | null | undefined) {
  if (!value) return '--'
  return trainingJobKindLabels[value.toLowerCase()] ?? value
}

export function formatTrainingStatusLabel(value: string | null | undefined) {
  if (!value) return '--'
  return trainingStatusLabels[value.toLowerCase()] ?? value
}

export function formatTrainingStageLabel(value: string | null | undefined) {
  if (!value) return '--'
  return trainingStageLabels[value.toLowerCase()] ?? value
}

export function replaceTechnicalTerms(text: string | null | undefined) {
  if (!text) return ''
  return Object.entries(featureLabels).reduce((result, [key, label]) => result.split(key).join(label), text)
}

export function formatValidationSummary(summary: Record<string, unknown>) {
  const items: Array<{ label: string; value: string }> = []
  if (typeof summary.battery_count === 'number') {
    items.push({ label: '导入电池', value: `${summary.battery_count} 节` })
  }
  if (typeof summary.imported_cycles === 'number') {
    items.push({ label: '周期点', value: `${summary.imported_cycles} 条` })
  }
  if (typeof summary.source === 'string') {
    items.push({ label: '识别来源', value: formatSourceLabel(summary.source) })
  }
  if (typeof summary.dataset_name === 'string') {
    items.push({ label: '数据集', value: summary.dataset_name })
  }
  if (typeof summary.file_type === 'string') {
    items.push({ label: '文件类型', value: summary.file_type.toUpperCase() })
  }
  if (typeof summary.ingestion_mode === 'string') {
    const ingestionMode = summary.ingestion_mode
    let label = '内置样例'
    if (ingestionMode === 'uploaded_file') {
      label = '上传文件'
    } else if (String(ingestionMode).includes('raw_converter')) {
      label = '原始资产转换'
    } else if (String(ingestionMode).includes('enhancement_assets')) {
      label = '增强资产建档'
    } else if (String(ingestionMode).includes('csv_ready')) {
      label = '标准 CSV 导入'
    }
    items.push({ label: '导入方式', value: label })
  }
  if (typeof summary.ready_for_immediate_analysis === 'boolean') {
    items.push({ label: '分析状态', value: summary.ready_for_immediate_analysis ? '可立即分析' : '待补充数据' })
  }
  if (typeof summary.include_in_training === 'boolean') {
    items.push({ label: '训练池', value: summary.include_in_training ? '已加入' : '仅导入' })
  }
  if (summary.source_distribution && typeof summary.source_distribution === 'object') {
    const labels = Object.entries(summary.source_distribution as Record<string, number>).map(([key, count]) => `${formatSourceLabel(key)} ${count} 节`)
    if (labels.length) {
      items.push({ label: '来源分布', value: labels.join(' / ') })
    }
  }
  return items
}
