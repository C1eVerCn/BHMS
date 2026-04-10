import React, { useMemo, useState } from 'react'
import { Button, Checkbox, Input, Progress, Select, Space, Upload, message } from 'antd'
import type { UploadFile } from 'antd/es/upload/interface'
import { CloudDownloadOutlined, InboxOutlined, RocketOutlined, UploadOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

import { EmptyStateBlock, PageHero, PanelCard, StatusTag, StructuredDataList } from '../components/ui'
import { useBhmsStore } from '../stores/useBhmsStore'
import type { DemoPreset, SupportedSource } from '../types/domain'
import { formatModelLabel, formatSourceLabel, formatValidationSummary } from '../utils/display'

const { Dragger } = Upload

const sourceOptions = [
  { label: '自动识别', value: 'auto' },
  { label: 'NASA', value: 'nasa' },
  { label: 'CALCE', value: 'calce' },
  { label: 'Kaggle', value: 'kaggle' },
  { label: 'HUST', value: 'hust' },
  { label: 'MATR', value: 'matr' },
  { label: 'Oxford', value: 'oxford' },
  { label: 'PulseBat', value: 'pulsebat' },
] as const

const builtinSources = sourceOptions.filter((item) => item.value !== 'auto') as Array<{ label: string; value: SupportedSource }>

const DataUpload: React.FC = () => {
  const navigate = useNavigate()
  const [batteryId, setBatteryId] = useState('')
  const [selectedSource, setSelectedSource] = useState<SupportedSource | 'auto'>('auto')
  const [includeInTraining, setIncludeInTraining] = useState(false)
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [localProgress, setLocalProgress] = useState(0)
  const [activePresetName, setActivePresetName] = useState<string | null>(null)

  const actionLoading = useBhmsStore((state) => state.actionLoading)
  const demoPresets = useBhmsStore((state) => state.demoPresets)
  const lastUpload = useBhmsStore((state) => state.lastUpload)
  const lifecycleRequestConfig = useBhmsStore((state) => state.lifecycleRequestConfig)
  const importSource = useBhmsStore((state) => state.importSource)
  const importPreset = useBhmsStore((state) => state.importPreset)
  const uploadFile = useBhmsStore((state) => state.uploadFile)
  const runDiagnosisWorkflow = useBhmsStore((state) => state.runDiagnosisWorkflow)
  const selectBattery = useBhmsStore((state) => state.selectBattery)
  const loadBatteryContext = useBhmsStore((state) => state.loadBatteryContext)

  const presetCards = useMemo(
    () => [...demoPresets].sort((left, right) => Number(right.recommended) - Number(left.recommended) || left.source.localeCompare(right.source) || left.name.localeCompare(right.name)),
    [demoPresets],
  )

  const uploadProps = {
    name: 'file',
    multiple: false,
    accept: '.csv,.mat,.zip',
    beforeUpload: () => false,
    fileList,
    onChange: ({ fileList: nextFileList }: { fileList: UploadFile[] }) => setFileList(nextFileList.slice(-1)),
    onRemove: () => setFileList([]),
  }

  const latestUploadedBattery = lastUpload?.battery_ids?.[0]
  const latestUploadItems = useMemo(
    () =>
      lastUpload
        ? [
            { label: '来源', value: formatSourceLabel(lastUpload.detected_source ?? lastUpload.source) },
            { label: '数据集', value: lastUpload.dataset_name ?? '--' },
            { label: '文件名', value: lastUpload.file_name },
            { label: '导入电池', value: lastUpload.battery_ids },
            { label: '周期点', value: `${lastUpload.imported_cycles} 条` },
            { label: '用途', value: lastUpload.include_in_training ? '训练数据' : '展示样本' },
          ]
        : [],
    [lastUpload],
  )
  const validationItems = useMemo(() => formatValidationSummary(lastUpload?.validation_summary ?? {}), [lastUpload])

  const handleUpload = async () => {
    const currentFile = fileList[0]?.originFileObj
    if (!currentFile) {
      message.warning('请先选择要导入的文件')
      return
    }
    setLocalProgress(20)
    try {
      await uploadFile(currentFile, {
        batteryId: batteryId || undefined,
        source: selectedSource,
        includeInTraining,
      })
      setLocalProgress(100)
      setFileList([])
      setBatteryId('')
      message.success('文件已成功导入数据库')
    } catch (error) {
      setLocalProgress(0)
      message.error(error instanceof Error ? error.message : '上传失败')
    }
  }

  const handleImportPreset = async (preset: DemoPreset) => {
    setActivePresetName(preset.name)
    try {
      await importPreset(preset.name, false)
      message.success(`预置样本 ${preset.name} 已导入`)
    } catch (error) {
      message.error(error instanceof Error ? error.message : `预置样本 ${preset.name} 导入失败`)
    } finally {
      setActivePresetName(null)
    }
  }

  const handleImportSource = async (source: SupportedSource) => {
    setLocalProgress(10)
    try {
      await importSource(source, includeInTraining)
      setLocalProgress(100)
      message.success(`${formatSourceLabel(source)} 数据已导入`)
    } catch (error) {
      setLocalProgress(0)
      message.error(error instanceof Error ? error.message : `${source} 导入失败`)
    }
  }

  const goToPrediction = () => {
    if (!latestUploadedBattery) return
    selectBattery(latestUploadedBattery)
    void loadBatteryContext(latestUploadedBattery)
    navigate('/prediction')
  }

  const goToFullAnalysis = () => {
    if (!latestUploadedBattery) return
    selectBattery(latestUploadedBattery)
    void loadBatteryContext(latestUploadedBattery)
    void runDiagnosisWorkflow(latestUploadedBattery, lifecycleRequestConfig)
      .then(() => navigate('/analysis'))
      .catch((error: Error) => message.error(error.message))
  }

  return (
    <div className="page-shell page-shell--stacked">
      <PageHero
        title="数据导入"
        description="支持导入预置样本、本地文件和整源数据，导入完成后可直接进入预测与分析流程。"
        pills={[
          { label: '预置样本', value: presetCards.length, tone: 'teal' },
          { label: '支持格式', value: 'CSV / MAT / ZIP', tone: 'slate' },
          { label: '最近导入周期点', value: lastUpload?.imported_cycles ?? 0, tone: 'amber' },
        ]}
      />

      <PanelCard title="预置样本">
        <Space direction="vertical" size={18} style={{ width: '100%' }}>
          {presetCards.length ? (
            <div className="demo-preset-grid">
              {presetCards.map((preset) => (
                <article className="demo-preset-card" key={preset.name}>
                  <div className="demo-preset-card__head">
                    <div>
                      <div className="demo-preset-card__title">{preset.name}</div>
                      <div className="demo-preset-card__subtitle">{formatSourceLabel(preset.source)} · {formatScenarioLabel(preset.scenario)}</div>
                    </div>
                    <Space wrap>
                      <StatusTag tone="info">来源 {formatSourceLabel(preset.source)}</StatusTag>
                      {preset.recommended ? <StatusTag tone="good">推荐</StatusTag> : null}
                    </Space>
                  </div>
                  <p className="demo-preset-card__description">{preset.description}</p>
                  <div className="demo-preset-card__footer">
                    <TextRow label="场景" value={formatScenarioCopy(preset.scenario)} />
                    <Button type="primary" loading={actionLoading && activePresetName === preset.name} onClick={() => void handleImportPreset(preset)}>
                      一键导入
                    </Button>
                  </div>
                </article>
              ))}
            </div>
          ) : (
            <EmptyStateBlock compact title="暂无预置样本" description="如果预置样本还未准备好，可以先使用下面的本地文件上传或整源导入入口。" className="panel-empty-state" />
          )}
        </Space>
      </PanelCard>

      <PanelCard title="最新导入结果">
        {lastUpload ? (
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            <StructuredDataList items={latestUploadItems} />
            {validationItems.length ? (
              <div className="panel-section-block">
                <div className="panel-section-label">校验摘要</div>
                <StructuredDataList items={validationItems} compact />
              </div>
            ) : null}
            {latestUploadedBattery ? (
              <Space wrap>
                <Button type="primary" icon={<RocketOutlined />} onClick={goToPrediction}>
                  进入生命周期预测
                </Button>
                <Button icon={<RocketOutlined />} loading={actionLoading} onClick={goToFullAnalysis}>
                  一键进入完整分析链路（{formatModelLabel(lifecycleRequestConfig.modelName)}）
                </Button>
              </Space>
            ) : null}
          </Space>
        ) : (
          <EmptyStateBlock compact title="暂无导入记录" description="导入一次预置样本或本地文件后，这里会展示结构化的校验结果和后续操作入口。" className="panel-empty-state" />
        )}
      </PanelCard>

      <PanelCard title="本地文件上传">
        <Space direction="vertical" size={18} style={{ width: '100%' }}>
          <div className="stacked-form-grid stacked-form-grid--three">
            <div>
              <div className="panel-section-label">来源识别</div>
              <Select value={selectedSource} onChange={setSelectedSource} options={sourceOptions as unknown as { label: string; value: string }[]} />
            </div>
            <div>
              <div className="panel-section-label">指定电池 ID</div>
              <Input value={batteryId} onChange={(event) => setBatteryId(event.target.value)} placeholder="可选：文件里没有 battery_id 时再填写" />
            </div>
            <div>
              <div className="panel-section-label">训练用途</div>
              <Checkbox checked={includeInTraining} onChange={(event) => setIncludeInTraining(event.target.checked)}>
                导入后同时加入训练数据
              </Checkbox>
            </div>
          </div>
          <Dragger {...uploadProps} disabled={actionLoading} className="upload-dragger">
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">点击或拖拽 CSV / MAT / ZIP 文件到此区域上传</p>
            <p className="ant-upload-hint">支持 NASA / CALCE / Kaggle / HUST / MATR / Oxford / PulseBat，系统会尽量沿用现有导入链路自动识别。</p>
          </Dragger>
          <Space wrap>
            <Button type="primary" icon={<UploadOutlined />} loading={actionLoading} onClick={() => void handleUpload()}>
              上传并入库
            </Button>
            {localProgress > 0 ? <Progress percent={localProgress} strokeColor="#0f6fff" style={{ width: 240 }} /> : null}
          </Space>
        </Space>
      </PanelCard>

      <PanelCard title="整源导入">
        <Space direction="vertical" size={16} style={{ width: '100%' }}>
          <Space wrap>
            {builtinSources.map((item) => (
              <Button key={item.value} icon={<CloudDownloadOutlined />} loading={actionLoading} onClick={() => void handleImportSource(item.value)}>
                导入 {item.label}
              </Button>
            ))}
          </Space>
        </Space>
      </PanelCard>
    </div>
  )
}

function formatScenarioLabel(scenario: string) {
  return scenario === 'fault_case' ? '故障样本' : '未见样本'
}

function formatScenarioCopy(scenario: string) {
  return scenario === 'fault_case' ? '适合展示诊断与机理解释链路' : '适合展示全周期寿命预测链路'
}

function TextRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="mini-meta-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

export default DataUpload
