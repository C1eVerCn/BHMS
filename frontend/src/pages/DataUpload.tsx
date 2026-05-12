import React, { useMemo, useState } from 'react'
import { Alert, Button, Checkbox, Col, Input, List, Progress, Row, Select, Space, Table, Upload, message } from 'antd'
import type { UploadFile } from 'antd/es/upload/interface'
import { CloudDownloadOutlined, InboxOutlined, RocketOutlined, UploadOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

import { EmptyStateBlock, InsightCard, PageHero, PanelCard, StatusTag } from '../components/ui'
import { useBhmsStore } from '../stores/useBhmsStore'
import type { SupportedSource } from '../types/domain'
import { formatSourceLabel } from '../utils/display'

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
  const actionLoading = useBhmsStore((state) => state.actionLoading)
  const lastUpload = useBhmsStore((state) => state.lastUpload)
  const importSource = useBhmsStore((state) => state.importSource)
  const uploadFile = useBhmsStore((state) => state.uploadFile)
  const runLifecyclePrediction = useBhmsStore((state) => state.runLifecyclePrediction)
  const runDiagnosisWorkflow = useBhmsStore((state) => state.runDiagnosisWorkflow)
  const selectBattery = useBhmsStore((state) => state.selectBattery)
  const loadBatteryContext = useBhmsStore((state) => state.loadBatteryContext)
  const markTrainingCandidate = useBhmsStore((state) => state.markTrainingCandidate)

  const sampleColumns = useMemo(
    () => [
      { title: 'battery_id/source_battery_id', dataIndex: 'battery_id', key: 'battery_id' },
      { title: 'cycle_number', dataIndex: 'cycle_number', key: 'cycle_number' },
      { title: 'voltage_mean', dataIndex: 'voltage_mean', key: 'voltage_mean' },
      { title: 'current_mean', dataIndex: 'current_mean', key: 'current_mean' },
      { title: 'temperature_mean', dataIndex: 'temperature_mean', key: 'temperature_mean' },
      { title: 'capacity', dataIndex: 'capacity', key: 'capacity' },
    ],
    [],
  )

  const sampleData = [
    { key: '1', battery_id: 'CALCE_001', cycle_number: 1, voltage_mean: 3.74, current_mean: -1.92, temperature_mean: 24.9, capacity: 1.85 },
    { key: '2', battery_id: 'CALCE_001', cycle_number: 2, voltage_mean: 3.71, current_mean: -1.96, temperature_mean: 25.6, capacity: 1.84 },
    { key: '3', battery_id: 'CALCE_001', cycle_number: 3, voltage_mean: 3.69, current_mean: -2.01, temperature_mean: 25.8, capacity: 1.83 },
  ]

  const uploadProps = {
    name: 'file',
    multiple: false,
    accept: '.csv,.mat,.zip',
    beforeUpload: () => false,
    fileList,
    onChange: ({ fileList: nextFileList }: { fileList: UploadFile[] }) => setFileList(nextFileList.slice(-1)),
    onRemove: () => setFileList([]),
  }

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

  const handleImportSource = async (source: (typeof builtinSources)[number]['value']) => {
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

  const validationSummary = lastUpload?.validation_summary ?? {}
  const latestUploadedBattery = lastUpload?.battery_ids?.[0]

  return (
    <div className="page-shell">
      <PageHero
        kicker="Lifecycle Data Ingestion"
        title="把 7 个数据源接入，变得更完整"
        description="上传文件、导入内置源，并在导入后直接串起 lifecycle -> mechanism -> analysis 流程。"
        pills={[
          { label: '支持格式', value: 'CSV / MAT / ZIP', tone: 'teal' },
          { label: '可选来源', value: builtinSources.length, tone: 'slate' },
          { label: '最近导入周期点', value: lastUpload?.imported_cycles ?? 0, tone: 'amber' },
        ]}
        aside={
          <InsightCard compact label="训练池标记" value={includeInTraining ? 'ON' : 'OFF'} description="打开后，新导入数据会被标记为后续训练候选。" />
        }
      />

      <Row gutter={[18, 18]}>
        <Col xs={24} lg={16}>
          <PanelCard className="upload-panel" title="上传新数据并接入生命周期工作流" style={{ marginBottom: 18 }}>
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <Alert type="info" showIcon message="导入顺序建议：选来源 -> 选择文件 -> 决定是否入训练池 -> 上传" />
              <Select value={selectedSource} onChange={setSelectedSource} options={sourceOptions as unknown as { label: string; value: string }[]} />
              <Input value={batteryId} onChange={(event) => setBatteryId(event.target.value)} placeholder="可选：若文件不包含 battery_id/source_battery_id，可在此填写" />
              <Checkbox checked={includeInTraining} onChange={(event) => setIncludeInTraining(event.target.checked)}>
                将该文件标记为后续 lifecycle 训练数据池候选
              </Checkbox>
              <Dragger {...uploadProps} disabled={actionLoading} className="upload-dragger">
                <p className="ant-upload-drag-icon">
                  <InboxOutlined />
                </p>
                <p className="ant-upload-text">点击或拖拽 CSV / MAT / ZIP 文件到此区域上传</p>
                <p className="ant-upload-hint">支持 NASA / CALCE / Kaggle / HUST / MATR / Oxford / PulseBat；Oxford/PulseBat 可先保留原始 MAT/ZIP 资产。</p>
              </Dragger>
              <Button type="primary" icon={<UploadOutlined />} loading={actionLoading} onClick={() => void handleUpload()}>
                上传并入库
              </Button>
              {localProgress > 0 ? <Progress percent={localProgress} strokeColor="#0071e3" /> : null}
            </Space>
          </PanelCard>

          <PanelCard title="导入仓库内置数据源">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Alert
                type="info"
                showIcon
                message="按来源独立导入"
                description="可分别导入 NASA、CALCE、Kaggle、HUST、MATR、Oxford、PulseBat；是否进入训练池由上方复选框决定。"
              />
              <Space wrap>
                {builtinSources.map((item) => (
                  <Button key={item.value} icon={<CloudDownloadOutlined />} loading={actionLoading} onClick={() => void handleImportSource(item.value)}>
                    导入 {item.label}
                  </Button>
                ))}
              </Space>
            </Space>
          </PanelCard>
        </Col>

        <Col xs={24} lg={8}>
          <PanelCard title="最新导入结果" style={{ marginBottom: 18 }}>
            {lastUpload ? (
              <Space direction="vertical" size={14} style={{ width: '100%' }}>
                <List
                  dataSource={[
                    `来源: ${formatSourceLabel(lastUpload.detected_source ?? lastUpload.source)}`,
                    `数据集: ${lastUpload.dataset_name ?? '-'}`,
                    `文件: ${lastUpload.file_name}`,
                    `导入电池: ${lastUpload.battery_ids.join(', ')}`,
                    `导入周期点: ${lastUpload.imported_cycles}`,
                  ]}
                  renderItem={(item) => <List.Item>{item}</List.Item>}
                />
                {latestUploadedBattery ? (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    <Button
                      type="primary"
                      icon={<RocketOutlined />}
                      onClick={() => {
                        selectBattery(latestUploadedBattery)
                        void loadBatteryContext(latestUploadedBattery)
                        void runLifecyclePrediction(latestUploadedBattery, 'hybrid', 30)
                          .then(() => runDiagnosisWorkflow(latestUploadedBattery))
                          .then(() => navigate('/analysis'))
                          .catch((error: Error) => message.error(error.message))
                      }}
                    >
                      立即执行 lifecycle {'->'} mechanism {'->'} analysis
                    </Button>
                    {!lastUpload.include_in_training ? (
                      <Button
                        onClick={() => {
                          void markTrainingCandidate(latestUploadedBattery, true)
                            .then(() => message.success('已将新样本加入训练池'))
                            .catch((error: Error) => message.error(error.message))
                        }}
                      >
                        加入训练池并稍后训练
                      </Button>
                    ) : null}
                  </Space>
                ) : null}
              </Space>
            ) : (
              <EmptyStateBlock compact title="暂无导入记录" description="导入一次数据后，这里会展示来源识别和校验摘要。" className="panel-empty-state" />
            )}
            {lastUpload ? (
              <Space wrap style={{ marginTop: 12 }}>
                <StatusTag tone="info">source={formatSourceLabel(lastUpload.detected_source ?? lastUpload.source)}</StatusTag>
                <StatusTag tone={lastUpload.include_in_training ? 'good' : 'neutral'}>
                  {lastUpload.include_in_training ? '已标记入训练池' : '仅演示入库'}
                </StatusTag>
              </Space>
            ) : null}
            {Object.keys(validationSummary).length > 0 ? (
              <List
                className="validation-list"
                style={{ marginTop: 12 }}
                size="small"
                dataSource={Object.entries(validationSummary).map(([key, value]) => `${key}: ${typeof value === 'object' ? JSON.stringify(value) : String(value)}`)}
                renderItem={(item) => <List.Item>{item}</List.Item>}
              />
            ) : null}
          </PanelCard>

          <PanelCard title="CSV 模板示例">
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              <Alert
                type="info"
                showIcon
                message="建议优先演示 lifecycle-first 路径"
                description="导入后直接触发生命周期预测和机理解释，分析工作台会自动聚合案例材料。"
              />
              <Table className="data-table" columns={sampleColumns} dataSource={sampleData} pagination={false} size="small" scroll={{ x: 720 }} />
            </Space>
          </PanelCard>
        </Col>
      </Row>
    </div>
  )
}

export default DataUpload
