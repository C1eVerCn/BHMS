import React, { useMemo, useState } from 'react'
import { Alert, Button, Card, Checkbox, Col, Input, List, message, Progress, Row, Select, Space, Table, Tag, Typography, Upload } from 'antd'
import type { UploadFile } from 'antd/es/upload/interface'
import { CloudDownloadOutlined, InboxOutlined, UploadOutlined } from '@ant-design/icons'

import { useBhmsStore } from '../stores/useBhmsStore'

const { Title, Text } = Typography
const { Dragger } = Upload

const sourceOptions = [
  { label: '自动识别', value: 'auto' },
  { label: 'NASA', value: 'nasa' },
  { label: 'CALCE', value: 'calce' },
  { label: 'Kaggle', value: 'kaggle' },
] as const

const DataUpload: React.FC = () => {
  const [batteryId, setBatteryId] = useState('')
  const [selectedSource, setSelectedSource] = useState<string>('auto')
  const [includeInTraining, setIncludeInTraining] = useState(false)
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [localProgress, setLocalProgress] = useState(0)
  const actionLoading = useBhmsStore((state) => state.actionLoading)
  const lastUpload = useBhmsStore((state) => state.lastUpload)
  const importSource = useBhmsStore((state) => state.importSource)
  const uploadFile = useBhmsStore((state) => state.uploadFile)

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
    accept: '.csv,.mat',
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

  const handleImportSource = async (source: 'nasa' | 'calce' | 'kaggle') => {
    setLocalProgress(10)
    try {
      await importSource(source, includeInTraining)
      setLocalProgress(100)
      message.success(`${source.toUpperCase()} 数据已导入`)
    } catch (error) {
      setLocalProgress(0)
      message.error(error instanceof Error ? error.message : `${source} 导入失败`)
    }
  }

  const validationSummary = lastUpload?.validation_summary ?? {}

  return (
    <div>
      <Title level={2} className="page-title">
        数据导入
      </Title>
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          <Card title="上传新数据并导入演示系统" style={{ marginBottom: 16 }}>
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <Select value={selectedSource} onChange={setSelectedSource} options={sourceOptions as unknown as { label: string; value: string }[]} />
              <Input value={batteryId} onChange={(event) => setBatteryId(event.target.value)} placeholder="可选：若文件不包含 battery_id/source_battery_id，可在此填写" />
              <Checkbox checked={includeInTraining} onChange={(event) => setIncludeInTraining(event.target.checked)}>
                将该文件标记为后续训练数据池候选
              </Checkbox>
              <Dragger {...uploadProps} disabled={actionLoading}>
                <p className="ant-upload-drag-icon">
                  <InboxOutlined />
                </p>
                <p className="ant-upload-text">点击或拖拽 CSV / MAT 文件到此区域上传</p>
                <p className="ant-upload-hint">NASA 支持 MAT；CALCE/Kaggle 首轮支持统一周期级 CSV 子集</p>
              </Dragger>
              <Button type="primary" icon={<UploadOutlined />} loading={actionLoading} onClick={() => void handleUpload()}>
                上传并入库
              </Button>
              {localProgress > 0 && <Progress percent={localProgress} />}
            </Space>
          </Card>

          <Card title="导入仓库内置数据源样例">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Alert
                type="info"
                showIcon
                message="按来源独立导入"
                description="可分别导入 NASA、CALCE、Kaggle 样例数据；是否进入训练数据池由上方复选框决定。"
              />
              <Space wrap>
                <Button icon={<CloudDownloadOutlined />} loading={actionLoading} onClick={() => void handleImportSource('nasa')}>
                  导入 NASA
                </Button>
                <Button icon={<CloudDownloadOutlined />} loading={actionLoading} onClick={() => void handleImportSource('calce')}>
                  导入 CALCE
                </Button>
                <Button icon={<CloudDownloadOutlined />} loading={actionLoading} onClick={() => void handleImportSource('kaggle')}>
                  导入 Kaggle
                </Button>
              </Space>
            </Space>
          </Card>
        </Col>

        <Col xs={24} lg={8}>
          <Card title="最新导入结果" style={{ marginBottom: 16 }}>
            {lastUpload ? (
              <List
                dataSource={[
                  `来源: ${lastUpload.detected_source ?? lastUpload.source}`,
                  `数据集: ${lastUpload.dataset_name ?? '-'}`,
                  `文件: ${lastUpload.file_name}`,
                  `导入电池: ${lastUpload.battery_ids.join(', ')}`,
                  `导入周期点: ${lastUpload.imported_cycles}`,
                ]}
                renderItem={(item) => <List.Item>{item}</List.Item>}
              />
            ) : (
              <Text type="secondary">暂无导入记录</Text>
            )}
            {lastUpload && (
              <Space wrap style={{ marginTop: 12 }}>
                <Tag color="blue">source={lastUpload.detected_source ?? lastUpload.source}</Tag>
                <Tag color={lastUpload.include_in_training ? 'green' : 'default'}>
                  {lastUpload.include_in_training ? '已标记入训练池' : '仅演示入库'}
                </Tag>
              </Space>
            )}
            {Object.keys(validationSummary).length > 0 && (
              <List
                style={{ marginTop: 12 }}
                size="small"
                dataSource={Object.entries(validationSummary).map(([key, value]) => `${key}: ${typeof value === 'object' ? JSON.stringify(value) : String(value)}`)}
                renderItem={(item) => <List.Item>{item}</List.Item>}
              />
            )}
          </Card>

          <Card title="CSV 模板示例">
            <Table columns={sampleColumns} dataSource={sampleData} pagination={false} size="small" />
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default DataUpload
