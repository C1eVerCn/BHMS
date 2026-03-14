import React, { useMemo, useState } from 'react'
import { Alert, Button, Card, Col, Input, List, message, Progress, Row, Space, Table, Typography, Upload } from 'antd'
import type { UploadFile } from 'antd/es/upload/interface'
import { CloudDownloadOutlined, InboxOutlined, UploadOutlined } from '@ant-design/icons'

import { useBhmsStore } from '../stores/useBhmsStore'

const { Title, Text } = Typography
const { Dragger } = Upload

const DataUpload: React.FC = () => {
  const [batteryId, setBatteryId] = useState('')
  const [fileList, setFileList] = useState<UploadFile[]>([])
  const [localProgress, setLocalProgress] = useState(0)
  const actionLoading = useBhmsStore((state) => state.actionLoading)
  const lastUpload = useBhmsStore((state) => state.lastUpload)
  const importSampleNasa = useBhmsStore((state) => state.importSampleNasa)
  const uploadFile = useBhmsStore((state) => state.uploadFile)

  const sampleColumns = useMemo(
    () => [
      { title: 'cycle_number', dataIndex: 'cycle_number', key: 'cycle_number' },
      { title: 'voltage_mean', dataIndex: 'voltage_mean', key: 'voltage_mean' },
      { title: 'current_mean', dataIndex: 'current_mean', key: 'current_mean' },
      { title: 'temperature_mean', dataIndex: 'temperature_mean', key: 'temperature_mean' },
      { title: 'capacity', dataIndex: 'capacity', key: 'capacity' },
    ],
    [],
  )

  const sampleData = [
    { key: '1', cycle_number: 1, voltage_mean: 3.74, current_mean: -1.92, temperature_mean: 24.9, capacity: 1.85 },
    { key: '2', cycle_number: 2, voltage_mean: 3.71, current_mean: -1.96, temperature_mean: 25.6, capacity: 1.84 },
    { key: '3', cycle_number: 3, voltage_mean: 3.69, current_mean: -2.01, temperature_mean: 25.8, capacity: 1.83 },
  ]

  const uploadProps = {
    name: 'file',
    multiple: false,
    accept: '.csv',
    beforeUpload: (file: UploadFile) => {
      const isCsv = file.name.endsWith('.csv')
      if (!isCsv) {
        message.error('MVP 版本仅支持 CSV 文件上传')
        return Upload.LIST_IGNORE
      }
      return false
    },
    fileList,
    onChange: ({ fileList: nextFileList }: { fileList: UploadFile[] }) => setFileList(nextFileList.slice(-1)),
    onRemove: () => setFileList([]),
  }

  const handleUpload = async () => {
    const currentFile = fileList[0]?.originFileObj
    if (!currentFile) {
      message.warning('请先选择 CSV 文件')
      return
    }
    setLocalProgress(20)
    try {
      await uploadFile(currentFile, batteryId || undefined)
      setLocalProgress(100)
      setFileList([])
      setBatteryId('')
      message.success('文件已成功导入数据库')
    } catch (error) {
      setLocalProgress(0)
      message.error(error instanceof Error ? error.message : '上传失败')
    }
  }

  const handleImportNasa = async () => {
    setLocalProgress(10)
    try {
      await importSampleNasa()
      setLocalProgress(100)
      message.success('NASA 原始样例数据已导入')
    } catch (error) {
      setLocalProgress(0)
      message.error(error instanceof Error ? error.message : 'NASA 导入失败')
    }
  }

  return (
    <div>
      <Title level={2} className="page-title">
        数据导入
      </Title>
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={16}>
          <Card title="上传周期级 CSV 数据" style={{ marginBottom: 16 }}>
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <Input value={batteryId} onChange={(event) => setBatteryId(event.target.value)} placeholder="可选：若 CSV 不包含 battery_id，可在此填写" />
              <Dragger {...uploadProps} disabled={actionLoading}>
                <p className="ant-upload-drag-icon">
                  <InboxOutlined />
                </p>
                <p className="ant-upload-text">点击或拖拽 CSV 文件到此区域上传</p>
                <p className="ant-upload-hint">字段最少包含 cycle_number、voltage_mean、current_mean、temperature_mean、capacity</p>
              </Dragger>
              <Button type="primary" icon={<UploadOutlined />} loading={actionLoading} onClick={() => void handleUpload()}>
                上传并入库
              </Button>
              {localProgress > 0 && <Progress percent={localProgress} />}
            </Space>
          </Card>

          <Card title="直接导入仓库内 NASA 样例">
            <Space direction="vertical" style={{ width: '100%' }}>
              <Alert
                type="info"
                showIcon
                message="无需额外下载"
                description="仓库已包含 NASA PCoE 原始 MAT 文件，点击下方按钮即可完成预处理和数据库导入。"
              />
              <Button icon={<CloudDownloadOutlined />} loading={actionLoading} onClick={() => void handleImportNasa()}>
                导入 NASA 样例数据
              </Button>
            </Space>
          </Card>
        </Col>

        <Col xs={24} lg={8}>
          <Card title="最新导入结果" style={{ marginBottom: 16 }}>
            {lastUpload ? (
              <List
                dataSource={[
                  `文件: ${lastUpload.file_name}`,
                  `导入电池: ${lastUpload.battery_ids.join(', ')}`,
                  `导入周期点: ${lastUpload.imported_cycles}`,
                ]}
                renderItem={(item) => <List.Item>{item}</List.Item>}
              />
            ) : (
              <Text type="secondary">暂无导入记录</Text>
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
