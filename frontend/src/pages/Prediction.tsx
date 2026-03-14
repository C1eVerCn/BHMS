import React, { useEffect, useMemo, useState } from 'react'
import { Alert, Button, Card, Col, Descriptions, Empty, Form, InputNumber, Progress, Row, Select, Space, Typography } from 'antd'
import { PlayCircleOutlined } from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'

import { useBhmsStore } from '../stores/useBhmsStore'

const { Title, Text } = Typography

const Prediction: React.FC = () => {
  const batteries = useBhmsStore((state) => state.batteries)
  const selectedBatteryId = useBhmsStore((state) => state.selectedBatteryId)
  const selectBattery = useBhmsStore((state) => state.selectBattery)
  const loadBatteryContext = useBhmsStore((state) => state.loadBatteryContext)
  const runPrediction = useBhmsStore((state) => state.runPrediction)
  const actionLoading = useBhmsStore((state) => state.actionLoading)
  const batteryCycles = useBhmsStore((state) => (selectedBatteryId ? state.batteryCycles[selectedBatteryId] ?? [] : []))
  const batteryHistory = useBhmsStore((state) => (selectedBatteryId ? state.batteryHistory[selectedBatteryId] : undefined))
  const batteryHealth = useBhmsStore((state) => (selectedBatteryId ? state.batteryHealth[selectedBatteryId] : undefined))
  const latestPrediction = useBhmsStore((state) => (selectedBatteryId ? state.latestPrediction[selectedBatteryId] : undefined))
  const [resultMessage, setResultMessage] = useState<string | null>(null)

  useEffect(() => {
    const batteryId = selectedBatteryId ?? batteries[0]?.battery_id
    if (batteryId) {
      selectBattery(batteryId)
      void loadBatteryContext(batteryId)
    }
  }, [batteries, loadBatteryContext, selectBattery, selectedBatteryId])

  const actualSeries = useMemo(() => batteryCycles.map((item) => [item.cycle_number, item.capacity]), [batteryCycles])
  const projectedSeries = useMemo(() => {
    if (!latestPrediction || !batteryCycles.length) return [] as Array<[number, number]>
    const last = batteryCycles[batteryCycles.length - 1]
    const initialCapacity = batteryCycles[0]?.capacity ?? last.capacity
    const eolCapacity = initialCapacity * 0.8
    const targetCycle = last.cycle_number + latestPrediction.predicted_rul
    return [
      [last.cycle_number, last.capacity],
      [targetCycle, Math.max(eolCapacity, last.capacity - (last.capacity - eolCapacity))],
    ]
  }, [batteryCycles, latestPrediction])

  const chartOption = {
    tooltip: { trigger: 'axis' },
    legend: { top: 0 },
    xAxis: { type: 'value', name: '循环次数' },
    yAxis: { type: 'value', name: '容量(Ah)' },
    series: [
      {
        name: '真实容量轨迹',
        type: 'line',
        smooth: true,
        data: actualSeries,
        lineStyle: { color: '#1677ff', width: 3 },
      },
      {
        name: '预测到 EOL',
        type: 'line',
        data: projectedSeries,
        lineStyle: { color: '#52c41a', type: 'dashed', width: 3 },
      },
    ],
  }

  const gaugeOption = {
    series: [
      {
        type: 'gauge',
        startAngle: 180,
        endAngle: 0,
        min: 0,
        max: 100,
        progress: { show: true, width: 16 },
        axisLine: { lineStyle: { width: 16 } },
        pointer: { show: false },
        splitLine: { distance: -18, length: 8 },
        axisTick: { show: false },
        axisLabel: { distance: 18 },
        detail: { valueAnimation: true, formatter: '{value}%', fontSize: 30, offsetCenter: [0, '10%'] },
        title: { offsetCenter: [0, '-25%'] },
        data: [{ value: Number((batteryHealth?.health_score ?? 0).toFixed(1)), name: '健康度' }],
      },
    ],
  }

  return (
    <div>
      <Title level={2} className="page-title">
        RUL 预测
      </Title>
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={8}>
          <Card title="预测配置" style={{ marginBottom: 16 }}>
            <Form
              layout="vertical"
              initialValues={{ battery_id: selectedBatteryId ?? batteries[0]?.battery_id, seq_len: 30, model_name: 'hybrid' }}
              onFinish={(values: { battery_id: string; seq_len: number; model_name: string }) => {
                selectBattery(values.battery_id)
                void loadBatteryContext(values.battery_id)
                void runPrediction(values.battery_id, values.model_name, values.seq_len)
                  .then(() => setResultMessage('预测成功，历史记录已落库。'))
                  .catch((error: Error) => setResultMessage(error.message))
              }}
            >
              <Form.Item label="电池 ID" name="battery_id" rules={[{ required: true, message: '请选择电池' }]}>
                <Select
                  options={batteries.map((item) => ({ label: item.battery_id, value: item.battery_id }))}
                  onChange={(value) => {
                    selectBattery(value)
                    void loadBatteryContext(value)
                  }}
                />
              </Form.Item>
              <Form.Item label="模型" name="model_name">
                <Select
                  options={[
                    { label: 'xLSTM-Transformer', value: 'hybrid' },
                    { label: 'Bi-LSTM 基线', value: 'bilstm' },
                  ]}
                />
              </Form.Item>
              <Form.Item label="历史窗口长度" name="seq_len">
                <InputNumber min={10} max={200} style={{ width: '100%' }} />
              </Form.Item>
              <Button type="primary" htmlType="submit" icon={<PlayCircleOutlined />} loading={actionLoading} block>
                发起预测
              </Button>
            </Form>
            {resultMessage && (
              <Alert
                style={{ marginTop: 16 }}
                type={latestPrediction ? 'success' : 'info'}
                showIcon
                message={resultMessage}
              />
            )}
          </Card>

          <Card title="最新预测结果">
            {latestPrediction ? (
              <Descriptions column={1} size="small">
                <Descriptions.Item label="预测 RUL">{latestPrediction.predicted_rul.toFixed(1)} cycles</Descriptions.Item>
                <Descriptions.Item label="置信度">
                  <Progress percent={Number((latestPrediction.confidence * 100).toFixed(1))} size="small" />
                </Descriptions.Item>
                <Descriptions.Item label="模型版本">{latestPrediction.model_version}</Descriptions.Item>
                <Descriptions.Item label="推理模式">{latestPrediction.fallback_used ? '启发式 fallback' : '训练模型'}</Descriptions.Item>
              </Descriptions>
            ) : (
              <Empty description="请选择电池并执行预测" />
            )}
          </Card>
        </Col>

        <Col xs={24} lg={16}>
          <Card title="容量轨迹与 EOL 预测" className="chart-container" style={{ marginBottom: 16 }}>
            {batteryCycles.length ? <ReactECharts option={chartOption} style={{ height: 360 }} /> : <Empty description="暂无可视化数据" />}
          </Card>

          <Row gutter={[16, 16]}>
            <Col xs={24} md={10}>
              <Card title="电池健康度">
                <ReactECharts option={gaugeOption} style={{ height: 250 }} />
              </Card>
            </Col>
            <Col xs={24} md={14}>
              <Card title="预测历史">
                {batteryHistory?.predictions.length ? (
                  <Space direction="vertical" style={{ width: '100%' }}>
                    {batteryHistory.predictions.map((record) => (
                      <Alert
                        key={record.id}
                        type="info"
                        showIcon
                        message={`${record.model_name} · RUL ${record.predicted_rul.toFixed(1)} cycles`}
                        description={new Date(record.created_at).toLocaleString()}
                      />
                    ))}
                  </Space>
                ) : (
                  <Text type="secondary">暂无预测历史</Text>
                )}
              </Card>
            </Col>
          </Row>
        </Col>
      </Row>
    </div>
  )
}

export default Prediction
