import React, { useEffect, useMemo, useState } from 'react'
import {
  Alert,
  Button,
  Col,
  Descriptions,
  Form,
  InputNumber,
  Progress,
  Row,
  Select,
  Space,
} from 'antd'
import { PlayCircleOutlined, RightOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

import { ChartPanel, EmptyStateBlock, InsightCard, PageHero, PanelCard } from '../components/ui'
import { useBhmsStore } from '../stores/useBhmsStore'

const Prediction: React.FC = () => {
  const navigate = useNavigate()
  const [form] = Form.useForm()
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
      form.setFieldsValue({ battery_id: batteryId })
    }
  }, [batteries, form, loadBatteryContext, selectBattery, selectedBatteryId])

  const actualSeries = useMemo(
    () => latestPrediction?.projection?.actual_points.map((item) => [item.cycle, item.capacity]) ?? batteryCycles.map((item) => [item.cycle_number, item.capacity]),
    [batteryCycles, latestPrediction],
  )
  const projectedSeries = useMemo(
    () => latestPrediction?.projection?.forecast_points.map((item) => [item.cycle, item.capacity]) ?? ([] as Array<[number, number]>),
    [latestPrediction],
  )
  const confidenceUpper = useMemo(
    () => latestPrediction?.projection?.confidence_band.map((item) => [item.cycle, item.upper]) ?? ([] as Array<[number, number]>),
    [latestPrediction],
  )
  const confidenceLower = useMemo(
    () => latestPrediction?.projection?.confidence_band.map((item) => [item.cycle, item.lower]) ?? ([] as Array<[number, number]>),
    [latestPrediction],
  )
  const eolCapacity = latestPrediction?.projection?.eol_capacity ?? 0
  const lastProjectedPoint = projectedSeries.length ? projectedSeries[projectedSeries.length - 1] : undefined
  const lastActualPoint = actualSeries.length ? actualSeries[actualSeries.length - 1] : undefined

  const chartOption = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(16, 35, 63, 0.92)',
      borderWidth: 0,
      textStyle: { color: '#f8fafc' },
    },
    legend: { top: 4, textStyle: { color: '#5f6c7b' } },
    grid: { left: 28, right: 20, top: 56, bottom: 28, containLabel: true },
    xAxis: {
      type: 'value',
      name: '循环次数',
      nameTextStyle: { color: '#6b7280' },
      splitLine: { show: false },
      axisLine: { lineStyle: { color: 'rgba(16, 35, 63, 0.14)' } },
      axisLabel: { color: '#6b7280' },
    },
    yAxis: {
      type: 'value',
      name: '容量(Ah)',
      nameTextStyle: { color: '#6b7280' },
      splitLine: { lineStyle: { color: 'rgba(16, 35, 63, 0.08)' } },
      axisLabel: { color: '#6b7280' },
    },
    series: [
      {
        name: '真实容量轨迹',
        type: 'line',
        smooth: true,
        symbol: 'circle',
        symbolSize: 8,
        data: actualSeries,
        lineStyle: { color: '#0071e3', width: 3 },
        itemStyle: { color: '#0071e3', borderColor: '#ffffff', borderWidth: 2 },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(0, 113, 227, 0.18)' },
              { offset: 1, color: 'rgba(0, 113, 227, 0.02)' },
            ],
          },
        },
      },
      {
        name: '预测到 EOL',
        type: 'line',
        data: projectedSeries,
        showSymbol: false,
        lineStyle: { color: '#34c759', type: 'dashed', width: 3 },
        itemStyle: { color: '#34c759' },
      },
      {
        name: 'EOL 阈值',
        type: 'line',
        symbol: 'none',
        data:
          actualSeries.length || projectedSeries.length
            ? [
                [Number((actualSeries[0] ?? projectedSeries[0])[0]), eolCapacity],
                [Number((lastProjectedPoint ?? lastActualPoint)?.[0] ?? 0), eolCapacity],
              ]
            : [],
        lineStyle: { color: '#ff9f0a', type: 'dotted', width: 2 },
      },
      {
        name: '置信带上界',
        type: 'line',
        symbol: 'none',
        data: confidenceUpper,
        lineStyle: { opacity: 0 },
        stack: 'confidence',
      },
      {
        name: '置信带下界',
        type: 'line',
        symbol: 'none',
        data: confidenceLower,
        lineStyle: { opacity: 0 },
        areaStyle: { color: 'rgba(52, 199, 89, 0.12)' },
        stack: 'confidence',
      },
    ],
  }

  const gaugeOption = {
    backgroundColor: 'transparent',
    series: [
      {
        type: 'gauge',
        startAngle: 200,
        endAngle: -20,
        min: 0,
        max: 100,
        progress: {
          show: true,
          width: 20,
          itemStyle: { color: '#0071e3' },
        },
        axisLine: { lineStyle: { width: 20, color: [[1, 'rgba(16, 35, 63, 0.1)']] } },
        pointer: { show: false },
        axisTick: { show: false },
        splitLine: { distance: -24, length: 10, lineStyle: { color: 'rgba(16, 35, 63, 0.15)' } },
        axisLabel: { distance: 20, color: '#6b7280' },
        detail: {
          valueAnimation: true,
          formatter: '{value}%',
          fontSize: 28,
          fontWeight: 700,
          color: '#10233f',
          offsetCenter: [0, '12%'],
        },
        title: { offsetCenter: [0, '-20%'], color: '#5f6c7b' },
        data: [{ value: Number((batteryHealth?.health_score ?? 0).toFixed(1)), name: '健康度' }],
      },
    ],
  }

  return (
    <div className="page-shell">
      <PageHero
        kicker="Remaining Useful Life"
        title="把寿命预测，看得更清楚"
        description="从容量轨迹到预测结果，用更轻松的方式理解模型输出。"
        pills={[
          { label: '当前电池', value: selectedBatteryId ?? '未选择', tone: 'teal' },
          { label: '已载入周期', value: batteryCycles.length, tone: 'slate' },
          { label: '健康分', value: (batteryHealth?.health_score ?? 0).toFixed(1), tone: 'amber' },
        ]}
        aside={
          <InsightCard
            compact
            label="最近一次 RUL"
            value={latestPrediction ? latestPrediction.predicted_rul.toFixed(1) : '--'}
            description={
              latestPrediction
                ? `置信度 ${(latestPrediction.confidence * 100).toFixed(1)}%，模型 ${latestPrediction.model_name}，可进入分析中心查看完整证据链。`
                : '选择电池后即可开始预测。'
            }
          />
        }
      />

      <Row gutter={[18, 18]}>
        <Col xs={24} lg={8}>
          <PanelCard title="预测配置" style={{ marginBottom: 18 }}>
            <Form
              form={form}
              layout="vertical"
              initialValues={{ battery_id: selectedBatteryId ?? batteries[0]?.battery_id, seq_len: 30, model_name: 'hybrid' }}
              onFinish={(values: { battery_id: string; seq_len: number; model_name: string }) => {
                selectBattery(values.battery_id)
                void loadBatteryContext(values.battery_id)
                void runPrediction(values.battery_id, values.model_name, values.seq_len)
                  .then(() => setResultMessage('预测成功，历史记录已同步刷新。'))
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
              <Button icon={<RightOutlined />} block onClick={() => navigate('/analysis')}>
                查看完整分析
              </Button>
            </Form>
            {resultMessage && (
              <Alert
                className="inline-feedback"
                style={{ marginTop: 16 }}
                type={latestPrediction ? 'success' : 'info'}
                showIcon
                message={resultMessage}
              />
            )}
          </PanelCard>

          <PanelCard title="最新预测结果">
            {latestPrediction ? (
              <Descriptions column={1} size="small" className="details-grid">
                <Descriptions.Item label="预测 RUL">{latestPrediction.predicted_rul.toFixed(1)} cycles</Descriptions.Item>
                <Descriptions.Item label="置信度">
                  <Progress percent={Number((latestPrediction.confidence * 100).toFixed(1))} size="small" strokeColor="#0071e3" />
                </Descriptions.Item>
                <Descriptions.Item label="模型版本">{latestPrediction.model_version}</Descriptions.Item>
                <Descriptions.Item label="模型来源">{latestPrediction.model_source}</Descriptions.Item>
                <Descriptions.Item label="Checkpoint">{latestPrediction.checkpoint_id ?? '--'}</Descriptions.Item>
                <Descriptions.Item label="推理模式">{latestPrediction.fallback_used ? '启发式 fallback' : '训练模型'}</Descriptions.Item>
                <Descriptions.Item label="预测 EOL 周期">{latestPrediction.projection.predicted_eol_cycle.toFixed(1)}</Descriptions.Item>
                <Descriptions.Item label="投影方法">{latestPrediction.projection.projection_method ?? 'linear'}</Descriptions.Item>
              </Descriptions>
            ) : (
              <EmptyStateBlock compact title="请选择电池并执行预测" description="运行一次预测后，这里会展示模型版本、置信度和推理模式。" className="panel-empty-state" />
            )}
          </PanelCard>
        </Col>

        <Col xs={24} lg={16}>
          <ChartPanel
            title="容量轨迹与 EOL 预测"
            option={chartOption}
            hasData={batteryCycles.length > 0}
            height={380}
            emptyTitle="暂无可视化数据"
            emptyDescription="选择已有循环轨迹的电池后，这里会自动显示预测曲线。"
            style={{ marginBottom: 18 }}
          />

          <Row gutter={[18, 18]}>
            <Col xs={24} md={10}>
              <ChartPanel
                title="电池健康度"
                option={gaugeOption}
                hasData={Boolean(batteryHealth)}
                height={260}
                emptyTitle="暂无健康度数据"
                emptyDescription="加载电池上下文后，这里会显示实时健康仪表盘。"
              />
            </Col>
            <Col xs={24} md={14}>
              <PanelCard title="预测历史">
                {batteryHistory?.predictions.length ? (
                  <Space direction="vertical" size={12} style={{ width: '100%' }}>
                    {batteryHistory.predictions.map((record) => (
                      <Alert
                        key={record.id}
                        className="history-alert"
                        type="info"
                        showIcon
                        message={`${record.model_name} · RUL ${record.predicted_rul.toFixed(1)} cycles`}
                        description={new Date(record.created_at).toLocaleString()}
                      />
                    ))}
                  </Space>
                ) : (
                  <EmptyStateBlock compact title="暂无预测历史" description="运行预测后，这里会按时间沉淀历史记录。" className="panel-empty-state" />
                )}
              </PanelCard>
            </Col>
          </Row>
        </Col>
      </Row>
    </div>
  )
}

export default Prediction
