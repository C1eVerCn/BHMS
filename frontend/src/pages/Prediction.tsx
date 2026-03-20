import React, { useEffect, useMemo, useState } from 'react'
import { Alert, Button, Col, Descriptions, Form, InputNumber, Progress, Row, Select, Space } from 'antd'
import { PlayCircleOutlined, RightOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

import { ChartPanel, EmptyStateBlock, InsightCard, PageHero, PanelCard } from '../components/ui'
import { useBhmsStore } from '../stores/useBhmsStore'
import { formatModelLabel, formatSeverityLabel } from '../utils/display'

const Prediction: React.FC = () => {
  const navigate = useNavigate()
  const [form] = Form.useForm()
  const batteries = useBhmsStore((state) => state.batteries)
  const selectedBatteryId = useBhmsStore((state) => state.selectedBatteryId)
  const selectBattery = useBhmsStore((state) => state.selectBattery)
  const loadBatteryContext = useBhmsStore((state) => state.loadBatteryContext)
  const runLifecyclePrediction = useBhmsStore((state) => state.runLifecyclePrediction)
  const actionLoading = useBhmsStore((state) => state.actionLoading)
  const batteryCycles = useBhmsStore((state) => (selectedBatteryId ? state.batteryCycles[selectedBatteryId] ?? [] : []))
  const batteryHealth = useBhmsStore((state) => (selectedBatteryId ? state.batteryHealth[selectedBatteryId] : undefined))
  const latestPrediction = useBhmsStore((state) => (selectedBatteryId ? state.latestLifecyclePrediction[selectedBatteryId] : undefined))
  const [resultMessage, setResultMessage] = useState<string | null>(null)

  useEffect(() => {
    const batteryId = selectedBatteryId ?? batteries[0]?.battery_id
    if (batteryId) {
      selectBattery(batteryId)
      void loadBatteryContext(batteryId)
      form.setFieldsValue({ battery_id: batteryId })
    }
  }, [batteries, form, loadBatteryContext, selectBattery, selectedBatteryId])

  const observedSeries = useMemo(
    () => latestPrediction?.projection?.actual_points.map((item) => [item.cycle, item.capacity]) ?? batteryCycles.map((item) => [item.cycle_number, item.capacity]),
    [batteryCycles, latestPrediction],
  )
  const futureSeries = useMemo(
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

  const chartOption = useMemo(
    () => ({
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
        axisLabel: { color: '#6b7280' },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'value',
        name: '容量(Ah)',
        axisLabel: { color: '#6b7280' },
        splitLine: { lineStyle: { color: 'rgba(16, 35, 63, 0.08)' } },
      },
      series: [
        {
          name: '观测轨迹',
          type: 'line',
          smooth: true,
          showSymbol: false,
          data: observedSeries,
          lineStyle: { color: '#0071e3', width: 3 },
        },
        {
          name: '未来 trajectory',
          type: 'line',
          smooth: true,
          showSymbol: false,
          data: futureSeries,
          lineStyle: { color: '#34c759', type: 'dashed', width: 3 },
          markLine: {
            symbol: 'none',
            label: { color: '#334155' },
            data: [
              latestPrediction?.predicted_knee_cycle
                ? { xAxis: latestPrediction.predicted_knee_cycle, name: 'knee' }
                : undefined,
              latestPrediction?.predicted_eol_cycle
                ? { xAxis: latestPrediction.predicted_eol_cycle, name: 'EOL' }
                : undefined,
            ].filter(Boolean),
          },
          markArea: latestPrediction?.risk_windows?.length
            ? {
                itemStyle: { color: 'rgba(255, 159, 10, 0.08)' },
                data: latestPrediction.risk_windows.map((item) => [{ xAxis: item.start_cycle }, { xAxis: item.end_cycle, name: item.label }]),
              }
            : undefined,
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
    }),
    [confidenceLower, confidenceUpper, futureSeries, latestPrediction, observedSeries],
  )

  const metrics = [
    {
      label: '预测 RUL',
      value: latestPrediction ? `${latestPrediction.predicted_rul.toFixed(1)} cycles` : '--',
      description: latestPrediction ? `置信度 ${(latestPrediction.confidence * 100).toFixed(1)}%` : '运行生命周期预测后生成',
    },
    {
      label: 'EOL 周期',
      value: latestPrediction?.predicted_eol_cycle ? latestPrediction.predicted_eol_cycle.toFixed(1) : '--',
      description: '未来达到 EOL 阈值的预测 cycle',
    },
    {
      label: 'knee 周期',
      value: latestPrediction?.predicted_knee_cycle ? latestPrediction.predicted_knee_cycle.toFixed(1) : '--',
      description: '加速衰退拐点',
    },
    {
      label: '未来衰退模式',
      value: String(latestPrediction?.future_risks?.future_capacity_fade_pattern ?? '--'),
      description: `温度 ${String(latestPrediction?.future_risks?.temperature_risk ?? '--')} / 内阻 ${String(
        latestPrediction?.future_risks?.resistance_risk ?? '--',
      )}`,
    },
  ]

  return (
    <div className="page-shell">
      <PageHero
        kicker="Lifecycle Forecasting"
        title="把全生命周期预测，看得更清楚"
        description="从观测轨迹到未来 trajectory、knee、EOL 和风险窗口，把生命周期推理完整展示出来。"
        pills={[
          { label: '当前电池', value: selectedBatteryId ?? '未选择', tone: 'teal' },
          { label: '已载入周期', value: batteryCycles.length, tone: 'slate' },
          { label: '健康分', value: (batteryHealth?.health_score ?? 0).toFixed(1), tone: 'amber' },
        ]}
        aside={
          <InsightCard
            compact
            label="当前生命周期结果"
            value={latestPrediction ? latestPrediction.predicted_rul.toFixed(1) : '--'}
            description={
              latestPrediction
                ? `${formatModelLabel(latestPrediction.model_name)} · knee ${latestPrediction.predicted_knee_cycle ?? '--'} · EOL ${
                    latestPrediction.predicted_eol_cycle ?? '--'
                  }`
                : '选择电池后即可生成生命周期预测。'
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
                void runLifecyclePrediction(values.battery_id, values.model_name, values.seq_len)
                  .then(() => setResultMessage('生命周期预测成功，历史记录已同步刷新。'))
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
                    { label: 'Lifecycle Hybrid (xLSTM core)', value: 'hybrid' },
                    { label: 'Lifecycle Bi-LSTM', value: 'bilstm' },
                  ]}
                />
              </Form.Item>
              <Form.Item label="历史窗口长度" name="seq_len">
                <InputNumber min={10} max={200} style={{ width: '100%' }} />
              </Form.Item>
              <Button type="primary" htmlType="submit" icon={<PlayCircleOutlined />} loading={actionLoading} block>
                发起生命周期预测
              </Button>
              <Button icon={<RightOutlined />} block onClick={() => navigate('/analysis')}>
                查看分析工作台
              </Button>
            </Form>
            {resultMessage ? <Alert className="inline-feedback" style={{ marginTop: 16 }} type="info" showIcon message={resultMessage} /> : null}
          </PanelCard>

          <PanelCard title="关键生命周期指标">
            <Space direction="vertical" size={12} style={{ width: '100%' }}>
              {metrics.map((item) => (
                <InsightCard key={item.label} compact label={item.label} value={item.value} description={item.description} />
              ))}
            </Space>
          </PanelCard>
        </Col>

        <Col xs={24} lg={16}>
          <ChartPanel
            title="观测轨迹 / future trajectory / risk windows"
            option={chartOption}
            hasData={batteryCycles.length > 0}
            height={380}
            emptyTitle="暂无可视化数据"
            emptyDescription="选择已有循环轨迹的电池后，这里会显示生命周期轨迹与风险窗口。"
            style={{ marginBottom: 18 }}
          />

          <Row gutter={[18, 18]}>
            <Col xs={24} md={12}>
              <PanelCard title="模型证据">
                {latestPrediction ? (
                  <Descriptions column={1} size="small" className="details-grid">
                    <Descriptions.Item label="模型">{formatModelLabel(latestPrediction.model_name)}</Descriptions.Item>
                    <Descriptions.Item label="置信度">
                      <Progress percent={Number((latestPrediction.confidence * 100).toFixed(1))} size="small" strokeColor="#0071e3" />
                    </Descriptions.Item>
                    <Descriptions.Item label="Checkpoint">{latestPrediction.checkpoint_id ?? '--'}</Descriptions.Item>
                    <Descriptions.Item label="推理模式">{latestPrediction.fallback_used ? '启发式 fallback' : '训练模型'}</Descriptions.Item>
                    <Descriptions.Item label="风险窗口">{latestPrediction.risk_windows.length}</Descriptions.Item>
                    <Descriptions.Item label="温度/内阻/电压风险">
                      {String(latestPrediction.future_risks.temperature_risk ?? '--')} / {String(latestPrediction.future_risks.resistance_risk ?? '--')} /{' '}
                      {String(latestPrediction.future_risks.voltage_risk ?? '--')}
                    </Descriptions.Item>
                  </Descriptions>
                ) : (
                  <EmptyStateBlock compact title="请选择电池并执行预测" description="运行一次预测后，这里会显示模型、置信度和风险摘要。" className="panel-empty-state" />
                )}
              </PanelCard>
            </Col>
            <Col xs={24} md={12}>
              <PanelCard title="未来风险窗口">
                {latestPrediction?.risk_windows.length ? (
                  <Space direction="vertical" size={12} style={{ width: '100%' }}>
                    {latestPrediction.risk_windows.map((item) => (
                      <Alert
                        key={`${item.label}-${item.start_cycle}-${item.end_cycle}`}
                        type={item.severity === 'high' ? 'warning' : 'info'}
                        showIcon
                        message={`${item.label} · ${item.start_cycle} -> ${item.end_cycle}`}
                        description={`${item.description}（${formatSeverityLabel(item.severity)}）`}
                      />
                    ))}
                  </Space>
                ) : (
                  <EmptyStateBlock compact title="暂无风险窗口" description="如果未来 trajectory 出现加速衰退，这里会显示风险窗口。" className="panel-empty-state" />
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
