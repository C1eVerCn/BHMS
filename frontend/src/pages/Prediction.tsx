import React, { useEffect, useMemo, useState } from 'react'
import { Alert, Button, Form, InputNumber, List, Select, Space, Typography } from 'antd'
import { PlayCircleOutlined, RightOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

import { BatterySelect, ChartPanel, EmptyStateBlock, PageHero, PanelCard, StructuredDataList } from '../components/ui'
import { useBhmsStore } from '../stores/useBhmsStore'
import {
  buildBatteryProfileItems,
  formatFeatureLabel,
  formatModelLabel,
  formatProjectionMethod,
  formatSeverityLabel,
  formatSourceLabel,
  replaceTechnicalTerms,
} from '../utils/display'
import { buildLifecycleChartOption } from '../utils/lifecycle'

const { Paragraph, Text } = Typography

interface PredictionFormValues {
  battery_id: string
  seq_len: number
  model_name: 'hybrid' | 'bilstm'
}

const Prediction: React.FC = () => {
  const navigate = useNavigate()
  const [form] = Form.useForm<PredictionFormValues>()
  const [resultMessage, setResultMessage] = useState<string | null>(null)

  const batteryOptions = useBhmsStore((state) => state.batteryOptions)
  const batteryById = useBhmsStore((state) => state.batteryById)
  const selectedBatteryId = useBhmsStore((state) => state.selectedBatteryId)
  const selectBattery = useBhmsStore((state) => state.selectBattery)
  const loadBatteryContext = useBhmsStore((state) => state.loadBatteryContext)
  const lifecycleRequestConfig = useBhmsStore((state) => state.lifecycleRequestConfig)
  const setLifecycleRequestConfig = useBhmsStore((state) => state.setLifecycleRequestConfig)
  const runLifecyclePrediction = useBhmsStore((state) => state.runLifecyclePrediction)
  const actionLoading = useBhmsStore((state) => state.actionLoading)
  const batteryCycles = useBhmsStore((state) => (selectedBatteryId ? state.batteryCycles[selectedBatteryId] ?? [] : []))
  const batteryHealth = useBhmsStore((state) => (selectedBatteryId ? state.batteryHealth[selectedBatteryId] : undefined))
  const latestPrediction = useBhmsStore((state) => (selectedBatteryId ? state.latestLifecyclePrediction[selectedBatteryId] : undefined))

  useEffect(() => {
    const batteryId = selectedBatteryId ?? batteryOptions[0]?.battery_id
    if (!batteryId) return

    form.setFieldsValue({
      battery_id: batteryId,
      seq_len: lifecycleRequestConfig.seqLen,
      model_name: lifecycleRequestConfig.modelName,
    })

    if (selectedBatteryId !== batteryId) {
      selectBattery(batteryId)
    }

    void loadBatteryContext(batteryId)
  }, [batteryOptions, form, lifecycleRequestConfig.modelName, lifecycleRequestConfig.seqLen, loadBatteryContext, selectBattery, selectedBatteryId])

  const activeBattery = selectedBatteryId ? batteryById[selectedBatteryId] : undefined
  const sampleProfileItems = useMemo(() => buildBatteryProfileItems(activeBattery), [activeBattery])

  const chartOption = useMemo(() => buildLifecycleChartOption({ batteryCycles, prediction: latestPrediction }), [batteryCycles, latestPrediction])

  const milestoneItems = useMemo(
    () => [
      { label: '数据来源', value: formatSourceLabel(activeBattery?.source) },
      { label: '当前模型', value: latestPrediction ? formatModelLabel(latestPrediction.model_name) : '--' },
      { label: '预测 RUL', value: latestPrediction ? `${latestPrediction.predicted_rul.toFixed(1)} 次` : '--' },
      { label: 'knee 周期', value: latestPrediction?.predicted_knee_cycle ? latestPrediction.predicted_knee_cycle.toFixed(1) : '--' },
      { label: 'EOL 周期', value: latestPrediction?.predicted_eol_cycle ? latestPrediction.predicted_eol_cycle.toFixed(1) : '--' },
      { label: '容量归零周期', value: latestPrediction?.projection?.predicted_zero_cycle ? latestPrediction.projection.predicted_zero_cycle.toFixed(1) : '--' },
      { label: '投影方法', value: formatProjectionMethod(latestPrediction?.projection?.projection_method) },
      { label: '推理版本', value: latestPrediction?.checkpoint_id ?? '--' },
    ],
    [activeBattery?.source, latestPrediction],
  )

  const riskSummaryItems = useMemo(
    () => [
      { label: '衰退模式', value: String(latestPrediction?.future_risks?.future_capacity_fade_pattern ?? '--') },
      { label: '温度风险', value: String(latestPrediction?.future_risks?.temperature_risk ?? '--') },
      { label: '内阻风险', value: String(latestPrediction?.future_risks?.resistance_risk ?? '--') },
      { label: '电压风险', value: String(latestPrediction?.future_risks?.voltage_risk ?? '--') },
    ],
    [latestPrediction],
  )

  const confidenceFactors = ((latestPrediction?.explanation?.confidence_summary?.factors as string[] | undefined) ?? []).filter(Boolean)
  const projectionNotes = useMemo(
    () => [
      `预测曲线从当前状态连续延伸，直到容量降到 0。`,
      `EOL 周期 ${latestPrediction?.predicted_eol_cycle ? latestPrediction.predicted_eol_cycle.toFixed(1) : '--'} 仍然作为寿命判定的关键节点。`,
      `当前使用 ${formatProjectionMethod(latestPrediction?.projection?.projection_method)} 进行全周期投影。`,
    ],
    [latestPrediction],
  )

  return (
    <div className="page-shell page-shell--stacked">
      <PageHero
        title="生命周期预测"
        description="展示完整历史轨迹、从当前状态到容量归零的全周期预测，以及关键里程碑与模型解释证据。"
        pills={[
          { label: '当前电池', value: selectedBatteryId ?? '未选择', tone: 'teal' },
          { label: '已载入周期', value: batteryCycles.length, tone: 'slate' },
          { label: '健康分', value: (batteryHealth?.health_score ?? 0).toFixed(1), tone: 'amber' },
        ]}
      />

      <PanelCard title="预测设置">
        <Form
          form={form}
          layout="vertical"
          initialValues={{
            battery_id: selectedBatteryId ?? batteryOptions[0]?.battery_id,
            seq_len: lifecycleRequestConfig.seqLen,
            model_name: lifecycleRequestConfig.modelName,
          }}
          onValuesChange={(changedValues) => {
            const patch: Partial<{ modelName: 'hybrid' | 'bilstm'; seqLen: number }> = {}
            if (changedValues.model_name) {
              patch.modelName = changedValues.model_name
            }
            if (typeof changedValues.seq_len === 'number') {
              patch.seqLen = changedValues.seq_len
            }
            if (Object.keys(patch).length > 0) {
              setLifecycleRequestConfig(patch)
            }
          }}
          onFinish={(values) => {
            setResultMessage(null)
            selectBattery(values.battery_id)
            setLifecycleRequestConfig({ modelName: values.model_name, seqLen: values.seq_len })
            void loadBatteryContext(values.battery_id)
            void runLifecyclePrediction(values.battery_id, values.model_name, values.seq_len)
              .then(() => setResultMessage('全周期预测已更新。'))
              .catch((error: Error) => setResultMessage(error.message))
          }}
        >
          <div className="stacked-form-grid stacked-form-grid--three">
            <Form.Item label="电池 ID" name="battery_id" rules={[{ required: true, message: '请选择电池' }]}>
              <BatterySelect
                options={batteryOptions}
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
                  { label: 'Bi-LSTM', value: 'bilstm' },
                ]}
              />
            </Form.Item>
            <Form.Item label="历史窗口长度" name="seq_len">
              <InputNumber min={10} max={200} style={{ width: '100%' }} />
            </Form.Item>
          </div>
          <Space wrap>
            <Button type="primary" htmlType="submit" icon={<PlayCircleOutlined />} loading={actionLoading}>
              发起全周期预测
            </Button>
            <Button icon={<RightOutlined />} onClick={() => navigate('/analysis')}>
              去完整分析页
            </Button>
          </Space>
        </Form>
      </PanelCard>

      <PanelCard title="样本概况">
        {sampleProfileItems.length ? (
          <StructuredDataList items={sampleProfileItems} />
        ) : (
          <EmptyStateBlock compact title="暂无样本概况" description="选择电池后，这里会显示来源、协议、循环数和当前容量等基础信息。" className="panel-empty-state" />
        )}
      </PanelCard>

      <ChartPanel
        title="全周期生命轨迹"
        option={chartOption}
        hasData={batteryCycles.length > 0}
        height={440}
        emptyTitle="暂无可视化数据"
        emptyDescription="选择已有循环轨迹的电池后，这里会显示完整历史、全周期预测曲线以及关键里程碑。"
      />

      <PanelCard title="模型解释证据">
        {latestPrediction ? (
          <Space direction="vertical" size={20} style={{ width: '100%' }}>
            {latestPrediction.fallback_used ? (
              <Alert
                type="warning"
                showIcon
                message="当前结果使用了兼容推理路径"
                description="本次没有直接命中目标训练模型，因此系统回退到了可用的兼容推理路径。"
              />
            ) : null}
            {resultMessage ? <Alert type="info" showIcon message={resultMessage} /> : null}

            <div className="panel-section-block">
              <Text className="panel-section-label">关键里程碑</Text>
              <StructuredDataList items={milestoneItems} />
              <div style={{ marginTop: 12 }}>
                <Text className="panel-section-label">当前置信度</Text>
                <div className="prediction-confidence-bar">
                  <span className="prediction-confidence-bar__fill" style={{ width: `${(latestPrediction.confidence * 100).toFixed(1)}%` }} />
                </div>
                <Paragraph className="panel-subtle-copy">当前置信度 {(latestPrediction.confidence * 100).toFixed(1)}%。</Paragraph>
              </div>
            </div>

            <div className="panel-section-block">
              <Text className="panel-section-label">风险窗口</Text>
              <StructuredDataList items={riskSummaryItems} compact />
              {latestPrediction.risk_windows.length ? (
                <List
                  className="list-compact"
                  dataSource={latestPrediction.risk_windows}
                  renderItem={(item) => (
                    <List.Item>
                      <List.Item.Meta
                        title={`${item.label} · ${item.start_cycle} -> ${item.end_cycle}`}
                        description={`${replaceTechnicalTerms(item.description)}（${formatSeverityLabel(item.severity)}）`}
                      />
                    </List.Item>
                  )}
                />
              ) : (
                <EmptyStateBlock compact title="暂无额外风险窗口" description="当前预测没有识别到需要单独强调的加速衰退区段。" className="panel-empty-state" />
              )}
            </div>

            <div className="panel-section-block">
              <Text className="panel-section-label">关键特征贡献</Text>
              {latestPrediction.explanation?.feature_contributions?.length ? (
                <List
                  className="list-compact"
                  dataSource={latestPrediction.explanation.feature_contributions}
                  renderItem={(item) => (
                    <List.Item>
                      <List.Item.Meta
                        title={`${formatFeatureLabel(item.feature)} · 权重 ${item.impact.toFixed(3)}`}
                        description={replaceTechnicalTerms(item.description)}
                      />
                    </List.Item>
                  )}
                />
              ) : (
                <EmptyStateBlock compact title="暂无特征贡献" description="模型返回结构化解释后，这里会列出最关键的影响特征。" className="panel-empty-state" />
              )}
            </div>

            <div className="panel-section-block">
              <Text className="panel-section-label">关键时间窗口</Text>
              {latestPrediction.explanation?.window_contributions?.length ? (
                <List
                  className="list-compact"
                  dataSource={latestPrediction.explanation.window_contributions}
                  renderItem={(item) => (
                    <List.Item>
                      <List.Item.Meta
                        title={`${item.window_label} · 权重 ${item.impact.toFixed(3)}`}
                        description={replaceTechnicalTerms(item.description)}
                      />
                    </List.Item>
                  )}
                />
              ) : (
                <EmptyStateBlock compact title="暂无关键时间窗口" description="模型返回结构化解释后，这里会显示最关键的时间区段。" className="panel-empty-state" />
              )}
            </div>

            <div className="panel-section-block">
              <Text className="panel-section-label">置信度因素</Text>
              {confidenceFactors.length ? (
                <List className="list-compact" size="small" dataSource={confidenceFactors} renderItem={(item) => <List.Item>{item}</List.Item>} />
              ) : (
                <Paragraph className="panel-subtle-copy">当前暂无额外置信度说明。</Paragraph>
              )}
            </div>

            <div className="panel-section-block">
              <Text className="panel-section-label">投影方法</Text>
              <List className="list-compact" size="small" dataSource={projectionNotes} renderItem={(item) => <List.Item>{item}</List.Item>} />
            </div>
          </Space>
        ) : (
          <EmptyStateBlock compact title="请先执行预测" description="生成一次生命周期预测后，这里会显示关键里程碑、风险窗口和模型解释证据。" className="panel-empty-state" />
        )}
      </PanelCard>
    </div>
  )
}

export default Prediction
