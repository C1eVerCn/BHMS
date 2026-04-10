import React, { useEffect, useMemo, useState } from 'react'
import { Alert, Button, InputNumber, List, Select, Space, Steps, Typography } from 'antd'
import { AlertOutlined, FileTextOutlined, RightOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

import { BatterySelect, EmptyStateBlock, PageHero, PanelCard, SignalList, StatusTag, StructuredDataList, type StatusTone } from '../components/ui'
import { useBhmsStore } from '../stores/useBhmsStore'
import type { DiagnosisRecord, MechanismExplanationResult } from '../types/domain'
import { buildBatteryProfileItems, formatModelLabel, formatSeverityLabel, replaceTechnicalTerms } from '../utils/display'

const { Paragraph, Text } = Typography

const Diagnosis: React.FC = () => {
  const navigate = useNavigate()
  const [currentStep, setCurrentStep] = useState(0)

  const batteryOptions = useBhmsStore((state) => state.batteryOptions)
  const batteryById = useBhmsStore((state) => state.batteryById)
  const selectedBatteryId = useBhmsStore((state) => state.selectedBatteryId)
  const selectBattery = useBhmsStore((state) => state.selectBattery)
  const loadBatteryContext = useBhmsStore((state) => state.loadBatteryContext)
  const lifecycleRequestConfig = useBhmsStore((state) => state.lifecycleRequestConfig)
  const setLifecycleRequestConfig = useBhmsStore((state) => state.setLifecycleRequestConfig)
  const runDiagnosisWorkflow = useBhmsStore((state) => state.runDiagnosisWorkflow)
  const actionLoading = useBhmsStore((state) => state.actionLoading)
  const latestAnomaly = useBhmsStore((state) => (selectedBatteryId ? state.latestAnomaly[selectedBatteryId] : undefined))
  const latestMechanismExplanation = useBhmsStore((state) => (selectedBatteryId ? state.latestMechanismExplanation[selectedBatteryId] : undefined))
  const latestLifecyclePrediction = useBhmsStore((state) => (selectedBatteryId ? state.latestLifecyclePrediction[selectedBatteryId] : undefined))
  const batteryHistory = useBhmsStore((state) => (selectedBatteryId ? state.batteryHistory[selectedBatteryId] : undefined))

  useEffect(() => {
    const batteryId = selectedBatteryId ?? batteryOptions[0]?.battery_id
    if (!batteryId) return
    if (selectedBatteryId !== batteryId) {
      selectBattery(batteryId)
    }
    void loadBatteryContext(batteryId)
  }, [batteryOptions, loadBatteryContext, selectBattery, selectedBatteryId])

  const activeBattery = selectedBatteryId ? batteryById[selectedBatteryId] : undefined
  const diagnosis = latestMechanismExplanation ?? hydrateMechanismFromHistory(batteryHistory?.diagnoses?.[0])
  const anomalyEvents = latestAnomaly?.events ?? batteryHistory?.anomalies ?? []
  const sampleProfileItems = useMemo(() => buildBatteryProfileItems(activeBattery), [activeBattery])

  useEffect(() => {
    if (diagnosis) {
      setCurrentStep(3)
      return
    }
    if (latestLifecyclePrediction) {
      setCurrentStep(1)
      return
    }
    setCurrentStep(0)
  }, [diagnosis, latestLifecyclePrediction, selectedBatteryId])

  const riskEvidenceItems = useMemo(
    () => [
      { label: '衰退模式', value: String(diagnosis?.lifecycle_evidence?.future_capacity_fade_pattern ?? '--') },
      { label: '温度风险', value: String(diagnosis?.lifecycle_evidence?.temperature_risk ?? '--') },
      { label: '内阻风险', value: String(diagnosis?.lifecycle_evidence?.resistance_risk ?? '--') },
      { label: '电压风险', value: String(diagnosis?.lifecycle_evidence?.voltage_risk ?? '--') },
      { label: 'knee 周期', value: diagnosis?.lifecycle_evidence?.predicted_knee_cycle ? String(diagnosis.lifecycle_evidence.predicted_knee_cycle) : '--' },
      { label: 'EOL 周期', value: diagnosis?.lifecycle_evidence?.predicted_eol_cycle ? String(diagnosis.lifecycle_evidence.predicted_eol_cycle) : '--' },
    ],
    [diagnosis],
  )

  const modelEvidenceItems = useMemo(
    () => [
      { label: '当前模型', value: diagnosis?.model_evidence?.model_name ? formatModelLabel(String(diagnosis.model_evidence.model_name)) : '--' },
      { label: '关键特征', value: toStringList(diagnosis?.model_evidence?.top_features) },
      { label: '关键时间窗口', value: toStringList(diagnosis?.model_evidence?.critical_windows) },
      { label: '排序依据', value: diagnosis?.graph_trace?.ranking_basis ?? [] },
      { label: '决策依据', value: diagnosis?.decision_basis ?? [] },
    ],
    [diagnosis],
  )

  const diagnosisSeverity = diagnosis?.severity
  const diagnosisAlertType =
    diagnosisSeverity === 'critical' || diagnosisSeverity === 'high'
      ? 'error'
      : diagnosisSeverity === 'info' || diagnosis?.fault_type === '未发现明显故障'
        ? 'success'
        : 'warning'

  return (
    <div className="page-shell page-shell--stacked">
      <PageHero
        title="机理解释"
        description="围绕当前样本展示异常事件、GraphRAG 排序依据和模型证据，解释系统为什么给出当前故障判断。"
        pills={[
          { label: '当前电池', value: selectedBatteryId ?? '未选择', tone: 'teal' },
          { label: '异常事件', value: anomalyEvents.length, tone: anomalyEvents.length ? 'amber' : 'slate' },
          { label: '诊断结果', value: diagnosis ? '已生成' : '待生成', tone: diagnosis ? 'teal' : 'slate' },
        ]}
      />

      <PanelCard title="诊断入口">
        <Space direction="vertical" size={18} style={{ width: '100%' }}>
          <div className="stacked-form-grid stacked-form-grid--two">
            <div>
              <Text className="panel-section-label">电池 ID</Text>
              <BatterySelect
                value={selectedBatteryId ?? undefined}
                options={batteryOptions}
                onChange={(value) => {
                  selectBattery(value)
                  setCurrentStep(0)
                  void loadBatteryContext(value)
                }}
              />
            </div>
            <div>
              <Text className="panel-section-label">当前说明</Text>
              <Paragraph className="panel-subtle-copy">系统会结合寿命轨迹、异常信号和图谱证据生成机理解释结果。</Paragraph>
            </div>
          </div>
          <div className="stacked-form-grid stacked-form-grid--two">
            <div>
              <Text className="panel-section-label">生命周期模型</Text>
              <Select<'hybrid' | 'bilstm'>
                value={lifecycleRequestConfig.modelName}
                options={[
                  { label: 'xLSTM-Transformer', value: 'hybrid' },
                  { label: 'Bi-LSTM', value: 'bilstm' },
                ]}
                onChange={(value) => setLifecycleRequestConfig({ modelName: value })}
              />
            </div>
            <div>
              <Text className="panel-section-label">历史窗口长度</Text>
              <InputNumber
                min={10}
                max={200}
                value={lifecycleRequestConfig.seqLen}
                style={{ width: '100%' }}
                onChange={(value) => {
                  if (typeof value === 'number') {
                    setLifecycleRequestConfig({ seqLen: value })
                  }
                }}
              />
            </div>
          </div>
          <Space wrap>
            <Button
              type="primary"
              icon={<AlertOutlined />}
              loading={actionLoading}
              onClick={() => {
                if (!selectedBatteryId) return
                setCurrentStep(1)
                void runDiagnosisWorkflow(selectedBatteryId, lifecycleRequestConfig)
                  .then(() => setCurrentStep(3))
                  .catch(() => setCurrentStep(0))
              }}
            >
              生成机理解释
            </Button>
            <Button icon={<RightOutlined />} onClick={() => navigate('/analysis')}>
              去完整分析页
            </Button>
          </Space>
          <Steps
            current={currentStep}
            size="small"
            items={[
              { title: '生命周期预测', description: '先给出 knee、EOL 和未来衰退趋势' },
              { title: '异常检测', description: '把当前样本里的异常信号整理成症状事件' },
              { title: 'GraphRAG 排序', description: '结合知识图谱检索候选故障机理' },
              { title: '证据输出', description: '把故障判断、排序依据和处理建议整理出来' },
            ]}
          />
        </Space>
      </PanelCard>

      <PanelCard title="样本概况">
        {sampleProfileItems.length ? (
          <StructuredDataList items={sampleProfileItems} />
        ) : (
          <EmptyStateBlock compact title="暂无样本概况" description="选择电池后，这里会显示当前样本的来源、规格和循环状态。" className="panel-empty-state" />
        )}
      </PanelCard>

      {diagnosis ? (
        <>
          <PanelCard title="当前诊断结论">
            <Space direction="vertical" size={16} style={{ width: '100%' }}>
              <Alert
                className="diagnosis-banner"
                type={diagnosisAlertType}
                showIcon
                message={
                  <Space size={10} wrap>
                    <Text strong className="diagnosis-banner__title">
                      {diagnosis.fault_type}
                    </Text>
                    <StatusTag tone={diagnosisTone(diagnosis.severity)}>{formatSeverityLabel(diagnosis.severity)}</StatusTag>
                  </Space>
                }
                description={
                  <div>
                    <Text>置信度 {(diagnosis.confidence * 100).toFixed(1)}%</Text>
                    <br />
                    <Text type="secondary">{replaceTechnicalTerms(diagnosis.description)}</Text>
                  </div>
                }
              />
              <StructuredDataList items={riskEvidenceItems} />
            </Space>
          </PanelCard>

          <PanelCard title="异常事件与风险信号">
            <Space direction="vertical" size={16} style={{ width: '100%' }}>
              {anomalyEvents.length ? (
                <SignalList
                  items={anomalyEvents.map((event, index) => ({
                    key: `${event.code}-${event.symptom}-${index}`,
                    title: <Text strong>{event.symptom}</Text>,
                    tag: formatSeverityLabel(event.severity),
                    description: replaceTechnicalTerms(event.description),
                    tone: diagnosisTone(event.severity),
                  }))}
                />
              ) : (
                <Alert type="success" showIcon message="本次未检测到明显异常事件" description="系统主要依据寿命预测轨迹和历史状态给出解释结果。" />
              )}
              {diagnosis.related_symptoms.length ? (
                <StructuredDataList items={[{ label: '关联症状', value: diagnosis.related_symptoms }]} compact />
              ) : null}
            </Space>
          </PanelCard>

          <PanelCard title="候选机理排序">
            {diagnosis.candidate_faults.length ? (
              <List
                className="list-compact"
                dataSource={diagnosis.candidate_faults}
                renderItem={(item) => (
                  <List.Item>
                    <List.Item.Meta
                      title={`${item.name} · 分数 ${item.score.toFixed(3)}`}
                      description={
                        <Space direction="vertical" size={4} style={{ width: '100%' }}>
                          <Text type="secondary">{formatSeverityLabel(item.severity)}</Text>
                          <Text>{replaceTechnicalTerms(item.description)}</Text>
                          <Text type="secondary">匹配症状：{item.matched_symptoms.join('、') || '无'}</Text>
                          {item.confidence_basis?.length ? <Text type="secondary">排序依据：{item.confidence_basis.join('；')}</Text> : null}
                        </Space>
                      }
                    />
                  </List.Item>
                )}
              />
            ) : (
              <EmptyStateBlock compact title="暂无候选机理" description="执行机理解释后，这里会显示候选故障排序结果。" className="panel-empty-state" />
            )}
          </PanelCard>

          <PanelCard title="排序依据与模型证据">
            <Space direction="vertical" size={18} style={{ width: '100%' }}>
              <StructuredDataList items={modelEvidenceItems.filter((item) => Array.isArray(item.value) ? item.value.length > 0 : true)} />
              {diagnosis.root_causes.length ? (
                <div className="panel-section-block">
                  <Text className="panel-section-label">根因链</Text>
                  <List className="list-compact" size="small" dataSource={diagnosis.root_causes} renderItem={(item) => <List.Item>{item}</List.Item>} />
                </div>
              ) : null}
              <div className="panel-section-block">
                <Text className="panel-section-label">处理建议</Text>
                <List className="list-compact" size="small" dataSource={diagnosis.recommendations} renderItem={(item) => <List.Item>{item}</List.Item>} />
              </div>
              {diagnosis.evidence.length ? (
                <div className="panel-section-block">
                  <Text className="panel-section-label">证据条目</Text>
                  <List className="list-compact" size="small" dataSource={diagnosis.evidence} renderItem={(item) => <List.Item>{item}</List.Item>} />
                </div>
              ) : null}
            </Space>
          </PanelCard>
        </>
      ) : (
        <PanelCard className="empty-state-card">
          <EmptyStateBlock
            title="暂无机理解释结果"
            description="选择电池后即可执行“生命周期预测 → 异常检测 → 机理解释”流程。"
            icon={<FileTextOutlined className="empty-state-block__icon" />}
          />
        </PanelCard>
      )}
    </div>
  )
}

function diagnosisTone(severity?: string): StatusTone {
  if (severity === 'critical' || severity === 'high') return 'critical'
  if (severity === 'medium' || severity === 'warning') return 'warning'
  if (severity === 'info' || severity === 'low' || severity === 'success') return 'good'
  return 'neutral'
}

function toStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.map((item) => String(item)).filter(Boolean)
}

function hydrateMechanismFromHistory(record: DiagnosisRecord | undefined): MechanismExplanationResult | undefined {
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
    lifecycle_evidence: {},
    model_evidence: {},
    graph_backend: 'history',
  }
}

export default Diagnosis
