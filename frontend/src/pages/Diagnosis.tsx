import React, { useEffect, useState } from 'react'
import { Alert, Button, Descriptions, List, Select, Space, Steps, Typography } from 'antd'
import { AlertOutlined, FileTextOutlined, RightOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

import { EmptyStateBlock, InsightCard, PageHero, PanelCard, SignalList, StatusTag, type StatusTone } from '../components/ui'
import { useBhmsStore } from '../stores/useBhmsStore'
import { formatSeverityLabel } from '../utils/display'

const { Text, Title } = Typography

const Diagnosis: React.FC = () => {
  const navigate = useNavigate()
  const batteries = useBhmsStore((state) => state.batteries)
  const selectedBatteryId = useBhmsStore((state) => state.selectedBatteryId)
  const selectBattery = useBhmsStore((state) => state.selectBattery)
  const loadBatteryContext = useBhmsStore((state) => state.loadBatteryContext)
  const runDiagnosisWorkflow = useBhmsStore((state) => state.runDiagnosisWorkflow)
  const actionLoading = useBhmsStore((state) => state.actionLoading)
  const latestAnomaly = useBhmsStore((state) => (selectedBatteryId ? state.latestAnomaly[selectedBatteryId] : undefined))
  const latestMechanismExplanation = useBhmsStore((state) => (selectedBatteryId ? state.latestMechanismExplanation[selectedBatteryId] : undefined))
  const latestLifecyclePrediction = useBhmsStore((state) => (selectedBatteryId ? state.latestLifecyclePrediction[selectedBatteryId] : undefined))
  const batteryHistory = useBhmsStore((state) => (selectedBatteryId ? state.batteryHistory[selectedBatteryId] : undefined))
  const [currentStep, setCurrentStep] = useState(0)

  useEffect(() => {
    const batteryId = selectedBatteryId ?? batteries[0]?.battery_id
    if (batteryId) {
      selectBattery(batteryId)
      void loadBatteryContext(batteryId)
    }
  }, [batteries, loadBatteryContext, selectBattery, selectedBatteryId])

  const currentDiagnosis = latestMechanismExplanation ?? undefined
  const fallbackDiagnosis = batteryHistory?.diagnoses?.[0]
  const anomalyEvents = latestAnomaly?.events ?? batteryHistory?.anomalies ?? []
  const diagnosisSeverity = currentDiagnosis?.severity ?? fallbackDiagnosis?.severity
  const diagnosisAlertType =
    diagnosisSeverity === 'critical' || diagnosisSeverity === 'high'
      ? 'error'
      : diagnosisSeverity === 'info' || currentDiagnosis?.fault_type === '未发现明显故障'
        ? 'success'
        : 'warning'

  return (
    <div className="page-shell">
      <PageHero
        kicker="Mechanism Explanation"
        title="让机理解释，更值得信任"
        description="把异常事件、生命周期证据、未来风险窗口和 GraphRAG 证据链放到同一条路径里。"
        pills={[
          { label: '当前电池', value: selectedBatteryId ?? '未选择', tone: 'teal' },
          { label: '异常事件', value: anomalyEvents.length, tone: anomalyEvents.length ? 'amber' : 'slate' },
          { label: '生命周期结果', value: latestLifecyclePrediction ? '已生成' : '待生成', tone: latestLifecyclePrediction ? 'teal' : 'slate' },
        ]}
        aside={
          <InsightCard
            compact
            label="当前机理解释"
            value={currentDiagnosis?.fault_type ?? fallbackDiagnosis?.fault_type ?? '--'}
            description={
              currentDiagnosis
                ? `置信度 ${(currentDiagnosis.confidence * 100).toFixed(1)}%，可继续查看未来风险和证据链。`
                : '建议先生成生命周期预测，再执行机理解释。'
            }
          />
        }
      />

      <div className="diagnosis-grid">
        <div>
          <PanelCard title="解释配置" style={{ marginBottom: 18 }}>
            <Space direction="vertical" size="middle" style={{ width: '100%' }}>
              <Select
                value={selectedBatteryId ?? undefined}
                placeholder="选择电池"
                options={batteries.map((item) => ({ label: item.battery_id, value: item.battery_id }))}
                onChange={(value) => {
                  selectBattery(value)
                  void loadBatteryContext(value)
                  setCurrentStep(0)
                }}
              />
              <Button
                type="primary"
                icon={<AlertOutlined />}
                loading={actionLoading}
                onClick={() => {
                  if (!selectedBatteryId) return
                  setCurrentStep(1)
                  void runDiagnosisWorkflow(selectedBatteryId)
                    .then(() => setCurrentStep(3))
                    .catch(() => setCurrentStep(0))
                }}
                block
              >
                开始机理解释
              </Button>
              <Alert className="inline-feedback" type="info" showIcon message="建议按“生命周期预测 -> 异常检测 -> 机理解释 -> GraphRAG 证据链”的顺序演示" />
              <Button icon={<RightOutlined />} block onClick={() => navigate('/analysis')}>
                查看分析工作台
              </Button>
            </Space>
          </PanelCard>

          <PanelCard title="解释链路">
            <Steps
              direction="vertical"
              current={currentStep}
              size="small"
              items={[
                { title: '生命周期预测', description: '形成 trajectory / knee / EOL / RUL 证据' },
                { title: '异常检测', description: '标准化症状事件' },
                { title: '机理解释', description: '结合未来风险窗口排序候选机理' },
                { title: 'GraphRAG 证据链', description: '输出图谱证据、建议和报告' },
              ]}
            />
          </PanelCard>
        </div>

        <div>
          {currentDiagnosis ? (
            <>
              <Alert
                className="diagnosis-banner"
                type={diagnosisAlertType}
                showIcon
                message={
                  <Space size={10} wrap>
                    <Text strong className="diagnosis-banner__title">
                      {currentDiagnosis.fault_type}
                    </Text>
                    <StatusTag tone={diagnosisTone(currentDiagnosis.severity)}>{currentDiagnosis.severity}</StatusTag>
                  </Space>
                }
                description={
                  <div>
                    <Text>置信度 {(currentDiagnosis.confidence * 100).toFixed(1)}%</Text>
                    <br />
                    <Text type="secondary">{currentDiagnosis.description}</Text>
                  </div>
                }
              />

              <PanelCard title="异常事件与未来风险" style={{ marginBottom: 18 }}>
                {anomalyEvents.length ? (
                  <SignalList
                    items={anomalyEvents.map((event, index) => ({
                      key: `${event.code}-${event.symptom}-${index}`,
                      title: <Text strong>{event.symptom}</Text>,
                      tag: event.severity,
                      description: event.description,
                      tone: diagnosisTone(event.severity),
                    }))}
                  />
                ) : (
                  <Alert type="success" showIcon message="本次未检测到异常事件" description="系统已基于生命周期证据和历史状态给出解释结果。" />
                )}
                <Descriptions column={1} size="small" style={{ marginTop: 16 }}>
                  <Descriptions.Item label="future fade pattern">
                    {String(currentDiagnosis.lifecycle_evidence?.future_capacity_fade_pattern ?? '--')}
                  </Descriptions.Item>
                  <Descriptions.Item label="temperature / resistance / voltage risks">
                    {String(currentDiagnosis.lifecycle_evidence?.temperature_risk ?? '--')} / {String(currentDiagnosis.lifecycle_evidence?.resistance_risk ?? '--')} /{' '}
                    {String(currentDiagnosis.lifecycle_evidence?.voltage_risk ?? '--')}
                  </Descriptions.Item>
                  <Descriptions.Item label="predicted knee / EOL">
                    {String(currentDiagnosis.lifecycle_evidence?.predicted_knee_cycle ?? '--')} / {String(currentDiagnosis.lifecycle_evidence?.predicted_eol_cycle ?? '--')}
                  </Descriptions.Item>
                </Descriptions>
              </PanelCard>

              <div className="detail-grid detail-grid--two">
                <PanelCard title="候选机理与建议">
                  <Title level={5}>候选机理排序</Title>
                  <List
                    dataSource={currentDiagnosis.candidate_faults}
                    renderItem={(item) => (
                      <List.Item>
                        <Space direction="vertical" size={4} style={{ width: '100%' }}>
                          <Text strong>{item.name}</Text>
                          <Text type="secondary">
                            score {item.score.toFixed(3)} · {formatSeverityLabel(item.severity)}
                          </Text>
                          <Text>{item.description}</Text>
                        </Space>
                      </List.Item>
                    )}
                  />
                  <Title level={5}>处理建议</Title>
                  <List dataSource={currentDiagnosis.recommendations} renderItem={(item) => <List.Item>{item}</List.Item>} />
                </PanelCard>

                <PanelCard title="证据链">
                  <Title level={5}>GraphRAG 排序依据</Title>
                  <List dataSource={currentDiagnosis.graph_trace.ranking_basis} renderItem={(item) => <List.Item>{item}</List.Item>} />
                  {currentDiagnosis.model_evidence?.top_features ? (
                    <>
                      <Title level={5}>模型证据</Title>
                      <List
                        size="small"
                        dataSource={[
                          `关键特征：${(currentDiagnosis.model_evidence.top_features as string[]).join('、') || '--'}`,
                          `关键窗口：${((currentDiagnosis.model_evidence.critical_windows as string[]) ?? []).join('、') || '--'}`,
                        ]}
                        renderItem={(item) => <List.Item>{item}</List.Item>}
                      />
                    </>
                  ) : null}
                </PanelCard>
              </div>
            </>
          ) : (
            <PanelCard className="empty-state-card">
              <EmptyStateBlock
                title="暂无机理解释结果"
                description="选择电池后即可执行“生命周期预测 -> 异常检测 -> 机理解释”流程。"
                icon={<FileTextOutlined className="empty-state-block__icon" />}
              />
            </PanelCard>
          )}
        </div>
      </div>
    </div>
  )
}

function diagnosisTone(severity?: string): StatusTone {
  if (severity === 'critical' || severity === 'high') return 'critical'
  if (severity === 'medium' || severity === 'warning') return 'warning'
  if (severity === 'info' || severity === 'low' || severity === 'success') return 'good'
  return 'neutral'
}

export default Diagnosis
