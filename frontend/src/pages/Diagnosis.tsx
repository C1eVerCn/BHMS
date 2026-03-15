import React, { useEffect, useState } from 'react'
import { Alert, Button, List, Select, Space, Steps, Typography } from 'antd'
import { AlertOutlined, FileTextOutlined, RightOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

import { EmptyStateBlock, InsightCard, PageHero, PanelCard, SignalList, StatusTag, type StatusTone } from '../components/ui'
import { useBhmsStore } from '../stores/useBhmsStore'

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
  const latestDiagnosis = useBhmsStore((state) => (selectedBatteryId ? state.latestDiagnosis[selectedBatteryId] : undefined))
  const batteryHistory = useBhmsStore((state) => (selectedBatteryId ? state.batteryHistory[selectedBatteryId] : undefined))
  const [currentStep, setCurrentStep] = useState(0)

  useEffect(() => {
    const batteryId = selectedBatteryId ?? batteries[0]?.battery_id
    if (batteryId) {
      selectBattery(batteryId)
      void loadBatteryContext(batteryId)
    }
  }, [batteries, loadBatteryContext, selectBattery, selectedBatteryId])

  const currentDiagnosis = latestDiagnosis ?? batteryHistory?.diagnoses?.[0]
  const anomalyEvents = latestAnomaly?.events ?? []
  const diagnosisAlertType =
    currentDiagnosis?.severity === 'critical' || currentDiagnosis?.severity === 'high'
      ? 'error'
      : currentDiagnosis?.severity === 'info' || currentDiagnosis?.fault_type === '未发现明显故障'
        ? 'success'
        : 'warning'

  return (
    <div className="page-shell">
      <PageHero
        kicker="Fault Diagnosis"
        title="让诊断结果，更值得信任"
        description="把异常、依据与建议放进同一条清晰路径里。"
        pills={[
          { label: '当前电池', value: selectedBatteryId ?? '未选择', tone: 'teal' },
          { label: '异常事件', value: anomalyEvents.length, tone: anomalyEvents.length ? 'amber' : 'slate' },
          { label: '诊断级别', value: currentDiagnosis?.severity ?? '待生成', tone: diagnosisPillTone(currentDiagnosis?.severity) },
        ]}
        aside={
          <InsightCard
            compact
            label="当前诊断"
            value={currentDiagnosis?.fault_type ?? '--'}
            description={
              currentDiagnosis
                ? `置信度 ${(currentDiagnosis.confidence * 100).toFixed(1)}%，可继续查看根因和建议。`
                : '选择电池后即可生成诊断结果。'
            }
          />
        }
      />

      <div className="diagnosis-grid">
        <div>
          <PanelCard title="诊断配置" style={{ marginBottom: 18 }}>
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
                开始检测与诊断
              </Button>
              <Alert
                className="inline-feedback"
                type="info"
                showIcon
                message="建议按“选电池 -> 跑诊断 -> 看证据”的顺序演示"
              />
              <Button icon={<RightOutlined />} block onClick={() => navigate('/analysis')}>
                查看 GraphRAG 证据链
              </Button>
            </Space>
          </PanelCard>

          <PanelCard title="诊断链路">
            <Steps
              direction="vertical"
              current={currentStep}
              size="small"
              items={[
                { title: '选择电池', description: '读取周期级健康数据' },
                { title: '异常检测', description: '标准化症状事件' },
                { title: '知识检索', description: '匹配候选故障与规则' },
                { title: '诊断生成', description: '输出故障报告与建议' },
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

              <PanelCard title="最新异常事件" style={{ marginBottom: 18 }}>
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
                  <Alert
                    type="success"
                    showIcon
                    message="本次未检测到异常事件"
                    description="系统已按“健康状态说明”生成诊断结果，因此不会再因为缺少异常而报错。"
                  />
                )}
              </PanelCard>

              <div className="detail-grid detail-grid--two">
                <PanelCard title="根因与建议">
                  <Title level={5}>根本原因</Title>
                  <List dataSource={currentDiagnosis.root_causes} renderItem={(item) => <List.Item>{item}</List.Item>} />
                  <Title level={5}>处理建议</Title>
                  <List dataSource={currentDiagnosis.recommendations} renderItem={(item) => <List.Item>{item}</List.Item>} />
                </PanelCard>

                <PanelCard title="诊断依据">
                  <Space direction="vertical" size={14} style={{ width: '100%' }}>
                    <List dataSource={currentDiagnosis.evidence} renderItem={(item) => <List.Item>{item}</List.Item>} />
                    {currentDiagnosis.decision_basis?.length ? (
                      <>
                        <Title level={5}>为什么优先判断为这个故障</Title>
                        <List dataSource={currentDiagnosis.decision_basis} renderItem={(item) => <List.Item>{item}</List.Item>} />
                      </>
                    ) : null}
                    <Alert
                      type="info"
                      showIcon
                      message={`候选故障 ${currentDiagnosis.candidate_faults?.length ?? 0} 个，可进入分析中心查看完整子图与排序依据。`}
                    />
                  </Space>
                </PanelCard>
              </div>

              <PanelCard title="候选故障排序" style={{ marginTop: 18 }}>
                {currentDiagnosis.candidate_faults?.length ? (
                  <List
                    dataSource={currentDiagnosis.candidate_faults}
                    renderItem={(item) => (
                      <List.Item>
                        <Space direction="vertical" size={4} style={{ width: '100%' }}>
                          <Space size={8} wrap>
                            <Text strong>{item.name}</Text>
                            <StatusTag tone={diagnosisTone(item.severity)}>{item.severity}</StatusTag>
                            <Text type="secondary">score {item.score.toFixed(3)}</Text>
                            {item.rule_id ? <Text code>{item.rule_id}</Text> : null}
                          </Space>
                          <Text type="secondary">
                            匹配症状 {item.matched_symptom_count ?? item.matched_symptoms.length}/{item.all_symptoms?.length ?? item.matched_symptoms.length}
                            ，覆盖率 {typeof item.symptom_coverage === 'number' ? item.symptom_coverage.toFixed(3) : '--'}
                          </Text>
                          <Text>{item.description}</Text>
                          {item.confidence_basis?.length ? (
                            <Text type="secondary">排序依据：{item.confidence_basis.join('；')}</Text>
                          ) : null}
                          {item.evidence_source?.length ? (
                            <Text type="secondary">证据来源：{item.evidence_source.join('、')}</Text>
                          ) : null}
                        </Space>
                      </List.Item>
                    )}
                  />
                ) : (
                  <Alert type="info" showIcon message="暂无候选故障排序" />
                )}
              </PanelCard>
            </>
          ) : (
            <PanelCard className="empty-state-card">
              <EmptyStateBlock
                title="暂无诊断结果"
                description="选择电池后即可执行“异常检测 -&gt; GraphRAG 诊断”流程。"
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

function diagnosisPillTone(severity?: string) {
  if (severity === 'critical' || severity === 'high') return 'rose'
  if (severity === 'medium' || severity === 'warning') return 'amber'
  if (severity === 'info' || severity === 'low' || severity === 'success') return 'teal'
  return 'slate'
}
