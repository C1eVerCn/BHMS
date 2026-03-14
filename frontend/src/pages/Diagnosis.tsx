import React, { useEffect, useState } from 'react'
import { Alert, Button, Card, List, Select, Space, Steps, Tag, Typography } from 'antd'
import { AlertOutlined, CheckCircleOutlined, FileTextOutlined } from '@ant-design/icons'

import { useBhmsStore } from '../stores/useBhmsStore'

const { Title, Text } = Typography

const Diagnosis: React.FC = () => {
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

  return (
    <div>
      <Title level={2} className="page-title">
        故障诊断
      </Title>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: 16 }}>
        <div>
          <Card title="诊断配置" style={{ marginBottom: 16 }}>
            <Space direction="vertical" style={{ width: '100%' }}>
              <Select
                value={selectedBatteryId ?? undefined}
                placeholder="选择电池"
                options={batteries.map((item) => ({ label: item.battery_id, value: item.battery_id }))}
                onChange={(value) => {
                  selectBattery(value)
                  void loadBatteryContext(value)
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
            </Space>
          </Card>

          <Card title="诊断链路">
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
          </Card>
        </div>

        <div>
          {currentDiagnosis ? (
            <>
              <Alert
                style={{ marginBottom: 16 }}
                type={currentDiagnosis.severity === 'critical' || currentDiagnosis.severity === 'high' ? 'error' : 'warning'}
                showIcon
                message={
                  <Space>
                    <Text strong style={{ fontSize: 16 }}>
                      {currentDiagnosis.fault_type}
                    </Text>
                    <Tag color={currentDiagnosis.severity === 'critical' ? 'red' : currentDiagnosis.severity === 'high' ? 'orange' : 'blue'}>
                      {currentDiagnosis.severity}
                    </Tag>
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

              <Card title="最新异常事件" style={{ marginBottom: 16 }}>
                {latestAnomaly?.events.length ? (
                  <List
                    dataSource={latestAnomaly.events}
                    renderItem={(event) => (
                      <List.Item>
                        <List.Item.Meta
                          avatar={<CheckCircleOutlined style={{ color: event.severity === 'high' ? '#ff4d4f' : '#faad14' }} />}
                          title={`${event.symptom} · ${event.severity}`}
                          description={event.description}
                        />
                      </List.Item>
                    )}
                  />
                ) : (
                  <Text type="secondary">暂无新异常事件</Text>
                )}
              </Card>

              <Card title="根因与建议" style={{ marginBottom: 16 }}>
                <Title level={5}>根本原因</Title>
                <List dataSource={currentDiagnosis.root_causes} renderItem={(item) => <List.Item>{item}</List.Item>} />
                <Title level={5}>处理建议</Title>
                <List dataSource={currentDiagnosis.recommendations} renderItem={(item) => <List.Item>{item}</List.Item>} />
              </Card>

              <Card title="诊断依据">
                <List dataSource={currentDiagnosis.evidence} renderItem={(item) => <List.Item>{item}</List.Item>} />
              </Card>
            </>
          ) : (
            <Card style={{ textAlign: 'center', padding: '60px 0' }}>
              <FileTextOutlined style={{ fontSize: 56, color: '#d9d9d9' }} />
              <Title level={4} style={{ marginTop: 24 }}>
                暂无诊断结果
              </Title>
              <Text type="secondary">选择电池后即可执行“异常检测 → GraphRAG 诊断”流程。</Text>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}

export default Diagnosis
