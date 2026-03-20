import React, { useEffect, useMemo, useState } from 'react'
import { Alert, Button, Descriptions, List, Select, Space, Tabs, Typography } from 'antd'
import { DownloadOutlined, PlayCircleOutlined, RadarChartOutlined } from '@ant-design/icons'

import { ChartPanel, EmptyStateBlock, InsightCard, PageHero, PanelCard, StatusTag } from '../components/ui'
import { useBhmsStore } from '../stores/useBhmsStore'
import type {
  AblationResult,
  CandidateFault,
  CaseBundle,
  CaseBundleExportResult,
  DatasetProfile,
  DiagnosisRecord,
  ExperimentDetail,
  LifecyclePredictionResult,
  MechanismExplanationResult,
  PredictionRecord,
  SupportedSource,
  SystemStatus,
  TrainingComparison,
} from '../types/domain'
import {
  formatFeatureLabel,
  formatModelLabel,
  formatSourceLabel,
  replaceTechnicalTerms,
} from '../utils/display'

const { Paragraph, Text, Title } = Typography

const Analysis: React.FC = () => {
  const batteries = useBhmsStore((state) => state.batteries)
  const selectedBatteryId = useBhmsStore((state) => state.selectedBatteryId)
  const selectBattery = useBhmsStore((state) => state.selectBattery)
  const loadBatteryContext = useBhmsStore((state) => state.loadBatteryContext)
  const latestPrediction = useBhmsStore((state) => (selectedBatteryId ? state.latestLifecyclePrediction[selectedBatteryId] : undefined))
  const latestMechanismExplanation = useBhmsStore((state) => (selectedBatteryId ? state.latestMechanismExplanation[selectedBatteryId] : undefined))
  const batteryCycles = useBhmsStore((state) => (selectedBatteryId ? state.batteryCycles[selectedBatteryId] ?? [] : []))
  const batteryHistory = useBhmsStore((state) => (selectedBatteryId ? state.batteryHistory[selectedBatteryId] : undefined))
  const trainingJobs = useBhmsStore((state) => state.trainingJobs)
  const trainingComparisonMap = useBhmsStore((state) => state.trainingComparison)
  const trainingOverview = useBhmsStore((state) => state.trainingOverview)
  const experimentDetails = useBhmsStore((state) => state.experimentDetails)
  const ablationMap = useBhmsStore((state) => state.ablationSummary)
  const systemStatus = useBhmsStore((state) => state.systemStatus)
  const datasetProfiles = useBhmsStore((state) => state.datasetProfiles)
  const knowledgeSummary = useBhmsStore((state) => state.knowledgeSummary)
  const caseBundles = useBhmsStore((state) => state.caseBundles)
  const caseBundleExports = useBhmsStore((state) => state.caseBundleExports)
  const loadTrainingJobs = useBhmsStore((state) => state.loadTrainingJobs)
  const loadTrainingComparison = useBhmsStore((state) => state.loadTrainingComparison)
  const loadTrainingOverview = useBhmsStore((state) => state.loadTrainingOverview)
  const loadExperimentDetail = useBhmsStore((state) => state.loadExperimentDetail)
  const loadAblationSummary = useBhmsStore((state) => state.loadAblationSummary)
  const startTrainingJob = useBhmsStore((state) => state.startTrainingJob)
  const loadSystemStatus = useBhmsStore((state) => state.loadSystemStatus)
  const loadDatasetProfile = useBhmsStore((state) => state.loadDatasetProfile)
  const loadKnowledgeSummary = useBhmsStore((state) => state.loadKnowledgeSummary)
  const loadCaseBundle = useBhmsStore((state) => state.loadCaseBundle)
  const exportCaseBundleAction = useBhmsStore((state) => state.exportCaseBundle)
  const getPredictionReportText = useBhmsStore((state) => state.getPredictionReportText)
  const getDiagnosisReportText = useBhmsStore((state) => state.getDiagnosisReportText)
  const trainingLoading = useBhmsStore((state) => state.trainingLoading)
  const insightLoading = useBhmsStore((state) => state.insightLoading)
  const [activeTab, setActiveTab] = useState('trajectory')
  const [modelScope, setModelScope] = useState<'all' | 'bilstm' | 'hybrid'>('all')
  const [jobKind, setJobKind] = useState<'baseline' | 'multi_seed' | 'ablation' | 'full_suite'>('full_suite')

  useEffect(() => {
    const batteryId = selectedBatteryId ?? batteries[0]?.battery_id
    if (batteryId) {
      selectBattery(batteryId)
      void loadBatteryContext(batteryId)
    }
  }, [batteries, loadBatteryContext, selectBattery, selectedBatteryId])

  const activeBattery = batteries.find((item) => item.battery_id === selectedBatteryId) ?? batteries[0]
  const source = activeBattery?.source ?? 'nasa'
  const comparison = trainingComparisonMap[source] ?? null
  const experimentDetail = experimentDetails[source]
  const ablation = ablationMap[source]
  const datasetProfile = datasetProfiles[source]
  const caseBundle = selectedBatteryId ? caseBundles[selectedBatteryId] : undefined
  const caseExportResult = selectedBatteryId ? caseBundleExports[selectedBatteryId] : undefined
  const sourceOverview = trainingOverview?.sources.find((item) => item.source === source)

  useEffect(() => {
    void loadSystemStatus()
    void loadTrainingOverview()
    void loadKnowledgeSummary()
  }, [loadKnowledgeSummary, loadSystemStatus, loadTrainingOverview])

  useEffect(() => {
    if (!source) return
    void loadTrainingJobs(source)
    void loadTrainingComparison(source)
    void loadExperimentDetail(source)
    void loadAblationSummary(source)
    void loadDatasetProfile(source)
    const intervalId = window.setInterval(() => {
      void loadTrainingJobs(source)
      void loadTrainingComparison(source)
    }, 5000)
    return () => window.clearInterval(intervalId)
  }, [loadAblationSummary, loadDatasetProfile, loadExperimentDetail, loadTrainingComparison, loadTrainingJobs, source])

  useEffect(() => {
    if (!selectedBatteryId) return
    void loadCaseBundle(selectedBatteryId)
  }, [loadCaseBundle, selectedBatteryId])

  const prediction = latestPrediction ?? hydratePredictionFromHistory(batteryHistory?.predictions?.[0])
  const mechanism = latestMechanismExplanation ?? hydrateMechanismFromHistory(batteryHistory?.diagnoses?.[0])

  const lifecycleChartOption = useMemo(() => buildLifecycleOption(prediction, batteryCycles), [batteryCycles, prediction])
  const graphOption = useMemo(() => buildGraphOption(mechanism), [mechanism])

  const pills = [
    { label: '当前电池', value: selectedBatteryId ?? '未选择', tone: 'teal' as const },
    { label: '当前来源', value: formatSourceLabel(source), tone: 'slate' as const },
    { label: '训练任务', value: trainingJobs.length, tone: 'amber' as const },
  ]

  return (
    <div className="page-shell">
      <PageHero
        kicker="Lifecycle Analysis Center"
        title="生命周期轨迹、机理解释、benchmark、案例包都放到同一个工作台"
        description="这里统一承载 lifecycle 证据链、GraphRAG 解释、实验结论、数据画像与案例导出，直接服务答辩演示。"
        pills={pills}
        aside={
          <InsightCard
            compact
            label="工作台状态"
            value={trainingLoading || insightLoading ? '刷新中' : 'Ready'}
            description="建议先跑 lifecycle 预测 / 机理解释，再进入分析工作台查看完整证据链与案例材料。"
          />
        }
      />

      <PanelCard style={{ marginBottom: 18 }}>
        <Space wrap align="center" style={{ width: '100%', justifyContent: 'space-between' }}>
          <Space wrap>
            <Text strong>分析对象</Text>
            <Select
              value={selectedBatteryId ?? undefined}
              placeholder="选择电池"
              options={batteries.map((item) => ({ label: item.battery_id, value: item.battery_id }))}
              style={{ minWidth: 240 }}
              onChange={(value) => {
                selectBattery(value)
                void loadBatteryContext(value)
              }}
            />
            <StatusTag tone="info">source={formatSourceLabel(source)}</StatusTag>
            <StatusTag tone={activeBattery?.include_in_training ? 'good' : 'neutral'}>
              {activeBattery?.include_in_training ? '已入训练池' : '未入训练池'}
            </StatusTag>
            {sourceOverview?.best_model ? <StatusTag tone="info">最佳模型：{formatModelLabel(sourceOverview.best_model)}</StatusTag> : null}
          </Space>
        </Space>
      </PanelCard>

      <Tabs
        activeKey={activeTab}
        onChange={setActiveTab}
        items={[
          {
            key: 'trajectory',
            label: '生命周期轨迹',
            children: (
              <div className="analysis-grid">
                <div>
                  <ChartPanel
                    title="观测轨迹与 future trajectory"
                    option={lifecycleChartOption}
                    hasData={Boolean(prediction?.projection?.actual_points.length || batteryCycles.length)}
                    height={400}
                    emptyTitle="暂无生命周期结果"
                    emptyDescription="先在生命周期预测页执行预测，再回到这里查看 trajectory 与关键节点。"
                    style={{ marginBottom: 18 }}
                  />

                  <div className="detail-grid detail-grid--two">
                    <PanelCard title="关键特征贡献">
                      {prediction?.explanation?.feature_contributions?.length ? (
                        <List
                          dataSource={prediction.explanation.feature_contributions}
                          renderItem={(item) => (
                            <List.Item>
                              <List.Item.Meta
                                title={`${formatFeatureLabel(item.feature)} · impact ${item.impact.toFixed(3)}`}
                                description={replaceTechnicalTerms(item.description)}
                              />
                            </List.Item>
                          )}
                        />
                      ) : (
                        <EmptyStateBlock compact title="暂无特征贡献" description="先执行生命周期预测，再查看结构化证据链。" className="panel-empty-state" />
                      )}
                    </PanelCard>

                    <PanelCard title="关键时间窗口">
                      {prediction?.explanation?.window_contributions?.length ? (
                        <List
                          dataSource={prediction.explanation.window_contributions}
                          renderItem={(item) => (
                            <List.Item>
                              <List.Item.Meta title={`${item.window_label} · impact ${item.impact.toFixed(3)}`} description={replaceTechnicalTerms(item.description)} />
                            </List.Item>
                          )}
                        />
                      ) : (
                        <EmptyStateBlock compact title="暂无窗口贡献" description="预测后会显示最关键的时间窗口。" className="panel-empty-state" />
                      )}
                    </PanelCard>
                  </div>
                </div>

                <div>
                  <PanelCard title="生命周期证据链" style={{ marginBottom: 18 }}>
                    {prediction ? (
                      <Descriptions column={1} size="small" className="details-grid">
                        <Descriptions.Item label="模型">{formatModelLabel(prediction.model_name)}</Descriptions.Item>
                        <Descriptions.Item label="预测 RUL">{prediction.predicted_rul.toFixed(1)} cycles</Descriptions.Item>
                        <Descriptions.Item label="knee 周期">{prediction.predicted_knee_cycle?.toFixed(1) ?? '--'}</Descriptions.Item>
                        <Descriptions.Item label="EOL 周期">{prediction.predicted_eol_cycle?.toFixed(1) ?? '--'}</Descriptions.Item>
                        <Descriptions.Item label="风险窗口">{prediction.risk_windows.length}</Descriptions.Item>
                        <Descriptions.Item label="衰退模式">{String(prediction.future_risks.future_capacity_fade_pattern ?? '--')}</Descriptions.Item>
                      </Descriptions>
                    ) : (
                      <EmptyStateBlock compact title="暂无生命周期证据" description="执行一次生命周期预测后，这里会展示模型与轨迹解释。" className="panel-empty-state" />
                    )}
                  </PanelCard>

                  <PanelCard title="风险窗口与报告导出">
                    {prediction ? (
                      <Space direction="vertical" size={16} style={{ width: '100%' }}>
                        {prediction.risk_windows.length ? (
                          <List
                            size="small"
                            dataSource={prediction.risk_windows}
                            renderItem={(item) => <List.Item>{`${item.label}: ${item.start_cycle} -> ${item.end_cycle} (${item.severity})`}</List.Item>}
                          />
                        ) : (
                          <Alert type="info" showIcon message="当前预测未标记额外风险窗口" />
                        )}
                        <Button
                          icon={<DownloadOutlined />}
                          onClick={() => void downloadTextReport(prediction.id, prediction.report_markdown, getPredictionReportText, 'lifecycle-prediction-report')}
                        >
                          导出生命周期报告
                        </Button>
                      </Space>
                    ) : (
                      <EmptyStateBlock compact title="暂无可导出报告" description="先执行生命周期预测。" className="panel-empty-state" />
                    )}
                  </PanelCard>
                </div>
              </div>
            ),
          },
          {
            key: 'mechanism',
            label: '机理解释',
            children: (
              <div className="analysis-grid">
                <div>
                  <ChartPanel
                    title="GraphRAG 证据子图"
                    option={graphOption}
                    hasData={Boolean(mechanism?.graph_trace?.nodes?.length)}
                    height={420}
                    emptyTitle="暂无机理解释"
                    emptyDescription="先在机理解释页执行一次流程，再回到这里查看 GraphRAG 子图。"
                    style={{ marginBottom: 18 }}
                  />

                  <div className="detail-grid detail-grid--two">
                    <PanelCard title="候选机理排序">
                      {mechanism?.candidate_faults?.length ? (
                        <List
                          dataSource={mechanism.candidate_faults}
                          renderItem={(item: CandidateFault) => (
                            <List.Item>
                              <List.Item.Meta
                                avatar={<RadarChartOutlined />}
                                title={`${item.name} · score ${item.score.toFixed(3)}`}
                                description={
                                  <Space direction="vertical" size={2} style={{ width: '100%' }}>
                                    <Text type="secondary">{item.description}</Text>
                                    <Text type="secondary">匹配症状：{item.matched_symptoms.join('、') || '无'}</Text>
                                    {item.confidence_basis?.length ? <Text type="secondary">排序依据：{item.confidence_basis.join('；')}</Text> : null}
                                  </Space>
                                }
                              />
                            </List.Item>
                          )}
                        />
                      ) : (
                        <EmptyStateBlock compact title="暂无候选机理" description="GraphRAG 检索后会显示 Top-K 候选机理。" className="panel-empty-state" />
                      )}
                    </PanelCard>

                    <PanelCard title="未来风险与模型证据">
                      {mechanism ? (
                        <Space direction="vertical" size={12} style={{ width: '100%' }}>
                          <List
                            size="small"
                            dataSource={[
                              `future fade pattern: ${String(mechanism.lifecycle_evidence.future_capacity_fade_pattern ?? '--')}`,
                              `temperature / resistance / voltage: ${String(mechanism.lifecycle_evidence.temperature_risk ?? '--')} / ${String(
                                mechanism.lifecycle_evidence.resistance_risk ?? '--',
                              )} / ${String(mechanism.lifecycle_evidence.voltage_risk ?? '--')}`,
                              `top features: ${((mechanism.model_evidence.top_features as string[]) ?? []).join('、') || '--'}`,
                              `critical windows: ${((mechanism.model_evidence.critical_windows as string[]) ?? []).join('、') || '--'}`,
                            ]}
                            renderItem={(item) => <List.Item>{item}</List.Item>}
                          />
                        </Space>
                      ) : (
                        <EmptyStateBlock compact title="暂无未来风险摘要" description="机理解释后会显示未来风险与模型证据。" className="panel-empty-state" />
                      )}
                    </PanelCard>
                  </div>
                </div>

                <div>
                  <PanelCard title="机理解释摘要" style={{ marginBottom: 18 }}>
                    {mechanism ? (
                      <Space direction="vertical" size={14} style={{ width: '100%' }}>
                        <Alert
                          type={mechanism.severity === 'critical' || mechanism.severity === 'high' ? 'error' : 'warning'}
                          showIcon
                          message={`${mechanism.fault_type} · 置信度 ${(mechanism.confidence * 100).toFixed(1)}%`}
                          description={mechanism.description}
                        />
                        <Title level={5}>根因链</Title>
                        <List size="small" dataSource={mechanism.root_causes} renderItem={(item) => <List.Item>{item}</List.Item>} />
                        <Title level={5}>处理建议</Title>
                        <List size="small" dataSource={mechanism.recommendations} renderItem={(item) => <List.Item>{item}</List.Item>} />
                        <Button
                          icon={<DownloadOutlined />}
                          onClick={() => void downloadTextReport(mechanism.id, mechanism.report_markdown, getDiagnosisReportText, 'mechanism-report')}
                        >
                          导出机理解释报告
                        </Button>
                      </Space>
                    ) : (
                      <EmptyStateBlock compact title="暂无机理解释" description="请先执行 GraphRAG 机理解释。" className="panel-empty-state" />
                    )}
                  </PanelCard>

                  <PanelCard title="证据条目">
                    {mechanism?.evidence?.length ? (
                      <List size="small" dataSource={mechanism.evidence} renderItem={(item) => <List.Item>{item}</List.Item>} />
                    ) : (
                      <EmptyStateBlock compact title="暂无证据条目" description="执行机理解释后显示异常摘要和图谱证据。" className="panel-empty-state" />
                    )}
                  </PanelCard>
                </div>
              </div>
            ),
          },
          {
            key: 'benchmark',
            label: 'benchmark',
            children: (
              <div className="analysis-grid">
                <div>
                  <PanelCard title="训练控制台" style={{ marginBottom: 18 }}>
                    <Space direction="vertical" size={16} style={{ width: '100%' }}>
                      <Alert
                        type="info"
                        showIcon
                        message="训练流程支持 lifecycle 基线、多随机种子、消融和完整实验套件"
                        description="完整套件会按 prepare -> baseline -> multi-seed -> ablation -> plots 的固定顺序执行。"
                      />
                      {sourceOverview ? (
                        <Alert
                          type={sourceOverview.warnings.length ? 'warning' : 'success'}
                          showIcon
                          message={sourceOverview.headline}
                          description={`${formatSourceLabel(source)} 来源当前最佳模型：${formatModelLabel(sourceOverview.best_model ?? '--')}。`}
                        />
                      ) : null}
                      <Space wrap>
                        <StatusTag tone="info">source={formatSourceLabel(source)}</StatusTag>
                        <Select<'baseline' | 'multi_seed' | 'ablation' | 'full_suite'>
                          value={jobKind}
                          style={{ minWidth: 180 }}
                          options={[
                            { label: '完整实验套件', value: 'full_suite' },
                            { label: '基线训练', value: 'baseline' },
                            { label: '多随机种子', value: 'multi_seed' },
                            { label: '消融实验', value: 'ablation' },
                          ]}
                          onChange={setJobKind}
                        />
                        <Select<'all' | 'bilstm' | 'hybrid'>
                          value={modelScope}
                          style={{ minWidth: 180 }}
                          options={[
                            { label: '全部模型', value: 'all' },
                            { label: 'Lifecycle Bi-LSTM', value: 'bilstm' },
                            { label: 'Lifecycle Hybrid', value: 'hybrid' },
                          ]}
                          onChange={setModelScope}
                        />
                        <Button
                          type="primary"
                          icon={<PlayCircleOutlined />}
                          loading={trainingLoading}
                          onClick={() => void startTrainingJob(source as SupportedSource, modelScope, jobKind, false, 3)}
                        >
                          发起训练任务
                        </Button>
                      </Space>
                    </Space>
                  </PanelCard>

                  <PanelCard title="实验概览" style={{ marginBottom: 18 }}>
                    {trainingOverview ? (
                      <Space direction="vertical" size={14} style={{ width: '100%' }}>
                        <List
                          size="small"
                          dataSource={trainingOverview.sources}
                          renderItem={(item) => (
                            <List.Item>
                              <List.Item.Meta
                                avatar={<RadarChartOutlined />}
                                title={`${formatSourceLabel(item.source)} · ${formatModelLabel(item.best_model ?? '--')}`}
                                description={`${item.headline}（样本电池 ${item.dataset_batteries} 节）`}
                              />
                            </List.Item>
                          )}
                        />
                        <List size="small" dataSource={trainingOverview.summary_notes} renderItem={(item) => <List.Item>{item}</List.Item>} />
                      </Space>
                    ) : (
                      <EmptyStateBlock compact title="暂无实验概览" description="等待后端聚合 overview 后，这里会统一显示各来源状态。" className="panel-empty-state" />
                    )}
                  </PanelCard>

                  <PanelCard title="当前来源实验详情">
                    {experimentDetail ? <ExperimentDetailPanel comparison={comparison} detail={experimentDetail} /> : <EmptyStateBlock compact title="暂无实验详情" description="等待 experiments 接口返回后，这里会展示 benchmark 摘要。" className="panel-empty-state" />}
                  </PanelCard>
                </div>

                <div>
                  <PanelCard title="消融实验与数据画像" style={{ marginBottom: 18 }}>
                    {ablation ? <AblationPanel ablation={ablation} /> : <EmptyStateBlock compact title="暂无消融结果" description="等待 ablations 接口返回后，这里会显示消融方案与建议命令。" className="panel-empty-state" />}
                    {datasetProfile ? <DatasetProfilePanel profile={datasetProfile} /> : null}
                  </PanelCard>

                  <PanelCard title="系统状态与知识库摘要">
                    {systemStatus ? <SystemStatusPanel status={systemStatus} /> : <EmptyStateBlock compact title="暂无系统状态" description="系统状态会在这里展示数据库、知识库和演示链路准备情况。" className="panel-empty-state" />}
                    {knowledgeSummary ? <KnowledgePanel graphBackend={systemStatus?.graph_backend} summary={knowledgeSummary} /> : null}
                  </PanelCard>
                </div>
              </div>
            ),
          },
          {
            key: 'case',
            label: '案例包',
            children: (
              <div className="analysis-grid">
                <div>
                  <PanelCard title="当前电池案例包" style={{ marginBottom: 18 }}>
                    {caseBundle ? <CaseBundlePanel bundle={caseBundle} /> : <EmptyStateBlock compact title="暂无案例包" description="选择电池后，系统会自动聚合案例材料。" className="panel-empty-state" />}
                  </PanelCard>

                  <PanelCard title="案例材料导出">
                    {caseBundle ? (
                      <Space direction="vertical" size={14} style={{ width: '100%' }}>
                        <Alert
                          type="info"
                          showIcon
                          message="案例包用于论文附录和答辩留档"
                          description="目录化导出会生成 lifecycle / mechanism Markdown、JSON 元数据和生命周期 / GraphRAG / benchmark 图表。"
                        />
                        <Space wrap>
                          <Button type="primary" icon={<DownloadOutlined />} onClick={() => downloadMarkdown(caseBundle.bundle_markdown, `${selectedBatteryId ?? 'battery'}-case-bundle.md`)}>
                            导出案例包
                          </Button>
                          <Button icon={<DownloadOutlined />} loading={insightLoading} onClick={() => selectedBatteryId && void exportCaseBundleAction(selectedBatteryId, true)}>
                            导出目录化案例
                          </Button>
                          {caseBundle.prediction?.id ? (
                            <Button
                              icon={<DownloadOutlined />}
                              onClick={() => void downloadTextReport(caseBundle.prediction!.id, caseBundle.prediction?.report_markdown ?? '', getPredictionReportText, 'lifecycle-prediction-report')}
                            >
                              导出生命周期报告
                            </Button>
                          ) : null}
                          {caseBundle.diagnosis?.id ? (
                            <Button
                              icon={<DownloadOutlined />}
                              onClick={() => void downloadTextReport(caseBundle.diagnosis!.id, caseBundle.diagnosis?.report_markdown ?? '', getDiagnosisReportText, 'mechanism-report')}
                            >
                              导出机理解释报告
                            </Button>
                          ) : null}
                        </Space>
                        {caseExportResult ? <CaseExportPanel exportResult={caseExportResult} /> : null}
                        <Paragraph className="log-block" copyable={{ text: caseBundle.bundle_markdown }}>
                          {previewText(caseBundle.bundle_markdown, 24)}
                        </Paragraph>
                      </Space>
                    ) : (
                      <EmptyStateBlock compact title="暂无导出材料" description="当生命周期预测和机理解释完成后，这里可以一键导出案例包。" className="panel-empty-state" />
                    )}
                  </PanelCard>
                </div>

                <div>
                  <PanelCard title="答辩讲解建议" style={{ marginBottom: 18 }}>
                    {caseBundle?.recommended_story?.length ? <List size="small" dataSource={caseBundle.recommended_story} renderItem={(item) => <List.Item>{item}</List.Item>} /> : <EmptyStateBlock compact title="暂无讲解脚本" description="案例包生成后会给出推荐讲解顺序。" className="panel-empty-state" />}
                  </PanelCard>

                  <PanelCard title="案例链路完整性">
                    {caseBundle ? (
                      <List
                        dataSource={caseBundle.artifacts}
                        renderItem={(item) => (
                          <List.Item>
                            <List.Item.Meta title={`${item.title} · ${item.available ? '已就绪' : '待补充'}`} description={item.description} />
                          </List.Item>
                        )}
                      />
                    ) : (
                      <EmptyStateBlock compact title="暂无案例链路" description="案例包会展示样本画像、报告和 benchmark 背景是否都已就绪。" className="panel-empty-state" />
                    )}
                  </PanelCard>
                </div>
              </div>
            ),
          },
        ]}
      />
    </div>
  )
}

function ExperimentDetailPanel({ comparison, detail }: { comparison: TrainingComparison | null; detail: ExperimentDetail }) {
  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Alert
        type={detail.warnings.length ? 'warning' : 'success'}
        showIcon
        message={detail.headline}
        description={`${detail.academic_status} · 当前最佳模型：${formatModelLabel(detail.best_model ?? '--')}`}
      />
      {detail.warnings.length ? <List size="small" dataSource={detail.warnings} renderItem={(item) => <List.Item>{item}</List.Item>} /> : null}
      <ComparisonPanel comparison={comparison} />
      <List
        dataSource={Object.values(detail.models)}
        renderItem={(item) => (
          <List.Item>
            <List.Item.Meta
              avatar={<RadarChartOutlined />}
              title={`${formatModelLabel(item.model_type)} · ${item.multi_seed_summary ? '多随机种子已具备' : '单次实验'}`}
              description={
                <Space direction="vertical" size={2}>
                  <Text>单次 trajectory RMSE：{formatMetric(item.test_metrics.rmse)}</Text>
                  <Text>单次 trajectory MAE：{formatMetric(item.test_metrics.mae)}</Text>
                  <Text>单次 trajectory R2：{formatMetric(item.test_metrics.r2)}</Text>
                  <Text>
                    聚合 RMSE：{formatMetric(item.aggregate_metrics?.mean?.rmse)} ± {formatMetric(item.aggregate_metrics?.std?.rmse)}
                  </Text>
                  <Text>
                    聚合 R2：{formatMetric(item.aggregate_metrics?.mean?.r2)} ± {formatMetric(item.aggregate_metrics?.std?.r2)}
                  </Text>
                  <Text>seed 运行数：{item.per_seed_runs?.length ?? 0}</Text>
                  <Text>Checkpoint：{item.best_checkpoint ?? '--'}</Text>
                  <Text>{item.assessment}</Text>
                </Space>
              }
            />
          </List.Item>
        )}
      />
      <Title level={5}>关键结论</Title>
      <List size="small" dataSource={detail.key_findings} renderItem={(item) => <List.Item>{item}</List.Item>} />
      {detail.plots?.length ? (
        <>
          <Title level={5}>论文图表产物</Title>
          <List size="small" dataSource={detail.plots} renderItem={(item) => <List.Item>{item.title}：{item.path}</List.Item>} />
        </>
      ) : null}
      <Title level={5}>建议命令</Title>
      <Paragraph className="log-block" copyable>
        {Object.values(detail.recommended_commands).join('\n')}
      </Paragraph>
    </Space>
  )
}

function AblationPanel({ ablation }: { ablation: AblationResult }) {
  if (!ablation.available) {
    return (
      <Space direction="vertical" size={14} style={{ width: '100%', marginBottom: 18 }}>
        <Alert type="warning" showIcon message="当前来源尚未生成 ablation_summary.json" description="系统已给出默认消融方案和建议命令。" />
        <List size="small" dataSource={ablation.notes} renderItem={(item) => <List.Item>{item}</List.Item>} />
        {ablation.recommended_command ? <Paragraph className="log-block" copyable>{ablation.recommended_command}</Paragraph> : null}
      </Space>
    )
  }

  return (
    <List
      dataSource={ablation.variants}
      renderItem={(item) => (
        <List.Item>
          <List.Item.Meta
            title={`${item.label} · ${item.status ?? 'available'}`}
            description={
              <Space direction="vertical" size={2}>
                <Text>{item.description}</Text>
                <Text>
                  聚合 RMSE：{formatMetric(item.aggregate_metrics?.mean?.rmse)} ± {formatMetric(item.aggregate_metrics?.std?.rmse)}
                </Text>
                <Text>相对 full_hybrid 的 RMSE 变化：{formatMetric(item.delta_vs_full?.rmse)}</Text>
                <Text>seed 数：{item.seeds?.length ?? 0}</Text>
              </Space>
            }
          />
        </List.Item>
      )}
    />
  )
}

function DatasetProfilePanel({ profile }: { profile: DatasetProfile }) {
  return (
    <Space direction="vertical" size={16} style={{ width: '100%', marginTop: 18 }}>
      <Descriptions column={1} size="small">
        <Descriptions.Item label="来源">{formatSourceLabel(profile.source)}</Descriptions.Item>
        <Descriptions.Item label="电池数量">{profile.battery_count} 节</Descriptions.Item>
        <Descriptions.Item label="训练候选">{profile.training_candidate_count} 节</Descriptions.Item>
        <Descriptions.Item label="周期点">{profile.cycle_point_count} 条</Descriptions.Item>
        <Descriptions.Item label="训练/验证/测试划分">{formatSplitSummary(profile.split)}</Descriptions.Item>
      </Descriptions>
      <Title level={5}>可用特征列</Title>
      <Space wrap>
        {profile.available_feature_columns.map((item) => (
          <StatusTag key={item} tone="neutral">
            {formatFeatureLabel(item)}
          </StatusTag>
        ))}
      </Space>
    </Space>
  )
}

function SystemStatusPanel({ status }: { status: SystemStatus }) {
  return (
    <Space direction="vertical" size={16} style={{ width: '100%', marginBottom: 18 }}>
      <Descriptions column={1} size="small">
        <Descriptions.Item label="图谱后端">{status.graph_backend}</Descriptions.Item>
        <Descriptions.Item label="数据库就绪">{status.database_ready ? '是' : '否'}</Descriptions.Item>
        <Descriptions.Item label="知识库就绪">{status.knowledge_ready ? '是' : '否'}</Descriptions.Item>
        <Descriptions.Item label="演示预设">{status.demo_preset_count} 份</Descriptions.Item>
      </Descriptions>
      <Title level={5}>标准验收流</Title>
      <List size="small" dataSource={status.demo_acceptance_flow} renderItem={(item) => <List.Item>{item}</List.Item>} />
      {status.warnings.length ? <List size="small" dataSource={status.warnings} renderItem={(item) => <List.Item>{item}</List.Item>} /> : null}
    </Space>
  )
}

function KnowledgePanel({
  summary,
  graphBackend,
}: {
  summary: {
    fault_count: number
    symptom_alias_count: number
    categories: Record<string, number>
    top_symptoms: Array<[string, number]>
    coverage_notes: string[]
    source_coverage?: Record<string, number>
    evidence_sources?: Array<[string, number]>
    rule_count?: number
    threshold_rule_count?: number
  }
  graphBackend?: string
}) {
  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Alert
        type="info"
        showIcon
        message={`当前知识库包含 ${summary.fault_count} 类故障、${summary.symptom_alias_count} 个症状别名、${summary.rule_count ?? 0} 条规则`}
        description={`GraphRAG 运行后端：${graphBackend ?? 'unknown'}；带阈值提示规则 ${summary.threshold_rule_count ?? 0} 条`}
      />
      <List size="small" dataSource={summary.coverage_notes} renderItem={(item) => <List.Item>{item}</List.Item>} />
    </Space>
  )
}

function CaseBundlePanel({ bundle }: { bundle: CaseBundle }) {
  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Descriptions column={1} size="small">
        <Descriptions.Item label="电池 ID">{bundle.battery_id}</Descriptions.Item>
        <Descriptions.Item label="来源">{formatSourceLabel(bundle.source ?? '--')}</Descriptions.Item>
        <Descriptions.Item label="数据集">{bundle.dataset_name ?? '--'}</Descriptions.Item>
        <Descriptions.Item label="健康分">{formatMetric(bundle.health_score)}</Descriptions.Item>
        <Descriptions.Item label="循环次数">{bundle.cycle_count ?? '--'}</Descriptions.Item>
        <Descriptions.Item label="数据划分">{bundle.dataset_position?.split_name ?? '--'}</Descriptions.Item>
      </Descriptions>
      {bundle.prediction ? (
        <Alert
          type="info"
          showIcon
          message={`生命周期预测：RUL ${formatMetric(bundle.prediction.predicted_rul)} cycles`}
          description={`knee ${formatMetric(bundle.prediction.predicted_knee_cycle)} / EOL ${formatMetric(bundle.prediction.predicted_eol_cycle)}`}
        />
      ) : (
        <Alert type="warning" showIcon message="当前尚无生命周期预测记录" />
      )}
      {bundle.diagnosis ? (
        <Alert
          type={bundle.diagnosis.severity === 'critical' || bundle.diagnosis.severity === 'high' ? 'error' : 'warning'}
          showIcon
          message={`机理解释：${bundle.diagnosis.fault_type}`}
          description={`根因 ${bundle.diagnosis.root_causes.slice(0, 2).join('；') || '--'}`}
        />
      ) : (
        <Alert type="warning" showIcon message="当前尚无机理解释记录" />
      )}
      <Title level={5}>案例依赖资产</Title>
      <List size="small" dataSource={bundle.artifacts} renderItem={(item) => <List.Item>{item.title}：{item.available ? '已就绪' : '待补充'}，{item.description}</List.Item>} />
      {bundle.last_export ? (
        <Alert
          type="success"
          showIcon
          message={`最近一次目录导出：${bundle.last_export.export_dir}`}
          description={`导出时间 ${bundle.last_export.generated_at ?? '--'}，文件数 ${bundle.last_export.files?.length ?? 0}`}
        />
      ) : null}
    </Space>
  )
}

function CaseExportPanel({ exportResult }: { exportResult: CaseBundleExportResult }) {
  return (
    <Space direction="vertical" size={12} style={{ width: '100%' }}>
      <Alert
        type="success"
        showIcon
        message={`目录已导出到 ${exportResult.export_dir}`}
        description={`自动补预测：${exportResult.generated_artifacts.prediction_generated ? '是' : '否'}；自动补诊断：${exportResult.generated_artifacts.diagnosis_generated ? '是' : '否'}`}
      />
      <List size="small" dataSource={exportResult.files} renderItem={(item) => <List.Item>{item.kind}：{item.path}</List.Item>} />
    </Space>
  )
}

function buildLifecycleOption(prediction: LifecyclePredictionResult | undefined, batteryCycles: Array<{ cycle_number: number; capacity: number }>) {
  const actual = prediction?.projection?.actual_points ?? batteryCycles.map((item) => ({ cycle: item.cycle_number, capacity: item.capacity }))
  const forecast = prediction?.projection?.forecast_points ?? []
  const band = prediction?.projection?.confidence_band ?? []
  return {
    backgroundColor: 'transparent',
    tooltip: { trigger: 'axis', backgroundColor: 'rgba(16, 35, 63, 0.92)', borderWidth: 0, textStyle: { color: '#f8fafc' } },
    legend: { top: 4, textStyle: { color: '#5f6c7b' } },
    grid: { left: 28, right: 24, top: 56, bottom: 28, containLabel: true },
    xAxis: { type: 'value', name: '循环次数', axisLabel: { color: '#6b7280' }, splitLine: { show: false } },
    yAxis: { type: 'value', name: '容量(Ah)', axisLabel: { color: '#6b7280' }, splitLine: { lineStyle: { color: 'rgba(16, 35, 63, 0.08)' } } },
    series: [
      {
        name: '观测轨迹',
        type: 'line',
        smooth: true,
        showSymbol: false,
        data: actual.map((item) => [item.cycle, item.capacity]),
        lineStyle: { color: '#0071e3', width: 3 },
      },
      {
        name: '未来 trajectory',
        type: 'line',
        smooth: true,
        showSymbol: false,
        data: forecast.map((item) => [item.cycle, item.capacity]),
        lineStyle: { color: '#34c759', width: 3, type: 'dashed' },
        markLine: {
          symbol: 'none',
          data: [
            prediction?.predicted_knee_cycle ? { xAxis: prediction.predicted_knee_cycle, name: 'knee' } : undefined,
            prediction?.predicted_eol_cycle ? { xAxis: prediction.predicted_eol_cycle, name: 'EOL' } : undefined,
          ].filter(Boolean),
        },
      },
      {
        name: '置信带上界',
        type: 'line',
        symbol: 'none',
        data: band.map((item) => [item.cycle, item.upper]),
        lineStyle: { opacity: 0 },
        stack: 'confidence-band',
      },
      {
        name: '置信带下界',
        type: 'line',
        symbol: 'none',
        data: band.map((item) => [item.cycle, item.lower]),
        lineStyle: { opacity: 0 },
        areaStyle: { color: 'rgba(52, 199, 89, 0.12)' },
        stack: 'confidence-band',
      },
    ],
  }
}

function buildGraphOption(diagnosis: MechanismExplanationResult | undefined) {
  const nodes = diagnosis?.graph_trace?.nodes ?? []
  const edges = diagnosis?.graph_trace?.edges ?? []
  return {
    backgroundColor: 'transparent',
    tooltip: { backgroundColor: 'rgba(16, 35, 63, 0.92)', borderWidth: 0, textStyle: { color: '#f8fafc' } },
    series: [
      {
        type: 'graph',
        layout: 'force',
        roam: true,
        force: { repulsion: 180, edgeLength: 120 },
        data: nodes.map((node) => ({
          id: node.id,
          name: node.label,
          symbolSize: node.node_type === 'Fault' ? 64 : node.node_type === 'Symptom' ? 52 : 44,
          itemStyle: {
            color:
              node.node_type === 'Fault'
                ? '#0071e3'
                : node.node_type === 'Symptom'
                  ? '#ff9f0a'
                  : node.node_type === 'Cause'
                    ? '#ff453a'
                    : '#34c759',
          },
          label: { show: true, color: '#10233f' },
        })),
        links: edges.map((edge) => ({ source: edge.source, target: edge.target, value: edge.relation })),
        lineStyle: { color: 'rgba(16, 35, 63, 0.18)', curveness: 0.08 },
        emphasis: { focus: 'adjacency', lineStyle: { width: 2 } },
      },
    ],
  }
}

function ComparisonPanel({ comparison }: { comparison: TrainingComparison | null }) {
  const currentModels = extractModels(comparison?.current)
  const previousModels = extractModels(comparison?.previous)
  const modelKeys = Array.from(new Set([...Object.keys(currentModels), ...Object.keys(previousModels)]))

  return modelKeys.length ? (
    <List
      dataSource={modelKeys}
      renderItem={(modelKey) => {
        const current = currentModels[modelKey]
        const previous = previousModels[modelKey]
        return (
          <List.Item>
            <List.Item.Meta
              avatar={<RadarChartOutlined />}
              title={formatModelLabel(modelKey)}
              description={
                <Space direction="vertical" size={4}>
                  <Text>当前 best val loss：{formatMetric(current?.best_val_loss)}</Text>
                  <Text>之前 best val loss：{formatMetric(previous?.best_val_loss)}</Text>
                  <Text>当前 RMSE：{formatMetric((current?.test_metrics as Record<string, number> | undefined)?.rmse)}</Text>
                  <Text>当前 MAE：{formatMetric((current?.test_metrics as Record<string, number> | undefined)?.mae)}</Text>
                  <Text>当前 R2：{formatMetric((current?.test_metrics as Record<string, number> | undefined)?.r2)}</Text>
                </Space>
              }
            />
          </List.Item>
        )
      }}
    />
  ) : (
    <EmptyStateBlock compact title="暂无模型对比" description="等待 comparison_summary 生成后，这里会展示训练前后对比。" className="panel-empty-state" />
  )
}

function extractModels(payload: Record<string, unknown> | null | undefined) {
  if (!payload || typeof payload !== 'object') return {} as Record<string, Record<string, unknown>>
  const models = (payload as { models?: Record<string, Record<string, unknown>> }).models
  return models ?? {}
}

function formatMetric(value: unknown) {
  if (typeof value !== 'number' || Number.isNaN(value)) return '--'
  return value.toFixed(4)
}

function formatSplitSummary(split: Record<string, unknown>) {
  const entries = Object.entries(split)
  if (!entries.length) return '--'
  return entries.map(([key, value]) => `${key}:${String(value)}`).join(' / ')
}

function previewText(value: string, lines: number) {
  return value.split('\n').slice(0, lines).join('\n')
}

function hydratePredictionFromHistory(record: PredictionRecord | undefined) {
  if (!record || !record.projection || !record.model_version || !record.model_source) return undefined
  return {
    ...record,
    model_version: record.model_version,
    model_source: record.model_source,
    fallback_used: Boolean(record.fallback_used),
    prediction_time: record.prediction_time ?? record.created_at,
    trajectory: record.trajectory ?? [],
    risk_windows: record.risk_windows ?? [],
    future_risks: record.future_risks ?? {},
    model_evidence: record.model_evidence ?? {},
    projection: record.projection,
    explanation: record.explanation ?? null,
    report_markdown: record.report_markdown ?? '',
  } as LifecyclePredictionResult
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

function downloadMarkdown(content: string, fileName: string) {
  const safeName = fileName.replace(/[^a-zA-Z0-9._-]+/g, '-')
  const blob = new Blob([content], { type: 'text/markdown;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = safeName
  anchor.click()
  URL.revokeObjectURL(url)
}

async function downloadTextReport(id: number, content: string, fetcher: (id: number) => Promise<string>, fileStem: string) {
  const fallback = content || (await fetcher(id))
  downloadMarkdown(fallback, `${fileStem}-${id}.md`)
}

export default Analysis
