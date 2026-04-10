import React, { useEffect, useMemo } from 'react'
import { Alert, Button, List, Space, Typography } from 'antd'
import { DownloadOutlined } from '@ant-design/icons'

import { BatterySelect, ChartPanel, EmptyStateBlock, PageHero, PanelCard, StatusTag, StructuredDataList } from '../components/ui'
import { useBhmsStore } from '../stores/useBhmsStore'
import type {
  AblationResult,
  CaseBundleExportResult,
  DiagnosisRecord,
  ExperimentDetail,
  KnowledgeSummary,
  LifecyclePredictionResult,
  MechanismExplanationResult,
  PredictionRecord,
  TrainingComparison,
} from '../types/domain'
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

const { Paragraph, Text, Title } = Typography

const sectionLinks = [
  { id: 'analysis-profile', label: '样本概况' },
  { id: 'analysis-lifecycle', label: '生命周期全景预测' },
  { id: 'analysis-evidence', label: '模型解释证据' },
  { id: 'analysis-graphrag', label: 'GraphRAG 机理解释' },
  { id: 'analysis-experiments', label: '实验结果' },
  { id: 'analysis-case', label: '案例导出' },
]

const Analysis: React.FC = () => {
  const batteryOptions = useBhmsStore((state) => state.batteryOptions)
  const batteryById = useBhmsStore((state) => state.batteryById)
  const selectedBatteryId = useBhmsStore((state) => state.selectedBatteryId)
  const selectBattery = useBhmsStore((state) => state.selectBattery)
  const loadBatteryContext = useBhmsStore((state) => state.loadBatteryContext)
  const latestPrediction = useBhmsStore((state) => (selectedBatteryId ? state.latestLifecyclePrediction[selectedBatteryId] : undefined))
  const latestMechanismExplanation = useBhmsStore((state) => (selectedBatteryId ? state.latestMechanismExplanation[selectedBatteryId] : undefined))
  const batteryCycles = useBhmsStore((state) => (selectedBatteryId ? state.batteryCycles[selectedBatteryId] ?? [] : []))
  const batteryHistory = useBhmsStore((state) => (selectedBatteryId ? state.batteryHistory[selectedBatteryId] : undefined))
  const trainingComparisonMap = useBhmsStore((state) => state.trainingComparison)
  const trainingOverview = useBhmsStore((state) => state.trainingOverview)
  const experimentDetails = useBhmsStore((state) => state.experimentDetails)
  const ablationMap = useBhmsStore((state) => state.ablationSummary)
  const systemStatus = useBhmsStore((state) => state.systemStatus)
  const datasetProfiles = useBhmsStore((state) => state.datasetProfiles)
  const knowledgeSummary = useBhmsStore((state) => state.knowledgeSummary)
  const caseBundles = useBhmsStore((state) => state.caseBundles)
  const caseBundleExports = useBhmsStore((state) => state.caseBundleExports)
  const loadTrainingComparison = useBhmsStore((state) => state.loadTrainingComparison)
  const loadTrainingOverview = useBhmsStore((state) => state.loadTrainingOverview)
  const loadExperimentDetail = useBhmsStore((state) => state.loadExperimentDetail)
  const loadAblationSummary = useBhmsStore((state) => state.loadAblationSummary)
  const loadSystemStatus = useBhmsStore((state) => state.loadSystemStatus)
  const loadDatasetProfile = useBhmsStore((state) => state.loadDatasetProfile)
  const loadKnowledgeSummary = useBhmsStore((state) => state.loadKnowledgeSummary)
  const loadCaseBundle = useBhmsStore((state) => state.loadCaseBundle)
  const exportCaseBundleAction = useBhmsStore((state) => state.exportCaseBundle)
  const insightLoading = useBhmsStore((state) => state.insightLoading)

  useEffect(() => {
    const batteryId = selectedBatteryId ?? batteryOptions[0]?.battery_id
    if (!batteryId) return
    if (selectedBatteryId !== batteryId) {
      selectBattery(batteryId)
    }
    void loadBatteryContext(batteryId)
  }, [batteryOptions, loadBatteryContext, selectBattery, selectedBatteryId])

  const activeBattery = selectedBatteryId ? batteryById[selectedBatteryId] : undefined
  const source = activeBattery?.source ?? batteryOptions[0]?.source ?? 'nasa'

  useEffect(() => {
    void loadSystemStatus()
    void loadTrainingOverview()
    void loadKnowledgeSummary()
  }, [loadKnowledgeSummary, loadSystemStatus, loadTrainingOverview])

  useEffect(() => {
    if (!source) return
    void loadTrainingComparison(source)
    void loadExperimentDetail(source)
    void loadAblationSummary(source)
    void loadDatasetProfile(source)
  }, [loadAblationSummary, loadDatasetProfile, loadExperimentDetail, loadTrainingComparison, source])

  useEffect(() => {
    if (!selectedBatteryId) return
    void loadCaseBundle(selectedBatteryId)
  }, [loadCaseBundle, selectedBatteryId])

  const prediction = latestPrediction ?? hydratePredictionFromHistory(batteryHistory?.predictions?.[0])
  const mechanism = latestMechanismExplanation ?? hydrateMechanismFromHistory(batteryHistory?.diagnoses?.[0])
  const comparison = trainingComparisonMap[source] ?? null
  const experimentDetail = experimentDetails[source]
  const ablation = ablationMap[source]
  const datasetProfile = datasetProfiles[source]
  const caseBundle = selectedBatteryId ? caseBundles[selectedBatteryId] : undefined
  const caseExportResult = selectedBatteryId ? caseBundleExports[selectedBatteryId] : undefined
  const sourceOverview = trainingOverview?.sources.find((item) => item.source === source)
  const experimentModels = useMemo(() => Object.values(experimentDetail?.models ?? {}), [experimentDetail])

  const sampleProfileItems = useMemo(() => {
    const items = buildBatteryProfileItems(activeBattery)
    return [
      ...items,
      { label: '历史预测记录', value: `${batteryHistory?.predictions.length ?? 0} 条` },
      { label: '历史诊断记录', value: `${batteryHistory?.diagnoses.length ?? 0} 条` },
      { label: '当前状态', value: formatSeverityLabel(activeBattery?.status, '--') },
    ]
  }, [activeBattery, batteryHistory])

  const lifecycleOption = useMemo(() => buildLifecycleChartOption({ batteryCycles, prediction }), [batteryCycles, prediction])
  const graphOption = useMemo(() => buildGraphOption(mechanism), [mechanism])

  const lifecycleSummaryItems = useMemo(
    () => [
      { label: '当前模型', value: prediction ? formatModelLabel(prediction.model_name) : '--' },
      { label: '预测 RUL', value: prediction ? `${prediction.predicted_rul.toFixed(1)} 次` : '--' },
      { label: 'knee 周期', value: prediction?.predicted_knee_cycle ? prediction.predicted_knee_cycle.toFixed(1) : '--' },
      { label: 'EOL 周期', value: prediction?.predicted_eol_cycle ? prediction.predicted_eol_cycle.toFixed(1) : '--' },
      { label: '容量归零周期', value: prediction?.projection?.predicted_zero_cycle ? prediction.projection.predicted_zero_cycle.toFixed(1) : '--' },
      { label: '投影方法', value: formatProjectionMethod(prediction?.projection?.projection_method) },
    ],
    [prediction],
  )

  const explanationItems = useMemo(
    () => [
      { label: '关键特征', value: (prediction?.explanation?.feature_contributions ?? []).slice(0, 5).map((item) => `${formatFeatureLabel(item.feature)} (${item.impact.toFixed(3)})`) },
      { label: '关键时间窗口', value: (prediction?.explanation?.window_contributions ?? []).slice(0, 5).map((item) => `${item.window_label} (${item.impact.toFixed(3)})`) },
      { label: '置信度因素', value: toStringList(prediction?.explanation?.confidence_summary?.factors) },
      { label: '未来风险摘要', value: [
        `衰退模式：${String(prediction?.future_risks?.future_capacity_fade_pattern ?? '--')}`,
        `温度风险：${String(prediction?.future_risks?.temperature_risk ?? '--')}`,
        `内阻风险：${String(prediction?.future_risks?.resistance_risk ?? '--')}`,
        `电压风险：${String(prediction?.future_risks?.voltage_risk ?? '--')}`,
      ] },
    ],
    [prediction],
  )

  const graphSummaryItems = useMemo(
    () => [
      { label: '当前判断', value: mechanism?.fault_type ?? '--' },
      { label: '严重程度', value: formatSeverityLabel(mechanism?.severity) },
      { label: '置信度', value: mechanism ? `${(mechanism.confidence * 100).toFixed(1)}%` : '--' },
      { label: '关联症状', value: mechanism?.related_symptoms ?? [] },
      { label: '排序依据', value: mechanism?.graph_trace?.ranking_basis ?? [] },
      { label: '模型证据', value: [
        ...toStringList(mechanism?.model_evidence?.top_features),
        ...toStringList(mechanism?.model_evidence?.critical_windows),
      ] },
    ],
    [mechanism],
  )

  const experimentSummaryItems = useMemo(
    () => [
      { label: '数据来源', value: formatSourceLabel(source) },
      { label: '样本电池', value: sourceOverview?.dataset_batteries ? `${sourceOverview.dataset_batteries} 节` : datasetProfile?.battery_count ? `${datasetProfile.battery_count} 节` : '--' },
      { label: '当前数据集', value: datasetProfile?.dataset_names?.length ? datasetProfile.dataset_names : activeBattery?.dataset_name ?? '--' },
      { label: '最佳模型', value: formatModelLabel(experimentDetail?.best_model ?? sourceOverview?.best_model ?? sourceOverview?.best_models?.within_source) },
      { label: '可用模型', value: experimentModels.length ? experimentModels.map((item) => formatModelLabel(item.model_type)) : '--' },
      { label: '结构对比', value: ablation?.available ? `已生成 ${ablation.variants.length} 组` : '未提供' },
      { label: '图表资产', value: experimentDetail?.plots?.length ? `${experimentDetail.plots.length} 张` : '--' },
    ],
    [ablation, activeBattery?.dataset_name, datasetProfile, experimentDetail, experimentModels, source, sourceOverview],
  )

  const datasetProfileItems = useMemo(
    () =>
      datasetProfile
        ? [
            { label: '训练数据', value: `${datasetProfile.training_candidate_count} 节` },
            { label: '周期点', value: `${datasetProfile.cycle_point_count} 条` },
            { label: '数据集', value: datasetProfile.dataset_names },
            { label: '训练/验证/测试', value: formatSplitSummary(datasetProfile.split) },
            { label: '预置样本', value: datasetProfile.demo_files.map((item) => item.name) },
          ]
        : [],
    [datasetProfile],
  )

  const systemStatusItems = useMemo(
    () =>
      systemStatus
        ? [
            { label: '图谱后端', value: systemStatus.graph_backend },
            { label: '数据库', value: systemStatus.database_ready ? '已就绪' : '未就绪' },
            { label: '知识库', value: systemStatus.knowledge_ready ? '已就绪' : '未就绪' },
            { label: '预置样本', value: `${systemStatus.demo_preset_count} 份` },
          ]
        : [],
    [systemStatus],
  )

  const caseItems = useMemo(
    () =>
      caseBundle
        ? [
            { label: '电池 ID', value: caseBundle.battery_id },
            { label: '来源', value: formatSourceLabel(caseBundle.source) },
            { label: '数据集', value: caseBundle.dataset_name ?? '--' },
            { label: '健康分', value: formatMetric(caseBundle.health_score) },
            { label: '循环次数', value: caseBundle.cycle_count ? `${caseBundle.cycle_count}` : '--' },
            { label: '数据划分', value: caseBundle.dataset_position?.split_name ?? '--' },
          ]
        : [],
    [caseBundle],
  )

  return (
    <div className="page-shell page-shell--stacked">
      <PageHero
        title="分析工作台"
        description="围绕当前样本组织生命周期预测、模型解释、GraphRAG 机理证据、实验结果与案例导出。"
        pills={[
          { label: '当前电池', value: selectedBatteryId ?? '未选择', tone: 'teal' },
          { label: '数据来源', value: formatSourceLabel(source), tone: 'slate' },
        ]}
      />

      <PanelCard title="分析对象">
        <Space direction="vertical" size={18} style={{ width: '100%' }}>
          <div className="stacked-form-grid stacked-form-grid--two">
            <div>
              <Text className="panel-section-label">电池 ID</Text>
              <BatterySelect
                value={selectedBatteryId ?? undefined}
                options={batteryOptions}
                onChange={(value) => {
                  selectBattery(value)
                  void loadBatteryContext(value)
                }}
              />
            </div>
            <div>
              <Text className="panel-section-label">当前来源</Text>
              <StatusTag tone="info">{formatSourceLabel(source)}</StatusTag>
            </div>
          </div>

          <div className="sequence-nav">
            {sectionLinks.map((item) => (
              <button
                key={item.id}
                type="button"
                className="sequence-nav__button"
                onClick={() => document.getElementById(item.id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })}
              >
                {item.label}
              </button>
            ))}
          </div>
        </Space>
      </PanelCard>

      <section id="analysis-profile" className="sequence-section">
        <SectionHeading title="样本概况" description="展示当前样本的来源、数据集、规格和衰减状态。" />
        <PanelCard title="样本概况">
          {sampleProfileItems.length ? (
            <StructuredDataList items={sampleProfileItems} />
          ) : (
            <EmptyStateBlock compact title="暂无样本概况" description="选择电池后，这里会展示来源、协议、循环数和当前容量。" className="panel-empty-state" />
          )}
        </PanelCard>
      </section>

      <section id="analysis-lifecycle" className="sequence-section">
        <SectionHeading title="生命周期全景预测" description="展示完整历史轨迹、从当前状态到容量归零的连续预测，以及 knee 和 EOL 关键节点。" />
        <ChartPanel
          title="全周期生命轨迹"
          option={lifecycleOption}
          hasData={Boolean(prediction?.projection?.actual_points.length || batteryCycles.length)}
          height={460}
          emptyTitle="暂无生命周期结果"
          emptyDescription="执行一次生命周期预测后，这里会显示完整历史、全周期预测、knee 和 EOL 关键节点。"
        />
        <PanelCard title="关键里程碑与风险窗口">
          {prediction ? (
            <Space direction="vertical" size={18} style={{ width: '100%' }}>
              <StructuredDataList items={lifecycleSummaryItems} />
              {prediction.risk_windows.length ? (
                <List
                  className="list-compact"
                  dataSource={prediction.risk_windows}
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
                <Alert type="info" showIcon message="当前没有额外风险窗口" description="本次主预测没有识别出需要额外强调的加速衰退区段。" />
              )}
            </Space>
          ) : (
            <EmptyStateBlock compact title="暂无生命周期摘要" description="先在生命周期预测页完成一次预测。" className="panel-empty-state" />
          )}
        </PanelCard>
      </section>

      <section id="analysis-evidence" className="sequence-section">
        <SectionHeading title="模型解释证据" description="把关键特征、关键时间窗口和置信度因素整理成结构化证据。" />
        <PanelCard title="解释证据总览">
          {prediction ? (
            <Space direction="vertical" size={18} style={{ width: '100%' }}>
              <StructuredDataList items={explanationItems.filter((item) => Array.isArray(item.value) ? item.value.length > 0 : true)} />
              <div className="panel-section-block">
                <Text className="panel-section-label">关键特征贡献</Text>
                {prediction.explanation?.feature_contributions?.length ? (
                  <List
                    className="list-compact"
                    dataSource={prediction.explanation.feature_contributions}
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
                  <Paragraph className="panel-subtle-copy">当前没有返回更细粒度的特征贡献。</Paragraph>
                )}
              </div>
              <div className="panel-section-block">
                <Text className="panel-section-label">关键时间窗口</Text>
                {prediction.explanation?.window_contributions?.length ? (
                  <List
                    className="list-compact"
                    dataSource={prediction.explanation.window_contributions}
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
                  <Paragraph className="panel-subtle-copy">当前没有返回更细粒度的时间窗口证据。</Paragraph>
                )}
              </div>
            </Space>
          ) : (
            <EmptyStateBlock compact title="暂无模型解释证据" description="预测完成后，这里会显示关键特征、关键窗口和置信度因素。" className="panel-empty-state" />
          )}
        </PanelCard>
      </section>

      <section id="analysis-graphrag" className="sequence-section">
        <SectionHeading title="GraphRAG 机理解释" description="这一段用来回答“为什么系统判断是这个故障”，既展示证据网络，也把排序依据和模型证据拆出来讲。" />
        <ChartPanel
          className="graph-panel"
          title="机理证据网络 / 为什么系统判断是这个故障"
          option={graphOption}
          hasData={Boolean(mechanism?.graph_trace?.nodes?.length)}
          height={500}
          emptyTitle="暂无 GraphRAG 机理解释"
          emptyDescription="先执行机理解释流程，这里会展示候选故障与证据节点构成的关系网络。"
        />
        <PanelCard title="机理解释摘要">
          {mechanism ? (
            <Space direction="vertical" size={18} style={{ width: '100%' }}>
              <Alert
                type={mechanism.severity === 'critical' || mechanism.severity === 'high' ? 'error' : 'warning'}
                showIcon
                message={`${mechanism.fault_type} · 置信度 ${(mechanism.confidence * 100).toFixed(1)}%`}
                description={replaceTechnicalTerms(mechanism.description)}
              />
              <StructuredDataList items={graphSummaryItems.filter((item) => Array.isArray(item.value) ? item.value.length > 0 : true)} />
              {mechanism.candidate_faults.length ? (
                <div className="panel-section-block">
                  <Text className="panel-section-label">候选机理排序</Text>
                  <List
                    className="list-compact"
                    dataSource={mechanism.candidate_faults}
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
                </div>
              ) : null}
              {mechanism.recommendations.length ? (
                <div className="panel-section-block">
                  <Text className="panel-section-label">处理建议</Text>
                  <List className="list-compact" size="small" dataSource={mechanism.recommendations} renderItem={(item) => <List.Item>{item}</List.Item>} />
                </div>
              ) : null}
            </Space>
          ) : (
            <EmptyStateBlock compact title="暂无机理解释摘要" description="先执行机理解释流程，再回到这里展示故障判断与证据链。" className="panel-empty-state" />
          )}
        </PanelCard>
      </section>

      <section id="analysis-experiments" className="sequence-section">
        <SectionHeading title="实验结果" description="汇总当前来源的模型对比、结构对比和数据画像信息。" />
        <PanelCard title="当前来源实验摘要">
          <Space direction="vertical" size={18} style={{ width: '100%' }}>
            <StructuredDataList items={experimentSummaryItems} />
            {!experimentDetail && !sourceOverview ? (
              <EmptyStateBlock compact title="暂无实验摘要" description="等待实验结果聚合完成后，这里会显示当前来源的整体结论。" className="panel-empty-state" />
            ) : null}
          </Space>
        </PanelCard>

        <PanelCard title="模型对比与结构对比">
          <Space direction="vertical" size={18} style={{ width: '100%' }}>
            {experimentDetail ? <ExperimentModelsPanel detail={experimentDetail} comparison={comparison} /> : null}
            {ablation ? <AblationSummaryPanel ablation={ablation} /> : null}
            {!experimentDetail && !ablation ? (
              <EmptyStateBlock compact title="暂无模型对比结果" description="等待 comparison 和 ablation 结果准备完成后，这里会补齐。" className="panel-empty-state" />
            ) : null}
          </Space>
        </PanelCard>

        <PanelCard title="数据画像与系统准备情况">
          <Space direction="vertical" size={18} style={{ width: '100%' }}>
            {datasetProfileItems.length ? (
              <>
                <StructuredDataList items={datasetProfileItems} />
                <StructuredDataList items={[{ label: '可用特征', value: datasetProfile?.available_feature_columns.map((item) => formatFeatureLabel(item)) ?? [] }]} compact />
              </>
            ) : null}
            {systemStatusItems.length ? <StructuredDataList items={systemStatusItems} /> : null}
            {knowledgeSummary ? <KnowledgePanel summary={knowledgeSummary} /> : null}
            {!datasetProfileItems.length && !systemStatusItems.length && !knowledgeSummary ? (
              <EmptyStateBlock compact title="暂无系统与数据画像信息" description="等待 insight 接口完成加载。" className="panel-empty-state" />
            ) : null}
          </Space>
        </PanelCard>
      </section>

      <section id="analysis-case" className="sequence-section">
        <SectionHeading title="案例导出" description="整理当前样本的案例材料、导出文件和链路完整性信息。" />
        <PanelCard title="当前案例材料">
          {caseBundle ? (
            <Space direction="vertical" size={18} style={{ width: '100%' }}>
              <StructuredDataList items={caseItems} />
              <Space wrap>
                <Button type="primary" icon={<DownloadOutlined />} onClick={() => downloadMarkdown(caseBundle.bundle_markdown, `${selectedBatteryId ?? 'battery'}-case-bundle.md`)}>
                  导出案例包
                </Button>
                <Button icon={<DownloadOutlined />} loading={insightLoading} onClick={() => selectedBatteryId && void exportCaseBundleAction(selectedBatteryId, true)}>
                  导出目录化案例
                </Button>
                {caseBundle.prediction?.report_markdown ? (
                  <Button
                    icon={<DownloadOutlined />}
                    onClick={() => downloadTextReport(caseBundle.prediction?.report_markdown ?? '', `${selectedBatteryId ?? 'battery'}-lifecycle-prediction-report.md`)}
                  >
                    导出生命周期报告
                  </Button>
                ) : null}
                {caseBundle.diagnosis?.report_markdown ? (
                  <Button
                    icon={<DownloadOutlined />}
                    onClick={() => downloadTextReport(caseBundle.diagnosis?.report_markdown ?? '', `${selectedBatteryId ?? 'battery'}-mechanism-report.md`)}
                  >
                    导出机理解释报告
                  </Button>
                ) : null}
              </Space>
              {caseBundle.artifacts.length ? (
                <div className="panel-section-block">
                  <Text className="panel-section-label">案例链路完整性</Text>
                  <List
                    className="list-compact"
                    dataSource={caseBundle.artifacts}
                    renderItem={(item) => (
                      <List.Item>
                        <List.Item.Meta title={`${item.title} · ${item.available ? '已就绪' : '待补充'}`} description={item.description} />
                      </List.Item>
                    )}
                  />
                </div>
              ) : null}
              {caseExportResult ? <CaseExportPanel exportResult={caseExportResult} /> : null}
            </Space>
          ) : (
            <EmptyStateBlock compact title="暂无案例材料" description="案例包会在聚合预测、诊断和数据画像后自动生成。" className="panel-empty-state" />
          )}
        </PanelCard>
      </section>
    </div>
  )
}

function SectionHeading({ title, description }: { title: string; description: string }) {
  return (
    <div className="section-heading">
      <span className="section-heading__eyebrow">分析模块</span>
      <Title level={4} className="section-heading__title">
        {title}
      </Title>
      <Paragraph className="section-heading__description">{description}</Paragraph>
    </div>
  )
}

function ExperimentModelsPanel({ detail, comparison: _comparison }: { detail: ExperimentDetail; comparison: TrainingComparison | null }) {
  const displayFindings = filterDisplayNotes(detail.key_findings)
  const loadedModels = Object.values(detail.models)
  const bestModel = formatModelLabel(detail.best_model ?? detail.best_models?.within_source)

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      <Alert
        type="info"
        showIcon
        message={bestModel !== '--' ? `当前来源最佳结果：${bestModel}` : '已载入当前来源的模型结果'}
        description={`共加载 ${loadedModels.length} 个模型结果，这里按统一指标展示各模型的整体表现。`}
      />
      <List
        className="list-compact"
        dataSource={loadedModels}
        renderItem={(item) => (
          <List.Item>
            <List.Item.Meta
              title={`${formatModelLabel(item.model_type)} · ${item.multi_seed_available ? '多随机种子结果' : '单次实验结果'}`}
              description={
                <Space direction="vertical" size={4} style={{ width: '100%' }}>
                  <Text>RMSE：{formatMetric(item.test_metrics.rmse)}</Text>
                  <Text>MAE：{formatMetric(item.test_metrics.mae)}</Text>
                  <Text>R2：{formatMetric(item.test_metrics.r2)}</Text>
                  <Text>聚合 RMSE：{formatMetric(item.aggregate_metrics?.mean?.rmse)} ± {formatMetric(item.aggregate_metrics?.std?.rmse)}</Text>
                  <Text>聚合 R2：{formatMetric(item.aggregate_metrics?.mean?.r2)} ± {formatMetric(item.aggregate_metrics?.std?.r2)}</Text>
                  {item.preferred_metrics?.trajectory_rmse != null ? <Text>轨迹 RMSE：{formatMetric(item.preferred_metrics.trajectory_rmse)}</Text> : null}
                </Space>
              }
            />
          </List.Item>
        )}
      />
      {displayFindings.length ? (
        <div className="panel-section-block">
          <Text className="panel-section-label">关键结论</Text>
          <List className="list-compact" size="small" dataSource={displayFindings} renderItem={(item) => <List.Item>{item}</List.Item>} />
        </div>
      ) : null}
    </Space>
  )
}

function AblationSummaryPanel({ ablation }: { ablation: AblationResult }) {
  if (!ablation.available) {
    return (
      <Alert
        type="info"
        showIcon
        message="当前来源还没有生成结构对比结果"
        description="结构对比用于说明去掉某个模块后，模型效果会出现怎样的变化。"
      />
    )
  }

  return (
    <Space direction="vertical" size={16} style={{ width: '100%' }}>
      {ablation.guardrail ? (
        <Alert
          type={ablation.guardrail.passed ? 'success' : 'info'}
          showIcon
          message={`结构对比已加载 ${ablation.variants.length} 组结果`}
          description={`已检查 ${ablation.guardrail.checked_variants.length} 个变体。`}
        />
      ) : null}
      <List
        className="list-compact"
        dataSource={ablation.variants}
        renderItem={(item) => (
          <List.Item>
            <List.Item.Meta
              title={item.label}
              description={
                <Space direction="vertical" size={4} style={{ width: '100%' }}>
                  <Text>{item.description}</Text>
                  <Text>聚合 RMSE：{formatMetric(item.aggregate_metrics?.mean?.rmse)} ± {formatMetric(item.aggregate_metrics?.std?.rmse)}</Text>
                  <Text>聚合 R2：{formatMetric(item.aggregate_metrics?.mean?.r2)} ± {formatMetric(item.aggregate_metrics?.std?.r2)}</Text>
                  <Text>相对完整模型的 RMSE 变化：{formatMetric(item.delta_vs_full?.rmse)}</Text>
                </Space>
              }
            />
          </List.Item>
        )}
      />
    </Space>
  )
}

function KnowledgePanel({ summary }: { summary: KnowledgeSummary }) {
  const displayNotes = filterDisplayNotes(summary.coverage_notes)

  return (
    <div className="panel-section-block">
      <Text className="panel-section-label">知识库与图谱覆盖</Text>
      <StructuredDataList
        items={[
          { label: '故障类型', value: `${summary.fault_count} 类` },
          { label: '症状别名', value: `${summary.symptom_alias_count} 个` },
          { label: '规则数量', value: `${summary.rule_count ?? 0} 条` },
          { label: '阈值规则', value: `${summary.threshold_rule_count ?? 0} 条` },
          { label: '高频症状', value: summary.top_symptoms.map(([item, count]) => `${item} (${count})`) },
        ]}
      />
      {displayNotes.length ? <List className="list-compact" size="small" dataSource={displayNotes} renderItem={(item) => <List.Item>{item}</List.Item>} /> : null}
    </div>
  )
}

function CaseExportPanel({ exportResult }: { exportResult: CaseBundleExportResult }) {
  return (
    <div className="panel-section-block">
      <Text className="panel-section-label">最近一次目录导出</Text>
      <Alert
        type="success"
        showIcon
        message={`目录已导出到 ${exportResult.export_dir}`}
        description={`自动补预测：${exportResult.generated_artifacts.prediction_generated ? '是' : '否'}；自动补诊断：${exportResult.generated_artifacts.diagnosis_generated ? '是' : '否'}`}
      />
      <List className="list-compact" size="small" dataSource={exportResult.files} renderItem={(item) => <List.Item>{item.kind}：{item.path}</List.Item>} />
    </div>
  )
}

function buildGraphOption(diagnosis: MechanismExplanationResult | undefined) {
  const nodes = diagnosis?.graph_trace?.nodes ?? []
  const edges = diagnosis?.graph_trace?.edges ?? []

  const palette: Record<string, string> = {
    Fault: '#7dd3fc',
    Symptom: '#f8c25c',
    Cause: '#fb7185',
    Context: '#7dd3a7',
  }

  return {
    backgroundColor: 'transparent',
    tooltip: {
      backgroundColor: 'rgba(10, 16, 30, 0.94)',
      borderWidth: 0,
      textStyle: { color: '#f8fafc' },
      formatter: (params: { dataType: string; data?: { name?: string; category?: string }; value?: string }) => {
        if (params.dataType === 'edge') return params.value ?? '关联关系'
        return `${params.data?.name ?? ''}<br/>${params.data?.category ?? ''}`
      },
    },
    series: [
      {
        type: 'graph',
        layout: 'force',
        roam: true,
        draggable: false,
        force: {
          repulsion: 260,
          edgeLength: [110, 220],
          gravity: 0.08,
          friction: 0.2,
        },
        data: nodes.map((node) => ({
          id: node.id,
          name: node.label,
          category: node.node_type,
          symbolSize: node.node_type === 'Fault' ? 72 : node.node_type === 'Symptom' ? 58 : node.node_type === 'Cause' ? 52 : 46,
          itemStyle: {
            color: palette[node.node_type] ?? '#94a3b8',
            borderColor: 'rgba(255,255,255,0.18)',
            borderWidth: 1,
            shadowBlur: 18,
            shadowColor: 'rgba(9, 18, 38, 0.32)',
          },
          label: {
            show: true,
            color: '#e5eef8',
            fontSize: 11,
            backgroundColor: 'rgba(10, 18, 35, 0.72)',
            borderRadius: 12,
            padding: [6, 10],
          },
        })),
        links: edges.map((edge) => ({ source: edge.source, target: edge.target, value: edge.relation })),
        lineStyle: { color: 'rgba(148, 163, 184, 0.22)', width: 1.2, curveness: 0.1 },
        emphasis: {
          focus: 'adjacency',
          lineStyle: { width: 2, color: 'rgba(148, 163, 184, 0.5)' },
        },
      },
    ],
  }
}

function toStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.map((item) => String(item)).filter(Boolean)
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

function downloadTextReport(content: string, fileName: string) {
  downloadMarkdown(content, fileName)
}

function filterDisplayNotes(items: string[] | undefined) {
  if (!items?.length) return []
  const blockedTerms = ['论文', '补实验', '答辩', '老师', 'legacy', 'benchmark truth', 'rebuild', 'within-source', 'transfer', '实验状态', '累计实验', '建议']
  return items.filter((item) => {
    const normalized = item.toLowerCase()
    return !blockedTerms.some((term) => normalized.includes(term.toLowerCase()))
  })
}

export default Analysis
