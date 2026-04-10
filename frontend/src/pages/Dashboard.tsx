import React, { useEffect } from 'react'
import { Col, Progress, Row, Typography } from 'antd'
import { AlertOutlined, CheckCircleOutlined, DatabaseOutlined, WarningOutlined } from '@ant-design/icons'

import {
  ChartPanel,
  EmptyStateBlock,
  InsightCard,
  MetricCard,
  PageHero,
  PanelCard,
  SignalList,
  type MetricTone,
  type StatusTone,
} from '../components/ui'
import { useBhmsStore } from '../stores/useBhmsStore'

const { Text } = Typography

const Dashboard: React.FC = () => {
  const dashboard = useBhmsStore((state) => state.dashboard)
  const loadDashboard = useBhmsStore((state) => state.loadDashboard)

  useEffect(() => {
    if (!dashboard) {
      void loadDashboard()
    }
  }, [dashboard, loadDashboard])

  const stats: Array<{
    title: string
    value: number
    icon: React.ReactNode
    tone: MetricTone
    caption: string
  }> = [
    {
      title: '电池总数',
      value: dashboard?.total_batteries ?? 0,
      icon: <DatabaseOutlined />,
      tone: 'teal',
      caption: '统一管理多来源样本与真实演示数据',
    },
    {
      title: '健康电池',
      value: dashboard?.good_batteries ?? 0,
      icon: <CheckCircleOutlined />,
      tone: 'emerald',
      caption: '状态稳定，可继续跟踪寿命衰减趋势',
    },
    {
      title: '预警电池',
      value: dashboard?.warning_batteries ?? 0,
      icon: <WarningOutlined />,
      tone: 'amber',
      caption: '建议优先进入诊断流程做根因排查',
    },
    {
      title: '故障电池',
      value: dashboard?.critical_batteries ?? 0,
      icon: <AlertOutlined />,
      tone: 'rose',
      caption: '高风险样本，需重点查看异常证据链',
    },
  ]

  const trendData = dashboard?.capacity_trend ?? []
  const distributionData = dashboard?.health_distribution ?? []
  const sourceData = dashboard?.batteries_by_source ?? []
  const alertData = dashboard?.recent_alerts ?? []
  const averageScore = dashboard?.average_health_score ?? 0

  const trendOption = {
    backgroundColor: 'transparent',
    grid: { left: 24, right: 24, top: 32, bottom: 28, containLabel: true },
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(16, 35, 63, 0.92)',
      borderWidth: 0,
      textStyle: { color: '#f8fafc' },
    },
    xAxis: {
      type: 'category',
      data: trendData.map((item) => item.cycle_number),
      name: '循环次数',
      nameTextStyle: { color: '#6b7280' },
      axisLine: { lineStyle: { color: 'rgba(16, 35, 63, 0.14)' } },
      axisLabel: { color: '#6b7280' },
      axisTick: { show: false },
    },
    yAxis: {
      type: 'value',
      name: '平均容量(Ah)',
      nameTextStyle: { color: '#6b7280' },
      splitLine: { lineStyle: { color: 'rgba(16, 35, 63, 0.08)' } },
      axisLabel: { color: '#6b7280' },
    },
    series: [
      {
        name: '平均容量',
        type: 'line',
        smooth: true,
        symbol: 'circle',
        symbolSize: 8,
        data: trendData.map((item) => Number(item.avg_capacity.toFixed(3))),
        lineStyle: { width: 3, color: '#0071e3' },
        itemStyle: { color: '#0071e3', borderColor: '#ffffff', borderWidth: 2 },
        areaStyle: {
          color: {
            type: 'linear',
            x: 0,
            y: 0,
            x2: 0,
            y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(0, 113, 227, 0.22)' },
              { offset: 1, color: 'rgba(0, 113, 227, 0.02)' },
            ],
          },
        },
      },
    ],
  }

  const distributionOption = {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'item',
      backgroundColor: 'rgba(16, 35, 63, 0.92)',
      borderWidth: 0,
      textStyle: { color: '#f8fafc' },
    },
    color: ['#0071e3', '#34c759', '#ff9f0a', '#ff453a'],
    series: [
      {
        type: 'pie',
        radius: ['46%', '74%'],
        center: ['50%', '54%'],
        itemStyle: { borderColor: '#f8faf7', borderWidth: 4 },
        data: distributionData,
        label: { formatter: '{b}\n{c}', color: '#4b5563', fontWeight: 600 },
      },
    ],
  }

  return (
    <div className="page-shell dashboard-page">
      <PageHero
        title="总览"
        description="快速查看状态、趋势和告警。"
        pills={[
          { label: '收录电池', value: dashboard?.total_batteries ?? 0, tone: 'teal' },
          { label: '平均健康分', value: averageScore.toFixed(1), tone: 'slate' },
          { label: '最近告警', value: alertData.length, tone: 'amber' },
        ]}
        aside={
          <InsightCard
            label="关注"
            value={dashboard?.critical_batteries ?? 0}
            description="个高风险电池，建议优先进入诊断流程。"
            footer={
              <>
                <span>数据源覆盖</span>
                <strong>{sourceData.length} 个来源</strong>
              </>
            }
          />
        }
      />

      <Row gutter={[18, 18]}>
        {stats.map((stat) => (
          <Col xs={24} sm={12} xl={6} key={stat.title}>
            <MetricCard icon={stat.icon} title={stat.title} value={stat.value} caption={stat.caption} tone={stat.tone} />
          </Col>
        ))}
      </Row>

      <Row gutter={[18, 18]}>
        <Col xs={24} lg={15}>
          <ChartPanel
            title="容量趋势"
            option={trendOption}
            hasData={trendData.length > 0}
            height={340}
            emptyTitle="暂无趋势数据"
            emptyDescription="导入足够的循环数据后，这里会显示容量衰减趋势。"
          />
        </Col>
        <Col xs={24} lg={9}>
          <ChartPanel
            title="状态分布"
            option={distributionOption}
            hasData={distributionData.length > 0}
            height={340}
            emptyTitle="暂无分布数据"
            emptyDescription="系统生成健康状态汇总后，会自动渲染分布图。"
          />
        </Col>
      </Row>

      <Row gutter={[18, 18]}>
        <Col xs={24} lg={9}>
          <PanelCard title="来源分布">
            {sourceData.length ? (
              <div className="source-stack">
                {sourceData.map((item) => {
                  const percent = totalBatteries(dashboard?.total_batteries ?? 0, item.battery_count)
                  return (
                    <div key={item.source} className="source-stack__item">
                      <div className="source-stack__head">
                        <div>
                          <Text strong>{item.source.toUpperCase()}</Text>
                          <Text className="source-stack__subtext">{item.battery_count} 节电池</Text>
                        </div>
                        <Text>{percent}%</Text>
                      </div>
                      <Progress percent={percent} showInfo={false} strokeColor="#0071e3" trailColor="rgba(29, 29, 31, 0.08)" />
                    </div>
                  )
                })}
              </div>
            ) : (
              <EmptyStateBlock compact title="暂无来源分布" description="当前还没有可展示的来源聚合信息。" className="panel-empty-state" />
            )}
          </PanelCard>
        </Col>
        <Col xs={24} lg={15}>
          <PanelCard title="告警" extra={<Text className="panel-extra">平均健康分 {averageScore.toFixed(1)}</Text>}>
            {alertData.length ? (
              <SignalList
                items={alertData.map((item) => ({
                  key: `${item.battery_id}-${item.symptom}`,
                  title: <Text strong>{item.battery_id}</Text>,
                  tag: item.symptom,
                  description: item.description,
                  tone: mapSeverityTone(item.severity),
                }))}
              />
            ) : (
              <EmptyStateBlock compact title="暂无异常事件" description="当前没有触发新的告警信号。" className="panel-empty-state" />
            )}
          </PanelCard>
        </Col>
      </Row>
    </div>
  )
}

function totalBatteries(total: number, count: number) {
  if (!total) return 0
  return Number(((count / total) * 100).toFixed(1))
}

function mapSeverityTone(severity: string): StatusTone {
  if (severity === 'critical' || severity === 'high') return 'critical'
  if (severity === 'medium' || severity === 'warning') return 'warning'
  return 'info'
}

export default Dashboard
