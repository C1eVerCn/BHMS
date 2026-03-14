import React, { useEffect } from 'react'
import { Badge, Card, Col, Empty, List, Row, Space, Statistic, Typography } from 'antd'
import { AlertOutlined, CheckCircleOutlined, DatabaseOutlined, WarningOutlined } from '@ant-design/icons'
import ReactECharts from 'echarts-for-react'

import { useBhmsStore } from '../stores/useBhmsStore'

const { Title, Text } = Typography

const Dashboard: React.FC = () => {
  const dashboard = useBhmsStore((state) => state.dashboard)
  const loadDashboard = useBhmsStore((state) => state.loadDashboard)

  useEffect(() => {
    if (!dashboard) {
      void loadDashboard()
    }
  }, [dashboard, loadDashboard])

  const stats = [
    { title: '电池总数', value: dashboard?.total_batteries ?? 0, icon: <DatabaseOutlined />, color: '#1677ff' },
    { title: '健康电池', value: dashboard?.good_batteries ?? 0, icon: <CheckCircleOutlined />, color: '#52c41a' },
    { title: '预警电池', value: dashboard?.warning_batteries ?? 0, icon: <WarningOutlined />, color: '#faad14' },
    { title: '故障电池', value: dashboard?.critical_batteries ?? 0, icon: <AlertOutlined />, color: '#ff4d4f' },
  ]

  const trendOption = {
    tooltip: { trigger: 'axis' },
    xAxis: {
      type: 'category',
      data: (dashboard?.capacity_trend ?? []).map((item) => item.cycle_number),
      name: '循环次数',
    },
    yAxis: { type: 'value', name: '平均容量(Ah)' },
    series: [
      {
        name: '平均容量',
        type: 'line',
        smooth: true,
        data: (dashboard?.capacity_trend ?? []).map((item) => item.avg_capacity.toFixed(3)),
        areaStyle: { opacity: 0.12 },
        lineStyle: { width: 3, color: '#1677ff' },
      },
    ],
  }

  const distributionOption = {
    tooltip: { trigger: 'item' },
    series: [
      {
        type: 'pie',
        radius: ['40%', '68%'],
        data: dashboard?.health_distribution ?? [],
        label: { formatter: '{b}: {c}' },
      },
    ],
  }

  return (
    <div>
      <Title level={2} className="page-title">
        数据概览
      </Title>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        {stats.map((stat) => (
          <Col xs={24} sm={12} xl={6} key={stat.title}>
            <Card className="stat-card">
              <Statistic
                title={stat.title}
                value={stat.value}
                prefix={React.cloneElement(stat.icon as React.ReactElement, { style: { color: stat.color } })}
              />
            </Card>
          </Col>
        ))}
      </Row>

      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} lg={14}>
          <Card title="NASA 导入样本容量趋势" className="chart-container">
            {dashboard?.capacity_trend.length ? (
              <ReactECharts option={trendOption} style={{ height: 320 }} />
            ) : (
              <Empty description="暂无趋势数据" />
            )}
          </Card>
        </Col>
        <Col xs={24} lg={10}>
          <Card title="健康状态分布" className="chart-container">
            {dashboard?.health_distribution.length ? (
              <ReactECharts option={distributionOption} style={{ height: 320 }} />
            ) : (
              <Empty description="暂无分布数据" />
            )}
          </Card>
        </Col>
      </Row>

      <Row gutter={[16, 16]}>
        <Col span={24}>
          <Card title="最近告警" extra={<Text type="secondary">平均健康分 {dashboard?.average_health_score.toFixed(1) ?? '0.0'}</Text>}>
            {dashboard?.recent_alerts.length ? (
              <List
                dataSource={dashboard.recent_alerts}
                renderItem={(item) => (
                  <List.Item>
                    <List.Item.Meta
                      title={
                        <Space>
                          <Text strong>{item.battery_id}</Text>
                          <Badge color={item.severity === 'high' ? 'red' : item.severity === 'medium' ? 'orange' : 'green'} text={item.symptom} />
                        </Space>
                      }
                      description={item.description}
                    />
                  </List.Item>
                )}
              />
            ) : (
              <Empty description="暂无异常事件" />
            )}
          </Card>
        </Col>
      </Row>
    </div>
  )
}

export default Dashboard
