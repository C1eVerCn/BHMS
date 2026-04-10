import React, { useEffect, useMemo, useState } from 'react'
import { Button, Input, Select, Space, Table, Typography } from 'antd'
import { EyeOutlined, LineChartOutlined, SearchOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

import { InsightCard, PageHero, PanelCard, StatusTag, type StatusTone } from '../components/ui'
import { useBhmsStore } from '../stores/useBhmsStore'
import type { Battery, BatteryStatus } from '../types/domain'
import { formatSourceLabel } from '../utils/display'

const { Text } = Typography

const statusMap: Record<BatteryStatus, { tone: StatusTone; text: string }> = {
  good: { tone: 'good', text: '健康' },
  warning: { tone: 'warning', text: '预警' },
  critical: { tone: 'critical', text: '故障' },
  unknown: { tone: 'neutral', text: '未知' },
}

const BatteryList: React.FC = () => {
  const navigate = useNavigate()
  const batteryPageItems = useBhmsStore((state) => state.batteryPageItems)
  const batteryOptions = useBhmsStore((state) => state.batteryOptions)
  const pagination = useBhmsStore((state) => state.pagination)
  const loadBatteries = useBhmsStore((state) => state.loadBatteries)
  const selectBattery = useBhmsStore((state) => state.selectBattery)
  const loadBatteryContext = useBhmsStore((state) => state.loadBatteryContext)
  const [searchText, setSearchText] = useState('')
  const [usageFilter, setUsageFilter] = useState<'all' | 'display' | 'training'>('all')

  useEffect(() => {
    if (!batteryPageItems.length) {
      void loadBatteries(pagination.page, pagination.pageSize)
    }
  }, [batteryPageItems.length, loadBatteries, pagination.page, pagination.pageSize])

  const filteredData = useMemo(
    () =>
      batteryPageItems.filter((item) => {
        const keyword = searchText.toLowerCase()
        const matchesUsage =
          usageFilter === 'all' ? true : usageFilter === 'training' ? Boolean(item.include_in_training) : !item.include_in_training
        return matchesUsage && (item.battery_id.toLowerCase().includes(keyword) || (item.chemistry ?? '').toLowerCase().includes(keyword))
      }),
    [batteryPageItems, searchText, usageFilter],
  )

  const quickStats = useMemo(
    () => ({
      critical: batteryOptions.filter((item) => item.status === 'critical').length,
      training: batteryOptions.filter((item) => item.include_in_training).length,
      display: batteryOptions.filter((item) => !item.include_in_training).length,
      bestHealth: batteryPageItems.reduce((best, item) => Math.max(best, item.health_score), 0),
    }),
    [batteryOptions, batteryPageItems],
  )

  const columns = [
    {
      title: '电池 ID',
      dataIndex: 'battery_id',
      key: 'battery_id',
      width: 340,
      render: (value: string, record: Battery) => (
        <button
          type="button"
          className="table-link-button"
          onClick={() => {
            selectBattery(record.battery_id)
            void loadBatteryContext(record.battery_id)
            navigate('/diagnosis')
          }}
        >
          {value}
        </button>
      ),
    },
    {
      title: '来源',
      dataIndex: 'source',
      key: 'source',
      width: 140,
      ellipsis: true,
      render: (value: string) => (
        <Text strong className="table-nowrap">
          {formatSourceLabel(value)}
        </Text>
      ),
    },
    {
      title: '类型',
      dataIndex: 'chemistry',
      key: 'chemistry',
      width: 120,
      render: (value: string | null) => value ?? '--',
    },
    {
      title: '用途',
      key: 'usage',
      width: 120,
      render: (_: unknown, record: Battery) =>
        record.include_in_training ? <StatusTag tone="warning">训练数据</StatusTag> : <StatusTag tone="info">展示样本</StatusTag>,
    },
    {
      title: '循环次数',
      dataIndex: 'cycle_count',
      key: 'cycle_count',
      width: 120,
      sorter: (a: Battery, b: Battery) => a.cycle_count - b.cycle_count,
    },
    {
      title: '当前容量(Ah)',
      dataIndex: 'latest_capacity',
      key: 'latest_capacity',
      width: 150,
      render: (value: number | null) => (value ?? 0).toFixed(3),
    },
    {
      title: '健康分',
      dataIndex: 'health_score',
      key: 'health_score',
      width: 120,
      sorter: (a: Battery, b: Battery) => a.health_score - b.health_score,
      render: (value: number) => <span className="health-score-pill">{value.toFixed(1)}</span>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      width: 110,
      render: (value: BatteryStatus) => <StatusTag tone={statusMap[value]?.tone}>{statusMap[value]?.text ?? value}</StatusTag>,
    },
    {
      title: '操作',
      key: 'action',
      width: 240,
      render: (_: unknown, record: Battery) => (
        <Space>
          <Button
            icon={<EyeOutlined />}
            onClick={() => {
              selectBattery(record.battery_id)
              void loadBatteryContext(record.battery_id)
              navigate('/diagnosis')
            }}
          >
            诊断
          </Button>
          <Button
            type="primary"
            icon={<LineChartOutlined />}
            onClick={() => {
              selectBattery(record.battery_id)
              void loadBatteryContext(record.battery_id)
              navigate('/prediction')
            }}
          >
            预测
          </Button>
        </Space>
      ),
    },
  ]

  return (
    <div className="page-shell">
      <PageHero
        title="电池"
        description="浏览、筛选并定位目标电池。"
        pills={[
          { label: '展示样本', value: quickStats.display, tone: 'teal' },
          { label: '高风险样本', value: quickStats.critical, tone: 'rose' },
          { label: '训练数据', value: quickStats.training, tone: 'amber' },
        ]}
        aside={
          <InsightCard
            compact
            label="概况"
            value={pagination.total}
            description={`条可浏览记录，当前页最高健康分 ${quickStats.bestHealth.toFixed(1)}。`}
          />
        }
      />

      <PanelCard>
        <div className="toolbar-row">
          <div>
            <Text className="toolbar-row__label">快速检索</Text>
            <Input
              placeholder="搜索电池 ID 或电池类型"
              prefix={<SearchOutlined />}
              value={searchText}
              onChange={(event) => setSearchText(event.target.value)}
              className="filter-input"
            />
          </div>
          <Select<'all' | 'display' | 'training'>
            value={usageFilter}
            onChange={setUsageFilter}
            options={[
              { label: '全部', value: 'all' },
              { label: '展示样本', value: 'display' },
              { label: '训练数据', value: 'training' },
            ]}
            style={{ minWidth: 168 }}
          />
          <div className="toolbar-row__summary">
            <StatusTag tone="info">共 {pagination.total} 条记录</StatusTag>
            <StatusTag tone="neutral">筛选后 {filteredData.length} 条</StatusTag>
          </div>
        </div>

        <Table
          rowKey="battery_id"
          className="data-table"
          columns={columns}
          dataSource={filteredData}
          scroll={{ x: 1100 }}
          pagination={{
            current: pagination.page,
            pageSize: pagination.pageSize,
            total: pagination.total,
            showSizeChanger: true,
            pageSizeOptions: [20, 40, 80],
            onChange: (page, pageSize) => {
              void loadBatteries(page, pageSize)
            },
          }}
        />
      </PanelCard>
    </div>
  )
}

export default BatteryList
