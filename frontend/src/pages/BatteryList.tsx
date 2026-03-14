import React, { useEffect, useMemo, useState } from 'react'
import { Button, Card, Input, Space, Table, Tag, Typography } from 'antd'
import { EyeOutlined, LineChartOutlined, SearchOutlined } from '@ant-design/icons'
import { useNavigate } from 'react-router-dom'

import type { Battery, BatteryStatus } from '../types/domain'
import { useBhmsStore } from '../stores/useBhmsStore'

const { Title } = Typography

const statusMap: Record<BatteryStatus, { color: string; text: string }> = {
  good: { color: 'success', text: '健康' },
  warning: { color: 'warning', text: '预警' },
  critical: { color: 'error', text: '故障' },
  unknown: { color: 'default', text: '未知' },
}

const BatteryList: React.FC = () => {
  const navigate = useNavigate()
  const batteries = useBhmsStore((state) => state.batteries)
  const pagination = useBhmsStore((state) => state.pagination)
  const loadBatteries = useBhmsStore((state) => state.loadBatteries)
  const selectBattery = useBhmsStore((state) => state.selectBattery)
  const loadBatteryContext = useBhmsStore((state) => state.loadBatteryContext)
  const [searchText, setSearchText] = useState('')

  useEffect(() => {
    if (!batteries.length) {
      void loadBatteries(pagination.page, pagination.pageSize)
    }
  }, [batteries.length, loadBatteries, pagination.page, pagination.pageSize])

  const filteredData = useMemo(
    () =>
      batteries.filter((item) => {
        const keyword = searchText.toLowerCase()
        return item.battery_id.toLowerCase().includes(keyword) || (item.chemistry ?? '').toLowerCase().includes(keyword)
      }),
    [batteries, searchText],
  )

  const columns = [
    {
      title: '电池 ID',
      dataIndex: 'battery_id',
      key: 'battery_id',
      render: (value: string) => <a>{value}</a>,
    },
    {
      title: '来源',
      dataIndex: 'source',
      key: 'source',
    },
    {
      title: '类型',
      dataIndex: 'chemistry',
      key: 'chemistry',
      render: (value: string | null) => value ?? '--',
    },
    {
      title: '循环次数',
      dataIndex: 'cycle_count',
      key: 'cycle_count',
      sorter: (a: Battery, b: Battery) => a.cycle_count - b.cycle_count,
    },
    {
      title: '当前容量(Ah)',
      dataIndex: 'latest_capacity',
      key: 'latest_capacity',
      render: (value: number | null) => (value ?? 0).toFixed(3),
    },
    {
      title: '健康分',
      dataIndex: 'health_score',
      key: 'health_score',
      sorter: (a: Battery, b: Battery) => a.health_score - b.health_score,
      render: (value: number) => <span style={{ fontWeight: 700 }}>{value.toFixed(1)}</span>,
    },
    {
      title: '状态',
      dataIndex: 'status',
      key: 'status',
      render: (value: BatteryStatus) => <Tag color={statusMap[value]?.color}>{statusMap[value]?.text ?? value}</Tag>,
    },
    {
      title: '操作',
      key: 'action',
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
    <div>
      <Title level={2} className="page-title">
        电池管理
      </Title>

      <Card>
        <div style={{ display: 'flex', justifyContent: 'space-between', gap: 16, marginBottom: 16, flexWrap: 'wrap' }}>
          <Input
            placeholder="搜索电池 ID 或电池类型"
            prefix={<SearchOutlined />}
            value={searchText}
            onChange={(event) => setSearchText(event.target.value)}
            style={{ width: 320 }}
          />
          <Tag color="blue">共 {pagination.total} 条记录</Tag>
        </div>

        <Table
          rowKey="battery_id"
          columns={columns}
          dataSource={filteredData}
          pagination={{
            current: pagination.page,
            pageSize: pagination.pageSize,
            total: pagination.total,
            showSizeChanger: true,
            onChange: (page, pageSize) => {
              void loadBatteries(page, pageSize)
            },
          }}
        />
      </Card>
    </div>
  )
}

export default BatteryList
