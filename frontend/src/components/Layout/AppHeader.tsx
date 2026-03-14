import React from 'react'
import { Avatar, Badge, Layout, Space, Tag, Typography } from 'antd'
import { BellOutlined, DatabaseOutlined, UserOutlined } from '@ant-design/icons'

import { useBhmsStore } from '../../stores/useBhmsStore'

const { Header } = Layout
const { Title, Text } = Typography

const AppHeader: React.FC = () => {
  const batteries = useBhmsStore((state) => state.batteries)
  const dashboard = useBhmsStore((state) => state.dashboard)

  return (
    <Header className="app-header">
      <Space align="center" size="middle">
        <div className="brand-mark">
          <DatabaseOutlined />
        </div>
        <div>
          <Title level={4} style={{ margin: 0 }}>
            BHMS 锂电池健康管理系统
          </Title>
          <Text type="secondary">毕业设计 MVP · 真实数据闭环演示</Text>
        </div>
      </Space>

      <Space size="large">
        <Tag color="processing">电池数 {dashboard?.total_batteries ?? batteries.length}</Tag>
        <Badge count={dashboard?.recent_alerts.length ?? 0} size="small">
          <BellOutlined style={{ fontSize: 18 }} />
        </Badge>
        <Space>
          <Avatar icon={<UserOutlined />} />
          <span>答辩演示账号</span>
        </Space>
      </Space>
    </Header>
  )
}

export default AppHeader
