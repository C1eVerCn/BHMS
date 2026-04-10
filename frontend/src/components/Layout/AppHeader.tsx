import React from 'react'
import { Avatar, Badge, Button, Grid, Layout, Space, Typography } from 'antd'
import { BellOutlined, DatabaseOutlined, MenuOutlined, UserOutlined } from '@ant-design/icons'
import { useLocation } from 'react-router-dom'

import { useBhmsStore } from '../../stores/useBhmsStore'
import { compactBatteryId } from '../../utils/display'

const { Header } = Layout
const { Text, Title } = Typography
const { useBreakpoint } = Grid

const routeLabels: Record<string, string> = {
  '/': '总览',
  '/batteries': '电池',
  '/prediction': '预测',
  '/diagnosis': '诊断',
  '/analysis': '分析',
  '/upload': '导入',
}

interface AppHeaderProps {
  onOpenNavigation?: () => void
}

const AppHeader: React.FC<AppHeaderProps> = ({ onOpenNavigation }) => {
  const screens = useBreakpoint()
  const location = useLocation()
  const batteryOptions = useBhmsStore((state) => state.batteryOptions)
  const dashboard = useBhmsStore((state) => state.dashboard)
  const selectedBatteryId = useBhmsStore((state) => state.selectedBatteryId)

  const isMobile = !screens.lg
  const currentSection = routeLabels[location.pathname] ?? '控制台'
  const focusBattery = compactBatteryId(selectedBatteryId ?? batteryOptions[0]?.battery_id)
  const averageScore = dashboard?.average_health_score ?? 0
  const alertCount = dashboard?.recent_alerts.length ?? 0

  return (
    <Header className="app-header">
      <div className="app-header__left">
        <Space align="center" size={14}>
          {isMobile && (
            <Button
              type="text"
              shape="circle"
              icon={<MenuOutlined />}
              className="mobile-nav-trigger"
              aria-label="打开导航"
              onClick={onOpenNavigation}
            />
          )}
          <div className="brand-mark">
            <DatabaseOutlined />
          </div>
          <div className="brand-copy">
            <Text className="brand-copy__eyebrow">BHMS</Text>
            <Title level={4}>BHMS</Title>
            <Text className="brand-copy__subline">{currentSection}</Text>
          </div>
        </Space>
      </div>

      <div className="app-header__right">
        <div className="header-chip header-chip--accent">
          <span>平均健康</span>
          <strong>{averageScore.toFixed(1)}</strong>
        </div>
        <div className="header-chip">
          <span>当前电池</span>
          <strong>{focusBattery}</strong>
        </div>
        <div className="header-chip header-chip--icon">
          <Badge count={alertCount} size="small">
            <BellOutlined className="header-chip__icon" />
          </Badge>
          <div>
            <span>告警</span>
            <strong>{alertCount}</strong>
          </div>
        </div>
        <div className="header-profile">
          <Avatar icon={<UserOutlined />} />
          <div>
            <span>环境</span>
            <strong>演示模式</strong>
          </div>
        </div>
      </div>
    </Header>
  )
}

export default AppHeader
