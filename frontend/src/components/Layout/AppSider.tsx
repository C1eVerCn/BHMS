import React from 'react'
import { Drawer, Grid, Layout, Menu, Typography } from 'antd'
import { useLocation, useNavigate } from 'react-router-dom'
import {
  AlertOutlined,
  ApartmentOutlined,
  CloudUploadOutlined,
  DashboardOutlined,
  DatabaseOutlined,
  LineChartOutlined,
} from '@ant-design/icons'

import { useBhmsStore } from '../../stores/useBhmsStore'
import { compactBatteryId } from '../../utils/display'

const { Sider } = Layout
const { Text, Title } = Typography
const { useBreakpoint } = Grid

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: '总览' },
  { key: '/batteries', icon: <DatabaseOutlined />, label: '电池' },
  { key: '/prediction', icon: <LineChartOutlined />, label: '预测' },
  { key: '/diagnosis', icon: <AlertOutlined />, label: '诊断' },
  { key: '/analysis', icon: <ApartmentOutlined />, label: '分析' },
  { key: '/upload', icon: <CloudUploadOutlined />, label: '导入' },
]

interface AppSiderProps {
  mobileOpen?: boolean
  onCloseMobile?: () => void
}

const AppSider: React.FC<AppSiderProps> = ({ mobileOpen = false, onCloseMobile }) => {
  const navigate = useNavigate()
  const location = useLocation()
  const screens = useBreakpoint()
  const dashboard = useBhmsStore((state) => state.dashboard)
  const selectedBatteryId = useBhmsStore((state) => state.selectedBatteryId)

  const isDesktop = Boolean(screens.lg)
  const totalBatteries = dashboard?.total_batteries ?? 0
  const averageHealth = dashboard?.average_health_score ?? 0
  const criticalCount = dashboard?.critical_batteries ?? 0

  const navContent = (
    <div className="app-sider__inner">
      <div className="sider-overview">
        <Text className="sider-overview__eyebrow">Workspace</Text>
        <Title level={5}>BHMS</Title>
        <Text className="sider-overview__description">状态、预测、诊断统一查看。</Text>
        <div className="sider-overview__stats">
          <div>
            <span>电池</span>
            <strong>{totalBatteries}</strong>
          </div>
          <div>
            <span>健康</span>
            <strong>{averageHealth.toFixed(1)}</strong>
          </div>
          <div>
            <span>风险</span>
            <strong>{criticalCount}</strong>
          </div>
          <div>
            <span>焦点</span>
            <strong>{compactBatteryId(selectedBatteryId)}</strong>
          </div>
        </div>
      </div>

      <Menu
        mode="inline"
        selectedKeys={[location.pathname]}
        className="app-menu"
        style={{ borderRight: 0 }}
        items={menuItems}
        onClick={({ key }) => {
          navigate(key)
          onCloseMobile?.()
        }}
      />

      <div className="sider-footnote">
        <Text>先看总览，再看预测和诊断。</Text>
      </div>
    </div>
  )

  if (!isDesktop) {
    return (
      <Drawer
        placement="left"
        width={320}
        open={mobileOpen}
        onClose={onCloseMobile}
        closable={false}
        rootClassName="app-nav-drawer"
        bodyStyle={{ padding: 0 }}
      >
        {navContent}
      </Drawer>
    )
  }

  return (
    <Sider width={296} className="app-sider" trigger={null}>
      {navContent}
    </Sider>
  )
}

export default AppSider
