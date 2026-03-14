import React from 'react'
import { Layout, Menu } from 'antd'
import { useLocation, useNavigate } from 'react-router-dom'
import { AlertOutlined, CloudUploadOutlined, DashboardOutlined, DatabaseOutlined, LineChartOutlined } from '@ant-design/icons'

const { Sider } = Layout

const menuItems = [
  { key: '/', icon: <DashboardOutlined />, label: '数据概览' },
  { key: '/batteries', icon: <DatabaseOutlined />, label: '电池管理' },
  { key: '/prediction', icon: <LineChartOutlined />, label: 'RUL 预测' },
  { key: '/diagnosis', icon: <AlertOutlined />, label: '故障诊断' },
  { key: '/upload', icon: <CloudUploadOutlined />, label: '数据导入' },
]

const AppSider: React.FC = () => {
  const navigate = useNavigate()
  const location = useLocation()

  return (
    <Sider breakpoint="lg" collapsedWidth="0" width={220} className="app-sider">
      <Menu
        mode="inline"
        selectedKeys={[location.pathname]}
        style={{ height: '100%', borderRight: 0 }}
        items={menuItems}
        onClick={({ key }) => navigate(key)}
      />
    </Sider>
  )
}

export default AppSider
