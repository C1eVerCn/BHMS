import React, { Suspense, lazy, useEffect, useState } from 'react'
import { Alert, FloatButton, Layout, Spin } from 'antd'
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom'
import { ArrowUpOutlined } from '@ant-design/icons'

import AppHeader from './components/Layout/AppHeader'
import AppSider from './components/Layout/AppSider'
import { PageLoader } from './components/ui'
import './App.css'
import { useBhmsStore } from './stores/useBhmsStore'

const { Content } = Layout
const Dashboard = lazy(() => import('./pages/Dashboard'))
const BatteryList = lazy(() => import('./pages/BatteryList'))
const Prediction = lazy(() => import('./pages/Prediction'))
const Diagnosis = lazy(() => import('./pages/Diagnosis'))
const DataUpload = lazy(() => import('./pages/DataUpload'))
const Analysis = lazy(() => import('./pages/Analysis'))

const App: React.FC = () => {
  const init = useBhmsStore((state) => state.init)
  const loading = useBhmsStore((state) => state.loading)
  const error = useBhmsStore((state) => state.error)
  const clearError = useBhmsStore((state) => state.clearError)
  const [mobileNavOpen, setMobileNavOpen] = useState(false)

  useEffect(() => {
    void init()
  }, [init])

  return (
    <Router
      future={{
        v7_startTransition: true,
        v7_relativeSplatPath: true,
      }}
    >
      <Layout className="app-shell">
        <AppHeader onOpenNavigation={() => setMobileNavOpen(true)} />
        <Layout className="workspace-shell">
          <AppSider mobileOpen={mobileNavOpen} onCloseMobile={() => setMobileNavOpen(false)} />
          <Layout className="content-shell">
            <Content className="content-area">
              {error && (
                <Alert
                  className="global-alert"
                  message="接口请求异常"
                  description={error}
                  type="error"
                  closable
                  onClose={clearError}
                />
              )}

              <div className="route-stage">
                <Spin spinning={loading} tip="正在加载 BHMS 数据..." size="large">
                  <Suspense fallback={<PageLoader label="正在切换工作台视图..." />}>
                    <Routes>
                      <Route path="/" element={<Dashboard />} />
                      <Route path="/batteries" element={<BatteryList />} />
                      <Route path="/prediction" element={<Prediction />} />
                      <Route path="/diagnosis" element={<Diagnosis />} />
                      <Route path="/upload" element={<DataUpload />} />
                      <Route path="/analysis" element={<Analysis />} />
                    </Routes>
                  </Suspense>
                </Spin>
              </div>
            </Content>
          </Layout>
        </Layout>
        <FloatButton.BackTop icon={<ArrowUpOutlined />} />
      </Layout>
    </Router>
  )
}

export default App
