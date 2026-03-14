import React, { useEffect } from 'react'
import { Alert, Layout, Spin } from 'antd'
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom'

import AppHeader from './components/Layout/AppHeader'
import AppSider from './components/Layout/AppSider'
import './App.css'
import Dashboard from './pages/Dashboard'
import BatteryList from './pages/BatteryList'
import Prediction from './pages/Prediction'
import Diagnosis from './pages/Diagnosis'
import DataUpload from './pages/DataUpload'
import { useBhmsStore } from './stores/useBhmsStore'

const { Content } = Layout

const App: React.FC = () => {
  const init = useBhmsStore((state) => state.init)
  const loading = useBhmsStore((state) => state.loading)
  const error = useBhmsStore((state) => state.error)
  const clearError = useBhmsStore((state) => state.clearError)

  useEffect(() => {
    void init()
  }, [init])

  return (
    <Router>
      <Layout className="app-shell">
        <AppHeader />
        <Layout>
          <AppSider />
          <Layout className="content-shell">
            <Content className="content-card">
              {error && (
                <Alert
                  message="接口请求异常"
                  description={error}
                  type="error"
                  closable
                  onClose={clearError}
                  style={{ marginBottom: 16 }}
                />
              )}
              <Spin spinning={loading} tip="正在加载 BHMS 数据...">
                <Routes>
                  <Route path="/" element={<Dashboard />} />
                  <Route path="/batteries" element={<BatteryList />} />
                  <Route path="/prediction" element={<Prediction />} />
                  <Route path="/diagnosis" element={<Diagnosis />} />
                  <Route path="/upload" element={<DataUpload />} />
                </Routes>
              </Spin>
            </Content>
          </Layout>
        </Layout>
      </Layout>
    </Router>
  )
}

export default App
