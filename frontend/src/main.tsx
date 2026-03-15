import React from 'react'
import ReactDOM from 'react-dom/client'
import { ConfigProvider } from 'antd'
import zhCN from 'antd/locale/zh_CN'

import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ConfigProvider
      locale={zhCN}
      theme={{
        token: {
          colorPrimary: '#0071e3',
          colorInfo: '#0071e3',
          colorSuccess: '#34c759',
          colorWarning: '#ff9f0a',
          colorError: '#ff453a',
          colorText: '#1d1d1f',
          colorTextSecondary: '#6e6e73',
          colorBgBase: '#f5f5f7',
          colorBorderSecondary: 'rgba(29, 29, 31, 0.08)',
          borderRadius: 16,
          borderRadiusLG: 22,
          controlHeight: 42,
          fontFamily:
            "'SF Pro Display', 'SF Pro Text', 'PingFang SC', 'Helvetica Neue', 'Hiragino Sans GB', sans-serif",
        },
      }}
    >
      <App />
    </ConfigProvider>
  </React.StrictMode>,
)
