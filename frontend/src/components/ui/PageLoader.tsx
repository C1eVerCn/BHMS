import React from 'react'
import { Spin, Typography } from 'antd'

const { Text } = Typography

interface PageLoaderProps {
  label?: string
}

const PageLoader: React.FC<PageLoaderProps> = ({ label = '页面加载中...' }) => {
  return (
    <div className="route-loader">
      <div className="route-loader__panel">
        <Spin size="large" />
        <Text className="route-loader__label">{label}</Text>
      </div>
    </div>
  )
}

export default PageLoader
