import React from 'react'
import { Typography } from 'antd'

import PanelCard from './PanelCard'

const { Text } = Typography

export type MetricTone = 'teal' | 'emerald' | 'amber' | 'rose'

interface MetricCardProps {
  icon: React.ReactNode
  title: string
  value: React.ReactNode
  caption: React.ReactNode
  tone: MetricTone
}

const MetricCard: React.FC<MetricCardProps> = ({ icon, title, value, caption, tone }) => {
  return (
    <PanelCard className={`metric-card metric-card--${tone}`}>
      <div className="metric-card__icon">{icon}</div>
      <Text className="metric-card__label">{title}</Text>
      <div className="metric-card__value">{value}</div>
      <Text className="metric-card__caption">{caption}</Text>
    </PanelCard>
  )
}

export default MetricCard
