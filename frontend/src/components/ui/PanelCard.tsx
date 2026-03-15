import React from 'react'
import { Card, type CardProps } from 'antd'

const PanelCard: React.FC<CardProps> = ({ className, ...props }) => {
  return <Card className={['panel-card', className].filter(Boolean).join(' ')} {...props} />
}

export default PanelCard
