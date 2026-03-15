import React from 'react'
import { Card, Typography } from 'antd'

const { Paragraph, Text } = Typography

interface InsightCardProps {
  label: string
  value: React.ReactNode
  description: React.ReactNode
  footer?: React.ReactNode
  compact?: boolean
  className?: string
}

const InsightCard: React.FC<InsightCardProps> = ({ label, value, description, footer, compact = false, className }) => {
  return (
    <Card className={['hero-insight-card', compact ? 'hero-insight-card--compact' : '', className].filter(Boolean).join(' ')}>
      <Text className="hero-insight-card__label">{label}</Text>
      <div className="hero-insight-card__value">{value}</div>
      <Paragraph className="hero-insight-card__text">{description}</Paragraph>
      {footer ? <div className="hero-insight-card__footer">{footer}</div> : null}
    </Card>
  )
}

export default InsightCard
