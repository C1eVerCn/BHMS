import React from 'react'
import { FileTextOutlined } from '@ant-design/icons'
import { Typography } from 'antd'

const { Text, Title } = Typography

interface EmptyStateBlockProps {
  title: React.ReactNode
  description?: React.ReactNode
  icon?: React.ReactNode
  compact?: boolean
  className?: string
}

const EmptyStateBlock: React.FC<EmptyStateBlockProps> = ({
  title,
  description,
  icon = <FileTextOutlined className="empty-state-block__icon" />,
  compact = false,
  className,
}) => {
  return (
    <div className={['empty-state-block', compact ? 'empty-state-block--compact' : '', className].filter(Boolean).join(' ')}>
      <div className="empty-state-block__visual">{icon}</div>
      <Title level={compact ? 5 : 4}>{title}</Title>
      {description ? <Text className="empty-state-block__description">{description}</Text> : null}
    </div>
  )
}

export default EmptyStateBlock
