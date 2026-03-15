import React from 'react'
import { Tag } from 'antd'

export type StatusTone = 'good' | 'warning' | 'critical' | 'neutral' | 'info'

interface StatusTagProps {
  tone: StatusTone
  children: React.ReactNode
  className?: string
}

const StatusTag: React.FC<StatusTagProps> = ({ tone, children, className }) => {
  return <Tag className={['status-tag', `status-tag--${tone}`, className].filter(Boolean).join(' ')}>{children}</Tag>
}

export default StatusTag
