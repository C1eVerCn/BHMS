import React from 'react'
import { List, Space } from 'antd'

import StatusTag, { type StatusTone } from './StatusTag'

interface SignalListItem {
  key: React.Key
  title: React.ReactNode
  description?: React.ReactNode
  tone: StatusTone
  tag?: React.ReactNode
  meta?: React.ReactNode
}

interface SignalListProps {
  items: SignalListItem[]
  className?: string
}

const SignalList: React.FC<SignalListProps> = ({ items, className }) => {
  return (
    <List
      className={['signal-list', className].filter(Boolean).join(' ')}
      dataSource={items}
      renderItem={(item) => (
        <List.Item key={item.key}>
          <div className="signal-list__item">
            <div className={`severity-dot severity-dot--${item.tone}`} />
            <div className="signal-list__content">
              <Space size={8} wrap>
                {item.title}
                {item.tag ? <StatusTag tone={item.tone}>{item.tag}</StatusTag> : null}
                {item.meta}
              </Space>
              {item.description ? <div className="signal-list__description">{item.description}</div> : null}
            </div>
          </div>
        </List.Item>
      )}
    />
  )
}

export default SignalList
