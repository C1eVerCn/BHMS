import React from 'react'
import { Space, Typography } from 'antd'

const { Text } = Typography

export interface StructuredDataItem {
  label: string
  value: React.ReactNode
}

interface StructuredDataListProps {
  items: StructuredDataItem[]
  compact?: boolean
  className?: string
}

const StructuredDataList: React.FC<StructuredDataListProps> = ({ items, compact = false, className }) => {
  return (
    <div className={['structured-data-list', compact ? 'structured-data-list--compact' : '', className].filter(Boolean).join(' ')}>
      {items.map((item) => (
        <div className="structured-data-list__row" key={item.label}>
          <Text className="structured-data-list__label">{item.label}</Text>
          <div className="structured-data-list__value">
            {Array.isArray(item.value) ? (
              <Space wrap size={[8, 8]}>
                {item.value.map((entry, index) => (
                  <span key={`${item.label}-${index}`} className="structured-data-list__chip">
                    {entry}
                  </span>
                ))}
              </Space>
            ) : (
              item.value
            )}
          </div>
        </div>
      ))}
    </div>
  )
}

export default StructuredDataList
