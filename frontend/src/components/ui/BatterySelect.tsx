import React, { useMemo } from 'react'
import { Select, Typography } from 'antd'

import type { BatteryOption } from '../../types/domain'
import { formatSourceLabel } from '../../utils/display'

const { Text } = Typography

interface BatterySelectProps {
  value?: string
  options: BatteryOption[]
  onChange?: (value: string) => void
  placeholder?: string
  style?: React.CSSProperties
  disabled?: boolean
  listHeight?: number
}

const BatterySelect: React.FC<BatterySelectProps> = ({
  value,
  options,
  onChange,
  placeholder = '选择电池',
  style,
  disabled = false,
  listHeight = 420,
}) => {
  const groupedOptions = useMemo(() => {
    const displaySamples = options.filter((item) => !item.include_in_training)
    const trainingSamples = options.filter((item) => item.include_in_training)
    return [
      {
        label: `展示样本（${displaySamples.length}）`,
        options: displaySamples.map((item) => ({
          value: item.battery_id,
          searchText: `${item.battery_id} ${item.source} ${item.dataset_name ?? ''}`.toLowerCase(),
          label: (
            <div className="battery-select-option">
              <div className="battery-select-option__title-row">
                <Text strong className="battery-select-option__title">
                  {item.battery_id}
                </Text>
                <span className="battery-select-option__badge">{formatSourceLabel(item.source)}</span>
              </div>
              <Text className="battery-select-option__meta">
                {item.dataset_name ?? '未命名数据集'} · {item.cycle_count} cycles
              </Text>
            </div>
          ),
        })),
      },
      {
        label: `训练数据（${trainingSamples.length}）`,
        options: trainingSamples.map((item) => ({
          value: item.battery_id,
          searchText: `${item.battery_id} ${item.source} ${item.dataset_name ?? ''}`.toLowerCase(),
          label: (
            <div className="battery-select-option">
              <div className="battery-select-option__title-row">
                <Text strong className="battery-select-option__title">
                  {item.battery_id}
                </Text>
                <div className="battery-select-option__badges">
                  <span className="battery-select-option__badge">{formatSourceLabel(item.source)}</span>
                  <span className="battery-select-option__badge battery-select-option__badge--training">训练</span>
                </div>
              </div>
              <Text className="battery-select-option__meta">
                {item.dataset_name ?? '未命名数据集'} · {item.cycle_count} cycles
              </Text>
            </div>
          ),
        })),
      },
    ].filter((group) => group.options.length > 0)
  }, [options])

  return (
    <Select
      className="battery-select"
      value={value}
      virtual
      showSearch
      listHeight={listHeight}
      placeholder={placeholder}
      disabled={disabled}
      style={style}
      popupClassName="battery-select-dropdown"
      optionLabelProp="value"
      optionFilterProp="searchText"
      options={groupedOptions}
      onChange={onChange}
    />
  )
}

export default BatterySelect
