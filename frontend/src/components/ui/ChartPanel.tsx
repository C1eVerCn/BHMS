import React from 'react'
import type { CardProps } from 'antd'

import type { CoreChartProps } from './CoreChart'
import EmptyStateBlock from './EmptyStateBlock'
import LazyChart from './LazyChart'
import PanelCard from './PanelCard'

interface ChartPanelProps extends Omit<CardProps, 'children'> {
  option: CoreChartProps['option']
  hasData?: boolean
  height?: number
  emptyTitle?: React.ReactNode
  emptyDescription?: React.ReactNode
  chartProps?: Omit<CoreChartProps, 'option' | 'style'>
}

const ChartPanel: React.FC<ChartPanelProps> = ({
  option,
  hasData = true,
  height = 320,
  emptyTitle = '暂无数据',
  emptyDescription,
  chartProps,
  className,
  ...cardProps
}) => {
  return (
    <PanelCard className={['chart-card', className].filter(Boolean).join(' ')} {...cardProps}>
      {hasData ? (
        <LazyChart option={option} style={{ height }} {...chartProps} />
      ) : (
        <EmptyStateBlock compact title={emptyTitle} description={emptyDescription} className="chart-empty-state" />
      )}
    </PanelCard>
  )
}

export default ChartPanel
