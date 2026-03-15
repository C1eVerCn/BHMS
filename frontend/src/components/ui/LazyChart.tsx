import React, { Suspense, lazy } from 'react'
import type { CoreChartProps } from './CoreChart'

const CoreChart = lazy(() => import('./CoreChart'))

const LazyChart: React.FC<CoreChartProps> = ({ style, ...props }) => {
  const height = style?.height ?? 320

  return (
    <Suspense
      fallback={
        <div className="chart-placeholder" style={{ height }}>
          <div className="chart-placeholder__grid" />
          <span>图表模块加载中...</span>
        </div>
      }
    >
      <CoreChart style={style} {...props} />
    </Suspense>
  )
}

export default LazyChart
