import React, { useEffect, useRef } from 'react'
import { GaugeChart, GraphChart, LineChart, PieChart } from 'echarts/charts'
import { GridComponent, LegendComponent, TooltipComponent } from 'echarts/components'
import { init, use, type EChartsCoreOption, type EChartsInitOpts, type EChartsType, type SetOptionOpts } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'

use([CanvasRenderer, GridComponent, LegendComponent, TooltipComponent, LineChart, PieChart, GaugeChart, GraphChart])

export interface CoreChartProps {
  option: EChartsCoreOption
  style?: React.CSSProperties
  className?: string
  initOpts?: EChartsInitOpts
  setOptionOpts?: SetOptionOpts
}

const CoreChart: React.FC<CoreChartProps> = ({ option, style, className, initOpts, setOptionOpts }) => {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const chartRef = useRef<EChartsType | null>(null)

  useEffect(() => {
    if (!containerRef.current) return

    const chart = init(containerRef.current, undefined, initOpts)
    chartRef.current = chart

    const resizeObserver = new ResizeObserver(() => {
      chart.resize()
    })

    resizeObserver.observe(containerRef.current)

    return () => {
      resizeObserver.disconnect()
      chart.dispose()
      chartRef.current = null
    }
  }, [initOpts])

  useEffect(() => {
    chartRef.current?.setOption(option, setOptionOpts)
  }, [option, setOptionOpts])

  return <div ref={containerRef} className={className} style={{ width: '100%', minHeight: 240, ...style }} />
}

export default CoreChart
