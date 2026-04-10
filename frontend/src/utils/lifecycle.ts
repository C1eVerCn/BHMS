import type { CyclePoint, LifecyclePredictionResult, PredictionProjection, RiskWindow } from '../types/domain'

type ChartTuple = [number, number]

interface LifecycleChartPayload {
  batteryCycles?: CyclePoint[]
  prediction?: LifecyclePredictionResult
}

function pushUniquePoint(target: ChartTuple[], point: ChartTuple) {
  const last = target[target.length - 1]
  if (last && last[0] === point[0] && last[1] === point[1]) return
  target.push(point)
}

function readSeriesValue(payload: unknown) {
  if (!Array.isArray(payload)) return undefined
  const value = payload[1]
  return typeof value === 'number' ? value : undefined
}

function formatAxisValue(value: unknown) {
  return typeof value === 'number' ? value.toFixed(1) : String(value ?? '--')
}

function formatCapacity(value: number | undefined) {
  return typeof value === 'number' ? `${value.toFixed(4)} Ah` : '--'
}

export function buildObservedSeries({ batteryCycles = [], prediction }: LifecycleChartPayload): ChartTuple[] {
  const actualPoints = prediction?.projection?.actual_points
  if (actualPoints?.length) {
    return actualPoints.map((item) => [item.cycle, item.capacity])
  }
  return batteryCycles.map((item) => [item.cycle_number, item.capacity])
}

export function buildPredictionSeries(projection?: PredictionProjection, observedSeries: ChartTuple[] = []): ChartTuple[] {
  if (!projection) return []
  const series: ChartTuple[] = []
  const lastObserved = observedSeries[observedSeries.length - 1]
  if (lastObserved) {
    pushUniquePoint(series, lastObserved)
  }
  const displayPoints = projection.display_points?.length ? projection.display_points : projection.forecast_points
  for (const item of displayPoints ?? []) {
    pushUniquePoint(series, [item.cycle, item.capacity])
  }
  if (projection.predicted_zero_cycle != null && series.length) {
    const last = series[series.length - 1]
    if (last[0] < projection.predicted_zero_cycle || last[1] !== 0) {
      pushUniquePoint(series, [projection.predicted_zero_cycle, 0])
    }
  }
  return series
}

export function buildConfidenceBand(projection?: PredictionProjection) {
  if (!projection?.confidence_band?.length) {
    return {
      lower: [] as ChartTuple[],
      range: [] as ChartTuple[],
    }
  }

  const lower = projection.confidence_band.map((item) => [item.cycle, item.lower] as ChartTuple)
  const range = projection.confidence_band.map((item) => [item.cycle, Math.max(0, item.upper - item.lower)] as ChartTuple)
  return { lower, range }
}

export function buildLifecycleChartOption({ batteryCycles = [], prediction }: LifecycleChartPayload) {
  const observedSeries = buildObservedSeries({ batteryCycles, prediction })
  const predictionSeries = buildPredictionSeries(prediction?.projection, observedSeries)
  const confidenceBand = buildConfidenceBand(prediction?.projection)
  const riskWindows = prediction?.risk_windows ?? []

  return {
    backgroundColor: 'transparent',
    tooltip: {
      trigger: 'axis',
      backgroundColor: 'rgba(9, 18, 38, 0.92)',
      borderWidth: 0,
      textStyle: { color: '#f8fafc' },
      formatter: (params: any) => {
        const rows = Array.isArray(params) ? params : [params]
        if (!rows.length) return ''
        const observed = rows.find((item) => item.seriesName === '历史轨迹')
        const forecast = rows.find((item) => item.seriesName === '全周期预测')
        const lowerBand = rows.find((item) => item.seriesName === '置信带下界')
        const rangeBand = rows.find((item) => item.seriesName === '置信区间')
        const lower = readSeriesValue(lowerBand?.data)
        const range = readSeriesValue(rangeBand?.data)
        const upper = typeof lower === 'number' && typeof range === 'number' ? lower + range : undefined
        const lines = [`<div style="margin-bottom:6px;font-weight:600;">循环 ${formatAxisValue(rows[0]?.axisValue)}</div>`]
        if (observed) {
          lines.push(`<div>${observed.marker}历史轨迹：${formatCapacity(readSeriesValue(observed.data))}</div>`)
        }
        if (forecast) {
          lines.push(`<div>${forecast.marker}全周期预测：${formatCapacity(readSeriesValue(forecast.data))}</div>`)
        }
        if (typeof lower === 'number' && typeof upper === 'number') {
          lines.push(`<div>置信区间：${lower.toFixed(4)} ~ ${upper.toFixed(4)} Ah</div>`)
        }
        return lines.join('')
      },
    },
    legend: {
      top: 6,
      textStyle: { color: '#5f6c7b' },
      data: ['历史轨迹', '全周期预测', '置信区间'],
    },
    grid: { left: 28, right: 24, top: 64, bottom: 28, containLabel: true },
    xAxis: {
      type: 'value',
      name: '循环次数',
      axisLabel: { color: '#6b7280' },
      splitLine: { show: false },
    },
    yAxis: {
      type: 'value',
      name: '容量(Ah)',
      axisLabel: { color: '#6b7280' },
      splitLine: { lineStyle: { color: 'rgba(16, 35, 63, 0.08)' } },
    },
    series: [
      {
        name: '历史轨迹',
        type: 'line',
        showSymbol: false,
        data: observedSeries,
        lineStyle: { color: '#0f6fff', width: 3 },
        z: 3,
      },
      {
        name: '全周期预测',
        type: 'line',
        showSymbol: false,
        data: predictionSeries,
        lineStyle: { color: '#21b26f', width: 3 },
        z: 4,
        markLine: {
          symbol: 'none',
          label: { color: '#334155' },
          lineStyle: { color: 'rgba(51, 65, 85, 0.28)' },
          data: [
            prediction?.predicted_knee_cycle ? { xAxis: prediction.predicted_knee_cycle, name: 'knee' } : undefined,
            prediction?.predicted_eol_cycle ? { xAxis: prediction.predicted_eol_cycle, name: 'EOL' } : undefined,
            prediction?.projection?.predicted_zero_cycle ? { xAxis: prediction.projection.predicted_zero_cycle, name: '0' } : undefined,
          ].filter(Boolean),
        },
        markArea: riskWindows.length
          ? {
              itemStyle: { color: 'rgba(255, 159, 10, 0.08)' },
              data: riskWindows.map((item: RiskWindow) => [{ xAxis: item.start_cycle }, { xAxis: item.end_cycle, name: item.label }]),
            }
          : undefined,
      },
      {
        name: '置信带下界',
        type: 'line',
        symbol: 'none',
        silent: true,
        data: confidenceBand.lower,
        lineStyle: { opacity: 0 },
        stack: 'confidence-band',
        z: 1,
      },
      {
        name: '置信区间',
        type: 'line',
        symbol: 'none',
        silent: true,
        data: confidenceBand.range,
        lineStyle: { opacity: 0 },
        areaStyle: { color: 'rgba(33, 178, 111, 0.12)' },
        stack: 'confidence-band',
        z: 1,
      },
    ],
  }
}
