import { useMemo } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

interface Props {
  opinions: number[]
  currentTick: number
  maxTick: number
}

const BINS = 24

function opinionColor(opinion: number): string {
  const t = (opinion + 1) / 2
  const r = Math.round(220 * t + 80 * (1 - t))
  const g = Math.round(60 * t + 80 * (1 - t))
  const b = Math.round(60 * t + 220 * (1 - t))
  return `rgb(${r},${g},${b})`
}

export function OpinionHistogram({ opinions }: Props) {
  const histogram = useMemo(() => {
    if (opinions.length === 0) return []

    const bins: number[] = new Array(BINS).fill(0)
    for (const o of opinions) {
      const idx = Math.min(Math.floor(((o + 1) / 2) * BINS), BINS - 1)
      bins[idx]++
    }

    const max = Math.max(1, ...bins)
    const binWidth = 2 / BINS
    return bins.map((count, i) => ({
      binStart: -1 + i * binWidth,
      binCenter: -1 + (i + 0.5) * binWidth,
      count,
      fraction: count / opinions.length,
      maxFraction: count / max,
    }))
  }, [opinions])

  return (
    <div className="chart-box">
      <div className="chart-heading">Opinion Distribution ({opinions.length} agents)</div>
      {histogram.length > 0 ? (
        <div className="chart-area">
          <ResponsiveContainer width="100%" height="100%">
          <BarChart data={histogram} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" />
            <XAxis
              dataKey="binCenter"
              stroke="#555"
              tick={{ fontSize: 10, fill: '#787888' }}
              tickLine={false}
              domain={[-1, 1]}
              type="number"
              tickFormatter={(v: number) => v.toFixed(1)}
            />
            <YAxis
              stroke="#555"
              tick={{ fontSize: 10, fill: '#787888' }}
              tickLine={false}
            />
            <Tooltip
              contentStyle={{
                background: '#1a1a24',
                border: '1px solid #2a2a3a',
                borderRadius: 4,
                fontSize: 11,
                color: '#d0d0d8',
              }}
              formatter={(value: number) => [value, 'Agents']}
              labelFormatter={(v: number) => `Opinion: ${v.toFixed(2)}`}
            />
            <Bar dataKey="count" radius={[1, 1, 0, 0]} isAnimationActive={false}>
              {histogram.map((entry, idx) => (
                <Cell key={idx} fill={opinionColor(entry.binCenter)} fillOpacity={0.8} />
              ))}
            </Bar>
            <ReferenceLine x={0} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" />
          </BarChart>
          </ResponsiveContainer>
        </div>
      ) : (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#555' }}>
          No data
        </div>
      )}
    </div>
  )
}
