import { useMemo } from 'react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { MetricSnapshot } from '../lib/types'

interface Props {
  snapshots: MetricSnapshot[]
  currentTick: number
  maxTick: number
}

interface MetricDef {
  key: string
  label: string
  description: string
  color: string
}

const METRICS: MetricDef[] = [
  {
    key: 'polarization_index',
    label: 'Polarization',
    description: 'Mean absolute opinion difference across connected agents. Higher = more divided.',
    color: '#e0556a',
  },
  {
    key: 'assortativity',
    label: 'Assortativity',
    description: 'Pearson correlation of opinions across edges. +1 = only similar agents linked (echo chambers).',
    color: '#6c8cff',
  },
  {
    key: 'opinion_entropy',
    label: 'Entropy',
    description: 'Shannon entropy of the opinion distribution. Low = concentrated clusters, high = diverse.',
    color: '#5abc7a',
  },
  {
    key: 'misinfo_prevalence',
    label: 'Misinfo',
    description: 'Fraction of agents currently in the Infected SIR state — actively believing false content.',
    color: '#f0a060',
  },
]

export function MetricsChart({ snapshots, currentTick, maxTick }: Props) {
  const { data, yMax } = useMemo(() => {
    if (snapshots.length === 0) return { data: [], yMax: 1 }

    let max = 0
    const rows = snapshots.map(s => {
      const row: Record<string, number> = {}
      for (const m of METRICS) {
        const v = (s as Record<string, number>)[m.key] ?? 0
        if (v > max) max = v
        row[m.key] = v
      }
      row.tick = s.tick
      return row
    })

    return { data: rows, yMax: Math.ceil(max * 1.15 * 100) / 100 || 0.1 }
  }, [snapshots])

  return (
    <div className="chart-box">
      <div className="chart-legend">
        {METRICS.map(m => (
          <span
            key={m.key}
            className="chart-legend-item"
            title={m.description}
          >
            <span className="legend-swatch" style={{ background: m.color }} />
            {m.label}
          </span>
        ))}
      </div>
      {data.length > 0 ? (
        <div className="chart-area">
          <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.04)" strokeDasharray="3 3" />
            <XAxis
              dataKey="tick"
              stroke="#555"
              tick={{ fontSize: 10, fill: '#787888' }}
              tickLine={false}
              domain={[0, maxTick]}
              type="number"
              allowDataOverflow
            />
            <YAxis
              stroke="#555"
              tick={{ fontSize: 10, fill: '#787888' }}
              tickLine={false}
              domain={[0, yMax]}
              type="number"
              allowDataOverflow
            />
            <Tooltip
              contentStyle={{
                background: '#1a1a24',
                border: '1px solid #2a2a3a',
                borderRadius: 4,
                fontSize: 11,
                color: '#d0d0d8',
              }}
              labelFormatter={(v: number) => `Tick ${v}`}
              formatter={(value: number, name: string) => {
                const def = METRICS.find(m => m.key === name)
                return [value.toFixed(4), def?.label ?? name]
              }}
            />
            {METRICS.map(m => (
              <Line
                key={m.key}
                type="monotone"
                dataKey={m.key}
                stroke={m.color}
                dot={false}
                strokeWidth={1.5}
                isAnimationActive={false}
              />
            ))}
            {maxTick > 0 && (
              <ReferenceLine
                x={currentTick}
                stroke="rgba(255,255,255,0.2)"
                strokeWidth={1}
                strokeDasharray="3 3"
              />
            )}
          </LineChart>
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
