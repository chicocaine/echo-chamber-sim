import { useMemo, useState } from 'react'
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import type { AgentState, MetricSnapshot } from '../lib/types'

interface OpinionHistogramProps {
  finalAgents: AgentState[]
  snapshots: MetricSnapshot[]
}

interface HistogramBin {
  bucket: string
  count: number
}

function buildHistogram(finalAgents: AgentState[]): HistogramBin[] {
  const binCount = 20
  const min = -1
  const max = 1
  const width = (max - min) / binCount

  const counts = new Array<number>(binCount).fill(0)
  for (const agent of finalAgents) {
    const rawIndex = Math.floor((agent.opinion - min) / width)
    const index = Math.max(0, Math.min(binCount - 1, rawIndex))
    counts[index] += 1
  }

  return counts.map((count, index) => {
    const start = min + index * width
    const end = start + width
    return {
      bucket: `${start.toFixed(1)}..${end.toFixed(1)}`,
      count,
    }
  })
}

export function OpinionHistogram({ finalAgents, snapshots }: OpinionHistogramProps) {
  const [selectedSnapshotIndex, setSelectedSnapshotIndex] = useState(0)

  const histogram = useMemo(() => buildHistogram(finalAgents), [finalAgents])
  const maxIndex = Math.max(0, snapshots.length - 1)
  const selectedTick = snapshots[selectedSnapshotIndex]?.tick ?? 0

  return (
    <section className="panel">
      <header className="panel-header">
        <h2>Opinion Histogram</h2>
        <p>
          Tick {selectedTick} (MVP displays final-state distribution for all scrubber positions).
        </p>
      </header>

      <label className="scrubber">
        <span>Snapshot index: {selectedSnapshotIndex}</span>
        <input
          type="range"
          min={0}
          max={maxIndex}
          step={1}
          value={selectedSnapshotIndex}
          onChange={(event) => setSelectedSnapshotIndex(Number(event.target.value))}
          disabled={maxIndex === 0}
        />
      </label>

      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={histogram} margin={{ top: 8, right: 12, left: 4, bottom: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
            <XAxis dataKey="bucket" angle={-30} textAnchor="end" interval={1} height={64} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="count" fill="#0f766e" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
