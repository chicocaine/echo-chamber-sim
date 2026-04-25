import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { MetricSnapshot } from '../lib/types'

interface MetricsPanelProps {
  snapshots: MetricSnapshot[]
}

export function MetricsPanel({ snapshots }: MetricsPanelProps) {
  return (
    <section className="panel metrics-panel">
      <header className="panel-header">
        <h2>Metrics Timeline</h2>
        <p>Five MVP metrics across simulation ticks.</p>
      </header>

      <div className="chart-wrap">
        <ResponsiveContainer width="100%" height={320}>
          <LineChart data={snapshots} margin={{ top: 8, right: 12, left: 4, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--grid)" />
            <XAxis dataKey="tick" stroke="var(--muted)" />
            <YAxis stroke="var(--muted)" />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="opinion_variance" stroke="#166534" dot={false} />
            <Line type="monotone" dataKey="polarization_index" stroke="#b45309" dot={false} />
            <Line type="monotone" dataKey="assortativity" stroke="#0f766e" dot={false} />
            <Line type="monotone" dataKey="opinion_entropy" stroke="#1d4ed8" dot={false} />
            <Line type="monotone" dataKey="misinfo_prevalence" stroke="#b91c1c" dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
