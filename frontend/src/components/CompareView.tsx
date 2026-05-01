import { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts'

import { runCompare } from '../lib/api'
import type { AggregatedMetrics, CompareRequest, CompareResult, SimConfig } from '../lib/types'

type CompareStatus = 'idle' | 'running' | 'done' | 'error'

interface ConfigFormProps {
  label: string
  config: SimConfig
  onChange: (c: SimConfig) => void
  disabled: boolean
}

const METRIC_LABELS: Record<string, string> = {
  assortativity: 'Assortativity',
  misinfo_prevalence: 'Misinfo Prevalence',
  polarization_index: 'Polarization',
  opinion_entropy: 'Opinion Entropy',
  opinion_variance: 'Opinion Variance',
  ei_index: 'E-I Index',
  modularity_q: 'Modularity Q',
  cascade_mean: 'Cascade Mean',
  exposure_disparity: 'Exposure Disparity',
}

function ConfigForm({ label, config, onChange, disabled }: ConfigFormProps) {
  return (
    <fieldset className="compare-config" disabled={disabled}>
      <legend>{label}</legend>
      <label>
        N:
        <input
          type="number" min={20} max={1000} step={20}
          value={config.N}
          onChange={(e) => onChange({ ...config, N: Number(e.target.value) })}
        />
      </label>
      <label>
        alpha:
        <input
          type="range" min={0} max={1} step={0.05}
          value={config.alpha}
          onChange={(e) => onChange({ ...config, alpha: Number(e.target.value) })}
        />
        <span>{config.alpha.toFixed(2)}</span>
      </label>
      <label>
        diversity:
        <input
          type="range" min={0} max={0.3} step={0.05}
          value={config.diversity_ratio ?? 0}
          onChange={(e) => onChange({ ...config, diversity_ratio: Number(e.target.value) })}
        />
        <span>{(config.diversity_ratio ?? 0).toFixed(2)}</span>
      </label>
      <label>
        lambda:
        <input
          type="range" min={0} max={1} step={0.1}
          value={config.lambda_penalty ?? 0}
          onChange={(e) => onChange({ ...config, lambda_penalty: Number(e.target.value) })}
        />
        <span>{(config.lambda_penalty ?? 0).toFixed(2)}</span>
      </label>
      <label>
        rewiring:
        <input
          type="range" min={0} max={0.5} step={0.05}
          value={config.dynamic_rewire_rate ?? 0}
          onChange={(e) => onChange({ ...config, dynamic_rewire_rate: Number(e.target.value) })}
        />
        <span>{(config.dynamic_rewire_rate ?? 0).toFixed(2)}</span>
      </label>
      <label>
        T:
        <input
          type="number" min={20} max={500} step={20}
          value={config.T}
          onChange={(e) => onChange({ ...config, T: Number(e.target.value) })}
        />
      </label>
      <label>
        seed:
        <input
          type="number" min={1} max={999}
          value={config.seed}
          onChange={(e) => onChange({ ...config, seed: Number(e.target.value) })}
        />
      </label>
    </fieldset>
  )
}

function IESBadge({ metric, ies }: { metric: string; ies: number }) {
  const pct = Math.round(ies * 100)
  const isGood = ies > 0.05
  const isBad = ies < -0.05
  const color = isGood ? '#16a34a' : isBad ? '#dc2626' : '#6b7280'
  const arrow = isGood ? '↓' : isBad ? '↑' : '→'

  return (
    <span
      className="ies-badge"
      style={{
        display: 'inline-block',
        padding: '2px 8px',
        borderRadius: 4,
        backgroundColor: color,
        color: '#fff',
        fontSize: '0.8rem',
        fontWeight: 600,
        marginLeft: 6,
      }}
      title={`${METRIC_LABELS[metric] ?? metric}: ${pct}% ${isGood ? 'improvement' : isBad ? 'worsening' : 'no change'}`}
    >
      {arrow} {pct}%
    </span>
  )
}

export function CompareView() {
  const [baseline, setBaseline] = useState<SimConfig>({
    N: 100, avg_degree: 12, rewire_prob: 0.1, T: 80, snapshot_interval: 6,
    alpha: 0.65, beta_pop: 0.2, k_exp: 20,
    agent_mix: { stubborn: 0.6, flexible: 0.2, zealot: 0.1, bot: 0.1 },
    sir_beta: 0.3, sir_gamma: 0.05, initial_opinion_distribution: 'uniform',
    emotional_decay: 0.85, arousal_share_weight: 0.3, valence_share_weight: 0.4,
    arousal_tolerance_effect: 0.4, seed: 42,
  })
  const [intervention, setIntervention] = useState<SimConfig>({
    ...baseline,
    diversity_ratio: 0.2,
  })
  const [nRuns, setNRuns] = useState(3)
  const [status, setStatus] = useState<CompareStatus>('idle')
  const [result, setResult] = useState<CompareResult | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function handleCompare() {
    setStatus('running')
    setError(null)
    try {
      const req: CompareRequest = { baseline, intervention, n_runs: nRuns }
      const res = await runCompare(req)
      setResult(res)
      setStatus('done')
    } catch (err) {
      setStatus('error')
      setError(err instanceof Error ? err.message : 'Unknown error')
    }
  }

  // Build chart data from aggregated results.
  function buildChartData(agg: AggregatedMetrics | undefined): Record<string, number>[] {
    if (!agg || !agg.tick) return []
    return agg.tick.map((t, i) => {
      const row: Record<string, number> = { tick: t }
      for (const key of Object.keys(agg)) {
        if (key === 'tick') continue
        row[key] = (agg[key] as number[])[i]
      }
      return row
    })
  }

  const baselineChart = buildChartData(result?.baseline?.aggregated)
  const interventionChart = buildChartData(result?.intervention?.aggregated)
  const ies = result?.ies ?? {}

  const metricKeys = ['assortativity', 'misinfo_prevalence', 'polarization_index', 'ei_index']

  return (
    <section className="compare-view">
      <h2>Scenario Comparison</h2>
      <p>
        Configure baseline and intervention scenarios, then run replicated
        simulations to compare effectiveness.
      </p>

      <div className="compare-forms">
        <ConfigForm
          label="Baseline"
          config={baseline}
          onChange={setBaseline}
          disabled={status === 'running'}
        />
        <ConfigForm
          label="Intervention"
          config={intervention}
          onChange={setIntervention}
          disabled={status === 'running'}
        />
      </div>

      <div className="compare-actions">
        <label>
          Replicates:
          <input
            type="number" min={1} max={20} step={1}
            value={nRuns}
            onChange={(e) => setNRuns(Number(e.target.value))}
            disabled={status === 'running'}
            style={{ width: 60, marginLeft: 8 }}
          />
        </label>
        <button
          type="button"
          onClick={() => void handleCompare()}
          disabled={status === 'running'}
          className="run-button"
        >
          {status === 'running' ? 'Comparing...' : 'Run Comparison'}
        </button>
      </div>

      {status === 'error' && <p className="error-banner">{error}</p>}

      {status === 'done' && result && (
        <>
          <div className="ies-summary">
            <h3>Intervention Effectiveness</h3>
            <div className="ies-grid">
              {Object.entries(ies).map(([metric, value]) => (
                <div key={metric} className="ies-item">
                  <span>{METRIC_LABELS[metric] ?? metric}</span>
                  <IESBadge metric={metric} ies={value} />
                </div>
              ))}
            </div>
          </div>

          <div className="compare-charts">
            {metricKeys.map((metric) => (
              <div key={metric} className="compare-chart-pair">
                <h4>
                  {METRIC_LABELS[metric] ?? metric}
                  {ies[metric] !== undefined && (
                    <IESBadge metric={metric} ies={ies[metric]} />
                  )}
                </h4>
                <LineChart width={350} height={200} data={baselineChart}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="tick" label={{ value: 'tick', position: 'insideBottom', offset: -5 }} />
                  <YAxis width={60} />
                  <Tooltip />
                  <Legend />
                  <Line
                    type="monotone" dataKey={`${metric}_mean`} stroke="#3b82f6"
                    dot={false} name="Baseline"
                  />
                  <Line
                    type="monotone"
                    data={interventionChart}
                    dataKey={`${metric}_mean`} stroke="#ef4444"
                    dot={false} name="Intervention"
                  />
                </LineChart>
              </div>
            ))}
          </div>
        </>
      )}
    </section>
  )
}
