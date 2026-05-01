import { useEffect, useMemo, useState } from 'react'

import { CompareView } from './components/CompareView'
import { ControlPanel } from './components/ControlPanel'
import { MetricsPanel } from './components/MetricsPanel'
import { NetworkGraph } from './components/NetworkGraph'
import { OpinionHistogram } from './components/OpinionHistogram'
import { useSimulation } from './hooks/useSimulation'
import { getDefaults } from './lib/api'
import type { SimConfig } from './lib/types'
import './App.css'

const FALLBACK_CONFIG: SimConfig = {
  N: 200,
  avg_degree: 16,
  rewire_prob: 0.1,
  T: 200,
  snapshot_interval: 6,
  alpha: 0.65,
  beta_pop: 0.2,
  k_exp: 20,
  agent_mix: {
    stubborn: 0.6,
    flexible: 0.2,
    zealot: 0.15,
    bot: 0.05,
  },
  sir_beta: 0.3,
  sir_gamma: 0.05,
  initial_opinion_distribution: 'uniform',
  seed: 42,
}

type Tab = 'simulate' | 'compare'

function App() {
  const [tab, setTab] = useState<Tab>('simulate')
  const [config, setConfig] = useState<SimConfig>(FALLBACK_CONFIG)
  const [defaultsLoaded, setDefaultsLoaded] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)

  const { status, result, errorMessage, run } = useSimulation()

  useEffect(() => {
    let cancelled = false

    async function loadDefaults(): Promise<void> {
      try {
        const defaults = await getDefaults()
        if (!cancelled) {
          setConfig(defaults)
          setDefaultsLoaded(true)
        }
      } catch (error) {
        if (!cancelled) {
          setLoadError(error instanceof Error ? error.message : 'Unable to load defaults')
          setDefaultsLoaded(true)
        }
      }
    }

    void loadDefaults()
    return () => {
      cancelled = true
    }
  }, [])

  const snapshots = result?.snapshots ?? []
  const finalAgents = result?.final_agents ?? []
  const finalGraph = result?.final_graph ?? { nodes: [], edges: [] }
  const canRun = defaultsLoaded && status !== 'running'

  const statusLabel = useMemo(() => {
    if (status === 'running') {
      return 'Simulation running...'
    }
    if (status === 'done') {
      return 'Simulation complete'
    }
    if (status === 'error') {
      return `Simulation failed: ${errorMessage ?? 'unknown error'}`
    }
    return 'Ready'
  }, [errorMessage, status])

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <h1>Echo Chamber Simulation</h1>
          <p>Agent-based social media dynamics dashboard</p>
        </div>
        <div className="status-pill" data-status={status}>
          {statusLabel}
        </div>
      </header>

      {loadError && <p className="error-banner">Defaults unavailable: {loadError}</p>}

      <nav className="tab-nav">
        <button
          type="button"
          className={tab === 'simulate' ? 'tab-active' : ''}
          onClick={() => setTab('simulate')}
        >
          Simulate
        </button>
        <button
          type="button"
          className={tab === 'compare' ? 'tab-active' : ''}
          onClick={() => setTab('compare')}
        >
          Compare
        </button>
      </nav>

      {tab === 'simulate' ? (
        <section className="layout-grid">
          <div className="left-column">
            <ControlPanel
              config={config}
              status={status}
              onConfigChange={setConfig}
              onRun={() => {
                if (canRun) {
                  void run(config)
                }
              }}
            />
            <NetworkGraph nodes={finalGraph.nodes} edges={finalGraph.edges} />
          </div>
          <div className="right-column">
            <MetricsPanel snapshots={snapshots} />
            <OpinionHistogram finalAgents={finalAgents} snapshots={snapshots} />
          </div>
        </section>
      ) : (
        <CompareView />
      )}
    </main>
  )
}

export default App
