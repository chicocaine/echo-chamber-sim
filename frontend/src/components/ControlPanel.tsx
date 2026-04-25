import type { ChangeEvent } from 'react'

import type { SimConfig } from '../lib/types'

interface ControlPanelProps {
  config: SimConfig
  status: 'idle' | 'running' | 'done' | 'error'
  onConfigChange: (nextConfig: SimConfig) => void
  onRun: () => void
}

function roundToTwo(value: number): number {
  return Math.round(value * 100) / 100
}

function handleRange(
  event: ChangeEvent<HTMLInputElement>,
  updater: (value: number) => SimConfig,
  onConfigChange: (nextConfig: SimConfig) => void,
): void {
  onConfigChange(updater(Number(event.target.value)))
}

export function ControlPanel({ config, status, onConfigChange, onRun }: ControlPanelProps) {
  const updateBotAndZealot = (botFraction: number, zealotFraction: number): SimConfig => {
    const flexible = 0.2
    const stubborn = 1 - (flexible + botFraction + zealotFraction)

    return {
      ...config,
      N: Math.round(config.N),
      agent_mix: {
        stubborn: roundToTwo(stubborn),
        flexible,
        zealot: roundToTwo(zealotFraction),
        bot: roundToTwo(botFraction),
      },
    }
  }

  return (
    <section className="panel">
      <header className="panel-header">
        <h2>Simulation Controls</h2>
        <p>Tune parameters and run the model.</p>
      </header>

      <div className="control-grid">
        <label>
          <span>Personalization (alpha): {config.alpha.toFixed(2)}</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={config.alpha}
            onChange={(event) =>
              handleRange(
                event,
                (value) => ({ ...config, alpha: roundToTwo(value) }),
                onConfigChange,
              )
            }
          />
        </label>

        <label>
          <span>Population (N): {config.N}</span>
          <input
            type="range"
            min={100}
            max={1000}
            step={100}
            value={config.N}
            onChange={(event) =>
              handleRange(event, (value) => ({ ...config, N: Math.round(value) }), onConfigChange)
            }
          />
        </label>

        <label>
          <span>Misinformation transmission (sir_beta): {config.sir_beta.toFixed(2)}</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={config.sir_beta}
            onChange={(event) =>
              handleRange(
                event,
                (value) => ({ ...config, sir_beta: roundToTwo(value) }),
                onConfigChange,
              )
            }
          />
        </label>

        <label>
          <span>Bot fraction: {(config.agent_mix.bot ?? 0).toFixed(2)}</span>
          <input
            type="range"
            min={0}
            max={0.2}
            step={0.01}
            value={config.agent_mix.bot ?? 0}
            onChange={(event) => {
              const nextBot = Number(event.target.value)
              const next = updateBotAndZealot(nextBot, config.agent_mix.zealot ?? 0)
              onConfigChange(next)
            }}
          />
        </label>

        <label>
          <span>Zealot fraction: {(config.agent_mix.zealot ?? 0).toFixed(2)}</span>
          <input
            type="range"
            min={0}
            max={0.2}
            step={0.01}
            value={config.agent_mix.zealot ?? 0}
            onChange={(event) => {
              const nextZealot = Number(event.target.value)
              const next = updateBotAndZealot(config.agent_mix.bot ?? 0, nextZealot)
              onConfigChange(next)
            }}
          />
        </label>
      </div>

      <button type="button" onClick={onRun} disabled={status === 'running'} className="run-button">
        {status === 'running' ? 'Running...' : 'Run Simulation'}
      </button>

      <dl className="config-readout">
        <div>
          <dt>stubborn</dt>
          <dd>{(config.agent_mix.stubborn ?? 0).toFixed(2)}</dd>
        </div>
        <div>
          <dt>flexible</dt>
          <dd>{(config.agent_mix.flexible ?? 0).toFixed(2)}</dd>
        </div>
        <div>
          <dt>zealot</dt>
          <dd>{(config.agent_mix.zealot ?? 0).toFixed(2)}</dd>
        </div>
        <div>
          <dt>bot</dt>
          <dd>{(config.agent_mix.bot ?? 0).toFixed(2)}</dd>
        </div>
      </dl>
    </section>
  )
}
