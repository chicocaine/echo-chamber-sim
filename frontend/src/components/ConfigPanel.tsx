import { useCallback, useState } from 'react'
import { AGENT_TYPES, CONFIG_SECTIONS } from '../lib/configDefinitions'
import type { SimConfig } from '../lib/types'

interface Props {
  config: SimConfig
  onChange: (config: SimConfig) => void
  onRun: () => void
  canRun: boolean
  isRunning: boolean
}

export function ConfigPanel({ config, onChange, onRun, canRun, isRunning }: Props) {
  const [openSections, setOpenSections] = useState<Set<string>>(
    new Set(['network', 'population', 'simulation', 'recommendation'])
  )

  const toggleSection = (key: string) => {
    setOpenSections(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  const update = useCallback(
    (key: string, value: number | string | boolean) => {
      onChange({ ...config, [key]: value })
    },
    [config, onChange]
  )

  const updateAgentMix = useCallback(
    (agentType: string, value: number) => {
      onChange({
        ...config,
        agent_mix: { ...config.agent_mix, [agentType]: value },
      })
    },
    [config, onChange]
  )

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2>Simulation Config</h2>
        <p>Configure parameters and run the echo chamber model</p>
      </div>

      <div className="sidebar-scroll">
        {CONFIG_SECTIONS.map(section => (
          <div key={section.key} className="config-section">
            <div
              className="config-section-header"
              onClick={() => toggleSection(section.key)}
            >
              <span>{section.label}</span>
              <span className={`chevron ${openSections.has(section.key) ? 'open' : ''}`}>
                ▶
              </span>
            </div>

            {openSections.has(section.key) && (
              <div className="config-section-body">
                {section.key === 'population' && (
                  <AgentMixEditor
                    agentMix={config.agent_mix}
                    onChange={updateAgentMix}
                  />
                )}
                {section.params.map(param => (
                  <ParamField
                    key={param.key}
                    param={param}
                    value={(config as Record<string, unknown>)[param.key]}
                    onChange={update}
                  />
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      <div className="sidebar-footer">
        <button className="run-btn" onClick={onRun} disabled={!canRun}>
          {isRunning ? 'Running...' : 'Run Simulation'}
        </button>
      </div>
    </aside>
  )
}

function ParamField({
  param,
  value,
  onChange,
}: {
  param: { key: string; label: string; tooltip: string; type: string; min?: number; max?: number; step?: number; options?: { value: string; label: string }[] }
  value: unknown
  onChange: (key: string, value: number | string | boolean) => void
}) {
  return (
    <div className="config-field">
      <div className="config-field-row">
        <span className="config-label" title={param.tooltip}>
          {param.label}
        </span>

        {param.type === 'select' && param.options && (
          <select
            className="config-select"
            value={String(value ?? '')}
            onChange={e => onChange(param.key, e.target.value)}
          >
            {param.options.map(opt => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        )}

        {param.type === 'boolean' && (
          <label className="config-toggle">
            <input
              type="checkbox"
              checked={Boolean(value)}
              onChange={e => onChange(param.key, e.target.checked)}
            />
            <span className="toggle-track" />
            <span className="toggle-thumb" />
          </label>
        )}

        {param.type === 'number' && (
          <input
            className="config-input"
            type="number"
            value={String(value ?? '')}
            min={param.min}
            max={param.max}
            step={param.step ?? 1}
            onChange={e => {
              const v = parseFloat(e.target.value)
              if (!isNaN(v)) onChange(param.key, v)
            }}
          />
        )}

        {param.type === 'slider' && (
          <div className="config-slider">
            <input
              type="range"
              value={Number(value ?? 0)}
              min={param.min ?? 0}
              max={param.max ?? 1}
              step={param.step ?? 0.01}
              onChange={e => onChange(param.key, parseFloat(e.target.value))}
            />
            <span className="slider-value">
              {typeof value === 'number' ? value.toFixed(3) : String(value ?? '')}
            </span>
          </div>
        )}
      </div>
    </div>
  )
}

const AGENT_MIX_TOOLTIP = `Fraction of the population for each agent type. Must sum to 1.0.

Stubborn (FJ): anchored to initial belief, partially influenced by neighbors
Flexible (DeGroot): pure social averaging, no memory of starting position
Passive: lurkers, very low activity rate (0.05-0.1 per tick)
Zealot: fixed extreme opinion, never changes
Bot: automated misinformation spreaders, high activity, coordinated campaigns
HK Bounded: only listen to neighbors within confidence bound ε
Contrarian: probabilistically move opposite to neighbor consensus
Influencer: outsized reach, high-degree hub nodes`

function AgentMixEditor({
  agentMix,
  onChange,
}: {
  agentMix: Record<string, number>
  onChange: (agentType: string, value: number) => void
}) {
  const total = Object.values(agentMix).reduce((a, b) => a + b, 0)

  return (
    <div className="config-field">
      <span className="config-label" title={AGENT_MIX_TOOLTIP}>
        Agent Mix (sum: {total.toFixed(2)})
      </span>
      {AGENT_TYPES.map(at => (
        <div key={at.key} className="agent-mix-row">
          <span className="agent-mix-label" style={{ color: at.color }}>
            {at.label}
          </span>
          <input
            className="agent-mix-input"
            type="number"
            value={agentMix[at.key] ?? 0}
            min={0}
            max={1}
            step={0.01}
            onChange={e => {
              const v = parseFloat(e.target.value)
              if (!isNaN(v)) onChange(at.key, v)
            }}
          />
          <div className="agent-mix-bar">
            <div
              style={{
                width: `${((agentMix[at.key] ?? 0) / Math.max(total, 1)) * 100}%`,
                background: at.color,
                height: '100%',
                borderRadius: 3,
                transition: 'width 0.15s',
              }}
            />
          </div>
        </div>
      ))}
    </div>
  )
}
