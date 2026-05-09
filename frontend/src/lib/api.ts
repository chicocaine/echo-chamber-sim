import type { Preset, SimConfig, SimResult } from './types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

function prepareConfig(config: SimConfig): SimConfig {
  const cfg = { ...config }
  if (cfg.topology === 'stochastic_block') {
    // Backend requires community_sizes and community_p for stochastic_block.
    // Provide sensible defaults: two equal communities with moderate homophily.
    if (!cfg.community_sizes || cfg.community_sizes.length === 0) {
      const half = Math.floor(cfg.N / 2)
      cfg.community_sizes = [half, cfg.N - half]
    }
    if (!cfg.community_p || cfg.community_p.length === 0) {
      const n = cfg.community_sizes.length
      cfg.community_p = Array.from({ length: n }, (_, i) =>
        Array.from({ length: n }, (_, j) => (i === j ? 0.4 : 0.02))
      )
    }
  }
  return cfg
}

export async function runSimulation(config: SimConfig): Promise<SimResult> {
  const response = await fetch(`${API_BASE}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(prepareConfig(config)),
  })
  if (!response.ok) {
    const text = await response.text().catch(() => '')
    throw new Error(`Simulation failed (${response.status}): ${text}`)
  }
  return (await response.json()) as SimResult
}

export async function getDefaults(): Promise<SimConfig> {
  const response = await fetch(`${API_BASE}/defaults`)
  if (!response.ok) {
    throw new Error(`Defaults request failed (${response.status})`)
  }
  return (await response.json()) as SimConfig
}

export async function getPresets(): Promise<Preset[]> {
  const response = await fetch(`${API_BASE}/presets`)
  if (!response.ok) {
    throw new Error(`Presets request failed (${response.status})`)
  }
  return (await response.json()) as Preset[]
}
