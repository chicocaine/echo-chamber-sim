import type { SimConfig, SimResult } from './types'

const API_BASE = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

export async function runSimulation(config: SimConfig): Promise<SimResult> {
  const response = await fetch(`${API_BASE}/run`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(config),
  })

  if (!response.ok) {
    throw new Error(`Simulation request failed (${response.status})`)
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