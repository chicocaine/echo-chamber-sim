import { useCallback, useState } from 'react'

import { runSimulation } from '../lib/api'
import type { SimConfig, SimResult } from '../lib/types'

export type SimulationStatus = 'idle' | 'running' | 'done' | 'error'

interface UseSimulationState {
  status: SimulationStatus
  result: SimResult | null
  errorMessage: string | null
  run: (config: SimConfig) => Promise<void>
}

export function useSimulation(): UseSimulationState {
  const [status, setStatus] = useState<SimulationStatus>('idle')
  const [result, setResult] = useState<SimResult | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const run = useCallback(async (config: SimConfig) => {
    setStatus('running')
    setErrorMessage(null)

    try {
      const payload = await runSimulation(config)
      setResult(payload)
      setStatus('done')
    } catch (error) {
      setStatus('error')
      setResult(null)
      setErrorMessage(error instanceof Error ? error.message : 'Unknown error')
    }
  }, [])

  return {
    status,
    result,
    errorMessage,
    run,
  }
}
