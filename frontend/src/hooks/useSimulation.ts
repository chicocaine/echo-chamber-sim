import { useCallback, useRef, useState } from 'react'
import { runSimulation } from '../lib/api'
import type { SimConfig, SimResult } from '../lib/types'

export type Status = 'idle' | 'running' | 'done' | 'error'

export function useSimulation() {
  const [status, setStatus] = useState<Status>('idle')
  const [result, setResult] = useState<SimResult | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const run = useCallback(async (config: SimConfig) => {
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setStatus('running')
    setErrorMessage(null)
    setResult(null)

    try {
      const res = await runSimulation(config)
      if (controller.signal.aborted) return
      setResult(res)
      setStatus('done')
    } catch (err) {
      if (controller.signal.aborted) return
      setErrorMessage(err instanceof Error ? err.message : 'Unknown error')
      setStatus('error')
    }
  }, [])

  const reset = useCallback(() => {
    abortRef.current?.abort()
    setStatus('idle')
    setResult(null)
    setErrorMessage(null)
  }, [])

  return { status, result, errorMessage, run, reset } as const
}
