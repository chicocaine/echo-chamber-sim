import { useCallback, useRef, useState } from 'react'

import { runSimulation } from '../lib/api'
import type { GraphEdge, MetricSnapshot, SimConfig, SimResult } from '../lib/types'

export type SimulationStatus = 'idle' | 'running' | 'done' | 'error'

const WS_BASE = import.meta.env.VITE_WS_BASE_URL ?? 'ws://localhost:8000'

interface StreamingState {
  status: SimulationStatus
  snapshots: MetricSnapshot[]
  agentOpinions: number[]
  edges: GraphEdge[]
  errorMessage: string | null
}

interface UseSimulationReturn {
  status: SimulationStatus
  result: SimResult | null
  streaming: StreamingState
  errorMessage: string | null
  run: (config: SimConfig) => Promise<void>
  runStreaming: (config: SimConfig) => void
  sendCommand: (cmd: Record<string, unknown>) => void
  streamingMode: boolean
}

export function useSimulation(): UseSimulationReturn {
  const [status, setStatus] = useState<SimulationStatus>('idle')
  const [result, setResult] = useState<SimResult | null>(null)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [streamingMode, setStreamingMode] = useState(false)
  const [streaming, setStreaming] = useState<StreamingState>({
    status: 'idle', snapshots: [], agentOpinions: [], edges: [], errorMessage: null,
  })
  const wsRef = useRef<WebSocket | null>(null)

  const run = useCallback(async (config: SimConfig) => {
    setStreamingMode(false)
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

  const sendCommand = useCallback((cmd: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(cmd))
    }
  }, [])

  const runStreaming = useCallback((config: SimConfig) => {
    // Close any existing connection.
    wsRef.current?.close()

    setStreamingMode(true)
    setResult(null)
    setStreaming({
      status: 'running', snapshots: [], agentOpinions: [],
      edges: [], errorMessage: null,
    })

    const ws = new WebSocket(`${WS_BASE}/run/stream`)
    wsRef.current = ws

    ws.onopen = () => {
      ws.send(JSON.stringify(config))
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data as string)

      if (data.type === 'complete') {
        setStreaming((prev) => ({ ...prev, status: 'done' }))
        ws.close()
        return
      }
      if (data.type === 'error') {
        setStreaming((prev) => ({
          ...prev, status: 'error',
          errorMessage: data.message ?? 'Unknown streaming error',
        }))
        ws.close()
        return
      }

      // Tick data.
      setStreaming((prev) => ({
        ...prev,
        status: 'running',
        snapshots: [...prev.snapshots, data.metrics as MetricSnapshot],
        agentOpinions: data.agent_opinions as number[],
        edges: data.edges_changed
          ? (data.edge_list as number[][]).map(([s, t]) => ({ source: s, target: t, weight: 0 }))
          : prev.edges,
      }))
    }

    ws.onerror = () => {
      setStreaming((prev) => ({
        ...prev, status: 'error', errorMessage: 'WebSocket connection failed',
      }))
    }

    ws.onclose = () => {
      setStreaming((prev) =>
        prev.status === 'running' ? { ...prev, status: 'done' } : prev,
      )
    }
  }, [])

  return { status, result, streaming, errorMessage, run, runStreaming, sendCommand, streamingMode }
}
