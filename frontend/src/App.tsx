import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ConfigPanel } from './components/ConfigPanel'
import { MetricsChart } from './components/MetricsChart'
import { NetworkGraph } from './components/NetworkGraph'
import { OpinionHistogram } from './components/OpinionHistogram'
import { PlaybackBar } from './components/PlaybackBar'
import { setTickOpinions } from './components/networkGraphState'
import { useSimulation } from './hooks/useSimulation'
import { getDefaults, getPresets } from './lib/api'
import type { Preset, SimConfig } from './lib/types'
import './App.css'

const FALLBACK_CONFIG: SimConfig = {
  N: 200,
  avg_degree: 16,
  rewire_prob: 0.1,
  topology: 'watts_strogatz',
  T: 200,
  snapshot_interval: 6,
  alpha: 0.65,
  beta_pop: 0.2,
  k_exp: 20,
  agent_mix: { stubborn: 0.6, flexible: 0.2, passive: 0.1, zealot: 0.05, bot: 0.05, hk: 0, contrarian: 0, influencer: 0 },
  sir_beta: 0.3,
  sir_gamma: 0.05,
  reinforcement_factor: 0,
  recommender_type: 'content_based',
  cf_blend_ratio: 0.5,
  dynamic_rewire_rate: 0.01,
  homophily_threshold: 0.3,
  enable_churn: false,
  churn_base: -4.0,
  churn_weight: 1.0,
  T_detect: 24,
  s_thresh: 0.7,
  p_detect_remove: 0,
  rate_limit_factor: 0,
  media_literacy_boost: 0,
  diversity_ratio: 0,
  lambda_penalty: 0,
  virality_dampening: 0,
  initial_opinion_distribution: 'uniform',
  emotional_decay: 0.85,
  arousal_share_weight: 0.3,
  valence_share_weight: 0.4,
  arousal_tolerance_effect: 0.4,
  seed: 42,
}

export default function App() {
  const [config, setConfig] = useState<SimConfig>(FALLBACK_CONFIG)
  const [defaultsLoaded, setDefaultsLoaded] = useState(false)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [presets, setPresets] = useState<Preset[]>([])
  const [activePresetId, setActivePresetId] = useState<string | null>(null)
  const { status, result, errorMessage, run } = useSimulation()

  // Playback state
  const [currentTick, setCurrentTick] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackSpeed, setPlaybackSpeed] = useState(1)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const maxTick = result?.config?.T ?? 0

  useEffect(() => {
    let cancelled = false
    Promise.all([
      getDefaults().then(
        defaults => { if (!cancelled) { setConfig(defaults); setDefaultsLoaded(true) } },
        err => { if (!cancelled) { setLoadError(err instanceof Error ? err.message : 'Unknown'); setDefaultsLoaded(true) } }
      ),
      getPresets().then(
        p => { if (!cancelled) setPresets(p) },
        () => { /* presets are non-critical; ignore fetch errors */ }
      ),
    ])
    return () => { cancelled = true }
  }, [])

  // When result arrives, go to final tick
  useEffect(() => {
    if (result) {
      setCurrentTick(result.config.T)
      setIsPlaying(false)
    }
  }, [result])

  // Playback interval
  useEffect(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    if (!isPlaying || maxTick <= 0) return
    const ms = Math.max(20, 1000 / (playbackSpeed * 4))
    intervalRef.current = setInterval(() => {
      setCurrentTick(prev => {
        if (prev >= maxTick) {
          setIsPlaying(false)
          return maxTick
        }
        return prev + 1
      })
    }, ms)
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
  }, [isPlaying, playbackSpeed, maxTick])

  // Build per-tick opinion map: agentId -> opinion at currentTick
  const tickOpinions = useMemo(() => {
    if (!result) return {}
    const map: Record<number, number> = {}
    for (const agent of result.final_agents) {
      const hist = agent.opinion_history
      if (hist.length > 0) {
        const idx = Math.min(currentTick, hist.length - 1)
        map[agent.id] = hist[idx]
      } else {
        map[agent.id] = agent.opinion
      }
    }
    return map
  }, [result, currentTick])

  // Push opinions to the graph via module-level channel — avoids React
  // re-renders on the ForceGraph2D component.
  useEffect(() => {
    setTickOpinions(tickOpinions)
  }, [tickOpinions])

  // Current tick opinions for histogram
  const currentOpinions = useMemo(() => {
    return Object.values(tickOpinions)
  }, [tickOpinions])

  const handleRun = useCallback(() => {
    setIsPlaying(false)
    setCurrentTick(0)
    void run(config)
  }, [config, run])

  const handleStop = useCallback(() => {
    setIsPlaying(false)
    setCurrentTick(0)
  }, [])

  const handlePresetSelect = useCallback((preset: Preset) => {
    setConfig(preset.config)
    setActivePresetId(preset.id)
  }, [])

  const canRun = defaultsLoaded && status !== 'running'

  return (
    <div className="app-shell">
      <ConfigPanel
        config={config}
        presets={presets}
        activePresetId={activePresetId}
        onChange={setConfig}
        onPresetSelect={handlePresetSelect}
        onRun={handleRun}
        canRun={canRun}
        isRunning={status === 'running'}
      />

      <div className="main-area">
        {loadError && <div className="error-banner">Defaults unavailable: {loadError}</div>}
        {errorMessage && <div className="error-banner">{errorMessage}</div>}

        {result ? (
          <div className="graph-container">
            <NetworkGraph
              nodes={result.final_graph.nodes}
              edges={result.final_graph.edges}
            />
          </div>
        ) : status === 'running' ? (
          <div className="loading-overlay">
            <div className="loading-spinner" />
            <span className="loading-text">Running simulation...</span>
          </div>
        ) : (
          <div className="empty-state">
            <div className="empty-icon">◉</div>
            <p>Configure parameters and run the simulation</p>
            <p style={{ fontSize: 11, color: '#555' }}>
              The network graph will appear here after the run completes
            </p>
          </div>
        )}
      </div>

      <div className="bottom-panel">
        {result && (
          <>
            <PlaybackBar
              tick={currentTick}
              maxTick={maxTick}
              isPlaying={isPlaying}
              playbackSpeed={playbackSpeed}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onStop={handleStop}
              onStepLeft={() => setCurrentTick(t => Math.max(0, t - 1))}
              onStepRight={() => setCurrentTick(t => Math.min(maxTick, t + 1))}
              onSeek={setCurrentTick}
              onSpeedChange={setPlaybackSpeed}
              disabled={status === 'running'}
            />
            <div className="charts-panel">
              <MetricsChart
                snapshots={result.snapshots}
                currentTick={currentTick}
                maxTick={maxTick}
              />
              <OpinionHistogram
                opinions={currentOpinions}
                currentTick={currentTick}
                maxTick={maxTick}
              />
            </div>
          </>
        )}
      </div>
    </div>
  )
}
