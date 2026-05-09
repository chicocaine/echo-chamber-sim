export interface SimConfig {
  N: number
  avg_degree: number
  rewire_prob: number
  topology: 'watts_strogatz' | 'barabasi_albert' | 'erdos_renyi' | 'stochastic_block'
  community_sizes?: number[] | null
  community_p?: number[][] | null
  T: number
  snapshot_interval: number
  alpha: number
  beta_pop: number
  k_exp: number
  agent_mix: Record<string, number>
  sir_beta: number
  sir_gamma: number
  reinforcement_factor: number
  recommender_type: 'content_based' | 'cf' | 'graph' | 'hybrid'
  cf_blend_ratio: number
  dynamic_rewire_rate: number
  homophily_threshold: number
  enable_churn: boolean
  churn_base: number
  churn_weight: number
  T_detect: number
  s_thresh: number
  p_detect_remove: number
  rate_limit_factor: number
  media_literacy_boost: number
  diversity_ratio: number
  lambda_penalty: number
  virality_dampening: number
  initial_opinion_distribution: 'uniform' | 'bimodal'
  emotional_decay: number
  arousal_share_weight: number
  valence_share_weight: number
  arousal_tolerance_effect: number
  seed: number
}

export interface MetricSnapshot {
  tick: number
  opinion_variance: number
  polarization_index: number
  assortativity: number
  opinion_entropy: number
  misinfo_prevalence: number
  modularity_q?: number
  ei_index?: number
  cascade_mean?: number
  cascade_max?: number
  exposure_disparity?: number
}

export interface AgentState {
  id: number
  agent_type: string
  opinion: number
  initial_opinion: number
  stubbornness: number
  susceptibility: number
  trust: number
  expertise: number
  activity_rate: number
  emotional_arousal: number
  media_literacy: number
  confidence_bound: number
  arousal_tolerance_effect: number
  contrarian_prob: number
  influence_weight_multiplier: number
  suspicion_score: number
  is_active: boolean
  sir_states: Record<number, 'S' | 'I' | 'R'>
  opinion_history: number[]
  misinfo_rate: number
  exposure_count: Record<number, number>
}

export interface GraphNode {
  id: number
  opinion: number
  activity_rate: number
  agent_type: string
  is_active: boolean
}

export interface GraphEdge {
  source: number
  target: number
  weight: number
}

export interface GraphSnapshot {
  nodes: GraphNode[]
  edges: GraphEdge[]
}

export interface SimResult {
  config: SimConfig
  snapshots: MetricSnapshot[]
  final_agents: AgentState[]
  final_graph: GraphSnapshot
}

export type SimStatus = 'idle' | 'running' | 'done' | 'error'

export interface ConfigSectionDef {
  key: string
  label: string
  params: ConfigParamDef[]
}

export interface Preset {
  id: string
  label: string
  description: string
  config: SimConfig
}

export interface ConfigParamDef {
  key: string
  label: string
  tooltip: string
  type: 'number' | 'select' | 'boolean' | 'slider' | 'agent_mix'
  min?: number
  max?: number
  step?: number
  options?: { value: string; label: string }[]
  agentType?: string
  showIf?: (config: SimConfig) => boolean
}
