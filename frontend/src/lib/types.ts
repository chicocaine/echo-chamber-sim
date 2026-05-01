export interface SimConfig {
  N: number
  avg_degree: number
  rewire_prob: number
  T: number
  snapshot_interval: number
  alpha: number
  beta_pop: number
  k_exp: number
  agent_mix: Record<string, number>
  sir_beta: number
  sir_gamma: number
  initial_opinion_distribution: 'uniform' | 'bimodal'
  emotional_decay: number
  arousal_share_weight: number
  valence_share_weight: number
  arousal_tolerance_effect: number
  seed: number
  diversity_ratio?: number
  lambda_penalty?: number
  dynamic_rewire_rate?: number
}

export interface MetricSnapshot {
  tick: number
  opinion_variance: number
  polarization_index: number
  assortativity: number
  opinion_entropy: number
  misinfo_prevalence: number
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

export interface CompareRequest {
  baseline: SimConfig
  intervention: SimConfig
  n_runs: number
}

export interface AggregatedMetrics {
  tick: number[]
  [key: string]: number[]
}

export interface ReplicatedResult {
  config: SimConfig
  n_runs: number
  aggregated: AggregatedMetrics
  all_runs: MetricSnapshot[][]
}

export interface CompareResult {
  baseline: ReplicatedResult
  intervention: ReplicatedResult
  ies: Record<string, number>
}
