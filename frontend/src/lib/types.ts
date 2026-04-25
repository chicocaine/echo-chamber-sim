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
  seed: number
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
  contrarian_prob: number
  suspicion_score: number
  is_active: boolean
  sir_state: 'S' | 'I' | 'R'
  opinion_history: number[]
  misinfo_rate: number
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
