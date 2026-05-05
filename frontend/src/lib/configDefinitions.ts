import type { ConfigParamDef, ConfigSectionDef } from './types'

const NETWORK_PARAMS: ConfigParamDef[] = [
  {
    key: 'N',
    label: 'Number of agents',
    tooltip: 'Total number of agents in the simulation.\nRange: 100–1,000,000\nDefault: 200\nLarger = more realistic but slower.',
    type: 'number',
    min: 10,
    max: 1000000,
    step: 10,
  },
  {
    key: 'avg_degree',
    label: 'Average degree',
    tooltip: 'Average number of connections per agent.\nRange: 5–50\nDefault: 16\nHigher = more connected, faster consensus.',
    type: 'number',
    min: 2,
    max: 100,
  },
  {
    key: 'topology',
    label: 'Network topology',
    tooltip: 'How the social graph is wired.\n• watts_strogatz: small-world with high clustering (realistic)\n• barabasi_albert: scale-free with hub nodes\n• erdos_renyi: random uniform connections\n• stochastic_block: pre-defined community clusters',
    type: 'select',
    options: [
      { value: 'watts_strogatz', label: 'Watts–Strogatz' },
      { value: 'barabasi_albert', label: 'Barabási–Albert' },
      { value: 'erdos_renyi', label: 'Erdős–Rényi' },
      { value: 'stochastic_block', label: 'Stochastic Block' },
    ],
  },
  {
    key: 'rewire_prob',
    label: 'Rewiring probability',
    tooltip: 'Watts–Strogatz rewiring probability.\nRange: 0–1\nDefault: 0.1\nHigher = more random edges, less clustering.',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: 'homophily_threshold',
    label: 'Homophily threshold',
    tooltip: 'Maximum opinion difference allowed for forming a new tie.\nRange: 0–1\nDefault: 0.3\nLower = ties only between very similar agents → strong echo chambers.',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: 'dynamic_rewire_rate',
    label: 'Dynamic rewire rate',
    tooltip: 'Per-tick probability an agent rewires one edge.\nRange: 0–1\nDefault: 0.01\nAgents unfollow disagreeing neighbors and follow similar ones.',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.001,
  },
]

const POPULATION_PARAMS: ConfigParamDef[] = [
  {
    key: 'initial_opinion_distribution',
    label: 'Initial opinion distribution',
    tooltip: 'How opinions are sampled at t=0.\n• uniform: evenly spread across [-1, 1]\n• bimodal: two peaks at ±0.7 (pre-polarized population)',
    type: 'select',
    options: [
      { value: 'uniform', label: 'Uniform' },
      { value: 'bimodal', label: 'Bimodal (pre-polarized)' },
    ],
  },
]

const SIM_PARAMS: ConfigParamDef[] = [
  {
    key: 'T',
    label: 'Total ticks',
    tooltip: 'Number of discrete time steps to simulate.\nDefault: 200\n720 ticks ≈ 30 days at 1 tick/hour.\nHigher = longer simulation.',
    type: 'number',
    min: 1,
    max: 10000,
    step: 10,
  },
  {
    key: 'snapshot_interval',
    label: 'Snapshot interval',
    tooltip: 'How often (in ticks) metrics are logged.\nDefault: 6\nLower = more data points in charts but more memory.',
    type: 'number',
    min: 1,
    max: 100,
  },
  {
    key: 'seed',
    label: 'Random seed',
    tooltip: 'Random seed for reproducibility.\nDefault: 42\nSame seed + same config = identical results.',
    type: 'number',
    min: 0,
    max: 999999,
  },
]

const REC_PARAMS: ConfigParamDef[] = [
  {
    key: 'recommender_type',
    label: 'Recommender type',
    tooltip: 'Algorithm that decides what content each agent sees.\n• content_based: scores by similarity to past likes (high echo)\n• cf: recommends what similar users liked (user-cluster loops)\n• graph: random walk on user-item graph\n• hybrid: blends CF + content-based',
    type: 'select',
    options: [
      { value: 'content_based', label: 'Content-Based' },
      { value: 'cf', label: 'Collaborative Filtering' },
      { value: 'graph', label: 'Graph-Based' },
      { value: 'hybrid', label: 'Hybrid' },
    ],
  },
  {
    key: 'alpha',
    label: 'Personalization strength α',
    tooltip: 'How much the recommender prioritizes similar content.\nRange: 0–1\nDefault: 0.65\n0 = fully random feed (no echo chamber).\n1 = fully personalized (maximum filter bubble).',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: 'beta_pop',
    label: 'Popularity bias β',
    tooltip: 'How much viral content is boosted in rankings.\nRange: 0–1\nDefault: 0.2\nHigher = popular content dominates feeds.',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: 'k_exp',
    label: 'Exposure cap',
    tooltip: 'Maximum content items shown per agent per tick.\nRange: 1–100\nDefault: 20',
    type: 'number',
    min: 1,
    max: 100,
  },
  {
    key: 'cf_blend_ratio',
    label: 'CF blend ratio',
    tooltip: 'Fraction of feed from collaborative filtering when using hybrid recommender.\nRange: 0–1\nDefault: 0.5',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
]

const SIR_PARAMS: ConfigParamDef[] = [
  {
    key: 'sir_beta',
    label: 'Transmission rate β',
    tooltip: 'Probability a Susceptible agent becomes Infected per exposure to misinformation.\nRange: 0–1\nDefault: 0.3\nHigher = misinformation spreads faster.',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: 'sir_gamma',
    label: 'Recovery rate γ',
    tooltip: 'Probability an Infected agent Recovers per tick (fact-checking / skepticism).\nRange: 0–1\nDefault: 0.05\nHigher = faster recovery from misinformation.',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: 'reinforcement_factor',
    label: 'Reinforcement factor',
    tooltip: 'How much repeated exposure boosts infection probability.\nRange: 0–1\nDefault: 0.0\nHigher = seeing misinformation multiple times makes infection more likely.',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
]

const EMOTION_PARAMS: ConfigParamDef[] = [
  {
    key: 'emotional_decay',
    label: 'Emotional decay rate λ',
    tooltip: 'How quickly emotional arousal fades per tick.\nRange: 0–1\nDefault: 0.85\nHigher = arousal fades slower.',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: 'arousal_share_weight',
    label: 'Arousal → share weight',
    tooltip: 'How much an agent\'s emotional arousal boosts sharing probability.\nRange: 0–1\nDefault: 0.3',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: 'valence_share_weight',
    label: 'Valence → share weight',
    tooltip: 'How much content emotional intensity boosts sharing probability.\nRange: 0–1\nDefault: 0.4',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: 'arousal_tolerance_effect',
    label: 'Arousal → tolerance effect γₑ',
    tooltip: 'How much arousal narrows an agent\'s confidence bound (making them more closed-minded).\nRange: 0–1\nDefault: 0.4\nHigher = aroused agents ignore more of their neighbors.',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
]

const INTERVENTION_PARAMS: ConfigParamDef[] = [
  {
    key: 'diversity_ratio',
    label: 'Diversity ratio',
    tooltip: 'Fraction of feed reserved for diverse/opposing content.\nRange: 0–1\nDefault: 0.0\n0 = no diversity injection. Higher = more cross-cutting exposure.',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: 'lambda_penalty',
    label: 'Misinfo penalty λ',
    tooltip: 'How much misinformation content is penalized in feed rankings.\nRange: 0–1\nDefault: 0.0\n0 = no penalty. Higher = misinformation ranked lower.',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: 'virality_dampening',
    label: 'Virality dampening',
    tooltip: 'How much viral/emotional content sharing is suppressed.\nRange: 0–1\nDefault: 0.0\nHigher = less viral sharing.',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: 'media_literacy_boost',
    label: 'Media literacy boost',
    tooltip: 'Flat increase to all agents\' media literacy.\nRange: 0–1\nDefault: 0.0\nHigher = population better at spotting misinformation.',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: 'rate_limit_factor',
    label: 'Rate limit factor',
    tooltip: 'How much flagged/bot accounts are rate-limited.\nRange: 0–1\nDefault: 0.0\nHigher = flagged accounts post less.',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
]

const BOT_PARAMS: ConfigParamDef[] = [
  {
    key: 'T_detect',
    label: 'Detection interval',
    tooltip: 'How often (in ticks) the bot detection module runs.\nDefault: 24\nScans all agents for behavioral bot signals.',
    type: 'number',
    min: 1,
    max: 200,
  },
  {
    key: 's_thresh',
    label: 'Suspicion threshold',
    tooltip: 'Suspicion score above which an agent is flagged as a bot.\nRange: 0–1\nDefault: 0.7\n≥ 0.7 = flagged.',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: 'p_detect_remove',
    label: 'Detection removal rate',
    tooltip: 'Probability a flagged bot is removed when detected.\nRange: 0–1\nDefault: 0.0\n0 = never remove. Higher = more aggressive removal.',
    type: 'slider',
    min: 0,
    max: 1,
    step: 0.01,
  },
]

const CHURN_PARAMS: ConfigParamDef[] = [
  {
    key: 'enable_churn',
    label: 'Enable churn',
    tooltip: 'Whether dissatisfied agents can leave the platform.\nDefault: false\nWhen enabled, agents with high dissatisfaction may become inactive.',
    type: 'boolean',
  },
  {
    key: 'churn_base',
    label: 'Churn base log-odds',
    tooltip: 'Base log-odds of churning.\nDefault: -4.0\nMore negative = less churn.',
    type: 'number',
    min: -10,
    max: 0,
    step: 0.1,
  },
  {
    key: 'churn_weight',
    label: 'Churn weight',
    tooltip: 'How strongly dissatisfaction drives churn.\nDefault: 1.0\nHigher = dissatisfaction matters more.',
    type: 'number',
    min: 0,
    max: 10,
    step: 0.1,
  },
]

export const CONFIG_SECTIONS: ConfigSectionDef[] = [
  { key: 'network', label: 'Network', params: NETWORK_PARAMS },
  { key: 'population', label: 'Population', params: POPULATION_PARAMS },
  { key: 'simulation', label: 'Simulation', params: SIM_PARAMS },
  { key: 'recommendation', label: 'Recommendation', params: REC_PARAMS },
  { key: 'sir', label: 'SIR / Misinformation', params: SIR_PARAMS },
  { key: 'emotion', label: 'Emotion', params: EMOTION_PARAMS },
  { key: 'intervention', label: 'Interventions', params: INTERVENTION_PARAMS },
  { key: 'bot', label: 'Bot Detection', params: BOT_PARAMS },
  { key: 'churn', label: 'Churn', params: CHURN_PARAMS },
]

export const AGENT_TYPES = [
  { key: 'stubborn', label: 'Stubborn (FJ)', color: '#8b9dc3' },
  { key: 'flexible', label: 'Flexible', color: '#9bc88b' },
  { key: 'passive', label: 'Passive', color: '#a0a0a0' },
  { key: 'zealot', label: 'Zealot', color: '#e0556a' },
  { key: 'bot', label: 'Bot', color: '#f0a060' },
  { key: 'hk', label: 'HK Bounded', color: '#7bc8c8' },
  { key: 'contrarian', label: 'Contrarian', color: '#c8a0e0' },
  { key: 'influencer', label: 'Influencer', color: '#f0d060' },
] as const
