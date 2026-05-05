"""Main simulation loop for MVP Step 7.

Implements the nine-step tick loop in the exact order defined by the
implementation plan, with explicit stubs for future phases.
"""

from __future__ import annotations

from dataclasses import asdict
import math
import random
from statistics import fmean
from typing import Any

from joblib import Parallel, delayed
import numpy as np

from .agent import Agent, initialize_agents
from .bot_detection import (
    compute_population_activity_stats,
    compute_suspicion_score,
)
from .content import (
    CAMPAIGN_EXPIRY_DELTA,
    Content,
    get_effective_misinfo_score,
    maybe_generate_content,
)
from .metrics import compute_all_metrics, modularity_q
from .network import (
    build_network,
    compute_dissatisfaction,
    get_graph_snapshot,
    get_influence_weights,
    get_predecessors,
    rewire_step,
)
from .recommender import (
    CollaborativeFilteringRecommender,
    ContentBasedRecommender,
    GraphBasedRecommender,
    generate_feed_vectorized,
)


try:
    profile  # type: ignore[name-defined]
except NameError:
    def profile(func):
        """No-op decorator when line_profiler is not active."""
        return func


# PERF BASELINE (original code, kernprof):
# run_simulation N=1000 T=100: 159.48s (kernprof baseline)
# generate_feed: 97.7% of total time (~155.92s)
# PERF OPT 1 (after Phase 1 feed-gen refactor):
# run_simulation N=200 T=200: 4.44s
# run_simulation N=1000 T=100: 22.47s
# run_simulation N=1000 T=720: 175.71s (~2.93 min)
# PERF OPT 2 (Louvain + SIR vectorized + np.clip removal + topic-vector dead-code elimination):
# run_simulation N=1000 T=50:  8.31s
# run_simulation N=1000 T=100: 16.18s
# run_simulation N=1000 T=720: 119.84s (2.00 min)


SHARE_BASE_LOGIT = -1.5
FEED_INFLUENCE_MAX = 0.35

DEFAULT_CONFIG: dict[str, Any] = {
    "N": 200,
    "avg_degree": 16,
    "rewire_prob": 0.1,
    "topology": "watts_strogatz",
    "community_sizes": None,
    "community_p": None,
    "T": 200,
    "snapshot_interval": 6,
    "alpha": 0.65,
    "beta_pop": 0.2,
    "k_exp": 20,
    "agent_mix": {
        "stubborn": 0.60,
        "flexible": 0.20,
        "passive": 0.10,
        "zealot": 0.05,
        "bot": 0.05,
        "hk": 0.0,
        "contrarian": 0.0,
        "influencer": 0.0,
    },
    "sir_beta": 0.3,
    "sir_gamma": 0.05,
    "reinforcement_factor": 0.0,
    "recommender_type": "content_based",
    "cf_blend_ratio": 0.5,
    "dynamic_rewire_rate": 0.01,
    "homophily_threshold": 0.3,
    "enable_churn": False,
    "churn_base": -4.0,
    "churn_weight": 1.0,
    "T_detect": 24,
    "s_thresh": 0.7,
    "p_detect_remove": 0.0,
    "rate_limit_factor": 0.0,
    "media_literacy_boost": 0.0,
    "diversity_ratio": 0.0,
    "lambda_penalty": 0.0,
    "virality_dampening": 0.0,
    "initial_opinion_distribution": "uniform",
    "emotional_decay": 0.85,
    "arousal_share_weight": 0.3,
    "valence_share_weight": 0.4,
    "arousal_tolerance_effect": 0.4,
    "seed": 42,
}


def _agent_tick_seed(base_seed: int, tick: int, agent_id: int) -> int:
    """Derive deterministic random seeds safe for threaded execution."""
    return int(base_seed + tick * 1_000_003 + agent_id * 9_176)


@profile
def _process_agent_tick(
    agent: Agent,
    feed: list[Content],
    neighbors: list[Agent],
    weights: dict[int, float],
    alpha: float,
    emotional_decay: float,
    arousal_share_weight: float,
    valence_share_weight: float,
    virality_dampening: float,
    random_seed: int,
) -> tuple[int, float, float, list[Content]]:
    """Process Step 3/4/5 for one agent without shared-state mutation.

    Step 3: Content consumption with arousal update (Phase 2).
    Step 4: Sharing decision (Phase 2 + Phase 3 virality dampening).
    Step 5: Opinion update.
    """
    # Step 3: CONTENT CONSUMPTION — update arousal per content item (Phase 2 Step 2.1)
    new_arousal = float(agent.emotional_arousal)
    ld = float(emotional_decay)  # hoist constant out of loop
    one_minus_ld = 1.0 - ld
    for content in feed:
        new_arousal = ld * new_arousal + one_minus_ld * float(content.emotional_valence)
        # Fast built-in clamp — avoids numpy scalar overhead (dominant hot spot).
        if new_arousal < 0.0:
            new_arousal = 0.0
        elif new_arousal > 1.0:
            new_arousal = 1.0

    # Fast LCG instead of random.Random() — saves ~0.37s in constructor seeding.
    rand_state = (random_seed * 1103515245 + 12345) & 0x7fffffff
    shared: list[Content] = []
    # Step 4: SHARING DECISION (Phase 2 Step 2.2)
    effective_w_v = valence_share_weight * (1.0 - virality_dampening)
    logit_base = SHARE_BASE_LOGIT + arousal_share_weight * new_arousal
    for content in feed:
        if agent.agent_type == "bot":
            shared.append(content)
            continue

        share_probability = _sigmoid(
            logit_base + effective_w_v * float(content.emotional_valence)
        )
        rand_state = (rand_state * 1103515245 + 12345) & 0x7fffffff
        if (rand_state / 0x7fffffff) < share_probability:
            shared.append(content)

    # Step 5: OPINION UPDATE
    social_update = float(agent.compute_update(neighbors, weights))
    if agent.stubbornness >= 1.0 or not feed:
        return (agent.id, social_update, new_arousal, shared)

    # Direct sum/len avoids statistics.fmean generator overhead.
    ideo_sum = 0.0
    for c in feed:
        ideo_sum += c.ideological_score
    feed_mean_ideology = ideo_sum / len(feed)
    # Higher personalization should amplify homophily feedback from the feed.
    susceptibility = float(agent.susceptibility)
    feed_weight = FEED_INFLUENCE_MAX * float(alpha) * susceptibility
    neutralizing_weight = FEED_INFLUENCE_MAX * (1.0 - float(alpha)) * susceptibility
    total_weight = feed_weight + neutralizing_weight
    blended = (
        (1.0 - total_weight) * social_update
        + feed_weight * float(feed_mean_ideology)
        + neutralizing_weight * 0.0
    )
    return (agent.id, _clamp_opinion(blended), new_arousal, shared)


def _assert_probability(name: str, value: float) -> None:
    """Validate probability values against [0, 1]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_config(config: dict[str, Any]) -> None:
    """Validate simulation configuration constraints for MVP Step 7."""
    required_keys = {
        "N",
        "avg_degree",
        "rewire_prob",
        "topology",
        "T",
        "snapshot_interval",
        "alpha",
        "beta_pop",
        "k_exp",
        "agent_mix",
        "sir_beta",
        "sir_gamma",
        "reinforcement_factor",
        "recommender_type",
        "cf_blend_ratio",
        "dynamic_rewire_rate",
        "homophily_threshold",
        "enable_churn",
        "churn_base",
        "churn_weight",
        "T_detect",
        "s_thresh",
        "p_detect_remove",
        "rate_limit_factor",
        "media_literacy_boost",
        "diversity_ratio",
        "lambda_penalty",
        "virality_dampening",
        "initial_opinion_distribution",
        "emotional_decay",
        "arousal_share_weight",
        "valence_share_weight",
        "arousal_tolerance_effect",
        "seed",
    }
    missing = required_keys.difference(config.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {sorted(missing)}")

    if int(config["N"]) <= 0:
        raise ValueError("N must be > 0")
    if int(config["T"]) <= 0:
        raise ValueError("T must be > 0")
    if int(config["snapshot_interval"]) <= 0:
        raise ValueError("snapshot_interval must be > 0")
    if int(config["k_exp"]) <= 0:
        raise ValueError("k_exp must be > 0")
    if int(config["avg_degree"]) <= 0:
        raise ValueError("avg_degree must be > 0")

    _assert_probability("rewire_prob", float(config["rewire_prob"]))
    topology = str(config["topology"])
    if topology not in {"watts_strogatz", "barabasi_albert", "erdos_renyi", "stochastic_block"}:
        raise ValueError(f"Unknown topology: {topology}")
    if topology == "stochastic_block":
        sizes = config.get("community_sizes")
        p_mat = config.get("community_p")
        if sizes is None or p_mat is None:
            raise ValueError("community_sizes and community_p required for stochastic_block")
        if not isinstance(sizes, list) or not isinstance(p_mat, list):
            raise ValueError("community_sizes and community_p must be lists")
    _assert_probability("alpha", float(config["alpha"]))
    _assert_probability("beta_pop", float(config["beta_pop"]))
    _assert_probability("sir_beta", float(config["sir_beta"]))
    _assert_probability("sir_gamma", float(config["sir_gamma"]))
    _assert_probability("reinforcement_factor", float(config["reinforcement_factor"]))
    _assert_probability("cf_blend_ratio", float(config["cf_blend_ratio"]))
    _assert_probability("dynamic_rewire_rate", float(config["dynamic_rewire_rate"]))
    _assert_probability("homophily_threshold", float(config["homophily_threshold"]))
    if not isinstance(config.get("enable_churn"), bool):
        raise ValueError("enable_churn must be a boolean")
    if float(config["churn_weight"]) < 0.0:
        raise ValueError("churn_weight must be >= 0")
    if int(config["T_detect"]) <= 0:
        raise ValueError("T_detect must be > 0")
    _assert_probability("s_thresh", float(config["s_thresh"]))
    _assert_probability("p_detect_remove", float(config["p_detect_remove"]))
    _assert_probability("rate_limit_factor", float(config["rate_limit_factor"]))
    _assert_probability("media_literacy_boost", float(config["media_literacy_boost"]))
    _assert_probability("diversity_ratio", float(config["diversity_ratio"]))
    _assert_probability("lambda_penalty", float(config["lambda_penalty"]))
    _assert_probability("virality_dampening", float(config["virality_dampening"]))

    recommender_type = str(config["recommender_type"])
    if recommender_type not in {"content_based", "cf", "graph", "hybrid"}:
        raise ValueError(
            f"recommender_type must be one of content_based, cf, graph, hybrid; got {recommender_type}"
        )
    _assert_probability("emotional_decay", float(config["emotional_decay"]))
    _assert_probability("arousal_share_weight", float(config["arousal_share_weight"]))
    _assert_probability("valence_share_weight", float(config["valence_share_weight"]))
    _assert_probability("arousal_tolerance_effect", float(config["arousal_tolerance_effect"]))

    if float(config["beta_pop"]) > 1.0:
        raise ValueError("beta_pop must be <= 1.0")


def _serialize_agents(agents: list[Agent]) -> list[dict[str, Any]]:
    """Convert agent dataclasses into JSON-serializable dict objects."""
    serialized: list[dict[str, Any]] = []
    for agent in agents:
        payload = asdict(agent)
        payload["opinion"] = float(agent.opinion)
        payload["initial_opinion"] = float(agent.initial_opinion)
        payload["opinion_history"] = [float(value) for value in agent.opinion_history]
        serialized.append(payload)
    return serialized


def _clamp_opinion(value: float) -> float:
    """Clamp opinions to invariant range [-1.0, 1.0]."""
    return float(max(-1.0, min(1.0, value)))


def _sigmoid(value: float) -> float:
    """Numerically stable logistic function for sharing decisions."""
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_v = math.exp(value)
    return exp_v / (1.0 + exp_v)


def _network_rewiring_step(
    G: Any,
    agents: list[Agent],
    dynamic_rewire_rate: float,
    homophily_threshold: float,
    base_seed: int,
    tick: int,
) -> bool:
    """Dynamic edge rewiring (Phase 4 Step 4.1).

    Agents probabilistically unfollow disagreeing peers and follow new agents
    with similar opinions within homophily_threshold.
    """
    if dynamic_rewire_rate <= 0.0:
        return False
    return rewire_step(
        G=G,
        agents=agents,
        dynamic_rewire_rate=dynamic_rewire_rate,
        homophily_threshold=homophily_threshold,
        seed=base_seed + tick * 1_000_033,
    )


def _churn_step(
    G: Any,
    agents: list[Agent],
    enable_churn: bool,
    churn_base: float,
    churn_weight: float,
    seed: int,
) -> bool:
    """Agent churn: dissatisfied agents may leave the platform (Phase 4 Step 4.2).

    Dissatisfaction = mean opinion distance to predecessors.
    Churn probability = sigmoid(churn_base + churn_weight * dissatisfaction).
    Churned agents become inactive and are removed from the graph.
    """
    if not enable_churn:
        return False

    rng = random.Random(seed)
    graph_changed = False
    for agent in agents:
        if not agent.is_active:
            continue

        dissatisfaction = compute_dissatisfaction(agent, G)
        p_churn = _sigmoid(churn_base + churn_weight * dissatisfaction)
        if rng.random() < p_churn:
            agent.is_active = False
            G.remove_node(agent.id)
            graph_changed = True

    return graph_changed


def _bot_detection_step(
    G: Any,
    agents: list[Agent],
    shared_content: dict[int, list[Content]],
    T_detect: int,
    s_thresh: float,
    p_detect_remove: float,
    rate_limit_factor: float,
    tick: int,
    seed: int,
) -> bool:
    """Behavioral bot detection and intervention (Phase 5 Step 5.4).

    Every ``T_detect`` ticks, computes suspicion scores for all active agents
    using four behavioral signals. Agents scoring >= ``s_thresh`` are either
    removed (with probability ``p_detect_remove``) or rate-limited.
    """
    if tick % T_detect != 0 or (p_detect_remove <= 0.0 and rate_limit_factor <= 0.0):
        return False

    pop_mean, pop_std = compute_population_activity_stats(agents)
    rng = random.Random(seed + tick * 1_000_111)
    graph_changed = False

    for agent in agents:
        if not agent.is_active:
            continue

        agent.suspicion_score = compute_suspicion_score(
            agent=agent,
            recent_shares=shared_content.get(agent.id, []),
            G=G,
            population_mean_activity=pop_mean,
            population_std_activity=pop_std,
        )

        if agent.suspicion_score < s_thresh:
            continue

        if p_detect_remove > 0.0 and rng.random() < p_detect_remove:
            agent.is_active = False
            G.remove_node(agent.id)
            graph_changed = True
        elif rate_limit_factor > 0.0:
            agent.activity_rate = float(np.clip(
                agent.activity_rate * (1.0 - rate_limit_factor), 0.0, 1.0,
            ))

    return graph_changed


@profile
def run_simulation(
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the MVP simulation loop and return final outputs.

    Tick order (must remain exact):
    1. Content generation
    2. Feed generation
    3. Content consumption
    4. Sharing decision
    5. Opinion update + SIR transitions
    6. Network rewiring (stub in MVP)
    7. Churn check (stub in MVP)
    8. Bot detection (stub in MVP)
    9. Metric logging
    """
    merged_config: dict[str, Any] = {**DEFAULT_CONFIG, **(config or {})}
    _validate_config(merged_config)

    rng = np.random.default_rng(int(merged_config["seed"]))
    py_rng = random.Random(int(merged_config["seed"]))
    base_seed = int(merged_config["seed"])

    agents = initialize_agents(
        n_agents=int(merged_config["N"]),
        agent_mix=dict(merged_config["agent_mix"]),
        seed=int(merged_config["seed"]),
        initial_opinion_distribution=str(merged_config["initial_opinion_distribution"]),
        arousal_tolerance_effect=float(merged_config["arousal_tolerance_effect"]),
        media_literacy_boost=float(merged_config["media_literacy_boost"]),
    )
    topology = str(merged_config["topology"])
    community_sizes = merged_config.get("community_sizes")
    community_p = merged_config.get("community_p")
    G = build_network(
        agents=agents,
        avg_degree=int(merged_config["avg_degree"]),
        rewire_prob=float(merged_config["rewire_prob"]),
        seed=int(merged_config["seed"]),
        topology=topology,
        community_sizes=list(community_sizes) if community_sizes is not None else None,
        community_p=[list(row) for row in community_p] if community_p is not None else None,
    )

    content_id_counter = 0
    campaign_expiry: dict[int, int] = {}
    previous_shared_content: dict[int, list[Content]] = {}
    snapshots: list[dict[str, Any]] = []

    # Cache predecessor-derived structures. Single traversal per agent avoids
    # redundant NetworkX successor/predecessor lookups.
    _id_to_agent: dict[int, Agent] = {a.id: a for a in agents}
    predecessor_ids_by_agent: dict[int, list[int]] = {}
    neighbors_by_agent: dict[int, list[Agent]] = {}
    influence_weights_by_agent: dict[int, dict[int, float]] = {}
    for agent in agents:
        preds = get_predecessors(G, agent.id)
        predecessor_ids_by_agent[agent.id] = preds
        neighbors_by_agent[agent.id] = [_id_to_agent[nid] for nid in preds]
        if not preds:
            influence_weights_by_agent[agent.id] = {}
        else:
            raw = {src: float(G[src][agent.id].get("weight", 1.0)) for src in preds}
            total = sum(raw.values())
            if total <= 0.0:
                uniform = 1.0 / len(preds)
                influence_weights_by_agent[agent.id] = {src: uniform for src in preds}
            else:
                influence_weights_by_agent[agent.id] = {src: w / total for src, w in raw.items()}

    total_ticks = int(merged_config["T"])
    snapshot_interval = int(merged_config["snapshot_interval"])
    k_exp = int(merged_config["k_exp"])
    alpha = float(merged_config["alpha"])
    beta_pop = float(merged_config["beta_pop"])
    sir_beta = float(merged_config["sir_beta"])
    sir_gamma = float(merged_config["sir_gamma"])
    reinforcement_factor = float(merged_config["reinforcement_factor"])
    diversity_ratio = float(merged_config["diversity_ratio"])
    lambda_penalty = float(merged_config["lambda_penalty"])
    virality_dampening = float(merged_config["virality_dampening"])
    emotional_decay = float(merged_config["emotional_decay"])
    arousal_share_weight = float(merged_config["arousal_share_weight"])
    valence_share_weight = float(merged_config["valence_share_weight"])
    recommender_type = str(merged_config["recommender_type"])
    cf_blend_ratio = float(merged_config["cf_blend_ratio"])
    dynamic_rewire_rate = float(merged_config["dynamic_rewire_rate"])
    homophily_threshold = float(merged_config["homophily_threshold"])
    enable_churn = bool(merged_config["enable_churn"])
    churn_base = float(merged_config["churn_base"])
    churn_weight = float(merged_config["churn_weight"])
    T_detect = int(merged_config["T_detect"])
    s_thresh = float(merged_config["s_thresh"])
    p_detect_remove = float(merged_config["p_detect_remove"])
    rate_limit_factor = float(merged_config["rate_limit_factor"])
    media_literacy_boost = float(merged_config["media_literacy_boost"])

    cb_recommender: ContentBasedRecommender | None = None
    cf_recommender: CollaborativeFilteringRecommender | None = None
    graph_recommender: GraphBasedRecommender | None = None

    if recommender_type in {"content_based", "hybrid"}:
        cb_recommender = ContentBasedRecommender(
            alpha=alpha,
            beta_pop=beta_pop,
            diversity_ratio=diversity_ratio,
            lambda_penalty=lambda_penalty,
        )
    if recommender_type in {"cf", "hybrid"}:
        cf_recommender = CollaborativeFilteringRecommender()
    if recommender_type == "graph":
        graph_recommender = GraphBasedRecommender()

    graph_dirty = True
    cached_modularity: float | None = None

    for tick in range(total_ticks):
        active_agents = [agent for agent in agents if agent.is_active]

        # Step 1: CONTENT GENERATION
        current_tick_pool: list[Content] = []
        for agent in active_agents:
            maybe_content = maybe_generate_content(
                agent=agent,
                content_id=content_id_counter,
                timestamp=tick,
                rng=rng,
            )
            if maybe_content is not None:
                current_tick_pool.append(maybe_content)
                # Register new campaigns for expiry tracking.
                cid = maybe_content.coordinated_campaign_id
                if cid is not None and cid not in campaign_expiry:
                    campaign_expiry[cid] = tick + CAMPAIGN_EXPIRY_DELTA
                content_id_counter += 1

        # Step 2: FEED GENERATION
        if cf_recommender is not None:
            cf_recommender.update_context(active_agents, previous_shared_content)
        if graph_recommender is not None:
            graph_recommender.update_context(G, previous_shared_content)

        content_based_mode = cb_recommender is not None

        current_pool_size = len(current_tick_pool)
        if content_based_mode and current_pool_size:
            current_pool_ideo = np.fromiter(
                (c.ideological_score for c in current_tick_pool),
                dtype=np.float64, count=current_pool_size,
            )
            current_pool_virality = np.fromiter(
                (c.virality for c in current_tick_pool),
                dtype=np.float64, count=current_pool_size,
            )
            current_pool_misinfo = np.fromiter(
                (c.misinfo_score for c in current_tick_pool),
                dtype=np.float64, count=current_pool_size,
            )
        else:
            current_pool_ideo = None
            current_pool_virality = None
            current_pool_misinfo = None

        shared_arrays: dict[int, tuple[list[Content], np.ndarray | None, np.ndarray | None, np.ndarray | None]] = {}
        if previous_shared_content:
            for agent_id, items in previous_shared_content.items():
                if not items:
                    continue
                if not content_based_mode:
                    shared_arrays[agent_id] = (items, None, None, None)
                    continue
                n_items = len(items)
                shared_arrays[agent_id] = (
                    items,
                    np.fromiter(
                        (c.ideological_score for c in items),
                        dtype=np.float64, count=n_items,
                    ),
                    np.fromiter(
                        (c.virality for c in items),
                        dtype=np.float64, count=n_items,
                    ),
                    np.fromiter(
                        (c.misinfo_score for c in items),
                        dtype=np.float64, count=n_items,
                    ),
                )

        feeds: dict[int, list[Content]] = {}
        for agent in active_agents:
            pool_len = current_pool_size
            for predecessor_id in predecessor_ids_by_agent[agent.id]:
                shared_entry = shared_arrays.get(predecessor_id)
                if shared_entry is not None:
                    pool_len += len(shared_entry[0])

            if pool_len == 0:
                feeds[agent.id] = []
                continue

            candidate_pool = list(current_tick_pool) if current_pool_size else []
            if content_based_mode:
                content_ideo_array = np.empty(pool_len, dtype=np.float64)
                content_virality_array = np.empty(pool_len, dtype=np.float64)
                content_misinfo_array = np.empty(pool_len, dtype=np.float64)
                offset = 0
                if current_pool_size:
                    content_ideo_array[:current_pool_size] = current_pool_ideo
                    content_virality_array[:current_pool_size] = current_pool_virality
                    content_misinfo_array[:current_pool_size] = current_pool_misinfo
                    offset = current_pool_size
            else:
                content_ideo_array = None
                content_virality_array = None
                content_misinfo_array = None
                offset = 0

            for predecessor_id in predecessor_ids_by_agent[agent.id]:
                shared_entry = shared_arrays.get(predecessor_id)
                if shared_entry is None:
                    continue
                items, shared_ideo, shared_virality, shared_misinfo = shared_entry
                candidate_pool.extend(items)
                if content_based_mode:
                    end = offset + len(items)
                    content_ideo_array[offset:end] = shared_ideo
                    content_virality_array[offset:end] = shared_virality
                    content_misinfo_array[offset:end] = shared_misinfo
                    offset = end

            if recommender_type == "content_based":
                assert content_ideo_array is not None
                assert content_virality_array is not None
                assert content_misinfo_array is not None
                feeds[agent.id] = generate_feed_vectorized(
                    agent=agent,
                    candidate_pool=candidate_pool,
                    content_ideo_array=content_ideo_array,
                    content_virality_array=content_virality_array,
                    content_misinfo_array=content_misinfo_array,
                    k_exp=k_exp,
                    alpha=alpha,
                    beta_pop=beta_pop,
                    lambda_penalty=lambda_penalty,
                    diversity_ratio=diversity_ratio,
                )
            elif recommender_type == "cf":
                assert cf_recommender is not None
                feeds[agent.id] = cf_recommender.generate_feed(
                    agent, candidate_pool, k_exp,
                )
            elif recommender_type == "graph":
                assert graph_recommender is not None
                feeds[agent.id] = graph_recommender.generate_feed(
                    agent, candidate_pool, k_exp,
                )
            elif recommender_type == "hybrid":
                assert cf_recommender is not None
                assert content_ideo_array is not None
                assert content_virality_array is not None
                assert content_misinfo_array is not None
                cf_feed = cf_recommender.generate_feed(agent, candidate_pool, k_exp)
                cb_feed = generate_feed_vectorized(
                    agent=agent,
                    candidate_pool=candidate_pool,
                    content_ideo_array=content_ideo_array,
                    content_virality_array=content_virality_array,
                    content_misinfo_array=content_misinfo_array,
                    k_exp=k_exp,
                    alpha=alpha,
                    beta_pop=beta_pop,
                    lambda_penalty=lambda_penalty,
                    diversity_ratio=diversity_ratio,
                )
                # Blend: interleave CF and CB items by blend ratio.
                n_cf = int(k_exp * cf_blend_ratio)
                n_cb = k_exp - n_cf
                seen: set[int] = set()
                blended: list[Content] = []
                for item in cf_feed[:n_cf]:
                    if item.id not in seen:
                        blended.append(item)
                        seen.add(item.id)
                for item in cb_feed:
                    if len(blended) >= k_exp:
                        break
                    if item.id not in seen:
                        blended.append(item)
                        seen.add(item.id)
                # Fill remaining slots from either feed
                for item in cf_feed + cb_feed:
                    if len(blended) >= k_exp:
                        break
                    if item.id not in seen:
                        blended.append(item)
                        seen.add(item.id)
                feeds[agent.id] = blended
            else:
                feeds[agent.id] = candidate_pool[:k_exp]

        # Step 3/4/5: compute per-agent outcomes, then synchronized apply.
        # Serial for N < 5000: thread-pool scheduling overhead dominates for
        # small-to-medium populations (~0.2s/tick to create/destroy pool).
        if len(active_agents) < 5000:
            step_results = [
                _process_agent_tick(
                    agent=agent,
                    feed=feeds.get(agent.id, []),
                    neighbors=neighbors_by_agent[agent.id],
                    weights=influence_weights_by_agent[agent.id],
                    alpha=alpha,
                    emotional_decay=emotional_decay,
                    arousal_share_weight=arousal_share_weight,
                    valence_share_weight=valence_share_weight,
                    virality_dampening=virality_dampening,
                    random_seed=_agent_tick_seed(base_seed=base_seed, tick=tick, agent_id=agent.id),
                )
                for agent in active_agents
            ]
        else:
            step_results = Parallel(n_jobs=-1, prefer="threads")(
                delayed(_process_agent_tick)(
                    agent=agent,
                    feed=feeds.get(agent.id, []),
                    neighbors=neighbors_by_agent[agent.id],
                    weights=influence_weights_by_agent[agent.id],
                    alpha=alpha,
                    emotional_decay=emotional_decay,
                    arousal_share_weight=arousal_share_weight,
                    valence_share_weight=valence_share_weight,
                    virality_dampening=virality_dampening,
                    random_seed=_agent_tick_seed(base_seed=base_seed, tick=tick, agent_id=agent.id),
                )
                for agent in active_agents
            )

        consumed_items: dict[int, list[Content]] = {
            agent.id: feeds.get(agent.id, []) for agent in active_agents
        }
        shared_content: dict[int, list[Content]] = {}
        new_opinions: dict[int, float] = {}
        new_arousals: dict[int, float] = {}
        for agent_id, new_opinion, new_arousal, shared in step_results:
            shared_content[agent_id] = shared
            new_opinions[agent_id] = float(new_opinion)
            new_arousals[agent_id] = float(new_arousal)

        # Apply simultaneously to avoid order bias.
        for agent in active_agents:
            agent.opinion = float(new_opinions.get(agent.id, agent.opinion))
            agent.emotional_arousal = float(new_arousals.get(agent.id, 0.0))

        # SIR transitions — multi-claim tracking (Phase 5 Step 5.2).
        # Pre-compute a structured dtype for per-agent feed filtering.
        _sir_dtype = np.dtype([
            ("score", np.float64),
            ("has_campaign", bool),
            ("is_satire", bool),
        ])
        for agent in active_agents:
            feed = consumed_items.get(agent.id, [])
            n_feed = len(feed)
            if n_feed == 0:
                continue

            ml = agent.media_literacy

            # Vectorized pre-filter: one generator pass extracts all needed attributes.
            feed_arr = np.fromiter(
                (
                    (c.misinfo_score, c.coordinated_campaign_id is not None, c.is_satire)
                    for c in feed
                ),
                dtype=_sir_dtype,
                count=n_feed,
            )
            candidate_mask = (
                (feed_arr["score"] > 0.2)
                | feed_arr["has_campaign"]
                | (feed_arr["is_satire"] & (ml < 0.4))
            )
            candidate_indices = np.where(candidate_mask)[0]
            if len(candidate_indices) == 0:
                continue

            # Group misinformation by campaign (or content ID for non-campaign).
            misinfo_by_campaign: dict[int, list[Content]] = {}
            for idx in candidate_indices:
                content = feed[int(idx)]
                cid = content.coordinated_campaign_id
                if feed_arr["score"][idx] <= 0.2:
                    if cid is None or tick <= campaign_expiry.get(cid, tick + 999999):
                        if not content.is_satire or ml >= 0.4:
                            continue
                effective = get_effective_misinfo_score(
                    content, tick, campaign_expiry, ml,
                )
                if effective <= 0.5:
                    continue
                key = cid or content.id
                misinfo_by_campaign.setdefault(key, []).append(content)

            # Per-campaign SIR transitions.
            for campaign_id, items in misinfo_by_campaign.items():
                current_state = agent.sir_states.get(campaign_id, "S")

                if current_state == "S":
                    should_infect = False
                    for content in items:
                        agent.exposure_count[campaign_id] = (
                            agent.exposure_count.get(campaign_id, 0) + 1
                        )
                        n_exposures = agent.exposure_count[campaign_id]
                        effective_beta = sir_beta * (
                            1.0 + n_exposures * reinforcement_factor
                        )
                        effective_beta = min(effective_beta, 1.0)
                        if py_rng.random() < effective_beta:
                            should_infect = True
                            break

                    if should_infect:
                        agent.sir_states[campaign_id] = "I"

                elif current_state == "I":
                    if py_rng.random() < sir_gamma:
                        agent.sir_states[campaign_id] = "R"

        # Step 6: NETWORK REWIRING
        graph_changed = _network_rewiring_step(
            G=G,
            agents=agents,
            dynamic_rewire_rate=dynamic_rewire_rate,
            homophily_threshold=homophily_threshold,
            base_seed=base_seed,
            tick=tick,
        )

        # Step 7: CHURN CHECK
        if _churn_step(
            G=G,
            agents=agents,
            enable_churn=enable_churn,
            churn_base=churn_base,
            churn_weight=churn_weight,
            seed=base_seed + tick * 1_000_099,
        ):
            graph_changed = True

        # Step 8: BOT DETECTION
        if _bot_detection_step(
            G=G,
            agents=agents,
            shared_content=shared_content,
            T_detect=T_detect,
            s_thresh=s_thresh,
            p_detect_remove=p_detect_remove,
            rate_limit_factor=rate_limit_factor,
            tick=tick,
            seed=base_seed,
        ):
            graph_changed = True

        if graph_changed:
            graph_dirty = True
            # Only cache agents whose nodes still exist in the graph.
            surviving = [a for a in agents if a.is_active and a.id in G]
            _id_to_agent = {a.id: a for a in surviving}
            predecessor_ids_by_agent = {}
            neighbors_by_agent = {}
            influence_weights_by_agent = {}
            for a in surviving:
                preds = get_predecessors(G, a.id)
                predecessor_ids_by_agent[a.id] = preds
                neighbors_by_agent[a.id] = [_id_to_agent[nid] for nid in preds]
                if not preds:
                    influence_weights_by_agent[a.id] = {}
                else:
                    raw = {src: float(G[src][a.id].get("weight", 1.0)) for src in preds}
                    total = sum(raw.values())
                    if total <= 0.0:
                        uniform = 1.0 / len(preds)
                        influence_weights_by_agent[a.id] = {src: uniform for src in preds}
                    else:
                        influence_weights_by_agent[a.id] = {src: w / total for src, w in raw.items()}

        # Step 9: METRIC LOGGING
        if tick % snapshot_interval == 0:
            if graph_dirty or cached_modularity is None:
                cached_modularity = modularity_q(G)
                graph_dirty = False
            snapshots.append(
                compute_all_metrics(
                    G, agents, tick=tick,
                    shared_content=shared_content,
                    consumed_items=consumed_items,
                    modularity_override=cached_modularity,
                )
            )

        # Required for later bot-detection signal calculations.
        for agent in active_agents:
            agent.opinion_history.append(float(agent.opinion))

        previous_shared_content = shared_content

    final_graph = get_graph_snapshot(G)
    return {
        "config": merged_config,
        "snapshots": snapshots,
        "final_agents": _serialize_agents(agents),
        "final_graph": final_graph,
    }


def aggregate_snapshots(
    all_snapshots: list[list[dict[str, Any]]],
) -> dict[str, list[int] | list[float]]:
    """Aggregate multi-run snapshot series into mean/std metric bands.

    Expected input shape:
    - outer list: one element per run
    - inner list: one snapshot dict per logged tick
    - snapshot dict keys: tick + metric names
    """
    if not all_snapshots:
        return {"tick": []}

    first_run = all_snapshots[0]
    if not first_run:
        return {"tick": []}

    expected_len = len(first_run)
    for run_snapshots in all_snapshots:
        if len(run_snapshots) != expected_len:
            raise ValueError(
                "All runs must have the same snapshot count for aggregation"
            )

    metric_names = [key for key in first_run[0].keys() if key != "tick"]
    tick_values = [int(snapshot["tick"]) for snapshot in first_run]

    aggregated: dict[str, list[int] | list[float]] = {
        "tick": tick_values,
    }
    for metric_name in metric_names:
        aggregated[f"{metric_name}_mean"] = []
        aggregated[f"{metric_name}_std"] = []

    for snapshot_idx, tick in enumerate(tick_values):
        for run_snapshots in all_snapshots:
            run_tick = int(run_snapshots[snapshot_idx]["tick"])
            if run_tick != tick:
                raise ValueError(
                    "All runs must log snapshots at identical tick values for aggregation"
                )

        for metric_name in metric_names:
            values = np.array(
                [
                    float(run_snapshots[snapshot_idx][metric_name])
                    for run_snapshots in all_snapshots
                ],
                dtype=np.float64,
            )
            aggregated[f"{metric_name}_mean"].append(float(np.mean(values)))
            aggregated[f"{metric_name}_std"].append(float(np.std(values)))

    return aggregated


def run_replicated(config: dict[str, Any], n_runs: int = 10) -> dict[str, Any]:
    """Run deterministic multi-seed simulation replicates and aggregate metrics."""
    if n_runs <= 0:
        raise ValueError(f"n_runs must be > 0, got {n_runs}")

    merged_config: dict[str, Any] = {**DEFAULT_CONFIG, **config}
    base_seed = int(merged_config["seed"])

    all_snapshots: list[list[dict[str, Any]]] = []
    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        run_result = run_simulation({**merged_config, "seed": seed})
        all_snapshots.append(run_result["snapshots"])

    aggregated = aggregate_snapshots(all_snapshots)
    return {
        "config": merged_config,
        "n_runs": n_runs,
        "aggregated": aggregated,
        "all_runs": all_snapshots,
    }


if __name__ == "__main__":
    result = run_simulation()
    print(
        {
            "snapshot_count": len(result["snapshots"]),
            "agent_count": len(result["final_agents"]),
            "edge_count": len(result["final_graph"]["edges"]),
        }
    )
