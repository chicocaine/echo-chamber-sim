"""Main simulation loop for MVP Step 7.

Implements the nine-step tick loop in the exact order defined by the
implementation plan, with explicit stubs for future phases.
"""

from __future__ import annotations

from dataclasses import asdict
import random
from statistics import fmean
from typing import Any

from joblib import Parallel, delayed
import numpy as np

from .agent import Agent, initialize_agents
from .content import Content, maybe_generate_content
from .metrics import compute_all_metrics
from .network import (
    build_network,
    get_graph_snapshot,
    get_influence_weights,
    get_predecessors,
)
from .recommender import generate_feed_vectorized


try:
    profile  # type: ignore[name-defined]
except NameError:
    def profile(func):
        """No-op decorator when line_profiler is not active."""
        return func


# PERF BASELINE (recorded before optimization):
# run_simulation N=1000 T=100: 159.48s total (kernprof baseline)
# generate_feed: 97.7% of total time (~155.92s)
# opinion_update: 2.1% of total time (~3.36s)
# Baseline command: `kernprof -l sim/simulation.py`
# PERF OPT (update after each optimization pass):
# run_simulation N=200 T=200: 4.44s (local benchmark)
# run_simulation N=1000 T=100: 22.47s (local benchmark)
# run_simulation N=1000 T=720: 175.71s (~2.93 min, local benchmark)
# profile command: `kernprof -l -m sim.simulation`


BASE_SHARE_PROB = 0.18
FEED_INFLUENCE_MAX = 0.35

DEFAULT_CONFIG: dict[str, Any] = {
    "N": 200,
    "avg_degree": 16,
    "rewire_prob": 0.1,
    "T": 200,
    "snapshot_interval": 6,
    "alpha": 0.65,
    "beta_pop": 0.2,
    "k_exp": 20,
    "agent_mix": {
        "stubborn": 0.60,
        "flexible": 0.20,
        "zealot": 0.15,
        "bot": 0.05,
    },
    "sir_beta": 0.3,
    "sir_gamma": 0.05,
    "initial_opinion_distribution": "uniform",
    "seed": 42,
}


def _agent_tick_seed(base_seed: int, tick: int, agent_id: int) -> int:
    """Derive deterministic random seeds safe for threaded execution."""
    return int(base_seed + tick * 1_000_003 + agent_id * 9_176)


def _process_agent_tick(
    agent: Agent,
    feed: list[Content],
    neighbors: list[Agent],
    weights: dict[int, float],
    alpha: float,
    random_seed: int,
) -> tuple[int, float, float, list[Content]]:
    """Process Step 3/4/5 for one agent without shared-state mutation."""
    # STUB: Phase 2 consumption/arousal model.
    new_arousal = 0.0

    local_rng = random.Random(random_seed)
    shared: list[Content] = []
    for content in feed:
        # STUB: Phase 2 sigmoid sharing probability.
        if local_rng.random() < BASE_SHARE_PROB:
            shared.append(content)

    social_update = float(agent.compute_update(neighbors, weights))
    if agent.stubbornness >= 1.0 or not feed:
        return (agent.id, social_update, new_arousal, shared)

    feed_mean_ideology = fmean(content.ideological_score for content in feed)
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
        "T",
        "snapshot_interval",
        "alpha",
        "beta_pop",
        "k_exp",
        "agent_mix",
        "sir_beta",
        "sir_gamma",
        "initial_opinion_distribution",
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
    _assert_probability("alpha", float(config["alpha"]))
    _assert_probability("beta_pop", float(config["beta_pop"]))
    _assert_probability("sir_beta", float(config["sir_beta"]))
    _assert_probability("sir_gamma", float(config["sir_gamma"]))

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


def _network_rewiring_step() -> None:
    """Placeholder for Phase 4 rewiring logic."""
    # STUB: Phase 4 — Network rewiring step
    # See implementation plan Phase 4, Step 4.1 for full implementation.
    pass


def _churn_step() -> None:
    """Placeholder for Phase 4 churn logic."""
    # STUB: Phase 4 — Churn step
    # See implementation plan Phase 4, Step 4.2 for full implementation.
    pass


def _bot_detection_step() -> None:
    """Placeholder for Phase 5 bot detection logic."""
    # STUB: Phase 5 — Bot detection step
    # See implementation plan Phase 5, Step 5.4 for full implementation.
    pass


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
    )
    G = build_network(
        agents=agents,
        avg_degree=int(merged_config["avg_degree"]),
        rewire_prob=float(merged_config["rewire_prob"]),
        seed=int(merged_config["seed"]),
    )

    content_id_counter = 0
    previous_shared_content: dict[int, list[Content]] = {}
    snapshots: list[dict[str, Any]] = []

    # MVP network is static (rewiring/churn are stubs), so cache predecessor-derived
    # structures once instead of rebuilding them for every agent every tick.
    predecessor_ids_by_agent: dict[int, list[int]] = {
        agent.id: get_predecessors(G, agent.id) for agent in agents
    }
    neighbors_by_agent: dict[int, list[Agent]] = {
        agent.id: [G.nodes[node_id]["agent"] for node_id in predecessor_ids_by_agent[agent.id]]
        for agent in agents
    }
    influence_weights_by_agent: dict[int, dict[int, float]] = {
        agent.id: get_influence_weights(G, agent.id) for agent in agents
    }

    total_ticks = int(merged_config["T"])
    snapshot_interval = int(merged_config["snapshot_interval"])
    k_exp = int(merged_config["k_exp"])
    alpha = float(merged_config["alpha"])
    beta_pop = float(merged_config["beta_pop"])
    sir_beta = float(merged_config["sir_beta"])
    sir_gamma = float(merged_config["sir_gamma"])

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
                content_id_counter += 1

        # Step 2: FEED GENERATION
        def _generate_agent_feed(agent: Agent) -> tuple[int, list[Content]]:
            predecessor_ids = predecessor_ids_by_agent[agent.id]
            candidate_pool: list[Content] = list(current_tick_pool)
            for predecessor_id in predecessor_ids:
                candidate_pool.extend(previous_shared_content.get(predecessor_id, []))

            if not candidate_pool:
                return (agent.id, [])

            content_ideo_array = np.fromiter(
                (content.ideological_score for content in candidate_pool),
                dtype=np.float64,
                count=len(candidate_pool),
            )
            content_virality_array = np.fromiter(
                (content.virality for content in candidate_pool),
                dtype=np.float64,
                count=len(candidate_pool),
            )
            feed = generate_feed_vectorized(
                agent=agent,
                candidate_pool=candidate_pool,
                content_ideo_array=content_ideo_array,
                content_virality_array=content_virality_array,
                k_exp=k_exp,
                alpha=alpha,
                beta_pop=beta_pop,
            )
            return (agent.id, feed)

        feed_pairs = [_generate_agent_feed(agent) for agent in active_agents]
        feeds: dict[int, list[Content]] = dict(feed_pairs)

        # Step 3/4/5: compute per-agent outcomes, then synchronized apply.
        # For smaller populations, serial execution avoids joblib scheduling overhead.
        if len(active_agents) < 300:
            step_results = [
                _process_agent_tick(
                    agent=agent,
                    feed=feeds.get(agent.id, []),
                    neighbors=neighbors_by_agent[agent.id],
                    weights=influence_weights_by_agent[agent.id],
                    alpha=alpha,
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

        # SIR transitions (triggered by misinformation in consumed feed).
        for agent in active_agents:
            if agent.sir_state == "S":
                feed = consumed_items.get(agent.id, [])
                if not feed:
                    continue
                misinfo_count = sum(1 for content in feed if content.is_misinformation)
                if misinfo_count <= 0:
                    continue

                # Exposure-scaled infection risk avoids immediate saturation at tick 0
                # while preserving higher risk under heavier misinformation exposure.
                misinfo_fraction = misinfo_count / max(1, len(feed))
                infection_prob = sir_beta * misinfo_fraction
                if py_rng.random() < infection_prob:
                    agent.sir_state = "I"
            elif agent.sir_state == "I":
                if py_rng.random() < sir_gamma:
                    agent.sir_state = "R"

        # Step 6: NETWORK REWIRING (MVP stub)
        _network_rewiring_step()

        # Step 7: CHURN CHECK (MVP stub)
        _churn_step()

        # Step 8: BOT DETECTION (MVP stub)
        _bot_detection_step()

        # Step 9: METRIC LOGGING
        if tick % snapshot_interval == 0:
            snapshots.append(compute_all_metrics(G, agents, tick=tick))

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


if __name__ == "__main__":
    result = run_simulation()
    print(
        {
            "snapshot_count": len(result["snapshots"]),
            "agent_count": len(result["final_agents"]),
            "edge_count": len(result["final_graph"]["edges"]),
        }
    )
