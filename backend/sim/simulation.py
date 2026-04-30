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


SHARE_BASE_LOGIT = -1.5
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


def _process_agent_tick(
    agent: Agent,
    feed: list[Content],
    neighbors: list[Agent],
    weights: dict[int, float],
    alpha: float,
    emotional_decay: float,
    arousal_share_weight: float,
    valence_share_weight: float,
    random_seed: int,
) -> tuple[int, float, float, list[Content]]:
    """Process Step 3/4/5 for one agent without shared-state mutation.

    Step 3: Content consumption with arousal update (Phase 2).
    Step 4: Sharing decision.
    Step 5: Opinion update.
    """
    # Step 3: CONTENT CONSUMPTION — update arousal per content item (Phase 2 Step 2.1)
    new_arousal = float(agent.emotional_arousal)
    for content in feed:
        # Arousal decay + valence injection: e_i(t+1) = λ * e_i(t) + (1 - λ) * v_c
        lambda_decay = float(emotional_decay)
        new_arousal = lambda_decay * new_arousal + (1.0 - lambda_decay) * float(
            content.emotional_valence
        )
        # Clamp to [0, 1]
        new_arousal = float(np.clip(new_arousal, 0.0, 1.0))

    local_rng = random.Random(random_seed)
    shared: list[Content] = []
    # Step 4: SHARING DECISION (Phase 2 Step 2.2)
    for content in feed:
        if agent.agent_type == "bot":
            shared.append(content)
            continue

        share_probability = _sigmoid(
            SHARE_BASE_LOGIT
            + arousal_share_weight * new_arousal
            + valence_share_weight * float(content.emotional_valence)
        )
        if local_rng.random() < share_probability:
            shared.append(content)

    # Step 5: OPINION UPDATE
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
        "reinforcement_factor",
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
    _assert_probability("alpha", float(config["alpha"]))
    _assert_probability("beta_pop", float(config["beta_pop"]))
    _assert_probability("sir_beta", float(config["sir_beta"]))
    _assert_probability("sir_gamma", float(config["sir_gamma"]))
    _assert_probability("reinforcement_factor", float(config["reinforcement_factor"]))
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
        z = float(np.exp(-value))
        return 1.0 / (1.0 + z)
    z = float(np.exp(value))
    return z / (1.0 + z)


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
        arousal_tolerance_effect=float(merged_config["arousal_tolerance_effect"]),
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
    reinforcement_factor = float(merged_config["reinforcement_factor"])
    emotional_decay = float(merged_config["emotional_decay"])
    arousal_share_weight = float(merged_config["arousal_share_weight"])
    valence_share_weight = float(merged_config["valence_share_weight"])

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
                    emotional_decay=emotional_decay,
                    arousal_share_weight=arousal_share_weight,
                    valence_share_weight=valence_share_weight,
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

        # SIR transitions with exposure-count reinforcement (Phase 2b Step 2b.2).
        for agent in active_agents:
            if agent.sir_state == "S":
                feed = consumed_items.get(agent.id, [])
                if not feed:
                    continue
                misinfo_items = [c for c in feed if c.is_misinformation]
                if not misinfo_items:
                    continue

                should_infect = False
                for content in misinfo_items:
                    agent.exposure_count[content.id] = (
                        agent.exposure_count.get(content.id, 0) + 1
                    )
                    n_exposures = agent.exposure_count[content.id]
                    effective_beta = sir_beta * (
                        1.0 + n_exposures * reinforcement_factor
                    )
                    effective_beta = min(effective_beta, 1.0)
                    if py_rng.random() < effective_beta:
                        should_infect = True
                        break

                if should_infect:
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
