"""Metric computations for the echo chamber simulation.

MVP Step 6 + Phase 6 Step 6.1:
- opinion variance, polarization index, opinion assortativity
- opinion entropy (20 bins), misinformation prevalence
- E-I index (opinion quartile groups)
- modularity Q (greedy community detection)
- cascade size (per-content sharing reach)
- exposure disparity (misinfo exposure by opinion quartile)
- IES (intervention effectiveness score)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import networkx as nx
import numpy as np

from .agent import Agent


ENTROPY_BIN_COUNT = 20
OPINION_MIN = -1.0
OPINION_MAX = 1.0
ENTROPY_EPSILON = 1e-10

# E-I Index quartile boundaries.
QUARTILE_EDGES = [-1.0, -0.5, 0.0, 0.5, 1.0]


def _agent_map(agents: list[Agent]) -> dict[int, Agent]:
    """Build an ID->Agent mapping for lookup by graph node id."""
    return {agent.id: agent for agent in agents}


def _node_opinion(G: nx.DiGraph, agent_lookup: dict[int, Agent], node_id: int) -> float:
    """Resolve node opinion from graph-attached agent first, then agent list fallback."""
    maybe_agent = G.nodes[node_id].get("agent")
    if isinstance(maybe_agent, Agent):
        return float(maybe_agent.opinion)
    agent = agent_lookup.get(node_id)
    if agent is None:
        raise KeyError(f"No agent opinion found for node {node_id}")
    return float(agent.opinion)


def opinion_variance(G: nx.DiGraph, agents: list[Agent]) -> float:
    """Compute opinion variance: (1/N) * sum((opinion_i - mean)^2)."""
    if not agents:
        return 0.0

    opinions = np.array([float(agent.opinion) for agent in agents], dtype=np.float64)
    return float(np.var(opinions))


def polarization_index(G: nx.DiGraph, agents: list[Agent]) -> float:
    """Compute mean absolute opinion difference across all directed edges."""
    if G.number_of_edges() == 0:
        return 0.0

    lookup = _agent_map(agents)
    distances: list[float] = []
    for source, target in G.edges:
        source_opinion = _node_opinion(G, lookup, source)
        target_opinion = _node_opinion(G, lookup, target)
        distances.append(abs(source_opinion - target_opinion))

    if not distances:
        return 0.0
    return float(np.mean(np.array(distances, dtype=np.float64)))


def opinion_assortativity(G: nx.DiGraph, agents: list[Agent]) -> float:
    """Compute Pearson correlation of opinion pairs across directed edges."""
    if G.number_of_edges() < 2:
        return 0.0

    lookup = _agent_map(agents)
    source_vals: list[float] = []
    target_vals: list[float] = []

    for source, target in G.edges:
        source_vals.append(_node_opinion(G, lookup, source))
        target_vals.append(_node_opinion(G, lookup, target))

    x = np.array(source_vals, dtype=np.float64)
    y = np.array(target_vals, dtype=np.float64)

    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 0.0 or y_std <= 0.0:
        return 0.0

    # Manual Pearson r over edge-endpoint opinion pairs (MVP Step 6 requirement).
    x_centered = x - float(np.mean(x))
    y_centered = y - float(np.mean(y))
    numerator = float(np.sum(x_centered * y_centered))
    denominator = math.sqrt(float(np.sum(x_centered * x_centered) * np.sum(y_centered * y_centered)))
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def opinion_entropy(G: nx.DiGraph, agents: list[Agent]) -> float:
    """Compute Shannon entropy over 20 bins spanning [-1, 1]."""
    if not agents:
        return 0.0

    opinions = np.array([float(agent.opinion) for agent in agents], dtype=np.float64)
    hist, _ = np.histogram(
        opinions,
        bins=ENTROPY_BIN_COUNT,
        range=(OPINION_MIN, OPINION_MAX),
    )
    total = int(np.sum(hist))
    if total == 0:
        return 0.0

    probabilities = hist.astype(np.float64) / float(total)
    nonzero_probs = probabilities[probabilities > 0.0]
    return float(-np.sum(nonzero_probs * np.log(nonzero_probs + ENTROPY_EPSILON)))


def misinformation_prevalence(G: nx.DiGraph, agents: list[Agent]) -> float:
    """Compute fraction of agents currently in misinformation state I."""
    if not agents:
        return 0.0
    infected = sum(
        1 for agent in agents
        if any(state == "I" for state in agent.sir_states.values())
    )
    return float(infected / len(agents))


def _quartile(opinion: float) -> int:
    """Map an opinion value to its quartile index (0-3)."""
    if opinion < -0.5:
        return 0
    if opinion < 0.0:
        return 1
    if opinion < 0.5:
        return 2
    return 3


def ei_index(G: nx.DiGraph, agents: list[Agent]) -> float:
    """E-I Index: (external_ties - internal_ties) / (external_ties + internal_ties).

    Groups are opinion quartiles: [-1,-0.5), [-0.5,0), [0,0.5), [0.5,1].
    -1 = pure echo chambers (all ties within groups).
    +1 = pure cross-cutting (all ties between groups).
    """
    if G.number_of_edges() == 0:
        return 0.0

    lookup = _agent_map(agents)
    internal = 0
    external = 0

    for source, target in G.edges:
        s_opinion = _node_opinion(G, lookup, source)
        t_opinion = _node_opinion(G, lookup, target)
        if _quartile(s_opinion) == _quartile(t_opinion):
            internal += 1
        else:
            external += 1

    total = internal + external
    if total == 0:
        return 0.0
    return float((external - internal) / total)


def modularity_q(G: nx.DiGraph) -> float:
    """Modularity Q using greedy community detection on the undirected graph.

    Values 0.3-0.7 indicate strong community structure (echo chambers).
    Returns 0.0 for graphs with fewer than 2 nodes.
    """
    if G.number_of_nodes() < 2:
        return 0.0

    try:
        ug = G.to_undirected()
        communities = nx.community.greedy_modularity_communities(ug)
        return float(nx.community.modularity(ug, communities))
    except Exception:
        return 0.0


def cascade_stats(
    shared_content: dict[int, list[Any]],
) -> dict[str, float]:
    """Compute mean and max cascade size from per-agent sharing data.

    Cascade size = number of unique agents who shared a content item.
    Takes ``shared_content`` dict mapping agent_id -> list of shared Content.
    Returns dict with ``cascade_mean`` and ``cascade_max``.
    """
    content_sharers: dict[int, set[int]] = defaultdict(set)
    for agent_id, items in shared_content.items():
        for content in items:
            content_sharers[content.id].add(agent_id)

    if not content_sharers:
        return {"cascade_mean": 0.0, "cascade_max": 0}

    sizes = [len(sharers) for sharers in content_sharers.values()]
    return {
        "cascade_mean": float(np.mean(sizes)),
        "cascade_max": int(np.max(sizes)) if sizes else 0,
    }


def exposure_disparity(
    G: nx.DiGraph,
    agents: list[Agent],
    consumed_items: dict[int, list[Any]],
) -> float:
    """Exposure disparity: difference in misinfo exposure across opinion quartiles.

    For each quartile group, computes mean misinformation exposure rate
    (fraction of consumed feed items that are misinformation).
    Returns max(group_rate) - min(group_rate).
    0 = equal exposure across groups.
    """
    # Group agents by opinion quartile.
    quartile_agents: dict[int, list[Agent]] = {0: [], 1: [], 2: [], 3: []}
    for agent in agents:
        if not agent.is_active:
            continue
        quartile_agents[_quartile(agent.opinion)].append(agent)

    # Compute mean misinfo exposure per quartile.
    quartile_rates: list[float] = []
    for q in range(4):
        group = quartile_agents[q]
        if not group:
            quartile_rates.append(0.0)
            continue
        rates: list[float] = []
        for agent in group:
            feed = consumed_items.get(agent.id, [])
            if not feed:
                continue
            misinfo_frac = sum(1 for c in feed if c.is_misinformation) / len(feed)
            rates.append(misinfo_frac)
        quartile_rates.append(float(np.mean(rates)) if rates else 0.0)

    return float(max(quartile_rates) - min(quartile_rates))


def compute_ies(
    baseline_metric: float,
    intervention_metric: float,
) -> float:
    """Intervention Effectiveness Score.

    IES = (M_control - M_intervention) / (M_control + epsilon).
    Positive = intervention reduced the metric (good for misinfo/polarization).
    Negative = intervention made it worse.
    """
    return float(
        (baseline_metric - intervention_metric) / (baseline_metric + 1e-10)
    )


def compute_all_metrics(
    G: nx.DiGraph,
    agents: list[Agent],
    tick: int,
    shared_content: dict[int, list[Any]] | None = None,
    consumed_items: dict[int, list[Any]] | None = None,
) -> dict[str, Any]:
    """Return the full metric snapshot payload for a simulation tick.

    Phase 6 Step 6.1: adds E-I index, modularity Q, cascade stats, and
    exposure disparity when the optional data dicts are provided.
    """
    snapshot: dict[str, Any] = {
        "tick": int(tick),
        "opinion_variance": opinion_variance(G, agents),
        "polarization_index": polarization_index(G, agents),
        "assortativity": opinion_assortativity(G, agents),
        "opinion_entropy": opinion_entropy(G, agents),
        "misinfo_prevalence": misinformation_prevalence(G, agents),
        "ei_index": ei_index(G, agents),
        "modularity_q": modularity_q(G),
    }

    if shared_content is not None:
        cascade = cascade_stats(shared_content)
        snapshot["cascade_mean"] = cascade["cascade_mean"]
        snapshot["cascade_max"] = cascade["cascade_max"]

    if consumed_items is not None:
        snapshot["exposure_disparity"] = exposure_disparity(
            G, agents, consumed_items,
        )

    return snapshot


__all__ = [
    "cascade_stats",
    "compute_all_metrics",
    "compute_ies",
    "ei_index",
    "exposure_disparity",
    "misinformation_prevalence",
    "modularity_q",
    "opinion_assortativity",
    "opinion_entropy",
    "opinion_variance",
    "polarization_index",
]
