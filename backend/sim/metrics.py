"""Metric computations for the echo chamber simulation.

MVP Step 6 scope:
- opinion variance
- polarization index
- opinion assortativity (manual Pearson across edges)
- opinion entropy (20 bins in [-1, 1])
- misinformation prevalence
"""

from __future__ import annotations

import math
from typing import Any

import networkx as nx
import numpy as np

from .agent import Agent


ENTROPY_BIN_COUNT = 20
OPINION_MIN = -1.0
OPINION_MAX = 1.0
ENTROPY_EPSILON = 1e-10


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
    infected = sum(1 for agent in agents if agent.sir_state == "I")
    return float(infected / len(agents))


def compute_all_metrics(G: nx.DiGraph, agents: list[Agent], tick: int) -> dict[str, Any]:
    """Return the MVP metric snapshot payload for a simulation tick."""
    return {
        "tick": int(tick),
        "opinion_variance": opinion_variance(G, agents),
        "polarization_index": polarization_index(G, agents),
        "assortativity": opinion_assortativity(G, agents),
        "opinion_entropy": opinion_entropy(G, agents),
        "misinfo_prevalence": misinformation_prevalence(G, agents),
    }


__all__ = [
    "compute_all_metrics",
    "misinformation_prevalence",
    "opinion_assortativity",
    "opinion_entropy",
    "opinion_variance",
    "polarization_index",
]
