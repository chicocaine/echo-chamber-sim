"""Network construction and graph helper utilities.

MVP Step 4 scope:
- Build a directed Watts-Strogatz social graph
- Attach agents to nodes
- Compute/store normalized edge weights for incoming influence
- Provide safe graph access helpers and snapshot serialization
"""

from __future__ import annotations

from typing import Any

import networkx as nx

from .agent import Agent


DEFAULT_N = 200
DEFAULT_AVG_DEGREE = 16
DEFAULT_REWIRE_PROB = 0.1


def _validate_graph_params(n_agents: int, avg_degree: int, rewire_prob: float) -> None:
    """Validate basic graph construction parameters."""
    if n_agents <= 0:
        raise ValueError(f"n_agents must be > 0, got {n_agents}")
    if avg_degree <= 0:
        raise ValueError(f"avg_degree must be > 0, got {avg_degree}")
    if avg_degree >= n_agents:
        raise ValueError("avg_degree must be smaller than n_agents for Watts-Strogatz graph")
    if avg_degree % 2 != 0:
        raise ValueError("avg_degree must be even for Watts-Strogatz graph")
    if not 0.0 <= rewire_prob <= 1.0:
        raise ValueError(f"rewire_prob must be in [0, 1], got {rewire_prob}")


def assign_agents_to_graph(G: nx.DiGraph, agents: list[Agent]) -> None:
    """Attach each agent object to its node as ``G.nodes[id]['agent']``."""
    if len(agents) != G.number_of_nodes():
        raise ValueError(
            "Agent count must equal graph node count "
            f"(agents={len(agents)} nodes={G.number_of_nodes()})"
        )

    for node_id in G.nodes:
        G.nodes[node_id]["agent"] = agents[node_id]


def initialize_edge_weights(G: nx.DiGraph) -> None:
    """Initialize incoming influence weights as uniform 1 / in_degree(i).

    For every edge (j -> i), ``G[j][i]['weight']`` is the influence weight that node i
    assigns to source j.
    """
    for agent_id in G.nodes:
        predecessors = get_predecessors(G, agent_id)
        if not predecessors:
            continue

        uniform_weight = 1.0 / len(predecessors)
        for source_id in predecessors:
            G[source_id][agent_id]["weight"] = uniform_weight


def build_network(
    agents: list[Agent],
    avg_degree: int = DEFAULT_AVG_DEGREE,
    rewire_prob: float = DEFAULT_REWIRE_PROB,
    seed: int | None = None,
) -> nx.DiGraph:
    """Build a directed Watts-Strogatz graph and attach agents.

    The undirected base graph is converted to ``nx.DiGraph`` so each undirected
    edge becomes two directed edges, matching the simulation's information-flow model.
    """
    n_agents = len(agents)
    _validate_graph_params(n_agents=n_agents, avg_degree=avg_degree, rewire_prob=rewire_prob)

    undirected_graph = nx.watts_strogatz_graph(
        n=n_agents,
        k=avg_degree,
        p=rewire_prob,
        seed=seed,
    )
    G = nx.DiGraph(undirected_graph)

    assign_agents_to_graph(G, agents)
    initialize_edge_weights(G)
    return G


def build_network_from_size(
    n_agents: int = DEFAULT_N,
    avg_degree: int = DEFAULT_AVG_DEGREE,
    rewire_prob: float = DEFAULT_REWIRE_PROB,
    seed: int | None = None,
) -> nx.DiGraph:
    """Build a directed graph without attached agents.

    This helper is primarily for network-only diagnostics and tests.
    """
    _validate_graph_params(n_agents=n_agents, avg_degree=avg_degree, rewire_prob=rewire_prob)
    undirected_graph = nx.watts_strogatz_graph(
        n=n_agents,
        k=avg_degree,
        p=rewire_prob,
        seed=seed,
    )
    G = nx.DiGraph(undirected_graph)
    initialize_edge_weights(G)
    return G


# =============================================================================
# DIRECTED GRAPH CONVENTION — READ THIS BEFORE TOUCHING ANY GRAPH TRAVERSAL
# =============================================================================
# The graph is a nx.DiGraph. An edge (j -> i) means "i follows j", i.e. j is
# an influence source for i. The direction encodes information flow, NOT who
# initiated the relationship.
#
#   G.predecessors(i)  -> agents that i LISTENS TO   (influence sources for i)
#   G.successors(i)    -> agents that i BROADCASTS TO (i is an influence source for them)
#   G.neighbors(i)     -> DO NOT USE. On DiGraph this returns successors only,
#                        which is the WRONG direction for opinion updates. It is
#                        banned in this codebase to prevent silent correctness bugs.
#
# Rule of thumb:
#   Opinion update / feed generation  ->  G.predecessors(i)   ✓
#   Broadcast / virality propagation  ->  G.successors(i)     ✓
#   G.neighbors(i)                    ->  NEVER               ✗
#
# All public helpers below enforce this. Never call the NetworkX API directly
# outside of network.py.
# =============================================================================


def get_predecessors(G: nx.DiGraph, agent_id: int) -> list[int]:
    """Agents that ``agent_id`` listens to (incoming neighbors)."""
    return list(G.predecessors(agent_id))


def get_successors(G: nx.DiGraph, agent_id: int) -> list[int]:
    """Agents that receive broadcasts from ``agent_id`` (outgoing neighbors)."""
    return list(G.successors(agent_id))


def get_influence_weights(G: nx.DiGraph, agent_id: int) -> dict[int, float]:
    """Return normalized incoming influence weights for one agent.

    The returned weights always sum to 1.0 when predecessors exist.
    """
    predecessors = get_predecessors(G, agent_id)
    if not predecessors:
        return {}

    raw = {source_id: float(G[source_id][agent_id].get("weight", 1.0)) for source_id in predecessors}
    total = sum(raw.values())
    if total <= 0.0:
        uniform = 1.0 / len(predecessors)
        return {source_id: uniform for source_id in predecessors}
    return {source_id: weight / total for source_id, weight in raw.items()}


def get_graph_snapshot(G: nx.DiGraph) -> dict[str, Any]:
    """Serialize current node state and directed edges for frontend rendering."""
    nodes: list[dict[str, Any]] = []
    for node_id in G.nodes:
        maybe_agent = G.nodes[node_id].get("agent")
        if isinstance(maybe_agent, Agent):
            nodes.append(
                {
                    "id": node_id,
                    "opinion": float(maybe_agent.opinion),
                    "activity_rate": float(maybe_agent.activity_rate),
                    "agent_type": maybe_agent.agent_type,
                    "is_active": bool(maybe_agent.is_active),
                }
            )
        else:
            nodes.append(
                {
                    "id": node_id,
                    "opinion": 0.0,
                    "activity_rate": 0.0,
                    "agent_type": "unknown",
                    "is_active": True,
                }
            )

    edges = [
        {
            "source": source,
            "target": target,
            "weight": float(G[source][target].get("weight", 0.0)),
        }
        for source, target in G.edges
    ]

    return {
        "nodes": nodes,
        "edges": edges,
    }


__all__ = [
    "assign_agents_to_graph",
    "build_network",
    "build_network_from_size",
    "get_graph_snapshot",
    "get_influence_weights",
    "get_predecessors",
    "get_successors",
    "initialize_edge_weights",
]
