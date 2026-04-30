"""Network construction and graph helper utilities.

MVP Step 4 scope:
- Build a directed Watts-Strogatz social graph
- Attach agents to nodes
- Compute/store normalized edge weights for incoming influence
- Provide safe graph access helpers and snapshot serialization
"""

from __future__ import annotations

import random
from typing import Any

import networkx as nx

from .agent import Agent


DEFAULT_N = 200
DEFAULT_AVG_DEGREE = 16
DEFAULT_REWIRE_PROB = 0.1
INFLUENCER_DEGREE_MULTIPLIER = 3


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


def compute_edge_weights(G: nx.DiGraph, agent_id: int, agents: list[Agent]) -> dict[int, float]:
    """Compute normalized incoming edge weights for one agent.

    Influence weights are proportional to each predecessor's
    ``influence_weight_multiplier`` and normalized to sum to 1.0.
    """
    predecessors = get_predecessors(G, agent_id)
    if not predecessors:
        return {}

    raw = {
        source_id: float(agents[source_id].influence_weight_multiplier)
        for source_id in predecessors
    }
    total = sum(raw.values())
    if total <= 0.0:
        uniform = 1.0 / len(predecessors)
        return {source_id: uniform for source_id in predecessors}

    return {source_id: weight / total for source_id, weight in raw.items()}


def initialize_edge_weights(G: nx.DiGraph, agents: list[Agent] | None = None) -> None:
    """Initialize incoming influence weights as uniform 1 / in_degree(i).

    For every edge (j -> i), ``G[j][i]['weight']`` is the influence weight that node i
    assigns to source j.
    """
    for agent_id in G.nodes:
        predecessors = get_predecessors(G, agent_id)
        if not predecessors:
            continue

        if agents is None:
            uniform_weight = 1.0 / len(predecessors)
            for source_id in predecessors:
                G[source_id][agent_id]["weight"] = uniform_weight
            continue

        weights = compute_edge_weights(G, agent_id, agents)
        for source_id, weight in weights.items():
            G[source_id][agent_id]["weight"] = weight


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
    _rewire_influencer_hubs(
        G,
        agents,
        avg_degree=avg_degree,
        seed=seed,
    )
    initialize_edge_weights(G, agents)
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


def _rewire_influencer_hubs(
    G: nx.DiGraph,
    agents: list[Agent],
    avg_degree: int,
    seed: int | None,
) -> None:
    """Boost influencer out-degree to at least 3x avg_degree.

    Outgoing edges represent followers (influence targets), so we expand
    influencer successors to increase broadcast reach.
    """
    influencer_ids = [agent.id for agent in agents if agent.agent_type == "influencer"]
    if not influencer_ids:
        return

    rng = random.Random(seed)
    target_degree = int(INFLUENCER_DEGREE_MULTIPLIER * avg_degree)
    max_targets = max(0, G.number_of_nodes() - 1)
    target_degree = min(target_degree, max_targets)

    for influencer_id in influencer_ids:
        current_successors = set(get_successors(G, influencer_id))
        needed = target_degree - len(current_successors)
        if needed <= 0:
            continue

        candidates = [
            node_id
            for node_id in G.nodes
            if node_id != influencer_id and node_id not in current_successors
        ]
        if not candidates:
            continue

        if needed >= len(candidates):
            chosen = candidates
        else:
            chosen = rng.sample(candidates, k=needed)

        for target_id in chosen:
            G.add_edge(influencer_id, target_id)


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


def renormalize_weights(G: nx.DiGraph, agent_id: int) -> None:
    """Recompute normalized incoming edge weights for one agent.

    Called after any edge add/remove that changes ``agent_id``'s predecessor set.
    Uses uniform 1/in_degree by default; uses ``compute_edge_weights`` for
    influencer agents whose ``influence_weight_multiplier != 1.0``.

    Per the directed graph convention:
    - Opinion updates consume INCOMING weights (predecessors -> agent_id).
    - Normalize incoming weights for agent_id, NOT outgoing.
    - After G.remove_edge(u, v): call renormalize_weights(G, v).
    - After G.add_edge(u, v): call renormalize_weights(G, v).
    - Never call on u — u's incoming weights are unaffected.
    """
    preds = list(G.predecessors(agent_id))
    if not preds:
        return

    # Check if this node's agent uses non-uniform influence weights.
    node_agent = G.nodes[agent_id].get("agent")
    if node_agent is not None and hasattr(node_agent, "influence_weight_multiplier"):
        multiplier = float(node_agent.influence_weight_multiplier)
    else:
        multiplier = 1.0

    if multiplier != 1.0:
        # Recompute using the full weight formula for non-uniform agents.
        agents_list = [
            G.nodes[nid]["agent"] for nid in preds if "agent" in G.nodes[nid]
        ]
        if len(agents_list) == len(preds):
            weights = compute_edge_weights(G, agent_id, agents_list)
            for source_id, weight in weights.items():
                G[source_id][agent_id]["weight"] = weight
            return

    # Uniform weight strategy.
    uniform = 1.0 / len(preds)
    for source_id in preds:
        G[source_id][agent_id]["weight"] = uniform


def rewire_step(
    G: nx.DiGraph,
    agents: list[Agent],
    dynamic_rewire_rate: float,
    homophily_threshold: float,
    seed: int | None = None,
) -> None:
    """Dynamic edge rewiring: agents unfollow disagreeing peers and follow similar ones.

    Each active agent, with probability ``dynamic_rewire_rate``, evaluates its
    outgoing edges (who it follows). The successor with the largest opinion
    distance beyond ``homophily_threshold`` is unfollowed, and a new agent
    within the threshold is followed instead.

    Directed graph convention (Phase 4 Step 4.1):
    - Rewiring changes OUTGOING edges (who this agent follows = successors).
    - After remove_edge(agent, worst): renormalize_weights(G, worst).
    - After add_edge(agent, new_follow): renormalize_weights(G, new_follow).
    """
    if dynamic_rewire_rate <= 0.0:
        return

    rng = random.Random(seed)
    agent_map = {a.id: a for a in agents}

    for agent in agents:
        if not agent.is_active:
            continue

        if rng.random() >= dynamic_rewire_rate:
            continue

        following_ids = get_successors(G, agent.id)
        if not following_ids:
            continue

        # Find the followed agent with the largest opinion distance.
        worst_id = max(
            following_ids,
            key=lambda nid: abs(agent_map[nid].opinion - agent.opinion),
        )
        worst_agent = agent_map[worst_id]
        if abs(worst_agent.opinion - agent.opinion) <= homophily_threshold:
            continue

        # Unfollow.
        G.remove_edge(agent.id, worst_id)
        renormalize_weights(G, worst_id)

        # Find a new agent within homophily_threshold to follow.
        current_following = set(get_successors(G, agent.id))
        candidates = [
            a
            for a in agents
            if a.id != agent.id
            and a.id not in current_following
            and a.is_active
            and abs(a.opinion - agent.opinion) <= homophily_threshold
        ]
        if candidates:
            new_follow = rng.choice(candidates)
            G.add_edge(agent.id, new_follow.id)
            # Initialize edge weight for the new connection.
            G[agent.id][new_follow.id]["weight"] = 1.0
            renormalize_weights(G, new_follow.id)


def compute_dissatisfaction(agent: Agent, G: nx.DiGraph) -> float:
    """Mean opinion distance to predecessors (influence sources).

    High disagreement with who an agent listens to drives churn.
    Returns 0.0 when the agent has no predecessors.
    """
    preds = get_predecessors(G, agent.id)
    if not preds:
        return 0.0

    total_distance = 0.0
    for source_id in preds:
        source_agent = G.nodes[source_id].get("agent")
        if source_agent is not None:
            total_distance += abs(agent.opinion - source_agent.opinion)
    return total_distance / len(preds)


__all__ = [
    "assign_agents_to_graph",
    "build_network",
    "build_network_from_size",
    "compute_dissatisfaction",
    "compute_edge_weights",
    "get_graph_snapshot",
    "get_influence_weights",
    "get_predecessors",
    "get_successors",
    "initialize_edge_weights",
    "renormalize_weights",
    "rewire_step",
]
