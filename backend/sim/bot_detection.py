"""Behavioral bot detection using suspicion scores (Phase 5 Step 5.3).

Computes suspicion scores from four observable behavioral signals.
Never uses ground-truth ``agent.agent_type`` — detection is behavioral only.
"""

from __future__ import annotations

from statistics import fmean, pstdev
from typing import TYPE_CHECKING

import numpy as np

from .network import get_predecessors, get_successors

if TYPE_CHECKING:
    from .agent import Agent
    from .content import Content
    from networkx import DiGraph


def _sigmoid(x: float) -> float:
    """Standard logistic sigmoid."""
    z = float(np.exp(np.clip(x, -20.0, 20.0)))
    return z / (1.0 + z)


def compute_suspicion_score(
    agent: "Agent",
    recent_shares: list["Content"],
    G: "DiGraph",
    population_mean_activity: float,
    population_std_activity: float,
) -> float:
    """Compute behavioral bot suspicion score from four signals.

    All signals are in [0, 1]. The final score is the unweighted mean.

    Signal 1 — Abnormal post frequency:
        Z-score of agent activity_rate vs population, passed through sigmoid.

    Signal 2 — Near-zero opinion variance:
        Variance of ``opinion_history[-50:]``. Bots post consistently from a
        fixed extreme position.

    Signal 3 — High emotional valence in shared content:
        Bots share high-arousal content to maximize engagement.

    Signal 4 — Low reciprocity ratio:
        Bots follow many agents but have few followers in common.
        in_degree / out_degree, inverted (1 - ratio) so higher = more suspicious.
    """
    signals: list[float] = []

    # Signal 1: abnormal post frequency.
    if population_std_activity > 0.0:
        z_score = (float(agent.activity_rate) - population_mean_activity) / population_std_activity
    else:
        z_score = 0.0
    signals.append(_sigmoid(z_score))

    # Signal 2: near-zero opinion variance (requires at least 2 data points).
    history = agent.opinion_history[-50:]
    if len(history) >= 2:
        variance = float(np.var(history))
    else:
        variance = 0.0
    signals.append(1.0 - min(variance * 10.0, 1.0))

    # Signal 3: high average emotional valence of shared content.
    if recent_shares:
        mean_valence = fmean(float(c.emotional_valence) for c in recent_shares)
    else:
        mean_valence = 0.0
    signals.append(float(np.clip(mean_valence, 0.0, 1.0)))

    # Signal 4: low reciprocity ratio.
    following = set(get_successors(G, agent.id))
    followers = set(get_predecessors(G, agent.id))
    if len(following) > 0:
        reciprocity = len(followers & following) / len(following)
    else:
        reciprocity = 1.0
    signals.append(1.0 - float(np.clip(reciprocity, 0.0, 1.0)))

    return float(np.clip(fmean(signals), 0.0, 1.0))


def compute_population_activity_stats(
    agents: list["Agent"],
) -> tuple[float, float]:
    """Compute population mean and std of activity rates."""
    rates = [float(a.activity_rate) for a in agents if a.is_active]
    if not rates:
        return 0.0, 0.0
    return float(fmean(rates)), float(pstdev(rates)) if len(rates) >= 2 else 0.0


__all__ = [
    "compute_population_activity_stats",
    "compute_suspicion_score",
]
