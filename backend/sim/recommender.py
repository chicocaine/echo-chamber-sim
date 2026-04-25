"""Recommender baseline module for MVP Step 1b profiling.

This module intentionally uses a naive Python-loop scoring approach so the
line profiler can capture baseline runtime hotspots before optimization phases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


try:
    profile  # type: ignore[name-defined]
except NameError:
    def profile(func):
        """No-op decorator when line_profiler is not active."""
        return func


class AgentLike(Protocol):
    """Protocol describing the fields used by feed scoring."""

    opinion: float


@dataclass(slots=True)
class CandidateContent:
    """Minimal content shape required for profiling feed generation."""

    ideological_score: float
    virality: float


def _score(content: CandidateContent, agent: AgentLike, alpha: float, beta_pop: float) -> float:
    """Compute MVP feed score per implementation plan Step 5 formula."""
    ideological_similarity = 1.0 - abs(content.ideological_score - agent.opinion)
    popularity_component = beta_pop * math.log1p(content.virality * 9.0)
    return alpha * ideological_similarity + (1.0 - alpha) * popularity_component


@profile
def generate_feed(
    agent: AgentLike,
    candidate_pool: list[CandidateContent],
    k_exp: int,
    alpha: float,
    beta_pop: float = 0.2,
) -> list[CandidateContent]:
    """Return top-k scored content with score normalization to [0, 1]."""
    if not candidate_pool or k_exp <= 0:
        return []

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not 0.0 <= beta_pop <= 1.0:
        raise ValueError("beta_pop must be in [0, 1]")

    scored: list[tuple[float, CandidateContent]] = [
        (_score(content, agent, alpha, beta_pop), content) for content in candidate_pool
    ]

    max_score = max(score for score, _ in scored)
    if max_score > 0:
        scored = [(score / max_score, content) for score, content in scored]

    scored.sort(key=lambda item: item[0], reverse=True)
    return [content for _, content in scored[:k_exp]]
