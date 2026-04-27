"""Content-based recommender for MVP Step 5.

This module implements the MVP scoring contract used to rank candidate content
for each agent feed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


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
    """Minimal content shape used by the recommender and profiling harness."""

    ideological_score: float
    virality: float


def _score(content: CandidateContent, agent: AgentLike, alpha: float, beta_pop: float) -> float:
    """Compute feed score with distinct personalization and popularity terms.

    MVP Step 5 formula:
        score = alpha * (1 - abs(content.ideological_score - agent.opinion))
                + (1 - alpha) * beta_pop * log1p(content.virality * 9)
    """
    ideological_similarity = 1.0 - abs(content.ideological_score - agent.opinion)
    popularity_component = beta_pop * math.log1p(content.virality * 9.0)
    return alpha * ideological_similarity + (1.0 - alpha) * popularity_component


def _validate_params(alpha: float, beta_pop: float, k_exp: int) -> None:
    if k_exp <= 0:
        raise ValueError("k_exp must be > 0")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not 0.0 <= beta_pop <= 1.0:
        raise ValueError("beta_pop must be in [0, 1]")


@profile
def generate_feed(
    agent: AgentLike,
    candidate_pool: list[CandidateContent],
    k_exp: int,
    alpha: float,
    beta_pop: float = 0.2,
) -> list[CandidateContent]:
    """Rank candidate content and return the top-k feed for an agent.

    Scores are normalized to [0, 1] by dividing by the maximum score in the
    candidate pool before ranking, as required by MVP Step 5.
    """
    if not candidate_pool:
        return []

    _validate_params(alpha=alpha, beta_pop=beta_pop, k_exp=k_exp)

    scored: list[tuple[float, CandidateContent]] = [
        (_score(content, agent, alpha, beta_pop), content) for content in candidate_pool
    ]

    max_score = max(score for score, _ in scored)
    if max_score > 0:
        scored = [(score / max_score, content) for score, content in scored]

    scored.sort(key=lambda item: item[0], reverse=True)
    return [content for _, content in scored[:k_exp]]


def generate_feed_vectorized(
    agent: AgentLike,
    candidate_pool: list[CandidateContent],
    content_ideo_array: np.ndarray,
    content_virality_array: np.ndarray,
    k_exp: int,
    alpha: float,
    beta_pop: float = 0.2,
) -> list[CandidateContent]:
    """Vectorized top-k feed selection using NumPy arrays.

    This mirrors the MVP score formula while moving O(C) scoring into vectorized
    array operations and using argpartition for efficient top-k extraction.
    """
    if not candidate_pool:
        return []

    _validate_params(alpha=alpha, beta_pop=beta_pop, k_exp=k_exp)

    if len(candidate_pool) != int(content_ideo_array.shape[0]) or len(candidate_pool) != int(content_virality_array.shape[0]):
        raise ValueError("candidate_pool and content arrays must have matching lengths")

    similarity = 1.0 - np.abs(float(agent.opinion) - content_ideo_array)
    popularity_component = beta_pop * np.log1p(content_virality_array * 9.0)
    scores = alpha * similarity + (1.0 - alpha) * popularity_component

    max_score = float(np.max(scores))
    if max_score > 0.0:
        scores = scores / max_score

    n_candidates = len(candidate_pool)
    if k_exp >= n_candidates:
        top_indices = np.argsort(scores)[::-1]
    else:
        top_indices_unsorted = np.argpartition(scores, -k_exp)[-k_exp:]
        top_indices = top_indices_unsorted[np.argsort(scores[top_indices_unsorted])[::-1]]

    return [candidate_pool[int(idx)] for idx in top_indices]


__all__ = ["CandidateContent", "generate_feed", "generate_feed_vectorized"]
