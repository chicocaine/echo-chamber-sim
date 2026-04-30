"""Recommender class hierarchy for feed generation.

Phase 3 Step 3.1 — refactored from standalone functions into:
- BaseRecommender: abstract base with score() and loop-based generate_feed()
- ContentBasedRecommender: vectorized content-based scoring with intervention hooks
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .agent import Agent
    from .content import Content


try:
    profile  # type: ignore[name-defined]
except NameError:
    def profile(func):
        """No-op decorator when line_profiler is not active."""
        return func


class BaseRecommender(ABC):
    """Abstract base for all recommender algorithms.

    Subclasses must implement score() and may override generate_feed()
    for optimized implementations.
    """

    @abstractmethod
    def score(self, content: "Content", agent: "Agent") -> float:
        """Score a single content item for an agent. Higher is better."""
        ...

    def generate_feed(
        self,
        agent: "Agent",
        candidate_pool: list["Content"],
        k_exp: int,
    ) -> list["Content"]:
        """Loop-based fallback using score() for non-vectorized subclasses.

        CF and graph-based recommenders use this path. ContentBasedRecommender
        overrides with the vectorized implementation (Phase Opt Step Opt.2).
        """
        if not candidate_pool:
            return []

        scored = [(self.score(c, agent), c) for c in candidate_pool]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [c for _, c in scored[:k_exp]]


class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender using ideological similarity and virality.

    Scoring formula (ref doc Part 4):
        score = alpha * (1 - |content.ideological_score - agent.opinion|)
                + (1 - alpha) * beta_pop * log1p(content.virality * 9)

    Intervention hooks (diversity_ratio, lambda_penalty) are accepted at init
    but wired in Phase 3 Step 3.2.
    """

    def __init__(
        self,
        alpha: float = 0.65,
        beta_pop: float = 0.2,
        diversity_ratio: float = 0.0,
        lambda_penalty: float = 0.0,
    ) -> None:
        self.alpha = float(alpha)
        self.beta_pop = float(beta_pop)
        self.diversity_ratio = float(diversity_ratio)
        self.lambda_penalty = float(lambda_penalty)

        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= self.beta_pop <= 1.0:
            raise ValueError("beta_pop must be in [0, 1]")
        if not 0.0 <= self.diversity_ratio <= 1.0:
            raise ValueError("diversity_ratio must be in [0, 1]")
        if not 0.0 <= self.lambda_penalty <= 1.0:
            raise ValueError("lambda_penalty must be in [0, 1]")

    def score(self, content: "Content", agent: "Agent") -> float:
        """Per-item score using the MVP content-based formula with misinfo penalty."""
        ideological_similarity = 1.0 - abs(content.ideological_score - agent.opinion)
        popularity_component = self.beta_pop * math.log1p(content.virality * 9.0)
        base = self.alpha * ideological_similarity + (1.0 - self.alpha) * popularity_component
        return base - self.lambda_penalty * content.misinfo_score

    @profile
    def generate_feed(
        self,
        agent: "Agent",
        candidate_pool: list["Content"],
        k_exp: int,
    ) -> list["Content"]:
        """Vectorized feed generation with intervention hooks (Phase 3 Step 3.2).

        Delegates to the module-level vectorized function with all intervention
        parameters. simulation.py may also call generate_feed_vectorized directly
        with pre-extracted arrays for batch performance.
        """
        if not candidate_pool:
            return []

        if k_exp <= 0:
            raise ValueError("k_exp must be > 0")

        n_candidates = len(candidate_pool)
        content_ideo_array = np.fromiter(
            (c.ideological_score for c in candidate_pool),
            dtype=np.float64,
            count=n_candidates,
        )
        content_virality_array = np.fromiter(
            (c.virality for c in candidate_pool),
            dtype=np.float64,
            count=n_candidates,
        )
        content_misinfo_array = np.fromiter(
            (c.misinfo_score for c in candidate_pool),
            dtype=np.float64,
            count=n_candidates,
        )
        return generate_feed_vectorized(
            agent=agent,
            candidate_pool=candidate_pool,
            content_ideo_array=content_ideo_array,
            content_virality_array=content_virality_array,
            content_misinfo_array=content_misinfo_array,
            k_exp=k_exp,
            alpha=self.alpha,
            beta_pop=self.beta_pop,
            lambda_penalty=self.lambda_penalty,
            diversity_ratio=self.diversity_ratio,
        )


@profile
def generate_feed_vectorized(
    agent: "Agent",
    candidate_pool: list["Content"],
    content_ideo_array: np.ndarray,
    content_virality_array: np.ndarray,
    k_exp: int,
    alpha: float,
    beta_pop: float = 0.2,
    lambda_penalty: float = 0.0,
    content_misinfo_array: np.ndarray | None = None,
    diversity_ratio: float = 0.0,
) -> list["Content"]:
    """Vectorized top-k feed selection using pre-extracted NumPy arrays.

    Performs O(C) scoring in vectorized array operations and uses argpartition
    for efficient top-k extraction (Phase Opt Step Opt.2).

    Intervention hooks (Phase 3 Step 3.2):
    - lambda_penalty: subtracted from score per content item
    - diversity_ratio: fraction of feed slots filled with dissimilar content
    """
    if not candidate_pool:
        return []

    n_candidates = len(candidate_pool)
    if n_candidates != int(content_ideo_array.shape[0]) or n_candidates != int(content_virality_array.shape[0]):
        raise ValueError("candidate_pool and content arrays must have matching lengths")
    if content_misinfo_array is not None and n_candidates != int(content_misinfo_array.shape[0]):
        raise ValueError("content_misinfo_array length must match candidate_pool")

    similarity = 1.0 - np.abs(float(agent.opinion) - content_ideo_array)
    popularity_component = beta_pop * np.log1p(content_virality_array * 9.0)
    scores = alpha * similarity + (1.0 - alpha) * popularity_component

    if lambda_penalty > 0.0 and content_misinfo_array is not None:
        scores = scores - lambda_penalty * content_misinfo_array

    max_score = float(np.max(scores))
    if max_score > 0.0:
        scores = scores / max_score

    # Diversity injection: reserve k_div slots for low-scoring (dissimilar) content.
    k_sim = int(k_exp * (1.0 - diversity_ratio))
    k_div = k_exp - k_sim

    if k_div == 0 or k_exp >= n_candidates:
        if k_exp >= n_candidates:
            top_indices = list(range(n_candidates))
        else:
            top_indices_unsorted = np.argpartition(scores, -k_exp)[-k_exp:]
            top_indices = list(top_indices_unsorted)
        top_indices.sort(key=lambda i: scores[i], reverse=True)
        return [candidate_pool[i] for i in top_indices[:k_exp]]

    # Full rank-ordering for correct diversity partition.
    ranked = list(range(n_candidates))
    ranked.sort(key=lambda i: scores[i], reverse=True)

    sim_feed = [candidate_pool[i] for i in ranked[:k_sim]]
    remaining = ranked[k_sim:]
    # Dissimilar = lowest-scoring from remaining
    remaining_sorted = sorted(remaining, key=lambda i: scores[i])
    div_feed = [candidate_pool[i] for i in remaining_sorted[:k_div]]

    feed = sim_feed + div_feed
    rng = np.random.default_rng(abs(hash(agent.id)))
    rng.shuffle(feed)
    return feed


__all__ = [
    "BaseRecommender",
    "ContentBasedRecommender",
    "generate_feed_vectorized",
]
