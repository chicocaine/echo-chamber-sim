"""Content schema and generation logic for the echo chamber simulation.

MVP Step 3 scope:
- Content dataclass and required fields
- Per-agent content generation rules
- Belief update weight computation (stored on content metadata only in MVP)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


# Content generation defaults from implementation plan MVP Step 3 and Phase 2.
IDEOLOGICAL_NOISE_STD = 0.1
NORMAL_MISINFO_SCORE = 0.05
BOT_MISINFO_MIN = 0.7
BOT_MISINFO_MAX = 1.0
VIRALITY_MIN = 0.1
VIRALITY_MAX = 0.9
# Phase 2: Emotional valence beta distribution parameters (ref doc Part 6)
MISINFO_VALENCE_ALPHA = 3.0  # skewed toward high valence
MISINFO_VALENCE_BETA = 1.0
NORMAL_VALENCE_ALPHA = 1.0  # skewed toward low valence
NORMAL_VALENCE_BETA = 3.0


class AgentForContent(Protocol):
    """Minimal agent interface needed by content generation."""

    id: int
    agent_type: str
    opinion: float
    trust: float
    media_literacy: float
    activity_rate: float
    misinfo_rate: float


def _clamp_opinion(value: float) -> float:
    """Clamp values to opinion range [-1.0, 1.0]."""
    return float(np.clip(value, -1.0, 1.0))


def _clamp_probability(value: float) -> float:
    """Clamp values to probability range [0.0, 1.0]."""
    return float(np.clip(value, 0.0, 1.0))


@dataclass(slots=True)
class Content:
    """Single content item in the simulation."""

    id: int
    creator_id: int
    timestamp: int
    ideological_score: float
    emotional_valence: float
    misinfo_score: float
    virality: float
    source_credibility: float
    is_misinformation: bool
    belief_update_weight: float
    topic_vector: list[float]
    coordinated_campaign_id: int | None
    is_satire: bool

    def __post_init__(self) -> None:
        self.ideological_score = _clamp_opinion(self.ideological_score)
        self.emotional_valence = _clamp_probability(self.emotional_valence)
        self.misinfo_score = _clamp_probability(self.misinfo_score)
        self.virality = _clamp_probability(self.virality)
        self.source_credibility = _clamp_probability(self.source_credibility)
        self.belief_update_weight = _clamp_probability(self.belief_update_weight)

        assert self.id >= 0, f"content id must be >= 0, got {self.id}"
        assert self.creator_id >= 0, f"creator_id must be >= 0, got {self.creator_id}"
        assert self.timestamp >= 0, f"timestamp must be >= 0, got {self.timestamp}"


def compute_belief_update_weight(
    misinfo_score: float,
    source_credibility: float,
    media_literacy_i: float,
) -> float:
    """Compute belief weight from reference formula (agent reference Part 3).

    belief_update_weight = (1 - misinfo_score) * source_credibility
                           * (1 - (1 - media_literacy_i) * misinfo_score)
    """
    misinfo_score = _clamp_probability(misinfo_score)
    source_credibility = _clamp_probability(source_credibility)
    media_literacy_i = _clamp_probability(media_literacy_i)

    weight = (1.0 - misinfo_score) * source_credibility * (
        1.0 - (1.0 - media_literacy_i) * misinfo_score
    )
    return _clamp_probability(weight)


def sample_misinfo_score(agent: AgentForContent, rng: np.random.Generator) -> float:
    """Sample misinformation score according to MVP Step 3 rules."""
    if agent.agent_type == "bot":
        bot_misinfo = float(rng.uniform(BOT_MISINFO_MIN, BOT_MISINFO_MAX))
        if rng.random() <= _clamp_probability(agent.misinfo_rate):
            return bot_misinfo
        return NORMAL_MISINFO_SCORE
    return NORMAL_MISINFO_SCORE


def sample_emotional_valence(is_misinformation: bool, rng: np.random.Generator) -> float:
    """Sample emotional valence using Beta distribution (Phase 2 Step 2.1).

    Misinformation content: Beta(3, 1) — skewed toward high valence.
    Normal content: Beta(1, 3) — skewed toward low valence.
    """
    if is_misinformation:
        return _clamp_probability(
            float(rng.beta(MISINFO_VALENCE_ALPHA, MISINFO_VALENCE_BETA))
        )
    return _clamp_probability(
        float(rng.beta(NORMAL_VALENCE_ALPHA, NORMAL_VALENCE_BETA))
    )


def generate_content_item(
    agent: AgentForContent,
    content_id: int,
    timestamp: int,
    rng: np.random.Generator,
) -> Content:
    """Create a content item for a specific agent at the current tick."""
    ideological_score = _clamp_opinion(
        float(rng.normal(loc=agent.opinion, scale=IDEOLOGICAL_NOISE_STD))
    )
    misinfo_score = sample_misinfo_score(agent, rng)
    is_misinformation = bool(misinfo_score > 0.5)
    emotional_valence = sample_emotional_valence(is_misinformation, rng)
    virality = float(rng.uniform(VIRALITY_MIN, VIRALITY_MAX))
    source_credibility = _clamp_probability(agent.trust)
    belief_update_weight = compute_belief_update_weight(
        misinfo_score=misinfo_score,
        source_credibility=source_credibility,
        media_literacy_i=agent.media_literacy,
    )

    return Content(
        id=content_id,
        creator_id=agent.id,
        timestamp=timestamp,
        ideological_score=ideological_score,
        emotional_valence=emotional_valence,
        misinfo_score=misinfo_score,
        virality=virality,
        source_credibility=source_credibility,
        is_misinformation=is_misinformation,
        belief_update_weight=belief_update_weight,
        topic_vector=[],
        coordinated_campaign_id=None,
        is_satire=False,
    )


def maybe_generate_content(
    agent: AgentForContent,
    content_id: int,
    timestamp: int,
    rng: np.random.Generator,
) -> Content | None:
    """Generate content with probability equal to the agent's activity rate."""
    activity_rate = _clamp_probability(agent.activity_rate)
    if rng.random() >= activity_rate:
        return None
    return generate_content_item(agent=agent, content_id=content_id, timestamp=timestamp, rng=rng)


# TODO (Phase 2): wire belief_update_weight into compute_update() as a per-content
# modulator on influence weights after emotion/media-literacy integration is added.


__all__ = [
    "Content",
    "compute_belief_update_weight",
    "generate_content_item",
    "maybe_generate_content",
    "sample_emotional_valence",
    "sample_misinfo_score",
]
