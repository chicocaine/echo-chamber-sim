"""Agent models and initialization utilities for the echo chamber simulation.

MVP Step 2 scope:
- Base agent state schema
- Stubborn (Friedkin-Johnsen) and Flexible (DeGroot) update rules
- Zealot/Bot behavior via parameterized StubbornAgent instances
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import floor
import random
from typing import Literal

import numpy as np


AgentType = Literal[
    "stubborn",
    "flexible",
    "zealot",
    "bot",
    "hk",
    "contrarian",
    "influencer",
    "passive",
]
SIRState = Literal["S", "I", "R"]


# Default means/stds and ranges from agent reference Part 1.3 and Part 8.
STUBBORNNESS_MIN = 0.1
STUBBORNNESS_MAX = 0.3
SUSCEPTIBILITY_MEAN = 0.5
SUSCEPTIBILITY_STD = 0.1
TRUST_MEAN = 0.5
TRUST_STD = 0.2
EXPERTISE_MEAN = 0.5
EXPERTISE_STD = 0.2
STANDARD_ACTIVITY_MEAN = 0.5
STANDARD_ACTIVITY_STD = 0.15
BOT_ACTIVITY_MEAN = 0.9
BOT_ACTIVITY_STD = 0.05
PASSIVE_ACTIVITY_MEAN = 0.07
PASSIVE_ACTIVITY_STD = 0.03
MEDIA_LITERACY_MIN = 0.2
MEDIA_LITERACY_MAX = 0.8
HK_CONFIDENCE_BOUND_DEFAULT = 0.3
AROUSAL_TOLERANCE_EFFECT_DEFAULT = 0.4
CONTRARIAN_PROB_DEFAULT = 0.3
INFLUENCE_WEIGHT_MULTIPLIER_DEFAULT = 2.0
BIMODAL_PEAK = 0.7
BIMODAL_STD = 0.15


def _clamp_opinion(value: float) -> float:
    """Clamp opinions to the invariant simulation range [-1.0, 1.0]."""
    return float(max(-1.0, min(1.0, value)))


def _clip_probability(value: float) -> float:
    """Clip scalar values to [0.0, 1.0]."""
    return float(np.clip(value, 0.0, 1.0))


def _assert_probability(name: str, value: float) -> None:
    """Enforce probability invariants at initialization time."""
    assert 0.0 <= value <= 1.0, f"{name} must be in [0, 1], got {value}"


def _normalize_influence_weights(
    neighbors: list["Agent"],
    influence_weights: dict[int, float],
) -> dict[int, float]:
    """Return normalized influence weights for the provided neighbors.

    Falls back to uniform weights if all provided weights are missing or zero.
    """
    if not neighbors:
        return {}

    raw_weights = {
        neighbor.id: max(0.0, float(influence_weights.get(neighbor.id, 0.0)))
        for neighbor in neighbors
    }
    total = sum(raw_weights.values())

    if total <= 0.0:
        uniform = 1.0 / len(neighbors)
        return {neighbor.id: uniform for neighbor in neighbors}

    return {neighbor_id: weight / total for neighbor_id, weight in raw_weights.items()}


@dataclass(slots=True)
class Agent:
    """Base agent schema shared by all archetypes.

    Subclasses must implement ``compute_update`` to return the next opinion.
    """

    id: int
    agent_type: AgentType
    opinion: float
    initial_opinion: float
    stubbornness: float
    susceptibility: float
    trust: float
    expertise: float
    activity_rate: float
    emotional_arousal: float
    media_literacy: float
    confidence_bound: float
    arousal_tolerance_effect: float
    contrarian_prob: float
    influence_weight_multiplier: float
    suspicion_score: float
    is_active: bool
    sir_states: dict[int, str] = field(default_factory=dict)  # campaign_id -> SIR state
    opinion_history: list[float] = field(default_factory=list)
    misinfo_rate: float = 0.0
    exposure_count: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.opinion = _clamp_opinion(self.opinion)
        self.initial_opinion = _clamp_opinion(self.initial_opinion)

        _assert_probability("stubbornness", self.stubbornness)
        _assert_probability("susceptibility", self.susceptibility)
        _assert_probability("trust", self.trust)
        _assert_probability("expertise", self.expertise)
        _assert_probability("activity_rate", self.activity_rate)
        _assert_probability("emotional_arousal", self.emotional_arousal)
        _assert_probability("media_literacy", self.media_literacy)
        _assert_probability("confidence_bound", self.confidence_bound)
        _assert_probability("arousal_tolerance_effect", self.arousal_tolerance_effect)
        _assert_probability("contrarian_prob", self.contrarian_prob)
        _assert_probability("suspicion_score", self.suspicion_score)
        _assert_probability("misinfo_rate", self.misinfo_rate)

        assert self.id >= 0, f"id must be non-negative, got {self.id}"
        assert self.agent_type in {
            "stubborn",
            "flexible",
            "zealot",
            "bot",
            "hk",
            "contrarian",
            "influencer",
            "passive",
        }
        for cid, state in self.sir_states.items():
            assert isinstance(cid, int), f"campaign_id must be int, got {type(cid)}"
            assert state in {"S", "I", "R"}, f"invalid SIR state for campaign {cid}: {state}"

        if not self.opinion_history:
            # Tick-0 history is required for downstream bot detection signals.
            self.opinion_history.append(self.opinion)

    def compute_update(
        self,
        neighbors: list["Agent"],
        influence_weights: dict[int, float],
    ) -> float:
        """Compute and return the next opinion value.

        Subclasses override this method.
        """
        raise NotImplementedError("Subclasses must implement compute_update")


@dataclass(slots=True)
class StubbornAgent(Agent):
    """Friedkin-Johnsen agent update model.

    Formula (agent reference Part 5):
        x_i(t+1) = g_i * x_i(0) + (1 - g_i) * sum_j(a_ij * x_j(t))
    """

    def compute_update(
        self,
        neighbors: list[Agent],
        influence_weights: dict[int, float],
    ) -> float:
        if self.stubbornness >= 1.0:
            # Zealots and bots are configured as fully stubborn.
            return self.opinion
        if not neighbors:
            return self.opinion

        weighted_avg = sum(
            influence_weights.get(neighbor.id, 0.0) * neighbor.opinion
            for neighbor in neighbors
        )
        updated = self.stubbornness * self.initial_opinion + (1.0 - self.stubbornness) * weighted_avg
        return _clamp_opinion(updated)


@dataclass(slots=True)
class FlexibleAgent(Agent):
    """DeGroot social learner update model.

    Formula (agent reference Part 5):
        x_i(t+1) = sum_j(a_ij * x_j(t))
    """

    def __post_init__(self) -> None:
        # Flexible agents are pure DeGroot learners with no self-weight.
        self.stubbornness = 0.0
        Agent.__post_init__(self)

    def compute_update(
        self,
        neighbors: list[Agent],
        influence_weights: dict[int, float],
    ) -> float:
        if not neighbors:
            return self.opinion

        weighted_avg = sum(
            influence_weights.get(neighbor.id, 0.0) * neighbor.opinion
            for neighbor in neighbors
        )
        return _clamp_opinion(weighted_avg)


@dataclass(slots=True)
class HKAgent(Agent):
    """Hegselmann-Krause bounded-confidence agent update model.

    Formula (agent reference Part 5):
        N_i(t) = {j : |x_j(t) - x_i(t)| <= epsilon}
        x_i(t+1) = (1 / |N_i(t)|) * sum_{j in N_i(t)} x_j(t)
    """

    def compute_update(
        self,
        neighbors: list[Agent],
        influence_weights: dict[int, float],
    ) -> float:
        if not neighbors:
            return self.opinion

        epsilon_eff = float(self.confidence_bound) * (
            1.0 - float(self.arousal_tolerance_effect) * float(self.emotional_arousal)
        )
        epsilon = float(np.clip(epsilon_eff, 0.0, 1.0))
        opinion = float(self.opinion)
        in_bound = [neighbor.opinion for neighbor in neighbors if abs(neighbor.opinion - opinion) <= epsilon]
        if not in_bound:
            return self.opinion

        return _clamp_opinion(float(np.mean(in_bound)))


@dataclass(slots=True)
class ContrarianAgent(Agent):
    """Contrarian agent update model.

    With probability p_c, move away from neighbor social pressure; otherwise
    follow the standard Friedkin-Johnsen update rule (agent reference Part 5).
    """

    def compute_update(
        self,
        neighbors: list[Agent],
        influence_weights: dict[int, float],
    ) -> float:
        if not neighbors:
            return self.opinion

        neighbor_avg = sum(
            influence_weights.get(neighbor.id, 0.0) * neighbor.opinion
            for neighbor in neighbors
        )
        influence_delta = float(neighbor_avg) - float(self.opinion)

        if random.random() < float(self.contrarian_prob):
            updated = float(self.opinion) - (1.0 - float(self.stubbornness)) * influence_delta
        else:
            updated = float(self.stubbornness) * float(self.initial_opinion) + (
                1.0 - float(self.stubbornness)
            ) * float(neighbor_avg)

        return _clamp_opinion(updated)


@dataclass(slots=True)
class InfluencerAgent(StubbornAgent):
    """Influencer agent with standard FJ update and boosted influence weight."""


def sample_initial_opinion(
    rng: np.random.Generator,
    distribution: Literal["uniform", "bimodal"] = "uniform",
) -> float:
    """Sample initial opinion using supported start distributions."""
    if distribution == "uniform":
        return _clamp_opinion(float(rng.uniform(-1.0, 1.0)))
    if distribution == "bimodal":
        peak = BIMODAL_PEAK if float(rng.random()) >= 0.5 else -BIMODAL_PEAK
        return _clamp_opinion(float(rng.normal(loc=peak, scale=BIMODAL_STD)))
    raise ValueError(f"Unsupported initial opinion distribution: {distribution}")


def _sample_truncated_normal(rng: np.random.Generator, mean: float, std: float) -> float:
    """Sample a normal random variable clipped to [0, 1]."""
    return _clip_probability(float(rng.normal(loc=mean, scale=std)))


def _sample_activity_rate(rng: np.random.Generator, agent_type: AgentType) -> float:
    """Sample activity rate based on archetype defaults."""
    if agent_type == "bot":
        return _sample_truncated_normal(rng, BOT_ACTIVITY_MEAN, BOT_ACTIVITY_STD)
    if agent_type == "passive":
        return _sample_truncated_normal(rng, PASSIVE_ACTIVITY_MEAN, PASSIVE_ACTIVITY_STD)
    return _sample_truncated_normal(rng, STANDARD_ACTIVITY_MEAN, STANDARD_ACTIVITY_STD)


def create_agent(
    agent_id: int,
    agent_type: AgentType,
    rng: np.random.Generator,
    initial_opinion_distribution: Literal["uniform", "bimodal"] = "uniform",
    bot_misinfo_rate: float = 1.0,
    arousal_tolerance_effect: float = AROUSAL_TOLERANCE_EFFECT_DEFAULT,
    media_literacy_boost: float = 0.0,
) -> Agent:
    """Create a single agent with Step 2 defaults.

    Zealot and bot behavior are parameter configurations of ``StubbornAgent``.
    """
    _assert_probability("bot_misinfo_rate", bot_misinfo_rate)

    opinion = sample_initial_opinion(rng, initial_opinion_distribution)
    initial_opinion = opinion
    stubbornness = float(rng.uniform(STUBBORNNESS_MIN, STUBBORNNESS_MAX))
    misinfo_rate = 0.0

    contrarian_prob = 0.0

    influence_weight_multiplier = 1.0

    if agent_type in {"zealot", "bot"}:
        extreme = 1.0 if float(rng.random()) >= 0.5 else -1.0
        opinion = extreme
        initial_opinion = extreme
        stubbornness = 1.0
    elif agent_type == "flexible":
        stubbornness = 0.0
    elif agent_type == "contrarian":
        contrarian_prob = CONTRARIAN_PROB_DEFAULT
    elif agent_type == "influencer":
        influence_weight_multiplier = INFLUENCE_WEIGHT_MULTIPLIER_DEFAULT

    if agent_type == "bot":
        misinfo_rate = bot_misinfo_rate

    common_kwargs = dict(
        id=agent_id,
        agent_type=agent_type,
        opinion=opinion,
        initial_opinion=initial_opinion,
        stubbornness=stubbornness,
        susceptibility=_sample_truncated_normal(rng, SUSCEPTIBILITY_MEAN, SUSCEPTIBILITY_STD),
        trust=_sample_truncated_normal(rng, TRUST_MEAN, TRUST_STD),
        expertise=_sample_truncated_normal(rng, EXPERTISE_MEAN, EXPERTISE_STD),
        activity_rate=_sample_activity_rate(rng, agent_type),
        emotional_arousal=0.0,
        media_literacy=_clip_probability(
            float(rng.uniform(MEDIA_LITERACY_MIN, MEDIA_LITERACY_MAX))
            + media_literacy_boost
        ),
        confidence_bound=HK_CONFIDENCE_BOUND_DEFAULT,
        arousal_tolerance_effect=arousal_tolerance_effect,
        contrarian_prob=contrarian_prob,
        influence_weight_multiplier=influence_weight_multiplier,
        suspicion_score=0.0,
        is_active=True,
        sir_states={},
        opinion_history=[opinion],
        misinfo_rate=misinfo_rate,
    )

    if agent_type == "flexible":
        return FlexibleAgent(**common_kwargs)
    if agent_type == "hk":
        return HKAgent(**common_kwargs)
    if agent_type == "contrarian":
        return ContrarianAgent(**common_kwargs)
    if agent_type == "influencer":
        return InfluencerAgent(**common_kwargs)
    return StubbornAgent(**common_kwargs)


def initialize_agents(
    n_agents: int,
    agent_mix: dict[AgentType, float],
    seed: int,
    initial_opinion_distribution: Literal["uniform", "bimodal"] = "uniform",
    bot_misinfo_rate: float = 1.0,
    arousal_tolerance_effect: float = AROUSAL_TOLERANCE_EFFECT_DEFAULT,
    media_literacy_boost: float = 0.0,
) -> list[Agent]:
    """Initialize a population of agents from a fractional type mix.

    Agent IDs are unique, sequential integers in [0, n_agents-1].
    """
    if n_agents <= 0:
        raise ValueError("n_agents must be > 0")
    if not agent_mix:
        raise ValueError("agent_mix must not be empty")

    mix_total = 0.0
    for agent_type, fraction in agent_mix.items():
        if agent_type not in {
            "stubborn",
            "flexible",
            "zealot",
            "bot",
            "hk",
            "contrarian",
            "influencer",
            "passive",
        }:
            raise ValueError(f"Unsupported agent type in mix: {agent_type}")
        _assert_probability(f"agent_mix[{agent_type}]", fraction)
        mix_total += fraction

    if not np.isclose(mix_total, 1.0):
        raise ValueError(f"agent_mix fractions must sum to 1.0, got {mix_total}")

    rng = np.random.default_rng(seed)

    exact_counts = {agent_type: fraction * n_agents for agent_type, fraction in agent_mix.items()}
    base_counts = {agent_type: int(floor(count)) for agent_type, count in exact_counts.items()}

    assigned = sum(base_counts.values())
    remainder = n_agents - assigned
    if remainder > 0:
        fractional_parts = sorted(
            (
                (agent_type, exact_counts[agent_type] - base_counts[agent_type])
                for agent_type in agent_mix
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        for idx in range(remainder):
            base_counts[fractional_parts[idx][0]] += 1

    type_assignments: list[AgentType] = []
    for agent_type, count in base_counts.items():
        type_assignments.extend([agent_type] * count)
    rng.shuffle(type_assignments)

    agents = [
        create_agent(
            agent_id=agent_id,
            agent_type=type_assignments[agent_id],
            rng=rng,
            initial_opinion_distribution=initial_opinion_distribution,
            bot_misinfo_rate=bot_misinfo_rate,
            arousal_tolerance_effect=arousal_tolerance_effect,
            media_literacy_boost=media_literacy_boost,
        )
        for agent_id in range(n_agents)
    ]

    return agents


__all__ = [
    "Agent",
    "AgentType",
    "ContrarianAgent",
    "FlexibleAgent",
    "HKAgent",
    "InfluencerAgent",
    "SIRState",
    "StubbornAgent",
    "create_agent",
    "initialize_agents",
    "sample_initial_opinion",
]
