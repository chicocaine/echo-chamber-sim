"""Pydantic schemas for the simulation API (MVP Step 8)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

RecommenderType = Literal["content_based", "cf", "graph", "hybrid"]
TopologyType = Literal["watts_strogatz", "barabasi_albert", "erdos_renyi", "stochastic_block"]


class SimConfig(BaseModel):
    """Configuration payload for running a simulation."""

    N: int = 200
    avg_degree: int = 16
    rewire_prob: float = 0.1
    topology: TopologyType = "watts_strogatz"
    community_sizes: list[int] | None = None
    community_p: list[list[float]] | None = None
    T: int = 200
    snapshot_interval: int = 6
    alpha: float = 0.65
    beta_pop: float = 0.2
    k_exp: int = 20
    agent_mix: dict[str, float] = Field(
        default_factory=lambda: {
            "stubborn": 0.60,
            "flexible": 0.20,
            "passive": 0.10,
            "zealot": 0.05,
            "bot": 0.05,
            "hk": 0.0,
            "contrarian": 0.0,
            "influencer": 0.0,
        }
    )
    sir_beta: float = 0.3
    sir_gamma: float = 0.05
    reinforcement_factor: float = 0.0
    recommender_type: RecommenderType = "content_based"
    cf_blend_ratio: float = 0.5
    dynamic_rewire_rate: float = 0.01
    homophily_threshold: float = 0.3
    enable_churn: bool = False
    churn_base: float = -4.0
    churn_weight: float = 1.0
    diversity_ratio: float = 0.0
    lambda_penalty: float = 0.0
    virality_dampening: float = 0.0
    initial_opinion_distribution: Literal["uniform", "bimodal"] = "uniform"
    emotional_decay: float = 0.85
    arousal_share_weight: float = 0.3
    valence_share_weight: float = 0.4
    arousal_tolerance_effect: float = 0.4
    seed: int = 42

    @model_validator(mode="after")
    def validate_ranges(self) -> "SimConfig":
        """Validate scalar ranges and agent mix invariants."""
        if self.N <= 0:
            raise ValueError("N must be > 0")
        if self.avg_degree <= 0:
            raise ValueError("avg_degree must be > 0")
        if self.T <= 0:
            raise ValueError("T must be > 0")
        if self.snapshot_interval <= 0:
            raise ValueError("snapshot_interval must be > 0")
        if self.k_exp <= 0:
            raise ValueError("k_exp must be > 0")

        probability_fields = {
            "rewire_prob": self.rewire_prob,
            "alpha": self.alpha,
            "beta_pop": self.beta_pop,
            "sir_beta": self.sir_beta,
            "sir_gamma": self.sir_gamma,
            "reinforcement_factor": self.reinforcement_factor,
            "cf_blend_ratio": self.cf_blend_ratio,
            "dynamic_rewire_rate": self.dynamic_rewire_rate,
            "homophily_threshold": self.homophily_threshold,
            "diversity_ratio": self.diversity_ratio,
            "lambda_penalty": self.lambda_penalty,
            "virality_dampening": self.virality_dampening,
            "emotional_decay": self.emotional_decay,
            "arousal_share_weight": self.arousal_share_weight,
            "valence_share_weight": self.valence_share_weight,
            "arousal_tolerance_effect": self.arousal_tolerance_effect,
        }
        for name, value in probability_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        if self.topology == "stochastic_block":
            if self.community_sizes is None or self.community_p is None:
                raise ValueError("community_sizes and community_p required for stochastic_block")
            if sum(self.community_sizes) != self.N:
                raise ValueError(
                    f"community_sizes must sum to N ({self.N}), got {sum(self.community_sizes)}"
                )

        if self.churn_weight < 0.0:
            raise ValueError("churn_weight must be >= 0")

        for agent_type, fraction in self.agent_mix.items():
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(f"agent_mix[{agent_type}] must be in [0, 1]")
        mix_sum = sum(self.agent_mix.values())
        if abs(mix_sum - 1.0) > 1e-9:
            raise ValueError(f"agent_mix must sum to 1.0, got {mix_sum}")
        return self


class MetricSnapshot(BaseModel):
    """Metric values logged at a simulation tick."""

    tick: int
    opinion_variance: float
    polarization_index: float
    assortativity: float
    opinion_entropy: float
    misinfo_prevalence: float


class SimResult(BaseModel):
    """Simulation execution output payload."""

    config: SimConfig
    snapshots: list[MetricSnapshot]
    final_agents: list[dict]
    final_graph: dict
