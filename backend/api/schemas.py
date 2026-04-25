"""Pydantic schemas for the simulation API (MVP Step 8)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SimConfig(BaseModel):
    """Configuration payload for running a simulation."""

    N: int = 200
    avg_degree: int = 16
    rewire_prob: float = 0.1
    T: int = 200
    snapshot_interval: int = 6
    alpha: float = 0.65
    beta_pop: float = 0.2
    k_exp: int = 20
    agent_mix: dict[str, float] = Field(
        default_factory=lambda: {
            "stubborn": 0.60,
            "flexible": 0.20,
            "zealot": 0.15,
            "bot": 0.05,
        }
    )
    sir_beta: float = 0.3
    sir_gamma: float = 0.05
    initial_opinion_distribution: Literal["uniform", "bimodal"] = "uniform"
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
        }
        for name, value in probability_fields.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

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
