"""Simulation baseline harness for MVP Step 1b profiling.

This file provides a profiling target before optimization work begins.
It will be replaced with full simulation logic in MVP Steps 2-7.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from statistics import fmean
from time import perf_counter

try:
    from sim.recommender import CandidateContent, generate_feed
except ModuleNotFoundError:
    from recommender import CandidateContent, generate_feed


try:
    profile  # type: ignore[name-defined]
except NameError:
    def profile(func):
        """No-op decorator when line_profiler is not active."""
        return func


# PERF BASELINE (recorded before optimization):
# run_simulation N=1000 T=100: 159.48s total (kernprof baseline)
# generate_feed: 97.7% of total time (~155.92s)
# opinion_update: 2.1% of total time (~3.36s)
# Baseline command: `kernprof -l sim/simulation.py`


@dataclass(slots=True)
class _AgentState:
    """Minimal internal agent state used for Step 1b profiling."""

    id: int
    opinion: float


def _clamp_opinion(value: float) -> float:
    """Clamp an opinion value to the simulation invariant range [-1.0, 1.0]."""
    return max(-1.0, min(1.0, value))


def _build_candidate_pool(size: int, rng: random.Random) -> list[CandidateContent]:
    """Create synthetic content pool for baseline feed-scoring load."""
    return [
        CandidateContent(
            ideological_score=rng.uniform(-1.0, 1.0),
            virality=rng.uniform(0.0, 1.0),
        )
        for _ in range(size)
    ]


@profile
def run_simulation(
    N: int = 1000,
    T: int = 100,
    k_exp: int = 20,
    alpha: float = 0.65,
    beta_pop: float = 0.2,
    candidate_pool_size: int = 500,
    seed: int = 42,
) -> dict[str, float | int]:
    """Run a baseline loop to profile feed generation and opinion update load.

    Returns summary timings so the same script can be used with or without
    line_profiler.
    """
    if N <= 0 or T <= 0:
        raise ValueError("N and T must be positive integers")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not 0.0 <= beta_pop <= 1.0:
        raise ValueError("beta_pop must be in [0, 1]")

    rng = random.Random(seed)
    agents = [_AgentState(id=i, opinion=rng.uniform(-1.0, 1.0)) for i in range(N)]

    feed_seconds = 0.0
    opinion_seconds = 0.0
    total_start = perf_counter()

    for _tick in range(T):
        candidate_pool = _build_candidate_pool(candidate_pool_size, rng)

        feed_start = perf_counter()
        feeds: dict[int, list[CandidateContent]] = {
            agent.id: generate_feed(
                agent=agent,
                candidate_pool=candidate_pool,
                k_exp=k_exp,
                alpha=alpha,
                beta_pop=beta_pop,
            )
            for agent in agents
        }
        feed_seconds += perf_counter() - feed_start

        update_start = perf_counter()
        new_opinions: dict[int, float] = {}
        for agent in agents:
            feed = feeds[agent.id]
            if not feed:
                new_opinions[agent.id] = agent.opinion
                continue

            neighbor_pressure = fmean(content.ideological_score for content in feed)
            updated = 0.2 * agent.opinion + 0.8 * neighbor_pressure
            new_opinions[agent.id] = _clamp_opinion(updated)

        for agent in agents:
            agent.opinion = new_opinions[agent.id]
        opinion_seconds += perf_counter() - update_start

    total_seconds = perf_counter() - total_start
    return {
        "N": N,
        "T": T,
        "total_seconds": round(total_seconds, 6),
        "feed_seconds": round(feed_seconds, 6),
        "opinion_update_seconds": round(opinion_seconds, 6),
    }


if __name__ == "__main__":
    summary = run_simulation()
    print(summary)
