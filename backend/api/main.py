"""FastAPI entrypoint for the simulation backend.

Endpoints:
  POST /run             — single simulation, returns full result with snapshots
  POST /run/replicated  — multi-seed replication with aggregated metrics
  POST /run/compare     — baseline vs intervention comparison with IES scores
  GET  /defaults        — default simulation configuration
  GET  /presets         — curated preset configurations for standout experiments
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import orjson
from pydantic import BaseModel

from api.schemas import CompareRequest, IES_METRIC_NAMES, SimConfig
from sim.metrics import compute_ies
from sim.simulation import run_replicated, run_simulation


app = FastAPI(title="Echo Chamber Simulation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Presets — curated configurations for standout experiment scenarios
# ---------------------------------------------------------------------------

def _mix(**overrides: float) -> dict[str, float]:
    m: dict[str, float] = {
        "stubborn": 0.60, "flexible": 0.20, "passive": 0.10,
        "zealot": 0.05, "bot": 0.05,
        "hk": 0.0, "contrarian": 0.0, "influencer": 0.0,
    }
    m.update(overrides)
    return m

PRESETS: list[dict[str, Any]] = [
    {
        "id": "baseline",
        "label": "Baseline (Default)",
        "description": "Standard configuration with 5% bots, alpha=0.65, Watts-Strogatz topology. "
                       "Polarization ~0.24, Assortativity ~0.30, Misinfo ~30%. "
                       "Use as a reference point for comparing other presets.",
        "config": SimConfig().model_dump(),
    },
    {
        "id": "no_bots",
        "label": "No Bots — Clean Platform",
        "description": "Zero bot agents — the only configuration that achieves 0% misinformation prevalence. "
                       "Polarization drops to 0.18 (lowest without structural changes). "
                       "Demonstrates that bot removal is the single most effective intervention.",
        "config": {
            **SimConfig().model_dump(),
            "agent_mix": _mix(bot=0.0, stubborn=0.63, flexible=0.21, passive=0.11, zealot=0.05),
        },
    },
    {
        "id": "misinfo_storm",
        "label": "Misinfo Storm — Worst Case",
        "description": "Highest polarization of all experiments (0.54). "
                       "25% bots, alpha=0.8, high infection rate (β=0.5), slow recovery (γ=0.02), "
                       "no virality dampening. Shows multiplicative risk when multiple factors combine.",
        "config": {
            **SimConfig().model_dump(),
            "agent_mix": _mix(bot=0.25, stubborn=0.40, flexible=0.15, passive=0.10, zealot=0.10),
            "alpha": 0.8,
            "sir_beta": 0.5,
            "sir_gamma": 0.02,
            "virality_dampening": 0.0,
        },
    },
    {
        "id": "echo_chamber_max",
        "label": "Echo Chamber — Maximum",
        "description": "Highest assortativity (0.87) — the structural signature of extreme echo chambers. "
                       "80% stubborn agents, alpha=0.7, 10% rewire rate, tight homophily (0.15). "
                       "Agents aggressively cluster with like-minded peers creating ideological silos. "
                       "Polarization is paradoxically low (0.15) because within-cluster agreement is high.",
        "config": {
            **SimConfig().model_dump(),
            "agent_mix": _mix(stubborn=0.80, flexible=0.05, passive=0.05, bot=0.05, zealot=0.05),
            "alpha": 0.7,
            "dynamic_rewire_rate": 0.10,
            "homophily_threshold": 0.15,
        },
    },
    {
        "id": "echo_chamber_min",
        "label": "Echo Chamber — Minimum",
        "description": "Lowest assortativity (0.17) achieved via high diversity ratio. "
                       "80% flexible agents, alpha=0.1, diversity_ratio=0.5, no rewiring. "
                       "⚠ WARNING: diversity_ratio=0.5 causes 99.8% misinformation prevalence. "
                       "Good structural metrics but catastrophic misinformation outcome.",
        "config": {
            **SimConfig().model_dump(),
            "agent_mix": _mix(stubborn=0.05, flexible=0.80, passive=0.05, bot=0.05, zealot=0.05),
            "alpha": 0.1,
            "diversity_ratio": 0.5,
            "dynamic_rewire_rate": 0.0,
        },
    },
    {
        "id": "churn_enabled",
        "label": "Churn Enabled — User Exodus",
        "description": "Lowest polarization of all experiments (0.10). "
                       "Dissatisfied agents leave the platform, removing cross-cutting edges. "
                       "Polarization plummets but assortativity soars to 0.83 and E-I drops to -0.54 "
                       "as remaining agents form ideologically pure clusters. "
                       "Demonstrates the polarization/assortativity tradeoff.",
        "config": {
            **SimConfig().model_dump(),
            "enable_churn": True,
            "churn_base": -4.0,
            "churn_weight": 1.0,
        },
    },
    {
        "id": "high_rewire",
        "label": "High Rewire — Aggressive Clustering",
        "description": "Dynamic rewire rate 0.10 with default parameters. "
                       "Assortativity 0.79, polarization 0.17 — the classic echo chamber signature. "
                       "Agents rapidly unfollow disagreeing neighbors and seek similar ones. "
                       "Low polarization hides the fact that cross-cutting exposure has collapsed (E-I = -0.43).",
        "config": {
            **SimConfig().model_dump(),
            "dynamic_rewire_rate": 0.10,
        },
    },
    {
        "id": "bimodal_polarized",
        "label": "Bimodal + High Alpha — Pre-Polarized",
        "description": "Pre-polarized population (opinions clustered at ±0.7) with strong personalization (alpha=0.9) "
                       "and 70% stubborn agents. Polarization 0.28 — 18% higher than uniform-start baseline. "
                       "Shows how pre-existing societal polarization is amplified by algorithmic curation.",
        "config": {
            **SimConfig().model_dump(),
            "initial_opinion_distribution": "bimodal",
            "alpha": 0.9,
            "agent_mix": _mix(stubborn=0.70, flexible=0.10, passive=0.10, bot=0.05, zealot=0.05),
        },
    },
    {
        "id": "media_literate",
        "label": "Media Literacy — Educated Population",
        "description": "All agents receive +0.4 media literacy boost, improving their ability to spot misinformation. "
                       "10% bots. Misinfo prevalence 44% (vs 65% in misinfo storm) despite high bot presence. "
                       "A moderate but net-positive intervention with no adverse structural side effects.",
        "config": {
            **SimConfig().model_dump(),
            "media_literacy_boost": 0.4,
            "agent_mix": _mix(bot=0.10, stubborn=0.50, flexible=0.20, passive=0.10, zealot=0.10),
        },
    },
    {
        "id": "diverse_backfire",
        "label": "Diverse Content — ⚠ Backfire Warning",
        "description": "Diversity ratio 0.5 — exposes agents to opposing content across the network. "
                       "Reduces assortativity to 0.17 and improves E-I to +0.06 (near-perfect cross-cutting). "
                       "⚠ CRITICAL: misinformation prevalence reaches 99.8%. "
                       "Naive content diversity floods feeds with bot-generated misinformation from distant peers. "
                       "Demonstrates that diversity interventions must be paired with content quality filters.",
        "config": {
            **SimConfig().model_dump(),
            "diversity_ratio": 0.5,
        },
    },
    {
        "id": "balanced_optimal",
        "label": "Balanced — Recommended",
        "description": "The recommended balanced configuration from experimental analysis. "
                       "Zero bots, alpha=0.3, content-based recommender, uniform initial opinions, no churn. "
                       "Expected: polarization ~0.18, assortativity ~0.28, misinfo 0%, entropy ~1.9. "
                       "Optimizes across all metrics with the fewest tradeoffs.",
        "config": {
            **SimConfig().model_dump(),
            "agent_mix": _mix(bot=0.0, stubborn=0.68, flexible=0.22, passive=0.08, zealot=0.02),
            "alpha": 0.3,
            "recommender_type": "content_based",
            "dynamic_rewire_rate": 0.01,
            "diversity_ratio": 0.0,
            "enable_churn": False,
            "initial_opinion_distribution": "uniform",
            "media_literacy_boost": 0.2,
        },
    },
    {
        "id": "ba_topology",
        "label": "Scale-Free Network — Barabási–Albert",
        "description": "Barabási–Albert topology with hub nodes and power-law degree distribution. "
                       "Assortativity drops to 0.17 (vs 0.30 for Watts-Strogatz) and modularity to 0.25 (vs 0.56). "
                       "Hub agents create natural cross-cutting connections. "
                       "More realistic for large social platforms with influencer dynamics.",
        "config": {
            **SimConfig().model_dump(),
            "topology": "barabasi_albert",
        },
    },
    {
        "id": "graph_recommender",
        "label": "Graph Recommender — Network Amplifier",
        "description": "Graph-based random-walk recommender. Produces the second-highest assortativity (0.40) "
                       "after the echo chamber max preset. Misinfo prevalence 75% — more than double content-based. "
                       "Shows how graph-structure-aware recommenders can amplify existing network segregation.",
        "config": {
            **SimConfig().model_dump(),
            "recommender_type": "graph",
        },
    },
]


@app.post("/run")
async def run_endpoint(config: SimConfig) -> Response:
    """Run a simulation and return the full result including snapshots for replay."""
    result = await asyncio.to_thread(run_simulation, config.model_dump())
    return Response(
        content=orjson.dumps(result, option=orjson.OPT_NON_STR_KEYS),
        media_type="application/json",
    )


class ReplicatedRequest(BaseModel):
    config: SimConfig
    n_runs: int = 10


@app.post("/run/replicated")
async def run_replicated_endpoint(req: ReplicatedRequest) -> Response:
    """Run n_runs replicates with deterministic seeds and return aggregated metrics."""
    cfg = req.config.model_dump()
    n_runs = int(req.n_runs)

    header_warning = None
    if n_runs * int(cfg.get("T", 0)) > 50000:
        header_warning = "Warning: large request (n_runs * T > 50000)"

    result = await asyncio.to_thread(run_replicated, cfg, n_runs=n_runs)

    headers = {}
    if header_warning:
        headers["X-Experiment-Warning"] = header_warning

    return Response(
        content=orjson.dumps(result, option=orjson.OPT_NON_STR_KEYS),
        media_type="application/json",
        headers=headers,
    )


@app.post("/run/compare")
async def run_compare_endpoint(req: CompareRequest) -> Response:
    """Compare baseline vs intervention with multi-run replication.

    Returns aggregated results for both scenarios and IES scores per metric.
    """
    baseline_cfg = req.baseline.model_dump()
    intervention_cfg = req.intervention.model_dump()
    n_runs = int(req.n_runs)

    baseline_result = await asyncio.to_thread(run_replicated, baseline_cfg, n_runs=n_runs)
    intervention_result = await asyncio.to_thread(run_replicated, intervention_cfg, n_runs=n_runs)

    base_agg = baseline_result["aggregated"]
    int_agg = intervention_result["aggregated"]
    ies_scores: dict[str, float] = {}
    for metric in IES_METRIC_NAMES:
        mean_key = f"{metric}_mean"
        if mean_key in base_agg and mean_key in int_agg:
            base_final = float(base_agg[mean_key][-1])
            int_final = float(int_agg[mean_key][-1])
            ies_scores[metric] = compute_ies(base_final, int_final)

    return Response(
        content=orjson.dumps(
            {
                "baseline": baseline_result,
                "intervention": intervention_result,
                "ies": ies_scores,
            },
            option=orjson.OPT_NON_STR_KEYS,
        ),
        media_type="application/json",
    )


@app.get("/defaults")
async def defaults_endpoint() -> Response:
    """Return the default simulation configuration."""
    defaults = SimConfig().model_dump()
    return Response(
        content=orjson.dumps(defaults, option=orjson.OPT_NON_STR_KEYS),
        media_type="application/json",
    )


@app.get("/presets")
async def presets_endpoint() -> Response:
    """Return curated preset configurations for standout experiment scenarios."""
    return Response(
        content=orjson.dumps(PRESETS, option=orjson.OPT_NON_STR_KEYS),
        media_type="application/json",
    )
