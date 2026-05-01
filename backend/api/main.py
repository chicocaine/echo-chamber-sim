"""FastAPI entrypoint for the simulation backend (MVP Step 8)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import orjson

from api.schemas import CompareRequest, IES_METRIC_NAMES, SimConfig
from sim.metrics import compute_ies
from sim.simulation import run_replicated, run_simulation
from pydantic import BaseModel


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


@app.post("/run")
async def run_endpoint(config: SimConfig) -> Response:
    """Run a simulation synchronously and return the full result payload."""
    result = run_simulation(config.model_dump())
    return Response(
        content=orjson.dumps(result, option=orjson.OPT_NON_STR_KEYS),
        media_type="application/json",
    )


class ReplicatedRequest(BaseModel):
    config: SimConfig
    n_runs: int = 10


@app.post("/run/replicated")
async def run_replicated_endpoint(req: ReplicatedRequest) -> Response:
    """Run `n_runs` replicates of the simulation and return aggregated results.

    If the requested work is large (n_runs * T > 50000) the response will include
    a warning header to indicate the request may be slow.
    """
    cfg = req.config.model_dump()
    n_runs = int(req.n_runs)

    # Warning header when the work is large
    header_warning = None
    if n_runs * int(cfg.get("T", 0)) > 50000:
        header_warning = "Warning: large request (n_runs * T > 50000)"

    result = run_replicated(cfg, n_runs=n_runs)

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

    Returns aggregated results for both scenarios and an IES dict
    measuring intervention effectiveness per metric (Phase 6 Step 6.2).
    """
    baseline_cfg = req.baseline.model_dump()
    intervention_cfg = req.intervention.model_dump()
    n_runs = int(req.n_runs)

    baseline_result = run_replicated(baseline_cfg, n_runs=n_runs)
    intervention_result = run_replicated(intervention_cfg, n_runs=n_runs)

    # Compute IES per metric from the final-tick mean values.
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
    """Return default simulation configuration values."""
    defaults = SimConfig().model_dump()
    return Response(
        content=orjson.dumps(defaults, option=orjson.OPT_NON_STR_KEYS),
        media_type="application/json",
    )
