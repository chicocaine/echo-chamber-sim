"""FastAPI entrypoint for the simulation backend (MVP Step 8)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import orjson

from api.schemas import SimConfig
from sim.simulation import run_simulation
from sim.simulation import run_replicated
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


@app.get("/defaults")
async def defaults_endpoint() -> Response:
    """Return default simulation configuration values."""
    defaults = SimConfig().model_dump()
    return Response(
        content=orjson.dumps(defaults, option=orjson.OPT_NON_STR_KEYS),
        media_type="application/json",
    )
