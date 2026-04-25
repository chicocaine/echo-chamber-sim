"""FastAPI entrypoint for the simulation backend (MVP Step 8)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import orjson

from api.schemas import SimConfig
from sim.simulation import run_simulation


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
    return Response(content=orjson.dumps(result), media_type="application/json")


@app.get("/defaults")
async def defaults_endpoint() -> Response:
    """Return default simulation configuration values."""
    defaults = SimConfig().model_dump()
    return Response(content=orjson.dumps(defaults), media_type="application/json")
