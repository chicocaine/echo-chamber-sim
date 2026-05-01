"""FastAPI entrypoint for the simulation backend (MVP Step 8)."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import orjson

import asyncio
import json

from fastapi import WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from api.schemas import CompareRequest, IES_METRIC_NAMES, SimConfig
from sim.metrics import compute_ies
from sim.simulation import run_replicated, run_simulation, run_simulation_streaming


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


def _sim_worker(
    config: dict[str, object],
    queue: asyncio.Queue[dict[str, object] | None],
    control_event: asyncio.Event,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Run the simulation generator in a background thread, feeding results to the queue."""
    def put(item: dict[str, object] | None) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, item)

    gen = run_simulation_streaming(config)
    try:
        for tick_data in gen:
            put(tick_data)
            control_event.wait()  # Block here if paused.
            control_event.clear()
        put({"type": "complete"})
    except Exception as exc:
        put({"type": "error", "message": str(exc)})


@app.websocket("/run/stream")
async def stream_simulation(websocket: WebSocket) -> None:
    """Stream simulation tick-by-tick over WebSocket (Phase 7 Step 7.1).

    Receives a SimConfig as JSON on connect, then streams one message per tick.
    Supports pause/resume/step/set_speed commands from the frontend.
    """
    await websocket.accept()
    config_raw = await websocket.receive_text()
    config = json.loads(config_raw)

    queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue(maxsize=10)
    control_event = asyncio.Event()
    control_event.set()  # Start in running state.
    ticks_per_second = 0
    step_mode = False

    loop = asyncio.get_running_loop()
    loop.run_in_executor(None, _sim_worker, config, queue, control_event, loop)

    # Coroutine that listens for control messages without blocking.
    async def recv_commands() -> None:
        nonlocal ticks_per_second, step_mode
        while True:
            try:
                raw = await asyncio.wait_for(websocket.receive_text(), timeout=0.1)
                cmd = json.loads(raw)
                action = cmd.get("command", "")
                if action == "pause":
                    control_event.clear()
                elif action == "resume":
                    control_event.set()
                elif action == "step":
                    step_mode = True
                    control_event.set()
                elif action == "set_speed":
                    ticks_per_second = cmd.get("ticks_per_second", 0)
            except asyncio.TimeoutError:
                continue
            except WebSocketDisconnect:
                break
            except Exception:
                break

    recv_task = asyncio.create_task(recv_commands())

    try:
        while True:
            tick_data = await queue.get()
            msg_type = tick_data.get("type", "")

            if msg_type == "complete":
                break
            if msg_type == "error":
                await websocket.send_json(tick_data)
                break

            await websocket.send_json(tick_data)

            if step_mode:
                control_event.clear()
                step_mode = False

            if ticks_per_second > 0:
                await asyncio.sleep(1.0 / ticks_per_second)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        recv_task.cancel()
        try:
            await websocket.close()
        except Exception:
            pass


@app.get("/defaults")
async def defaults_endpoint() -> Response:
    """Return default simulation configuration values."""
    defaults = SimConfig().model_dump()
    return Response(
        content=orjson.dumps(defaults, option=orjson.OPT_NON_STR_KEYS),
        media_type="application/json",
    )
