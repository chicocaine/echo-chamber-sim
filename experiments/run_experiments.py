"""Experiment runner for echo chamber simulation parameter sweeps.

Taps into backend.sim to run parameterized simulations, saves per-run JSON
results with metadata, and identifies standout runs for further analysis.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure backend is importable from the experiments/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))

from sim.simulation import run_simulation, run_replicated, DEFAULT_CONFIG


RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _short_hash(params: dict[str, Any], length: int = 8) -> str:
    raw = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:length]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _make_run_id(params: dict[str, Any]) -> str:
    ts = _now_iso()
    h = _short_hash(params)
    return f"{ts}-{h}"


def run_one(config: dict[str, Any], n_runs: int = 3) -> dict[str, Any]:
    """Run a single replicated experiment and return the result envelope."""
    t0 = time.perf_counter()
    result = run_replicated(config, n_runs=n_runs)
    elapsed = time.perf_counter() - t0

    run_id = _make_run_id(config)
    envelope: dict[str, Any] = {
        "run_id": run_id,
        "datetime_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(elapsed, 3),
        "n_runs": n_runs,
        "parameters": config,
        "aggregated": result["aggregated"],
        "all_runs": result["all_runs"],
    }
    return envelope


def save_result(envelope: dict[str, Any]) -> Path:
    path = RESULTS_DIR / f"{envelope['run_id']}.json"
    with open(path, "w") as f:
        json.dump(envelope, f, indent=2, default=str)
    return path


# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

BASE: dict[str, Any] = {**DEFAULT_CONFIG, "N": 200, "T": 100, "snapshot_interval": 5}
N_RUNS = 3


def _mix(**overrides: float) -> dict[str, float]:
    m = {
        "stubborn": 0.60,
        "flexible": 0.20,
        "passive": 0.10,
        "zealot": 0.05,
        "bot": 0.05,
        "hk": 0.0,
        "contrarian": 0.0,
        "influencer": 0.0,
    }
    m.update(overrides)
    return m


EXPERIMENTS: list[tuple[str, dict[str, Any]]] = []

# 1 — Alpha sweep (personalization strength: core echo-chamber dial)
for a in [0.1, 0.3, 0.5, 0.65, 0.9]:
    EXPERIMENTS.append((f"alpha_{a}", {**BASE, "alpha": a}))

# 2 — Bot percentage sweep
for b_pct in [0.0, 0.05, 0.10, 0.20, 0.30]:
    mix = _mix(bot=b_pct, stubborn=0.60 - b_pct * 0.5, flexible=0.20 - b_pct * 0.3, passive=0.10 - b_pct * 0.1, zealot=0.05 - b_pct * 0.1)
    # Ensure non-negative
    mix = {k: max(0.0, v) for k, v in mix.items()}
    total = sum(mix.values())
    mix = {k: v / total for k, v in mix.items()}
    EXPERIMENTS.append((f"bots_{b_pct}", {**BASE, "agent_mix": mix}))

# 3 — Recommender type comparison
for rt in ["content_based", "cf", "graph", "hybrid"]:
    EXPERIMENTS.append((f"recommender_{rt}", {**BASE, "recommender_type": rt}))

# 4 — Topology comparison
for topo in ["watts_strogatz", "barabasi_albert", "erdos_renyi"]:
    EXPERIMENTS.append((f"topology_{topo}", {**BASE, "topology": topo}))

# 5 — Initial opinion distribution
for dist in ["uniform", "bimodal"]:
    EXPERIMENTS.append((f"initdist_{dist}", {**BASE, "initial_opinion_distribution": dist}))

# 6 — Virality dampening sweep
for vd in [0.0, 0.3, 0.6, 0.9]:
    EXPERIMENTS.append((f"viraldamp_{vd}", {**BASE, "virality_dampening": vd}))

# 7 — Diversity ratio sweep
for dr in [0.0, 0.2, 0.5]:
    EXPERIMENTS.append((f"diversity_{dr}", {**BASE, "diversity_ratio": dr}))

# 8 — Dynamic rewire rate sweep
for rr in [0.0, 0.01, 0.05, 0.10]:
    EXPERIMENTS.append((f"rewire_{rr}", {**BASE, "dynamic_rewire_rate": rr}))

# 9 — Churn enabled
EXPERIMENTS.append(("churn_on", {**BASE, "enable_churn": True}))

# 10 — High stubbornness (echo chamber maximizer)
EXPERIMENTS.append((
    "max_echo",
    {
        **BASE,
        "agent_mix": _mix(stubborn=0.80, flexible=0.05, passive=0.05, bot=0.05, zealot=0.05),
        "alpha": 0.9,
        "diversity_ratio": 0.0,
        "dynamic_rewire_rate": 0.05,
    },
))

# 11 — High flexibility (echo chamber minimizer)
EXPERIMENTS.append((
    "min_echo",
    {
        **BASE,
        "agent_mix": _mix(stubborn=0.05, flexible=0.80, passive=0.05, bot=0.05, zealot=0.05),
        "alpha": 0.1,
        "diversity_ratio": 0.5,
        "dynamic_rewire_rate": 0.0,
    },
))

# 12 — Bimodal + high alpha (worst-case polarization)
EXPERIMENTS.append((
    "bimodal_high_alpha",
    {
        **BASE,
        "initial_opinion_distribution": "bimodal",
        "alpha": 0.9,
        "agent_mix": _mix(stubborn=0.70, flexible=0.10, passive=0.10, bot=0.05, zealot=0.05),
    },
))

# 13 — High misinformation (bot heavy + no dampening)
EXPERIMENTS.append((
    "misinfo_storm",
    {
        **BASE,
        "agent_mix": _mix(bot=0.25, stubborn=0.40, flexible=0.15, passive=0.10, zealot=0.10),
        "virality_dampening": 0.0,
        "alpha": 0.8,
        "sir_beta": 0.5,
        "sir_gamma": 0.02,
    },
))

# 14 — Media literacy boost
EXPERIMENTS.append((
    "media_literate",
    {
        **BASE,
        "media_literacy_boost": 0.4,
        "agent_mix": _mix(bot=0.10, stubborn=0.50, flexible=0.20, passive=0.10, zealot=0.10),
    },
))

# 15 — High rewire + high homophily
EXPERIMENTS.append((
    "rewire_homophily",
    {
        **BASE,
        "dynamic_rewire_rate": 0.10,
        "homophily_threshold": 0.15,
        "alpha": 0.7,
    },
))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"Running {len(EXPERIMENTS)} experiment configurations...")
    print(f"Base: N={BASE['N']}, T={BASE['T']}, n_runs={N_RUNS}")
    print(f"Results dir: {RESULTS_DIR}")
    print("-" * 60)

    for idx, (label, config) in enumerate(EXPERIMENTS):
        print(f"\n[{idx + 1}/{len(EXPERIMENTS)}] {label} ...", flush=True)
        try:
            envelope = run_one(config, n_runs=N_RUNS)
            path = save_result(envelope)
            # Extract a few summary stats for immediate feedback
            agg = envelope["aggregated"]
            final_pol = agg.get("polarization_index_mean", [0])[-1] if agg.get("polarization_index_mean") else 0
            final_assort = agg.get("assortativity_mean", [0])[-1] if agg.get("assortativity_mean") else 0
            final_misinfo = agg.get("misinfo_prevalence_mean", [0])[-1] if agg.get("misinfo_prevalence_mean") else 0
            print(f"  -> {path.name}  ({envelope['runtime_seconds']:.1f}s)")
            print(f"     final_polarization={final_pol:.4f}  final_assortativity={final_assort:.4f}  final_misinfo={final_misinfo:.4f}")
        except Exception as exc:
            print(f"  FAILED: {exc}")

    print("\n" + "=" * 60)
    print("All experiments complete.")


if __name__ == "__main__":
    main()
