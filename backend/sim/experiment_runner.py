"""Experiment runner for Phase 0 Step 0.2.

Provides `run_experiment` which enumerates a cartesian product over a
parameter grid, runs replicated simulations per combination (via
`run_replicated`) and writes standardized JSON outputs.

Outputs are written to `output_path/{scenario_name}.json`.
"""
from __future__ import annotations

import os
import itertools
from typing import Any, Dict, List
import orjson

from .simulation import run_replicated


def _make_scenario_name(prefix: str, params: Dict[str, Any]) -> str:
    """Create a compact, filesystem-safe scenario name.

    Sort params by key to ensure deterministic names.
    """
    parts: List[str] = [prefix] if prefix else []
    for k, v in sorted(params.items()):
        # Replace spaces and slashes in stringified values
        sval = str(v).replace(" ", "_").replace("/", "_")
        parts.append(f"{k}={sval}")
    return "__".join(parts)


def run_experiment(
    base_config: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    n_runs: int = 10,
    output_path: str = "results/",
) -> List[Dict[str, Any]]:
    """Run experiments over the cartesian product of `param_grid`.

    For each parameter combination:
    - merge with `base_config`
    - call `run_replicated(config, n_runs)`
    - write a JSON file to `output_path/{scenario_name}.json`

    Returns list of summary dictionaries for each scenario.
    """
    os.makedirs(output_path, exist_ok=True)

    # Create list of parameter names and corresponding lists
    keys = list(param_grid.keys())
    value_lists = [param_grid[k] for k in keys]

    results: List[Dict[str, Any]] = []

    for combo in itertools.product(*value_lists):
        params = dict(zip(keys, combo))
        config = {**base_config, **params}

        scenario_name = _make_scenario_name("scenario", params)
        filename = os.path.join(output_path, f"{scenario_name}.json")

        # Run replicated experiment (deterministic seeds inside)
        replicated = run_replicated(config, n_runs=n_runs)

        aggregated = replicated.get("aggregated", {})

        # Derive a compact metrics summary (final values + misinfo peak if present)
        metrics_summary: Dict[str, Any] = {}
        # final values: use last element of each *_mean/_std for base metrics
        for key in list(aggregated.keys()):
            if key == "tick":
                continue
            # we expect keys like "opinion_variance_mean" and "opinion_variance_std"
            # collect final values per metric base name
        base_metric_names = set()
        for k in aggregated.keys():
            if k == "tick":
                continue
            if k.endswith("_mean"):
                base_metric_names.add(k[: -5])

        for base in sorted(base_metric_names):
            mean_series = aggregated.get(f"{base}_mean", [])
            std_series = aggregated.get(f"{base}_std", [])
            if mean_series:
                metrics_summary[f"final_{base}_mean"] = float(mean_series[-1])
            if std_series:
                metrics_summary[f"final_{base}_std"] = float(std_series[-1])

        # misinfo peak (if misinfo_prevalence_mean exists)
        if "misinfo_prevalence_mean" in aggregated:
            metrics_summary["misinfo_peak_mean"] = float(
                max(aggregated["misinfo_prevalence_mean"])
            )
            metrics_summary["misinfo_peak_std"] = float(
                max(aggregated.get("misinfo_prevalence_std", [0.0]))
            )

        scenario_output = {
            "scenario_name": scenario_name,
            "parameters": params,
            "n_runs": n_runs,
            "aggregated": aggregated,
            "metrics": metrics_summary,
        }

        # Write JSON to disk using orjson for speed
        with open(filename, "wb") as f:
            f.write(orjson.dumps(scenario_output, option=orjson.OPT_NON_STR_KEYS))

        results.append(scenario_output)

    return results
