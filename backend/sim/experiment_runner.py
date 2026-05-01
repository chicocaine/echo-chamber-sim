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

from .metrics import compute_ies
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


def format_policy_finding(
    intervention_result: dict[str, Any],
    baseline_result: dict[str, Any] | None = None,
    intervention_name: str = "Intervention",
) -> str:
    """Produce a standardized policy-finding text block (Phase 6 Step 6.3).

    Extracts key metrics from aggregated replicated results and generates
    a human-readable interpretation with tradeoff detection.
    """
    agg = intervention_result.get("aggregated", {})
    if not agg:
        return f"Intervention: {intervention_name}\n\nNo aggregated data available."

    def _final(metric: str) -> float:
        series = agg.get(f"{metric}_mean", [])
        return float(series[-1]) if series else 0.0

    def _final_std(metric: str) -> float:
        series = agg.get(f"{metric}_std", [])
        return float(series[-1]) if series else 0.0

    def _peak(metric: str) -> float:
        series = agg.get(f"{metric}_mean", [])
        return float(max(series)) if series else 0.0

    # Compute IES vs baseline if provided.
    ies_scores: dict[str, float] = {}
    if baseline_result is not None:
        base_agg = baseline_result.get("aggregated", {})
        for metric in [
            "assortativity",
            "misinfo_prevalence",
            "polarization_index",
            "opinion_entropy",
            "ei_index",
            "modularity_q",
        ]:
            base_mean = base_agg.get(f"{metric}_mean", [])
            int_mean = agg.get(f"{metric}_mean", [])
            if base_mean and int_mean:
                ies_scores[metric] = compute_ies(
                    float(base_mean[-1]), float(int_mean[-1])
                )

    peak_misinfo = _peak("misinfo_prevalence")
    assort = _final("assortativity")
    assort_std = _final_std("assortativity")
    entropy = _final("opinion_entropy")
    entropy_std = _final_std("opinion_entropy")
    ies_misinfo = ies_scores.get("misinfo_prevalence", 0.0)

    # Build output.
    lines: list[str] = []
    lines.append(f"Intervention: {intervention_name}")
    lines.append("")

    # Findings block.
    lines.append("Findings:")
    if ies_misinfo != 0.0:
        direction = "reduced" if ies_misinfo > 0 else "increased"
        pct = abs(ies_misinfo) * 100
        lines.append(
            f"- Peak misinformation {direction} by {pct:.1f}% (IES = {ies_misinfo:.2f})"
        )
    else:
        lines.append(f"- Peak misinformation: {peak_misinfo:.3f}")
    lines.append(f"- Final assortativity: {assort:.3f} ± {assort_std:.3f}")
    lines.append(f"- Opinion entropy: {entropy:.3f} ± {entropy_std:.3f}")
    lines.append("")

    # Auto-generated interpretation.
    lines.append("Interpretation:")
    if ies_misinfo > 0.1:
        lines.append(
            f"The intervention effectively reduced misinformation spread "
            f"({ies_misinfo * 100:.0f}% reduction vs baseline)."
        )
    elif ies_misinfo < -0.1:
        lines.append(
            f"The intervention unexpectedly increased misinformation spread "
            f"({-ies_misinfo * 100:.0f}% increase vs baseline)."
        )
    else:
        ies_assort = ies_scores.get("assortativity", 0.0)
        if ies_assort > 0.1:
            lines.append(
                "The intervention reduced echo chamber formation "
                "(lower assortativity vs baseline) but had limited effect on misinformation."
            )
        elif ies_assort < -0.1:
            lines.append(
                "The intervention increased echo chamber formation "
                "(higher assortativity vs baseline). Consider adjusting parameters."
            )
        else:
            lines.append(
                "The intervention showed mixed or negligible effects across all metrics."
            )
    lines.append("")

    # Tradeoff detection.
    tradeoffs = [
        metric
        for metric, ies in ies_scores.items()
        if ies < -0.05 and metric != "misinfo_prevalence"
    ]
    lines.append("Tradeoffs detected:")
    if tradeoffs:
        for metric in tradeoffs:
            lines.append(
                f"- {metric}: moved in the wrong direction "
                f"(IES = {ies_scores[metric]:.3f})"
            )
    else:
        lines.append("- No significant tradeoffs detected.")

    return "\n".join(lines)


__all__ = [
    "format_policy_finding",
    "run_experiment",
]
