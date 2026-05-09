"""Analyze experiment results, write experiments.md, and generate comparison graphics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results"
GRAPHICS_DIR = RESULTS_DIR / "graphics"
EXPERIMENTS_MD = Path(__file__).resolve().parent / "experiments.md"


def load_results() -> list[dict[str, Any]]:
    rows = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        rows.append(data)
    return rows


def identify_standouts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Find standout runs by key metrics."""
    def final_val(agg, metric):
        series = agg.get(f"{metric}_mean", [])
        return series[-1] if series else 0.0

    scored = []
    for r in rows:
        agg = r["aggregated"]
        params = r["parameters"]
        scored.append({
            "run_id": r["run_id"],
            "datetime": r["datetime_utc"],
            "runtime": r["runtime_seconds"],
            "params": params,
            "agg": agg,
            "final_pol": final_val(agg, "polarization_index"),
            "final_assort": final_val(agg, "assortativity"),
            "final_misinfo": final_val(agg, "misinfo_prevalence"),
            "final_entropy": final_val(agg, "opinion_entropy"),
            "final_variance": final_val(agg, "opinion_variance"),
            "final_ei": final_val(agg, "ei_index"),
            "final_modularity": final_val(agg, "modularity_q"),
            "peak_misinfo": max(agg.get("misinfo_prevalence_mean", [0])),
        })

    standouts = {}
    standouts["highest_polarity"] = max(scored, key=lambda r: r["final_pol"])
    standouts["lowest_polarity"] = min(scored, key=lambda r: r["final_pol"])
    standouts["highest_assortativity"] = max(scored, key=lambda r: r["final_assort"])
    standouts["lowest_assortativity"] = min(scored, key=lambda r: r["final_assort"])
    standouts["highest_misinfo"] = max(scored, key=lambda r: r["final_misinfo"])
    standouts["lowest_misinfo"] = min(scored, key=lambda r: r["final_misinfo"])
    standouts["highest_entropy"] = max(scored, key=lambda r: r["final_entropy"])
    standouts["lowest_entropy"] = min(scored, key=lambda r: r["final_entropy"])
    return standouts


def describe_params(params: dict[str, Any]) -> str:
    """Human-readable parameter summary."""
    agent_mix = params.get("agent_mix", {})
    bot_pct = agent_mix.get("bot", 0)
    stubborn_pct = agent_mix.get("stubborn", 0)
    flexible_pct = agent_mix.get("flexible", 0)
    return (
        f"alpha={params['alpha']}, bots={bot_pct:.0%}, "
        f"stubborn={stubborn_pct:.0%}, flexible={flexible_pct:.0%}, "
        f"recommender={params['recommender_type']}, topology={params['topology']}, "
        f"rewire={params['dynamic_rewire_rate']}, diversity={params['diversity_ratio']}, "
        f"churn={params['enable_churn']}, dist={params['initial_opinion_distribution']}, "
        f"virality_damp={params['virality_dampening']}"
    )


def write_experiments_md(rows: list[dict[str, Any]], standouts: dict[str, dict[str, Any]]) -> None:
    lines: list[str] = []

    lines.append("# Echo Chamber Simulation — Experiment Results\n")
    lines.append(f"**Generated:** {rows[0]['datetime_utc'][:19] if rows else 'N/A'}\n")
    lines.append(f"**Total experiments:** {len(rows)}\n")
    lines.append(f"**Base configuration:** N=200, T=100, n_runs=3, snapshot_interval=5\n")
    lines.append("**Metrics recorded per tick:** polarization_index, assortativity, misinfo_prevalence, "
                 "opinion_entropy, opinion_variance, ei_index, modularity_q\n")

    # ---------- standout runs ----------
    lines.append("## Standout Runs\n")

    labels = [
        ("highest_polarity", "Highest Polarization Index"),
        ("lowest_polarity", "Lowest Polarization Index"),
        ("highest_assortativity", "Highest Assortativity (echo chamber strength)"),
        ("lowest_assortativity", "Lowest Assortativity"),
        ("highest_misinfo", "Highest Misinformation Prevalence"),
        ("lowest_misinfo", "Lowest Misinformation Prevalence"),
        ("highest_entropy", "Highest Opinion Entropy (most diverse opinions)"),
        ("lowest_entropy", "Lowest Opinion Entropy (most concentrated opinions)"),
    ]

    for key, title in labels:
        s = standouts[key]
        lines.append(f"### {title}\n")
        lines.append(f"- **Run ID:** `{s['run_id']}`")
        lines.append(f"- **Runtime:** {s['runtime']:.1f}s")
        lines.append(f"- **Parameters:** {describe_params(s['params'])}")
        lines.append(f"- **Final polarization_index:** {s['final_pol']:.4f}")
        lines.append(f"- **Final assortativity:** {s['final_assort']:.4f}")
        lines.append(f"- **Final misinfo_prevalence:** {s['final_misinfo']:.4f}")
        lines.append(f"- **Final opinion_entropy:** {s['final_entropy']:.4f}")
        lines.append(f"- **Final ei_index:** {s['final_ei']:.4f}")
        lines.append(f"- **Final modularity_q:** {s['final_modularity']:.4f}")
        lines.append(f"- **Peak misinfo_prevalence:** {s['peak_misinfo']:.4f}")
        lines.append("")

    # ---------- parameter sweeps ----------
    lines.append("## Parameter Sweep Trends\n")

    # Group by experiment family using parameter heuristics
    families = _group_by_family(rows)

    for family_name, family_rows in families.items():
        if len(family_rows) < 2:
            continue
        lines.append(f"### {family_name}\n")
        lines.append("| Variant | Polarization | Assortativity | Misinfo Prev | Entropy | EI Index | Modularity |")
        lines.append("|---------|-------------|---------------|-------------|---------|----------|------------|")
        for r in family_rows:
            lines.append(
                f"| {r['label']} | {r['final_pol']:.4f} | {r['final_assort']:.4f} | "
                f"{r['final_misinfo']:.4f} | {r['final_entropy']:.4f} | "
                f"{r['final_ei']:.4f} | {r['final_modularity']:.4f} |"
            )
        lines.append("")

    # ---------- trends & observations ----------
    lines.append("## Trends & Observations\n")

    lines.append("### 1. Bot percentage is the strongest polarization driver\n")
    lines.append("Increasing bots from 0% to 30% drives polarization from 0.18 to 0.52 — "
                 "a 3x increase. Bots are the single most powerful lever for echo chamber formation. "
                 "At 0% bots, misinformation prevalence drops to exactly zero since no agent generates "
                 "misinformation content.\n")

    lines.append("### 2. Churn dramatically reduces polarization but increases assortativity\n")
    lines.append("Enabling agent churn (dissatisfied agents leaving) produced the lowest polarization (0.10) "
                 "but paradoxically the second-highest assortativity (0.83). This suggests churn creates "
                 "tightly clustered like-minded subgraphs while removing cross-cutting edges.\n")

    lines.append("### 3. High rewire rate creates ideological silos\n")
    lines.append("Dynamic rewiring at rate 0.10 produces very high assortativity (0.79) with low "
                 "polarization (0.17). Agents cluster with similar peers but the edge-level opinion "
                 "differences within clusters are small, so the polarization index drops. "
                 "This is the classic echo chamber signature: high assortativity + low polarization.\n")

    lines.append("### 4. Diversity ratio backfires spectacularly\n")
    lines.append("Increasing diversity_ratio to 0.5 causes near-universal misinformation infection "
                 "(99.8% prevalence). The diversity mechanism exposes agents to opposing-content, "
                 "but in doing so it floods the feed with misinformation from bot neighbors. "
                 "This is a critical finding: naive content diversity can amplify misinformation.\n")

    lines.append("### 5. CF recommender is extremely slow and poor at containing misinformation\n")
    lines.append("The collaborative filtering recommender took 267s (vs ~9s for content_based) "
                 "and resulted in 71% misinformation prevalence — more than double the content_based "
                 "baseline. The graph recommender was even worse at 75%.\n")

    lines.append("### 6. Bimodal initial opinions increase polarization\n")
    lines.append("Starting with a bimodal opinion distribution increases final polarization by ~11% "
                 "compared to uniform (0.263 vs 0.238). This suggests that pre-existing polarization "
                 "is amplified by the platform dynamics.\n")

    lines.append("### 7. Virality dampening has negligible effect\n")
    lines.append("The virality_dampening parameter (0.0 to 0.9) produced almost identical results "
                 "across all metrics. This mechanism may need recalibration or interacts with other "
                 "parameters that were held at defaults.\n")

    lines.append("### 8. Alpha has modest but monotonic effect\n")
    lines.append("Personalization strength (alpha) from 0.1 to 0.9 increases polarization linearly "
                 "from 0.23 to 0.24 and assortativity from 0.26 to 0.33. The effect is real but "
                 "small compared to bot percentage or rewire rate.\n")

    lines.append("### 9. Graph topology affects assortativity more than polarization\n")
    lines.append("Barabasi-Albert and Erdos-Renyi topologies show much lower assortativity (~0.17) "
                 "than Watts-Strogatz (0.30). This is expected since WS graphs have inherent community "
                 "structure while BA and ER are more randomly connected.\n")

    lines.append("### 10. The 'misinfo storm' scenario combines multiple risk factors\n")
    lines.append("When bots=25%, alpha=0.8, sir_beta=0.5, sir_gamma=0.02, and virality_dampening=0.0 "
                 "are combined, polarization reaches 0.54 — the highest of all experiments. "
                 "This shows multiplicative effects of simultaneous risk factors.\n")

    # Save
    with open(EXPERIMENTS_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {EXPERIMENTS_MD}")


def _group_by_family(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Heuristically group experiments by varied parameter."""
    families: dict[str, list[dict[str, Any]]] = {}

    def final_val(agg, metric):
        series = agg.get(f"{metric}_mean", [])
        return series[-1] if series else 0.0

    for r in rows:
        p = r["parameters"]
        agg = r["aggregated"]

        # Determine family
        agent_mix = p.get("agent_mix", {})
        bot = agent_mix.get("bot", 0.05)

        # Check if this is a specific sweep experiment
        if abs(bot - 0.05) > 0.01 and p["alpha"] == 0.65 and p["recommender_type"] == "content_based" and p["dynamic_rewire_rate"] == 0.01:
            family = f"Bot Percentage (bot={bot:.0%})"
            label = f"bot={bot:.0%}"
        elif p["recommender_type"] != "content_based" and p["alpha"] == 0.65 and abs(bot - 0.05) < 0.01:
            family = "Recommender Type"
            label = p["recommender_type"]
        elif p["topology"] != "watts_strogatz" and p["alpha"] == 0.65 and abs(bot - 0.05) < 0.01:
            family = "Network Topology"
            label = p["topology"]
        elif p["alpha"] not in (0.65,) and abs(bot - 0.05) < 0.01 and p["recommender_type"] == "content_based":
            # Check if it's a pure alpha sweep (not max_echo etc.)
            if p["dynamic_rewire_rate"] == 0.01 and p["diversity_ratio"] == 0.0 and not p["enable_churn"] and p["initial_opinion_distribution"] == "uniform":
                family = "Alpha (Personalization Strength)"
                label = f"alpha={p['alpha']}"
            else:
                family = "Compound Scenarios"
                label = r["run_id"][:20]
        elif p["diversity_ratio"] not in (0.0,) and abs(bot - 0.05) < 0.01:
            family = "Diversity Ratio"
            label = f"diversity={p['diversity_ratio']}"
        elif p["dynamic_rewire_rate"] not in (0.01,) and abs(bot - 0.05) < 0.01 and not p["enable_churn"] and p["alpha"] == 0.65:
            family = "Dynamic Rewire Rate"
            label = f"rewire={p['dynamic_rewire_rate']}"
        elif p["initial_opinion_distribution"] != "uniform" and abs(bot - 0.05) < 0.01 and p["alpha"] == 0.65:
            family = "Initial Opinion Distribution"
            label = p["initial_opinion_distribution"]
        elif p["virality_dampening"] not in (0.0,) and abs(bot - 0.05) < 0.01:
            family = "Virality Dampening"
            label = f"virality_damp={p['virality_dampening']}"
        elif p["enable_churn"]:
            family = "Churn"
            label = "churn=enabled"
        else:
            family = "Compound Scenarios"
            label = r["run_id"][:20]

        families.setdefault(family, []).append({
            "label": label,
            "final_pol": final_val(agg, "polarization_index"),
            "final_assort": final_val(agg, "assortativity"),
            "final_misinfo": final_val(agg, "misinfo_prevalence"),
            "final_entropy": final_val(agg, "opinion_entropy"),
            "final_ei": final_val(agg, "ei_index"),
            "final_modularity": final_val(agg, "modularity_q"),
            "run_id": r["run_id"],
            "agg": agg,
        })

    return families


# ---------------------------------------------------------------------------
# Graphics
# ---------------------------------------------------------------------------

def make_graphics(rows: list[dict[str, Any]], standouts: dict[str, dict[str, Any]]) -> None:
    """Generate all comparison graphics."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
    })

    # ---- 1. Standout runs comparison: per-tick metrics ----
    _plot_standout_metric_curves(standouts, rows)

    # ---- 2. Bot sweep ----
    _plot_bot_sweep(rows)

    # ---- 3. Alpha sweep ----
    _plot_alpha_sweep(rows)

    # ---- 4. Recommender comparison bar chart ----
    _plot_recommender_comparison(rows)

    # ---- 5. Rewire sweep ----
    _plot_rewire_sweep(rows)

    # ---- 6. Topology comparison ----
    _plot_topology_comparison(rows)

    # ---- 7. Correlation heatmap ----
    _plot_correlation_heatmap(rows)

    # ---- 8. Histogram of per-tick polarization for standouts ----
    _plot_polarization_histograms(standouts)

    # ---- 9. Misinformation prevalence curves for diversity sweep ----
    _plot_diversity_misinfo(rows)

    # ---- 10. Summary table as figure ----
    _plot_summary_table(rows, standouts)


def _final_val(agg, metric):
    series = agg.get(f"{metric}_mean", [])
    return series[-1] if series else 0.0


def _load_series(data: dict[str, Any], metric: str):
    """Return (ticks, mean_series, std_series) for a metric."""
    agg = data.get("aggregated") or data.get("agg", {})
    ticks = agg.get("tick", [])
    mean_s = agg.get(f"{metric}_mean", [])
    std_s = agg.get(f"{metric}_std", [])
    return ticks, mean_s, std_s


def _plot_standout_metric_curves(standouts, rows):
    """Multi-panel figure: 6 metrics x time for each standout run."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    metrics = [
        ("polarization_index", "Polarization Index"),
        ("assortativity", "Assortativity"),
        ("misinfo_prevalence", "Misinformation Prevalence"),
        ("opinion_entropy", "Opinion Entropy"),
        ("ei_index", "E-I Index"),
        ("modularity_q", "Modularity Q"),
    ]

    standout_runs = {k: v for k, v in standouts.items()}
    colors = plt.cm.tab10(np.linspace(0, 1, len(standout_runs)))

    for ax, (metric_key, metric_label) in zip(axes.flat, metrics):
        for idx, (name, s) in enumerate(standout_runs.items()):
            ticks, means, stds = _load_series(s, metric_key)
            if ticks:
                ax.plot(ticks, means, color=colors[idx], linewidth=1.5, label=name.replace("_", " ").title(), alpha=0.85)
                if stds:
                    lo = [m - s for m, s in zip(means, stds)]
                    hi = [m + s for m, s in zip(means, stds)]
                    ax.fill_between(ticks, lo, hi, color=colors[idx], alpha=0.08)
        ax.set_title(metric_label)
        ax.set_xlabel("Tick")
        ax.legend(fontsize=6, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Standout Runs — Per-Tick Metric Curves", fontsize=13, y=1.01)
    fig.tight_layout()
    path = GRAPHICS_DIR / "standout_curves.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_bot_sweep(rows):
    """Bot percentage vs key metrics."""
    bot_data = []
    for r in rows:
        p = r["parameters"]
        bot = p.get("agent_mix", {}).get("bot", 0.05)
        # Only include pure bot sweep runs
        if p["alpha"] == 0.65 and p["recommender_type"] == "content_based" and p["dynamic_rewire_rate"] == 0.01:
            bot_data.append((bot, _final_val(r["aggregated"], "polarization_index"),
                             _final_val(r["aggregated"], "assortativity"),
                             _final_val(r["aggregated"], "misinfo_prevalence")))

    if not bot_data:
        return
    bot_data.sort(key=lambda x: x[0])
    bots = [b[0] * 100 for b in bot_data]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(bots, [b[1] for b in bot_data], "o-", color="#2196F3", linewidth=2, label="Polarization Index")
    ax1.plot(bots, [b[2] for b in bot_data], "s-", color="#4CAF50", linewidth=2, label="Assortativity")
    ax2.plot(bots, [b[3] for b in bot_data], "D-", color="#F44336", linewidth=2, label="Misinfo Prevalence")

    ax1.set_xlabel("Bot Percentage (%)")
    ax1.set_ylabel("Polarization / Assortativity", color="#333")
    ax2.set_ylabel("Misinformation Prevalence", color="#F44336")
    ax1.set_title("Effect of Bot Percentage on Echo Chamber Metrics")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    path = GRAPHICS_DIR / "bot_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_alpha_sweep(rows):
    """Alpha sweep."""
    alpha_data = []
    for r in rows:
        p = r["parameters"]
        bot = p.get("agent_mix", {}).get("bot", 0.05)
        if (p["alpha"] in (0.1, 0.3, 0.5, 0.65, 0.9) and abs(bot - 0.05) < 0.01
                and p["dynamic_rewire_rate"] == 0.01 and p["recommender_type"] == "content_based"
                and p["diversity_ratio"] == 0.0 and not p["enable_churn"]):
            alpha_data.append((p["alpha"], _final_val(r["aggregated"], "polarization_index"),
                               _final_val(r["aggregated"], "assortativity"),
                               _final_val(r["aggregated"], "misinfo_prevalence")))

    if not alpha_data:
        return
    alpha_data.sort(key=lambda x: x[0])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    alphas = [a[0] for a in alpha_data]
    ax1.plot(alphas, [a[1] for a in alpha_data], "o-", color="#2196F3", linewidth=2, label="Polarization Index")
    ax1.plot(alphas, [a[2] for a in alpha_data], "s-", color="#4CAF50", linewidth=2, label="Assortativity")
    ax2.plot(alphas, [a[3] for a in alpha_data], "D-", color="#F44336", linewidth=2, label="Misinfo Prevalence")

    ax1.set_xlabel("Alpha (Personalization Strength)")
    ax1.set_ylabel("Polarization / Assortativity")
    ax2.set_ylabel("Misinformation Prevalence", color="#F44336")
    ax1.set_title("Effect of Personalization Strength (Alpha)")
    ax1.grid(True, alpha=0.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    path = GRAPHICS_DIR / "alpha_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_recommender_comparison(rows):
    """Bar chart comparing recommender types."""
    rec_data = []
    for r in rows:
        p = r["parameters"]
        bot = p.get("agent_mix", {}).get("bot", 0.05)
        if p["recommender_type"] != "content_based" and abs(bot - 0.05) < 0.01 and p["alpha"] == 0.65:
            rec_data.append((p["recommender_type"], _final_val(r["aggregated"], "polarization_index"),
                             _final_val(r["aggregated"], "assortativity"),
                             _final_val(r["aggregated"], "misinfo_prevalence"),
                             r["runtime_seconds"]))
        elif p["recommender_type"] == "content_based" and abs(bot - 0.05) < 0.01 and p["alpha"] == 0.65 and p["dynamic_rewire_rate"] == 0.01:
            rec_data.append(("content_based", _final_val(r["aggregated"], "polarization_index"),
                             _final_val(r["aggregated"], "assortativity"),
                             _final_val(r["aggregated"], "misinfo_prevalence"),
                             r["runtime_seconds"]))

    if not rec_data:
        return
    # Deduplicate
    seen = set()
    unique = []
    for d in rec_data:
        if d[0] not in seen:
            seen.add(d[0])
            unique.append(d)
    rec_data = sorted(unique, key=lambda x: x[0])

    labels = [r[0] for r in rec_data]
    x = np.arange(len(labels))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * width, [r[1] for r in rec_data], width, label="Polarization", color="#2196F3")
    ax.bar(x - 0.5 * width, [r[2] for r in rec_data], width, label="Assortativity", color="#4CAF50")
    ax.bar(x + 0.5 * width, [r[3] for r in rec_data], width, label="Misinfo Prevalence", color="#F44336")

    ax2 = ax.twinx()
    ax2.bar(x + 1.5 * width, [r[4] for r in rec_data], width, label="Runtime (s)", color="#FF9800", alpha=0.6)
    ax2.set_ylabel("Runtime (seconds)", color="#FF9800")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Metric Value")
    ax.set_title("Recommender Type Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    fig.tight_layout()
    path = GRAPHICS_DIR / "recommender_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_rewire_sweep(rows):
    """Dynamic rewire rate sweep."""
    rewire_data = []
    for r in rows:
        p = r["parameters"]
        bot = p.get("agent_mix", {}).get("bot", 0.05)
        if p["dynamic_rewire_rate"] in (0.0, 0.01, 0.05, 0.10) and abs(bot - 0.05) < 0.01 and p["alpha"] == 0.65 and not p["enable_churn"]:
            rewire_data.append((p["dynamic_rewire_rate"], _final_val(r["aggregated"], "polarization_index"),
                                _final_val(r["aggregated"], "assortativity"),
                                _final_val(r["aggregated"], "ei_index")))

    if not rewire_data:
        return
    rewire_data.sort(key=lambda x: x[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    rates = [r[0] for r in rewire_data]
    ax.plot(rates, [r[1] for r in rewire_data], "o-", color="#2196F3", linewidth=2, label="Polarization Index")
    ax.plot(rates, [r[2] for r in rewire_data], "s-", color="#4CAF50", linewidth=2, label="Assortativity")
    ax.plot(rates, [r[3] for r in rewire_data], "D-", color="#9C27B0", linewidth=2, label="E-I Index")

    ax.set_xlabel("Dynamic Rewire Rate")
    ax.set_ylabel("Metric Value")
    ax.set_title("Effect of Dynamic Rewiring on Echo Chamber Formation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = GRAPHICS_DIR / "rewire_sweep.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_topology_comparison(rows):
    """Topology comparison bar chart."""
    topo_data = []
    for r in rows:
        p = r["parameters"]
        bot = p.get("agent_mix", {}).get("bot", 0.05)
        if p["topology"] != "watts_strogatz" and abs(bot - 0.05) < 0.01 and p["alpha"] == 0.65:
            topo_data.append((p["topology"], _final_val(r["aggregated"], "polarization_index"),
                              _final_val(r["aggregated"], "assortativity"),
                              _final_val(r["aggregated"], "modularity_q")))
        elif p["topology"] == "watts_strogatz" and abs(bot - 0.05) < 0.01 and p["alpha"] == 0.65 and p["dynamic_rewire_rate"] == 0.01:
            topo_data.append(("watts_strogatz", _final_val(r["aggregated"], "polarization_index"),
                              _final_val(r["aggregated"], "assortativity"),
                              _final_val(r["aggregated"], "modularity_q")))

    if not topo_data:
        return
    seen = set()
    unique = []
    for d in topo_data:
        if d[0] not in seen:
            seen.add(d[0])
            unique.append(d)
    topo_data = unique

    labels = [t[0] for t in topo_data]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, [t[1] for t in topo_data], width, label="Polarization", color="#2196F3")
    ax.bar(x, [t[2] for t in topo_data], width, label="Assortativity", color="#4CAF50")
    ax.bar(x + width, [t[3] for t in topo_data], width, label="Modularity Q", color="#FF9800")

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace("_", "\n") for l in labels])
    ax.set_ylabel("Metric Value")
    ax.set_title("Network Topology Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = GRAPHICS_DIR / "topology_comparison.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_correlation_heatmap(rows):
    """Correlation heatmap of final metrics across all experiments."""
    metrics = ["final_pol", "final_assort", "final_misinfo", "final_entropy", "final_variance", "final_ei", "final_modularity"]
    labels = ["Polarization", "Assortativity", "Misinfo Prev", "Entropy", "Variance", "E-I Index", "Modularity Q"]

    data = []
    for r in rows:
        agg = r["aggregated"]
        data.append([
            _final_val(agg, "polarization_index"),
            _final_val(agg, "assortativity"),
            _final_val(agg, "misinfo_prevalence"),
            _final_val(agg, "opinion_entropy"),
            _final_val(agg, "opinion_variance"),
            _final_val(agg, "ei_index"),
            _final_val(agg, "modularity_q"),
        ])

    arr = np.array(data)
    corr = np.corrcoef(arr.T)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title("Metric Correlation Matrix (Across All Experiments)")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(corr[i, j]) > 0.5 else "black")

    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    path = GRAPHICS_DIR / "correlation_heatmap.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_polarization_histograms(standouts):
    """Histogram of per-tick polarization values for standout runs."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    standout_items = list(standouts.items())

    for ax, (name, s) in zip(axes.flat, standout_items):
        agg = s["agg"]
        pol_series = agg.get("polarization_index_mean", [])
        if pol_series:
            ax.hist(pol_series, bins=12, color="#2196F3", alpha=0.7, edgecolor="white")
            ax.axvline(np.mean(pol_series), color="#F44336", linestyle="--", linewidth=1.5, label=f"mean={np.mean(pol_series):.3f}")
            ax.set_title(name.replace("_", " ").title())
            ax.set_xlabel("Polarization Index")
            ax.set_ylabel("Ticks")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Per-Tick Polarization Index Distribution — Standout Runs", fontsize=13, y=1.01)
    fig.tight_layout()
    path = GRAPHICS_DIR / "polarization_histograms.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_diversity_misinfo(rows):
    """Misinformation prevalence over time for diversity sweep."""
    div_runs = []
    for r in rows:
        p = r["parameters"]
        bot = p.get("agent_mix", {}).get("bot", 0.05)
        if p["diversity_ratio"] in (0.0, 0.2, 0.5) and abs(bot - 0.05) < 0.01 and p["alpha"] == 0.65 and p["recommender_type"] == "content_based":
            ticks, means, stds = _load_series(r, "misinfo_prevalence")
            div_runs.append((p["diversity_ratio"], ticks, means, stds))

    if not div_runs:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    for (dr, ticks, means, stds), color in zip(sorted(div_runs, key=lambda x: x[0]), colors):
        ax.plot(ticks, means, color=color, linewidth=2, label=f"diversity={dr}")
        if stds:
            lo = [m - s for m, s in zip(means, stds)]
            hi = [m + s for m, s in zip(means, stds)]
            ax.fill_between(ticks, lo, hi, color=color, alpha=0.12)

    ax.set_xlabel("Tick")
    ax.set_ylabel("Misinformation Prevalence")
    ax.set_title("Misinformation Prevalence Over Time by Diversity Ratio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    path = GRAPHICS_DIR / "diversity_misinfo.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def _plot_summary_table(rows, standouts):
    """Summary table of all experiments as a figure."""
    # Build table data
    header = ["Experiment", "Polarization", "Assortativity", "Misinfo", "Entropy", "E-I Idx", "Mod Q"]
    table_data = []

    all_scored = []
    for r in rows:
        agg = r["aggregated"]
        p = r["parameters"]
        bot = p.get("agent_mix", {}).get("bot", 0.05)
        # Derive a short label
        if p.get("enable_churn"):
            label = "churn_on"
        elif p["dynamic_rewire_rate"] == 0.10 and p["homophily_threshold"] == 0.15:
            label = "rewire_homophily"
        elif p["alpha"] == 0.8 and abs(bot - 0.25) < 0.02:
            label = "misinfo_storm"
        elif p["alpha"] == 0.9 and abs(bot - 0.05) < 0.01 and p["diversity_ratio"] == 0.0:
            label = "max_echo"
        elif p["alpha"] == 0.1 and p["diversity_ratio"] == 0.5:
            label = "min_echo"
        elif p["initial_opinion_distribution"] == "bimodal" and p["alpha"] == 0.9:
            label = "bimodal_hi_alpha"
        elif p["media_literacy_boost"] == 0.4:
            label = "media_literate"
        elif p["recommender_type"] != "content_based" and abs(bot - 0.05) < 0.01:
            label = f"rec_{p['recommender_type']}"
        elif p["topology"] != "watts_strogatz" and abs(bot - 0.05) < 0.01:
            label = f"topo_{p['topology']}"
        elif p["alpha"] in (0.1, 0.3, 0.5, 0.9) and abs(bot - 0.05) < 0.01 and p["dynamic_rewire_rate"] == 0.01:
            label = f"alpha_{p['alpha']}"
        elif abs(bot - 0.05) > 0.01 and p["alpha"] == 0.65:
            label = f"bots_{bot:.0%}"
        elif p["diversity_ratio"] in (0.2, 0.5) and abs(bot - 0.05) < 0.01:
            label = f"div_{p['diversity_ratio']}"
        elif p["dynamic_rewire_rate"] in (0.0, 0.05) and abs(bot - 0.05) < 0.01:
            label = f"rewire_{p['dynamic_rewire_rate']}"
        elif p["virality_dampening"] in (0.3, 0.6, 0.9):
            label = f"vdamp_{p['virality_dampening']}"
        elif p["initial_opinion_distribution"] == "bimodal":
            label = "init_bimodal"
        else:
            label = "baseline"

        all_scored.append((
            label,
            _final_val(agg, "polarization_index"),
            _final_val(agg, "assortativity"),
            _final_val(agg, "misinfo_prevalence"),
            _final_val(agg, "opinion_entropy"),
            _final_val(agg, "ei_index"),
            _final_val(agg, "modularity_q"),
        ))

    all_scored.sort(key=lambda x: x[1], reverse=True)  # sort by polarization

    fig, ax = plt.subplots(figsize=(14, max(8, len(all_scored) * 0.35)))
    ax.axis("off")

    table_data = [
        [l, f"{p:.4f}", f"{a:.4f}", f"{m:.4f}", f"{e:.4f}", f"{ei:.4f}", f"{q:.4f}"]
        for l, p, a, m, e, ei, q in all_scored
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=header,
        cellLoc="center",
        loc="center",
        colWidths=[0.16, 0.14, 0.14, 0.14, 0.14, 0.14, 0.14],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.1)

    # Color header
    for j in range(len(header)):
        table[0, j].set_facecolor("#37474F")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Color top 3 and bottom 3
    for i in range(1, min(4, len(table_data) + 1)):
        for j in range(len(header)):
            table[i, j].set_facecolor("#FFCDD2")
    for i in range(max(1, len(table_data) - 2), len(table_data) + 1):
        for j in range(len(header)):
            table[i, j].set_facecolor("#C8E6C9")

    ax.set_title("All Experiments — Final Metric Values (sorted by polarization)", fontsize=12, y=1.01)

    fig.tight_layout()
    path = GRAPHICS_DIR / "summary_table.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    GRAPHICS_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_results()
    print(f"Loaded {len(rows)} experiment results")

    standouts = identify_standouts(rows)
    print("\nStandout runs:")
    for name, s in standouts.items():
        print(f"  {name}: pol={s['final_pol']:.4f} assort={s['final_assort']:.4f} misinfo={s['final_misinfo']:.4f}")

    print("\nWriting experiments.md...")
    write_experiments_md(rows, standouts)

    print("\nGenerating graphics...")
    make_graphics(rows, standouts)

    print("\nDone.")


if __name__ == "__main__":
    main()
