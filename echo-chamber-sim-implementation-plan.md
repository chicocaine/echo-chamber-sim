# Echo Chamber Simulation — LLM Implementation Plan

> **How to use this document**
> This is a phase-by-phase implementation guide for an agent-based echo chamber simulation.
> Each phase is self-contained. An LLM coder can be given any single phase section and execute it
> without needing the full conversation history. Phases must be completed in order — each builds
> on the previous. Read the CONTEXT and CONSTRAINTS sections of each phase before writing any code.
>
> **Canonical reference:** `echo-chamber-sim-agent-reference.md` is the source of truth for all
> agent mechanics, parameters, equations, and defaults. When in doubt, that document wins.

---

## Project Overview

**What this is:** A social media echo chamber simulator using agent-based modeling (ABM).
Agents hold opinions, consume content from a recommender algorithm, update their opinions,
and form (or break) social connections over time. The goal is to observe echo chamber formation
and test interventions that reduce polarization and misinformation spread.

**Stack:**
- Backend: Python 3.11+, NumPy, NetworkX, SciPy, FastAPI, Uvicorn, Pydantic, joblib, orjson, line_profiler
- Experimentation: Jupyter notebooks
- Frontend: React 18 + TypeScript + Vite, Recharts, react-force-graph-2d
- Transport: REST (MVP), WebSocket (Phase 7)

**Project structure:**
```
echo-chamber-sim/
├── backend/
│   ├── sim/
│   │   ├── __init__.py
│   │   ├── agent.py
│   │   ├── content.py
│   │   ├── network.py
│   │   ├── recommender.py
│   │   ├── simulation.py
│   │   └── metrics.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── schemas.py
│   └── requirements.txt
├── notebooks/
│   ├── 01_agent_init.ipynb
│   ├── 02_network_viz.ipynb
│   ├── 03_opinion_dynamics.ipynb
│   ├── 04_recommender.ipynb
│   └── 05_metrics.ipynb
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── NetworkGraph.tsx
│   │   │   ├── MetricsPanel.tsx
│   │   │   ├── ControlPanel.tsx
│   │   │   └── OpinionHistogram.tsx
│   │   ├── hooks/
│   │   │   └── useSimulation.ts
│   │   └── lib/
│   │       ├── api.ts
│   │       └── types.ts
│   ├── package.json
│   └── vite.config.ts
└── README.md
```

**Global invariants — never violate these:**
- Opinions are always floats in `[-1.0, 1.0]`. Clamp after every update.
- All probability values are in `[0.0, 1.0]`. Assert this at init.
- Agent `id` values are unique integers assigned at creation, never reused.
- Inactive agents (`is_active = False`) are skipped in every tick loop step.
- All parameter defaults come from the agent reference document. Do not invent defaults.

---

## MVP — Two-Week Baseline

### Goal
A working simulation with two agent types, two opinion update rules, a simplified content-based
recommender, SIR misinformation tracking, five metrics, a FastAPI backend, and a React dashboard
that runs a simulation and displays results.

### What is explicitly excluded from MVP
- HK, Contrarian, Influencer, Passive agent types
- Emotion system (arousal, sharing probability formula)
- Collaborative filtering or graph-based recommenders
- Dynamic network rewiring or churn
- Bot detection module
- Monte Carlo replication (added in Phase 0 immediately after)
- WebSocket streaming (added in Phase 7)

---

### MVP Step 1 — Environment setup

**Commands to run (WSL2, plain venv):**
```bash
mkdir echo-chamber-sim && cd echo-chamber-sim
python3 -m venv .venv
source .venv/bin/activate
pip install numpy networkx matplotlib scipy jupyter fastapi uvicorn pydantic joblib orjson line_profiler
pip freeze > backend/requirements.txt

npm create vite@latest frontend -- --template react-ts
cd frontend
npm install recharts react-force-graph-2d
npm install -D @types/react @types/react-dom
```

Create all `__init__.py` files. The `backend/` directory must be importable as a package.
Notebooks import from `backend/sim/` — never the reverse.

---

### MVP Step 1b — Performance baseline and profiling setup

**This step establishes the profiling harness used throughout all phases.**
Do not skip it. You cannot optimize what you haven't measured.

**Why this matters:** The two dominant performance bottlenecks in an ABM of this type are:
1. Feed generation — O(N × C) where C = candidate pool size. At N=1000 and C=500 this is 500,000 score computations per tick, done in a Python loop by default.
2. Opinion update collection — O(N × avg_degree) neighbor lookups per tick.

Both are fixable with vectorization and parallelization, but only after profiling confirms they are actually the bottleneck in your specific run.

**Install the profiling tool:**
```bash
pip install line_profiler  # already in requirements if Step 1 was followed
```

**Add the `@profile` decorator pattern to `simulation.py`:**
```python
# At the top of simulation.py, add this guard so @profile works both
# with and without the line_profiler tool active:
try:
    profile  # defined by line_profiler when active
except NameError:
    def profile(func):  # no-op decorator when not profiling
        return func
```

Apply `@profile` to `run_simulation()` and `recommender.generate_feed()` before any
optimization work. Run a baseline profile before touching any code:
```bash
kernprof -l -v backend/sim/simulation.py
```

Record the baseline timings in a comment block at the top of `simulation.py`:
```python
# PERF BASELINE (recorded before optimization):
# run_simulation N=1000 T=100: Xs total
# generate_feed: X% of total time
# opinion_update: X% of total time
# Record actuals here after first profile run.
```

This comment block must be updated after each optimization phase.

---

### MVP Step 2 — `backend/sim/agent.py`

**What to build:** `Agent` dataclass with all state variables. Two active agent types via
a `compute_update()` method that subclasses override. Zealot and Bot as parameter configs,
not separate classes.

**Agent types in MVP:**
- `StubbornAgent` (Friedkin–Johnsen): `x_i(t+1) = g_i * x_i(0) + (1 - g_i) * weighted_avg(neighbors)`
- `FlexibleAgent` (DeGroot): `x_i(t+1) = sum(a_ij * x_j(t))` where `a_ij` are normalized weights
- Zealot: `StubbornAgent` with `stubbornness = 1.0`, opinion fixed at ±1.0
- Bot: `StubbornAgent` with `stubbornness = 1.0`, `activity_rate ~ N(0.9, 0.05)`, `opinion` fixed extreme,
  and `misinfo_rate: float = 1.0` (m5: fraction of posts that are misinformation, from ref doc Part 1.1).
  During content generation, use `random() < agent.misinfo_rate` to decide if a bot's content is
  misinformation. Expose `misinfo_rate` as a config parameter to enable partial-misinfo bot experiments.

**State variables every agent carries (all types):**
```python
id: int
agent_type: str          # "stubborn" | "flexible" | "zealot" | "bot"
opinion: float           # [-1, 1], current
initial_opinion: float   # [-1, 1], frozen at t=0
stubbornness: float      # g_i in [0, 1]; 0 = DeGroot, 1 = Zealot
susceptibility: float    # [0, 1]
trust: float             # [0, 1]
expertise: float         # [0, 1]
activity_rate: float     # [0, 1]
emotional_arousal: float # [0, 1], always 0.0 at init in MVP
media_literacy: float    # [0, 1]
confidence_bound: float  # [0, 1], unused in MVP (placeholder for HK)
contrarian_prob: float   # [0, 1], unused in MVP (placeholder for Contrarian)
suspicion_score: float   # [0, 1], unused in MVP (placeholder for bot detection)
is_active: bool          # False = churned, skip in all loops
sir_state: str           # "S" | "I" | "R" for misinformation tracking
opinion_history: list[float]  # M3: populated each tick for bot detection in Phase 5 (Step 5.3).
                               # Must start from tick 0 — do not add this field later.
                               # Append agent.opinion at the end of every tick in the main loop.
```

**Initialization distributions (from reference doc Part 1.3):**
```
opinion:        U(-1, 1) for neutral; bimodal N(±0.7, 0.15) for pre-polarized
                # C4: Bimodal peaks at ±0.7 from ref doc Part 1.3.
                # σ=0.15 is a design choice — not specified in ref doc.
stubbornness:   U(0.1, 0.3) for stubborn/flexible; 1.0 for zealot/bot
susceptibility: N(0.5, 0.1) clipped to [0, 1]
trust:          N(0.5, 0.2) clipped to [0, 1]
expertise:      N(0.5, 0.2) clipped to [0, 1]
activity_rate:  N(0.5, 0.15) standard; N(0.9, 0.05) bots; N(0.07, 0.03) passive
                # M2: Passive agent activity: REF gives range 0.05–0.1 (Part 1.1).
                # N(0.07, 0.03) clipped to [0.01, 0.2] is a design choice.
emotional_arousal: 0.0
media_literacy: U(0.2, 0.8)
```

**`compute_update()` contract:**
- Takes `neighbors: list[Agent]`, `influence_weights: dict[int, float]`
- Returns a float: the new opinion value (not a delta)
- Must clamp return value to `[-1.0, 1.0]`
- Zealots return `self.opinion` unchanged always

**Future expansion note:** When adding HKAgent, ContrarianAgent, InfluencerAgent in Phase 1,
each will be a new subclass overriding `compute_update()`. Do not add branching `if/elif` logic
to this base method — keep it clean for subclassing.

---

### MVP Step 3 — `backend/sim/content.py`

**What to build:** `Content` dataclass representing a single piece of content in the simulation.

**Fields:**
```python
id: int
creator_id: int
timestamp: int                # tick at which content was created
ideological_score: float      # [-1, 1]
emotional_valence: float      # [0, 1] — placeholder in MVP, used fully in Phase 2
misinfo_score: float          # [0, 1]: 0=credible, 1=false
virality: float               # [0, 1]
source_credibility: float     # [0, 1]
is_misinformation: bool       # ground truth label
# Placeholders — not used in MVP, do not remove:
topic_vector: list[float]     # will be 128-dim in Phase 5; empty list [] in MVP
coordinated_campaign_id: int | None  # Phase 5
is_satire: bool               # Phase 5
# M8: valid_until is NOT a content field. Campaign expiry tracked via campaign_expiry
#     dict in simulation.py (Phase 5). Do not add valid_until to content schema.
```

**Content generation at each tick:**
- Agent generates content with probability = `agent.activity_rate`
- `ideological_score` sampled near agent's current `opinion`: `N(opinion, 0.1)` clipped to `[-1, 1]`
- `emotional_valence`: `U(0.2, 0.6)` for normal content in MVP
- `misinfo_score`: 0.05 for normal agents; `U(0.7, 1.0)` for bots
- `is_misinformation`: True if `misinfo_score > 0.5`
- `virality`: `U(0.1, 0.9)`
- `source_credibility`: agent's `trust` value

**`belief_update_weight` formula (from reference doc Part 3):**
```python
belief_update_weight = (1 - misinfo_score) * source_credibility \
                       * (1 - (1 - media_literacy_i) * misinfo_score)
```
Compute this value and store it as a field on the `Content` object at generation time.

> **M1 (MVP decision):** `belief_update_weight` is **content metadata only** in the MVP.
> It is NOT passed into `compute_update()` and does NOT influence the FJ/DeGroot opinion
> update in this phase. Do not wire it into the opinion update loop.
> # TODO (Phase 2): wire belief_update_weight into compute_update() as a per-content
> # modulator on the influence weight a_ij. A credible source with misleading content
> # can still shift a low-media-literacy agent — but this requires Phase 2 emotion
> # and media_literacy integration before it becomes meaningful.

---

### MVP Step 4 — `backend/sim/network.py`

**What to build:** Graph initialization and helper functions. No dynamic rewiring in MVP.

**Default topology:** Watts–Strogatz small-world graph, converted to directed.
```python
import networkx as nx
G = nx.watts_strogatz_graph(N, k=avg_degree, p=rewire_prob)
G = nx.DiGraph(G)   # m6: convert to directed. Each undirected edge becomes two directed edges.
```
Default params: `N=200` for demo runs, `N=1000` for experiment runs, `avg_degree=15`, `rewire_prob=0.1`

> **m6 — Directed graph: performance and resource implications.**
>
> *Memory:* `nx.DiGraph` stores both directions per edge. At N=1000, avg_degree=15 this is
> ~30k directed edges vs ~15k undirected — roughly 2× edge memory. Negligible at this scale.
>
> *Speed:* All neighbor iteration now uses `G.predecessors(i)` (who do I listen to?) vs
> `G.successors(i)` (who do I broadcast to?). Make this convention explicit everywhere in
> `network.py` and the tick loop — confusing the two causes silent correctness bugs.
> For opinion updates, use **`G.predecessors(agent_id)`** (incoming edges = influence sources).
>
> *Rewiring:* Dynamic rewiring in Phase 4 is now asymmetric — an agent can unfollow without
> being unfollowed back. This is more realistic but doubles the edge-set cases to handle.
>
> *Bot detection Signal 4 (reciprocity ratio):* Now **meaningful and valid**.
> Bots follow many agents but few follow back → low `in_degree / out_degree` ratio.
> Signal 4 implementation in Phase 5 Step 5.3 can proceed as designed.
>
> *Vectorized ops (Phase Opt):* NumPy adjacency matrix must be built as a directed matrix
> (not symmetric). Use `nx.to_numpy_array(G)` which handles DiGraph correctly.
> Ensure `generate_feed_vectorized` uses the predecessor submatrix, not the full adjacency.

**After graph creation:**
- Assign one agent to each node: `G.nodes[node_id]['agent'] = agents[node_id]`
- Compute normalized influence weights `a_ij` for each edge: uniform `1/in_degree(i)` in MVP
- Store weights as edge attributes: `G[i][j]['weight'] = a_ij`

**Helper functions to expose:**
```python
# =============================================================================
# DIRECTED GRAPH CONVENTION — READ THIS BEFORE TOUCHING ANY GRAPH TRAVERSAL
# =============================================================================
# The graph is a nx.DiGraph. An edge (j → i) means "i follows j", i.e. j is
# an influence source for i. The direction encodes information flow, NOT who
# initiated the relationship.
#
#   G.predecessors(i)  → agents that i LISTENS TO   (influence sources for i)
#   G.successors(i)    → agents that i BROADCASTS TO (i is an influence source for them)
#   G.neighbors(i)     → DO NOT USE. On DiGraph this returns successors only,
#                        which is the WRONG direction for opinion updates. It is
#                        banned in this codebase to prevent silent correctness bugs.
#
# Rule of thumb:
#   Opinion update / feed generation  →  G.predecessors(i)   ✓
#   Broadcast / virality propagation  →  G.successors(i)     ✓
#   G.neighbors(i)                    →  NEVER               ✗
#
# All public helpers below enforce this. Never call the NetworkX API directly
# outside of network.py.
# =============================================================================

def get_predecessors(G: nx.DiGraph, agent_id: int) -> list[int]:
    """Agents that agent_id listens to. Use for opinion updates and feed generation."""
    return list(G.predecessors(agent_id))

def get_successors(G: nx.DiGraph, agent_id: int) -> list[int]:
    """Agents that agent_id broadcasts to. Use for virality and content propagation."""
    return list(G.successors(agent_id))

def get_influence_weights(G: nx.DiGraph, agent_id: int) -> dict[int, float]:
    """Normalized influence weights from predecessors. Rows sum to 1.0 (DeGroot constraint)."""
    preds = list(G.predecessors(agent_id))
    if not preds:
        return {}
    raw = {j: G[j][agent_id].get('weight', 1.0) for j in preds}
    total = sum(raw.values())
    return {j: w / total for j, w in raw.items()}

def get_graph_snapshot(G: nx.DiGraph) -> dict:
    """Returns node opinions and directed edge list for frontend rendering."""
    return {
        "nodes": [
            {"id": n, "opinion": G.nodes[n]['agent'].opinion}
            for n in G.nodes
        ],
        "edges": [
            {"source": u, "target": v}   # u → v: v listens to u
            for u, v in G.edges
        ],
    }
```

**Future expansion note:** `network.py` will gain a `rewire_step(G, agents, config)` function
in Phase 4. A `topology_factory(config)` function dispatching to BA or SBM graphs also goes here.
Design `get_graph_snapshot()` to include edge list from the start — rewiring changes edges and
the frontend needs to know.

---

### MVP Step 5 — `backend/sim/recommender.py`

**What to build:** Content-based recommender only. Scores candidate content for each agent
and returns the top `k_exp` items.

**Similarity score formula (simplified for MVP):**
```python
import math
sim(content, agent) = alpha * (1 - abs(content.ideological_score - agent.opinion)) \
                    + (1 - alpha) * beta_pop * math.log1p(content.virality * 9)
```
where `alpha` is personalization strength (default `0.65`), `beta_pop` is virality boost (default `0.2`).

> **M4 (Option B):** `beta_pop` is a distinct parameter from `alpha` (ref doc Part 4: β_pop).
> `log1p(v * 9)` compresses virality from [0,1] → [0, ~2.30], preventing winner-takes-all
> viral dominance. Assert `beta_pop <= 1.0` in config validation. Re-normalize scores to
> [0,1] before ranking (divide by max score in candidate pool).
>
> **Risk:** If β_pop is raised above ~0.5 in experiments, expect runaway viral cascades.
> Log this as a known model risk and flag it in experiment result outputs.

> **MVP Simplification Note (C2):** This formula collapses the REF's three-component
> similarity (ref doc Part 4: topic + ideology + sentiment, weighted by `α_t`, `α_id`, `α_s`)
> into a two-term approximation using ideology and log-compressed virality.
> The full formula with `α_t`, `α_id`, `α_s` blend weights and `pref_vector` (accumulated
> topic preference via exponential moving average) is implemented in Phase 3 when
> `topic_vector` becomes available. Do NOT use the three-weight formula here.

**Feed generation:**
```python
def generate_feed(agent, candidate_pool, k_exp, alpha, beta_pop=0.2) -> list[Content]:
    scored = [(sim(c, agent, alpha, beta_pop), c) for c in candidate_pool]
    scored.sort(reverse=True)
    return [c for _, c in scored[:k_exp]]
```

**Candidate pool:** All content generated in the current tick by all agents,
plus content shared by the agent's neighbors in the previous tick.

**Future expansion note:** In Phase 3, `recommender.py` is refactored into a base class
`BaseRecommender` with a `score(content, agent) -> float` method. The current function
becomes `ContentBasedRecommender`. Plan your function signatures to make this extraction
clean — avoid global state.

---

### MVP Step 6 — `backend/sim/metrics.py`

**What to build:** Five metric functions. Each takes the current graph `G` and agent list,
returns a single float. Called every snapshot interval.

**Metrics to implement:**

1. **Opinion variance**
   ```
   (1/N) * sum((opinion_i - mean_opinion)^2)
   ```

2. **Polarization index**
   ```
   mean(|opinion_i - opinion_j|) across all edges (i,j) in G
   ```

3. **Opinion assortativity**
   Pearson correlation of opinion values across edges.
   Do NOT use `nx.degree_assortativity_coefficient` — that's for degree.
   Compute manually: for each edge (i,j), collect pairs `(opinion_i, opinion_j)`,
   then compute Pearson r.

4. **Opinion entropy**
   Bin opinions into 20 buckets over `[-1, 1]`. Compute Shannon entropy:
   ```
   -sum(p(bin) * log(p(bin) + 1e-10))
   ```

5. **Misinformation prevalence**
   ```
   count(agents where sir_state == "I") / N
   ```

**Output format per snapshot:**
```python
{
    "tick": int,
    "opinion_variance": float,
    "polarization_index": float,
    "assortativity": float,
    "opinion_entropy": float,
    "misinfo_prevalence": float
}
```

**Future expansion note:** E-I index, Modularity Q, Cascade Size, Exposure Disparity,
and IES are added in Phase 6. Each is a new function in this file — do not restructure,
just append.

---

### MVP Step 7 — `backend/sim/simulation.py`

**What to build:** The main tick loop. Takes a config dict, runs T ticks, returns
a list of metric snapshots and the final agent state.

**Config dict shape:**
```python
{
    "N": 200,
    "avg_degree": 15,
    "rewire_prob": 0.1,
    "T": 200,               # m3: reduced for MVP demos; full default is 720 (Appendix B / ref doc Part 8)
    "snapshot_interval": 6,   # m2: Appendix B default is authoritative (6). REF Part 8 range: 1–6.
                               # (was 5 in earlier draft — corrected to match Appendix B)
    "alpha": 0.65,
    "k_exp": 20,
    "agent_mix": {
        "stubborn": 0.60,
        "flexible": 0.20,
        "passive": 0.10,
        "zealot": 0.05,
        "bot": 0.05
    },
    "sir_beta": 0.3,
    "sir_gamma": 0.05,
    "initial_opinion_distribution": "uniform",  # or "bimodal"
    "seed": 42
}
```

**Tick loop — execute steps in this exact order:**

```
Step 1: CONTENT GENERATION
  For each active agent:
    if random() < agent.activity_rate:
      generate Content item, add to current_tick_pool

Step 2: FEED GENERATION
  For each active agent:
    candidate_pool = current_tick_pool + agent's neighbors' shared content from t-1
    agent.feed = recommender.generate_feed(agent, candidate_pool, k_exp, alpha)

Step 3: CONTENT CONSUMPTION
  For each active agent:
    For each content item in agent.feed:
      update agent.emotional_arousal  ← placeholder in MVP, emotion system in Phase 2

Step 4: SHARING DECISION
  For each active agent:
    For each content item in agent.feed:
      if random() < base_share_prob (default 0.18):
        add to agent's shared_content (available to neighbors next tick)

Step 5: OPINION UPDATE
  new_opinions = {}
  For each active agent:
    # DIRECTED GRAPH CONVENTION: use get_predecessors() — agents this agent listens to.
    # Do NOT use get_successors() or G.neighbors() here — both give the wrong direction.
    # get_influence_weights() also uses predecessors internally; do not reimplement inline.
    neighbors = get_predecessors(G, agent.id)
    weights = get_influence_weights(G, agent.id)
    new_opinions[agent.id] = agent.compute_update(neighbors, weights)
  Apply all new_opinions simultaneously (not sequentially — prevents order bias)

  SIR transitions:
    # C3 fix: transmission is triggered by misinformation content in the agent's feed,
    # NOT by whether a neighbor happens to be in state I. The creator's SIR state is
    # irrelevant — an agent becomes infected when they SEE misinfo content (is_misinformation=True),
    # regardless of who created it. Ref doc Part 5: "S→I when a Susceptible agent sees
    # an Infected neighbor's content" means content exposure, not neighbor SIR state check.
    for agent in active_agents:
        if agent.sir_state == "S":
            misinfo_in_feed = [c for c in agent.feed if c.is_misinformation]
            if misinfo_in_feed:
                if random() < sir_beta:
                    agent.sir_state = "I"
        elif agent.sir_state == "I":
            if random() < sir_gamma:
                agent.sir_state = "R"

Step 6: NETWORK REWIRING — SKIP in MVP

Step 7: CHURN CHECK — SKIP in MVP

Step 8: BOT DETECTION — SKIP in MVP

Step 9: METRIC LOGGING
  if tick % snapshot_interval == 0:
    snapshot = compute_all_metrics(G, agents)
    snapshot["tick"] = tick
    snapshots.append(snapshot)
  # M3: Append current opinion to history EVERY tick (not just snapshot ticks).
  # Required for bot detection Signal 2 in Phase 5. Must start at tick 0.
  for agent in active_agents:
    agent.opinion_history.append(agent.opinion)
```

**Return value:**
```python
{
    "config": config,
    "snapshots": list[dict],       # one per snapshot interval
    "final_agents": list[dict],    # serialized agent states at tick T
    "final_graph": dict            # node positions + edge list for frontend
}
```

**Important:** Apply opinion updates simultaneously. Collect all new opinions first,
then apply in a second pass. Sequential updates introduce order bias.

**Future expansion note:** Steps 6, 7, 8 are stubbed with `pass` and a `# TODO: PhaseN`
comment. The emotion update in Step 3 is stubbed as `agent.emotional_arousal = 0.0`.
The sharing probability in Step 4 is a fixed float — Phase 2 replaces this with the
sigmoid formula. Keep these as clearly labeled stubs so Phase implementations are obvious.

---

### MVP Step 8 — `backend/api/schemas.py` and `backend/api/main.py`

**`schemas.py`** — Pydantic models:
```python
class SimConfig(BaseModel):
    N: int = 200             # m7: demo default; use N=1000 for experiment runs (ref doc Part 11 baseline)
    avg_degree: int = 15
    rewire_prob: float = 0.1
    T: int = 200             # m3: reduced for MVP demos; full default is 720 (Appendix B / ref doc Part 8)
    snapshot_interval: int = 6   # m2: Appendix B default; REF Part 8 range 1–6
    alpha: float = 0.65
    k_exp: int = 20
    agent_mix: dict[str, float] = {...}
    sir_beta: float = 0.3
    sir_gamma: float = 0.05
    initial_opinion_distribution: str = "uniform"
    seed: int = 42

class MetricSnapshot(BaseModel):
    tick: int
    opinion_variance: float
    polarization_index: float
    assortativity: float
    opinion_entropy: float
    misinfo_prevalence: float

class SimResult(BaseModel):
    config: SimConfig
    snapshots: list[MetricSnapshot]
    final_agents: list[dict]
    final_graph: dict
```

**`main.py`** — Two endpoints only in MVP:
```python
POST /run        # accepts SimConfig, returns SimResult
GET  /defaults   # returns default SimConfig as JSON
```

Add CORS middleware for `localhost:5173` (Vite default port).
Run sim synchronously in MVP — no background tasks yet.

---

### MVP Step 9 — Frontend

**`lib/types.ts`** — Mirror `schemas.py` exactly as TypeScript interfaces.
Write these by hand from the Pydantic models — do not auto-generate.

**`lib/api.ts`** — Two functions:
```typescript
runSimulation(config: SimConfig): Promise<SimResult>
getDefaults(): Promise<SimConfig>
```

**`hooks/useSimulation.ts`** — State machine with four states: `idle | running | done | error`.
Calls `runSimulation`, stores `SimResult`, exposes to components.

**Components — build in this order:**

1. `ControlPanel.tsx`
   - Sliders for: `alpha` (0–1), `N` (100–1000, step 100), `sir_beta` (0–1),
     `bot fraction` (0–0.2), `zealot fraction` (0–0.2)
   - Run button, disabled while `status === 'running'`
   - Shows current config values

2. `MetricsPanel.tsx`
   - Recharts `LineChart` with 5 lines, one per metric
   - X-axis: tick number. Y-axis: metric value.
   - Data source: `result.snapshots`

3. `OpinionHistogram.tsx`
   - Recharts `BarChart`, 20 bins over `[-1, 1]`
   - Tick scrubber: slider to select which snapshot to display
   - Reconstructs opinion distribution from `final_agents` at selected tick
     (Note: to show intermediate ticks accurately, `final_agents` needs per-tick
     opinion history — either store it in snapshots or only show tick T in MVP)

4. `NetworkGraph.tsx`
   - `react-force-graph-2d`, Canvas-based
   - Node color: opinion mapped to red (−1) → white (0) → blue (+1)
   - Node size: `activity_rate * 5 + 3`
   - Edge opacity: 0.2 (static in MVP)
   - Freeze layout after first render — only update node colors, not positions
   - Data source: `result.final_graph`
   - Cap render at N=300. Show warning if N>300.

**Layout:** Two-column. Left: `ControlPanel` + `NetworkGraph`. Right: `MetricsPanel` + `OpinionHistogram`.

---

### MVP Validation Checklist

Before declaring MVP done, verify all of these:

- [ ] Pure DeGroot population (all flexible) converges to a single opinion value
- [ ] Pure FJ population with high stubbornness does NOT fully converge
- [ ] Zealot population pulls neighbors toward their opinion over 200 ticks
- [ ] Bot agents post more frequently than standard agents (check activity in logs)
- [ ] Misinformation prevalence rises then falls (SIR dynamics visible in chart)
- [ ] Higher `alpha` produces higher assortativity (echo chamber effect visible)
- [ ] Lower `alpha` (≈0) produces lower assortativity
- [ ] Network graph node colors shift visibly across a 200-tick run
- [ ] API returns a result within 10 seconds for N=200, T=200

---

## MVP Phase Opt — Performance Optimization

**Insert after MVP validation checklist passes. Before Phase 0.**

### Goal
Make the simulation fast enough to be useful. Target: N=1000, T=720 completes in under
60 seconds on a modern CPU. Without optimization, a naive Python implementation at this
scale takes 10–30 minutes. This phase makes Monte Carlo replication (Phase 0) practical.

### Context
- Prerequisites: Full MVP tick loop working and passing the validation checklist.
- Profile before optimizing. Do not guess. The bottleneck is almost always feed generation,
  but confirm it in your specific implementation before rewriting anything.
- Optimizations are applied in three sequential passes: vectorization first, then
  parallelization, then serialization. Do not apply parallelization before vectorization —
  parallelizing a slow inner loop is less effective than vectorizing it first.
- The simultaneous opinion update rule (all updates collected before any are applied)
  is compatible with parallelization — see Step Opt.3 for the correct pattern.

---

### Step Opt.1 — Profile and identify bottlenecks

Run the profiler on a N=1000, T=50 run and record actual timings:

```bash
# From backend/ directory with venv active:
kernprof -l -v -o profile.lprof sim/simulation.py
python -m line_profiler profile.lprof
```

Expected findings (confirm these match your output before proceeding):
- `generate_feed()` should account for 60–90% of total tick time
- `compute_update()` (opinion updates) should account for 5–20%
- Metric computation should account for <5%

If the distribution is very different from this, investigate before optimizing.
The optimization strategy below is tuned for the expected profile.

Update the `# PERF BASELINE` comment block in `simulation.py` with actual numbers.

---

### Step Opt.2 — Vectorize feed generation in `recommender.py`

This is the highest-impact change. Replace the Python loop over content items with
NumPy matrix operations.

**Before (naive — O(N×C) Python loop):**
```python
def generate_feed(self, agent, candidate_pool, k_exp):
    scored = [(self.score(c, agent), c) for c in candidate_pool]
    scored.sort(reverse=True)
    return [c for _, c in scored[:k_exp]]
```

**After (vectorized — single NumPy operation):**
```python
def generate_feed_vectorized(
    self,
    agent_opinion: float,
    content_ideological_scores: np.ndarray,  # shape (C,)
    content_virality: np.ndarray,            # shape (C,)
    candidate_pool: list[Content],
    k_exp: int,
    alpha: float
) -> list[Content]:
    # Score = alpha * ideological_similarity + (1-alpha) * virality
    # Similarity = 1 - |agent_opinion - content_score|  (ref doc Part 4)
    similarity = 1.0 - np.abs(agent_opinion - content_ideological_scores)
    scores = alpha * similarity + (1.0 - alpha) * content_virality
    # argpartition is faster than argsort when k_exp << C
    if len(scores) <= k_exp:
        return candidate_pool
    top_k_indices = np.argpartition(scores, -k_exp)[-k_exp:]
    return [candidate_pool[i] for i in top_k_indices]
```

**How to wire this in:**
Before the feed generation loop in `simulation.py`, pre-extract the content arrays once
per tick (not once per agent — that defeats the purpose):

```python
# Extract content score arrays ONCE per tick, outside the agent loop
if candidate_pool:
    content_ideo_array = np.array([c.ideological_score for c in candidate_pool])
    content_virality_array = np.array([c.virality for c in candidate_pool])
else:
    content_ideo_array = np.array([])
    content_virality_array = np.array([])

# Then inside the agent loop:
for agent in active_agents:
    agent.feed = recommender.generate_feed_vectorized(
        agent.opinion,
        content_ideo_array,
        content_virality_array,
        candidate_pool,
        k_exp,
        alpha
    )
```

**Correctness constraint:** The output must be the same set of top-k items as the naive
implementation. Verify by running both on the same input and comparing results before
removing the naive version.

**For diversity injection (Phase 3):** Vectorize the dissimilar ranking too:
```python
# Bottom-k for dissimilar content:
bottom_k_indices = np.argpartition(scores, k_div)[:k_div]
```

**For misinformation downranking (Phase 3):** Add misinfo penalty to the score array:
```python
content_misinfo_array = np.array([c.misinfo_score for c in candidate_pool])
scores = alpha * similarity + (1.0 - alpha) * content_virality \
         - lambda_penalty * content_misinfo_array
```
This is a single vectorized line — no loop change required.

---

### Step Opt.3 — Parallelize the tick loop with joblib

After vectorization, parallelize Steps 3, 4, and 5 of the tick loop across CPU cores.
The key constraint is that opinion updates must still be collected before being applied
(the simultaneous update invariant). `joblib` handles this correctly because each worker
returns a value rather than mutating shared state.

**Safe parallelization pattern:**

```python
from joblib import Parallel, delayed

def process_agent_tick(agent, feed, neighbors, influence_weights, config):
    """
    Pure function — reads agent state, returns updates. Does NOT mutate agent.
    This is safe to run in parallel because no shared state is written.
    Returns: (agent_id, new_opinion, shared_content_ids, new_arousal)
    """
    # Step 3: Compute new arousal (read-only on agent state)
    new_arousal = compute_arousal_update(agent, feed, config)

    # Step 4: Compute sharing decisions (read-only)
    shared = [c for c in feed if compute_share_prob(agent, c, config) > random.random()]

    # Step 5: Compute new opinion (read-only)
    new_opinion = agent.compute_update(neighbors, influence_weights)

    return agent.id, new_opinion, shared, new_arousal

# In the tick loop (Step 5 block in simulation.py):
results = Parallel(n_jobs=-1, prefer="threads")(
    delayed(process_agent_tick)(
        agent,
        agent.feed,
        get_predecessors(G, agent.id),   # DIRECTED GRAPH CONVENTION: predecessors = influence sources
        get_influence_weights(G, agent.id),
        config
    )
    for agent in active_agents
)

# Apply all results AFTER parallel computation completes (simultaneous update):
for agent_id, new_opinion, shared, new_arousal in results:
    agent_map[agent_id].opinion = new_opinion
    agent_map[agent_id].emotional_arousal = new_arousal
    shared_content_map[agent_id] = shared
```

**`prefer="threads"` vs `prefer="processes"`:**
Use `prefer="threads"` first. Python's GIL is released during NumPy operations, so
threaded parallelism works for vectorized code without the overhead of process spawning.
If profiling shows GIL contention is still a bottleneck (rare for this workload),
switch to `prefer="processes"` — but note that process-based parallelism requires all
arguments to be picklable, so ensure `Agent`, `Content`, and NetworkX graphs are picklable
before making that switch.

**`n_jobs=-1`:** Uses all available CPU cores. For Monte Carlo replication in Phase 0,
parallelize at the run level instead (one core per replicate), not at the agent level.
Do not nest `Parallel` calls — it causes overhead without benefit.

**Correctness constraint:** Run the MVP validation checklist again after adding parallelism.
Results must be deterministic when using the same seed. If they are not, the parallel
workers are sharing mutable state — find and fix the shared reference before proceeding.

---

### Step Opt.4 — Fast JSON serialization with orjson

Standard `json` cannot serialize NumPy arrays or numpy scalar types (float32, int64, etc.)
without custom encoders. `orjson` handles them natively and is 5–10× faster than `json`
for large payloads.

**In `api/main.py`**, replace the FastAPI default JSON response with orjson:

```python
import orjson
from fastapi.responses import Response

@app.post("/run")
async def run_simulation_endpoint(config: SimConfig):
    result = run_simulation(config.model_dump())
    # orjson serializes numpy arrays, numpy scalars, and dataclasses natively
    return Response(
        content=orjson.dumps(result, option=orjson.OPT_NON_STR_KEYS),
        media_type="application/json"
    )
```

**For WebSocket streaming (Phase 7)**, use orjson in the tick stream:
```python
await websocket.send_bytes(orjson.dumps(tick_snapshot))
# On the TypeScript side, parse with: JSON.parse(new TextDecoder().decode(data))
```

**NumPy array serialization rule:** Never pass raw NumPy arrays in snapshot dicts.
Convert arrays to Python lists before the orjson boundary:
```python
# BAD — works with orjson but breaks standard json and is unclear:
snapshot["agent_opinions"] = opinion_array  # np.ndarray

# GOOD — explicit, works everywhere:
snapshot["agent_opinions"] = opinion_array.tolist()
```
The one exception is `topic_vector` in content objects (Phase 5) — these are 128-dim
arrays that are never serialized to the frontend directly. Keep them as NumPy arrays
internally and exclude them from snapshot dicts.

---

### Step Opt.5 — Monte Carlo parallelization (preview of Phase 0)

When Phase 0 adds multi-seed replication, parallelize at the run level:

```python
from joblib import Parallel, delayed

def run_replicated(config: dict, n_runs: int = 10) -> dict:
    # Each replicate is independent — embarrassingly parallel
    results = Parallel(n_jobs=-1, prefer="processes")(
        delayed(run_simulation)({**config, "seed": config["seed"] + i})
        for i in range(n_runs)
    )
    return aggregate_results(results)
```

Use `prefer="processes"` here (not threads) because each replicate runs the full
simulation independently — process isolation prevents any accidental state sharing
between replicates, which would corrupt results.

---

### Optimization Validation Checklist

- [ ] Profiler output recorded in `# PERF BASELINE` comment before any changes
- [ ] Vectorized `generate_feed_vectorized()` produces identical top-k results to naive version (test with same seed)
- [ ] N=1000, T=100 run completes in under 15 seconds after vectorization (rough target — hardware dependent)
- [ ] N=1000, T=720 full run completes in under 60 seconds after parallelization (rough target)
- [ ] MVP validation checklist still passes after parallelization (results are still deterministic with same seed)
- [ ] orjson serializes a full N=1000 snapshot without errors, including float arrays
- [ ] No `NameError` on `profile` decorator when running without `kernprof`

---

## Phase 0 — Monte Carlo + Experiment Runner

**Insert immediately after MVP. All subsequent phases depend on this.**

### Goal
Make simulation results scientifically reliable by adding multi-seed replication
and a config-driven experiment runner that sweeps parameter grids.

### Context
A single-seed run is not publishable. With n_runs=10–30 you get mean ± std per metric,
which lets you make claims about system behavior and compare interventions statistically.

### Step 0.1 — Multi-seed replication in `simulation.py`

Add a `run_replicated(config, n_runs) -> ReplicatedResult` function:

```python
def run_replicated(config: dict, n_runs: int = 10) -> dict:
    all_snapshots = []
    for run_idx in range(n_runs):
        seed = config["seed"] + run_idx  # deterministic seed ladder
        result = run_simulation({**config, "seed": seed})
        all_snapshots.append(result["snapshots"])

    # Aggregate: for each metric at each tick, compute mean and std
    aggregated = aggregate_snapshots(all_snapshots)
    return {
        "config": config,
        "n_runs": n_runs,
        "aggregated": aggregated,   # mean + std per metric per tick
        "all_runs": all_snapshots   # raw data for distribution plots
    }
```

**Aggregation shape:**
```python
{
    "tick": [0, 5, 10, ...],
    "opinion_variance_mean": [...],
    "opinion_variance_std": [...],
    "polarization_index_mean": [...],
    "polarization_index_std": [...],
    # ... same pattern for all 5 metrics
}
```

### Step 0.2 — `backend/sim/experiment_runner.py` (new file)

```python
def run_experiment(
    base_config: dict,
    param_grid: dict[str, list],   # e.g. {"alpha": [0.0, 0.3, 0.65, 1.0]}
    n_runs: int = 10,
    output_path: str = "results/"
) -> list[dict]:
```

- Generates cartesian product of `param_grid` values
- For each combination, merges with `base_config` and calls `run_replicated`
- Writes each result to `output_path/{scenario_name}.json`
- Returns list of all aggregated results

**Standardized output JSON per scenario:**
```json
{
  "scenario_name": "alpha_sweep_0.65",
  "parameters": {"alpha": 0.65, "N": 200},
  "n_runs": 10,
  "metrics": {
    "misinfo_peak_mean": 0.34,
    "misinfo_peak_std": 0.04,
    "final_assortativity_mean": 0.61,
    "final_assortativity_std": 0.03,
    "final_entropy_mean": 1.82,
    "final_entropy_std": 0.11,
    "final_polarization_mean": 0.72,
    "final_polarization_std": 0.05
  }
}
```

### Step 0.3 — Add `POST /run/replicated` API endpoint

Accepts `SimConfig` + `n_runs: int`. Returns `ReplicatedResult`.
This is a slow endpoint — add a warning in the response headers if `n_runs * T > 50000`.

### Step 0.4 — Baseline experiment notebook

`notebooks/06_baseline_experiment.ipynb`:
- Run the reference baseline config 10 times
- Plot metric time-series with shaded std bands (mean ± 1σ)
- Confirm results are stable across seeds (std < 0.1 * mean for all metrics)
- This notebook is your scientific baseline — save and commit its outputs

### Validation
- [ ] Two runs with same seed produce identical results
- [ ] Two runs with different seeds produce different but similar results
- [ ] Std bands are visible but narrow on metric plots (system is consistent, not chaotic)
- [ ] Experiment runner produces one JSON file per parameter combination

---

## Phase 1 — Agent Richness

### Goal
Add HK bounded-confidence, Contrarian, Passive, and Influencer agent types.
Refactor Agent into a proper base class with `compute_update()` override pattern.

### Context
- Prerequisite: MVP `agent.py` must have `compute_update()` as a method, not a module-level function.
- Do NOT add `if agent_type == "hk"` branching in the tick loop. New types are new subclasses.
- All new types use the same state variable schema as the MVP base agent.

### Step 1.1 — Refactor `agent.py` base class

If MVP used a dataclass with type-based branching, refactor now:
- `Agent` becomes the base class with `compute_update()` raising `NotImplementedError`
- `StubbornAgent` and `FlexibleAgent` become explicit subclasses
- Update `simulation.py` to call `agent.compute_update()` polymorphically
- Run MVP validation checklist again to confirm nothing broke

### Step 1.2 — `HKAgent` (Hegselmann–Krause)

**Update rule:**
```
N_i(t) = {j : |x_j(t) - x_i(t)| <= epsilon}
x_i(t+1) = (1 / |N_i(t)|) * sum(x_j(t) for j in N_i(t))
```
- Uses `self.confidence_bound` (ε) — already in state variables, was a placeholder in MVP
- If `N_i(t)` is empty (no neighbors within ε), opinion is unchanged
- Default ε = 0.3
- Effect: low ε → opinion clusters. high ε → consensus. This is the primary echo chamber mechanism.

### Step 1.3 — `ContrarianAgent`

**Update rule:**
```python
# Compute the direction and magnitude of neighbor social pressure:
neighbor_avg = sum(a_ij * x_j for j in neighbors)  # weighted neighbor average
influence_delta = neighbor_avg - self.opinion        # what neighbors push toward

if random() < self.contrarian_prob:
    # Oppose the social pressure — move AWAY from neighbor average
    new_opinion = self.opinion - (1 - self.stubbornness) * influence_delta
else:
    # Normal FJ update — move toward neighbor average
    new_opinion = self.stubbornness * self.initial_opinion \
                  + (1 - self.stubbornness) * neighbor_avg

return clamp(new_opinion, -1.0, 1.0)
```
# NOTE (C1 fix): The contrarian flips the *direction of social pressure* (moves away from
# neighbor_avg), NOT the sign of the absolute opinion value. Do NOT use `-base_update`.
# Ref doc Part 1.1: "adopt the opposite of what neighbors would normally push them toward."

- Uses `self.contrarian_prob` (p_c) — already in state variables
- Default p_c = 0.3 for contrarian agents
- Clamp result to [-1, 1] as always

### Step 1.4 — `InfluencerAgent`

- Same `compute_update()` as `StubbornAgent` (FJ rule)
- Difference is at network construction time: initialized as a hub node
- In `network.py`, after graph creation, re-wire influencer nodes to have degree ≥ 3× avg_degree
- Add `influence_weight_multiplier` to state (default 2.0): neighbors weight influencer's
  opinion `influence_weight_multiplier` times more than normal when computing their own updates
- **M4 fix:** Apply `influence_weight_multiplier` BEFORE normalization, not as a post-hoc
  multiplier (post-hoc breaks DeGroot row-sum = 1 constraint, causing opinion drift).
  Use this pattern in `network.py`:
  ```python
  def compute_edge_weights(G, agent_id, agents):
      # DIRECTED GRAPH CONVENTION: predecessors = agents that agent_id listens to.
      neighbors = list(G.predecessors(agent_id))
      raw_weights = {}
      for n_id in neighbors:
          multiplier = agents[n_id].influence_weight_multiplier  # 1.0 normal, 2.0 influencer
          raw_weights[n_id] = multiplier
      total = sum(raw_weights.values())
      return {n_id: w / total for n_id, w in raw_weights.items()}
  ```
  This ensures row sums always equal 1.0 as required by the DeGroot rule (ref doc Part 5).

### Step 1.5 — `PassiveAgent`

- Identical to `StubbornAgent` in update logic
- Only difference: `activity_rate ~ N(0.07, 0.03)` clipped to [0.01, 0.2]
- Still consumes content and updates opinion — just rarely posts
- Not a new subclass required; can be a `StubbornAgent` initialized with low `activity_rate`
  if the codebase is cleaner that way

### Step 1.6 — Update population factory

In `simulation.py`, the agent population factory must handle the new `agent_mix` keys:
```python
"agent_mix": {
    "stubborn": 0.60,
    "flexible": 0.20,
    "passive": 0.10,
    "zealot": 0.05,
    "bot": 0.05,
    # New in Phase 1:
    "hk": 0.0,        # default 0, set in config to use
    "contrarian": 0.0,
    "influencer": 0.0
}
```
Fractions must sum to 1.0. Assert this at initialization.

### Step 1.7 — Validation notebook

`notebooks/07_agent_types.ipynb`:
- Test each new agent type in isolation (population of 100 identical agents)
- HK with ε=0.1: should form 3+ distinct clusters by tick 100
- HK with ε=0.8: should converge to near-consensus by tick 100
- Contrarian: should show opinion oscillation rather than convergence
- Influencer: neighbors' opinions should track influencer's opinion more closely than normal

---

## Phase 2 — Emotion System

### Goal
Add emotional arousal dynamics that make high-valence content spread faster and
make aroused agents less receptive to opposing opinions.

### Context
- Prerequisite: Phase 1 complete (HK agents need arousal-adjusted ε)
- Changes the tick loop at Steps 3 and 4
- The MVP stubbed both of these — this phase fills them in
- Do not change the sharing probability for Zealots or Bots (they have fixed behavior)

### Step 2.1 — Arousal update (tick loop Step 3)

Replace the MVP stub with the full update rule from reference doc Part 6:
```
e_i(t+1) = lambda * e_i(t) + (1 - lambda) * v_c
```
- `lambda` = emotional decay rate, default 0.85
- `v_c` = `emotional_valence` of the content item consumed this tick
- If agent consumed multiple items, apply the update once per item sequentially
- Clamp `emotional_arousal` to [0, 1] after each update

**Emotional valence generation in `content.py` (update from MVP):**
- Misinformation content (`is_misinformation=True`): sample from `Beta(3, 1)` (skewed high)
- Normal content: sample from `Beta(1, 3)` (skewed low)

### Step 2.2 — Sharing probability (tick loop Step 4)

Replace the fixed `base_share_prob` with the sigmoid formula:
```
P(share | content, agent) = sigmoid(base_share + w_e * e_i(t) + w_v * v_c)
```
where:
- `base_share` = −1.5 (log-odds ≈ 18% base rate)
- `w_e` = 0.3 (arousal weight)
- `w_v` = 0.4 (content valence weight)
- `sigmoid(x) = 1 / (1 + exp(-x))`

Bots: ignore this formula. Bots share everything with probability 1.0.

### Step 2.3 — Arousal effect on HK confidence bound

In `HKAgent.compute_update()`, replace `self.confidence_bound` with effective bound:
```
epsilon_eff = self.confidence_bound * (1 - gamma_e * self.emotional_arousal)
```
- `gamma_e` = 0.4 (default)
- Use `epsilon_eff` in the neighbor filtering step, not raw `self.confidence_bound`
- Effect: highly aroused HK agents become more closed-minded

### Step 2.4 — Expose emotion params in config

Add to `SimConfig`:
```python
emotional_decay: float = 0.85     # lambda
arousal_share_weight: float = 0.3  # w_e
valence_share_weight: float = 0.4  # w_v
arousal_tolerance_effect: float = 0.4  # gamma_e
```

### Step 2.5 — Validation notebook

`notebooks/08_emotion_system.ipynb`:
- Show that high-valence misinformation spreads faster than low-valence misinformation
- Show that agents with high arousal share more
- Show that aroused HK agents form tighter clusters (lower effective ε)
- Plot `emotional_arousal` distribution over time

---

## Phase 2b — SIR Reinforcement Effect

### Goal
Make misinformation persistent by modeling repeated exposure increasing infection probability.

### Context
- Small addition to the SIR module in `simulation.py`
- Can be done alongside Phase 2 or immediately after
- Prerequisite: MVP SIR must be working correctly

### Step 2b.1 — Add `exposure_count` to agent state

```python
exposure_count: dict[int, int] = field(default_factory=dict)
# key: content_id (or campaign_id), value: number of times exposed
```

### Step 2b.2 — Update SIR transition in tick loop Step 5

Replace the MVP SIR S→I transition:
```python
# MVP:
if random() < sir_beta:
    agent.sir_state = "I"

# Phase 2b:
agent.exposure_count[misinfo_id] = agent.exposure_count.get(misinfo_id, 0) + 1
n_exposures = agent.exposure_count[misinfo_id]
effective_beta = sir_beta * (1 + n_exposures * reinforcement_factor)
effective_beta = min(effective_beta, 1.0)  # cap at 1.0
if random() < effective_beta:
    agent.sir_state = "I"
```

### Step 2b.3 — Expose in config

```python
reinforcement_factor: float = 0.0  # default off; set to 0.2-0.5 to enable
```

### Validation
- [ ] With `reinforcement_factor=0`, behavior matches MVP SIR exactly
- [ ] With `reinforcement_factor=0.3`, misinformation peak is higher and later
- [ ] Agents exposed 3+ times have meaningfully higher effective beta than first-exposure agents

---

## Phase 3 — Recommender Variants + Intervention Hooks

### Goal
Add collaborative filtering and graph-based recommenders. Add recommender-level
intervention hooks: diversity injection, misinformation downranking, virality dampening.

### Context
- Prerequisite: Phases 1–2 complete
- **Architectural rework required:** `recommender.py` must be refactored from a function
  into a class hierarchy before adding new variants. Do this first.
- Intervention hooks are NOT separate modules — they are parameters on the recommender classes.

### Step 3.1 — Refactor `recommender.py` into class hierarchy

```python
class BaseRecommender:
    def score(self, content: Content, agent: Agent) -> float:
        raise NotImplementedError

    def generate_feed(self, agent, candidate_pool, k_exp) -> list[Content]:
        # Fallback loop for non-vectorized subclasses (CF, graph-based).
        # ContentBasedRecommender overrides this with the vectorized implementation.
        scored = [(self.score(c, agent), c) for c in candidate_pool]
        scored.sort(reverse=True)
        return [c for _, c in scored[:k_exp]]

class ContentBasedRecommender(BaseRecommender):
    def __init__(self, alpha, diversity_ratio=0.0, lambda_penalty=0.0):
        ...
    def score(self, content, agent) -> float:
        ...
    def generate_feed(self, agent, candidate_pool, k_exp) -> list[Content]:
        # M5 fix (Option A): ContentBasedRecommender overrides generate_feed() with the
        # vectorized implementation from Phase Opt (generate_feed_vectorized). The score()
        # method is kept for testing and for non-vectorized subclasses. CF and graph-based
        # recommenders use the BaseRecommender loop fallback via score(). Do NOT try to call
        # a per-item score() method from within a vectorized batch path — the two patterns
        # are incompatible. See Phase Opt Step Opt.2 for the vectorized implementation.
        ...
```

### Step 3.2 — Add intervention hooks to `ContentBasedRecommender`

**A. Diversity injection** (`diversity_ratio ∈ {0.0, 0.1, 0.2, 0.3}`):
```python
def generate_feed(self, agent, candidate_pool, k_exp):
    k_sim = int(k_exp * (1 - self.diversity_ratio))
    k_div = k_exp - k_sim
    similar = sorted by similarity (high to low)
    dissimilar = sorted by similarity (low to high)
    feed = similar[:k_sim] + dissimilar[:k_div]
    return shuffle(feed)
```

**B. Misinformation downranking** (`lambda_penalty ∈ {0.0, 0.25, 0.5, 1.0}`):
```python
def score(self, content, agent):
    base_score = alpha * (1 - abs(content.ideological_score - agent.opinion))
                + (1 - alpha) * content.virality
    return base_score - self.lambda_penalty * content.misinfo_score
```

**C. Virality dampening** (`virality_dampening ∈ {0.0, 0.3, 0.6}`):
Do NOT add a separate multiplier. Instead, adjust `w_v` in the sharing probability formula
(Phase 2 sigmoid). Dampening reduces the influence of valence on sharing:
```python
effective_w_v = w_v * (1 - virality_dampening)
```
Add `virality_dampening` to `SimConfig`. Apply in simulation.py Step 4.

### Step 3.3 — `CollaborativeFilteringRecommender`

Simple k-nearest-neighbors approach (not matrix factorization):
- Cluster agents by opinion similarity: agents with `|opinion_i - opinion_j| < 0.2` are peers
- Recommend content that peer agents engaged with (appeared in their feed and they shared it)
- Score: `(1 - opinion_distance_to_peer) * peer_content_virality`
- Fallback to content-based scoring if peer cluster is empty

### Step 3.4 — `GraphBasedRecommender`

Random walk on the social graph:
- From agent i, take a random walk of length `walk_length` (default 3) on the follow graph
- Collect content from visited agents
- Score by proximity (shorter walk = higher score) and ideological similarity
- `restart_probability` (default 0.15): at each step, restart from origin with this probability

### Step 3.5 — Add recommender type to config

```python
recommender_type: str = "content_based"  # "content_based" | "cf" | "graph" | "hybrid"
cf_blend_ratio: float = 0.5              # only for "hybrid": fraction from CF vs content-based
diversity_ratio: float = 0.0
lambda_penalty: float = 0.0
virality_dampening: float = 0.0
```

### Step 3.6 — Experiment: recommender comparison

Run the reference baseline config under each recommender type. Compare:
- Final assortativity
- Misinformation peak
- Time to opinion clustering

### Validation
- [ ] `diversity_ratio=0.3` produces lower assortativity than `diversity_ratio=0.0`
- [ ] `lambda_penalty=1.0` produces lower misinformation peak than `lambda_penalty=0.0`
- [ ] CF recommender produces faster echo chamber formation than content-based
- [ ] Content-based recommender with `alpha=0` produces a virality-sorted feed
  (verify highest-virality content appears most). CF and graph-based recommenders
  are unaffected by `alpha` — do not test them with alpha=0 as a baseline check.
  # M7 fix: "alpha=0 gives random feed" is WRONG. alpha=0 gives virality-ranked feed
  # (score = 0*ideology + 1.0*virality). For a truly random feed, use a dedicated
  # RandomRecommender or set all content virality to a constant.

---

## Phase 4 — Network Dynamics + Rewiring Intervention

### Goal
Add dynamic edge rewiring (agents unfollow disagreeing peers and follow similar ones),
agent churn, and additional network topology options. Rewiring itself is treated as an
intervention with its own experiment grid.

### Context
- Prerequisite: Phase 0 (Monte Carlo) must be complete — rewiring effects have high variance
- **Metric warning:** Rewiring changes graph structure mid-run. Assortativity and modularity
  must recompute from the live graph, not a cached version. Verify `metrics.py` always
  operates on the current `G`, never a snapshot of it.

### Step 4.1 — Dynamic rewiring (tick loop Step 6)

Fill in the MVP stub. Define `renormalize_weights()` first — it is called by `rewire_step()`
and must live in `network.py` alongside the other graph helpers:

```python
def renormalize_weights(G: nx.DiGraph, agent_id: int) -> None:
    """
    Recompute normalized edge weights for all edges INCOMING to agent_id.
    Called after any edge add/remove that changes agent_id's predecessor set.

    # DIRECTED GRAPH CONVENTION:
    # Opinion updates consume INCOMING weights (predecessors → agent_id).
    # These are the weights that must sum to 1.0 for the DeGroot constraint.
    # Outgoing weights (agent_id → successors) are stored on those edges for
    # the successors' own normalization — do NOT touch them here.
    # Each agent is responsible for normalizing its own incoming weights
    # whenever its predecessor set changes.

    Weight strategy: uniform 1/in_degree by default in MVP.
    In Phase 1 Step 1.4, InfluencerAgent nodes carry influence_weight_multiplier != 1.0;
    call compute_edge_weights() instead of this function for those nodes.
    """
    preds = list(G.predecessors(agent_id))
    if not preds:
        return  # isolated node — nothing to normalize

    # Read existing raw weights (or 1.0 if not yet set)
    raw = {j: G[j][agent_id].get('weight', 1.0) for j in preds}
    total = sum(raw.values())
    if total == 0:
        return  # degenerate — leave weights untouched to avoid div-by-zero

    for j in preds:
        G[j][agent_id]['weight'] = raw[j] / total
```

> **When to call `renormalize_weights()`:**
> - After `G.remove_edge(u, v)`: call on `v` (its predecessor set shrank)
> - After `G.add_edge(u, v)`: call on `v` (its predecessor set grew)
> - Do NOT call on `u` in either case — `u`'s incoming weights are unaffected.
> - `rewire_step()` removes edge `(agent.id → worst.id)` and adds `(agent.id → new_follow.id)`.
>   This changes the predecessor sets of `worst.id` and `new_follow.id`, not of `agent.id`.
>   The call site in `rewire_step()` is therefore: `renormalize_weights(G, worst.id)` (after
>   remove) and `renormalize_weights(G, new_follow.id)` (after add).

```python
def rewire_step(G, agents, config):
    for agent in active_agents:
        if random() < config["dynamic_rewire_rate"]:  # default 0.01
            # DIRECTED GRAPH CONVENTION:
            # Rewiring = changing who this agent FOLLOWS (outgoing edges = successors).
            # get_predecessors() = who follows ME (don't touch those edges here).
            following = get_successors(G, agent.id)   # agents this agent currently follows
            if not following:
                continue
            worst = max(following, key=lambda n: abs(n.opinion - agent.opinion))
            if abs(worst.opinion - agent.opinion) > config["homophily_threshold"]:
                G.remove_edge(agent.id, worst.id)
                renormalize_weights(G, worst.id)      # worst.id lost a predecessor
                # Find a new agent within homophily_threshold to follow
                candidates = [a for a in agents
                              if a.id != agent.id
                              and a.id not in get_successors(G, agent.id)
                              and abs(a.opinion - agent.opinion) <= config["homophily_threshold"]]
                if candidates:
                    new_follow = random.choice(candidates)
                    G.add_edge(agent.id, new_follow.id, weight=1/G.out_degree(agent.id))
                    renormalize_weights(G, new_follow.id)  # new_follow.id gained a predecessor
```

**Rewiring intervention experiment grid:**
```
dynamic_rewire_rate ∈ {0.0, 0.2, 0.5}  # M6: use dynamic_rewire_rate everywhere.
                                         # r_dyn in ref doc Part 7, Part 8.
                                         # (was incorrectly labeled p_unfollow — removed)
homophily_threshold ∈ {0.3, 0.5}
```

### Step 4.2 — Agent churn (tick loop Step 7)

```python
def churn_step(G, agents, config):
    if not config.get("enable_churn", False):
        return
    for agent in active_agents:
        dissatisfaction = compute_dissatisfaction(agent, G)
        p_churn = sigmoid(config["churn_base"] + config["churn_weight"] * dissatisfaction)
        if random() < p_churn:
            agent.is_active = False
            G.remove_node(agent.id)
```

`dissatisfaction` = mean opinion distance to all neighbors. High disagreement → more likely to leave.

### Step 4.3 — Additional topology options in `network.py`

```python
def build_graph(config) -> nx.DiGraph:
    topology = config.get("topology", "watts_strogatz")
    if topology == "watts_strogatz":
        G = nx.watts_strogatz_graph(N, k, p)
    elif topology == "barabasi_albert":
        G = nx.barabasi_albert_graph(N, m=int(avg_degree/2))
    elif topology == "erdos_renyi":
        G = nx.erdos_renyi_graph(N, p=avg_degree/N)
    elif topology == "stochastic_block":
        # Pre-seeded community structure
        sizes = config["community_sizes"]   # e.g. [500, 500]
        p_matrix = config["community_p"]    # e.g. [[0.1, 0.01], [0.01, 0.1]]
        G = nx.stochastic_block_model(sizes, p_matrix)
    # m6: All topologies returned as DiGraph. Undirected generators produce symmetric
    # edge pairs; directed rewiring in Phase 4 then makes the graph asymmetric over time.
    return nx.DiGraph(G)
```

### Validation
- [ ] With `dynamic_rewire_rate=0.5` and `homophily_threshold=0.3`, assortativity rises
  faster than a static network run
- [ ] Barabási–Albert graph has a small number of very high-degree nodes (verify with degree distribution plot)
- [ ] Churn does not crash the simulation when the graph becomes sparse
- [ ] Metrics still compute correctly on a rewired graph (no cached graph state bugs)

---

## Phase 5 — Bot Detection + Full Misinformation

### Goal
Add behavioral bot detection using suspicion scores (not ground-truth labels).
Extend content schema with full fields. Add multi-claim SIR tracking.

### Context
- Prerequisites: Phases 1 and 2 (bot detection signals require activity rate and emotion data)
- **Do not use `agent.agent_type == "bot"` in detection logic.** Detection must use
  observable behavioral signals only. Using ground truth is scientifically invalid.
- Bot removal and rate limiting are interventions with their own experiment grids.

### Step 5.1 — Full content schema

Add to `content.py` (these were placeholders in MVP):
```python
topic_vector: np.ndarray    # shape (128,), random unit vector for now
coordinated_campaign_id: int | None   # links to bot campaign; None if organic
is_satire: bool
# M8: valid_until removed — misinfo aging is tracked via campaign_expiry dict
#     in simulation.py, not per content object. See get_effective_misinfo_score().
```

For now, `topic_vector` is a random unit vector (not semantically meaningful).
In a future research extension, replace with embeddings from a real topic model.

**`misinfo_score` aging (M8 — Option B: campaign-level expiry):**

Do NOT age individual content objects. Instead, maintain a simulation-level dict:
```python
# In simulation.py, alongside agents and G:
campaign_expiry: dict[int, int] = {}  # campaign_id -> tick at which misinfo becomes "exposed"
```
At content consumption time (inside `generate_feed` or the feed-processing loop), apply:
```python
def get_effective_misinfo_score(content: Content, tick: int, campaign_expiry: dict) -> float:
    cid = content.coordinated_campaign_id
    if cid is not None and tick > campaign_expiry.get(cid, float('inf')):
        # Campaign debunked/expired — elevate misinfo signal
        return min(content.misinfo_score + 0.3, 1.0)
    return content.misinfo_score
```
Use `get_effective_misinfo_score()` wherever `content.misinfo_score` would otherwise be
read during agent processing. The `content.misinfo_score` field itself is never mutated.

> **Rationale:** Content objects are short-lived (1–2 ticks). Aging per-object would only
> ever fire in the same tick the content was created, making it a no-op in practice.
> The campaign dict approach is O(1) per lookup, zero memory overhead, and correctly
> models debunking as a campaign-level event. `valid_until` on individual `Content` objects
> is removed from the Phase 5 schema — do NOT add it back.

**Satire handling:** Agents with `media_literacy < 0.4` treat satire as factual:
```python
effective_misinfo_score = 0.5 if content.is_satire and agent.media_literacy < 0.4
                          else content.misinfo_score
```

### Step 5.2 — Multi-claim SIR

Replace the single global SIR state with per-claim tracking:
```python
# In agent state, replace sir_state: str with:
sir_states: dict[int, str] = {}  # key: campaign_id, value: "S"|"I"|"R"
```

SIR transitions now happen per campaign. An agent can be I for campaign 1 and S for campaign 2.
Default state for any unseen campaign_id is "S".

### Step 5.3 — `backend/sim/bot_detection.py` (new file)

**Suspicion score computation** — behavioral signals only:

```python
def compute_suspicion_score(agent, recent_posts, recent_shares, G) -> float:
    signals = []

    # Signal 1: abnormally high post frequency
    # Compare agent's posts-per-tick to population mean
    post_rate_z = (agent_post_rate - population_mean_post_rate) / population_std_post_rate
    signals.append(sigmoid(post_rate_z))

    # Signal 2: near-zero opinion variance over last 50 ticks
    opinion_variance = variance(agent.opinion_history[-50:])
    signals.append(1.0 - min(opinion_variance * 10, 1.0))

    # Signal 3: high average emotional valence of shared content
    mean_valence = mean(c.emotional_valence for c in recent_shares)
    signals.append(mean_valence)

    # Signal 4: low reciprocity ratio (bots follow but aren't followed back)
    # DIRECTED GRAPH CONVENTION:
    #   following  = get_successors()   — agents this agent follows (outgoing edges)
    #   followers  = get_predecessors() — agents that follow this agent (incoming edges)
    following  = set(get_successors(G, agent.id))
    followers  = set(get_predecessors(G, agent.id))
    reciprocity = len(followers & following) / (len(following) + 1e-6)
    signals.append(1.0 - reciprocity)

    return mean(signals)
```

Run every `T_detect = 24` ticks (configurable). Flag agents with `suspicion_score >= 0.7`.

**Note:** Add `opinion_history: list[float]` to agent state to support Signal 2.
Append current opinion at the end of each tick.

### Step 5.4 — Bot intervention hooks (tick loop Step 8)

```python
def bot_detection_step(G, agents, config, tick):
    if tick % config["T_detect"] != 0:
        return
    for agent in active_agents:
        agent.suspicion_score = compute_suspicion_score(agent, ...)
        if agent.suspicion_score >= config["s_thresh"]:
            if random() < config["p_detect_remove"]:
                agent.is_active = False
                G.remove_node(agent.id)
            elif config["rate_limit_factor"] > 0:
                agent.activity_rate *= (1 - config["rate_limit_factor"])
```

**Experiment grid:**
```
p_detect_remove ∈ {0.0, 0.3, 0.6}
rate_limit_factor ∈ {0.0, 0.5}
```

### Step 5.5 — Media literacy intervention

In `SimConfig`:
```python
media_literacy_boost: float = 0.0  # ∈ {0.0, 0.2, 0.4}
```

At initialization, after sampling base `media_literacy`:
```python
agent.media_literacy = clamp(agent.media_literacy + config["media_literacy_boost"], 0, 1)
```

### Validation
- [ ] Bot suspicion scores are meaningfully higher than human suspicion scores by tick 100
  (without using agent_type as a signal)
- [ ] `p_detect_remove=0.6` reduces bot prevalence in the active agent population over time
- [ ] `media_literacy_boost=0.4` reduces misinformation peak compared to boost=0.0
- [ ] Multi-claim SIR: agent can be I for one campaign and S for another simultaneously

---

## Phase 6 — Full Metrics + Interventions + Result Interpretation

### Goal
Add remaining metrics from the reference doc. Add the intervention effectiveness score (IES).
Add standardized result interpretation output for policy-style findings.

### Context
- Prerequisites: Phases 3–5 (interventions need recommender + bot infra)
- Metrics are additive — new functions in `metrics.py`, no restructuring

### Step 6.1 — Add remaining metrics to `metrics.py`

**E-I Index:**
```
EI = (external_ties - internal_ties) / (external_ties + internal_ties)
```
Requires defining groups. Use opinion quartiles as groups: [−1,−0.5), [−0.5,0), [0,0.5), [0.5,1].
External ties = edges between agents in different quartiles.
Result: −1 = pure echo chambers, +1 = pure cross-cutting network.

**Modularity Q:**
```python
communities = nx.community.greedy_modularity_communities(G)
Q = nx.community.modularity(G, communities)
```
Values 0.3–0.7 indicate strong community structure (echo chambers).

**Cascade Size:**
Track per content item: how many agents shared it (directly or transitively).
Implement as a breadth-first traversal from the original creator through share events.
Report mean cascade size and max cascade size per tick window.

**Exposure Disparity:**
For each opinion quartile group, compute mean misinformation exposure rate.
`exposure_disparity = max(group_rates) - min(group_rates)`
0 = equal exposure, high = one group sees much more misinformation.

**Intervention Effectiveness Score (IES):**
```python
def compute_IES(baseline_result, intervention_result, metric_name) -> float:
    M_control = baseline_result["metrics"][metric_name]
    M_intervention = intervention_result["metrics"][metric_name]
    return (M_control - M_intervention) / (M_control + 1e-10)
```
Positive IES = intervention reduced the metric (good for misinfo/polarization).
Negative IES = intervention made it worse.

### Step 6.2 — Scenario comparison API endpoint

```python
POST /run/compare
# Body: { "baseline": SimConfig, "intervention": SimConfig, "n_runs": int }
# Returns: { "baseline": ReplicatedResult, "intervention": ReplicatedResult, "IES": dict }
```

### Step 6.3 — Standardized result interpretation output

Add to `experiment_runner.py` a `format_policy_finding(result, intervention_name)` function
that produces this text block:
```
Intervention: {intervention_name} ({param_name} = {value})

Findings:
- Peak misinformation {reduced|increased} by {X}% (IES = {ies:.2f})
- Final assortativity: {value:.3f} ± {std:.3f}
- Opinion entropy: {value:.3f} ± {std:.3f}

Interpretation:
{auto-generated one sentence based on IES direction and magnitude}

Tradeoffs detected:
{list any metrics that moved in the wrong direction}
```

### Step 6.4 — Frontend: scenario comparison view

Add a "Compare" tab to the dashboard:
- Two `SimConfig` forms side by side (baseline left, intervention right)
- After running: side-by-side `MetricsPanel` components
- IES badge per metric showing % change with color (green = improvement, red = worse)

### Validation
- [ ] Modularity Q increases over time in a run with high homophily and high alpha
- [ ] E-I index is negative (within-group ties dominant) under same conditions
- [ ] IES of 0.0 when baseline config == intervention config (sanity check)
- [ ] Diversity injection shows positive IES for misinformation but check for negative IES on polarization

---

## Phase 7 — Live Streaming Frontend

### Goal
Upgrade from request/response to WebSocket streaming. Simulation updates the frontend
on every tick in real time. Add play/pause/step controls.

### Context
- This is the biggest frontend architectural change. It replaces the `useSimulation` hook model.
- Real-time graph rendering is only viable at N≤200. For N>200, graph updates every 10 ticks.
- Prerequisite: none — this can be done after MVP if demo quality is the priority.
  But it is cleaner to have Phase 0 done first (stable simulation output).

### Step 7.1 — WebSocket endpoint in `api/main.py`

```python
@app.websocket("/run/stream")
async def stream_simulation(websocket: WebSocket):
    await websocket.accept()
    config = await websocket.receive_json()
    # Run simulation tick by tick, yielding control between ticks
    for tick_snapshot in run_simulation_streaming(config):
        await websocket.send_json(tick_snapshot)
        await asyncio.sleep(0)  # yield to event loop
    await websocket.close()
```

`run_simulation_streaming(config)` is a generator that yields one snapshot dict per tick
(not per snapshot_interval — send every tick for live feel, frontend can downsample).

Each streamed message shape:
```json
{
  "tick": int,
  "metrics": { ...MetricSnapshot... },
  "agent_opinions": [float],        // array indexed by agent id
  "edges_changed": bool,            // true if rewiring happened this tick
  "edge_list": [[int,int]]          // only included if edges_changed = true
}
```

### Step 7.2 — `hooks/useSimulation.ts` upgrade

Replace the `fetch`-based hook with a WebSocket-based one:
```typescript
const ws = new WebSocket('ws://localhost:8000/run/stream')
ws.onopen = () => ws.send(JSON.stringify(config))
ws.onmessage = (event) => {
    const snapshot = JSON.parse(event.data)
    setSnapshots(prev => [...prev, snapshot.metrics])
    setCurrentOpinions(snapshot.agent_opinions)
    if (snapshot.edges_changed) setEdgeList(snapshot.edge_list)
}
ws.onclose = () => setStatus('done')
```

### Step 7.3 — Live network graph

In `NetworkGraph.tsx`:
- On receiving new `agent_opinions`, update node colors only (no layout recalculation)
- On receiving `edges_changed=true`, trigger a layout re-render
- Add `N > 200` guard: if N > 200, only update graph every 10 ticks
- Use `react-force-graph-2d` with `cooldownTicks={0}` to freeze layout after initial settle

### Step 7.4 — Playback controls

Add to `ControlPanel.tsx`:
- Play/Pause button: sends `{"command": "pause"}` or `{"command": "resume"}` through WebSocket
- Step button: sends `{"command": "step"}` — advances one tick while paused
- Speed slider: sends `{"command": "set_speed", "ticks_per_second": N}`
- Handle these commands in the FastAPI WebSocket handler with an asyncio Event or Queue

### Validation
- [ ] Network node colors update smoothly at N=150 (no visible lag)
- [ ] Pause/resume works correctly (sim stops and resumes at the right tick)
- [ ] At N=500, graph only updates every 10 ticks but metric charts update every tick
- [ ] WebSocket closes cleanly when simulation completes (no hanging connections)

---

## Phase 8 — Multi-Platform + RL Recommender

### Goal
Support M parallel platform graphs with cross-platform opinion coupling.
Add a reinforcement learning recommender that learns from agent engagement.

### Context
- **Major architectural change.** The simulation loop must manage M graphs simultaneously.
- Prerequisites: All previous phases. This is the research-grade endgame.
- RL recommender requires a training loop separate from the main sim loop.
- Estimated effort: 4–6 weeks solo.

### Step 8.1 — Multi-platform simulation architecture

Introduce a `Platform` class:
```python
@dataclass
class Platform:
    id: int
    graph: nx.Graph
    agents: list[Agent]        # agents on this platform
    recommender: BaseRecommender
    alpha: float
    topology: str
```

Refactor `simulation.py`:
- `run_simulation()` now instantiates M `Platform` objects
- Each tick, run the full tick loop for each platform independently
- After each platform's tick loop, apply cross-platform coupling

**Cross-platform coupling (φ parameter, default 0.1):**
```python
for agent in platform_a.agents:
    if random() < phi:
        # Agent is exposed to content from a random other platform this tick
        other_platform = random.choice(other_platforms)
        cross_content = sample(other_platform.content_pool, k=3)
        agent.feed += cross_content
```

### Step 8.2 — Shared agents (agents on multiple platforms)

Add `platforms: list[int]` to agent state. Shared agents appear in multiple `Platform.agents` lists.
Their opinion is updated as a weighted average of updates from each platform they're on.
Their `emotional_arousal` is shared across platforms (one global state).

### Step 8.3 — RL Recommender

Use a simple tabular Q-learning or policy gradient — no deep RL required.

**State space:** Discretized agent opinion (5 bins) × arousal level (3 bins) = 15 states
**Action space:** Content ideological distance buckets: [similar, neutral, opposing]
**Reward:** Engagement rate (sharing probability) minus polarization penalty

```python
class RLRecommender(BaseRecommender):
    def __init__(self, learning_rate=0.01, exploration_rate=0.1, polarization_penalty=0.5):
        self.Q = defaultdict(lambda: np.zeros(3))  # Q[state][action]

    def score(self, content, agent):
        state = self.discretize_state(agent)
        action = self.select_action(state)
        # Map action to content ideological distance
        ...

    def update(self, agent, content, engagement_signal, polarization_delta):
        reward = engagement_signal - self.polarization_penalty * max(0, polarization_delta)
        # Q-learning update
        ...
```

### Step 8.4 — Multi-platform metrics

Add to `metrics.py`:
- `cross_platform_opinion_divergence`: mean opinion difference between same agents across platforms
- Per-platform assortativity (run existing metric per Platform object)
- Platform-specific misinformation prevalence

### Validation
- [ ] With `phi=0.0`, platforms develop independent opinion distributions
- [ ] With `phi=1.0`, platforms converge to similar opinion distributions
- [ ] RL recommender's Q-table shows clear preference patterns after 500+ ticks
- [ ] Multi-platform run with M=2 produces same results as two independent single-platform runs
  when `phi=0.0` (correctness check)

---

## Appendix A — Intervention Experiment Reference

Use these grids when running Phase 0 experiment sweeps. Run each with `n_runs=10`, `N=200`, `T=500`.

| Experiment | Variable | Values |
|---|---|---|
| Baseline | — | reference config |
| Personalization | `alpha` | 0.0, 0.3, 0.65, 1.0 |
| Diversity injection | `diversity_ratio` | 0.0, 0.1, 0.2, 0.3 |
| Misinfo downranking | `lambda_penalty` | 0.0, 0.25, 0.5, 1.0 |
| Virality dampening | `virality_dampening` | 0.0, 0.3, 0.6 |
| Media literacy | `media_literacy_boost` | 0.0, 0.2, 0.4 |
| Bot removal | `p_detect_remove` | 0.0, 0.3, 0.6 |
| Bot rate limiting | `rate_limit_factor` | 0.0, 0.5 |
| Rewiring rate | `dynamic_rewire_rate` | 0.0, 0.2, 0.5 |
| Homophily threshold | `homophily_threshold` | 0.3, 0.5 |
| Reinforcement factor | `reinforcement_factor` | 0.0, 0.2, 0.5 |

**Design principle:** Change ONE variable at a time vs. baseline. Always include baseline
as a control condition in every batch.

**Key tradeoffs to watch:**
- Diversity injection may ↓ misinformation but ↑ polarization (hostile exposure effect)
- Bot removal may ↓ misinfo but ↑ network sparsity (graph fragmentation)
- High media literacy boost may ↓ misinfo prevalence but not affect assortativity

---

## Appendix B — Parameter Defaults Quick Reference

All defaults from the canonical agent reference document. Do not change these without justification.

```python
# Network
N = 1000                  # agents (use 200 for demos, 1000 for experiments)
avg_degree = 15
rewire_prob = 0.1          # Watts-Strogatz p
homophily_threshold = 0.3
dynamic_rewire_rate = 0.01

# Agent
stubbornness = U(0.1, 0.3)
susceptibility = N(0.5, 0.1)
trust = N(0.5, 0.2)
expertise = N(0.5, 0.2)
activity_rate = N(0.5, 0.15)
media_literacy = U(0.2, 0.8)
confidence_bound = 0.3    # HK agents only
contrarian_prob = 0.3     # Contrarian agents only

# Recommender
alpha = 0.65              # personalization strength
k_exp = 20                # feed size
popularity_bias = 0.2      # β_pop: virality boost multiplier in scoring formula (ref doc Part 4)
                           # M4 (Option B): β_pop is a DISTINCT parameter from alpha.
                           # Scoring formula: score = alpha * ideological_sim
                           #                        + (1 - alpha) * beta_pop * log1p(virality * 9)
                           #                        (log1p(v*9) maps [0,1] → [0, ln10] ≈ [0, 2.30])
                           # WARNING: If β_pop is large, viral content dominates and the
                           # recommender collapses into winner-takes-all dynamics. Mitigations
                           # applied by default:
                           #   1. log(virality) compression — log1p(v*9) is used, NOT raw virality.
                           #      This prevents a v=0.99 item from drowning out v=0.50 items.
                           #   2. β_pop is capped at 1.0 in config validation (assert β_pop <= 1.0).
                           #   3. The full score is re-normalized to [0,1] before ranking.
                           # If you remove these mitigations in experiments, expect runaway
                           # virality cascades. Log this as a known model risk.

# SIR
sir_beta = 0.3            # transmission rate (calibrate to empirical data in research use)
sir_gamma = 0.05          # recovery rate

# Emotion
emotional_decay = 0.85    # lambda
arousal_share_weight = 0.3   # w_e
valence_share_weight = 0.4   # w_v
arousal_tolerance_effect = 0.4  # gamma_e
base_share_logodds = -1.5    # ~18% base sharing rate

# Temporal
T = 720                   # ticks (720 ≈ 30 days at 1 tick/hr)
snapshot_interval = 6
T_detect = 24             # bot detection interval

# Bot detection
s_thresh = 0.7            # suspicion score threshold for flagging
```

---

## Appendix C — Visualization Notes

**Demo mode (N≤200):** Live WebSocket streaming, real-time node color updates,
force-directed layout frozen after tick 0, metric charts update every tick.

**Experiment mode (N=1000):** Run completes fully, results replayed in frontend.
Graph renders final state only. Metric charts show full time series with std bands.

**Node color encoding:** Linear interpolation from red (opinion=−1) to white (opinion=0)
to blue (opinion=+1). Use HSL: `hsl(${opinion > 0 ? 220 : 0}, 80%, ${50 + (1-abs(opinion))*40}%)`.

**Network graph library:** `react-force-graph-2d` (Canvas-based, not SVG).
Do not use SVG-based force graph libraries — they degrade past N=100.
Cap live rendering at N=200. Show static snapshot for N>200.

**Bot detection overlay:** When bot detection is active, add a pulsing red ring around
flagged nodes (suspicion_score >= s_thresh). Implement as a custom node canvas renderer.