---
name: gpubid
description: Use when working on the GPUBid project (OIT 277 "Digital Platforms in the Age of AI" at Stanford GSB) — an agentic GPU marketplace prototype. Covers running the notebook locally or in Colab, baking LLM presets, modifying the synthetic market generator, tweaking agent strategies, the four design tradeoffs the professor cares about, and the provider-agnostic LLM client architecture. Trigger when the user mentions GPUBid, gpubid, the GPU marketplace project, OIT 277, or asks to ramp up on this codebase.
---

# GPUBid — Agentic GPU Marketplace

This skill is both a ramp-up doc for new teammates and the machine-readable context Claude uses when helping you work on the project. Read top to bottom on first encounter (~10 min); skim from the table of contents thereafter.

## TL;DR

GPU cloud markets force buyers into three rigid contract types: long-term reserved, on-demand, or interruptible spot. A team that needs 8 H100s in the next hour pays the same on-demand rate as a team willing to wait 12 hours, even though they impose very different claims on perishable capacity. The middle of the demand curve is unserved.

GPUBid is a working prototype of an alternative: buyer agents (representing AI/ML jobs) and seller agents (representing GPU capacity slots) negotiate over rounds via *structured offer tuples plus free-form reasoning*. A central engine clears agreed deals, enforces a per-buyer concentration cap, and validates against private reserves. Two non-agentic baselines — VCG-optimal welfare (mixed-integer program) and a static posted-price — bracket what the mechanism is achieving.

The deliverable is `notebooks/demo.ipynb`. Every cell produces a visual or interactive output.

## Three run modes

The notebook's "Mode" dropdown picks one of three backends. **The same code path drives all three** — only the agent backend swaps. This is the most common point of confusion.

### Fast (default, no API key required)
Deterministic rule-based agents in `src/gpubid/agents/deterministic.py`. Sellers post asks at reserve × markup and decay each round; buyers post bids at value × markdown and climb each round. No LLM calls. Instant. **This is why the notebook ran end-to-end without an API key** — fast mode is the default, and the agents are pure Python rules.

Use this when:
- Exploring the sliders / tradeoff sandbox.
- Iterating on the mechanism (clearing rules, concentration cap).
- Running the headline experiments in `experiments/run_sweep.py`.

### Preset (no API key required, but presets must exist)
Replays a previously baked LLM negotiation from `data/presets/*.json`. The animation looks identical to live mode but is reading from a JSON file. **No presets are committed to the repo by default — the team bakes them once with their keys, then they ship for everyone else.**

Use this when:
- Demoing real LLM behavior without API costs (during the video, the live class showcase).
- Showing the heterogeneity comparison (`heterogeneous_mix` preset uses Claude buyers + OpenAI sellers).

### Live (requires API key)
Real LLM agents call Anthropic or OpenAI on every round. The provider is auto-detected from the key prefix (`sk-ant-…` → Anthropic, `sk-…` → OpenAI). Costs roughly $0.10-$0.50 per market run depending on model and market size. Slower (~30-60s per run).

Use this when:
- Stress-testing prompts.
- Validating that the mechanism still recovers >85% of VCG when LLMs are in the loop.
- Generating new presets.

## How to run

### Local

```bash
pip install -e ".[dev]"
pytest                                                          # 60 tests, ~2s
PYTHONPATH=src python -m gpubid.experiments.run_sweep --seeds 15 # regen the 4 figures
jupyter lab notebooks/demo.ipynb
```

### Colab — three paths

The notebook's setup cell tries each in order and uses whichever finds the package.

**Path A — Open directly from GitHub (slickest):**
```
https://colab.research.google.com/github/meghna810/GPUBid/blob/main/notebooks/demo.ipynb
```
Auto-imports any notebook from a public GitHub repo. No upload, no zip. Bookmark this URL.

**Path B — Git clone in Colab:**
```python
!git clone -q https://github.com/meghna810/GPUBid.git
import sys; sys.path.insert(0, '/content/GPUBid/src')
!pip install -q pydantic numpy pulp plotly matplotlib ipywidgets anthropic openai tenacity
import gpubid
```

**Path C — Zip upload (no GitHub needed):**
```bash
# on your Mac:
cd /Users/meghna/Desktop && zip -r GPUBid.zip GPUBid
```
Drag `GPUBid.zip` into the Colab Files panel, then in a notebook cell:
```python
!unzip -q -o GPUBid.zip
```
The setup cell finds `/content/GPUBid` and proceeds.

**Path D — Google Drive:**
Place the `GPUBid` folder in your Drive, then in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
```
The setup cell finds `/content/drive/MyDrive/GPUBid`.

## Provider-agnostic LLM client

Live mode supports either Anthropic or OpenAI from the same code. The visitor pastes a key, `gpubid.llm.make_client` detects the provider from the prefix, and instantiates the right adapter:

```python
def detect_provider(api_key: str) -> str:
    if api_key.startswith("sk-ant-"): return "anthropic"
    if api_key.startswith("sk-"):     return "openai"
    raise ProviderUnknownError(...)
```

Both adapters (`AnthropicClient`, `OpenAIClient` in `src/gpubid/llm.py`) implement the same `LLMClient` Protocol. Buyer and seller agents (`agents/buyer.py`, `agents/seller.py`) call `client.generate(...)` and never branch on provider — the adapters translate to and from a single internal `ToolCall` shape.

Default models (cheap, set in `llm.py`):
- Anthropic: `claude-haiku-4-5-20251001`
- OpenAI: `gpt-4o-mini`

## Bake presets (one-time)

```bash
ANTHROPIC_API_KEY=sk-ant-...  OPENAI_API_KEY=sk-...  PYTHONPATH=src \
  python -m gpubid.experiments.bake_presets all
```

This costs ~$5-10 of API credits total and produces six JSON files in `data/presets/`. Configured presets live in `src/gpubid/experiments/bake_presets.py:PRESET_SPECS`:

| scenario_id | description |
|---|---|
| tight_h100_rush | Tight H100 supply, urgent buyers compete |
| slack_offpeak | Slack supply, buyers fill off-peak slots |
| homogeneous_collusion | All-Claude sellers — watch prices drift up |
| heterogeneous_mix | Claude buyers vs OpenAI sellers — symmetry broken (requires both keys) |
| fairness_cap_bites | Concentration cap blocks one big buyer |
| free_form_breakdown | Reduced-structure mode (M5 toggle showcase) |

Presets are committed to the repo so the demo plays back without API calls during a class showcase.

## Headline figures

```bash
PYTHONPATH=src python -m gpubid.experiments.run_sweep --seeds 30
```

Saves four PNGs into `data/figures/`:
1. `welfare_vs_rounds.png` — recovery vs N rounds (curve flattens at 5).
2. `welfare_vs_cap.png` — fairness/Gini vs welfare as cap tightens.
3. `agentic_vs_posted.png` — boxplot across seeds in tight + slack regimes.
4. `offpeak_utilization.png` — off-peak slot filling, agentic vs posted.

These power the one-pager. Regenerate any time.

## The four design tradeoffs (the heart of the demo)

The professor explicitly wants the showcase to surface *A vs. B* decisions with evidence. Each is a toggle, a preset, or a figure.

### 1. Structure vs autonomy

**Choice:** structured offer tuples (tool calls) + free-form `reasoning` strings.

**Why:** Pure free-text chat collapses into looping, drift, JSON-validity failures, and incoherent acceptances. Sealed-bid + VCG eliminates the reason to use agents at all. The middle preserves strategic reasoning while bounding the action space.

**Evidence in the demo:** the `free_form_breakdown` preset shows the failure mode of pure chat. Compare its `n_deals` and reserve-violation count to a structured-mode preset.

### 2. Welfare vs collusion

**Choice:** heterogeneous seller backends — Claude buyers vs OpenAI sellers in `heterogeneous_mix`, all-Claude in `homogeneous_collusion`.

**Why:** When all sellers are the same model with the same system prompt, they tacitly converge on a higher markup (effectively colluding without explicit communication). Mixing model families breaks the symmetry that drives this drift.

**Evidence in the demo:** play `homogeneous_collusion` and `heterogeneous_mix` back-to-back; clearing prices in homogeneous run higher.

### 3. Expressiveness vs tractability

**Choice:** 6-dimensional bid language (price, qty, GPU type, time slot, duration, interruption tolerance), default 8 buyers × 4 sellers.

**Why:** More dimensions make the market feel realistic but slow the VCG MIP and make agent negotiation harder. We cap markets at 8×4 so VCG solves sub-second and a full LLM run finishes in under a minute of API time. Larger markets are stress tests, not the default.

**Evidence in the demo:** market sliders go up to 12×6; VCG solve time stays sub-second across the range.

### 4. Fairness vs revenue (a.k.a. concentration cap)

**Choice:** per-buyer concentration cap (default off in fast mode, configurable in the sandbox).

**Why:** Without a cap, large buyers can effectively bundle demand and capture a disproportionate share of capacity, displacing smaller buyers. With a cap, total welfare drops slightly but Gini drops materially. We accept marginal welfare loss for a more defensible allocation.

**Evidence in the demo:** the tradeoff sandbox (cell 5) cap slider. Tighten to 5-10% on tight-supply seeds where it visibly bites — Gini moves, welfare moves slightly, sometimes one buyer gets blocked entirely.

### Two engineering decisions worth showing

These aren't in the original proposal but the professor wants honesty about real implementation choices:

**5. Round depth N=5.** Welfare-recovery vs N flattens at 5 (`welfare_vs_rounds.png`). Below 3, deals don't have time to converge; above 7, marginal gain isn't worth the API cost.

**6. Hard reserve guard outside agents.** The clearing engine validates every accept: a buyer cannot pay above their max value; a seller cannot accept below their slot's reserve. If an LLM agent produces a violating accept, we *reject* the deal and log the violation — we don't trust the agent to police itself. The violation rate is a metric we report.

## Notebook tour (8 sections, top to bottom)

1. **Setup** — package import; auto-detects local / Colab / Drive layouts.
2. **Mode + API key** — Mode dropdown (fast/preset/live), password field for the key, dropdown of available presets.
3. **Pick a market** — sliders for # buyers, # sellers, regime, seed, max rounds.
4. **Render market** — buyer/seller cards as HTML (private values shown to the notebook user, *not* to other agents).
5. **Watch the negotiation** — `animate_negotiation()` updates an `ipywidgets.Output` each round. Asks left, bids right, deals pane below flashes green for new agreements.
6. **Compare to baselines** — Plotly bar chart with hover tooltips: Agentic / VCG / Posted-price, plus a metric table.
7. **Tradeoff sandbox** — `@interact` on `cap_pct` and `max_rounds`; metrics update live.
8. **Inspect a deal** — dropdown of deals, click one to see the surplus split between buyer and seller, plus chat-bubble reasoning when LLM modes were used.
9. **Headline figures** — the four `data/figures/*.png` displayed inline.
10. **Writeup** — the one-pager copy.

## Where things live (component → file)

| Component | File |
|---|---|
| Pydantic models, GPU types, interruption levels | `src/gpubid/schema.py` |
| Synthetic market generator (tight/slack regimes, seeded) | `src/gpubid/market.py` |
| Provider-agnostic LLM client | `src/gpubid/llm.py` |
| Versioned system prompts + tool specs | `src/gpubid/agents/prompts.py` |
| Deterministic rule-based agents (fast mode) | `src/gpubid/agents/deterministic.py` |
| LLM-backed agents (live + preset) | `src/gpubid/agents/buyer.py`, `seller.py` |
| Round-loop generator + agent factories | `src/gpubid/engine/round_runner.py` |
| Public board state + redaction | `src/gpubid/engine/board.py` |
| Match offers, finalize, cap, reserve guard | `src/gpubid/engine/clearing.py` |
| VCG MIP solver (PuLP + CBC) | `src/gpubid/benchmark/vcg.py` |
| Posted-price baseline | `src/gpubid/benchmark/posted_price.py` |
| Surplus, Gini, recovery, off-peak utilization | `src/gpubid/eval/metrics.py` |
| HTML rendering for market cards, trading floor, trace | `src/gpubid/viz/market_view.py`, `trading_floor.py`, `trace_view.py` |
| Plotly + matplotlib figures | `src/gpubid/viz/charts.py`, `figures.py` |
| Preset baking (LLM → JSON) | `src/gpubid/experiments/bake_presets.py` |
| Headline experiments → 4 figures | `src/gpubid/experiments/run_sweep.py` |

## Common tasks

### Add a new preset scenario

Add a new `PresetSpec` to `PRESET_SPECS` in `src/gpubid/experiments/bake_presets.py`:

```python
PresetSpec("my_scenario", "What this shows", n_buyers=6, n_sellers=3, regime="tight", seed=99),
```

Then `python -m gpubid.experiments.bake_presets my_scenario`. The notebook's preset dropdown picks it up automatically.

### Tweak the deterministic agent's strategy

Both `DeterministicBuyer` and `DeterministicSeller` live in `src/gpubid/agents/deterministic.py`. The seller's markup factor (1.5x tight, 1.2x slack), the decay rate per round, and the buyer's markdown / climb rates are all in `decide()`. After tweaking, run the sweep to see how the strategy affects the welfare-recovery curve.

### Add a new GPU type

1. Add a value to `GPUType` in `schema.py`.
2. Add a base reserve to `GPU_BASE_RESERVE` in `market.py`.
3. Add color entries to `GPU_COLOR` and `GPU_BG` in `viz/market_view.py`.
4. Add the new value to the enum in the tool specs (`buyer_tool_specs`, `seller_tool_specs` in `agents/prompts.py`).

### Run a custom experiment

`src/gpubid/experiments/run_sweep.py` has four functions (`experiment_*`) you can call directly. Each takes a list of seeds and returns a dict of results. Plot with matplotlib or repurpose `viz/charts.py`.

### Tighten the round limit

`max_rounds` is a parameter on `run_rounds()` and `animate_negotiation()`. The notebook exposes it as a slider in cell 1 (default 5). For experiments, change `rounds_grid` in `run_sweep.py:experiment_welfare_vs_rounds`.

## Testing

```bash
pytest                  # all 60 tests, ~2s
pytest tests/test_engine.py -v       # mechanism + clearing
pytest tests/test_benchmark.py -v    # VCG against hand-computable cases
pytest tests/test_llm.py -v          # provider detection (no LLM calls)
```

The VCG tests are the most critical: they verify the MIP against 2×2 markets we can solve by hand (`_two_by_two_market_simple`, `_two_by_two_one_seller_only`, `_incompatible_market`). If those break, every "% of VCG recovered" number above is suspect.

## Submission deliverables

- **3-4 minute video**: screen-record the notebook running in Colab. Walk through cells 1→4 (pick a market, animate, compare baselines), then cells 5–7 (sandbox toggles, deal inspection, the four figures). Voiceover hits the four tradeoffs.
- **One-pager**: problem, mechanism diagram, headline result (the bar chart from cell 4 or `agentic_vs_posted.png`), four tradeoff bullets each with the number from the figures. QR code to the Colab notebook URL above.
- **Optional UI port (M7)**: if voted top, wrap `viz/*` render functions in a Gradio app on Hugging Face Spaces or a Next.js app on Vercel. Same render code, different shell. The package is set up to make that easy — every renderer returns an HTML string.

## Plan + memory

The full implementation plan: `/Users/meghna/.claude/plans/jolly-hopping-sonnet.md`.
Project / user memory: `/Users/meghna/.claude/projects/-Users-meghna-Desktop-GPUBid/memory/`.
