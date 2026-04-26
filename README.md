# GPUBid

Agentic GPU auction prototype for OIT 277 (Stanford GSB).

Buyer agents represent AI/ML jobs. Seller agents represent GPU capacity slots. They negotiate over rounds via structured offer tuples plus free-form reasoning. A central engine clears agreed deals. A VCG mixed-integer program and a static posted-price serve as benchmarks.

## Quick start (local)

```bash
pip install -e ".[dev]"
pytest                          # 60 tests, ~2s
PYTHONPATH=src python -m gpubid.experiments.run_sweep --seeds 15   # generate the four figures
jupyter lab notebooks/demo.ipynb
```

## Quick start (Colab)

Open `notebooks/demo.ipynb` in Colab and run all cells.

The first cell installs the package and enables widgets. In the **Mode** cell you can:
- Pick **fast** (deterministic agents, no key) — instant.
- Pick **preset** to play back a baked LLM trace (no key, no API cost).
- Pick **live** and paste an Anthropic OR OpenAI API key — the provider is auto-detected.

## Layout

```
src/gpubid/
  schema.py         — Pydantic models with rich HTML reprs
  market.py         — synthetic market generator (tight + slack regimes)
  llm.py            — provider-agnostic client (auto-detect by API key prefix)
  agents/           — buyer / seller / deterministic agents + versioned prompts
  engine/           — round runner, public board, clearing, concentration cap, reserve guard
  benchmark/        — VCG (PuLP+CBC) and posted-price baselines
  eval/             — surplus, recovery, Gini, off-peak utilization
  viz/              — render functions returning HTML strings or Plotly/Matplotlib figures
  experiments/      — preset baking + headline experiment sweeps
data/
  scenarios/        — seeded synthetic markets (JSON)
  presets/          — pre-baked LLM negotiation traces (JSON)
  figures/          — rendered plots used in the one-pager and notebook cell 7
notebooks/
  demo.ipynb        — the canonical artifact
tests/              — pytest, mostly the deterministic mechanism + VCG
```

## Modes

- **Fast** — deterministic agents, no API key required. Drives the tradeoff sandbox so it stays reactive.
- **Preset** — load a JSON trace from `data/presets/`, animate playback. No key required. Used in the video walkthrough.
- **Live** — agents call the LLM with the visitor's pasted key. Costs a few cents per run.

## Baking presets (one-time, requires keys)

Run once with the team's API keys; the resulting JSONs ship with the repo so the demo plays back without any LLM calls during a class showcase.

```bash
ANTHROPIC_API_KEY=sk-ant-…  OPENAI_API_KEY=sk-…  PYTHONPATH=src \
  python -m gpubid.experiments.bake_presets all
```

The six configured presets are listed in `src/gpubid/experiments/bake_presets.py:PRESET_SPECS`. The `heterogeneous_mix` preset uses both keys — buyer agents on Anthropic, seller agents on OpenAI — to demonstrate the welfare-vs-collusion tradeoff.

## Headline figures

Four figures land in `data/figures/`:
- `welfare_vs_rounds.png` — recovery vs N rounds (diminishing returns; N=5 sits on the elbow).
- `welfare_vs_cap.png` — fairness/Gini vs welfare as the concentration cap tightens.
- `agentic_vs_posted.png` — boxplot across seeds in tight + slack regimes.
- `offpeak_utilization.png` — off-peak filling, agentic vs posted-price.

Regenerate any time:

```bash
PYTHONPATH=src python -m gpubid.experiments.run_sweep --seeds 30
```

## Plan

Implementation plan: `/Users/meghna/.claude/plans/jolly-hopping-sonnet.md`.

## Submission deliverables (M6 — team-side)

- 3-4 minute video: screen-record the notebook running in Colab. Walk through cells 1→4 (pick a market, animate, compare baselines), then cells 5–7 (sandbox toggles, deal inspection, the four figures). Voiceover hits the four tradeoffs.
- One-pager: problem, mechanism diagram, headline result (the bar chart from cell 4), four tradeoff bullets each with the number from the figures. QR code to the Colab notebook.
- Optional UI port (M7): if voted top, wrap `viz/*` render functions in a Gradio app on Hugging Face Spaces or a Next.js app on Vercel. Same render code, different shell.
