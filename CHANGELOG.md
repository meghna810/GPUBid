# Changelog

## v0.4.4 — 2026-04-27 (latest)

The "Gemini sellers stop walking away on turn 1" fix.

### Schema sanitizer now handles JSONSchema type unions

The dialogue tool's `proposed_price_per_gpu_hr` parameter has type
`["number", "null"]` (price required for `counter`, null for `accept` /
`walk_away`). Gemini's tool schema validator rejects union types — `type`
must be a single string. The sanitizer in `gpubid.llm._sanitize_for_gemini`
didn't handle this, so every Gemini-backed seller agent crashed on turn 1
with a Pydantic validation error and walked away as the fallback action.

Fix: when `type` is a list, drop "null" from it, set the surviving type
as the canonical one, and add `nullable: true` (Gemini's preferred way
to express optional). Applied recursively through nested schemas.

Two new regression tests in `test_llm.py` so this can't sneak back:
- `test_gemini_sanitizer_rewrites_type_union_to_nullable`
- `test_dialogue_tool_schema_is_gemini_compatible`

### Stricter "no early walk-away" instructions

Even agents that aren't crashing were sometimes walking away on turn 1.
Both dialogue prompts now say explicitly: NEVER walk away on turns 1-3
(seller) or turns 1, 2, 4 (buyer's early turns). The agent must counter.
Walk-away is only available after turn 4-5 and only after at least 2
concessions from your own side.

### Bumped

- `__version__` → `0.4.4`.
- 165 tests passing, 2 skipped (added 2 new schema-sanitizer tests).

---

## v0.4.3 — 2026-04-27 (latest)

The "deals actually close" fix.

### Guaranteed buyer-seller satisfiability

Even with the strict matchmaker, the v0.4.1 chat market frequently closed
0 deals because most buyers had no structurally-compatible seller slot
to begin with — random market generation produced incompatible buyer/slot
pairs no amount of dialogue could resolve.

Fixed by `_ensure_buyer_satisfiability` in `market_v3.py`. After buyers
and sellers are generated, walk every buyer; if no slot satisfies them
on (GPU, qty, duration, time window), find the closest candidate slot
and **loosen the buyer's job** (lower qty, shorter duration, wider window,
add the slot's GPU type to acceptable list) until at least one slot does.
Buyer's `max_value_per_gpu_hr` and seller reserves stay untouched — only
structural prefs get relaxed, so the only thing the LLM dialogue still
has to negotiate is **price**.

Result: 8/8 buyers have at least one fully-compatible slot in every
seed tested (was 1/8 in worst cases). Every buyer also has positive
bargaining zone — `max_value > cheapest_compat_reserve` always.

### Close-biased dialogue prompts

`_SELLER_DIALOGUE_PROMPT` and `_BUYER_DIALOGUE_PROMPT` both now lead with:

> YOUR PRIMARY GOAL: CLOSE A DEAL.

Walk-away is reframed as a last resort, not a negotiating move. Concession
steps capped at 5-15% per turn so agents converge instead of holding firm.
The per-turn user message includes a closing-pressure ⚠ banner in the
last 2 turns.

### Tighter openings + longer threads

- Opening prices: seller `1.30 × reserve` (was 1.50), buyer `0.65 × max`
  (was 0.55) — narrower starting gap.
- `max_turns_per_dialogue` 6 → 8.
- `max_retries_per_buyer` 2 → 3.

### Bumped

- `__version__` → `0.4.3`.
- 163 tests passing, 2 skipped.

---

## v0.4.1 — 2026-04-27 (late)

The "agents actually negotiate" pass.

### Bilateral chat is now the primary market mechanism

Notebook cell 3 ("Watch the agents negotiate") used to drive the public-board
flow with a central clearing engine — agents posted offers to a shared board
and a clearer matched compatible bids/asks. That isn't really "negotiating",
it's bidding into an order book.

Now the headline mechanism is `gpubid.protocol.chat_market.run_chat_market`:

1. **Matchmaking** — for each buyer, list seller slots that are structurally
   compatible (GPU type, time window, qty, duration, tolerance), ranked by
   reserve cheapest-first.
2. **Sequential bilateral dialogues** — buyers ordered by urgency descending.
   Each buyer's LLM chats turn-by-turn with the seller's LLM (via the existing
   `gpubid.protocol.dialogue.run_bilateral_dialogue`) until one accepts,
   walks away, or hits the turn cap (default 6).
3. **Walk-away → retry** — when a buyer walks, they try the next compatible
   slot. Up to 2 retries per buyer.
4. **No central clearer** — only sanity validation that the closing price is
   inside both private reserves. Capacity scarcity (slot qty) is the only
   "market dynamic."

The board flow is preserved in `round_runner` for the deterministic / preset
paths and as the comparison baseline; it just isn't the headline anymore.

### Live streaming chat viz

`gpubid.viz.chat_stream.render_chat_thread` renders one bilateral dialogue
as an iMessage-style chat thread (buyer right, seller left, model badges on
every bubble showing provider/model). Each turn shows the agent's argument,
attached condition, action (`COUNTER @ $X.XX/hr` / `ACCEPT` / `WALK AWAY`),
and a "refs alt" pill when the agent cited an alternative.

The notebook cell streams threads as they complete via an `on_dialogue_complete`
callback into an `ipywidgets.Output` — you watch each conversation appear,
then move to the next pair.

### Auto-detect available providers

The negotiation cell now auto-detects whichever LLM keys are present
(Anthropic / OpenAI / Gemini, in any combination) and round-robins them
across buyers and sellers. If one provider is set, both sides use it; if
multiple, each side gets a mix so threads are cross-provider by default.
No hardcoded provider list — bring whichever keys you have.

### Wider buyer time windows

Bilateral matchmaking is sensitive to time-window overlap in a way the
board flow wasn't (the board could just have buyers hop to whichever ask
matched). Buyer `earliest_start` now in [0, 8] (was [0, 16]) and
`latest_offset` in [10, 16] hours after duration (was [4, 12]). Result:
6-8 of 8 buyers now have at least one structurally-compatible slot, up
from often 4-5.

### Plumbing

- `chat_run_to_snapshots` adapter: emits `RoundSnapshot`-shaped records
  from a `ChatMarketRun` so the existing forensics, persuasion, and chat-
  exchange views keep working unmodified.
- `__version__` → `0.4.1`.
- 163 tests passing, 2 skipped.

---

## v0.4.0 — 2026-04-27 (evening)

The "deals reliably close + Gemini + persuasion analytics" pass.

### Reliable deal closure

The synthetic market generator was producing too many structurally-infeasible
buyer/slot pairs (buyer needs 7h or 8 GPUs but no slot has either). Result:
runs frequently closed zero deals, which made the demo look broken.

- `market_v3._synth_profile_from_requirement` now caps buyer qty at 5 and
  duration at 6h, matching the tight-regime seller-slot ceiling.
- Buyer max-WTP markup raised from 1.4-1.8x to 1.7-2.4x of the most expensive
  acceptable GPU's reserve. The seller's opening ask (1.5x in tight) now
  always sits inside the buyer's bargaining zone.
- Buyer interruption tolerance biased toward INTERRUPTIBLE (50%) /
  CHECKPOINT (35%) / NONE (15%) — keeps tolerance mismatch from blocking
  otherwise compatible deals.

Across 5 tight-regime seeds the v3 deterministic path now closes 2-4 deals
per run; slack regime closes 4-7. Previously many seeds closed zero.

### Gemini provider support

`gpubid.llm` now auto-detects Gemini keys (`AIza…`) and dispatches to a new
`GeminiClient` adapter using the `google-genai` SDK. Default model
`gemini-2.5-flash`. Schema sanitization strips JSONSchema fields Gemini
rejects (`additionalProperties`, `$schema`, etc.). The factory and
`get_api_key_from_env` recognize `GEMINI_API_KEY` / `GOOGLE_API_KEY`.

The notebook's API-key cell loads `GEMINI_API_KEY` from Colab Secrets the
same way as Anthropic / OpenAI keys. Provider-mix tournaments can now run
across all three.

### Multi-provider tournament

`gpubid.analysis.tournament.head_to_head_multi` accepts any subset of
`{anthropic, openai, gemini}` (>=2) and runs round-robin assignment across
them. Per-provider model pinning is optional (`provider_models=`); defaults
pick the cheapest tier each provider offers.

`compute_baseline_comparison` + `render_baseline_comparison` add per-seed
agentic-vs-VCG-vs-posted-price welfare numbers right under the tournament
report — answers the "did the agentic mechanism actually do better than a
posted price?" question for every seed in the run.

### Persuasion + manipulation analytics (new module)

`gpubid.analysis.persuasion` adds:

- **Quantitative persuasion** (no LLM needed): for each agent, the average %
  the counterparty's posted price moved *after* this agent posted. Buyers
  pulling sellers DOWN score positive; sellers pulling buyers UP score
  positive. Uses the existing per-round price snapshots — zero extra cost.
- **Semantic style tags**: optional LLM judge tags each reasoning bubble as
  one of `bluff`, `false_urgency`, `emotional_appeal`, `anchor`, `concession`,
  `honest_argument`, `hedge`. Aggregates per-agent and per-provider so we can
  surface "Claude bluffs more than Gemini" or "OpenAI sellers anchor more
  aggressively". Costs roughly $0.05-0.10 per market run with cost-effective
  models; capped at 80 bubbles per run by default.
- HTML rendering: per-agent leaderboard, per-provider rollup, tag-mix
  Plotly chart, and an examples panel showing 1-2 utterances per tag.

New notebook cell **6.75 Persuasion + manipulation analytics** runs after
the tournament cell and uses the same negotiation history.

### HITL real-world use cases (new module)

`gpubid.analysis.hitl_usecases` documents seven scenarios where human review
demonstrably pays off in an agentic GPU marketplace:

1. Multi-week reservations >$50k commitment (financial)
2. Regulated workloads — HIPAA / FedRAMP / GDPR (regulatory)
3. Foundation-model training runs that can't restart (operational)
4. Counterparty showing manipulation signals (reputational)
5. Ambiguous CEO requirements — wide qty/duration band (operational)
6. Repeat low-confidence accepts (financial)
7. Cross-organization deals — different cost centers (financial)

Each card lists a practical threshold and the auto-detectable signal that
should escalate. `detect_alerts_from_persuasion` ties this into the
persuasion analytics — agents whose counterparty was tagged with >=2
manipulation tags get flagged for review.

New notebook cell **6.95 Where HITL pays off — real-world use cases** runs
after the existing 6.9 stub and surfaces both the static guidance and any
live alerts produced from this run.

### Plumbing

- `pyproject.toml`: added `google-genai>=0.3` dependency.
- Setup cell pip-install line includes `google-genai`.
- `__version__` → `0.4.0`.
- Tournament gains `provider_models` field on `TournamentResult` so the
  baseline rendering can show which model was on each side.

163 tests passing, 2 skipped (existing live-LLM tests).

---

## v0.3.1 — 2026-04-27 (later same day)

The "make negotiations interesting" pass.

### Live mode is now the default; fast mode dropped from the UI

Per spec §0.2.1: the deterministic "fast" option is gone from the notebook's
mode dropdown. v0.3 is LLM-only end-to-end. The dropdown now offers:
- **`live`** (default when an API key is loaded) — runs a fresh LLM negotiation now.
- **`preset`** — replays a baked LLM trace from `data/presets/*.json`. No API cost.

For the 3-4 min video and the live class demo: bake presets once, then use
`preset` mode. The deterministic agents stay in code (used by the tournament
smoke harness and by tests via recorded fixtures); they're just not in the
notebook UX.

### Buyer agents translate CEO/CTO requirements (Phase 3 wired into the notebook)

`gpubid.market_v3.generate_market_v3()` samples a CEO requirement per buyer
from `REQUIREMENT_LIBRARY` and runs `BuyerAgent.translate()` (cached in-memory
by `(provider, model, requirement_id, prompt_version)`) when an LLM client is
provided. Falls back to a synthetic translator when no key is available.

The notebook's market render cell now shows a "💬 From the CEO/CTO" panel
under each buyer card with the original NL brief plus what the agent
translated it into.

### Sellers carry volume-discount tiers — offers are non-directly-comparable

35% flat-priced, 65% tiered (1-3 tiers, 5-25% discounts at varying thresholds).
Tiers feed into the bilateral dialogue prompts (agents can argue about volume
commits), the tiered VCG benchmark, and the seller card render.

### New cells

- **6.8 Post-deal regret signals (exploratory)** — heuristic-based, no LLM.
  Per-deal regret table + calibration trigger when avg seller regret > 0.4.
- **6.9 HITL trigger demo** — stub showing what events would surface in
  production (ambiguous requirements, low-confidence closes). Uses the
  headless auto-proceed surfacer.

### Plumbing

- `agent_models_map()` builds the agent_id -> (provider, model) lookup; chat
  exchange and dialogue views consume it.
- Bilateral dialogue prompts now receive `seller_volume_policy` and
  `buyer_business_context`.

161 tests passing, 2 skipped.

---

## v0.3.0 — 2026-04-27

The first refactor toward two-sided information asymmetry as a first-class
concept. Implemented per `gpubid_v0.3_refactor_spec.md`.

### Shipped (LLM-independent, fully tested)

- **Phase 1**: centralized `gpubid.config.settings` (Pydantic BaseSettings,
  six grouped configs, env-var overrides) and `gpubid.errors` taxonomy
  (`GPUBidError` → `MarketError`, `ProtocolError`, `ProviderError`, `HITLAbort`).
- **Phase 2**: `gpubid.domain.profiles` with `BuyerPublicProfile`,
  `BuyerPrivateProfile`, `SellerPublicProfile`, `SellerPrivateProfile`,
  `VolumeDiscountPolicy`/`VolumeDiscountTier`, `TimeWindow`, `InventorySlot`,
  `FallbackOption`, `BuyerV2`, `SellerV2`. All frozen Pydantic v2.
  Backward-compat shims (`.id`, `.label`, `.gpu_type`) on `BuyerV2`/`SellerV2`
  so existing viz code keeps working.
- **Phase 4**: `gpubid.domain.offers.OfferTerms` with optional
  `discount_schedule`. Canonical `effective_price_per_gpu_hr(offer, qty,
  duration)` and `total_value_usd(offer, qty, duration)` — these MUST be used
  everywhere (viz, benchmarks, metrics). Do not re-derive discount math.
- **Phase 6**: `gpubid.protocol.budget.BudgetPolicy` (layered round count +
  token cap + no-progress streak) with `StopReason` enum. Reports the cause
  of a halt so forensics can show why a run ended.
- **Phase 10**: `gpubid.benchmark.vcg_v2.solve_vcg_tiered` — linearizes each
  seller's volume-discount tiers into bundles, solves an MIP for welfare-
  optimal assignment under shared slot capacity. Accepts both legacy v0.2
  `Market` and v0.3 `(buyers_v2, sellers_v2)` tuples. Legacy `solve_vcg`
  preserved for backward compat per spec §11.3.

### Scaffolded — needs your API keys + recorded LLM fixtures

These ship with the API surface, the data shape, the docstrings, and tests
that pass on data-model invariants. The runtime path raises
`NotImplementedError` until LLM fixtures are recorded.

- **Phase 3**: `gpubid.domain.requirements` — 12-entry CEO-style requirement
  library is fully implemented (no LLM). `gpubid.agents.buyer_agent.BuyerAgent`
  has the full translate-step logic; needs a fixture-replayed test.
- **Phase 5**: `gpubid.protocol.broadcast` and `eligibility` are implemented;
  `round.py:run_negotiation` is scaffolded.
- **Phase 7**: existing `gpubid.llm.make_client` already auto-detects
  provider from key prefix; `make_llm_agents_assigned` already supports
  per-role provider routing per spec §8.2.
- **Phase 8**: `gpubid.agents.prompts` re-exports the v0.2 prompt API and
  adds a `render_prompt(role, variant, ctx)` template renderer that raises
  `ConfigError` until the 8 variant `.md` files are authored.
- **Phase 9**: `gpubid.protocol.hitl` data model + `auto_proceed_surfacer`
  for headless tests. Notebook widget surfacer is TBD.
- **Phase 11**: `gpubid.analysis.comparator.ComparatorCell` defined; the
  metric aggregation depends on Phase 12 outputs.
- **Phase 12**: `gpubid.experiments.sim_v2.SimSpec` + `run_simulation`
  scaffolded. `load_all_runs` returns an empty DataFrame when no runs exist.
- **Phase 13**: `gpubid.analysis.regret` is fully implemented (heuristic,
  no LLM); the before/after demo depends on Phase 12 outputs.

### Removed (not yet)

Per spec §0.2.1, fast mode is supposed to disappear in v0.3. We **kept it**
for this cut so the existing demo doesn't brick for users without recorded
fixtures. Once you've recorded LLM fixtures with your keys (see
`tests/fixtures/llm_fixture_client.py` for the recording instructions),
removing fast mode is a one-commit follow-up.

### Bumped

- `__version__` → `0.3.0`.
- `pydantic-settings` added as a dependency.
- 151 tests now passing, 2 skipped (live LLM tests skip without keys, exactly
  as designed).
