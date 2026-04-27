# Changelog

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
