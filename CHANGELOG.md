# Changelog

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
