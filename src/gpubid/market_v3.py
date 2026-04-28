"""v0.3 market generation: NL CEO requirements -> translate -> structured profiles,
plus volume-discount-aware sellers.

Two-tier output:
  1. ``Market`` (v0.2-compatible) — what the existing engine consumes.
  2. ``V3Enrichment`` — side-channel data for v0.3 features:
       - Per-buyer CEO requirement (raw_text, persona, requirement_id)
       - Per-buyer public+private profile pair (from translate, when LLM available)
       - Per-seller volume-discount policy (tiered pricing)

Existing notebook code that just wants to run a negotiation uses ``Market``
unchanged. New cells (CEO requirement panel, tiered VCG, dialogue prompts that
mention volume commits) read from ``V3Enrichment``.

Translate-cache: in-memory dict keyed by ``(provider, model, requirement_id, prompt_version)``.
Re-running the same market in the same Python process pays zero LLM cost on
the second call. Persistent disk cache is a follow-up (spec §13.5).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from gpubid.agents.buyer_agent import BuyerAgent
from gpubid.agents.prompts import PROMPT_VERSION
from gpubid.domain.profiles import (
    BuyerPrivateProfile,
    BuyerPublicProfile,
    InventorySlot,
    SellerPrivateProfile,
    SellerPublicProfile,
    SellerV2,
    TimeWindow,
    VolumeDiscountPolicy,
    VolumeDiscountTier,
)
from gpubid.domain.requirements import EndRequirement, sample_requirements
from gpubid.llm import LLMClient
from gpubid.market import (
    BUYER_NAMES,
    GPU_BASE_RESERVE,
    SELLER_NAMES,
    _generate_sellers as _generate_v02_sellers,
    generate_market as _generate_market_v02,
)
from gpubid.schema import (
    Buyer,
    CapacitySlot,
    GPUType,
    InterruptionTolerance,
    Job,
    Market,
    Seller,
)


# ---------------------------------------------------------------------------
# Translate cache (in-memory; survives only within one Python process)
# ---------------------------------------------------------------------------

_TRANSLATE_CACHE: dict[str, tuple[BuyerPublicProfile, BuyerPrivateProfile]] = {}


def _cache_key(client: LLMClient, requirement: EndRequirement) -> str:
    blob = (
        f"{getattr(client, 'provider', '?')}|"
        f"{getattr(client, 'model', '?')}|"
        f"{requirement.requirement_id}|"
        f"{PROMPT_VERSION}"
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def clear_translate_cache() -> None:
    """Wipe the in-memory translate cache. Useful between experiment runs."""
    _TRANSLATE_CACHE.clear()


# ---------------------------------------------------------------------------
# Enrichment carrier
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class V3Enrichment:
    """Side-channel data layered on a v0.2 Market for v0.3 features.

    All fields default to empty so a v0.2-only Market still gets a well-formed
    enrichment object.
    """

    buyer_requirements: dict[str, EndRequirement] = field(default_factory=dict)
    buyer_public_profiles: dict[str, BuyerPublicProfile] = field(default_factory=dict)
    buyer_private_profiles: dict[str, BuyerPrivateProfile] = field(default_factory=dict)
    seller_volume_policies: dict[str, VolumeDiscountPolicy] = field(default_factory=dict)
    seller_v3: dict[str, SellerV2] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Volume discount policy generation
# ---------------------------------------------------------------------------


def _generate_volume_discount_policy(rng: np.random.Generator) -> VolumeDiscountPolicy:
    """Mix of flat-priced and tiered sellers. Tiers vary so offers are non-comparable.

    Distribution:
      - 35% flat (no discount)
      - 65% have 1-3 tiers, with first-tier thresholds in {2, 3, 4} GPUs and
        durations in {3, 4, 5} hours, discounts ramping 5%-25%.
    """
    if rng.random() < 0.35:
        return VolumeDiscountPolicy()

    n_tiers = int(rng.integers(1, 4))
    tiers: list[VolumeDiscountTier] = []

    qty = int(rng.integers(2, 5))
    dur = float(rng.choice([3.0, 4.0, 5.0]))
    disc = 0.05 + 0.05 * float(rng.random())  # first tier 5-10%

    for _ in range(n_tiers):
        tiers.append(VolumeDiscountTier(
            min_qty_gpus=qty,
            min_duration_hours=dur,
            discount_pct=round(disc, 3),
        ))
        qty += int(rng.integers(2, 5))
        dur += float(rng.integers(2, 5))
        disc += 0.04 + 0.06 * float(rng.random())
        if disc > 0.30:
            disc = 0.30

    return VolumeDiscountPolicy(
        tiers=tuple(tiers),
        is_negotiable=bool(rng.random() > 0.5),
    )


def _build_seller_v3(seller: Seller, policy: VolumeDiscountPolicy) -> SellerV2:
    """Wrap a v0.2 Seller as a v0.3 SellerV2 with the given volume discount policy."""
    inv: list[InventorySlot] = []
    reserve_per_slot: dict[str, float] = {}
    for slot in seller.capacity_slots:
        inv.append(InventorySlot(
            slot_id=slot.id,
            gpu_type=slot.gpu_type,
            qty_gpus=slot.qty,
            start_slot=slot.start,
            duration_hours=float(slot.duration),
        ))
        reserve_per_slot[slot.id] = slot.reserve_per_gpu_hr

    list_prices: dict[GPUType, float] = {}
    for slot in seller.capacity_slots:
        # Synthetic v0.2 sellers don't carry an explicit list price; use 1.5x reserve.
        if slot.gpu_type not in list_prices:
            list_prices[slot.gpu_type] = round(slot.reserve_per_gpu_hr * 1.5, 2)

    public = SellerPublicProfile(
        seller_id=seller.id,
        display_name=seller.label,
        inventory_slots=tuple(inv),
        list_price_per_gpu_hr=list_prices,
        volume_discount_policy=policy,
        min_commitment_hours=2.0,
    )
    private = SellerPrivateProfile(
        reserve_per_slot=reserve_per_slot,
        marginal_cost_per_gpu_hr={g: r * 0.7 for g, r in list_prices.items()},
        target_utilization=0.75,
        competing_demand_signal="medium",
    )
    return SellerV2(public=public, private=private)


# ---------------------------------------------------------------------------
# Buyer translation (with cache)
# ---------------------------------------------------------------------------


def _profiles_to_v02_buyer(
    pub: BuyerPublicProfile,
    priv: BuyerPrivateProfile,
    urgency_score: float,
    fallback_label_idx: int,
) -> Buyer:
    """Convert a v0.3 (public, private) profile pair into a v0.2 Buyer for the engine.

    Maps semantic interruption tolerance to the legacy enum:
      none -> NONE; checkpoint_15min/checkpoint_60min -> CHECKPOINT; any -> INTERRUPTIBLE.
    """
    tol_map = {
        "none": InterruptionTolerance.NONE,
        "checkpoint_15min": InterruptionTolerance.CHECKPOINT,
        "checkpoint_60min": InterruptionTolerance.CHECKPOINT,
        "any": InterruptionTolerance.INTERRUPTIBLE,
    }
    job = Job(
        qty=pub.qty_gpus,
        acceptable_gpus=pub.gpu_type_preferences,
        earliest_start=pub.time_window.earliest_start_slot,
        latest_finish=pub.time_window.latest_finish_slot,
        duration=int(round(pub.duration_hours)),
        interruption_tolerance=tol_map.get(pub.interruption_tolerance, InterruptionTolerance.NONE),
        max_value_per_gpu_hr=priv.max_willingness_to_pay,
    )
    return Buyer(
        id=pub.buyer_id,
        label=pub.display_name or BUYER_NAMES[fallback_label_idx % len(BUYER_NAMES)],
        job=job,
        urgency=urgency_score,
    )


def _translate_with_cache(
    agent: BuyerAgent,
    requirement: EndRequirement,
    rng: np.random.Generator,
) -> tuple[BuyerPublicProfile, BuyerPrivateProfile]:
    key = _cache_key(agent.llm_client, requirement)
    if key in _TRANSLATE_CACHE:
        return _TRANSLATE_CACHE[key]
    pub, priv = agent.translate(requirement, rng)
    _TRANSLATE_CACHE[key] = (pub, priv)
    return pub, priv


# ---------------------------------------------------------------------------
# Synthetic-fallback buyer (when no LLM client provided)
# ---------------------------------------------------------------------------


def _synth_profile_from_requirement(
    requirement: EndRequirement,
    buyer_idx: int,
    rng: np.random.Generator,
) -> tuple[BuyerPublicProfile, BuyerPrivateProfile]:
    """Deterministic fallback when no LLM client is available.

    Uses the requirement's expected_qty_range / expected_duration_range /
    expected_urgency_band as the structured-profile values directly. This is
    what the LLM would land on for an obvious requirement; not a substitute for
    real translate but lets the demo run without keys.
    """
    # Cap qty + duration to what the synthetic seller slot generator actually
    # produces — otherwise most buyer/slot pairs are structurally infeasible
    # and the demo never closes a deal.
    # Seller slots in tight regime: qty 2..5, duration 3..6; slack: qty 3..8,
    # duration 4..8. Use the tight-regime caps as the safe upper bound.
    qty = int(rng.integers(requirement.expected_qty_range[0], requirement.expected_qty_range[1] + 1))
    qty = max(1, min(qty, 5))
    duration = float(rng.uniform(*requirement.expected_duration_range))
    duration = max(2.0, min(duration, 6.0))

    # GPU preferences: derive from workload category
    gpu_prefs_by_category: dict[str, tuple[GPUType, ...]] = {
        "training":           (GPUType.H100, GPUType.A100),
        "fine_tuning":        (GPUType.H100, GPUType.A100, GPUType.L40S),
        "inference_realtime": (GPUType.H100, GPUType.A100),
        "inference_batch":    (GPUType.A100, GPUType.L40S),
        "evaluation_sweep":   (GPUType.A100, GPUType.L40S),
    }
    acceptable = gpu_prefs_by_category.get(
        requirement.workload_category, (GPUType.A100, GPUType.L40S)
    )

    # Time window: wide enough that some seller slot will plausibly fit.
    # See note in market.py about the chat-market mechanism's sensitivity to
    # this — narrow windows cause structural mismatches that no amount of
    # negotiation can fix.
    earliest = int(rng.integers(0, 9))
    latest = min(24, earliest + int(round(duration)) + int(rng.integers(10, 16)))

    # Most workloads in the demo prefer permissive tolerance so they don't
    # bounce off seller offers on a tolerance-mismatch alone. Real-time
    # inference still demands `none` because that's the realistic ask.
    interruption_map = {
        "training":           "checkpoint_60min",
        "fine_tuning":        "checkpoint_60min",
        "inference_realtime": "none",
        "inference_batch":    "any",
        "evaluation_sweep":   "any",
    }
    urgency_to_score = {"routine": 0.2, "soon": 0.5, "urgent": 0.85}
    urgency_score = urgency_to_score[requirement.expected_urgency_band] + 0.1 * float(rng.random())

    # Max WTP scales with the most expensive acceptable GPU. Pick a markup
    # band that clearly exceeds the seller opening markup (1.5x in tight,
    # 1.2x in slack) so the bargaining zone is positive — otherwise even a
    # well-behaved LLM run can fail to close any deals.
    most_expensive_reserve = max(GPU_BASE_RESERVE[g] for g in acceptable)
    markup = 1.75 + 0.65 * float(rng.random())  # 1.75x..2.4x
    max_wtp = round(most_expensive_reserve * markup, 2)

    public = BuyerPublicProfile(
        buyer_id=f"B{buyer_idx}",
        display_name=requirement.persona.split(",")[0].split(" of ")[-1].strip().title()[:32]
                     or BUYER_NAMES[buyer_idx % len(BUYER_NAMES)],
        workload_category=requirement.workload_category,
        gpu_type_preferences=acceptable,
        qty_gpus=qty,
        duration_hours=duration,
        time_window=TimeWindow(earliest_start_slot=earliest, latest_finish_slot=latest),
        interruption_tolerance=interruption_map.get(
            requirement.workload_category, "checkpoint_60min"
        ),
        urgency_band=requirement.expected_urgency_band,
    )
    private = BuyerPrivateProfile(
        max_willingness_to_pay=max_wtp,
        urgency_score=min(0.99, urgency_score),
        budget_remaining_usd=max_wtp * qty * duration * 1.5,
        business_context_summary=requirement.raw_text,
    )
    return public, private


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def generate_market_v3(
    n_buyers: int = 8,
    n_sellers: int = 4,
    regime: str = "tight",
    seed: int = 42,
    *,
    llm_client: Optional[LLMClient] = None,
) -> tuple[Market, V3Enrichment]:
    """Generate a v0.2-compatible Market plus v0.3 enrichment.

    Buyers come from the CEO-requirement library:
      - With ``llm_client`` set: the buyer agent translates each NL requirement
        into a structured (public, private) profile via the LLM. Cached by
        ``(provider, model, requirement_id, prompt_version)``.
      - Without: a deterministic ``_synth_profile_from_requirement`` derives
        plausible profile fields from the requirement's expected ranges.

    Sellers always get volume-discount policies (random mix of flat-priced and
    tiered). The v0.2 Seller objects (passed to the existing engine) carry only
    capacity+reserve; the tier policy is in the enrichment dict.
    """
    if regime not in ("tight", "slack"):
        raise ValueError(f"regime must be 'tight' or 'slack', got {regime!r}")

    rng = np.random.default_rng(seed)

    # Sellers first (synthetic + tier policies)
    sellers_v02 = _generate_v02_sellers(n_sellers, regime, rng)
    seller_policies: dict[str, VolumeDiscountPolicy] = {}
    seller_v3: dict[str, SellerV2] = {}
    for seller in sellers_v02:
        policy = _generate_volume_discount_policy(rng)
        seller_policies[seller.id] = policy
        seller_v3[seller.id] = _build_seller_v3(seller, policy)

    # Buyers from CEO requirements
    requirements = sample_requirements(n_buyers, rng)
    buyer_agent = BuyerAgent(llm_client=llm_client) if llm_client is not None else None

    buyers_v02: list[Buyer] = []
    buyer_requirements: dict[str, EndRequirement] = {}
    buyer_public: dict[str, BuyerPublicProfile] = {}
    buyer_private: dict[str, BuyerPrivateProfile] = {}

    for idx, req in enumerate(requirements):
        if buyer_agent is not None:
            pub_raw, priv_raw = _translate_with_cache(buyer_agent, req, rng)
            # Force the buyer_id to match our index so downstream lookup is stable.
            pub = pub_raw.model_copy(update={"buyer_id": f"B{idx}"})
            priv = priv_raw  # keep as-is; max_wtp / urgency / context are LLM judgment
            urgency_score = priv.urgency_score
        else:
            pub, priv = _synth_profile_from_requirement(req, idx, rng)
            urgency_score = priv.urgency_score

        buyer_v02 = _profiles_to_v02_buyer(pub, priv, urgency_score, idx)
        buyers_v02.append(buyer_v02)
        buyer_requirements[buyer_v02.id] = req
        buyer_public[buyer_v02.id] = pub
        buyer_private[buyer_v02.id] = priv

    market = Market(
        id=f"mkt-v3-{regime}-s{seed}-{n_buyers}x{n_sellers}",
        regime=regime,
        seed=seed,
        buyers=tuple(buyers_v02),
        sellers=tuple(sellers_v02),
    )
    enrichment = V3Enrichment(
        buyer_requirements=buyer_requirements,
        buyer_public_profiles=buyer_public,
        buyer_private_profiles=buyer_private,
        seller_volume_policies=seller_policies,
        seller_v3=seller_v3,
    )
    return market, enrichment


__all__ = [
    "V3Enrichment",
    "generate_market_v3",
    "clear_translate_cache",
]
