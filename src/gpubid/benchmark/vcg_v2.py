"""Tiered-VCG benchmark for v0.3 markets with volume-discount policies.

Classic VCG assumes each bidder reports a single value for each allocation.
With volume discounts, the seller's "value function" is piecewise-linear in
qty: the social-welfare-maximizing allocation depends on which discount tier
fires, which itself depends on the allocation. This breaks direct VCG.

We solve it by *linearization*: each seller-slot × discount-tier combination
becomes a discrete bundle. The MIP picks at most one bundle per (buyer,
seller-slot) pair. Document this in the notebook: it is an upper bound that
is itself approximate when tiers are coarse, but it is a fair comparison
because the agentic mechanism is ALSO restricted to the same discrete tiers.

Per the spec §11: keep the legacy ``solve_vcg`` (gpubid.benchmark.vcg)
functioning for backwards compatibility on legacy presets. The metric table
in cell 12 shows BOTH and labels them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pulp

from gpubid.domain.profiles import (
    BuyerV2,
    InventorySlot,
    SellerV2,
    VolumeDiscountTier,
)
from gpubid.engine.clearing import ask_satisfies_buyer
from gpubid.schema import (
    Buyer,
    Deal,
    GPUType,
    InterruptionTolerance,
    Market,
    Offer,
    OfferKind,
    Seller,
)


# ---------------------------------------------------------------------------
# Bundle materialization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Bundle:
    """One discrete (slot, tier) option.

    The bundle records the effective price PER GPU-HOUR after this tier's
    discount applies. The MIP chooses bundles to maximize total welfare.
    """

    slot_id: str
    seller_id: str
    gpu_type: GPUType
    qty_available: int
    duration_hours: float
    start_slot: int
    list_price_per_gpu_hr: float
    reserve_per_gpu_hr: float
    discount_pct: float                  # 0 = no-discount tier
    min_qty_for_tier: int
    min_duration_for_tier: float

    @property
    def effective_price(self) -> float:
        return self.list_price_per_gpu_hr * (1.0 - self.discount_pct)


def _materialize_bundles_v2(seller: SellerV2) -> list[_Bundle]:
    """Expand a v0.3 SellerV2 into one bundle per (slot × tier) combination.

    Always includes the no-discount tier (discount_pct=0) so a buyer who can't
    or won't commit enough volume still has a price to face.
    """
    bundles: list[_Bundle] = []
    list_prices = seller.public.list_price_per_gpu_hr
    reserves = seller.private.reserve_per_slot

    for slot in seller.public.inventory_slots:
        list_price = list_prices.get(slot.gpu_type)
        reserve = reserves.get(slot.slot_id)
        if list_price is None or reserve is None:
            continue

        # No-discount baseline tier — always materialized.
        bundles.append(_Bundle(
            slot_id=slot.slot_id,
            seller_id=seller.public.seller_id,
            gpu_type=slot.gpu_type,
            qty_available=slot.qty_gpus,
            duration_hours=slot.duration_hours,
            start_slot=slot.start_slot,
            list_price_per_gpu_hr=list_price,
            reserve_per_gpu_hr=reserve,
            discount_pct=0.0,
            min_qty_for_tier=1,
            min_duration_for_tier=0.0,
        ))

        # Each discount tier in the policy becomes its own bundle.
        for tier in seller.public.volume_discount_policy.tiers:
            bundles.append(_Bundle(
                slot_id=slot.slot_id,
                seller_id=seller.public.seller_id,
                gpu_type=slot.gpu_type,
                qty_available=slot.qty_gpus,
                duration_hours=slot.duration_hours,
                start_slot=slot.start_slot,
                list_price_per_gpu_hr=list_price,
                reserve_per_gpu_hr=reserve,
                discount_pct=tier.discount_pct,
                min_qty_for_tier=tier.min_qty_gpus,
                min_duration_for_tier=tier.min_duration_hours,
            ))

    return bundles


def _materialize_bundles_legacy(seller: Seller) -> list[_Bundle]:
    """Legacy v0.2 sellers don't have volume discounts; one bundle per slot."""
    bundles: list[_Bundle] = []
    for slot in seller.capacity_slots:
        bundles.append(_Bundle(
            slot_id=slot.id,
            seller_id=seller.id,
            gpu_type=slot.gpu_type,
            qty_available=slot.qty,
            duration_hours=float(slot.duration),
            start_slot=slot.start,
            # Legacy doesn't carry a separate list price; use a markup so the
            # bundle is well-formed. Welfare math below uses reserve, not list,
            # so the chosen markup doesn't affect VCG welfare.
            list_price_per_gpu_hr=slot.reserve_per_gpu_hr * 1.5,
            reserve_per_gpu_hr=slot.reserve_per_gpu_hr,
            discount_pct=0.0,
            min_qty_for_tier=1,
            min_duration_for_tier=0.0,
        ))
    return bundles


# ---------------------------------------------------------------------------
# Compatibility check (works for both v0.2 Buyer and v0.3 BuyerV2)
# ---------------------------------------------------------------------------


def _bundle_compatible_with_buyer(bundle: _Bundle, buyer) -> bool:
    """Buyer can use this bundle iff GPU type, qty, time window, duration, and
    tier qualifications all align.

    Reuses the existing v0.2 ``ask_satisfies_buyer`` for the time/qty/gpu
    checks via a hypothetical ASK. The tier-qualification check is bundle-
    specific.
    """
    # Tier qualification: buyer must commit at least the tier's min qty/duration.
    buyer_qty = buyer.job.qty if hasattr(buyer, "job") else buyer.public.qty_gpus
    buyer_dur = (
        float(buyer.job.duration) if hasattr(buyer, "job")
        else buyer.public.duration_hours
    )
    if buyer_qty < bundle.min_qty_for_tier:
        return False
    if buyer_dur < bundle.min_duration_for_tier:
        return False

    if hasattr(buyer, "job"):
        # Legacy v0.2 buyer
        hypothetical = Offer(
            id="vcg2-probe", round_n=1, from_id="vcg2",
            kind=OfferKind.ASK, slot_id=bundle.slot_id,
            price_per_gpu_hr=bundle.reserve_per_gpu_hr,
            qty=bundle.qty_available, gpu_type=bundle.gpu_type,
            start=bundle.start_slot, duration=int(bundle.duration_hours),
            interruption_tolerance=InterruptionTolerance.NONE,
        )
        # Need a CapacitySlot-shaped object for ask_satisfies_buyer
        from gpubid.schema import CapacitySlot
        slot = CapacitySlot(
            id=bundle.slot_id, gpu_type=bundle.gpu_type,
            start=bundle.start_slot, duration=int(bundle.duration_hours),
            qty=bundle.qty_available, reserve_per_gpu_hr=bundle.reserve_per_gpu_hr,
        )
        return ask_satisfies_buyer(hypothetical, buyer, slot, bundle.qty_available)

    # v0.3 BuyerV2 — explicit checks
    pub = buyer.public
    if bundle.gpu_type not in pub.gpu_type_preferences:
        return False
    if bundle.qty_available < pub.qty_gpus:
        return False
    if bundle.duration_hours < pub.duration_hours:
        return False
    if bundle.start_slot < pub.time_window.earliest_start_slot:
        return False
    if bundle.start_slot + pub.duration_hours > pub.time_window.latest_finish_slot:
        return False
    return True


# ---------------------------------------------------------------------------
# MIP
# ---------------------------------------------------------------------------


@dataclass
class TieredVCGResult:
    """Welfare-optimal allocation under linearized-tier VCG."""

    assignments: list[tuple[str, str, str, float]]   # (buyer_id, seller_id, slot_id, effective_price)
    welfare: float
    deals: list[Deal]
    solver_status: str
    linearization_notes: str = ""


def solve_vcg_tiered(
    market_or_v2: object,
    *,
    time_limit_seconds: float = 10.0,
) -> TieredVCGResult:
    """Solve the welfare-optimal MIP for either a legacy Market or v0.3 sellers.

    ``market_or_v2`` may be:
      - the legacy ``gpubid.schema.Market`` (v0.2; one bundle per slot), or
      - a tuple ``(buyers_v2, sellers_v2)`` for v0.3 markets with volume
        discounts.

    The MIP variables are ``x[buyer_id, bundle_index] in {0, 1}``. Each buyer is
    assigned at most one bundle. Per slot, the bundles draw from a shared
    capacity pool (the slot's qty_gpus). The objective is total welfare:
    ``sum (buyer_value − reserve) × qty × duration`` over chosen bundles,
    where buyer_value uses the buyer's private max WTP.
    """
    buyers, sellers, is_v2 = _normalize_inputs(market_or_v2)

    # Materialize bundles
    all_bundles: list[_Bundle] = []
    for s in sellers:
        bundles = _materialize_bundles_v2(s) if is_v2 else _materialize_bundles_legacy(s)
        all_bundles.extend(bundles)

    if not all_bundles or not buyers:
        return TieredVCGResult(
            assignments=[], welfare=0.0, deals=[], solver_status="Trivial",
            linearization_notes=f"materialized {len(all_bundles)} bundles, {len(buyers)} buyers",
        )

    prob = pulp.LpProblem("gpubid_vcg_tiered", pulp.LpMaximize)

    # x[buyer_id, bundle_idx] = 1 if buyer takes this bundle
    x: dict[tuple[str, int], pulp.LpVariable] = {}
    buyer_ids = [_buyer_id(b) for b in buyers]
    buyer_by_id = dict(zip(buyer_ids, buyers))

    for b_idx, buyer in enumerate(buyers):
        bid = _buyer_id(buyer)
        for k, bundle in enumerate(all_bundles):
            if _bundle_compatible_with_buyer(bundle, buyer):
                x[(bid, k)] = pulp.LpVariable(
                    f"x_{bid}_{k}", lowBound=0, upBound=1, cat=pulp.LpBinary,
                )

    # Objective: total welfare
    obj_terms = []
    for (bid, k), var in x.items():
        bundle = all_bundles[k]
        buyer = buyer_by_id[bid]
        bv = _buyer_value(buyer)
        bq = _buyer_qty(buyer)
        bd = _buyer_duration(buyer)
        welfare_contribution = (bv - bundle.reserve_per_gpu_hr) * bq * bd
        obj_terms.append(welfare_contribution * var)

    prob += pulp.lpSum(obj_terms) if obj_terms else 0

    # Each buyer at most one bundle
    for bid in buyer_ids:
        terms = [v for (b2, k), v in x.items() if b2 == bid]
        if terms:
            prob += pulp.lpSum(terms) <= 1, f"one_assignment_{bid}"

    # Slot capacity — across ALL bundles for the same slot, total qty must fit
    by_slot: dict[str, list[tuple[str, int]]] = {}
    for (bid, k) in x.keys():
        by_slot.setdefault(all_bundles[k].slot_id, []).append((bid, k))
    for slot_id, key_list in by_slot.items():
        cap = next(b.qty_available for b in all_bundles if b.slot_id == slot_id)
        terms = [_buyer_qty(buyer_by_id[bid]) * x[(bid, k)] for (bid, k) in key_list]
        prob += pulp.lpSum(terms) <= cap, f"capacity_{slot_id}"

    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds)
    status = prob.solve(solver)
    status_name = pulp.LpStatus[status]

    assignments: list[tuple[str, str, str, float]] = []
    for (bid, k), var in x.items():
        if var.value() is not None and var.value() > 0.5:
            bundle = all_bundles[k]
            assignments.append((bid, bundle.seller_id, bundle.slot_id, bundle.effective_price))

    welfare = float(pulp.value(prob.objective) or 0.0)

    # Synthesize Deals
    deals: list[Deal] = []
    for (bid, sid, slot_id, eff_price) in assignments:
        buyer = buyer_by_id[bid]
        bundle = next(b for b in all_bundles if b.slot_id == slot_id)
        deals.append(Deal(
            id=f"vcg2-{bid}-{slot_id}",
            round_n=0,
            buyer_id=bid,
            seller_id=sid,
            slot_id=slot_id,
            qty=_buyer_qty(buyer),
            price_per_gpu_hr=eff_price,
            start=max(bundle.start_slot, _buyer_earliest_start(buyer)),
            duration=int(_buyer_duration(buyer)),
            gpu_type=bundle.gpu_type,
            interruption_tolerance=InterruptionTolerance.NONE,
        ))

    notes = (
        f"materialized {len(all_bundles)} bundles "
        f"(no-discount + tiers across {len(sellers)} sellers); "
        f"{len([k for k in x.keys()])} compatible (buyer, bundle) pairs"
    )

    return TieredVCGResult(
        assignments=assignments,
        welfare=welfare,
        deals=deals,
        solver_status=status_name,
        linearization_notes=notes,
    )


# ---------------------------------------------------------------------------
# Normalization helpers (handle both legacy and v0.3 inputs)
# ---------------------------------------------------------------------------


def _normalize_inputs(arg: object) -> tuple[list, list, bool]:
    """Return (buyers, sellers, is_v2)."""
    if isinstance(arg, Market):
        return list(arg.buyers), list(arg.sellers), False
    if isinstance(arg, tuple) and len(arg) == 2:
        buyers_v2, sellers_v2 = arg
        return list(buyers_v2), list(sellers_v2), True
    raise TypeError(
        "solve_vcg_tiered expects either a v0.2 Market or a (buyers_v2, sellers_v2) tuple"
    )


def _buyer_id(buyer) -> str:
    return getattr(buyer, "id", None) or buyer.public.buyer_id


def _buyer_value(buyer) -> float:
    if hasattr(buyer, "job"):
        return float(buyer.job.max_value_per_gpu_hr)
    return float(buyer.private.max_willingness_to_pay)


def _buyer_qty(buyer) -> int:
    if hasattr(buyer, "job"):
        return int(buyer.job.qty)
    return int(buyer.public.qty_gpus)


def _buyer_duration(buyer) -> float:
    if hasattr(buyer, "job"):
        return float(buyer.job.duration)
    return float(buyer.public.duration_hours)


def _buyer_earliest_start(buyer) -> int:
    if hasattr(buyer, "job"):
        return int(buyer.job.earliest_start)
    return int(buyer.public.time_window.earliest_start_slot)


__all__ = ["TieredVCGResult", "solve_vcg_tiered"]
