"""Compatibility checks, accept-processing, concentration cap, reserve guard.

The clearing engine sits *outside* the agents — it enforces the rules even
when an agent's logic (rule-based or LLM) tries to do something invalid. The
accept-below-reserve violation rate is itself a metric we report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gpubid.engine.board import RunState
from gpubid.schema import (
    Buyer,
    CapacitySlot,
    Deal,
    InterruptionTolerance,
    Market,
    Offer,
    OfferKind,
)


# ---------------------------------------------------------------------------
# Tolerance ordering
# ---------------------------------------------------------------------------

TOLERANCE_RANK: dict[InterruptionTolerance, int] = {
    InterruptionTolerance.NONE: 0,            # most strict (continuous guarantee)
    InterruptionTolerance.CHECKPOINT: 1,
    InterruptionTolerance.INTERRUPTIBLE: 2,   # most relaxed
}


def buyer_accepts_tolerance(buyer: Buyer, offer_tolerance: InterruptionTolerance) -> bool:
    """A buyer accepts an offer iff the offer's guarantees are at least as strict as the buyer needs.

    Buyer with NONE only accepts NONE. Buyer with INTERRUPTIBLE accepts anything.
    """
    return TOLERANCE_RANK[offer_tolerance] <= TOLERANCE_RANK[buyer.job.interruption_tolerance]


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


def ask_satisfies_buyer(ask: Offer, buyer: Buyer, slot: CapacitySlot, remaining_qty: int) -> bool:
    """Can `buyer` actually use this seller ASK on this slot?"""
    if ask.kind != OfferKind.ASK:
        return False
    if ask.gpu_type not in buyer.job.acceptable_gpus:
        return False
    if remaining_qty < buyer.job.qty:
        return False
    if slot.duration < buyer.job.duration:
        return False
    if ask.start < buyer.job.earliest_start:
        return False
    if ask.start + buyer.job.duration > buyer.job.latest_finish:
        return False
    if not buyer_accepts_tolerance(buyer, ask.interruption_tolerance):
        return False
    return True


def bid_satisfies_slot(bid: Offer, slot: CapacitySlot, remaining_qty: int) -> bool:
    """Can `slot` actually fulfill this buyer BID?"""
    if bid.kind != OfferKind.BID:
        return False
    if slot.gpu_type != bid.gpu_type:
        return False
    if remaining_qty < bid.qty:
        return False
    if slot.duration < bid.duration:
        return False
    if slot.start > bid.start:
        return False
    if slot.start + slot.duration < bid.start + bid.duration:
        return False
    return True


# ---------------------------------------------------------------------------
# Acceptance processing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClearingResult:
    """Outcome of processing this round's accepts."""

    new_deals: list[Deal]
    rejected_accepts: list[tuple[str, str]]   # (accepter_id, reason)
    cap_blocked: list[tuple[str, str]]        # (buyer_id, reason)
    reserve_violations: list[tuple[str, str]] # (agent_id, reason)


def buyer_accepts_ask(
    *,
    market: Market,
    buyer: Buyer,
    ask: Offer,
    state: RunState,
    round_n: int,
    deals_for_buyer_so_far: list[Deal],
    concentration_cap_pct: Optional[float],
) -> tuple[Optional[Deal], Optional[str]]:
    """Validate and (if valid) build the Deal that results from a buyer accepting an ask.

    Returns (deal_or_None, rejection_reason_or_None).
    """
    if ask.slot_id is None:
        return None, "ask missing slot_id"

    slot = state.slot_by_id(ask.slot_id)
    if slot is None:
        return None, f"unknown slot {ask.slot_id}"

    seller_id = state.seller_id_for_slot(ask.slot_id)
    if seller_id is None:
        return None, f"could not resolve seller for slot {ask.slot_id}"

    remaining = state.slot_remaining_qty.get(ask.slot_id, 0)
    if not ask_satisfies_buyer(ask, buyer, slot, remaining):
        return None, "ask does not satisfy buyer"

    # Reserve guard — buyer side: never pay above your value
    if ask.price_per_gpu_hr > buyer.job.max_value_per_gpu_hr:
        return None, "would pay above max value (buyer reserve violation)"

    # Reserve guard — seller side: should not have posted below reserve, but enforce here
    if ask.price_per_gpu_hr < slot.reserve_per_gpu_hr:
        return None, "ask below seller reserve"

    # Concentration cap
    if concentration_cap_pct is not None:
        cap = compute_concentration_cap(market, concentration_cap_pct)
        already = sum(d.qty * d.duration for d in deals_for_buyer_so_far)
        if already + buyer.job.qty * buyer.job.duration > cap:
            return None, f"concentration cap {concentration_cap_pct*100:.0f}% exceeded"

    deal = Deal(
        id=f"deal-{buyer.id}-{ask.slot_id}-r{round_n}",
        round_n=round_n,
        buyer_id=buyer.id,
        seller_id=seller_id,
        slot_id=ask.slot_id,
        qty=buyer.job.qty,
        price_per_gpu_hr=ask.price_per_gpu_hr,
        start=ask.start,
        duration=buyer.job.duration,
        gpu_type=ask.gpu_type,
        interruption_tolerance=ask.interruption_tolerance,
    )
    return deal, None


def seller_accepts_bid(
    *,
    market: Market,
    seller_id: str,
    slot: CapacitySlot,
    bid: Offer,
    state: RunState,
    round_n: int,
    deals_for_buyer_so_far: list[Deal],
    concentration_cap_pct: Optional[float],
) -> tuple[Optional[Deal], Optional[str]]:
    """Build the Deal from a seller's slot accepting a buyer's bid."""
    remaining = state.slot_remaining_qty.get(slot.id, 0)
    if not bid_satisfies_slot(bid, slot, remaining):
        return None, "bid does not satisfy slot"

    # Reserve guard — seller side: never accept below reserve
    if bid.price_per_gpu_hr < slot.reserve_per_gpu_hr:
        return None, "bid below seller reserve"

    # Find the buyer to check their reserve too
    buyer = next((b for b in market.buyers if b.id == bid.from_id), None)
    if buyer is None:
        return None, f"unknown buyer {bid.from_id}"
    if bid.price_per_gpu_hr > buyer.job.max_value_per_gpu_hr:
        return None, "bid above buyer max value (rare, but rejected)"

    # Concentration cap
    if concentration_cap_pct is not None:
        cap = compute_concentration_cap(market, concentration_cap_pct)
        already = sum(d.qty * d.duration for d in deals_for_buyer_so_far)
        if already + bid.qty * bid.duration > cap:
            return None, f"concentration cap {concentration_cap_pct*100:.0f}% exceeded"

    deal = Deal(
        id=f"deal-{bid.from_id}-{slot.id}-r{round_n}",
        round_n=round_n,
        buyer_id=bid.from_id,
        seller_id=seller_id,
        slot_id=slot.id,
        qty=bid.qty,
        price_per_gpu_hr=bid.price_per_gpu_hr,
        start=bid.start,
        duration=bid.duration,
        gpu_type=bid.gpu_type,
        interruption_tolerance=bid.interruption_tolerance,
    )
    return deal, None


def compute_concentration_cap(market: Market, cap_pct: float) -> int:
    """How many GPU-hours can any one buyer win? `cap_pct` ∈ (0, 1]."""
    return int(market.total_supply_gpu_hours * cap_pct)


def commit_deal(state: RunState, deal: Deal) -> None:
    """Apply a deal to state: decrement capacity, mark buyer fulfilled."""
    state.deals.append(deal)
    state.slot_remaining_qty[deal.slot_id] -= deal.qty
    state.active_buyer_ids.discard(deal.buyer_id)

    # Remove the buyer's bid (job filled)
    state.buyer_bids.pop(deal.buyer_id, None)

    # Remove the slot's ask if exhausted
    if state.slot_remaining_qty[deal.slot_id] <= 0:
        state.seller_asks.pop(deal.slot_id, None)

    # Mark a seller exhausted only when *all* their slots are empty
    seller_slot_ids = state.slot_ids_for_seller(deal.seller_id)
    if all(state.slot_remaining_qty.get(sid, 0) <= 0 for sid in seller_slot_ids):
        state.active_seller_ids.discard(deal.seller_id)


__all__ = [
    "TOLERANCE_RANK",
    "ask_satisfies_buyer",
    "bid_satisfies_slot",
    "buyer_accepts_ask",
    "seller_accepts_bid",
    "compute_concentration_cap",
    "commit_deal",
    "buyer_accepts_tolerance",
    "ClearingResult",
]
