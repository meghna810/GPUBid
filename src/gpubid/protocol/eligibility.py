"""Phase 5 — Eligibility filter (deterministic, NOT LLM-driven).

Each seller checks the buyer's public profile against its own private inventory
to decide whether it can plausibly fulfill. This is mechanical: GPU type
intersection, capacity intersection, time-window overlap, interruption
compatibility. LLM judgment enters at the *negotiation* step (per spec §6.3),
not eligibility.
"""

from __future__ import annotations

from gpubid.domain.profiles import BuyerPublicProfile, SellerV2
from gpubid.protocol.broadcast import BroadcastMessage


# Map semantic interruption tolerance levels to numeric ranks
# (lower = more strict).
_TOL_RANK: dict[str, int] = {
    "none": 0,
    "checkpoint_15min": 1,
    "checkpoint_60min": 2,
    "any": 3,
}


def is_eligible(seller: SellerV2, broadcast: BroadcastMessage) -> tuple[bool, str]:
    """Return ``(eligible, reason)`` for forensics logging.

    A seller is eligible iff, for at least one inventory slot:
      - the slot's GPU type is in buyer.gpu_type_preferences, AND
      - slot capacity ≥ buyer.qty_gpus, AND
      - slot duration ≥ buyer.duration_hours, AND
      - slot's start_slot ≥ buyer.time_window.earliest_start_slot, AND
      - slot's start_slot + buyer.duration ≤ buyer.time_window.latest_finish_slot, AND
      - the seller can offer interruption tolerance the buyer accepts.
    """
    pub = broadcast.buyer_public
    eligible_slots = []
    rejection_reasons: list[str] = []

    for slot in seller.public.inventory_slots:
        if slot.gpu_type not in pub.gpu_type_preferences:
            rejection_reasons.append(f"slot {slot.slot_id}: gpu type {slot.gpu_type.value} not preferred")
            continue
        if slot.qty_gpus < pub.qty_gpus:
            rejection_reasons.append(f"slot {slot.slot_id}: capacity {slot.qty_gpus} < buyer {pub.qty_gpus}")
            continue
        if slot.duration_hours < pub.duration_hours:
            rejection_reasons.append(f"slot {slot.slot_id}: duration {slot.duration_hours} < buyer {pub.duration_hours}")
            continue
        if slot.start_slot < pub.time_window.earliest_start_slot:
            rejection_reasons.append(f"slot {slot.slot_id}: starts before buyer window")
            continue
        if slot.start_slot + pub.duration_hours > pub.time_window.latest_finish_slot:
            rejection_reasons.append(f"slot {slot.slot_id}: finishes after buyer window")
            continue
        eligible_slots.append(slot.slot_id)

    if eligible_slots:
        return True, f"eligible via {len(eligible_slots)} slot(s): {','.join(eligible_slots)}"
    return False, "; ".join(rejection_reasons) if rejection_reasons else "no inventory"


__all__ = ["is_eligible"]
