"""Phase 5 — Broadcast: each seller receives only buyer.public.

SCAFFOLDED. This module is small and protocol-only (no LLM dependency
itself), but it ships in the same phase as eligibility + round-runner which
do need LLM fixtures, so it lives behind the same TODO marker.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from gpubid.domain.profiles import BuyerPublicProfile, BuyerV2, SellerV2


@dataclass(frozen=True)
class BroadcastMessage:
    """One seller's view of one buyer's announcement."""

    broadcast_id: str
    buyer_public: BuyerPublicProfile
    target_seller_id: str
    market_round: int = 0  # broadcast happens before round 1


def broadcast_buyer_to_sellers(
    buyer: BuyerV2,
    sellers: list[SellerV2],
    rng=None,  # rng parameter kept for spec parity; not used yet
) -> list[BroadcastMessage]:
    """Emit one BroadcastMessage per seller, each carrying ONLY buyer.public.

    Sellers see no information from buyer.private. The broadcast_id is shared
    across all messages (one buyer = one broadcast event); seller responses
    correlate via this id.
    """
    bid = str(uuid.uuid4())
    return [
        BroadcastMessage(
            broadcast_id=bid,
            buyer_public=buyer.public,
            target_seller_id=s.public.seller_id,
        )
        for s in sellers
    ]


__all__ = ["BroadcastMessage", "broadcast_buyer_to_sellers"]
