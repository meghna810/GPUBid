"""Public board state — what every agent sees each round, plus the mutable run state.

Crucially, only structured offer tuples are public. Free-form `reasoning` text
is *not* propagated between agents (cleared in board snapshots) so a buyer can't
leak their max WTP to other agents through a justification string.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from gpubid.schema import (
    BoardSnapshot,
    CapacitySlot,
    Deal,
    Market,
    Offer,
    OfferKind,
)


@dataclass
class RunState:
    """Mutable state across rounds.

    Tracks the most recent ask per seller-slot, the most recent bid per buyer,
    cumulative deals, remaining capacity per slot, and which agents are still
    active (not yet fully filled / not yet exhausted).
    """

    market: Market
    deals: list[Deal] = field(default_factory=list)
    seller_asks: dict[str, Offer] = field(default_factory=dict)   # slot_id  -> latest ASK
    buyer_bids: dict[str, Offer] = field(default_factory=dict)    # buyer_id -> latest BID
    slot_remaining_qty: dict[str, int] = field(default_factory=dict)
    active_buyer_ids: set[str] = field(default_factory=set)
    active_seller_ids: set[str] = field(default_factory=set)

    @classmethod
    def initial(cls, market: Market) -> "RunState":
        slot_remaining = {
            slot.id: slot.qty
            for s in market.sellers
            for slot in s.capacity_slots
        }
        return cls(
            market=market,
            deals=[],
            seller_asks={},
            buyer_bids={},
            slot_remaining_qty=slot_remaining,
            active_buyer_ids={b.id for b in market.buyers},
            active_seller_ids={s.id for s in market.sellers},
        )

    # ---------- lookups ----------

    def slot_by_id(self, slot_id: str) -> Optional[CapacitySlot]:
        for s in self.market.sellers:
            for slot in s.capacity_slots:
                if slot.id == slot_id:
                    return slot
        return None

    def seller_id_for_slot(self, slot_id: str) -> Optional[str]:
        for s in self.market.sellers:
            for slot in s.capacity_slots:
                if slot.id == slot_id:
                    return s.id
        return None

    def slot_ids_for_seller(self, seller_id: str) -> list[str]:
        for s in self.market.sellers:
            if s.id == seller_id:
                return [slot.id for slot in s.capacity_slots]
        return []

    # ---------- public board snapshot (passed to agents) ----------

    def public_snapshot(self, round_n: int) -> BoardSnapshot:
        """A snapshot agents see at the start of a round.

        Reasoning text is stripped from offers — only structured tuples are public.
        """
        def _strip(o: Offer) -> Offer:
            return o.model_copy(update={"reasoning": ""})

        asks = tuple(
            _strip(a)
            for a in self.seller_asks.values()
            if self.slot_remaining_qty.get(a.slot_id or "", 0) > 0
        )
        bids = tuple(
            _strip(b)
            for b in self.buyer_bids.values()
            if b.from_id in self.active_buyer_ids
        )

        return BoardSnapshot(
            round_n=round_n,
            asks=asks,
            bids=bids,
            deals_so_far=tuple(self.deals),
            active_buyer_ids=tuple(sorted(self.active_buyer_ids)),
            active_seller_ids=tuple(sorted(self.active_seller_ids)),
        )


@dataclass(frozen=True)
class AgentActionRecord:
    """What one agent did in one round — used for forensics and tournament analysis."""

    agent_id: str
    new_offers: tuple[Offer, ...] = ()
    accept_offer_ids: tuple[str, ...] = ()
    reasoning: str = ""


@dataclass(frozen=True)
class RoundSnapshot:
    """A frozen, yieldable snapshot used by the visualization layer."""

    round_n: int
    max_rounds: int
    asks: tuple[Offer, ...]
    bids: tuple[Offer, ...]
    new_deals: tuple[Deal, ...]
    all_deals: tuple[Deal, ...]
    active_buyer_ids: tuple[str, ...]
    active_seller_ids: tuple[str, ...]
    is_final: bool
    # Forensic record — what each agent *attempted* this round (regardless of whether
    # the action survived clearing). Empty in older preset files.
    actions: tuple[AgentActionRecord, ...] = ()


def make_offer_id(kind: OfferKind, agent_or_slot_id: str, round_n: int, suffix: str = "") -> str:
    """Build a stable, human-readable offer id."""
    parts = [kind.value, agent_or_slot_id, f"r{round_n}"]
    if suffix:
        parts.append(suffix)
    return "-".join(parts)


__all__ = ["RunState", "RoundSnapshot", "AgentActionRecord", "make_offer_id"]
