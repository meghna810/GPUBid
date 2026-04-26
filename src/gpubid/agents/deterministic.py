"""Rule-based agents that drive fast mode (no LLM calls).

Behavior is intentionally interesting, not naive:

  - Sellers post asks at reserve × markup. Markup is higher in tight regimes.
  - Sellers decay asks each round, more aggressively near the round deadline.
  - Sellers accept the highest compatible bid that clears their reserve.

  - Buyers post bids at value × markdown. Markdown shrinks (= bid rises) with urgency.
  - Buyers raise bids each round, more aggressively if urgent or near deadline.
  - Buyers accept the cheapest compatible ask at or below their max value.

These dynamics produce a watchable trading-floor demo even without any LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from gpubid.engine.board import RunState, make_offer_id
from gpubid.engine.clearing import (
    ask_satisfies_buyer,
    bid_satisfies_slot,
    buyer_accepts_tolerance,
)
from gpubid.market import GPU_BASE_RESERVE
from gpubid.schema import (
    Buyer,
    GPUType,
    InterruptionTolerance,
    Offer,
    OfferKind,
    Seller,
)


@dataclass(frozen=True)
class AgentAction:
    """What an agent does in a single round."""

    new_offers: tuple[Offer, ...] = ()
    accept_offer_ids: tuple[str, ...] = ()
    reasoning: str = ""


# ---------------------------------------------------------------------------
# Deterministic seller
# ---------------------------------------------------------------------------


@dataclass
class DeterministicSeller:
    seller_id: str

    def decide(self, state: RunState, round_n: int, max_rounds: int) -> AgentAction:
        seller = next(s for s in state.market.sellers if s.id == self.seller_id)
        regime = state.market.regime
        markup = 1.50 if regime == "tight" else 1.20

        rounds_left = max_rounds - round_n
        decay_per_round = 0.05 + (0.05 if rounds_left <= 1 else 0.0)

        new_offers: list[Offer] = []
        accepts: list[str] = []
        reasoning_bits: list[str] = []

        for slot in seller.capacity_slots:
            remaining = state.slot_remaining_qty.get(slot.id, 0)
            if remaining <= 0:
                continue

            # 1) Try to accept any compatible bid that clears reserve.
            best_bid: Optional[Offer] = None
            for bid in state.buyer_bids.values():
                if bid_satisfies_slot(bid, slot, remaining) and bid.price_per_gpu_hr >= slot.reserve_per_gpu_hr:
                    if best_bid is None or bid.price_per_gpu_hr > best_bid.price_per_gpu_hr:
                        best_bid = bid
            if best_bid is not None and round_n > 1:
                accepts.append(best_bid.id)
                reasoning_bits.append(
                    f"slot {slot.id}: accepting bid {best_bid.id} at ${best_bid.price_per_gpu_hr:.2f}"
                )
                continue

            # 2) Otherwise post (or update) the ask for this slot.
            prev_ask = state.seller_asks.get(slot.id)
            if prev_ask is None:
                price = round(slot.reserve_per_gpu_hr * markup, 2)
                reasoning_bits.append(f"slot {slot.id}: opening ask at ${price:.2f}")
            else:
                # Decay toward reserve, but never below reserve × 1.05
                floor = round(slot.reserve_per_gpu_hr * 1.05, 2)
                proposed = round(prev_ask.price_per_gpu_hr * (1.0 - decay_per_round), 2)
                price = max(floor, proposed)
                reasoning_bits.append(
                    f"slot {slot.id}: ask {prev_ask.price_per_gpu_hr:.2f} -> {price:.2f}"
                )

            offer = Offer(
                id=make_offer_id(OfferKind.ASK, slot.id, round_n),
                round_n=round_n,
                from_id=self.seller_id,
                kind=OfferKind.ASK,
                slot_id=slot.id,
                price_per_gpu_hr=price,
                qty=remaining,
                gpu_type=slot.gpu_type,
                start=slot.start,
                duration=slot.duration,
                interruption_tolerance=InterruptionTolerance.NONE,  # default: most strict
                reasoning=f"deterministic seller {self.seller_id}",
            )
            new_offers.append(offer)

        return AgentAction(
            new_offers=tuple(new_offers),
            accept_offer_ids=tuple(accepts),
            reasoning="; ".join(reasoning_bits),
        )


# ---------------------------------------------------------------------------
# Deterministic buyer
# ---------------------------------------------------------------------------


@dataclass
class DeterministicBuyer:
    buyer_id: str

    def decide(self, state: RunState, round_n: int, max_rounds: int) -> AgentAction:
        buyer = next(b for b in state.market.buyers if b.id == self.buyer_id)

        # 1) Accept any compatible ask we can afford. Pick cheapest.
        compatible: list[Offer] = []
        for ask in state.seller_asks.values():
            if ask.slot_id is None:
                continue
            slot = state.slot_by_id(ask.slot_id)
            remaining = state.slot_remaining_qty.get(ask.slot_id, 0) if ask.slot_id else 0
            if slot is None:
                continue
            if not ask_satisfies_buyer(ask, buyer, slot, remaining):
                continue
            if ask.price_per_gpu_hr > buyer.job.max_value_per_gpu_hr:
                continue
            compatible.append(ask)

        if compatible and round_n > 1:
            best = min(compatible, key=lambda a: a.price_per_gpu_hr)
            return AgentAction(
                accept_offer_ids=(best.id,),
                reasoning=f"accepting cheapest compatible ask {best.id} at ${best.price_per_gpu_hr:.2f}",
            )

        # 2) Otherwise post (or raise) a bid.
        prev_bid = state.buyer_bids.get(buyer.id)
        ceiling = buyer.job.max_value_per_gpu_hr * 0.95  # leave a thin margin

        if prev_bid is None:
            # Initial markdown: more urgent buyers bid closer to their value.
            markdown = 0.55 + 0.30 * buyer.urgency
            price = round(buyer.job.max_value_per_gpu_hr * markdown, 2)
            reasoning = (
                f"opening bid at ${price:.2f} "
                f"(urgency {buyer.urgency:.2f}, max ${buyer.job.max_value_per_gpu_hr:.2f})"
            )
        else:
            # Climb toward value, faster if urgent or close to deadline.
            rounds_left = max_rounds - round_n
            climb = 0.05 + 0.05 * buyer.urgency + (0.05 if rounds_left <= 1 else 0.0)
            proposed = prev_bid.price_per_gpu_hr * (1.0 + climb)
            price = round(min(proposed, ceiling), 2)
            reasoning = f"bid {prev_bid.price_per_gpu_hr:.2f} -> {price:.2f} (climb {climb:.2f})"

        # Choose GPU type: cheapest acceptable visible ask, else most-expensive acceptable
        gpu_type = _pick_target_gpu(buyer, state)

        offer = Offer(
            id=make_offer_id(OfferKind.BID, buyer.id, round_n),
            round_n=round_n,
            from_id=buyer.id,
            kind=OfferKind.BID,
            price_per_gpu_hr=price,
            qty=buyer.job.qty,
            gpu_type=gpu_type,
            start=buyer.job.earliest_start,
            duration=buyer.job.duration,
            interruption_tolerance=buyer.job.interruption_tolerance,
            reasoning=f"deterministic buyer {self.buyer_id}: {reasoning}",
        )
        return AgentAction(new_offers=(offer,), reasoning=reasoning)


def _pick_target_gpu(buyer: Buyer, state: RunState) -> GPUType:
    """Cheapest acceptable GPU based on visible asks; fallback to most expensive acceptable."""
    visible = [
        ask for ask in state.seller_asks.values()
        if ask.gpu_type in buyer.job.acceptable_gpus
    ]
    if visible:
        return min(visible, key=lambda a: a.price_per_gpu_hr).gpu_type
    return max(buyer.job.acceptable_gpus, key=lambda g: GPU_BASE_RESERVE[g])


__all__ = ["AgentAction", "DeterministicBuyer", "DeterministicSeller"]
