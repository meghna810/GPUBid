"""Static posted-price baseline.

A fixed $/GPU-hr per GPU type (median seller reserve marked up 25%).
Buyers greedily accept any compatible slot priced at or below their value,
in deterministic order. The slot price for the deal is the posted price.

This is the "what current cloud markets give you" comparison: a single take-it
or-leave-it price, no flexibility on either side. Inefficiency comes from:
  - Buyers whose value is below the posted price walk away even when they
    could clear a low-reserve seller.
  - Sellers whose reserve is below the posted price take it but never get the
    upside that flexible pricing would extract from high-value buyers.
"""

from __future__ import annotations

from dataclasses import dataclass

from gpubid.benchmark.vcg import _is_compatible, compute_welfare
from gpubid.schema import (
    Deal,
    GPUType,
    InterruptionTolerance,
    Market,
)


POSTED_PRICE_MARKUP = 1.25


@dataclass(frozen=True)
class PostedPriceResult:
    posted_prices: dict[GPUType, float]
    deals: list[Deal]
    welfare: float


def median(xs: list[float]) -> float:
    s = sorted(xs)
    if not s:
        return 0.0
    n = len(s)
    if n % 2 == 1:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def compute_posted_prices(market: Market, markup: float = POSTED_PRICE_MARKUP) -> dict[GPUType, float]:
    """Per-GPU-type price = median seller reserve × markup."""
    by_gpu: dict[GPUType, list[float]] = {}
    for s in market.sellers:
        for slot in s.capacity_slots:
            by_gpu.setdefault(slot.gpu_type, []).append(slot.reserve_per_gpu_hr)
    return {gpu: round(median(rs) * markup, 2) for gpu, rs in by_gpu.items()}


def solve_posted_price(market: Market) -> PostedPriceResult:
    """Greedy allocation under a fixed per-GPU-type price.

    Iteration order is buyer.id then slot.id (alphabetical), so results are
    reproducible across runs.
    """
    posted = compute_posted_prices(market)

    # Track remaining capacity
    remaining = {sl.id: sl.qty for s in market.sellers for sl in s.capacity_slots}
    deals: list[Deal] = []
    used_buyers: set[str] = set()

    # Process buyers in deterministic order
    for buyer in sorted(market.buyers, key=lambda b: b.id):
        if buyer.id in used_buyers:
            continue
        # Find compatible slots whose posted price is within buyer's value AND
        # at or above the slot's reserve (otherwise the seller refuses).
        candidates = []
        for s in market.sellers:
            for slot in s.capacity_slots:
                if remaining[slot.id] < buyer.job.qty:
                    continue
                if not _is_compatible(buyer, slot):
                    continue
                price = posted.get(slot.gpu_type)
                if price is None:
                    continue
                if price > buyer.job.max_value_per_gpu_hr:
                    continue
                if price < slot.reserve_per_gpu_hr:
                    continue
                candidates.append((s.id, slot, price))

        if not candidates:
            continue

        # Pick the slot with the lowest posted price, then by slot id.
        candidates.sort(key=lambda t: (t[2], t[1].id))
        seller_id, slot, price = candidates[0]
        deal = Deal(
            id=f"pp-{buyer.id}-{slot.id}",
            round_n=0,
            buyer_id=buyer.id,
            seller_id=seller_id,
            slot_id=slot.id,
            qty=buyer.job.qty,
            price_per_gpu_hr=price,
            start=max(slot.start, buyer.job.earliest_start),
            duration=buyer.job.duration,
            gpu_type=slot.gpu_type,
            interruption_tolerance=InterruptionTolerance.NONE,
        )
        deals.append(deal)
        remaining[slot.id] -= buyer.job.qty
        used_buyers.add(buyer.id)

    welfare = compute_welfare(market, deals)
    return PostedPriceResult(posted_prices=posted, deals=deals, welfare=welfare)


__all__ = ["POSTED_PRICE_MARKUP", "PostedPriceResult", "compute_posted_prices", "solve_posted_price"]
