"""Surplus, % of VCG recovered, Gini of buyer welfare, off-peak utilization."""

from __future__ import annotations

from dataclasses import dataclass

from gpubid.benchmark.vcg import compute_welfare
from gpubid.schema import Deal, Market


@dataclass(frozen=True)
class RunMetrics:
    """Aggregate metrics for a single negotiation outcome."""

    n_deals: int
    n_buyers_filled: int
    welfare: float
    transacted_value: float        # sum of buyer-paid totals (price × qty × dur)
    seller_revenue: float          # = transacted_value (one-sided market)
    avg_clearing_price: float
    offpeak_utilization: float     # fraction of off-peak slot capacity used
    gini_buyer_welfare: float


def gini(values: list[float]) -> float:
    """Standard Gini coefficient. Returns 0 for empty inputs."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    n = len(sorted_v)
    cum = 0.0
    for i, v in enumerate(sorted_v, start=1):
        cum += i * v
    total = sum(sorted_v)
    if total == 0:
        return 0.0
    return (2 * cum) / (n * total) - (n + 1) / n


def offpeak_utilization(market: Market, deals: list[Deal]) -> float:
    """Fraction of off-peak slot GPU-hour capacity that ended up in deals."""
    offpeak_slots = {
        sl.id: sl.qty * sl.duration
        for s in market.sellers
        for sl in s.capacity_slots
        if sl.is_offpeak
    }
    if not offpeak_slots:
        return 0.0
    total_offpeak_capacity = sum(offpeak_slots.values())
    used = 0
    for d in deals:
        if d.slot_id in offpeak_slots:
            used += d.qty * d.duration
    return used / total_offpeak_capacity if total_offpeak_capacity else 0.0


def per_buyer_welfare(market: Market, deals: list[Deal]) -> dict[str, float]:
    """Each buyer's surplus = (value − price) · qty · duration. Unmatched = 0."""
    welfare = {b.id: 0.0 for b in market.buyers}
    for d in deals:
        buyer = next((b for b in market.buyers if b.id == d.buyer_id), None)
        if buyer is None:
            continue
        welfare[d.buyer_id] = (buyer.job.max_value_per_gpu_hr - d.price_per_gpu_hr) * d.qty * d.duration
    return welfare


def compute_metrics(market: Market, deals: list[Deal]) -> RunMetrics:
    n_deals = len(deals)
    n_buyers_filled = len({d.buyer_id for d in deals})
    welfare = compute_welfare(market, deals)
    transacted = sum(d.price_per_gpu_hr * d.qty * d.duration for d in deals)
    avg_price = (
        sum(d.price_per_gpu_hr for d in deals) / n_deals if n_deals else 0.0
    )
    util = offpeak_utilization(market, deals)
    buyer_welfare = list(per_buyer_welfare(market, deals).values())
    return RunMetrics(
        n_deals=n_deals,
        n_buyers_filled=n_buyers_filled,
        welfare=welfare,
        transacted_value=transacted,
        seller_revenue=transacted,
        avg_clearing_price=avg_price,
        offpeak_utilization=util,
        gini_buyer_welfare=gini(buyer_welfare),
    )


__all__ = [
    "RunMetrics",
    "gini",
    "offpeak_utilization",
    "per_buyer_welfare",
    "compute_metrics",
]
