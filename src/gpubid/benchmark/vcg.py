"""Welfare-optimal allocation via mixed-integer program (PuLP + CBC).

Used as the upper bound for the headline "% of VCG recovered" metric.
Tested against hand-computable 2x2 markets before being trusted.

Formulation:
  Variables:
    x[b, slot] ∈ {0, 1}    1 if buyer b is assigned to slot
  Constraints:
    Σ_slot x[b, slot] ≤ 1                                  (each buyer at most once)
    Σ_buyer qty[b] · x[b, slot] ≤ qty[slot]                (slot capacity in GPUs)
    x[b, slot] = 0 when (b, slot) is not compatible        (gpu, time, tolerance)
  Objective:
    maximize  Σ x[b, slot] · (value[b] − reserve[slot]) · qty[b] · duration[b]

The objective measures *welfare* (buyer value minus seller reserve cost).
Specific transfer prices are irrelevant for VCG welfare — only allocation matters.
"""

from __future__ import annotations

from dataclasses import dataclass

import pulp

from gpubid.engine.clearing import ask_satisfies_buyer
from gpubid.schema import (
    Buyer,
    CapacitySlot,
    Deal,
    GPUType,
    InterruptionTolerance,
    Market,
    Offer,
    OfferKind,
)


@dataclass(frozen=True)
class VCGResult:
    """Welfare-optimal allocation outcome."""

    assignments: list[tuple[str, str]]   # (buyer_id, slot_id) pairs
    welfare: float
    deals: list[Deal]                    # synthesized at the slot's reserve price
    solver_status: str


def _is_compatible(buyer: Buyer, slot: CapacitySlot) -> bool:
    """Wrap `ask_satisfies_buyer` for the case where we're checking the slot itself.

    We construct a hypothetical ASK at the slot's reserve to reuse the existing
    compatibility logic — including time window, GPU type, qty, and tolerance.
    """
    hypothetical = Offer(
        id="vcg-probe",
        round_n=1,
        from_id="vcg",
        kind=OfferKind.ASK,
        slot_id=slot.id,
        price_per_gpu_hr=slot.reserve_per_gpu_hr,
        qty=slot.qty,
        gpu_type=slot.gpu_type,
        start=slot.start,
        duration=slot.duration,
        # VCG considers the *welfare-maximizing* allocation: assume the seller would
        # offer their strictest available tolerance (NONE) so the buyer's compat check
        # is the binding constraint.
        interruption_tolerance=InterruptionTolerance.NONE,
    )
    return ask_satisfies_buyer(hypothetical, buyer, slot, slot.qty)


def solve_vcg(market: Market, time_limit_seconds: float = 10.0) -> VCGResult:
    """Solve the welfare-optimal MIP for the given market."""

    prob = pulp.LpProblem("gpubid_vcg", pulp.LpMaximize)

    # All slots flat-listed for indexing convenience
    all_slots = [(s.id, sl) for s in market.sellers for sl in s.capacity_slots]

    # Decision variables: only create x[b, slot] for compatible pairs (rest fixed at 0).
    x: dict[tuple[str, str], pulp.LpVariable] = {}
    for buyer in market.buyers:
        for seller_id, slot in all_slots:
            if _is_compatible(buyer, slot):
                x[(buyer.id, slot.id)] = pulp.LpVariable(
                    f"x_{buyer.id}_{slot.id}", lowBound=0, upBound=1, cat=pulp.LpBinary,
                )

    # Objective: maximize total welfare contribution.
    obj_terms = []
    for buyer in market.buyers:
        for seller_id, slot in all_slots:
            if (buyer.id, slot.id) in x:
                surplus_per_var = (
                    (buyer.job.max_value_per_gpu_hr - slot.reserve_per_gpu_hr)
                    * buyer.job.qty
                    * buyer.job.duration
                )
                obj_terms.append(surplus_per_var * x[(buyer.id, slot.id)])
    prob += pulp.lpSum(obj_terms) if obj_terms else 0

    # Constraint: each buyer assigned at most once
    for buyer in market.buyers:
        terms = [x[(buyer.id, slot.id)] for _, slot in all_slots if (buyer.id, slot.id) in x]
        if terms:
            prob += pulp.lpSum(terms) <= 1, f"one_assignment_{buyer.id}"

    # Constraint: slot capacity (sum of buyer GPU demands ≤ slot qty)
    for seller_id, slot in all_slots:
        terms = [
            buyer.job.qty * x[(buyer.id, slot.id)]
            for buyer in market.buyers
            if (buyer.id, slot.id) in x
        ]
        if terms:
            prob += pulp.lpSum(terms) <= slot.qty, f"capacity_{slot.id}"

    # Solve. CBC is bundled with PuLP — no Gurobi license needed.
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_seconds)
    status = prob.solve(solver)
    status_name = pulp.LpStatus[status]

    # Extract assignments
    assignments: list[tuple[str, str]] = []
    for (buyer_id, slot_id), var in x.items():
        if var.value() is not None and var.value() > 0.5:
            assignments.append((buyer_id, slot_id))

    welfare = pulp.value(prob.objective) if prob.objective is not None else 0.0
    welfare = float(welfare or 0.0)

    deals = _assignments_to_deals(market, assignments)
    return VCGResult(assignments=assignments, welfare=welfare, deals=deals, solver_status=status_name)


def _assignments_to_deals(market: Market, assignments: list[tuple[str, str]]) -> list[Deal]:
    """Synthesize Deals from VCG assignments. Price is the slot's reserve (lower bound).

    For VCG welfare comparisons we only care about the *allocation*, not the price.
    Synthesizing deals at the seller reserve gives the buyer all the surplus, which
    is conventional VCG (seller breaks even). Other transfer schemes give the same
    welfare.
    """
    deals: list[Deal] = []
    for buyer_id, slot_id in assignments:
        buyer = next(b for b in market.buyers if b.id == buyer_id)
        slot = next(sl for s in market.sellers for sl in s.capacity_slots if sl.id == slot_id)
        seller_id = next(s.id for s in market.sellers if any(sl.id == slot_id for sl in s.capacity_slots))
        deals.append(
            Deal(
                id=f"vcg-{buyer_id}-{slot_id}",
                round_n=0,                       # benchmark, no round
                buyer_id=buyer_id,
                seller_id=seller_id,
                slot_id=slot_id,
                qty=buyer.job.qty,
                price_per_gpu_hr=slot.reserve_per_gpu_hr,
                start=max(slot.start, buyer.job.earliest_start),
                duration=buyer.job.duration,
                gpu_type=slot.gpu_type,
                interruption_tolerance=InterruptionTolerance.NONE,
            )
        )
    return deals


def compute_welfare(market: Market, deals: list[Deal]) -> float:
    """Welfare contribution of an allocation: Σ (value − reserve) · qty · duration."""
    total = 0.0
    for d in deals:
        buyer = next((b for b in market.buyers if b.id == d.buyer_id), None)
        slot = next(
            (sl for s in market.sellers for sl in s.capacity_slots if sl.id == d.slot_id),
            None,
        )
        if buyer is None or slot is None:
            continue
        total += (buyer.job.max_value_per_gpu_hr - slot.reserve_per_gpu_hr) * d.qty * d.duration
    return total


__all__ = ["VCGResult", "solve_vcg", "compute_welfare"]
