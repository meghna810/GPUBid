"""Phase 13 — Post-deal regret signals (exploratory; for class discussion).

Synthetic per-side satisfaction scoring. In a real marketplace this would
come from post-deal user feedback; here we simulate it from private profiles
to demonstrate the framing.

The demo of correction injects a calibration line into the next-run system
prompt for systematically high-regret sellers, then re-runs to show
directional improvement. This is prompt-time calibration, NOT a real RLHF loop.

Implemented; depends on Phase 12 outputs to plot before/after.
"""

from __future__ import annotations

from dataclasses import dataclass

from gpubid.domain.profiles import BuyerV2, SellerV2
from gpubid.schema import Deal


@dataclass(frozen=True)
class RegretSignal:
    buyer_score: float       # 0 = no regret, 1 = severe regret
    seller_score: float
    notes: str = ""


def synthesize_regret(deal: Deal, buyer: BuyerV2, seller: SellerV2) -> RegretSignal:
    """Compute synthetic regret signals for one deal.

    Heuristic per spec §14.2:
      - Buyer regret rises if effective_price > 0.85 * max_willingness_to_pay (overpaid).
      - Seller regret rises if effective_price < reserve * 1.10 (left money on table).
    """
    eff_price = deal.price_per_gpu_hr
    buyer_max = buyer.private.max_willingness_to_pay
    reserve = seller.private.reserve_per_slot.get(deal.slot_id, eff_price)

    # Buyer regret — 0 below 60% of max, ramps to 1 at max.
    buyer_score = max(0.0, min(1.0, (eff_price - 0.60 * buyer_max) / (0.40 * buyer_max)))

    # Seller regret — 0 above 1.6x reserve, ramps to 1 at exactly reserve.
    seller_floor = reserve
    seller_ceiling = reserve * 1.6
    if seller_ceiling <= seller_floor:
        seller_score = 0.0
    else:
        seller_score = max(0.0, min(1.0, (seller_ceiling - eff_price) / (seller_ceiling - seller_floor)))

    notes = ""
    if buyer_score > 0.7:
        notes += "buyer overpaid; "
    if seller_score > 0.7:
        notes += "seller left money on table"
    return RegretSignal(buyer_score=buyer_score, seller_score=seller_score, notes=notes.strip("; "))


def calibration_line_for_seller(avg_seller_regret: float) -> str | None:
    """Return a one-line calibration prompt to inject for systematically high regret.

    Returns None when avg_seller_regret < 0.4 (no intervention needed).
    """
    if avg_seller_regret < 0.4:
        return None
    return (
        "Recent runs suggest you have been accepting prices close to your reserve. "
        "Hold firmer in this run."
    )


__all__ = ["RegretSignal", "synthesize_regret", "calibration_line_for_seller"]
