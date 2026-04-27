"""Volume-discount-aware offer terms — the v0.3 OfferTerms schema.

The v0.2 6-dim tuple (price, qty, gpu_type, time, duration, interruption)
becomes a degenerate case of v0.3's ``OfferTerms`` (empty discount_schedule).
Pricing math goes through ``effective_price_per_gpu_hr`` and ``total_value_usd``
EVERYWHERE — viz, benchmarks, metrics. Do not re-derive the discount math.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from gpubid.domain.profiles import VolumeDiscountTier
from gpubid.schema import GPUType


OfferSide = Literal["buy", "sell"]


class OfferTerms(BaseModel):
    """A single offer (bid or ask) on the public board.

    Represents what an agent has put in front of its counterparty in one round.
    Volume-discount-aware: when ``discount_schedule`` is non-empty, the
    ``base_price_per_gpu_hr`` is the headline list price and tiers below it
    apply when the buyer commits to enough volume.

    Reasoning text is REQUIRED on every offer (even an empty string is
    explicit). Free-form reasoning is NEVER propagated cross-agent through
    the public board — it stays in the emitting agent's transcript only.
    """

    model_config = ConfigDict(frozen=True)

    offer_id: str
    round_n: int = Field(ge=0)
    side: OfferSide
    agent_id: str
    counterparty_id: Optional[str] = None  # populated once paired

    gpu_type: GPUType
    qty_gpus: int = Field(ge=1)
    duration_hours: float = Field(gt=0)
    start_slot: int = Field(ge=0, le=23)
    interruption_tolerance: Literal["none", "checkpoint_15min", "checkpoint_60min", "any"]

    base_price_per_gpu_hr: float = Field(gt=0)
    discount_schedule: tuple[VolumeDiscountTier, ...] = ()
    expires_after_round: Optional[int] = None

    reasoning: str = ""
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Canonical pricing helpers
# ---------------------------------------------------------------------------


def effective_price_per_gpu_hr(offer: OfferTerms, qty: int, duration: float) -> float:
    """Apply ``discount_schedule`` (if any) to ``base_price_per_gpu_hr`` for the
    given (qty, duration). Returns the per-GPU-hour rate the buyer would pay.

    A flat-pricing offer (empty schedule) returns ``base_price_per_gpu_hr``
    unchanged. Use this anywhere a price needs to be computed — viz, surplus
    accounting, MIP coefficients, headline metrics.
    """
    if not offer.discount_schedule:
        return offer.base_price_per_gpu_hr

    best_discount = 0.0
    for tier in reversed(offer.discount_schedule):
        if qty >= tier.min_qty_gpus and duration >= tier.min_duration_hours:
            best_discount = tier.discount_pct
            break

    return offer.base_price_per_gpu_hr * (1.0 - best_discount)


def total_value_usd(offer: OfferTerms, qty: int, duration: float) -> float:
    """Total contract value the buyer pays = effective price × qty × duration."""
    return effective_price_per_gpu_hr(offer, qty, duration) * qty * duration


__all__ = [
    "OfferSide",
    "OfferTerms",
    "effective_price_per_gpu_hr",
    "total_value_usd",
]
