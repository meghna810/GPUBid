"""Pydantic models for offers, jobs, capacity, deals, and market state.

Every public class defines `_repr_html_` so a notebook cell can `display(obj)`
or simply put `obj` on the last line to render rich HTML cards instead of a
`repr` string. This is the single biggest lever that makes the notebook feel
intuitive instead of loggy.

The HTML rendering itself lives in `gpubid.viz.market_view` so it can also be
called directly (e.g. `render_market(m)` from a Gradio app or Next.js port).
"""

from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class GPUType(str, Enum):
    """GPU model. Three discrete tiers keep the bid language tractable."""

    H100 = "H100"
    A100 = "A100"
    L40S = "L40S"


class InterruptionTolerance(str, Enum):
    """How much disruption a buyer's job can tolerate (or a seller will offer)."""

    NONE = "none"
    CHECKPOINT = "checkpoint"
    INTERRUPTIBLE = "interruptible"


class OfferKind(str, Enum):
    ASK = "ask"
    BID = "bid"
    ACCEPT = "accept"
    COUNTER = "counter"


# ---------------------------------------------------------------------------
# Core market objects
# ---------------------------------------------------------------------------


class Job(BaseModel):
    """A buyer's compute job. `max_value_per_gpu_hr` is the buyer's *private* value."""

    model_config = ConfigDict(frozen=True)

    qty: int = Field(ge=1, description="Number of GPUs the job needs")
    acceptable_gpus: tuple[GPUType, ...] = Field(min_length=1)
    earliest_start: int = Field(ge=0, le=23)
    latest_finish: int = Field(ge=1, le=24)
    duration: int = Field(ge=1, description="Hours of compute")
    interruption_tolerance: InterruptionTolerance
    max_value_per_gpu_hr: float = Field(gt=0, description="Private — never appears on the public board")


class Buyer(BaseModel):
    """A buyer agent's persistent state."""

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    job: Job
    urgency: float = Field(ge=0.0, le=1.0, description="0 = patient, 1 = panic")

    def _repr_html_(self) -> str:
        from gpubid.viz.market_view import render_buyer_card
        return render_buyer_card(self)


class CapacitySlot(BaseModel):
    """A seller's capacity slot. `reserve_per_gpu_hr` is the seller's *private* reserve."""

    model_config = ConfigDict(frozen=True)

    id: str
    gpu_type: GPUType
    start: int = Field(ge=0, le=23)
    duration: int = Field(ge=1)
    qty: int = Field(ge=1)
    reserve_per_gpu_hr: float = Field(gt=0, description="Private — seller will not accept below this")

    @property
    def is_offpeak(self) -> bool:
        """Heuristic: midnight–6am or 10pm–midnight slots are off-peak."""
        return self.start < 6 or self.start >= 22


class Seller(BaseModel):
    """A seller agent's persistent state."""

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    capacity_slots: tuple[CapacitySlot, ...]

    def _repr_html_(self) -> str:
        from gpubid.viz.market_view import render_seller_card
        return render_seller_card(self)


class Market(BaseModel):
    """The synthetic market a single negotiation runs on. Reproducible from `seed`."""

    model_config = ConfigDict(frozen=True)

    id: str
    regime: Literal["tight", "slack"]
    seed: int
    buyers: tuple[Buyer, ...]
    sellers: tuple[Seller, ...]
    time_slots: int = 24

    @property
    def total_demand_gpu_hours(self) -> int:
        return sum(b.job.qty * b.job.duration for b in self.buyers)

    @property
    def total_supply_gpu_hours(self) -> int:
        return sum(slot.qty * slot.duration for s in self.sellers for slot in s.capacity_slots)

    @property
    def supply_demand_ratio(self) -> float:
        if self.total_demand_gpu_hours == 0:
            return float("inf")
        return self.total_supply_gpu_hours / self.total_demand_gpu_hours

    def _repr_html_(self) -> str:
        from gpubid.viz.market_view import render_market
        return render_market(self)


# ---------------------------------------------------------------------------
# Negotiation objects
# ---------------------------------------------------------------------------


class Offer(BaseModel):
    """A single ask, bid, accept, or counter posted in one round.

    Public components (everything except `reasoning`) appear on the board.
    Free-form `reasoning` is private to the agent that wrote it; it is *not*
    propagated to other agents in subsequent rounds, to prevent leakage of
    private values.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    round_n: int = Field(ge=1)
    from_id: str            # the buyer_id or seller_id who posted the offer
    kind: OfferKind
    slot_id: Optional[str] = None         # set when kind=ASK — points at the seller's CapacitySlot
    target_offer_id: Optional[str] = None  # set when kind=ACCEPT or COUNTER
    price_per_gpu_hr: float = Field(gt=0)
    qty: int = Field(ge=1)
    gpu_type: GPUType
    start: int = Field(ge=0, le=23)
    duration: int = Field(ge=1)
    interruption_tolerance: InterruptionTolerance
    reasoning: str = ""


class Deal(BaseModel):
    """A finalized agreement between a buyer and a seller for a specific slot."""

    model_config = ConfigDict(frozen=True)

    id: str
    round_n: int
    buyer_id: str
    seller_id: str
    slot_id: str
    qty: int
    price_per_gpu_hr: float
    start: int
    duration: int
    gpu_type: GPUType
    interruption_tolerance: InterruptionTolerance

    @property
    def total_value(self) -> float:
        return self.price_per_gpu_hr * self.qty * self.duration

    def _repr_html_(self) -> str:
        from gpubid.viz.market_view import render_deal_row
        return render_deal_row(self)


class BoardSnapshot(BaseModel):
    """Public state visible to every agent at the start of a round."""

    model_config = ConfigDict(frozen=True)

    round_n: int
    asks: tuple[Offer, ...] = ()
    bids: tuple[Offer, ...] = ()
    deals_so_far: tuple[Deal, ...] = ()
    active_buyer_ids: tuple[str, ...] = ()
    active_seller_ids: tuple[str, ...] = ()


__all__ = [
    "GPUType",
    "InterruptionTolerance",
    "OfferKind",
    "Job",
    "Buyer",
    "CapacitySlot",
    "Seller",
    "Market",
    "Offer",
    "Deal",
    "BoardSnapshot",
]
