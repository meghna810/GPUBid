"""Two-sided information asymmetry as a first-class concept.

Each agent carries a public profile (visible to its counterparty) and a private
profile (hidden). The market generator emits both; agent runtime code must
respect the asymmetry — only public profiles flow across the protocol boundary.

Volume-discount policies live on the seller's public profile because they are
publicly advertised and what makes offers non-directly-comparable. The
``applicable_discount`` helper is the canonical lookup.

All profile types are frozen Pydantic models. Equality is structural.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gpubid.schema import GPUType


WorkloadCategory = Literal[
    "training",
    "fine_tuning",
    "inference_batch",
    "inference_realtime",
    "evaluation_sweep",
]

InterruptionToleranceSemantic = Literal[
    "none",                  # job must run to completion uninterrupted
    "checkpoint_15min",      # can pause if checkpoint flushed within 15 min
    "checkpoint_60min",      # can pause if checkpoint flushed within 60 min
    "any",                   # spot-style; OK with arbitrary interruption
]

UrgencyBand = Literal["routine", "soon", "urgent"]

CompetingDemandSignal = Literal["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TimeWindow(BaseModel):
    """Slot-index time window (hours-of-day, integer slots, [earliest, latest))."""

    model_config = ConfigDict(frozen=True)

    earliest_start_slot: int = Field(ge=0, le=23)
    latest_finish_slot: int = Field(ge=1, le=24)

    @model_validator(mode="after")
    def _check_range(self) -> "TimeWindow":
        if self.latest_finish_slot <= self.earliest_start_slot:
            raise ValueError(
                f"latest_finish_slot ({self.latest_finish_slot}) must be > "
                f"earliest_start_slot ({self.earliest_start_slot})"
            )
        return self


class FallbackOption(BaseModel):
    """What a buyer falls back to if no deal closes — used in private profile only."""

    model_config = ConfigDict(frozen=True)

    provider: str
    est_price_per_gpu_hr: float = Field(gt=0)
    friction_cost_usd: float = Field(ge=0)


# ---------------------------------------------------------------------------
# Volume discount policy (lives on seller public profile; v0.3 headline feature)
# ---------------------------------------------------------------------------


class VolumeDiscountTier(BaseModel):
    """One tier in a seller's volume-discount schedule.

    A tier triggers when the buyer commits at least ``min_qty_gpus`` GPUs AND
    at least ``min_duration_hours`` hours. The discount is a fraction of the
    list price (0.0 = no discount, 0.25 = 25% off list).
    """

    model_config = ConfigDict(frozen=True)

    min_qty_gpus: int = Field(ge=1)
    min_duration_hours: float = Field(gt=0)
    discount_pct: float = Field(ge=0.0, le=1.0)


class VolumeDiscountPolicy(BaseModel):
    """A seller's publicly advertised tiered pricing policy.

    ``tiers`` is sorted by ``min_qty_gpus`` ascending. Empty list = flat
    pricing (no discount). When ``is_negotiable`` is True, the seller agent
    may propose a custom tier mid-negotiation that isn't in the policy yet.
    """

    model_config = ConfigDict(frozen=True)

    tiers: tuple[VolumeDiscountTier, ...] = ()
    is_negotiable: bool = False

    @field_validator("tiers")
    @classmethod
    def _tiers_sorted_ascending(
        cls, v: tuple[VolumeDiscountTier, ...]
    ) -> tuple[VolumeDiscountTier, ...]:
        if v and any(v[i].min_qty_gpus > v[i + 1].min_qty_gpus for i in range(len(v) - 1)):
            raise ValueError("tiers must be sorted by min_qty_gpus ascending")
        return v

    def applicable_discount(self, qty: int, duration: float) -> float:
        """Return the best discount fraction that applies to (qty, duration). 0 if none.

        Walks tiers from highest qty threshold down; returns the first whose
        ``min_qty_gpus`` and ``min_duration_hours`` are both satisfied.
        """
        for tier in reversed(self.tiers):
            if qty >= tier.min_qty_gpus and duration >= tier.min_duration_hours:
                return tier.discount_pct
        return 0.0


# ---------------------------------------------------------------------------
# Inventory (seller side)
# ---------------------------------------------------------------------------


class InventorySlot(BaseModel):
    """One available capacity slot a seller can sell from."""

    model_config = ConfigDict(frozen=True)

    slot_id: str
    gpu_type: GPUType
    qty_gpus: int = Field(ge=1)
    start_slot: int = Field(ge=0, le=23)
    duration_hours: float = Field(gt=0)


# ---------------------------------------------------------------------------
# Buyer profiles
# ---------------------------------------------------------------------------


class BuyerPublicProfile(BaseModel):
    """The slice of buyer state that ALL sellers see at broadcast.

    Notable absences (these live on `BuyerPrivateProfile`): exact willingness-to-pay,
    exact urgency score, internal deadline, fallback options, budget remaining,
    business-context narrative.
    """

    model_config = ConfigDict(frozen=True)

    buyer_id: str
    display_name: str
    workload_category: WorkloadCategory
    gpu_type_preferences: tuple[GPUType, ...] = Field(min_length=1)
    qty_gpus: int = Field(ge=1)
    duration_hours: float = Field(gt=0)
    time_window: TimeWindow
    interruption_tolerance: InterruptionToleranceSemantic
    urgency_band: UrgencyBand


class BuyerPrivateProfile(BaseModel):
    """Hidden from sellers. The notebook may render this for inspection only."""

    model_config = ConfigDict(frozen=True)

    max_willingness_to_pay: float = Field(gt=0)
    urgency_score: float = Field(ge=0.0, le=1.0)
    internal_deadline_slot: Optional[int] = Field(default=None, ge=0, le=24)
    fallback_options: tuple[FallbackOption, ...] = ()
    budget_remaining_usd: float = Field(gt=0)
    business_context_summary: str = ""


# ---------------------------------------------------------------------------
# Seller profiles
# ---------------------------------------------------------------------------


class SellerPublicProfile(BaseModel):
    """The slice of seller state that ALL buyers see at broadcast."""

    model_config = ConfigDict(frozen=True)

    seller_id: str
    display_name: str
    inventory_slots: tuple[InventorySlot, ...]
    list_price_per_gpu_hr: dict[GPUType, float]
    volume_discount_policy: VolumeDiscountPolicy = VolumeDiscountPolicy()
    min_commitment_hours: float = Field(default=1.0, gt=0)


class SellerPrivateProfile(BaseModel):
    """Hidden from buyers. ``reserve_per_slot`` keys reference ``InventorySlot.slot_id``."""

    model_config = ConfigDict(frozen=True)

    reserve_per_slot: dict[str, float]
    marginal_cost_per_gpu_hr: dict[GPUType, float]
    target_utilization: float = Field(ge=0.0, le=1.0)
    competing_demand_signal: CompetingDemandSignal = "medium"


# ---------------------------------------------------------------------------
# Composite types — Buyer / Seller
# ---------------------------------------------------------------------------


class BuyerV2(BaseModel):
    """v0.3 buyer: public + private profiles. The notebook viz reads ``public``."""

    model_config = ConfigDict(frozen=True)

    public: BuyerPublicProfile
    private: BuyerPrivateProfile

    # Backward-compat shims — TODO(v0.4): remove once viz migrates to public.*
    @property
    def id(self) -> str:
        return self.public.buyer_id

    @property
    def label(self) -> str:
        return self.public.display_name

    @property
    def gpu_type(self) -> GPUType:
        """Convenience: the primary preferred GPU type."""
        return self.public.gpu_type_preferences[0]


class SellerV2(BaseModel):
    """v0.3 seller: public + private profiles."""

    model_config = ConfigDict(frozen=True)

    public: SellerPublicProfile
    private: SellerPrivateProfile

    # Backward-compat shims — TODO(v0.4): remove once viz migrates to public.*
    @property
    def id(self) -> str:
        return self.public.seller_id

    @property
    def label(self) -> str:
        return self.public.display_name


__all__ = [
    "WorkloadCategory",
    "InterruptionToleranceSemantic",
    "UrgencyBand",
    "CompetingDemandSignal",
    "TimeWindow",
    "FallbackOption",
    "VolumeDiscountTier",
    "VolumeDiscountPolicy",
    "InventorySlot",
    "BuyerPublicProfile",
    "BuyerPrivateProfile",
    "SellerPublicProfile",
    "SellerPrivateProfile",
    "BuyerV2",
    "SellerV2",
]
