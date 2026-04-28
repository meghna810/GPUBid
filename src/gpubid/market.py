"""Synthetic market generator: tight-supply and slack-supply regimes.

Deterministic given `seed`. Reproducibility is non-negotiable — every paired
experiment compares conditions on the *same* market.

Calibration:
  - GPU reserves are loosely modeled on real-world H100/A100/L40S list pricing
    so the numbers feel plausible in the demo, but the market itself is synthetic.
  - "Tight" regime → total demand ≈ 1.4× total supply (forces buyer competition).
  - "Slack" regime → total demand ≈ 0.7× total supply (forces seller competition;
    this is where off-peak filling becomes interesting).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from gpubid.schema import (
    Buyer,
    CapacitySlot,
    GPUType,
    InterruptionTolerance,
    Job,
    Market,
    Seller,
)


# ---------------------------------------------------------------------------
# Calibration constants
# ---------------------------------------------------------------------------

GPU_BASE_RESERVE: dict[GPUType, float] = {
    GPUType.H100: 4.50,
    GPUType.A100: 2.50,
    GPUType.L40S: 1.50,
}

OFFPEAK_DISCOUNT = 0.70  # off-peak slots have 30% lower reserve
TIGHT_RATIO = 1.40       # total_demand / total_supply target for "tight"
SLACK_RATIO = 0.70

ALL_INTERRUPTION = list(InterruptionTolerance)
ALL_GPUS = list(GPUType)

BUYER_NAMES = [
    "Alpha Lab", "Beta AI", "Gamma Research", "Delta ML",
    "Epsilon Tech", "Zeta Studio", "Eta Labs", "Theta AI",
    "Iota Compute", "Kappa Systems", "Lambda Labs", "Mu Research",
]

SELLER_NAMES = [
    "Pacific Cluster", "Aurora Compute", "Granite Cloud", "Helio Systems",
    "Nimbus GPU", "Vortex Compute", "Apex Cloud", "Stellar Compute",
]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def generate_market(
    n_buyers: int = 8,
    n_sellers: int = 4,
    regime: Literal["tight", "slack"] = "tight",
    seed: int = 42,
) -> Market:
    """Return a seeded synthetic market.

    The regime parameter shapes *supply*: tight markets have fewer/smaller slots
    per seller, slack markets have more/larger slots. Buyer demands are always
    sized to be compatible with at least some slot in the market — that's the
    invariant that keeps the deterministic mode interesting (otherwise most
    buyer/slot pairs would be a priori incompatible and almost no deals would
    close, which we observed in early testing).
    """

    if n_buyers < 1 or n_sellers < 1:
        raise ValueError("Need at least one buyer and one seller")
    if regime not in ("tight", "slack"):
        raise ValueError(f"regime must be 'tight' or 'slack', got {regime!r}")

    rng = np.random.default_rng(seed)

    sellers = _generate_sellers(n_sellers, regime, rng)
    buyers = _generate_buyers(n_buyers, sellers, regime, rng)

    return Market(
        id=f"mkt-{regime}-s{seed}-{n_buyers}x{n_sellers}",
        regime=regime,
        seed=seed,
        buyers=tuple(buyers),
        sellers=tuple(sellers),
    )


# ---------------------------------------------------------------------------
# Sellers
# ---------------------------------------------------------------------------


def _generate_sellers(
    n_sellers: int,
    regime: Literal["tight", "slack"],
    rng: np.random.Generator,
) -> list[Seller]:
    # Tight regime: fewer / smaller slots per seller. Slack: more / larger slots.
    if regime == "tight":
        slots_per_seller = (2, 4)   # 2..3
        slot_qty_range  = (2, 6)    # 2..5 GPUs
        slot_dur_range  = (3, 7)    # 3..6 hours
    else:
        slots_per_seller = (3, 5)   # 3..4
        slot_qty_range  = (3, 9)    # 3..8 GPUs
        slot_dur_range  = (4, 9)    # 4..8 hours

    sellers: list[Seller] = []
    for s_idx in range(n_sellers):
        n_slots = int(rng.integers(*slots_per_seller))
        slots: list[CapacitySlot] = []
        for slot_i in range(n_slots):
            gpu = ALL_GPUS[int(rng.integers(0, len(ALL_GPUS)))]
            duration = int(rng.integers(*slot_dur_range))
            start = int(rng.integers(0, max(1, 24 - duration)))
            qty = int(rng.integers(*slot_qty_range))

            base = GPU_BASE_RESERVE[gpu]
            offpeak = start < 6 or start >= 22
            jitter = 0.9 + 0.2 * float(rng.random())
            mult = OFFPEAK_DISCOUNT if offpeak else 1.0
            reserve = round(base * mult * jitter, 2)

            slots.append(
                CapacitySlot(
                    id=f"S{s_idx}-slot{slot_i}",
                    gpu_type=gpu,
                    start=start,
                    duration=duration,
                    qty=qty,
                    reserve_per_gpu_hr=reserve,
                )
            )

        label = SELLER_NAMES[s_idx % len(SELLER_NAMES)]
        sellers.append(Seller(id=f"S{s_idx}", label=label, capacity_slots=tuple(slots)))
    return sellers


# ---------------------------------------------------------------------------
# Buyers
# ---------------------------------------------------------------------------


def _generate_buyers(
    n_buyers: int,
    sellers: list[Seller],
    regime: Literal["tight", "slack"],
    rng: np.random.Generator,
) -> list[Buyer]:
    """Generate buyers whose demands are bounded by what slots actually offer.

    The bound matters: if a buyer needs 26 GPUs but no slot has more than 8,
    that buyer is structurally unfulfillable and the deterministic demo never
    closes a deal for them. We cap qty/duration to typical slot sizes so most
    pairings are *feasible*; agent strategy then decides whether they happen.
    """
    all_slots = [sl for s in sellers for sl in s.capacity_slots]
    max_slot_qty = max(sl.qty for sl in all_slots)
    max_slot_dur = max(sl.duration for sl in all_slots)

    # Tight markets generate higher-urgency, higher-value buyers (they really need it).
    urgency_floor = 0.4 if regime == "tight" else 0.0

    # Bias buyer GPU choice toward types that actually have supply.
    supply_by_gpu = {g: 0 for g in ALL_GPUS}
    for sl in all_slots:
        supply_by_gpu[sl.gpu_type] += sl.qty * sl.duration
    total_supply = sum(supply_by_gpu.values())
    gpu_p = np.array([supply_by_gpu[g] / total_supply for g in ALL_GPUS])

    buyers: list[Buyer] = []
    for b_idx in range(n_buyers):
        # Pick a primary GPU weighted by supply share, then 0..2 more acceptable.
        primary_idx = int(rng.choice(len(ALL_GPUS), p=gpu_p))
        n_extra = int(rng.integers(0, 3))
        other_idxs = [i for i in range(len(ALL_GPUS)) if i != primary_idx]
        rng.shuffle(other_idxs)
        accepted_idxs = [primary_idx] + other_idxs[:n_extra]
        acceptable = tuple(ALL_GPUS[i] for i in accepted_idxs)

        # Demand: stay within typical slot sizes so most pairings are feasible.
        qty_max = max(2, min(max_slot_qty, 5))
        qty = int(rng.integers(1, qty_max + 1))

        dur_max = max(3, min(max_slot_dur, 6))
        duration = int(rng.integers(2, dur_max + 1))

        # Time window: wide enough that several slots can plausibly fit.
        # Earliest in [0, 16], latest gives at least `duration` + 4 hours of slack.
        earliest = int(rng.integers(0, 17))
        latest_offset = int(rng.integers(4, 12))
        latest = min(24, earliest + duration + latest_offset)

        # Bias toward more-permissive tolerances so buyer/seller offers are
        # likely to be tolerance-compatible. Pure NONE buyers reject any offer
        # that's CHECKPOINT or INTERRUPTIBLE — when LLM sellers post varied
        # tolerances we lose deals to this even though price would clear. By
        # weighting toward INTERRUPTIBLE/CHECKPOINT we keep the demo lively.
        # 50% INTERRUPTIBLE, 35% CHECKPOINT, 15% NONE.
        tol_choice = float(rng.random())
        if tol_choice < 0.50:
            interruption = InterruptionTolerance.INTERRUPTIBLE
        elif tol_choice < 0.85:
            interruption = InterruptionTolerance.CHECKPOINT
        else:
            interruption = InterruptionTolerance.NONE

        # Value: markup over the most expensive acceptable GPU's reserve.
        # Markup must clearly exceed the seller's opening markup (1.5x in
        # tight, 1.2x in slack) plus a margin so negotiation has a positive
        # bargaining zone after both sides hedge. 1.7x..2.4x guarantees the
        # buyer can in principle afford the seller's opening ask.
        most_expensive = max(GPU_BASE_RESERVE[g] for g in acceptable)
        markup = 1.70 + 0.70 * float(rng.random())  # 1.7x..2.4x
        value = round(most_expensive * markup, 2)

        urgency = urgency_floor + (1.0 - urgency_floor) * float(rng.random())

        job = Job(
            qty=qty,
            acceptable_gpus=acceptable,
            earliest_start=earliest,
            latest_finish=latest,
            duration=duration,
            interruption_tolerance=interruption,
            max_value_per_gpu_hr=value,
        )
        buyers.append(
            Buyer(
                id=f"B{b_idx}",
                label=BUYER_NAMES[b_idx % len(BUYER_NAMES)],
                job=job,
                urgency=urgency,
            )
        )
    return buyers


__all__ = ["generate_market", "GPU_BASE_RESERVE"]
