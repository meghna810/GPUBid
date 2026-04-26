"""Tests for the round-runner, clearing engine, and deterministic agents."""

import pytest

from gpubid.engine.board import RunState
from gpubid.engine.clearing import (
    TOLERANCE_RANK,
    ask_satisfies_buyer,
    bid_satisfies_slot,
    buyer_accepts_tolerance,
    commit_deal,
    compute_concentration_cap,
)
from gpubid.engine.round_runner import make_deterministic_agents, run_rounds
from gpubid.market import generate_market
from gpubid.schema import (
    Buyer,
    CapacitySlot,
    Deal,
    GPUType,
    InterruptionTolerance,
    Job,
    Market,
    Offer,
    OfferKind,
    Seller,
)


# ---------------------------------------------------------------------------
# Tolerance ordering
# ---------------------------------------------------------------------------


def test_tolerance_ordering():
    assert TOLERANCE_RANK[InterruptionTolerance.NONE] == 0
    assert TOLERANCE_RANK[InterruptionTolerance.CHECKPOINT] == 1
    assert TOLERANCE_RANK[InterruptionTolerance.INTERRUPTIBLE] == 2


def test_buyer_accepts_tolerance():
    job = Job(
        qty=1, acceptable_gpus=(GPUType.H100,),
        earliest_start=0, latest_finish=10, duration=2,
        interruption_tolerance=InterruptionTolerance.CHECKPOINT,
        max_value_per_gpu_hr=5.0,
    )
    b = Buyer(id="B0", label="x", job=job, urgency=0.5)
    assert buyer_accepts_tolerance(b, InterruptionTolerance.NONE)
    assert buyer_accepts_tolerance(b, InterruptionTolerance.CHECKPOINT)
    assert not buyer_accepts_tolerance(b, InterruptionTolerance.INTERRUPTIBLE)


# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------


def _hand_built_market() -> Market:
    """Tiny 1-buyer, 1-seller market we can hand-check."""
    job = Job(
        qty=2, acceptable_gpus=(GPUType.H100,),
        earliest_start=8, latest_finish=14, duration=4,
        interruption_tolerance=InterruptionTolerance.NONE,
        max_value_per_gpu_hr=8.0,
    )
    buyer = Buyer(id="B0", label="Acme", job=job, urgency=0.5)
    slot = CapacitySlot(
        id="S0-slot0", gpu_type=GPUType.H100, start=8, duration=6, qty=4,
        reserve_per_gpu_hr=4.0,
    )
    seller = Seller(id="S0", label="Cloud A", capacity_slots=(slot,))
    return Market(id="hand", regime="tight", seed=0, buyers=(buyer,), sellers=(seller,))


def test_ask_satisfies_buyer_compatible_case():
    market = _hand_built_market()
    buyer = market.buyers[0]
    slot = market.sellers[0].capacity_slots[0]
    ask = Offer(
        id="ask1", round_n=1, from_id="S0", kind=OfferKind.ASK, slot_id=slot.id,
        price_per_gpu_hr=5.0, qty=4, gpu_type=GPUType.H100,
        start=8, duration=6, interruption_tolerance=InterruptionTolerance.NONE,
    )
    assert ask_satisfies_buyer(ask, buyer, slot, remaining_qty=4)


def test_ask_satisfies_buyer_rejects_wrong_gpu():
    market = _hand_built_market()
    buyer = market.buyers[0]
    slot = market.sellers[0].capacity_slots[0]
    ask = Offer(
        id="ask1", round_n=1, from_id="S0", kind=OfferKind.ASK, slot_id=slot.id,
        price_per_gpu_hr=5.0, qty=4, gpu_type=GPUType.A100,  # buyer wants H100 only
        start=8, duration=6, interruption_tolerance=InterruptionTolerance.NONE,
    )
    assert not ask_satisfies_buyer(ask, buyer, slot, remaining_qty=4)


def test_ask_satisfies_buyer_rejects_outside_window():
    market = _hand_built_market()
    buyer = market.buyers[0]
    slot = market.sellers[0].capacity_slots[0]
    ask = Offer(
        id="ask1", round_n=1, from_id="S0", kind=OfferKind.ASK, slot_id=slot.id,
        price_per_gpu_hr=5.0, qty=4, gpu_type=GPUType.H100,
        start=20, duration=6, interruption_tolerance=InterruptionTolerance.NONE,  # too late
    )
    assert not ask_satisfies_buyer(ask, buyer, slot, remaining_qty=4)


def test_bid_satisfies_slot():
    market = _hand_built_market()
    slot = market.sellers[0].capacity_slots[0]
    bid = Offer(
        id="b1", round_n=1, from_id="B0", kind=OfferKind.BID,
        price_per_gpu_hr=6.0, qty=2, gpu_type=GPUType.H100,
        start=8, duration=4, interruption_tolerance=InterruptionTolerance.NONE,
    )
    assert bid_satisfies_slot(bid, slot, remaining_qty=4)


# ---------------------------------------------------------------------------
# Capacity decrement
# ---------------------------------------------------------------------------


def test_commit_deal_decrements_capacity():
    market = _hand_built_market()
    state = RunState.initial(market)
    deal = Deal(
        id="d1", round_n=1, buyer_id="B0", seller_id="S0", slot_id="S0-slot0",
        qty=2, price_per_gpu_hr=5.0, start=8, duration=4,
        gpu_type=GPUType.H100, interruption_tolerance=InterruptionTolerance.NONE,
    )
    commit_deal(state, deal)
    assert state.slot_remaining_qty["S0-slot0"] == 2
    assert "B0" not in state.active_buyer_ids
    assert state.deals == [deal]


def test_commit_deal_exhausts_seller():
    """Buyer takes all remaining GPUs on seller's only slot → seller becomes inactive."""
    market = _hand_built_market()
    state = RunState.initial(market)
    deal = Deal(
        id="d1", round_n=1, buyer_id="B0", seller_id="S0", slot_id="S0-slot0",
        qty=4, price_per_gpu_hr=5.0, start=8, duration=4,
        gpu_type=GPUType.H100, interruption_tolerance=InterruptionTolerance.NONE,
    )
    commit_deal(state, deal)
    assert state.slot_remaining_qty["S0-slot0"] == 0
    assert "S0" not in state.active_seller_ids


# ---------------------------------------------------------------------------
# Concentration cap
# ---------------------------------------------------------------------------


def test_concentration_cap_value():
    market = generate_market(8, 4, "tight", seed=1)
    cap_30pct = compute_concentration_cap(market, 0.30)
    assert cap_30pct == int(market.total_supply_gpu_hours * 0.30)


# ---------------------------------------------------------------------------
# Round runner — full negotiation
# ---------------------------------------------------------------------------


def test_run_rounds_yields_max_rounds_at_most():
    market = generate_market(6, 3, "tight", seed=42)
    buyers, sellers = make_deterministic_agents(market)
    snaps = list(run_rounds(market, buyers, sellers, max_rounds=5))
    assert 1 <= len(snaps) <= 5
    assert all(snap.round_n == i + 1 for i, snap in enumerate(snaps))


def test_run_rounds_terminates_when_all_buyers_filled():
    """Slack supply with simple compatibility should fill everyone fast."""
    market = generate_market(4, 4, "slack", seed=42)
    buyers, sellers = make_deterministic_agents(market)
    snaps = list(run_rounds(market, buyers, sellers, max_rounds=10))
    assert snaps[-1].is_final


def test_no_deal_below_reserve():
    """Across many runs, no committed deal should price below the seller's slot reserve."""
    for seed in range(20):
        market = generate_market(8, 4, "tight", seed=seed)
        buyers, sellers = make_deterministic_agents(market)
        snaps = list(run_rounds(market, buyers, sellers, max_rounds=5))
        for snap in snaps:
            for deal in snap.new_deals:
                slot = next(
                    sl for s in market.sellers for sl in s.capacity_slots if sl.id == deal.slot_id
                )
                assert deal.price_per_gpu_hr >= slot.reserve_per_gpu_hr, (
                    f"seed {seed}: deal {deal.id} priced ${deal.price_per_gpu_hr} below "
                    f"reserve ${slot.reserve_per_gpu_hr}"
                )


def test_no_deal_above_buyer_value():
    """No committed deal should price above the buyer's max willingness-to-pay."""
    for seed in range(20):
        market = generate_market(8, 4, "tight", seed=seed)
        buyers, sellers = make_deterministic_agents(market)
        snaps = list(run_rounds(market, buyers, sellers, max_rounds=5))
        for snap in snaps:
            for deal in snap.new_deals:
                buyer = next(b for b in market.buyers if b.id == deal.buyer_id)
                assert deal.price_per_gpu_hr <= buyer.job.max_value_per_gpu_hr, (
                    f"seed {seed}: deal {deal.id} priced ${deal.price_per_gpu_hr} above "
                    f"buyer max ${buyer.job.max_value_per_gpu_hr}"
                )


def test_capacity_never_oversold():
    """Across many runs, no slot should be oversold past its initial qty."""
    for seed in range(10):
        market = generate_market(8, 4, "tight", seed=seed)
        buyers, sellers = make_deterministic_agents(market)
        snaps = list(run_rounds(market, buyers, sellers, max_rounds=5))
        # Compare per-slot total qty sold to slot's initial qty.
        slot_qty: dict[str, int] = {}
        for s in market.sellers:
            for sl in s.capacity_slots:
                slot_qty[sl.id] = sl.qty
        sold: dict[str, int] = {sid: 0 for sid in slot_qty}
        for d in snaps[-1].all_deals:
            sold[d.slot_id] += d.qty
        for sid, total in sold.items():
            assert total <= slot_qty[sid], f"seed {seed}: slot {sid} oversold"


def test_tight_market_has_some_deals():
    """Sanity check — deterministic agents should close at least *some* deals."""
    market = generate_market(8, 4, "tight", seed=7)
    buyers, sellers = make_deterministic_agents(market)
    snaps = list(run_rounds(market, buyers, sellers, max_rounds=5))
    final = snaps[-1]
    assert len(final.all_deals) > 0, "expected at least one deal in 5 rounds"


def test_reasoning_redacted_from_public_snapshot():
    """`reasoning` should not be visible to other agents through the public board."""
    market = generate_market(4, 2, "tight", seed=3)
    state = RunState.initial(market)
    fake = Offer(
        id="x", round_n=1, from_id="B0", kind=OfferKind.BID,
        price_per_gpu_hr=5.0, qty=1, gpu_type=GPUType.H100,
        start=0, duration=2, interruption_tolerance=InterruptionTolerance.NONE,
        reasoning="MY MAX IS $9.99 PLEASE DON'T LEAK",
    )
    state.buyer_bids["B0"] = fake
    snap = state.public_snapshot(round_n=2)
    for o in snap.bids:
        assert "MY MAX" not in o.reasoning, "reasoning should be redacted from public board"


# ---------------------------------------------------------------------------
# Concentration cap binds
# ---------------------------------------------------------------------------


def test_concentration_cap_can_block_deals():
    """With a tiny cap (1%), almost no buyer can win — total deals should drop."""
    market = generate_market(8, 4, "slack", seed=11)
    buyers, sellers = make_deterministic_agents(market)
    snaps_uncapped = list(run_rounds(market, buyers, sellers, max_rounds=5, concentration_cap_pct=None))
    snaps_capped = list(run_rounds(market, buyers, sellers, max_rounds=5, concentration_cap_pct=0.01))
    # With a 1% cap the cap should bind *at least* sometimes.
    # We can't guarantee strictly fewer deals (some buyers fit under 1% naturally) but we
    # can at least say capped <= uncapped in deal count.
    assert len(snaps_capped[-1].all_deals) <= len(snaps_uncapped[-1].all_deals)
