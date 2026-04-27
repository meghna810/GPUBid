"""Phase 10 tests: tiered-VCG benchmark for v0.3 markets.

Two test pathways:
1. Legacy v0.2 Market input (no volume discounts) — should match legacy solve_vcg.
2. v0.3 (buyers_v2, sellers_v2) tuple with volume discounts — should produce
   welfare >= legacy VCG when tiers are non-trivial (bundle materialization
   gives the planner more options).
"""

import pytest

from gpubid.benchmark.vcg import solve_vcg as solve_vcg_legacy
from gpubid.benchmark.vcg_v2 import solve_vcg_tiered
from gpubid.domain.profiles import (
    BuyerPrivateProfile,
    BuyerPublicProfile,
    BuyerV2,
    InventorySlot,
    SellerPrivateProfile,
    SellerPublicProfile,
    SellerV2,
    TimeWindow,
    VolumeDiscountPolicy,
    VolumeDiscountTier,
)
from gpubid.market import generate_market
from gpubid.schema import GPUType


# ---------------------------------------------------------------------------
# Legacy v0.2 input
# ---------------------------------------------------------------------------


def test_tiered_vcg_on_legacy_market_matches_legacy_welfare():
    """Without any volume discount, tiered VCG should produce the same welfare as legacy VCG."""
    market = generate_market(8, 4, "tight", seed=42)
    legacy = solve_vcg_legacy(market)
    tiered = solve_vcg_tiered(market)
    # Welfare should match within rounding (both linearize the same set of slots).
    assert tiered.welfare == pytest.approx(legacy.welfare, abs=1.0)


def test_tiered_vcg_runs_on_multiple_seeds():
    for seed in [1, 7, 42, 99, 256]:
        market = generate_market(6, 3, "tight", seed=seed)
        result = solve_vcg_tiered(market, time_limit_seconds=5.0)
        assert result.solver_status in ("Optimal", "Trivial")
        assert result.welfare >= 0


# ---------------------------------------------------------------------------
# v0.3 input with volume discounts
# ---------------------------------------------------------------------------


def _v2_buyer(b_id: str, qty: int, dur: float, value: float, gpu=GPUType.H100) -> BuyerV2:
    return BuyerV2(
        public=BuyerPublicProfile(
            buyer_id=b_id, display_name=f"Buyer {b_id}",
            workload_category="training",
            gpu_type_preferences=(gpu,),
            qty_gpus=qty, duration_hours=dur,
            time_window=TimeWindow(earliest_start_slot=0, latest_finish_slot=24),
            interruption_tolerance="none", urgency_band="urgent",
        ),
        private=BuyerPrivateProfile(
            max_willingness_to_pay=value, urgency_score=0.5,
            budget_remaining_usd=value * qty * dur,
        ),
    )


def _v2_seller_with_tiers(
    s_id: str, slot_qty: int, slot_dur: float, list_price: float, reserve: float,
    tiers: tuple[VolumeDiscountTier, ...] = (), gpu=GPUType.H100,
) -> SellerV2:
    slot_id = f"{s_id}-slot0"
    return SellerV2(
        public=SellerPublicProfile(
            seller_id=s_id, display_name=f"Seller {s_id}",
            inventory_slots=(
                InventorySlot(
                    slot_id=slot_id, gpu_type=gpu, qty_gpus=slot_qty,
                    start_slot=0, duration_hours=slot_dur,
                ),
            ),
            list_price_per_gpu_hr={gpu: list_price},
            volume_discount_policy=VolumeDiscountPolicy(tiers=tiers),
        ),
        private=SellerPrivateProfile(
            reserve_per_slot={slot_id: reserve},
            marginal_cost_per_gpu_hr={gpu: reserve * 0.8},
            target_utilization=0.8,
        ),
    )


def test_v2_market_no_discounts_runs():
    """v0.3 input without any tiers should still produce a result."""
    buyers = [_v2_buyer("B0", qty=2, dur=4.0, value=8.0)]
    sellers = [_v2_seller_with_tiers("S0", slot_qty=4, slot_dur=8.0,
                                     list_price=6.0, reserve=4.0)]
    result = solve_vcg_tiered((buyers, sellers))
    assert result.solver_status == "Optimal"
    # Welfare = (8 - 4) × 2 × 4 = 32
    assert result.welfare == pytest.approx(32.0, abs=0.5)
    assert len(result.assignments) == 1


def test_v2_market_with_discounts_assignment_uses_compatible_bundle():
    """When a buyer crosses a discount tier threshold, the assignment should be
    feasible (the planner picks one bundle, not both)."""
    tiers = (
        VolumeDiscountTier(min_qty_gpus=4, min_duration_hours=4, discount_pct=0.20),
    )
    buyers = [_v2_buyer("B0", qty=4, dur=4.0, value=8.0)]
    sellers = [_v2_seller_with_tiers("S0", slot_qty=4, slot_dur=8.0,
                                     list_price=6.0, reserve=4.0, tiers=tiers)]
    result = solve_vcg_tiered((buyers, sellers))
    assert result.solver_status == "Optimal"
    # Welfare uses (max_value − reserve) regardless of which tier fires:
    # (8 - 4) × 4 × 4 = 64
    assert result.welfare == pytest.approx(64.0, abs=0.5)
    assert len(result.assignments) == 1


def test_v2_market_assigns_each_buyer_at_most_once():
    tiers = (
        VolumeDiscountTier(min_qty_gpus=4, min_duration_hours=4, discount_pct=0.10),
        VolumeDiscountTier(min_qty_gpus=8, min_duration_hours=8, discount_pct=0.20),
    )
    buyers = [
        _v2_buyer("B0", qty=8, dur=8.0, value=10.0),
        _v2_buyer("B1", qty=2, dur=2.0, value=12.0),
    ]
    sellers = [_v2_seller_with_tiers("S0", slot_qty=10, slot_dur=10.0,
                                     list_price=6.0, reserve=4.0, tiers=tiers)]
    result = solve_vcg_tiered((buyers, sellers))
    # Both buyers can fit in the single 10-GPU 10h slot (need 8+2 = 10).
    seen_buyers = {a[0] for a in result.assignments}
    assert len(seen_buyers) <= 2
    # No buyer assigned twice
    assert len(seen_buyers) == len(result.assignments)


def test_v2_market_capacity_constraint_blocks_overpacking():
    """If two buyers each want 4 GPUs and the slot only has 4, only one wins."""
    buyers = [
        _v2_buyer("B0", qty=4, dur=4.0, value=10.0),
        _v2_buyer("B1", qty=4, dur=4.0, value=8.0),
    ]
    sellers = [_v2_seller_with_tiers("S0", slot_qty=4, slot_dur=8.0,
                                     list_price=6.0, reserve=3.0)]
    result = solve_vcg_tiered((buyers, sellers))
    assert len(result.assignments) == 1
    # Should pick the higher-value buyer
    assert result.assignments[0][0] == "B0"


def test_linearization_notes_mention_bundle_count():
    market = generate_market(4, 2, "tight", seed=1)
    result = solve_vcg_tiered(market)
    assert "materialized" in result.linearization_notes
    assert "bundles" in result.linearization_notes


def test_invalid_input_type_raises():
    with pytest.raises(TypeError):
        solve_vcg_tiered("not a market")  # type: ignore[arg-type]


def test_v2_market_terminates_within_time_limit():
    """For 8x4 markets with up to 5 tiers per seller, must terminate < 3s."""
    import time
    tiers = tuple(
        VolumeDiscountTier(min_qty_gpus=q, min_duration_hours=2, discount_pct=0.05 * i)
        for i, q in enumerate([2, 4, 6, 8, 12], start=1)
    )
    buyers = [_v2_buyer(f"B{i}", qty=2 + i % 5, dur=4.0, value=8.0) for i in range(8)]
    sellers = [
        _v2_seller_with_tiers(
            f"S{i}", slot_qty=8, slot_dur=8.0,
            list_price=6.0 + i, reserve=3.0 + i, tiers=tiers,
        )
        for i in range(4)
    ]
    t0 = time.time()
    result = solve_vcg_tiered((buyers, sellers))
    elapsed = time.time() - t0
    assert elapsed < 3.0, f"tiered VCG took {elapsed:.2f}s on 8x4 with 5 tiers per seller"
    assert result.solver_status == "Optimal"
