"""Phase 2 tests: public/private profile schema, volume discount policy."""

import pytest

from gpubid.domain.profiles import (
    BuyerPrivateProfile,
    BuyerPublicProfile,
    BuyerV2,
    FallbackOption,
    InventorySlot,
    SellerPrivateProfile,
    SellerPublicProfile,
    SellerV2,
    TimeWindow,
    VolumeDiscountPolicy,
    VolumeDiscountTier,
)
from gpubid.schema import GPUType


# ---------------------------------------------------------------------------
# TimeWindow validation
# ---------------------------------------------------------------------------


def test_time_window_valid():
    tw = TimeWindow(earliest_start_slot=8, latest_finish_slot=16)
    assert tw.earliest_start_slot == 8


def test_time_window_rejects_inverted_range():
    with pytest.raises(ValueError, match="must be"):
        TimeWindow(earliest_start_slot=10, latest_finish_slot=8)


def test_time_window_rejects_zero_width():
    with pytest.raises(ValueError):
        TimeWindow(earliest_start_slot=10, latest_finish_slot=10)


# ---------------------------------------------------------------------------
# Volume discount policy
# ---------------------------------------------------------------------------


def test_empty_policy_returns_zero_discount():
    pol = VolumeDiscountPolicy()
    assert pol.applicable_discount(qty=100, duration=999) == 0.0


def test_single_tier_applies_above_thresholds():
    pol = VolumeDiscountPolicy(
        tiers=(VolumeDiscountTier(min_qty_gpus=4, min_duration_hours=4, discount_pct=0.10),),
    )
    assert pol.applicable_discount(qty=4, duration=4.0) == 0.10
    assert pol.applicable_discount(qty=10, duration=10.0) == 0.10


def test_single_tier_does_not_apply_below_thresholds():
    pol = VolumeDiscountPolicy(
        tiers=(VolumeDiscountTier(min_qty_gpus=4, min_duration_hours=4, discount_pct=0.10),),
    )
    assert pol.applicable_discount(qty=3, duration=4.0) == 0.0
    assert pol.applicable_discount(qty=4, duration=3.9) == 0.0


def test_multi_tier_picks_best_applicable():
    pol = VolumeDiscountPolicy(
        tiers=(
            VolumeDiscountTier(min_qty_gpus=4, min_duration_hours=4, discount_pct=0.05),
            VolumeDiscountTier(min_qty_gpus=8, min_duration_hours=4, discount_pct=0.15),
            VolumeDiscountTier(min_qty_gpus=16, min_duration_hours=8, discount_pct=0.25),
        ),
    )
    # qty 4: only first tier applies
    assert pol.applicable_discount(qty=4, duration=4.0) == 0.05
    # qty 8: first and second apply; second wins
    assert pol.applicable_discount(qty=8, duration=4.0) == 0.15
    # qty 16, dur 8: all three apply; third wins
    assert pol.applicable_discount(qty=16, duration=8.0) == 0.25
    # qty 16 but only dur 4: third tier needs dur 8 → fall back to second
    assert pol.applicable_discount(qty=16, duration=4.0) == 0.15


def test_tiers_must_be_sorted_ascending():
    with pytest.raises(ValueError, match="sorted"):
        VolumeDiscountPolicy(
            tiers=(
                VolumeDiscountTier(min_qty_gpus=8, min_duration_hours=4, discount_pct=0.15),
                VolumeDiscountTier(min_qty_gpus=4, min_duration_hours=4, discount_pct=0.05),
            ),
        )


def test_discount_pct_bounded_zero_to_one():
    with pytest.raises(ValueError):
        VolumeDiscountTier(min_qty_gpus=1, min_duration_hours=1, discount_pct=1.5)
    with pytest.raises(ValueError):
        VolumeDiscountTier(min_qty_gpus=1, min_duration_hours=1, discount_pct=-0.1)


# ---------------------------------------------------------------------------
# Buyer public profile
# ---------------------------------------------------------------------------


def _sample_buyer_public(**overrides) -> BuyerPublicProfile:
    base = dict(
        buyer_id="B0", display_name="Acme Lab",
        workload_category="training",
        gpu_type_preferences=(GPUType.H100, GPUType.A100),
        qty_gpus=8, duration_hours=6.0,
        time_window=TimeWindow(earliest_start_slot=8, latest_finish_slot=18),
        interruption_tolerance="none", urgency_band="urgent",
    )
    base.update(overrides)
    return BuyerPublicProfile(**base)


def test_buyer_public_minimal_valid():
    p = _sample_buyer_public()
    assert p.qty_gpus == 8


def test_buyer_public_rejects_negative_qty():
    with pytest.raises(ValueError):
        _sample_buyer_public(qty_gpus=0)


def test_buyer_public_requires_at_least_one_gpu_preference():
    with pytest.raises(ValueError):
        _sample_buyer_public(gpu_type_preferences=())


# ---------------------------------------------------------------------------
# Buyer private profile
# ---------------------------------------------------------------------------


def test_buyer_private_minimal_valid():
    p = BuyerPrivateProfile(
        max_willingness_to_pay=5.0, urgency_score=0.7,
        budget_remaining_usd=1000.0,
    )
    assert p.fallback_options == ()  # empty allowed
    assert p.business_context_summary == ""


def test_buyer_private_with_fallbacks():
    p = BuyerPrivateProfile(
        max_willingness_to_pay=5.0, urgency_score=0.5, budget_remaining_usd=500,
        fallback_options=(
            FallbackOption(provider="aws", est_price_per_gpu_hr=6.0, friction_cost_usd=200),
        ),
    )
    assert len(p.fallback_options) == 1


def test_buyer_private_urgency_score_bounds():
    with pytest.raises(ValueError):
        BuyerPrivateProfile(
            max_willingness_to_pay=5.0, urgency_score=1.5, budget_remaining_usd=100,
        )


# ---------------------------------------------------------------------------
# Seller profiles
# ---------------------------------------------------------------------------


def _sample_seller_v2() -> SellerV2:
    inv = (
        InventorySlot(slot_id="S0-slot0", gpu_type=GPUType.H100, qty_gpus=8,
                      start_slot=8, duration_hours=10.0),
        InventorySlot(slot_id="S0-slot1", gpu_type=GPUType.A100, qty_gpus=4,
                      start_slot=14, duration_hours=6.0),
    )
    pub = SellerPublicProfile(
        seller_id="S0", display_name="Pacific Cluster",
        inventory_slots=inv,
        list_price_per_gpu_hr={GPUType.H100: 6.0, GPUType.A100: 3.5},
        volume_discount_policy=VolumeDiscountPolicy(
            tiers=(
                VolumeDiscountTier(min_qty_gpus=4, min_duration_hours=4, discount_pct=0.05),
                VolumeDiscountTier(min_qty_gpus=8, min_duration_hours=8, discount_pct=0.15),
            ),
        ),
    )
    priv = SellerPrivateProfile(
        reserve_per_slot={"S0-slot0": 4.0, "S0-slot1": 2.5},
        marginal_cost_per_gpu_hr={GPUType.H100: 3.0, GPUType.A100: 2.0},
        target_utilization=0.75,
        competing_demand_signal="medium",
    )
    return SellerV2(public=pub, private=priv)


def test_seller_v2_round_trips_through_json():
    s = _sample_seller_v2()
    raw = s.model_dump_json()
    restored = SellerV2.model_validate_json(raw)
    assert restored == s


def test_seller_v2_backward_compat_shims():
    s = _sample_seller_v2()
    assert s.id == "S0"
    assert s.label == "Pacific Cluster"


# ---------------------------------------------------------------------------
# BuyerV2 backward-compat shims
# ---------------------------------------------------------------------------


def test_buyer_v2_shims():
    pub = _sample_buyer_public()
    priv = BuyerPrivateProfile(
        max_willingness_to_pay=8.0, urgency_score=0.7, budget_remaining_usd=2000,
    )
    b = BuyerV2(public=pub, private=priv)
    assert b.id == "B0"
    assert b.label == "Acme Lab"
    assert b.gpu_type == GPUType.H100


def test_buyer_v2_round_trips_through_json():
    pub = _sample_buyer_public()
    priv = BuyerPrivateProfile(
        max_willingness_to_pay=8.0, urgency_score=0.7, budget_remaining_usd=2000,
    )
    b = BuyerV2(public=pub, private=priv)
    raw = b.model_dump_json()
    restored = BuyerV2.model_validate_json(raw)
    assert restored == b


# ---------------------------------------------------------------------------
# Frozen guarantees
# ---------------------------------------------------------------------------


def test_buyer_public_is_frozen():
    p = _sample_buyer_public()
    with pytest.raises(Exception):
        p.qty_gpus = 99  # type: ignore[misc]


def test_volume_discount_tier_is_frozen():
    t = VolumeDiscountTier(min_qty_gpus=4, min_duration_hours=4, discount_pct=0.10)
    with pytest.raises(Exception):
        t.discount_pct = 0.5  # type: ignore[misc]
