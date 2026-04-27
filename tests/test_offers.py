"""Phase 4 tests: volume-discount-aware offer schema and pricing helpers."""

import pytest

from gpubid.domain.offers import (
    OfferTerms,
    effective_price_per_gpu_hr,
    total_value_usd,
)
from gpubid.domain.profiles import VolumeDiscountTier
from gpubid.schema import GPUType


def _flat_offer(price: float = 5.0) -> OfferTerms:
    return OfferTerms(
        offer_id="o1", round_n=1, side="sell", agent_id="S0",
        gpu_type=GPUType.H100, qty_gpus=8, duration_hours=6.0, start_slot=8,
        interruption_tolerance="none",
        base_price_per_gpu_hr=price,
    )


def _tiered_offer() -> OfferTerms:
    return OfferTerms(
        offer_id="o2", round_n=1, side="sell", agent_id="S0",
        gpu_type=GPUType.H100, qty_gpus=16, duration_hours=10.0, start_slot=8,
        interruption_tolerance="none",
        base_price_per_gpu_hr=10.0,
        discount_schedule=(
            VolumeDiscountTier(min_qty_gpus=4, min_duration_hours=4, discount_pct=0.05),
            VolumeDiscountTier(min_qty_gpus=8, min_duration_hours=8, discount_pct=0.15),
            VolumeDiscountTier(min_qty_gpus=16, min_duration_hours=10, discount_pct=0.25),
        ),
    )


def test_flat_offer_returns_base_price_for_any_qty():
    o = _flat_offer(price=5.0)
    assert effective_price_per_gpu_hr(o, qty=1, duration=1.0) == 5.0
    assert effective_price_per_gpu_hr(o, qty=100, duration=100.0) == 5.0


def test_tiered_offer_picks_best_applicable_tier():
    o = _tiered_offer()
    # qty=4, dur=4: only first tier applies (5%)
    assert effective_price_per_gpu_hr(o, qty=4, duration=4.0) == pytest.approx(9.50)
    # qty=8, dur=8: first and second apply; second wins (15%)
    assert effective_price_per_gpu_hr(o, qty=8, duration=8.0) == pytest.approx(8.50)
    # qty=16, dur=10: all three apply; third wins (25%)
    assert effective_price_per_gpu_hr(o, qty=16, duration=10.0) == pytest.approx(7.50)
    # qty=16 but dur=4 only: third tier needs dur 10, falls back to second's dur 8 — also fails;
    # falls back to first tier's dur 4 — applies. Discount 5%.
    assert effective_price_per_gpu_hr(o, qty=16, duration=4.0) == pytest.approx(9.50)


def test_tiered_offer_below_first_tier_returns_base():
    o = _tiered_offer()
    assert effective_price_per_gpu_hr(o, qty=1, duration=1.0) == 10.0


def test_total_value_includes_discount():
    o = _tiered_offer()
    # qty=8, dur=8: 8.50 × 8 × 8 = 544
    assert total_value_usd(o, qty=8, duration=8.0) == pytest.approx(544.0)


def test_total_value_monotonic_at_tier_thresholds():
    """Crossing a tier boundary should not produce a price hike for the buyer."""
    o = _tiered_offer()
    # As qty grows past tier thresholds, effective price should be non-increasing.
    prices = [effective_price_per_gpu_hr(o, qty=q, duration=10.0) for q in [3, 4, 7, 8, 15, 16]]
    for i in range(len(prices) - 1):
        assert prices[i + 1] <= prices[i] + 1e-6, f"price increased at index {i}: {prices}"


def test_offer_is_frozen():
    o = _flat_offer()
    with pytest.raises(Exception):
        o.base_price_per_gpu_hr = 99.0  # type: ignore[misc]


def test_offer_round_trips_through_json():
    o = _tiered_offer()
    raw = o.model_dump_json()
    restored = OfferTerms.model_validate_json(raw)
    assert restored == o


def test_offer_rejects_negative_qty():
    with pytest.raises(ValueError):
        OfferTerms(
            offer_id="bad", round_n=0, side="sell", agent_id="S0",
            gpu_type=GPUType.H100, qty_gpus=0, duration_hours=1.0, start_slot=0,
            interruption_tolerance="none", base_price_per_gpu_hr=1.0,
        )


def test_offer_confidence_bounds():
    with pytest.raises(ValueError):
        OfferTerms(
            offer_id="bad", round_n=0, side="sell", agent_id="S0",
            gpu_type=GPUType.H100, qty_gpus=1, duration_hours=1.0, start_slot=0,
            interruption_tolerance="none", base_price_per_gpu_hr=1.0,
            confidence=1.5,
        )
