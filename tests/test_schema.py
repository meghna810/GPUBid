"""Schema and market-generator correctness."""

from gpubid.market import GPU_BASE_RESERVE, generate_market
from gpubid.schema import (
    Buyer,
    GPUType,
    InterruptionTolerance,
    Market,
    Seller,
)


def test_market_is_reproducible_from_seed():
    a = generate_market(8, 4, "tight", seed=1234)
    b = generate_market(8, 4, "tight", seed=1234)
    assert a.model_dump() == b.model_dump()


def test_different_seeds_yield_different_markets():
    a = generate_market(8, 4, "tight", seed=1)
    b = generate_market(8, 4, "tight", seed=2)
    assert a.model_dump() != b.model_dump()


def test_tight_regime_has_less_supply_than_slack():
    """Regime now shapes supply: tight = fewer/smaller slots, slack = more/larger."""
    avg_tight = sum(
        generate_market(8, 4, "tight", seed=i).total_supply_gpu_hours for i in range(20)
    ) / 20
    avg_slack = sum(
        generate_market(8, 4, "slack", seed=i).total_supply_gpu_hours for i in range(20)
    ) / 20
    assert avg_tight < avg_slack


def test_tight_regime_has_higher_average_urgency():
    """Tight markets generate buyers with a non-zero urgency floor."""
    avg_tight = sum(
        b.urgency
        for i in range(10)
        for b in generate_market(8, 4, "tight", seed=i).buyers
    ) / (10 * 8)
    avg_slack = sum(
        b.urgency
        for i in range(10)
        for b in generate_market(8, 4, "slack", seed=i).buyers
    ) / (10 * 8)
    assert avg_tight > avg_slack


def test_buyer_values_clear_their_acceptable_gpus():
    """A buyer's max value should at least be markup over the cheapest acceptable GPU's reserve."""
    m = generate_market(10, 4, "tight", seed=7)
    for b in m.buyers:
        cheapest_acceptable = min(GPU_BASE_RESERVE[g] for g in b.job.acceptable_gpus)
        # Markup is 1.2x at minimum on the most expensive; should clear the cheapest.
        assert b.job.max_value_per_gpu_hr > cheapest_acceptable


def test_capacity_slot_offpeak_detection():
    m = generate_market(8, 4, "slack", seed=99)
    for s in m.sellers:
        for slot in s.capacity_slots:
            expected = slot.start < 6 or slot.start >= 22
            assert slot.is_offpeak == expected


def test_offpeak_slots_have_lower_reserves_on_average():
    """Across many seeds, off-peak reserves should be systematically lower."""
    peak_reserves = []
    offpeak_reserves = []
    for seed in range(50):
        m = generate_market(8, 4, "tight", seed=seed)
        for s in m.sellers:
            for slot in s.capacity_slots:
                bucket = offpeak_reserves if slot.is_offpeak else peak_reserves
                # Normalize by GPU base so we're comparing apples to apples.
                bucket.append(slot.reserve_per_gpu_hr / GPU_BASE_RESERVE[slot.gpu_type])
    assert sum(offpeak_reserves) / len(offpeak_reserves) < sum(peak_reserves) / len(peak_reserves)


def test_models_are_frozen():
    m = generate_market(4, 2, "tight", seed=1)
    import pydantic
    try:
        m.buyers[0].job.qty = 999  # type: ignore[misc]
    except (pydantic.ValidationError, AttributeError, TypeError):
        return
    raise AssertionError("expected frozen model to reject mutation")


def test_repr_html_returns_nonempty_strings():
    m = generate_market(4, 2, "tight", seed=1)
    assert "<div" in m._repr_html_()
    assert "<div" in m.buyers[0]._repr_html_()
    assert "<div" in m.sellers[0]._repr_html_()


def test_invalid_regime_raises():
    import pytest
    with pytest.raises(ValueError):
        generate_market(4, 2, "weird", seed=1)  # type: ignore[arg-type]


def test_smallest_market_works():
    """Tiny markets are useful for hand-checking VCG later."""
    m = generate_market(1, 1, "tight", seed=1)
    assert len(m.buyers) == 1
    assert len(m.sellers) == 1
    assert isinstance(m.buyers[0], Buyer)
    assert isinstance(m.sellers[0], Seller)
