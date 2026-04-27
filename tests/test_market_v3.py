"""Tests for v0.3 market generation (translate + tiered sellers).

The translate step itself requires an LLM client; tests use the synthetic
fallback path so they run without API keys.
"""

import pytest

from gpubid.market_v3 import V3Enrichment, clear_translate_cache, generate_market_v3
from gpubid.schema import Market


def setup_function(_fn):
    clear_translate_cache()


def test_generate_returns_market_and_enrichment():
    market, enrichment = generate_market_v3(n_buyers=4, n_sellers=2, regime="tight", seed=1)
    assert isinstance(market, Market)
    assert isinstance(enrichment, V3Enrichment)
    assert len(market.buyers) == 4
    assert len(market.sellers) == 2


def test_every_buyer_carries_a_requirement():
    market, enrichment = generate_market_v3(n_buyers=6, n_sellers=3, regime="tight", seed=42)
    for buyer in market.buyers:
        assert buyer.id in enrichment.buyer_requirements
        assert buyer.id in enrichment.buyer_public_profiles
        assert buyer.id in enrichment.buyer_private_profiles


def test_every_seller_carries_a_volume_policy():
    market, enrichment = generate_market_v3(n_buyers=4, n_sellers=4, regime="slack", seed=7)
    for seller in market.sellers:
        assert seller.id in enrichment.seller_volume_policies
        assert seller.id in enrichment.seller_v3


def test_business_context_summary_is_pass_through_of_raw_text():
    market, enrichment = generate_market_v3(n_buyers=4, n_sellers=2, regime="tight", seed=99)
    for buyer in market.buyers:
        req = enrichment.buyer_requirements[buyer.id]
        priv = enrichment.buyer_private_profiles[buyer.id]
        assert priv.business_context_summary == req.raw_text


def test_seed_is_deterministic():
    a, ea = generate_market_v3(n_buyers=4, n_sellers=2, regime="tight", seed=12345)
    b, eb = generate_market_v3(n_buyers=4, n_sellers=2, regime="tight", seed=12345)
    assert a.id == b.id
    assert [b.id for b in a.buyers] == [b.id for b in b.buyers]
    # Same requirements assigned in same order
    assert [ea.buyer_requirements[b.id].requirement_id for b in a.buyers] == \
           [eb.buyer_requirements[b.id].requirement_id for b in b.buyers]


def test_some_sellers_have_tiers_some_dont():
    """Across many seeds we should see a mix of flat and tiered sellers."""
    n_with_tiers = 0
    n_without = 0
    for seed in range(30):
        _market, enrichment = generate_market_v3(n_buyers=2, n_sellers=4, regime="tight", seed=seed)
        for policy in enrichment.seller_volume_policies.values():
            if len(policy.tiers) == 0:
                n_without += 1
            else:
                n_with_tiers += 1
    assert n_with_tiers > 0
    assert n_without > 0
    # And neither dominates absolutely
    assert n_with_tiers / (n_with_tiers + n_without) > 0.4


def test_tiered_seller_v3_has_correct_inventory_slots():
    market, enrichment = generate_market_v3(n_buyers=2, n_sellers=2, regime="tight", seed=1)
    for seller in market.sellers:
        v3 = enrichment.seller_v3[seller.id]
        # Slot count and IDs must match
        assert len(v3.public.inventory_slots) == len(seller.capacity_slots)
        v3_slot_ids = {s.slot_id for s in v3.public.inventory_slots}
        v02_slot_ids = {s.id for s in seller.capacity_slots}
        assert v3_slot_ids == v02_slot_ids


def test_buyer_qty_does_not_exceed_typical_slot_capacity():
    """Synthetic fallback caps qty at 8 so the market is solvable."""
    market, _ = generate_market_v3(n_buyers=8, n_sellers=4, regime="tight", seed=1)
    for buyer in market.buyers:
        assert buyer.job.qty <= 8


def test_translate_cache_returns_same_profile_on_repeat():
    """Running the same seed back-to-back should reuse the cache.

    For the synthetic-fallback path (no LLM), there's no cache hit (the cache
    is for LLM responses), but determinism still holds — same seed, same RNG,
    same profile.
    """
    a, ea = generate_market_v3(n_buyers=2, n_sellers=2, regime="tight", seed=42)
    b, eb = generate_market_v3(n_buyers=2, n_sellers=2, regime="tight", seed=42)
    for buyer in a.buyers:
        pa = ea.buyer_public_profiles[buyer.id]
        pb = eb.buyer_public_profiles[buyer.id]
        assert pa == pb


def test_invalid_regime_rejected():
    with pytest.raises(ValueError):
        generate_market_v3(n_buyers=2, n_sellers=2, regime="invalid", seed=1)  # type: ignore[arg-type]
