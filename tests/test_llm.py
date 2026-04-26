"""Provider auto-detection tests for the LLM client wrapper.

These tests verify the dispatch logic without making any actual LLM calls.
"""

import pytest

from gpubid.agents.prompts import (
    PROMPT_VERSION,
    buyer_system_prompt,
    buyer_tool_specs,
    seller_system_prompt,
    seller_tool_specs,
)
from gpubid.llm import (
    DEFAULT_ANTHROPIC_MODEL,
    DEFAULT_OPENAI_MODEL,
    AnthropicClient,
    OpenAIClient,
    ProviderUnknownError,
    detect_provider,
    make_client,
)
from gpubid.market import generate_market


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


def test_anthropic_key_detected():
    assert detect_provider("sk-ant-abc123") == "anthropic"


def test_openai_key_detected():
    assert detect_provider("sk-abc123") == "openai"


def test_anthropic_takes_priority_over_short_sk_prefix():
    """`sk-ant-` is more specific than `sk-`; both starting with sk-ant- map to anthropic."""
    assert detect_provider("sk-ant-anything") == "anthropic"


def test_unknown_key_raises():
    with pytest.raises(ProviderUnknownError):
        detect_provider("garbage-key")
    with pytest.raises(ProviderUnknownError):
        detect_provider("")


def test_make_client_returns_anthropic_on_ant_key():
    """make_client constructs (but does not call) the right backend."""
    # We don't actually validate the key — Anthropic SDK is lazy; key check happens at API time.
    client = make_client("sk-ant-test-fake-key-for-testing")
    assert client.provider == "anthropic"
    assert isinstance(client, AnthropicClient)
    assert client.model == DEFAULT_ANTHROPIC_MODEL


def test_make_client_returns_openai_on_sk_key():
    client = make_client("sk-test-fake-key-for-testing")
    assert client.provider == "openai"
    assert isinstance(client, OpenAIClient)
    assert client.model == DEFAULT_OPENAI_MODEL


def test_make_client_respects_model_override():
    client = make_client("sk-ant-test", model="custom-claude-id")
    assert client.model == "custom-claude-id"


# ---------------------------------------------------------------------------
# Tool specs (provider-neutral)
# ---------------------------------------------------------------------------


def test_buyer_tool_specs_have_required_fields():
    specs = buyer_tool_specs()
    names = {s["name"] for s in specs}
    assert names == {"post_bid", "accept_ask", "do_nothing"}
    for s in specs:
        assert "name" in s and "description" in s and "parameters" in s
        assert s["parameters"]["type"] == "object"


def test_seller_tool_specs_have_required_fields():
    specs = seller_tool_specs()
    names = {s["name"] for s in specs}
    assert names == {"post_ask", "accept_bid", "do_nothing"}


def test_post_ask_requires_slot_id():
    """The seller's post_ask MUST include slot_id so clearing knows which slot is offered."""
    specs = seller_tool_specs()
    post_ask = next(s for s in specs if s["name"] == "post_ask")
    assert "slot_id" in post_ask["parameters"]["required"]


# ---------------------------------------------------------------------------
# System prompts (sanity checks — content matters, structure matters)
# ---------------------------------------------------------------------------


def test_buyer_prompt_includes_max_value_but_warns_not_to_reveal():
    market = generate_market(4, 2, "tight", seed=42)
    buyer = market.buyers[0]
    prompt = buyer_system_prompt(buyer, max_rounds=5)
    assert f"${buyer.job.max_value_per_gpu_hr:.2f}" in prompt
    # The prompt should warn against revealing the value.
    assert "do NOT reveal" in prompt or "do not reveal" in prompt.lower()


def test_seller_prompt_lists_all_slots():
    market = generate_market(4, 2, "tight", seed=42)
    seller = market.sellers[0]
    prompt = seller_system_prompt(seller, regime="tight", max_rounds=5)
    for slot in seller.capacity_slots:
        assert slot.id in prompt
        assert f"${slot.reserve_per_gpu_hr:.2f}" in prompt


def test_prompt_version_is_set():
    assert PROMPT_VERSION  # non-empty string


# ---------------------------------------------------------------------------
# Tool-call → action translation (deterministic, no LLM call)
# ---------------------------------------------------------------------------


def test_buyer_tool_call_post_bid_round_trips_to_offer():
    from gpubid.agents.buyer import _buyer_tool_call_to_action
    from gpubid.llm import ToolCall

    market = generate_market(4, 2, "tight", seed=42)
    buyer = market.buyers[0]
    tc = ToolCall(
        tool_name="post_bid",
        arguments={
            "price_per_gpu_hr": 3.50,
            "qty": buyer.job.qty,
            "gpu_type": buyer.job.acceptable_gpus[0].value,
            "start": buyer.job.earliest_start,
            "duration": buyer.job.duration,
            "interruption_tolerance": buyer.job.interruption_tolerance.value,
            "reasoning": "test bid",
        },
    )
    action = _buyer_tool_call_to_action(tc, buyer, round_n=1)
    assert len(action.new_offers) == 1
    assert action.new_offers[0].price_per_gpu_hr == 3.50
    assert action.new_offers[0].reasoning == "test bid"


def test_seller_tool_call_post_ask_validates_slot_belongs_to_seller():
    """post_ask with a slot_id from a different seller should be rejected."""
    from gpubid.agents.seller import _seller_tool_call_to_action
    from gpubid.llm import ToolCall

    market = generate_market(4, 2, "tight", seed=42)
    seller = market.sellers[0]
    other_seller = market.sellers[1]
    foreign_slot = other_seller.capacity_slots[0]
    tc = ToolCall(
        tool_name="post_ask",
        arguments={
            "slot_id": foreign_slot.id,  # not seller's own
            "price_per_gpu_hr": 3.0,
            "qty": foreign_slot.qty,
            "gpu_type": foreign_slot.gpu_type.value,
            "start": foreign_slot.start,
            "duration": foreign_slot.duration,
            "interruption_tolerance": "none",
            "reasoning": "stealing slot",
        },
    )
    action = _seller_tool_call_to_action(tc, seller, round_n=1)
    # Action should have NO new_offers — the foreign slot was rejected.
    assert action.new_offers == ()


def test_buyer_tool_call_accept_ask():
    from gpubid.agents.buyer import _buyer_tool_call_to_action
    from gpubid.llm import ToolCall

    market = generate_market(4, 2, "tight", seed=42)
    buyer = market.buyers[0]
    tc = ToolCall(
        tool_name="accept_ask",
        arguments={"target_offer_id": "ask-S0-slot0-r2", "reasoning": "good price"},
    )
    action = _buyer_tool_call_to_action(tc, buyer, round_n=2)
    assert action.accept_offer_ids == ("ask-S0-slot0-r2",)
