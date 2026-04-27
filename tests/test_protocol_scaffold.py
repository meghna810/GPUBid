"""Phase 5 + 9 scaffold tests: broadcast + eligibility + HITL data model.

Negotiation runner tests (run_negotiation end-to-end) are deferred until LLM
fixtures are recorded — those tests live in test_negotiation_live.py with a
skip marker.
"""

import pytest

from gpubid.domain.profiles import (
    BuyerPrivateProfile,
    BuyerPublicProfile,
    BuyerV2,
    InventorySlot,
    SellerPrivateProfile,
    SellerPublicProfile,
    SellerV2,
    TimeWindow,
)
from gpubid.protocol.broadcast import broadcast_buyer_to_sellers
from gpubid.protocol.eligibility import is_eligible
from gpubid.protocol.hitl import (
    HITLDecision,
    HITLEvent,
    HITLPolicy,
    HITLTrigger,
    auto_proceed_surfacer,
)
from gpubid.schema import GPUType


def _make_buyer(qty=4, dur=4.0, gpus=(GPUType.H100,), tw=(0, 24)) -> BuyerV2:
    return BuyerV2(
        public=BuyerPublicProfile(
            buyer_id="B0", display_name="Acme",
            workload_category="training",
            gpu_type_preferences=gpus,
            qty_gpus=qty, duration_hours=dur,
            time_window=TimeWindow(earliest_start_slot=tw[0], latest_finish_slot=tw[1]),
            interruption_tolerance="none", urgency_band="urgent",
        ),
        private=BuyerPrivateProfile(
            max_willingness_to_pay=8.0, urgency_score=0.7,
            budget_remaining_usd=1000,
        ),
    )


def _make_seller(slot_qty=8, slot_dur=8.0, gpu=GPUType.H100, slot_start=0) -> SellerV2:
    sid = "S0-slot0"
    return SellerV2(
        public=SellerPublicProfile(
            seller_id="S0", display_name="Pacific",
            inventory_slots=(
                InventorySlot(slot_id=sid, gpu_type=gpu, qty_gpus=slot_qty,
                              start_slot=slot_start, duration_hours=slot_dur),
            ),
            list_price_per_gpu_hr={gpu: 6.0},
        ),
        private=SellerPrivateProfile(
            reserve_per_slot={sid: 4.0},
            marginal_cost_per_gpu_hr={gpu: 3.0},
            target_utilization=0.8,
        ),
    )


# ---------------------------------------------------------------------------
# Broadcast
# ---------------------------------------------------------------------------


def test_broadcast_emits_one_message_per_seller():
    buyer = _make_buyer()
    sellers = [_make_seller(), _make_seller()]
    sellers[1].public.__dict__["seller_id"] = "S1"  # cheating; profile is frozen, can't mutate
    msgs = broadcast_buyer_to_sellers(buyer, sellers)
    assert len(msgs) == 2


def test_broadcast_carries_only_public_profile():
    buyer = _make_buyer()
    sellers = [_make_seller()]
    msgs = broadcast_buyer_to_sellers(buyer, sellers)
    msg = msgs[0]
    # The broadcast must carry buyer.public; private profile must NOT leak.
    assert msg.buyer_public.buyer_id == "B0"
    # Trying to attribute-access `private` on the message should fail.
    assert not hasattr(msg, "buyer_private")


def test_broadcast_id_shared_across_messages():
    buyer = _make_buyer()
    sellers = [_make_seller(), _make_seller()]
    msgs = broadcast_buyer_to_sellers(buyer, sellers)
    assert len({m.broadcast_id for m in msgs}) == 1


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def test_eligibility_accepts_compatible_pair():
    buyer = _make_buyer(qty=4, dur=4.0, gpus=(GPUType.H100,))
    seller = _make_seller(slot_qty=8, slot_dur=8.0, gpu=GPUType.H100)
    msgs = broadcast_buyer_to_sellers(buyer, [seller])
    eligible, reason = is_eligible(seller, msgs[0])
    assert eligible is True
    assert "eligible" in reason.lower()


def test_eligibility_rejects_wrong_gpu_type():
    buyer = _make_buyer(qty=4, dur=4.0, gpus=(GPUType.H100,))
    seller = _make_seller(slot_qty=8, slot_dur=8.0, gpu=GPUType.A100)
    msgs = broadcast_buyer_to_sellers(buyer, [seller])
    eligible, reason = is_eligible(seller, msgs[0])
    assert eligible is False
    assert "gpu" in reason.lower()


def test_eligibility_rejects_undersized_capacity():
    buyer = _make_buyer(qty=10)
    seller = _make_seller(slot_qty=4)
    msgs = broadcast_buyer_to_sellers(buyer, [seller])
    eligible, reason = is_eligible(seller, msgs[0])
    assert eligible is False
    assert "capacity" in reason.lower() or "slot" in reason.lower()


def test_eligibility_rejects_outside_time_window():
    buyer = _make_buyer(tw=(8, 12))
    seller = _make_seller(slot_start=20, slot_dur=4.0)
    msgs = broadcast_buyer_to_sellers(buyer, [seller])
    eligible, _ = is_eligible(seller, msgs[0])
    assert eligible is False


# ---------------------------------------------------------------------------
# HITL
# ---------------------------------------------------------------------------


def test_hitl_policy_disabled_always_proceeds():
    policy = HITLPolicy(enabled=False, surfacer=auto_proceed_surfacer)
    event = HITLEvent(
        trigger=HITLTrigger.DEADLOCK, agent_id="B0", round_n=3,
        proposed_offer=None, note="test",
    )
    decision = policy.maybe_intervene(event)
    assert decision.action == "proceed_as_proposed"


def test_hitl_auto_proceed_surfacer_is_safe_for_headless():
    event = HITLEvent(
        trigger=HITLTrigger.LOW_CONFIDENCE_CLOSE, agent_id="B0", round_n=2,
        proposed_offer=None,
    )
    decision = auto_proceed_surfacer(event)
    assert isinstance(decision, HITLDecision)
    assert "auto-proceed" in decision.note


def test_hitl_decision_is_frozen():
    d = HITLDecision(action="proceed_as_proposed", note="x")
    with pytest.raises(Exception):
        d.note = "y"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Scaffolded modules raise NotImplementedError until LLM fixtures recorded
# ---------------------------------------------------------------------------


def test_run_negotiation_is_scaffolded_and_raises_not_implemented():
    from gpubid.protocol.round import run_negotiation
    with pytest.raises(NotImplementedError, match="Phase 5"):
        run_negotiation(None, None, None, None)  # type: ignore[arg-type]


def test_sim_v2_run_simulation_is_scaffolded():
    from gpubid.experiments.sim_v2 import run_simulation, SimSpec
    spec = SimSpec(name="test")
    with pytest.raises(NotImplementedError, match="Phase 12"):
        run_simulation(spec)


def test_load_all_runs_returns_empty_df_when_no_runs():
    from gpubid.experiments.sim_v2 import load_all_runs
    df = load_all_runs()
    assert df.empty
