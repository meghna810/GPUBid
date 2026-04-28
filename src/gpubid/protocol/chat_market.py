"""Chat-based market — bilateral dialogues are the PRIMARY mechanism.

This module replaces the public-board + central-clearer flow (`round_runner`)
with what users actually expect when they hear "agents negotiate": real 1:1
conversations between a buyer and a seller, turn by turn, until one accepts,
walks, or hits the turn cap.

Architecture (pair → chat → deal, no order book):

    1. MATCHMAKING — for each buyer, list seller slots that are structurally
       compatible (GPU type, time window, qty, duration, tolerance). Rank by
       reserve ascending so the cheapest matching slot is the buyer's first
       date.

    2. SEQUENTIAL DIALOGUES — order buyers by urgency desc; for each buyer in
       turn, run a bilateral chat with their top remaining compatible slot.
       The dialogue itself lives in `gpubid.protocol.dialogue.run_bilateral_dialogue`.

    3. CLOSURE — if the chat ends in `accept`, validate against private
       reserves (sanity, not market mechanism) and commit a deal. If it ends
       in `walk_away`, the buyer is freed to chat with the next compatible
       slot, up to `max_retries_per_buyer`.

    4. SLOTS DEPLETE — once a slot is exhausted (qty == 0) it falls out of
       all buyers' compatibility lists. This is the only "market dynamic" left:
       capacity scarcity. There's no central matching engine; the only "engine"
       is FIFO-by-urgency.

That's it. Every deal is the product of an actual conversation, not a
spreadsheet match.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from gpubid.engine.clearing import ask_satisfies_buyer, buyer_accepts_tolerance
from gpubid.llm import LLMClient
from gpubid.protocol.dialogue import (
    BilateralDialogueResult,
    DialogueTurn,
    run_bilateral_dialogue,
)
from gpubid.schema import (
    Buyer,
    CapacitySlot,
    Deal,
    GPUType,
    InterruptionTolerance,
    Market,
    Offer,
    OfferKind,
    Seller,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class ChatMarketRun:
    """The full record of one chat-based market run."""

    market: Market
    deals: list[Deal] = field(default_factory=list)
    dialogues: list[BilateralDialogueResult] = field(default_factory=list)
    walked_pairs: list[tuple[str, str]] = field(default_factory=list)  # (buyer_id, slot_id)
    skipped_buyers: list[tuple[str, str]] = field(default_factory=list)  # (buyer_id, reason)


# ---------------------------------------------------------------------------
# Compatibility — pure structural match, no price involved
# ---------------------------------------------------------------------------


def _compatible_slots_for_buyer(buyer: Buyer, sellers: list[Seller]) -> list[CapacitySlot]:
    """Return seller slots structurally compatible with this buyer's job, cheapest first."""
    matches: list[CapacitySlot] = []
    for s in sellers:
        for sl in s.capacity_slots:
            if sl.gpu_type not in buyer.job.acceptable_gpus:
                continue
            if sl.qty < buyer.job.qty:
                continue
            if sl.duration < buyer.job.duration:
                continue
            # The slot's start window must be able to accommodate the buyer's
            # required duration inside their [earliest, latest] window.
            if sl.start < buyer.job.earliest_start:
                continue
            if sl.start + buyer.job.duration > buyer.job.latest_finish:
                continue
            # Tolerance — sellers in chat mode default to NONE (strictest); the
            # buyer must accept that. (Same convention as the deterministic flow.)
            if not buyer_accepts_tolerance(buyer, InterruptionTolerance.NONE):
                continue
            matches.append(sl)
    matches.sort(key=lambda sl: sl.reserve_per_gpu_hr)
    return matches


def _seller_for_slot(slot_id: str, market: Market) -> Optional[Seller]:
    for s in market.sellers:
        for sl in s.capacity_slots:
            if sl.id == slot_id:
                return s
    return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_chat_market(
    market: Market,
    *,
    buyer_clients: dict[str, LLMClient],
    seller_clients: dict[str, LLMClient],
    max_turns_per_dialogue: int = 6,
    max_retries_per_buyer: int = 2,
    posted_price_estimate_factor: float = 1.4,
    seller_volume_policies: Optional[dict[str, object]] = None,
    buyer_business_contexts: Optional[dict[str, str]] = None,
    on_dialogue_complete: Optional[callable] = None,
) -> ChatMarketRun:
    """Run a chat-based market. Each deal is the product of a real bilateral chat.

    Buyers are processed in urgency-descending order. Each buyer chats with
    their top remaining compatible seller slot; on walk-away they retry with
    the next-cheapest compatible slot up to `max_retries_per_buyer` times.

    `buyer_clients` / `seller_clients` map agent_id → LLMClient. Pass any
    LLMClient subclass — Anthropic, OpenAI, Gemini all work the same.

    `on_dialogue_complete` is an optional callback `fn(dialogue_result, deal)`
    called after each thread closes — used by the streaming chat viz to render
    each conversation as it happens.
    """
    # State
    slot_remaining: dict[str, int] = {
        sl.id: sl.qty for s in market.sellers for sl in s.capacity_slots
    }
    fulfilled_buyers: set[str] = set()
    deals: list[Deal] = []
    dialogues: list[BilateralDialogueResult] = []
    walked: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []

    # Order buyers by urgency desc; ties break by id for reproducibility.
    buyers_ordered = sorted(market.buyers, key=lambda b: (-b.urgency, b.id))

    for buyer in buyers_ordered:
        if buyer.id in fulfilled_buyers:
            continue
        if buyer.id not in buyer_clients:
            skipped.append((buyer.id, "no LLM client assigned"))
            continue

        compatible = _compatible_slots_for_buyer(buyer, list(market.sellers))
        # Filter to slots that still have remaining capacity for this buyer.
        compatible = [sl for sl in compatible if slot_remaining.get(sl.id, 0) >= buyer.job.qty]
        if not compatible:
            skipped.append((buyer.id, "no compatible slot with remaining capacity"))
            continue

        retries_left = max_retries_per_buyer
        for slot in compatible:
            if slot_remaining.get(slot.id, 0) < buyer.job.qty:
                continue
            seller = _seller_for_slot(slot.id, market)
            if seller is None or seller.id not in seller_clients:
                continue

            # Opening positions: seller starts at reserve × markup, buyer at max × markdown.
            # These are the *opening* moves — the LLMs will counter from here.
            opening_seller = round(slot.reserve_per_gpu_hr * 1.5, 2)
            opening_buyer = round(buyer.job.max_value_per_gpu_hr * 0.55, 2)

            policy = (seller_volume_policies or {}).get(seller.id)
            biz_ctx = (buyer_business_contexts or {}).get(buyer.id)

            dialogue = run_bilateral_dialogue(
                buyer=buyer,
                seller=seller,
                slot=slot,
                opening_seller_price=opening_seller,
                opening_buyer_price=opening_buyer,
                max_turns=max_turns_per_dialogue,
                buyer_client=buyer_clients[buyer.id],
                seller_client=seller_clients[seller.id],
                market=market,
                posted_price_estimate=slot.reserve_per_gpu_hr * posted_price_estimate_factor,
                seller_volume_policy=policy,
                buyer_business_context=biz_ctx,
            )
            dialogues.append(dialogue)

            if dialogue.closed and dialogue.closing_price is not None:
                # Validate against private reserves — sanity check, not a market move.
                price = float(dialogue.closing_price)
                if price < slot.reserve_per_gpu_hr:
                    walked.append((buyer.id, slot.id))
                    if on_dialogue_complete is not None:
                        on_dialogue_complete(dialogue, None)
                    continue
                if price > buyer.job.max_value_per_gpu_hr:
                    walked.append((buyer.id, slot.id))
                    if on_dialogue_complete is not None:
                        on_dialogue_complete(dialogue, None)
                    continue

                deal = Deal(
                    id=f"deal-{buyer.id}-{slot.id}-chat",
                    round_n=dialogue.turns[-1].turn_n if dialogue.turns else 0,
                    buyer_id=buyer.id,
                    seller_id=seller.id,
                    slot_id=slot.id,
                    qty=buyer.job.qty,
                    price_per_gpu_hr=price,
                    start=slot.start,
                    duration=buyer.job.duration,
                    gpu_type=slot.gpu_type,
                    interruption_tolerance=InterruptionTolerance.NONE,
                )
                deals.append(deal)
                slot_remaining[slot.id] -= buyer.job.qty
                fulfilled_buyers.add(buyer.id)
                if on_dialogue_complete is not None:
                    on_dialogue_complete(dialogue, deal)
                break

            # Walk away — buyer can try another compatible slot.
            walked.append((buyer.id, slot.id))
            if on_dialogue_complete is not None:
                on_dialogue_complete(dialogue, None)
            retries_left -= 1
            if retries_left <= 0:
                break

    return ChatMarketRun(
        market=market,
        deals=deals,
        dialogues=dialogues,
        walked_pairs=walked,
        skipped_buyers=skipped,
    )


# ---------------------------------------------------------------------------
# Adapter — emit a board-style snapshot list so persuasion / forensics keep working
# ---------------------------------------------------------------------------


def chat_run_to_snapshots(run: ChatMarketRun) -> list:
    """Convert a ChatMarketRun into a list of RoundSnapshot-like records.

    The persuasion / forensics modules iterate over `RoundSnapshot.actions`
    looking for `new_offers` and `accept_offer_ids` with `reasoning` strings.
    We synthesize equivalent records from each dialogue's turns so the existing
    analytics keep working without modification.
    """
    from gpubid.engine.board import AgentActionRecord, RoundSnapshot

    # Each dialogue → one synthetic "round" (so the round number is unique
    # per dialogue). Turns within the dialogue become actions in that round.
    snapshots: list[RoundSnapshot] = []
    cumulative_deals: list[Deal] = []
    deal_by_pair = {(d.buyer_id, d.slot_id): d for d in run.deals}

    for idx, dialogue in enumerate(run.dialogues, start=1):
        actions: list[AgentActionRecord] = []
        new_deals_this = []
        for turn in dialogue.turns:
            if turn.action == "counter" and turn.proposed_price_per_gpu_hr is not None:
                # Build a synthetic Offer mirroring this turn's price.
                if turn.speaker == "seller":
                    offer = Offer(
                        id=f"chat-ask-{turn.speaker_id}-{idx}-t{turn.turn_n}",
                        round_n=idx,
                        from_id=turn.speaker_id,
                        kind=OfferKind.ASK,
                        slot_id=_slot_id_from_dialogue(dialogue),
                        price_per_gpu_hr=turn.proposed_price_per_gpu_hr,
                        qty=1,
                        gpu_type=GPUType.A100,  # filled from actual slot below
                        start=0,
                        duration=1,
                        interruption_tolerance=InterruptionTolerance.NONE,
                        reasoning=_format_turn_reasoning(turn),
                    )
                else:
                    offer = Offer(
                        id=f"chat-bid-{turn.speaker_id}-{idx}-t{turn.turn_n}",
                        round_n=idx,
                        from_id=turn.speaker_id,
                        kind=OfferKind.BID,
                        price_per_gpu_hr=turn.proposed_price_per_gpu_hr,
                        qty=1,
                        gpu_type=GPUType.A100,
                        start=0,
                        duration=1,
                        interruption_tolerance=InterruptionTolerance.NONE,
                        reasoning=_format_turn_reasoning(turn),
                    )
                actions.append(AgentActionRecord(
                    agent_id=turn.speaker_id,
                    new_offers=(offer,),
                    accept_offer_ids=(),
                    reasoning=_format_turn_reasoning(turn),
                ))
            elif turn.action == "accept":
                actions.append(AgentActionRecord(
                    agent_id=turn.speaker_id,
                    new_offers=(),
                    accept_offer_ids=(f"chat-accept-{idx}-t{turn.turn_n}",),
                    reasoning=_format_turn_reasoning(turn),
                ))

        # If this dialogue closed in a deal, attach it.
        buyer_id, _, slot_id = (dialogue.pair_id.partition("↔")[2].partition("/")) if dialogue.pair_id else ("", "", "")
        buyer_id_clean = dialogue.pair_id.split("↔")[0] if "↔" in dialogue.pair_id else ""
        slot_id_clean = dialogue.pair_id.split("/")[-1] if "/" in dialogue.pair_id else ""
        deal = deal_by_pair.get((buyer_id_clean, slot_id_clean))
        if deal is not None:
            cumulative_deals.append(deal)
            new_deals_this.append(deal)

        snapshots.append(RoundSnapshot(
            round_n=idx,
            max_rounds=len(run.dialogues),
            asks=(),
            bids=(),
            new_deals=tuple(new_deals_this),
            all_deals=tuple(cumulative_deals),
            active_buyer_ids=(),
            active_seller_ids=(),
            is_final=(idx == len(run.dialogues)),
            actions=tuple(actions),
        ))
    return snapshots


def _format_turn_reasoning(turn: DialogueTurn) -> str:
    """Combine argument + condition into a single reasoning string for analytics."""
    parts = [turn.argument or ""]
    if turn.condition and turn.condition.strip().lower() not in ("flat", "flat — no condition", "none", ""):
        parts.append(f"(condition: {turn.condition})")
    return " ".join(p for p in parts if p).strip()


def _slot_id_from_dialogue(dialogue: BilateralDialogueResult) -> str:
    if "/" in dialogue.pair_id:
        return dialogue.pair_id.split("/")[-1]
    return ""


__all__ = [
    "ChatMarketRun",
    "run_chat_market",
    "chat_run_to_snapshots",
]
