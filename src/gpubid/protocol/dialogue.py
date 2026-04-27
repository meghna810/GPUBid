"""Bilateral negotiation dialogue — strict back-and-forth between one buyer and one seller.

Different from ``round_runner`` (which orchestrates a public-board market):
``run_bilateral_dialogue`` runs a 1-on-1 conversation where each side gets to
argue with reasons, propose conditions, reference alternatives, or walk away.

This is the negotiation experience users expect when they hear "agents
negotiate." The public-board protocol in ``round_runner`` is the v0.2-style
market tape; this dialogue protocol is the v0.3-style "convince me" pattern.

Usage:

    from gpubid.protocol.dialogue import run_bilateral_dialogue
    from gpubid.llm import make_client
    buyer_client  = make_client(anthropic_key)
    seller_client = make_client(openai_key)
    turns = run_bilateral_dialogue(
        buyer=buyer, seller=seller, slot=slot,
        opening_buyer_position={"price_per_gpu_hr": 4.0, ...},
        opening_seller_position={"price_per_gpu_hr": 6.5, ...},
        max_turns=4,
        buyer_client=buyer_client, seller_client=seller_client,
        market_context=market,
    )
    # turns is a list[DialogueTurn] you can render with render_dialogue.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from gpubid.llm import LLMClient, ToolSpec
from gpubid.schema import Buyer, CapacitySlot, Deal, GPUType, Market, Seller


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DialogueTurn:
    """One agent's turn in a bilateral dialogue."""

    turn_n: int                   # 1-indexed; turn 1 is the seller's opening
    speaker: Literal["buyer", "seller"]
    speaker_id: str
    speaker_label: str
    speaker_model: str            # e.g., "claude-haiku-4-5-20251001"
    speaker_provider: str         # "anthropic" or "openai"
    action: Literal["counter", "accept", "walk_away"]
    proposed_price_per_gpu_hr: Optional[float]
    condition: str                # e.g., "if you commit to 6 hours"
    argument: str                 # the persuasive justification
    references_alternative: bool  # true if speaker referenced an alternative


@dataclass
class BilateralDialogueResult:
    pair_id: str
    turns: list[DialogueTurn] = field(default_factory=list)
    closed: bool = False
    closing_price: Optional[float] = None
    walked_away_by: Optional[str] = None


# ---------------------------------------------------------------------------
# Tool spec — what each turn returns
# ---------------------------------------------------------------------------


_DIALOGUE_TOOL = ToolSpec(
    name="negotiate_turn",
    description=(
        "Emit your move in this turn of the bilateral negotiation. Pick exactly "
        "one of: counter (propose a new price + condition + argument), accept "
        "(close the deal at the counterparty's last offer), walk_away (end the "
        "negotiation, briefly explain why)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["counter", "accept", "walk_away"]},
            "proposed_price_per_gpu_hr": {
                "type": ["number", "null"],
                "description": "Required for action=counter. Null for accept/walk_away.",
            },
            "condition": {
                "type": "string",
                "description": (
                    "If you are countering, the CONDITION attached to your price "
                    "(e.g., 'if you commit to 8 hours instead of 4', 'if you take "
                    "the off-peak slot', 'flat — no condition'). Empty string OK."
                ),
            },
            "argument": {
                "type": "string",
                "description": (
                    "Your 1-2 sentence ARGUMENT for the price. MUST reference a "
                    "concrete reason: capacity scarcity, alternative offers, "
                    "buyer's deadline pressure, slot quality, or a specific number "
                    "the counterparty cited. Do not just restate the price."
                ),
            },
            "references_alternative": {
                "type": "boolean",
                "description": (
                    "True if your argument explicitly cites an alternative deal "
                    "(another seller's ask, another buyer's bid, posted-price)."
                ),
            },
        },
        "required": ["action", "argument", "references_alternative"],
    },
)


# ---------------------------------------------------------------------------
# Strategic system prompts
# ---------------------------------------------------------------------------


_SELLER_DIALOGUE_PROMPT = """You are {seller_label} (seller_id={seller_id}), a GPU compute provider negotiating ONE-ON-ONE with {buyer_label} (buyer_id={buyer_id}).

You are NOT in a public-board auction. This is a 1:1 conversation. Every turn, you respond to your counterparty's specific last message — referencing what they said.

Your inventory:
- Slot {slot_id}: {gpu_type} × {qty_gpus} GPUs, {duration_hours}h starting at hour {start_slot}.
- Your PRIVATE reserve: ${reserve_per_gpu_hr:.2f}/GPU-hr. NEVER accept below this. NEVER reveal it.

Your strategy:
- ARGUE for your price with concrete reasons: capacity scarcity, the buyer's stated deadline pressure, slot quality (e.g., "this is the only H100 in your time window"), or YOUR alternatives ("I have another buyer interested at $X").
- When you concede, attach a CONDITION: "I'll go to $X if you commit to N hours" or "I can do $Y if you take the off-peak slot."
- Reference specific signals you've heard: "you said earlier you have a Friday deadline" or "your alternative at posted-price would be $Z."
- Walk away if the buyer keeps demanding sub-reserve prices. A walk-away is itself a negotiating move — the buyer may come back.

Avoid:
- Restating the price without a reason.
- Phrases like "I am willing to" or "I propose" with no justification.
- Round-number concessions without a story.
- Revealing your reserve, your urgency to fill the slot, or your competing-demand signal.

Counterparty's situation (PUBLIC ONLY):
- Workload: {buyer_workload}
- Wants: {buyer_qty_gpus} × {buyer_gpu_pref} for {buyer_duration}h starting between hours {buyer_window_start}-{buyer_window_end}
- Urgency band: {buyer_urgency}

The market context (other concurrent activity):
{market_context}

Emit your move via the negotiate_turn tool."""


_BUYER_DIALOGUE_PROMPT = """You are {buyer_label} (buyer_id={buyer_id}), a {buyer_workload} buyer negotiating ONE-ON-ONE with {seller_label} (seller_id={seller_id}).

You are NOT in a public-board auction. This is a 1:1 conversation. Every turn, you respond to your counterparty's specific last message — referencing what they said.

Your job:
- Need {buyer_qty_gpus} × {buyer_gpu_pref} for {buyer_duration}h starting between hours {buyer_window_start}-{buyer_window_end}.
- Urgency band: {buyer_urgency}.
- Your PRIVATE max willingness-to-pay: ${max_wtp:.2f}/GPU-hr. NEVER pay above this. NEVER reveal it.
- Your fallback if this fails: posted-price at roughly ${posted_price:.2f}/GPU-hr (with friction cost ~${friction_cost:.0f}).

Your strategy:
- ARGUE down their price with concrete reasons: posted-price comparison, other sellers' offers in the market, your time-window flexibility (or lack of it).
- When you raise your bid, attach a CONDITION: "I'll go to $X if you guarantee no interruption" or "I'll pay $Y if you bundle the second slot at the same rate."
- Reference specific signals: "another seller is at $Z for the same GPU" or "your stated min commitment is N hours; I can do that if you drop the rate."
- Walk away if the seller stays well above your max — your fallback is real.

Avoid:
- Restating your bid without a reason.
- Round-number concessions ("OK $5") without a story.
- Revealing your max WTP, your real urgency score, or your internal deadline.

Counterparty's situation (PUBLIC ONLY):
- Slot {slot_id}: {gpu_type} × {seller_qty_gpus} GPUs, {seller_duration_hours}h at hour {seller_start_slot}
- Their advertised list price: ${list_price:.2f}/GPU-hr
- Volume discounts: {volume_discount_summary}

The market context (other concurrent activity):
{market_context}

Emit your move via the negotiate_turn tool."""


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_bilateral_dialogue(
    *,
    buyer: Buyer,
    seller: Seller,
    slot: CapacitySlot,
    opening_seller_price: float,
    opening_buyer_price: float,
    max_turns: int = 6,
    buyer_client: LLMClient,
    seller_client: LLMClient,
    market: Market | None = None,
    posted_price_estimate: float | None = None,
    friction_cost_estimate: float = 200.0,
) -> BilateralDialogueResult:
    """Run a strict alternating 1:1 dialogue between buyer and seller.

    Turn order: seller opens (turn 1), then buyer (turn 2), seller (turn 3), ...
    Each turn produces a `negotiate_turn` tool call. The dialogue ends when:
    1. One side accepts the other's last counter-price → ``closed=True``.
    2. One side walks away → ``walked_away_by`` set.
    3. ``max_turns`` reached → ``closed=False``.

    Returns a ``BilateralDialogueResult`` with the full turn list (renderable
    via ``render_dialogue``).
    """
    pair_id = f"{buyer.id}↔{seller.id}/{slot.id}"
    result = BilateralDialogueResult(pair_id=pair_id)

    # Conversation memory — what each agent has seen so far.
    last_other_price = opening_buyer_price  # seller goes first; "their" last is the opening buyer pos
    last_other_argument = (
        f"My opening bid: ${opening_buyer_price:.2f}/GPU-hr. "
        f"My fallback is posted-price at roughly ${posted_price_estimate or 0:.2f}."
    )

    # Persuasive system prompts pre-rendered with this pair's facts.
    seller_system = _build_seller_prompt(seller, slot, buyer, market)
    buyer_system  = _build_buyer_prompt(
        buyer, seller, slot, market,
        posted_price_estimate=posted_price_estimate or slot.reserve_per_gpu_hr * 1.4,
        friction_cost_estimate=friction_cost_estimate,
    )

    # Each side maintains its own message history.
    seller_history: list[dict[str, str]] = []
    buyer_history: list[dict[str, str]] = []

    for turn_n in range(1, max_turns + 1):
        is_seller_turn = (turn_n % 2 == 1)
        speaker = "seller" if is_seller_turn else "buyer"
        client = seller_client if is_seller_turn else buyer_client
        history = seller_history if is_seller_turn else buyer_history
        system = seller_system if is_seller_turn else buyer_system

        user_msg = _render_user_message(
            turn_n=turn_n,
            opening_seller_price=opening_seller_price,
            opening_buyer_price=opening_buyer_price,
            counterparty_last_price=last_other_price,
            counterparty_last_argument=last_other_argument,
            is_seller_turn=is_seller_turn,
        )
        history.append({"role": "user", "content": user_msg})

        try:
            tc = client.generate(
                system_prompt=system,
                messages=history,
                tools=[_DIALOGUE_TOOL],
                max_tokens=400,
                temperature=0.5,
            )
        except Exception as e:
            argument = f"(LLM error: {e})"
            tc_args: dict[str, Any] = {
                "action": "walk_away",
                "argument": argument,
                "references_alternative": False,
            }
            tc_name = "negotiate_turn"
        else:
            tc_args = tc.arguments or {}
            tc_name = tc.tool_name or "__no_tool__"

        action = str(tc_args.get("action", "walk_away"))
        proposed_price = tc_args.get("proposed_price_per_gpu_hr")
        condition = str(tc_args.get("condition", "") or "")
        argument = str(tc_args.get("argument", "") or "")
        refs_alt = bool(tc_args.get("references_alternative", False))

        # Push assistant turn into history (compact form).
        history.append({
            "role": "assistant",
            "content": f"[turn {turn_n}: {tc_name}({json.dumps(tc_args, default=str)})]",
        })

        if is_seller_turn:
            speaker_id, speaker_label = seller.id, seller.label
            speaker_model = getattr(client, "model", "?")
            speaker_provider = getattr(client, "provider", "?")
        else:
            speaker_id, speaker_label = buyer.id, buyer.label
            speaker_model = getattr(client, "model", "?")
            speaker_provider = getattr(client, "provider", "?")

        turn = DialogueTurn(
            turn_n=turn_n,
            speaker=speaker,
            speaker_id=speaker_id,
            speaker_label=speaker_label,
            speaker_model=speaker_model,
            speaker_provider=speaker_provider,
            action=action if action in ("counter", "accept", "walk_away") else "walk_away",
            proposed_price_per_gpu_hr=(
                float(proposed_price) if isinstance(proposed_price, (int, float)) else None
            ),
            condition=condition,
            argument=argument,
            references_alternative=refs_alt,
        )
        result.turns.append(turn)

        if turn.action == "accept":
            result.closed = True
            result.closing_price = last_other_price
            return result
        if turn.action == "walk_away":
            result.walked_away_by = speaker_id
            return result

        # counter — feed the new price/argument into the OTHER side's user message.
        if turn.proposed_price_per_gpu_hr is not None:
            last_other_price = turn.proposed_price_per_gpu_hr
        last_other_argument = argument
        if condition and condition.strip().lower() not in ("flat", "flat — no condition", "none", ""):
            last_other_argument += f" Condition: {condition}."

    return result


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------


def _build_seller_prompt(seller: Seller, slot: CapacitySlot, buyer: Buyer, market: Market | None) -> str:
    market_ctx = _render_market_context(market, exclude_buyer_id=buyer.id)
    return _SELLER_DIALOGUE_PROMPT.format(
        seller_label=seller.label, seller_id=seller.id,
        buyer_label=buyer.label, buyer_id=buyer.id,
        slot_id=slot.id, gpu_type=slot.gpu_type.value,
        qty_gpus=slot.qty, duration_hours=slot.duration, start_slot=slot.start,
        reserve_per_gpu_hr=slot.reserve_per_gpu_hr,
        buyer_workload="training/inference",
        buyer_qty_gpus=buyer.job.qty,
        buyer_gpu_pref=", ".join(g.value for g in buyer.job.acceptable_gpus),
        buyer_duration=buyer.job.duration,
        buyer_window_start=buyer.job.earliest_start,
        buyer_window_end=buyer.job.latest_finish,
        buyer_urgency=("urgent" if buyer.urgency > 0.66 else "soon" if buyer.urgency > 0.33 else "routine"),
        market_context=market_ctx,
    )


def _build_buyer_prompt(
    buyer: Buyer, seller: Seller, slot: CapacitySlot, market: Market | None,
    posted_price_estimate: float, friction_cost_estimate: float,
) -> str:
    market_ctx = _render_market_context(market, exclude_buyer_id=buyer.id)
    volume_discount = "none advertised"  # legacy v0.2 sellers don't carry tiers
    return _BUYER_DIALOGUE_PROMPT.format(
        buyer_label=buyer.label, buyer_id=buyer.id,
        buyer_workload="training/inference",
        seller_label=seller.label, seller_id=seller.id,
        buyer_qty_gpus=buyer.job.qty,
        buyer_gpu_pref=", ".join(g.value for g in buyer.job.acceptable_gpus),
        buyer_duration=buyer.job.duration,
        buyer_window_start=buyer.job.earliest_start,
        buyer_window_end=buyer.job.latest_finish,
        buyer_urgency=("urgent" if buyer.urgency > 0.66 else "soon" if buyer.urgency > 0.33 else "routine"),
        max_wtp=buyer.job.max_value_per_gpu_hr,
        posted_price=posted_price_estimate,
        friction_cost=friction_cost_estimate,
        slot_id=slot.id, gpu_type=slot.gpu_type.value,
        seller_qty_gpus=slot.qty, seller_duration_hours=slot.duration,
        seller_start_slot=slot.start,
        list_price=slot.reserve_per_gpu_hr * 1.5,  # synthetic v0.2 list price
        volume_discount_summary=volume_discount,
        market_context=market_ctx,
    )


def _render_market_context(market: Market | None, *, exclude_buyer_id: str | None) -> str:
    """A short text summary of the surrounding market state. Public info only."""
    if market is None:
        return "(no market context provided)"
    other_buyers = [b for b in market.buyers if b.id != exclude_buyer_id]
    lines = []
    if other_buyers:
        lines.append(f"{len(other_buyers)} other buyers in the market:")
        for b in other_buyers[:4]:
            gpus = "/".join(g.value for g in b.job.acceptable_gpus)
            lines.append(
                f"  - {b.id} {b.label} wants {b.job.qty}× {gpus} for {b.job.duration}h"
            )
    sellers = market.sellers
    if sellers:
        lines.append(f"{len(sellers)} sellers active:")
        for s in sellers[:4]:
            slot_summary = ", ".join(
                f"{sl.gpu_type.value}×{sl.qty}@hr{sl.start}" for sl in s.capacity_slots[:2]
            )
            lines.append(f"  - {s.id} {s.label}: {slot_summary}")
    return "\n".join(lines) if lines else "(empty market)"


def _render_user_message(
    *, turn_n: int,
    opening_seller_price: float, opening_buyer_price: float,
    counterparty_last_price: float, counterparty_last_argument: str,
    is_seller_turn: bool,
) -> str:
    if turn_n == 1:
        return (
            f"It's your opening turn.\n\n"
            f"The buyer's opening position: ${opening_buyer_price:.2f}/GPU-hr "
            f"(public bid). They argued: \"{counterparty_last_argument}\"\n\n"
            f"Make your opening counter. Justify your price with a CONCRETE reason "
            f"(scarcity, slot quality, or an alternative). Use the negotiate_turn tool."
        )
    if turn_n == 2:
        return (
            f"Turn 2 — your reply.\n\n"
            f"The seller opened at ${counterparty_last_price:.2f}/GPU-hr. "
            f"They said: \"{counterparty_last_argument}\"\n\n"
            f"Argue down. Reference your fallback or other sellers' prices. "
            f"Use the negotiate_turn tool."
        )
    side = "seller" if is_seller_turn else "buyer"
    other = "buyer" if is_seller_turn else "seller"
    return (
        f"Turn {turn_n}.\n\n"
        f"The {other}'s most recent counter: ${counterparty_last_price:.2f}/GPU-hr.\n"
        f"They argued: \"{counterparty_last_argument}\"\n\n"
        f"Respond. Either counter (with a NEW price + condition + argument), "
        f"accept their last offer, or walk away. As the {side}, your strategic "
        f"position is: don't restate, ARGUE."
    )


__all__ = [
    "DialogueTurn",
    "BilateralDialogueResult",
    "run_bilateral_dialogue",
]
