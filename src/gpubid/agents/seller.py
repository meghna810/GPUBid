"""LLM-backed seller agent.

Symmetric to `LLMBuyer`. The seller's per-slot reserves go in the system prompt;
only the public board is passed each round. Tool-call output is translated to
the same `AgentAction` shape used by the deterministic seller.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from gpubid.agents.deterministic import AgentAction
from gpubid.agents.prompts import seller_system_prompt, seller_tool_specs
from gpubid.engine.board import RunState, make_offer_id
from gpubid.llm import LLMClient, ToolCall, ToolSpec
from gpubid.schema import (
    GPUType,
    InterruptionTolerance,
    Offer,
    OfferKind,
    Seller,
)


@dataclass
class LLMSeller:
    seller_id: str
    client: LLMClient
    max_tokens: int = 768
    temperature: float = 0.3
    history: list[dict[str, str]] = field(default_factory=list)

    def decide(self, state: RunState, round_n: int, max_rounds: int) -> AgentAction:
        seller = next(s for s in state.market.sellers if s.id == self.seller_id)

        board_text = _render_board_for_seller(seller, state, round_n, max_rounds)
        self.history.append({"role": "user", "content": board_text})

        system = seller_system_prompt(seller, regime=state.market.regime, max_rounds=max_rounds)
        tools = [ToolSpec(**t) for t in seller_tool_specs()]

        try:
            tc = self.client.generate(
                system_prompt=system,
                messages=self.history,
                tools=tools,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        except Exception as e:
            self.history.append({"role": "assistant", "content": f"[error: {e}]"})
            return AgentAction(reasoning=f"(LLM error: {e})")

        self.history.append({
            "role": "assistant",
            "content": f"[tool: {tc.tool_name}({json.dumps(tc.arguments, default=str)})]",
        })

        return _seller_tool_call_to_action(tc, seller, round_n)


# ---------------------------------------------------------------------------
# Tool call → AgentAction
# ---------------------------------------------------------------------------


def _seller_tool_call_to_action(tc: ToolCall, seller: Seller, round_n: int) -> AgentAction:
    if tc.tool_name == "post_ask":
        offer = _try_build_ask(tc.arguments, seller, round_n)
        if offer is not None:
            return AgentAction(new_offers=(offer,), reasoning=offer.reasoning)
        return AgentAction(reasoning=f"(malformed post_ask: {tc.arguments})")

    if tc.tool_name == "accept_bid":
        target = tc.arguments.get("target_offer_id")
        if isinstance(target, str) and target:
            return AgentAction(
                accept_offer_ids=(target,),
                reasoning=str(tc.arguments.get("reasoning", "")),
            )
        return AgentAction(reasoning="(accept_bid missing target_offer_id)")

    if tc.tool_name == "do_nothing":
        return AgentAction(reasoning=str(tc.arguments.get("reasoning", "")))

    return AgentAction(reasoning=f"(unknown tool: {tc.tool_name})")


def _try_build_ask(args: dict, seller: Seller, round_n: int) -> Optional[Offer]:
    slot_id = args.get("slot_id")
    # Validate slot_id belongs to this seller.
    valid_slot_ids = {sl.id for sl in seller.capacity_slots}
    if not isinstance(slot_id, str) or slot_id not in valid_slot_ids:
        return None
    try:
        return Offer(
            id=make_offer_id(OfferKind.ASK, slot_id, round_n),
            round_n=round_n,
            from_id=seller.id,
            kind=OfferKind.ASK,
            slot_id=slot_id,
            price_per_gpu_hr=float(args["price_per_gpu_hr"]),
            qty=int(args["qty"]),
            gpu_type=GPUType(args["gpu_type"]),
            start=int(args["start"]),
            duration=int(args["duration"]),
            interruption_tolerance=InterruptionTolerance(args["interruption_tolerance"]),
            reasoning=str(args.get("reasoning", "")),
        )
    except (KeyError, ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Board rendering for the LLM
# ---------------------------------------------------------------------------


def _render_board_for_seller(seller: Seller, state: RunState, round_n: int, max_rounds: int) -> str:
    """Textual board snapshot for the seller. Reasoning text is redacted from offers."""
    lines: list[str] = [f"Round {round_n} of {max_rounds}.\n"]

    snap = state.public_snapshot(round_n)

    own_slot_ids = {sl.id for sl in seller.capacity_slots}
    own_asks = [a for a in snap.asks if a.slot_id in own_slot_ids]
    other_asks = [a for a in snap.asks if a.slot_id not in own_slot_ids]

    lines.append(f"Your current asks ({len(own_asks)}):")
    if own_asks:
        for a in own_asks:
            lines.append(
                f"  - {a.slot_id} at ${a.price_per_gpu_hr:.2f}/GPU-hr (round {a.round_n})"
            )
    else:
        lines.append("  (none — your slots are unposted)")

    if other_asks:
        lines.append(f"\nOther sellers' asks ({len(other_asks)}):")
        for a in other_asks:
            lines.append(
                f"  - id={a.id} from={a.from_id} {a.gpu_type.value}×{a.qty} "
                f"at ${a.price_per_gpu_hr:.2f}/GPU-hr "
                f"slot{a.start:02d}+{a.duration}h"
            )

    if snap.bids:
        lines.append(f"\nActive bids from buyers ({len(snap.bids)}):")
        for b in snap.bids:
            lines.append(
                f"  - id={b.id} from={b.from_id} {b.gpu_type.value}×{b.qty} "
                f"at ${b.price_per_gpu_hr:.2f}/GPU-hr "
                f"slot{b.start:02d}+{b.duration}h tol={b.interruption_tolerance.value}"
            )
    else:
        lines.append("\nActive bids: (none yet)")

    # Remaining capacity per slot
    remaining_lines = []
    for sl in seller.capacity_slots:
        rem = state.slot_remaining_qty.get(sl.id, 0)
        marker = " [SOLD OUT]" if rem == 0 else f" ({rem}/{sl.qty} GPUs left)"
        remaining_lines.append(
            f"  - {sl.id}: {sl.gpu_type.value} ×{sl.qty} reserve=${sl.reserve_per_gpu_hr:.2f}{marker}"
        )
    lines.append("\nYour slot capacities (remaining):")
    lines.extend(remaining_lines)

    if snap.deals_so_far:
        lines.append(f"\n{len(snap.deals_so_far)} deals struck so far in this market.")

    lines.append(
        f"\n{max_rounds - round_n} rounds remain after this one. Choose ONE tool action."
    )
    return "\n".join(lines)


__all__ = ["LLMSeller"]
