"""LLM-backed buyer agent.

Uses an `LLMClient` (provider auto-detected by API key prefix in `gpubid.llm`).
The buyer's max value, urgency, and budget go in the system prompt; only the
public board state is passed in subsequent user messages. Tool-call output is
translated into the same `AgentAction` shape as the deterministic buyer, so
the round runner doesn't need to know the difference.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from gpubid.agents.deterministic import AgentAction
from gpubid.agents.prompts import buyer_system_prompt, buyer_tool_specs
from gpubid.engine.board import RunState, make_offer_id
from gpubid.llm import LLMClient, ToolCall, ToolSpec
from gpubid.schema import (
    Buyer,
    GPUType,
    InterruptionTolerance,
    Offer,
    OfferKind,
)


@dataclass
class LLMBuyer:
    buyer_id: str
    client: LLMClient
    max_tokens: int = 512
    temperature: float = 0.3
    history: list[dict[str, str]] = field(default_factory=list)

    def decide(self, state: RunState, round_n: int, max_rounds: int) -> AgentAction:
        buyer = next(b for b in state.market.buyers if b.id == self.buyer_id)

        board_text = _render_board_for_buyer(buyer, state, round_n, max_rounds)
        self.history.append({"role": "user", "content": board_text})

        system = buyer_system_prompt(buyer, max_rounds)
        tools = [ToolSpec(**t) for t in buyer_tool_specs()]

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

        # Append assistant turn to history (compact representation).
        self.history.append({
            "role": "assistant",
            "content": f"[tool: {tc.tool_name}({json.dumps(tc.arguments, default=str)})]",
        })

        return _buyer_tool_call_to_action(tc, buyer, round_n)


# ---------------------------------------------------------------------------
# Tool call → AgentAction
# ---------------------------------------------------------------------------


def _buyer_tool_call_to_action(tc: ToolCall, buyer: Buyer, round_n: int) -> AgentAction:
    if tc.tool_name == "post_bid":
        offer = _try_build_bid(tc.arguments, buyer, round_n)
        if offer is not None:
            return AgentAction(new_offers=(offer,), reasoning=offer.reasoning)
        return AgentAction(reasoning=f"(malformed post_bid: {tc.arguments})")

    if tc.tool_name == "accept_ask":
        target = tc.arguments.get("target_offer_id")
        if isinstance(target, str) and target:
            return AgentAction(
                accept_offer_ids=(target,),
                reasoning=str(tc.arguments.get("reasoning", "")),
            )
        return AgentAction(reasoning="(accept_ask missing target_offer_id)")

    if tc.tool_name == "do_nothing":
        return AgentAction(reasoning=str(tc.arguments.get("reasoning", "")))

    # Unknown / no tool — be safe and pass.
    return AgentAction(reasoning=f"(unknown tool: {tc.tool_name})")


def _try_build_bid(args: dict, buyer: Buyer, round_n: int) -> Optional[Offer]:
    try:
        return Offer(
            id=make_offer_id(OfferKind.BID, buyer.id, round_n),
            round_n=round_n,
            from_id=buyer.id,
            kind=OfferKind.BID,
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


def _render_board_for_buyer(buyer: Buyer, state: RunState, round_n: int, max_rounds: int) -> str:
    """Compact textual board snapshot for the LLM. Reasoning text is redacted from offers."""
    lines: list[str] = [f"Round {round_n} of {max_rounds}.\n"]

    snap = state.public_snapshot(round_n)

    if snap.asks:
        lines.append(f"Active asks ({len(snap.asks)}):")
        for a in snap.asks:
            lines.append(
                f"  - id={a.id} from={a.from_id} {a.gpu_type.value}×{a.qty} "
                f"at ${a.price_per_gpu_hr:.2f}/GPU-hr "
                f"slot{a.start:02d}+{a.duration}h tol={a.interruption_tolerance.value}"
            )
    else:
        lines.append("Active asks: (none)")

    other_bids = [b for b in snap.bids if b.from_id != buyer.id]
    if other_bids:
        lines.append(f"\nActive bids from other buyers ({len(other_bids)}):")
        for b in other_bids:
            lines.append(
                f"  - id={b.id} from={b.from_id} {b.gpu_type.value}×{b.qty} "
                f"at ${b.price_per_gpu_hr:.2f}/GPU-hr "
                f"slot{b.start:02d}+{b.duration}h tol={b.interruption_tolerance.value}"
            )
    else:
        lines.append("\nActive bids from other buyers: (none)")

    if snap.deals_so_far:
        lines.append(f"\n{len(snap.deals_so_far)} deals struck so far in this market.")

    own = state.buyer_bids.get(buyer.id)
    if own is not None:
        lines.append(
            f"\nYour previous bid: {own.gpu_type.value}×{own.qty} "
            f"at ${own.price_per_gpu_hr:.2f}/GPU-hr "
            f"slot{own.start:02d}+{own.duration}h tol={own.interruption_tolerance.value} "
            f"(round {own.round_n})."
        )
    else:
        lines.append("\nYou have no standing bid yet.")

    lines.append(
        f"\n{max_rounds - round_n} rounds remain after this one. Choose ONE tool action."
    )
    return "\n".join(lines)


__all__ = ["LLMBuyer"]
