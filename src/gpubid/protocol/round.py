"""Phase 5 — Round protocol scaffold.

SCAFFOLDED — needs LLM-fixture recording before tests can validate end-to-end
multi-round transcripts. The data model and orchestration shape are here; the
actual ``run_negotiation`` body raises NotImplementedError until the fixture
client is wired through buyer/seller agent calls.

Per spec §6.4: each (buyer, eligible-seller) pair exchanges at most one bid
and one ask per round. A pair closes when one side accepts the other's last
offer. A pair can also walk away (one side emits a `walk_away` action).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from gpubid.domain.offers import OfferTerms
from gpubid.protocol.budget import BudgetPolicy, StopReason
from gpubid.schema import Deal, Market


@dataclass
class NegotiationState:
    round_n: int = 0
    active_pairs: list[tuple[str, str]] = field(default_factory=list)
    offers_this_round: list[OfferTerms] = field(default_factory=list)
    deals_closed: list[Deal] = field(default_factory=list)
    history: list[OfferTerms] = field(default_factory=list)
    token_usage_by_agent: dict[str, int] = field(default_factory=dict)
    no_progress_streak_by_pair: dict[tuple[str, str], int] = field(default_factory=dict)


@dataclass
class NegotiationResult:
    deals: list[Deal]
    final_state: NegotiationState
    stop_reason: StopReason


def run_negotiation(
    market: Market,
    buyer_runtime: Any,            # BuyerAgentRuntime — defined when LLM agents wired
    seller_runtime: Any,           # SellerAgentRuntime
    budget_policy: BudgetPolicy,
    hitl: Any | None = None,       # HITLPolicy
    on_round_end: Optional[Callable[[NegotiationState], None]] = None,
) -> NegotiationResult:
    """End-to-end negotiation across multiple rounds.

    SCAFFOLDED. The full implementation requires:
      - The LLM seller agent's eligibility-and-ask loop (Phase 5.4).
      - The LLM buyer agent's accept-or-counter loop (Phase 5.4).
      - Hooking into HITL trigger conditions (Phase 9).
      - Emitting on_round_end snapshots for the trading-floor animation.

    Until those are recorded against fixtures, this raises explicitly so
    callers don't silently get a no-op result.
    """
    raise NotImplementedError(
        "Phase 5 run_negotiation is scaffolded. The buyer/seller agent runtimes "
        "and the round loop need LLM fixtures recorded with API keys. See "
        "tests/fixtures/responses/ and the recording instructions in the v0.3 "
        "refactor spec §16.2."
    )


__all__ = ["NegotiationState", "NegotiationResult", "run_negotiation"]
