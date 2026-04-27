"""Layered budget policy for multi-round LLM negotiations.

We use THREE simultaneous bounds. Whichever fires first wins:

1. ``max_rounds`` — caps round count. Predictable, comparable across seeds.
2. ``per_run_token_cap`` — caps total token spend across all agents in this run.
   This is the operationally important cost control: dollars are token-shaped.
3. ``no_progress_streak_cap`` — if a (buyer, seller) pair has not moved closer
   for this many rounds, we declare deadlock and stop. Catches wheel-spinning
   without paying for it.

Why all three: pure round count wastes calls when agents converge in 2 rounds
and truncates real haggling on 6. Pure token caps produce uneven round counts
across seeds (hurts comparability). No-progress alone misses runaway-token
scenarios. Layered = belt + suspenders + brace.

The ``StopReason`` returned alongside the stop signal is reported in the
negotiation result and shown in forensics so we can see *why* a run ended.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class StopReason(str, Enum):
    """Why a negotiation run halted."""

    MAX_ROUNDS = "max_rounds"
    TOKEN_CAP = "token_cap"
    ALL_PAIRS_RESOLVED = "all_pairs_resolved"
    NO_PROGRESS = "no_progress"


class _NegotiationStateProtocol(Protocol):
    """The minimum interface ``BudgetPolicy.should_stop`` reads.

    Phase 5's full ``NegotiationState`` will satisfy this. Defining a Protocol
    here lets us write tests for the budget without depending on Phase 5's full
    state object.
    """

    round_n: int
    active_pairs: list[Any]
    token_usage_by_agent: dict[str, int]
    no_progress_streak_by_pair: dict[Any, int]


class BudgetPolicy(BaseModel):
    """Configuration for the three layered limits.

    Defaults align with ``gpubid.config.settings.negotiation``. Pass overrides
    explicitly for experiments (notebook cell 6 exposes max_rounds + token cap +
    no-progress threshold as sliders).
    """

    model_config = ConfigDict(frozen=True)

    max_rounds: int = Field(ge=1)
    per_run_token_cap: int = Field(ge=0)
    no_progress_streak_cap: int = Field(ge=1)

    def should_stop(self, state: _NegotiationStateProtocol) -> tuple[bool, StopReason | None]:
        """Decide whether the negotiation should halt now.

        Order of checks matters for the reported reason:

        1. ``ALL_PAIRS_RESOLVED`` — natural completion; reported even if other
           caps would also fire, so the user sees "everyone closed/walked"
           rather than "we hit our budget."
        2. ``MAX_ROUNDS`` — round count exceeded.
        3. ``TOKEN_CAP`` — total tokens across all agents exceeded.
        4. ``NO_PROGRESS`` — at least one pair has hit the no-progress streak.

        Returns ``(False, None)`` when the run should continue.
        """

        if not state.active_pairs:
            return True, StopReason.ALL_PAIRS_RESOLVED

        if state.round_n >= self.max_rounds:
            return True, StopReason.MAX_ROUNDS

        total_tokens = sum(state.token_usage_by_agent.values())
        if total_tokens >= self.per_run_token_cap:
            return True, StopReason.TOKEN_CAP

        if state.no_progress_streak_by_pair:
            worst = max(state.no_progress_streak_by_pair.values())
            if worst >= self.no_progress_streak_cap:
                return True, StopReason.NO_PROGRESS

        return False, None

    @classmethod
    def from_settings(cls) -> "BudgetPolicy":
        """Construct using defaults from ``gpubid.config.settings``."""
        from gpubid.config import settings as _settings
        return cls(
            max_rounds=_settings.negotiation.default_max_rounds,
            per_run_token_cap=_settings.negotiation.total_run_token_budget,
            no_progress_streak_cap=_settings.negotiation.no_progress_rounds_threshold,
        )


__all__ = ["BudgetPolicy", "StopReason"]
