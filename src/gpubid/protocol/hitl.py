"""Phase 9 — Human-in-the-loop intervention scaffold.

SCAFFOLDED. Triggers and decision types are implemented; the notebook-side
ipywidgets surfacer and threading event are stubbed because they need a live
notebook environment plus LLM-driven negotiation flow to demonstrate.

Per spec §10.1: triggers are
    constraint_violation_imminent
    deadlock
    low_confidence_close
    ambiguous_requirement
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Literal, Optional

from pydantic import BaseModel, ConfigDict

from gpubid.domain.offers import OfferTerms


class HITLTrigger(str, Enum):
    CONSTRAINT_VIOLATION_IMMINENT = "constraint_violation_imminent"
    DEADLOCK = "deadlock"
    LOW_CONFIDENCE_CLOSE = "low_confidence_close"
    AMBIGUOUS_REQUIREMENT = "ambiguous_requirement"


@dataclass
class HITLEvent:
    trigger: HITLTrigger
    agent_id: str
    round_n: int
    proposed_offer: OfferTerms | None
    note: str = ""


class HITLDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    action: Literal["proceed_as_proposed", "override_offer", "walk_away", "abort_run"]
    override_offer: Optional[OfferTerms] = None
    note: str = ""


@dataclass
class HITLPolicy:
    """Wraps a surfacer (notebook widget or auto-proceed for headless tests)."""

    enabled: bool
    surfacer: Callable[[HITLEvent], HITLDecision]

    def maybe_intervene(self, event: HITLEvent) -> HITLDecision:
        """Surface the event if enabled; auto-proceed if disabled."""
        if not self.enabled:
            return HITLDecision(action="proceed_as_proposed")
        return self.surfacer(event)


def auto_proceed_surfacer(event: HITLEvent) -> HITLDecision:
    """Test/headless surfacer that always proceeds. Logged via the event note."""
    return HITLDecision(
        action="proceed_as_proposed",
        note=f"auto-proceed (headless): {event.trigger.value}",
    )


__all__ = [
    "HITLTrigger",
    "HITLEvent",
    "HITLDecision",
    "HITLPolicy",
    "auto_proceed_surfacer",
]
