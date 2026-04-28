"""Where human-in-the-loop pays off — real-world use cases + automatic triggers.

The notebook's existing HITL cell shows the trigger taxonomy. This module
focuses on the *use cases*: which workloads / contracts genuinely need HITL,
and how the persuasion + regret analytics feed back into automatic escalation.

Used by the notebook's "HITL guidance" cell. No external dependencies — pure
data + a small render function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from gpubid.analysis.persuasion import PersuasionReport
from gpubid.viz.market_view import FONT_STACK


@dataclass(frozen=True)
class HITLUseCase:
    """One high-stakes scenario where automatic agent decisions need human review."""

    name: str
    why: str
    practical_threshold: str
    automation_signal: str   # which auto-detected signal should escalate
    risk_class: str          # "financial" | "regulatory" | "reputational" | "operational"


HITL_USE_CASES: tuple[HITLUseCase, ...] = (
    HITLUseCase(
        name="Multi-week reserved capacity (>$50k commitment)",
        why=(
            "Reserved-instance contracts lock in pricing and capacity for weeks. "
            "An agent that under-prices the reserve or over-commits the buyer's "
            "budget creates losses that can't be unwound by cancelling the next round."
        ),
        practical_threshold=(
            "Total contract value above ~$50k, or duration > 7 days. The "
            "marginal cost of 30s of human review is small relative to the size."
        ),
        automation_signal=(
            "Trigger HITL whenever proposed_offer.qty * duration * price > "
            "$50k OR duration > 168 hours."
        ),
        risk_class="financial",
    ),
    HITLUseCase(
        name="Regulated-industry workloads (HIPAA / FedRAMP / GDPR)",
        why=(
            "A buyer's compliance posture (region, attestation, audit trail) "
            "is qualitative — agents struggle to verify it from natural-language "
            "claims. Mistakes here are regulatory exposure, not just cost."
        ),
        practical_threshold=(
            "Any deal where the buyer's CEO requirement mentions HIPAA, "
            "FedRAMP, SOC-2, GDPR, or 'regulated', or where the seller's "
            "region attestation is uncertain."
        ),
        automation_signal=(
            "Keyword match in the requirement raw_text, OR seller compliance "
            "field is None."
        ),
        risk_class="regulatory",
    ),
    HITLUseCase(
        name="Foundation-model training runs (single-use, 24h+, all-or-nothing)",
        why=(
            "A training run that loses capacity mid-curriculum often has to "
            "restart from a checkpoint — sometimes from scratch. The cost of "
            "a wrong slot pick (interruptible vs strict) cascades into wasted "
            "GPU-hours that dwarf the negotiation savings."
        ),
        practical_threshold=(
            "duration > 24 hours AND interruption_tolerance == 'none' AND "
            "qty >= 8 GPUs."
        ),
        automation_signal=(
            "Combination of structural fields. Surface once before the buyer "
            "agent commits to an accept; never escalate twice for the same job."
        ),
        risk_class="operational",
    ),
    HITLUseCase(
        name="Counterparty showing manipulation signals",
        why=(
            "When the persuasion analytics flag the counterparty's reasoning "
            "as bluff / false_urgency / emotional_appeal, the platform should "
            "give the human a chance to counter — agents tend to reciprocate "
            "manipulative tactics rather than de-escalate."
        ),
        practical_threshold=(
            ">=2 distinct manipulation tags from the counterparty in the "
            "current run (bluff / false_urgency / emotional_appeal)."
        ),
        automation_signal=(
            "Persuasion judge tags counterparty's bubble; raise for review "
            "before next acceptance."
        ),
        risk_class="reputational",
    ),
    HITLUseCase(
        name="Ambiguous CEO requirement (translate confidence low)",
        why=(
            "If the buyer agent's translate step yielded a wide qty/duration "
            "range relative to the midpoint, downstream pricing decisions are "
            "very sensitive to where the agent landed. A 30-second clarification "
            "from the CEO eliminates expensive misalignment."
        ),
        practical_threshold=(
            "qty range width / midpoint > 50% OR duration range width > 24 hours."
        ),
        automation_signal=(
            "Compute (max - min) / mean on the EndRequirement.expected_qty_range."
        ),
        risk_class="operational",
    ),
    HITLUseCase(
        name="Repeat low-confidence accept",
        why=(
            "A buyer agent that closes 3 deals in a row at the bottom of its "
            "willingness-to-pay band may be miscalibrated. A human review every "
            "3rd deal of this shape catches systematic over-payment / under-payment."
        ),
        practical_threshold=(
            "n_consecutive_close_to_max_value >= 3 OR n_consecutive_close_to_reserve >= 3."
        ),
        automation_signal=(
            "Track distance from max_value (buyer) or reserve (seller) over a "
            "rolling window of 3 deals."
        ),
        risk_class="financial",
    ),
    HITLUseCase(
        name="Cross-organization deals (different cost centers)",
        why=(
            "When a buyer's job is funded by a different cost center than its "
            "default budget (research grant vs. eng OpEx vs. sales budget), the "
            "constraints are organizational, not technical. Agents shouldn't "
            "decide which budget to draw on."
        ),
        practical_threshold=(
            "Buyer's BuyerPrivateProfile.budget_remaining_usd is below the deal "
            "total OR job is tagged 'research' vs 'production'."
        ),
        automation_signal=(
            "Budget shortfall check at accept time."
        ),
        risk_class="financial",
    ),
)


# ---------------------------------------------------------------------------
# Detection from the persuasion analytics
# ---------------------------------------------------------------------------


@dataclass
class HITLAlert:
    """One escalation candidate produced from the live run's analytics."""

    use_case: HITLUseCase
    agent_id: str
    detail: str


def detect_alerts_from_persuasion(
    persuasion: PersuasionReport,
    *,
    manipulation_tags: tuple[str, ...] = ("bluff", "false_urgency", "emotional_appeal"),
    threshold: int = 2,
) -> list[HITLAlert]:
    """Scan the persuasion report for agents whose counterparty showed manipulation.

    Returns one HITLAlert per agent who, in this run, was facing a counterparty
    that the judge flagged with `>= threshold` manipulation tags.
    """
    use_case = next(
        (uc for uc in HITL_USE_CASES if uc.risk_class == "reputational"),
        HITL_USE_CASES[0],
    )

    # Per-agent count of manipulation tags
    manipulation_count: dict[str, int] = {}
    for ag in persuasion.agents.values():
        n = sum(ag.tag_counts.get(t, 0) for t in manipulation_tags)
        if n >= threshold:
            manipulation_count[ag.agent_id] = n

    alerts: list[HITLAlert] = []
    for aid, n in manipulation_count.items():
        ag = persuasion.agents[aid]
        # Find counterparties whose deals are at risk — anyone of opposite role
        for other in persuasion.agents.values():
            if other.role != ag.role:
                alerts.append(HITLAlert(
                    use_case=use_case,
                    agent_id=other.agent_id,
                    detail=(
                        f"counterparty {ag.agent_id} ({ag.label}) shows "
                        f"{n} manipulation tags this run — review before "
                        f"{other.label} commits."
                    ),
                ))
                break  # one alert per manipulator is enough
    return alerts


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


_RISK_COLOR = {
    "financial":     "#dc2626",
    "regulatory":    "#a855f7",
    "reputational":  "#ea580c",
    "operational":   "#0369a1",
}


def render_hitl_use_cases(use_cases: Optional[tuple[HITLUseCase, ...]] = None) -> str:
    """HTML guide to where HITL pays off in a real agentic GPU marketplace."""
    use_cases = use_cases or HITL_USE_CASES

    cards = []
    for uc in use_cases:
        color = _RISK_COLOR.get(uc.risk_class, "#6b7280")
        cards.append(
            f'<div style="background:#fff;border:1px solid #e5e7eb;border-left:4px solid {color};'
            f'border-radius:6px;padding:10px 14px;margin-bottom:8px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'<strong style="font-size:13px;color:#111;">{uc.name}</strong>'
            f'<span style="background:{color};color:#fff;padding:1px 8px;border-radius:8px;'
            f'font-size:9px;font-weight:600;text-transform:uppercase;">{uc.risk_class}</span>'
            f'</div>'
            f'<div style="font-size:12px;color:#374151;margin-top:6px;line-height:1.5;">'
            f'<strong>Why HITL:</strong> {uc.why}'
            f'</div>'
            f'<div style="font-size:11px;color:#6b7280;margin-top:4px;font-family:monospace;'
            f'background:#f9fafb;padding:6px 8px;border-radius:4px;">'
            f'<strong>Threshold:</strong> {uc.practical_threshold}'
            f'</div>'
            f'<div style="font-size:11px;color:#0369a1;margin-top:4px;font-family:monospace;'
            f'background:#eff6ff;padding:6px 8px;border-radius:4px;">'
            f'<strong>Signal:</strong> {uc.automation_signal}'
            f'</div>'
            f'</div>'
        )

    return (
        f'<div style="font-family:{FONT_STACK};">'
        f'<h3 style="margin:0 0 6px;">When HITL is worth the latency</h3>'
        f'<div style="font-size:12px;color:#6b7280;margin-bottom:10px;">'
        f'Each card lists a real-world scenario, the threshold for when human '
        f'review pays off, and the auto-detected signal that should escalate.'
        f'</div>'
        f'{"".join(cards)}'
        f'</div>'
    )


def render_hitl_alerts(alerts: list[HITLAlert]) -> str:
    """HTML for live alerts produced from the persuasion report this run."""
    if not alerts:
        return (
            '<div style="background:#dcfce7;border-left:3px solid #16a34a;'
            'padding:8px 12px;border-radius:4px;font-family:-apple-system,sans-serif;'
            'font-size:12px;color:#14532d;">'
            'No HITL escalations triggered this run — agents stayed within '
            'normal manipulation thresholds.'
            '</div>'
        )

    rows = []
    for a in alerts:
        color = _RISK_COLOR.get(a.use_case.risk_class, "#6b7280")
        rows.append(
            f'<div style="background:#fef3c7;border-left:3px solid {color};'
            f'padding:8px 12px;margin-bottom:6px;border-radius:4px;font-size:12px;">'
            f'<strong style="color:{color};">⚠ {a.use_case.name}</strong> '
            f'<span style="font-family:monospace;color:#9ca3af;font-size:10px;">'
            f'agent {a.agent_id}</span>'
            f'<div style="color:#374151;margin-top:4px;">{a.detail}</div>'
            f'</div>'
        )
    return f'<div style="font-family:{FONT_STACK};">{"".join(rows)}</div>'


__all__ = [
    "HITLUseCase",
    "HITLAlert",
    "HITL_USE_CASES",
    "detect_alerts_from_persuasion",
    "render_hitl_use_cases",
    "render_hitl_alerts",
]
