"""Phase 3 — Natural-language end-requirements (CEO-style buyer briefs).

A 12-entry library of synthetic CEO/CTO requirements. Buyer agents receive
one of these as input and run their LLM-driven ``translate`` step to produce
the structured ``BuyerPublicProfile`` + ``BuyerPrivateProfile`` pair.

This module is fully implemented (no LLM required) — it's just data + a
sampler. The translate step itself lives in ``gpubid.agents.buyer_agent``
and DOES need an LLM.

Diversity rubric (per spec §4.2):
- (training, urgent), (training, soon), (training, routine)
- (fine_tuning, urgent), (fine_tuning, routine)
- (inference_realtime, urgent)
- (inference_batch, soon), (inference_batch, routine)
- (evaluation_sweep, soon)
- 3 narrative-variety entries: research lab burst, regulated industry, academia.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from gpubid.domain.profiles import UrgencyBand, WorkloadCategory


class EndRequirement(BaseModel):
    """One CEO-style brief that becomes a buyer profile after translation."""

    model_config = ConfigDict(frozen=True)

    requirement_id: str
    persona: str
    raw_text: str
    workload_category: WorkloadCategory
    expected_qty_range: tuple[int, int]
    expected_duration_range: tuple[float, float]
    expected_urgency_band: UrgencyBand


# 12 requirements covering the diversity rubric.
REQUIREMENT_LIBRARY: list[EndRequirement] = [
    EndRequirement(
        requirement_id="req_series_a_robotics_001",
        persona="CTO of a Series A robotics startup",
        raw_text=(
            "I'm the CTO of a Series A robotics startup. We're shipping autonomous "
            "fulfillment robots to two warehouses on Friday and need to run a final "
            "policy distillation pass over 80GB of trajectory data this week. H100s "
            "preferred but we can stomach A100s if the price difference is meaningful. "
            "I'd rather pay more than miss the deadline."
        ),
        workload_category="fine_tuning",
        expected_qty_range=(4, 8),
        expected_duration_range=(8.0, 24.0),
        expected_urgency_band="urgent",
    ),
    EndRequirement(
        requirement_id="req_climate_foundation_002",
        persona="ML lead at a climate-modeling research lab",
        raw_text=(
            "We're training a foundation weather model. 32 H100s ideally, but we "
            "have a soft 2-week window and don't need anything immediately. Cost is "
            "the binding constraint, not time."
        ),
        workload_category="training",
        expected_qty_range=(16, 32),
        expected_duration_range=(48.0, 168.0),
        expected_urgency_band="routine",
    ),
    EndRequirement(
        requirement_id="req_chatbot_eval_003",
        persona="evaluation engineer at a chatbot company",
        raw_text=(
            "Need to run a 5,000-prompt regression sweep on our latest checkpoint "
            "over the next 48 hours. Inference workload, A100s are fine, can checkpoint."
        ),
        workload_category="evaluation_sweep",
        expected_qty_range=(2, 4),
        expected_duration_range=(8.0, 24.0),
        expected_urgency_band="soon",
    ),
    EndRequirement(
        requirement_id="req_realtime_speech_004",
        persona="head of platform at a real-time speech startup",
        raw_text=(
            "Our speech model needs serving capacity for a customer pilot starting "
            "tomorrow morning. 4 GPUs, low latency, no interruptions. H100s only."
        ),
        workload_category="inference_realtime",
        expected_qty_range=(4, 4),
        expected_duration_range=(12.0, 24.0),
        expected_urgency_band="urgent",
    ),
    EndRequirement(
        requirement_id="req_quarterly_finetune_005",
        persona="staff ML engineer at a fintech",
        raw_text=(
            "Quarterly fraud-detection model fine-tune. Always plan ahead — running "
            "next week. A100s preferred, can use L40S. We checkpoint every 30 minutes."
        ),
        workload_category="fine_tuning",
        expected_qty_range=(4, 8),
        expected_duration_range=(12.0, 36.0),
        expected_urgency_band="routine",
    ),
    EndRequirement(
        requirement_id="req_reinforcement_006",
        persona="CEO of a Series B AI startup, on the eve of fundraising",
        raw_text=(
            "We need to demonstrate a 10x training-time improvement over our last "
            "result by next Tuesday for the investor pitch. 16 H100s, urgency is "
            "'whatever it costs'. Training run, can't be interrupted mid-curriculum."
        ),
        workload_category="training",
        expected_qty_range=(8, 16),
        expected_duration_range=(48.0, 96.0),
        expected_urgency_band="urgent",
    ),
    EndRequirement(
        requirement_id="req_overnight_inference_007",
        persona="data engineer at an e-commerce platform",
        raw_text=(
            "Overnight catalog re-embedding. 8M product images. Tonight or tomorrow "
            "night both work. L40S is the sweet spot — fully interruptible, will retry."
        ),
        workload_category="inference_batch",
        expected_qty_range=(2, 4),
        expected_duration_range=(8.0, 16.0),
        expected_urgency_band="soon",
    ),
    EndRequirement(
        requirement_id="req_academic_burst_008",
        persona="grad student two days before an ICLR rebuttal deadline",
        raw_text=(
            "Need to run three ablation experiments before Friday at 5pm. Each is "
            "about a 6-hour run on 4 A100s. I have lab credits but they're under "
            "review. Reviewer asked for them, can't say no."
        ),
        workload_category="training",
        expected_qty_range=(2, 4),
        expected_duration_range=(4.0, 8.0),
        expected_urgency_band="urgent",
    ),
    EndRequirement(
        requirement_id="req_compliance_inference_009",
        persona="CISO at a regulated healthcare company",
        raw_text=(
            "Inference workload for our HIPAA-compliant clinical-decision tool. "
            "Need uninterrupted A100s in a SOC-2 region for batch processing of "
            "patient charts every weekend. Routine operational need."
        ),
        workload_category="inference_batch",
        expected_qty_range=(2, 4),
        expected_duration_range=(6.0, 12.0),
        expected_urgency_band="routine",
    ),
    EndRequirement(
        requirement_id="req_research_lab_burst_010",
        persona="head of an academic NLP lab",
        raw_text=(
            "Funded research burst — we have 72 hours of compute we need to use "
            "before the grant period ends. 8 GPUs, any tier, any time within the "
            "next 3 days. Please don't let it expire."
        ),
        workload_category="training",
        expected_qty_range=(4, 8),
        expected_duration_range=(24.0, 72.0),
        expected_urgency_band="soon",
    ),
    EndRequirement(
        requirement_id="req_finetune_quick_011",
        persona="indie ML developer",
        raw_text=(
            "Side project. Need to fine-tune a 7B model on a tiny dataset, urgent — "
            "demo with a customer in 6 hours. Single H100 or A100 will do."
        ),
        workload_category="fine_tuning",
        expected_qty_range=(1, 2),
        expected_duration_range=(4.0, 6.0),
        expected_urgency_band="urgent",
    ),
    EndRequirement(
        requirement_id="req_eval_sweep_routine_012",
        persona="ML platform team at a mid-stage SaaS company",
        raw_text=(
            "Weekly regression evaluation sweep. We run it every Sunday night. "
            "8 hours, 2 GPUs, can use L40S — fully interruptible workload."
        ),
        workload_category="evaluation_sweep",
        expected_qty_range=(2, 2),
        expected_duration_range=(6.0, 10.0),
        expected_urgency_band="soon",
    ),
]


def sample_requirements(n: int, rng: np.random.Generator) -> list[EndRequirement]:
    """Sample without replacement, falling back to with-replacement only if n > library size."""
    lib = REQUIREMENT_LIBRARY
    if n <= len(lib):
        idxs = rng.choice(len(lib), size=n, replace=False)
    else:
        idxs = rng.choice(len(lib), size=n, replace=True)
    return [lib[int(i)] for i in idxs]


__all__ = ["EndRequirement", "REQUIREMENT_LIBRARY", "sample_requirements"]
