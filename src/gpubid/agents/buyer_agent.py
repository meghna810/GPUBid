"""Phase 3 — LLM-driven buyer agent with translate step.

The translate step takes a natural-language ``EndRequirement`` and emits a
schema-valid (``BuyerPublicProfile``, ``BuyerPrivateProfile``) pair via
structured-output (Anthropic tool-use / OpenAI JSON mode).

SCAFFOLDED — needs LLM-fixture recording before tests can run live. The
translate logic itself is implemented; what's missing is the fixture data
under ``tests/fixtures/responses/`` that lets CI run without keys.

To complete:
1. Set ``ANTHROPIC_API_KEY`` (or ``OPENAI_API_KEY``).
2. Run ``pytest tests/test_buyer_agent.py --live`` once with --live to record
   real LLM responses into ``tests/fixtures/responses/``.
3. Commit those fixtures. CI will replay them via ``RecordedFixtureClient``.

Per spec §4.3: there is no deterministic path. Calling ``BuyerAgent(None)``
must raise ``MissingAPIKeyError``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from gpubid.config import settings
from gpubid.domain.profiles import (
    BuyerPrivateProfile,
    BuyerPublicProfile,
)
from gpubid.domain.requirements import EndRequirement
from gpubid.errors import MissingAPIKeyError, ProfileValidationError
from gpubid.llm import LLMClient, ToolSpec


# Tool schema for the LLM's structured output.
_TRANSLATE_TOOL = ToolSpec(
    name="emit_buyer_profile",
    description=(
        "Emit a structured buyer public+private profile pair derived from the "
        "natural-language requirement. The public profile is what sellers will see; "
        "the private profile stays with the buyer agent. Be honest about urgency, "
        "qty, and duration based on the persona's stated needs."
    ),
    parameters={
        "type": "object",
        "properties": {
            "public": {
                "type": "object",
                "properties": {
                    "buyer_id": {"type": "string"},
                    "display_name": {"type": "string"},
                    "workload_category": {"type": "string", "enum": [
                        "training", "fine_tuning", "inference_batch",
                        "inference_realtime", "evaluation_sweep",
                    ]},
                    "gpu_type_preferences": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["H100", "A100", "L40S"]},
                        "minItems": 1,
                    },
                    "qty_gpus": {"type": "integer", "minimum": 1},
                    "duration_hours": {"type": "number", "exclusiveMinimum": 0},
                    "time_window": {
                        "type": "object",
                        "properties": {
                            "earliest_start_slot": {"type": "integer", "minimum": 0, "maximum": 23},
                            "latest_finish_slot": {"type": "integer", "minimum": 1, "maximum": 24},
                        },
                        "required": ["earliest_start_slot", "latest_finish_slot"],
                    },
                    "interruption_tolerance": {"type": "string", "enum": [
                        "none", "checkpoint_15min", "checkpoint_60min", "any",
                    ]},
                    "urgency_band": {"type": "string", "enum": ["routine", "soon", "urgent"]},
                },
                "required": [
                    "buyer_id", "display_name", "workload_category",
                    "gpu_type_preferences", "qty_gpus", "duration_hours",
                    "time_window", "interruption_tolerance", "urgency_band",
                ],
            },
            "private": {
                "type": "object",
                "properties": {
                    "max_willingness_to_pay": {"type": "number", "exclusiveMinimum": 0},
                    "urgency_score": {"type": "number", "minimum": 0, "maximum": 1},
                    "internal_deadline_slot": {"type": ["integer", "null"]},
                    "budget_remaining_usd": {"type": "number", "exclusiveMinimum": 0},
                    "business_context_summary": {"type": "string"},
                },
                "required": [
                    "max_willingness_to_pay", "urgency_score",
                    "budget_remaining_usd", "business_context_summary",
                ],
            },
        },
        "required": ["public", "private"],
    },
)


_SYSTEM_PROMPT = """You are a buyer agent in a GPU-compute marketplace.

Your job: take a CEO's natural-language requirement and emit a structured pair of profiles.
- The PUBLIC profile is what sellers will see during broadcast.
- The PRIVATE profile stays with you only. Sellers MUST NOT see your max willingness-to-pay or your exact urgency score.

Be conservative when extracting numeric ranges from fuzzy language. Err on the
side of producing valid profiles over guessing. Explicit cues like "must finish by Friday"
become the time_window or internal_deadline_slot. Phrases like "whatever it costs"
mean high willingness-to-pay AND urgent urgency_band.

ALWAYS set business_context_summary to the requirement's raw_text verbatim.
Downstream prompts ground on this string."""


@dataclass
class BuyerAgent:
    """LLM-backed buyer agent. Owns the translate step.

    Per spec §4.3: there is no deterministic fallback. Constructing with
    ``llm_client=None`` raises ``MissingAPIKeyError``.
    """

    llm_client: LLMClient

    def __post_init__(self) -> None:
        if self.llm_client is None:
            raise MissingAPIKeyError(
                "BuyerAgent requires a real LLM client. v0.3 has no deterministic fallback. "
                "Pass either an AnthropicClient or OpenAIClient via gpubid.llm.make_client()."
            )

    def translate(
        self,
        requirement: EndRequirement,
        rng: np.random.Generator,
    ) -> tuple[BuyerPublicProfile, BuyerPrivateProfile]:
        """Convert NL requirement -> structured public+private profiles.

        Failure modes:
            ProfileValidationError: after ``settings.provider.max_retries`` attempts the LLM
                still returned a payload that fails Pydantic validation. The error
                carries the last raw response.
            MissingAPIKeyError: not raised here directly (already raised in __post_init__).

        ``business_context_summary`` is set to ``requirement.raw_text`` verbatim;
        we do not let the LLM rewrite it. Downstream prompts ground on this string.
        """
        last_raw = ""
        last_validation_error: Exception | None = None

        for attempt in range(settings.provider.max_retries + 1):
            user_msg = self._build_user_message(requirement, last_validation_error, last_raw)
            temperature = 0.2 + 0.05 * float(rng.random())  # small jitter avoids identical retries

            tc = self.llm_client.generate(
                system_prompt=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
                tools=[_TRANSLATE_TOOL],
                max_tokens=1024,
                temperature=temperature,
            )
            last_raw = json.dumps(tc.arguments) if tc.arguments else tc.raw_text

            try:
                pub = BuyerPublicProfile(**tc.arguments["public"])
                priv_kwargs = dict(tc.arguments["private"])
                # Force exact pass-through of raw_text per spec §4.3.
                priv_kwargs["business_context_summary"] = requirement.raw_text
                priv = BuyerPrivateProfile(**priv_kwargs)
                return pub, priv
            except (KeyError, ValueError, TypeError) as e:
                last_validation_error = e
                continue

        raise ProfileValidationError(
            f"BuyerAgent.translate failed Pydantic validation after "
            f"{settings.provider.max_retries + 1} attempts on requirement "
            f"{requirement.requirement_id!r}. Last error: {last_validation_error}",
            last_raw_response=last_raw,
        )

    def _build_user_message(
        self,
        requirement: EndRequirement,
        last_error: Exception | None,
        last_raw: str,
    ) -> str:
        retry_block = ""
        if last_error is not None:
            retry_block = (
                f"\n\nRETRY: your previous response did not validate. "
                f"Error: {last_error}\n"
                f"Previous raw response (truncated): {last_raw[:500]!r}\n"
                f"Try again. Make sure required fields are present and types match."
            )
        return (
            f"Persona: {requirement.persona}\n"
            f"Requirement ID: {requirement.requirement_id}\n\n"
            f"--- raw_text ---\n{requirement.raw_text}\n\n"
            f"Emit a `emit_buyer_profile` tool call with both `public` and `private` "
            f"objects. Set `private.business_context_summary` to the raw_text above verbatim.{retry_block}"
        )


__all__ = ["BuyerAgent"]
