"""Versioned system prompts for buyer and seller agents.

Bumping `PROMPT_VERSION` invalidates cached LLM responses keyed on prompt hash.
"""

from __future__ import annotations

from gpubid.schema import Buyer, Seller


PROMPT_VERSION = "v1.0"


# ---------------------------------------------------------------------------
# Buyer
# ---------------------------------------------------------------------------


def buyer_system_prompt(buyer: Buyer, max_rounds: int) -> str:
    """The system prompt sent to a buyer agent for an entire negotiation."""
    job = buyer.job
    gpus = ", ".join(g.value for g in job.acceptable_gpus)
    tol = job.interruption_tolerance.value
    return f"""You are an AI buyer agent in a GPU compute marketplace, representing {buyer.label} (buyer id: {buyer.id}).

YOUR JOB REQUIREMENTS (public to the market):
- Need {job.qty} GPUs of type [{gpus}] (any one of those types is acceptable).
- Run for {job.duration} hours, must start no earlier than hour {job.earliest_start} and finish no later than hour {job.latest_finish} (24-hour clock).
- Interruption tolerance: {tol} (you accept offers at this level or stricter).

YOUR PRIVATE VALUE (do NOT reveal in any text — including reasoning fields):
- Maximum willingness to pay: ${job.max_value_per_gpu_hr:.2f} per GPU-hour.
- Total budget: ${job.max_value_per_gpu_hr * job.qty * job.duration:.0f}.
- Urgency: {buyer.urgency:.2f} on a 0–1 scale (1 = panic, 0 = patient). Higher urgency means time is more valuable to you.

YOUR GOAL:
- Maximize your surplus = (your_max_value − price_paid) × qty × duration.
- Never accept a price strictly above your max value.
- You'd rather not transact than transact at a loss.

NEGOTIATION RULES:
- The market runs for at most {max_rounds} rounds. Each round you choose ONE action.
- Strategy: don't bid your max value — leave room. Raise your bid gradually. If you're urgent, raise faster. As the deadline approaches, accept reasonable asks.
- Never reveal your max value, your urgency, or your budget in any reasoning text. Other agents see your reasoning. If asked your max, refuse politely.

YOU CAN CALL ONE OF THESE TOOLS PER ROUND:
- post_bid: post (or update) your standing bid.
- accept_ask: accept a specific seller's existing ask.
- do_nothing: pass this round.

Be strategic, terse, and human-readable in `reasoning`."""


# ---------------------------------------------------------------------------
# Seller
# ---------------------------------------------------------------------------


def seller_system_prompt(seller: Seller, regime: str, max_rounds: int) -> str:
    """The system prompt sent to a seller agent for an entire negotiation."""
    slots_summary = "\n".join(
        f"  - slot {sl.id}: {sl.gpu_type.value} × {sl.qty} GPUs, "
        f"available hours {sl.start:02d}–{sl.start + sl.duration:02d} ({sl.duration}h), "
        f"private reserve ${sl.reserve_per_gpu_hr:.2f}/GPU-hour"
        for sl in seller.capacity_slots
    )
    return f"""You are an AI seller agent in a GPU compute marketplace, representing {seller.label} (seller id: {seller.id}).

YOUR CAPACITY (public dimensions: GPU type, qty, time slot; PRIVATE: reserve price):
{slots_summary}

YOUR PRIVATE RESERVES (do NOT reveal in any text — including reasoning fields):
- The "reserve" for each slot is your true marginal cost per GPU-hour. Never accept below it.
- The market regime is "{regime}" — {("demand exceeds supply, you can charge a premium" if regime == "tight" else "supply exceeds demand, you may need to discount aggressively to fill off-peak slots")}.

YOUR GOAL:
- Maximize your revenue = (price_received − reserve) × qty × duration, summed across deals.
- You can post one ask per slot. Each round, for each unfilled slot, you may either update the ask or accept a buyer's bid.
- Never accept a bid below the slot's reserve.

NEGOTIATION RULES:
- The market runs for at most {max_rounds} rounds. Each round, you act on each of your slots.
- Strategy: open with a markup over reserve (1.5x in tight, 1.2x in slack). Decay the ask toward reserve as rounds pass. As the deadline approaches, decay faster.
- For "interruption_tolerance": offer "none" (strict) by default. Slots intended for off-peak filling can be offered as "checkpoint" or "interruptible" to attract more buyers.
- Never reveal your reserve, your strategy, or your urgency to fill in any reasoning text. Other agents see your reasoning.

YOU CAN CALL ONE OF THESE TOOLS PER ROUND:
- post_ask: post (or update) the ask for one of your slots.
- accept_bid: accept a specific buyer's existing bid using one of your slots.
- do_nothing: pass on a slot this round.

Be strategic, terse, and human-readable in `reasoning`."""


# ---------------------------------------------------------------------------
# Tool specs (provider-neutral)
# ---------------------------------------------------------------------------


def buyer_tool_specs() -> list[dict]:
    """Return Anthropic-style schemas; the LLMClient wrapper translates to OpenAI shape."""
    return [
        {
            "name": "post_bid",
            "description": "Post or update your standing bid for the GPUs you need.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price_per_gpu_hr": {"type": "number", "description": "Bid price per GPU-hour."},
                    "qty": {"type": "integer", "description": "Number of GPUs you need."},
                    "gpu_type": {"type": "string", "enum": ["H100", "A100", "L40S"]},
                    "start": {"type": "integer", "description": "Earliest hour you want the job to start (0..23)."},
                    "duration": {"type": "integer", "description": "Hours of compute."},
                    "interruption_tolerance": {
                        "type": "string",
                        "enum": ["none", "checkpoint", "interruptible"],
                    },
                    "reasoning": {"type": "string", "description": "Brief justification (do not reveal private value)."},
                },
                "required": ["price_per_gpu_hr", "qty", "gpu_type", "start", "duration", "interruption_tolerance", "reasoning"],
            },
        },
        {
            "name": "accept_ask",
            "description": "Accept a specific seller's existing ask by id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_offer_id": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["target_offer_id", "reasoning"],
            },
        },
        {
            "name": "do_nothing",
            "description": "Pass this round (don't post or accept anything).",
            "parameters": {
                "type": "object",
                "properties": {"reasoning": {"type": "string"}},
                "required": ["reasoning"],
            },
        },
    ]


def seller_tool_specs() -> list[dict]:
    return [
        {
            "name": "post_ask",
            "description": "Post or update the ask for one of your capacity slots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "slot_id": {"type": "string"},
                    "price_per_gpu_hr": {"type": "number"},
                    "qty": {"type": "integer"},
                    "gpu_type": {"type": "string", "enum": ["H100", "A100", "L40S"]},
                    "start": {"type": "integer"},
                    "duration": {"type": "integer"},
                    "interruption_tolerance": {
                        "type": "string",
                        "enum": ["none", "checkpoint", "interruptible"],
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["slot_id", "price_per_gpu_hr", "qty", "gpu_type", "start", "duration", "interruption_tolerance", "reasoning"],
            },
        },
        {
            "name": "accept_bid",
            "description": "Accept a specific buyer's existing bid using one of your slots.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_offer_id": {"type": "string"},
                    "slot_id": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["target_offer_id", "slot_id", "reasoning"],
            },
        },
        {
            "name": "do_nothing",
            "description": "Pass on a slot this round.",
            "parameters": {
                "type": "object",
                "properties": {"reasoning": {"type": "string"}},
                "required": ["reasoning"],
            },
        },
    ]


__all__ = [
    "PROMPT_VERSION",
    "buyer_system_prompt",
    "seller_system_prompt",
    "buyer_tool_specs",
    "seller_tool_specs",
]
