"""Prompt-variation experiment — does prompt engineering change negotiation outcomes?

We know the *model* matters (the cell 6.77 model-version tournament shows
this). The next interesting question is: holding both model and market
constant, does PROMPT STYLE change outcomes?

This module defines four variants of the buyer/seller dialogue prompt and a
driver that runs the same market four times — once per variant — so the user
can read off "aggressive prompts walk away more often", "few-shot prompts
close more deals at midpoint prices", etc.

Variants:
  - **standard**   The current production prompt (close-biased, polite).
  - **aggressive** Anchors hard, holds firm, willing to walk away earlier.
  - **cooperative** Concedes generously, signals flexibility, accepts faster.
  - **few_shot**   Standard prompt + 2 worked examples of past successful
                   bilateral negotiations the agent can pattern-match against.

Each variant is applied to BOTH sides (so a "cooperative buyer" plays a
"cooperative seller" in that run) — keeps the comparison clean. Mixed-variant
matchups are an obvious follow-up but produce 2× the runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from gpubid.llm import LLMClient
from gpubid.protocol.chat_market import ChatMarketRun, run_chat_market
from gpubid.schema import Market


# ---------------------------------------------------------------------------
# Variant prompts
# ---------------------------------------------------------------------------

# Each variant overrides the BLOCK between "YOUR PRIMARY GOAL" and "Avoid:" in
# the standard prompt. We monkeypatch in a controlled way: load the standard
# prompt module, swap the strategy section, run the market, swap it back.

_STANDARD_BUYER_STRATEGY = "STANDARD"  # sentinel — uses current prompt unchanged
_STANDARD_SELLER_STRATEGY = "STANDARD"


_AGGRESSIVE_SELLER_STRATEGY = """YOUR PRIMARY GOAL: MAXIMIZE MARGIN. The buyer needs your GPU more than you need their dollar this round. Hold firm; let them come to you. Open at a high markup (1.6-1.8x reserve), concede in TINY steps (3-5% per turn), and reference your alternatives ("I have other buyers interested at this rate") aggressively.

Your strategy:
- OPEN HIGH (1.6-1.8x reserve). Anchor hard — the midpoint moves toward you.
- ARGUE for your price aggressively: capacity scarcity is YOUR leverage. The buyer's deadline is THEIR problem, not yours.
- CONCEDE only 3-5% per turn. Force the buyer to make most of the movement.
- Walk away if buyer goes below 1.20x your reserve at any point — that signals they're not serious.

When to walk away:
- If buyer keeps insisting on prices below 1.20x your reserve after turn 3.
- The point of walk-away is to BOUNDARY-SET. You are not a charity."""


_AGGRESSIVE_BUYER_STRATEGY = """YOUR PRIMARY GOAL: MINIMIZE PRICE. The seller has perishable inventory; an unsold GPU-hour is worth zero to them. Hold firm; let them come to you. Open low (0.50-0.55× your max), climb in TINY steps (3-5% per turn), and reference posted-price/alternatives aggressively.

Your strategy:
- OPEN LOW (0.50-0.55× your max). Anchor hard.
- ARGUE down with concrete reasons: posted-price benchmark, walk-away threat, time-window flexibility.
- CLIMB only 3-5% per turn. Force the seller to make most of the movement.
- Walk away if seller stays above 0.85x your max after turn 3 — your fallback is real.

When to walk away:
- If seller refuses to come below 0.85x your max after turn 3.
- The point of walk-away is to BOUNDARY-SET. The seller has perishable inventory."""


_COOPERATIVE_SELLER_STRATEGY = """YOUR PRIMARY GOAL: CLOSE A DEAL. Every dollar above your reserve is pure margin. Build trust, signal flexibility, and accept early when the buyer is anywhere near reasonable.

Your strategy:
- OPEN MODERATE (1.20-1.30x reserve) — leave room but not too much.
- ARGUE briefly with concrete reasons; don't belabor the point.
- CONCEDE generously (10-20% per turn) when the buyer signals movement.
- ACCEPT EAGERLY when the buyer's price is 1.05x reserve or above — that's a fine deal.

When to walk away:
- Almost never. Walk-away is reserved for buyers who refuse to come above your reserve at all after multiple concessions."""


_COOPERATIVE_BUYER_STRATEGY = """YOUR PRIMARY GOAL: CLOSE A DEAL. The friction of going to posted-price (no negotiation) costs you real money. Build trust, signal flexibility, and accept early when the seller is anywhere near reasonable.

Your strategy:
- OPEN MODERATE (0.65-0.75× your max) — leave room but not too much.
- ARGUE briefly with concrete reasons; don't belabor the point.
- CLIMB generously (10-20% per turn) when the seller signals movement.
- ACCEPT EAGERLY when the seller's price is 0.85x your max or below — that's a fine deal.

When to walk away:
- Almost never. Walk-away is reserved for sellers who stay above your max even after multiple concessions."""


# Few-shot examples — concrete worked dialogues the agent can pattern-match.
_FEW_SHOT_EXAMPLES = """Below are two examples of past successful negotiations on this platform. Pattern your moves on these.

EXAMPLE 1 — Closed at workable price:
  Seller turn 1: "Opening at $5.40/hr. This is the only A100 in your time window with no interruption guarantee, which matches your training run."
  Buyer turn 2: "Posted-price is $4.80, so I'd expect a discount. Can you do $4.30 if I commit to the full 6 hours?"
  Seller turn 3: "I can do $4.80 if you commit to 6 hours and take the off-peak start. Otherwise $5.20."
  Buyer turn 4: "Off-peak works. Done at $4.80."
  ✓ Deal: $4.80/hr × 6h.

EXAMPLE 2 — Closed via volume commit:
  Seller turn 1: "Opening at $3.10/hr for 4 GPUs."
  Buyer turn 2: "I have a Friday deadline so I value this. Can you do $2.60 if I take 6 GPUs instead of 4?"
  Seller turn 3: "Yes — my volume tier kicks in at 5+ GPUs. I can do $2.50 for 6 GPUs at 5 hours."
  Buyer turn 4: "Accept at $2.50."
  ✓ Deal: $2.50/hr × 6 GPUs × 5h.

Now respond to the actual negotiation."""


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class PromptVariantOutcome:
    """Aggregate stats for one prompt variant run on the same market."""

    variant_name: str
    n_dialogues: int
    n_deals: int
    avg_turns_per_dialogue: float
    avg_closing_price_pct_of_max: float   # avg(closing_price / buyer.max_value)
    avg_buyer_surplus: float              # avg surplus per closed deal
    avg_seller_revenue: float
    n_walked: int

    @property
    def close_rate(self) -> float:
        return self.n_deals / max(1, self.n_dialogues)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


VARIANT_NAMES: tuple[str, ...] = ("standard", "aggressive", "cooperative", "few_shot")


def _patch_dialogue_prompts(variant: str):
    """Monkeypatch the dialogue module's prompts for one variant.

    Returns a tuple of (original_seller, original_buyer) which the caller must
    restore after the run.
    """
    from gpubid.protocol import dialogue

    if variant == "standard":
        return None  # no patching needed

    orig_seller = dialogue._SELLER_DIALOGUE_PROMPT
    orig_buyer = dialogue._BUYER_DIALOGUE_PROMPT

    if variant == "aggressive":
        new_seller = orig_seller.replace(
            "YOUR PRIMARY GOAL: CLOSE A DEAL.",
            _AGGRESSIVE_SELLER_STRATEGY.split("Your strategy:")[0].rstrip()
        ).replace(
            orig_seller.split("Your strategy:")[1].split("Avoid:")[0],
            "\n" + _AGGRESSIVE_SELLER_STRATEGY.split("Your strategy:")[1].split("When to walk away:")[0]
            + "\nWhen to walk away:" + _AGGRESSIVE_SELLER_STRATEGY.split("When to walk away:")[1] + "\n",
        )
        new_buyer = orig_buyer.replace(
            "YOUR PRIMARY GOAL: CLOSE A DEAL.",
            _AGGRESSIVE_BUYER_STRATEGY.split("Your strategy:")[0].rstrip()
        ).replace(
            orig_buyer.split("Your strategy:")[1].split("Avoid:")[0],
            "\n" + _AGGRESSIVE_BUYER_STRATEGY.split("Your strategy:")[1].split("When to walk away:")[0]
            + "\nWhen to walk away:" + _AGGRESSIVE_BUYER_STRATEGY.split("When to walk away:")[1] + "\n",
        )
    elif variant == "cooperative":
        new_seller = orig_seller.replace(
            "YOUR PRIMARY GOAL: CLOSE A DEAL.",
            _COOPERATIVE_SELLER_STRATEGY.split("Your strategy:")[0].rstrip()
        ).replace(
            orig_seller.split("Your strategy:")[1].split("Avoid:")[0],
            "\n" + _COOPERATIVE_SELLER_STRATEGY.split("Your strategy:")[1].split("When to walk away:")[0]
            + "\nWhen to walk away:" + _COOPERATIVE_SELLER_STRATEGY.split("When to walk away:")[1] + "\n",
        )
        new_buyer = orig_buyer.replace(
            "YOUR PRIMARY GOAL: CLOSE A DEAL.",
            _COOPERATIVE_BUYER_STRATEGY.split("Your strategy:")[0].rstrip()
        ).replace(
            orig_buyer.split("Your strategy:")[1].split("Avoid:")[0],
            "\n" + _COOPERATIVE_BUYER_STRATEGY.split("Your strategy:")[1].split("When to walk away:")[0]
            + "\nWhen to walk away:" + _COOPERATIVE_BUYER_STRATEGY.split("When to walk away:")[1] + "\n",
        )
    elif variant == "few_shot":
        # Append few-shot examples to the standard prompt.
        new_seller = orig_seller + "\n\n" + _FEW_SHOT_EXAMPLES
        new_buyer = orig_buyer + "\n\n" + _FEW_SHOT_EXAMPLES
    else:
        raise ValueError(f"Unknown variant: {variant!r}")

    dialogue._SELLER_DIALOGUE_PROMPT = new_seller
    dialogue._BUYER_DIALOGUE_PROMPT = new_buyer

    return (orig_seller, orig_buyer)


def _restore_dialogue_prompts(restore: Optional[tuple]) -> None:
    if restore is None:
        return
    from gpubid.protocol import dialogue
    dialogue._SELLER_DIALOGUE_PROMPT, dialogue._BUYER_DIALOGUE_PROMPT = restore


def _summarize_run(name: str, run: ChatMarketRun) -> PromptVariantOutcome:
    n_dialogues = len(run.dialogues)
    n_deals = len(run.deals)
    n_walked = sum(1 for d in run.dialogues if d.walked_away_by)
    avg_turns = sum(len(d.turns) for d in run.dialogues) / max(1, n_dialogues)

    # avg closing price as fraction of buyer max
    pct_of_max = []
    surpluses = []
    revenues = []
    for d in run.deals:
        buyer = next(b for b in run.market.buyers if b.id == d.buyer_id)
        slot = next(
            sl for s in run.market.sellers for sl in s.capacity_slots if sl.id == d.slot_id
        )
        pct_of_max.append(d.price_per_gpu_hr / max(0.01, buyer.job.max_value_per_gpu_hr))
        surpluses.append((buyer.job.max_value_per_gpu_hr - d.price_per_gpu_hr) * d.qty * d.duration)
        revenues.append((d.price_per_gpu_hr - slot.reserve_per_gpu_hr) * d.qty * d.duration)

    return PromptVariantOutcome(
        variant_name=name,
        n_dialogues=n_dialogues,
        n_deals=n_deals,
        avg_turns_per_dialogue=avg_turns,
        avg_closing_price_pct_of_max=(sum(pct_of_max) / len(pct_of_max)) if pct_of_max else 0.0,
        avg_buyer_surplus=(sum(surpluses) / len(surpluses)) if surpluses else 0.0,
        avg_seller_revenue=(sum(revenues) / len(revenues)) if revenues else 0.0,
        n_walked=n_walked,
    )


def run_prompt_variant_tournament(
    market: Market,
    *,
    buyer_clients: dict[str, LLMClient],
    seller_clients: dict[str, LLMClient],
    variants: tuple[str, ...] = VARIANT_NAMES,
    seller_volume_policies: Optional[dict[str, object]] = None,
    buyer_business_contexts: Optional[dict[str, str]] = None,
    max_turns_per_dialogue: int = 8,
    progress: bool = True,
) -> tuple[dict[str, PromptVariantOutcome], dict[str, ChatMarketRun]]:
    """Run the SAME market under each prompt variant. Returns per-variant outcomes
    plus the raw `ChatMarketRun` so callers can dig into specific dialogues.

    Cost: one full chat-market run per variant. If you have 4 variants and each
    run is ~$0.10, the experiment costs $0.40. Cheap.
    """
    outcomes: dict[str, PromptVariantOutcome] = {}
    runs: dict[str, ChatMarketRun] = {}

    for variant in variants:
        if progress:
            print(f"--- Running prompt variant: {variant} ---", flush=True)
        restore = _patch_dialogue_prompts(variant)
        try:
            run = run_chat_market(
                market,
                buyer_clients=buyer_clients,
                seller_clients=seller_clients,
                max_turns_per_dialogue=max_turns_per_dialogue,
                seller_volume_policies=seller_volume_policies,
                buyer_business_contexts=buyer_business_contexts,
            )
            runs[variant] = run
            outcomes[variant] = _summarize_run(variant, run)
            if progress:
                o = outcomes[variant]
                print(
                    f"   {variant}: {o.n_deals}/{o.n_dialogues} deals "
                    f"(close rate {o.close_rate*100:.0f}%), "
                    f"avg buyer surplus ${o.avg_buyer_surplus:.0f}, "
                    f"avg seller revenue ${o.avg_seller_revenue:.0f}",
                    flush=True,
                )
        finally:
            _restore_dialogue_prompts(restore)

    return outcomes, runs


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_prompt_variant_report(outcomes: dict[str, PromptVariantOutcome]) -> str:
    """HTML table comparing the four prompt variants on the same market."""
    if not outcomes:
        return '<em style="color:#9ca3af;">No outcomes to render.</em>'

    rows = []
    for v in VARIANT_NAMES:
        if v not in outcomes:
            continue
        o = outcomes[v]
        close_color = "#16a34a" if o.close_rate >= 0.7 else "#d97706" if o.close_rate >= 0.4 else "#dc2626"
        rows.append(
            f'<tr>'
            f'<td style="padding:8px 12px;font-weight:600;color:#111;">{o.variant_name}</td>'
            f'<td style="padding:8px 12px;font-family:monospace;text-align:right;">'
            f'{o.n_deals}/{o.n_dialogues}</td>'
            f'<td style="padding:8px 12px;font-family:monospace;text-align:right;color:{close_color};">'
            f'{o.close_rate*100:.0f}%</td>'
            f'<td style="padding:8px 12px;font-family:monospace;text-align:right;">'
            f'{o.avg_turns_per_dialogue:.1f}</td>'
            f'<td style="padding:8px 12px;font-family:monospace;text-align:right;">'
            f'{o.avg_closing_price_pct_of_max*100:.0f}%</td>'
            f'<td style="padding:8px 12px;font-family:monospace;text-align:right;color:#16a34a;">'
            f'${o.avg_buyer_surplus:.0f}</td>'
            f'<td style="padding:8px 12px;font-family:monospace;text-align:right;color:#0369a1;">'
            f'${o.avg_seller_revenue:.0f}</td>'
            f'</tr>'
        )

    return (
        '<div style="font-family:-apple-system,sans-serif;max-width:920px;">'
        '<h3 style="margin:0 0 6px;">Prompt-variant tournament</h3>'
        '<div style="font-size:12px;color:#6b7280;margin-bottom:10px;">'
        'Same market, same model, four prompt strategies. Both sides use the '
        'same variant per run, so a "cooperative buyer" plays a "cooperative '
        'seller" — keeps the comparison clean.'
        '</div>'
        '<table style="border-collapse:collapse;width:100%;font-size:12px;">'
        '<thead><tr style="background:#f3f4f6;font-size:11px;color:#374151;">'
        '<th style="padding:8px 12px;text-align:left;">variant</th>'
        '<th style="padding:8px 12px;text-align:right;">deals/threads</th>'
        '<th style="padding:8px 12px;text-align:right;">close rate</th>'
        '<th style="padding:8px 12px;text-align:right;">avg turns</th>'
        '<th style="padding:8px 12px;text-align:right;">price (% of buyer max)</th>'
        '<th style="padding:8px 12px;text-align:right;">avg buyer surplus</th>'
        '<th style="padding:8px 12px;text-align:right;">avg seller revenue</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody>'
        '</table>'
        '<div style="margin-top:8px;font-size:11px;color:#6b7280;">'
        'How to read: lower "price (% of max)" = better for buyers; higher = better for sellers.'
        '</div>'
        '</div>'
    )


__all__ = [
    "VARIANT_NAMES",
    "PromptVariantOutcome",
    "run_prompt_variant_tournament",
    "render_prompt_variant_report",
]
