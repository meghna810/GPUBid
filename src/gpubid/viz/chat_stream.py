"""Streaming chat-bubble renderer for the chat-based market.

Used by the notebook's negotiation cell when `run_chat_market` is the engine.
Each bilateral dialogue is rendered as a chat thread (iMessage-style); threads
appear sequentially as the LLMs converge on each deal.

The renderer is dumb (just HTML) — the live streaming is handled by the
caller via `ipywidgets.Output` plus `time.sleep` between threads.
"""

from __future__ import annotations

from gpubid.protocol.dialogue import BilateralDialogueResult, DialogueTurn
from gpubid.schema import Deal
from gpubid.viz.market_view import FONT_STACK


_PROVIDER_BG = {
    "anthropic": "#dc7c2f",
    "openai": "#10a37f",
    "gemini": "#4285f4",
    "deterministic": "#6b7280",
}


def render_chat_thread(
    dialogue: BilateralDialogueResult,
    deal: Deal | None,
    *,
    buyer_label: str = "",
    seller_label: str = "",
) -> str:
    """Render one bilateral dialogue as a chat thread + outcome banner."""
    bubbles: list[str] = []

    # Header — who's chatting
    header_left = seller_label or "Seller"
    header_right = buyer_label or "Buyer"
    bubbles.append(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:8px 12px;margin-bottom:6px;background:#f3f4f6;border-radius:8px 8px 0 0;'
        f'border-bottom:2px solid #e5e7eb;">'
        f'<strong style="font-size:12px;color:#0369a1;">🖥️ {header_left}</strong>'
        f'<span style="font-size:10px;color:#9ca3af;font-family:monospace;">{dialogue.pair_id}</span>'
        f'<strong style="font-size:12px;color:#16a34a;">🛒 {header_right}</strong>'
        f'</div>'
    )

    for turn in dialogue.turns:
        bubbles.append(_render_turn(turn))

    # Outcome banner
    if deal is not None:
        bubbles.append(
            f'<div style="text-align:center;background:#dcfce7;color:#14532d;'
            f'padding:8px 14px;border-radius:8px;margin-top:8px;font-size:12px;font-weight:600;">'
            f'✓ DEAL — {deal.gpu_type.value}×{deal.qty} '
            f'@ ${deal.price_per_gpu_hr:.2f}/hr × {deal.duration}h '
            f'(total ${deal.total_value:.0f})'
            f'</div>'
        )
    elif dialogue.walked_away_by:
        bubbles.append(
            f'<div style="text-align:center;background:#fee2e2;color:#7f1d1d;'
            f'padding:8px 14px;border-radius:8px;margin-top:8px;font-size:12px;font-weight:600;">'
            f'✗ {dialogue.walked_away_by} walked away'
            f'</div>'
        )
    else:
        bubbles.append(
            f'<div style="text-align:center;background:#fef3c7;color:#78350f;'
            f'padding:8px 14px;border-radius:8px;margin-top:8px;font-size:12px;font-weight:600;">'
            f'… no deal — turn cap reached'
            f'</div>'
        )

    return (
        f'<div style="font-family:{FONT_STACK};max-width:780px;'
        f'border:1px solid #e5e7eb;border-radius:8px;padding:0 12px 12px;'
        f'background:#fafafa;margin-bottom:14px;">'
        + ''.join(bubbles)
        + '</div>'
    )


def _render_turn(turn: DialogueTurn) -> str:
    side = "right" if turn.speaker == "buyer" else "left"
    align = "flex-end" if side == "right" else "flex-start"
    bg = "#dcfce7" if side == "right" else "#dbeafe"
    border = "#bbf7d0" if side == "right" else "#bfdbfe"

    # Provider badge
    bg_pill = _PROVIDER_BG.get(turn.speaker_provider, "#6b7280")
    short_model = (turn.speaker_model or "?").replace("-latest", "").replace("claude-", "c-").replace("gpt-", "")
    model_pill = (
        f'<span style="background:{bg_pill};color:#fff;padding:1px 6px;border-radius:6px;'
        f'font-size:9px;font-weight:600;font-family:monospace;margin-left:6px;">'
        f'{turn.speaker_provider}/{short_model}</span>'
    )

    # Action label + price
    action_color = {
        "counter": "#0369a1",
        "accept": "#16a34a",
        "walk_away": "#dc2626",
    }.get(turn.action, "#6b7280")
    if turn.action == "counter" and turn.proposed_price_per_gpu_hr is not None:
        action_str = f'COUNTER @ ${turn.proposed_price_per_gpu_hr:.2f}/hr'
    else:
        action_str = turn.action.upper().replace("_", " ")

    cond_html = ""
    if turn.condition and turn.condition.strip().lower() not in ("flat", "flat — no condition", "none", ""):
        cond_html = (
            f'<div style="font-size:11px;color:#7c2d12;margin-top:4px;font-style:italic;">'
            f'condition: {_escape(turn.condition)}'
            f'</div>'
        )

    refs_pill = ""
    if turn.references_alternative:
        refs_pill = (
            '<span style="background:#fde68a;color:#78350f;padding:1px 5px;'
            'border-radius:5px;font-size:9px;margin-left:4px;">refs alt</span>'
        )

    role_emoji = "🛒" if turn.speaker == "buyer" else "🖥️"
    speaker_name = turn.speaker_label or turn.speaker_id

    arg_html = ""
    if turn.argument and turn.argument.strip():
        arg_html = (
            f'<div style="font-size:12px;color:#1f2937;margin-top:6px;line-height:1.45;">'
            f'{_escape(turn.argument)}</div>'
        )

    return (
        f'<div style="display:flex;justify-content:{align};margin:6px 0;">'
        f'<div style="max-width:78%;background:{bg};border:1px solid {border};'
        f'border-radius:12px;padding:8px 12px;">'
        f'<div style="display:flex;justify-content:space-between;gap:10px;align-items:baseline;">'
        f'<strong style="font-size:11px;color:#111;">'
        f'turn {turn.turn_n} · {role_emoji} {_escape(speaker_name)}{model_pill}{refs_pill}</strong>'
        f'<span style="font-size:10px;color:{action_color};font-weight:600;font-family:monospace;">{action_str}</span>'
        f'</div>'
        f'{arg_html}'
        f'{cond_html}'
        f'</div></div>'
    )


def render_chat_market_summary(deals: list[Deal], n_dialogues: int, n_walked: int) -> str:
    """Headline summary box at the top/bottom of the streaming view."""
    n_deals = len(deals)
    total = sum(d.total_value for d in deals)
    return (
        f'<div style="font-family:{FONT_STACK};background:#fff;border:1px solid #e5e7eb;'
        f'border-radius:8px;padding:12px 16px;margin-top:8px;display:flex;gap:14px;">'
        f'<div><div style="font-size:10px;color:#6b7280;text-transform:uppercase;">Threads run</div>'
        f'<div style="font-size:22px;font-weight:600;color:#111;">{n_dialogues}</div></div>'
        f'<div><div style="font-size:10px;color:#6b7280;text-transform:uppercase;">Deals closed</div>'
        f'<div style="font-size:22px;font-weight:600;color:#16a34a;">{n_deals}</div></div>'
        f'<div><div style="font-size:10px;color:#6b7280;text-transform:uppercase;">Walked away</div>'
        f'<div style="font-size:22px;font-weight:600;color:#dc2626;">{n_walked}</div></div>'
        f'<div><div style="font-size:10px;color:#6b7280;text-transform:uppercase;">Total transacted</div>'
        f'<div style="font-size:22px;font-weight:600;color:#111;">${total:.0f}</div></div>'
        f'</div>'
    )


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("\n", "<br>"))


def render_seller_menu(buyer, candidate_slots, sellers) -> str:
    """Render the "shop window" of seller-slot options the buyer is choosing from.

    The first slot in `candidate_slots` is the buyer's first pick (cheapest by
    reserve); the rest are fall-back options if their first chat walks away.

    Highlights this thread's "winner" choice (slot 0) with a green pill so the
    audience can see exactly what the buyer is engaging with first.
    """
    seller_lookup = {sl.id: s for s in sellers for sl in s.capacity_slots}

    rows = []
    for i, sl in enumerate(candidate_slots[:6]):  # cap at 6 to keep view tidy
        seller = seller_lookup.get(sl.id)
        seller_label = seller.label if seller else "?"
        is_first = (i == 0)
        bg = "#dcfce7" if is_first else "#fff"
        border = "#16a34a" if is_first else "#e5e7eb"
        badge = (
            '<span style="background:#16a34a;color:#fff;padding:1px 7px;border-radius:8px;'
            'font-size:9px;font-weight:600;margin-left:6px;">first pick</span>'
        ) if is_first else ""
        rows.append(
            f'<div style="background:{bg};border:1px solid {border};border-radius:6px;'
            f'padding:8px 12px;margin-bottom:4px;font-size:12px;">'
            f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
            f'<strong style="color:#111;">#{i+1} · {seller_label} · slot {sl.id}{badge}</strong>'
            f'<span style="font-family:monospace;color:#374151;">'
            f'{sl.gpu_type.value}×{sl.qty} · hr{sl.start:02d}+{sl.duration}h · '
            f'reserve ${sl.reserve_per_gpu_hr:.2f}/hr'
            f'</span>'
            f'</div>'
            f'</div>'
        )

    n_more = max(0, len(candidate_slots) - 6)
    more_html = (
        f'<div style="font-size:10px;color:#9ca3af;margin-top:4px;">'
        f'…+{n_more} more compatible options not shown</div>'
        if n_more > 0 else ''
    )

    return (
        f'<div style="font-family:{FONT_STACK};max-width:780px;'
        f'border:1px dashed #6366f1;border-radius:8px;padding:10px 14px;'
        f'background:#eef2ff;margin-bottom:8px;">'
        f'<div style="font-size:12px;font-weight:600;color:#4338ca;margin-bottom:6px;">'
        f'🛍 {buyer.id} ({buyer.label}) is choosing from {len(candidate_slots)} compatible seller slot(s)'
        f'</div>'
        f'<div style="font-size:11px;color:#6b7280;margin-bottom:8px;line-height:1.4;">'
        f'Buyer needs {buyer.job.qty}× {",".join(g.value for g in buyer.job.acceptable_gpus)} '
        f'for {buyer.job.duration}h between hr {buyer.job.earliest_start}-{buyer.job.latest_finish}. '
        f'They engage the cheapest slot first (green); on walk-away they try the next.'
        f'</div>'
        f'{"".join(rows)}'
        f'{more_html}'
        f'</div>'
    )


__all__ = [
    "render_chat_thread",
    "render_chat_market_summary",
    "render_seller_menu",
]
