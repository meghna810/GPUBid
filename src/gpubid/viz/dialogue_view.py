"""Render a bilateral dialogue as a chat-style HTML thread with model badges.

Buyers right (green-tinted), sellers left (blue-tinted), each bubble shows:
- Speaker label
- ``provider/model`` badge so it's obvious whether the agent is on Anthropic
  or OpenAI (and which model variant).
- Action label (COUNTER / ACCEPT / WALK_AWAY).
- Proposed price + condition (if any).
- Argument paragraph.
- A small "↪ refs alt" indicator when the agent cited an alternative.
"""

from __future__ import annotations

from gpubid.protocol.dialogue import BilateralDialogueResult, DialogueTurn
from gpubid.viz.market_view import FONT_STACK


_PROVIDER_COLOR: dict[str, str] = {
    "anthropic": "#dc7c2f",
    "openai": "#10a37f",
    "fixture": "#6b7280",
    "?": "#9ca3af",
}


def _model_badge(provider: str, model: str) -> str:
    bg = _PROVIDER_COLOR.get(provider, "#9ca3af")
    short_model = (model or "?").replace("-latest", "").replace("claude-", "c-").replace("gpt-", "")
    return (
        f'<span style="background:{bg};color:#fff;padding:1px 6px;border-radius:8px;'
        f'font-size:9px;font-weight:600;font-family:monospace;">'
        f'{provider}/{short_model}'
        f'</span>'
    )


def _action_badge(action: str) -> str:
    color, label = {
        "counter":    ("#0369a1", "COUNTER"),
        "accept":     ("#16a34a", "✓ ACCEPT"),
        "walk_away":  ("#dc2626", "✗ WALK AWAY"),
    }.get(action, ("#6b7280", action.upper()))
    return (
        f'<span style="font-size:10px;color:{color};font-weight:700;letter-spacing:0.05em;">'
        f'{label}</span>'
    )


def render_dialogue_turn(turn: DialogueTurn) -> str:
    """One turn -> one chat bubble."""
    is_buyer = turn.speaker == "buyer"
    align = "flex-end" if is_buyer else "flex-start"
    bg = "#dcfce7" if is_buyer else "#dbeafe"
    border = "#bbf7d0" if is_buyer else "#bfdbfe"
    role_emoji = "🛒" if is_buyer else "🖥️"

    price_html = ""
    if turn.proposed_price_per_gpu_hr is not None:
        cond = f' <span style="color:#6b7280;">{turn.condition}</span>' if turn.condition.strip() else ""
        price_html = (
            f'<div style="font-family:monospace;font-size:12px;color:#1f2937;'
            f'margin-top:4px;background:rgba(255,255,255,0.5);padding:3px 6px;'
            f'border-radius:4px;display:inline-block;">'
            f'${turn.proposed_price_per_gpu_hr:.2f}/GPU-hr{cond}'
            f'</div>'
        )

    refs_marker = ""
    if turn.references_alternative:
        refs_marker = (
            ' <span style="font-size:9px;color:#7c3aed;background:#f5f3ff;'
            'padding:1px 5px;border-radius:3px;margin-left:4px;">↪ refs alt</span>'
        )

    return (
        f'<div style="display:flex;justify-content:{align};margin:6px 0;">'
        f'<div style="max-width:75%;background:{bg};border:1px solid {border};'
        f'border-radius:10px;padding:10px 12px;">'
        f'<div style="display:flex;justify-content:space-between;gap:10px;align-items:baseline;'
        f'flex-wrap:wrap;">'
        f'<strong style="font-size:11px;color:#111;">'
        f'{role_emoji} {turn.speaker_label} <span style="font-family:monospace;font-weight:400;'
        f'color:#9ca3af;">{turn.speaker_id}</span>'
        f'</strong>'
        f'<span style="display:flex;gap:4px;align-items:center;">'
        f'{_model_badge(turn.speaker_provider, turn.speaker_model)}'
        f'{_action_badge(turn.action)}'
        f'</span>'
        f'</div>'
        f'<div style="font-size:10px;color:#9ca3af;margin-top:2px;">turn {turn.turn_n}{refs_marker}</div>'
        f'{price_html}'
        f'<div style="font-size:12px;color:#1f2937;margin-top:6px;line-height:1.4;font-style:italic;">'
        f'"{turn.argument}"'
        f'</div>'
        f'</div></div>'
    )


def render_dialogue(result: BilateralDialogueResult, max_height_px: int = 700) -> str:
    """Render the full bilateral dialogue thread, plus a closing summary."""
    header = (
        f'<div style="font-family:{FONT_STACK};margin-bottom:8px;">'
        f'<div style="display:flex;align-items:baseline;gap:8px;">'
        f'<strong style="font-size:13px;color:#111;">Bilateral negotiation</strong>'
        f'<span style="font-family:monospace;color:#6b7280;font-size:11px;">{result.pair_id}</span>'
        f'</div>'
        f'<div style="font-size:11px;color:#6b7280;margin-top:2px;">'
        f'{len(result.turns)} turns · '
        f'{"closed at $%.2f/GPU-hr" % result.closing_price if result.closed and result.closing_price else "no deal"}'
        f'{" · walked away by " + result.walked_away_by if result.walked_away_by else ""}'
        f'</div>'
        f'</div>'
    )

    bubbles = "".join(render_dialogue_turn(t) for t in result.turns)

    closer = ""
    if result.closed:
        closer = (
            f'<div style="text-align:center;background:#dcfce7;color:#14532d;'
            f'padding:6px 12px;border-radius:12px;display:inline-block;margin:8px auto;'
            f'font-size:11px;font-weight:600;">'
            f'✓ DEAL CLOSED at ${result.closing_price:.2f}/GPU-hr'
            f'</div>'
        )
    elif result.walked_away_by:
        closer = (
            f'<div style="text-align:center;background:#fee2e2;color:#7f1d1d;'
            f'padding:6px 12px;border-radius:12px;display:inline-block;margin:8px auto;'
            f'font-size:11px;font-weight:600;">'
            f'✗ {result.walked_away_by} walked away'
            f'</div>'
        )

    return (
        f'<div style="font-family:{FONT_STACK};max-height:{max_height_px}px;'
        f'overflow-y:auto;padding:10px 14px;background:#fafafa;border-radius:8px;'
        f'border:1px solid #e5e7eb;">'
        f'{header}{bubbles}'
        f'<div style="text-align:center;">{closer}</div>'
        f'</div>'
    )


__all__ = ["render_dialogue", "render_dialogue_turn"]
