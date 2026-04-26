"""Render a deal's history as a chat-bubble HTML view for the inspect cell.

Given the buyer, seller, slot, and the deal itself, render a side-by-side
narrative: buyer's job + private value, seller's slot + private reserve, the
final deal terms, and the surplus split.

If `reasoning` text is available on the offers that led to the deal (only true
in live LLM mode or preset playback), it can be shown as actual chat bubbles.
For deterministic mode there's no rich reasoning, so we render the structured
summary only.
"""

from __future__ import annotations

from gpubid.schema import Buyer, CapacitySlot, Deal, Market, Offer, Seller
from gpubid.viz.market_view import (
    FONT_STACK,
    GPU_COLOR,
    INTERRUPTION_COLOR,
    INTERRUPTION_LABEL,
    _pill,
    render_buyer_card,
    render_seller_card,
)


def render_trace(deal: Deal, market: Market, all_offers: list[Offer] | None = None) -> str:
    """Render the inspect view for one deal."""
    buyer = next(b for b in market.buyers if b.id == deal.buyer_id)
    seller = next(s for s in market.sellers if s.id == deal.seller_id)
    slot = next(sl for sl in seller.capacity_slots if sl.id == deal.slot_id)

    header = _render_deal_header(deal, buyer, slot)
    surplus = _render_surplus_split(deal, buyer, slot)
    side_by_side = _render_buyer_seller_side_by_side(buyer, seller)
    chat = _render_negotiation_chat(deal, all_offers or [])

    return (
        f'<div style="font-family:{FONT_STACK};max-width:1100px;">'
        f"{header}{surplus}{chat}{side_by_side}"
        f"</div>"
    )


def _render_deal_header(deal: Deal, buyer: Buyer, slot: CapacitySlot) -> str:
    color = GPU_COLOR[deal.gpu_type]
    return (
        f'<div style="background:#f0fdf4;border-left:4px solid #16a34a;padding:10px 14px;'
        f'border-radius:6px;margin-bottom:12px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
        f'<strong style="font-size:14px;color:#111;">'
        f'Deal {deal.id} · round {deal.round_n}'
        f'</strong>'
        f'<span style="font-family:monospace;color:#16a34a;font-weight:600;">'
        f'${deal.price_per_gpu_hr:.2f}/GPU-hr × {deal.qty} × {deal.duration}h = '
        f'${deal.total_value:.0f} total'
        f'</span>'
        f'</div>'
        f'<div style="margin-top:6px;display:flex;gap:6px;flex-wrap:wrap;">'
        f'{_pill(f"{deal.gpu_type.value} ×{deal.qty}", bg=color, fg="#fff")}'
        f'{_pill(f"slot {deal.start:02d}+{deal.duration}h")}'
        f'{_pill(INTERRUPTION_LABEL[deal.interruption_tolerance], bg=INTERRUPTION_COLOR[deal.interruption_tolerance], fg="#fff")}'
        f'</div>'
        f'</div>'
    )


def _render_surplus_split(deal: Deal, buyer: Buyer, slot: CapacitySlot) -> str:
    """Bar showing how the surplus is split between buyer and seller."""
    buyer_paid = deal.price_per_gpu_hr
    buyer_value = buyer.job.max_value_per_gpu_hr
    seller_reserve = slot.reserve_per_gpu_hr

    # Per-GPU-hour amounts (each visualized on the same scale)
    seller_revenue_per = max(buyer_paid - seller_reserve, 0)
    buyer_savings_per = max(buyer_value - buyer_paid, 0)

    total = buyer_value - seller_reserve
    if total <= 0:
        return ""

    seller_pct = seller_revenue_per / total * 100
    buyer_pct = buyer_savings_per / total * 100

    return (
        f'<div style="margin-bottom:14px;">'
        f'<div style="font-size:11px;color:#374151;font-weight:600;text-transform:uppercase;'
        f'letter-spacing:0.05em;margin-bottom:4px;">Surplus split per GPU-hour</div>'
        f'<div style="display:flex;height:24px;border-radius:4px;overflow:hidden;'
        f'border:1px solid #e5e7eb;">'
        f'<div style="background:#bfdbfe;width:{seller_pct:.0f}%;display:flex;align-items:center;'
        f'justify-content:center;color:#1e3a8a;font-size:11px;font-weight:600;">'
        f'seller +${seller_revenue_per:.2f}'
        f'</div>'
        f'<div style="background:#bbf7d0;width:{buyer_pct:.0f}%;display:flex;align-items:center;'
        f'justify-content:center;color:#14532d;font-size:11px;font-weight:600;">'
        f'buyer +${buyer_savings_per:.2f}'
        f'</div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:10px;color:#6b7280;margin-top:2px;">'
        f'<span>Reserve ${seller_reserve:.2f} (private)</span>'
        f'<span>Paid ${buyer_paid:.2f}</span>'
        f'<span>Buyer max ${buyer_value:.2f} (private)</span>'
        f'</div>'
        f'</div>'
    )


def _render_buyer_seller_side_by_side(buyer: Buyer, seller: Seller) -> str:
    return (
        f'<div style="display:flex;gap:12px;">'
        f'<div style="flex:1;">'
        f'<div style="font-size:11px;font-weight:600;color:#374151;'
        f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">Buyer</div>'
        f'{render_buyer_card(buyer)}'
        f'</div>'
        f'<div style="flex:1;">'
        f'<div style="font-size:11px;font-weight:600;color:#374151;'
        f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">Seller</div>'
        f'{render_seller_card(seller)}'
        f'</div>'
        f'</div>'
    )


def _render_negotiation_chat(deal: Deal, all_offers: list[Offer]) -> str:
    """Render reasoning-bearing offers as chat bubbles, if any are available."""
    related = [
        o for o in all_offers
        if (o.slot_id == deal.slot_id and o.from_id == deal.seller_id)
        or (o.from_id == deal.buyer_id)
    ]
    related = sorted(related, key=lambda o: (o.round_n, o.kind.value))
    if not related:
        return ""

    bubbles: list[str] = []
    for o in related:
        is_buyer = o.from_id == deal.buyer_id
        side = "left" if not is_buyer else "right"
        bg = "#eff6ff" if is_buyer else "#f1f5f9"
        align = "flex-end" if is_buyer else "flex-start"
        label = "Buyer " + o.from_id if is_buyer else "Seller " + (o.slot_id or o.from_id)
        text_inside = (
            f'<div style="font-size:10px;color:#6b7280;">round {o.round_n} · {o.kind.value}</div>'
            f'<div style="font-size:12px;color:#111;margin-top:2px;">'
            f'{o.gpu_type.value} ×{o.qty} @ ${o.price_per_gpu_hr:.2f}/hr · slot {o.start:02d}+{o.duration}h'
            f'</div>'
        )
        if o.reasoning:
            text_inside += (
                f'<div style="font-size:11px;color:#374151;margin-top:4px;font-style:italic;">'
                f'{o.reasoning}'
                f'</div>'
            )

        bubbles.append(
            f'<div style="display:flex;justify-content:{align};margin:4px 0;">'
            f'<div style="background:{bg};padding:8px 10px;border-radius:8px;max-width:70%;">'
            f'<div style="font-size:10px;color:#6b7280;font-weight:600;">{label}</div>'
            f'{text_inside}'
            f'</div></div>'
        )

    return (
        f'<div style="margin-bottom:14px;">'
        f'<div style="font-size:11px;font-weight:600;color:#374151;text-transform:uppercase;'
        f'letter-spacing:0.05em;margin-bottom:6px;">Negotiation</div>'
        + "".join(bubbles)
        + "</div>"
    )


__all__ = ["render_trace"]
