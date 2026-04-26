"""Render Market, Buyer, Seller, CapacitySlot, Deal as HTML strings.

Inline styles only — no external CSS — so the same string works in Jupyter,
Colab, Gradio, or a Next.js page. Every public function returns an HTML
fragment safe to drop into `display(HTML(...))` or `gr.HTML(...)`.
"""

from __future__ import annotations

from gpubid.schema import (
    Buyer,
    CapacitySlot,
    Deal,
    GPUType,
    InterruptionTolerance,
    Market,
    Seller,
)


# ---------------------------------------------------------------------------
# Color palette (inline so HTML is self-contained)
# ---------------------------------------------------------------------------

GPU_COLOR: dict[GPUType, str] = {
    GPUType.H100: "#dc2626",   # red-600   — top-tier
    GPUType.A100: "#2563eb",   # blue-600  — mid
    GPUType.L40S: "#16a34a",   # green-600 — entry
}

GPU_BG: dict[GPUType, str] = {
    GPUType.H100: "#fef2f2",   # red-50
    GPUType.A100: "#eff6ff",   # blue-50
    GPUType.L40S: "#f0fdf4",   # green-50
}

INTERRUPTION_COLOR: dict[InterruptionTolerance, str] = {
    InterruptionTolerance.NONE: "#dc2626",
    InterruptionTolerance.CHECKPOINT: "#d97706",
    InterruptionTolerance.INTERRUPTIBLE: "#16a34a",
}

INTERRUPTION_LABEL: dict[InterruptionTolerance, str] = {
    InterruptionTolerance.NONE: "no interrupt",
    InterruptionTolerance.CHECKPOINT: "checkpoint OK",
    InterruptionTolerance.INTERRUPTIBLE: "interruptible",
}

FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
    "'Helvetica Neue', Arial, sans-serif"
)


# ---------------------------------------------------------------------------
# Pills and small components
# ---------------------------------------------------------------------------


def _pill(text: str, bg: str = "#f3f4f6", fg: str = "#374151") -> str:
    return (
        f'<span style="background:{bg};color:{fg};padding:1px 7px;border-radius:10px;'
        f'font-size:11px;font-weight:500;white-space:nowrap;">{text}</span>'
    )


def _gpu_pill(gpu: GPUType, qty: int | None = None) -> str:
    text = f"{qty}× {gpu.value}" if qty else gpu.value
    return _pill(text, bg=GPU_COLOR[gpu], fg="#fff")


def _interruption_pill(it: InterruptionTolerance) -> str:
    return _pill(INTERRUPTION_LABEL[it], bg=INTERRUPTION_COLOR[it], fg="#fff")


def _urgency_color(urgency: float) -> str:
    if urgency < 0.33:
        return "#22c55e"  # green
    if urgency < 0.66:
        return "#f59e0b"  # amber
    return "#dc2626"      # red


def _urgency_label(urgency: float) -> str:
    if urgency < 0.33:
        return "patient"
    if urgency < 0.66:
        return "moderate"
    return "urgent"


# ---------------------------------------------------------------------------
# Buyer card
# ---------------------------------------------------------------------------


def render_buyer_card(b: Buyer) -> str:
    border = _urgency_color(b.urgency)
    job = b.job

    gpu_pills = "".join(_gpu_pill(g) for g in job.acceptable_gpus)
    pills = (
        gpu_pills
        + _pill(f"{job.qty} GPUs")
        + _pill(f"{job.duration}h")
        + _pill(f"slot {job.earliest_start:02d}–{job.latest_finish:02d}")
        + _interruption_pill(job.interruption_tolerance)
    )

    return (
        f'<div style="background:#fff;border-left:4px solid {border};border:1px solid #e5e7eb;'
        f'border-left-width:4px;border-radius:6px;padding:10px 12px;margin:0;'
        f'font-family:{FONT_STACK};box-shadow:0 1px 2px rgba(0,0,0,0.04);">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
        f'<strong style="font-size:13px;color:#111;">{b.label}</strong>'
        f'<span style="font-size:10px;color:#9ca3af;font-family:monospace;">{b.id}</span>'
        f'</div>'
        f'<div style="display:flex;gap:4px;margin-top:6px;flex-wrap:wrap;">{pills}</div>'
        f'<div style="margin-top:8px;display:flex;align-items:center;gap:6px;">'
        f'<div style="flex:1;height:4px;background:#e5e7eb;border-radius:2px;overflow:hidden;">'
        f'<div style="height:100%;width:{b.urgency*100:.0f}%;background:{border};"></div>'
        f'</div>'
        f'<span style="font-size:10px;color:#6b7280;">{_urgency_label(b.urgency)} ({b.urgency:.2f})</span>'
        f'</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Seller card
# ---------------------------------------------------------------------------


def _slot_chip(slot: CapacitySlot) -> str:
    bg = GPU_BG[slot.gpu_type]
    border = GPU_COLOR[slot.gpu_type]
    offpeak_badge = (
        '<span style="font-size:9px;color:#6366f1;margin-left:4px;">●off-peak</span>'
        if slot.is_offpeak
        else ""
    )
    return (
        f'<div style="background:{bg};border-left:3px solid {border};padding:6px 8px;'
        f'border-radius:4px;margin:0;font-size:11px;display:flex;'
        f'justify-content:space-between;align-items:center;gap:8px;">'
        f'<span><strong style="color:{border};">{slot.gpu_type.value}</strong> '
        f'×{slot.qty} · {slot.duration}h @ slot {slot.start:02d}{offpeak_badge}</span>'
        f'<span style="font-family:monospace;color:#6b7280;font-size:10px;" '
        f'title="private reserve, not shown to other agents">'
        f'≥${slot.reserve_per_gpu_hr:.2f}</span>'
        f'</div>'
    )


def render_seller_card(s: Seller) -> str:
    total_qty = sum(slot.qty for slot in s.capacity_slots)
    total_gpu_hours = sum(slot.qty * slot.duration for slot in s.capacity_slots)

    slot_html = "".join(_slot_chip(slot) for slot in s.capacity_slots)
    return (
        f'<div style="background:#fff;border-left:4px solid #1d4ed8;border:1px solid #e5e7eb;'
        f'border-left-width:4px;border-radius:6px;padding:10px 12px;margin:0;'
        f'font-family:{FONT_STACK};box-shadow:0 1px 2px rgba(0,0,0,0.04);">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
        f'<strong style="font-size:13px;color:#111;">{s.label}</strong>'
        f'<span style="font-size:10px;color:#9ca3af;font-family:monospace;">{s.id}</span>'
        f'</div>'
        f'<div style="font-size:10px;color:#6b7280;margin-top:2px;">'
        f'{len(s.capacity_slots)} slots · {total_qty} GPUs · {total_gpu_hours} GPU-hours'
        f'</div>'
        f'<div style="margin-top:8px;display:flex;flex-direction:column;gap:4px;">'
        f'{slot_html}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Market — the killer cell-2 output
# ---------------------------------------------------------------------------


def render_market(m: Market) -> str:
    regime_color = "#dc2626" if m.regime == "tight" else "#16a34a"
    ratio = m.supply_demand_ratio

    header = (
        f'<div style="font-family:{FONT_STACK};margin-bottom:12px;">'
        f'<div style="display:flex;align-items:center;gap:8px;flex-wrap:wrap;">'
        f'<strong style="font-size:14px;color:#111;">Market {m.id}</strong>'
        f'{_pill(m.regime + " supply", bg=regime_color, fg="#fff")}'
        f'{_pill(f"seed {m.seed}")}'
        f'{_pill(f"{len(m.buyers)} buyers · {len(m.sellers)} sellers")}'
        f'{_pill(f"S/D = {ratio:.2f}", bg="#eef2ff", fg="#4338ca")}'
        f'</div>'
        f'<div style="font-size:11px;color:#6b7280;margin-top:4px;">'
        f'Total demand: {m.total_demand_gpu_hours} GPU-hours · '
        f'Total supply: {m.total_supply_gpu_hours} GPU-hours'
        f'</div>'
        f'</div>'
    )

    buyer_cards = "".join(render_buyer_card(b) for b in m.buyers)
    seller_cards = "".join(render_seller_card(s) for s in m.sellers)

    columns = (
        f'<div style="display:flex;gap:16px;font-family:{FONT_STACK};align-items:flex-start;">'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-size:12px;font-weight:600;color:#374151;'
        f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">'
        f'Buyers ({len(m.buyers)})</div>'
        f'<div style="display:flex;flex-direction:column;gap:6px;">{buyer_cards}</div>'
        f'</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-size:12px;font-weight:600;color:#374151;'
        f'text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">'
        f'Sellers ({len(m.sellers)})</div>'
        f'<div style="display:flex;flex-direction:column;gap:6px;">{seller_cards}</div>'
        f'</div>'
        f'</div>'
    )

    return f'<div>{header}{columns}</div>'


# ---------------------------------------------------------------------------
# Deal row — compact, used in the deals pane during animation
# ---------------------------------------------------------------------------


def render_deal_row(d: Deal) -> str:
    return (
        f'<div style="display:flex;align-items:center;gap:8px;padding:6px 10px;'
        f'background:#f0fdf4;border-left:3px solid #16a34a;border-radius:4px;'
        f'font-family:{FONT_STACK};font-size:12px;">'
        f'<span style="font-family:monospace;color:#6b7280;font-size:10px;">'
        f'r{d.round_n}</span>'
        f'<span><strong>{d.buyer_id}</strong> ↔ <strong>{d.seller_id}</strong></span>'
        f'{_gpu_pill(d.gpu_type, d.qty)}'
        f'<span style="color:#374151;">{d.duration}h @ slot {d.start:02d}</span>'
        f'<span style="margin-left:auto;font-family:monospace;color:#16a34a;font-weight:600;">'
        f'${d.price_per_gpu_hr:.2f}/GPU-hr · ${d.total_value:.0f} total'
        f'</span>'
        f'</div>'
    )


__all__ = [
    "render_market",
    "render_buyer_card",
    "render_seller_card",
    "render_deal_row",
    "GPU_COLOR",
    "INTERRUPTION_COLOR",
]
