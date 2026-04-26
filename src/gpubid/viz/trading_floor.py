"""Animated trading-floor view used by the notebook's negotiation cell.

`render_round(snapshot, market)` returns an HTML string with the round counter,
active asks (left), active bids (right), and the deals struck so far.

`animate_negotiation(market, ...)` drives the animation in a notebook by writing
into a single `ipywidgets.Output` container and stepping `time.sleep` between
rounds. No flicker, no scroll-spam.
"""

from __future__ import annotations

import time
from typing import Iterator, Optional

from gpubid.engine.board import RoundSnapshot
from gpubid.engine.round_runner import (
    make_deterministic_agents,
    make_llm_agents,
    run_rounds,
)
from gpubid.schema import Deal, GPUType, Market, Offer
from gpubid.viz.market_view import (
    FONT_STACK,
    GPU_BG,
    GPU_COLOR,
    INTERRUPTION_COLOR,
    INTERRUPTION_LABEL,
    _pill,
    render_deal_row,
)


# ---------------------------------------------------------------------------
# Per-offer rendering
# ---------------------------------------------------------------------------


def _label_for_id(market: Market, agent_id: str) -> str:
    for b in market.buyers:
        if b.id == agent_id:
            return b.label
    for s in market.sellers:
        if s.id == agent_id:
            return s.label
    return agent_id


def _ask_row(offer: Offer, market: Market, is_new: bool) -> str:
    seller_label = _label_for_id(market, offer.from_id)
    bg = GPU_BG[offer.gpu_type]
    border = GPU_COLOR[offer.gpu_type]
    new_badge = (
        '<span style="font-size:9px;color:#fff;background:#2563eb;padding:1px 5px;'
        'border-radius:3px;margin-left:4px;">new</span>'
        if is_new else ""
    )
    return (
        f'<div style="background:{bg};border-left:3px solid {border};padding:6px 9px;'
        f'border-radius:4px;font-family:{FONT_STACK};font-size:12px;'
        f'display:flex;align-items:center;gap:8px;">'
        f'<span style="font-family:monospace;color:#6b7280;font-size:10px;min-width:60px;">'
        f'{offer.from_id}/{offer.slot_id or "?"}</span>'
        f'<strong style="color:{border};">{offer.gpu_type.value}</strong>'
        f'<span>×{offer.qty}</span>'
        f'<span style="color:#6b7280;">{offer.duration}h@{offer.start:02d}</span>'
        f'<span style="margin-left:auto;font-family:monospace;font-weight:600;color:{border};">'
        f'${offer.price_per_gpu_hr:.2f}</span>'
        f'{new_badge}'
        f'<span style="font-size:10px;color:#9ca3af;">— {seller_label}</span>'
        f'</div>'
    )


def _bid_row(offer: Offer, market: Market, is_new: bool) -> str:
    buyer_label = _label_for_id(market, offer.from_id)
    border = GPU_COLOR[offer.gpu_type]
    new_badge = (
        '<span style="font-size:9px;color:#fff;background:#16a34a;padding:1px 5px;'
        'border-radius:3px;margin-left:4px;">new</span>'
        if is_new else ""
    )
    tol_color = INTERRUPTION_COLOR[offer.interruption_tolerance]
    tol_short = {"none": "strict", "checkpoint": "ckpt-OK", "interruptible": "any"}[
        offer.interruption_tolerance.value
    ]
    return (
        f'<div style="background:#f8fafc;border-left:3px solid #64748b;padding:6px 9px;'
        f'border-radius:4px;font-family:{FONT_STACK};font-size:12px;'
        f'display:flex;align-items:center;gap:8px;">'
        f'<span style="font-family:monospace;color:#6b7280;font-size:10px;min-width:60px;">'
        f'{offer.from_id}</span>'
        f'<strong style="color:{border};">{offer.gpu_type.value}</strong>'
        f'<span>×{offer.qty}</span>'
        f'<span style="color:#6b7280;">{offer.duration}h@{offer.start:02d}</span>'
        f'<span style="font-size:10px;color:{tol_color};font-weight:500;">{tol_short}</span>'
        f'<span style="margin-left:auto;font-family:monospace;font-weight:600;color:#0369a1;">'
        f'${offer.price_per_gpu_hr:.2f}</span>'
        f'{new_badge}'
        f'<span style="font-size:10px;color:#9ca3af;">— {buyer_label}</span>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Round snapshot — the cell-3 output
# ---------------------------------------------------------------------------


def render_round(snapshot: RoundSnapshot, market: Market) -> str:
    progress_pct = snapshot.round_n / max(snapshot.max_rounds, 1) * 100

    # Header: round counter + progress bar + summary stats
    header = (
        f'<div style="font-family:{FONT_STACK};margin-bottom:10px;">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;">'
        f'<strong style="font-size:14px;color:#111;">'
        f'Round {snapshot.round_n} <span style="color:#9ca3af;font-weight:400;">/ {snapshot.max_rounds}</span>'
        f'</strong>'
        f'<span style="font-size:11px;color:#6b7280;">'
        f'{len(snapshot.all_deals)} deals · '
        f'{len(snapshot.active_buyer_ids)} buyers active · '
        f'{len(snapshot.active_seller_ids)} sellers active'
        f'{" · <strong style=\"color:#16a34a\">FINAL</strong>" if snapshot.is_final else ""}'
        f'</span>'
        f'</div>'
        f'<div style="height:4px;background:#e5e7eb;border-radius:2px;overflow:hidden;margin-top:4px;">'
        f'<div style="height:100%;width:{progress_pct}%;background:#2563eb;'
        f'transition:width 200ms ease;"></div>'
        f'</div>'
        f'</div>'
    )

    # Asks column (sellers' offers on the floor)
    asks_sorted = sorted(snapshot.asks, key=lambda o: (o.price_per_gpu_hr, o.from_id))
    if asks_sorted:
        asks_html = "".join(
            _ask_row(o, market, is_new=(o.round_n == snapshot.round_n)) for o in asks_sorted
        )
    else:
        asks_html = '<div style="color:#9ca3af;font-style:italic;font-size:12px;">no active asks</div>'

    # Bids column (buyers' offers on the floor)
    bids_sorted = sorted(snapshot.bids, key=lambda o: (-o.price_per_gpu_hr, o.from_id))
    if bids_sorted:
        bids_html = "".join(
            _bid_row(o, market, is_new=(o.round_n == snapshot.round_n)) for o in bids_sorted
        )
    else:
        bids_html = '<div style="color:#9ca3af;font-style:italic;font-size:12px;">no active bids</div>'

    columns = (
        f'<div style="display:flex;gap:12px;font-family:{FONT_STACK};">'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-size:11px;font-weight:600;color:#374151;text-transform:uppercase;'
        f'letter-spacing:0.05em;margin-bottom:6px;">'
        f'Asks ({len(snapshot.asks)}) — sellers offering'
        f'</div>'
        f'<div style="display:flex;flex-direction:column;gap:4px;">{asks_html}</div>'
        f'</div>'
        f'<div style="flex:1;min-width:0;">'
        f'<div style="font-size:11px;font-weight:600;color:#374151;text-transform:uppercase;'
        f'letter-spacing:0.05em;margin-bottom:6px;">'
        f'Bids ({len(snapshot.bids)}) — buyers willing'
        f'</div>'
        f'<div style="display:flex;flex-direction:column;gap:4px;">{bids_html}</div>'
        f'</div>'
        f'</div>'
    )

    # Deals pane
    deals_label = (
        f'<div style="font-size:11px;font-weight:600;color:#374151;text-transform:uppercase;'
        f'letter-spacing:0.05em;margin:14px 0 6px;">Deals ({len(snapshot.all_deals)})</div>'
    )
    if snapshot.all_deals:
        # Newest first; mark this round's deals with a glow.
        rows: list[str] = []
        for d in reversed(snapshot.all_deals):
            base = render_deal_row(d)
            if d.round_n == snapshot.round_n:
                base = base.replace(
                    'background:#f0fdf4;',
                    'background:#dcfce7;box-shadow:0 0 0 2px #16a34a;',
                    1,
                )
            rows.append(base)
        deals_html = "".join(rows)
    else:
        deals_html = (
            '<div style="color:#9ca3af;font-style:italic;font-size:12px;">no deals struck yet</div>'
        )

    deals_pane = (
        f'{deals_label}'
        f'<div style="display:flex;flex-direction:column;gap:4px;font-family:{FONT_STACK};">'
        f'{deals_html}'
        f'</div>'
    )

    return f'<div>{header}{columns}{deals_pane}</div>'


# ---------------------------------------------------------------------------
# Animation driver — used by the notebook's negotiation cell
# ---------------------------------------------------------------------------


def animate_negotiation(
    market: Optional[Market] = None,
    mode: str = "fast",
    max_rounds: int = 5,
    step_seconds: float = 1.0,
    concentration_cap_pct: Optional[float] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    seller_api_key: Optional[str] = None,
    preset_path: Optional[str] = None,
) -> tuple[RoundSnapshot, Market]:
    """Run a negotiation and animate it in the current notebook cell.

    Three modes:
      - "fast"   — deterministic agents. No API key required. Instant.
      - "live"   — LLM agents. Requires `api_key`. Provider auto-detected by prefix.
                   Optionally pass `seller_api_key` from a different provider for the
                   heterogeneity experiment.
      - "preset" — Replay a baked LLM trace from `preset_path`. No API key required.

    Returns (final_snapshot, market) so downstream cells can compute baselines on
    exactly the market that was animated (especially in preset mode where the market
    comes from the preset, not from the caller).
    """
    from IPython.display import HTML, display
    import ipywidgets as widgets

    if mode == "preset":
        from gpubid.experiments.bake_presets import load_preset
        if preset_path is None:
            raise ValueError("mode='preset' requires preset_path")
        preset = load_preset(preset_path)
        market = preset["market"]
        snapshots = preset["snapshots"]
        snapshot_iterable = iter(snapshots)
    else:
        if market is None:
            raise ValueError(f"mode={mode!r} requires a Market")
        if mode == "fast":
            buyer_agents, seller_agents = make_deterministic_agents(market)
        elif mode == "live":
            if not api_key:
                raise ValueError(
                    "mode='live' requires api_key (Anthropic 'sk-ant-...' or OpenAI 'sk-...'). "
                    "Paste yours in the settings cell."
                )
            buyer_agents, seller_agents = make_llm_agents(
                market, api_key=api_key, model=model, seller_api_key=seller_api_key,
            )
        else:
            raise ValueError(f"mode={mode!r} not in {{'fast', 'live', 'preset'}}")
        snapshot_iterable = run_rounds(
            market,
            buyer_agents,
            seller_agents,
            max_rounds=max_rounds,
            concentration_cap_pct=concentration_cap_pct,
        )

    out = widgets.Output()
    display(out)

    final_snapshot: Optional[RoundSnapshot] = None
    for snap in snapshot_iterable:
        final_snapshot = snap
        with out:
            out.clear_output(wait=True)
            display(HTML(render_round(snap, market)))
        if not snap.is_final:
            time.sleep(step_seconds)

    assert final_snapshot is not None and market is not None
    return final_snapshot, market


def collect_snapshots(
    market: Market,
    mode: str = "fast",
    max_rounds: int = 5,
    concentration_cap_pct: Optional[float] = None,
    api_key: Optional[str] = None,
    seller_api_key: Optional[str] = None,
) -> list[RoundSnapshot]:
    """Headless variant for tests/scripts — returns all snapshots without animation."""
    if mode == "fast":
        buyer_agents, seller_agents = make_deterministic_agents(market)
    elif mode == "live":
        if not api_key:
            raise ValueError("mode='live' requires api_key")
        buyer_agents, seller_agents = make_llm_agents(
            market, api_key=api_key, seller_api_key=seller_api_key,
        )
    else:
        raise ValueError(f"mode={mode!r} not supported here")
    return list(
        run_rounds(
            market, buyer_agents, seller_agents,
            max_rounds=max_rounds,
            concentration_cap_pct=concentration_cap_pct,
        )
    )


__all__ = ["render_round", "animate_negotiation", "collect_snapshots"]
