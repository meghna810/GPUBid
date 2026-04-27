"""Negotiation forensics: per-agent timelines, round-by-round logs, aggression scores.

After a run, ask the natural questions:
  - WHO bid what each round? Who held out?
  - Who conceded most aggressively?
  - Why did some buyers go unmatched — too high a reserve, or just no compatible slot?

`extract_history(snapshots)` walks the action records captured by the round runner
and returns a structured `NegotiationHistory` with per-agent price trajectories,
acceptances, and aggression metrics. `render_timeline(...)` produces a Plotly chart;
`render_log(...)` produces a per-round HTML log of every action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

from gpubid.engine.board import RoundSnapshot
from gpubid.schema import Deal, Market, Offer, OfferKind
from gpubid.viz.market_view import FONT_STACK, GPU_COLOR


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class AgentTimeline:
    """One agent's price trajectory across rounds."""

    agent_id: str
    label: str
    role: str                            # "buyer" or "seller"
    slot_id: Optional[str] = None        # for sellers, which slot this trajectory is for
    gpu_type: Optional[str] = None
    prices_by_round: dict[int, float] = field(default_factory=dict)
    accepts_at_round: dict[int, str] = field(default_factory=dict)  # round_n -> target_offer_id
    won_deal: Optional[Deal] = None
    initial_price: Optional[float] = None
    final_price: Optional[float] = None
    private_value_or_reserve: Optional[float] = None  # for the timeline tooltip

    @property
    def aggression(self) -> float:
        """Concession toward the middle, normalized.

        Buyer aggression  = (last_bid − first_bid) / first_bid × 100  (climbing toward seller).
        Seller aggression = (first_ask − last_ask) / first_ask × 100  (decaying toward buyer).
        Always positive when the agent conceded; 0 when they held firm; negative if they
        moved *away* from a deal (rare but possible in LLM mode).
        """
        if self.initial_price in (None, 0) or self.final_price is None:
            return 0.0
        if self.role == "buyer":
            return (self.final_price - self.initial_price) / self.initial_price * 100
        return (self.initial_price - self.final_price) / self.initial_price * 100


@dataclass
class NegotiationHistory:
    """Forensic record of a complete negotiation."""

    market: Market
    snapshots: list[RoundSnapshot]
    buyer_timelines: dict[str, AgentTimeline] = field(default_factory=dict)
    seller_timelines: dict[str, AgentTimeline] = field(default_factory=dict)  # keyed by slot_id

    @property
    def deals(self) -> list[Deal]:
        return list(self.snapshots[-1].all_deals) if self.snapshots else []


# ---------------------------------------------------------------------------
# Extract history from snapshots
# ---------------------------------------------------------------------------


def extract_history(market: Market, snapshots: list[RoundSnapshot]) -> NegotiationHistory:
    """Walk action records and build per-agent timelines."""
    history = NegotiationHistory(market=market, snapshots=list(snapshots))

    # Seed buyer timelines (one per buyer)
    for b in market.buyers:
        history.buyer_timelines[b.id] = AgentTimeline(
            agent_id=b.id,
            label=b.label,
            role="buyer",
            gpu_type=None,
            private_value_or_reserve=b.job.max_value_per_gpu_hr,
        )

    # Seed seller timelines (one per slot)
    for s in market.sellers:
        for sl in s.capacity_slots:
            history.seller_timelines[sl.id] = AgentTimeline(
                agent_id=s.id,
                label=f"{s.label} · {sl.id}",
                role="seller",
                slot_id=sl.id,
                gpu_type=sl.gpu_type.value,
                private_value_or_reserve=sl.reserve_per_gpu_hr,
            )

    # Walk each round's action records
    for snap in snapshots:
        for action in snap.actions:
            for offer in action.new_offers:
                if offer.kind == OfferKind.BID and offer.from_id in history.buyer_timelines:
                    tl = history.buyer_timelines[offer.from_id]
                    tl.prices_by_round[snap.round_n] = offer.price_per_gpu_hr
                    if tl.initial_price is None:
                        tl.initial_price = offer.price_per_gpu_hr
                        tl.gpu_type = offer.gpu_type.value
                    tl.final_price = offer.price_per_gpu_hr
                elif offer.kind == OfferKind.ASK and offer.slot_id in history.seller_timelines:
                    tl = history.seller_timelines[offer.slot_id]
                    tl.prices_by_round[snap.round_n] = offer.price_per_gpu_hr
                    if tl.initial_price is None:
                        tl.initial_price = offer.price_per_gpu_hr
                    tl.final_price = offer.price_per_gpu_hr
            for accept_id in action.accept_offer_ids:
                # Buyer accepting an ASK or seller accepting a BID — record on the actor's timeline.
                if action.agent_id in history.buyer_timelines:
                    history.buyer_timelines[action.agent_id].accepts_at_round[snap.round_n] = accept_id
                # Sellers do not have a single timeline; we just record the accept on each of their slots.
                else:
                    seller_slot_ids = [sid for sid, tl in history.seller_timelines.items() if tl.agent_id == action.agent_id]
                    for sid in seller_slot_ids:
                        history.seller_timelines[sid].accepts_at_round[snap.round_n] = accept_id

    # Wire up which timeline won which deal (for chart annotations and surplus calc).
    # NOTE: we do NOT overwrite `final_price` with the deal price — `final_price`
    # tracks the last posted offer, which is what "aggression" measures. The deal
    # price lives on `won_deal.price_per_gpu_hr` for surplus reporting.
    for d in history.deals:
        if d.buyer_id in history.buyer_timelines:
            history.buyer_timelines[d.buyer_id].won_deal = d
        if d.slot_id in history.seller_timelines:
            history.seller_timelines[d.slot_id].won_deal = d

    return history


# ---------------------------------------------------------------------------
# Plotly timeline
# ---------------------------------------------------------------------------


def render_timeline(history: NegotiationHistory) -> "go.Figure":
    """Plotly line chart: each agent's price over rounds, with deals marked as stars."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # Buyer trajectories — green if they won a deal, gray otherwise.
    for tl in history.buyer_timelines.values():
        if not tl.prices_by_round:
            continue
        rounds = sorted(tl.prices_by_round.keys())
        prices = [tl.prices_by_round[r] for r in rounds]
        won = tl.won_deal is not None
        color = "#16a34a" if won else "#9ca3af"
        if won:
            rounds = rounds + [tl.won_deal.round_n]
            prices = prices + [tl.won_deal.price_per_gpu_hr]
        fig.add_trace(go.Scatter(
            x=rounds, y=prices, mode="lines+markers",
            line=dict(color=color, width=2 if won else 1, dash="solid" if won else "dot"),
            marker=dict(size=8 if won else 6, symbol="circle"),
            name=f"{tl.label} (buyer)",
            legendgroup="buyers",
            hovertemplate=(f"<b>{tl.label}</b> (buyer)<br>round %{{x}}<br>bid $%{{y:.2f}}/hr<br>"
                           f"max value: ${tl.private_value_or_reserve:.2f}<extra></extra>"),
        ))
        if won:
            fig.add_trace(go.Scatter(
                x=[tl.won_deal.round_n], y=[tl.won_deal.price_per_gpu_hr],
                mode="markers", marker=dict(size=14, symbol="star", color="#16a34a", line=dict(color="#0b3d20", width=1)),
                name=f"{tl.label} won", legendgroup="buyers", showlegend=False,
                hovertemplate=f"<b>DEAL</b><br>{tl.label}<br>$%{{y:.2f}}/hr<extra></extra>",
            ))

    # Seller trajectories — colored by GPU type.
    for tl in history.seller_timelines.values():
        if not tl.prices_by_round:
            continue
        rounds = sorted(tl.prices_by_round.keys())
        prices = [tl.prices_by_round[r] for r in rounds]
        gpu = tl.gpu_type or "?"
        color = GPU_COLOR.get(__import__("gpubid.schema", fromlist=["GPUType"]).GPUType(gpu), "#6b7280") if gpu in {"H100", "A100", "L40S"} else "#6b7280"
        won = tl.won_deal is not None
        fig.add_trace(go.Scatter(
            x=rounds, y=prices, mode="lines+markers",
            line=dict(color=color, width=2 if won else 1, dash="solid" if won else "dash"),
            marker=dict(size=8 if won else 6, symbol="square"),
            name=f"{tl.label} (seller)",
            legendgroup="sellers",
            hovertemplate=(f"<b>{tl.label}</b> (seller)<br>round %{{x}}<br>ask $%{{y:.2f}}/hr<br>"
                           f"reserve: ${tl.private_value_or_reserve:.2f}<extra></extra>"),
        ))

    fig.update_layout(
        title="Offer trajectories — solid = won a deal, dashed = held out",
        xaxis_title="Round",
        yaxis_title="Price ($/GPU-hour)",
        plot_bgcolor="white",
        height=440,
        hovermode="closest",
        legend=dict(font=dict(size=10), groupclick="toggleitem"),
        margin=dict(l=50, r=20, t=50, b=40),
        font=dict(family="-apple-system, sans-serif"),
    )
    fig.update_xaxes(gridcolor="#e5e7eb", dtick=1)
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    return fig


# ---------------------------------------------------------------------------
# Round-by-round HTML log
# ---------------------------------------------------------------------------


def render_log(history: NegotiationHistory) -> str:
    """Human-readable round-by-round log of every action with arrow indicators."""

    def _ago(r: AgentTimeline, current: int) -> str:
        prev = max((rr for rr in r.prices_by_round if rr < current), default=None)
        if prev is None:
            return ""
        delta = r.prices_by_round[current] - r.prices_by_round[prev]
        if delta > 0.005:
            return f' <span style="color:#dc2626;">↑ +${delta:.2f}</span>'
        if delta < -0.005:
            return f' <span style="color:#16a34a;">↓ ${delta:.2f}</span>'
        return ' <span style="color:#9ca3af;">·</span>'

    rounds_html: list[str] = []
    for snap in history.snapshots:
        round_lines: list[str] = []
        for action in snap.actions:
            for offer in action.new_offers:
                if offer.kind == OfferKind.BID:
                    tl = history.buyer_timelines.get(offer.from_id)
                    label = tl.label if tl else offer.from_id
                    delta_html = _ago(tl, snap.round_n) if tl else ""
                    round_lines.append(
                        f'<div style="display:flex;gap:6px;font-size:12px;">'
                        f'<span style="font-family:monospace;color:#9ca3af;min-width:36px;">{offer.from_id}</span>'
                        f'<strong>{label}</strong>'
                        f'<span style="color:#0369a1;">bids</span>'
                        f'<span style="font-family:monospace;">${offer.price_per_gpu_hr:.2f}</span>'
                        f'<span style="color:#9ca3af;">for {offer.gpu_type.value}×{offer.qty}</span>'
                        f'{delta_html}'
                        f'</div>'
                    )
                elif offer.kind == OfferKind.ASK:
                    tl = history.seller_timelines.get(offer.slot_id)
                    label = tl.label if tl else f"{offer.from_id}/{offer.slot_id}"
                    delta_html = _ago(tl, snap.round_n) if tl else ""
                    round_lines.append(
                        f'<div style="display:flex;gap:6px;font-size:12px;">'
                        f'<span style="font-family:monospace;color:#9ca3af;min-width:36px;">{offer.slot_id}</span>'
                        f'<strong>{label}</strong>'
                        f'<span style="color:#dc2626;">asks</span>'
                        f'<span style="font-family:monospace;">${offer.price_per_gpu_hr:.2f}</span>'
                        f'<span style="color:#9ca3af;">for {offer.gpu_type.value}×{offer.qty}</span>'
                        f'{delta_html}'
                        f'</div>'
                    )
            for accept_id in action.accept_offer_ids:
                round_lines.append(
                    f'<div style="display:flex;gap:6px;font-size:12px;background:#fef3c7;'
                    f'padding:2px 6px;border-radius:3px;">'
                    f'<span style="font-family:monospace;color:#92400e;min-width:36px;">{action.agent_id}</span>'
                    f'<strong>accepts</strong>'
                    f'<span style="font-family:monospace;color:#92400e;">{accept_id}</span>'
                    f'</div>'
                )
        # New deals struck this round
        for d in snap.new_deals:
            round_lines.append(
                f'<div style="display:flex;gap:6px;font-size:12px;background:#dcfce7;'
                f'padding:3px 6px;border-radius:3px;border-left:3px solid #16a34a;font-weight:600;">'
                f'<span style="color:#14532d;">✓ DEAL</span>'
                f'<span>{d.buyer_id} ↔ {d.seller_id}</span>'
                f'<span style="color:#14532d;">{d.gpu_type.value}×{d.qty}</span>'
                f'<span style="font-family:monospace;color:#14532d;">${d.price_per_gpu_hr:.2f}/hr</span>'
                f'<span style="color:#6b7280;font-weight:400;">total ${d.total_value:.0f}</span>'
                f'</div>'
            )

        if not round_lines:
            round_lines.append('<div style="color:#9ca3af;font-style:italic;font-size:12px;">no actions this round</div>')

        rounds_html.append(
            f'<div style="margin-bottom:12px;">'
            f'<div style="font-weight:600;color:#374151;text-transform:uppercase;font-size:11px;'
            f'letter-spacing:0.05em;margin-bottom:4px;">Round {snap.round_n}</div>'
            f'<div style="display:flex;flex-direction:column;gap:2px;">{"".join(round_lines)}</div>'
            f'</div>'
        )

    return f'<div style="font-family:{FONT_STACK};">{"".join(rounds_html)}</div>'


# ---------------------------------------------------------------------------
# Aggression scoreboard
# ---------------------------------------------------------------------------


def render_aggression_scoreboard(history: NegotiationHistory) -> str:
    """Two columns: top buyers and top sellers by concession-toward-middle."""

    def _row(tl: AgentTimeline) -> str:
        won_badge = '<span style="background:#16a34a;color:#fff;padding:1px 6px;border-radius:8px;font-size:9px;">WON</span>' if tl.won_deal else ''
        agg = tl.aggression
        bar_pct = max(0, min(40, abs(agg))) / 40 * 100
        bar_color = "#16a34a" if agg > 0 else "#dc2626" if agg < 0 else "#9ca3af"
        return (
            f'<div style="display:flex;gap:8px;align-items:center;font-size:12px;padding:4px 0;'
            f'border-bottom:1px solid #f3f4f6;">'
            f'<span style="font-family:monospace;color:#6b7280;min-width:40px;font-size:10px;">{tl.agent_id}</span>'
            f'<span style="min-width:140px;">{tl.label}</span>'
            f'<span style="font-family:monospace;font-size:11px;min-width:80px;color:#6b7280;">'
            f'${tl.initial_price or 0:.2f} → ${tl.final_price or 0:.2f}</span>'
            f'<div style="flex:1;height:5px;background:#f3f4f6;border-radius:2px;overflow:hidden;">'
            f'<div style="height:100%;width:{bar_pct}%;background:{bar_color};"></div></div>'
            f'<span style="font-family:monospace;font-weight:600;color:{bar_color};min-width:48px;text-align:right;">'
            f'{agg:+.1f}%</span>'
            f'{won_badge}'
            f'</div>'
        )

    buyers = sorted(history.buyer_timelines.values(), key=lambda t: -t.aggression)
    sellers = sorted(history.seller_timelines.values(), key=lambda t: -t.aggression)

    def _section(title: str, rows: list[AgentTimeline]) -> str:
        body = "".join(_row(r) for r in rows if r.initial_price is not None) or '<div style="color:#9ca3af;">no offers posted</div>'
        return (
            f'<div style="flex:1;">'
            f'<div style="font-size:11px;font-weight:600;color:#374151;text-transform:uppercase;'
            f'letter-spacing:0.05em;margin-bottom:6px;">{title}</div>'
            f'{body}</div>'
        )

    return (
        f'<div style="display:flex;gap:16px;font-family:{FONT_STACK};">'
        f'{_section("Buyer aggression (climbed)", buyers)}'
        f'{_section("Seller aggression (decayed)", sellers)}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Chat-exchange view — the reasoning bubbles
# ---------------------------------------------------------------------------


def render_chat_exchange(
    history: NegotiationHistory,
    only_with_reasoning: bool = False,
    max_height_px: int = 600,
    agent_models: dict[str, tuple[str, str]] | None = None,
) -> str:
    """Render the negotiation as a chat-app conversation — every offer's reasoning
    as a bubble, in chronological order.

    Buyers on the right (green-tinted), sellers on the left (blue-tinted), like
    iMessage. The reasoning text is the headline — that's where LLM strategy lives.
    Pass `only_with_reasoning=True` to skip deterministic stub reasoning if you
    only care about the LLM-written messages.

    Scrollable container — long negotiations (many rounds × many agents) stay
    contained instead of pushing the rest of the notebook off-screen.
    """
    bubbles: list[str] = []

    for snap in history.snapshots:
        # Round divider — like a date stamp in a chat app.
        bubbles.append(
            f'<div style="text-align:center;color:#9ca3af;font-size:11px;font-weight:600;'
            f'text-transform:uppercase;letter-spacing:0.05em;margin:16px 0 8px;">'
            f'— Round {snap.round_n} —'
            f'</div>'
        )

        for action in snap.actions:
            agent_id = action.agent_id
            is_buyer = agent_id in history.buyer_timelines
            label = (history.buyer_timelines[agent_id].label if is_buyer
                     else next((s.label for s in history.market.sellers if s.id == agent_id), agent_id))
            role_emoji = "🛒" if is_buyer else "🖥️"

            model_badge = _build_model_badge(agent_id, agent_models)

            for offer in action.new_offers:
                if only_with_reasoning and not (offer.reasoning or "").strip():
                    continue
                bubbles.append(
                    _chat_bubble(
                        agent_id=agent_id,
                        agent_label=label,
                        role_emoji=role_emoji,
                        side="right" if is_buyer else "left",
                        action_label=("BID" if is_buyer else "ASK"),
                        action_color=("#16a34a" if is_buyer else "#0369a1"),
                        offer_summary=(
                            f'{offer.gpu_type.value} ×{offer.qty} @ '
                            f'${offer.price_per_gpu_hr:.2f}/hr · '
                            f'{offer.duration}h@slot {offer.start:02d}'
                            + (f' · slot_id={offer.slot_id}' if offer.slot_id else '')
                        ),
                        reasoning=offer.reasoning,
                        model_badge_html=model_badge,
                    )
                )

            for accept_id in action.accept_offer_ids:
                bubbles.append(
                    _chat_bubble(
                        agent_id=agent_id,
                        agent_label=label,
                        role_emoji=role_emoji,
                        side="right" if is_buyer else "left",
                        action_label=f"ACCEPTS {accept_id}",
                        action_color="#d97706",
                        offer_summary="",
                        reasoning=action.reasoning,
                        is_accept=True,
                        model_badge_html=model_badge,
                    )
                )

        for d in snap.new_deals:
            bubbles.append(
                f'<div style="text-align:center;background:#dcfce7;color:#14532d;'
                f'padding:6px 12px;border-radius:12px;display:inline-block;margin:6px auto;'
                f'font-size:11px;font-weight:600;">'
                f'✓ DEAL · {d.buyer_id} ↔ {d.seller_id} · {d.gpu_type.value}×{d.qty} '
                f'@ ${d.price_per_gpu_hr:.2f}/hr (total ${d.total_value:.0f})'
                f'</div>'
            )

    if not bubbles:
        bubbles.append(
            '<div style="color:#9ca3af;font-style:italic;text-align:center;">'
            'No actions captured. (Older preset files predate the action-capture wiring; '
            'use a fresh fast/live run instead.)'
            '</div>'
        )

    body = "".join(bubbles)
    return (
        f'<div style="font-family:{FONT_STACK};max-height:{max_height_px}px;'
        f'overflow-y:auto;padding:8px 12px;background:#fafafa;border-radius:8px;'
        f'border:1px solid #e5e7eb;">'
        f'{body}'
        f'</div>'
    )


_PROVIDER_BADGE_COLOR: dict[str, str] = {
    "anthropic": "#dc7c2f",
    "openai": "#10a37f",
    "fixture": "#6b7280",
}


def _build_model_badge(
    agent_id: str,
    agent_models: dict[str, tuple[str, str]] | None,
) -> str:
    """Inline-styled badge showing 'provider/model' for an agent. Empty if unknown."""
    if not agent_models or agent_id not in agent_models:
        return ""
    provider, model = agent_models[agent_id]
    bg = _PROVIDER_BADGE_COLOR.get(provider, "#9ca3af")
    short_model = (model or "?").replace("-latest", "").replace("claude-", "c-").replace("gpt-", "")
    return (
        f'<span style="background:{bg};color:#fff;padding:1px 5px;border-radius:6px;'
        f'font-size:9px;font-weight:600;font-family:monospace;margin-left:4px;">'
        f'{provider}/{short_model}</span>'
    )


def _chat_bubble(
    *,
    agent_id: str,
    agent_label: str,
    role_emoji: str,
    side: str,
    action_label: str,
    action_color: str,
    offer_summary: str,
    reasoning: str,
    is_accept: bool = False,
    model_badge_html: str = "",
) -> str:
    align = "flex-end" if side == "right" else "flex-start"
    bg = "#dcfce7" if side == "right" else "#dbeafe"   # green = buyer, blue = seller
    border = "#bbf7d0" if side == "right" else "#bfdbfe"

    reasoning_html = ""
    if reasoning and reasoning.strip():
        # Quote it; preserve newlines as <br>.
        safe = (reasoning.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                .replace("\n", "<br>"))
        reasoning_html = (
            f'<div style="font-size:12px;color:#1f2937;margin-top:6px;line-height:1.4;">'
            f'{safe}'
            f'</div>'
        )

    summary_html = ""
    if offer_summary:
        summary_html = (
            f'<div style="font-family:monospace;font-size:11px;color:#374151;margin-top:4px;">'
            f'{offer_summary}'
            f'</div>'
        )

    accept_marker = ""
    if is_accept:
        accept_marker = '<div style="font-size:10px;color:#92400e;font-weight:600;">✓ ACCEPT</div>'

    return (
        f'<div style="display:flex;justify-content:{align};margin:4px 0;">'
        f'<div style="max-width:75%;background:{bg};border:1px solid {border};'
        f'border-radius:10px;padding:8px 12px;">'
        f'<div style="display:flex;justify-content:space-between;gap:10px;align-items:baseline;">'
        f'<strong style="font-size:11px;color:#111;">{role_emoji} {agent_label}{model_badge_html}</strong>'
        f'<span style="font-size:10px;color:{action_color};font-weight:600;">{action_label}</span>'
        f'</div>'
        f'<div style="font-family:monospace;font-size:9px;color:#9ca3af;">{agent_id}</div>'
        f'{summary_html}'
        f'{accept_marker}'
        f'{reasoning_html}'
        f'</div></div>'
    )


__all__ = [
    "AgentTimeline",
    "NegotiationHistory",
    "extract_history",
    "render_timeline",
    "render_log",
    "render_aggression_scoreboard",
    "render_chat_exchange",
]
