"""Plotly figures for in-notebook interactivity (hoverable, zoomable)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

from gpubid.eval.metrics import RunMetrics


def baseline_comparison(
    *,
    agentic: RunMetrics,
    vcg: RunMetrics,
    posted: RunMetrics,
    title: str = "Welfare comparison",
) -> "go.Figure":
    """Three-bar chart: Agentic / VCG / Posted-price welfare."""
    import plotly.graph_objects as go

    labels = ["Posted-price", "Agentic", "VCG (optimal)"]
    welfare = [posted.welfare, agentic.welfare, vcg.welfare]
    colors = ["#94a3b8", "#2563eb", "#16a34a"]

    # Recovery vs VCG for hover
    if vcg.welfare > 0:
        recovery_pct = [w / vcg.welfare * 100 for w in welfare]
    else:
        recovery_pct = [0.0 for _ in welfare]

    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=welfare,
                marker_color=colors,
                text=[f"${w:.0f}<br>({p:.0f}% of VCG)" for w, p in zip(welfare, recovery_pct)],
                textposition="auto",
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Welfare: $%{y:.2f}<br>"
                    "Recovery: %{customdata:.1f}% of VCG<extra></extra>"
                ),
                customdata=recovery_pct,
            )
        ]
    )
    fig.update_layout(
        title=title,
        yaxis_title="Welfare ($)",
        xaxis_title=None,
        showlegend=False,
        plot_bgcolor="white",
        height=380,
        margin=dict(l=50, r=20, t=50, b=40),
        font=dict(family="-apple-system, sans-serif"),
    )
    fig.update_yaxes(gridcolor="#e5e7eb", zerolinecolor="#e5e7eb")
    return fig


def metric_table(
    *,
    agentic: RunMetrics,
    vcg: RunMetrics,
    posted: RunMetrics,
) -> str:
    """A small HTML table summarizing the headline metrics across the three mechanisms."""
    rows = [
        ("Welfare ($)", posted.welfare, agentic.welfare, vcg.welfare),
        ("Buyers filled", posted.n_buyers_filled, agentic.n_buyers_filled, vcg.n_buyers_filled),
        ("Avg clearing price", posted.avg_clearing_price, agentic.avg_clearing_price, vcg.avg_clearing_price),
        ("Off-peak utilization (%)", posted.offpeak_utilization * 100, agentic.offpeak_utilization * 100, vcg.offpeak_utilization * 100),
        ("Gini (buyer welfare)", posted.gini_buyer_welfare, agentic.gini_buyer_welfare, vcg.gini_buyer_welfare),
    ]
    header = (
        '<tr style="background:#f3f4f6;">'
        '<th style="text-align:left;padding:6px 10px;border-bottom:1px solid #d1d5db;">Metric</th>'
        '<th style="text-align:right;padding:6px 10px;border-bottom:1px solid #d1d5db;">Posted</th>'
        '<th style="text-align:right;padding:6px 10px;border-bottom:1px solid #d1d5db;">Agentic</th>'
        '<th style="text-align:right;padding:6px 10px;border-bottom:1px solid #d1d5db;">VCG</th>'
        "</tr>"
    )
    body = "".join(
        f'<tr><td style="padding:6px 10px;border-bottom:1px solid #f3f4f6;">{label}</td>'
        f'<td style="padding:6px 10px;text-align:right;font-family:monospace;">{p:.2f}</td>'
        f'<td style="padding:6px 10px;text-align:right;font-family:monospace;">{a:.2f}</td>'
        f'<td style="padding:6px 10px;text-align:right;font-family:monospace;">{v:.2f}</td></tr>'
        for label, p, a, v in rows
    )
    return (
        '<table style="border-collapse:collapse;font-family:-apple-system,sans-serif;'
        'font-size:12px;margin-top:8px;width:100%;max-width:560px;">'
        f"{header}{body}</table>"
    )


__all__ = ["baseline_comparison", "metric_table"]
