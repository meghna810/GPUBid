"""Multi-simulation provider tournament.

Run the same market under different LLM-provider assignments to answer:
"Who wins more — Claude buyers or OpenAI buyers?" and "Are Claude sellers
more aggressive at decaying prices than OpenAI sellers?"

Typical use:

    from gpubid.analysis.tournament import head_to_head_alternating, render_tournament_report
    result = head_to_head_alternating(
        n_seeds=10,
        api_keys={
            'anthropic': os.environ['ANTHROPIC_API_KEY'],
            'openai':    os.environ['OPENAI_API_KEY'],
        },
        n_buyers=8, n_sellers=4, regime='tight', max_rounds=5,
    )
    display(HTML(render_tournament_report(result)))
    fig = render_tournament_chart(result)
    fig.show()

Each seed runs once. A tournament with 10 seeds × 12 agents × 5 rounds ≈
600 LLM calls. At Haiku 4.5 / GPT-4o-mini pricing that's roughly $1-3 total.

For a fast-mode tournament (no API keys) use `head_to_head_deterministic` —
"providers" are simulated with two parameter variants of the deterministic
agent, useful for pipeline testing without API costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import plotly.graph_objects as go

from gpubid.analysis.forensics import NegotiationHistory, extract_history
from gpubid.engine.round_runner import (
    make_deterministic_agents,
    make_llm_agents_assigned,
    run_rounds,
)
from gpubid.llm import make_client
from gpubid.market import generate_market
from gpubid.schema import Market
from gpubid.viz.market_view import FONT_STACK


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class SeedResult:
    """One seed's run within a tournament."""

    seed: int
    market: Market
    history: NegotiationHistory
    buyer_assignment: dict[str, str]   # buyer_id -> provider
    seller_assignment: dict[str, str]


@dataclass
class TournamentResult:
    """Aggregate stats across multiple seeds."""

    name: str
    description: str
    seeds: list[SeedResult] = field(default_factory=list)
    provider_models: dict[str, str] = field(default_factory=dict)

    def per_provider_buyer_stats(self) -> dict[str, dict]:
        """For each provider on the buyer side: total buyers, # who won, avg deal price, avg aggression."""
        agg: dict[str, dict] = {}
        for seed in self.seeds:
            for b in seed.market.buyers:
                provider = seed.buyer_assignment.get(b.id, "?")
                stats = agg.setdefault(provider, {
                    "n_buyers": 0, "n_won": 0, "deal_prices": [],
                    "buyer_surplus": [], "aggressions": [],
                })
                stats["n_buyers"] += 1
                tl = seed.history.buyer_timelines.get(b.id)
                if tl is None:
                    continue
                stats["aggressions"].append(tl.aggression)
                if tl.won_deal:
                    stats["n_won"] += 1
                    stats["deal_prices"].append(tl.won_deal.price_per_gpu_hr)
                    stats["buyer_surplus"].append(
                        (b.job.max_value_per_gpu_hr - tl.won_deal.price_per_gpu_hr)
                        * tl.won_deal.qty * tl.won_deal.duration
                    )
        # Reduce
        out: dict[str, dict] = {}
        for provider, s in agg.items():
            n_buyers = s["n_buyers"]
            n_won = s["n_won"]
            out[provider] = {
                "n_buyers": n_buyers,
                "n_won": n_won,
                "win_rate": n_won / n_buyers if n_buyers else 0,
                "avg_deal_price": (sum(s["deal_prices"]) / len(s["deal_prices"])) if s["deal_prices"] else 0,
                "avg_buyer_surplus": (sum(s["buyer_surplus"]) / len(s["buyer_surplus"])) if s["buyer_surplus"] else 0,
                "avg_aggression": (sum(s["aggressions"]) / len(s["aggressions"])) if s["aggressions"] else 0,
            }
        return out

    def per_provider_seller_stats(self) -> dict[str, dict]:
        """For each provider on the seller side."""
        agg: dict[str, dict] = {}
        for seed in self.seeds:
            for s in seed.market.sellers:
                provider = seed.seller_assignment.get(s.id, "?")
                stats = agg.setdefault(provider, {
                    "n_slots": 0, "n_sold": 0, "deal_prices": [],
                    "seller_revenue": [], "aggressions": [],
                })
                for slot in s.capacity_slots:
                    stats["n_slots"] += 1
                    tl = seed.history.seller_timelines.get(slot.id)
                    if tl is None:
                        continue
                    stats["aggressions"].append(tl.aggression)
                    if tl.won_deal:
                        stats["n_sold"] += 1
                        stats["deal_prices"].append(tl.won_deal.price_per_gpu_hr)
                        stats["seller_revenue"].append(
                            (tl.won_deal.price_per_gpu_hr - slot.reserve_per_gpu_hr)
                            * tl.won_deal.qty * tl.won_deal.duration
                        )
        out: dict[str, dict] = {}
        for provider, s in agg.items():
            n_slots = s["n_slots"]
            n_sold = s["n_sold"]
            out[provider] = {
                "n_slots": n_slots,
                "n_sold": n_sold,
                "sell_rate": n_sold / n_slots if n_slots else 0,
                "avg_deal_price": (sum(s["deal_prices"]) / len(s["deal_prices"])) if s["deal_prices"] else 0,
                "avg_seller_revenue": (sum(s["seller_revenue"]) / len(s["seller_revenue"])) if s["seller_revenue"] else 0,
                "avg_aggression": (sum(s["aggressions"]) / len(s["aggressions"])) if s["aggressions"] else 0,
            }
        return out


# ---------------------------------------------------------------------------
# Assignment helpers
# ---------------------------------------------------------------------------


def alternating_assignment(ids: list[str], providers: list[str]) -> dict[str, str]:
    """Round-robin assignment — id i gets providers[i % len(providers)]."""
    return {agent_id: providers[i % len(providers)] for i, agent_id in enumerate(ids)}


def uniform_assignment(ids: list[str], provider: str) -> dict[str, str]:
    return {agent_id: provider for agent_id in ids}


# ---------------------------------------------------------------------------
# Live LLM tournament
# ---------------------------------------------------------------------------


def head_to_head_alternating(
    n_seeds: int,
    api_keys: dict[str, str],
    *,
    n_buyers: int = 8,
    n_sellers: int = 4,
    regime: str = "tight",
    max_rounds: int = 5,
    model: Optional[str] = None,
    progress: bool = True,
) -> TournamentResult:
    """Alternating-providers tournament: half the buyers Claude, half OpenAI; same for sellers.

    Backward-compatible 2-provider entry point. For an N-provider tournament
    (e.g. Anthropic vs OpenAI vs Gemini), use `head_to_head_multi`.

    Returns a TournamentResult covering all seeds. Each seed uses a different
    synthetic market so we're averaging over market conditions, not just one shape.
    """
    if not all(p in api_keys for p in ("anthropic", "openai")):
        raise ValueError("Need both 'anthropic' and 'openai' keys in api_keys for the alternating tournament.")

    result = TournamentResult(
        name="alternating",
        description=f"{n_seeds} seeds × {n_buyers}b×{n_sellers}s {regime}; buyers and sellers alternate Anthropic/OpenAI",
    )

    for i, seed in enumerate(range(n_seeds)):
        market = generate_market(n_buyers, n_sellers, regime, seed=seed)  # type: ignore[arg-type]
        buyer_assignment = alternating_assignment([b.id for b in market.buyers], ["anthropic", "openai"])
        seller_assignment = alternating_assignment([s.id for s in market.sellers], ["anthropic", "openai"])

        if progress:
            print(f"[{i+1}/{n_seeds}] seed={seed} running…", flush=True)

        buyers, sellers = make_llm_agents_assigned(
            market,
            api_keys=api_keys,
            buyer_assignment=buyer_assignment,
            seller_assignment=seller_assignment,
            model=model,
        )
        snaps = list(run_rounds(market, buyers, sellers, max_rounds=max_rounds))
        history = extract_history(market, snaps)

        result.seeds.append(SeedResult(
            seed=seed,
            market=market,
            history=history,
            buyer_assignment=buyer_assignment,
            seller_assignment=seller_assignment,
        ))

        if progress:
            n_deals = len(snaps[-1].all_deals)
            print(f"          → {n_deals} deals struck", flush=True)

    return result


# ---------------------------------------------------------------------------
# N-provider tournament (e.g. Anthropic vs OpenAI vs Gemini)
# ---------------------------------------------------------------------------


def head_to_head_multi(
    n_seeds: int,
    api_keys: dict[str, str],
    *,
    n_buyers: int = 8,
    n_sellers: int = 4,
    regime: str = "tight",
    max_rounds: int = 5,
    provider_models: Optional[dict] = None,
    intra_provider_mode: bool = False,
    progress: bool = True,
) -> TournamentResult:
    """Round-robin tournament — across providers OR (with `intra_provider_mode`)
    across model versions within ONE provider.

    Two modes:

    1. **Cross-provider (default).** `api_keys` includes any subset of
       {'anthropic', 'openai', 'gemini'} (>=2 entries). Buyers and sellers
       are round-robined across the providers. `provider_models` is an
       optional `{provider: model_string}` map for per-provider model pinning.

    2. **Intra-provider** (`intra_provider_mode=True`). `api_keys` has ONE
       provider entry. `provider_models[provider]` is a list of model strings
       to round-robin across. Use this to compare e.g.
       Haiku vs Sonnet vs Opus within Anthropic, or 4o-mini vs 4o within
       OpenAI. Buyers and sellers are split across the model list, and
       per-agent (provider, model) shows up in the persuasion analytics so
       you can see "Opus buyers extracted N% more surplus than Haiku buyers".

    Cost note: 3 entrants × 5 seeds × 12 agents × 5 rounds ≈ 900 LLM calls.
    Roughly $1-3 with cost-effective default models, more with frontier
    models.
    """
    if intra_provider_mode:
        # Pool every (provider, model) tuple across all providers in api_keys
        # into one entrant list. Backward-compatible: a single-provider-with-
        # list-of-models call still works; a multi-provider-with-lists call
        # now produces the "cross-tier" tournament (e.g. Haiku vs Sonnet vs
        # Opus vs 4o-mini vs 4o all in one round-robin).
        if not provider_models:
            raise ValueError(
                "intra_provider_mode needs provider_models={provider: [model, ...]}"
            )

        entrants: list[tuple[str, str]] = []
        for prov, models in provider_models.items():
            if prov not in api_keys:
                raise ValueError(
                    f"provider_models has {prov!r} but no api_keys[{prov!r}]"
                )
            if not isinstance(models, (list, tuple)):
                raise ValueError(
                    f"intra_provider_mode needs provider_models[{prov!r}] = list of models; "
                    f"got {type(models).__name__}"
                )
            for m in models:
                entrants.append((prov, m))

        if len(entrants) < 2:
            raise ValueError(
                f"intra_provider_mode needs >=2 (provider, model) entrants total; got {entrants}"
            )

        # Sanity-construct each (provider, model) once so bad combos fail fast.
        for prov, m in entrants:
            try:
                make_client(api_keys[prov], model=m)
            except Exception as e:
                raise ValueError(
                    f"Could not construct {prov}/{m} client: {e}"
                )

        # Each entrant labeled "anthropic/claude-haiku-4-5", "openai/gpt-4o-mini", etc.
        entrant_labels = [f"{prov}/{m}" for prov, m in entrants]
        providers_used = sorted({p for p, _ in entrants})

        result = TournamentResult(
            name=("intra-" + providers_used[0]) if len(providers_used) == 1
                 else "cross-tier-" + "-".join(providers_used),
            description=(
                f"{n_seeds} seeds × {n_buyers}b×{n_sellers}s {regime}; "
                f"round-robin across {len(entrants)} (provider, model) entrants: "
                f"{', '.join(entrant_labels)}"
            ),
            provider_models={lbl: m for lbl, (_, m) in zip(entrant_labels, entrants)},
        )

        for i, seed in enumerate(range(n_seeds)):
            market = generate_market(n_buyers, n_sellers, regime, seed=seed)
            buyer_assignment = alternating_assignment(
                [b.id for b in market.buyers], entrant_labels
            )
            seller_assignment = alternating_assignment(
                [s.id for s in market.sellers], entrant_labels
            )

            if progress:
                print(f"[{i+1}/{n_seeds}] seed={seed} running…", flush=True)

            from gpubid.agents.buyer import LLMBuyer
            from gpubid.agents.seller import LLMSeller

            # One client per (provider, model) entrant.
            clients = {
                lbl: make_client(api_keys[prov], model=m)
                for lbl, (prov, m) in zip(entrant_labels, entrants)
            }
            buyers = {
                b.id: LLMBuyer(buyer_id=b.id, client=clients[buyer_assignment[b.id]])
                for b in market.buyers if b.id in buyer_assignment
            }
            sellers = {
                s.id: LLMSeller(seller_id=s.id, client=clients[seller_assignment[s.id]])
                for s in market.sellers if s.id in seller_assignment
            }

            snaps = list(run_rounds(market, buyers, sellers, max_rounds=max_rounds))
            history = extract_history(market, snaps)

            result.seeds.append(SeedResult(
                seed=seed, market=market, history=history,
                buyer_assignment=buyer_assignment,
                seller_assignment=seller_assignment,
            ))
            if progress:
                print(f"          → {len(snaps[-1].all_deals)} deals struck", flush=True)
        return result

    # ---- Cross-provider mode (default) ----
    providers = sorted(api_keys.keys())
    if len(providers) < 2:
        raise ValueError(
            f"head_to_head_multi needs at least 2 providers; got {providers}"
        )

    # Pre-build the LLMClient per provider once per tournament so we don't pay
    # client init cost on every seed.
    pinned_models: dict[str, str] = {}
    if provider_models:
        for prov, m in provider_models.items():
            # Defensively reject lists in cross-provider mode; only intra
            # mode accepts lists per provider.
            if isinstance(m, str) and m:
                pinned_models[prov] = m

    # Sanity-construct each client once so a bad key fails fast.
    for prov in providers:
        try:
            make_client(api_keys[prov], model=pinned_models.get(prov))
        except Exception as e:
            raise ValueError(
                f"Could not construct {prov} client (bad key or SDK missing): {e}"
            )

    result = TournamentResult(
        name="multi-" + "-".join(providers),
        description=(
            f"{n_seeds} seeds × {n_buyers}b×{n_sellers}s {regime}; "
            f"round-robin across {', '.join(providers)}"
        ),
        provider_models=pinned_models or {p: "<default>" for p in providers},
    )

    for i, seed in enumerate(range(n_seeds)):
        market = generate_market(n_buyers, n_sellers, regime, seed=seed)  # type: ignore[arg-type]
        buyer_assignment = alternating_assignment(
            [b.id for b in market.buyers], providers
        )
        seller_assignment = alternating_assignment(
            [s.id for s in market.sellers], providers
        )

        if progress:
            print(f"[{i+1}/{n_seeds}] seed={seed} running…", flush=True)

        # Per-provider client pool, with model overrides preserved.
        from gpubid.agents.buyer import LLMBuyer
        from gpubid.agents.seller import LLMSeller

        clients = {
            p: make_client(api_keys[p], model=pinned_models.get(p))
            for p in providers
        }
        buyers = {
            b.id: LLMBuyer(buyer_id=b.id, client=clients[buyer_assignment[b.id]])
            for b in market.buyers if b.id in buyer_assignment
        }
        sellers = {
            s.id: LLMSeller(seller_id=s.id, client=clients[seller_assignment[s.id]])
            for s in market.sellers if s.id in seller_assignment
        }

        snaps = list(run_rounds(market, buyers, sellers, max_rounds=max_rounds))
        history = extract_history(market, snaps)

        result.seeds.append(SeedResult(
            seed=seed,
            market=market,
            history=history,
            buyer_assignment=buyer_assignment,
            seller_assignment=seller_assignment,
        ))

        if progress:
            n_deals = len(snaps[-1].all_deals)
            print(f"          → {n_deals} deals struck", flush=True)

    return result


# ---------------------------------------------------------------------------
# Fast-mode tournament (no API keys, for pipeline testing)
# ---------------------------------------------------------------------------


def head_to_head_deterministic(
    n_seeds: int,
    *,
    n_buyers: int = 8,
    n_sellers: int = 4,
    regime: str = "tight",
    max_rounds: int = 5,
) -> TournamentResult:
    """No-key tournament that exercises the framework with deterministic agents.

    All "providers" are the same deterministic strategy here — useful only to
    confirm the report renders. For real provider comparison, use
    `head_to_head_alternating` with both API keys.
    """
    result = TournamentResult(
        name="deterministic_smoke",
        description=f"{n_seeds} seeds × {n_buyers}b×{n_sellers}s {regime}; deterministic agents (synthetic 'providers')",
    )

    for seed in range(n_seeds):
        market = generate_market(n_buyers, n_sellers, regime, seed=seed)  # type: ignore[arg-type]
        # Tag agents with synthetic provider labels so the aggregation works.
        buyer_assignment = alternating_assignment([b.id for b in market.buyers], ["anthropic", "openai"])
        seller_assignment = alternating_assignment([s.id for s in market.sellers], ["anthropic", "openai"])

        buyers, sellers = make_deterministic_agents(market)
        snaps = list(run_rounds(market, buyers, sellers, max_rounds=max_rounds))
        history = extract_history(market, snaps)

        result.seeds.append(SeedResult(
            seed=seed,
            market=market,
            history=history,
            buyer_assignment=buyer_assignment,
            seller_assignment=seller_assignment,
        ))
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def render_tournament_report(result: TournamentResult, *, title: Optional[str] = None) -> str:
    """HTML summary table — the headline output for the notebook.

    `title` overrides the default `result.name` heading (useful for the
    intra-provider variant: "Within Anthropic: model-version tournament").
    """
    buyer_stats = result.per_provider_buyer_stats()
    seller_stats = result.per_provider_seller_stats()

    def _table(rows: list[tuple], header: list[str], note: str) -> str:
        header_html = "".join(
            f'<th style="text-align:left;padding:6px 10px;border-bottom:1px solid #d1d5db;'
            f'font-size:11px;color:#374151;">{h}</th>'
            for h in header
        )
        body_html = ""
        for row in rows:
            cells = "".join(
                f'<td style="padding:6px 10px;border-bottom:1px solid #f3f4f6;'
                f'font-family:monospace;font-size:12px;">{c}</td>'
                for c in row
            )
            body_html += f"<tr>{cells}</tr>"
        return (
            f'<div style="margin:8px 0;">'
            f'<div style="font-size:11px;color:#6b7280;margin-bottom:2px;">{note}</div>'
            f'<table style="border-collapse:collapse;width:100%;max-width:760px;">'
            f'<thead><tr style="background:#f3f4f6;">{header_html}</tr></thead>'
            f'<tbody>{body_html}</tbody></table>'
            f'</div>'
        )

    buyer_rows = []
    for prov, s in sorted(buyer_stats.items()):
        buyer_rows.append((
            prov,
            f"{s['n_won']}/{s['n_buyers']}",
            f"{s['win_rate']*100:.0f}%",
            f"${s['avg_deal_price']:.2f}",
            f"${s['avg_buyer_surplus']:.0f}",
            f"{s['avg_aggression']:+.1f}%",
        ))
    seller_rows = []
    for prov, s in sorted(seller_stats.items()):
        seller_rows.append((
            prov,
            f"{s['n_sold']}/{s['n_slots']}",
            f"{s['sell_rate']*100:.0f}%",
            f"${s['avg_deal_price']:.2f}",
            f"${s['avg_seller_revenue']:.0f}",
            f"{s['avg_aggression']:+.1f}%",
        ))

    header_buyer = ["Provider", "Won/Total", "Win rate", "Avg deal price", "Avg surplus", "Avg aggression (climb)"]
    header_seller = ["Provider", "Sold/Total", "Sell rate", "Avg deal price", "Avg revenue", "Avg aggression (decay)"]

    return (
        f'<div style="font-family:{FONT_STACK};">'
        f'<h3 style="margin:0 0 6px;">{title or result.name} — {len(result.seeds)} seeds</h3>'
        f'<div style="font-size:12px;color:#6b7280;margin-bottom:8px;">{result.description}</div>'
        f'{_table(buyer_rows, header_buyer, "Per-provider buyer outcomes (across all buyers in all seeds)")}'
        f'{_table(seller_rows, header_seller, "Per-provider seller outcomes (across all slots in all seeds)")}'
        f'</div>'
    )


def render_tournament_chart(result: TournamentResult) -> "go.Figure":
    """Two-panel bar chart: buyer win rate by provider, seller sell rate by provider."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    buyer_stats = result.per_provider_buyer_stats()
    seller_stats = result.per_provider_seller_stats()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Buyers — win rate (%)", "Sellers — sell rate (%)"),
        horizontal_spacing=0.15,
    )

    providers = sorted(set(buyer_stats) | set(seller_stats))
    colors = {"anthropic": "#dc7c2f", "openai": "#10a37f"}

    fig.add_trace(go.Bar(
        x=providers,
        y=[buyer_stats.get(p, {}).get("win_rate", 0) * 100 for p in providers],
        marker_color=[colors.get(p, "#6b7280") for p in providers],
        text=[f"{buyer_stats.get(p, {}).get('n_won', 0)}/{buyer_stats.get(p, {}).get('n_buyers', 0)}" for p in providers],
        textposition="outside",
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Win rate: %{y:.1f}%<extra></extra>",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=providers,
        y=[seller_stats.get(p, {}).get("sell_rate", 0) * 100 for p in providers],
        marker_color=[colors.get(p, "#6b7280") for p in providers],
        text=[f"{seller_stats.get(p, {}).get('n_sold', 0)}/{seller_stats.get(p, {}).get('n_slots', 0)}" for p in providers],
        textposition="outside",
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>Sell rate: %{y:.1f}%<extra></extra>",
    ), row=1, col=2)

    fig.update_yaxes(range=[0, 110], gridcolor="#e5e7eb")
    fig.update_layout(
        title=f"{result.name} — {len(result.seeds)} seeds",
        height=380,
        plot_bgcolor="white",
        font=dict(family="-apple-system, sans-serif"),
        margin=dict(l=40, r=20, t=70, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Baseline comparison — agentic vs VCG vs posted-price across a tournament
# ---------------------------------------------------------------------------


@dataclass
class BaselineComparisonRow:
    """Per-seed baseline numbers for the tournament summary."""

    seed: int
    agentic_welfare: float
    vcg_welfare: float
    posted_welfare: float
    agentic_deals: int
    vcg_deals: int
    posted_deals: int

    @property
    def recovery_pct(self) -> float:
        return (self.agentic_welfare / self.vcg_welfare * 100) if self.vcg_welfare else 0.0

    @property
    def vs_posted_pct(self) -> float:
        return (
            self.agentic_welfare / self.posted_welfare * 100
            if self.posted_welfare else 0.0
        )


def compute_baseline_comparison(result: TournamentResult) -> list[BaselineComparisonRow]:
    """For every seed in the tournament, compute VCG + posted-price welfare.

    Used to surface 'tournament agents recovered X% of optimal vs Y% for
    posted-price' alongside the per-provider win-rate numbers.
    """
    from gpubid.benchmark.posted_price import solve_posted_price
    from gpubid.benchmark.vcg import compute_welfare, solve_vcg

    rows: list[BaselineComparisonRow] = []
    for seed_result in result.seeds:
        deals = list(seed_result.history.snapshots[-1].all_deals) if seed_result.history.snapshots else []
        agentic_w = compute_welfare(seed_result.market, deals)
        vcg_r = solve_vcg(seed_result.market)
        posted_r = solve_posted_price(seed_result.market)
        rows.append(BaselineComparisonRow(
            seed=seed_result.seed,
            agentic_welfare=agentic_w,
            vcg_welfare=vcg_r.welfare,
            posted_welfare=compute_welfare(seed_result.market, posted_r.deals),
            agentic_deals=len(deals),
            vcg_deals=len(vcg_r.deals),
            posted_deals=len(posted_r.deals),
        ))
    return rows


def render_baseline_comparison(rows: list[BaselineComparisonRow]) -> str:
    """HTML table: per-seed welfare across agentic / VCG / posted, plus aggregate."""
    if not rows:
        return '<em style="color:#9ca3af;">No seeds.</em>'

    body_rows = []
    for r in rows:
        body_rows.append(
            f'<tr>'
            f'<td style="padding:6px 10px;font-family:monospace;color:#6b7280;">{r.seed}</td>'
            f'<td style="padding:6px 10px;font-family:monospace;text-align:right;">${r.agentic_welfare:.0f}</td>'
            f'<td style="padding:6px 10px;font-family:monospace;text-align:right;">${r.vcg_welfare:.0f}</td>'
            f'<td style="padding:6px 10px;font-family:monospace;text-align:right;">${r.posted_welfare:.0f}</td>'
            f'<td style="padding:6px 10px;font-family:monospace;text-align:right;color:'
            f'{"#16a34a" if r.recovery_pct >= 80 else "#d97706" if r.recovery_pct >= 50 else "#dc2626"};">'
            f'{r.recovery_pct:.0f}%</td>'
            f'<td style="padding:6px 10px;font-family:monospace;text-align:right;color:'
            f'{"#16a34a" if r.vs_posted_pct >= 100 else "#dc2626"};">{r.vs_posted_pct:.0f}%</td>'
            f'<td style="padding:6px 10px;font-family:monospace;text-align:right;color:#6b7280;">'
            f'{r.agentic_deals}/{r.vcg_deals}</td>'
            f'</tr>'
        )

    avg_recov = sum(r.recovery_pct for r in rows) / len(rows)
    avg_vsposted = sum(r.vs_posted_pct for r in rows) / len(rows)

    return (
        f'<div style="font-family:{FONT_STACK};margin-top:10px;">'
        f'<div style="font-size:13px;font-weight:600;color:#374151;margin-bottom:6px;">'
        f'Baseline comparison — agentic vs VCG vs posted-price (per seed)'
        f'</div>'
        f'<table style="border-collapse:collapse;width:100%;max-width:880px;font-size:12px;">'
        f'<thead><tr style="background:#f3f4f6;color:#374151;text-align:right;font-size:11px;">'
        f'<th style="padding:6px 10px;text-align:left;">seed</th>'
        f'<th style="padding:6px 10px;">agentic ($)</th>'
        f'<th style="padding:6px 10px;">VCG ($)</th>'
        f'<th style="padding:6px 10px;">posted ($)</th>'
        f'<th style="padding:6px 10px;">% of VCG</th>'
        f'<th style="padding:6px 10px;">vs posted</th>'
        f'<th style="padding:6px 10px;">deals (agentic/VCG)</th>'
        f'</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody>'
        f'</table>'
        f'<div style="margin-top:8px;padding:8px 12px;background:#f9fafb;border-left:3px solid #2563eb;'
        f'border-radius:4px;font-size:12px;color:#374151;">'
        f'Across {len(rows)} seeds: <strong>agentic recovered {avg_recov:.0f}% of VCG welfare</strong>, '
        f'<strong>{avg_vsposted:.0f}% of posted-price welfare</strong> '
        f'(>100% = agentic strictly better than posted).'
        f'</div>'
        f'</div>'
    )


__all__ = [
    "SeedResult",
    "TournamentResult",
    "BaselineComparisonRow",
    "alternating_assignment",
    "uniform_assignment",
    "head_to_head_alternating",
    "head_to_head_multi",
    "head_to_head_deterministic",
    "render_tournament_report",
    "render_tournament_chart",
    "compute_baseline_comparison",
    "render_baseline_comparison",
]
