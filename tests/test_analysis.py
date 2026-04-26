"""Tests for forensics + tournament analysis (deterministic mode — no LLM calls)."""

from gpubid.analysis.forensics import (
    extract_history,
    render_aggression_scoreboard,
    render_log,
    render_timeline,
)
from gpubid.analysis.tournament import (
    alternating_assignment,
    head_to_head_deterministic,
    render_tournament_chart,
    render_tournament_report,
    uniform_assignment,
)
from gpubid.market import generate_market
from gpubid.viz.trading_floor import collect_snapshots


# ---------------------------------------------------------------------------
# Forensics
# ---------------------------------------------------------------------------


def test_extract_history_captures_all_buyers():
    market = generate_market(8, 4, "tight", seed=42)
    snaps = collect_snapshots(market, max_rounds=5)
    history = extract_history(market, snaps)
    assert len(history.buyer_timelines) == 8
    # Each seller has 2..4 slots; total is variable but at least n_sellers
    assert len(history.seller_timelines) >= 4


def test_extract_history_records_initial_and_final_prices():
    market = generate_market(8, 4, "tight", seed=42)
    snaps = collect_snapshots(market, max_rounds=5)
    history = extract_history(market, snaps)
    for tl in history.buyer_timelines.values():
        if tl.prices_by_round:
            assert tl.initial_price is not None
            assert tl.final_price is not None


def test_buyer_aggression_signs_correctly():
    """A buyer who climbed should have aggression > 0; one who never re-bid should have 0."""
    market = generate_market(8, 4, "tight", seed=42)
    snaps = collect_snapshots(market, max_rounds=5)
    history = extract_history(market, snaps)
    # At least one buyer should have climbed (positive aggression)
    climbed = [tl for tl in history.buyer_timelines.values() if tl.aggression > 0.5]
    assert climbed, "expected at least one buyer to climb their bid"


def test_seller_aggression_signs_correctly():
    """Deterministic sellers all decay their asks → aggression > 0."""
    market = generate_market(8, 4, "tight", seed=42)
    snaps = collect_snapshots(market, max_rounds=5)
    history = extract_history(market, snaps)
    decayed = [tl for tl in history.seller_timelines.values() if tl.aggression > 0]
    # Most slots that posted multiple asks should show positive aggression
    posted = [tl for tl in history.seller_timelines.values() if len(tl.prices_by_round) >= 2]
    assert len(decayed) >= len(posted) - 1  # allow one edge case


def test_render_timeline_returns_plotly_figure():
    market = generate_market(6, 3, "tight", seed=7)
    snaps = collect_snapshots(market, max_rounds=5)
    history = extract_history(market, snaps)
    fig = render_timeline(history)
    # Plotly figure has a `data` attribute with traces
    assert hasattr(fig, "data")
    assert len(fig.data) > 0


def test_render_log_includes_round_headers():
    market = generate_market(4, 2, "tight", seed=1)
    snaps = collect_snapshots(market, max_rounds=3)
    history = extract_history(market, snaps)
    html = render_log(history)
    assert "Round 1" in html
    assert "Round 2" in html
    assert "Round 3" in html


def test_render_aggression_scoreboard_lists_both_sides():
    market = generate_market(4, 2, "tight", seed=1)
    snaps = collect_snapshots(market, max_rounds=3)
    history = extract_history(market, snaps)
    html = render_aggression_scoreboard(history)
    assert "Buyer aggression" in html
    assert "Seller aggression" in html


# ---------------------------------------------------------------------------
# Tournament
# ---------------------------------------------------------------------------


def test_alternating_assignment_round_robins():
    ids = ["a", "b", "c", "d", "e"]
    out = alternating_assignment(ids, ["X", "Y"])
    assert out == {"a": "X", "b": "Y", "c": "X", "d": "Y", "e": "X"}


def test_uniform_assignment_all_one_provider():
    ids = ["a", "b", "c"]
    out = uniform_assignment(ids, "Z")
    assert all(v == "Z" for v in out.values())
    assert set(out.keys()) == {"a", "b", "c"}


def test_head_to_head_deterministic_runs_and_aggregates():
    result = head_to_head_deterministic(n_seeds=3, n_buyers=6, n_sellers=3, regime="tight")
    assert len(result.seeds) == 3
    bs = result.per_provider_buyer_stats()
    ss = result.per_provider_seller_stats()
    assert "anthropic" in bs and "openai" in bs
    assert "anthropic" in ss and "openai" in ss
    # Across 3 seeds × 6 buyers, alternating gives 9 anthropic + 9 openai (or 9+9 / 12+6 depending on parity).
    total_buyers = bs["anthropic"]["n_buyers"] + bs["openai"]["n_buyers"]
    assert total_buyers == 18  # 3 seeds × 6 buyers


def test_tournament_report_renders_html():
    result = head_to_head_deterministic(n_seeds=2, n_buyers=4, n_sellers=2)
    html = render_tournament_report(result)
    assert "<table" in html
    assert "anthropic" in html and "openai" in html


def test_tournament_chart_returns_plotly_figure():
    result = head_to_head_deterministic(n_seeds=2, n_buyers=4, n_sellers=2)
    fig = render_tournament_chart(result)
    assert hasattr(fig, "data")
    # Two bar traces (buyers + sellers)
    assert len(fig.data) >= 2


# ---------------------------------------------------------------------------
# Action capture round-trip (preset save/load)
# ---------------------------------------------------------------------------


def test_actions_round_trip_through_preset_serialization():
    """Saving + loading a preset must preserve action records."""
    import tempfile
    from pathlib import Path
    from gpubid.experiments.bake_presets import (
        PresetSpec, deserialize_snapshot, save_preset, serialize_snapshot,
    )

    market = generate_market(4, 2, "tight", seed=1)
    snaps = collect_snapshots(market, max_rounds=3)
    spec = PresetSpec(scenario_id="action_rt", description="rt", n_buyers=4, n_sellers=2,
                      regime="tight", seed=1, max_rounds=3)
    with tempfile.TemporaryDirectory() as tmp:
        path = save_preset(spec=spec, market=market, snapshots=list(snaps),
                           metadata={}, output_dir=Path(tmp))
        # Round-trip a single snapshot
        import json
        raw = json.loads(path.read_text())
        rebuilt = deserialize_snapshot(raw["snapshots"][1])
        original = snaps[1]
    # Same number of action records, same first-action agent_id
    assert len(rebuilt.actions) == len(original.actions)
    if rebuilt.actions:
        assert rebuilt.actions[0].agent_id == original.actions[0].agent_id
