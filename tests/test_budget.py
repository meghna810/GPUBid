"""Phase 6 tests: layered budget policy."""

from dataclasses import dataclass, field

import pytest

from gpubid.protocol.budget import BudgetPolicy, StopReason


@dataclass
class _FakeState:
    round_n: int = 0
    active_pairs: list = field(default_factory=lambda: [("B0", "S0")])
    token_usage_by_agent: dict = field(default_factory=dict)
    no_progress_streak_by_pair: dict = field(default_factory=dict)


def test_continues_when_under_all_caps():
    policy = BudgetPolicy(max_rounds=5, per_run_token_cap=100_000, no_progress_streak_cap=3)
    state = _FakeState(round_n=2, token_usage_by_agent={"B0": 500, "S0": 500})
    stop, reason = policy.should_stop(state)
    assert stop is False
    assert reason is None


def test_stops_when_no_active_pairs():
    """All pairs resolved (closed or walked) → ALL_PAIRS_RESOLVED."""
    policy = BudgetPolicy(max_rounds=5, per_run_token_cap=100_000, no_progress_streak_cap=3)
    state = _FakeState(active_pairs=[], round_n=4)
    stop, reason = policy.should_stop(state)
    assert stop is True
    assert reason is StopReason.ALL_PAIRS_RESOLVED


def test_resolved_takes_priority_over_max_rounds():
    """If pairs are all done AND we hit max_rounds in the same call, report resolved."""
    policy = BudgetPolicy(max_rounds=5, per_run_token_cap=100_000, no_progress_streak_cap=3)
    state = _FakeState(active_pairs=[], round_n=10)
    stop, reason = policy.should_stop(state)
    assert reason is StopReason.ALL_PAIRS_RESOLVED


def test_stops_when_max_rounds_reached():
    policy = BudgetPolicy(max_rounds=5, per_run_token_cap=100_000, no_progress_streak_cap=3)
    state = _FakeState(round_n=5)
    stop, reason = policy.should_stop(state)
    assert stop is True
    assert reason is StopReason.MAX_ROUNDS


def test_stops_when_token_cap_exceeded():
    policy = BudgetPolicy(max_rounds=10, per_run_token_cap=1000, no_progress_streak_cap=3)
    state = _FakeState(round_n=2, token_usage_by_agent={"B0": 600, "S0": 500})
    stop, reason = policy.should_stop(state)
    assert stop is True
    assert reason is StopReason.TOKEN_CAP


def test_stops_when_no_progress_streak_hit():
    policy = BudgetPolicy(max_rounds=10, per_run_token_cap=100_000, no_progress_streak_cap=2)
    state = _FakeState(
        round_n=3,
        no_progress_streak_by_pair={("B0", "S0"): 2},
    )
    stop, reason = policy.should_stop(state)
    assert stop is True
    assert reason is StopReason.NO_PROGRESS


def test_no_progress_does_not_fire_below_cap():
    policy = BudgetPolicy(max_rounds=10, per_run_token_cap=100_000, no_progress_streak_cap=3)
    state = _FakeState(
        round_n=3,
        no_progress_streak_by_pair={("B0", "S0"): 2, ("B1", "S1"): 1},
    )
    stop, _ = policy.should_stop(state)
    assert stop is False


def test_max_rounds_takes_priority_over_token_cap_when_simultaneous():
    """When both fire on the same step, report whichever the policy checks first."""
    policy = BudgetPolicy(max_rounds=5, per_run_token_cap=1000, no_progress_streak_cap=10)
    state = _FakeState(round_n=5, token_usage_by_agent={"B0": 5000})
    _, reason = policy.should_stop(state)
    # max_rounds is checked first per the docstring.
    assert reason is StopReason.MAX_ROUNDS


def test_from_settings_uses_config_defaults():
    policy = BudgetPolicy.from_settings()
    assert policy.max_rounds == 5
    assert policy.no_progress_streak_cap == 2
    assert policy.per_run_token_cap == 60_000


def test_constant_disagreement_triggers_no_progress_before_max_rounds():
    """A pair stuck in disagreement should hit NO_PROGRESS first."""
    policy = BudgetPolicy(max_rounds=5, per_run_token_cap=100_000, no_progress_streak_cap=2)
    # Round 1: progressing
    state = _FakeState(round_n=1, no_progress_streak_by_pair={("B0", "S0"): 0})
    assert policy.should_stop(state) == (False, None)
    # Round 2: still progressing
    state = _FakeState(round_n=2, no_progress_streak_by_pair={("B0", "S0"): 1})
    assert policy.should_stop(state) == (False, None)
    # Round 3: stuck, streak hits cap
    state = _FakeState(round_n=3, no_progress_streak_by_pair={("B0", "S0"): 2})
    stop, reason = policy.should_stop(state)
    assert stop is True
    assert reason is StopReason.NO_PROGRESS


def test_max_rounds_fires_when_token_cap_is_huge():
    """Slow movers eventually hit MAX_ROUNDS."""
    policy = BudgetPolicy(max_rounds=3, per_run_token_cap=10_000_000, no_progress_streak_cap=99)
    state = _FakeState(round_n=3, token_usage_by_agent={"B0": 100, "S0": 100})
    stop, reason = policy.should_stop(state)
    assert stop is True
    assert reason is StopReason.MAX_ROUNDS


def test_budget_policy_is_frozen():
    p = BudgetPolicy(max_rounds=5, per_run_token_cap=100, no_progress_streak_cap=2)
    with pytest.raises(Exception):
        p.max_rounds = 10  # type: ignore[misc]
