"""Phase 12 — Simulation harness with persistent results. SCAFFOLDED.

Per spec §13: SimSpec dataclass + run_simulation that writes to
``data/runs/<UTC-timestamp>__<spec.name>/`` with results.parquet,
deals.parquet, metrics.json, transcript_seed_<s>.jsonl (first 3 only),
summary.md.

Awaits the Phase 5 negotiation runner being completed against fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from gpubid.config import settings
from gpubid.protocol.budget import BudgetPolicy


@dataclass
class SimSpec:
    """Specification for one simulation run."""

    name: str
    n_seeds: int = 10
    seeds: list[int] | None = None
    seed_base: int = 1000
    market_kwargs: dict = field(default_factory=dict)
    provider_matrix: list[tuple[str, str]] = field(default_factory=list)
    prompt_variants: list[tuple[str, str]] = field(default_factory=list)
    budget_policy: BudgetPolicy | None = None
    cache_enabled: bool = True
    record_hitl: bool = False


@dataclass
class SimulationRunResult:
    run_dir: Path
    n_seeds: int
    n_deals_total: int
    cost_estimate_usd: float


def run_simulation(spec: SimSpec) -> SimulationRunResult:
    """SCAFFOLDED. Implementation depends on Phase 5 negotiation runner being live.

    The persistence layer (parquet files, metrics.json, summary.md) is
    independent and could be implemented now, but without runs to persist
    there's no surface to test against.
    """
    raise NotImplementedError(
        "Phase 12 run_simulation is scaffolded. Wire after Phase 5's "
        "run_negotiation is recording-fixture-backed."
    )


def load_all_runs(runs_dir: Path | None = None):
    """Concatenate metrics.json across all runs in runs_dir into a DataFrame.

    SCAFFOLDED. Returns an empty DataFrame when no runs exist (so the trend
    cell renders cleanly on first use).
    """
    import pandas as pd
    rd = runs_dir or settings.simulation.runs_dir
    if not rd.exists():
        return pd.DataFrame()
    rows = []
    # When Phase 12 lands, walk rd/*/metrics.json and append.
    return pd.DataFrame(rows)


__all__ = ["SimSpec", "SimulationRunResult", "run_simulation", "load_all_runs"]
