"""Phase 11 — Cross-run, cross-model comparator. SCAFFOLDED.

Per spec §12.2, computes per-(buyer_provider, seller_provider, prompt_variant) cell:
  - mean_buyer_surplus_pct, mean_seller_surplus_pct
  - deal_close_rate, walk_away_rate, mean_rounds_to_close
  - volume_discount_invocation_rate
  - concession_first_rate, aggression_index
  - guardrail_violation_count, hitl_intervention_count

Renders 2D heatmap, walk-away bar chart, Sankey, and a templated headline text
block. The forensics module covers per-RUN insights; the comparator covers
cross-run trends across the simulation harness output.

SCAFFOLDED — needs Phase 12 (sim_v2.py) outputs to consume. Stubs the
metric computation API; visualization is a thin wrapper once the metrics exist.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComparatorCell:
    buyer_provider: str
    seller_provider: str
    prompt_variant: str
    mean_buyer_surplus_pct: float = 0.0
    mean_seller_surplus_pct: float = 0.0
    deal_close_rate: float = 0.0
    walk_away_rate: float = 0.0
    mean_rounds_to_close: float = 0.0
    volume_discount_invocation_rate: float = 0.0
    concession_first_rate: float = 0.0
    aggression_index: float = 0.0
    guardrail_violation_count: int = 0
    hitl_intervention_count: int = 0


def compute_cells_from_runs(runs_df) -> list[ComparatorCell]:
    """Aggregate metrics across runs into per-cell ComparatorCells.

    SCAFFOLDED. Awaits Phase 12 sim_v2 output schema (parquet rows of
    pair-round-action) to be defined and populated.
    """
    raise NotImplementedError(
        "Phase 11 comparator is scaffolded. Implement after Phase 12 (sim_v2) "
        "produces the runs_df schema."
    )


def render_headline_text(cells: list[ComparatorCell]) -> str:
    """Programmatic headline finding — no LLM, deterministic from metrics.

    SCAFFOLDED.
    """
    raise NotImplementedError("Phase 11 comparator render_headline_text scaffolded.")


__all__ = ["ComparatorCell", "compute_cells_from_runs", "render_headline_text"]
