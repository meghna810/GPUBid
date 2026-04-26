"""Bake LLM negotiation traces into JSON preset files.

Run once with the team's API keys (Anthropic and/or OpenAI). The resulting
JSONs ship with the repo so the notebook's preset mode plays back without any
LLM calls — keeping the demo instant and free for visitors.

Usage (from repo root):
    ANTHROPIC_API_KEY=sk-ant-... python -m gpubid.experiments.bake_presets all
    OPENAI_API_KEY=sk-... python -m gpubid.experiments.bake_presets headline_tight
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gpubid.engine.round_runner import make_llm_agents, run_rounds
from gpubid.llm import make_client
from gpubid.market import generate_market
from gpubid.schema import Deal, Market, Offer


@dataclass(frozen=True)
class PresetSpec:
    """Definition of a preset scenario to bake."""

    scenario_id: str
    description: str
    n_buyers: int
    n_sellers: int
    regime: str
    seed: int
    max_rounds: int = 5
    concentration_cap_pct: Optional[float] = None
    # If set, use a different provider for sellers — used for the collusion experiment.
    seller_api_env: Optional[str] = None


# Six headline scenarios used by the video walkthrough and the live demo dropdown.
PRESET_SPECS: list[PresetSpec] = [
    PresetSpec("tight_h100_rush", "Tight H100 supply, urgent buyers compete", 8, 4, "tight", seed=42),
    PresetSpec("slack_offpeak",   "Slack supply, buyers fill off-peak slots", 8, 4, "slack", seed=42),
    PresetSpec("homogeneous_collusion", "All-Claude sellers (same prompts) — watch prices drift up",
               6, 4, "tight", seed=7),
    PresetSpec("heterogeneous_mix", "Claude buyers vs OpenAI sellers — symmetry broken",
               6, 4, "tight", seed=7, seller_api_env="OPENAI_API_KEY"),
    PresetSpec("fairness_cap_bites", "Concentration cap blocks one big buyer",
               5, 3, "tight", seed=11, concentration_cap_pct=0.20),
    PresetSpec("free_form_breakdown", "Reduced-structure mode (showcased in M5 toggle)",
               6, 3, "tight", seed=13),
]


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def serialize_offer(o: Offer) -> dict:
    return o.model_dump(mode="json")


def serialize_deal(d: Deal) -> dict:
    return d.model_dump(mode="json")


def serialize_snapshot(snap) -> dict:
    return {
        "round_n": snap.round_n,
        "max_rounds": snap.max_rounds,
        "asks": [serialize_offer(o) for o in snap.asks],
        "bids": [serialize_offer(o) for o in snap.bids],
        "new_deals": [serialize_deal(d) for d in snap.new_deals],
        "all_deals": [serialize_deal(d) for d in snap.all_deals],
        "active_buyer_ids": list(snap.active_buyer_ids),
        "active_seller_ids": list(snap.active_seller_ids),
        "is_final": snap.is_final,
    }


def deserialize_snapshot(data: dict):
    """Build a RoundSnapshot back from JSON."""
    from gpubid.engine.board import RoundSnapshot
    return RoundSnapshot(
        round_n=data["round_n"],
        max_rounds=data["max_rounds"],
        asks=tuple(Offer(**o) for o in data["asks"]),
        bids=tuple(Offer(**o) for o in data["bids"]),
        new_deals=tuple(Deal(**d) for d in data["new_deals"]),
        all_deals=tuple(Deal(**d) for d in data["all_deals"]),
        active_buyer_ids=tuple(data["active_buyer_ids"]),
        active_seller_ids=tuple(data["active_seller_ids"]),
        is_final=data["is_final"],
    )


def save_preset(
    *,
    spec: PresetSpec,
    market: Market,
    snapshots: list,
    metadata: dict,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{spec.scenario_id}.json"
    payload = {
        "scenario_id": spec.scenario_id,
        "description": spec.description,
        "spec": {
            "n_buyers": spec.n_buyers,
            "n_sellers": spec.n_sellers,
            "regime": spec.regime,
            "seed": spec.seed,
            "max_rounds": spec.max_rounds,
            "concentration_cap_pct": spec.concentration_cap_pct,
        },
        "market": market.model_dump(mode="json"),
        "snapshots": [serialize_snapshot(s) for s in snapshots],
        "metadata": metadata,
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_preset(path: Path) -> dict:
    """Load a preset JSON. Returns dict with `market` (Market) and `snapshots` (list[RoundSnapshot])."""
    raw = json.loads(Path(path).read_text())
    market = Market(**raw["market"])
    snapshots = [deserialize_snapshot(s) for s in raw["snapshots"]]
    return {
        "scenario_id": raw["scenario_id"],
        "description": raw.get("description", ""),
        "spec": raw["spec"],
        "market": market,
        "snapshots": snapshots,
        "metadata": raw.get("metadata", {}),
    }


def list_presets(presets_dir: Path = Path("data/presets")) -> list[Path]:
    if not presets_dir.exists():
        return []
    return sorted(presets_dir.glob("*.json"))


# ---------------------------------------------------------------------------
# Bake
# ---------------------------------------------------------------------------


def bake_one(spec: PresetSpec, output_dir: Path = Path("data/presets")) -> Path:
    """Generate one preset by running an LLM negotiation. Costs API credits."""
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if api_key is None:
        raise RuntimeError(
            "Set ANTHROPIC_API_KEY or OPENAI_API_KEY to bake presets. "
            "(Live LLM calls cost money; presets are baked once and shipped with the repo.)"
        )

    seller_api_key = None
    if spec.seller_api_env:
        seller_api_key = os.environ.get(spec.seller_api_env)
        if seller_api_key is None:
            raise RuntimeError(
                f"Preset {spec.scenario_id} requires {spec.seller_api_env} for the seller-side."
            )

    market = generate_market(spec.n_buyers, spec.n_sellers, spec.regime, seed=spec.seed)  # type: ignore[arg-type]
    buyer_agents, seller_agents = make_llm_agents(
        market,
        api_key=api_key,
        seller_api_key=seller_api_key,
    )

    snapshots = list(
        run_rounds(
            market,
            buyer_agents,
            seller_agents,
            max_rounds=spec.max_rounds,
            concentration_cap_pct=spec.concentration_cap_pct,
        )
    )

    # Best-effort metadata so we can audit what model produced the trace.
    buyer_client = next(iter(buyer_agents.values())).client  # type: ignore[union-attr]
    seller_client = next(iter(seller_agents.values())).client  # type: ignore[union-attr]
    metadata = {
        "buyer_provider": buyer_client.provider,
        "buyer_model": getattr(buyer_client, "model", "?"),
        "seller_provider": seller_client.provider,
        "seller_model": getattr(seller_client, "model", "?"),
        "prompt_version": __import__("gpubid.agents.prompts", fromlist=["PROMPT_VERSION"]).PROMPT_VERSION,
    }
    return save_preset(spec=spec, market=market, snapshots=snapshots, metadata=metadata, output_dir=output_dir)


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python -m gpubid.experiments.bake_presets <scenario_id|all>")
        print("Available scenarios:")
        for spec in PRESET_SPECS:
            print(f"  - {spec.scenario_id}: {spec.description}")
        return 1

    target = argv[1]
    specs_to_bake = PRESET_SPECS if target == "all" else [s for s in PRESET_SPECS if s.scenario_id == target]
    if not specs_to_bake:
        print(f"Unknown scenario: {target}")
        return 1

    for spec in specs_to_bake:
        print(f"Baking {spec.scenario_id}…", flush=True)
        try:
            path = bake_one(spec)
            print(f"  ✓ {path}")
        except Exception as e:
            print(f"  ✗ {e}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))


__all__ = [
    "PresetSpec",
    "PRESET_SPECS",
    "bake_one",
    "save_preset",
    "load_preset",
    "list_presets",
    "deserialize_snapshot",
]
