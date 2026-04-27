"""Centralized configuration for GPUBid v0.3.

A single ``Settings`` object exposes every tunable in the package. Callers
import the module-level singleton::

    from gpubid.config import settings
    rounds = settings.negotiation.default_max_rounds

Environment variables override defaults when prefixed with ``GPUBID_`` and
nested with double underscores, e.g.::

    GPUBID_NEGOTIATION__DEFAULT_MAX_ROUNDS=7

This module has no side effects beyond instantiating ``settings``. Importing
it must not perform IO, network calls, or directory creation.

Important: do NOT hard-code numbers like ``5`` or ``0.20`` in business logic.
If a value belongs in policy, put it here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class MarketConfig(BaseModel):
    n_buyers_default: int = 8
    n_sellers_default: int = 4
    regime_default: Literal["tight", "slack"] = "tight"
    seed_default: int = 42
    concentration_cap_pct_default: float = 0.20


class NegotiationConfig(BaseModel):
    max_rounds_hard_cap: int = 10
    default_max_rounds: int = 5
    no_progress_rounds_threshold: int = 2  # see Phase 6 budget policy
    per_round_token_budget: int = 1500     # per agent per round
    total_run_token_budget: int = 60_000


class ProviderConfig(BaseModel):
    anthropic_default_model: str = "claude-3-5-sonnet-latest"
    openai_default_model: str = "gpt-4o-mini"
    request_timeout_s: float = 30.0
    max_retries: int = 3
    backoff_base_s: float = 1.5


class SimulationConfig(BaseModel):
    default_n_seeds: int = 10
    cache_dir: Path = Path("data/llm_cache")
    runs_dir: Path = Path("data/runs")
    figures_dir: Path = Path("data/figures")


class PromptConfig(BaseModel):
    available_variants: list[str] = [
        "vanilla",
        "few_shot",
        "guardrails_repeated",
        "chain_of_thought",
    ]
    default_variant: str = "vanilla"


class HITLConfig(BaseModel):
    enabled_default: bool = True
    triggers: list[str] = [
        "constraint_violation_imminent",
        "deadlock",
        "low_confidence_close",
        "ambiguous_requirement",
    ]


class Settings(BaseSettings):
    market: MarketConfig = MarketConfig()
    negotiation: NegotiationConfig = NegotiationConfig()
    provider: ProviderConfig = ProviderConfig()
    simulation: SimulationConfig = SimulationConfig()
    prompts: PromptConfig = PromptConfig()
    hitl: HITLConfig = HITLConfig()

    model_config = SettingsConfigDict(env_prefix="GPUBID_", env_nested_delimiter="__")


# Module-level singleton. Other code imports `from gpubid.config import settings`.
settings = Settings()


__all__ = [
    "settings",
    "Settings",
    "MarketConfig",
    "NegotiationConfig",
    "ProviderConfig",
    "SimulationConfig",
    "PromptConfig",
    "HITLConfig",
]
