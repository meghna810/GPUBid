"""Prompt module — re-exports the v0.2 prompts (still in use) and adds Phase 8 scaffolding.

The v0.2 prompt API (``PROMPT_VERSION``, ``buyer_system_prompt``,
``seller_system_prompt``, ``buyer_tool_specs``, ``seller_tool_specs``) is
preserved by importing from ``gpubid.agents._v02_prompts``. The Phase 8
variant harness (``render_prompt``) is added below — it raises ``ConfigError``
when a variant template hasn't been authored yet.

To complete Phase 8:
1. Author the 8 variant .md files in this directory per spec §9.2.
2. Author ``few_shot_examples.json``.
3. Wire the sweep harness in a notebook cell.
"""

from __future__ import annotations

from pathlib import Path

from gpubid.errors import ConfigError

# Re-export the v0.2 prompt API. Existing callers (LLMBuyer, LLMSeller, tests)
# keep working unchanged.
from gpubid.agents._v02_prompts import (
    PROMPT_VERSION,
    buyer_system_prompt,
    buyer_tool_specs,
    seller_system_prompt,
    seller_tool_specs,
)


_PROMPT_DIR = Path(__file__).parent


def render_prompt(role: str, variant: str, ctx: dict) -> str:
    """Render the v0.3 variant prompt template for the given role+variant.

    Replaces ``{{key}}`` placeholders with values from ``ctx``. Raises
    ``ConfigError`` if the template file doesn't exist (variant not yet authored).

    Supported placeholders include: ``public_profile``, ``private_profile``,
    ``counterparty_public``, ``round_n``, ``recent_offers``, ``budget_remaining``.
    """
    if role not in ("buyer", "seller"):
        raise ValueError(f"role must be 'buyer' or 'seller', got {role!r}")

    path = _PROMPT_DIR / f"{role}_{variant}.md"
    if not path.exists():
        raise ConfigError(
            f"Prompt template not authored yet: {path}. Authoring the 8 variant "
            f"files (4 buyer + 4 seller) is part of Phase 8 of the v0.3 refactor."
        )

    template = path.read_text()
    rendered = template
    for key, value in ctx.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered


__all__ = [
    "PROMPT_VERSION",
    "buyer_system_prompt",
    "buyer_tool_specs",
    "seller_system_prompt",
    "seller_tool_specs",
    "render_prompt",
]
