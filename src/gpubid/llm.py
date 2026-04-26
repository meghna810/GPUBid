"""Provider-agnostic LLM client.

The user pastes either an Anthropic key (`sk-ant-...`) or an OpenAI key
(`sk-...`); `make_client` picks the right backend and returns a uniform
`LLMClient` interface. Agent code never branches on provider.

Both providers converge on the same conceptual flow (system prompt, messages,
tool calls). The two adapters translate to and from a single internal `ToolCall`
shape — the agent stays provider-neutral.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Protocol


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------


def detect_provider(api_key: str) -> str:
    """Return 'anthropic' or 'openai' based on the key prefix.

    Raises ProviderUnknownError on unrecognized keys.
    """
    if not api_key or not isinstance(api_key, str):
        raise ProviderUnknownError("API key is empty")
    key = api_key.strip()
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("sk-"):
        return "openai"
    raise ProviderUnknownError(
        f"Unrecognized key format. Expected 'sk-ant-...' (Anthropic) "
        f"or 'sk-...' (OpenAI), got prefix {key[:8]!r}."
    )


class ProviderUnknownError(ValueError):
    """Raised when a provided API key doesn't match any known provider."""


# ---------------------------------------------------------------------------
# Provider-neutral types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """A normalized tool-call result that both providers map to."""

    tool_name: str
    arguments: dict[str, Any]
    raw_text: str = ""           # the assistant's text reply (often empty when tool-using)


@dataclass(frozen=True)
class ToolSpec:
    """Provider-neutral tool description."""

    name: str
    description: str
    parameters: dict[str, Any]   # JSON-schema dict


class LLMClient(Protocol):
    provider: str

    def generate(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        tools: list[ToolSpec],
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ToolCall: ...


# ---------------------------------------------------------------------------
# Default model picks (cost-conscious)
# ---------------------------------------------------------------------------

DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------


class AnthropicClient:
    provider = "anthropic"

    def __init__(self, api_key: str, model: str = DEFAULT_ANTHROPIC_MODEL):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic package not installed. `pip install anthropic`."
            ) from e
        self._anthropic = anthropic
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        tools: list[ToolSpec],
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ToolCall:
        anthropic_tools = [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            tools=anthropic_tools,
            messages=messages,
        )
        # Find the first tool_use block; fall back to text.
        text_chunks: list[str] = []
        for block in resp.content:
            if getattr(block, "type", "") == "tool_use":
                return ToolCall(
                    tool_name=block.name,
                    arguments=dict(block.input),
                    raw_text="".join(text_chunks),
                )
            if getattr(block, "type", "") == "text":
                text_chunks.append(block.text)
        return ToolCall(tool_name="__no_tool__", arguments={}, raw_text="".join(text_chunks))


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------


class OpenAIClient:
    provider = "openai"

    def __init__(self, api_key: str, model: str = DEFAULT_OPENAI_MODEL):
        try:
            import openai
        except ImportError as e:
            raise RuntimeError("openai package not installed. `pip install openai`.") from e
        self._openai = openai
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model

    def generate(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        tools: list[ToolSpec],
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ToolCall:
        openai_tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]
        all_messages = [{"role": "system", "content": system_prompt}] + messages
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=all_messages,
            tools=openai_tools,
            tool_choice="required",
            max_tokens=max_tokens,
            temperature=temperature,
        )
        choice = resp.choices[0]
        msg = choice.message
        if msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            return ToolCall(
                tool_name=tc.function.name,
                arguments=args,
                raw_text=msg.content or "",
            )
        return ToolCall(tool_name="__no_tool__", arguments={}, raw_text=msg.content or "")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_client(api_key: str, model: Optional[str] = None) -> LLMClient:
    """Return the right `LLMClient` for `api_key`.

    `model` overrides the default for whichever provider was detected. If unset,
    each provider uses the cost-effective default (Haiku for Anthropic,
    `gpt-4o-mini` for OpenAI).
    """
    provider = detect_provider(api_key)
    if provider == "anthropic":
        return AnthropicClient(api_key=api_key, model=model or DEFAULT_ANTHROPIC_MODEL)
    if provider == "openai":
        return OpenAIClient(api_key=api_key, model=model or DEFAULT_OPENAI_MODEL)
    raise ProviderUnknownError(provider)


def get_api_key_from_env() -> Optional[str]:
    """Convenience for notebooks: read either ANTHROPIC_API_KEY or OPENAI_API_KEY."""
    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(env)
        if v:
            return v
    return None


__all__ = [
    "ToolCall",
    "ToolSpec",
    "LLMClient",
    "AnthropicClient",
    "OpenAIClient",
    "ProviderUnknownError",
    "detect_provider",
    "make_client",
    "get_api_key_from_env",
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_OPENAI_MODEL",
]
