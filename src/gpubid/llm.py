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
    """Return 'anthropic', 'openai', or 'gemini' based on the key prefix.

    Raises ProviderUnknownError on unrecognized keys.
    """
    if not api_key or not isinstance(api_key, str):
        raise ProviderUnknownError("API key is empty")
    key = api_key.strip()
    if key.startswith("sk-ant-"):
        return "anthropic"
    if key.startswith("sk-"):
        return "openai"
    # Google AI Studio keys begin with "AIza" (40 chars total).
    if key.startswith("AIza"):
        return "gemini"
    raise ProviderUnknownError(
        f"Unrecognized key format. Expected 'sk-ant-...' (Anthropic), "
        f"'sk-...' (OpenAI), or 'AIza...' (Gemini), got prefix {key[:8]!r}."
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
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


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
# Gemini adapter
# ---------------------------------------------------------------------------


class GeminiClient:
    """Gemini adapter using the google-genai SDK (the 2024+ Google AI SDK).

    Tool-call protocol uses Gemini's `function_declarations` + `function_call`
    response shape, which we translate to the same internal `ToolCall`. The
    response part with a `function_call` (if any) wins; otherwise falls back
    to text and emits the sentinel `__no_tool__` tool name (matching how the
    Anthropic / OpenAI adapters behave when the model fails to emit a tool).
    """

    provider = "gemini"

    def __init__(self, api_key: str, model: str = DEFAULT_GEMINI_MODEL):
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError as e:
            raise RuntimeError(
                "google-genai package not installed. `pip install google-genai`."
            ) from e
        self._genai = genai
        self._types = genai_types
        self.client = genai.Client(api_key=api_key)
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
        types = self._types

        # Translate ToolSpec → google.genai function_declarations. Gemini's
        # JSONSchema dialect doesn't accept `additionalProperties` and a few
        # other JSONSchema-Draft-7 fields; strip them from each parameter
        # subtree before sending.
        function_declarations = [
            {
                "name": t.name,
                "description": t.description,
                "parameters": _sanitize_for_gemini(t.parameters),
            }
            for t in tools
        ]
        gemini_tools = [types.Tool(function_declarations=function_declarations)]

        # Translate `messages` (OpenAI-style list of role/content) into Gemini
        # `contents`. Gemini uses 'user' and 'model' as roles (no 'assistant').
        contents = []
        for m in messages:
            role = m.get("role", "user")
            text = m.get("content", "") or ""
            gemini_role = "model" if role == "assistant" else "user"
            contents.append(types.Content(
                role=gemini_role,
                parts=[types.Part.from_text(text=text)],
            ))

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=gemini_tools,
            temperature=temperature,
            max_output_tokens=max_tokens,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(mode="ANY"),
            ),
        )

        resp = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )

        text_chunks: list[str] = []
        for cand in (resp.candidates or []):
            content = getattr(cand, "content", None)
            if content is None:
                continue
            for part in (content.parts or []):
                fc = getattr(part, "function_call", None)
                if fc is not None and getattr(fc, "name", None):
                    args = dict(fc.args) if getattr(fc, "args", None) else {}
                    return ToolCall(
                        tool_name=fc.name,
                        arguments=args,
                        raw_text="".join(text_chunks),
                    )
                text = getattr(part, "text", None)
                if text:
                    text_chunks.append(text)
        return ToolCall(tool_name="__no_tool__", arguments={}, raw_text="".join(text_chunks))


def _sanitize_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip JSONSchema fields Gemini doesn't accept, recursively.

    Gemini rejects `additionalProperties`, `$schema`, `exclusiveMinimum` (in
    some forms), and a few others. Strip them rather than wrestling with the
    SDK's strict validator. Mutates a copy, not the input.
    """
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    drop = {"additionalProperties", "$schema", "$id", "$ref"}
    for k, v in schema.items():
        if k in drop:
            continue
        if k == "properties" and isinstance(v, dict):
            out[k] = {prop_name: _sanitize_for_gemini(prop_schema)
                      for prop_name, prop_schema in v.items()}
        elif k == "items" and isinstance(v, dict):
            out[k] = _sanitize_for_gemini(v)
        elif isinstance(v, dict):
            out[k] = _sanitize_for_gemini(v)
        elif isinstance(v, list):
            out[k] = [_sanitize_for_gemini(x) if isinstance(x, dict) else x for x in v]
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_client(api_key: str, model: Optional[str] = None) -> LLMClient:
    """Return the right `LLMClient` for `api_key`.

    `model` overrides the default for whichever provider was detected. If unset,
    each provider uses the cost-effective default (Haiku for Anthropic,
    `gpt-4o-mini` for OpenAI, `gemini-2.5-flash` for Gemini).
    """
    provider = detect_provider(api_key)
    if provider == "anthropic":
        return AnthropicClient(api_key=api_key, model=model or DEFAULT_ANTHROPIC_MODEL)
    if provider == "openai":
        return OpenAIClient(api_key=api_key, model=model or DEFAULT_OPENAI_MODEL)
    if provider == "gemini":
        return GeminiClient(api_key=api_key, model=model or DEFAULT_GEMINI_MODEL)
    raise ProviderUnknownError(provider)


def get_api_key_from_env() -> Optional[str]:
    """Convenience for notebooks: read ANTHROPIC_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY."""
    for env in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
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
    "GeminiClient",
    "ProviderUnknownError",
    "detect_provider",
    "make_client",
    "get_api_key_from_env",
    "DEFAULT_ANTHROPIC_MODEL",
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_GEMINI_MODEL",
]
