"""Replay-only LLMClient for tests.

Reads canned responses from ``tests/fixtures/responses/<test_name>__<call_index>.json``
keyed identically to the production cache. Lives under ``tests/`` (not in the
shipped package) per spec §16.2 — there is NO deterministic client in src/.

To record fixtures (run-once with API keys):

    pytest tests/test_buyer_agent.py --live

That will exercise the real LLMClient and write each response to a
fixture file. CI then replays them via this client.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from gpubid.llm import ToolCall, ToolSpec


_FIXTURE_DIR = Path(__file__).parent / "responses"


@dataclass
class RecordedFixtureClient:
    provider: str = "fixture"
    model: str = "fixture"
    test_name: str = "default"
    _call_index: int = field(default=0, init=False)

    def generate(
        self,
        *,
        system_prompt: str,
        messages: list[dict[str, str]],
        tools: list[ToolSpec],
        max_tokens: int = 1024,
        temperature: float = 0.5,
    ) -> ToolCall:
        path = _FIXTURE_DIR / f"{self.test_name}__{self._call_index:03d}.json"
        self._call_index += 1
        if not path.exists():
            raise FileNotFoundError(
                f"Fixture missing: {path}. Re-record with `pytest <test> --live` "
                "and an API key set."
            )
        data = json.loads(path.read_text())
        return ToolCall(
            tool_name=data.get("tool_name", "__no_tool__"),
            arguments=data.get("arguments", {}),
            raw_text=data.get("raw_text", ""),
        )

    @staticmethod
    def hash_key(provider: str, model: str, system: str, messages: list, temperature: float) -> str:
        """Stable key matching the production cache. Used by the recorder."""
        blob = json.dumps(
            {"provider": provider, "model": model, "system": system,
             "messages": messages, "temperature": temperature},
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


__all__ = ["RecordedFixtureClient"]
