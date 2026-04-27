"""Phase 3 tests for BuyerAgent. Live tests skip without API keys.

To record fixtures:

    ANTHROPIC_API_KEY=sk-ant-... pytest tests/test_buyer_agent.py --live

That writes responses to ``tests/fixtures/responses/``. CI replays them via
``RecordedFixtureClient``. Without --live and without fixtures recorded,
fixture-based tests skip with a clear message.
"""

import os

import pytest

from gpubid.agents.buyer_agent import BuyerAgent
from gpubid.errors import MissingAPIKeyError


def test_buyer_agent_rejects_none_client():
    """Per spec §4.3: there is no deterministic fallback."""
    with pytest.raises(MissingAPIKeyError, match="real LLM client"):
        BuyerAgent(llm_client=None)


@pytest.mark.skipif(
    not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")),
    reason="live LLM test — set ANTHROPIC_API_KEY or OPENAI_API_KEY to run",
)
def test_buyer_agent_translate_against_live_llm():
    """Live test: actually call the LLM and validate the returned profiles.

    This is the path that records fixtures. Once fixtures exist under
    tests/fixtures/responses/, a corresponding fixture-replay test should be
    added that runs without --live.
    """
    import numpy as np
    from gpubid.domain.requirements import REQUIREMENT_LIBRARY
    from gpubid.llm import make_client

    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    client = make_client(api_key)
    agent = BuyerAgent(llm_client=client)

    rng = np.random.default_rng(42)
    requirement = REQUIREMENT_LIBRARY[0]
    pub, priv = agent.translate(requirement, rng)

    # Schema-valid is enforced by Pydantic; sanity-check semantic alignment:
    assert priv.business_context_summary == requirement.raw_text
    assert pub.qty_gpus >= 1
    assert pub.urgency_band in ("routine", "soon", "urgent")


@pytest.mark.skip(reason="awaits fixture recording — run --live test once with API key, commit fixtures")
def test_buyer_agent_translate_against_recorded_fixture():
    """Replay-based test that runs in CI without keys.

    Fixture file: tests/fixtures/responses/buyer_translate_req_series_a_robotics_001__000.json
    """
    pass
