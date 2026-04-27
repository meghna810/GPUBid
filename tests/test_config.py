"""Phase 1 smoke tests: settings load, env var overrides, error taxonomy importable."""

import importlib

import pytest


def test_settings_loads_with_defaults():
    from gpubid.config import settings
    assert settings.market.n_buyers_default == 8
    assert settings.market.regime_default == "tight"
    assert settings.negotiation.default_max_rounds == 5
    assert settings.negotiation.max_rounds_hard_cap == 10
    assert settings.provider.anthropic_default_model.startswith("claude-")
    assert settings.provider.openai_default_model == "gpt-4o-mini"
    assert "vanilla" in settings.prompts.available_variants
    assert settings.hitl.enabled_default is True


def test_env_var_override_via_double_underscore(monkeypatch):
    """GPUBID_NEGOTIATION__DEFAULT_MAX_ROUNDS=7 should propagate."""
    monkeypatch.setenv("GPUBID_NEGOTIATION__DEFAULT_MAX_ROUNDS", "7")

    # Re-import the config module so Settings re-reads env.
    import gpubid.config as cfg
    importlib.reload(cfg)
    assert cfg.settings.negotiation.default_max_rounds == 7


def test_env_var_override_returns_to_default_after(monkeypatch):
    """When the env var is removed, defaults restore on reimport."""
    import gpubid.config as cfg
    monkeypatch.delenv("GPUBID_NEGOTIATION__DEFAULT_MAX_ROUNDS", raising=False)
    importlib.reload(cfg)
    assert cfg.settings.negotiation.default_max_rounds == 5


def test_error_hierarchy_is_importable():
    from gpubid.errors import (
        BudgetExhaustedError,
        ConfigError,
        DeadlockError,
        GPUBidError,
        HITLAbort,
        InfeasibleMarketError,
        MarketError,
        MissingAPIKeyError,
        NoEligibleSellersError,
        ProfileValidationError,
        ProtocolError,
        ProviderError,
        ProviderRateLimitError,
        ProviderResponseError,
    )
    assert issubclass(MissingAPIKeyError, ProviderError)
    assert issubclass(ProviderError, GPUBidError)
    assert issubclass(NoEligibleSellersError, ProtocolError)
    assert issubclass(InfeasibleMarketError, MarketError)
    assert issubclass(MarketError, GPUBidError)
    assert issubclass(HITLAbort, GPUBidError)
    assert issubclass(DeadlockError, ProtocolError)
    assert issubclass(BudgetExhaustedError, ProtocolError)
    assert issubclass(ConfigError, GPUBidError)
    assert issubclass(ProfileValidationError, MarketError)
    assert issubclass(ProviderRateLimitError, ProviderError)
    assert issubclass(ProviderResponseError, ProviderError)


def test_profile_validation_error_carries_raw_response():
    from gpubid.errors import ProfileValidationError
    err = ProfileValidationError("bad json", last_raw_response="{not json}")
    assert err.last_raw_response == "{not json}"
    assert "bad json" in str(err)


def test_missing_api_key_message_is_actionable():
    from gpubid.errors import MissingAPIKeyError
    with pytest.raises(MissingAPIKeyError, match="ANTHROPIC_API_KEY"):
        raise MissingAPIKeyError("ANTHROPIC_API_KEY not set; v0.3 requires real LLM access")


def test_version_is_v030():
    import gpubid
    assert gpubid.__version__ == "0.3.0"
