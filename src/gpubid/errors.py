"""Exception taxonomy for GPUBid.

Catch broad ``Exception`` only at the outermost layer (notebook cell, CLI
entrypoint). Internal modules raise specific subclasses below so callers can
recover where it makes sense (e.g., retry on `ProviderRateLimitError`,
abort on `MissingAPIKeyError`).
"""

from __future__ import annotations


class GPUBidError(Exception):
    """Base for every error this package raises."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class ConfigError(GPUBidError):
    """Invalid or missing configuration discovered at import or settings load time."""


# ---------------------------------------------------------------------------
# Market generation
# ---------------------------------------------------------------------------


class MarketError(GPUBidError):
    """Base for market-construction failures."""


class InfeasibleMarketError(MarketError):
    """The requested market shape produces no compatible buyer/seller pair."""


class ProfileValidationError(MarketError):
    """An LLM-emitted public/private profile failed Pydantic validation after retries."""

    def __init__(self, message: str, last_raw_response: str | None = None) -> None:
        super().__init__(message)
        self.last_raw_response = last_raw_response


# ---------------------------------------------------------------------------
# Negotiation protocol
# ---------------------------------------------------------------------------


class ProtocolError(GPUBidError):
    """Base for negotiation-protocol failures."""


class NoEligibleSellersError(ProtocolError):
    """The eligibility filter rejected every seller for a given buyer broadcast."""


class DeadlockError(ProtocolError):
    """No-progress streak hit the configured cap without a deal closing."""


class BudgetExhaustedError(ProtocolError):
    """Per-run token budget exceeded before all pairs resolved."""


# ---------------------------------------------------------------------------
# LLM provider
# ---------------------------------------------------------------------------


class ProviderError(GPUBidError):
    """Base for LLM-provider failures."""


class MissingAPIKeyError(ProviderError):
    """No API key was supplied where one is required.

    v0.3 has no key-free mode for LLM-dependent paths. Raise this immediately,
    before doing any other work, so callers see a clear cause.
    """


class ProviderRateLimitError(ProviderError):
    """Provider returned a 429 / rate-limit signal. Caller may retry with backoff."""


class ProviderResponseError(ProviderError):
    """Provider returned a malformed or unparseable response after retries."""


# ---------------------------------------------------------------------------
# Human-in-the-loop
# ---------------------------------------------------------------------------


class HITLAbort(GPUBidError):
    """Raised when a human-in-the-loop surfacer asks for the run to stop."""


__all__ = [
    "GPUBidError",
    "ConfigError",
    "MarketError",
    "InfeasibleMarketError",
    "ProfileValidationError",
    "ProtocolError",
    "NoEligibleSellersError",
    "DeadlockError",
    "BudgetExhaustedError",
    "ProviderError",
    "MissingAPIKeyError",
    "ProviderRateLimitError",
    "ProviderResponseError",
    "HITLAbort",
]
