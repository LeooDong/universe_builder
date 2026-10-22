"""Live credential and round-trip check. Deselected by default.

Every other test in this suite is mocked. This one is the exception: it spends
real money to answer one question the mocks cannot — do the configured keys
still work, and does a request survive the whole path from `LLMRequest` through
the provider to a validated Pydantic instance?

    pytest tests/test_live_smoke.py -m integration

Cost is a fraction of a cent per run on `gpt-5-nano`. The budget is deliberately
generous: `gpt-5-nano` is a reasoning model and spends output tokens on
reasoning before it emits any content, so a tight limit returns an empty string
rather than a short answer. See `tests/test_truncation.py`.
"""

from __future__ import annotations

import os

import pytest
from pydantic import BaseModel, Field

from universe_builder.settings import load_environment
from universe_builder.providers.registry import get_provider_by_name
from universe_builder.providers.types import LLMRequest

pytestmark = pytest.mark.integration

# A reasoning model needs room for reasoning tokens before it writes anything.
SMOKE_MAX_TOKENS = 2000


class SmokeAnswer(BaseModel):
    """Two fields of different types, so validation is doing real work."""

    city: str = Field(description="The capital city.")
    founded_century: int = Field(description="Century founded, negative for BCE.")


def require_key(name: str) -> None:
    load_environment()
    if not os.environ.get(name):
        pytest.skip(f"{name} is not set")


def test_openai_key_and_round_trip():
    require_key("OPENAI_API_KEY")

    provider = get_provider_by_name("openai")
    response = provider.complete(
        LLMRequest(
            model="gpt-5-nano",
            schema_model=SmokeAnswer,
            system="Answer with the capital of the country given.",
            user="France",
            max_tokens=SMOKE_MAX_TOKENS,
            cache=False,
        )
    )

    assert response.ok, f"live call failed: {response.error}\nraw: {response.raw_text[:400]}"
    assert response.parsed.city == "Paris"
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


def test_anthropic_key_and_round_trip():
    require_key("ANTHROPIC_API_KEY")

    provider = get_provider_by_name("anthropic")
    response = provider.complete(
        LLMRequest(
            model="claude-sonnet-5",
            schema_model=SmokeAnswer,
            system="Answer with the capital of the country given.",
            user="France",
            max_tokens=SMOKE_MAX_TOKENS,
            cache=False,
        )
    )

    assert response.ok, f"live call failed: {response.error}\nraw: {response.raw_text[:400]}"
    assert response.parsed.city == "Paris"
