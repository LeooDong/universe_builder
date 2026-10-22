"""Multi-provider LLM layer.

Providers transport requests; they do not know about atom types, filenames, or
the pipeline. Build an `LLMRequest` (see `providers.requests` for the pipeline-
aware builder), hand it to a provider, get an `LLMResponse` back.

    from universe_builder.providers import build_request, get_provider

    request = build_request("person", "Genghis Khan", model="claude-haiku-4-5-20251001")
    response = get_provider(request.model).complete(request)
    if response.ok:
        print(response.parsed.name.common)

For bulk work use the batch path, which is 50% cheaper on every provider that
offers it:

    handle = submit_batch_for("person", topics, model="gpt-5-nano")
    # ... hours later, in a different process ...
    collect(handle)
"""

from universe_builder.providers.base import Provider
from universe_builder.providers.errors import (
    BatchNotSupported,
    ProviderError,
    ProviderNotConfigured,
    ProviderNotInstalled,
    UnknownModel,
)
from universe_builder.providers.pricing import ModelInfo, cache_warning, estimate_cost, known_models, lookup
from universe_builder.providers.registry import (
    get_provider,
    get_provider_by_name,
    provider_names,
    schema_model_by_name,
)
from universe_builder.providers.types import (
    BatchHandle,
    BatchStatus,
    Example,
    LLMRequest,
    LLMResponse,
    Usage,
)

__all__ = [
    "BatchHandle",
    "BatchNotSupported",
    "BatchStatus",
    "Example",
    "LLMRequest",
    "LLMResponse",
    "ModelInfo",
    "Provider",
    "ProviderError",
    "ProviderNotConfigured",
    "ProviderNotInstalled",
    "UnknownModel",
    "Usage",
    "cache_warning",
    "estimate_cost",
    "get_provider",
    "get_provider_by_name",
    "known_models",
    "lookup",
    "provider_names",
    "schema_model_by_name",
]
