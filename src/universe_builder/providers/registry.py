"""Provider and schema lookup.

Providers are constructed lazily and memoised, so importing this module does not
require any SDK to be installed or any key to be set. You only pay for the
provider you actually use.
"""

from __future__ import annotations

import logging
from typing import Callable, Type

from pydantic import BaseModel

from universe_builder.providers.base import Provider
from universe_builder.providers.errors import ProviderError, UnknownModel
from universe_builder.providers.pricing import provider_for

log = logging.getLogger(__name__)


def _anthropic() -> Provider:
    from universe_builder.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider()


def _openai() -> Provider:
    from universe_builder.providers.openai_provider import OpenAIProvider

    return OpenAIProvider()


def _openai_responses() -> Provider:
    from universe_builder.providers.openai_provider import OpenAIResponsesProvider

    return OpenAIResponsesProvider()


def _openai_search() -> Provider:
    from universe_builder.providers.openai_provider import OpenAISearchProvider

    return OpenAISearchProvider()


def _openrouter() -> Provider:
    from universe_builder.providers.openai_provider import OpenRouterProvider

    return OpenRouterProvider()


def _deepseek() -> Provider:
    from universe_builder.providers.openai_provider import DeepSeekProvider

    return DeepSeekProvider()


def _qwen() -> Provider:
    from universe_builder.providers.openai_provider import QwenProvider

    return QwenProvider()


def _google() -> Provider:
    from universe_builder.providers.google_provider import GoogleProvider

    return GoogleProvider()


_FACTORIES: dict[str, Callable[[], Provider]] = {
    "anthropic": _anthropic,
    "openai": _openai,
    "openai-responses": _openai_responses,
    "openai-search": _openai_search,
    "openrouter": _openrouter,
    "google": _google,
    "deepseek": _deepseek,
    "qwen": _qwen,
}

_CACHE: dict[str, Provider] = {}


def get_provider_by_name(name: str) -> Provider:
    if name in _CACHE:
        return _CACHE[name]
    try:
        factory = _FACTORIES[name]
    except KeyError as e:
        raise ProviderError(
            f"Unknown provider {name!r}. Known: {', '.join(sorted(_FACTORIES))}"
        ) from e
    provider = factory()
    _CACHE[name] = provider
    return provider


def get_provider(model: str) -> Provider:
    """Provider that serves `model`. A `vendor/model` slug routes to OpenRouter."""
    return get_provider_by_name(provider_for(model))


def provider_names() -> list[str]:
    return sorted(_FACTORIES)


#: Provider name -> (module under `providers`, class): the class without its
#: client, for what a caller can ask before any key is read (v0.6.0: its
#: `EFFORT_LEVELS`, for `effort.refusal`). Kept beside `_FACTORIES`;
#: `tests/test_effort.py` holds the two to the same names.
_CLASSES: dict[str, tuple[str, str]] = {
    "anthropic": ("anthropic_provider", "AnthropicProvider"),
    "openai": ("openai_provider", "OpenAIProvider"),
    "openai-responses": ("openai_provider", "OpenAIResponsesProvider"),
    "openai-search": ("openai_provider", "OpenAISearchProvider"),
    "openrouter": ("openai_provider", "OpenRouterProvider"),
    "google": ("google_provider", "GoogleProvider"),
    "deepseek": ("openai_provider", "DeepSeekProvider"),
    "qwen": ("openai_provider", "QwenProvider"),
}


def provider_class(name: str) -> type:
    """The provider class registered under `name`, not constructed (importing
    a provider module needs no SDK and no key; its `__init__` does)."""
    from importlib import import_module

    try:
        module, cls = _CLASSES[name]
    except KeyError as e:
        raise ProviderError(f"Unknown provider {name!r}. Known: {', '.join(sorted(_CLASSES))}") from e
    return getattr(import_module(f"universe_builder.providers.{module}"), cls)


# -- schema models --------------------------------------------------------------
#
# Batch results arrive long after the process that submitted them has exited, so
# the handle stores the schema class *name* and we resolve it back here.


def _schema_models() -> dict[str, Type[BaseModel]]:
    """Every schema a batch reply might need resolved by name, derived from the
    active pack: every stored/seed class (`class_mapping`), plus each atom's
    expander entry model (`Population` — a sub-object, not a stored atom, so it
    is absent from `class_mapping` but a batched expand reply validates against
    it), plus any declared non-atom `response_models`.
    """
    from universe_builder import atoms
    from universe_builder.registry import active_pack

    pack = active_pack()
    models: dict[str, Type[BaseModel]] = dict(atoms.class_mapping)
    for spec in pack.all_specs():
        if spec.expand:
            models[spec.expand.entry_model.__name__] = spec.expand.entry_model
    for model in pack.response_models:
        models[model.__name__] = model
    return models


def schema_model_by_name(name: str | None) -> Type[BaseModel]:
    if not name:
        raise UnknownModel("Batch handle has no schema_model recorded")
    models = _schema_models()
    try:
        return models[name]
    except KeyError as e:
        raise UnknownModel(
            f"Unknown schema model {name!r}. Known: {', '.join(sorted(models))}"
        ) from e
