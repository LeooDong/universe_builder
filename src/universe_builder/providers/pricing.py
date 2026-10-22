"""Model registry and cost estimation.

The pricing table lives in `model_pricing.json` next to this module and is loaded
through `universe_builder.settings`, so it resolves regardless of the working
directory. Prices go stale; treat estimates as indicative and re-check the table
before making a decision based on it.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

from universe_builder import settings
from universe_builder.providers.errors import UnknownModel
from universe_builder.providers.types import Usage

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    model: str
    provider: str
    input: float
    cached_input: float
    cache_write: Optional[float]
    output: float
    batch_discount: float
    cache_min_tokens: int
    context: int
    #: The model's output cap in tokens (`max_tokens` above it is a 400, not a
    #: truncation). Recorded for the Anthropic rows, from the Models API; None
    #: where unrecorded, which means "not checked here".
    max_output: Optional[int] = None
    #: False when the model takes no effort setting (v0.6.0): not a reasoning
    #: model (gpt-4o-mini, gpt-4.1), or one the API refuses `effort` on (Haiku
    #: 4.5). None, the usual case, leaves it to the provider (`providers.effort`).
    effort: Optional[bool] = None
    #: The effort levels this model takes, where they differ from its
    #: provider's `EFFORT_LEVELS` (v0.6.0): e.g. an OpenAI model that takes
    #: `xhigh`. None: the provider's set.
    effort_levels: Optional[tuple] = None
    #: True when the model takes Anthropic's task budget (v0.6.1,
    #: `providers.task_budget`); None or False: it takes none, and a budget
    #: sent to it is refused before the call.
    task_budget: Optional[bool] = None


def _load() -> dict[str, ModelInfo]:
    with open(settings.MODEL_PRICING, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        name: ModelInfo(model=name, **spec)
        for name, spec in data["models"].items()
    }


_MODELS: dict[str, ModelInfo] = _load()


def lookup(model: str) -> ModelInfo:
    """Registry entry for a model ID."""
    try:
        return _MODELS[model]
    except KeyError as e:
        raise UnknownModel(
            f"{model!r} is not in the pricing table. "
            f"Add it to {settings.MODEL_PRICING.name} or pick one of: {', '.join(sorted(_MODELS))}"
        ) from e


def provider_for(model: str) -> str:
    """Which provider serves this model. OpenRouter slugs (`vendor/model`) route to openrouter."""
    if "/" in model:
        return "openrouter"
    return lookup(model).provider


def known_models() -> list[str]:
    return sorted(_MODELS)


def max_output(model: str) -> Optional[int]:
    """The model's output cap, or None when the table does not record one."""
    try:
        return lookup(model).max_output
    except UnknownModel:
        return None


def estimate_cost(usage: Usage, model: str, batch: bool = False) -> Optional[float]:
    """Estimated USD for one call. Returns None for models missing from the table."""
    try:
        info = lookup(model)
    except UnknownModel:
        return None

    discount = info.batch_discount if batch else 1.0
    write_rate = info.cache_write if info.cache_write is not None else info.input
    total = (
        usage.input_tokens * info.input
        + usage.cache_read_tokens * info.cached_input
        + usage.cache_write_tokens * write_rate
        + usage.output_tokens * info.output
    )
    return total / 1_000_000 * discount


def cache_warning(model: str, prefix_tokens: int) -> Optional[str]:
    """Warn when a prefix is too short to cache on this model.

    Cache minimums are not monotonic across model generations - Haiku 4.5 needs a
    4096-token prefix while Opus 5 needs only 512 - and a prefix under the minimum
    fails to cache silently, with no error and no cache_creation tokens reported.
    """
    try:
        info = lookup(model)
    except UnknownModel:
        return None
    if prefix_tokens < info.cache_min_tokens:
        return (
            f"Prefix is ~{prefix_tokens} tokens but {model} needs "
            f"{info.cache_min_tokens} to cache. It will not cache, silently."
        )
    return None
