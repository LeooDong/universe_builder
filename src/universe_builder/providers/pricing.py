"""Model registry and cost estimation.

The pricing table lives in `model_pricing.json` next to this module and is loaded
through `universe_builder.settings`, so it resolves regardless of the working
directory. Prices go stale; treat estimates as indicative and re-check the table
before making a decision based on it.

Rows can be added or replaced without an engine release (v0.9.0): the active
pack's `model_pricing_path`, then the file `UNIVERSE_BUILDER_MODEL_PRICING`
names, each in the same shape, are laid over the engine's rows, the later
winning. A model priced there is callable everywhere a model id is taken.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
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


#: Names a pricing file laid over the engine's rows and the pack's.
PRICING_ENV = "UNIVERSE_BUILDER_MODEL_PRICING"


def _load(path: Path) -> dict[str, ModelInfo]:
    """The rows of one pricing file. A malformed file or row raises ValueError
    naming the file: a declared table that cannot be read stops the run rather
    than silently pricing nothing."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"pricing file {path} cannot be read: {e}") from e
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, dict):
        raise ValueError(f"pricing file {path} has no \"models\" object")
    rows = {}
    for name, spec in models.items():
        if not isinstance(spec, dict):
            raise ValueError(f"pricing file {path}, model {name!r}: a row is an object")
        try:
            info = ModelInfo(model=name, **spec)
        except TypeError as e:
            raise ValueError(f"pricing file {path}, model {name!r}: {e}") from e
        problem = _row_problem(info)
        if problem:
            raise ValueError(f"pricing file {path}, model {name!r}: {problem}")
        rows[name] = info
    return rows


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_count(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _row_problem(info: ModelInfo) -> Optional[str]:
    """What is wrong with one row's values, or None. A price as a string or a
    negative number would otherwise surface as a wrong cost or a TypeError
    deep in a run."""
    if not isinstance(info.provider, str) or not info.provider:
        return "provider must be a non-empty string"
    for field in ("input", "cached_input", "output"):
        value = getattr(info, field)
        if not _is_number(value) or value < 0:
            return f"{field} must be a non-negative number, not {value!r}"
    if info.cache_write is not None and (not _is_number(info.cache_write) or info.cache_write < 0):
        return f"cache_write must be null or a non-negative number, not {info.cache_write!r}"
    if not _is_number(info.batch_discount) or not 0 < info.batch_discount <= 1:
        return f"batch_discount must be a number in (0, 1], not {info.batch_discount!r}"
    if not _is_count(info.cache_min_tokens) or info.cache_min_tokens < 0:
        return f"cache_min_tokens must be a non-negative integer, not {info.cache_min_tokens!r}"
    if not _is_count(info.context) or info.context <= 0:
        return f"context must be a positive integer, not {info.context!r}"
    if info.max_output is not None and (not _is_count(info.max_output) or info.max_output <= 0):
        return f"max_output must be null or a positive integer, not {info.max_output!r}"
    for field in ("effort", "task_budget"):
        value = getattr(info, field)
        if value is not None and not isinstance(value, bool):
            return f"{field} must be null or true/false, not {value!r}"
    levels = info.effort_levels
    if levels is not None and (not isinstance(levels, (list, tuple))
                               or not all(isinstance(level, str) for level in levels)):
        return f"effort_levels must be null or a list of strings, not {levels!r}"
    return None


def _load_layer(path: Path) -> dict[str, ModelInfo]:
    """One layered file's rows, checked also for a provider the engine has a
    factory for (the engine's own table is checked by its tests; importing the
    factory registry while this module loads would be circular)."""
    from universe_builder.providers.registry import provider_names

    rows = _load(path)
    known = provider_names()
    for name, info in rows.items():
        if info.provider not in known:
            raise ValueError(f"pricing file {path}, model {name!r}: provider {info.provider!r} "
                             f"is not one of {', '.join(known)}")
    return rows


#: The engine's own rows. Tests patch entries here; `_table` reads it on every
#: lookup, so a patch shows through the layered table too.
_MODELS: dict[str, ModelInfo] = _load(settings.MODEL_PRICING)
#: The layered rows, keyed by each file's path, mtime and size: an edited or
#: replaced file is read again. Holds the last key only.
_LAYERS: dict[tuple, dict[str, ModelInfo]] = {}


def extra_pricing_paths() -> list[Path]:
    """The pricing files laid over the engine's rows, lowest first: the active
    pack's `model_pricing_path`, then `UNIVERSE_BUILDER_MODEL_PRICING`. Read at
    use time, so a pack switch or an env change applies."""
    from universe_builder.registry import active_pack

    paths = []
    pack_path = getattr(active_pack(), "model_pricing_path", None)
    if pack_path:
        paths.append(Path(pack_path))
    env_path = os.environ.get(PRICING_ENV)
    if env_path:
        paths.append(Path(env_path))
    return paths


def _layer_key(paths: list[Path]) -> tuple:
    key = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError as e:
            raise ValueError(f"pricing file {path} cannot be read: {e}") from e
        key.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(key)


def _table() -> dict[str, ModelInfo]:
    paths = extra_pricing_paths()
    if not paths:
        return _MODELS
    key = _layer_key(paths)
    if key not in _LAYERS:
        layered: dict[str, ModelInfo] = {}
        for path in paths:
            layered.update(_load_layer(path))
        _LAYERS.clear()
        _LAYERS[key] = layered
    return {**_MODELS, **_LAYERS[key]}


def lookup(model: str) -> ModelInfo:
    """Registry entry for a model ID."""
    table = _table()
    try:
        return table[model]
    except KeyError as e:
        raise UnknownModel(
            f"{model!r} is not in the pricing table. Add it to the pack's "
            f"model_pricing_path or {PRICING_ENV} (no engine release needed), or to "
            f"{settings.MODEL_PRICING.name}; or pick one of: {', '.join(sorted(table))}"
        ) from e


def provider_for(model: str) -> str:
    """Which provider serves this model. OpenRouter slugs (`vendor/model`) route to openrouter."""
    if "/" in model:
        return "openrouter"
    return lookup(model).provider


def known_models() -> list[str]:
    """Every priced model: the engine's rows and any the pack or env adds."""
    return sorted(_table())


def engine_models() -> list[str]:
    """The engine's own rows only. For import-time callers (`config`), which
    run before a pack may have registered."""
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
