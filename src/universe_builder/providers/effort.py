"""How hard a model is asked to think: one provider-neutral setting (v0.6.0).

Until v0.6.0 the engine sent no reasoning setting at all, so every model ran at
its vendor's default: OpenAI's reasoning models at the API's default reasoning
effort, Claude at the model's default (`high` on Haiku 5.5 and Sonnet 5.5,
`medium` on Opus 5.5). Comparing two vendors' models fairly needs the same
knob on both, so `effort` is one value the engine carries on every request
(`LLMRequest.effort`) and each provider translates:

* OpenAI (Chat Completions and its batch lines): `reasoning_effort`; over
  `/v1/responses`: `reasoning.effort`. `low`, `medium`, `high`; `xhigh` only
  for a model whose `model_pricing.json` row lists it in `effort_levels`. A model the pricing table marks
  `"effort": false` (gpt-4o-mini, gpt-4.1, gpt-4.1-mini: not reasoning models)
  gets nothing, with one warning per model.
* Anthropic: `output_config.effort`, merged beside the structured-output
  `format` when that is sent too; the same params go out streamed and in a
  Message Batch. Haiku 4.5 takes no effort (the API refuses it) and is marked
  `"effort": false`.
* Google, DeepSeek, Qwen and OpenRouter: ignored, with one warning per model.

**Unset is the default and sends nothing**, so every request built without an
effort is byte for byte what it was before v0.6.0.

The levels are `low`, `medium` and `high`, which both vendors take, plus
`xhigh` and `max`, which pass through only where the model has them
(`Provider.EFFORT_LEVELS`, narrowed or widened per model by a row's
`effort_levels`): Claude takes all five, OpenAI has no `max`, and `xhigh`
only on a model recorded as taking it.

**What is recorded is what was sent** (`sent`): a run that asked for `low`
on Gemini or gpt-4o-mini sent nothing and is journalled with no effort, so
`costs --by effort`, experiment records and batch handles show the wire, not
the request. A level
a provider lacks is refused, never quietly mapped to a neighbour, since a run
labelled `max` that ran at `xhigh` would mislabel every comparison drawn from
it.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional, Sequence

log = logging.getLogger(__name__)

#: The levels every effort-taking provider understands.
LEVELS = ("low", "medium", "high")
#: Levels that pass through to a provider that has them.
EXTENDED = ("xhigh", "max")
ALL = LEVELS + EXTENDED

_warned: set = set()
_warned_lock = threading.Lock()


def validate(value: Optional[str]) -> Optional[str]:
    """`value` as a level, None when unset; ValueError for anything else.

    Empty and None are unset. Case is folded (`High` is `high`)."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text not in ALL:
        raise ValueError(f"effort must be one of {', '.join(ALL)} (or unset), not {value!r}")
    return text


def add_argument(parser, flag: str = "--effort", dest: Optional[str] = None, help: Optional[str] = None) -> None:
    """Add an effort flag to an argparse parser, the same choices everywhere.
    For a pack's own verbs too (`<pack> index-append --effort low`)."""
    kwargs = {"dest": dest} if dest else {}
    parser.add_argument(flag, choices=ALL, default=None, metavar="LEVEL", **kwargs,
                        help=help or f"reasoning effort ({', '.join(ALL)}); unset sends none, "
                                     "the provider's default")


def takes_effort(model: str) -> Optional[bool]:
    """False when the pricing table marks `model` as taking no effort; None
    when it says nothing (the provider's rule applies)."""
    from universe_builder.providers import pricing

    try:
        return pricing.lookup(model).effort
    except Exception:  # noqa: BLE001 - an unpriced or slug model: nothing recorded
        return None


def warn_once(key: tuple, message: str, *args) -> None:
    """Log `message` once per process for `key`."""
    with _warned_lock:
        if key in _warned:
            return
        _warned.add(key)
    log.warning(message, *args)


def model_levels(model: str) -> Optional[tuple]:
    """The levels the pricing table records for `model` (`effort_levels`), or
    None when it records none and the provider's `EFFORT_LEVELS` apply. A row
    lists them where the model's own set differs from its provider's: an
    OpenAI model takes `xhigh` only where its row says so."""
    from universe_builder.providers import pricing

    try:
        levels = pricing.lookup(model).effort_levels
    except Exception:  # noqa: BLE001 - an unpriced or slug model: nothing recorded
        return None
    return tuple(levels) if levels is not None else None


def wire(provider: str, levels: Sequence[str], model: str, effort: Optional[str]) -> Optional[str]:
    """The effort a provider puts on the wire for `model`, or None to send none.

    `levels` is the provider's `EFFORT_LEVELS`; empty means the provider has
    no effort setting, and the request goes without one (warned once). A model
    marked `"effort": false` likewise. A model row's `effort_levels` replaces
    the provider's set for that model. A level not in the set raises
    `EffortNotSupported`."""
    if effort is None:
        return None
    if not levels:
        warn_once((provider, model, "provider"),
                  "%s takes no effort setting; effort %r ignored for %s (it runs at its default)",
                  provider, effort, model)
        return None
    if takes_effort(model) is False:
        warn_once((provider, model, "model"),
                  "%s takes no effort setting (not a reasoning model, per model_pricing.json); "
                  "effort %r ignored, nothing sent", model, effort)
        return None
    allowed = model_levels(model) or tuple(levels)
    if effort not in allowed:
        from universe_builder.providers.errors import EffortNotSupported

        raise EffortNotSupported(f"{model} ({provider}) has no effort level {effort!r}; "
                                 f"it takes {', '.join(allowed)}")
    return effort


def sent(model: str, effort: Optional[str], provider: Optional[str] = None) -> Optional[str]:
    """The effort that goes on the wire for `model`, from the provider class
    its id names (or `provider`), without building a client: the value the
    journal, batch handles, experiment records and `costs --by effort` keep.
    None when none is sent (unset, or a provider or model that takes none).
    Raises `EffortNotSupported` for a level the model lacks."""
    if effort is None:
        return None
    from universe_builder.providers.pricing import provider_for
    from universe_builder.providers.registry import provider_class

    try:
        name = provider or provider_for(model)
        levels = provider_class(name).EFFORT_LEVELS
    except Exception:  # noqa: BLE001 - an unknown model or provider: the call fails on its own
        return None
    return wire(name, levels, model, effort)


def sent_or_none(model: str, effort: Optional[str]) -> Optional[str]:
    """`sent`, with a refused level read as none sent (a refused request sends
    nothing at all). For recording a failure, never for deciding to send."""
    from universe_builder.providers.errors import EffortNotSupported

    try:
        return sent(model, effort)
    except EffortNotSupported:
        return None


def refusal(model: str, effort: Optional[str], provider: Optional[str] = None) -> Optional[str]:
    """Why `effort` cannot go to `model`, or None: the pre-flight a CLI makes
    before its first call, so a run is refused rather than failing per request.

    Only a level the model lacks is refused. A provider or model that takes
    no effort at all is not (the request goes without one, warned once), so
    one `--effort` can cover a run over several vendors' models."""
    from universe_builder.providers.errors import EffortNotSupported

    try:
        sent(model, effort, provider)
    except EffortNotSupported as e:
        return str(e)
    return None


def label(model: str, effort: Optional[str]) -> str:
    """`model` or `model@effort`: one name for a (model, effort) pair, as the
    experiment judge keys and `costs --by effort` show them."""
    return f"{model}@{effort}" if effort else model


def split_label(text: str) -> tuple:
    """`(model, effort)` from `model` or `model@effort`; the effort validated."""
    model, sep, effort = text.partition("@")
    return model, (validate(effort) if sep else None)
