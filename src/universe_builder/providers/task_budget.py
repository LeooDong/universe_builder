"""An advisory output budget the model paces itself against (v0.6.1).

Anthropic's task budget (beta `task-budgets-2026-03-13`) is
`output_config.task_budget = {"type": "tokens", "total": N}`, merged beside
`output_config.effort` and `output_config.format`. The model sees a countdown
of the budget as it generates and plans to finish inside it; `max_tokens`
stays the hard per-response cap, which the model does not see. Measured live:
Fable 5.1 at effort `high`, with a budget well under `max_tokens`, finished
far inside it (`end_turn`) where the same kind of request with no budget
thought through its whole output limit and was cut off.

The engine carries it as `LLMRequest.task_budget` (an int, the `total`).
**Unset is the default and sends nothing**: the request is byte for byte what
it was before v0.6.1, on the same (non-beta) endpoint.

Rules, checked before anything goes on the wire (`wire`), and refused with
`TaskBudgetRefused` (a 400 to `providers.run.complete`, so never retried):

* the total is at least `MINIMUM` (20,000, the API's floor);
* the model takes one: its `model_pricing.json` row says `"task_budget": true`
  (Opus 5, Opus 5.5, Fable 5.1, Sonnet 5, Opus 4.8 at the time of writing; not
  Haiku, not Sonnet 5.5, nothing older). Anthropic only: every other provider
  ignores it with one warning per model, as there is no equivalent;
* the total is no more than the `max_tokens` the request sends (held to the
  model's output cap). Every engine call is a single turn, so a budget over
  the cap can never be spent: the model would pace itself for room it does
  not have. A pack driving its own agent loop, where the budget spans several
  responses, sets `LLMRequest.allow_task_budget_over_max_tokens`.

Sizing: the budget is advisory and is sized by measuring the task (the output
tokens a good reply takes), not guessed. For a single call keep it under
`max_tokens` with headroom, one half to two thirds of it, so a reply that
overruns its plan still lands before the hard cap; a budget over two thirds
of the cap is allowed and warned about once.

**What is recorded is what was sent** (`sent`): journalled as
`sent_task_budget`, stamped on batch handles (`extra["task_budget"]`) and in
`run.json` when set; a request to a provider that takes none is recorded as
none.
"""

from __future__ import annotations

import argparse
from typing import Optional

from universe_builder.providers.effort import warn_once

#: The beta header the task budget needs, on the stream, the create and the
#: Message Batches call alike.
BETA = "task-budgets-2026-03-13"
#: The API's floor for `total`.
MINIMUM = 20_000
#: Above this share of `max_tokens` a single call has little headroom left.
HEADROOM_SHARE = 2 / 3


def validate(value) -> Optional[int]:
    """`value` as a budget, None when unset; ValueError for anything else.

    None and empty are unset. An int, or text spelling one (`20000`, `20_000`),
    of at least `MINIMUM`."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise ValueError(f"task budget must be a whole number of tokens, not {value!r}")
    try:
        total = int(str(value).strip().replace("_", "").replace(",", ""))
    except ValueError:
        raise ValueError(f"task budget must be a whole number of tokens, not {value!r}") from None
    if isinstance(value, float) and value != total:
        raise ValueError(f"task budget must be a whole number of tokens, not {value!r}")
    if total < MINIMUM:
        raise ValueError(f"task budget must be at least {MINIMUM:,} tokens (the API's minimum), "
                         f"not {total:,}")
    return total


def _argument(text: str) -> int:
    try:
        return validate(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def add_argument(parser, flag: str = "--task-budget", dest: Optional[str] = None,
                 help: Optional[str] = None) -> None:
    """Add a task-budget flag to an argparse parser, validated the same way
    everywhere. For a pack's own verbs too (`<pack> index-append --task-budget N`)."""
    kwargs = {"dest": dest} if dest else {}
    parser.add_argument(flag, type=_argument, default=None, metavar="N", **kwargs,
                        help=help or f"advisory output-token budget (Anthropic task budget, at least "
                                     f"{MINIMUM:,}; keep it under max_tokens with headroom); unset sends none")


def takes_task_budget(model: str) -> bool:
    """True when the pricing table marks `model` as taking a task budget."""
    from universe_builder.providers import pricing

    try:
        return pricing.lookup(model).task_budget is True
    except Exception:  # noqa: BLE001 - an unpriced or slug model: nothing recorded
        return False


def effective_max_tokens(model: str, max_tokens: int) -> int:
    """The `max_tokens` a request to `model` sends: held to the model's output
    cap where the table records one (`AnthropicProvider.max_tokens_for`)."""
    from universe_builder.providers.pricing import max_output

    cap = max_output(model)
    return min(max_tokens, cap) if cap else max_tokens


def wire(provider: str, supported: bool, model: str, budget: Optional[int], max_tokens: int,
         allow_over_max_tokens: bool = False) -> Optional[int]:
    """The task budget a provider puts on the wire for `model`, or None.

    `supported` is the provider's `TASK_BUDGET`; False means it has no such
    setting and the request goes without one (warned once per model).
    `max_tokens` is what the request sends. Raises `TaskBudgetRefused` for a
    total under the minimum, a model that takes none, or a total over
    `max_tokens` unless `allow_over_max_tokens`."""
    if budget is None:
        return None
    if not supported:
        warn_once((provider, model, "task_budget"),
                  "%s has no task budget; task budget %s ignored for %s (nothing sent)",
                  provider, budget, model)
        return None
    from universe_builder.providers.errors import TaskBudgetRefused

    if budget < MINIMUM:
        raise TaskBudgetRefused(f"task budget {budget:,} is under the API's minimum of {MINIMUM:,}")
    if not takes_task_budget(model):
        raise TaskBudgetRefused(f"{model} ({provider}) takes no task budget (its model_pricing.json "
                                "row does not say \"task_budget\": true)")
    if budget > max_tokens and not allow_over_max_tokens:
        raise TaskBudgetRefused(
            f"task budget {budget:,} is over max_tokens {max_tokens:,} for {model}: a single "
            f"response can never spend it, so the model would pace itself for room it does not "
            f"have. Keep it under max_tokens with headroom (one half to two thirds), or raise "
            f"max_tokens")
    if budget > max_tokens * HEADROOM_SHARE and not allow_over_max_tokens:
        warn_once((provider, model, "task_budget_headroom", budget, max_tokens),
                  "task budget %s is over two thirds of max_tokens %s for %s: little headroom "
                  "if a reply overruns its plan", f"{budget:,}", f"{max_tokens:,}", model)
    return budget


def sent(model: str, budget: Optional[int], max_tokens: int, provider: Optional[str] = None,
         allow_over_max_tokens: bool = False) -> Optional[int]:
    """The task budget that goes on the wire for `model` under `max_tokens`,
    from the provider class its id names (or `provider`), without building a
    client: the value the journal, batch handles and `run.json` keep. None
    when none is sent. Raises `TaskBudgetRefused` as `wire` does."""
    if budget is None:
        return None
    from universe_builder.providers.pricing import provider_for
    from universe_builder.providers.registry import provider_class

    try:
        name = provider or provider_for(model)
        supported = bool(getattr(provider_class(name), "TASK_BUDGET", False))
    except Exception:  # noqa: BLE001 - an unknown model or provider: the call fails on its own
        return None
    return wire(name, supported, model, budget, effective_max_tokens(model, max_tokens),
                allow_over_max_tokens)


def sent_or_none(model: str, budget: Optional[int], max_tokens: int,
                 allow_over_max_tokens: bool = False) -> Optional[int]:
    """`sent`, with a refusal read as none sent (a refused request sends
    nothing at all). For recording, never for deciding to send."""
    from universe_builder.providers.errors import TaskBudgetRefused

    try:
        return sent(model, budget, max_tokens, allow_over_max_tokens=allow_over_max_tokens)
    except TaskBudgetRefused:
        return None


def of_request(request) -> Optional[int]:
    """What a request put on the wire (`sent_or_none` on its own fields)."""
    budget = getattr(request, "task_budget", None)
    model = getattr(request, "model", None)
    if not budget or not isinstance(model, str):
        return None
    return sent_or_none(model, budget, request.max_tokens,
                        bool(getattr(request, "allow_task_budget_over_max_tokens", False)))


def refusal(model: str, budget: Optional[int], max_tokens: int,
            provider: Optional[str] = None) -> Optional[str]:
    """Why `budget` cannot go to `model` under `max_tokens`, or None: the
    pre-flight a CLI makes before its first call, so a run is refused rather
    than failing per request. A provider with no task budget is not refused
    (ignored, warned once), so one `--task-budget` can cover a mixed run."""
    from universe_builder.providers.errors import TaskBudgetRefused

    try:
        sent(model, budget, max_tokens, provider)
    except TaskBudgetRefused as e:
        return str(e)
    return None


def output_config(budget: int) -> dict:
    """The `output_config` entry for `budget`."""
    return {"type": "tokens", "total": int(budget)}
