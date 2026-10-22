"""Synchronous execution helpers: retries, bounded concurrency, cost reporting.

This module owns the retry loop — the SDKs' own retries are disabled on the
synchronous call (see the provider `complete` implementations), because two
retry loops nested invisibly is how a single failure used to take half an hour
to surface: a 600s default read timeout × the SDK's 2 internal retries ×
our 3 attempts before our own logic saw anything. One loop, this one, with its
attempts and backoff in `pipeline.yaml`'s `retry:` block and every attempt
logged.
"""

from __future__ import annotations

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Mapping, Optional, Sequence

from universe_builder.providers.pricing import estimate_cost
from universe_builder.providers.registry import get_provider, get_provider_by_name
from universe_builder.providers.types import LLMRequest, LLMResponse, Usage

log = logging.getLogger(__name__)


@dataclass
class Policy:
    """The retry and timeout policy `complete()` runs under.

    Field names match `pipeline.yaml`'s `retry:` block, because that block is
    where these values are supposed to come from — it was once
    *displayed* by `universe_builder config` and applied by nothing, so the file promised
    three transport attempts while the code hardcoded its own. Entry points
    push the block down with `configure()`; providers never import pipeline
    config, so the layering survives.

    `timeout_seconds` is the client-side ceiling per attempt, handed to the
    SDK per request. None keeps the SDK default (600s for OpenAI), which is
    the original behaviour and almost never what a pipeline run wants.
    """

    transport_attempts: int = 3
    validation_attempts: int = 2
    backoff_seconds: float = 1.0
    backoff_max_seconds: float = 30.0
    timeout_seconds: Optional[float] = None


_policy = Policy()


def configure(settings: Optional[Mapping[str, Any]] = None) -> None:
    """Install the run policy, usually `pipeline.Config.retry` verbatim.

    Replaces the whole policy each call — defaults plus the keys given — so a
    second `configure` cannot inherit stale values from a first, and tests
    reset with `configure()`. Unknown keys are warned about rather than
    ignored: a typo in `pipeline.yaml` must not silently mean "default".
    """
    global _policy
    settings = dict(settings or {})
    known = {f for f in Policy.__dataclass_fields__}
    unknown = set(settings) - known
    if unknown:
        log.warning(
            "retry settings %s are not recognised and have no effect; known "
            "keys: %s", sorted(unknown), sorted(known),
        )
    _policy = Policy(**{k: v for k, v in settings.items() if k in known})


@dataclass
class RunReport:
    responses: list[LLMResponse] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    cost: Optional[float] = None

    @property
    def ok_count(self) -> int:
        return sum(1 for r in self.responses if r.ok)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.responses if not r.ok)


#: HTTP statuses a second identical request cannot fix: a malformed or
#: refused request (400, 413, 422), a bad key or permission (401, 403), an
#: unknown model (404). Not retried, since the retry only delays the error.
#: 408, 409, 429 and 5xx (Anthropic's 529 overload among them) are.
NOT_RETRYABLE_STATUS = frozenset({400, 401, 403, 404, 413, 422})


def _retryable(error: BaseException) -> bool:
    """False for an SDK error whose HTTP status says a retry cannot help.
    An error without a status (a socket, a timeout) is retried."""
    return getattr(error, "status_code", None) not in NOT_RETRYABLE_STATUS


def _retry_after(error: BaseException) -> float:
    """Seconds the server asked us to wait (`retry-after` on a 429 or 529),
    else 0. Capped by the policy's `backoff_max_seconds` at the call site."""
    headers = getattr(getattr(error, "response", None), "headers", None)
    try:
        return max(float(headers.get("retry-after")), 0.0) if headers else 0.0
    except (TypeError, ValueError):
        return 0.0


def complete(
    request: LLMRequest,
    *,
    validation_retries: Optional[int] = None,
    transport_retries: Optional[int] = None,
    provider: Optional[str] = None,
) -> LLMResponse:
    """Run one request, retrying on transport errors and schema failures.

    Retry counts default to the installed `Policy` (see `configure`); passing
    them explicitly overrides it for this call. The policy's timeout lands on
    the request itself, so the provider hands it to the SDK — unless the
    request already carries one, which then wins.

    `provider` names a registered provider when the model id alone does not
    pick the transport — `openai-search` serves the same model ids as
    `openai`, with the `web_search` tool attached.
    """
    if transport_retries is None:
        transport_retries = max(0, _policy.transport_attempts - 1)
    if validation_retries is None:
        validation_retries = max(0, _policy.validation_attempts - 1)
    if request.timeout is None and _policy.timeout_seconds is not None:
        request = replace(request, timeout=_policy.timeout_seconds)

    provider = get_provider_by_name(provider) if provider else get_provider(request.model)
    attempt = 0
    # Every attempt costs tokens, including the ones that failed validation.
    # Reporting only the last attempt's usage under-reports the real spend.
    spent = Usage()

    while True:
        try:
            response = provider.complete(request)
        except Exception as e:  # noqa: BLE001 - transport failures are reported, not raised
            final = attempt >= transport_retries or not _retryable(e)
            if final:
                log.error("%s failed after %d attempts: %s", request.custom_id, attempt + 1, e)
                return LLMResponse("", None, request.model, spent, request.custom_id, str(e),
                                   transport_failed=True)
            delay = min(
                max(_policy.backoff_seconds * 2**attempt + random.random(), _retry_after(e)),
                _policy.backoff_max_seconds,
            )
            log.warning("Transport error (%s), retrying in %.1fs", e, delay)
            time.sleep(delay)
            attempt += 1
            continue

        spent = spent + response.usage
        response.usage = spent

        if response.ok or validation_retries <= 0:
            return response

        # A truncated response is not a schema disagreement, so re-asking with
        # the same budget cannot succeed - it just spends the budget again.
        # Observed live: gpt-5-nano hit an 8192 limit, was re-asked,
        # and burned a second 8192 tokens reaching the identical wall.
        if response.error and response.error.startswith("Response truncated"):
            log.error(
                "%s was truncated at max_tokens=%d; not re-asking, since the "
                "same budget would hit the same limit. Raise max_tokens.",
                request.custom_id,
                request.max_tokens,
            )
            return response

        log.warning(
            "Schema validation failed for %s, re-asking (%d left)",
            request.custom_id,
            validation_retries,
        )
        validation_retries -= 1
        attempt = 0


def run_many(
    requests: Sequence[LLMRequest],
    *,
    concurrency: int = 1,
    on_result: Optional[Callable[[LLMResponse], None]] = None,
    **kwargs,
) -> RunReport:
    """Run requests, optionally with bounded concurrency.

    Concurrency defaults to 1 because parallel requests that share a prefix all
    miss the cache - an entry only becomes readable once the first response has
    started streaming. Warm the cache with a single call, then raise concurrency.
    """
    report = RunReport()
    if not requests:
        return report

    def one(request: LLMRequest) -> LLMResponse:
        response = complete(request, **kwargs)
        if on_result:
            on_result(response)
        return response

    if concurrency <= 1:
        results = [one(r) for r in requests]
    else:
        # Warm the shared prefix first so the pool reads cache instead of writing it.
        first = one(requests[0])
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            rest = list(pool.map(one, requests[1:]))
        results = [first, *rest]

    report.responses = results
    for r in results:
        report.usage = report.usage + r.usage
    report.cost = estimate_cost(report.usage, requests[0].model)
    log.warning(
        "Ran %d requests: %d ok, %d failed, ~$%s (cache reads: %d tokens)",
        len(results),
        report.ok_count,
        report.failed_count,
        f"{report.cost:.4f}" if report.cost is not None else "?",
        report.usage.cache_read_tokens,
    )
    return report
