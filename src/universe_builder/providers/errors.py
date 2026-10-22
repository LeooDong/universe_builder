"""Exceptions raised by the provider layer."""


class ProviderError(Exception):
    """Base for every provider-layer failure."""


class ProviderNotInstalled(ProviderError):
    """The provider's SDK is not installed."""


class ProviderNotConfigured(ProviderError):
    """The provider's API key is missing."""


class UnknownModel(ProviderError):
    """The model is not in the registry."""


class BatchNotSupported(ProviderError, NotImplementedError):
    """The provider has no batch job API.

    The one exception for it: `batch.submit` raised a bare
    `NotImplementedError` for the same condition. It is still one, so a
    caller catching that sees no difference."""


class TransportFailed(ProviderError):
    """Every transport attempt failed; no response arrived at all.

    Raised by the synchronous call adapter (`llm_classes.provider_call`) so an
    operation takes its `Result.failed` path and journals the attempt at the
    `transport` stage, instead of writing `{"error": ...}` as if it were the
    model's answer."""


class EffortNotSupported(ProviderError, ValueError):
    """The provider has no effort level by this name (v0.6.0): `max` on OpenAI.

    Carries `status_code = 400` so `providers.run.complete` treats it as a
    request no retry can fix, as it treats the API's own 400: it is raised
    before anything goes on the wire, and three identical attempts would only
    log it three times. The CLI checks the same thing up front
    (`effort.refusal`), so a run is refused before its first call."""

    status_code = 400


class TaskBudgetRefused(ProviderError, ValueError):
    """A task budget that cannot be sent as asked (v0.6.1, `providers.task_budget`):
    under the API's 20,000 minimum, over the request's `max_tokens` on a single
    turn, or to a model that takes none.

    `status_code = 400`, as for `EffortNotSupported`: raised before anything
    goes on the wire, and no retry can fix it. The CLI checks the same thing
    up front (`task_budget.refusal`), so a run is refused before its first call."""

    status_code = 400
