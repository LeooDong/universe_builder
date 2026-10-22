"""A scripted `Provider`, installed in the provider registry by `conftest.fake_provider`.

`fake_call_factory` replaces the whole `LLMCall`, so a test
built on it never runs `ProviderCall.build_request`, `providers.run.complete`
(retries, re-asks, the truncation guard, usage accounting) or
`Provider._validate`. This fake replaces only the transport: everything above
`Provider.complete` / `submit_batch` is the real code.

Replies are scripted in order. Each is one of:

* a `str` — the raw model output; validated against the request's schema by
  the real `Provider._validate`, exactly as a vendor reply would be;
* an `Exception` instance — raised from `complete`, i.e. a transport failure;
* a callable `(request) -> str | Exception | LLMResponse` — for replies that
  depend on what was asked;
* an `LLMResponse` — returned as is (truncation, provider-side refusals).

Every request is recorded on `.requests`, so a test can assert on the exact
prompt the pipeline built.
"""

from __future__ import annotations

from typing import Any, Iterator, List, Optional, Sequence

from universe_builder.providers.base import Provider
from universe_builder.providers.types import BatchHandle, BatchStatus, LLMRequest, LLMResponse, Usage


class FakeProvider(Provider):
    name = "fake"
    batch_max_requests = 10_000
    batch_max_bytes = 1 << 40

    def __init__(self, usage: Optional[Usage] = None) -> None:
        self.requests: List[LLMRequest] = []
        self.replies: List[Any] = []
        self.batches: List[List[LLMRequest]] = []
        #: Batch replies by custom_id; a missing id comes back as a failure.
        self.batch_replies: dict = {}
        self.usage = usage or Usage(input_tokens=100, output_tokens=50)

    # -- scripting --------------------------------------------------------------

    def script(self, *replies: Any) -> "FakeProvider":
        self.replies.extend(replies)
        return self

    def without_batch(self) -> "FakeProvider":
        """Behave like a provider with no batch API (`supports_batch` False)."""
        self.batch_max_requests = 0
        return self

    # -- transport ----------------------------------------------------------------

    def _answer(self, reply: Any, request: LLMRequest) -> LLMResponse:
        if callable(reply) and not isinstance(reply, (str, LLMResponse)):
            reply = reply(request)
        if isinstance(reply, BaseException):
            raise reply
        if isinstance(reply, LLMResponse):
            return reply
        return self._validate(str(reply), request, Usage(**vars(self.usage)))

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.replies:
            raise AssertionError(f"fake provider ran out of scripted replies at {request.custom_id!r}")
        return self._answer(self.replies.pop(0), request)

    def submit_batch(self, requests: Sequence[LLMRequest]) -> BatchHandle:
        chunk = list(requests)
        self.batches.append(chunk)
        self.requests.extend(chunk)
        return BatchHandle(
            provider=self.name, batch_id=f"fake-batch-{len(self.batches)}", model=chunk[0].model,
            submitted_at="2026-10-03T00:00:00Z", request_count=len(chunk),
            routing={r.custom_id: dict(r.metadata) for r in chunk},
        )

    def poll_batch(self, handle: BatchHandle) -> BatchStatus:
        return BatchStatus(done=True, state="completed", succeeded=handle.request_count)

    def fetch_batch_results(self, handle: BatchHandle) -> Iterator[LLMResponse]:
        index = int(handle.batch_id.rsplit("-", 1)[1]) - 1
        for request in self.batches[index]:
            reply = self.batch_replies.get(request.custom_id)
            if reply is None:
                yield LLMResponse("", None, request.model, Usage(), request.custom_id, "no scripted reply")
                continue
            response = self._answer(reply, request)
            response.custom_id = request.custom_id
            yield response
