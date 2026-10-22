"""OpenAI provider: sync completions, the Batch API, and automatic prompt caching.

OpenAI caching needs no `cache_control` markers - prefixes over the model's
minimum are cached automatically. What it does need is ordering: stable content
(system text, schema, examples) must come first, and anything volatile must come
last. `LLMRequest` already enforces that shape, so this class just lays the
messages out in order.

Structured output goes through `response_format={"type": "json_schema", ...}`
with `strict: true` rather than the SDK's `.parse()` helper, because the raw
parameter is stable across SDK versions and identical in sync and batch paths.
"""

from __future__ import annotations

import io
import json
import logging
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence

from universe_builder.providers.base import Provider, require
from universe_builder.providers.errors import BatchNotSupported, ProviderNotConfigured
from universe_builder.providers.types import (
    BatchHandle,
    BatchStatus,
    LLMRequest,
    LLMResponse,
    Usage,
    require_matching_schema,
    schema_fingerprint,
)
from universe_builder.providers.util import expected_custom_ids

log = logging.getLogger(__name__)

_TERMINAL = {"completed", "failed", "expired", "cancelled"}


class OpenAIProvider(Provider):
    name = "openai"
    #: Stream `complete()`. See its docstring — this is what keeps a generation
    #: longer than about a minute from having its idle socket closed. Batch
    #: requests are unaffected: their results are fetched from a file, which is
    #: why the batch path has been reliable while the synchronous one was not.
    stream = True

    batch_max_requests = 50_000
    batch_max_bytes = 200 * 1024 * 1024
    endpoint = "/v1/chat/completions"
    #: Whether the API enforces a JSON schema, as opposed to loose JSON mode.
    supports_json_schema = True
    #: Use `/v1/responses` instead of `/v1/chat/completions`. Off here: the
    #: whole corpus was generated over Chat Completions and there is no defect
    #: pushing us off it. `OpenAIResponsesProvider` turns it on for callers who
    #: opt in by provider name.
    use_responses = False
    #: Attach the `web_search` tool. Responses only, and off by default —
    #: search costs $10/1k calls plus content tokens, and a generation pass
    #: that silently started searching would change both the bill and the
    #: meaning of every `sources` array.
    web_search = False
    #: Newer OpenAI models take `max_completion_tokens`; compatible APIs mostly
    #: still take `max_tokens`.
    max_tokens_field = "max_completion_tokens"
    #: Environment variables holding this provider's own key. Empty means
    #: first-party OpenAI, where the SDK's own OPENAI_API_KEY lookup is correct.
    env_vars: tuple[str, ...] = ()
    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    #: v0.6.0: sent as `reasoning_effort` (Chat Completions, and every batch
    #: line, which is a Chat Completions body) or `reasoning.effort`
    #: (`/v1/responses`). No `max`: OpenAI has none, so it is refused rather
    #: than mapped. `xhigh` only for a model whose `model_pricing.json` row
    #: lists it in `effort_levels` (none does yet: unverified). A model marked
    #: `"effort": false` (gpt-4o-mini, gpt-4.1) is not a reasoning model and
    #: gets nothing.
    EFFORT_LEVELS: tuple = ("low", "medium", "high")

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        openai = require("openai")
        key = api_key or self._key_from_env()

        # An OpenAI-compatible provider must never fall through to the SDK's own
        # environment lookup: with no key of its own it would pick up
        # OPENAI_API_KEY and send that secret to a third party's endpoint.
        if self.env_vars and not key:
            names = " or ".join(self.env_vars)
            raise ProviderNotConfigured(
                f"No API key for {self.name}. Set {names}, or pass api_key=. "
                f"Refusing to fall back to OPENAI_API_KEY - it would be sent to "
                f"{base_url or self.DEFAULT_BASE_URL}."
            )

        kwargs: dict[str, Any] = {}
        if key:
            kwargs["api_key"] = key
        if base_url:
            kwargs["base_url"] = base_url
        try:
            self._client = openai.OpenAI(**kwargs)
        except Exception as e:  # noqa: BLE001
            raise ProviderNotConfigured(f"{self.name} client init failed: {e}") from e

    @classmethod
    def _key_from_env(cls) -> str | None:
        import os

        for name in cls.env_vars:
            value = os.getenv(name)
            if value:
                return value
        return None

    # -- request assembly -------------------------------------------------------

    def _messages(self, request: LLMRequest) -> list[dict[str, str]]:
        system = self.system_with_schema(request)
        if not self.supports_json_schema:
            # DeepSeek's json_object mode requires the literal lowercase token
            # "json" somewhere in the prompt, or it declines to produce output.
            system += "\n\nRespond with a single json object and nothing else."
        messages = [{"role": "system", "content": system}]
        for ex in request.examples:
            messages.append({"role": "user", "content": ex.user})
            messages.append({"role": "assistant", "content": ex.assistant_json})
        messages.append({"role": "user", "content": request.user})
        return messages

    def _response_format(self, request: LLMRequest) -> dict[str, Any]:
        """Strict schema enforcement, where the provider supports it."""
        if not self.supports_json_schema:
            # Loose JSON mode: the model promises valid JSON but nothing constrains
            # it to the schema, so the schema in the system text is all we have.
            return {"type": "json_object"}

        return {
            "type": "json_schema",
            "json_schema": {
                "name": request.schema_model.__name__,
                "strict": True,
                "schema": self.strict_json_schema(request.schema_model),
            },
        }

    def _body(self, request: LLMRequest) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": request.model,
            "messages": self._messages(request),
            self.max_tokens_field: request.max_tokens,
            "response_format": self._response_format(request),
        }
        if request.temperature is not None:
            body["temperature"] = request.temperature
        effort = self.effort_for(request)
        if effort:
            body["reasoning_effort"] = effort
        # No task budget on OpenAI (v0.6.1): ignored, warned once per model.
        self.task_budget_for(request)
        return body

    @staticmethod
    def _usage(raw: Any) -> Usage:
        if raw is None:
            return Usage()
        details = getattr(raw, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details else 0
        if isinstance(raw, dict):
            details = raw.get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens", 0)
            prompt = raw.get("prompt_tokens", 0)
            completion = raw.get("completion_tokens", 0)
        else:
            prompt = getattr(raw, "prompt_tokens", 0) or 0
            completion = getattr(raw, "completion_tokens", 0) or 0
        return Usage(
            input_tokens=max(prompt - (cached or 0), 0),
            output_tokens=completion or 0,
            cache_read_tokens=cached or 0,
        )

    # -- sync -------------------------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        """One completion, streamed (or over `/v1/responses` when opted in).

        **Streaming is not an optimisation here, it is what makes long
        generations possible at all.** A non-streamed request holds an idle
        socket for the whole generation, and something in the path closes it at
        roughly 60 seconds. Diagnosed live: a `history` atom on
        `gpt-6-astra` takes about three minutes to generate and failed on every
        attempt with

            APIConnectionError: Connection error.
              caused by RemoteProtocolError: Server disconnected without
              sending a response.

        at 62s each time. The identical request with `stream=True` succeeded,
        first token at 7s. The failure looked like a flaky network and was
        deterministic.

        The assembled text goes through exactly the same `_validate` as before,
        so truncation reporting and schema handling are unchanged. Usage arrives
        in a final chunk that carries no choices, which is why
        `stream_options.include_usage` is set — without it a streamed call
        reports no tokens and every cost figure silently becomes zero.
        """
        if self.use_responses:
            return self._complete_responses(request)
        if not self.stream:
            return self._complete_unstreamed(request)

        body = self._body(request)
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}

        parts: list[str] = []
        usage = None
        finish_reason = None
        for event in self._sync_client(request).chat.completions.create(**body):
            if getattr(event, "usage", None):
                usage = event.usage
            for choice in getattr(event, "choices", None) or []:
                delta = getattr(choice, "delta", None)
                if delta is not None and getattr(delta, "content", None):
                    parts.append(delta.content)
                if getattr(choice, "finish_reason", None):
                    finish_reason = choice.finish_reason

        text = "".join(parts)

        # A stream that ends without a `finish_reason` did not finish: the server
        # stopped sending and the iterator simply ran out, with no exception to
        # notice. The unstreamed path could not do this — it returned a complete
        # response or raised — so the check is new because the failure is.
        #
        # Left unchecked it is worse than a crash. A stream cut after the last
        # brace yields JSON that parses, validates and gets written as a *good*
        # atom, with `Usage()` zeros making the call look free. Reproduced:
        # ok=True, parsed=a=1, error=None, 0 tokens.
        if finish_reason is None:
            log.error(
                "Stream for %s ended after %d chars without a finish_reason; "
                "treating it as incomplete rather than trusting the partial text",
                request.model, len(text),
            )
            return LLMResponse(
                text, None, request.model, self._usage(usage), request.custom_id,
                "the stream ended before the model signalled completion, so the "
                "response is incomplete however complete it may look",
            )

        if usage is None:
            # Costs would silently read zero. Not fatal — the text is whole — but
            # it must not pass for a free call.
            log.warning(
                "Stream for %s carried no usage chunk; token counts and cost for "
                "this call are unknown, not zero", request.model,
            )

        return self._validate(text, request, self._usage(usage), finish_reason)

    def _complete_unstreamed(self, request: LLMRequest) -> LLMResponse:
        """The single-response path. Kept for providers or tests that want it."""
        completion = self._sync_client(request).chat.completions.create(**self._body(request))
        choice = completion.choices[0]
        text = choice.message.content or ""
        return self._validate(
            text, request, self._usage(completion.usage), getattr(choice, "finish_reason", None)
        )

    # -- Responses API ----------------------------------------------------------

    def _responses_body(self, request: LLMRequest) -> dict:
        """The `/v1/responses` equivalent of `_body`.

        Three shape differences from Chat Completions, all of them load-bearing:
        the schema goes under `text.format` rather than `response_format`, the
        budget is `max_output_tokens`, and system/user turns become `input`
        messages of the same shape. Examples are carried through unchanged,
        because they are the cached prefix and reordering them throws the cache
        away.
        """
        body: dict = {
            "model": request.model,
            "input": self._messages(request),
            "max_output_tokens": request.max_tokens,
        }
        if self.supports_json_schema:
            body["text"] = {"format": {
                "type": "json_schema",
                "name": request.schema_model.__name__,
                "strict": True,
                "schema": self.strict_json_schema(request.schema_model),
            }}
        else:
            body["text"] = {"format": {"type": "json_object"}}
        if request.temperature is not None:
            body["temperature"] = request.temperature
        effort = self.effort_for(request)
        if effort:
            body["reasoning"] = {"effort": effort}
        self.task_budget_for(request)   # none here (v0.6.1): warned once per model
        if self.web_search:
            # Verified to compose with strict `json_schema` in one call, so a
            # searched generation needs no summarise-then-structure second pass.
            body["tools"] = [{"type": "web_search"}]
        return body

    @staticmethod
    def _responses_usage(raw) -> Usage:
        """Usage from `/v1/responses`, which already speaks our field names.

        Chat Completions says `prompt_tokens`/`completion_tokens` and hides
        cached tokens inside the prompt count; Responses reports
        `input_tokens`/`output_tokens` directly with cached tokens in
        `input_tokens_details`. Cached tokens are subtracted out of the input
        count here for the same reason as in `_usage`: they bill at a different
        rate, and folding them together loses that.
        """
        if raw is None:
            return Usage()
        details = getattr(raw, "input_tokens_details", None)
        cached = getattr(details, "cached_tokens", 0) if details else 0
        if isinstance(raw, dict):
            details = raw.get("input_tokens_details") or {}
            cached = details.get("cached_tokens", 0) or 0
            total_in = raw.get("input_tokens", 0) or 0
            total_out = raw.get("output_tokens", 0) or 0
        else:
            total_in = getattr(raw, "input_tokens", 0) or 0
            total_out = getattr(raw, "output_tokens", 0) or 0
        return Usage(
            input_tokens=max(total_in - (cached or 0), 0),
            output_tokens=total_out,
            cache_read_tokens=cached or 0,
        )

    def _complete_responses(self, request: LLMRequest) -> LLMResponse:
        """One completion over `/v1/responses`.

        Truncation is reported differently and must be translated, or a cut-off
        response reaches `_validate` looking like a schema failure — which is
        how a dead API key came to be recorded as 417 repair jobs earlier in
        this corpus's life. Responses sets `status == "incomplete"` with
        `incomplete_details.reason == "max_output_tokens"`; the rest of the
        stack understands the Chat Completions spelling, `"length"`.
        """
        response = self._sync_client(request).responses.create(
            **self._responses_body(request))
        finish_reason = None
        if getattr(response, "status", None) == "incomplete":
            detail = getattr(response, "incomplete_details", None)
            reason = getattr(detail, "reason", None) if detail else None
            finish_reason = "length" if reason == "max_output_tokens" else reason
        return self._validate(response.output_text or "", request,
                              self._responses_usage(response.usage), finish_reason)

    def _sync_client(self, request: LLMRequest):
        """The client for one synchronous call: our timeout, no SDK retries.

        `max_retries=0` is deliberate and only for this path. `providers.run`
        owns the retry loop — it logs every attempt and its backoff comes from
        `pipeline.yaml` — and the SDK's own retries nested invisibly inside it,
        so one dead call could take (600s timeout × 3 SDK attempts × 3 of ours)
        before anything surfaced. Batch endpoints keep the SDK's retries: file
        uploads and polls are not under `run.complete`, and a retried upload
        costs nothing.

        The timeout is per attempt and, on a streamed call, bounds the wait
        between chunks rather than the whole generation — a 182s `gpt-6-astra`
        response streams tokens continuously and is untouched, while the idle
        socket that used to hang for the SDK's 600s default now fails fast
        enough for the retry loop to matter.
        """
        options: dict[str, Any] = {"max_retries": 0}
        if request.timeout is not None:
            options["timeout"] = request.timeout
        return self._client.with_options(**options)

    # -- batch ------------------------------------------------------------------

    def request_size_bytes(self, request: LLMRequest) -> int:
        """Exact size of this request's JSONL line, plus the newline."""
        return len(self._jsonl([request]))

    def _jsonl(self, requests: Sequence[LLMRequest]) -> bytes:
        lines = [
            json.dumps(
                {
                    "custom_id": r.custom_id,
                    "method": "POST",
                    "url": self.endpoint,
                    "body": self._body(r),
                }
            )
            for r in requests
        ]
        return ("\n".join(lines) + "\n").encode("utf-8")

    def submit_batch(self, requests: Sequence[LLMRequest]) -> BatchHandle:
        if len(requests) > self.batch_max_requests:
            raise ValueError(
                f"{len(requests)} requests exceeds the {self.batch_max_requests} limit; "
                "chunk before calling submit_batch"
            )
        payload = self._jsonl(requests)
        upload = self._client.files.create(
            file=("batch_input.jsonl", io.BytesIO(payload)), purpose="batch"
        )
        batch = self._client.batches.create(
            input_file_id=upload.id, endpoint=self.endpoint, completion_window="24h"
        )
        log.warning("Submitted OpenAI batch %s (%d requests)", batch.id, len(requests))
        return BatchHandle(
            provider=self.name,
            batch_id=batch.id,
            model=requests[0].model,
            submitted_at=datetime.now(timezone.utc).isoformat(),
            request_count=len(requests),
            routing={r.custom_id: dict(r.metadata) for r in requests if r.custom_id},
            extra={
                **schema_fingerprint(requests[0].schema_model),
                "input_file_id": upload.id,
            },
        )

    def poll_batch(self, handle: BatchHandle) -> BatchStatus:
        batch = self._client.batches.retrieve(handle.batch_id)
        counts = getattr(batch, "request_counts", None)
        if batch.status == "completed" and getattr(batch, "output_file_id", None):
            handle.extra["output_file_id"] = batch.output_file_id
        return BatchStatus(
            done=batch.status in _TERMINAL,
            state=batch.status,
            succeeded=getattr(counts, "completed", 0) if counts else 0,
            errored=getattr(counts, "failed", 0) if counts else 0,
            processing=getattr(counts, "total", 0) if counts else 0,
        )

    def cancel_batch(self, handle: BatchHandle) -> BatchStatus:
        batch = self._client.batches.cancel(handle.batch_id)
        counts = getattr(batch, "request_counts", None)
        return BatchStatus(
            done=batch.status in _TERMINAL,
            state=batch.status,
            succeeded=getattr(counts, "completed", 0) if counts else 0,
            errored=getattr(counts, "failed", 0) if counts else 0,
            processing=getattr(counts, "total", 0) if counts else 0,
        )

    def _read_jsonl(self, file_id: str) -> Iterator[dict]:
        content = self._client.files.content(file_id).text
        for line in content.splitlines():
            if line.strip():
                yield json.loads(line)

    def fetch_batch_results(self, handle: BatchHandle) -> Iterator[LLMResponse]:
        from universe_builder.providers.registry import schema_model_by_name

        schema_model = schema_model_by_name(handle.extra.get("schema_model"))
        # Before a single result is read: a schema that moved while the batch was
        # in flight makes every validation below meaningless.
        require_matching_schema(handle, schema_model)
        batch = self._client.batches.retrieve(handle.batch_id)
        output_file_id = handle.extra.get("output_file_id") or getattr(batch, "output_file_id", None)
        # Requests that fail at the API level never reach the output file - OpenAI
        # puts them in a separate error file. Reading only the output file makes
        # those requests vanish silently.
        error_file_id = getattr(batch, "error_file_id", None)

        if not output_file_id and not error_file_id:
            error = f"batch ended as {batch.status} with no output or error file"
            log.error("Batch %s: %s", handle.batch_id, error)
            for custom_id in expected_custom_ids(handle):
                yield LLMResponse("", None, handle.model, Usage(), custom_id, error)
            return

        if output_file_id:
            for record in self._read_jsonl(output_file_id):
                yield self._response_from(record, handle, schema_model)

        if error_file_id:
            for record in self._read_jsonl(error_file_id):
                custom_id = record.get("custom_id")
                error = record.get("error") or (record.get("response") or {}).get("body")
                log.error("Batch request %s failed: %s", custom_id, error)
                yield LLMResponse("", None, handle.model, Usage(), custom_id, f"batch error: {error}")

    def _response_from(self, record: dict, handle: BatchHandle, schema_model) -> LLMResponse:
        custom_id = record.get("custom_id")
        response = record.get("response") or {}
        if response.get("status_code") != 200:
            error = record.get("error") or response.get("body")
            return LLMResponse("", None, handle.model, Usage(), custom_id, f"batch error: {error}")

        body = response.get("body", {})
        usage = self._usage(body.get("usage"))
        choice = body["choices"][0]
        text = choice["message"].get("content") or ""
        # Same truncation reporting as the sync path; a batch line that ran
        # out of budget is otherwise indistinguishable from a schema fault.
        return self._validate_stored(text, schema_model, handle.model, usage, custom_id,
                                     choice.get("finish_reason"))


class _NoBatchMixin:
    """Explicitly disowns the inherited OpenAI batch job API.

    Leaving `OpenAIProvider.submit_batch` in place on a provider with
    `batch_max_requests = 0` fails with a confusing "exceeds the 0 limit"
    ValueError instead of saying the API does not exist.
    """

    batch_max_requests = 0
    #: The OpenAI-compatible APIs (OpenRouter, DeepSeek, Qwen) take no
    #: `reasoning_effort` that has been checked here; an effort is ignored with
    #: one warning per model (`providers.effort`), never sent unverified.
    EFFORT_LEVELS: tuple = ()

    def submit_batch(self, requests):
        raise BatchNotSupported(f"{self.name} has no batch job API")

    def poll_batch(self, handle):
        raise BatchNotSupported(f"{self.name} has no batch job API")

    def fetch_batch_results(self, handle):
        raise BatchNotSupported(f"{self.name} has no batch job API")

    def cancel_batch(self, handle):
        raise BatchNotSupported(f"{self.name} has no batch job API")


class OpenRouterProvider(_NoBatchMixin, OpenAIProvider):
    """OpenRouter, reached through the OpenAI SDK against a different base URL.

    Useful as an escape hatch for models the first-party providers do not serve.
    There is no batch *job* API - OpenRouter exposes batch-priced models as
    separate `vendor/model:batch` slugs that you still call one request at a time -
    so `submit_batch` is deliberately not implemented here.
    """

    name = "openrouter"
    env_vars = ("OPENROUTER_API_KEY",)
    # OpenRouter fans out to many non-OpenAI backends; `max_tokens` is the field
    # every one of them accepts.
    max_tokens_field = "max_tokens"
    DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        super().__init__(api_key=api_key, base_url=base_url or self.DEFAULT_BASE_URL)


class DeepSeekProvider(_NoBatchMixin, OpenAIProvider):
    """DeepSeek, via its OpenAI-compatible endpoint.

    Two things to know before using it for this pipeline.

    **No schema enforcement.** DeepSeek supports only `response_format:
    {"type": "json_object"}` - loose JSON mode. The model promises syntactically
    valid JSON but nothing constrains it to the schema, so deeply nested atoms
    like `Person` will fail Pydantic validation more often than on a provider
    with strict `json_schema`. Every failure costs a fix-pass call, which eats
    into the price advantage. The schema is still in the system text, and
    `providers.run.complete()` re-asks once on a validation failure.

    **Prices swing with the clock.** Off-peak rates roughly halve the bill, and
    peak hours are 01:00-04:00 and 06:00-10:00 UTC on weekdays. The pricing table
    records off-peak; double it for peak.

    There is no batch job API, so `submit_batch` raises.
    """

    name = "deepseek"
    supports_json_schema = False
    max_tokens_field = "max_tokens"
    env_vars = ("DEEPSEEK_API_KEY",)
    DEFAULT_BASE_URL = "https://api.deepseek.com"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        super().__init__(api_key=api_key, base_url=base_url or self.DEFAULT_BASE_URL)

    @staticmethod
    def _usage(raw: Any) -> Usage:
        """DeepSeek reports cache hits under its own field names."""
        if raw is None:
            return Usage()
        get = raw.get if isinstance(raw, dict) else lambda k, d=0: getattr(raw, k, d)
        hit = get("prompt_cache_hit_tokens", 0) or 0
        miss = get("prompt_cache_miss_tokens", 0) or 0
        prompt = get("prompt_tokens", 0) or 0
        return Usage(
            input_tokens=miss if (hit or miss) else prompt,
            output_tokens=get("completion_tokens", 0) or 0,
            cache_read_tokens=hit,
        )


class QwenProvider(_NoBatchMixin, OpenAIProvider):
    """Alibaba Qwen, via the Model Studio OpenAI-compatible endpoint.

    Uses the international endpoint by default; set `base_url` to the mainland
    one if that is where your account lives. Model Studio does document a batch
    facility, but it is not wired up here - `batch_max_requests` is 0 until the
    job API has been checked against a live key rather than assumed.
    """

    name = "qwen"
    supports_json_schema = True
    max_tokens_field = "max_tokens"
    env_vars = ("DASHSCOPE_API_KEY", "QWEN_API_KEY")
    DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        super().__init__(api_key=api_key, base_url=base_url or self.DEFAULT_BASE_URL)


class OpenAIResponsesProvider(OpenAIProvider):
    """OpenAI over `/v1/responses`, opt-in by provider name.

    A separate provider rather than a flag on the default one, because the
    choice of transport is exactly the kind of thing that should be visible in
    `universe_builder config` and in a run's recorded configuration. `--provider
    openai-responses` says what happened; a boolean buried in a yaml does not.

    Why anyone would want it: reasoning models report reasoning tokens
    separately here and their budget can be capped, which is the actual cause
    of every truncation this corpus has suffered — Chat Completions gives one
    `max_tokens` covering reasoning and output together, so the only remedy is
    to raise it blindly. It is also the only transport where `web_search`
    works with general models.

    Why it is not the default: the corpus was generated over Chat Completions,
    no defect pushes us off it, and this provider layer serves three vendors —
    Responses is OpenAI-only, so making it the default would put a
    vendor-shaped exception in the one place that exists to hide vendor shapes.
    """

    name = "openai-responses"
    endpoint = "/v1/responses"
    use_responses = True

    #: Zero capacity, which is what makes `Provider.supports_batch` False.
    #: Overriding the three batch methods alone left `supports_batch` returning
    #: True — derived from `batch_max_requests > 0` — so `run_batch` still
    #: believed this transport could batch and only discovered otherwise when
    #: `submit_batch` raised. The capability flag is what callers gate on, so
    #: it is the thing that has to tell the truth.
    batch_max_requests = 0

    def chunk_for_batch(self, requests: Sequence[LLMRequest]):
        """Refused: with zero capacity the base implementation would emit one
        chunk per request rather than raising, which reads as success."""
        raise BatchNotSupported(f"{self.name} does not support --batch")

    def poll_batch(self, handle: BatchHandle):
        """Refused for the same reason as `submit_batch`."""
        raise BatchNotSupported(f"{self.name} does not support --batch")

    def cancel_batch(self, handle: BatchHandle):
        """Refused for the same reason as `submit_batch`."""
        raise BatchNotSupported(f"{self.name} does not support --batch")

    def fetch_batch_results(self, handle: BatchHandle):
        """Refused: `_response_from` reads a Chat Completions shape.

        This is the method `run_batch` actually calls to collect. An earlier
        attempt at this fix overrode a `collect_batch` that does not exist on
        the interface — dead code that looked like a guard while the real
        collect path stayed inherited and broken.
        """
        raise BatchNotSupported(f"{self.name} does not support --batch")

    def submit_batch(self, requests: Sequence[LLMRequest]) -> BatchHandle:
        """Refused: the batch path speaks Chat Completions at both ends.

        Inheriting it would be silently wrong in two places, and both cost a
        24-hour window to discover. `_batch_lines` pairs `"url": self.endpoint`
        — now `/v1/responses` — with `"body": self._body(r)`, which is a Chat
        Completions body carrying `response_format` and
        `max_completion_tokens`; the Responses endpoint accepts neither. And
        `_response_from` reads `body["choices"][0]["message"]["content"]`, a
        shape a Responses result does not have, so collect would raise
        `KeyError` on every record of a batch that had already been paid for.

        Making it work means a Responses-shaped `_batch_lines` *and* a
        Responses-shaped `_response_from`, and neither can be verified without
        submitting a real batch and waiting out the window. An unverified batch
        path is precisely the kind of thing this corpus has been bitten by
        before, so it fails here — before the upload, before the bill — rather
        than at collect time.

        Use the synchronous path: `universe_builder run --phase <p> --provider
        openai-responses` (no `--batch`).
        """
        raise BatchNotSupported(
            f"{self.name} does not support --batch: the batch submit and "
            f"collect paths both speak Chat Completions, so a batch would be "
            f"built wrong and parsed wrong. Run this provider synchronously, "
            f"or use --provider openai for batch work."
        )


class OpenAISearchProvider(OpenAIResponsesProvider):
    """`/v1/responses` with the `web_search` tool attached.

    Separate again, and for a sharper reason: with search on, a `sources` array
    means "the model looked this up" rather than "the model recalled it". Those
    are different claims about the same field, and which one a given atom
    carries should be recoverable from the run's configuration rather than
    guessed at later.
    """

    name = "openai-search"
    web_search = True
