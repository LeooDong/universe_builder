"""Anthropic provider: streamed completions, Message Batches, and prompt caching.

Cache layout, which is the whole point of the class:

    system   [base instructions + JSON schema]   <- breakpoint 1
    user     [one-shot example prompt]
    assistant[one-shot example JSON]             <- breakpoint 2
    user     [the actual topic]                     (volatile, after both)

Breakpoint 2 matters. The one-shot examples in `prompts/examples/` are several
thousand tokens and are identical for every topic of a given atom type, so
leaving them after the last breakpoint re-bills them at full input price on every
single call.

**Structured output is best effort, and the prompt is the floor** (v0.5.9). The schema goes on the wire as
`output_config.format`, through the SDK's `transform_schema`, which moves the
keywords the API refuses (`minimum`/`maximum` on every pack's coordinates, a
400 without it) into the field's description. Even so, the API compiles the
schema into a grammar and refuses one that is too large, and most history atoms
are (`City`, `Figure`, `Polity`: "The compiled grammar is too large", observed
on Haiku 4.5; `Work` and `Population` compile). Size alone does not predict it,
so the provider asks: on that 400 it remembers the (model, schema) pair and
sends the same request without `output_config`. The schema is still in the
cached system text, and `Provider._validate` is the check either way, as it is
for DeepSeek's loose JSON mode. A batch cannot fall back per line, so
`submit_batch` probes each new (model, schema) once with a one-token request
first (a refused schema is a free 400; an accepted one bills its few thousand
input tokens once per process).

**A task budget goes through the beta endpoint** (v0.6.1,
`providers.task_budget`): `output_config.task_budget` beside the effort and
the format, with the beta header `task-budgets-2026-03-13`, on
`client.beta.messages.stream` (the streamed call and its prompt-only
fallback) and `client.beta.messages.batches.create` (a handle so submitted
records `extra["betas"]`, and is polled and collected through the beta
batches endpoints too). A request with no budget takes the endpoints it
always took, byte for byte. The schema probe stays on the plain endpoint:
it asks only whether a grammar compiles.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Iterator, Optional, Sequence

from universe_builder.providers.base import Provider, require
from universe_builder.providers.errors import ProviderNotConfigured
from universe_builder.providers.types import (
    BatchHandle,
    BatchStatus,
    LLMRequest,
    LLMResponse,
    Usage,
    require_matching_schema,
    schema_fingerprint,
)

log = logging.getLogger(__name__)

_EPHEMERAL = "ephemeral"

#: What the API says when it will not compile a schema into a grammar. Both
#: spellings are documented; the first is the one observed live.
_GRAMMAR_REJECTION = re.compile(
    r"compiled grammar is too large|schema is too complex", re.IGNORECASE)

#: Message Batches accept a `custom_id` of 1-64 characters from this set only.
#: The engine's ids are `<atom>-<index>-<slug>`, and the slug keeps any
#: alphanumeric character, so "Gorée" or a 48-character slug behind
#: `append_city` would be refused; such ids are swapped for a wire id on submit
#: and swapped back on collect (`BatchHandle.extra["wire_ids"]`).
_WIRE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class AnthropicProvider(Provider):
    name = "anthropic"
    batch_max_requests = 100_000
    batch_max_bytes = 256 * 1024 * 1024
    #: v0.6.0: `output_config.effort`, beside `output_config.format` when the
    #: schema goes too; the same params stream and batch. All five levels
    #: (`xhigh` from the 4.7 models, `max` from 4.6). Haiku 4.5 refuses effort
    #: and is marked `"effort": false` in `model_pricing.json`, so it gets none.
    #: Unset, nothing is sent and the model runs at its default (`high` on
    #: Haiku 5.5 and Sonnet 5.5, `medium` on Opus 5.5).
    EFFORT_LEVELS: tuple = ("low", "medium", "high", "xhigh", "max")
    #: v0.6.1: `output_config.task_budget`, on the beta endpoint with the
    #: `task-budgets-2026-03-13` header; only for a model whose
    #: `model_pricing.json` row says `"task_budget": true` (others are refused).
    TASK_BUDGET: bool = True

    def __init__(self, api_key: str | None = None) -> None:
        anthropic = require("anthropic")
        try:
            self._client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
        except Exception as e:  # noqa: BLE001
            raise ProviderNotConfigured(f"Anthropic client init failed: {e}") from e

    # -- schema -----------------------------------------------------------------

    @property
    def _unconstrained(self) -> set[tuple[str, str]]:
        """(model, schema) pairs whose schema the API would not compile: those
        go out without `output_config`. Per provider instance, so per process."""
        return self.__dict__.setdefault("_unconstrained_pairs", set())

    @property
    def _probed(self) -> set[tuple[str, str]]:
        return self.__dict__.setdefault("_probed_pairs", set())

    @classmethod
    def wire_schema(cls, schema_model: Any) -> Optional[dict[str, Any]]:
        """The schema as `output_config` takes it, or None if it cannot be sent.

        The strict normalisation (every property required, `additionalProperties:
        false`, `$ref` siblings inlined) and then the SDK's `transform_schema`,
        which strips what the API rejects (`minimum`, `maximum`, `minLength`, an
        unsupported `format`, ...) into the description; the reply is still
        validated against the full Pydantic model.
        """
        try:
            return require("anthropic").transform_schema(cls.strict_json_schema(schema_model))
        except (ValueError, TypeError) as e:
            log.warning("%s cannot be sent as an Anthropic output schema (%s); the prompt "
                        "carries it instead", getattr(schema_model, "__name__", schema_model), e)
            return None

    @staticmethod
    def _schema_key(request: LLMRequest) -> tuple[str, str]:
        name = request.schema_model.__name__
        digest = schema_fingerprint(request.schema_model)["schema_sha256"][:12]
        return (request.model, f"{name}:{digest}")

    def _constrained(self, request: LLMRequest) -> bool:
        return self._schema_key(request) not in self._unconstrained

    @staticmethod
    def _is_grammar_rejection(error: BaseException) -> bool:
        return getattr(error, "status_code", None) == 400 and bool(
            _GRAMMAR_REJECTION.search(str(error)))

    def _give_up_schema(self, request: LLMRequest, error: BaseException) -> None:
        key = self._schema_key(request)
        if key not in self._unconstrained:
            log.warning("%s will not compile %s as an output grammar (%s); sending it "
                        "through the prompt only, validated on return", key[0], key[1], error)
        self._unconstrained.add(key)

    # -- request assembly -------------------------------------------------------

    def _system_blocks(self, request: LLMRequest) -> list[dict[str, Any]]:
        text = self.system_with_schema(request)
        block: dict[str, Any] = {"type": "text", "text": text}
        if request.cache:
            block["cache_control"] = {"type": _EPHEMERAL, "ttl": request.cache_ttl}
        return [block]

    def _messages(self, request: LLMRequest) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for i, ex in enumerate(request.examples):
            messages.append({"role": "user", "content": ex.user})
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": [{"type": "text", "text": ex.assistant_json}],
            }
            # Breakpoint on the final example so the whole one-shot block caches.
            if request.cache and i == len(request.examples) - 1:
                assistant["content"][0]["cache_control"] = {
                    "type": _EPHEMERAL,
                    "ttl": request.cache_ttl,
                }
            messages.append(assistant)
        messages.append({"role": "user", "content": request.user})
        return messages

    @staticmethod
    def max_tokens_for(request: LLMRequest) -> int:
        """`request.max_tokens`, held to the model's output cap.

        A `max_tokens` above the cap is a 400, not a truncation, so the cap
        (`max_output` in `model_pricing.json`, from the Models API) wins: 64K on
        Haiku 4.5, 128K on the 4.6-and-later models, both above the engine's
        16,384 and `pipeline.yaml`'s 32,768.
        """
        from universe_builder.providers.pricing import max_output

        cap = max_output(request.model)
        if cap and request.max_tokens > cap:
            log.warning("max_tokens %d is over %s's %d-token output cap; sending %d",
                        request.max_tokens, request.model, cap, cap)
            return cap
        return request.max_tokens

    def sent_max_tokens(self, request: LLMRequest) -> int:
        """`max_tokens_for`, without its warning: what a task budget is held
        under (`Provider.task_budget_for`)."""
        from universe_builder.providers.task_budget import effective_max_tokens

        return effective_max_tokens(request.model, request.max_tokens)

    def _params(self, request: LLMRequest, constrained: bool = False) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": request.model,
            "max_tokens": self.max_tokens_for(request),
            "system": self._system_blocks(request),
            "messages": self._messages(request),
        }
        # Sampling parameters are rejected outright on the 4.7-and-later models,
        # so only send one when the caller explicitly asked for it.
        if request.temperature is not None:
            params["temperature"] = request.temperature
        if constrained:
            schema = self.wire_schema(request.schema_model)
            if schema is not None:
                params["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        effort = self.effort_for(request)
        if effort:
            # Merged, not replaced: the format above and the effort share the
            # one `output_config` object.
            params["output_config"] = {**params.get("output_config", {}), "effort": effort}
        budget = self.task_budget_for(request)
        if budget:
            # Merged likewise (v0.6.1); `_betas` reads it back to pick the beta
            # endpoint and header.
            from universe_builder.providers.task_budget import output_config

            params["output_config"] = {**params.get("output_config", {}),
                                       "task_budget": output_config(budget)}
        return params

    @staticmethod
    def _betas(params: dict[str, Any]) -> list[str]:
        """The beta headers `params` need: the task budget's when it carries
        one (v0.6.1), else none, and the request takes the plain endpoint."""
        from universe_builder.providers.task_budget import BETA

        return [BETA] if "task_budget" in (params.get("output_config") or {}) else []

    def _structured_params(self, request: LLMRequest) -> dict[str, Any]:
        """`_params` with the schema on the wire unless this (model, schema) is
        known not to compile: one shape for the streamed call and the batch."""
        return self._params(request, constrained=self._constrained(request))

    #: The batch path's name for `_structured_params`, kept for callers and tests.
    _batch_params = _structured_params

    @staticmethod
    def _usage(raw: Any) -> Usage:
        """Anthropic's `input_tokens` already excludes cache reads and writes,
        which is the engine's convention (OpenAI's is subtracted to match)."""
        return Usage(
            input_tokens=getattr(raw, "input_tokens", 0) or 0,
            output_tokens=getattr(raw, "output_tokens", 0) or 0,
            cache_read_tokens=getattr(raw, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(raw, "cache_creation_input_tokens", 0) or 0,
        )

    @staticmethod
    def _text(message: Any) -> str:
        """The reply's JSON text: every `text` block joined (a thinking model's
        `thinking` blocks come first and are skipped), and, for a reply sent
        without a grammar, a Markdown fence or a line of prose around the
        object removed."""
        text = "".join(getattr(b, "text", "") or "" for b in getattr(message, "content", None) or []
                       if getattr(b, "type", None) == "text")
        return _json_text(text)

    # -- sync -------------------------------------------------------------------

    def _stream(self, request: LLMRequest, params: dict[str, Any]) -> Any:
        # Same contract as `OpenAIProvider._sync_client`: `providers.run` owns
        # the retry loop, so the SDK's own retries are off, and the policy
        # timeout is handed through per attempt. Streamed, so that timeout bounds
        # the wait between events rather than the whole generation (a thinking
        # model's atom runs past 120s), and so the SDK's ten-minute guard on a
        # large non-streamed `max_tokens` never fires.
        options: dict[str, Any] = {"max_retries": 0}
        if request.timeout is not None:
            options["timeout"] = request.timeout
        client = self._client.with_options(**options)
        betas = self._betas(params)
        if betas:
            # A task budget (v0.6.1): the beta endpoint, the header passed as
            # `betas`. Without one, the call is exactly what it was.
            with client.beta.messages.stream(**params, betas=betas) as stream:
                return stream.get_final_message()
        with client.messages.stream(**params) as stream:
            return stream.get_final_message()

    def complete(self, request: LLMRequest) -> LLMResponse:
        """One streamed completion, validated by `Provider._validate`.

        The raw text is always kept, so a reply that does not parse can be
        repaired from, and a cut-off or refused reply is named as such from its
        `stop_reason`. A schema the API will not compile is
        retried at once without `output_config` and remembered (module doc).
        """
        constrained = self._constrained(request)
        try:
            message = self._stream(request, self._params(request, constrained))
        except Exception as e:  # noqa: BLE001 - only the grammar refusal is handled here
            if not (constrained and self._is_grammar_rejection(e)):
                raise
            self._give_up_schema(request, e)
            message = self._stream(request, self._params(request, constrained=False))
        usage = self._usage(message.usage)
        return self._validate(self._text(message), request, usage,
                              getattr(message, "stop_reason", None))

    # -- batch ------------------------------------------------------------------

    def _probe_schema(self, request: LLMRequest) -> None:
        """Find out once whether this (model, schema) compiles, before a batch
        carries it: a batch line cannot fall back the way `complete` does."""
        key = self._schema_key(request)
        if key in self._probed or key in self._unconstrained:
            return
        self._probed.add(key)
        schema = self.wire_schema(request.schema_model)
        if schema is None:
            self._unconstrained.add(key)
            return
        try:
            self._client.messages.create(
                model=request.model, max_tokens=1,
                messages=[{"role": "user", "content": "{}"}],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
        except Exception as e:  # noqa: BLE001 - anything else fails the submit below, loudly
            if not self._is_grammar_rejection(e):
                raise
            self._give_up_schema(request, e)

    def request_size_bytes(self, request: LLMRequest) -> int:
        payload = {"custom_id": request.custom_id, "params": self._batch_params(request)}
        return len(self._dumps(payload).encode("utf-8"))

    @staticmethod
    def _wire_ids(requests: Sequence[LLMRequest]) -> dict[int, str]:
        """Position -> the custom_id each request goes out under: its own where
        the API accepts it, else `r<index>-<hash>` (module note on `_WIRE_ID`)."""
        ids: dict[int, str] = {}
        taken = {r.custom_id for r in requests if r.custom_id and _WIRE_ID.match(r.custom_id)}
        for i, r in enumerate(requests):
            cid = r.custom_id or ""
            if cid and _WIRE_ID.match(cid):
                ids[i] = cid
                continue
            digest = hashlib.blake2b(cid.encode("utf-8"), digest_size=6).hexdigest()
            wire = f"r{i:06d}-{digest}"
            while wire in taken:
                wire += "x"
            taken.add(wire)
            ids[i] = wire
        return ids

    def submit_batch(self, requests: Sequence[LLMRequest]) -> BatchHandle:
        if len(requests) > self.batch_max_requests:
            raise ValueError(
                f"{len(requests)} requests exceeds the {self.batch_max_requests} limit; "
                "chunk before calling submit_batch"
            )
        seen: set[tuple[str, str]] = set()
        for r in requests:
            key = self._schema_key(r)
            if key not in seen:
                seen.add(key)
                self._probe_schema(r)
        wire = self._wire_ids(requests)
        payload = [
            {"custom_id": wire[i], "params": self._batch_params(r)} for i, r in enumerate(requests)
        ]
        betas = sorted({b for line in payload for b in self._betas(line["params"])})
        if betas:
            # A task budget (v0.6.1): Message Batches take it through the beta
            # endpoint, the header on the create, each line's params unchanged
            # in shape (verified live: accepted).
            batch = self._client.beta.messages.batches.create(betas=betas, requests=payload)
        else:
            batch = self._client.messages.batches.create(requests=payload)
        log.warning("Submitted Anthropic batch %s (%d requests)", batch.id, len(payload))
        extra: dict[str, Any] = dict(schema_fingerprint(requests[0].schema_model))
        if betas:
            extra["betas"] = betas
        renamed = {wire[i]: r.custom_id for i, r in enumerate(requests)
                   if r.custom_id and wire[i] != r.custom_id}
        if renamed:
            extra["wire_ids"] = renamed
        return BatchHandle(
            provider=self.name,
            batch_id=batch.id,
            model=requests[0].model,
            submitted_at=datetime.now(timezone.utc).isoformat(),
            request_count=len(payload),
            routing={r.custom_id: dict(r.metadata) for r in requests if r.custom_id},
            extra=extra,
        )

    def _batches(self, handle: BatchHandle) -> tuple[Any, dict[str, Any]]:
        """The batches endpoint a handle was submitted through, and the kwargs
        it takes: the beta one, with its headers, for a handle that records
        `betas` (v0.6.1); otherwise the plain one, as before."""
        betas = handle.extra.get("betas") or []
        if betas:
            return self._client.beta.messages.batches, {"betas": list(betas)}
        return self._client.messages.batches, {}

    def poll_batch(self, handle: BatchHandle) -> BatchStatus:
        batches, kwargs = self._batches(handle)
        batch = batches.retrieve(handle.batch_id, **kwargs)
        counts = batch.request_counts
        # A canceled or expired line has no result either; count it with the
        # errored ones so `succeeded + errored` is everything that is settled.
        settled_badly = sum(getattr(counts, k, 0) or 0 for k in ("errored", "canceled", "expired"))
        return BatchStatus(
            done=batch.processing_status == "ended",
            state=batch.processing_status,
            succeeded=getattr(counts, "succeeded", 0),
            errored=settled_badly,
            processing=getattr(counts, "processing", 0),
        )

    def cancel_batch(self, handle: BatchHandle) -> BatchStatus:
        batches, kwargs = self._batches(handle)
        batches.cancel(handle.batch_id, **kwargs)
        # The cancel reply is the batch mid-cancel; poll for the counts.
        return self.poll_batch(handle)

    @staticmethod
    def _batch_error(result: Any, kind: str) -> str:
        """`batch errored: invalid_request_error: <message>`. The SDK nests the
        error object (`result.error.error`); the outer `.type` is just "error"."""
        outer = getattr(result, "error", None)
        inner = getattr(outer, "error", None) or outer
        etype = getattr(inner, "type", None) or kind
        message = getattr(inner, "message", None)
        return f"batch {kind}: {etype}" + (f": {message}" if message else "")

    def fetch_batch_results(self, handle: BatchHandle) -> Iterator[LLMResponse]:
        from universe_builder.providers.registry import schema_model_by_name

        schema_model = schema_model_by_name(handle.extra.get("schema_model"))
        # Before a single result is read: a schema that moved while the batch was
        # in flight makes every validation below meaningless.
        require_matching_schema(handle, schema_model)
        renamed = handle.extra.get("wire_ids") or {}
        batches, kwargs = self._batches(handle)
        for result in batches.results(handle.batch_id, **kwargs):
            custom_id = renamed.get(result.custom_id, result.custom_id)
            kind = result.result.type
            if kind != "succeeded":
                yield LLMResponse("", None, handle.model, Usage(), custom_id,
                                  self._batch_error(result.result, kind))
                continue

            message = result.result.message
            usage = self._usage(message.usage)
            yield self._validate_stored(self._text(message), schema_model, handle.model, usage,
                                        custom_id, getattr(message, "stop_reason", None))


def _json_text(text: str) -> str:
    """The JSON object in a reply sent without a grammar: the text itself,
    byte for byte, when it opens with `{` (a truncated object is kept whole,
    for the truncation report and for repair), else the inside of a Markdown fence, else from the
    first `{` to the last `}`. A reply with no object is returned as it came.

    Either way, a complete object followed by nothing but closing braces or
    brackets loses them (`_without_extra_closers`)."""
    stripped = text.strip()
    if not stripped:
        return text
    if stripped.startswith("{"):
        return _without_extra_closers(text)
    fence = re.match(r"^```[A-Za-z]*\s*\n(.*?)\n?```\s*$", stripped, re.DOTALL)
    if fence:
        return _without_extra_closers(fence.group(1).strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if 0 <= start < end:
        return _without_extra_closers(stripped[start:end + 1])
    return text


#: What may follow a complete object and be dropped: whitespace and closers.
_EXTRA_CLOSERS = re.compile(r"[\s}\]]+")


def _without_extra_closers(text: str) -> str:
    """`text` less any closing braces or brackets after a complete JSON object.

    Found live (v0.6.0): Haiku 5.5, writing a `Discovery` or
    `Person` without a grammar (both schemas are too large to compile), ended
    a valid object with one `}` too many in a large share of replies, and a re-ask
    repeated it: `Invalid JSON: trailing characters at line 1 column 4142`,
    the input ending `...firearms."}}` where the object had closed one brace
    earlier. The first complete object is kept (`json.JSONDecoder.raw_decode`)
    only when everything after it is whitespace and `}`/`]`, with a warning
    naming how many characters went; anything else after it (prose, a second
    object) is left in place and still fails validation, as does a truncated
    object, which does not decode at all. A grammar-constrained reply never
    has the tail, so this only ever changes a prompt-only one."""
    stripped = text.strip()
    try:
        json.loads(stripped)
        return text
    except ValueError:
        pass
    try:
        _, end = json.JSONDecoder().raw_decode(stripped)
    except ValueError:
        return text
    tail = stripped[end:]
    if not tail or not _EXTRA_CLOSERS.fullmatch(tail):
        return text
    dropped = len(tail.strip())
    log.warning("Reply carried %d extra closing character(s) after a complete JSON object (%r); "
                "kept the object, dropped them", dropped, tail.strip()[:20])
    return stripped[:end]
