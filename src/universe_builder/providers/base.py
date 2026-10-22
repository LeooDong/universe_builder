"""The Provider contract.

A provider does one job: transport. It takes an `LLMRequest` that prompt-assembly
already built and puts it on the wire. It does not know about atom types, file
naming, or the pipeline. Adding a provider means implementing `complete()` plus,
optionally, the three batch methods.
"""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from abc import ABC, abstractmethod
from typing import Any, Iterable, Iterator, Optional, Sequence

from pydantic import BaseModel

from universe_builder.providers.errors import BatchNotSupported
from universe_builder.providers.types import BatchHandle, BatchStatus, LLMRequest, LLMResponse, Usage

log = logging.getLogger(__name__)


class Provider(ABC):
    """Base class for all providers."""

    name: str = "unset"
    #: Maximum requests in a single batch submission.
    batch_max_requests: int = 0
    #: Maximum serialised payload for one batch submission, in bytes. Each batch
    #: line carries the whole prompt - system text, schema and examples - so the
    #: byte limit binds long before the request-count limit does.
    batch_max_bytes: int = 0

    #: The effort levels this provider translates (v0.6.0, `providers.effort`).
    #: Empty: it has no effort setting, and a request's effort is ignored with
    #: one warning per model.
    EFFORT_LEVELS: tuple = ()
    #: Whether this provider sends a task budget (v0.6.1,
    #: `providers.task_budget`). False: a request's budget is ignored with one
    #: warning per model. Anthropic only.
    TASK_BUDGET: bool = False

    @property
    def supports_batch(self) -> bool:
        return self.batch_max_requests > 0

    def effort_for(self, request: LLMRequest) -> Optional[str]:
        """The effort this provider puts on the wire for `request`, or None to
        send none (`providers.effort.wire`). Raises `EffortNotSupported` for a
        level the provider lacks."""
        from universe_builder.providers.effort import wire

        return wire(self.name, self.EFFORT_LEVELS, request.model, request.effort)

    def sent_max_tokens(self, request: LLMRequest) -> int:
        """The `max_tokens` this provider puts on the wire for `request`."""
        return request.max_tokens

    def task_budget_for(self, request: LLMRequest) -> Optional[int]:
        """The task budget this provider puts on the wire for `request`, or
        None to send none (`providers.task_budget.wire`). Raises
        `TaskBudgetRefused` for one that cannot be sent as asked."""
        from universe_builder.providers.task_budget import wire

        return wire(self.name, self.TASK_BUDGET, request.model, request.task_budget,
                    self.sent_max_tokens(request), request.allow_task_budget_over_max_tokens)

    # -- synchronous ------------------------------------------------------------

    @abstractmethod
    def complete(self, request: LLMRequest) -> LLMResponse:
        """Run one request and return the validated result."""

    # -- batch ------------------------------------------------------------------

    def submit_batch(self, requests: Sequence[LLMRequest]) -> BatchHandle:
        raise BatchNotSupported(f"{self.name} has no batch API")

    def poll_batch(self, handle: BatchHandle) -> BatchStatus:
        raise BatchNotSupported(f"{self.name} has no batch API")

    def fetch_batch_results(self, handle: BatchHandle) -> Iterator[LLMResponse]:
        raise BatchNotSupported(f"{self.name} has no batch API")

    def cancel_batch(self, handle: BatchHandle) -> BatchStatus:
        """Ask the provider to stop the batch (v0.6.2). What already finished
        stays collectable: the batch ends cancelled with its partial results."""
        raise BatchNotSupported(f"{self.name} has no batch API")

    # -- shared helpers ---------------------------------------------------------

    #: Provider-specific values of the "ran out of output budget" stop signal.
    #: OpenAI says `length`, Anthropic `max_tokens`, Google `MAX_TOKENS`.
    TRUNCATION_REASONS = frozenset({"length", "max_tokens", "MAX_TOKENS"})

    @classmethod
    def _validate(
        cls,
        raw_text: str,
        request: LLMRequest,
        usage: Usage,
        finish_reason: Optional[str] = None,
    ) -> LLMResponse:
        """Parse raw model output against the request's Pydantic schema.

        `finish_reason` is the provider's own stop signal, passed through so a
        truncated response can be reported as truncation. Without it a response
        cut off mid-object surfaces as `Invalid JSON: EOF while parsing`, which
        points the reader at the schema when the actual fault is the token
        budget - and on a reasoning model the budget can be consumed entirely by
        reasoning tokens, leaving the content empty and the cause invisible.
        """
        try:
            parsed = request.schema_model.model_validate_json(raw_text)
            return LLMResponse(raw_text, parsed, request.model, usage, request.custom_id)
        except Exception as e:  # noqa: BLE001 - any validation failure is reported, not raised
            error = cls._truncation_error(
                request.model, request.max_tokens, usage, finish_reason
            ) or cls._refusal_error(request.model, finish_reason) or str(e)
            log.warning("Validation failed for %s: %s", request.custom_id or "<no id>", error)
            return LLMResponse(raw_text, None, request.model, usage, request.custom_id, error)

    @classmethod
    def _validate_stored(
        cls,
        text: str,
        schema_model: type[BaseModel],
        model: str,
        usage: Usage,
        custom_id: Optional[str],
        finish_reason: Optional[str] = None,
    ) -> LLMResponse:
        """`_validate` for a batch result read back from the provider.

        One place for every provider's collect path. The
        stored result does not echo `max_tokens`, so the truncation message
        says "its output limit". Every provider passes its line's stop signal
        as `finish_reason` (Anthropic's `stop_reason`, Google's candidate
        `finish_reason`, OpenAI's choice `finish_reason`; v0.4.0), so a reply
        cut off by its budget is named as truncation, not as a schema fault.
        """
        try:
            parsed = schema_model.model_validate_json(text)
            return LLMResponse(text, parsed, model, usage, custom_id)
        except Exception as e:  # noqa: BLE001 - any validation failure is reported, not raised
            error = (cls._truncation_error(model, None, usage, finish_reason)
                     or cls._refusal_error(model, finish_reason) or str(e))
            return LLMResponse(text, None, model, usage, custom_id, error)

    @classmethod
    def _truncation_error(
        cls,
        model: str,
        max_tokens: Optional[int],
        usage: Usage,
        finish_reason: Optional[str],
    ) -> Optional[str]:
        """Describe a budget exhaustion, or None if that is not what happened.

        `max_tokens` is Optional because the batch result path reads a stored
        response, which does not echo the limit the request was submitted with.
        """
        if finish_reason not in cls.TRUNCATION_REASONS:
            return None
        limit = f"the {max_tokens}-token limit" if max_tokens else "its output limit"
        detail = (
            f"Response truncated: {model} stopped at {limit} ({finish_reason!r}) "
            f"after emitting {usage.output_tokens} output tokens"
        )
        if not usage.output_tokens or (max_tokens and usage.output_tokens >= max_tokens):
            detail += (
                ". On a reasoning model the whole budget can go to reasoning tokens, "
                "leaving the content empty"
            )
        advice = (
            f"Raise max_tokens above {max_tokens} and retry."
            if max_tokens
            else "Resubmit with a higher max_tokens."
        )
        return f"{detail}. {advice}"

    #: The stop signal of a reply a safety classifier declined (Anthropic's
    #: `stop_reason`; HTTP 200, billed, and the content is not an answer).
    REFUSAL_REASONS = frozenset({"refusal"})

    @classmethod
    def _refusal_error(cls, model: str, finish_reason: Optional[str]) -> Optional[str]:
        """Name a refused reply, or None if that is not what happened. Without
        it a refusal reads as `Invalid JSON`, which points at the schema."""
        if finish_reason not in cls.REFUSAL_REASONS:
            return None
        return (f"Response refused: {model} stopped with {finish_reason!r} (a safety "
                "classifier declined the request), so the output is not an answer. A "
                "re-ask may pass; a repeated refusal means the subject or prompt needs a look.")

    #: What follows the request's system text in every provider's system turn.
    SCHEMA_PREAMBLE = "Return JSON matching this schema exactly:\n"

    def system_with_schema(self, request: LLMRequest) -> str:
        """The request's system text with its JSON schema appended: the
        stable, cached head of every provider's prompt. One spelling for all
        three providers; each had its own copy, byte for
        byte the same."""
        return (
            f"{request.system}\n\n"
            f"{self.SCHEMA_PREAMBLE}"
            f"{self._dumps(self.json_schema(request.schema_model))}"
        )

    @staticmethod
    def json_schema(model: type[BaseModel]) -> dict[str, Any]:
        return model.model_json_schema()

    @classmethod
    def strict_json_schema(cls, model: type[BaseModel]) -> dict[str, Any]:
        """Pydantic schema normalised for OpenAI-style strict structured outputs.

        Strict mode requires every object to set `additionalProperties: false` and
        to list every property in `required`. Pydantic already emits Optional
        fields as `anyOf: [..., {"type": "null"}]`, which satisfies strict mode's
        "use a nullable type rather than omitting the key" rule.

        It also rejects `allOf` outright, and rejects `$ref` carrying sibling
        keys. Pydantic emits exactly that combination for any field that both
        references another model and has a description::

            "ageOfRecord": {"allOf": [{"$ref": "#/$defs/Date"}],
                            "description": "yyyy-MM-dd of the earliest record"}

        `_flatten_refs` rewrites those nodes by inlining the referenced
        definition and keeping the field's own description. Dropping the
        description would also satisfy the API, but descriptions are the
        per-field instructions the model actually reads, so they are worth the
        duplication. Verified against gpt-5-nano: sibling keys are
        a 400, inlining is accepted.
        """
        schema = cls.json_schema(model)
        cls._flatten_refs(schema, schema.get("$defs") or {})

        def walk(node: Any) -> None:
            if isinstance(node, dict):
                if node.get("type") == "object" or "properties" in node:
                    props = node.get("properties")
                    if isinstance(props, dict):
                        node["additionalProperties"] = False
                        node["required"] = list(props.keys())
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for item in node:
                    walk(item)

        walk(schema)
        return schema

    @classmethod
    def _flatten_refs(cls, node: Any, defs: dict[str, Any], _seen: frozenset = frozenset()) -> None:
        """Remove `allOf`, and inline any `$ref` that carries sibling keys.

        `_seen` tracks the definitions already inlined on the current branch. A
        self-referencing model would otherwise expand forever; on a cycle the
        siblings are dropped and the bare `$ref` kept, which the API accepts.
        """
        if isinstance(node, list):
            for item in node:
                cls._flatten_refs(item, defs, _seen)
            return
        if not isinstance(node, dict):
            return

        # `allOf: [X]` with siblings is Pydantic's "$ref plus metadata" shape.
        subschemas = node.pop("allOf", None)
        if isinstance(subschemas, list):
            for sub in subschemas:
                if isinstance(sub, dict):
                    for key, value in sub.items():
                        node.setdefault(key, value)

        ref = node.get("$ref")
        if isinstance(ref, str) and len(node) > 1:
            name = ref.rsplit("/", 1)[-1]
            target = defs.get(name)
            if target is None or name in _seen:
                # Unresolvable, or we are inside this definition already. Keep
                # the reference and drop the siblings the API will not accept.
                for key in [k for k in node if k != "$ref"]:
                    del node[key]
            else:
                del node["$ref"]
                inlined = deepcopy(target)
                cls._flatten_refs(inlined, defs, _seen | {name})
                for key, value in inlined.items():
                    node.setdefault(key, value)

        for value in node.values():
            cls._flatten_refs(value, defs, _seen)

    @staticmethod
    def _chunk(items: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
        for i in range(0, len(items), size):
            yield items[i:i + size]

    def request_size_bytes(self, request: LLMRequest) -> int:
        """Serialised size of one request inside a batch submission.

        Subclasses override this where their wire format differs; the default is
        a reasonable proxy for anything JSON-shaped.
        """
        payload = {
            "system": request.system,
            "user": request.user,
            "examples": [(e.user, e.assistant_json) for e in request.examples],
            "schema": self.json_schema(request.schema_model),
        }
        return len(self._dumps(payload).encode("utf-8"))

    def chunk_for_batch(self, requests: Sequence[LLMRequest]) -> list[Sequence[LLMRequest]]:
        """Split requests into submittable batches, honouring both limits.

        Chunking on request count alone silently builds an oversized upload:
        every batch line carries the whole prompt, so at a ~4k-token prefix
        50,000 OpenAI requests serialise to roughly 1.4 GB against a 200 MB cap.
        """
        if not requests:
            return []

        chunks: list[Sequence[LLMRequest]] = []
        current: list[LLMRequest] = []
        current_bytes = 0

        for request in requests:
            size = self.request_size_bytes(request)
            if self.batch_max_bytes and size > self.batch_max_bytes:
                raise ValueError(
                    f"A single request serialises to {size:,} bytes, over "
                    f"{self.name}'s {self.batch_max_bytes:,} byte batch limit"
                )
            over_count = len(current) >= self.batch_max_requests
            over_bytes = bool(self.batch_max_bytes) and current_bytes + size > self.batch_max_bytes
            if current and (over_count or over_bytes):
                chunks.append(current)
                current, current_bytes = [], 0
            current.append(request)
            current_bytes += size

        if current:
            chunks.append(current)

        if len(chunks) > 1:
            log.warning(
                "Split %d requests into %d batches (limits: %s requests, %s bytes)",
                len(requests),
                len(chunks),
                f"{self.batch_max_requests:,}",
                f"{self.batch_max_bytes:,}" if self.batch_max_bytes else "none",
            )
        return chunks

    @staticmethod
    def _dumps(obj: Any) -> str:
        """Deterministic JSON. Unsorted keys would change the cached prefix bytes."""
        return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def require(module: str, package: Optional[str] = None):
    """Import an optional SDK, raising a useful error if it is missing."""
    from importlib import import_module

    from universe_builder.providers.errors import ProviderNotInstalled

    try:
        return import_module(module)
    except ImportError as e:
        raise ProviderNotInstalled(
            f"{module} is not installed. Run: pip install {package or module}"
        ) from e
