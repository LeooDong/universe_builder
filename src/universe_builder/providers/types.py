"""Provider-agnostic request/response types.

The important design constraint here is prompt-cache friendliness. Every provider
caches on an exact *prefix* match, so a request is split into content that is
stable across calls (system text, JSON schema, one-shot example) and content that
varies per call (the topic, or the JSON being fixed). Providers place their cache
breakpoint at the boundary. Nothing volatile may appear before that boundary.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Type

from pydantic import BaseModel

from universe_builder.providers.errors import ProviderError

log = logging.getLogger(__name__)


#: Output budget for one atom, shared by the synchronous and batch paths.
#: Measured, not guessed: a full `Discovery` or `City` from gpt-5-nano
#: consumed a little over 8K output tokens — almost all of it
#: reasoning, since a reasoning model thinks before it emits. At 8192 both
#: truncated and returned nothing, wasting the whole call. 16384 leaves roughly
#: 2x headroom over the worst case observed.
#:
#: **It lives here, not in `llm_classes`, because the batch path never imports
#: that module.** It used to, and the batch path silently kept the old 8192 —
#: so a default `run_batch submit --atom discovery` would have truncated nearly
#: every record, and a batch reports that 24 hours later as a validation failure
#: rather than immediately. Found during the first live batch run.
#:
#: Checked against Anthropic (the Models API's `max_tokens`): every
#: current Claude model caps output at 64K (Haiku 4.5, and the 4.5 Sonnet and
#: Opus) or 128K (4.6 and later), above this and above `pipeline.yaml`'s 32,768.
#: The caps are recorded as `max_output` in `model_pricing.json`, and the
#: Anthropic provider holds a request to its model's cap, since a request over
#: it is rejected outright rather than truncated.
DEFAULT_MAX_TOKENS = 16384


class SchemaMismatch(ProviderError):
    """A batch's results no longer match the schema it was submitted under."""


@dataclass(frozen=True)
class Example:
    """One-shot example turn. Stable per atom type, so it sits inside the cached prefix."""

    user: str
    assistant_json: str


@dataclass(frozen=True)
class LLMRequest:
    """A single completion request, independent of provider.

    Attributes are ordered by cache stability: `system` and `examples` are the
    stable prefix, `user` is the volatile suffix.
    """

    model: str
    schema_model: Type[BaseModel]
    system: str
    user: str
    examples: tuple[Example, ...] = ()
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: Optional[float] = None
    cache: bool = True
    cache_ttl: str = "1h"
    custom_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Client-side timeout in seconds for the synchronous call, applied per
    #: attempt. None means "the run policy decides" (`providers.run.Policy`);
    #: whatever reaches the provider is handed to the SDK per request. With
    #: streaming this bounds the wait *between* chunks, not the generation as a
    #: whole, so a long response is fine and a dead socket is not.
    timeout: Optional[float] = None
    #: How hard the model is asked to think (v0.6.0): `low`, `medium`, `high`,
    #: or `xhigh`/`max` where the provider has them (`providers.effort`). None,
    #: the default, sends nothing: the request is what it was before v0.6.0.
    #: Not part of `cache_key`, which fingerprints the cached prefix only.
    effort: Optional[str] = None
    #: An advisory output budget in tokens (v0.6.1, `providers.task_budget`):
    #: Anthropic's task budget, which the model sees as a countdown and paces
    #: itself against; `max_tokens` stays the hard cap. At least 20,000. None,
    #: the default, sends nothing: the request is what it was before v0.6.1.
    #: Not part of `cache_key`, like `effort`.
    task_budget: Optional[int] = None
    #: Let `task_budget` exceed `max_tokens` (v0.6.1). Off: an engine call is
    #: a single response, which can never spend more than its cap, so such a
    #: budget is refused. On only for a pack's own agent loop, where the
    #: budget spans several responses.
    allow_task_budget_over_max_tokens: bool = False

    def __post_init__(self) -> None:
        if self.effort is not None:
            from universe_builder.providers.effort import validate

            object.__setattr__(self, "effort", validate(self.effort))
        if self.task_budget is not None:
            from universe_builder.providers.task_budget import validate as validate_budget

            object.__setattr__(self, "task_budget", validate_budget(self.task_budget))

    def cache_key(self) -> str:
        """Fingerprint of the stable prefix.

        Two requests sharing this value share a cache entry. Use it to assert in
        tests (and in logs) that a batch of calls really does share a prefix.
        """
        parts = [
            self.model,
            self.schema_model.__name__,
            json.dumps(self.schema_model.model_json_schema(), sort_keys=True),
            self.system,
            *(f"{e.user}\x00{e.assistant_json}" for e in self.examples),
        ]
        return hashlib.sha256("\x01".join(parts).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class LLMResponse:
    """Result of one completion.

    `parsed` is None when the provider returned text that failed schema validation;
    `raw_text` always holds what came back so the caller can write an _ERROR file.
    """

    raw_text: str
    parsed: Optional[BaseModel]
    model: str
    usage: Usage
    custom_id: Optional[str] = None
    error: Optional[str] = None
    #: True when no response arrived at all: every transport attempt raised
    #: (`providers.run.complete`). `raw_text` is then empty and `error` is the
    #: last exception, and there is nothing to repair from.
    transport_failed: bool = False

    @property
    def ok(self) -> bool:
        return self.parsed is not None


#: Keys `schema_fingerprint` writes into `BatchHandle.extra`.
SCHEMA_PIN_KEYS = ("schema_model", "schema_id", "schema_sha256")


def schema_fingerprint(model: Type[BaseModel]) -> dict[str, str]:
    """Pin the identity of the schema a batch is being submitted under.

    A batch can take 24 hours to come back. Recording only `schema_model` — the
    bare class name — means a model edited in the meantime is indistinguishable
    from the one that was submitted: `fetch_batch_results` looks the name up in
    the registry, gets the *new* class, and validates day-old responses against
    it. A field added since submission then fails every record as "missing", and
    a field removed since submission is silently dropped from every record that
    still carries it. Neither failure names its cause.

    `schema_id` catches the class being moved or replaced by a same-named class
    in another module; `schema_sha256` catches its shape changing in place.

    **The hash is taken over `model.model_json_schema()`, the plain Pydantic
    form — deliberately not `Provider.strict_json_schema()`.** The strict form is
    a provider-side transform (`additionalProperties`, exhaustive `required`,
    `$ref` inlining) applied on the way out. Hashing it would make an edit to
    that transform — a change in this repo's transport code, not in the data
    shape — read as a schema change, and every batch in flight would be refused
    for a difference that cannot affect how the results validate. What the
    collect path validates against is the model, so the model's own schema is
    what the pin has to track.

    `sort_keys` normalises mapping order, so the same schema emitted with its
    properties in a different order hashes the same. It does not touch list
    order, and `required` is a list in field-declaration order — so before
    the normalisation reordering a model's fields tripped the pin and refused a batch
    whose results validate identically. `_normalised` now sorts every
    `required` list, at every depth (`$defs` included), before hashing. Other
    lists keep their order: in `enum`, `anyOf` or `prefixItems` it can carry
    meaning. Handles pinned before carry the unsorted hash, which
    `BatchHandle.schema_mismatch` still accepts (`legacy_schema_sha256`).
    """
    return {
        "schema_model": model.__name__,
        "schema_id": f"{model.__module__}:{model.__qualname__}",
        "schema_sha256": _schema_sha256(_normalised(model.model_json_schema())),
    }


def _normalised(schema: Any) -> Any:
    """`schema` with every `required` list sorted, recursively. A property
    that happens to be called `required` maps to a schema object, not a list
    of names, and is walked rather than sorted."""
    if isinstance(schema, dict):
        return {key: (sorted(value) if key == "required" and isinstance(value, list)
                      and all(isinstance(v, str) for v in value)
                      else _normalised(value))
                for key, value in schema.items()}
    if isinstance(schema, list):
        return [_normalised(value) for value in schema]
    return schema


def _schema_sha256(schema: Any) -> str:
    canonical = json.dumps(schema, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def legacy_schema_sha256(model: Type[BaseModel]) -> str:
    """The pre-normalisation hash, over `required` in declaration order.

    Kept only so a handle submitted by an engine before the normalisation (a
    batch in flight across the upgrade) still matches the unchanged model it
    was pinned against. New handles never carry it.
    """
    return _schema_sha256(model.model_json_schema())


@dataclass
class BatchHandle:
    """Reference to a submitted batch, durable enough to survive process exit."""

    provider: str
    batch_id: str
    model: str
    submitted_at: str
    request_count: int
    # custom_id -> caller-supplied routing info (e.g. the output file path)
    routing: dict[str, dict[str, Any]] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "batch_id": self.batch_id,
            "model": self.model,
            "submitted_at": self.submitted_at,
            "request_count": self.request_count,
            "routing": self.routing,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BatchHandle":
        return cls(
            provider=d["provider"],
            batch_id=d["batch_id"],
            model=d["model"],
            submitted_at=d["submitted_at"],
            request_count=d["request_count"],
            routing=d.get("routing", {}),
            extra=d.get("extra", {}),
        )

    def schema_mismatch(self, model: Type[BaseModel]) -> Optional[str]:
        """Say how `model` differs from the schema pinned at submit time, or None.

        None also means "cannot tell": a handle written before pinning existed
        carries `schema_model` and nothing else. Those are reported once, at
        warning level, and allowed through — refusing them would strand batches
        that were submitted and paid for under the old handle format, which is a
        worse outcome than validating them the way the code always has.
        """
        pinned_sha = self.extra.get("schema_sha256")
        if not pinned_sha:
            log.warning(
                "Batch %s predates schema pinning (extra has no schema_sha256); its results "
                "will be validated against the current %s, which may have changed since "
                "submission on %s.",
                self.batch_id,
                model.__name__,
                self.submitted_at,
            )
            return None

        current = schema_fingerprint(model)
        differences = []
        pinned_id = self.extra.get("schema_id")
        if pinned_id and pinned_id != current["schema_id"]:
            differences.append(f"identity {pinned_id} -> {current['schema_id']}")
        if pinned_sha not in (current["schema_sha256"], legacy_schema_sha256(model)):
            differences.append(
                f"sha256 {pinned_sha[:12]} -> {current['schema_sha256'][:12]}"
            )
        if not differences:
            return None

        return (
            f"Batch {self.batch_id} was submitted on {self.submitted_at} against "
            f"{self.extra.get('schema_model', '?')}, whose schema has since changed "
            f"({'; '.join(differences)}). Collecting it now would validate those results "
            f"against a model that no longer describes them. Restore the schema, or "
            f"re-record the pin on the handle once you have confirmed the change is "
            f"backwards compatible."
        )


def require_matching_schema(handle: BatchHandle, model: Type[BaseModel]) -> None:
    """Refuse to collect a batch whose schema moved underneath it.

    Raised before any result is read, so a refusal costs nothing: the results
    stay on the provider and the batch can be collected again once the schema
    question is settled. Validating them silently against the new model, by
    contrast, writes files that look correct and are not.
    """
    problem = handle.schema_mismatch(model)
    if problem:
        log.error("%s", problem)
        raise SchemaMismatch(problem)


@dataclass(frozen=True)
class BatchStatus:
    """Poll result. `done` means results are ready to fetch."""

    done: bool
    state: str
    succeeded: int = 0
    errored: int = 0
    processing: int = 0
