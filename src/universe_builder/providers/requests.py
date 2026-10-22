"""Pipeline-aware request building.

This is where the prompt-cache split is decided, and it is the reason the module
exists separately from the providers. Everything stable for an atom type - the
historian instructions, the operation instructions, the JSON schema, the one-shot
example - goes in `system`/`examples`. Everything that varies per call - the
topic, the JSON being fixed - goes in `user`.

The existing `execute_prompt.py` gets this backwards for fix/verify/expand: it
interpolates the per-file topic into `base_system_text`, which then goes into the
cached system block. Since the cached prefix must match byte for byte, a topic in
the system block means the cache never hits on any of those passes.

All paths here resolve through `universe_builder.settings`, so this module works regardless of the
working directory.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import logging
from functools import lru_cache
from typing import Any, Optional, Sequence

from universe_builder import settings
from universe_builder.providers.errors import ProviderError
from universe_builder.providers.registry import schema_model_by_name
from universe_builder.providers.types import DEFAULT_MAX_TOKENS, Example, LLMRequest

log = logging.getLogger(__name__)

#: Roughly the average characters per token for English prose plus JSON. Used
#: only to warn about prefixes too short to cache - never for billing.
_CHARS_PER_TOKEN = 4


@lru_cache(maxsize=1)
def base_system_text() -> str:
    """The domain persona. The active pack's `base_system` wins; otherwise the
    package default file — so history is unchanged and a second domain speaks in
    its own voice rather than as a historian."""
    from universe_builder.registry import active_pack

    pack_text = active_pack().base_system
    if pack_text:
        return pack_text.strip()
    return settings.BASE_SYSTEM.read_text(encoding="utf-8").strip()


def _examples_dir():
    """Where to load one-shot examples from: the active pack's dir, else the
    package default. A second domain ships its own examples."""
    from universe_builder.registry import active_pack

    return active_pack().examples_dir or settings.EXAMPLES_DIR


def clear_pack_caches() -> None:
    """Drop caches keyed on pack-specific resources, after a pack switch. Called
    by `atoms.reload_from_pack`, so a domain change is fully visible here."""
    base_system_text.cache_clear()
    _example.cache_clear()
    operation_text.cache_clear()


@lru_cache(maxsize=8)
def operation_text(operation: str) -> str:
    """The active pack's text for `operation` (v0.5.0), else the engine's file."""
    from universe_builder.registry import active_pack

    pack_text = (active_pack().operation_texts or {}).get(operation)
    if pack_text is not None:
        return pack_text.strip()
    path = settings.OPERATIONS_DIR / f"{operation}.txt"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=64)
def _example(filename: str, schema_model: Optional[type] = None) -> str:
    """Load an example, validating it against the atom's model.

    An example that does not match its schema silently teaches the model the
    wrong shape, so this fails loudly at build time rather than showing up as a
    pile of validation errors after a batch has been paid for. Cached per
    (file, model), so a 20,000-request batch validates each example once.
    """
    with open(_examples_dir() / filename, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if schema_model is not None:
        try:
            schema_model.model_validate(payload)
        except Exception as e:  # noqa: BLE001
            raise ProviderError(
                f"Example {filename} does not validate against "
                f"{schema_model.__name__}: {e}"
            ) from e
    return json.dumps(payload, ensure_ascii=False)


def _spec(atom_type: str) -> dict[str, Any]:
    """The generating metadata for an atom type, from the active domain pack.

    Was `prompt_config.json`; now derived from `atoms.atom_mapping` so the pack
    is the single source of truth (the JSON no longer feeds the runtime). Read
    live, not cached, because a pack switch rebuilds `atom_mapping` in place.
    """
    from universe_builder import atoms

    try:
        m = atoms.atom_mapping[atom_type]
    except KeyError as e:
        raise ProviderError(
            f"Unknown atom type {atom_type!r}. Known: {', '.join(sorted(atoms.atom_mapping))}"
        ) from e
    return {
        "atom_class": m.atom_class.__name__,
        "prompt_start_text": m.prompt_start_text,
        "example_json_file": m.example_json_file,
        "plural": m.plural,
    }


def _examples_for(spec: dict[str, Any], schema_model: Optional[type] = None) -> tuple[Example, ...]:
    """One-shot or few-shot examples for an atom type.

    `example_json_file` accepts a single filename or a list of them. Every
    example sits inside the cached prefix, so a second and third example cost
    roughly a tenth of what they would uncached - and they push the prefix past
    the cache minimums that most atoms currently fall short of.

    The order must be stable: caching is a byte-exact prefix match, so rotating
    or shuffling examples between calls throws the cache away entirely.
    """
    configured = spec.get("example_json_file")
    if not configured:
        return ()
    filenames = [configured] if isinstance(configured, str) else list(configured)
    examples = []
    for filename in filenames:
        pretty = filename.removesuffix(".json").replace("_", " ")
        examples.append(
            Example(
                user=f"{spec['prompt_start_text']}'{pretty}'",
                assistant_json=_example(filename, schema_model),
            )
        )
    return tuple(examples)


def build_request(
    atom_type: str,
    subject: str,
    model: str,
    *,
    operation: str = "create",
    payload: Optional[str] = None,
    user_instructions: str = "",
    context: Optional[str] = None,
    schema_model: Optional[Any] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: Optional[float] = None,
    cache: bool = True,
    custom_id: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
    prompt: Optional[Any] = None,
    effort: Optional[str] = None,
    task_budget: Optional[int] = None,
) -> LLMRequest:
    """Build one request for `atom_type`.

    Args:
        atom_type: an atom name in the active pack, e.g. "person" or "append_city".
        subject: the topic. Always volatile - goes in the user turn.
        model: model ID; decides which provider serves the request.
        operation: create | fix | verify | expand | append.
        payload: existing JSON, for the operations that rewrite a file.
        context: extra stable framing, e.g. the civilisation. Cached with the
            prefix, so keep it constant across a run or it defeats caching.
        schema_model: the model to ask for, by name (a batch handle records
            names) or as the class itself (the synchronous path holds the
            class it resolved from the stored shape).
        prompt: a `prompt_spec.PromptSpec` (v0.5.0, prompt experiments): the
            persona, operation text, examples and prompt start come from it,
            read fresh rather than through this module's caches, so two specs
            in one process can never hand each other their bytes. None, the
            only value the pipeline passes, is the live pack exactly as before.
        effort: the reasoning effort (v0.6.0, `providers.effort`); None sends
            none, and the request is byte for byte what it was before.
        task_budget: the advisory output budget (v0.6.1,
            `providers.task_budget`); None sends none, likewise.

    The one prompt builder: the synchronous path (`ProviderCall.build_request`)
    delegates here too, so a phase sends the same
    prompt whether it runs synchronously or as a batch.
    """
    spec = _spec(atom_type)
    # `schema_model` overrides the atom type's default class. The augment
    # operations need it: a city that has been through expand is stored as
    # `CityExpanded`, and asking the model for the seed `City` shape is what B7
    # was — the answer validates, overwrites the file, and silently collapses the
    # expansion back to a list of period names.
    if not isinstance(schema_model, type):
        schema_model = schema_model_by_name(schema_model or spec["atom_class"])

    if prompt is not None:
        spec = dict(spec, prompt_start_text=prompt.prompt_start(atom_type, spec["prompt_start_text"]))
    system_parts = [prompt.persona() if prompt is not None else base_system_text()]
    if context:
        system_parts.append(f"The context is {context}.")
    op_text = prompt.operation_text(operation) if prompt is not None else operation_text(operation)
    if op_text:
        system_parts.append(op_text)
    system = "\n\n".join(system_parts)

    examples = (prompt.examples(atom_type, spec, schema_model) if prompt is not None
                else _examples_for(spec, schema_model))

    user_parts = [f"{spec['prompt_start_text']}'{subject}'"]
    if payload:
        user_parts.append(f"The subject is {subject}. Here is the JSON to work on:\n{payload}")
    if user_instructions:
        user_parts.append(user_instructions)
    user = "\n\n".join(user_parts)

    return LLMRequest(
        model=model,
        schema_model=schema_model,
        system=system,
        user=user,
        examples=examples,
        max_tokens=max_tokens,
        temperature=temperature,
        cache=cache,
        custom_id=custom_id,
        metadata=metadata or {"atom_type": atom_type, "subject": subject},
        effort=effort,
        task_budget=task_budget,
    )


@dataclass(frozen=True)
class BatchItem:
    """One unit of batch work that is more than a bare topic.

    `create` only needs a subject — the destination follows from it, because ids
    are content-addressed. The augment operations do not: `fix`, `verify`,
    `expand` and `append` each rewrite *a particular file*, and that file's model
    version, decorations and `_ERROR` state cannot be re-derived from the subject
    alone. `target` carries it, and `collect` writes the result back there rather
    than to a path it reconstructed and hoped was the same one.
    """

    subject: str
    #: Existing JSON the operation works on. Goes in the user turn, never the
    #: cached system prefix.
    payload: Optional[str] = None
    #: Where the result must be written. Relative to the data directory, so a
    #: handle stays valid if the corpus moves.
    target: Optional[str] = None
    #: Extra routing merged into the request metadata (and so into the handle),
    #: for collect to read back. `expand` uses it to carry which period of the
    #: target file this request expands, since many requests share one target.
    routing: Optional[dict] = None
    #: Operation-specific volatile instructions (e.g. which list entry to expand).
    user_instructions: str = ""


def with_create_note(atom_type: str, subject: str, context: Optional[str] = None, notes=None, index=None):
    """`subject` as it is, or a `BatchItem` carrying what its create prompt
    adds (`topics.create_instructions`: list neighbours, then the note) in the
    user turn. Only subjects with either change, so without them every request
    in a batch is byte for byte what it was. `index` (a `topics.ListIndex`)
    saves re-reading the index atoms per subject."""
    from universe_builder import topics

    extra = topics.create_instructions(atom_type, subject, context, notes=notes, index=index)
    return BatchItem(subject=subject, user_instructions=extra) if extra else subject


def build_requests(
    atom_type: str,
    subjects: Sequence[Any],
    model: str,
    **kwargs: Any,
) -> list[LLMRequest]:
    """Build one request per subject, with stable `custom_id`s for batch routing.

    Accepts plain strings or `BatchItem`s. The per-item `payload` and `target`
    override anything passed in `kwargs`, so a caller cannot accidentally send
    one payload for a whole batch.
    """
    requests = []
    for i, entry in enumerate(subjects):
        item = entry if isinstance(entry, BatchItem) else BatchItem(subject=str(entry))
        slug = "".join(c if c.isalnum() else "_" for c in item.subject.lower())[:48]
        per_item = dict(kwargs)
        if item.user_instructions:
            per_item["user_instructions"] = item.user_instructions
        if item.payload is not None:
            per_item["payload"] = item.payload
        metadata = dict(per_item.pop("metadata", None) or {})
        metadata.setdefault("atom_type", atom_type)
        metadata.setdefault("subject", item.subject)
        # Routed into the handle, so `collect` knows what kind of work each
        # result answers: a failed *verify* must not queue its valid target for
        # repair, and collect cannot tell without this.
        metadata.setdefault("operation", str(per_item.get("operation") or "create"))
        if item.target is not None:
            metadata["target"] = item.target
        if item.routing:
            metadata.update(item.routing)
        requests.append(
            build_request(
                atom_type,
                item.subject,
                model,
                custom_id=f"{atom_type}-{i:05d}-{slug}",
                metadata=metadata,
                **per_item,
            )
        )
    return requests


def prefix_tokens(request: LLMRequest) -> int:
    """Approximate size of the cacheable prefix, for cache-minimum warnings."""
    chars = len(request.system)
    chars += len(json.dumps(request.schema_model.model_json_schema()))
    chars += sum(len(e.user) + len(e.assistant_json) for e in request.examples)
    return chars // _CHARS_PER_TOKEN
