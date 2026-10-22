"""Batch submission, polling and collection.

Batch inference is half price on every provider that offers it, and this
pipeline's create pass is the ideal shape for it: thousands of independent
requests with no latency requirement. The catch is that a batch outlives the
process that submitted it - Anthropic and OpenAI both allow up to 24 hours - so
the handle has to be durable. It is written to
`data/automated/<domain>/_batches/<name>.json` and can be collected from any later
process.

    submit("person", topics, model="claude-haiku-4-5-20251001")   # returns a path
    poll(path)                                                    # cheap status check
    collect(path)                                                 # writes the JSON files
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from universe_builder import journal
from universe_builder import settings
from universe_builder import store
from universe_builder.registry import active_pack
from universe_builder.providers.errors import BatchNotSupported, ProviderError, UnknownModel
from universe_builder.providers.pricing import estimate_cost
from universe_builder.providers.registry import get_provider
from universe_builder.providers.requests import BatchItem, build_requests, prefix_tokens
from universe_builder.providers.types import (
    DEFAULT_MAX_TOKENS, BatchHandle, BatchStatus, LLMRequest, SchemaMismatch, Usage,
)
from universe_builder.providers.util import expected_custom_ids

log = logging.getLogger(__name__)

#: Re-exported so existing importers keep working; `universe_builder.store` owns it.
ERROR_SUFFIX = store.ERROR_SUFFIX

_CUSTOM_ID = re.compile(r"^(?P<atom>.+)-(?P<index>\d{5})-(?P<slug>.*)$")


@dataclass
class CollectReport:
    written: int = 0
    failed: int = 0
    #: Submitted custom_ids that never came back. A non-empty list means results
    #: were lost, not that they failed - those are counted in `failed`.
    missing: list[str] = field(default_factory=list)
    usage: Usage = None  # type: ignore[assignment]
    cost: Optional[float] = None
    #: Where the handle was moved once its results were safely written. None
    #: means it stayed put, which is correct for anything incomplete.
    archived: Optional[Path] = None
    #: The batch ended cancelled or expired (v0.6.2): its missing requests never
    #: ran and never will, so the handle is archived with them listed in
    #: `missing_dir()` rather than left for every later `collect --all` to
    #: re-fetch, which would rewrite atoms corrected since.
    ended_short: Optional[str] = None
    #: Set when the batch was refused rather than fetched — currently only when
    #: its schema has moved since submission. Refusal is reported, not raised,
    #: so one bad handle does not abort a sweep over the others.
    refused: Optional[str] = None

    def __post_init__(self) -> None:
        if self.usage is None:
            self.usage = Usage()

    @property
    def complete(self) -> bool:
        """True when every submitted request was accounted for."""
        return not self.missing and self.refused is None


# -- handle persistence ---------------------------------------------------------


def _write(handle: BatchHandle, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(handle.to_dict(), f, indent=2)
    return path


def _save(handle: BatchHandle, name: str) -> Path:
    """Persist a handle under a name that is guaranteed not to be taken.

    Two submissions of the same atom within one second previously produced the
    same filename, and the second silently overwrote the first - orphaning a
    batch that had been submitted and paid for but could no longer be collected.
    """
    settings.batch_dir().mkdir(parents=True, exist_ok=True)
    candidate = settings.batch_dir() / f"{name}.json"
    suffix = 1
    while candidate.exists():
        candidate = settings.batch_dir() / f"{name}-{suffix:02d}.json"
        suffix += 1
    path = _write(handle, candidate)
    log.warning("Batch handle saved to %s", path)
    return path


_HANDLE_KEYS = {"provider", "batch_id", "model", "submitted_at", "request_count"}


def load(path: Path | str) -> BatchHandle:
    """Read a batch handle, failing with a useful message on anything else."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or not _HANDLE_KEYS.issubset(data):
        raise ProviderError(f"{path} is not a batch handle")
    return BatchHandle.from_dict(data)


def list_batches() -> list[Path]:
    """Every readable batch handle in settings.batch_dir().

    Files that are not handles are skipped rather than crashing the caller - the
    directory is on disk and anything could end up in it.
    """
    if not settings.batch_dir().exists():
        return []
    paths = []
    for path in sorted(settings.batch_dir().glob("*.json")):
        if not path.is_file():
            continue
        try:
            load(path)
        except (ProviderError, json.JSONDecodeError, OSError) as e:
            log.warning("Skipping %s: %s", path.name, e)
            continue
        paths.append(path)
    return paths


# -- submit ---------------------------------------------------------------------


def submit(
    atom_type: str,
    subjects: Sequence,
    model: str,
    *,
    operation: str = "create",
    context: Optional[str] = None,
    schema_model: Optional[str] = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    dry_run: bool = False,
    effort: Optional[str] = None,
    task_budget: Optional[int] = None,
) -> list[Path]:
    """Submit `subjects` as one or more batches. Returns the handle paths.

    `subjects` is a sequence of topic strings, or of `requests.BatchItem` when
    the work carries a payload and a destination — which is what the augment
    operations need, since they rewrite a specific existing file.

    Requests are chunked to the provider's per-batch limit. Every request in a
    chunk shares the same cached prefix, which is checked and warned about here
    rather than discovered later in a billing statement.

    `effort` (v0.6.0) goes on every request and into each handle's
    `extra["effort"]`, so `collect` journals it; None sends none. The task
    budget (v0.6.1) the same way, into `extra["task_budget"]`.
    """
    subjects = [
        s if isinstance(s, BatchItem) else s.strip()
        for s in subjects
        if s and (isinstance(s, BatchItem) or s.strip())
    ]
    if not subjects:
        raise ValueError("No subjects to submit")

    provider = get_provider(model)
    if not provider.supports_batch:
        raise BatchNotSupported(
            f"{provider.name} has no batch job API. Use the sync path, or pick a "
            "model served by anthropic, openai or google."
        )

    requests = build_requests(
        atom_type, subjects, model, operation=operation, context=context,
        schema_model=schema_model, max_tokens=max_tokens, effort=effort,
        **({"task_budget": task_budget} if task_budget is not None else {}),
    )
    _warn_about_cache(requests, model)

    if dry_run:
        # The count a real submit would make: chunked on request count *and*
        # serialised bytes (v0.4.0; it counted requests only, so a dry run
        # promised one batch where the byte cap made several).
        log.warning(
            "Dry run: would submit %d %s requests to %s (effort %s, task budget %s) in %d batch(es)",
            len(requests),
            atom_type,
            model,
            requests[0].effort or "unset",
            requests[0].task_budget or "unset",
            len(provider.chunk_for_batch(requests)),
        )
        return []

    # What the provider sends, not what was asked (raises, before anything is
    # submitted, for a level the model lacks): the handle records the wire.
    from universe_builder.providers.effort import sent

    wire_effort = sent(model, effort)
    from universe_builder.providers import task_budget as task_budget_mod

    wire_task_budget = task_budget_mod.sent(model, task_budget, max_tokens)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths: list[Path] = []
    # Chunk on request count *and* serialised bytes - each batch line carries the
    # whole prompt, so the byte cap binds first by a wide margin.
    chunks = provider.chunk_for_batch(requests)
    for i, chunk in enumerate(chunks):
        handle = provider.submit_batch(chunk)
        stamp_effort(handle, wire_effort)
        stamp_task_budget(handle, wire_task_budget)
        suffix = f"-{i:02d}" if len(chunks) > 1 else ""
        paths.append(_save(handle, f"{stamp}-{atom_type}{suffix}"))
    return paths


def stamp_effort(handle: BatchHandle, effort: Optional[str]) -> None:
    """Record the effort a batch's requests carried on the wire
    (`effort.sent`, v0.6.0) on its handle, where `collect` reads it back into
    each result's journal line. Nothing when none was sent, so a handle
    without one is what it always was."""
    if effort:
        handle.extra["effort"] = effort


def stamp_task_budget(handle: BatchHandle, task_budget: Optional[int]) -> None:
    """Record the task budget a batch's requests carried on the wire
    (`task_budget.sent`, v0.6.1) on its handle, for `collect` to journal.
    Nothing when none was sent."""
    if task_budget:
        handle.extra["task_budget"] = task_budget


def _warn_about_cache(requests: Sequence[LLMRequest], model: str) -> None:
    from universe_builder.providers.pricing import cache_warning

    keys = {r.cache_key() for r in requests}
    if len(keys) > 1:
        log.warning(
            "These %d requests span %d distinct cached prefixes - caching will be "
            "far less effective than it could be. Check that nothing per-topic is "
            "leaking into the system text.",
            len(requests),
            len(keys),
        )
    warning = cache_warning(model, prefix_tokens(requests[0]))
    if warning:
        log.warning("%s", warning)


# -- poll -----------------------------------------------------------------------


def poll(path: Path | str) -> BatchStatus:
    handle = load(path)
    status = get_provider(handle.model).poll_batch(handle)
    # poll_batch may enrich the handle (OpenAI records its output file id).
    # Write back to where it was loaded from - saving by stem would relocate a
    # handle passed by an out-of-tree path into settings.batch_dir().
    _write(handle, Path(path))
    log.warning(
        "Batch %s: state=%s succeeded=%d errored=%d",
        handle.batch_id,
        status.state,
        status.succeeded,
        status.errored,
    )
    return status


def cancel(path: Path | str) -> BatchStatus:
    """Cancel a batch still running at the provider (v0.6.2). Its finished
    results stay with the provider; `collect` fetches them once it reads done,
    and the requests that never ran are reported missing, for a sync create."""
    handle = load(path)
    status = get_provider(handle.model).cancel_batch(handle)
    log.warning("Batch %s: cancel requested, state=%s succeeded=%d errored=%d",
                handle.batch_id, status.state, status.succeeded, status.errored)
    return status


#: Terminal states in which the requests that did not run never will: OpenAI's
#: and Google's. Anthropic returns a cancelled or expired request as an errored
#: result, so it is accounted for and never missing.
ENDED_SHORT = frozenset({"cancelled", "expired", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"})


def _ended_short(handle: BatchHandle) -> Optional[str]:
    """The batch's state when it ended cancelled or expired, else None. A
    failed poll answers None: the handle stays in place, as before."""
    try:
        state = get_provider(handle.model).poll_batch(handle).state
    except Exception as e:  # noqa: BLE001 - any failure keeps the handle, the safe side
        log.warning("Batch %s: could not read its state (%s); handle kept in place", handle.batch_id, e)
        return None
    return state if isinstance(state, str) and state in ENDED_SHORT else None


def archive(path: Path | str) -> Path:
    """Move a finished handle out of the glob `collect --all` sweeps.

    Handles were never retired, and `_targets(--all)` returns every `*.json` in
    `batch_dir()`. So a batch collected in January stayed eligible forever: an
    unrelated `collect --all` in March re-fetched it and rewrote its atoms,
    silently reverting any correction made to them in between. Same shape as the
    other write-without-looking bugs found today.

    Moved rather than deleted — the handle is the only record of what a batch
    cost and which subjects it covered — and named non-destructively, because a
    deliberate re-collect produces a second handle for the same stem.
    """
    path = Path(path)
    settings.collected_dir().mkdir(parents=True, exist_ok=True)
    target = settings.collected_dir() / path.name
    suffix = 1
    while target.exists():
        target = settings.collected_dir() / f"{path.stem}-{suffix:02d}{path.suffix}"
        suffix += 1
    path.replace(target)
    log.warning("Handle archived to %s; `collect --all` will skip it from now on", target)
    return target


# -- collect --------------------------------------------------------------------


def _output_path(handle: BatchHandle, custom_id: str) -> Path:
    """Reproduce the pipeline's filename convention from the batch custom_id.

    Uses the same content-addressed id as the synchronous pass. It used to use
    the index embedded in the custom_id, which is the subject's position in the
    submitted list — stable within one submission but not across two. That meant
    generating a topic synchronously and then again in a batch produced two files
    for one topic, differing only in an id that encoded nothing about the
    content. Deriving the id from the subject makes the two paths agree.
    """
    match = _CUSTOM_ID.match(custom_id)
    routing = handle.routing.get(custom_id, {})
    atom_type = routing.get("atom_type") or (match.group("atom") if match else "unknown")
    subject = routing.get("subject") or (match.group("slug") if match else custom_id)

    # An augment operation rewrites *a particular file*, and that file's model
    # version, decorations and `_ERROR` state cannot be rebuilt from the subject
    # — `store.path_for` would reconstruct a plausible path rather than the one
    # the payload came from. When the submitter recorded a target, honour it.
    target = routing.get("target")
    if target:
        resolved = (settings.data_dir() / target).resolve()
        root = settings.data_dir().resolve()
        # A handle is a file on disk and could say anything. Refuse to be walked
        # out of the data directory by one.
        if resolved == root or root not in resolved.parents:
            log.error(
                "Batch %s routes %s to %s, which is outside %s; writing to the "
                "content-addressed path instead",
                handle.batch_id, custom_id, target, root,
            )
        else:
            return resolved

    return store.path_for(atom_type, None, subject, handle.model)


def _journal_result(handle: BatchHandle, response, out: Path) -> None:
    """Record a batch result's raw payload before anything can overwrite it.

    A batch is where losing evidence hurts most: the result arrives hours after
    the request, and the `_ERROR` sentinel it would otherwise live in is the same
    file a batched `fix` read its input from. Append-only means a later write
    cannot destroy an earlier payload, whichever route the write takes.

    Swallows its own failures — journalling is observation, and a failed
    observation must not cost the run a result that did arrive.
    """
    routing = handle.routing.get(response.custom_id or "", {})
    atom_type = routing.get("atom_type") or "unknown"
    subject = routing.get("subject") or response.custom_id or "unknown"
    # For a fix/verify batch the routed target names the file being worked on,
    # and its filename carries the true content id where `subject` is only the
    # slug the submitter could recover. Deriving the work item from it keeps
    # the repair on the same queue entry as the failure it repairs — without
    # this, the first live batched-fix cycle recorded its success under a
    # slug-derived id and never cleared the item its submission answered.
    target = routing.get("target")
    work_item = (journal.work_item_from_path(atom_type, target)
                 if target else None)
    try:
        journal.current().record_attempt(
            work_item_id=work_item or journal.work_item_id(atom_type, subject),
            raw_payload=response.raw_text or None,
            raw_payload_unavailable_reason=(
                None if response.raw_text else (response.error or "no text in the batch result")
            ),
            atom_type=atom_type,
            subject=subject,
            # The routed operation, so `costs` can attribute a batched create,
            # fix or verify to its phase: all three write the base atom type
            # with the base schema, and the collect run's own record says only
            # "collect". Absent on lines journalled before it was added.
            operation=routing.get("operation"),
            execution_mode="batch",
            batch_id=handle.batch_id,
            provider=handle.provider,
            requested_model=handle.model,
            sent_effort=handle.extra.get("effort"),
            sent_task_budget=handle.extra.get("task_budget"),
            returned_model=response.model,
            schema_id=handle.extra.get("schema_id"),
            schema_sha256=handle.extra.get("schema_sha256"),
            # Cached tokens too — see `JournalEntry.cache_read_tokens`. This
            # path recorded only input and output from the start, so even the
            # historical batch runs (the only ones that carried usage at all)
            # understate cost by whatever the cached prefix was worth.
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_tokens or None,
            cache_write_tokens=response.usage.cache_write_tokens or None,
            estimated_cost=estimate_cost(response.usage, handle.model, batch=True),
            validation_status="ok" if response.ok else "failed",
            validation_errors=[response.error] if response.error else None,
            # Same rule as `operations._support._record`: a failed verify
            # queues nothing, because its target is still a valid atom and a
            # `needs_fix` here would send the fix pass to overwrite good data.
            work_status=(journal.OK if response.ok
                         else None if (routing.get("operation") in ("verify", "expand")
                                       or store.target_of_error(out).exists())
                         else journal.NEEDS_FIX),
            target_pointer=str(out),
            # A good result is about to be written over `out` (collect writes
            # through `store.write_atom` whatever sits there, hours after the
            # submit read it): keep those bytes (v0.3.2). An expand reply is
            # one period, not a write; its splice keeps the city's bytes once
            # (`operations.expand.write_expanded`).
            replaces=(out if response.ok and routing.get("operation") != "expand" else None),
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Could not journal batch result %s: %s", response.custom_id, e)


def collect(path: Path | str) -> CollectReport:
    """Fetch a finished batch and write its results into `data/automated/`.

    Valid results are written as `.json`; schema failures are written as
    `.json_ERROR` so the existing fix pass can pick them up.
    """
    handle = load(path)
    report = CollectReport()
    expected = set(expected_custom_ids(handle))
    seen: set[str] = set()
    # expand fans out: many period-requests share one target city, and the city
    # is written once, spliced, after the loop. Accumulate here keyed by target.
    expand_results: dict[str, list[tuple[int, str, object]]] = {}
    #: target file -> the augment atom_type its replies came from (e.g.
    #: "expand_city", "expand_medication"), so the splice below can read the
    #: right ExpandSpec from the active pack rather than assuming city periods.
    expand_atom: dict[str, str] = {}

    try:
        provider = get_provider(handle.model)
        responses = list(provider.fetch_batch_results(handle))
    except (SchemaMismatch, UnknownModel, ValueError) as e:
        # The schema moved while the batch was in flight, so these results cannot
        # be trusted against the current model. Report it and return rather than
        # propagating: `run_batch collect` sweeps every handle, and raising here
        # would skip the healthy batches queued behind this one. Nothing is
        # fetched, so the results stay with the provider and can be collected
        # once the mismatch is resolved.
        log.error("Refusing to collect %s: %s; check UNIVERSE_BUILDER_DOMAIN matches the submitted pack",
                  handle.batch_id, e)
        report.refused = str(e)
        report.missing = sorted(expected)
        return report

    for response in responses:
        report.usage = report.usage + response.usage
        if not response.custom_id:
            log.error("Batch result with no custom_id, cannot route it: %s", response.error)
            report.failed += 1
            continue
        seen.add(response.custom_id)

        out = _output_path(handle, response.custom_id)
        _journal_result(handle, response, out)

        routing = handle.routing.get(response.custom_id, {})
        if routing.get("operation") == "expand":
            # One period of a city. Never a whole-atom write — `response.parsed`
            # is a single Population, and `out` is the city file; writing it
            # here would replace the city with one period. Accumulate by target;
            # the splice happens once per city after the loop, placing each
            # reply by the period NAME it expanded (positions are not stable
            # across a split or a concurrent expand). A failed period keeps its
            # name — best-effort per entry, as the sync pass does — no sentinel
            # over a valid city. `written` is counted from what actually splices.
            if response.ok:
                name = routing.get("period_name")
                idx = routing.get("period_index")
                if isinstance(idx, int) and name is not None:
                    expand_results.setdefault(str(out), []).append((idx, name, response.parsed))
                    expand_atom[str(out)] = routing.get("atom_type")
                else:
                    log.error("%s: expand result lacks period_index/period_name; cannot place it",
                              response.custom_id)
                    report.failed += 1
            else:
                log.error("%s: expand of a period failed (%s); the period keeps its name",
                          response.custom_id, response.error)
                report.failed += 1
            continue

        if response.ok:
            with store.path_lock(out):
                store.write_atom(out, response.parsed)
                store.clear_error(out)
            # Retire any sentinel this result supersedes, exactly as the
            # synchronous `operations._support.persist` does. Without it a batch
            # that successfully regenerates a previously-failed topic leaves
            # `X.json_ERROR` beside the new, valid `X.json` — and the next fix
            # pass then feeds the *stale* failed payload back to the model and
            # `commit_fix` writes the answer over the good batch result. So the
            # cost is not wasted work, it is a correct atom replaced by a fix
            # derived from evidence that no longer applies.
            report.written += 1
        elif handle.routing.get(response.custom_id, {}).get("operation") == "verify":
            # A failed verify leaves no sentinel: its target is still a valid
            # atom, and a sentinel beside it would queue a "repair" that
            # overwrites good data with output derived from this very failure.
            # The journal above holds the payload; the atom stays untouched.
            log.error(
                "%s: verification failed (%s); leaving the valid atom alone",
                response.custom_id, response.error,
            )
            report.failed += 1
        else:
            # Never overwrite a sentinel that is already there. For a batched
            # `fix` the existing sentinel holds the *input* to this very repair,
            # so replacing it with the repair's own failure destroys the only
            # copy of what needed fixing — and a transient "server overloaded",
            # which returns no text at all, would reduce it to a comment line.
            # The item stays queued with nothing left to repair from. B8 again,
            # one route further out.
            #
            # `write_new` is O_EXCL, so the decision and the write are one step;
            # two collectors racing cannot both conclude the sentinel is absent.
            sentinel = store.error_path(store.target_of_error(out))
            payload = response.raw_text or f"// {response.error}\n"
            if store.write_error(out, payload, preserve_existing=True):
                log.error("%s failed validation: %s", response.custom_id, response.error)
            else:
                log.error(
                    "%s failed validation (%s); preserving the existing atom or sentinel at %s",
                    response.custom_id, response.error, sentinel.name,
                )
            report.failed += 1

    # Splice each expand target once, now that every period for it is in hand.
    # Reads the file fresh and places only still-unexpanded indices, so a
    # re-collect or a concurrent sync run cannot double-write.
    if expand_results:
        from universe_builder.operations.expand import apply_expansions, expand_config

        for target, expansions in expand_results.items():
            # Derive the atom from the reply's own routing and read its ExpandSpec
            # from the active pack, so a non-city domain splices its own list
            # field/models. If this pack has no such atom (e.g. collecting a
            # medications batch without UNIVERSE_BUILDER_DOMAIN set), skip rather than force
            # the city shape — the periods keep their names and re-expand.
            atom_type = expand_atom.get(target)
            if not atom_type:
                # A handle written before routing carried the atom type. The
                # only default core could offer would be one domain's (it was
                # once history's city), so ask the pack instead:
                # if exactly one atom expands, it is that one.
                expandable = [s.name for s in active_pack().atoms if s.expand]
                if len(expandable) != 1:
                    log.error("%s: expand result names no atom type and the %s pack has %d "
                              "expandable atoms; cannot place it, periods keep their names",
                              target, active_pack().name, len(expandable))
                    continue
                atom_type = f"expand_{expandable[0]}"
            atom_name = atom_type[len("expand_"):] if atom_type.startswith("expand_") else atom_type
            try:
                cfg = expand_config(atom_name, handle.model)
            except KeyError as e:
                log.error("%s: cannot splice expansions (%s); periods keep their names", target, e)
                continue
            placed = apply_expansions(target, expansions, cfg)
            report.written += placed
            skipped = len(expansions) - placed
            if skipped:
                log.warning(
                    "%s: %d of %d expanded period(s) could not be placed (the file "
                    "changed since submit); they re-expand next run",
                    Path(target).name, skipped, len(expansions),
                )
            log.warning("Spliced %d expanded period(s) into %s", placed, Path(target).name)

    # A result that never arrives is not a failure we recorded - it is a silent
    # loss. Reconcile against the submitted ids so it cannot pass unnoticed.
    report.missing = sorted(expected - seen)
    if report.missing:
        log.error(
            "Batch %s: %d of %d submitted requests produced no result. "
            "First few: %s",
            handle.batch_id,
            len(report.missing),
            len(expected),
            ", ".join(report.missing[:5]),
        )
        settings.missing_dir().mkdir(parents=True, exist_ok=True)
        missing_log = settings.missing_dir() / f"{Path(path).stem}.json"
        missing_log.write_text(json.dumps(report.missing, indent=2), encoding="utf-8")
        log.error("Full list written to %s", missing_log)

    report.cost = estimate_cost(report.usage, handle.model, batch=True)

    if report.missing:
        report.ended_short = _ended_short(handle)

    # Only once everything is accounted for, or once nothing more can come: a
    # cancelled or expired batch's missing requests never ran. A batch that
    # lost results otherwise, or was refused, keeps its handle in place so it
    # stays visible to `list` and can be collected again after the cause is
    # dealt with.
    if report.complete or report.ended_short:
        report.archived = archive(path)

    log.warning(
        "Collected batch %s: %d written, %d failed, %d missing, ~$%s",
        handle.batch_id,
        report.written,
        report.failed,
        len(report.missing),
        f"{report.cost:.4f}" if report.cost is not None else "?",
    )
    return report
