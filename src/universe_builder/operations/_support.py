"""Shared plumbing for the operation modules.

The single job here is turning an `LLMCall` into a `Result`. Before this, each
operation ended with `call.get_valid_response().model_dump_json(indent=4)` and
every *caller* was responsible for noticing that the response might be an
`EmptyModel` sentinel rather than a real atom. Two callers forgot, which is
what B8 was.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Optional, Type

from universe_builder import journal, store
from universe_builder.operations.types import Result

log = logging.getLogger(__name__)


#: The per-path lock registry moved to `store.path_lock` when `write_error`
#: needed the same ordering domain as `persist` — a failing call's sentinel
#: racing a succeeding call's clear is decided by whoever holds the lock.


def call_budget(config) -> dict:
    """`max_tokens=` (and `effort=`, `task_budget=`) for an `llm_call_mapping` constructor, or nothing.

    Every operation spreads this into the call it builds, so the synchronous
    path asks for the same output budget as the batch path — both read
    `PipelineConfig.max_tokens`. Nothing rather than `max_tokens=None` when
    the config carries no budget, so the call class keeps its own default.
    The effort (v0.6.0) likewise: only when one was asked for, so a run
    without one builds exactly the call it built before.
    """
    budget = getattr(config, "max_tokens", None)
    out = {"max_tokens": int(budget)} if budget else {}
    effort = getattr(config, "effort", None)
    if isinstance(effort, str) and effort:
        out["effort"] = effort
    # The task budget (v0.6.1) the same way: only when one was asked for.
    task_budget = getattr(config, "task_budget", None)
    if isinstance(task_budget, int) and not isinstance(task_budget, bool) and task_budget:
        out["task_budget"] = task_budget
    return out


# ---------------------------------------------------------------------------
# Global API-call concurrency gate
# ---------------------------------------------------------------------------
#
# One number bounds how many model calls are in flight at once, *across* the
# whole run — not per pool. It exists because expand nests: the file loop runs
# several cities at once and each city expands several periods, so two
# independent pools would multiply into (cities x periods) concurrent calls and
# blow past the provider's rate limit. Every leaf call that wants to respect the
# bound takes a `call_slot()`; the file pool and the period pool can then both
# be sized generously without the product ever reaching the API. Default 1 —
# serial, matching the pre-concurrency behaviour when nobody set it (direct
# `process_expand_operation` calls in tests, say); the entry points set it from
# the resolved config's `concurrency`.
_call_concurrency = 1
_call_semaphore = threading.BoundedSemaphore(1)


def set_call_concurrency(n: int) -> None:
    global _call_concurrency, _call_semaphore
    _call_concurrency = max(1, int(n))
    _call_semaphore = threading.BoundedSemaphore(_call_concurrency)


def call_concurrency() -> int:
    """The configured bound, for sizing a pool against the same number."""
    return _call_concurrency


@contextmanager
def call_slot():
    """Hold one of the global API-call slots for the duration of a call."""
    _call_semaphore.acquire()
    try:
        yield
    finally:
        _call_semaphore.release()


def _usage_fields(call) -> dict:
    """`journal.usage_fields` of the call's reported usage, or nothing at all
    when the transport reported none (see that docstring for why zeros would
    be a lie)."""
    return journal.usage_fields(getattr(call, "usage", None),
                                str(getattr(call, "model_version", "") or "") or None)


def _effort_of(call) -> Optional[str]:
    """The effort the call put on the wire (`ProviderCall.sent_effort`: none
    for a provider or model that takes none), or None (a duck-typed test
    call's attribute is not one)."""
    effort = getattr(call, "sent_effort", None)
    return effort if isinstance(effort, str) and effort else None


def _task_budget_of(call) -> Optional[int]:
    """The task budget the call put on the wire (`ProviderCall.sent_task_budget`),
    or None (a duck-typed test call's attribute is not one)."""
    budget = getattr(call, "sent_task_budget", None)
    return budget if isinstance(budget, int) and not isinstance(budget, bool) and budget else None


def _record(call, result: Result, atom_class: Type, *, replaces=None, **metadata) -> None:
    """Put this attempt's raw payload in the run journal, whatever happened.

    The `_ERROR` sentinel is both the work queue and the only copy of a failed
    payload, and those two jobs keep colliding: B8 has now recurred four times,
    each through a different write path, and each time the fix was to stop one
    particular piece of code overwriting the evidence. The journal is the
    structural answer — it is append-only, so no later write can destroy an
    earlier payload, and a sentinel lost to a fifth route is then an
    inconvenience rather than the end of the record.

    Failures here are swallowed. Journalling is observation, and losing an
    observation must never cost the run the atom it was observing.

    `replaces` is the file a validated result is about to be written over in
    place; its bytes are kept with the attempt (`Journal.record_attempt`).
    Only a result that will be written passes it on, so a failure costs no
    copy.
    """
    subject = metadata.get("subject") or "unknown"
    atom_type = getattr(call, "atom_type", None) or metadata.get("atom_type") or "unknown"
    # The path the call is writing to carries the true identity; the subject may
    # be a lossy slug (augment calls read theirs off the filename). Deriving the
    # work item from the path keeps a repair on the same queue entry as the
    # failure it repairs — see `journal.work_item_from_path`.
    response_path = getattr(call, "response_path", None)
    work_item = (journal.work_item_from_path(atom_type, response_path)
                 if response_path is not None else None)
    # A failed verify or expand queues nothing: the target is still a valid
    # atom, and now that these attempts land on the *target's* work item rather
    # than a slug-minted one, a `needs_fix` here would send the fix pass to
    # "repair" good data from a failed improvement's payload. The same holds
    # for ANY failure whose target atom already sits on disk — a failed
    # re-create of a topic that has an atom must not queue that atom's
    # destruction (`store.write_error` refuses the sentinel for the same
    # reason; this is the journal-level half of one rule). Fresh create, fix
    # and append failures queue as before — there the payload really is the
    # atom's best current evidence.
    if result.should_write:
        work_status = journal.OK
    elif metadata.get("operation") in ("verify", "expand"):
        work_status = None
    elif (response_path is not None
          and store.target_of_error(response_path).exists()):
        work_status = None
    else:
        work_status = journal.NEEDS_FIX
    try:
        journal.current().record_attempt(
            work_item_id=work_item or journal.work_item_id(atom_type, subject),
            # `or None`, not the bare value. `Result.raw_text` defaults to the
            # empty string, so passing it through recorded "we have the bytes"
            # for a response that never arrived — `raw_payload_available: True`
            # beside a reason saying nothing came back. That is precisely the
            # confusion `record_attempt` refuses to allow, and the wiring was
            # smuggling it past the check.
            raw_payload=result.raw_text or None,
            raw_payload_unavailable_reason=(
                None if result.raw_text else "the provider returned no text"
            ),
            phase=metadata.get("phase"),
            operation=metadata.get("operation"),
            atom_type=atom_type,
            subject=subject,
            requested_model=str(getattr(call, "model_version", "") or "") or None,
            sent_effort=_effort_of(call),
            sent_task_budget=_task_budget_of(call),
            schema_id=f"{atom_class.__module__}:{atom_class.__qualname__}",
            validation_status="ok" if result.should_write else "failed",
            validation_errors=[result.error] if result.error else None,
            work_status=work_status,
            target_pointer=str(getattr(call, "response_path", "") or "") or None,
            replaces=replaces if result.should_write else None,
            # Only passed when the transport actually measured them: the
            # journal derives `usage_available` from whether these keys are
            # present, so sending zeros for a call that reported nothing would
            # record "this cost nothing" instead of "we do not know".
            **_usage_fields(call),
        )
    except Exception as e:  # noqa: BLE001 - observation must not break the run
        log.warning("Could not journal the attempt for %s: %s", subject, e)


def failure_stage(cause: Optional[BaseException]) -> str:
    """The journal's `validation_stage` for an exception raised building a call.

    `transport` only when the provider was asked and never answered
    (`TransportFailed`, raised by the call adapter after every attempt).
    Anything else raised while constructing the call — an example that does not
    validate, a missing key, an unknown model — happened before any request
    left, and calling it `transport` sent a reader looking at the network for a
    configuration fault.
    """
    from universe_builder.providers.errors import TransportFailed

    if cause is None or isinstance(cause, TransportFailed):
        return "transport"
    return "call_setup"


def record_failure(atom_type: str, subject: str, error: str, target=None, *,
                   cause: Optional[BaseException] = None, **entry_fields) -> None:
    """Record an attempt that never produced a response at all.

    Every operation catches transport errors and returns `Result.failed` *before*
    reaching `result_from_call`, so a call that died before its first byte left
    no journal entry whatsoever: the atom was invisible to the record and absent
    from `needs_fix()`. That is the worst thing for the queue to be wrong about,
    because a retry is exactly what a transport failure needs.

    `target` is the file the operation was working on, when there is one; it
    carries the true identity where `subject` may be a lossy slug — same rule
    as `_record`.

    `raw_payload=None` with a reason, never an empty string — the distinction
    between "the model returned nothing" and "we never heard back" is the one
    `record_attempt` exists to keep.

    `cause` is the exception, when there is one: it decides the stage
    (`failure_stage`), and only a transport failure is marked retryable.
    """
    stage = failure_stage(cause)
    work_item = (journal.work_item_from_path(atom_type, target)
                 if target is not None else None)
    # Same rule as `_record`, missed on the first pass: a verify or expand that
    # died at transport queues nothing, because its target is still a valid
    # atom and — now that failures land on the *real* work item — a `needs_fix`
    # here is a standing order to overwrite good data. The retry these
    # operations want is their own pass run again, not a repair.
    operation = entry_fields.get("operation")
    work_status = None if operation in ("verify", "expand") else journal.NEEDS_FIX
    try:
        journal.current().record_attempt(
            work_item_id=work_item or journal.work_item_id(atom_type, subject),
            raw_payload=None,
            raw_payload_unavailable_reason=error,
            atom_type=atom_type,
            subject=subject,
            validation_stage=stage,
            validation_status="failed",
            validation_errors=[error],
            retryable=stage == "transport",
            work_status=work_status,
            **entry_fields,
        )
    except Exception as e:  # noqa: BLE001 - observation must not break the run
        log.warning("Could not journal the transport failure for %s: %s", subject, e)


def fix_queue_targets(atom_type: str):
    """`(entry, target)` for each journal-queued `atom_type` item whose target
    and sentinel are both missing from disk. Reads only.

    Shared by `reconcile_fix_queue` (which restores those with a payload) and
    `work.select_work` (which counts them in a fix dry run), so the two cannot
    disagree about the queue.
    """
    found = []
    for entry in journal.needs_fix_across_runs():
        if entry.get("atom_type") != atom_type:
            continue
        # Relative pointers (v0.1.20 on) resolve in this data dir; absolute
        # ones from earlier lines are taken as written.
        target = journal.target_path(entry)
        if target is None:
            model = entry.get("requested_model")
            subject = entry.get("subject")
            if not model or not subject:
                continue
            target = store.path_for(atom_type, None, subject, model)
        target = store.target_of_error(target)
        if store.error_path(target).exists() or target.exists():
            continue
        found.append((entry, target))
    return found


def reconcile_fix_queue(atom_type: str) -> int:
    """Restore lost sentinels from the journal before a fix pass runs.

    The fix pass selects its work by globbing `_ERROR` files; the journal
    records the same queue. When they disagree, the glob silently under-fixes
    — a sentinel lost to a stray write (B8's fifth route) simply drops out of
    the work. The queue a fix pass actually serves is therefore the union:
    every sentinel on disk, plus every journal-queued item whose sentinel is
    gone but whose payload the journal kept, rewritten as a sentinel here so
    one repair path serves both. `store.write_error` still refuses if a real
    atom has appeared meanwhile.

    Items with neither a sentinel nor a payload are transport failures:
    nothing exists to repair from, and `needing_create` already queues their
    regeneration. An item whose atom exists needs nothing — the run that
    wrote the atom settles the queue entry with its own OK line.
    """
    restored = 0
    for entry, target in fix_queue_targets(atom_type):
        payload = journal.raw_payload_of(entry)
        if payload is None:
            log.info(
                "Fix queue holds %s with no payload (transport failure); "
                "regeneration is the create pass's job", entry.get("work_item_id"),
            )
            continue
        if store.write_error(target, payload):
            restored += 1
            log.warning(
                "Restored the lost sentinel for %s from the journal (run %s) — "
                "the glob alone would have skipped this repair",
                entry.get("work_item_id"), entry.get("run_id"),
            )
    return restored


def result_from_call(call, atom_class: Type, *, replaces=None, **metadata) -> Result:
    """Convert a completed `LLMCall` into a typed `Result`.

    `get_valid_response()` answers with an `EmptyModel` sentinel when validation
    failed, so the sentinel check has to happen somewhere. Doing it here, once,
    is the point: no operation and no caller has to remember.

    This is also where the attempt reaches the run journal, for the same reason:
    it is the one place every synchronous operation passes through with both the
    raw payload and the verdict in hand.

    `replaces`: the file the caller will write a validated atom over (create,
    verify, append pass their `call.response_path`; fix passes nothing, since
    `store.commit_fix` never writes over an atom; expand's per-entry calls
    pass nothing, its one splice is journaled where it is written). Its
    current bytes go into the journal before the caller writes (v0.3.2).
    """
    raw = getattr(call, "llm_response_json", "") or ""
    atom = call.get_valid_response()

    if isinstance(atom, store.get_empty_model().__class__):
        log.error("%s returned no usable data", metadata.get("subject", "call"))
        result = Result.invalid(raw, "response did not validate against the schema", **metadata)
    elif not isinstance(atom, atom_class):
        # Not expected, but a silently mistyped atom would be written to disk and
        # only fail much later, on read.
        result = Result.invalid(
            raw,
            f"expected {atom_class.__name__}, got {type(atom).__name__}",
            **metadata,
        )
    else:
        result = Result.success(atom, raw, **metadata)

    _record(call, result, atom_class, replaces=replaces, **metadata)
    return result


def persist(call, result: Result) -> None:
    """Write the result of a generating operation, without ever destroying data.

    `LLMCall.write_response_to_file` cannot be used for this. It writes whatever
    `parse_response_to_pydantic` produced, which is an `EmptyModel` sentinel when
    validation failed — so a failed call wrote the sentinel to the atom's own
    path.

    That was survivable only because ids used to be a per-process counter: a
    failed regeneration landed on a *different* filename each run, leaving junk
    but destroying nothing. Content-addressed ids made the path a
    function of the topic, so the same failure now lands squarely on the existing
    file. Reproduced: a valid `Discovery` was replaced by a
    `{"error": "No data found"}` simply by re-running create for the same
    topic and having the model return rubbish.

    So: a validated atom is written; a failure never overwrites an existing one.
    The `_ERROR` sentinel — already recorded by `result_from_call` — is what
    carries the evidence and queues the retry.
    """
    path = call.response_path
    subject = result.metadata.get("subject", "call")

    # Two calls for one topic resolve to one content-addressed path, and the
    # create pass can genuinely see a topic twice — two History files may list
    # the same city. Serialising per path keeps a success and a failure from
    # interleaving their writes.
    with store.path_lock(path):
        if result.should_write:
            store.write_atom(path, result.atom)
            # The topic generates cleanly now, so any sentinel from an earlier
            # attempt is resolved. Safe to retire only because we just wrote a
            # validated atom - the B8 ordering.
            store.clear_error(path)
            return

        # A failure writes nothing at the atom's path. It used to leave an
        # `EmptyModel` placeholder there, which was dead weight wearing an
        # atom's name: some had to be purged from the corpus, every
        # reader needed an `is_empty_model` guard to avoid mistaking one for
        # data, and the `_ERROR` sentinel plus the journal already record the
        # failure and queue the retry. `needing_create` treats the topic as
        # missing either way, so nothing changes about what gets regenerated.
        log.error(
            "%s did not validate; the _ERROR sentinel queues the fix and "
            "nothing is written at %s",
            subject, path.name,
        )
