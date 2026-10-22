"""The run journal: evidence for diagnosis, and the work queue for the fix pass.

**STATUS: wired in.** Entry points call `start_run()`; the
passes record through `current()`. Outside a run `current()` returns a
`NullJournal` that records nothing, so nothing requires a run to exist.

It was first left inert, on the grounds that `store.commit_fix` had closed B8
for a fraction of the code. B8 then recurred again and again, each time
through a different write path, because the `_ERROR` sentinel is
both the work queue and the only copy of the evidence and every new write path
eventually writes over it. Append-only is the structural answer; see
ARCHITECTURE.md 2.1e.

The fix pass still takes its queue from the `_ERROR` glob. The journal records
the same information alongside it and `universe_builder status` flags any disagreement.

`_ERROR` files do two jobs at once, and do both badly. As evidence they hold the
model's text but nothing about the request that produced it, the model that was
asked, the code that was running, or *why* validation failed - so a diagnosis
starts by guessing. As a work queue they are a filename glob, and the fix pass
retired an entry by deleting it, which meant a successful fix destroyed the only
record of the failure (B8). This module replaces both jobs with an append-only
journal.

Layout, one directory per run, under the generated-data directory::

    <data_dir>/_runs/<run_id>/
      run.json                        # resolved config, code revision, model resolution
      journal.jsonl                   # one line per attempt
      attempts/<attempt_id>/
        request.json                  # exactly what was sent
        response.raw.txt              # exactly what came back, BEFORE any parse
        response-envelope.json        # provider metadata, usage, ids
        replaced.json                 # the bytes an in-place rewrite replaced (v0.3.2)

`_runs` is underscore-prefixed deliberately: `store.iter_atoms` skips
underscore-prefixed directories, so journal data can never be walked as if it
were corpus data.

**The one property that matters.** `response.raw.txt` is written before
`json.loads` and before any Pydantic coercion. The current code loses the payload
precisely when it is malformed, which is the only time anyone needs it. That is
why `record_attempt` takes `raw_payload` as a *required* keyword argument and
writes it before it writes anything else: there is no ordering a caller can
choose that records an attempt without first preserving the bytes. A caller that
genuinely received nothing - a connection reset before the first byte - must say
so explicitly with `raw_payload_unavailable_reason`, which is recorded as a fact
rather than papered over with an empty string.

**Resolution is an append.** A fix that works writes a line naming the attempt it
resolves; `needs_fix()` then stops returning that item. Nothing is deleted, so
the failure and its repair are both still on disk afterwards. This is the
structural form of the B8 fix that `store.commit_fix` handles for the ordering.

This module deliberately imports nothing from `execute_prompt`. The journal has
to be usable from the batch path and from one-off scripts, and a dependency on
orchestration would make it unusable from both.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Optional

from universe_builder import settings, store

log = logging.getLogger(__name__)

#: Bumped when the meaning of a field changes, so an old journal stays readable.
LOG_SCHEMA_VERSION = 1

#: Underscore-prefixed so `store.iter_atoms` skips it. See the module docstring.
RUNS_DIRNAME = "_runs"

RUN_FILENAME = "run.json"
JOURNAL_FILENAME = "journal.jsonl"
ATTEMPTS_DIRNAME = "attempts"
REQUEST_FILENAME = "request.json"
RAW_FILENAME = "response.raw.txt"
ENVELOPE_FILENAME = "response-envelope.json"
#: The exact bytes an in-place rewrite replaced, saved before the write
#: (`record_attempt(replaces=...)`). Named `.json` because every file it holds
#: is an atom file; the bytes are copied, never parsed or re-encoded.
REPLACED_FILENAME = "replaced.json"

# -- event types ------------------------------------------------------------
EVENT_ATTEMPT = "attempt"
EVENT_RESOLUTION = "resolution"

#: `execution_mode` of an attempt line no model made (`record_mechanical`):
#: a pack's list verb, a label rewrite, a URL respelling. Beside `batch` and
#: the synchronous path's null, so `costs` can leave it out of the calls.
MECHANICAL = "mechanical"

# -- work_status values -----------------------------------------------------
#: The fix pass selects on this, in place of globbing for `*_ERROR`.
NEEDS_FIX = "needs_fix"
#: A later line said so; the item leaves the queue.
RESOLVED = "resolved"
#: Validated first time; never entered the queue.
OK = "ok"
#: Given up on deliberately. Distinct from resolved so it is auditable.
ABANDONED = "abandoned"

#: Statuses that take a work item *out* of the queue when seen on a later line.
SETTLED_STATUSES = frozenset({RESOLVED, OK, ABANDONED})


# ---------------------------------------------------------------------------
# The journal line
# ---------------------------------------------------------------------------

@dataclass
class JournalEntry:
    """One line of `journal.jsonl`. Its field names and grouping are the journal's schema.

    Every field defaults to None so that a line states only what was actually
    known at the time. `to_dict` drops the Nones rather than emitting them,
    because an absent field and a field the provider reported as null are
    different claims and only one of them is true. Falsy-but-real values (`0`,
    `False`, `""`) survive; only None is treated as "not applicable here".
    """

    # -- Identity
    event_id: Optional[str] = None
    event_type: Optional[str] = None
    event_sequence: Optional[int] = None
    timestamp_utc: Optional[str] = None
    log_schema_version: int = LOG_SCHEMA_VERSION

    # -- Correlation
    run_id: Optional[str] = None
    work_item_id: Optional[str] = None
    attempt_id: Optional[str] = None
    attempt_number: Optional[int] = None
    parent_attempt_id: Optional[str] = None

    # -- Operation
    phase: Optional[str] = None
    operation: Optional[str] = None
    atom_type: Optional[str] = None
    subject: Optional[str] = None
    context: Optional[str] = None
    target_pointer: Optional[str] = None

    # -- Input lineage
    input_artifact_id: Optional[str] = None
    input_sha256: Optional[str] = None
    source_history_id: Optional[str] = None

    # -- Execution
    provider: Optional[str] = None
    requested_model: Optional[str] = None
    #: The reasoning effort the call put on the wire (v0.6.0,
    #: `providers.effort.sent`): not the one asked for, so `low` asked of
    #: Gemini or gpt-4o-mini, which take none, is recorded as nothing. Absent
    #: when none was sent, which is every line before v0.6.0 too: the model
    #: ran at its provider's default.
    sent_effort: Optional[str] = None
    #: The task budget (output tokens) the call put on the wire (v0.6.1,
    #: `providers.task_budget.sent`): none for a provider that takes none.
    #: Absent when none was sent, which is every line before v0.6.1 too.
    sent_task_budget: Optional[int] = None
    returned_model: Optional[str] = None
    execution_mode: Optional[str] = None
    provider_request_id: Optional[str] = None
    batch_id: Optional[str] = None

    # -- Reproduction
    code_revision: Optional[str] = None
    resolved_config_sha256: Optional[str] = None
    request_ref: Optional[str] = None
    request_sha256: Optional[str] = None

    # -- Schema
    schema_id: Optional[str] = None
    schema_version: Optional[str] = None
    schema_sha256: Optional[str] = None

    # -- Response evidence
    raw_payload_ref: Optional[str] = None
    raw_payload_sha256: Optional[str] = None
    raw_payload_available: Optional[bool] = None
    raw_payload_unavailable_reason: Optional[str] = None

    # -- Validation
    validation_stage: Optional[str] = None
    validation_status: Optional[str] = None
    validation_errors: Optional[list] = None

    # -- Retry
    retryable: Optional[bool] = None
    retry_action: Optional[str] = None
    backoff_ms: Optional[int] = None
    attempt_limit: Optional[int] = None

    # -- Usage
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    #: Cached prompt tokens are **billable**, at `cached_input` rather than
    #: `input` (`providers/pricing.py:81`). `openai_provider._usage` splits them
    #: out of `input_tokens`, so a journal recording only input+output silently
    #: drops them — and this project leans hard on prompt caching, so that is
    #: not a rounding error. Recorded separately to keep the two rates apart.
    cache_read_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    usage_available: Optional[bool] = None
    estimated_cost: Optional[float] = None

    # -- Persistence
    output_artifact_id: Optional[str] = None
    commit_status: Optional[str] = None
    supersedes_artifact_id: Optional[str] = None
    #: The bytes this line's write replaced at `target_pointer`, saved in the
    #: attempt folder before the write (v0.3.2). Not `input_sha256`: that is
    #: the call's input, and on the batch path the input was read at submit
    #: while the bytes replaced are whatever sits there at collect — the very
    #: gap where a correction made in between was silently reverted. Absent
    #: when the write replaced nothing (a new file) and on every older line.
    replaced_ref: Optional[str] = None
    replaced_sha256: Optional[str] = None

    # -- Queue resolution
    work_status: Optional[str] = None
    resolves_attempt_id: Optional[str] = None
    resolution_reason: Optional[str] = None

    def to_dict(self, omit_none: bool = True) -> dict:
        data = asdict(self)
        if omit_none:
            data = {k: v for k, v in data.items() if v is not None}
        return data


ENTRY_FIELDS = frozenset(f.name for f in fields(JournalEntry))

#: Fields the journal derives and a caller may not set. Letting a caller pass its
#: own `raw_payload_sha256` would make the hash a claim rather than a measurement,
#: and a caller-set `event_sequence` would break the ordering `needs_fix` reduces
#: over.
DERIVED_FIELDS = frozenset({
    "event_id",
    "event_sequence",
    "timestamp_utc",
    "log_schema_version",
    "run_id",
    "raw_payload_ref",
    "raw_payload_sha256",
    "raw_payload_available",
    "request_ref",
    "request_sha256",
    "replaced_ref",
    "replaced_sha256",
})


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def sha256_hex(data) -> str:
    """Digest of the exact bytes recorded. Text is measured as UTF-8."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(moment: Optional[datetime] = None) -> datetime:
    """Normalise an injected timestamp to UTC.

    Callers may inject a timestamp - tests want deterministic ids - but a naive
    datetime would produce an id that silently depends on the machine's zone.
    """
    moment = moment or utc_now()
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def _iso(moment: datetime) -> str:
    return _stamp(moment).isoformat().replace("+00:00", "Z")


def safe_id(value: str) -> str:
    """Make an id usable as a single path segment.

    Run and attempt ids become directory names. A work item id does not, and is
    left exactly as the caller wrote it, because `city:memphis` is more useful in
    a query than `city-memphis`.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-.")
    if not cleaned:
        raise ValueError(f"{value!r} contains nothing usable as a path segment")
    return cleaned[:120]


def new_run_id(moment: Optional[datetime] = None) -> str:
    """Sortable and unique: two runs in the same second must not collide."""
    return f"{_stamp(moment):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"


def new_attempt_id(moment: Optional[datetime] = None) -> str:
    return f"{_stamp(moment):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:12]}"


def _dumps(payload) -> str:
    """Serialise an artefact, never raising on an object json does not know.

    A request dict can carry a Pydantic model class, a `Path`, or an enum. The
    journal is the thing that records failures; it must not become a new source
    of them, so unknown objects are recorded as their `repr` rather than
    aborting the run that was being logged.
    """
    return json.dumps(payload, indent=4, ensure_ascii=False, default=repr)


def _engine_package_dir() -> Path:
    """Where the `universe_builder` package was imported from."""
    return Path(__file__).resolve().parent


def _engine_checkout() -> Optional[Path]:
    """The engine's own source checkout, or None when it is not running from one.

    Only a src-layout root that holds `.git` and a `pyproject.toml` naming this
    project qualifies. Anything looser answers for the wrong repository: under a
    wheel the package sits in site-packages, git walks up from there to
    whatever encloses it, and the history corpus keeps its venv inside its own
    checkout — so its runs recorded the pack's HEAD as the engine's revision.
    """
    package = _engine_package_dir()
    if package.parent.name != "src":
        return None
    root = package.parent.parent
    if not (root / ".git").exists():
        return None
    try:
        project = (root / "pyproject.toml").read_text(encoding="utf-8")
    except OSError:
        return None
    if not re.search(r'^name\s*=\s*"universe_builder"\s*$', project, re.M):
        return None
    return root


def _git_revision() -> Optional[str]:
    """The engine checkout's current commit, or None when there is no checkout.

    Returns None rather than raising when this is an installed wheel rather
    than the engine's own working copy (`_engine_checkout`), when git is
    missing, or when the repository has no commits yet. An unknown revision is
    a small loss — `provenance()` still names the release — and a crashed run
    is not.
    """
    root = _engine_checkout()
    if root is None:
        return None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as e:
        log.debug("No code revision available: %s", e)
        return None
    if result.returncode != 0:
        log.debug("No code revision available: %s", result.stderr.strip())
        return None
    return result.stdout.strip() or None


_REVISION_UNSET = object()
_revision_cache: Any = _REVISION_UNSET


def code_revision(refresh: bool = False) -> Optional[str]:
    """Cached `git rev-parse HEAD`. The tree does not change mid-run."""
    global _revision_cache
    if refresh or _revision_cache is _REVISION_UNSET:
        _revision_cache = _git_revision()
    return _revision_cache


def provenance() -> dict:
    """What ran: the engine's release and the active pack's distribution.

    Written into every `run.json`. The distributions' names and versions are
    what identify the code under any install; `git_revision` is added only
    when the engine runs from its own checkout (`_engine_checkout`), where a
    version number alone does not say which commit. Never raises: a pack that
    cannot be resolved is recorded as None rather than costing the run.
    """
    from importlib import metadata

    import universe_builder

    try:
        engine_version = metadata.version("universe_builder")
    except metadata.PackageNotFoundError:
        engine_version = universe_builder.__version__
    engine = {"distribution": "universe_builder", "version": engine_version,
              "git_revision": code_revision()}
    if engine_version != universe_builder.__version__:
        # Stale install metadata, or a checkout shadowing the install on
        # `sys.path`: say which code actually ran as well as what pip thinks.
        engine["imported_version"] = universe_builder.__version__
    record = {"engine": engine, "pack": None}
    try:
        from universe_builder import registry

        record["pack"] = registry.pack_provenance()
    except Exception as e:  # noqa: BLE001 - provenance is observation
        log.debug("No pack provenance: %s", e)
    return record


def run_provenance(run_id: str) -> Optional[dict]:
    """A run's recorded `provenance`, or None for a run from before provenance
    was recorded (whose `run.json` has only `code_revision`) or one with no `run.json`."""
    path = run_dir(run_id) / RUN_FILENAME
    if not path.exists():
        return None
    return json.loads(store.read(path)).get("provenance")


def validation_errors(error) -> list:
    """Pydantic's `ValidationError` as the JSON-safe shape the journal records.

    `loc`/`type`/`msg`/`input`/`ctx` per error. `input` and `ctx` routinely hold
    values that `json.dumps` cannot serialise - a partially built model, an
    exception instance - so they are coerced here rather than at write time,
    where the failure would land on the wrong line and be blamed on the payload.
    """
    raw = error.errors() if hasattr(error, "errors") else list(error)
    out = []
    for item in raw:
        out.append({
            "loc": [str(part) for part in item.get("loc", ())],
            "type": item.get("type"),
            "msg": item.get("msg"),
            "input": _jsonable(item.get("input")),
            "ctx": {k: _jsonable(v) for k, v in (item.get("ctx") or {}).items()} or None,
        })
    return out


def _jsonable(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return repr(value)


def _append_line(path: Path, line: str) -> None:
    """Append one JSONL record.

    This cannot go through `store.write`. That function is atomic by writing a
    temporary file and `os.replace`-ing it over the destination, which by design
    replaces the *whole* file - so appending through it would mean reading the
    journal back, concatenating, and rewriting it for every attempt. That is
    quadratic in the number of attempts, and worse, it puts the entire
    accumulated journal at risk on every single write: an interruption between
    `open` and `replace` swaps a complete journal for a partial rebuild, losing
    exactly the evidence the file exists to keep.

    A single `O_APPEND` write is the right primitive instead. Records are
    guaranteed newline-free because `json.dumps` escapes newlines inside
    strings, so a line can never be split, and concurrent appenders cannot
    interleave a short line on POSIX. The write is flushed and fsynced because a
    journal that is still in the page cache when the process is killed did not
    record anything.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

def runs_dir() -> Path:
    """Read the data directory at call time, as `store` does, so tests isolate."""
    return settings.data_dir() / RUNS_DIRNAME


def run_dir(run_id: str) -> Path:
    return runs_dir() / safe_id(run_id)


def list_runs() -> list:
    """Run ids, oldest first. Ids start with a UTC stamp, so lexical is chronological."""
    root = runs_dir()
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def read_entries(run_id: str) -> list:
    """Parse one run's journal, skipping any line that is not JSON.

    A truncated final line - the process was killed mid-append - must not make
    the whole journal unreadable, which is the failure mode that would take the
    work queue down with it.
    """
    path = run_dir(run_id) / JOURNAL_FILENAME
    if not path.exists():
        return []
    entries = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except ValueError:
            log.warning("Skipping unparseable journal line %s:%d", path, number)
    return entries


# ---------------------------------------------------------------------------
# The queue reduction
# ---------------------------------------------------------------------------

def outstanding(entries: Iterable[Mapping]) -> list:
    """Work items still needing a fix, newest attempt per item.

    Replayed in sequence order rather than filtered, because the same work item
    can cycle: `needs_fix`, resolved by a fix, then `needs_fix` again when the
    fix itself failed to validate. A filter over `work_status == "needs_fix"`
    would return the stale first attempt in that case, and a filter that merely
    subtracted resolved ids would drop an item that is genuinely still broken.

    Within one entry, resolution is applied before status: a line that both
    resolves attempt A and reports its own failure leaves the item queued
    against the *new* attempt, which is the one holding the current evidence.

    A resolution that names an attempt which has since been superseded clears
    nothing. A fix pass that started before a later attempt failed must not be
    able to retire evidence it never looked at - that is the same mistake as
    deleting an `_ERROR` file unconditionally, one indirection further out.
    """
    queue: dict = {}
    ordered = sorted(entries, key=lambda e: (e.get("event_sequence") or 0))
    for entry in ordered:
        item = entry.get("work_item_id")
        if not item:
            continue
        resolves = entry.get("resolves_attempt_id")
        current = queue.get(item)
        applies = not resolves or current is None or current.get("attempt_id") == resolves
        if resolves and applies:
            queue.pop(item, None)
        status = entry.get("work_status")
        if status == NEEDS_FIX:
            queue[item] = entry
        elif status in SETTLED_STATUSES and applies:
            queue.pop(item, None)
    return sorted(queue.values(), key=lambda e: (e.get("event_sequence") or 0))


def relative_pointer(pointer) -> Optional[str]:
    """A `target_pointer` as stored: relative to the data dir when inside it.

    An absolute path recorded today points into *this* corpus wherever the
    journal is later read from: a copied or moved corpus (the parity harness's
    copy, a backup restored elsewhere) would have `reconcile_fix_queue` follow
    it and write `_ERROR` sentinels into the original.
    A path outside the data dir is kept as given; so is a relative one.
    """
    if not pointer:
        return pointer
    path = Path(pointer)
    if not path.is_absolute():
        return str(pointer)
    root = settings.data_dir()
    for candidate, base in ((path, root), (path.resolve(), root.resolve())):
        try:
            return candidate.relative_to(base).as_posix()
        except ValueError:
            continue
    return str(pointer)


def target_path(entry_or_pointer) -> Optional[Path]:
    """Where a journal line's `target_pointer` points, in *this* data dir.

    Reads both forms: a relative pointer (v0.1.20 on) is joined to
    `settings.data_dir()`; an absolute one (every line up to v0.1.19) is taken
    as written. None when the line has no pointer.
    """
    pointer = (entry_or_pointer.get("target_pointer") if isinstance(entry_or_pointer, Mapping)
               else entry_or_pointer)
    if not pointer:
        return None
    path = Path(pointer)
    return path if path.is_absolute() else settings.data_dir() / path


def raw_payload_of(entry: Mapping) -> Optional[str]:
    """The recorded payload behind a queue entry, from whichever run holds it.

    `Journal.read_raw` answers for the current run; the fix queue spans runs,
    so its entries carry `run_id` and this follows it. None means no payload
    was recorded (a transport failure) or the file cannot be read — either
    way, nothing to repair from.
    """
    run_id, attempt_id = entry.get("run_id"), entry.get("attempt_id")
    if not run_id or not attempt_id:
        return None
    path = run_dir(run_id) / ATTEMPTS_DIRNAME / safe_id(attempt_id) / RAW_FILENAME
    try:
        return store.read(path) if path.exists() else None
    except (OSError, UnicodeDecodeError):
        return None


def replaced_bytes(entry: Mapping) -> Optional[bytes]:
    """The bytes a journal line's write replaced, or None if it saved none.

    None for a line that replaced nothing (a new file) and for every line
    written before v0.3.2, which had no such field. A line that *claims*
    saved bytes is held to it: a copy that is missing raises
    `FileNotFoundError` and one whose hash disagrees raises `ValueError`,
    because a restore from evidence that is gone or altered must not
    quietly become a restore of something else.
    """
    ref, digest, run_id = entry.get("replaced_ref"), entry.get("replaced_sha256"), entry.get("run_id")
    if not ref and not digest:
        return None
    if not (ref and digest and run_id):
        raise ValueError(f"journal line {entry.get('attempt_id')!r} names replaced bytes "
                         f"incompletely (replaced_ref={ref!r}, replaced_sha256={digest!r})")
    data = (run_dir(run_id) / ref).read_bytes()
    if sha256_hex(data) != digest:
        raise ValueError(f"the replaced bytes saved for attempt {entry.get('attempt_id')!r} do not "
                         f"match the line's replaced_sha256; refusing to use them")
    return data


def find_attempt(attempt_id: str, run_id: Optional[str] = None) -> Optional[dict]:
    """The attempt line with this `attempt_id`, searched in `run_id` or in
    every run (newest first), or None."""
    runs = [run_id] if run_id else list(reversed(list_runs()))
    for rid in runs:
        for entry in read_entries(rid):
            if entry.get("attempt_id") == attempt_id and entry.get("event_type") == EVENT_ATTEMPT:
                return entry
    return None


def needs_fix_across_runs(run_ids: Optional[Iterable[str]] = None) -> list:
    """The queue spanning several runs, for a fix pass over everything on disk.

    Entries are pooled and reduced together rather than per run, so a fix
    recorded in today's run clears an item that failed in last week's.
    """
    ids = list(run_ids) if run_ids is not None else list_runs()
    pooled = []
    for offset, run_id in enumerate(ids):
        for entry in read_entries(run_id):
            # `event_sequence` restarts at 1 per run; offset it so the pooled
            # replay keeps runs in order instead of interleaving them.
            entry = dict(entry)
            entry["event_sequence"] = (offset + 1) * 10 ** 9 + (entry.get("event_sequence") or 0)
            pooled.append(entry)
    return outstanding(pooled)


# ---------------------------------------------------------------------------
# The journal
# ---------------------------------------------------------------------------

class Journal:
    """Append-only record of one run.

    Construct with `Journal.start` for a new run or `Journal.open` to reattach to
    an existing one - the fix pass reattaches, because its resolving lines belong
    with the failures they resolve.
    """

    def __init__(self, run_id: str, revision: Optional[str] = None,
                 resolved_config_sha256: Optional[str] = None):
        self.run_id = safe_id(run_id)
        self.directory = run_dir(self.run_id)
        self.code_revision = revision
        self.resolved_config_sha256 = resolved_config_sha256
        existing = read_entries(self.run_id)
        self._sequence = max((e.get("event_sequence") or 0) for e in existing) if existing else 0
        self._attempt_counts: dict = {}
        self._last_attempt: dict = {}
        for entry in sorted(existing, key=lambda e: (e.get("event_sequence") or 0)):
            self._remember_attempt(entry)

    # -- construction ---------------------------------------------------------

    @classmethod
    def start(cls, run_id: Optional[str] = None, config: Optional[Mapping] = None,
              models: Optional[Mapping] = None, now: Optional[datetime] = None) -> "Journal":
        """Create the run directory and write `run.json`.

        `config` is the *resolved* configuration - what the run actually used
        after defaults and overrides - not the file on disk. Its hash is stamped
        on every entry, so two runs that disagree can be told apart without
        diffing their config blobs.
        """
        moment = _stamp(now)
        run_id = safe_id(run_id) if run_id else new_run_id(moment)
        revision = code_revision()
        config_hash = sha256_hex(json.dumps(config, sort_keys=True, default=repr)) if config else None

        journal = cls(run_id, revision=revision, resolved_config_sha256=config_hash)
        store.write(journal.directory / RUN_FILENAME, _dumps({
            "run_id": journal.run_id,
            "started_utc": _iso(moment),
            "log_schema_version": LOG_SCHEMA_VERSION,
            "code_revision": revision,
            # Added later; `code_revision` stays beside it for older
            # readers, and is None outside the engine's own checkout.
            "provenance": provenance(),
            "resolved_config_sha256": config_hash,
            "config": config,
            "models": dict(models) if models else None,
        }))
        log.info("Journal run %s at %s", journal.run_id, journal.directory)
        return journal

    @classmethod
    def open(cls, run_id: str) -> "Journal":
        """Reattach to an existing run, inheriting its revision and config hash."""
        directory = run_dir(run_id)
        if not directory.exists():
            raise FileNotFoundError(f"No journal run at {directory}")
        meta = {}
        run_file = directory / RUN_FILENAME
        if run_file.exists():
            meta = json.loads(store.read(run_file))
        return cls(
            run_id,
            revision=meta.get("code_revision"),
            resolved_config_sha256=meta.get("resolved_config_sha256"),
        )

    # -- paths ----------------------------------------------------------------

    @property
    def journal_path(self) -> Path:
        return self.directory / JOURNAL_FILENAME

    @property
    def run_path(self) -> Path:
        return self.directory / RUN_FILENAME

    def attempt_dir(self, attempt_id: str) -> Path:
        return self.directory / ATTEMPTS_DIRNAME / safe_id(attempt_id)

    def _ref(self, path: Path) -> str:
        """Store artefact references relative to the run directory.

        An absolute path recorded today is wrong the moment the run directory is
        copied to another machine to be looked at, which is most of the times
        anyone reads a journal at all.
        """
        return path.relative_to(self.directory).as_posix()

    def read_raw(self, attempt_id: str) -> Optional[str]:
        """The exact text recorded for an attempt, or None if none was."""
        path = self.attempt_dir(attempt_id) / RAW_FILENAME
        return store.read(path) if path.exists() else None

    # -- writing --------------------------------------------------------------

    def record_attempt(self, *, work_item_id: str, raw_payload: Optional[str],
                       request: Optional[Mapping] = None, envelope: Optional[Mapping] = None,
                       attempt_id: Optional[str] = None, now: Optional[datetime] = None,
                       replaces=None, **entry_fields) -> JournalEntry:
        """Preserve one attempt's evidence, then append its line.

        `raw_payload` is required and is written first - before `request.json`,
        before the envelope, before the journal line, and, at the call site,
        before `json.loads` and before Pydantic sees it. There is deliberately no
        overload that records an attempt and adds the payload afterwards, because
        that is the version that loses the payload when it is malformed, which is
        the only time it is worth having.

        Pass `raw_payload=None` only when nothing arrived, and say why in
        `raw_payload_unavailable_reason`. Remaining keyword arguments are
        `JournalEntry` field names, spelled exactly as `ENTRY_FIELDS` has them; an
        unknown name raises rather than being dropped, because a typo that
        silently discards `validation_errors` produces a journal that looks
        complete and is not.

        `replaces` is the file this attempt's write is about to overwrite in
        place (v0.3.2). Its current bytes are copied, exactly, to
        `attempts/<id>/replaced.json` and their hash recorded as
        `replaced_sha256`, so `replaced_bytes` and `universe_builder restore`
        can put them back. It is the line's `target_pointer` (set from it when
        not given; a different pointer raises). A file that does not exist
        saves nothing: the write creates rather than replaces. Callers pass it
        only when the write will happen — a validated result — so a failure
        costs no copy.
        """
        moment = _stamp(now)
        if replaces is not None:
            pointer = entry_fields.get("target_pointer")
            if pointer is None:
                entry_fields["target_pointer"] = str(replaces)
            elif relative_pointer(str(pointer)) != relative_pointer(str(replaces)):
                raise ValueError(f"replaces={replaces!s} is not this line's target_pointer {pointer!s}")
        attempt_id = safe_id(attempt_id) if attempt_id else new_attempt_id(moment)
        reason = entry_fields.get("raw_payload_unavailable_reason")
        if raw_payload is None and not reason:
            raise ValueError(
                "raw_payload is None: pass raw_payload_unavailable_reason saying what "
                "went wrong, so the gap is recorded as a fact rather than as an empty response"
            )

        directory = self.attempt_dir(attempt_id)
        entry = self._new_entry(EVENT_ATTEMPT, moment, work_item_id=work_item_id,
                                attempt_id=attempt_id, **entry_fields)

        # 1. The raw bytes, before anything else can fail.
        if raw_payload is not None:
            text = raw_payload if isinstance(raw_payload, str) else raw_payload.decode("utf-8", "replace")
            store.write(directory / RAW_FILENAME, text)
            entry.raw_payload_ref = self._ref(directory / RAW_FILENAME)
            entry.raw_payload_sha256 = sha256_hex(text)
            entry.raw_payload_available = True
        else:
            entry.raw_payload_available = False

        # 2. What the write is about to replace, before the caller writes.
        if replaces is not None:
            self._save_replaced(entry, directory, Path(replaces))

        # 3. What was sent, and what the provider said about it.
        if request is not None:
            body = _dumps(request)
            store.write(directory / REQUEST_FILENAME, body)
            entry.request_ref = self._ref(directory / REQUEST_FILENAME)
            entry.request_sha256 = sha256_hex(body)
        store.write(directory / ENVELOPE_FILENAME, _dumps(self._envelope(entry, envelope)))

        self._append(entry)
        return entry

    def _save_replaced(self, entry: JournalEntry, directory: Path, target: Path) -> None:
        """Copy `target`'s bytes beside the attempt; nothing if it is absent."""
        try:
            data = target.read_bytes()
        except FileNotFoundError:
            return
        except OSError as e:
            log.error("Could not keep the bytes %s is about to replace: %s", target.name, e)
            return
        store.write_bytes(directory / REPLACED_FILENAME, data)
        entry.replaced_ref = self._ref(directory / REPLACED_FILENAME)
        entry.replaced_sha256 = sha256_hex(data)

    def resolve(self, work_item_id: str, resolves_attempt_id: str, *,
                resolution_reason: Optional[str] = None, now: Optional[datetime] = None,
                **entry_fields) -> JournalEntry:
        """Append a line taking a work item out of the queue.

        This is the whole of the `_ERROR`-deletion replacement. Nothing is
        removed: the failing attempt's line, its request and its raw payload are
        all still there afterwards, so a fix that turns out to be wrong can still
        be compared against what the model originally returned.
        """
        entry_fields.setdefault("work_status", RESOLVED)
        entry = self._new_entry(
            EVENT_RESOLUTION, _stamp(now),
            work_item_id=work_item_id,
            resolves_attempt_id=resolves_attempt_id,
            resolution_reason=resolution_reason,
            **entry_fields,
        )
        self._append(entry)
        return entry

    # -- reading --------------------------------------------------------------

    def entries(self) -> list:
        """Parsed journal lines, from disk. The file is the source of truth."""
        return read_entries(self.run_id)

    def entries_for(self, work_item_id: str) -> list:
        return [e for e in self.entries() if e.get("work_item_id") == work_item_id]

    def attempts_for(self, work_item_id: str) -> list:
        """This work item's attempts, ordered by `attempt_number`."""
        attempts = [e for e in self.entries_for(work_item_id) if e.get("event_type") == EVENT_ATTEMPT]
        return sorted(attempts, key=lambda e: (e.get("attempt_number") or 0, e.get("event_sequence") or 0))

    def needs_fix(self) -> list:
        """Outstanding work items in this run - the fix pass's queue.

        Replaces globbing for `*_ERROR`. An item resolved by a later line is not
        returned, and only the latest attempt for an item appears.
        """
        return outstanding(self.entries())

    # -- internals ------------------------------------------------------------

    def _new_entry(self, event_type: str, moment: datetime, **entry_fields) -> JournalEntry:
        unknown = set(entry_fields) - ENTRY_FIELDS
        if unknown:
            raise TypeError(
                f"Unknown journal field(s) {sorted(unknown)}. "
                f"Valid names are the journal's fields: {', '.join(sorted(ENTRY_FIELDS))}"
            )
        derived = set(entry_fields) & DERIVED_FIELDS
        if derived:
            raise TypeError(f"{sorted(derived)} are measured by the journal and cannot be passed in")

        entry_fields = {k: v for k, v in entry_fields.items() if v is not None}
        if "target_pointer" in entry_fields:
            # Stored relative to the data dir, whoever passed it (the passes,
            # batch collect, a pack's own verbs): see `relative_pointer`.
            entry_fields["target_pointer"] = relative_pointer(entry_fields["target_pointer"])
        work_item_id = entry_fields.get("work_item_id")
        entry = JournalEntry(**entry_fields)
        entry.event_id = uuid.uuid4().hex
        entry.event_type = event_type
        entry.timestamp_utc = _iso(moment)
        entry.run_id = self.run_id
        if entry.code_revision is None:
            entry.code_revision = self.code_revision
        if entry.resolved_config_sha256 is None:
            entry.resolved_config_sha256 = self.resolved_config_sha256
        if entry.usage_available is None and event_type == EVENT_ATTEMPT:
            # A fact about the provider's reply, not a guess: usage is available
            # exactly when the caller had token counts to pass.
            entry.usage_available = any(
                field in entry_fields for field in
                ("input_tokens", "output_tokens",
                 "cache_read_tokens", "cache_write_tokens"))
        if event_type == EVENT_ATTEMPT and work_item_id:
            if entry.attempt_number is None:
                entry.attempt_number = self._attempt_counts.get(work_item_id, 0) + 1
            if entry.parent_attempt_id is None:
                # Retry lineage, so a chain of attempts can be walked backwards
                # without re-deriving it from timestamps.
                entry.parent_attempt_id = self._last_attempt.get(work_item_id)
        return entry

    def _envelope(self, entry: JournalEntry, extra: Optional[Mapping]) -> dict:
        """Provider metadata, usage and ids, defaulting to what the line records."""
        envelope = {
            "attempt_id": entry.attempt_id,
            "provider": entry.provider,
            "requested_model": entry.requested_model,
            "returned_model": entry.returned_model,
            "execution_mode": entry.execution_mode,
            "provider_request_id": entry.provider_request_id,
            "batch_id": entry.batch_id,
            "usage": {
                "input_tokens": entry.input_tokens,
                "output_tokens": entry.output_tokens,
                "cache_read_tokens": entry.cache_read_tokens,
                "cache_write_tokens": entry.cache_write_tokens,
                "estimated_cost": entry.estimated_cost,
                "available": entry.raw_payload_available and entry.usage_available,
            },
            "raw_payload_sha256": entry.raw_payload_sha256,
        }
        if entry.sent_effort:
            # Only when one was sent (v0.6.0): an envelope without it is what
            # every attempt before it wrote.
            envelope["sent_effort"] = entry.sent_effort
        if entry.sent_task_budget:
            # Likewise only when one was sent (v0.6.1).
            envelope["sent_task_budget"] = entry.sent_task_budget
        if extra:
            envelope.update(extra)
        return envelope

    def _append(self, entry: JournalEntry) -> None:
        self._sequence += 1
        entry.event_sequence = self._sequence
        _append_line(self.journal_path, json.dumps(entry.to_dict(), ensure_ascii=False))
        self._remember_attempt(entry.to_dict())

    def _remember_attempt(self, entry: Mapping) -> None:
        item = entry.get("work_item_id")
        if not item or entry.get("event_type") != EVENT_ATTEMPT:
            return
        self._attempt_counts[item] = max(
            self._attempt_counts.get(item, 0), entry.get("attempt_number") or 0
        )
        if entry.get("attempt_id"):
            self._last_attempt[item] = entry["attempt_id"]


class NullJournal:
    """The journal when no run has been started. Records nothing, raises nothing.

    Wiring the journal in must not make the library unusable outside a run, and
    must not make every test set one up. A no-op default means `current()` is
    always safe to call, so call sites read as plain statements rather than as
    `if journal is not None:` at every point.

    It deliberately keeps `record_attempt`'s **required** `raw_payload` keyword.
    Dropping it here would let a call site forget the argument and still pass its
    tests, and preserving the payload before anything can fail is the one
    property this module exists for.
    """

    run_id = None
    directory = None

    def record_attempt(self, *, work_item_id, raw_payload, **kwargs):
        return None

    def resolve(self, work_item_id, resolves_attempt_id, **kwargs):
        return None

    def needs_fix(self) -> list:
        return []

    def __repr__(self) -> str:
        return "NullJournal()"


_current: Any = NullJournal()


def usage_fields(usage, model: Optional[str] = None) -> dict:
    """Token counts for `record_attempt`, plus `estimated_cost` when `model` is
    named, or nothing at all.

    Takes a `providers.types.Usage` (an `LLMResponse.usage`) or any object
    with the same attribute names, and omits every key when nothing was
    measured: `record_attempt` derives `usage_available` from their *presence*,
    so zeros would turn "we do not know what this cost" into "this cost
    nothing", and `costs` would price it as free. All four counts, because
    `openai_provider._usage` subtracts cached tokens out of `input_tokens`
    and reports them as `cache_read_tokens`, which are billable too.

    `estimated_cost` is stamped at write time, under the prices then in
    force: the field existed on `JournalEntry` from the start and was never
    populated, so "what did this run cost" meant re-deriving it against a
    table that may have moved since.

    Every model call a domain operation makes should spread this into its
    `record_attempt` (`**journal.usage_fields(response.usage, model)`); the
    history pack's `periods --split` once did not and was unpriceable.
    """
    if usage is None:
        return {}
    fields = {}
    for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens"):
        value = getattr(usage, name, None)
        if value:
            fields[name] = int(value)
    if not fields:
        return {}
    if model:
        from universe_builder.providers import pricing

        cost = pricing.estimate_cost(usage, model)
        if cost is not None:
            fields["estimated_cost"] = cost
    return fields


def _sent_effort(request) -> Optional[str]:
    """What a request put on the wire (`effort.sent_or_none`), or None."""
    asked = getattr(request, "effort", None)
    model = getattr(request, "model", None)
    if not asked or not model:
        return None
    from universe_builder.providers.effort import sent_or_none

    return sent_or_none(model, asked)


def _sent_task_budget(request) -> Optional[int]:
    """The task budget a request put on the wire (`task_budget.of_request`)."""
    if not getattr(request, "task_budget", None):
        return None
    from universe_builder.providers.task_budget import of_request

    return of_request(request)


def record_call(request, response=None, *, operation: str, atom_type: str, subject: str,
                error: Optional[str] = None, work_item: Optional[str] = None,
                target_pointer=None, schema_id: Optional[str] = None,
                replaces=None) -> None:
    """Journal one direct `providers.run.complete` call made outside an operation.

    For the core verbs that call a model themselves rather than through
    `result_from_call` — `collisions --via model` and `duplicates --via model`
    — and that were once invisible to `costs` for want of it.
    `response` is the `LLMResponse`, or None when the call raised (`error`
    says why). No `work_status`: a judgement about a name queues no repair.

    `target_pointer` (the file the call's answer is about or will be written
    to; stored relative to the data dir, like every pointer) and `schema_id`
    (the reply's schema, e.g. its model class name) are recorded when given
    (v0.2.0): a pack verb whose model call writes an atom
    journals it here, and `provenance`/`costs` then see it like a pass's.
    Omitted, the line is what it always was.

    `replaces` (v0.3.2): the file the answer is about to be written over, as
    `Journal.record_attempt` takes it — pass it only when the answer is good
    and will be written, so the bytes it replaces are kept for `restore`.

    Swallows its own failures, like the two recording points in the passes:
    journalling is observation, and a failed observation must not cost the
    verdict it was observing.
    """
    try:
        raw = getattr(response, "raw_text", None) or None
        model = getattr(request, "model", None)
        reason = None
        if raw is None:
            reason = error or getattr(response, "error", None) or "no text in the response"
        current().record_attempt(
            work_item_id=work_item or work_item_id(atom_type, subject),
            raw_payload=raw,
            raw_payload_unavailable_reason=reason,
            request={"model": model,
                     "system": getattr(request, "system", None),
                     "user": getattr(request, "user", None),
                     "max_tokens": getattr(request, "max_tokens", None),
                     **({"effort": request.effort} if getattr(request, "effort", None) else {}),
                     **({"task_budget": request.task_budget}
                        if getattr(request, "task_budget", None) else {})},
            operation=operation,
            atom_type=atom_type,
            subject=subject,
            requested_model=model,
            sent_effort=_sent_effort(request),
            sent_task_budget=_sent_task_budget(request),
            returned_model=getattr(response, "model", None),
            validation_status=("ok" if getattr(response, "parsed", None) is not None
                               else "failed"),
            validation_errors=[error] if error else None,
            target_pointer=str(target_pointer) if target_pointer is not None else None,
            schema_id=schema_id,
            replaces=replaces,
            **usage_fields(getattr(response, "usage", None), model),
        )
    except Exception as e:  # noqa: BLE001
        log.warning("Could not journal %s call for %r: %s", operation, subject, e)


def record_mechanical(operation: str, atom_type: str, path, what: str, *,
                      kind: str = "edit", save_replaced: bool = False, **entry_fields):
    """Journal a write no model made (after v0.3.0; an end-to-end finding).

    A pack verb that rewrites an atom mechanically — retires it, relabels a
    period, respells a URL — journals it so the atom's provenance
    (`index.build`) shows the write. It is an attempt line, because that is
    what provenance reads, but it is not a call: `execution_mode` is
    `MECHANICAL`, there is no payload (`mechanical <kind>, no model involved:
    <what>` in its place), no model and no usage, and `costs` counts it as a
    mechanical write rather than as an unpriceable call. The pack wrote these
    lines by hand before, unmarked; `is_mechanical` recognises those too.

    `path` is the atom written (its filename carries the work item);
    `entry_fields` are further `JournalEntry` fields. Swallows its own
    failures, like `record_call`: the edit it observes has already happened.

    `save_replaced` (v0.3.2): call this *before* the write and the bytes at
    `path` are kept beside the line (`record_attempt(replaces=)`), so the
    edit can be undone with `universe_builder restore`. Expand's splice,
    append's merge and `restore` itself pass it. Returns the entry, or None
    outside a run or when recording failed.
    """
    try:
        return current().record_attempt(
            work_item_id=(work_item_from_path(atom_type, path)
                          or work_item_id(atom_type, Path(path).stem)),
            raw_payload=None,
            raw_payload_unavailable_reason=f"mechanical {kind}, no model involved: {what}",
            operation=operation,
            atom_type=atom_type,
            subject=Path(path).name,
            validation_status=OK,
            target_pointer=str(path),
            execution_mode=MECHANICAL,
            replaces=path if save_replaced else None,
            **entry_fields,
        )
    except Exception as e:  # noqa: BLE001 - observation must not break the edit
        log.warning("Could not journal %s of %s: %s", operation, Path(path).name, e)
        return None


def is_mechanical(entry: Mapping) -> bool:
    """True for an attempt line no model made.

    Marked ones say so (`execution_mode == MECHANICAL`). Lines written before
    `record_mechanical` carry no marker; they are recognised by shape — no
    model, no provider, no usage, no payload, and a successful validation
    with no failure stage. A model call always names its model, and one that
    answered has a payload; a transport failure is `failed` with a stage. Older
    journals hold many such lines (a pack's list verbs, period relabels, a
    hand adjudication), every one previously counted by `costs` as an
    unpriceable synchronous call.
    """
    if entry.get("execution_mode") == MECHANICAL:
        return True
    if entry.get("execution_mode") or entry.get("batch_id"):
        return False
    evidence_of_a_call = ("requested_model", "returned_model", "provider",
                          "input_tokens", "output_tokens", "cache_read_tokens",
                          "cache_write_tokens", "raw_payload_available", "validation_stage")
    if any(entry.get(k) for k in evidence_of_a_call):
        return False
    return (entry.get("validation_status") or "").lower() == OK


def start_run(config: Optional[Mapping] = None, models: Optional[Mapping] = None,
              run_id: Optional[str] = None):
    """Begin a run and make it the one `current()` returns.

    Entry points call this; the passes call `current()`. Threading a journal
    through every operation signature was the alternative, and it would have
    meant changing every call site of every operation to wire in something most
    of them do not otherwise care about.
    """
    global _current
    _current = Journal.start(run_id=run_id, config=config, models=models)
    log.warning("Journal run %s", _current.run_id)
    return _current


def end_run() -> None:
    """Detach the current run. Safe to call when none was started."""
    global _current
    _current = NullJournal()


def current():
    """The active `Journal`, or a `NullJournal` when no run has been started."""
    return _current


def work_item_id(atom_type: str, subject: str) -> str:
    """A stable name for "this atom", independent of which pass is working on it.

    Built from the content id rather than the filename, so a fix attempt and the
    create attempt it is repairing share one work item even though one of them
    wrote an `_ERROR` name and the other did not — and so the id survives the
    corpus rename.
    """
    return f"{atom_type}:{store.atom_id(atom_type, subject)}"


def work_item_from_path(atom_type: str, path) -> Optional[str]:
    """The work item a content-addressed filename embeds, or None.

    For work that targets an existing file — fix, verify, a batch result routed
    back to its source — the filename is the only place the true identity
    survives: the slug in it is lossy (`Boutros Boutros-Ghali` slugs to
    `boutros_boutros_ghali`, and no reading of that slug hashes back to the
    original), so re-deriving the id from a slug-derived subject mints a
    *second* work item for the same atom. That is exactly what happened on the
    first live batched-fix cycle: the create failure was queued under the true
    subject's id and the repair's success was recorded under the slug's, so the
    repair never cleared the queue entry it answered.

    The filename's `h…` token *is* the id `work_item_id` computes from the true
    subject, so when a target exists it is authoritative. Returns None for
    legacy sequential ids and unparseable names — the caller falls back to the
    subject, which is the pre-existing behaviour.
    """
    try:
        name = store.target_of_error(Path(path)).name
        parsed = store.AtomName.decompose_filename(name)
    except (ValueError, TypeError):
        return None
    if not store.is_hash_id(parsed.padded_id):
        return None
    return f"{atom_type}:{parsed.padded_id}"


def iter_entries(run_ids: Optional[Iterable[str]] = None) -> Iterator[dict]:
    """Every journal line on disk, run by run, oldest run first."""
    for run_id in (list(run_ids) if run_ids is not None else list_runs()):
        yield from read_entries(run_id)
