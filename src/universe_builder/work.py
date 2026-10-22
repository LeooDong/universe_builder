"""What a phase works on, decided in one place.

Five places used to answer "which items does this phase touch, and how many
does `--limit` allow": the dry run (`cli._plan`), the create batch submitter,
the fix/verify batch submitter, the synchronous file pass
(`execute_prompt.augment_and_save_prompt`) and `run_batch submit
--from-history`. They disagreed. The dry run limited fix/verify/expand in
total while every real run limited per atom type; `--limit 0` meant "no limit"
in the dry run and `run_batch` and "nothing" elsewhere; the verify dry run
counted `{"error": ...}` placeholders the pass skips; the fix dry run missed
sentinels the pass restores from the journal first.

`select_work(phase, atom_type, limit)` is the one answer. **`--limit N` is per
atom type and `0` is nothing, on every path.** A work item is one subject
(index, create) or one file (fix, verify, expand, append): a limited expand
batch takes N *files* and asks for every pending entry of each.

`submit_groups` is the one batch-submit loop: each group submits on its own,
a failing group is reported rather than raised, and the caller learns what did
go so a partial submission is never mistaken for none.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple

from universe_builder import pipeline, store, topics

log = logging.getLogger(__name__)

#: Phases whose work items are stored files rather than subjects.
FILE_PHASES = ("fix", "verify", "expand", "append")


def index_refusal(phase: str, atom_type: Optional[str]) -> Optional[str]:
    """Why `phase` refuses the index atom type even when `--atom` names it,
    or None. Only `verify` does: it would re-ask each index atom on another
    tier and rewrite, in place, the list every other atom type is generated
    from. An index atom is replaced by retiring it and running the index
    phase, which a reviewer can diff; fix (a failed index generation's
    sentinel) and a pack's own expand/append on its index stay available by
    name."""
    from universe_builder.registry import active_pack

    if pipeline.modern_phase(phase) != "verify" or atom_type != active_pack().index.name:
        return None
    return (f"verify does not run on the index atom type ({atom_type}): it would re-ask "
            f"every {atom_type} atom and rewrite the list every other atom type is "
            "generated from. Replace an index atom by retiring it and running the index phase.")


def takes(phase: str, atom_type: Optional[str], explicit: bool = False) -> bool:
    """Whether `phase` works on `atom_type` at all (an end-to-end finding).

    The index atom type is never a file phase's work unless the caller named
    it (`explicit`, i.e. `--atom`): a bare `verify` used to re-ask every
    History atom on the balanced tier. Named, `verify` is refused outright
    (`index_refusal`, raised by `select_work`); fix, expand and append go on
    as for any type. Every other type and phase: True.
    """
    from universe_builder.registry import active_pack

    phase = pipeline.modern_phase(phase)
    if phase not in FILE_PHASES or atom_type != active_pack().index.name:
        return True
    return explicit and index_refusal(phase, atom_type) is None


@dataclass(frozen=True)
class WorkItem:
    """One unit of a phase's work."""

    phase: str
    atom_type: str
    #: index: the seed subject; create: the topic. None for a file phase.
    subject: Optional[str] = None
    #: create: the civilisation (or other index subject) that listed it.
    context: Optional[str] = None
    #: A file phase's file: the sentinel for fix, the atom otherwise.
    path: Optional[Path] = None
    #: fix only: the sentinel is not on disk yet; the journal holds its payload
    #: and `reconcile_fix_queue` restores it before the pass walks the folder.
    restore: bool = False
    #: expand only: how many list entries are still names.
    pending: int = 0

    @property
    def topic(self) -> topics.Topic:
        return topics.Topic(self.subject, self.context)

    def label(self) -> str:
        """The dry run's line for this item."""
        if self.phase == "index":
            return self.subject
        if self.phase == "create":
            return f"{self.atom_type:<10} {self.topic}"
        if self.phase == "expand":
            return f"{self.path.name}  ({self.pending} to expand)"
        if self.restore:
            return f"{self.path.name}  (restored from the journal)"
        return self.path.name


def check_limit(limit: Optional[int]) -> Optional[int]:
    """None (no limit) or a count >= 0; a negative limit is refused."""
    if limit is not None and limit < 0:
        raise ValueError(f"--limit must be 0 or more, not {limit}")
    return limit


def apply_limit(items: Sequence, limit: Optional[int], what: str = "") -> list:
    """The first `limit` items (all when None, none when 0), saying what was cut."""
    items = list(items)
    if check_limit(limit) is None or len(items) <= limit:
        return items
    log.warning("Stopping at --limit %d%s; %d not taken", limit,
                f" for {what}" if what else "", len(items) - limit)
    return items[:limit]


def select_work(phase: str, atom_type: Optional[str], limit: Optional[int] = None, *,
                explicit: bool = False) -> List[WorkItem]:
    """What `phase` would act on for `atom_type`, at most `limit` of it.

    Reads the filesystem and the journal; writes nothing and calls nothing.
    The phase's own run makes exactly this selection (after the fix pass has
    restored journal-queued sentinels), so a dry run counts what a run pays for.

    `explicit` says the caller named `atom_type` (`--atom`) rather than
    walking every type: a file phase takes the index atom type only then
    (`takes`), and `verify` refuses it with a ValueError (`index_refusal`).
    """
    phase = pipeline.modern_phase(phase)
    refusal = index_refusal(phase, atom_type) if explicit else None
    if refusal:
        raise ValueError(refusal)
    if not takes(phase, atom_type, explicit):
        return []
    return apply_limit(_select(phase, atom_type), limit, atom_type or phase)


def _select(phase: str, atom_type: Optional[str]) -> List[WorkItem]:
    from universe_builder.registry import active_pack

    index_name = active_pack().index.name
    if phase == "index":
        # Civilisations that already have an index atom are skipped, as
        # `_run_history` skips them: regenerating one would overwrite it.
        return [WorkItem("index", index_name, subject=c) for c in topics.civilisations()
                if not topics.has_real_atom(index_name, c)]
    if phase == "create":
        if atom_type == index_name:
            return []   # produced by the index phase, not by create
        # `topics.needing_create` looked up at call time: only topics without a
        # real atom, since `collect` and `persist` would replace an existing one.
        return [WorkItem("create", atom_type, subject=t.subject, context=t.context)
                for t in topics.needing_create(atom_type)]
    if phase not in FILE_PHASES:
        raise ValueError(f"no work selection for phase {phase!r}")
    return list(_select_files(phase, atom_type))


def _files(atom_type: str) -> List[Path]:
    folder = store.dir_for(atom_type)
    if not folder.exists():
        # A type with nothing stored yet has no work: not an error (1.11).
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.name.startswith(f"{atom_type}_"))


def _usable_text(path: Path) -> Optional[str]:
    """The file's text, or None when it is unreadable or an `{"error": ...}`
    placeholder — which has nothing to verify and nothing to repair from, so
    every pass skips it (create regenerates the topic instead)."""
    return store.usable_text(
        path, unreadable=lambda p, e: log.warning("Skipping %s, which cannot be read: %s", p.name, e))


def _select_files(phase: str, atom_type: str) -> Iterator[WorkItem]:
    from universe_builder.registry import active_pack

    spec = next((s for s in active_pack().all_specs() if s.name == atom_type), None)
    # `expand`/`append` run only on atom types that declare the matching spec.
    if phase == "expand" and not (spec and spec.expand):
        return
    if phase == "append" and not (spec and spec.append):
        return

    if phase == "fix":
        items = [WorkItem("fix", atom_type, path=p) for p in _files(atom_type)
                 if store.is_error_path(p) and _usable_text(p) is not None]
        restorable = [WorkItem("fix", atom_type, path=store.error_path(target), restore=True)
                      for target in restorable_fix_targets(atom_type)]
        # In the order the pass will meet them once restored: by sentinel name.
        yield from sorted(items + restorable, key=lambda w: w.path.name)
        return

    expanding = None
    if phase == "expand":
        from universe_builder.operations.expand import expand_config

        expanding = expand_config(atom_type)
    for path in _files(atom_type):
        if store.is_error_path(path):
            continue
        if phase == "verify":
            if _usable_text(path) is not None:
                yield WorkItem("verify", atom_type, path=path)
            continue
        if expanding is not None:
            from universe_builder.operations.expand import entries_to_expand

            data = store.read_json(path)
            try:
                left = entries_to_expand(data, expanding) if isinstance(data, dict) else []
            except Exception as e:  # noqa: BLE001 - one bad file must not end the selection
                log.warning("expand: skipping %s: %s", path.name, e)
                continue
            if left:
                yield WorkItem("expand", atom_type, path=path, pending=len(left))
            continue
        yield WorkItem(phase, atom_type, path=path)


def restorable_fix_targets(atom_type: str) -> List[Path]:
    """Targets whose sentinel the fix pass would restore from the journal.

    The read-only half of `operations._support.reconcile_fix_queue`: a
    journal-queued item whose target and sentinel are both gone and whose
    payload the journal kept.
    """
    from universe_builder import journal
    from universe_builder.operations._support import fix_queue_targets

    return [target for entry, target in fix_queue_targets(atom_type)
            if journal.raw_payload_of(entry) is not None]


# ---------------------------------------------------------------------------
# The one batch-submit loop
# ---------------------------------------------------------------------------

@dataclass
class Submission:
    """What a batch submission did: the handles that went, the groups that failed."""

    submitted: List[str]
    failed: List[Tuple[str, str]]

    @property
    def partial(self) -> bool:
        return bool(self.submitted and self.failed)


def submit_groups(groups: Iterable[Tuple[str, Callable[[], Iterable]]]) -> Submission:
    """Run each `(label, submit)` group; a group that raises is recorded, not raised.

    One provider rejecting one batch used to raise out of the CLI after other
    groups had already been submitted (and would be billed), with no word of
    which. Every group is attempted; the caller reports both lists.
    """
    submitted: List[str] = []
    failed: List[Tuple[str, str]] = []
    for label, submit in groups:
        try:
            paths = [str(p) for p in submit()]
        except Exception as e:  # noqa: BLE001 - one group must not hide the others
            log.error("Submitting %s failed: %s", label, e)
            failed.append((label, str(e)))
            continue
        submitted.extend(paths)
    return Submission(submitted, failed)
