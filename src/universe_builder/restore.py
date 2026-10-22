"""Put back the bytes an in-place rewrite replaced: `universe_builder restore`.

Every write over an existing atom on behalf of a model call — verify, a batch
collect, expand's splice, append's merge, a create over a file that was there
— keeps the bytes it replaced in its journal attempt folder (`replaced.json`,
hash on the line as `replaced_sha256`; `journal.record_attempt(replaces=)`,
v0.3.2). This reads them back for one attempt and writes them over the atom
again, through `store`.

The restore is itself an in-place rewrite, so it is journaled the same way:
a mechanical `restore` line that keeps the bytes *it* replaced. Restoring
that line's attempt undoes the restore. Dry by default, like `retire`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from universe_builder import journal, settings, store

log = logging.getLogger(__name__)


@dataclass
class RestorePlan:
    """What restoring one attempt would do. Built by `plan`; read only."""

    entry: dict
    #: The atom the attempt wrote over.
    target: Optional[Path]
    #: The bytes it replaced, hash-checked; None when it replaced nothing.
    data: Optional[bytes]
    #: What sits at `target` now; None when the file is absent.
    current: Optional[bytes]

    @property
    def attempt_id(self) -> str:
        return self.entry.get("attempt_id")

    @property
    def replaced_nothing(self) -> bool:
        return self.data is None

    @property
    def already_restored(self) -> bool:
        return self.data is not None and self.current == self.data

    def describe(self) -> list:
        e = self.entry
        rows = [
            f"attempt   {self.attempt_id}  (run {e.get('run_id')}, {e.get('timestamp_utc')})",
            f"operation {e.get('operation') or '?'}"
            + (f" ({e['execution_mode']})" if e.get("execution_mode") else "")
            + (f", model {e.get('requested_model')}" if e.get("requested_model") else ""),
            f"target    {_shown(self.target)}",
        ]
        if self.data is not None:
            rows.append(f"saved     {len(self.data):>8} bytes  sha256 {journal.sha256_hex(self.data)}")
        if self.current is None:
            rows.append("now       (absent)")
        else:
            rows.append(f"now       {len(self.current):>8} bytes  sha256 {journal.sha256_hex(self.current)}")
        return rows


def _shown(path: Optional[Path]) -> str:
    if path is None:
        return "(none recorded)"
    try:
        return path.relative_to(settings.data_dir()).as_posix()
    except ValueError:
        return str(path)


def _inside_data_dir(path: Path) -> bool:
    root = settings.data_dir().resolve()
    resolved = path.resolve()
    return resolved != root and root in resolved.parents


def plan(attempt_id: str, run_id: Optional[str] = None) -> RestorePlan:
    """The restore of one attempt, read only.

    Raises `LookupError` when no attempt line has that id; `ValueError` when
    the saved bytes fail their hash, or the target lies outside the data dir
    (a pointer from before v0.1.20 is absolute, and in a copied corpus it
    names the original — the copied-corpus hazard); `FileNotFoundError`
    when the line names saved bytes that are gone.
    """
    entry = journal.find_attempt(attempt_id, run_id)
    if entry is None:
        where = f"run {run_id}" if run_id else "any run"
        raise LookupError(f"no attempt {attempt_id!r} in {where} under {journal.runs_dir()}")
    data = journal.replaced_bytes(entry)
    target = journal.target_path(entry)
    if data is not None:
        if target is None:
            raise ValueError(f"attempt {attempt_id!r} saved replaced bytes but names no target")
        if not _inside_data_dir(target):
            raise ValueError(f"attempt {attempt_id!r} points outside the data dir "
                             f"({target}); refusing to write there")
    current = target.read_bytes() if target is not None and target.is_file() else None
    return RestorePlan(entry=entry, target=target, data=data, current=current)


def apply(restore_plan: RestorePlan):
    """Write the saved bytes back over the target. Returns the restore's own
    journal entry, or None when the target already holds them.

    Needs an open journal run (`journal.start_run`) and refuses without one:
    the restore replaces the atom's current bytes, and they are kept only by
    journaling the restore. A restore whose journal line could not be
    written is refused for the same reason, before anything is written.
    """
    if restore_plan.data is None:
        raise ValueError(f"attempt {restore_plan.attempt_id!r} replaced nothing; nothing to restore")
    if isinstance(journal.current(), journal.NullJournal):
        raise RuntimeError("restore needs an open journal run, to keep the bytes it replaces")
    target = restore_plan.target
    entry = restore_plan.entry
    atom_type = entry.get("atom_type")
    if not atom_type:
        atom_type = store.atom_type_of(target)
    with store.path_lock(target):
        if target.is_file() and target.read_bytes() == restore_plan.data:
            return None
        line = journal.record_mechanical(
            "restore", atom_type, target,
            f"restored the bytes attempt {restore_plan.attempt_id} (run {entry.get('run_id')}) replaced",
            kind="restore", save_replaced=True)
        if line is None:
            raise RuntimeError(f"could not journal the restore of {target.name}; nothing written")
        store.write_bytes(target, restore_plan.data)
    log.warning("Restored %s from attempt %s", target.name, restore_plan.attempt_id)
    return line
