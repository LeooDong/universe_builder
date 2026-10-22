"""Retire the atoms an older model wrote, so a better one regenerates them.

The corpus accumulates model generations. Once a later model is producing
materially better atoms, the old ones are not duplicates to adjudicate — there
is exactly one atom per subject and it simply came from a worse model. Moving it
to `_retired/` leaves its topic with no atom, which is precisely what
`topics.needing_create` looks for, so the next create run fills the gap.

**The index atom is never retired, and that rule is the whole reason this is a
command rather than a one-line `find`.** Deleting a `history` atom does not
schedule a regeneration; it removes topics from the work list. Egyptian History
has one atom per model by design, and the gpt-4-turbo one uniquely lists
hundreds of topics the newer one does not. Retire it and those atoms become
*orphans* — still on disk, listed by nothing, absent from coverage and from `links.json`
memberships, and invisible to every future pass. Much of a civilisation would
quietly stop existing while the bytes sat there looking fine. Splitting wrongly
costs a duplicate someone can see; this would cost an invisible hole, which is
the direction this project consistently refuses.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from universe_builder import store
from universe_builder.registry import active_pack

log = logging.getLogger(__name__)


@dataclass
class Plan:
    """Which atoms would be retired, and which were refused."""

    #: atom type -> paths, sorted.
    by_type: Dict[str, List[Path]] = field(default_factory=dict)
    #: Index-atom files matching the model, never retired. See the module docstring.
    refused: List[Path] = field(default_factory=list)

    @property
    def paths(self) -> List[Path]:
        return [p for group in self.by_type.values() for p in group]

    def is_empty(self) -> bool:
        return not self.paths and not self.refused

    def summary(self) -> str:
        counts = ", ".join(f"{atom} {len(paths)}"
                           for atom, paths in sorted(self.by_type.items()))
        return f"{len(self.paths)} atom(s) to retire" + (f" ({counts})" if counts else "")


def plan(model: str, atoms: Optional[Sequence[str]] = None) -> Plan:
    """Every stored atom whose model version contains `model`. Reads only.

    Matching is a case-insensitive substring of the filename's model segment, so
    `gpt-4` catches `gpt-4o-mini` and `gpt-4-turbo` while `gpt-4o` catches only
    the first. The segment is parsed by `AtomName`, never by splitting the
    filename here — a topic slug containing the text must not match.

    `atoms` restricts the plan to those atom types, because retiring by model
    alone is too blunt where a later pass has enriched the atom: a stored city
    holds its expanded periods, so retiring and regenerating it discards that
    work and it must be paid for again — and the periods can cost more than
    regenerating the cities themselves. Excluding one
    atom type from a sweep should not require hand-moving files past
    `store.py`.
    """
    wanted = (model or "").strip().lower()
    result = Plan()
    if not wanted:
        return result

    selected = {a.strip() for a in atoms if a and a.strip()} if atoms else None
    index_name = active_pack().index.name
    for spec in active_pack().all_specs():
        if selected is not None and spec.name not in selected:
            continue
        for path in store.iter_atoms(spec.name):
            try:
                version = store.AtomName.decompose_filename(path.name).model_version
            except ValueError:
                continue
            if wanted not in (version or "").lower():
                continue
            if spec.name == index_name:
                result.refused.append(path)
                continue
            result.by_type.setdefault(spec.name, []).append(path)
    for paths in result.by_type.values():
        paths.sort()
    result.refused.sort()
    return result


def apply(plan_: Plan, *, dry_run: bool = True) -> dict:
    """Move the planned atoms to `_retired/`. Dry by default."""
    if dry_run:
        return {"retired": [], "dry_run": True, "count": len(plan_.paths)}
    retired: List[str] = []
    for path in plan_.paths:
        if not path.exists():
            log.warning("%s is already gone; nothing to retire", path.name)
            continue
        retired.append(str(store.retire_superseded(path)))
    return {"retired": retired, "dry_run": False, "count": len(retired)}


def models_on_disk() -> Counter:
    """Every model version in the corpus, with counts — for the report."""
    found: Counter = Counter()
    for spec in active_pack().all_specs():
        for path in store.iter_atoms(spec.name):
            try:
                found[store.AtomName.decompose_filename(path.name).model_version] += 1
            except ValueError:
                continue
    return found
