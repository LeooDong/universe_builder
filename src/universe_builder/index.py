"""Derived cross-reference index: who lists an atom, and what an atom names.

**Never store what you can derive.** An atom does not record which Histories
list it — the History atoms *are* the membership edges, and writing
"civilisations: [egyptian, greek]" into Alexandria would go stale the moment a
third History lists it or the atom regenerates. Likewise a city's
`notableFigures` are free-text names at generation time, because models are
good at names and would hallucinate ids. This module turns both into edges
*after the fact*, through `topics.resolve_subject` — the same grouping, alias
and prefer-stored-atoms machinery the work list runs on, so a name resolves to
the same atom here as everywhere else.

The output lands in `data/automated/<domain>/_index/links.json`: machinery, not atoms,
skipped by `store.iter_atoms` like every underscore directory, and always
recomputable — a stale index is fixed by running `universe_builder index` again, never by
editing it.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict

from universe_builder import settings, store, topics
from universe_builder.registry import active_pack

log = logging.getLogger(__name__)


def load_atom_from_json(file_path, atom):
    """Read a stored atom file and construct `atom` from it (raises on a bad file).

    Moved here from `functions/json_helper.py` in v0.3.0;
    the old module re-exports it for one release. `topics` imports it lazily,
    since this module imports `topics`.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
        return atom(**data)


def values_at(payload, path: str):
    """Yield `(value, occurrence)` for every string a wire path reaches.

    `path` is dotted for nested objects (`birth.location.city`) and uses `[]`
    for a list of objects (`cityPeriodsOfHistory[].tradingPartners`). A
    one-segment path is a plain field, so declarations written before paths
    existed behave exactly as they did.

    The leaf may hold a string or a list of strings; both yield individual
    values, which is what a scalar link like `originalCreator` and a list link
    like `notableFigures` have always needed between them.

    `occurrence` is the concrete path with list indices filled in
    (`cityPeriodsOfHistory[3].tradingPartners[0]`), so a consumer can say which
    period made an assertion rather than only which atom did.

    Tolerant by construction: a missing key, a `None`, a scalar where an object
    was expected, or a still-unexpanded string inside a `[]` list all yield
    nothing instead of raising. The index reads whatever is on disk, including
    half-migrated and partially-expanded atoms, and one odd shape must not cost
    the run every edge behind it.
    """
    def walk(node, segments, trail):
        if node is None:
            return
        if not segments:
            if isinstance(node, list):
                for position, item in enumerate(node):
                    if isinstance(item, str):
                        yield item, f"{trail}[{position}]"
            elif isinstance(node, str):
                yield node, trail
            return
        if not isinstance(node, dict):
            return
        head, *rest = segments
        listed = head.endswith("[]")
        if listed:
            head = head[:-2]
        step = f"{trail}.{head}" if trail else head
        child = node.get(head)
        if not listed:
            yield from walk(child, rest, step)
        elif isinstance(child, list):
            for position, item in enumerate(child):
                yield from walk(item, rest, f"{step}[{position}]")

    yield from walk(payload, path.split("."), "")


def _resolver(aliases):
    """`resolve(atom_type, name, context=None)`, with one `topics.subject_resolver`
    per atom type built on first use and kept: each carries the full company
    of History-listed spellings, which is what lets `Abydos (Abdju)` find the
    atom stored under bare `Abydos` and bare `Narmer` find `Narmer (Menes)`."""
    resolvers: dict = {}
    answers: dict = {}   # (type, name, context) -> id: a build resolves the same name many times

    def resolve(atom_type, name, context=None):
        key = (atom_type, name, context)
        if key in answers:
            return answers[key]
        if atom_type not in resolvers:
            resolvers[atom_type] = topics.subject_resolver(atom_type, aliases=aliases)
        answers[key] = resolvers[atom_type](name, context)
        return answers[key]

    return resolve


def memberships(resolve=None, *, resolved: bool = True) -> dict:
    """`build()`'s `memberships` section on its own, without the reference walk.

    Per civilisation, per atom type, each listed subject and the
    `"type:id"` it resolves to (None = listed but never generated), sorted
    as `build` returns it; `memberships() == build()["memberships"]`. The
    first listing of a subject in one History wins. Public since v0.2.0
    for the history pack's fold-history and retire-list
    gates, which kept a line-for-line copy of this loop. Reads the History
    files through `topics.history_files`, so `topics.history_files_as`
    reaches it. `resolve` is `build`'s shared resolver; omitted, one is
    built over the alias file in force.

    `resolved` (v0.4.1): pass the result through the pack's
    `membership_resolver`, as `build` does, so every reader sees one answer;
    False gives what each index atom literally lists.
    """
    with store.read_snapshot():
        raw = _raw_memberships(resolve)
    if resolved and active_pack().membership_resolver:
        return active_pack().membership_resolver(raw)[0]
    return raw


def _raw_memberships(resolve=None) -> dict:
    """What each stored index atom lists, resolved to atom ids; `memberships`'s
    loop before the pack's resolver."""
    if resolve is None:
        resolve = _resolver(topics.load_aliases())
    found: dict = {}
    index_atom = active_pack().index          # History for the history domain
    for path in topics.history_files():
        try:
            instance = load_atom_from_json(str(path), index_atom.model)
        except Exception as e:  # noqa: BLE001 - one bad file costs one file
            log.error("Skipping unreadable history file %s: %s", path.name, e)
            continue
        context = topics.context_of(path.name)
        per_context = found.setdefault(context, {})
        # Iterate the pack's *base* atom specs, not `atom_mapping`. That set
        # excludes by construction the two things a membership must never count:
        # the index atom itself (`history`/`formulary` — the source of the lists,
        # not a member of them, and it can share its `plural` with a base atom, as
        # medications' Formulary and Medication both use "medications"), and the
        # `expand_*`/`append_*` variants (same subjects, different pass, would
        # double-count). Iterating base specs states that intent directly, where
        # the old `atom_type == index_atom.name or "_" in atom_type` guard would
        # also have silently dropped a future base atom named e.g. `side_effect`.
        for spec in active_pack().atoms:
            if not spec.plural:
                continue
            atom_type = spec.name
            per_type = per_context.setdefault(atom_type, {})
            for subject in getattr(instance, spec.plural, None) or []:
                subject = (subject or "").strip()
                if not subject or subject in per_type:
                    continue
                resolved = resolve(atom_type, subject, context=context)
                per_type[subject] = f"{atom_type}:{resolved}" if resolved else None
    return {
        context: {atom: dict(sorted(subjects.items()))
                  for atom, subjects in sorted(per.items())}
        for context, per in sorted(found.items())
    }


def build() -> dict:
    """See `_build`: the same, inside `store.read_snapshot` (a pure read, so one
    folder listing per atom type serves every lookup)."""
    with store.read_snapshot():
        return _build()


def _build() -> dict:
    """Compute the full index. Pure read: nothing on disk changes.

    Returns a dict with three sections, every list sorted so two builds over
    one corpus are byte-identical:

    - `memberships`: per civilisation, per atom type, each listed subject and
      the atom id it resolves to (None = listed but never generated — the same
      judgement as `status --coverage`, because it is the same machinery).
    - `references`: resolved name links between atoms, from `active_pack().link_fields`.
    - `unresolved`: name links nothing on disk answers to. Data for a report,
      not an error — a spouse with no atom is prose until someone decides the
      person deserves one.
    - `duplicates`: per atom type, names that *two stored atoms both declare*.
      One subject in two files, which every existing invariant misses because
      the two content ids genuinely differ — they were hashed from the two
      spellings two models happened to use. See `topics.duplicate_atoms`.
    """
    resolve = _resolver(topics.load_aliases())
    members = _raw_memberships(resolve)
    folds: dict = {}
    if active_pack().membership_resolver:
        members, folds = active_pack().membership_resolver(members)

    references: list = []
    unresolved: list = []
    for atom_type, fields in active_pack().link_fields.items():
        for path in store.iter_atoms(atom_type):
            payload = store.read_json(path)
            if not isinstance(payload, dict) or store.is_empty_model(payload):
                continue
            try:
                source = store.AtomName.decompose_filename(path.name)
            except ValueError:
                continue
            source_id = f"{atom_type}:{source.padded_id}"
            for field, target_type in fields.items():
                for name, at in values_at(payload, field):
                    if not name.strip():
                        continue
                    target = resolve(target_type, name)
                    edge = {"from": source_id, "field": field,
                            "name": name.strip(), "at": at}
                    if target:
                        references.append({**edge, "to": f"{target_type}:{target}"})
                    else:
                        unresolved.append({**edge, "target_type": target_type})

    def _sorted_edges(edges):
        # `at` is part of the key: one atom can cite one name at two
        # occurrences (a trading partner in two periods), and without it the
        # order of equal-name edges would depend on iteration order, which
        # would break "two builds are byte-identical".
        return sorted(edges, key=lambda e: (e["from"], e["field"], e["name"], e["at"]))

    duplicates = {}
    for spec in active_pack().atoms:
        found = topics.duplicate_atoms(spec.name)
        if found:
            duplicates[spec.name] = found

    return {
        "memberships": members,
        "references": _sorted_edges(references),
        "unresolved": _sorted_edges(unresolved),
        "duplicates": duplicates,
        "provenance": _provenance(),
        # Only when the pack's resolver changed something, so a pack without one
        # (or a corpus without folds) builds byte-identically to v0.4.0.
        **({"folds": folds} if folds else {}),
    }


def _provenance() -> dict:
    """Each atom's write history, derived from the journal — never stored.

    This is the answer to stale `certainty`. A `certainty` field is a model's
    self-assessment of how well a period is *known*; it was being read as if it
    also recorded *who wrote the period and when*, so a field rewritten by a
    later, different model on a different date left the certainty "overstating
    how much is known about its own record". The two are orthogonal. Provenance
    — operation, model, timestamp, ordered — belongs in the append-only
    journal, which records every write as it happens and cannot go stale;
    `certainty` stays a truth-confidence claim and answers only for itself.

    Keyed by work item (`atom_type:content_id`), each a list of successful
    writes oldest first: `{operation, model, timestamp, run}`. A period whose
    government was set by `adjudicate` or rewritten by `split`/`fix` shows that
    operation in its atom's history, which is where "was this touched after
    generation, and by what" is answered — not by squinting at `certainty`.
    """
    from universe_builder import journal

    history: dict = defaultdict(list)
    for run_id in journal.list_runs():
        for entry in journal.read_entries(run_id):
            item = entry.get("work_item_id")
            if not item or entry.get("validation_status") != "ok":
                continue
            history[item].append({
                "operation": entry.get("operation") or entry.get("event_type"),
                "model": entry.get("returned_model") or entry.get("requested_model"),
                "timestamp": entry.get("timestamp_utc"),
                "run": run_id,
            })
    # Oldest first within each atom; `timestamp_utc` is ISO-8601, so string
    # sort is chronological, and runs are already listed oldest first.
    return {
        item: sorted(writes, key=lambda w: w.get("timestamp") or "")
        for item, writes in sorted(history.items())
    }


def write(report: dict):
    """Persist the index at its one location. Returns the path."""
    path = settings.index_dir() / "links.json"
    store.write_json(path, report)
    log.warning("Index written to %s", path)
    return path
