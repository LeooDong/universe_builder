"""Expand each entry of a list field into a richer sub-object."""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from universe_builder.atoms import atom_mapping, resolve_stored_class
from universe_builder.config import EXPAND_SYSTEM_TEXT, OperationConfig, llm_call_mapping
from universe_builder.operations._support import (call_budget, call_concurrency, call_slot, record_failure,
                                                  result_from_call)
from universe_builder.operations.types import Result

log = logging.getLogger(__name__)


def prepare_expansion(data, config: OperationConfig):
    """The atom to expand and its current list, or `(None, None)` if done.

    Shared by the synchronous pass and the batch path so both agree on the
    seed-vs-expanded distinction (B7) and on which entries still need work: a
    file already through expand is loaded as its expanded class and only its
    remaining string entries are expanded; a fully-expanded file needs nothing.
    """
    seed_class = atom_mapping[config.augment_atom_name].atom_class
    stored_class = resolve_stored_class(config.atom_name, data)
    if stored_class is not seed_class:
        already = stored_class(**data)
        if already.is_fully_expanded():
            return None, None
        atom_to_expand = already
    else:
        atom_to_expand = seed_class(**data)
    return atom_to_expand, list(getattr(atom_to_expand, config.subatom_to_expand))


def entries_to_expand(data, config: OperationConfig) -> list:
    """The positions of the list entries an expand run would still call for.

    Empty for a fully-expanded atom, and for one whose list holds no names.
    `prepare_expansion` decides, so the dry run's count (`cli._plan`) asks the
    same question the sync pass and batch submit ask before they spend: both
    call only for the entries still stored as a name.
    """
    atom_to_expand, elements = prepare_expansion(data, config)
    if atom_to_expand is None:
        return []
    return [i for i, entry in enumerate(elements) if isinstance(entry, str)]


def splice_expansion(atom_to_expand, config: OperationConfig, expanded_elements) -> object:
    """Rebuild the expanded atom with `expanded_elements` in its list field.

    The one place the expanded atom is assembled — used by the sync pass and
    by batch collect — so the B7 rule (build the *expanded* class, never write
    objects into the seed) holds identically wherever a splice happens.
    """
    return config.expanded_class(
        **{**atom_to_expand.model_dump(), config.subatom_to_expand: expanded_elements}
    )


def write_expanded(target_path, atom, atom_name: str, placed: int) -> None:
    """Write a spliced atom over its file, keeping the bytes it replaces.

    The one write of an expand, sync or batch. Expand's model calls are one
    per list entry and its write is one per atom, so the bytes replaced are
    journaled here, on one mechanical `splice` line, rather than on every
    entry's attempt: N period calls, one rewrite, one copy (v0.3.2). The line
    is mechanical because it makes no call — each call has its own line.
    The caller holds `store.path_lock(target_path)`.
    """
    from universe_builder import journal, store

    journal.record_mechanical("expand", atom_name, target_path,
                              f"{placed} expanded entr{'y' if placed == 1 else 'ies'} spliced",
                              kind="splice", save_replaced=True)
    store.write_atom(target_path, atom)


def expand_config(atom_name: str, model: str = "", base_system_text: str = "",
                  max_tokens: int | None = None, effort: str | None = None,
                  task_budget: int | None = None) -> OperationConfig:
    """The `OperationConfig` for expanding `atom_name`, from the active pack.

    Both the synchronous pass and the batch path build their expand config here,
    reading the list field and seed/entry/expanded classes from the atom's
    `ExpandSpec` in `registry.active_pack()` — so neither hardcodes
    `cityPeriodsOfHistory`/`Population`/`CityExpanded`, and expand works for any
    domain's list field. `model` and `base_system_text` are only needed by a
    caller that makes calls (the sync pass passes the persona); a pure splice
    (batch collect) leaves them empty. `max_tokens` likewise: the sync pass
    passes the configured budget, which the batch path sends on its requests.
    """
    from universe_builder.registry import active_pack

    pack = active_pack()
    for spec in pack.all_specs():
        if spec.name == atom_name and spec.expand:
            ex = spec.expand
            return OperationConfig(
                atom_name=atom_name,
                model_version=(model, "augment"),
                base_system_text=base_system_text,
                subatom_to_expand=ex.list_field,
                expander_class=ex.entry_model,
                expanded_class=ex.expanded_model,
                operation_type="expand",
                # The one-shot example (a `Population` for city expand) belongs in
                # the cached prefix here too, not only on the batch path. It
                # validates against `expander_class` (the call's `atom_class`), the
                # same model it is an instance of. Omitting it shipped a smaller,
                # example-less prompt on the synchronous path than the batch path.
                example_json_file=ex.example,
                max_tokens=max_tokens,
                effort=effort,
                task_budget=task_budget,
            )
    raise KeyError(f"atom {atom_name!r} has no expand spec in pack {pack.name!r}")


def apply_expansions(target_path, expansions, config: OperationConfig) -> int:
    """Splice already-expanded periods into a city file, writing it once.

    Batch collect's counterpart to the synchronous splice. The model calls
    happened hours earlier, and the file may have changed since — the split
    loop inserts periods, a sync expand runs, adjudication edits — so a period
    cannot be placed by its submit-time *position*: an insertion shifts every
    later index, and a positional splice would then write a `Population` at the
    wrong period (corruption) or off the end (loss). Each reply therefore
    carries the period *name* it expanded, and it is placed only where that
    exact name is still an unexpanded string — at the recorded index if it
    still holds the name, else at the one still-string entry that matches it.
    A name that has moved ambiguously, been split away, or been expanded
    already is skipped and logged: the period keeps its string and re-expands
    next run, which is loss of a wasted call, never a wrong period on disk.

    `expansions` is a list of `(index, period_name, atom)`. Returns how many
    periods were placed.
    """
    from universe_builder import store

    with store.path_lock(target_path):
        data = store.read_json(target_path)
        if not isinstance(data, dict):
            log.error("Cannot splice expansions into %s: not a JSON object", target_path)
            return 0
        # `config` says which atom type and list field this is; there is no
        # default, because the only default core could offer would be one
        # domain's (it was once history's city).
        cfg = config
        try:
            atom_to_expand, seed_elements = prepare_expansion(data, cfg)
        except Exception as e:  # noqa: BLE001 - a changed/foreign file must not crash collect
            log.error("Cannot splice expansions into %s (%s); skipping it", target_path, e)
            return 0
        if atom_to_expand is None:
            return 0  # already fully expanded; nothing left to place
        expanded_elements = list(seed_elements)

        def _slot(index, name) -> int | None:
            if 0 <= index < len(expanded_elements) and expanded_elements[index] == name:
                return index
            matches = [i for i, e in enumerate(expanded_elements)
                       if isinstance(e, str) and e == name]
            return matches[0] if len(matches) == 1 else None

        placed = 0
        for index, name, atom in expansions:
            slot = _slot(index, name)
            if slot is None:
                log.warning(
                    "Period %r is no longer a unique unexpanded entry in %s "
                    "(the file changed since submit); skipping — it re-expands next run",
                    name, Path(target_path).name,
                )
                continue
            expanded_elements[slot] = atom
            placed += 1
        if placed:
            write_expanded(target_path, splice_expansion(atom_to_expand, cfg, expanded_elements),
                           cfg.atom_name, placed)
        return placed


def expansion_instructions(entries, index: int, list_field: str) -> str:
    """Shared volatile instructions for sync and batch; entries can belong to any domain."""
    if index + 1 == len(entries):
        following = "There is no following entry."
    else:
        neighbour = entries[index + 1]
        if not isinstance(neighbour, str):
            neighbour = neighbour.model_dump_json()
        following = f"The following entry is: {neighbour}."
    return f"Give a response for '{entries[index]}', one entry of {list_field}. {following}"


def process_expand_operation(data, config: OperationConfig, topic, target=None) -> Result:
    """Expand a list of names into a list of objects.

    The seed model's field holds names (`List[str]`); the expanded model's field
    holds objects. Those are two different shapes, so the result is built as
    `config.expanded_class`, not written back into the seed instance — doing the
    latter produced files that serialised with a warning and could never be read
    back (B7).

    Expansion is best-effort per entry: an entry whose call fails is left as its
    original name, which the expanded model accepts. That is why the operation
    can return `OK` while `metadata["unexpanded"]` is non-empty — a partial
    expansion is real progress and must be written, and a later run resumes it.

    **Expand cannot split a period.** Its shape is one name in, one
    `Population` out, so a seed period that spans a change of government
    cannot be divided here however clearly the schema forbids the span — the
    single-object response format wins. The rule is enforced where a whole
    period *list* is in play: the seed `City.cityPeriodsOfHistory` description
    (create authors the names) and `CityExpanded.cityPeriodsOfHistory`
    (verify/fix can restructure). `Population`'s own descriptions deliberately
    say nothing about splitting, so this call is never ordered to do what its
    schema forbids.
    """
    atom_to_expand, seed_elements = prepare_expansion(data, config)
    if atom_to_expand is None:
        log.info("%s: already fully expanded, skipping", topic)
        stored_class = resolve_stored_class(config.atom_name, data)
        return Result.skipped("already fully expanded", stored_class(**data), subject=topic)
    unexpanded_before = [e for e in seed_elements if isinstance(e, str)]
    if len(unexpanded_before) != len(seed_elements):
        log.warning("%s: resuming a partial expansion (%s)", topic, unexpanded_before)
    expanded_elements = list(seed_elements)

    def _expand_entry(i: int):
        """Expand one period. Returns (index, atom-or-None). Never raises."""
        curr_entry = seed_elements[i]
        try:
            # `call_slot` bounds the model calls in flight across the whole run,
            # so this per-period pool and the outer per-city pool cannot multiply
            # into a rate-limit breach. Construction is where the call happens.
            with call_slot():
                call = llm_call_mapping[config.model_version](
                    atom_type=config.augment_atom_name,
                    atom_class=config.expander_class,
                    model_version=config.model_version[0],
                    prompt_folder=config.prompt_folder,
                    prompt_subject=topic,
                    prompt_start_text=atom_mapping[config.augment_atom_name].prompt_start_text,
                    base_system_text=f"{config.base_system_text}\n{EXPAND_SYSTEM_TEXT}",
                    example_json_file=config.example_json_file,
                    # Serialised as `cli._submit_expand_batch` serialises it,
                    # so the two paths send one prompt.
                    augment_to_json=json.dumps(data, ensure_ascii=False),
                    topic=topic,
                    operation="expand",
                    target_path=target,
                    **call_budget(config),
                    user_instructions=expansion_instructions(seed_elements, i, config.subatom_to_expand),
                )
        except Exception as e:  # noqa: BLE001
            log.error("An error occurred for entry %d in '%s': %s", i, topic, e)
            # Expand is the fifth operation and was missed when the other four
            # were wired: a period whose call died at transport left no record at
            # all, so the entry stayed a bare name with nothing saying why.
            record_failure(config.augment_atom_name, topic, str(e), cause=e,
                           operation="expand", context=str(curr_entry), target=target)
            return i, None

        entry = result_from_call(call, config.expander_class, subject=topic, entry=curr_entry,
                                 operation="expand")
        if entry.ok:
            return i, entry.atom
        # Keeping the period name is strictly better than storing a sentinel
        # where a real object belongs — the name is what a later run resumes from.
        log.error("Expansion of entry %d in '%s' returned no usable data; keeping the name", i, topic)
        return i, None

    # Only the entries still needing expansion; already-expanded ones (a resumed
    # run) stay put. Warm-first then pool, the same trade the create and file
    # loops make: the periods share a cached prefix, and one call must land
    # before the rest read the cache. The global `call_slot` gate is the real
    # bound; this pool size just decides how many periods of THIS city queue.
    todo = [i for i, entry in enumerate(seed_elements) if isinstance(entry, str)]
    if not todo:
        return Result.skipped("no entries to expand", subject=topic)
    workers = min(call_concurrency(), len(todo))
    if workers <= 1:
        outcomes = [_expand_entry(i) for i in todo]
    else:
        outcomes = [_expand_entry(todo[0])]                 # warm the prefix
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes.extend(pool.map(_expand_entry, todo[1:]))

    failures = 0
    for i, atom in outcomes:
        if atom is not None:
            expanded_elements[i] = atom
        else:
            failures += 1

    if failures == len(todo):
        return Result.failed("every expansion failed", subject=topic, failures=failures, total=len(todo))

    expanded = splice_expansion(atom_to_expand, config, expanded_elements)
    unexpanded = [e for e in expanded_elements if isinstance(e, str)]
    if unexpanded:
        log.warning(
            "%s: %d of %d periods left unexpanded: %s",
            topic, len(unexpanded), len(seed_elements), unexpanded,
        )

    return Result.success(
        expanded, subject=topic, unexpanded=unexpanded, failures=failures,
        total=len(seed_elements), expanded=len(todo) - failures,
    )
