import logging
import re

from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from universe_builder import pipeline, registry, settings, store, topics, work
from universe_builder.providers import run as provider_run

# `merge_helper` keeps its old name here: callers and tests reach it as `ep.merge_helper`.
from universe_builder.operations import merge as merge_helper


log = logging.getLogger(__name__)

# Registries, configuration and the operations themselves now live in their own
# modules so that `operations/` can import them without importing this file. The
# names are re-exported here because that is where every caller and test expects
# them. The configuration half is meant to move into pipeline.yaml.

from universe_builder.atoms import (                                             # noqa: E402
    AtomMapping, atom_mapping, class_mapping, resolve_stored_class, STAGE_MODELS, BASE_ATOM_TYPES,
)
from universe_builder.config import (                                            # noqa: E402
    EXPAND_SYSTEM_TEXT, FIX_SYSTEM_TEXT, OperationConfig, PHASE_MODELS,
    PROVIDER_MODELS, VERIFY_SYSTEM_TEXT, base_system_text, llm_call_mapping,
)
from universe_builder.operations import (                                        # noqa: E402
    Outcome, Result, create_atom, process_append_operation,
    process_expand_operation, process_fix_operation, process_verify_operation,
)
from universe_builder.operations import _support as _ops_support                  # noqa: E402
from universe_builder.operations.expand import expand_config, write_expanded      # noqa: E402


def _base_system() -> str:
    """Use the same pack persona and fallback as batch requests."""
    from universe_builder.providers.requests import base_system_text

    return base_system_text()


__all__ = [
    "AtomMapping", "atom_mapping", "class_mapping", "resolve_stored_class",
    "STAGE_MODELS", "BASE_ATOM_TYPES", "OperationConfig", "PHASE_MODELS", "PROVIDER_MODELS",
    "model_for_atom",
    "base_system_text", "llm_call_mapping", "FIX_SYSTEM_TEXT",
    "VERIFY_SYSTEM_TEXT", "EXPAND_SYSTEM_TEXT", "Outcome", "Result",
    "create_atom", "process_append_operation", "process_expand_operation",
    "process_fix_operation", "process_verify_operation",
    "call_and_save_prompt", "augment_and_save_prompt", "run_phase",
]


def call_and_save_prompt(config: OperationConfig, topic_list, concurrency: int = None):
    """Create one atom per topic. Never raises: a bad topic must not end a run.

    Concurrency comes from `pipeline.yaml`'s `defaults.concurrency` (currently 4);
    the `1` here is only the fallback when that key is absent. At 1 (or a single
    topic) the run is serial with no pool at all. Above 1, the first
    topic is run alone before the pool starts, and that is not politeness — every
    call for an atom type shares a cached prefix, and a cache entry only becomes
    readable once the first response has begun streaming. Firing the whole batch
    at once means every request writes the cache and none reads it, which is
    slower *and* more expensive than running serially. `providers/run.run_many`
    makes the same trade for the same reason.
    """
    topics = list(topic_list)
    if concurrency is None:
        concurrency = int(_pipeline_config().defaults.get("concurrency", 1))
    results = []

    def one(topic):
        result = create_atom(config, topic)
        if result.outcome is not Outcome.OK:
            log.error("create %s: %s", result.metadata.get("subject", topic), result.error)
        return result

    if concurrency <= 1 or len(topics) <= 1:
        return [one(topic) for topic in topics]

    results.append(one(topics[0]))          # warm the shared prefix
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results.extend(pool.map(one, topics[1:]))
    return results


@lru_cache(maxsize=4)
def _pipeline_config(path: str = None):
    return pipeline.load(path)


def augment_and_save_prompt(config: OperationConfig, limit: int = None,
                            concurrency: int = None, explicit: bool = False) -> list:
    """Run one augment pass over every stored file for an atom type.

    Only the files the operation actually acts on are visited
    (`work.select_work`): `fix` sees `_ERROR` sentinels, `verify` valid atoms,
    `expand` atoms with entries left, `append` valid atoms. Filtering up front
    is not just tidiness — the warm-up call below must land on a file that does
    real work, and a `verify` pass whose folder held a sentinel used to waste
    the warm slot on it (sentinels sort first, verify skips them) and then fire
    the pool cold, defeating the caching the warm-up exists to protect.

    `concurrency` comes from the caller's resolved config; None falls back to
    the bundled default. It is a parameter rather than a `_pipeline_config()`
    read here so a `--config` run's concurrency is the one it selected, not the
    default's.

    There is no `output_folder` parameter: `universe_builder.store` resolves locations from
    the atom type, so passing a directory in could only ever disagree with where
    the files actually are.

    Returns every `Result` produced, so a caller can count what happened. It used
    to return `None`, which is why a pass that failed on every file was
    indistinguishable from one with nothing to do.

    `explicit`: the caller named this atom type (`--atom`). The index atom
    type is a file phase's work only then (`work.takes`), and the fix
    pass's journal restore is skipped with it, since it writes.
    """
    log.info("Augmenting prompt")
    if not work.takes(config.operation_type, config.atom_name, explicit):
        return []
    if config.operation_type == "fix":
        # The journal is the queue's authority; the glob below is its serving
        # mechanism. Restoring lost sentinels first means the walk serves the
        # union of the two rather than silently under-fixing.
        _ops_support.reconcile_fix_queue(config.atom_name)

    # `work.select_work` is the selection the dry run and the batch submitters
    # make too: `fix` sees `_ERROR` sentinels, `verify`
    # valid atoms, `expand` atoms with entries left, `append` atoms; never an
    # `{"error": ...}` placeholder; and `--limit` caps the files *visited* per
    # atom type, up front, because the pool below makes "stop after N
    # results" meaningless — several files are in flight at once.
    paths = [item.path for item in work.select_work(config.operation_type, config.atom_name, limit,
                                                    explicit=explicit)
             if not item.restore]   # one `reconcile_fix_queue` could not restore

    def _augment_one_unlocked(file_path):
        """Run the pass over one file. Returns a Result, or None to skip it."""
        log.info(f"Looking at file path: {file_path}")
        # One unreadable file must cost one record, not the pass. This walks
        # every file in an atom folder, and an unguarded
        # read aborts all of them on the first bad byte.
        try:
            data_literal = store.read(file_path)
        except (OSError, UnicodeDecodeError) as e:
            log.error("Skipping %s, which cannot be read: %s", file_path.name, e)
            return None

        data = store.read_json(file_path)
        if data is None:
            log.error(f"{file_path.name} is not valid JSON; passing the raw text through")
        elif store.is_empty_model(data_literal):
            log.info(f"Skipping {file_path.name} because it is empty")
            return None

        try:
            parsed = store.AtomName.decompose_filename(store.target_of_error(file_path).name)
        except ValueError as e:
            log.warning("Skipping %s: %s", file_path.name, e)
            return None
        topic = (
            re.sub(r"\.json.*$", "", parsed.readable_filename.lower().replace("_", " "))
            if parsed.readable_filename
            else None
        )

        if config.operation_type == "expand":
            result = process_expand_operation(data, config, topic, target=file_path)
            if result.should_write:
                # Journals the bytes it replaces, once for the whole splice.
                write_expanded(file_path, result.atom, config.atom_name,
                               result.metadata.get("expanded", 0))
            else:
                # SKIPPED means the file was already fully expanded, so rewriting
                # it would only churn its mtime for an identical payload.
                log.info("expand %s: %s (%s)", file_path.name, result.outcome.value, result.error)
            return result
        elif config.operation_type == "append":
            return process_append_operation(data, config, topic, target=file_path)
        elif config.operation_type == "fix":
            if store.is_error_path(file_path):
                return _apply_fix(file_path, data_literal, config, topic)
        elif config.operation_type == "verify":
            if not store.is_error_path(file_path):
                return _apply_verify(file_path, data_literal, config, topic)
        return None

    def _augment_one(file_path):
        if config.operation_type == "expand":
            # Serialize the whole read/modify/write, including resumed expansions.
            with store.path_lock(file_path):
                return _augment_one_unlocked(file_path)
        return _augment_one_unlocked(file_path)

    # Bounded concurrency, the same trade `call_and_save_prompt` makes for
    # create: every file's calls share one cached prefix, and a cache entry is
    # only readable once the first response has begun streaming — so the first
    # file runs alone to warm it, then the pool starts. Each file writes its
    # own path (per-path locked in `store`) and journals through an O_APPEND
    # line, so parallel files never contend for the same write.
    if concurrency is None:
        concurrency = int(_pipeline_config().defaults.get("concurrency", 1))
    if concurrency <= 1 or len(paths) <= 1:
        results = [_augment_one(p) for p in paths]
    else:
        results = [_augment_one(paths[0])]
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results.extend(pool.map(_augment_one, paths[1:]))
    return [r for r in results if r is not None]


def _apply_fix(sentinel, data_literal: str, config: OperationConfig, topic) -> Result:
    """Promote a fix only if it produced a real atom. This is the B8 fix.

    The check is now structural rather than a guess: `Result.should_write` is
    true only when the operation returned a validated atom. The previous version
    wrote whatever came back and then deleted the sentinel unconditionally, so an
    unsuccessful fix wrote a stub over the target *and* destroyed the `_ERROR`
    file holding the only copy of what the model had produced.
    """
    result = process_fix_operation(data_literal, config, topic, target=sentinel)

    if not result.should_write:
        log.error(
            "Fix for %s did not produce a valid atom (%s); keeping %s so the payload is not lost",
            topic, result.outcome.value, sentinel.name,
        )
        return result

    store.commit_fix(sentinel, result.atom)
    log.info("Fixed %s", store.target_of_error(sentinel).name)
    return result


def _apply_verify(file_path, data_literal: str, config: OperationConfig, topic) -> Result:
    """Rewrite a valid file only if verification produced a valid atom.

    The other half of B8: verify runs over files that are already good, so
    writing back anything less than a validated atom replaces real data.
    """
    result = process_verify_operation(data_literal, config, topic, target=file_path)

    if not result.should_write:
        log.error(
            "Verification of %s returned no usable data (%s); leaving %s unchanged",
            topic, result.outcome.value, file_path.name,
        )
        return result

    store.write_atom(file_path, result.atom)
    return result


def model_for_atom(phase: str, atom: str = None, override: str = None,
                   provider: str = None, config: str = None) -> str:
    """The model a phase will use for one atom type.

    The single place that answers it. It once had a twin: the `run_*` helpers
    behind `main()` (deleted in v0.3.0) indexed
    `PHASE_MODELS`, which is phase-level only, so `atoms.person.create` applied
    through `universe_builder run` and was silently ignored by `main()`.
    """
    # The override goes *through* `resolve_model`, not around it. Returning it
    # raw skipped tier resolution, so `--model balanced` reached
    # `llm_call_mapping["balanced"]` and raised KeyError — while the dry run,
    # which does resolve, displayed `gpt-5.6-luna`. Display and execution
    # disagreeing is worse than either being wrong on its own.
    return _pipeline_config(config).resolve_model(phase, atom, override, provider=provider).value


def run_phase(phase: str, atoms=None, model: str = None, limit: int = None,
              provider: str = None, config: str = None, explicit: bool = False,
              effort: str = None, task_budget: int = None) -> int:
    """Run exactly one phase, over one or all atom types.

    The `run_*` helpers behind the old `main()` (deleted in v0.3.0) each bundled
    two or three phases together, which is why there was no way to fix without
    also verifying. This is the single-phase entry the CLI drives. It returns a
    process exit status: non-zero when a phase produced no successful result at
    all, so a scripted run can tell "nothing to do" from "everything failed".

    `config` is the pipeline file the CLI was given (None = the bundled one),
    and it has to reach both decisions made here. It used not to: `cmd_run`
    resolved models and installed the retry policy from `--config`, then this
    function re-read the *default* file for both — so an alternate config
    governed the dry-run display and the journal record while the actual run
    resolved its models and retries from a file the user had pointed away from.

    `explicit`: `atoms` were named by the user (`--atom`), not every type. A
    file phase takes the index atom type only then (`work.takes`).

    `effort` is the CLI's `--effort` (v0.6.0); like `model` it is an override
    and is resolved per atom type against the config's `effort:` block. None
    everywhere sends none.

    `task_budget` is the CLI's `--task-budget` (v0.6.1), resolved the same way
    against the config's `task_budget:` block.
    """
    atoms = list(atoms) if atoms else [
        a for a in atom_mapping if not a.startswith(("append_", "expand_"))
    ]
    # A phase run may be entered directly (tests, scripts), not only through
    # the CLI, so the retry/timeout policy is installed here too — from the
    # same config the models resolve from.
    provider_run.configure(_pipeline_config(config).retry)
    concurrency = int(_pipeline_config(config).defaults.get("concurrency", 1))
    # The same number bounds the file pool (below) and expand's per-period pool
    # (inside the operation), which nest — the gate stops their product from
    # reaching the API. Set once here, from the selected config.
    _ops_support.set_call_concurrency(concurrency)
    # The output-token budget, from the same config — the one `cmd_run`'s batch
    # branch sends. Once only that branch read it, and every
    # synchronous call ran at `DEFAULT_MAX_TOKENS` whatever the config said.
    budget = _pipeline_config(config).max_tokens(phase)
    failures = successes = 0

    def model_for(atom_name=None) -> str:
        """Resolve per atom type, not once for the whole run.

        `model` is the CLI override and wins outright when given. Otherwise the
        answer genuinely depends on the atom: `pipeline.yaml` documents
        `atoms.<atom>.<phase>` overrides, and they were being skipped whenever a
        run covered more than one atom type — the CLI resolved once with
        `atom=None`, which never matches an atom override, and handed the single
        answer down. The config described a feature that only worked with
        `--atom`.
        """
        return model_for_atom(phase, atom_name, override=model, provider=provider,
                              config=config)

    def effort_for(atom_name=None):
        return _pipeline_config(config).resolve_effort(phase, atom_name, override=effort).value

    def task_budget_for(atom_name=None):
        return _pipeline_config(config).resolve_task_budget(phase, atom_name, override=task_budget).value

    # `history` generates the seed atoms every other phase reads, from a flat
    # list rather than from stored files, so it runs once rather than per atom
    # type.
    phase = pipeline.modern_phase(phase)   # `history` is the phase's old name
    if phase == "index":
        index_name = registry.active_pack().index.name
        return _tally(phase, _run_history(model_for(index_name), limit,
                                          max_tokens=budget, effort=effort_for(index_name),
                                          task_budget=task_budget_for(index_name)))

    specs = {s.name: s for s in registry.active_pack().all_specs()}
    for atom_name in atoms:
        if phase == "create" and atom_name == registry.active_pack().index.name:
            continue  # generated by the history phase, not from itself
        if atom_name not in atom_mapping:
            log.warning("Skipping unregistered atom type: %s", atom_name)
            continue

        atom_model = model_for(atom_name)
        atom_effort = effort_for(atom_name)
        atom_task_budget = task_budget_for(atom_name)

        if phase in ("fix", "verify"):
            results = augment_and_save_prompt(
                OperationConfig(
                    atom_name=atom_name,
                    model_version=(atom_model, "augment"),
                    base_system_text=_base_system(),
                    operation_type=phase,
                    max_tokens=budget,
                    effort=atom_effort,
                    task_budget=atom_task_budget,
                ),
                limit=limit,
                concurrency=concurrency,
                explicit=explicit,
            )
        elif phase == "expand":
            if atom_name not in specs or not specs[atom_name].expand:
                continue
            results = augment_and_save_prompt(
                expand_config(atom_name, atom_model, base_system_text=_base_system(),
                              max_tokens=budget, effort=atom_effort,
                              task_budget=atom_task_budget),
                limit=limit,
                concurrency=concurrency,
                explicit=explicit,
            )
        elif phase == "append":
            if atom_name not in specs or not specs[atom_name].append:
                continue
            results = augment_and_save_prompt(
                OperationConfig(
                    atom_name=atom_name,
                    model_version=(atom_model, "augment"),
                    base_system_text=_base_system(),
                    operation_type="append",
                    max_tokens=budget,
                    effort=atom_effort,
                    task_budget=atom_task_budget,
                ),
                limit=limit,
                concurrency=concurrency,
                explicit=explicit,
            )
            # The append pass is generate-then-merge (ARCHITECTURE §4): without
            # this, `universe_builder run --phase append` would leave `append_<atom>_*.json`
            # beside the atoms with nothing to consume them. `process_directory`
            # pairs and promotes in place, retiring each consumed append.
            merge_helper.process_directory(str(settings.data_dir()), atom_name)
        elif phase == "create":
            results = _run_create(atom_name, atom_model, limit, max_tokens=budget, effort=atom_effort,
                                  task_budget=atom_task_budget)
        else:
            log.error("Phase %r is not runnable on its own", phase)
            return 2

        for result in results or []:
            if result.outcome is Outcome.OK:
                successes += 1
            elif result.keeps_work_queued:
                failures += 1

    log.warning("phase %s: %d succeeded, %d still queued", phase, successes, failures)
    return 1 if failures and not successes else 0


def _tally(phase: str, results) -> int:
    """Collapse a phase's results into a process exit status."""
    successes = sum(1 for r in results or [] if r.outcome is Outcome.OK)
    failures = sum(1 for r in results or [] if r.outcome is not Outcome.OK and r.keeps_work_queued)
    log.warning("phase %s: %d succeeded, %d still queued", phase, successes, failures)
    return 1 if failures and not successes else 0


def _run_history(model: str, limit=None, max_tokens=None, effort=None, task_budget=None):
    """Generate one History atom per civilisation that does not have one.

    The same rule as `needing_create`, found the same way: adding a second
    civilisation to the seed list and running the phase would have re-bought
    the Egyptian frontier atom — and `persist` would have *overwritten* it
    with the rerun's output, on the most expensive call in the system.
    Deliberate regeneration is explicit: retire the stored History atom first.
    """
    index_name = registry.active_pack().index.name
    every = topics.civilisations()
    missing = work.select_work("index", index_name)
    if len(missing) != len(every):
        log.warning(
            "index: %d of %d seed subject(s) already have an atom; generating "
            "the %d missing", len(every) - len(missing), len(every), len(missing),
        )
    names = [item.subject for item in work.apply_limit(missing, limit, "index")]
    if not names:
        log.warning("index: nothing to generate — every seed subject has an atom")
        return []
    log.warning("index: %d seed subject(s) on %s", len(names), model)
    return call_and_save_prompt(
        config=OperationConfig(
            atom_name=registry.active_pack().index.name,
            model_version=model,
            base_system_text=_base_system(),
            max_tokens=max_tokens,
            effort=effort,
            task_budget=task_budget,
        ),
        topic_list=names,
    )


def _run_create(atom_name: str, model: str, limit=None, max_tokens=None, effort=None,
                task_budget=None):
    """Generate one atom per subject the stored History atoms list.

    One `OperationConfig` per context rather than one overall, because the
    context goes into the system text and the system text is the cached prefix.
    Mixing two civilisations into one config would mean neither reads the
    other's cache entry.
    """
    # Only topics without a real atom. The call in `create_atom` happens at
    # `LLMCall` construction, so handing this loop an already-generated topic
    # is money spent before anything can decide to skip it.
    found = [item.topic for item in work.select_work("create", atom_name, limit)]
    if not found:
        log.warning("create %s: nothing to generate", atom_name)
        return []

    results = []
    for context, subjects in topics.by_context(found).items():
        framing = f"{_base_system()}\n\nThe context is {context}." if context else _base_system()
        log.warning("create %s: %d subject(s) in %r on %s", atom_name, len(subjects), context, model)
        results.extend(
            call_and_save_prompt(
                config=OperationConfig(
                    atom_name=atom_name,
                    model_version=model,
                    base_system_text=framing,
                    max_tokens=max_tokens,
                    context=context,
                    effort=effort,
                    task_budget=task_budget,
                ),
                topic_list=subjects,
            ) or []
        )
    return results
