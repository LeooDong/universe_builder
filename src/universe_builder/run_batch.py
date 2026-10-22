"""CLI for batched, cached multi-provider generation.

Run after installing the package, from any directory:

    python -m universe_builder.run_batch models
    python -m universe_builder.run_batch estimate --atom person --model claude-haiku-4-5-20251001 --count 500
    python -m universe_builder.run_batch submit   --atom person --model gpt-5-nano --from-history
    python -m universe_builder.run_batch list
    python -m universe_builder.run_batch poll     --all
    python -m universe_builder.run_batch collect  --all
    python -m universe_builder.run_batch cancel   --all [--apply]

Batches take up to 24 hours, so submit and collect are separate commands: the
handle is written to `data/automated/_batches/` and survives process exit.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from universe_builder import journal, pipeline, settings, topics
from universe_builder.registry import active_pack
from universe_builder.providers import run as provider_run
from universe_builder.providers import batch as batch_mod
from universe_builder.providers import estimate_cost, known_models, lookup
from universe_builder.providers.pricing import cache_warning
from universe_builder.providers.requests import build_request, prefix_tokens, with_create_note
from universe_builder.providers.errors import UnknownModel
from universe_builder.providers.types import SchemaMismatch, Usage

log = logging.getLogger(__name__)


def _atom_types() -> list[str]:
    """Atom names accepted by `--atom`, from the active domain pack (was a
    hardcoded list, and `sorted(atom_config())` before that)."""
    from universe_builder import atoms

    return list(atoms.atom_mapping)


#: Default output size for `estimate`, measured rather than guessed: a full
#: `Discovery` or `City` from gpt-5-nano uses close to this many output tokens —
#: reasoning models spend most of the budget thinking before they emit, so the
#: figure is far larger than the JSON that comes out.
#:
#: A much smaller default once understated a large discovery batch several
#: times over. An estimate exists to be trusted before money is spent,
#: so it errs high: smaller atoms (`person`, `event`) will come in under this,
#: and being told a batch is cheaper than it is, is the failure that costs money.
MEASURED_OUTPUT_TOKENS = 9000


def _limit(value: str) -> int:
    """`--limit`: 0 or more; 0 submits nothing."""
    try:
        n = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"not a whole number: {value!r}") from e
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, not {n}")
    return n


def _subjects_from_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _subjects_from_history(atom_type: str, prog: str = "universe_builder") -> list[str]:
    """This atom type's topics, from every stored History atom.

    Delegates to `universe_builder.topics`, which the synchronous create pass also uses. This
    function used to walk the History files itself, and de-duplicated where
    `execute_prompt.run_base_atom_calls` did not — two implementations of one
    question, disagreeing about the answer.

    Note this flattens away each topic's context. `universe_builder run --phase create
    --batch` submits one batch per context instead, because the context sits in
    the cached prefix; a `--list` submission has no context to begin with.

    Only topics without a real atom are returned — a create submission for an
    existing atom is paid for and then *overwrites* it at collect. Deliberate
    regeneration is `--list PATH` with the subjects named.
    """
    found = topics.needing_create(atom_type)
    if not found:
        raise SystemExit(
            f"Nothing to generate for {atom_type}: every listed topic already "
            f"has an atom, or no {active_pack().index.name} atom exists under "
            f"{settings.data_dir() / active_pack().index.name}. Generate the index atoms "
            f"first ({prog} run --phase index), or name subjects with --list "
            f"to regenerate deliberately."
        )
    return [t.subject for t in found]


def cmd_models(_: argparse.Namespace) -> int:
    print(f"{'model':30} {'provider':11} {'in':>7} {'cached':>8} {'out':>7} {'cache min':>10}")
    print("-" * 78)
    for name in known_models():
        m = lookup(name)
        print(
            f"{name:30} {m.provider:11} {m.input:7.3f} {m.cached_input:8.3f} "
            f"{m.output:7.2f} {m.cache_min_tokens:10}"
        )
    print("\nPrices are USD per 1M tokens. Batch mode halves every column.")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    args.model = _model(args)
    request = build_request(args.atom, "Example Subject", args.model)
    prefix = prefix_tokens(request)
    warning = cache_warning(args.model, prefix)

    # Per call: the prefix is read from cache after the first, plus a small
    # volatile suffix, plus output. Output size is the guess here.
    out_guess = args.output_tokens
    cold = Usage(input_tokens=prefix + 100, output_tokens=out_guess)
    warm = Usage(input_tokens=100, cache_read_tokens=prefix, output_tokens=out_guess)

    for label, batch in (("sync", False), ("batch", True)):
        c_cold = estimate_cost(cold, args.model, batch=batch) or 0
        c_warm = estimate_cost(warm, args.model, batch=batch) or 0
        total_uncached = c_cold * args.count
        total_cached = c_cold + c_warm * (args.count - 1)
        print(f"{label:6} {args.count} calls: ${total_uncached:8.2f} uncached  "
              f"${total_cached:8.2f} with caching")

    source = "measured" if out_guess == MEASURED_OUTPUT_TOKENS else "your figure"
    print(f"\nprefix ~{prefix} tokens, assumed output {out_guess} tokens/call ({source})")
    if warning:
        print(f"WARNING: {warning}")
    return 0


def _model(args: argparse.Namespace) -> str:
    """Resolve `--model` through the tier table, so a tier name works here too.

    `pipeline.yaml` speaks tiers now, so `--model economy` is the natural thing
    to type — and it used to fail deep inside the pricing table with
    `'economy' is not in the pricing table`, which points at the wrong problem
    entirely. An exact model id still passes through untouched.
    """
    from universe_builder import pipeline

    config = pipeline.load()
    try:
        return config.resolve_tier(args.model)
    except pipeline.UnknownTier as e:
        raise SystemExit(str(e))


def cmd_submit(args: argparse.Namespace) -> int:
    """Submit create work for one atom type.

    `--from-history` reads the same selection `run --phase create --batch`
    makes (`work.select_work`) and submits **one batch per context**, each
    with its own framing. It used to flatten the topics and apply one
    `--context` (or none) to every subject, so a Greek subject could go out
    framed as Egyptian; `--context` is now refused with it. `--limit` is per
    atom type and `0` submits nothing.
    """
    from universe_builder import work

    if args.list:
        groups_of = {args.context: work.apply_limit(_subjects_from_list(Path(args.list)), args.limit)}
    elif args.from_history:
        if args.context:
            raise SystemExit("--context cannot be combined with --from-history: each subject "
                             "is submitted under the context of the index atom that listed it")
        _subjects_from_history(args.atom, getattr(args, "prog", None) or "universe_builder")
        found = [item.topic for item in work.select_work("create", args.atom, args.limit)]
        groups_of = topics.by_context(found)
    else:
        raise SystemExit("Pass --list PATH or --from-history")

    if not any(groups_of.values()):
        log.warning("--limit %s: nothing submitted", args.limit)
        return 0
    notes = topics.load_create_notes()
    index = topics.ListIndex(args.atom)
    model = _model(args)
    # The configured budget, as `universe_builder run --batch` and the
    # synchronous passes use; this default was once the bare constant, so the
    # same phase got two budgets by two doors.
    max_tokens = args.max_tokens or pipeline.load().max_tokens("create")
    # The effort likewise (v0.6.0): `--effort`, else the config's for create;
    # unset sends none. Refused up front for a level the provider lacks.
    from universe_builder.providers import effort as effort_mod

    effort = pipeline.load().resolve_effort("create", args.atom, getattr(args, "effort", None)).value
    problem = effort_mod.refusal(model, effort)
    if problem:
        raise SystemExit(f"refused: {problem}")
    # The task budget likewise (v0.6.1), refused up front under the minimum,
    # over max_tokens, or for a model that takes none.
    from universe_builder.providers import task_budget as task_budget_mod

    task_budget = pipeline.load().resolve_task_budget("create", args.atom,
                                                      getattr(args, "task_budget", None)).value
    problem = task_budget_mod.refusal(model, task_budget, max_tokens)
    if problem:
        raise SystemExit(f"refused: {problem}")

    groups = []
    for context, subjects in groups_of.items():
        items = [with_create_note(args.atom, s, context, notes, index) for s in subjects]

        def submit(items=items, context=context):
            log.warning("Submitting %d %s subjects to %s (%r)", len(items), args.atom, model, context)
            return batch_mod.submit(args.atom, items, model, context=context,
                                    max_tokens=max_tokens, dry_run=args.dry_run, effort=effort,
                                    task_budget=task_budget)

        groups.append((f"{args.atom} ({context})", submit))

    submission = work.submit_groups(groups)
    for p in submission.submitted:
        print(p)
    for label, error in submission.failed:
        print(f"NOT submitted: {label}: {error}")
    return 1 if submission.failed else 0


def _targets(args: argparse.Namespace) -> list[Path]:
    if args.all:
        paths = batch_mod.list_batches()
        if not paths:
            raise SystemExit(f"No batch handles in {settings.batch_dir()}")
        return paths
    if not args.name:
        raise SystemExit("Pass a handle name or --all")
    path = Path(args.name)
    if not path.exists():
        path = settings.batch_dir() / f"{args.name}.json"
    if not path.exists():
        # Named handles still resolve after archiving, so a deliberate re-collect
        # stays one command. `--all` deliberately does not look here.
        archived = settings.collected_dir() / f"{args.name}.json"
        if archived.exists():
            log.warning("%s was already collected; re-collecting from %s", args.name, archived)
            return [archived]
        raise SystemExit(f"No such batch handle: {args.name}")
    return [path]


def cmd_list(_: argparse.Namespace) -> int:
    paths = batch_mod.list_batches()
    # No early return on an empty list: once handles are archived after
    # collection, "nothing outstanding" is the normal state, and it is exactly
    # then that a reader needs to be told the archive and the missing-results
    # reports exist rather than being shown a bare "No batch handles".
    if paths:
        for path in paths:
            handle = batch_mod.load(path)
            model = handle.model + (f"@{handle.extra['effort']}" if handle.extra.get("effort") else "")
            if handle.extra.get("task_budget"):
                model += f" tb={handle.extra['task_budget']}"
            print(f"{path.stem:34} {handle.provider:10} {model:28} "
                  f"{handle.request_count:6} requests  {handle.batch_id}")
    else:
        print(f"Nothing outstanding in {settings.batch_dir()}")
    collected = sorted(settings.collected_dir().glob("*.json")) if settings.collected_dir().exists() else []
    if collected:
        print(f"\n{len(collected)} already collected, in {settings.collected_dir()}")
        print("  `collect --all` skips these; pass a handle name to re-collect one deliberately.")
    missing = sorted(settings.missing_dir().glob("*.json")) if settings.missing_dir().exists() else []
    if missing:
        print(f"\n{len(missing)} batch(es) lost results:")
        for path in missing:
            count = len(json.loads(path.read_text(encoding="utf-8")))
            print(f"  {path.stem:34} {count} missing")
    return 0


def cmd_poll(args: argparse.Namespace) -> int:
    for path in _targets(args):
        status = batch_mod.poll(path)
        print(f"{path.stem:34} {status.state:22} done={status.done} "
              f"ok={status.succeeded} err={status.errored}")
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    """Cancel the named batch, or with --all every one still running. Lists
    what it would cancel unless --apply: a cancel cannot be taken back."""
    running = []
    for path in _targets(args):
        status = batch_mod.poll(path)
        if status.done:
            print(f"{path.stem:34} {status.state:22} already done, left alone")
            continue
        running.append((path, status))
    for path, status in running:
        if not args.apply:
            print(f"{path.stem:34} {status.state:22} ok={status.succeeded} err={status.errored}"
                  " would be cancelled")
            continue
        after = batch_mod.cancel(path)
        print(f"{path.stem:34} {after.state:22} ok={after.succeeded} err={after.errored} cancel requested")
    if running and not args.apply:
        print(f"\n{len(running)} running; pass --apply to cancel them. Their finished results stay"
              " collectable (`collect` once they read cancelled).")
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    total_cost = 0.0
    incomplete = 0
    ended_short = 0
    for path in _targets(args):
        try:
            status = batch_mod.poll(path)
            if not status.done:
                print(f"{path.stem}: still {status.state}, skipping")
                continue
            report = batch_mod.collect(path)
        except (UnknownModel, SchemaMismatch, ValueError) as e:
            log.error("Skipping %s: %s; check UNIVERSE_BUILDER_DOMAIN matches the submitted pack", path.name, e)
            incomplete += 1
            continue
        total_cost += report.cost or 0
        if report.ended_short:
            # Never ran and never will; listed in missing_dir for a sync create.
            ended_short += 1
            suffix = f", {len(report.missing)} never ran ({report.ended_short}; handle archived)"
        else:
            incomplete += 0 if report.complete else 1
            suffix = "" if report.complete else f", {len(report.missing)} MISSING"
        print(f"{path.stem}: {report.written} written, {report.failed} failed{suffix}")
    print(f"\nTotal estimated cost: ${total_cost:.4f}")
    if ended_short:
        print(f"{ended_short} cancelled or expired batch(es): the requests that never ran are listed in "
              f"{settings.missing_dir()}; a synchronous create makes them")
    if incomplete:
        print(f"{incomplete} batch(es) lost results - see {settings.missing_dir()}")
        return 1
    return 0


def cmd_repin(args: argparse.Namespace) -> int:
    """Review/re-record a handle's schema pin locally after a deliberate schema change."""
    from universe_builder import store
    from universe_builder.providers.errors import ProviderError
    from universe_builder.providers.registry import schema_model_by_name
    from universe_builder.providers.types import SCHEMA_PIN_KEYS, legacy_schema_sha256, schema_fingerprint

    try:
        handle = batch_mod.load(args.path)
        model_name = handle.extra.get("schema_model")
        if not model_name:
            raise ValueError("handle has no schema_model to resolve")
        schema_model = schema_model_by_name(model_name)
        current = schema_fingerprint(schema_model)
    except (ProviderError, ValueError, OSError) as e:
        log.error("Cannot re-pin %s: %s; check UNIVERSE_BUILDER_DOMAIN", args.path, e)
        return 2
    previous = {key: handle.extra.get(key) for key in SCHEMA_PIN_KEYS}
    print(json.dumps({"recorded": previous, "current": current}, indent=2))
    # A pin in the pre-normalisation form of the same schema matches too
    # (`legacy_schema_sha256`); collect accepts it, so there is nothing to accept.
    legacy = dict(current, schema_sha256=legacy_schema_sha256(schema_model))
    if previous in (current, legacy):
        print("Schema pin already matches; nothing to write.")
    elif args.apply:
        handle.extra.setdefault("schema_pin_history", []).append(previous)
        handle.extra.update(current)
        store.write_json(args.path, handle.to_dict())
        print("Schema pin updated; previous pin retained in schema_pin_history.")
    else:
        print("Review the schema change before using --apply to accept the current schema.")
    return 0


def main(argv: list[str] | None = None, prog: str = "universe_builder") -> int:
    """`prog` is the CLI name hints tell the user to run next (`cli.main`'s
    counterpart): a wrapper installed under another name passes its own."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("models", help="show the pricing table").set_defaults(func=cmd_models)

    p = sub.add_parser("estimate", help="estimate cost before spending anything")
    p.add_argument("--atom", required=True, choices=sorted(_atom_types()))
    p.add_argument("--model", required=True,
                   help="a model id, or a tier name from pipeline.yaml "
                        "(frontier, strong, balanced, economy)")
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--output-tokens", type=int, default=MEASURED_OUTPUT_TOKENS)
    p.set_defaults(func=cmd_estimate)

    p = sub.add_parser("submit", help="submit a batch")
    p.add_argument("--atom", required=True, choices=sorted(_atom_types()))
    p.add_argument("--model", required=True,
                   help="a model id, or a tier name from pipeline.yaml "
                        "(frontier, strong, balanced, economy)")
    p.add_argument("--list", help="topic list file, one per line")
    p.add_argument("--from-history", action="store_true",
                   help="read topics from the generated index atoms (History atoms, for history)")
    p.add_argument("--context", help="stable context added to the cached prefix, "
                                     "e.g. a civilisation for history")
    p.add_argument("--limit", type=_limit, help="only submit the first N subjects (0: none)")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="output-token budget per request (default: pipeline.yaml's "
                        "defaults.max_tokens, the budget `run` uses)")
    p.add_argument("--dry-run", action="store_true")
    from universe_builder.providers import effort as effort_mod

    effort_mod.add_argument(p, help="reasoning effort (default: pipeline.yaml's for create; "
                                    "unset sends none)")
    from universe_builder.providers import task_budget as task_budget_mod

    task_budget_mod.add_argument(p, help="advisory output-token budget, Anthropic's task budget "
                                         "(default: pipeline.yaml's for create; at least 20,000 and "
                                         "no more than --max-tokens; unset sends none)")
    p.set_defaults(func=cmd_submit)

    sub.add_parser("list", help="list saved batch handles").set_defaults(func=cmd_list)

    p = sub.add_parser("poll", help="check batch status")
    p.add_argument("name", nargs="?")
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_poll)

    p = sub.add_parser("collect", help="fetch results and write JSON files")
    p.add_argument("name", nargs="?")
    p.add_argument("--all", action="store_true")
    p.set_defaults(func=cmd_collect)

    p = sub.add_parser("cancel", help="cancel running batches (lists them unless --apply)")
    p.add_argument("name", nargs="?")
    p.add_argument("--all", action="store_true")
    p.add_argument("--apply", action="store_true", help="cancel; without it, only list")
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser("repin", help="review a handle's current schema pin without fetching results")
    p.add_argument("path", type=Path)
    p.add_argument("--apply", action="store_true", help="accept the current schema; retain the old pin in the handle")
    p.set_defaults(func=cmd_repin)

    args = parser.parse_args(argv)
    args.prog = prog
    # Credentials come from the `.env`, and nothing on this path imports
    # `llm_classes.llm_call`, whose import side-effect is what loads it for the
    # synchronous pipeline. Without this, every command that builds a provider
    # fails with "Missing credentials" even though the key is on disk.
    settings.load_environment()
    # Install the config's retry/timeout policy for anything on this path that
    # makes a synchronous call. Batch submit/poll/collect keep the SDK's own
    # retries — they are cheap requests outside `run.complete`.
    provider_run.configure(pipeline.load().retry)
    # Only `collect` writes journal entries (per-result, via _journal_result);
    # `submit` records nothing there — the batch handle in `_batches/` is its
    # record — so opening a run for it just left an empty run dir behind.
    if args.command == "collect":
        journal.start_run(config={"command": args.command,
                                  "atom": getattr(args, "atom", None),
                                  "model": getattr(args, "model", None)})
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(name)s - %(levelname)s - %(message)s",
    )
    with topics.warnings_once():   # see `cli.main`
        return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
