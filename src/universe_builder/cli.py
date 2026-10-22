"""`universe_builder` — the command line entry point.

Before this, running anything meant `python -m universe_builder.execute_prompt`, which ran
*everything*: all four passes over every atom type, with model choices baked into
the source. There was no way to run one phase, no way to try a different model
without editing code, and no way to see what a run would do before paying for it.

`--dry-run` is the reason this exists. It resolves the same configuration the
real run would, prints which model each phase would use *and where that choice
came from*, and lists the work it would attempt — without making a single call.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Mapping, Optional, Sequence

from universe_builder import journal, pipeline, settings, store
from universe_builder.providers import effort
from universe_builder.providers import task_budget as task_budget_mod
from universe_builder.providers import run as provider_run

from universe_builder.registry import OperationSpec, active_pack

log = logging.getLogger(__name__)


PHASES = ("index", "create", "fix", "verify", "expand", "append")
#: The core verbs. A domain operation may not reuse one of these names; each
#: dispatches to the `cmd_<name>` function in this module.
COMMANDS = ("run", "config", "status", "index", "duplicates", "retire", "restore",
            "lint", "sources", "costs", "collisions", "experiment")
#: The seed phase was once called `history` — history's index
#: atom *is* a History — and every domain's seed phase went by that name.
#: Still accepted on the command line, normalised to `index` in `main`. The
#: table is `pipeline`'s; this name stays for callers that read it here.
LEGACY_PHASES = pipeline.LEGACY_PHASES


def costs_mod_groupings() -> List[str]:
    """`costs.GROUPINGS`, imported lazily so `--help` does not pay for it."""
    from universe_builder.costs import GROUPINGS

    return list(GROUPINGS)


def _atom_types() -> List[str]:
    from universe_builder.atoms import atom_mapping

    return sorted(k for k in atom_mapping if not k.startswith(("append_", "expand_")))


def _limit(value: str) -> int:
    """`--limit`: a count of 0 or more (0 means nothing)."""
    try:
        n = int(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"not a whole number: {value!r}") from e
    if n < 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more, not {n}")
    return n


def _pack_or_none():
    """The active pack, or None when it cannot be built — so `--help` and the
    error for an unknown `UNIVERSE_BUILDER_DOMAIN` still work. `main` reports the real
    problem before any command runs."""
    try:
        return active_pack()
    except KeyError:
        return None


def _check_atom(args) -> int:
    """0 if `--atom` names a type of the active domain (or was not given);
    otherwise say which types exist and return 2. An unknown type used to
    read as "0 atoms" (status) or a raw KeyError (run), and a command that
    reports nothing because it looked at nothing is the one answer it must
    never give."""
    atom = getattr(args, "atom", None)
    if atom and atom not in _atom_types():
        log.error("Unknown atom type %r. Known: %s", atom, ", ".join(_atom_types()))
        return 2
    return 0


def _prog(args) -> str:
    """The command name this invocation was run as, for "now run this" hints.

    `main` stamps it on `args` from its `prog=`; a wrapper (a pack that
    installs this engine under its own command name) passes its own, and a hint naming `universe_builder` under
    a wrapper points at a different command — one that loads neither the
    wrapper's `.env` nor its pack. Handlers called directly (tests) get the
    engine's name.
    """
    return getattr(args, "prog", None) or "universe_builder"


def _paid_setup(args, *, journal_config: dict, models_key: Optional[str] = None,
                model: Optional[str] = None, phase: str = "index", search: bool = False,
                announce=None) -> str:
    """The setup every paid core verb makes before its first call. Returns the model id.

    One place, in one order: load the config (`--config`),
    resolve the model — a tier name or a model id, from `model` or else
    `phase`'s configured choice, against `--provider`'s table; with `search`,
    against the `openai-search` table instead — install the retry policy, let
    the caller say what it is about to spend (`announce(model)`), and open a
    journal run so `costs` sees the spend. `journal_config` is the run's
    config with its `"model"` filled in here. Each verb used to repeat these
    steps, and two of them once forgot the journal.

    Raises `pipeline.UnknownTier` for a tier the provider does not define,
    before anything is configured or opened. The run is left open: the
    process ends with the verb.
    """
    config = pipeline.load(getattr(args, "config", None))
    if search:
        value = config.resolve_tier(model, provider="openai-search")
    else:
        value = config.resolve_model(phase, None, model,
                                     provider=getattr(args, "provider", None)).value
    provider_run.configure(config.retry)
    if announce is not None:
        announce(value)
    run_config = dict(journal_config)
    run_config["model"] = value
    journal.start_run(config=run_config, models={models_key: value} if models_key else None)
    return value


def _add_arm_b_flags(parser, prefix: str) -> None:
    """B's model and effort for a prompt experiment (v0.6.0): the model is then
    a variable too, and the variant may be `live` (the same prompt)."""
    parser.add_argument("--model-b", dest="model_b", metavar="MODEL_OR_TIER",
                        help=f"{prefix}B's model (default: A's); its provider is resolved from the id, "
                             "so A and B may be different vendors")
    effort.add_argument(parser, flag="--effort-b", dest="effort_b",
                        help=f"{prefix}B's reasoning effort (default: pipeline.yaml's for create)")
    parser.add_argument("--allow-dear-b", dest="allow_dear_b", action="store_true",
                        help=f"{prefix}let B's model exceed experiments.max_output_price "
                             "(A is always held to it)")


def build_parser(prog: str = "universe_builder",
                 operations: Optional[Mapping[str, OperationSpec]] = None) -> argparse.ArgumentParser:
    """`prog` is what usage and errors call the command: a wrapper installed
    under another name (a pack's own command) passes its own.

    `operations` are the active pack's verbs (`DomainPack.operation_specs()`),
    each given a subparser after the core ones. Core knows none of them by
    name: `periods` was once built here with its flags, and a
    second domain verb had nowhere to go."""
    parser = argparse.ArgumentParser(prog=prog, description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one phase of the pipeline")
    run.add_argument("--phase", required=True, choices=(*PHASES, *LEGACY_PHASES))
    run.add_argument("--atom", help="restrict to one atom type (default: all)")
    run.add_argument("--model", help="override the configured model or tier for this phase")
    run.add_argument("--provider", help="resolve tiers against this vendor's table "
                                        "(openai, anthropic, google, deepseek, qwen)")
    run.add_argument("--limit", type=_limit,
                     help="at most N items per atom type (0: none); the same on every "
                          "path, dry run and batch included")
    run.add_argument("--batch", action="store_true",
                     help="submit the phase as a batch (half price, up to 24h) instead of "
                          "running it synchronously; collect later with `run_batch collect`")
    run.add_argument("--dry-run", action="store_true",
                     help="resolve the config and list the work, without calling anything")
    run.add_argument("--config", help="path to a pipeline.yaml (default: the bundled one)")
    run.add_argument("--experiment", metavar="NAME",
                     help="prompt experiment (create --batch only): A, the real batch, goes "
                          "unchanged; a sample of the same subjects is also generated under "
                          "variant B into the experiments root, never the corpus")
    run.add_argument("--variant", help="with --experiment: the B prompt (default: the experiment's name)")
    run.add_argument("--sample-rate", dest="sample_rate", type=float, default=0.10,
                     help="with --experiment: share of each atom type's subjects B also runs (0.10)")
    run.add_argument("--min-per-type", dest="min_per_type", type=int, default=5,
                     help="with --experiment: at least this many per atom type (5)")
    effort.add_argument(run, help="reasoning effort for this phase (low, medium, high; xhigh, max "
                                  "where the provider has them); overrides pipeline.yaml's effort "
                                  "block; unset sends none (the provider's default)")
    task_budget_mod.add_argument(
        run, help="advisory output-token budget per call (Anthropic's task budget, Claude models "
                  "marked task_budget in model_pricing.json; others ignore it): at least 20,000 and "
                  "no more than the phase's max_tokens, ideally one half to two thirds of it; "
                  "overrides pipeline.yaml's task_budget block; unset sends none")
    _add_arm_b_flags(run, "with --experiment: ")

    sub.add_parser("config", help="print the resolved pipeline configuration")

    experiment = sub.add_parser(
        "experiment", help="prompt experiments: compare a prompt variant (B) with the live "
                           "prompt (A) on the same subjects, outside the corpus")
    verbs = experiment.add_subparsers(dest="experiment_command", required=True)
    replay = verbs.add_parser("replay", help="generate A and B for a sample of subjects that already "
                                             "have atoms (estimate unless --apply)")
    replay.add_argument("name")
    replay.add_argument("--variant", help="the B prompt (default: the experiment's name)")
    replay.add_argument("--atom", help="restrict to one atom type")
    replay.add_argument("--context", help="restrict to one index list (e.g. a civilisation)")
    replay.add_argument("--per-type", dest="per_type", type=int, default=15,
                        help="subjects per atom type (15)")
    replay.add_argument("--only", metavar="IDS_FILE",
                        help="sample only from these atoms, one kind:hID per line (# comments allowed)")
    replay.add_argument("--config", help="path to a pipeline.yaml")
    replay.add_argument("--apply", action="store_true", help="submit; without it, only the estimate")
    effort.add_argument(replay, flag="--effort-a", dest="effort_a",
                        help="A's reasoning effort (default: pipeline.yaml's for create; unset sends none)")
    _add_arm_b_flags(replay, "")
    replay.add_argument("--reuse-a", dest="reuse_a", metavar="FROM_EXPERIMENT",
                        help="take A, and the sample, from FROM_EXPERIMENT's collected A arm instead of "
                             "generating A again, so several B arms compare against one identical A")
    collect = verbs.add_parser("collect", help="fetch finished experiment batches (and, for a shadow "
                                               "experiment, A's replies from the corpus journal)")
    collect.add_argument("name")
    judge = verbs.add_parser("judge", help="judge complete pairs blind, both orders (paid; estimate "
                                           "unless --apply)")
    judge.add_argument("name")
    judge.add_argument("--model", default="balanced",
                       help="judge model or tier (default: balanced)")
    judge.add_argument("--sample", type=int, help="a seeded subset of N pairs (the calibration run)")
    judge.add_argument("--limit", type=int, help="at most N pairs")
    judge.add_argument("--concurrency", type=int, default=4)
    judge.add_argument("--config", help="path to a pipeline.yaml")
    judge.add_argument("--apply", action="store_true", help="send; without it, only the estimate")
    effort.add_argument(judge, help="the judge's reasoning effort; each (model, effort) is its own "
                                    "judge, stored beside the others (unset sends none)")
    report = verbs.add_parser("report", help="pairs, guardrails, judge tally and the verdict (free)")
    report.add_argument("name")
    report.add_argument("--judge-model", dest="judge_model",
                        help="whose judgements decide: MODEL or MODEL@EFFORT, a tier allowed "
                             "(default: the judge with the most)")
    report.add_argument("--config", help="path to a pipeline.yaml (for resolving a judge's tier)")
    report.add_argument("--panel", metavar="J1,J2",
                        help="two or more judges (MODEL or MODEL@EFFORT): each one's tally, their "
                             "agreement rate, and a verdict on the pairs they agree on")
    verbs.add_parser("list", help="experiments and variants")

    status = sub.add_parser("status", help="what is on disk, and what still needs fixing")
    status.add_argument("--atom", help="restrict to one atom type")
    status.add_argument("--coverage", action="store_true",
                        help="topics the index atoms list, against atoms on disk")
    status.add_argument("--gaps", action="store_true",
                        help="with --coverage, name the missing topics")

    collisions = sub.add_parser(
        "collisions",
        help="names an incoming index seed shares with the stored corpus, "
             "before `--phase create` turns them into a second atom. Detection "
             "is mechanical and noisy on purpose; `--via model` adjudicates")
    collisions.add_argument("--seed", metavar="NAME", dest="seed",
                            help="the index atom whose seed to check (history: e.g. 'Roman History')")
    # The flag was once `--history`; still accepted, no longer shown.
    collisions.add_argument("--history", metavar="NAME", dest="seed", help=argparse.SUPPRESS)
    collisions.add_argument("--atom", help="restrict to one atom type")
    collisions.add_argument("--review", metavar="FILE",
                            help="write the candidates out for adjudication; "
                                 f"act on them with `{prog} lint --apply-review FILE`")
    collisions.add_argument("--via", choices=("model",), metavar="SOURCE",
                            help="adjudicate with a model instead of leaving "
                                 "every row undecided. Paid: one call per "
                                 "candidate, on the frontier tier by default")
    collisions.add_argument("--model", help="model or tier for --via model "
                                            "(default: the index phase's)")
    collisions.add_argument("--config", help="path to a pipeline.yaml")
    collisions.add_argument("--provider", help="with --via model, resolve the tier against this "
                                               "vendor's table")

    costs = sub.add_parser(
        "costs",
        help="what each run, atom and model cost, read back out of the journal. "
             "Stores nothing and is always recomputable; see `universe_builder.costs`")
    costs.add_argument("--by", default="atom", choices=costs_mod_groupings(),
                       help="how to group the rows (default: atom)")
    costs.add_argument("--atom", help="restrict to one atom type")
    costs.add_argument("--run", metavar="RUN_ID", help="restrict to one journal run")
    costs.add_argument("--since", metavar="ISO",
                       help="only attempts at or after this timestamp, e.g. 2026-09-24")
    costs.add_argument("--limit", type=int, help="show only the N costliest rows")
    costs.add_argument("--json", action="store_true", dest="as_json",
                       help="machine-readable output instead of the table")

    sub.add_parser(
        "index",
        help="rebuild the derived cross-reference index (_index/links.json): "
             "which index atoms list each atom, and which atoms a name field "
             "points at — always recomputed, never stored in atoms")

    sources = sub.add_parser(
        "sources",
        help="append verified, resolvable source URLs to atoms (additive: "
             "nothing is removed). Every URL is fetched before it is written")
    sources.add_argument("--atom", help="restrict to one atom type")
    sources.add_argument("--limit", type=int, help="stop after this many atoms")
    sources.add_argument("--only", metavar="IDS_FILE",
                         help="only the atoms this file names, one kind:hID per line "
                              "(# comments allowed)")
    sources.add_argument("--missing", action="store_true",
                         help="only atoms that cite no URL at all. A free "
                              "resolver leaves a residue of names it cannot "
                              "match; `--missing --via search` is what the paid "
                              "model is for")
    # The free resolvers are the domain's (`DomainPack.url_resolvers`; history
    # ships `wikipedia`); `search` — a paid web-search model — is core. The
    # first free one is the default. **A pack with none has no default**: the
    # first cut defaulted to `search`, and a reviewer's bare `universe_builder sources`
    # under a test pack made three paid calls it never asked for. Validated in
    # `cmd_sources`, not by argparse, so `--help` needs no buildable pack.
    pack = _pack_or_none()
    free_resolvers = tuple(pack.url_resolvers) if pack else ()
    sources.add_argument("--via", metavar="RESOLVER",
                         default=free_resolvers[0] if free_resolvers else None,
                         help="a free lookup the domain provides"
                              + (f" ({', '.join(free_resolvers)}; no model; the default)"
                                 if free_resolvers else " (this domain provides none)")
                              + ", or search: a web-search model, ~$10/1k calls, "
                                "better judgement on which page is wanted"
                              + ("" if free_resolvers else ". Required: nothing is paid for by default"))
    sources.add_argument("--per-atom", type=int, default=1, dest="per_atom",
                         help="how many URLs to take per atom (default 1)")
    sources.add_argument("--model", default="economy",
                         help="with --via search: a model supporting web search, or a tier "
                              "name from pipeline.yaml's openai-search table (default economy)")
    sources.add_argument("--config", help="with --via search, path to a pipeline.yaml")
    sources.add_argument("--apply", action="store_true",
                         help="actually write; without this nothing is changed")

    lint = sub.add_parser(
        "lint",
        help="mechanical corpus checks — no model, no network, no writes: "
             "aliases that are over-broad, contested or contradicted, atoms "
             "that disown their topic, and dates whose era the atom's own "
             "prose contradicts")
    lint.add_argument("--check", dest="checks", action="append",
                      help="run only this check (repeatable)")
    lint.add_argument("--atom", help="restrict to one atom type")
    lint.add_argument("--review", metavar="FILE",
                      help="write the findings to FILE for adjudication instead "
                           "of printing them")
    # Identity sources are the domain's (`DomainPack.identity_resolvers`).
    # History's `wikipedia` decides by redirect equality: free, and it abstains
    # rather than guesses — every answer it committed to was correct when
    # measured, and it stayed silent on the rest.
    identity_sources = tuple(pack.identity_resolvers) if pack else ()
    lint.add_argument("--via", metavar="SOURCE",
                      help="with --review, pre-fill verdicts from a free identity "
                           "source the domain provides"
                           + (f": {', '.join(identity_sources)}" if identity_sources
                              else " (this domain provides none)"))
    lint.add_argument("--apply-review", dest="apply_review", metavar="FILE",
                      help="act on an adjudicated review file")
    lint.add_argument("--apply", action="store_true",
                      help="with --apply-review, actually write; without it the "
                           "plan is only printed")

    retire = sub.add_parser(
        "retire",
        help="retire the atoms an older model wrote so a better one regenerates "
             "them: moves to _retired/ (never deletes), leaving each topic with "
             "no atom for the next create run to fill")
    retire.add_argument("--model", required=True, metavar="TEXT",
                        help="match the filename's model segment, e.g. gpt-4o-mini")
    retire.add_argument("--atom", help="restrict to one atom type (default: all). "
                                       "Use it to exclude an atom a later pass has "
                                       "enriched -- retiring an expanded atom discards "
                                       "its expansion, which must then be paid for again")
    retire.add_argument("--apply", action="store_true",
                        help="actually retire; without this the plan is only printed")

    restore = sub.add_parser(
        "restore",
        help="write back the bytes one journal attempt's in-place rewrite replaced "
             "(verify, batch collect, expand's splice, append's merge). The restore "
             "is journaled and keeps what it replaces, so it can be undone the same way")
    restore.add_argument("--attempt", required=True, metavar="ID",
                         help="the attempt_id of the journal line whose write to undo")
    restore.add_argument("--run", metavar="RUN_ID",
                         help="look only in this run (default: every run, newest first)")
    restore_mode = restore.add_mutually_exclusive_group()
    restore_mode.add_argument("--dry-run", dest="apply", action="store_false",
                              help="show what would be written (the default)")
    restore_mode.add_argument("--apply", dest="apply", action="store_true",
                              help="actually write the saved bytes back")
    # Explicitly: `--dry-run`'s store_false would otherwise default `apply` to True.
    restore.set_defaults(apply=False)

    duplicates = sub.add_parser(
        "duplicates",
        help="atoms that declare a shared name — one subject, two files. "
             "--apply keeps the later model's atom, retires the other to "
             "_duplicates/, and records an alias line so the retired spelling "
             "is not regenerated")
    duplicates.add_argument("--via", choices=("model",), metavar="SOURCE",
                            help="triage the UNDECIDED pile with a model before "
                                 "anything is ranked: it holds components that are "
                                 "not duplicates at all (Abbas I, Abbas II and "
                                 "Abbas Mirza were grouped as one person). Pairs "
                                 "judged different become `!=` lines in "
                                 "not_duplicates.txt. Paid: one call per pair")
    duplicates.add_argument("--model", "--triage-model", dest="triage_model", metavar="MODEL",
                            help="model or tier for --via model (default: the "
                                 "index phase's, i.e. frontier)")
    duplicates.add_argument("--config", help="with --via model, path to a pipeline.yaml")
    duplicates.add_argument("--provider", help="with --via model, resolve the tier against "
                                               "this vendor's table")
    duplicates.add_argument("--limit", type=int,
                            help="with --via model, triage only the first N components")
    duplicates.add_argument("--out", metavar="FILE",
                            help="with --via model, save every verdict as JSON. A "
                                 "triage of the full pile costs real money; without "
                                 "this its only record is the terminal")
    duplicates.add_argument("--apply", action="store_true",
                            help="actually retire; without this the plan is only printed")
    duplicates.add_argument("--keep", dest="kept", action="append", metavar="TEXT",
                            help="name the survivor for a pair the model-family rule "
                                 "cannot rank, by a substring of its filename. "
                                 "Repeatable")
    duplicates.add_argument("--except", dest="excluded", action="append", metavar="TEXT",
                            help="drop any pair whose filenames contain TEXT — a shared "
                                 "name is evidence of a duplicate, not proof of one. "
                                 "Repeatable")
    duplicates.add_argument("--tiebreak-fill", dest="tiebreak_fill", action="store_true",
                            help="where two atoms share the highest model family, keep the "
                                 "one that fills more fields — but ONLY for a component an "
                                 "asserted, unscoped alias line covers in full (a person's "
                                 "verdict, e.g. via `lint --apply-review`). Everything else "
                                 "stays undecided: fill ranks atoms, it never proves identity")

    for name, spec in (operations or {}).items():
        if name in COMMANDS:
            raise ValueError(f"domain operation {name!r} collides with a core verb")
        op_parser = sub.add_parser(name, help=spec.help or f"the active domain's {name!r} operation")
        if spec.add_arguments:
            spec.add_arguments(op_parser)

    return parser


def cmd_config(args) -> int:
    print(f"data directory: {settings.data_dir()}\n")
    print(pipeline.load(getattr(args, "config", None)).describe())
    return 0


def _coverage(atoms: Sequence[str]):
    """Per civilisation, per atom type: listed, generated, failed, missing.

    Answers the question `status` could not: a count of atoms on disk says
    nothing about whether a civilisation is *finished*. Three states, not two —
    a topic that was attempted and failed is not the same as one never tried,
    and collapsing them is how work goes missing.

    An `EmptyModel` placeholder counts as **not** generated. It is a file, but
    treating it as coverage would report a civilisation complete when part of it
    is `{"error": "No data found"}`.
    """
    from universe_builder import topics

    report: dict = {}
    for atom in atoms:
        if atom == active_pack().index.name:
            continue
        for topic in topics.from_history(atom):
            context = topic.context or "(no context)"
            row = report.setdefault(context, {}).setdefault(
                atom, {"listed": 0, "have": 0, "failed": 0, "missing": []})
            row["listed"] += 1

            # One judgement, shared with `group_synonyms`' `prefer`. These were
            # separate and disagreed: coverage filtered empty placeholders and
            # the work list did not.
            if topics.has_real_atom(atom, topic.subject):
                row["have"] += 1
            elif store.sentinels_for(atom, topic.subject):
                row["failed"] += 1
                row["missing"].append(topic.subject)
            else:
                row["missing"].append(topic.subject)
    return report


def cmd_coverage(args, atoms: Sequence[str]) -> int:
    report = _coverage(atoms)
    if not report:
        print(f"No {active_pack().index.name} atoms on disk, so nothing lists any work.")
        print(f"Generate them first:  {_prog(args)} run --phase index")
        return 1

    print(f"data directory: {settings.data_dir()}\n")
    incomplete = 0
    for context in sorted(report):
        print(context)
        t_listed = t_have = t_failed = 0
        for atom in sorted(report[context]):
            row = report[context][atom]
            t_listed += row["listed"]
            t_have += row["have"]
            t_failed += row["failed"]
            note = "complete" if row["have"] == row["listed"] else ""
            if row["failed"]:
                note = f"{row['failed']} awaiting fix"
            gap = row["listed"] - row["have"] - row["failed"]
            if gap:
                note = (note + ", " if note else "") + f"{gap} never generated"
            print(f"  {atom:<12} {row['have']:>5} / {row['listed']:<5} {note}")
            if args.gaps and row["missing"]:
                from universe_builder import topics as topics_mod

                # Everything listed for this atom type, so a gap can be shown
                # against what you already have. Exact synonyms never reach here
                # — `from_history` groups those away — so anything flagged is a
                # *guess* that two entries are the same thing, offered for a
                # human to judge rather than acted on.
                listed = [t.subject for t in topics_mod.from_history(atom)]
                for subject in row["missing"]:
                    near = topics_mod.near_matches(subject, listed)
                    suffix = f"   ~ compare {', '.join(repr(n) for n in near[:2])}" if near else ""
                    print(f"        - {subject}{suffix}")
        print(f"  {'total':<12} {t_have:>5} / {t_listed:<5}")
        if t_have != t_listed:
            incomplete += 1
        print()

    # Non-zero when something is outstanding, so a script can gate on it.
    return 1 if incomplete else 0


def cmd_status(args) -> int:
    if _check_atom(args):
        return 2
    atoms = [args.atom] if args.atom else _atom_types()
    if getattr(args, "coverage", False):
        return cmd_coverage(args, atoms)
    print(f"data directory: {settings.data_dir()}")
    total_atoms = total_errors = total_unfixable = 0
    for atom in atoms:
        folder = store.dir_for(atom)
        if not folder.exists():
            continue
        files = [p for p in folder.iterdir() if p.is_file()]
        # Count only base atoms: an unmerged `append_<atom>_*.json` (or a stray
        # `expand_*`) shares the folder but is not itself an atom — counting it
        # would inflate the tally between an append pass and its merge.
        atoms_n = len([p for p in files if p.suffix == ".json"
                       and not p.name.startswith(("append_", "expand_"))])
        error_files = [p for p in files if store.is_error_path(p)]
        # Only count sentinels the fix pass will actually visit. `fix` filters
        # `p.name.startswith(config.atom_name)` (the base type), so an
        # `append_city_…_ERROR`/`expand_city_…_ERROR` sitting in this folder is
        # never repaired by `universe_builder run --phase fix` — reporting it as "awaiting fix"
        # is a promise the fix pass cannot keep. Count those separately.
        unfixable = [p for p in error_files
                     if p.name.startswith(("append_", "expand_"))]
        errors_n = len(error_files) - len(unfixable)
        total_atoms += atoms_n
        total_errors += errors_n
        total_unfixable += len(unfixable)
        print(f"  {atom:<12} {atoms_n:>5} atoms  {errors_n:>4} awaiting fix")
    print(f"  {'total':<12} {total_atoms:>5} atoms  {total_errors:>4} awaiting fix")
    if total_unfixable:
        print(f"  note: {total_unfixable} append/expand _ERROR sentinel(s) — `--phase "
              f"fix` does not clear these; regenerate the record instead.")

    # The journal's queue is derived by replaying every run's entries, so it can
    # disagree with the `_ERROR` count - and the disagreement is the interesting
    # part. More journal entries than sentinels means a sentinel was lost to a
    # write that should not have touched it, which is the failure mode B8 keeps
    # recurring as. Both are shown rather than one being treated as the truth.
    runs = journal.list_runs()
    if runs:
        queued = journal.needs_fix_across_runs()
        # The journal records every failure regardless of atom type, so compare it
        # against *all* sentinels on disk — the fixable ones plus the append/expand
        # ones broken out above — not just the "awaiting fix" subset. Comparing
        # against the reduced count would flag a phantom "lost sentinel" whenever an
        # append/expand failure was recorded.
        total_sentinels = total_errors + total_unfixable
        print(f"\njournal: {len(runs)} run(s) in {journal.runs_dir()}")
        print(f"  {len(queued):>5} work item(s) still needing a fix")
        # The queue is global, so the disk-vs-journal comparison is only meaningful
        # when the disk count is global too. Under `--atom` the disk count is one
        # folder's; a pending failure in any other type would make the two disagree
        # spuriously, so the note is suppressed.
        if not args.atom and len(queued) != total_sentinels:
            # Seen live with the causes reversed from what this note
            # used to claim: the surplus was a *journal* item kept by a repair
            # recorded under a mismatched id, not a lost sentinel. Name both
            # directions rather than guessing for the reader.
            print(f"  note: {total_sentinels} _ERROR sentinel(s) on disk — the two "
                  f"disagree. More sentinels than journal items means failures "
                  f"predating the journal (or an overwritten record); more "
                  f"journal items means a sentinel was lost, or a repair was "
                  f"recorded under a different work item than its failure.")
    return 0


def cmd_operation(name: str, args) -> int:
    """Run the active domain's operation `name`, if it declares one.

    Domain verbs (history's `periods`, the period lint/adjudicate/split
    machinery; its `media` and `signals` sidecars) live in their pack as
    `OperationSpec`s rather than in core here. This is the thin dispatcher: a
    domain without the operation (medications, say) has no behaviour to run,
    and says so. Normally unreachable through `main`, which only builds a
    subparser for a declared operation; reachable for a caller with a
    namespace in hand.
    """
    pack = active_pack()
    spec = pack.operation_specs().get(name)
    if spec is None:
        log.error("the %s domain has no %r operation", pack.name, name)
        return 2
    return spec.handler(args)


def cmd_periods(args) -> int:
    """`cmd_operation("periods", args)`, kept for callers that named it."""
    return cmd_operation("periods", args)


def cmd_index(args) -> int:
    """Rebuild the cross-reference index. Pure read of the corpus, one write."""
    from universe_builder import index

    report = index.build()
    path = index.write(report)

    listed = sum(len(subjects) for per in report["memberships"].values()
                 for subjects in per.values())
    resolved = sum(1 for per in report["memberships"].values()
                   for subjects in per.values() for v in subjects.values() if v)
    print(f"memberships  {resolved}/{listed} listed subjects resolve to an atom")
    print(f"references   {len(report['references'])} name links resolved")
    print(f"unresolved   {len(report['unresolved'])} names nothing on disk answers to")
    pairs = sum(len(found) for found in report["duplicates"].values())
    if pairs:
        print(f"duplicates   {pairs} name(s) declared by two stored atoms "
              f"— see `{_prog(args)} duplicates`")
    multi = sum(1 for w in report["provenance"].values() if len(w) > 1)
    print(f"provenance   {len(report['provenance'])} atoms tracked "
          f"({multi} written more than once)")
    print(f"\nwritten to {path}")
    return 0


def cmd_sources(args) -> int:
    """Propose and optionally append verified source URLs."""
    from universe_builder import sources as sources_mod

    if _check_atom(args):
        return 2
    atom_types = ([args.atom] if args.atom
                  else [spec.name for spec in active_pack().atoms])
    free = active_pack().url_resolvers
    if args.via is None:
        log.error("--via is required: the %s domain provides no free resolver, and "
                  "`--via search` is paid, so nothing runs by default", active_pack().name)
        return 2
    if args.via != "search" and args.via not in free:
        log.error("Unknown resolver %r. This domain provides: %s; `search` is always available",
                  args.via, ", ".join(free) or "none")
        return 2
    if args.via == "search":
        # A tier name resolves against the search provider's own table
        # (`economy` is gpt-4.1-mini there, not the generation tier's nano,
        # which cannot search), and the spend is journalled into a run of its
        # own, so `costs` sees it.
        try:
            model = _paid_setup(args, search=True, model=args.model,
                                journal_config={"command": "sources", "via": "search", "model": None,
                                                "atom": args.atom, "limit": args.limit})
        except pipeline.UnknownTier as e:
            log.error("%s", e)
            return 2
        try:
            return _run_sources(args, atom_types, sources_mod.searcher(model))
        finally:
            journal.end_run()
    return _run_sources(args, atom_types, free[args.via](per_atom=args.per_atom))


def _run_sources(args, atom_types, ask) -> int:
    """`sources.run`, printed as it goes."""
    from universe_builder import sources as sources_mod

    # Both files are read before anything is asked or fetched: a malformed
    # line is an error to fix, never a reject silently forgotten.
    try:
        rejected = sources_mod.load_rejected(active_pack().rejected_sources_path)
        only = sources_mod.load_only(args.only) if getattr(args, "only", None) else None
    except (OSError, ValueError) as e:
        log.error("%s", e)
        return 2

    def prefetched(matched, asked):
        print(f"prefetched {matched}/{asked} titles via {args.via}\n")

    def show(proposal):
        mark = "+" if proposal.verified else " "
        print(f" {mark} {proposal.subject[:40]:40} "
              f"{len(proposal.verified)} verified, "
              f"{len(proposal.rejected)} rejected")
        for url in proposal.verified:
            print(f"       {url[:96]}")
        for url, why in proposal.rejected:
            print(f"       -- {why}: {url[:80]}")
        for url in proposal.rejected_before:
            print(f"       -- rejected before: {url[:80]}")

    report = sources_mod.run(atom_types, ask, limit=args.limit, missing=args.missing,
                             write=args.apply, on_prefetch=prefetched, on_proposal=show,
                             rejected=rejected, only=only)
    if report.only_unmatched:
        print(f"\n{len(report.only_unmatched)} --only id(s) matched no stored atom: "
              + ", ".join(report.only_unmatched[:10])
              + (" …" if len(report.only_unmatched) > 10 else ""))
    if active_pack().rejected_sources_path is not None or report.rejected_before:
        print(f"\n{report.rejected_before} previously rejected URL(s) skipped unfetched "
              f"({sum(len(v) for v in rejected.values())} reject(s) on file).")
    print(f"\n{report.examined} atom(s) examined; "
          + (f"{report.updated} updated with {report.urls} URL(s)." if args.apply
             else "nothing written — re-run with --apply."))
    return 0 if args.apply else 1


def cmd_lint(args) -> int:
    """Report mechanical defects. Reads only; exits non-zero when any are found."""
    from universe_builder import lint as lint_mod

    if args.apply_review:
        try:
            rows = lint_mod.read_review(args.apply_review)
        except ValueError as e:
            print(e)
            return 2
        try:
            outcome = lint_mod.apply_review(rows, dry_run=not args.apply)
        except ValueError as e:
            print(e)
            return 2
        undecided = sum(1 for r in rows if r[0] in ("?", "skip"))
        print(f"{len(rows)} adjudicated row(s): {outcome['same']} same, "
              f"{outcome['wrong']} wrong, {undecided} left alone")
        for line in outcome["alias_lines"]:
            print(f"  alias: {line}")
        # The removals are the destructive half, so they are shown before the
        # write, not summarised as a count after it.
        for line in outcome.get("would_edit", []):
            print(f"  remove: {line}")
        if outcome["dry_run"]:
            print("\nNothing written. Re-run with --apply.")
            return 1
        print(f"  edited {len(outcome['edited'])} atom(s); "
              f"wrote {len(outcome['alias_lines'])} alias line(s)")
        return 0

    if args.via and args.via not in active_pack().identity_resolvers:
        log.error("Unknown identity source %r. This domain provides: %s", args.via,
                  ", ".join(active_pack().identity_resolvers) or "none")
        return 2
    try:
        report = lint_mod.run(checks=args.checks,
                              atom_types=[args.atom] if args.atom else None)
    except KeyError as e:
        print(str(e).strip('"'))
        return 2
    if report.is_clean():
        print("No findings.")
        return 0

    if args.review:
        decide = None
        if args.via:
            decide = lint_mod.identity_decider(
                report, active_pack().identity_resolvers[args.via], label=args.via)

        written = lint_mod.write_review(report, args.review, decide=decide)
        print(f"Wrote {written} finding(s) to {args.review}.")
        print(f"Set the verdict column, then: {_prog(args)} lint --apply-review "
              f"{args.review} --apply")
        return 1

    if report.errors:
        # Printed first and gated on below: a crashed check has established
        # nothing, and a clean-looking report is the one thing a linter must
        # never emit when it did not actually run.
        print(f"{len(report.errors)} check(s) FAILED TO RUN — this report is "
              f"incomplete:")
        for message in report.errors:
            print(f"  {message}")

    for check, findings in sorted(report.by_check().items()):
        print(f"\n{check}  ({len(findings)})")
        for f in findings[:40]:
            print(f"  {f.atom_type:9} {f.subject[:34]:34} {f.detail}")
        if len(findings) > 40:
            print(f"  ... and {len(findings) - 40} more")
    if report.findings:
        print(f"\n{len(report.findings)} finding(s). Nothing was changed — these "
              f"are judgement calls, not repairs.")
    return 1


def cmd_collisions(args) -> int:
    """Report, and optionally adjudicate, incoming-seed name collisions.

    Writes nothing to the corpus. `--review` produces a file for
    `universe_builder lint --apply-review`, which is where the alias lines are actually
    written and where the dry run lives.
    """
    from universe_builder import collisions as collisions_mod

    if not args.seed:
        log.error("--seed NAME is required: which index atom's seed to check")
        return 2
    seed, seed_file = collisions_mod.load_seed(args.seed)
    if seed is None:
        log.error("No %s atom for %r. Run `--phase index` first: the seed is "
                  "what names the collisions.", active_pack().index.name, args.seed)
        return 2

    atoms = [args.atom] if args.atom else list(collisions_mod.default_atoms())
    repins = collisions_mod.shared(seed, atoms, seed_file=seed_file)
    if repins:
        print(f"{len(repins)} topic(s) re-pin onto stored atoms. Usually one subject; "
              "where a name means something else here, write a scoped alias line "
              "before create:")
        for atom_type, topic, filename, listed_by in repins:
            also = ", ".join(listed_by) if listed_by else "no other History"
            print(f"    {atom_type:9} {topic}  ->  {filename}  (also in: {also})")
        print()
    candidates = collisions_mod.find(seed, atoms)
    if not candidates:
        print(f"No collision candidates for {args.seed!r}.")
        return 0

    by_source = {}
    for candidate in candidates:
        by_source[candidate.source] = by_source.get(candidate.source, 0) + 1
    print(f"{len(candidates)} candidate(s) for {args.seed!r}: "
          + ", ".join(f"{n} by {source}" for source, n in sorted(by_source.items())))

    verdicts = {}
    if args.via == "model":
        # Paid, so journalled: `adjudicate` records each call into the current
        # run, and without one open it recorded into a `NullJournal` — the
        # spend was invisible to `costs`.
        model = _paid_setup(
            args, model=args.model, models_key="collisions",
            journal_config={"command": "collisions", "seed": args.seed, "atom": args.atom},
            announce=lambda m: log.warning("Adjudicating %d candidate(s) on %s", len(candidates), m))
        verdicts = collisions_mod.adjudicate(candidates, model)
        counts = collisions_mod.summarise(verdicts)
        print("  verdicts: " + ", ".join(f"{v} {n}" for v, n in sorted(counts.items())))

    if args.review:
        written = collisions_mod.write_review(candidates, args.review, verdicts)
        print(f"\n{written} row(s) written to {args.review}")
        print("Review the verdicts, then: "
              f"{_prog(args)} lint --apply-review {args.review} --apply")
    else:
        for candidate in candidates:
            verdict, why = verdicts.get(candidate, ("?", ""))
            print(f"  [{verdict:>5}] {candidate.atom_type:9} "
                  f"{candidate.incoming!r} ~ {candidate.stored!r}"
                  + (f"  ({why})" if why else ""))
        print("\nPass --review FILE to adjudicate and apply these.")
    return 0


def cmd_costs(args) -> int:
    """Report journal-derived spend.

    Reads only. There is no `--apply` because there is nothing to apply: the
    journal is the record, and this recomputes a view of it every time.
    """
    import json as _json

    from universe_builder import costs as costs_mod

    report = costs_mod.build(args.by, atom=args.atom, run=args.run, since=args.since)
    if not report.total.calls:
        where = f" matching {args.atom!r}" if args.atom else ""
        print(f"No attempts{where} in {settings.data_dir() / '_runs'}.")
        if report.mechanical:
            print(costs_mod.mechanical_line(report.mechanical))
        return 0

    if args.as_json:
        print(_json.dumps(report.as_dict(), indent=2))
    else:
        print(costs_mod.format_report(report, limit=args.limit))
    return 0


def cmd_retire(args) -> int:
    """Show, and optionally apply, the retirement of an old model's atoms.

    Dry by default, like every other destructive verb. Exits non-zero while a
    dry run still has work, so a script can gate on it.
    """
    from universe_builder import retire as retire_mod

    plan = retire_mod.plan(args.model, atoms=[args.atom] if args.atom else None)
    if plan.is_empty():
        print(f"No atoms match model {args.model!r}. On disk:")
        for name, count in sorted(retire_mod.models_on_disk().items()):
            print(f"  {count:5}  {name}")
        return 0

    print(f"data directory: {settings.data_dir()}\n")
    for atom, paths in sorted(plan.by_type.items()):
        print(f"  {atom:10} {len(paths):5}")

    if plan.refused:
        print(f"\nREFUSED — {len(plan.refused)} index atom(s) match and are kept:")
        for path in plan.refused:
            print(f"  {path.name}")
        print("  Retiring an index atom does not schedule a regeneration: it "
              "removes topics\n  from the work list, orphaning the atoms they "
              "list. Those atoms stay on disk,\n  listed by nothing, absent from "
              "coverage and from the index.")

    outcome = retire_mod.apply(plan, dry_run=not args.apply)
    if not args.apply:
        print(f"\n{plan.summary()}. Nothing moved.")
        print(f"Re-run with --apply, then `{_prog(args)} run --phase create --dry-run` to "
              "see the regeneration work.")
        return 1
    print(f"\nRetired {outcome['count']} atom(s) to {settings.retired_dir()}.")
    print(f"Each retired topic now has no atom; `{_prog(args)} run --phase create` will "
          "regenerate it.")
    return 0


def cmd_restore(args) -> int:
    """Show, and with `--apply` make, the restore of one attempt's replaced bytes.

    Dry by default, exiting 1 while there is something to restore, like
    `retire`. Exits 2 when the attempt is unknown or its evidence cannot be
    trusted (a hash mismatch, saved bytes gone, a target outside the data
    dir); 1 when the attempt replaced nothing; 0 once the target holds the
    saved bytes.
    """
    from universe_builder import restore as restore_mod

    try:
        plan = restore_mod.plan(args.attempt, run_id=args.run)
    except LookupError as e:
        print(e.args[0] if e.args else str(e), file=sys.stderr)
        return 2
    except (ValueError, OSError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2

    print(f"data directory: {settings.data_dir()}\n")
    for row in plan.describe():
        print(f"  {row}")
    if plan.replaced_nothing:
        print(f"\nAttempt {plan.attempt_id} replaced nothing: no bytes were saved for it "
              "(a new file, a failed result, or a line from before v0.3.2).")
        return 1
    if plan.already_restored:
        print(f"\n{plan.target.name} already holds those bytes; nothing to do.")
        return 0
    if not args.apply:
        print("\nDry run; nothing written. Re-run with --apply to write the saved bytes back "
              "(the current ones are kept in the journal).")
        return 1

    run = journal.start_run(config={"command": "restore", "attempt": plan.attempt_id,
                                    "from_run": plan.entry.get("run_id")})
    try:
        line = restore_mod.apply(plan)
    finally:
        journal.end_run()
    if line is None:
        print(f"\n{plan.target.name} already holds those bytes; nothing to do.")
        return 0
    print(f"\nRestored {plan.target.name}. Journaled as attempt {line.attempt_id} in run "
          f"{run.run_id}; `{_prog(args)} restore --attempt {line.attempt_id} --apply` undoes it.")
    return 0


def cmd_duplicates(args) -> int:
    """Show, and optionally apply, the retirement of duplicate atoms.

    Dry by default, like `periods`: the plan prints, and nothing
    on disk moves without `--apply`. Exits non-zero while duplicates remain —
    a resolvable pair not yet applied, or an undecided one — so a script can
    gate on a clean corpus. Pairs declared distinct or excluded are reported
    and are not duplicates that remain: a plan holding only those exits 0.
    It once exited 1, so the gate could never pass on a corpus
    that had declared any pair distinct.
    """
    from universe_builder import duplicates as dup

    plan = dup.plan(exclude=args.excluded, keep=args.kept,
                    tiebreak_fill=getattr(args, "tiebreak_fill", False))
    if plan.is_clean():
        print("No duplicate atoms: no name is declared by two stored atoms.")
        return 0

    for resolution in plan.resolutions:
        print(f"\n{resolution.atom_type}: {resolution.name!r}")
        print(f"  keep    {resolution.keep.name}")
        for path in resolution.retire:
            print(f"  retire  {path.name}")
        print(f"  because {resolution.reason}")

    for entry in plan.undecided:
        print(f"\n{entry['atom_type']}: {entry['name']!r}  UNDECIDED")
        for name in entry["files"]:
            print(f"  keep?   {name}")
        print(f"  because {entry['why']}")

    for entry in plan.skipped:
        print(f"\n{entry['atom_type']}: {entry['name']!r}  DECLARED DISTINCT")
        for name in entry["files"]:
            print(f"  kept    {name}")

    if not plan.resolutions and not plan.undecided:
        print(f"\n{plan.summary()}. Nothing to retire and nothing to decide.")
        return 0

    if getattr(args, "via", None) == "model":
        def announce(model):
            components, pairs = dup.triage_scope(plan, args.limit)
            log.warning("Triaging %d undecided component(s), %d pair(s), on %s",
                        components, pairs, model)

        # Paid, so journalled — see the same call in `cmd_collisions`.
        model = _paid_setup(args, model=args.triage_model, models_key="duplicates",
                            journal_config={"command": "duplicates", "via": "model",
                                            "limit": args.limit},
                            announce=announce)
        records = dup.triage(plan, model, limit=args.limit)
        counts = dup.verdict_counts(records)
        print("\n  triage: " + ", ".join(f"{v} {n}" for v, n in sorted(counts.items())))
        for record in records:
            if record["verdict"] == "different":
                print(f"    distinct  {record['a']!r} != {record['b']!r}  ({record['why']})")
        if args.out:
            dup.save_verdicts(records, args.out)
            print(f"  {len(records)} verdict(s) saved to {args.out}")
        written = dup.write_distinct_lines(records, dry_run=not args.apply)
        if written["new_lines"]:
            verb = "would append" if written["dry_run"] else "appended"
            print(f"\n  {verb} {len(written['new_lines'])} line(s) to "
                  f"{written['path']}")
            if written["dry_run"]:
                print("  Re-run with --apply to write them.")
        if written["already_present"]:
            print(f"  {written['already_present']} already declared distinct")
        if written["contradicts_alias"]:
            print(f"\n  REFUSED {len(written['contradicts_alias'])} line(s): "
                  "aliases.txt already declares these pairs to be one topic, and "
                  "the two files must not disagree. Decide which is right by hand:")
            for line in written["contradicts_alias"]:
                print(f"    {line}")
        return 0

    outcome = dup.apply(plan, dry_run=not args.apply)
    print(f"\n{plan.summary()}")
    if not args.apply:
        print("\nNothing moved. Re-run with --apply to retire the losing atoms.")
        if outcome["aliases_path"]:
            print(f"It would append to {outcome['aliases_path']}:")
            for line in outcome["aliases"]:
                print(f"  {line}")
            print("\nThe alias lines are not optional decoration: without them the "
                  f"retired spelling is still listed in a {active_pack().index.name} "
                  "atom, so the next create run regenerates the duplicate.")
        else:
            print("This domain has no alias file, so the retired spelling would "
                  "be regenerated by the next create run; --apply is refused.")
    else:
        print(f"Retired {len(outcome['retired'])} file(s) to _duplicates/ "
              f"and wrote {len(outcome['aliases'])} alias line(s).")
        if plan.undecided:
            print(f"{len(plan.undecided)} pair(s) left untouched — decide those by hand.")
    return 1 if plan.undecided or not args.apply else 0


def _resolution(config, args, atoms: Sequence[str]):
    """What each atom type will actually run on, and why.

    One entry per atom, not one for the whole run: `atoms.<atom>.<phase>`
    overrides mean the answer genuinely differs between them, and a dry run that
    prints a single model hides exactly the thing you turned the dry run on to
    check.
    """
    if args.phase == "index":
        index_name = active_pack().index.name
        return [(index_name, config.resolve_model("index", index_name, args.model, provider=args.provider))]
    # `create` never generates a History atom — the history phase does — and
    # `_run_create` skips it. Listing it here would advertise a model for work
    # that will not happen. Nor does a file phase take it unless `--atom`
    # named it (`work.takes`).
    from universe_builder import work

    index_name = active_pack().index.name
    listed = [a for a in atoms if not (a == index_name and (
        args.phase == "create" or not work.takes(args.phase, a, bool(args.atom))))]
    return [(atom, config.resolve_model(args.phase, atom, args.model, provider=args.provider))
            for atom in listed]


def _efforts(config, args, resolution) -> dict:
    """atom -> the effort it will ask for and why (v0.6.0), per atom type as
    the models are (`PipelineConfig.resolve_effort`)."""
    return {atom: config.resolve_effort(args.phase, atom, getattr(args, "effort", None))
            for atom, _ in resolution}


def _effort_refusals(resolution, efforts) -> List[str]:
    """Pre-flight: an effort level some atom's provider lacks (`effort.refusal`)."""
    return [f"{atom}: {problem}" for atom, r in resolution
            for problem in [effort.refusal(r.value, efforts[atom].value)] if problem]


def _task_budgets(config, args, resolution) -> dict:
    """atom -> the task budget it will send and why (v0.6.1), per atom type
    (`PipelineConfig.resolve_task_budget`)."""
    return {atom: config.resolve_task_budget(args.phase, atom, getattr(args, "task_budget", None))
            for atom, _ in resolution}


def _task_budget_refusals(resolution, budgets, max_tokens: int) -> List[str]:
    """Pre-flight: a task budget some atom's model cannot take as asked
    (`task_budget.refusal`: under the minimum, over max_tokens, or a Claude
    model that takes none)."""
    return [f"{atom}: {problem}" for atom, r in resolution
            for problem in [task_budget_mod.refusal(r.value, budgets[atom].value, max_tokens)]
            if problem]


def _with_effort(resolved, effort_resolved, budget_resolved=None, max_tokens: Optional[int] = None) -> str:
    """A dry run's model line, with the effort beside it, and the task budget
    when one is asked for (v0.6.1)."""
    if effort_resolved.value:
        if not effort.sent_or_none(resolved.value, effort_resolved.value):
            line = (f"{resolved}   effort {effort_resolved.value} asked ({effort_resolved.source}), "
                    "none sent: this model takes no effort")
        else:
            line = f"{resolved}   effort {effort_resolved.value}  ({effort_resolved.source})"
    else:
        line = f"{resolved}   effort unset (none sent: the provider's default)"
    if budget_resolved is not None and budget_resolved.value:
        from universe_builder.providers.types import DEFAULT_MAX_TOKENS

        if not task_budget_mod.sent_or_none(resolved.value, budget_resolved.value,
                                            max_tokens or DEFAULT_MAX_TOKENS):
            line += (f"   task budget {budget_resolved.value} asked ({budget_resolved.source}), "
                     "none sent: this provider takes none")
        else:
            line += (f"   task budget {budget_resolved.value} of max_tokens {max_tokens}  "
                     f"({budget_resolved.source})")
    return line


def cmd_run(args) -> int:
    config = pipeline.load(args.config)
    if _check_atom(args):
        return 2
    from universe_builder import work as work_mod

    refusal = work_mod.index_refusal(args.phase, args.atom)
    if refusal:
        log.error("%s", refusal)
        return 2
    atoms = [args.atom] if args.atom else _atom_types()
    try:
        resolution = _resolution(config, args, atoms)
    except pipeline.UnknownTier as e:
        log.error("%s", e)
        return 2

    if not resolution:
        # `--phase create --atom history` is the way in: create never generates
        # History atoms, so `_resolution` filters the only atom asked for and
        # leaves nothing. Previously this indexed an empty list and printed a
        # traceback.
        log.error(
            "Nothing to do: %r does not run over %s. Index atoms are produced "
            "by `%s run --phase index`.",
            args.phase, ", ".join(atoms), _prog(args),
        )
        return 2
    resolved = resolution[0][1]
    try:
        efforts = _efforts(config, args, resolution)
    except ValueError as e:
        log.error("%s", e)
        return 2
    refused = _effort_refusals(resolution, efforts)
    if refused:
        log.error("effort refused:\n    %s", "\n    ".join(refused))
        return 2
    max_tokens = config.max_tokens(args.phase)
    try:
        budgets = _task_budgets(config, args, resolution)
    except ValueError as e:
        log.error("%s", e)
        return 2
    refused = _task_budget_refusals(resolution, budgets, max_tokens)
    if refused:
        log.error("task budget refused:\n    %s", "\n    ".join(refused))
        return 2
    if getattr(args, "experiment", None) and any(r.value for r in budgets.values()):
        # B (experiments.submit) carries no task budget yet, so A would run
        # with one and B without: the arms would differ in a variable nobody
        # named. Refused until experiments take one per arm (ARCHITECTURE §2.1g).
        log.error("--experiment does not take a task budget yet (--task-budget or pipeline.yaml's "
                  "task_budget block): B would run without the one A sends")
        return 2

    arm_b = [flag for flag, on in (("--model-b", getattr(args, "model_b", None)),
                                   ("--effort-b", getattr(args, "effort_b", None)),
                                   ("--allow-dear-b", getattr(args, "allow_dear_b", False))) if on]
    if arm_b and not getattr(args, "experiment", None):
        log.error("%s describe an experiment's B arm; they need --experiment", ", ".join(arm_b))
        return 2

    if getattr(args, "experiment", None):
        refusal = _experiment_refusal(args, config, resolution)
        if refusal:
            log.error("%s", refusal)
            return 2

    if args.dry_run:
        print("DRY RUN — nothing will be called and nothing will be written\n")
        print(f"phase      {args.phase}")
        print(f"mode       {'batch (submit now, collect later)' if args.batch else 'synchronous'}")
        print(f"provider   {args.provider or config.provider}")
        distinct = {(r.value, efforts[atom].value, budgets[atom].value) for atom, r in resolution}
        if len(distinct) == 1:
            first = resolution[0][0]
            print(f"model      {_with_effort(resolved, efforts[first], budgets[first], max_tokens)}")
        else:
            print("models     (they differ by atom type)")
            for atom, r in resolution:
                print(f"    {atom:<12} {_with_effort(r, efforts[atom], budgets[atom], max_tokens)}")
        print(f"atoms      {', '.join(atoms)}")
        print(f"data dir   {settings.data_dir()}")
        work = _plan(args.phase, atoms, args.limit, batch=bool(args.batch), explicit=bool(args.atom))
        print(f"work items {len(work)}")
        for item in work[:20]:
            print(f"    {item}")
        if len(work) > 20:
            print(f"    ... and {len(work) - 20} more")
        if getattr(args, "experiment", None):
            _print_shadow_plan(args, resolution)
        return 0

    # Install the retry/timeout policy the config promises. Until this call
    # existed, `universe_builder config` displayed the retry block and nothing applied it.
    provider_run.configure(config.retry)

    if args.batch:
        # No journal run for a submission: it records nothing there (the batch
        # handle in `_batches/` is its record, and `collect` journals each
        # result into a run of its own), so opening one left an empty run
        # directory behind every `run --batch`.
        # `config.max_tokens` once had no caller anywhere, so every
        # batch ran at `DEFAULT_MAX_TOKENS` (16,384) no matter what the config
        # said — the same failure `provider_run.configure` above was added to
        # fix for `retry`. It truncated cities at exactly that limit, after
        # `defaults.max_tokens` had been raised specifically to stop that.
        return _submit_batch(args, atoms, resolution,
                             max_tokens=max_tokens,
                             efforts={a: r.value for a, r in efforts.items()},
                             task_budgets={a: r.value for a, r in budgets.items()})

    run_config = {"phase": args.phase, "atoms": list(atoms), "model": resolved.value,
                  "source": resolved.source, "limit": args.limit, "batch": False}
    asked = {atom: r.value for atom, r in efforts.items() if r.value}
    if asked:
        # Only when one is sent (v0.6.0), so a run without one records the
        # config, and its hash, it always did.
        run_config["effort"] = (next(iter(asked.values())) if len(set(asked.values())) == 1
                                and len(asked) == len(efforts) else asked)
    sent_budgets = {atom: task_budget_mod.sent_or_none(r.value, budgets[atom].value, max_tokens)
                    for atom, r in resolution if budgets[atom].value}
    sent_budgets = {atom: b for atom, b in sent_budgets.items() if b}
    if sent_budgets:
        # Likewise only when one is sent (v0.6.1): the value on the wire, one
        # number when every atom type sends the same, else per atom type.
        run_config["task_budget"] = (next(iter(sent_budgets.values()))
                                     if len(set(sent_budgets.values())) == 1
                                     and len(sent_budgets) == len(budgets) else sent_budgets)
    journal.start_run(config=run_config, models={args.phase: resolved.value})

    # The real run still goes through the orchestrator; this entry point exists
    # to make the configuration explicit, not to reimplement the passes.
    #
    # `args.model` is passed through *unresolved* rather than the answer above:
    # `run_phase` resolves per atom type, and handing it one already-resolved
    # model is what silently discarded every `atoms.<atom>.<phase>` override.
    from universe_builder import execute_prompt

    for atom, r in resolution:
        log.warning("phase %s / %s -> %s", args.phase, atom,
                    _with_effort(r, efforts[atom], budgets[atom], max_tokens))
    return execute_prompt.run_phase(args.phase, atoms=atoms, model=args.model,
                                    limit=args.limit, provider=args.provider,
                                    config=args.config, explicit=bool(args.atom),
                                    **({"effort": args.effort} if getattr(args, "effort", None) else {}),
                                    **({"task_budget": args.task_budget}
                                       if getattr(args, "task_budget", None) else {}))


#: Phases that can be submitted as batch work.
#:
#: `create` needs only a subject: the destination follows from it, because ids
#: are content-addressed. `fix` and `verify` rewrite *a particular file*, which
#: for a long time is why they were refused — a batch result could not be routed
#: back to the file its payload came from, only to a path re-derived from the
#: subject, which is not the same thing for a file with a different model version
#: or an `_ERROR` decoration. `BatchItem.target` now records it.
#:
#: `expand` batches by a different shape than the other three: it fans out one
#: request per period and `_submit_expand_batch` splices the replies back per
#: record on collect (verified live). `append` stays out — it
#: produces a *separate* file that `merge_helper` later pairs with the original,
#: which is neither one-request-in, one-file-out nor a same-file rewrite.
#:
#: `index` joined later ("index via batch"): one request per seed
#: subject with no index atom yet, built like create's (the destination follows
#: from the subject) and so collected by the same path. It was refused while
#: the seed was a handful of subjects; at a few dozen seeds on the frontier tier the
#: half price is worth the wait. See `_submit_index_batch`.
BATCHABLE_PHASES = ("index", "create", "fix", "verify", "expand")

#: Phases whose requests carry the existing file's JSON.
_PAYLOAD_PHASES = ("fix", "verify")


def _submit_batch(args, atoms: Sequence[str], resolution, *, max_tokens: int,
                  efforts: Optional[Mapping[str, Optional[str]]] = None,
                  task_budgets: Optional[Mapping[str, Optional[int]]] = None) -> int:
    """Submit a phase as batch work instead of running it synchronously.

    One batch per (atom type, context) pair rather than one overall. The context
    is part of the system text and the system text is the cached prefix, so a
    batch spanning two civilisations carries two prefixes and neither request
    group reads the other's cache entry. `batch.submit` warns when that happens;
    grouping here means it does not have to.
    """
    from universe_builder import topics
    from universe_builder.providers import batch as batch_mod
    from universe_builder.providers.requests import with_create_note

    if args.phase not in BATCHABLE_PHASES:
        log.error(
            "--batch supports %s; %r is not one-request-in one-file-out, so it has "
            "no batch shape yet. Run it synchronously.",
            ", ".join(BATCHABLE_PHASES), args.phase,
        )
        return 2

    models = dict(resolution)
    efforts = dict(efforts or {})
    budgets = dict(task_budgets or {})

    if args.phase == "index":
        return _submit_index_batch(args, models, max_tokens=max_tokens, efforts=efforts,
                                   task_budgets=budgets)
    if args.phase in _PAYLOAD_PHASES:
        return _submit_payload_batch(args, atoms, models, max_tokens=max_tokens, efforts=efforts,
                                     task_budgets=budgets)
    if args.phase == "expand":
        return _submit_expand_batch(args, atoms, models, max_tokens=max_tokens, efforts=efforts,
                                    task_budgets=budgets)

    from universe_builder import work

    groups = []
    found_by_atom = {}
    for atom in atoms:
        if atom == active_pack().index.name:
            continue
        model = models[atom].value
        # Only topics without a real atom: `collect` writes through
        # `store.write_atom`, so a topic submitted twice is an existing atom
        # *replaced*, not merely a request paid for twice. `--limit` is per
        # atom type (`work.select_work`).
        found = [item.topic for item in work.select_work("create", atom, args.limit)]
        if not found:
            log.warning("create %s: nothing to generate", atom)
            continue
        found_by_atom[atom] = found
        notes = topics.load_create_notes()
        index = topics.ListIndex(atom)
        for context, subjects in topics.by_context(found).items():
            # A subject with a note goes as a `BatchItem` carrying it in the user
            # turn; the rest stay bare strings, their requests unchanged.
            items = [with_create_note(atom, subject, context, notes, index) for subject in subjects]

            def submit(atom=atom, items=items, model=model, context=context):
                log.warning("Submitting %d %s subject(s) for %r", len(items), atom, context)
                return batch_mod.submit(atom, items, model, context=context, max_tokens=max_tokens,
                                        effort=efforts.get(atom), task_budget=budgets.get(atom))

            groups.append((f"create {atom} ({context})", submit))

    submission = work.submit_groups(groups)
    status = _report_submission(submission,
                                f"Nothing submitted: no subjects found for {', '.join(atoms)}",
                                batch_prog=_batch_prog(args))
    if getattr(args, "experiment", None) and submission.submitted:
        # After A, never before: A's submission is the real work and must not
        # wait on, or be stopped by, anything the experiment does.
        shadow = _submit_shadow(args, found_by_atom, models, submission.submitted, max_tokens, efforts)
        status = status or shadow
    return status


def _experiment_refusal(args, config, resolution) -> Optional[str]:
    """Why `run --experiment` cannot go as asked, or None."""
    from universe_builder import experiments

    if args.phase != "create" or not args.batch:
        return ("--experiment runs with `--phase create --batch` only: B rides alongside a real "
                "create batch (use `experiment replay` for subjects that already have atoms)")
    models_a = {atom: r.value for atom, r in resolution}
    efforts_a = {atom: config.resolve_effort("create", atom, getattr(args, "effort", None)).value
                 for atom in models_a}
    models_b, efforts_b, refused = _arm_plan(args, config, models_a, efforts_a)
    if refused:
        return "--experiment refused:\n    " + "\n    ".join(refused)
    try:
        experiments.prepare(args.experiment, mode="shadow", variant=_shadow_variant(args),
                            rate=args.sample_rate, min_per_type=args.min_per_type, dry_run=True,
                            resolved=_resolved(models_a, efforts_a, models_b, efforts_b))
    except (experiments.ExperimentError, ValueError) as e:
        return str(e)
    return None


def _shadow_variant(args) -> str:
    """B's prompt: `--variant`, else `live` when B differs only in model or
    effort, else the experiment's name."""
    if args.variant:
        return args.variant
    return "live" if (getattr(args, "model_b", None) or getattr(args, "effort_b", None)) else args.experiment


def _print_shadow_plan(args, resolution) -> None:
    from universe_builder import experiments, work

    # From the work list itself: the dry run's `_plan` is printable labels.
    by_atom = {}
    for atom, _ in resolution:
        found = [item.topic for item in work.select_work("create", atom, args.limit)]
        if found:
            by_atom[atom] = found
    chosen = experiments.draw(args.experiment, by_atom, args.sample_rate, args.min_per_type)
    config = pipeline.load(args.config)
    models = {atom: r.value for atom, r in resolution}
    efforts_a = {atom: config.resolve_effort("create", atom, getattr(args, "effort", None)).value
                 for atom in models}
    models_b, efforts_b, _ = _arm_plan(args, config, models, efforts_a)
    shown = _show(_resolved(models, efforts_a, models_b, efforts_b)["B"])
    print(f"\nexperiment {args.experiment}: B = {_shadow_variant(args)} on {shown}, "
          f"into {experiments.folder(args.experiment)}")
    for atom, found in sorted(chosen.items()):
        print(f"    {atom:<12} {len(found)} of {len(by_atom[atom])}")
    print(f"    B generation ~${experiments.estimate(chosen, models_b, arms=1):.3f} (batch)")


def _submit_shadow(args, found_by_atom, models, submitted_paths, max_tokens: int, efforts=None) -> int:
    """Submit the B arm for a sample of what A just submitted (v0.6.0: on B's
    own model and effort when `--model-b`/`--effort-b` say so)."""
    from universe_builder import experiments
    from universe_builder.providers import batch as batch_mod

    config = pipeline.load(args.config)
    models_a = {a: m.value for a, m in models.items()}
    efforts_a = {a: (efforts or {}).get(a) for a in models_a}
    models_b, efforts_b, _ = _arm_plan(args, config, models_a, efforts_a)
    try:
        meta, spec_a, spec_b = experiments.prepare(
            args.experiment, mode="shadow", variant=_shadow_variant(args),
            rate=args.sample_rate, min_per_type=args.min_per_type,
            resolved=_resolved(models_a, efforts_a, models_b, efforts_b))
    except (experiments.ExperimentError, ValueError) as e:
        log.error("experiment not submitted (A went ahead): %s", e)
        return 1
    a_ids = []
    for path in submitted_paths:
        try:
            a_ids.append(batch_mod.load(path).batch_id)
        except Exception as e:  # noqa: BLE001
            log.error("cannot read A handle %s: %s", path, e)
    chosen = experiments.draw(args.experiment, found_by_atom, args.sample_rate, args.min_per_type)
    try:
        outcome = experiments.submit(
            meta, {"A": spec_a, "B": spec_b}, chosen, models_a,
            arms=("B",), max_tokens=max_tokens,
            population={a: len(v) for a, v in found_by_atom.items()}, a_batch_ids=a_ids,
            models_b=models_b, efforts={"A": efforts_a, "B": efforts_b})
    except experiments.ExperimentError as e:
        log.error("experiment not submitted (A went ahead): %s", e)
        return 1
    print(f"\nexperiment {args.experiment}: B submitted for {outcome['submitted'].get('B', 0)} "
          f"subject(s) into {experiments.folder(args.experiment)}")
    for arm, atom, context, error in outcome["failed"]:
        print(f"    NOT submitted: {arm} {atom} ({context}): {error}")
    print(f"After `collect --all` has collected A:  {_prog(args)} experiment collect {args.experiment}")
    return 1 if outcome["failed"] else 0


def cmd_experiment(args) -> int:
    from universe_builder import experiments, prompt_spec

    verb = args.experiment_command
    try:
        if verb == "list":
            for meta in experiments.list_experiments():
                b = meta["arms"]["B"]
                model_b = b.get("model") if isinstance(b.get("model"), str) else "per-atom models"
                effort_b = b.get("effort") if isinstance(b.get("effort"), str) else None
                on = f" on {effort.label(model_b, effort_b)}" if b.get("model") else ""
                print(f"{meta['name']:<28} {meta['mode']:<7} B={meta['variant'] + on:<24} "
                      f"runs {len(meta['runs'])}  since {meta['created']}")
            print(f"\nvariants: {', '.join(prompt_spec.available())}")
            print(f"root:     {settings.experiments_dir()}")
            return 0
        if verb == "collect":
            outcome = experiments.collect(args.name)
            print(f"{args.name}: {outcome['written']} valid and {outcome['failed']} invalid repl(ies) "
                  f"written, {outcome['pending']} batch(es) still running, {outcome['missing']} missing, "
                  f"{outcome['from_journal']} A repl(ies) copied from the corpus journal")
            for line in outcome["refused"]:
                print(f"    refused: {line}")
            return 1 if outcome["refused"] else 0
        if verb == "report":
            config = pipeline.load(args.config)
            judges = [_judge_arg(j, config) for j in (args.panel or "").split(",") if j.strip()]
            if args.panel and len(judges) < 2:
                log.error("--panel takes two or more judges, comma-separated")
                return 2
            result = experiments.report(args.name, _judge_arg(args.judge_model, config) if args.judge_model else None,
                                        panel_judges=judges or None)
            print(experiments.render(result))
            print(f"(written to {experiments.folder(args.name)})")
            return 0
        if verb == "judge":
            return _experiment_judge(args)
        if verb == "replay":
            return _experiment_replay(args)
    except (experiments.ExperimentError, prompt_spec.VariantError, ValueError, RuntimeError) as e:
        # RuntimeError: a pack's reference source that is down (history's
        # local Wikipedia) stops the judge, said plainly rather than traced.
        log.error("%s", e)
        return 2
    return 2


def _arm_plan(args, config, models_a: Mapping[str, str], efforts_a: Mapping[str, Optional[str]]):
    """B's model and effort per atom type (v0.6.0) and the refusals that stop
    an experiment before anything is spent: A over the price ceiling, B over
    it without `--allow-dear-b`, or an effort level a provider lacks.
    Returns `(models_b, efforts_b, refusals)`. B's model defaults to A's and
    its effort to A's, so a prompt experiment varies only the prompt."""
    from universe_builder import experiments

    model_b = config.resolve_tier(args.model_b) if getattr(args, "model_b", None) else None
    models_b = {atom: model_b or models_a[atom] for atom in models_a}
    efforts_b = {atom: getattr(args, "effort_b", None) or efforts_a.get(atom) for atom in models_a}
    refusals = []
    if not getattr(args, "reuse_a", None):
        refusals += [f"A {line}" for line in experiments.price_refusals(models_a, config)]
    if not getattr(args, "allow_dear_b", False):
        refusals += [f"B {line} (--allow-dear-b to accept it for B)"
                     for line in experiments.price_refusals(models_b, config)]
    for arm, models, efforts in (("A", models_a, efforts_a), ("B", models_b, efforts_b)):
        if arm == "A" and getattr(args, "reuse_a", None):
            continue
        refusals += [f"{arm} {atom}: {problem}" for atom in models
                     for problem in [effort.refusal(models[atom], efforts.get(atom))] if problem]
    return models_b, efforts_b, refusals


def _resolved(models_a, efforts_a, models_b, efforts_b) -> dict:
    """Both arms' resolution per atom type, with the effort each sends
    (`experiments.arm_resolution`), for `prepare`: what decides whether the
    arms differ, and what `meta.arms` records."""
    from universe_builder import experiments

    return {"A": experiments.arm_resolution(models_a, efforts_a),
            "B": experiments.arm_resolution(models_b, efforts_b)}


def _show(resolution: Mapping[str, Mapping[str, Optional[str]]]) -> str:
    """An arm's models and the efforts they send: `gpt-6-luna@medium, ...`."""
    pairs = sorted({effort.label(r["model"], r["effort"]) for r in resolution.values()})
    return ", ".join(pairs) or "—"


def _judge_arg(text: str, config=None) -> str:
    """A judge as `report` names it: MODEL or MODEL@EFFORT, a tier resolved
    against `config` (`--config`, else the bundled file)."""
    model, level = effort.split_label(text.strip())
    return effort.label((config or pipeline.load()).resolve_tier(model), level)


def _experiment_replay(args) -> int:
    from universe_builder import experiments

    config = pipeline.load(args.config)
    if args.reuse_a and (args.effort_a or args.only or args.context):
        log.error("--reuse-a takes A, its model and effort, and the sample from %s; it cannot be "
                  "combined with --effort-a, --only or --context", args.reuse_a)
        return 2
    atoms = [args.atom] if args.atom else experiments.generating_atoms()
    if args.reuse_a:
        # A's model and effort are what the source experiment recorded for
        # them, never guessed from today's pipeline.yaml; an atom type it did
        # not sample, or does not generate today, is left out.
        recorded = experiments.load_meta(args.reuse_a)["arms"]["A"].get("resolved")
        if not recorded:
            log.error("--reuse-a: %s records no model or effort for its A (it predates v0.6.0); "
                      "generate A again", args.reuse_a)
            return 2
        sampled = set(experiments.load_sample(args.reuse_a))
        atoms = [a for a in atoms if a in sampled]
        unknown = [a for a in atoms if a not in recorded]
        if unknown:
            log.error("--reuse-a: %s records no A model for %s", args.reuse_a, ", ".join(unknown))
            return 2
        models_a = {atom: recorded[atom]["model"] for atom in atoms}
        efforts_a = {atom: recorded[atom].get("effort") for atom in atoms}
    else:
        models_a = {atom: config.resolve_model("create", atom).value for atom in atoms}
        efforts_a = {atom: config.resolve_effort("create", atom, args.effort_a).value for atom in atoms}
    models_b, efforts_b, refused = _arm_plan(args, config, models_a, efforts_a)
    if refused:
        log.error("replay refused:\n    %s", "\n    ".join(refused))
        return 2
    resolved = _resolved(models_a, efforts_a, models_b, efforts_b)
    variant = args.variant or ("live" if (args.model_b or args.effort_b) else args.name)
    meta, spec_a, spec_b = experiments.prepare(args.name, mode="replay", variant=variant,
                                               rate=0.0, min_per_type=args.per_type, dry_run=not args.apply,
                                               resolved=resolved, reuse_a=args.reuse_a)
    if args.reuse_a:
        chosen, missing, skipped = experiments.reuse_a(meta, args.reuse_a, atoms, spec=spec_a,
                                                       max_tokens=config.max_tokens("create"),
                                                       write=args.apply)
        listed = chosen
        if missing:
            print(f"    {missing} sampled subject(s) in {args.reuse_a} have no A reply (collect it first); "
                  "B is not bought for them")
        if skipped:
            print(f"    {args.reuse_a}'s sample also holds {', '.join(skipped)}, not taken here")
    else:
        only = None
        if args.only:
            from universe_builder import sources
            only = sources.load_only(args.only)
        listed = experiments.replay_topics(atoms, args.context, only=only)
        chosen = experiments.draw(args.name, listed, 0.0, args.per_type)
    total = sum(len(v) for v in chosen.values())
    a_side = f"A reused from {args.reuse_a}" if args.reuse_a else "both arms"
    print(f"replay {args.name}: B = {meta['variant']}, {total} subject(s), {a_side}, into "
          f"{experiments.folder(args.name)}")
    print(f"    A on {_show(resolved['A'])}; B on {_show(resolved['B'])} (the efforts each sends)")
    for atom, found in sorted(chosen.items()):
        print(f"    {atom:<12} {len(found)} of {len(listed[atom])}")
    cost_a = 0.0 if args.reuse_a else experiments.estimate(chosen, models_a, arms=1)
    cost_b = experiments.estimate(chosen, models_b, arms=1)
    print(f"    generation ~${cost_a + cost_b:.3f} (batch; A ~${cost_a:.3f}, B ~${cost_b:.3f}; "
          "a nominal 2k output tokens per call, which an effort setting moves)")
    if not args.apply:
        print("\nEstimate only. Add --apply to submit.")
        return 0
    outcome = experiments.submit(meta, {"A": spec_a, "B": spec_b}, chosen, models_a,
                                 arms=("B",) if args.reuse_a else ("A", "B"),
                                 max_tokens=config.max_tokens("create"),
                                 population={a: len(v) for a, v in listed.items()},
                                 models_b=models_b, efforts={"A": efforts_a, "B": efforts_b})
    print(f"submitted: {outcome['submitted']}")
    for arm, atom, context, error in outcome["failed"]:
        print(f"    NOT submitted: {arm} {atom} ({context}): {error}")
    print(f"Collect when ready:  {_prog(args)} experiment collect {args.name}")
    return 1 if outcome["failed"] else 0


def _experiment_judge(args) -> int:
    from universe_builder import experiments

    config = pipeline.load(args.config)
    model = config.resolve_tier(args.model)
    level = getattr(args, "effort", None)
    problem = effort.refusal(model, level)
    if problem:
        log.error("judge refused: %s", problem)
        return 2
    plan = experiments.plan_judging(args.name, model, sample=args.sample, limit=args.limit, effort=level)
    requests = plan["requests"]
    cost = experiments.judge_estimate(requests, model) if requests else 0.0
    print(f"judge {args.name} on {effort.label(model, level)}: {plan['pairs']} complete pair(s), "
          f"{len(requests)} call(s) to make (both orders), ~${cost:.3f} (sync)")
    if plan["no_reference"]:
        print(f"    {len(plan['no_reference'])} pair(s) have no reference and are not judged: "
              f"{', '.join(plan['no_reference'][:8])}{' ...' if len(plan['no_reference']) > 8 else ''}")
    if not requests:
        return 0
    if not args.apply:
        print("\nEstimate only. Add --apply to send.")
        return 0
    provider_run.configure(config.retry)
    tally = experiments.judge(args.name, requests, concurrency=args.concurrency)
    print(f"judged: {tally['ok']} ok, {tally['failed']} failed, ${tally['cost']:.4f}")
    print(f"Next:  {_prog(args)} experiment report {args.name}")
    return 1 if tally["failed"] and not tally["ok"] else 0


def _submit_index_batch(args, models, *, max_tokens: int, efforts=None, task_budgets=None) -> int:
    """Submit the index phase as batch work: one request per seed subject
    with no index atom yet ("index via batch").

    The selection is `work.select_work("index", ...)`, the one `_run_history`
    makes, so a seed with an atom is skipped here as it is there (a collected
    result is written through `store.write_atom` and would replace it).
    `--limit` caps seeds.

    The request is what the synchronous index call sends, modulo the
    documented builder differences (`tests/test_index_batch.py`): no context
    in the system text, since `_run_history` frames none; the pack's create
    note for the seed, if any, in the user turn; the phase's own
    `max_tokens`; the index phase's model (frontier). It routes
    `operation="index"` — no operation text exists under that name, so the
    prompt is create's — which `collect` journals, and `costs --by phase`
    reports as `index (batch)` at the batch discount.

    One batch per seed, as create sends one per context: each seed is its own
    context, so a seed's result can be collected as soon as its batch is done,
    and a provider refusing one seed's batch is reported beside the others.
    """
    from universe_builder import topics, work
    from universe_builder.providers import batch as batch_mod
    from universe_builder.providers.requests import with_create_note

    index_name = active_pack().index.name
    model = models[index_name].value
    index_effort = (efforts or {}).get(index_name)
    index_task_budget = (task_budgets or {}).get(index_name)
    every = topics.civilisations()
    missing = work.select_work("index", index_name)
    if len(missing) != len(every):
        log.warning("index: %d of %d seed subject(s) already have an atom; they are skipped",
                    len(every) - len(missing), len(every))
    selected = work.apply_limit(missing, args.limit, "index")
    notes = topics.load_create_notes()
    groups = []
    for item in selected:
        # No context: the synchronous index call (`_run_history` ->
        # `create_atom`) frames none, so its note is looked up unscoped too.
        entry = with_create_note(index_name, item.subject, None, notes)

        def submit(entry=entry, subject=item.subject):
            log.warning("Submitting %s %r to %s", index_name, subject, model)
            return batch_mod.submit(index_name, [entry], model, operation="index",
                                    max_tokens=max_tokens, effort=index_effort,
                                    task_budget=index_task_budget)

        groups.append((f"index {item.subject}", submit))

    return _report_submission(work.submit_groups(groups),
                              f"index: nothing to submit; every seed subject has a {index_name} atom",
                              quiet_nothing=True,
                              batch_prog=_batch_prog(args))


def _batch_prog(args) -> str:
    """The batch entry point to name in a hint: `python -m <prog>.run_batch`.
    A wrapper passes its own prog (`<pack>`) and ships a `<prog>.run_batch`
    that loads its `.env` and pack; the engine's would load neither."""
    return f"python -m {_prog(args)}.run_batch"


def _report_submission(submission, nothing: str, *, quiet_nothing: bool = False,
                       batch_prog: str = "python -m universe_builder.run_batch") -> int:
    """Print what a batch submission did. Non-zero unless every group went.

    A group that failed is listed beside the handles that did go: those are
    submitted and will be billed, and a bare traceback used to hide them.
    """
    if submission.submitted:
        print(f"{len(submission.submitted)} batch(es) submitted:")
        for path in submission.submitted:
            print(f"    {path}")
    if submission.failed:
        print(f"\n{len(submission.failed)} group(s) NOT submitted:")
        for label, error in submission.failed:
            print(f"    {label}: {error}")
    if submission.submitted:
        print(f"\nCollect when ready:  {batch_prog} collect --all")
    if submission.failed:
        return 1
    if not submission.submitted:
        (log.warning if quiet_nothing else log.error)("%s", nothing)
        return 1
    return 0


def _submit_expand_batch(args, atoms: Sequence[str], models, *, max_tokens: int, efforts=None,
                         task_budgets=None) -> int:
    """Submit `expand` as batch work: one request per unexpanded list entry.

    Expand fans out where the other passes hand back one-file-out: every
    still-string entry of every atom becomes its own request, carrying the
    atom's JSON as payload and its path as target, tagged with the entry index
    and name so `collect` can splice the replies back into one file. Which list
    field is expanded, and into what, comes from each atom's `ExpandSpec` in the
    active pack — so this works for a city's periods, a medication's
    indications, or any domain's expandable atom, not city alone. Only atoms
    with work are submitted, and only their remaining entries, so a re-run after
    a partial expansion asks for just what is left. One batch per atom type.
    """
    import json

    from universe_builder.operations.expand import expand_config, expansion_instructions, prepare_expansion
    from universe_builder.providers import batch as batch_mod
    from universe_builder.providers.requests import BatchItem

    specs = [s for s in active_pack().all_specs()
             if s.expand and (not atoms or s.name in atoms)]
    if not specs:
        selected = ", ".join(atoms) if atoms else "the active pack"
        log.error("--phase expand --batch: no expandable atom types in %s", selected)
        return 2

    from universe_builder import work

    root = settings.data_dir()
    default_model = models[next(iter(models))].value
    groups = []

    for spec in specs:
        model = models.get(spec.name)
        model = model.value if model is not None else default_model
        cfg = expand_config(spec.name, model)
        entry_schema = spec.expand.entry_model.__name__

        # `work.select_work`: the files with entries left, at most `--limit` of
        # them for this atom type. Every pending entry of a selected file is
        # asked for, so a limited run finishes whole files.
        items: List[BatchItem] = []
        for selected in work.select_work("expand", spec.name, args.limit):
            path = selected.path
            data = store.read_json(path)
            if not isinstance(data, dict):
                continue
            try:
                atom_to_expand, seed_elements = prepare_expansion(data, cfg)
            except Exception as e:  # noqa: BLE001 - one bad file must not abort the submit
                log.error("Skipping %s: %s", path.name, e)
                continue
            if atom_to_expand is None:
                continue  # already fully expanded
            try:
                subject = store.AtomName.decompose_filename(path.name).getTopic()
            except ValueError as e:
                log.warning("Skipping %s: %s", path.name, e)
                continue
            payload = json.dumps(data, ensure_ascii=False)
            target = str(path.relative_to(root))
            for i, entry in enumerate(seed_elements):
                if not isinstance(entry, str):
                    continue  # already an object
                items.append(BatchItem(
                    subject=subject,
                    user_instructions=expansion_instructions(seed_elements, i, cfg.subatom_to_expand),
                    payload=payload,
                    target=target,
                    # The name is the stable key: positions shift if the file is
                    # split or re-expanded between submit and collect, so collect
                    # places the reply by name and treats the index as a hint.
                    routing={"period_index": i, "period_name": entry},
                ))

        if not items:
            log.warning("expand: no %s atom has entries left to expand", spec.name)
            continue

        def submit(spec=spec, items=items, model=model, entry_schema=entry_schema,
                   atom_effort=(efforts or {}).get(spec.name),
                   atom_task_budget=(task_budgets or {}).get(spec.name)):
            log.warning("Submitting %d %s entry(ies) for expansion", len(items), spec.name)
            return batch_mod.submit(f"expand_{spec.name}", items, model,
                                    operation="expand", schema_model=entry_schema,
                                    max_tokens=max_tokens, effort=atom_effort,
                                    task_budget=atom_task_budget)

        groups.append((f"expand {spec.name}", submit))

    return _report_submission(work.submit_groups(groups),
                              "expand: nothing to submit; every atom is already fully expanded",
                              quiet_nothing=True,
                              batch_prog=_batch_prog(args))


def _submit_payload_batch(args, atoms: Sequence[str], models, *, max_tokens: int, efforts=None,
                          task_budgets=None) -> int:
    """Submit `fix` or `verify` as batch work.

    Each request carries one file's JSON in the *user* turn and records that
    file's path as the target, so `collect` writes the answer back to where the
    payload came from rather than to a path reconstructed from the subject.

    Batched per atom type, not per context: the payload is what varies here, and
    it is volatile by construction, so every request for an atom type already
    shares one cached prefix.
    """
    import json

    from universe_builder.atoms import resolve_stored_class
    from universe_builder.providers import batch as batch_mod
    from universe_builder.providers.requests import BatchItem

    from universe_builder import work
    from universe_builder.operations import _support

    root = settings.data_dir()
    groups = []

    for atom in atoms:
        if atom == active_pack().index.name:
            continue
        if args.phase == "fix":
            # Same rule as the sync pass: restore journal-queued sentinels the
            # glob would otherwise silently skip, then select as usual.
            _support.reconcile_fix_queue(atom)
        model = models[atom].value

        # Grouped by the shape the file is actually stored in, not by atom type.
        # A city that has been through expand is a `CityExpanded`; asking the
        # model for the seed `City` shape is B7 — the reply validates, overwrites
        # the file, and collapses the expansion to a list of period names. One
        # batch per shape also keeps one JSON schema per cached prefix.
        #
        # `work.select_work` decides which files (sentinels for fix, atoms for
        # verify; never an `{"error": ...}` placeholder) and how many: at most
        # `--limit` per atom type.
        by_shape: dict = {}
        for selected in work.select_work(args.phase, atom, args.limit):
            if selected.restore:
                continue   # one `reconcile_fix_queue` could not restore
            path = selected.path
            is_error = store.is_error_path(path)
            # Guarded for the same reason as `topics.has_real_atom`: one
            # unreadable file must cost one record, not the submission.
            try:
                payload = store.read(path)
            except (OSError, UnicodeDecodeError) as e:
                log.warning("Skipping %s, which cannot be read: %s", path.name, e)
                continue

            # The fix pass writes the *atom*, then retires the sentinel — the
            # `store.commit_fix` ordering. Targeting the sentinel put the repair
            # at `X.json_ERROR`, which `collect`'s `clear_error` then deleted.
            target = store.target_of_error(path) if is_error else path
            try:
                parsed = store.AtomName.decompose_filename(target.name)
            except ValueError as e:
                log.warning("Skipping %s: %s", path.name, e)
                continue

            try:
                shape = resolve_stored_class(atom, json.loads(payload)).__name__
            except (ValueError, TypeError):
                shape = None        # unparseable: fall back to the atom's default

            by_shape.setdefault(shape, []).append(BatchItem(
                subject=parsed.getTopic(),
                payload=payload,
                target=str(target.relative_to(root)),
            ))

        if not by_shape:
            log.warning("%s %s: nothing to do", args.phase, atom)
            continue

        for shape, items in by_shape.items():
            def submit(atom=atom, items=items, model=model, shape=shape):
                log.warning("Submitting %d %s file(s) for %s as %s",
                            len(items), atom, args.phase, shape or "the default shape")
                return batch_mod.submit(atom, items, model, operation=args.phase,
                                        schema_model=shape, max_tokens=max_tokens,
                                        effort=(efforts or {}).get(atom),
                                        task_budget=(task_budgets or {}).get(atom))

            groups.append((f"{args.phase} {atom} ({shape or 'default shape'})", submit))

    return _report_submission(work.submit_groups(groups),
                              f"Nothing submitted: no {args.phase} work for {', '.join(atoms)}",
                              batch_prog=_batch_prog(args))


def _plan(phase: str, atoms: Sequence[str], limit: Optional[int], *, batch: bool = False,
          explicit: bool = False) -> List[str]:
    """What the phase would operate on, one line per item. Reads, calls nothing.

    `work.select_work` per atom type — the selection every run makes — so the
    count is what a run would pay for: at most `limit` per atom type, none for
    `0`, verify without `{"error": ...}` placeholders, fix with the sentinels
    the journal would restore. `batch` drops the index
    atom type from every phase but `index`, which the batch submitters never
    send there; `--phase index --batch` lists the same seeds as the sync run.
    `explicit` (`--atom` was given) lets a file phase list the index atom
    type, which it otherwise never selects (`work.takes`).

    `create` and `history` used to return the placeholder
    "(topics from HISTORY_list.txt)" and a work count of 1, which was wrong for
    both. A dry run that under-reports the work by two orders of magnitude is
    worse than no dry run, since the whole point is knowing what a run will do
    before paying for it.
    """
    from universe_builder import work

    index_name = active_pack().index.name
    if pipeline.modern_phase(phase) == "index":
        return [item.label() for item in work.select_work("index", index_name, limit)]
    items: List[str] = []
    for atom in atoms:
        if batch and atom == index_name:
            continue
        items.extend(item.label() for item in work.select_work(phase, atom, limit, explicit=explicit))
    return items


def main(argv: Optional[Sequence[str]] = None, prog: str = "universe_builder") -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    # See the note in `run_batch.main`: this entry point does not import
    # `llm_call` either, so the `.env` has to be loaded explicitly.
    settings.load_environment()
    # The pack is resolved before parsing, because its operations are verbs:
    # the parser cannot be built without knowing them. A pack that fails to
    # load still gets a parser (so `--help` works) and then the error.
    pack_error = None
    operations = {}
    try:
        operations = active_pack().operation_specs()
    except KeyError as e:
        pack_error = str(e).strip('"')
    args = build_parser(prog, operations).parse_args(argv)
    args.prog = prog
    if pack_error:
        log.error("%s", pack_error)
        return 2
    if getattr(args, "phase", None) is not None:
        args.phase = pipeline.modern_phase(args.phase)
    from universe_builder import topics

    # A work-list warning (an alias contradiction) is about the file, not the
    # atom type, and every type's list re-reads it: once per command.
    with topics.warnings_once():
        if args.command in operations:
            return operations[args.command].handler(args)
        # From module globals at call time, so a test can stand in for one verb.
        return globals()[f"cmd_{args.command}"](args)


if __name__ == "__main__":
    sys.exit(main())
