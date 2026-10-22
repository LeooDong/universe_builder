"""Prompt experiments: A on everything, B on a shadow sample (v0.5.0).

Two modes, one comparison:

* **shadow** — `run --phase create --batch --experiment NAME`. The real create
  batch (A, the live prompt) goes exactly as it would without the flag; a
  deterministic sample of the same subjects is also generated under variant B.
  A's side of each pair is read back, read-only, from the corpus journal
  after the ordinary `collect`.
* **replay** — `experiment replay NAME --variant V --apply`. A sample of
  subjects that already have atoms is generated twice, under the live prompt
  and under B, both here. Nothing in the corpus is read but the work lists.

**Experiment output is test data and lives outside the corpus**:
`settings.experiments_dir()/<name>/`, written only through
`store.write_experiment`, which refuses any path in the data directory (and
`store.write_atom` refuses any path in here). The handles live in this
folder's own `_batches/`, wrapped so `providers.batch.load` refuses them, so
`run_batch collect --all` can never write a B result into the corpus. No
experiment call is journalled into the corpus journal, so `index`'s
provenance, the fix queue and `costs` never see one; this folder keeps its own
spend.

    <experiments>/<name>/
        experiment.json       mode, variant, both fingerprints, acceptance, runs
        sample.json           atom -> content id -> subject, context
        _batches/             wrapped handles; collected/ once fetched
        results/<arm>/<atom>/<content_id>.json   raw reply, usage, cost
        references/<atom>/<content_id>.json      the judge's frozen reference
        judgements.jsonl      one line per (pair, order, judge model)
        report.json, report.md

**Model and effort as the variable** (v0.6.0). B may run on another model
(`--model-b`, any provider: its batches go to the provider its model id names)
and either arm at its own reasoning effort (`--effort-a`, `--effort-b`); both
are written into `meta.arms`, so changing either is a new experiment. The
variant may then be `live` (or any variant with no overrides): the prompt is
held fixed and the model is what is compared. `--reuse-a FROM` copies FROM's
sample and collected A replies instead of generating A again, so several B
arms are judged against one identical A. Judges are keyed by (model, effort)
(`judge_model`, `judge_effort` on each line), so two judges' verdicts sit side
by side, and `report --panel J1,J2` gives each one's tally, their agreement,
and a verdict on the pairs they agree on.

ARCHITECTURE §2.1g describes the design.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from pydantic import BaseModel

from universe_builder import experiment_stats as stats
from universe_builder import journal, prompt_spec, settings, store, topics

log = logging.getLogger(__name__)

MODES = ("shadow", "replay")
ARMS = ("A", "B")
EXPERIMENT_FILE = "experiment.json"
SAMPLE_FILE = "sample.json"
JUDGEMENTS_FILE = "judgements.jsonl"
#: The key that marks a wrapped experiment handle. A wrapped handle has none
#: of `providers.batch._HANDLE_KEYS` at its top level, so the corpus loader
#: refuses it whatever engine version reads it.
HANDLE_MARK = "experiment_handle"

DEFAULT_ACCEPTANCE: Dict[str, Any] = {
    "target_decisive": 120,     # the one pre-registered look for ADOPT
    "alpha": 0.05,              # one-sided sign test, B better
    "harm_alpha": 0.01,         # one-sided sign test, B worse: REJECT at any look
    "cost_cap": 1.25,           # B's uncached-equivalent $/atom over A's
    "fill_margin": 2.0,         # points of fill % B may lose
    "flag_margin_abs": 0.05,    # check flags per atom B may add ...
    "flag_margin_rel": 0.2,     # ... or this share of A's mean, whichever is larger
    "validity_margin_abs": 2,   # B-only failures over A-only failures ...
    "validity_margin_rel": 0.03,  # ... or this share of n, whichever is larger
    "noninferior_bound": 0.45,  # ADOPT IF CHEAPER: one-sided lower bound on B's score
    "cheaper_by": 0.10,         # ... and B at least this much cheaper
    "per_type_min_decisive": 8,
    "permutations": 10_000,
    "seed": 0,
}

JUDGE_INSTRUCTIONS = """\
You compare two generated records about the same subject against a reference text.
Assess each record on its own and fill the fields for it:
- wrong_subject: true if the record is about a different thing than the subject as listed
  (another person, place or work of the same name, or a broader or narrower subject).
- contradictions: each claim in the record that the reference contradicts, quoted briefly with
  what the reference says instead. A claim the reference does not mention is NOT a contradiction.
- false_certainty: each claim stated as certain where the reference hedges, disputes it, or gives
  a range. Again, a claim the reference is silent on does not count.
- coverage: 0 to 3, how much of what the reference treats as central the record includes.
Do not reward length, prose style or field order. Records are labelled 1 and 2; the order means
nothing. Use the reference only; if it does not settle a claim, leave the claim out."""


class ExperimentError(ValueError):
    """An experiment that cannot proceed as asked."""


# -- the judge's reply ----------------------------------------------------------


class RecordAssessment(BaseModel):
    wrong_subject: bool
    contradictions: List[str]
    false_certainty: List[str]
    coverage: int


class PairAssessment(BaseModel):
    record_1: RecordAssessment
    record_2: RecordAssessment
    reason: str


def _rank_key(a: Mapping[str, Any]) -> tuple:
    """Lower is better: wrong subject, then contradictions, then false
    certainty, then (less) coverage. The rubric's priority order, applied in
    code, so a verdict can be audited from the lists the judge wrote."""
    coverage = max(0, min(3, int(a.get("coverage") or 0)))
    return (bool(a.get("wrong_subject")), len(a.get("contradictions") or []),
            len(a.get("false_certainty") or []), -coverage)


def order_winner(parsed: Mapping[str, Any]) -> Optional[int]:
    """1 or 2 for the better record in one judged order, None for a tie."""
    one, two = _rank_key(parsed["record_1"]), _rank_key(parsed["record_2"])
    if one == two:
        return None
    return 1 if one < two else 2


# -- paths and files --------------------------------------------------------------


def folder(name: str) -> Path:
    return store.experiment_dir(name)


def _read(path: Path, default=None):
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def load_meta(name: str) -> dict:
    meta = _read(folder(name) / EXPERIMENT_FILE)
    if meta is None:
        raise ExperimentError(f"no experiment {name!r} at {folder(name)}")
    return meta


def list_experiments() -> List[dict]:
    root = settings.experiments_dir()
    if not root.is_dir():
        return []
    return [m for m in (_read(p) for p in sorted(root.glob(f"*/{EXPERIMENT_FILE}"))) if m]


def generating_atoms() -> List[str]:
    """The atom types create generates: the pack's, less index and helpers."""
    from universe_builder import atoms
    from universe_builder.registry import active_pack

    index = active_pack().index.name
    return [a for a in atoms.atom_mapping
            if a != index and not a.startswith(("append_", "expand_"))]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# -- setup --------------------------------------------------------------------------


def price_refusals(models: Mapping[str, str], config) -> List[str]:
    """Atom types whose model is dearer than the experiment ceiling."""
    from universe_builder.providers import pricing

    ceiling = float((config.experiments or {}).get("max_output_price", 1.0))
    refused = []
    for atom, model in sorted(models.items()):
        try:
            info = pricing.lookup(model)
        except Exception as e:  # noqa: BLE001 - an unpriced model cannot be shown to be cheap
            refused.append(f"{atom}: {model} is not in the pricing table ({e})")
            continue
        if info.output > ceiling:
            refused.append(f"{atom}: {model} outputs at ${info.output:g}/Mtok, over the "
                           f"experiments ceiling of ${ceiling:g} (pipeline.yaml experiments.max_output_price)")
    return refused


def arm_resolution(models: Mapping[str, str], efforts: Optional[Mapping[str, Optional[str]]] = None) -> dict:
    """`{atom: {"model", "effort"}}` for one arm (v0.6.0): the model and the
    effort it puts **on the wire** (`effort.sent`), so an effort a provider or
    model ignores (Gemini, gpt-4o-mini, Haiku 4.5) is recorded as none.
    Raises `EffortNotSupported` for a level a model lacks."""
    from universe_builder.providers.effort import sent

    return {atom: {"model": model, "effort": sent(model, (efforts or {}).get(atom))}
            for atom, model in models.items()}


def resolutions_differ(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """True when, for some atom type both arms resolved, B runs another model
    or sends another effort than A: the model, not only the prompt, is then a
    variable. Compared on resolved values, never on what the command line
    named: `--model-b` equal to A's model is no difference."""
    return any(a[atom] != b[atom] for atom in set(a) & set(b))


def arms_differ(arms: Mapping[str, Mapping[str, Any]]) -> bool:
    """`resolutions_differ` over an experiment's recorded `arms` (False for
    one from before v0.6.0, which records no resolution)."""
    return resolutions_differ((arms.get("A") or {}).get("resolved") or {},
                              (arms.get("B") or {}).get("resolved") or {})


def _summary(resolved: Mapping[str, Mapping[str, Any]], key: str):
    """An arm's model or effort for display: one value when every atom type
    agrees, else `{atom: value}`."""
    values = {atom: r.get(key) for atom, r in resolved.items()}
    distinct = set(values.values())
    return next(iter(distinct)) if len(distinct) == 1 else dict(sorted(values.items()))


def prepare(name: str, *, mode: str, variant: str, rate: float, min_per_type: int,
            dry_run: bool = False, resolved: Optional[Mapping[str, Mapping[str, Any]]] = None,
            reuse_a: Optional[str] = None) -> tuple:
    """`(meta, spec_a, spec_b)` for a new or existing experiment.

    An existing experiment is reattached only when its mode, variant, both
    fingerprints and both arms' resolved models and efforts are what they
    were: pairs from two runs pool only when nothing about either arm has
    moved (review R14). Anything else is a new experiment under a new name.

    `resolved` (v0.6.0): `{"A": arm_resolution(...), "B": ...}`, what each arm
    runs on per atom type. When B's differs from A's for some atom type, the
    variant may send the live prompt (`live`, `aa`, or a variant with no
    overrides): the model is the variable, and the cost ratio is reported
    beside the verdict rather than gating it (`verdict`). `reuse_a` names the
    experiment whose collected A arm this one copies (`reuse_a`): its A
    prompt must be today's live prompt and its A resolution recorded."""
    if mode not in MODES:
        raise ExperimentError(f"mode must be one of {MODES}")
    resolved = {arm: dict((resolved or {}).get(arm) or {}) for arm in ARMS}
    differ = resolutions_differ(resolved["A"], resolved["B"])
    if variant == "live" and not differ:
        raise ExperimentError("B cannot be 'live' unless B's model or effort differs from A's; "
                              "use 'aa' for the A/A control")
    spec_a = prompt_spec.load("live")
    spec_b = prompt_spec.load(variant)
    atoms = generating_atoms()
    fingerprints = {"A": spec_a.fingerprint(atoms), "B": spec_b.fingerprint(atoms)}
    if variant != "aa" and not differ and fingerprints["A"] == fingerprints["B"]:
        raise ExperimentError(f"variant {variant!r} sends exactly what the live pack sends; "
                              "there is nothing to compare (use 'aa' for a deliberate A/A, or "
                              "--model-b/--effort-b to compare models on the same prompt)")
    if reuse_a:
        if reuse_a == name:
            raise ExperimentError("--reuse-a names this experiment itself")
        source_a = load_meta(reuse_a)["arms"]["A"]
        if source_a.get("fingerprint") != fingerprints["A"]:
            raise ExperimentError(f"experiment {reuse_a!r}'s A ran under a live prompt that has moved since; "
                                  "its replies are not today's A. Generate A again (drop --reuse-a)")
        if not source_a.get("resolved"):
            raise ExperimentError(f"experiment {reuse_a!r} records no model or effort for its A (it predates "
                                  "v0.6.0), so what its A ran on is not known; generate A again")
        moved = [atom for atom in resolved["A"] if source_a["resolved"].get(atom) != resolved["A"][atom]]
        if moved:
            raise ExperimentError(f"A's resolution for {', '.join(sorted(moved))} is not {reuse_a!r}'s")
    acceptance = {**DEFAULT_ACCEPTANCE, **dict(spec_b.acceptance)}
    unknown = set(spec_b.acceptance) - set(DEFAULT_ACCEPTANCE)
    if unknown:
        raise ExperimentError(f"variant {variant!r}: unknown acceptance key(s) {sorted(unknown)}")
    path = folder(name) / EXPERIMENT_FILE
    existing = _read(path)
    if existing:
        problems = []
        if existing["mode"] != mode:
            problems.append(f"mode {existing['mode']} -> {mode}")
        if existing["variant"] != variant:
            problems.append(f"variant {existing['variant']} -> {variant}")
        for arm in ARMS:
            if existing["arms"][arm]["fingerprint"] != fingerprints[arm]:
                problems.append(f"arm {arm}'s prompt changed since this experiment began")
            was = existing["arms"][arm].get("resolved") or {}
            for atom in sorted(set(was) & set(resolved[arm])):
                if was[atom] != resolved[arm][atom]:
                    problems.append(f"arm {arm}'s {atom} {was[atom]} -> {resolved[arm][atom]}")
        if existing["arms"]["A"].get("reused_from") != reuse_a:
            problems.append(f"A reused from {existing['arms']['A'].get('reused_from') or 'nothing'} -> "
                            f"{reuse_a or 'nothing'}")
        if existing.get("acceptance") != acceptance:
            problems.append("the acceptance thresholds changed (they are fixed before B runs)")
        if problems:
            raise ExperimentError(f"experiment {name!r} cannot continue: {'; '.join(problems)}. "
                                  "Start a new experiment under a new name.")
        return existing, spec_a, spec_b
    arms_meta = {}
    for arm, spec_name in (("A", "live"), ("B", variant)):
        arms_meta[arm] = {"spec": spec_name, "fingerprint": fingerprints[arm]}
        if resolved[arm]:
            arms_meta[arm].update(model=_summary(resolved[arm], "model"), effort=_summary(resolved[arm], "effort"),
                                  resolved=resolved[arm])
    if reuse_a:
        arms_meta["A"]["reused_from"] = reuse_a
    meta = {
        "name": name, "mode": mode, "variant": variant, "created": _now(),
        "hypothesis": spec_b.hypothesis, "rate": rate, "min_per_type": min_per_type,
        "acceptance": acceptance,
        # Which thresholds the variant set itself (v0.6.0): a model A/B gates
        # on cost only when the variant set `cost_cap` (`verdict`).
        "acceptance_set": sorted(spec_b.acceptance),
        "arms": arms_meta,
        "runs": [], "a_batch_ids": [],
    }
    if not dry_run:
        store.write_experiment(path, meta)
    return meta, spec_a, spec_b


def _save_meta(meta: dict) -> None:
    store.write_experiment(folder(meta["name"]) / EXPERIMENT_FILE, meta)


# -- sampling -------------------------------------------------------------------------


def _rank(name: str, atom: str, content_id: str) -> int:
    digest = hashlib.blake2b(f"{name}\x00{atom}\x00{content_id}".encode("utf-8"), digest_size=8)
    return int.from_bytes(digest.digest(), "big")


def draw(name: str, topics_by_atom: Mapping[str, Sequence], rate: float, min_per_type: int) -> Dict[str, list]:
    """The sample: per atom type, the top `k` subjects by a hash of
    (experiment, atom, content id), `k = max(min_per_type, ceil(rate * N))`
    capped at N. Deterministic and independent of list order (review R9).
    Small types are oversampled; the report weights for it."""
    chosen: Dict[str, list] = {}
    for atom, found in topics_by_atom.items():
        unique = {}
        for topic in found:
            unique.setdefault(store.atom_id(atom, topic.subject), topic)
        if not unique:
            continue
        k = min(len(unique), max(min_per_type, math.ceil(rate * len(unique))))
        ranked = sorted(unique.items(), key=lambda kv: _rank(name, atom, kv[0]))[:k]
        chosen[atom] = [topic for _, topic in ranked]
    return chosen


def load_sample(name: str) -> Dict[str, Dict[str, dict]]:
    return _read(folder(name) / SAMPLE_FILE, {})


def _record_sample(meta: dict, chosen: Mapping[str, Sequence], population: Mapping[str, int]) -> None:
    sample = load_sample(meta["name"])
    run = len(meta["runs"])
    for atom, found in chosen.items():
        entries = sample.setdefault(atom, {})
        for topic in found:
            cid = store.atom_id(atom, topic.subject)
            entries.setdefault(cid, {"subject": topic.subject, "context": topic.context, "run": run,
                                     "weight": population[atom] / max(1, len(found))})
    store.write_experiment(folder(meta["name"]) / SAMPLE_FILE, sample)


# -- submitting ---------------------------------------------------------------------------


def _result_path(name: str, arm: str, atom: str, cid: str) -> Path:
    return folder(name) / "results" / arm / atom / f"{cid}.json"


def _open_handles(name: str) -> List[Path]:
    directory = folder(name) / "_batches"
    return sorted(directory.glob("*.json")) if directory.is_dir() else []


def _in_flight(name: str, arm: str) -> set:
    """(atom, content id) already submitted for `arm` and not yet collected."""
    pending = set()
    for path in _open_handles(name):
        wrapper = _read(path) or {}
        if wrapper.get("arm") != arm:
            continue
        for route in (wrapper.get("handle") or {}).get("routing", {}).values():
            pending.add((route.get("atom_type"), route.get("content_id")))
    return pending


def to_submit(name: str, arm: str, chosen: Mapping[str, Sequence]) -> Dict[str, list]:
    """`chosen` less what `arm` already has, collected or in flight (review R12)."""
    pending = _in_flight(name, arm)
    left: Dict[str, list] = {}
    for atom, found in chosen.items():
        keep = [t for t in found
                if not _result_path(name, arm, atom, store.atom_id(atom, t.subject)).exists()
                and (atom, store.atom_id(atom, t.subject)) not in pending]
        if keep:
            left[atom] = keep
    return left


def build_arm_requests(name: str, arm: str, spec, atom: str, context: Optional[str],
                       subjects: Sequence[str], model: str, *, max_tokens: int,
                       effort: Optional[str] = None) -> list:
    from universe_builder.providers.requests import BatchItem, build_requests

    notes = topics.load_create_notes() if spec.create_notes else None
    index = topics.ListIndex(atom)
    items = [BatchItem(subject=s, user_instructions=spec.user_instructions(atom, s, context, notes, index))
             for s in subjects]
    requests = build_requests(atom, items, model, operation="create", context=context,
                              max_tokens=max_tokens, prompt=spec, effort=effort,
                              metadata={"experiment": name, "arm": arm, "context": context})
    # The content id rides in the routing so collect files each reply by it,
    # and the prompt's digest (v0.6.0), so a reused A can be shown to be
    # what today's A would send, notes, neighbours and budget included.
    for request in requests:
        request.metadata["content_id"] = store.atom_id(atom, request.metadata["subject"])
        request.metadata["prompt_sha256"] = prompt_digest(request)
    return requests


def prompt_digest(request) -> str:
    """sha256 of everything a request sends but the model and effort: system
    text, examples, user turn (the subject, its create note, its list
    neighbours, any candidates), schema and output budget. Two arms with one
    digest asked the same question (v0.6.0, `reuse_a`)."""
    blob = json.dumps({"system": request.system, "user": request.user,
                       "examples": [[e.user, e.assistant_json] for e in request.examples],
                       "schema": request.schema_model.__name__, "max_tokens": request.max_tokens},
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def submit_arm(name: str, arm: str, spec, atom: str, context: Optional[str], subjects: Sequence[str],
               model: str, *, max_tokens: int, effort: Optional[str] = None) -> List[Path]:
    """Submit one arm's requests for one (atom type, context) as batches,
    straight to the provider its model names (so A and B may go to two
    vendors); the handles are wrapped and kept here, with the arm's model
    and effort."""
    from universe_builder.providers.batch import stamp_effort
    from universe_builder.providers.effort import sent
    from universe_builder.providers.errors import BatchNotSupported
    from universe_builder.providers.registry import get_provider

    requests = build_arm_requests(name, arm, spec, atom, context, subjects, model, max_tokens=max_tokens,
                                  effort=effort)
    provider = get_provider(model)
    if not provider.supports_batch:
        raise BatchNotSupported(f"{provider.name} has no batch API; experiments run as batches")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    paths = []
    for i, chunk in enumerate(provider.chunk_for_batch(requests)):
        handle = provider.submit_batch(chunk)
        stamp_effort(handle, sent(model, effort))
        path = folder(name) / "_batches" / f"{arm}-{atom}-{stamp}-{len(paths):02d}-{i:02d}.json"
        while path.exists():
            path = path.with_name(path.stem + "-x.json")
        wrapper = {HANDLE_MARK: 1, "experiment": name, "arm": arm, "atom_type": atom, "context": context,
                   "model": model, "handle": handle.to_dict()}
        if sent(model, effort):
            # What went on the wire, not what was asked (none for a model
            # that takes no effort): collect copies it to each reply.
            wrapper["effort"] = sent(model, effort)
        store.write_experiment(path, wrapper)
        log.warning("experiment %s: arm %s, %d %s request(s) for %r -> %s",
                    name, arm, len(chunk), atom, context, path.name)
        paths.append(path)
    return paths


def submit(meta: dict, specs: Mapping[str, Any], chosen: Mapping[str, Sequence], models: Mapping[str, str],
           *, arms: Sequence[str], max_tokens: int, population: Mapping[str, int],
           a_batch_ids: Sequence[str] = (), models_b: Optional[Mapping[str, str]] = None,
           efforts: Optional[Mapping[str, Mapping[str, Optional[str]]]] = None) -> dict:
    """Record the sample and submit `arms` for it. Returns per-arm counts and
    failures; a group that fails is reported, the others still go.

    `models` is A's model per atom type; `models_b` B's (v0.6.0; None: A's).
    `efforts` is `{arm: {atom: effort}}`, as asked (None: none). Both arms'
    resolution per atom type, with the effort each *sends*
    (`arm_resolution`), is recorded in `meta.arms[arm]["resolved"]` whether
    or not that arm is submitted here (shadow's A is the real batch, a reused
    A was bought elsewhere), and a run that would resolve one differently (the
    pipeline's tier moved, say) is refused before anything is submitted."""
    name = meta["name"]
    by_arm = {"A": dict(models), "B": dict(models_b or models)}
    efforts = efforts or {}
    now = {arm: arm_resolution({a: by_arm[arm][a] for a in chosen}, efforts.get(arm)) for arm in ARMS}
    problems = []
    for arm in ARMS:
        recorded = meta["arms"][arm].get("resolved") or {}
        for atom in chosen:
            if atom in recorded and recorded[atom] != now[arm][atom]:
                problems.append(f"arm {arm} {atom}: {recorded[atom]} -> {now[arm][atom]}")
    if problems:
        raise ExperimentError(f"experiment {name!r} cannot pool: an arm's model or effort moved "
                              f"({'; '.join(problems)}). Start a new experiment under a new name.")
    for arm in ARMS:
        recorded = meta["arms"][arm].setdefault("resolved", {})
        recorded.update(now[arm])
        meta["arms"][arm]["model"] = _summary(recorded, "model")
        meta["arms"][arm]["effort"] = _summary(recorded, "effort")
    _record_sample(meta, chosen, population)
    outcome = {"submitted": {}, "failed": []}
    for arm in arms:
        left = to_submit(name, arm, chosen)
        count = 0
        for atom, found in left.items():
            for context, subjects in topics.by_context(found).items():
                try:
                    submit_arm(name, arm, specs[arm], atom, context, subjects, by_arm[arm][atom],
                               max_tokens=max_tokens, effort=(efforts.get(arm) or {}).get(atom))
                    count += len(subjects)
                except Exception as e:  # noqa: BLE001 - one group must not hide the others
                    log.error("experiment %s: arm %s %s (%r) failed to submit: %s", name, arm, atom, context, e)
                    outcome["failed"].append((arm, atom, context, str(e)))
        outcome["submitted"][arm] = count
    meta["runs"].append({"at": _now(), "submitted": outcome["submitted"],
                         "sampled": {a: len(v) for a, v in chosen.items()},
                         "a_batch_ids": list(a_batch_ids)})
    meta["a_batch_ids"] = sorted(set(meta.get("a_batch_ids", [])) | set(a_batch_ids))
    _save_meta(meta)
    return outcome


def estimate(chosen: Mapping[str, Sequence], models: Mapping[str, str], arms: int) -> float:
    """A rough generation cost at batch prices: a nominal 4k tokens in and 2k
    out per create call (an economy-tier create usually costs less than this)."""
    from universe_builder.providers import pricing

    total = 0.0
    for atom, found in chosen.items():
        info = pricing.lookup(models[atom])
        per = (4000 * info.input + 2000 * info.output) / 1e6 * info.batch_discount
        total += per * len(found) * arms
    return total


def reuse_a(meta: dict, source: str, atoms: Sequence[str], *, spec, max_tokens: int,
            write: bool = True) -> tuple:
    """`(chosen, missing)`: take `source`'s sample and its collected A replies
    for this experiment (v0.6.0, `--reuse-a`), so B is judged against an A
    that is byte for byte the one another B arm was judged against.

    The sample is `source`'s, weights and all (restricted to `atoms` when
    given); only subjects whose A reply `source` holds are chosen, so B is
    never bought for a pair that cannot complete. `missing` counts the
    sampled subjects with no A reply there yet (collect `source` first). Each
    copied reply keeps its model, usage and cost and gains `reused_from`, so
    the report counts its spend apart (it was paid in `source`). With
    `write=False` (the estimate) nothing is copied.

    **The reused A must be today's A.** Each reply's recorded
    `prompt_sha256` must equal the digest of the request today's A (`spec`,
    `max_tokens`) would send for the subject, so B cannot be given a create
    note, list neighbours or a budget A lacked; any mismatch, or a reply with
    no digest (pre-v0.6.0, or a shadow A copied from the corpus journal),
    refuses the reuse. Only atom types in `atoms` are taken; the others in
    the source sample are named in the error-free return as skipped."""
    name = meta["name"]
    src_sample = load_sample(source)
    if not src_sample:
        raise ExperimentError(f"experiment {source!r} has no sample to reuse")
    skipped = sorted(set(src_sample) - set(atoms))
    sample = load_sample(name)
    chosen: Dict[str, list] = {}
    records: Dict[tuple, dict] = {}
    missing = 0
    for atom, entries in sorted(src_sample.items()):
        if atom not in atoms:
            continue
        for cid, entry in sorted(entries.items()):
            record = _read(_result_path(source, "A", atom, cid))
            if record is None:
                missing += 1
                continue
            records[(atom, cid)] = record
            chosen.setdefault(atom, []).append(topics.Topic(entry["subject"], entry.get("context")))
    stale = []
    for atom, found in chosen.items():
        for context, subjects in topics.by_context(found).items():
            for request in build_arm_requests(name, "A", spec, atom, context, subjects, "probe",
                                              max_tokens=max_tokens):
                cid = request.metadata["content_id"]
                if records[(atom, cid)].get("prompt_sha256") != request.metadata["prompt_sha256"]:
                    stale.append(request.metadata["subject"])
    if stale:
        raise ExperimentError(
            f"{len(stale)} of {sum(len(v) for v in chosen.values())} A repl(ies) in {source!r} were not asked "
            f"what today's A would ask (another note, neighbour list or budget, or no recorded prompt: "
            f"{', '.join(sorted(stale)[:5])}{' ...' if len(stale) > 5 else ''}); generate A again")
    for atom, entries in sorted(src_sample.items()):
        if atom not in chosen:
            continue
        for cid, entry in sorted(entries.items()):
            record = records.get((atom, cid))
            if record is None or not write:
                continue
            sample.setdefault(atom, {}).setdefault(cid, dict(entry))
            dest = _result_path(name, "A", atom, cid)
            if not dest.exists():
                copy = dict(record, reused_from=source,
                            source=f"experiment {source}: {record.get('source') or 'A'}")
                store.write_experiment(dest, copy)
    if write and chosen:
        store.write_experiment(folder(name) / SAMPLE_FILE, sample)
    return chosen, missing, skipped


# -- collecting -------------------------------------------------------------------------------


def _usage_dict(usage) -> dict:
    return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_tokens, "cache_write_tokens": usage.cache_write_tokens}


def collect(name: str) -> dict:
    """Fetch every finished experiment batch into `results/`, then (shadow)
    copy A's replies for the sample out of the corpus journal, read-only."""
    from universe_builder.providers.pricing import estimate_cost
    from universe_builder.providers.registry import get_provider
    from universe_builder.providers.types import BatchHandle
    from universe_builder.providers.util import expected_custom_ids

    meta = load_meta(name)
    summary = {"written": 0, "failed": 0, "pending": 0, "missing": 0, "refused": [], "from_journal": 0}
    for path in _open_handles(name):
        wrapper = _read(path) or {}
        if not wrapper.get(HANDLE_MARK):
            log.warning("skipping %s: not an experiment handle", path.name)
            continue
        handle = BatchHandle.from_dict(wrapper["handle"])
        provider = get_provider(handle.model)
        status = provider.poll_batch(handle)
        wrapper["handle"] = handle.to_dict()
        store.write_experiment(path, wrapper)
        if not status.done:
            summary["pending"] += 1
            continue
        try:
            responses = list(provider.fetch_batch_results(handle))
        except Exception as e:  # noqa: BLE001 - one handle must not stop the sweep
            log.error("experiment %s: %s refused: %s", name, path.name, e)
            summary["refused"].append(f"{path.name}: {e}")
            continue
        seen = set()
        for response in responses:
            route = handle.routing.get(response.custom_id or "", {})
            if not route.get("content_id"):
                summary["failed"] += 1
                continue
            seen.add(response.custom_id)
            record = {
                "arm": wrapper["arm"], "atom_type": route["atom_type"], "subject": route["subject"],
                "context": route.get("context"), "content_id": route["content_id"],
                "model": response.model or handle.model, "batch_id": handle.batch_id,
                "effort": wrapper.get("effort") or handle.extra.get("effort"),
                "prompt_sha256": route.get("prompt_sha256"),
                "raw_text": response.raw_text or "", "error": response.error,
                "usage": _usage_dict(response.usage),
                "cost": estimate_cost(response.usage, handle.model, batch=True),
                "batch": True, "source": "experiment batch", "collected": _now(),
            }
            store.write_experiment(_result_path(name, wrapper["arm"], route["atom_type"], route["content_id"]),
                                   record)
            summary["written" if response.ok else "failed"] += 1
        summary["missing"] += len(set(expected_custom_ids(handle)) - seen)
        store.write_experiment(path.parent / "collected" / path.name, wrapper)
        path.unlink()
    if meta["mode"] == "shadow":
        summary["from_journal"] = pull_corpus_a(meta)
    return summary


def pull_corpus_a(meta: dict) -> int:
    """Copy A's create replies for the sample out of the corpus journal.

    Read-only: the journal already holds each batch reply's raw payload,
    recorded before anything wrote the atom (review R13), so nothing about
    A's submission or collection changes. Only attempts from the A batches
    this experiment rode with count; a later fix of the same atom does not."""
    name = meta["name"]
    batch_ids = set(meta.get("a_batch_ids") or [])
    if not batch_ids:
        return 0
    wanted = {}
    for atom, entries in load_sample(name).items():
        for cid, entry in entries.items():
            if not _result_path(name, "A", atom, cid).exists():
                wanted[(atom, journal.work_item_id(atom, entry["subject"]))] = (atom, cid, entry)
    copied = 0
    for line in journal.iter_entries():
        if not wanted:
            break
        if line.get("event_type") != journal.EVENT_ATTEMPT or line.get("batch_id") not in batch_ids:
            continue
        if line.get("operation") not in (None, "create"):
            continue
        key = (line.get("atom_type"), line.get("work_item_id"))
        if key not in wanted:
            continue
        atom, cid, entry = wanted.pop(key)
        raw = journal.raw_payload_of(line)
        store.write_experiment(_result_path(name, "A", atom, cid), {
            "arm": "A", "atom_type": atom, "subject": entry["subject"], "context": entry.get("context"),
            "content_id": cid, "model": line.get("returned_model") or line.get("requested_model"),
            "effort": line.get("sent_effort"),
            "batch_id": line.get("batch_id"), "raw_text": raw or "",
            "error": None if raw else (line.get("raw_payload_unavailable_reason") or "no payload recorded"),
            "usage": {k: line.get(k) or 0 for k in ("input_tokens", "output_tokens",
                                                    "cache_read_tokens", "cache_write_tokens")},
            "cost": line.get("estimated_cost"), "batch": True,
            "source": f"corpus journal {line.get('run_id')}/{line.get('attempt_id')}", "collected": _now(),
        })
        copied += 1
    return copied


# -- pairing and metrics --------------------------------------------------------------------------


def _schema(atom: str):
    from universe_builder import atoms

    return atoms.atom_mapping[atom].atom_class


def parse(raw: str, atom: str) -> tuple:
    """`(payload dict | None, error | None)`: the same validation for both arms,
    on the raw text each arm's reply came back as."""
    if not raw:
        return None, "no reply"
    try:
        _schema(atom).model_validate_json(raw)
        return json.loads(raw), None
    except Exception as e:  # noqa: BLE001
        return None, str(e).splitlines()[0][:200]


def fill(payload: Any) -> float:
    """Percent of leaf fields holding something: the completeness measure the
    phase-default model comparisons used."""
    filled = total = 0

    def walk(node):
        nonlocal filled, total
        if isinstance(node, dict):
            if not node:
                total += 1
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            if not node:
                total += 1
            for value in node:
                walk(value)
        else:
            total += 1
            if node not in (None, ""):
                filled += 1

    walk(payload)
    return 100.0 * filled / total if total else 0.0


def uncached_cost(record: Mapping[str, Any]) -> Optional[float]:
    """$ for a reply as if no input token had been read from cache, so B's
    smaller batch and its own prefix cannot make it look cheaper or dearer
    (review R8)."""
    from universe_builder.providers import pricing

    try:
        info = pricing.lookup(record.get("model") or "")
    except Exception:  # noqa: BLE001
        return record.get("cost")
    usage = record.get("usage") or {}
    tokens_in = sum(int(usage.get(k) or 0) for k in ("input_tokens", "cache_read_tokens", "cache_write_tokens"))
    total = (tokens_in * info.input + int(usage.get("output_tokens") or 0) * info.output) / 1e6
    return total * (info.batch_discount if record.get("batch") else 1.0)


def _checks(atom: str, subject: str, context: Optional[str], payload: dict) -> Dict[str, int]:
    from universe_builder.registry import active_pack

    hook = active_pack().experiment_checks
    if hook is None or payload is None:
        return {}
    return {k: int(v) for k, v in (hook(atom, subject, context, payload) or {}).items()}


def pairs(name: str) -> List[dict]:
    """One row per sampled subject: each arm's validity, fill, cost, tokens and
    check flags, and the pair's state — complete (both valid), failed (an arm
    invalid: a real loss, counted by the validity guardrail) or missing (an
    arm has no reply yet: excluded, reported) (review R16)."""
    rows = []
    for atom, entries in sorted(load_sample(name).items()):
        for cid, entry in sorted(entries.items()):
            row = {"atom_type": atom, "content_id": cid, "subject": entry["subject"],
                   "context": entry.get("context"), "weight": entry.get("weight", 1.0)}
            for arm in ARMS:
                record = _read(_result_path(name, arm, atom, cid))
                if record is None:
                    row[arm] = None
                    continue
                payload, error = parse(record.get("raw_text") or "", atom)
                row[arm] = {
                    "valid": payload is not None, "error": error,
                    "fill": fill(payload) if payload is not None else None,
                    "cost": uncached_cost(record),
                    "output_tokens": int((record.get("usage") or {}).get("output_tokens") or 0),
                    "flags": _checks(atom, entry["subject"], entry.get("context"), payload),
                    "payload": payload,
                }
            if row["A"] is None or row["B"] is None:
                row["state"] = "missing"
            elif row["A"]["valid"] and row["B"]["valid"]:
                row["state"] = "complete"
            else:
                row["state"] = "failed"
            rows.append(row)
    return rows


# -- the judge -----------------------------------------------------------------------------------------


def reference(name: str, atom: str, cid: str, subject: str, context: Optional[str]) -> Optional[str]:
    """The pack's reference for the *listed* subject, frozen on first use so a
    re-judge reads the same text. Never chosen from either arm's citations
    (review R10)."""
    from universe_builder.registry import active_pack

    path = folder(name) / "references" / atom / f"{cid}.json"
    frozen = _read(path)
    if frozen is not None:
        return frozen.get("text")
    hook = active_pack().experiment_reference
    if hook is None:
        raise ExperimentError("the active pack declares no experiment_reference; nothing to judge against")
    text = hook(atom, subject, context)
    store.write_experiment(path, {"subject": subject, "context": context, "text": text, "at": _now()})
    return text


def judge_system() -> str:
    from universe_builder.registry import active_pack

    rubric = (active_pack().experiment_rubric or "").strip()
    return f"{rubric}\n\n{JUDGE_INSTRUCTIONS}" if rubric else JUDGE_INSTRUCTIONS


def judge_user(subject: str, context: Optional[str], ref: str, first: dict, second: dict) -> str:
    # The reference first: both orders of a pair then share everything up to
    # the records, which is what a provider's prefix cache can reuse.
    listed = f"\nListed under: {context}" if context else ""
    return (f"Subject: {subject}{listed}\n\nREFERENCE:\n{ref}\n\n"
            f"RECORD 1:\n{json.dumps(first, ensure_ascii=False)}\n\n"
            f"RECORD 2:\n{json.dumps(second, ensure_ascii=False)}")


def load_judgements(name: str) -> List[dict]:
    path = folder(name) / JUDGEMENTS_FILE
    if not path.is_file():
        return []
    lines = []
    for text in path.read_text(encoding="utf-8").splitlines():
        try:
            lines.append(json.loads(text))
        except ValueError:
            log.warning("skipping an unreadable line in %s", path)
    return lines


def judge_label(line: Mapping[str, Any]) -> str:
    """Which judge wrote a judgement line: its model, or `model@effort` when it
    ran at a set effort (v0.6.0). Lines from before carry no effort and read
    as their model, as they always did."""
    from universe_builder.providers.effort import label

    return label(line.get("judge_model") or "?", line.get("judge_effort"))


def plan_judging(name: str, model: str, *, sample: Optional[int] = None, limit: Optional[int] = None,
                 seed: int = 0, effort: Optional[str] = None) -> dict:
    """The requests a `judge` would send: both orders of every complete pair
    not yet judged by this judge, with its reference. `sample` takes a seeded
    subset of pairs (the calibration run); `limit` caps pairs.

    A judge is a (model, effort) pair (v0.6.0): the same model at another
    effort is another judge, whose verdicts are stored beside the first's, not
    over them. On Anthropic the reply goes through the same structured-output
    path (`output_config.format`, `PairAssessment`) as every other call."""
    from universe_builder.providers.effort import label
    from universe_builder.providers.types import LLMRequest

    me = label(model, effort)
    done = {(j["content_id"], j["atom_type"], j["order"]) for j in load_judgements(name)
            if judge_label(j) == me and j.get("parsed") is not None}
    complete = [r for r in pairs(name) if r["state"] == "complete"]
    if sample is not None:
        rng = random.Random(seed)
        complete = sorted(complete, key=lambda r: (r["atom_type"], r["content_id"]))
        complete = rng.sample(complete, min(sample, len(complete)))
    if limit is not None:
        complete = complete[:limit]
    system = judge_system()
    requests, no_reference = [], []
    for row in complete:
        ref = reference(name, row["atom_type"], row["content_id"], row["subject"], row["context"])
        if not ref:
            no_reference.append(row["subject"])
            continue
        for order in ("AB", "BA"):
            if (row["content_id"], row["atom_type"], order) in done:
                continue
            first, second = (row["A"], row["B"]) if order == "AB" else (row["B"], row["A"])
            requests.append(LLMRequest(
                model=model, schema_model=PairAssessment, system=system,
                user=judge_user(row["subject"], row["context"], ref, first["payload"], second["payload"]),
                max_tokens=8192, custom_id=f"judge-{row['atom_type']}-{row['content_id']}-{order}",
                effort=effort,
                metadata={"atom_type": row["atom_type"], "content_id": row["content_id"],
                          "subject": row["subject"], "order": order}))
    return {"requests": requests, "pairs": len(complete), "no_reference": no_reference}


def judge_estimate(requests: Sequence, model: str, output_tokens: int = 3000) -> float:
    """Sync price of `requests`: characters / 4 in, `output_tokens` out each
    (a reasoning judge spends 2-4k; review R11)."""
    from universe_builder.providers import pricing

    info = pricing.lookup(model)
    total = 0.0
    for r in requests:
        tokens_in = (len(r.system) + len(r.user)) / 4
        total += (tokens_in * info.input + output_tokens * info.output) / 1e6
    return total


def judge(name: str, requests: Sequence, *, concurrency: int = 4) -> dict:
    """Send `requests` (from `plan_judging`) and append each verdict, with its
    raw text and cost, to this experiment's `judgements.jsonl`."""
    from universe_builder.providers import run as provider_run
    from universe_builder.providers.pricing import estimate_cost

    by_id = {r.custom_id: r for r in requests}
    path = folder(name) / JUDGEMENTS_FILE
    tally = {"ok": 0, "failed": 0, "cost": 0.0}

    def record(response) -> None:
        request = by_id.get(response.custom_id)
        if request is None:
            log.error("a judge reply with an unknown id %r", response.custom_id)
            return
        # Sync prices for the judge's own model: judging runs synchronously.
        cost = estimate_cost(response.usage, request.model) or 0.0
        tally["cost"] += cost
        tally["ok" if response.ok else "failed"] += 1
        store.append_experiment_line(path, {
            **request.metadata, "judge_model": request.model,
            **({"judge_effort": request.effort} if request.effort else {}),
            "parsed": response.parsed.model_dump() if response.ok else None,
            "raw_text": response.raw_text, "error": response.error,
            "usage": _usage_dict(response.usage), "cost": cost, "at": _now(),
        })

    provider_run.run_many(list(requests), concurrency=concurrency, on_result=record)
    return tally


#: A pair the judge says neither record is about: the reference is the wrong
#: article, not two wrong atoms. Excluded from the tally and counted.
REFERENCE_MISS = "reference miss"


def _reference_miss(parsed: Mapping[str, Any]) -> bool:
    return bool(parsed["record_1"].get("wrong_subject")) and bool(parsed["record_2"].get("wrong_subject"))


def pair_outcomes(name: str, judge_model: str) -> Dict[tuple, str]:
    """(atom, content id) -> "A", "B", "tie" or `REFERENCE_MISS` under
    `judge_model`. A side wins a pair only by winning both orders;
    disagreement is a tie. A pair judged in one order only is left out.

    When either order says *both* records are the wrong subject, the
    reference is the wrong article (the first live run, v0.5.1: "Mendes
    Stela" read against the city of Mendes, a Korean novel against a 2015
    film). Two independently generated records being wrong the same way is
    far less likely than one wrong reference, and scoring the pair a tie
    would only dilute the count, so it is set aside and counted instead."""
    orders: Dict[tuple, Dict[str, Optional[str]]] = {}
    for line in load_judgements(name):
        if judge_label(line) != judge_model or line.get("parsed") is None:
            continue
        if _reference_miss(line["parsed"]):
            orders.setdefault((line["atom_type"], line["content_id"]), {})[line["order"]] = REFERENCE_MISS
            continue
        winner = order_winner(line["parsed"])
        arm = None
        if winner is not None:
            first, second = (("A", "B") if line["order"] == "AB" else ("B", "A"))
            arm = first if winner == 1 else second
        orders.setdefault((line["atom_type"], line["content_id"]), {})[line["order"]] = arm
    outcomes = {}
    for key, both in orders.items():
        if "AB" not in both or "BA" not in both:
            continue
        if REFERENCE_MISS in (both["AB"], both["BA"]):
            outcomes[key] = REFERENCE_MISS
            continue
        outcomes[key] = both["AB"] if both["AB"] is not None and both["AB"] == both["BA"] else "tie"
    return outcomes


def judge_models(name: str) -> Dict[str, int]:
    """Parsed judgements per judge (`judge_label`: model, or model@effort)."""
    counts: Dict[str, int] = {}
    for line in load_judgements(name):
        if line.get("parsed") is not None:
            counts[judge_label(line)] = counts.get(judge_label(line), 0) + 1
    return counts


# -- the verdict ------------------------------------------------------------------------------------------


def _guardrails(rows: Sequence[dict], acc: Mapping[str, Any]) -> dict:
    scored = [r for r in rows if r["state"] in ("complete", "failed")]
    n = len(scored)
    b_only = sum(1 for r in scored if r["A"]["valid"] and not r["B"]["valid"])
    a_only = sum(1 for r in scored if r["B"]["valid"] and not r["A"]["valid"])
    validity_margin = max(acc["validity_margin_abs"], acc["validity_margin_rel"] * n)
    out = {"n": n, "validity": {
        "a_only_failures": a_only, "b_only_failures": b_only, "margin": validity_margin,
        "passes": (b_only - a_only) <= validity_margin}}

    complete = [r for r in rows if r["state"] == "complete"]
    seed, perms = acc["seed"], acc["permutations"]

    fill_diffs = [r["B"]["fill"] - r["A"]["fill"] for r in complete]
    fill_mean = stats.mean(fill_diffs)
    fill_p = stats.sign_flip_p(fill_diffs, greater=False, permutations=perms, seed=seed)
    out["fill"] = {"mean_a": stats.mean(r["A"]["fill"] for r in complete),
                   "mean_b": stats.mean(r["B"]["fill"] for r in complete),
                   "mean_diff": fill_mean, "p_lower": fill_p, "margin": acc["fill_margin"],
                   "passes": not (fill_mean is not None and fill_mean < -acc["fill_margin"] and fill_p < 0.05)}

    from universe_builder.registry import active_pack

    # Report-only checks (v0.5.8) are counted in `by_check` but not in the guardrail.
    report_only = set(getattr(active_pack(), "experiment_report_only_checks", ()) or ())
    flags_a = [sum(v for k, v in r["A"]["flags"].items() if k not in report_only) for r in complete]
    flags_b = [sum(v for k, v in r["B"]["flags"].items() if k not in report_only) for r in complete]
    flag_diffs = [b - a for a, b in zip(flags_a, flags_b)]
    mean_a = stats.mean(flags_a) or 0.0
    flag_margin = max(acc["flag_margin_abs"], acc["flag_margin_rel"] * mean_a)
    flag_mean = stats.mean(flag_diffs)
    flag_p = stats.sign_flip_p(flag_diffs, greater=True, permutations=perms, seed=seed)
    per_check: Dict[str, list] = {}
    for r in complete:
        for check in set(r["A"]["flags"]) | set(r["B"]["flags"]):
            per_check.setdefault(check, [0, 0])
            per_check[check][0] += r["A"]["flags"].get(check, 0)
            per_check[check][1] += r["B"]["flags"].get(check, 0)
    out["flags"] = {"mean_a": mean_a, "mean_b": stats.mean(flags_b), "mean_diff": flag_mean,
                    "p_higher": flag_p, "margin": flag_margin,
                    "by_check": {k: {"A": v[0], "B": v[1]} for k, v in sorted(per_check.items())},
                    "report_only": sorted(report_only & set(per_check)),
                    "passes": not (flag_mean is not None and flag_mean > flag_margin and flag_p < 0.05)}

    costs = [(r["A"]["cost"], r["B"]["cost"]) for r in complete
             if r["A"]["cost"] is not None and r["B"]["cost"] is not None]
    cost_a = stats.mean(a for a, _ in costs)
    cost_b = stats.mean(b for _, b in costs)
    ratio = (cost_b / cost_a) if cost_a else None
    out["cost"] = {"mean_a": cost_a, "mean_b": cost_b, "ratio": ratio, "cap": acc["cost_cap"],
                   "passes": ratio is None or ratio <= acc["cost_cap"]}
    out["output_tokens"] = {"mean_a": stats.mean(r["A"]["output_tokens"] for r in complete),
                            "mean_b": stats.mean(r["B"]["output_tokens"] for r in complete)}
    return out


def _judged(rows: Sequence[dict], outcomes: Mapping[tuple, str], acc: Mapping[str, Any]) -> dict:
    by_key = {(r["atom_type"], r["content_id"]): r for r in rows if r["state"] == "complete"}
    misses = sum(1 for k, v in outcomes.items() if k in by_key and v == REFERENCE_MISS)
    results = [(by_key[k], v) for k, v in outcomes.items() if k in by_key and v != REFERENCE_MISS]
    w_b = sum(1 for _, v in results if v == "B")
    w_a = sum(1 for _, v in results if v == "A")
    n = len(results)
    t = n - w_a - w_b
    d = w_a + w_b
    out = {"n": n, "w_a": w_a, "w_b": w_b, "ties": t, "decisive": d, "reference_misses": misses,
           "p_better": stats.binom_sf(w_b, d) if d else None,
           "p_worse": stats.binom_sf(w_a, d) if d else None,
           "win_rate_b": (w_b / d) if d else None,
           "wilson95": stats.wilson(w_b, d) if d else None,
           "score_b": ((w_b + t / 2) / n) if n else None,
           "score_lower95": (stats.wilson(w_b + t / 2, n, z=stats.Z_ONE_SIDED_95)[0] if n else None)}
    # Inverse-probability weighted: each pair counts for the subjects it stands for.
    wb = sum(r["weight"] for r, v in results if v == "B")
    wa = sum(r["weight"] for r, v in results if v == "A")
    out["weighted_win_rate_b"] = (wb / (wa + wb)) if (wa + wb) else None
    per_type, eligible = {}, []
    for atom in sorted({r["atom_type"] for r, _ in results}):
        mine = [v for r, v in results if r["atom_type"] == atom]
        a, b = mine.count("A"), mine.count("B")
        per_type[atom] = {"w_a": a, "w_b": b, "ties": mine.count("tie")}
        if a + b >= acc["per_type_min_decisive"]:
            eligible.append(atom)
    if eligible:
        adjusted = stats.holm([stats.binom_sf(per_type[a]["w_a"], per_type[a]["w_a"] + per_type[a]["w_b"])
                               for a in eligible])
        for atom, p in zip(eligible, adjusted):
            per_type[atom]["p_a_better_holm"] = p
            per_type[atom]["flag"] = p < 0.05
    out["per_type"] = per_type
    return out


def criteria(name: str, judge_model: str, rows: Sequence[dict], acc: Mapping[str, Any]) -> dict:
    """Per-arm totals of the judge's lists (v0.5.7): for every complete pair
    judged in both orders by `judge_model`, each arm's count of contradictions,
    of false-certainty claims and its coverage, averaged over the two orders;
    then the paired B - A difference with a seeded sign-flip test either way.
    `wrong_subject` counts orders. Reference misses are left out, as in
    `pair_outcomes`. The tests that decide on what B claims, not on who wins,
    read these."""
    complete = {(r["atom_type"], r["content_id"]) for r in rows if r["state"] == "complete"}
    per: Dict[tuple, Dict[str, Dict[str, list]]] = {}
    wrong = {"A": 0, "B": 0}
    for line in load_judgements(name):
        parsed = line.get("parsed")
        key = (line.get("atom_type"), line.get("content_id"))
        if judge_label(line) != judge_model or parsed is None or key not in complete:
            continue
        if _reference_miss(parsed):
            per[key] = {"miss": {}}
            continue
        if "miss" in per.get(key, {}):
            continue
        arms = ("A", "B") if line["order"] == "AB" else ("B", "A")
        for record, arm in zip(("record_1", "record_2"), arms):
            r = parsed[record]
            slot = per.setdefault(key, {}).setdefault(arm, {"contradictions": [], "false_certainty": [],
                                                            "coverage": []})
            slot["contradictions"].append(len(r.get("contradictions") or []))
            slot["false_certainty"].append(len(r.get("false_certainty") or []))
            slot["coverage"].append(max(0, min(3, int(r.get("coverage") or 0))))
            wrong[arm] += int(bool(r.get("wrong_subject")))
    out: Dict[str, Any] = {"wrong_subject": wrong}
    pairs_used = [v for v in per.values() if "A" in v and "B" in v
                  and len(v["A"]["coverage"]) == 2 and len(v["B"]["coverage"]) == 2]
    out["pairs"] = len(pairs_used)
    for c in ("contradictions", "false_certainty", "coverage"):
        a = [stats.mean(v["A"][c]) for v in pairs_used]
        b = [stats.mean(v["B"][c]) for v in pairs_used]
        diffs = [y - x for x, y in zip(a, b)]
        out[c] = {"mean_a": stats.mean(a), "mean_b": stats.mean(b), "mean_diff": stats.mean(diffs),
                  "p_lower": stats.sign_flip_p(diffs, greater=False, permutations=acc["permutations"],
                                               seed=acc["seed"]),
                  "p_higher": stats.sign_flip_p(diffs, greater=True, permutations=acc["permutations"],
                                                seed=acc["seed"])}
    return out


#: A model A/B's verdicts (v0.6.0): on quality only, the cost ratio beside.
B_BETTER, A_BETTER = "B BETTER", "A BETTER"


def verdict(guard: Mapping[str, Any], judged: Optional[Mapping[str, Any]], acc: Mapping[str, Any],
            *, model_ab: bool = False, cost_gates: bool = True) -> tuple:
    """`(verdict, reasons)`. REJECT at any look on harm; ADOPT only at the
    pre-registered target; otherwise CONTINUE, or at target NO DIFFERENCE /
    ADOPT IF CHEAPER (review R4). Never applied: a person promotes a variant.

    `model_ab` (v0.6.0: the arms run different models or efforts) answers the
    quality question only: B BETTER, A BETTER (B significantly worse), NO
    DIFFERENCE at target, CONTINUE below it, REJECT on a failed validity,
    fill or flags guardrail. A dearer model is the point of asking, so the
    cost ratio is stated beside the verdict (`report`'s `cost_ratio`) and
    gates nothing unless `cost_gates` (the variant set `cost_cap` itself).
    Without `model_ab`, a prompt experiment, this is exactly as before."""
    gating = ("validity", "fill", "flags", "cost") if cost_gates else ("validity", "fill", "flags")
    failing = [k for k in gating if not guard[k]["passes"]]
    reasons = [f"guardrail failed: {k}" for k in failing]
    worse = judged and judged["p_worse"] is not None and judged["p_worse"] < acc["harm_alpha"]
    if worse:
        reasons.append(f"B significantly worse: P = {judged['p_worse']:.4f} < {acc['harm_alpha']}")
    if model_ab:
        if failing:
            return "REJECT", reasons
        if worse:
            return A_BETTER, reasons
    if reasons:
        return "REJECT", reasons
    if not judged or not judged["n"]:
        return "CONTINUE", ["not judged yet"]
    d, target = judged["decisive"], acc["target_decisive"]
    if d < target:
        return "CONTINUE", [f"{d} of {target} decisive pairs; pool more (same prompts) before deciding"]
    if judged["p_better"] < acc["alpha"]:
        return (B_BETTER if model_ab else "ADOPT"), [
            f"B better: {judged['w_b']}-{judged['w_a']} over {d} decisive pairs, "
            f"P = {judged['p_better']:.4f} < {acc['alpha']}"]
    if model_ab:
        return "NO DIFFERENCE", [f"at target ({d} decisive pairs) B is not detectably better "
                                 f"(P = {judged['p_better']:.4f}) nor worse (P = {judged['p_worse']:.4f})"]
    ratio = guard["cost"]["ratio"]
    if (ratio is not None and ratio <= 1 - acc["cheaper_by"]
            and judged["score_lower95"] is not None and judged["score_lower95"] >= acc["noninferior_bound"]):
        return "ADOPT IF CHEAPER", [f"no detectable difference, B non-inferior (score lower bound "
                                    f"{judged['score_lower95']:.3f}) and {1 - ratio:.0%} cheaper"]
    return "NO DIFFERENCE", [f"at target ({d} decisive pairs) B is not detectably better "
                             f"(P = {judged['p_better']:.4f})"]


def agreement(name: str, primary: str, reference_model: str) -> Optional[dict]:
    """How often `primary` agrees with `reference_model` on the pairs the
    reference finds decisive (the calibration: >= 0.8 keeps the cheap judge)."""
    ours, theirs = pair_outcomes(name, primary), pair_outcomes(name, reference_model)
    # Only the reference judge's own misses leave the denominator. A pair it
    # found decisive that `primary` called a miss is a disagreement, and
    # dropping it would flatter the cheap judge.
    decisive = [k for k, v in theirs.items() if v not in ("tie", REFERENCE_MISS) and k in ours]
    if not decisive:
        return None
    same = sum(1 for k in decisive if ours[k] == theirs[k])
    return {"against": reference_model, "pairs": len(decisive), "agree": same, "rate": same / len(decisive)}


def panel(name: str, judges: Sequence[str], rows: Sequence[dict], acc: Mapping[str, Any]) -> dict:
    """Two or more judges on one experiment (v0.6.0, `report --panel`).

    Each judge's own tally (`_judged` over its `pair_outcomes`), and an
    agreement-only one: a pair counts only when every judge gives it the same
    outcome, and each judge's outcome already needs both orders to agree, so
    an agreed win was won in all 2 x n judgements. Pairs judged by every
    judge in both orders are the denominator of the agreement rate; a pair
    one judge calls a reference miss and another does not is a disagreement.
    The verdict is drawn on the agreed pairs only (`report` adds it)."""
    outcomes = {j: pair_outcomes(name, j) for j in judges}
    complete = {(r["atom_type"], r["content_id"]) for r in rows if r["state"] == "complete"}
    common = sorted(k for k in complete if all(k in o for o in outcomes.values()))
    agreed: Dict[tuple, str] = {}
    for key in common:
        values = {o[key] for o in outcomes.values()}
        if len(values) == 1:
            agreed[key] = values.pop()
    decisive = [k for k in common if any(o[k] in ARMS for o in outcomes.values())]
    decisive_agree = [k for k in decisive if k in agreed]
    return {
        "judges": list(judges),
        "per_judge": {j: _judged(rows, outcomes[j], acc) for j in judges},
        "pairs_judged_by_all": len(common), "agree": len(agreed), "disagree": len(common) - len(agreed),
        "agreement_rate": (len(agreed) / len(common)) if common else None,
        "decisive_pairs": len(decisive), "decisive_agree": len(decisive_agree),
        "decisive_agreement_rate": (len(decisive_agree) / len(decisive)) if decisive else None,
        "agreed": _judged(rows, agreed, acc),
    }


def _arm_summary(name: str, meta: Mapping[str, Any]) -> dict:
    """What each arm ran as (v0.6.0): its prompt, overrides, and the models
    and efforts its stored replies actually came back from."""
    out = {}
    for arm in ARMS:
        recorded = meta["arms"][arm]
        models, efforts = set(), set()
        for atom, entries in load_sample(name).items():
            for cid in entries:
                record = _read(_result_path(name, arm, atom, cid))
                if record:
                    models.add(record.get("model") or "?")
                    efforts.add(record.get("effort") or "default")
        out[arm] = {"spec": recorded.get("spec"), "model": recorded.get("model"),
                    "effort": recorded.get("effort"), "reused_from": recorded.get("reused_from"),
                    "models": sorted(models), "efforts": sorted(efforts)}
    return out


def report(name: str, judge_model: Optional[str] = None, *, write: bool = True,
           panel_judges: Optional[Sequence[str]] = None) -> dict:
    meta = load_meta(name)
    acc = {**DEFAULT_ACCEPTANCE, **(meta.get("acceptance") or {})}
    rows = pairs(name)
    states = {s: sum(1 for r in rows if r["state"] == s) for s in ("complete", "failed", "missing")}
    guard = _guardrails(rows, acc)
    # A model or effort A/B (v0.6.0): quality decides, cost is reported beside
    # it and gates only through a `cost_cap` the variant set itself.
    model_ab = arms_differ(meta["arms"])
    cost_gates = not model_ab or "cost_cap" in (meta.get("acceptance_set") or [])
    guard["cost"]["gating"] = cost_gates
    if panel_judges:
        if len(set(panel_judges)) != len(panel_judges):
            raise ExperimentError(f"--panel names a judge twice: {', '.join(panel_judges)}")
        known = judge_models(name)
        absent = [j for j in panel_judges if not known.get(j)]
        if absent:
            raise ExperimentError(f"no judgements on {name!r} by {', '.join(absent)}; judges here: "
                                  f"{', '.join(sorted(known)) or 'none'} (MODEL or MODEL@EFFORT)")
    models = judge_models(name)
    if judge_model is None and models:
        judge_model = max(models, key=lambda m: models[m])
    judged = _judged(rows, pair_outcomes(name, judge_model), acc) if judge_model else None
    if judged is not None:
        judged["criteria"] = criteria(name, judge_model, rows, acc)
        # Complete pairs never judged because the reference resolved to nothing
        # (v0.5.8): the report's own measure of how much the judge could see.
        judged["no_reference"] = sum(
            1 for r in rows if r["state"] == "complete"
            and (_read(folder(name) / "references" / r["atom_type"] / f"{r['content_id']}.json") or {}).get("text")
            is None and (folder(name) / "references" / r["atom_type"] / f"{r['content_id']}.json").exists())
    calls, label = verdict(guard, judged, acc, model_ab=model_ab, cost_gates=cost_gates)
    # Cost per arm and per judge (v0.6.0). Each reply's cost was priced at
    # collect from its own model's row at batch rates, each judgement from the
    # judge's row at sync rates. An A copied by `--reuse-a` was paid for in
    # the experiment it came from and is counted apart.
    spend = {"A": 0.0, "B": 0.0, "A_reused": 0.0, "judge": 0.0, "judges": {}}
    for atom, entries in load_sample(name).items():
        for cid in entries:
            for arm in ARMS:
                record = _read(_result_path(name, arm, atom, cid))
                if not record or not record.get("cost"):
                    continue
                if arm == "A" and record.get("reused_from"):
                    spend["A_reused"] += record["cost"]
                elif arm == "B" or meta["mode"] == "replay":
                    spend[arm] += record["cost"]
    for line in load_judgements(name):
        spend["judges"][judge_label(line)] = spend["judges"].get(judge_label(line), 0.0) + (line.get("cost") or 0.0)
    spend["judge"] = sum(spend["judges"].values())
    panel_out = None
    if panel_judges:
        panel_out = panel(name, panel_judges, rows, acc)
        panel_out["verdict"], panel_out["reasons"] = verdict(guard, panel_out["agreed"], acc,
                                                             model_ab=model_ab, cost_gates=cost_gates)
    out = {
        "experiment": name, "mode": meta["mode"], "variant": meta["variant"],
        "hypothesis": meta.get("hypothesis"), "fingerprints": {a: meta["arms"][a]["fingerprint"] for a in ARMS},
        "generated": _now(), "pairs": states, "sampled_by_type": {a: len(e) for a, e in load_sample(name).items()},
        "guardrails": guard, "judge_model": judge_model, "judged": judged,
        "calibration": [agreement(name, judge_model, other) for other in models if other != judge_model]
        if judge_model else [],
        "verdict": calls, "reasons": label, "acceptance": acc, "spend": spend,
        "arms": _arm_summary(name, meta), "panel": panel_out,
        "model_ab": model_ab, "cost_ratio": guard["cost"]["ratio"],
        "failed_pairs": [{"atom_type": r["atom_type"], "subject": r["subject"],
                          "A": (r["A"] or {}).get("error"), "B": (r["B"] or {}).get("error")}
                         for r in rows if r["state"] == "failed"],
    }
    if meta["variant"] == "aa" and not arms_differ(meta["arms"]):
        out["aa_check"] = ("PASS" if calls in ("CONTINUE", "NO DIFFERENCE") else
                           "FAIL: an A/A run must never ADOPT or REJECT; the harness or the judge is biased")
    if write:
        store.write_experiment(folder(name) / "report.json", out)
        store.write_experiment(folder(name) / "report.md", render(out))
    return out


def _with_cost(r: Mapping[str, Any], label: str) -> str:
    """A model A/B's verdict with its cost ratio beside it: `B BETTER, 3.0x cost`."""
    ratio = r.get("cost_ratio")
    if r.get("model_ab") and ratio is not None:
        return f"{label}, {ratio:.1f}x cost"
    return label


def _f(value, spec=".3f") -> str:
    return "—" if value is None else format(value, spec)


def render(r: Mapping[str, Any]) -> str:
    g, j = r["guardrails"], r["judged"]
    lines = [f"# Experiment {r['experiment']} ({r['mode']}, variant {r['variant']})", ""]
    if r.get("hypothesis"):
        lines += [f"Hypothesis: {r['hypothesis']}", ""]
    lines += [f"**Verdict: {_with_cost(r, r['verdict'])}**  ", *[f"- {x}" for x in r["reasons"]], ""]
    if r.get("model_ab"):
        gate = ("gates through the variant's cost_cap" if r["guardrails"]["cost"].get("gating")
                else "is reported, not a gate")
        lines += [f"Model or effort A/B: the verdict is on quality; the cost ratio {gate}.", ""]
    if "aa_check" in r:
        lines += [f"A/A check: {r['aa_check']}", ""]
    for arm, a in (r.get("arms") or {}).items():
        ran = ", ".join(a.get("models") or []) or "no replies yet"
        efforts = ", ".join(a.get("efforts") or []) or "—"
        extra = f"; copied from experiment {a['reused_from']}" if a.get("reused_from") else ""
        lines.append(f"Arm {arm}: prompt {a.get('spec')}, model {ran}, effort {efforts}{extra}  ")
    if r.get("arms"):
        lines.append("")
    p = r["pairs"]
    lines += [f"Pairs: {p['complete']} complete, {p['failed']} failed, {p['missing']} missing "
              f"(sampled per type: {', '.join(f'{k} {v}' for k, v in sorted(r['sampled_by_type'].items()))})", ""]
    lines += ["| guardrail | A | B | test | passes |", "|---|---|---|---|---|",
              f"| validity (failures only in that arm) | {g['validity']['a_only_failures']} | "
              f"{g['validity']['b_only_failures']} | margin {_f(g['validity']['margin'], '.1f')} | "
              f"{g['validity']['passes']} |",
              f"| fill % | {_f(g['fill']['mean_a'], '.1f')} | {_f(g['fill']['mean_b'], '.1f')} | "
              f"P(lower) {_f(g['fill']['p_lower'])} | {g['fill']['passes']} |",
              f"| check flags / atom | {_f(g['flags']['mean_a'])} | {_f(g['flags']['mean_b'])} | "
              f"P(higher) {_f(g['flags']['p_higher'])}, margin {_f(g['flags']['margin'])} | "
              f"{g['flags']['passes']} |",
              f"| $ / atom (uncached-equivalent) | {_f(g['cost']['mean_a'], '.5f')} | "
              f"{_f(g['cost']['mean_b'], '.5f')} | ratio {_f(g['cost']['ratio'], '.2f')} ≤ "
              f"{g['cost']['cap']} | {g['cost']['passes'] if g['cost'].get('gating', True) else 'reported only'} |",
              f"| output tokens / atom | {_f(g['output_tokens']['mean_a'], '.0f')} | "
              f"{_f(g['output_tokens']['mean_b'], '.0f')} | | |", ""]
    if g["flags"]["by_check"]:
        lines += ["Flags by check: " + ", ".join(f"{k} A {v['A']} / B {v['B']}"
                                                 for k, v in g["flags"]["by_check"].items()), ""]
    if j:
        lines += [f"Judge {r['judge_model']}: B {j['w_b']} – A {j['w_a']}, {j['ties']} ties, "
                  f"{j.get('reference_misses', 0)} reference misses set aside, "
                  f"{j.get('no_reference', 0)} pairs with no reference "
                  f"({j['decisive']} decisive of {r['acceptance']['target_decisive']} needed). "
                  f"B's win rate {_f(j['win_rate_b'])} (Wilson 95% {_f((j['wilson95'] or [None])[0])}–"
                  f"{_f((j['wilson95'] or [None, None])[1])}; weighted {_f(j['weighted_win_rate_b'])}); "
                  f"P(better) {_f(j['p_better'], '.4f')}, P(worse) {_f(j['p_worse'], '.4f')}.", ""]
        c = j.get("criteria") or {}
        if c.get("pairs"):
            lines += ["Per record, mean over both orders (A / B; P that B is lower / higher): " + "; ".join(
                f"{k.replace('_', ' ')} {_f(c[k]['mean_a'], '.2f')} / {_f(c[k]['mean_b'], '.2f')} "
                f"(P {_f(c[k]['p_lower'])} / {_f(c[k]['p_higher'])})"
                for k in ("contradictions", "false_certainty", "coverage"))
                + f"; wrong subject A {c['wrong_subject']['A']} / B {c['wrong_subject']['B']} orders "
                  f"({c['pairs']} pairs).", ""]
        flagged = [a for a, v in j["per_type"].items() if v.get("flag")]
        lines += ["Per type: " + ", ".join(f"{a} B {v['w_b']}–A {v['w_a']}" for a, v in j["per_type"].items())
                  + (f". **A better on {', '.join(flagged)} (Holm)**" if flagged else ""), ""]
    for c in r.get("calibration") or []:
        if c:
            lines += [f"Calibration against {c['against']}: agrees on {c['agree']} of {c['pairs']} "
                      f"decisive pairs ({c['rate']:.0%}; keep the cheap judge at ≥ 80%).", ""]
    pnl = r.get("panel")
    if pnl:
        lines += [f"Panel of {len(pnl['judges'])} judges: " + "; ".join(
            f"{jm} B {t['w_b']} – A {t['w_a']}, {t['ties']} ties (win rate B {_f(t['win_rate_b'])}, "
            f"P(better) {_f(t['p_better'], '.4f')})" for jm, t in pnl["per_judge"].items()) + ".", "",
            f"Agreement: {pnl['agree']} of {pnl['pairs_judged_by_all']} pairs judged by every judge "
            f"({_f(pnl['agreement_rate'], '.0%')}); on pairs any "
            f"judge found decisive, {pnl['decisive_agree']} of {pnl['decisive_pairs']}"
            f" ({_f(pnl['decisive_agreement_rate'], '.0%')}).",
            "",
            f"**Panel verdict (agreed pairs only): {_with_cost(r, pnl['verdict'])}**: B {pnl['agreed']['w_b']} – "
            f"A {pnl['agreed']['w_a']}, {pnl['agreed']['ties']} ties, {pnl['agreed']['decisive']} decisive; "
            f"P(better) {_f(pnl['agreed']['p_better'], '.4f')}, P(worse) {_f(pnl['agreed']['p_worse'], '.4f')}.  ",
            *[f"- {x}" for x in pnl["reasons"]], ""]
    s = r["spend"]
    lines += [f"Spend: B ${s['B']:.4f}" + (f", A ${s['A']:.4f}" if r["mode"] == "replay" else "")
              + (f", A reused (paid in its own experiment) ${s['A_reused']:.4f}" if s.get("A_reused") else "")
              + f", judge ${s['judge']:.4f}"
              + (" (" + ", ".join(f"{k} ${v:.4f}" for k, v in sorted((s.get("judges") or {}).items())) + ")"
                 if s.get("judges") else "") + ".", ""]
    if r["failed_pairs"]:
        lines += ["Failed pairs:", *[f"- {x['atom_type']} {x['subject']}: A {x['A'] or 'ok'}; B {x['B'] or 'ok'}"
                                     for x in r["failed_pairs"][:20]], ""]
    return "\n".join(lines)


def replay_topics(atoms: Iterable[str], context: Optional[str] = None,
                  only: Optional[set] = None) -> Dict[str, list]:
    """Every listed subject that already has a real atom, per atom type;
    with `only`, just those whose `kind:hID` it names."""
    found = {}
    for atom in atoms:
        listed = [t for t in topics.from_history(atom) if topics.has_real_atom(atom, t.subject)]
        if only is not None:
            listed = [t for t in listed if f"{atom}:{store.atom_id(atom, t.subject)}" in only]
        if context:
            listed = [t for t in listed if t.context and topics.name_key(t.context) == topics.name_key(context)]
        if listed:
            found[atom] = listed
    return found
