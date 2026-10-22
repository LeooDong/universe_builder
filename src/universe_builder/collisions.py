"""Name collisions between an incoming index seed and the stored corpus.

A new civilisation's seed names several hundred subjects, and some of them are
already here under a different spelling. The atom store partitions by *content
id*, which hashes the topic string — so `Massalia (Marseille)` and
`Massilia (Marseille)` are one city with two ids and two files. Overlap is free
(a topic that re-pins onto an existing atom costs nothing);
**re-spelling is not**, and `aliases.txt` is what prevents it, by declaring the
identity before `topics.needing_create` builds the work list.

The seed is the only thing that can tell you which collisions exist, so this
runs after `--phase history` and before `--phase create`. A checklist that puts
alias lines first reads as "before the seed"; nothing in the
generation path reads `aliases.txt` — only `topics` does, and only to group the
create work — so the real constraint is *before create*, and doing it after the
seed means judging names you can see rather than names you guessed.

**Detection is mechanical and adjudication is not, because they fail in
opposite directions.** Recall is what detection needs: miss a pair and a
duplicate ships silently. Precision is what a verdict needs: assert wrongly and
two subjects merge into one, which is worse than the duplicate. So candidates
are generated generously — declared claims, token containment, character
similarity — and then judged, either by a human reading the review file or, with
`--via model`, by a frontier call.

On a real seed, most similarity candidates were **false** —
`Cannae`/`Carrhae`, `Pantheon`/`Parthenon`, `Justin I`/`Justinian I`. Every one
is obvious to anyone who knows the subjects and invisible to `difflib`. That
ratio is the argument for a model here and against automating the mechanical
signal: as a detector it is worth its noise, as a decider it would have merged
distinct pieces of history.

Output is the review-file format `universe_builder lint` already defines, so the alias
writing, the dry run and the typo'd-verdict refusal are all reused rather than
rebuilt.
"""

from __future__ import annotations

import difflib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from pydantic import BaseModel, Field

from universe_builder import store, topics
from universe_builder.registry import active_pack

log = logging.getLogger(__name__)

#: Character-similarity floor for a candidate. Lower than `lint.suggest_verdict`'s
#: 0.88 on purpose: that threshold gates an automatic `same`, this one gates
#: *asking*, and the costs are not symmetric. 0.82 admits `Massilia`/`Massalia`
#: and `Catana`/`Catane`, at the price of noise a verdict step removes.
SIMILARITY_FLOOR = 0.82


def default_atoms() -> tuple:
    """Every generated atom type of the active domain. All are checked; the
    caller restricts with `--atom`."""
    return tuple(spec.name for spec in active_pack().atoms)


class Judgement(BaseModel):
    """One adjudicated pair. The schema the model is held to."""

    same_subject: bool = Field(description=(
        "True only if these two names denote the SAME subject in different "
        "languages, spellings or periods. False if they are different subjects "
        "that merely look alike."
    ))
    confident: bool = Field(description=(
        "False if you are unsure. An unsure answer is recorded as undecided and "
        "shown to a human; a confident wrong answer merges two subjects "
        "permanently, so abstaining is always the better error."
    ))
    reason: str = Field(description=(
        "One short clause of justification, at most 15 words. Name what "
        "distinguishes them if they differ."
    ))


@dataclass(frozen=True)
class Candidate:
    """A possible collision, before anyone has judged it."""

    atom_type: str
    #: The name the incoming seed declares.
    incoming: str
    #: The stored atom's own `name.common`.
    stored: str
    #: The stored atom's filename, which `apply_review` edits for a `wrong`.
    filename: str
    #: How it was found: "claim" (the stored atom lists `incoming` among its
    #: `otherNames`) or "similar" (nothing claims it; the names merely look
    #: alike).
    source: str

    @property
    def can_be_wrong(self) -> bool:
        """Whether `wrong` is a meaningful verdict for this row.

        `wrong` deletes: `lint.apply_review` strips the alias from the named
        atom's `otherNames`. That is right for a *claim* — the corpus asserting
        an identity that is not true, which is how `Narseh` came to claim
        `Narses`, a different man four centuries later. It is meaningless for a
        *similarity* candidate, where no atom claims anything and there is
        nothing to remove; offering `wrong` there invites an edit that would
        silently do nothing, or worse, strip an unrelated name that happens to
        match.
        """
        return self.source == "claim"


def _id_of(path: Path) -> Optional[str]:
    """The content id in a stored filename, without its `h`, or None.

    `decompose_filename`, never `split("_")[1]`: that is the id only for an
    atom type without an underscore (`side_effect_h…` would yield `effect`).
    """
    try:
        return store.AtomName.decompose_filename(path.name).padded_id.lstrip("h")
    except ValueError:
        return None


def _stored_index(atom_type: str) -> List[tuple]:
    """`(folded topic, common name, filename, id)` for every stored atom.

    A list, not a dict keyed on the folded slug: two atoms can share one —
    every CJK subject slugs to the empty string — and keying on it kept only
    the last, so the corpus's own claim on the other was lost (1.7).
    """
    out: List[tuple] = []
    for path in store.iter_atoms(atom_type):
        try:
            topic = store.AtomName.decompose_filename(path.name).getTopic()
        except ValueError:
            continue
        try:
            data = store.read_json(path)
        except FileNotFoundError:
            continue        # retired or moved since the listing
        common = topic
        if isinstance(data, dict):
            common = ((data.get("name") or {}).get("common") or topic)
        out.append((topics.name_key(topic), common, path.name, _id_of(path)))
    return out


def _stored_ids(atom_type: str) -> set:
    return {i for i in (_id_of(path) for path in store.iter_atoms(atom_type)) if i}


def load_seed(name: str) -> tuple:
    """`(payload, filename)` of the stored index atom named `name`, or
    `(None, None)`: the seed `collisions --seed NAME` checks.

    Matched by content id first: the slug drops hyphens and accents, so
    `--seed "Anglo-Saxon History"` never matched its own file by name.
    """
    index_name = active_pack().index.name
    wanted = name.strip().casefold()
    wanted_id = store.atom_id(index_name, name)
    for path in store.iter_atoms(index_name):
        try:
            parsed = store.AtomName.decompose_filename(path.name)
        except ValueError:
            continue
        if parsed.padded_id == wanted_id or parsed.getTopic().strip().casefold() == wanted:
            data = store.read_json(path)
            if isinstance(data, dict):
                return data, path.name
    return None, None


def find(seed: dict, atoms: Optional[Sequence[str]] = None) -> List[Candidate]:
    """Every collision candidate between `seed`'s topics and the stored corpus.

    `seed` is an index atom's payload (a History atom, for history). A topic
    whose content id already exists is skipped outright: that is a re-pin, the
    thing content addressing is for, and it needs no alias.
    """
    pack = active_pack()
    plural = {spec.name: spec.plural for spec in pack.all_specs()}
    found: List[Candidate] = []

    for atom_type in (atoms if atoms is not None else default_atoms()):
        key = plural.get(atom_type)
        if not key:
            continue
        declared = [t for t in (seed.get(key) or []) if isinstance(t, str) and t.strip()]
        if not declared:
            continue

        existing_ids = _stored_ids(atom_type)
        index = _stored_index(atom_type)
        claims = topics.stored_name_claims(atom_type)
        by_id = {atom_id: (common, filename) for _f, common, filename, atom_id in index if atom_id}
        by_folded: Dict[str, List[tuple]] = {}
        for folded, common, filename, _id in index:
            by_folded.setdefault(folded, []).append((common, filename))
        folded_topics = list(by_folded)

        for topic in declared:
            if store.atom_id(atom_type, topic).lstrip("h") in existing_ids:
                continue  # a re-pin: one subject, one file, nothing to declare

            seen_files = set()

            # 1. The corpus naming it outright. Highest precision, and the only
            #    source for which `wrong` means anything.
            for owner in claims.get(topics.name_key(topic), ()) or ():
                hit = by_id.get(owner.lstrip("h"))
                if hit and hit[1] not in seen_files:
                    seen_files.add(hit[1])
                    found.append(Candidate(atom_type, topic, hit[0], hit[1], "claim"))

            # 2. Names that merely look alike. Noisy by design — see the module
            #    docstring's 32-of-43 measurement — and kept because the claim
            #    index is silent whenever the stored atom never listed the
            #    rival spelling, which is most of the time.
            close = difflib.get_close_matches(
                topics.name_key(topic), folded_topics, n=1, cutoff=SIMILARITY_FLOOR)
            for match in close:
                if match == topics.name_key(topic):
                    continue
                for common, filename in by_folded[match]:
                    if filename in seen_files:
                        continue
                    seen_files.add(filename)
                    found.append(Candidate(atom_type, topic, common, filename, "similar"))

    return found


def shared(seed: dict, atoms: Optional[Sequence[str]] = None,
           seed_file: Optional[str] = None) -> List[tuple]:
    """`(atom_type, topic, filename, [other civilisations])` for every topic
    `find` skips as a re-pin: its content id is already on disk.

    A re-pin is usually the point (Alexandria is one city in every History),
    and it is also exactly where one bare name meaning two things hides: a
    papal seed's `Leo I` lands on the stored Byzantine emperor, and `find`
    never says so, because to a hash the two are one topic. Nothing here can
    tell which case a re-pin is, so it is listed for a person, with the other
    Histories that name it; the fix for a wrong one is a scoped alias line
    before `create` (found by a new-history walk-through).
    """
    pack = active_pack()
    plural = {spec.name: spec.plural for spec in pack.all_specs()}
    others = []
    for path in store.iter_atoms(pack.index.name):
        if seed_file and path.name == seed_file:
            continue
        try:
            data = store.read_json(path)
        except FileNotFoundError:
            continue        # retired or moved since the listing
        if isinstance(data, dict):
            others.append((topics.context_of(path.name), data))
    out: List[tuple] = []
    for atom_type in (atoms if atoms is not None else default_atoms()):
        key = plural.get(atom_type)
        if not key:
            continue
        existing = {}
        for path in store.iter_atoms(atom_type):
            atom_id = _id_of(path)
            if atom_id:
                existing[atom_id] = path.name
        for topic in [t for t in (seed.get(key) or []) if isinstance(t, str) and t.strip()]:
            filename = existing.get(store.atom_id(atom_type, topic).lstrip("h"))
            if not filename:
                continue
            folded = topics.name_key(topic)
            listed_by = sorted(civ for civ, data in others
                               if any(isinstance(t, str) and topics.name_key(t) == folded
                                      for t in (data.get(key) or [])))
            out.append((atom_type, topic, filename, listed_by))
    return out


ADJUDICATION_SYSTEM = (
    "You decide whether two names refer to the same subject.\n"
    "\n"
    "You are shown a name an incoming source declares and the name of a record "
    "already held. Answer whether they are one subject under two names — "
    "different languages, transliterations, or ancient and modern forms — or "
    "two different subjects whose names resemble each other.\n"
    "\n"
    "Names frequently resemble each other without being related. If the two "
    "are distinct, say so plainly.\n"
    "\n"
    "If you are not sure, say you are not sure. An undecided pair is shown to a "
    "person; a confident wrong answer merges two subjects permanently."
)


def adjudicate(
    candidates: Iterable[Candidate],
    model: str,
    *,
    max_tokens: int = 2048,
) -> Dict[Candidate, tuple]:
    """Judge each candidate with one model call. Returns candidate -> (verdict, why).

    One call per pair rather than one call for all of them. The volume is tens,
    not thousands, so the saving would be trivial; and a single batched reply
    that fails validation loses every verdict at once, while a per-pair failure
    costs one row and leaves the rest. Each call is journalled under its own
    subject (`journal.record_call`, operation `alias-adjudication`) into
    whatever run is current — `cmd_collisions` opens one for `--via model` — so
    `universe_builder costs --by subject` can price a wrong alias later. This
    docstring once claimed that while nothing journalled at all: the call went
    straight to `provider_run.complete` and no run was open.

    A verdict is only ever `same`, `wrong` or `skip`. `wrong` is withheld unless
    the candidate came from a claim — see `Candidate.can_be_wrong`.
    """
    from universe_builder import journal
    from universe_builder.providers import run as provider_run
    from universe_builder.providers.types import LLMRequest

    verdicts: Dict[Candidate, tuple] = {}
    for candidate in candidates:
        user = (
            f"Subject type: {candidate.atom_type}\n"
            f"Name declared by the incoming source: {candidate.incoming}\n"
            f"Name of the record already held: {candidate.stored}\n\n"
            "Are these the same subject?"
        )
        request = LLMRequest(
            model=model,
            schema_model=Judgement,
            system=ADJUDICATION_SYSTEM,
            user=user,
            max_tokens=max_tokens,
            metadata={"operation": "alias-adjudication",
                      "atom_type": candidate.atom_type,
                      "subject": candidate.incoming},
        )
        journalled = dict(operation="alias-adjudication", atom_type=candidate.atom_type,
                          subject=candidate.incoming)
        try:
            response = provider_run.complete(request)
        except Exception as e:  # noqa: BLE001 - one bad pair must not lose the rest
            log.error("Adjudicating %r vs %r failed (%s); left undecided",
                      candidate.incoming, candidate.stored, e)
            journal.record_call(request, None, error=str(e), **journalled)
            verdicts[candidate] = ("?", "adjudication failed")
            continue
        journal.record_call(request, response, **journalled)

        judgement = response.parsed
        if not isinstance(judgement, Judgement):
            verdicts[candidate] = ("?", "no usable answer")
            continue

        why = (judgement.reason or "").strip().replace("\t", " ")[:110]
        if not judgement.confident:
            verdicts[candidate] = ("skip", f"model unsure: {why}")
        elif judgement.same_subject:
            verdicts[candidate] = ("same", why)
        elif candidate.can_be_wrong:
            verdicts[candidate] = ("wrong", why)
        else:
            # Different subjects, and nothing to delete: the corpus never
            # claimed the name, so there is no false claim to strip. Recording
            # `skip` says "create it separately", which is the correct outcome.
            verdicts[candidate] = ("skip", f"different subjects: {why}")
    return verdicts


def write_review(candidates: Sequence[Candidate], path,
                 verdicts: Optional[Dict[Candidate, tuple]] = None) -> int:
    """Emit candidates in `lint`'s review format. Returns rows written.

    Deliberately the same format, so `universe_builder lint --apply-review FILE --apply`
    writes the alias lines: that path already re-validates an edited atom,
    refuses the whole file on an unrecognised verdict, and previews both
    destructive and additive halves of a dry run. Rebuilding any of that here
    would mean two implementations of the one irreversible operation.
    """
    from universe_builder import lint

    verdicts = verdicts or {}
    lines = [lint.REVIEW_HEADER,
             "# Rows below come from `universe_builder collisions`: an incoming seed\n"
             "# against the stored corpus. `same` writes an alias line so create\n"
             "# makes one atom instead of two. `wrong` appears only where the\n"
             "# stored atom actually claims the name and that claim is false.\n"]
    written = 0
    for candidate in candidates:
        verdict, why = verdicts.get(candidate, ("?", f"found by {candidate.source}"))
        note = f"{candidate.source}: {why}" if why else candidate.source
        lines.append("\t".join([verdict, candidate.atom_type, candidate.filename,
                                candidate.incoming, candidate.stored, note]))
        written += 1
    path = str(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return written


def summarise(verdicts: Dict[Candidate, tuple]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for verdict, _why in verdicts.values():
        counts[verdict] = counts.get(verdict, 0) + 1
    return counts
