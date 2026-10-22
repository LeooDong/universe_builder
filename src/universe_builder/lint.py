"""Mechanical corpus checks: no model, no network, no writes.

Six checks, all of the same kind: defects no better model fixes, invisible
until something downstream breaks. `CHECKS` at the foot of this module is the
registry; `--check` names them.

Two are worked through below because they are the oldest and the clearest. The
other four came later and carry their reasoning at their own definitions:
`shared_aliases` (a name several atoms claim and none owns — the blind spot in
the first check, and where false duplicates come from), `topic_drift` (an atom
that disowns the topic it was generated for, which is how a fabricated index
topic becomes a real file), `alias_contradictions` (`aliases.txt` asserting
an identity `not_duplicates.txt` refuses), and `self_negations` (an atom whose
own summary says its subject did not happen or does not exist).

**Over-broad aliases.** An atom that claims another atom's name makes that name
ambiguous, and an ambiguous name resolves to nothing — correctly, because
guessing would silently merge two subjects. The gpt-5.6-luna `Karnak` atom
listed `Thebes` among its `otherNames`: true as geography, false as identity,
since Karnak is *in* Thebes. A containment relation is not an alias. Note the
honest scope: removing that one alias recovered **no** edges, because those
names were already failing a layer earlier — so this check is about keeping the
identity layer coherent, not about a measured edge count.

**Era-versus-prose contradictions.** The fact-check audit found that the
corpus's dominant genuine error is era confusion: a mask dated 550 **CE**
against a consensus of ~1300–1080 BCE, quarries dated 1500 **AD**. What makes
these catchable without a model is that the atom usually *contradicts itself* —
the structured `era` says CE while the atom's own prose says the year with BC or
BCE beside it. No external knowledge is needed to see that, which is exactly why
it belongs here and not in a paid verify pass.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from universe_builder import alias_files, store, topics
from universe_builder.registry import active_pack

log = logging.getLogger(__name__)


@dataclass
class Finding:
    """One flagged atom. Reported, never repaired — these are judgement calls."""

    check: str
    atom_type: str
    file: str
    subject: str
    detail: str
    #: The offending value, carried as data rather than recovered from `detail`.
    #: Parsing it back out of the human-readable sentence broke on every name
    #: containing an apostrophe — `Alexander the Great's conquest`,
    #: `Al-Muʿizz li-Dīn Allāh` — which is what happens whenever a message
    #: meant for a reader is used as a wire format.
    value: str = ""


@dataclass
class Report:
    findings: List[Finding] = field(default_factory=list)
    #: Checks that raised. **Never empty and clean at the same time** — a check
    #: that crashed has not established anything, and reporting it as "no
    #: findings" is the one output a linter must never produce. Swallowing the
    #: exception keeps one broken check from costing the others; letting the
    #: result still read as clean is what turns that kindness into a lie.
    errors: List[str] = field(default_factory=list)

    def by_check(self) -> Dict[str, List[Finding]]:
        grouped: Dict[str, List[Finding]] = {}
        for finding in self.findings:
            grouped.setdefault(finding.check, []).append(finding)
        return grouped

    def is_clean(self) -> bool:
        return not self.findings and not self.errors


# ---------------------------------------------------------------------------
# Check 1 — over-broad aliases
# ---------------------------------------------------------------------------

def overbroad_aliases(atom_type: str) -> List[Finding]:
    """Atoms whose `otherNames` claim a name another atom holds as `common`.

    The claim direction matters and is the whole point: an atom listing a name
    that is merely *also* used elsewhere is ordinary synonymy, but an atom
    listing a name that another stored atom calls itself is asserting those two
    atoms are one thing. When they are not, the name becomes ambiguous and
    stops resolving for everybody.

    An alias line in the domain's `aliases.txt` is a human saying "these really
    are one topic", so a pair it covers is not flagged — otherwise the file that
    exists to declare identity would be reported as a defect.
    """
    declared = {key for group in topics.load_aliases() for key in group.keys()}
    commons: Dict[str, str] = {}
    for path, payload in store.iter_payloads(atom_type):
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        if isinstance(name, dict) and isinstance(name.get("common"), str):
            folded = topics.name_key(name["common"])
            if folded:
                commons.setdefault(folded, path.name)

    findings: List[Finding] = []
    for path, payload in store.iter_payloads(atom_type):
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        if not isinstance(name, dict):
            continue
        mine = topics.name_key(name.get("common") or "")
        for other in name.get("otherNames") or []:
            if not isinstance(other, str) or not other.strip():
                continue
            folded = topics.name_key(other)
            owner = commons.get(folded)
            if not folded or folded == mine or owner is None or owner == path.name:
                continue
            if folded in declared:
                continue  # a human has declared these one topic
            findings.append(Finding(
                check="overbroad-alias", atom_type=atom_type, file=path.name,
                subject=name.get("common") or path.name, value=other,
                detail=f"claims {other!r}, which {owner.split('_', 2)[-1]} "
                       f"calls itself — an alias, or containment mistaken for one?",
            ))
    return findings


# ---------------------------------------------------------------------------
# Check 1b — one alias, several claimants, no owner
# ---------------------------------------------------------------------------

def _claim_index():
    """Corpus-wide `(folded alias) -> {(type, file): (common, raw)}`, plus owners.

    Built across **every** atom type, not the one being checked, because the
    collisions that matter cross types: `Usermaatre Setepenre` is claimed by a
    person (`Ramesses II`) and by an event (`Reign of Ramesses II`), and a
    per-type index cannot see the pair.

    Deliberately uncached. A module-level memo would be read by the next call in
    the same process, which in tests means the previous test's `UNIVERSE_BUILDER_DATA_DIR` —
    the class of bug this project has paid for more than once. Rebuilding costs
    one corpus read per atom type and the corpus is small.
    """
    claims: Dict[str, Dict[tuple, tuple]] = {}
    owners: Dict[str, str] = {}
    for spec in active_pack().atoms:
        for path, payload in store.iter_payloads(spec.name):
            if not isinstance(payload, dict):
                continue
            name = payload.get("name")
            if not isinstance(name, dict):
                continue
            common = name.get("common")
            if not isinstance(common, str) or not common.strip():
                continue
            mine = topics.name_key(common)
            if mine:
                owners.setdefault(mine, common)
            for other in name.get("otherNames") or []:
                if not isinstance(other, str) or not other.strip():
                    continue
                folded = topics.name_key(other)
                if not folded or folded == mine:
                    continue
                claims.setdefault(folded, {})[(spec.name, path.name)] = (common, other)
    return claims, owners


def shared_aliases(atom_type: str) -> List[Finding]:
    """Aliases that several atoms claim and **no atom owns**.

    `overbroad_aliases` catches a claim on a name another atom calls itself. It
    is blind, structurally, to the case where two atoms claim the same name and
    nothing is called it — there is no `common` for the claim to collide with.
    That blind spot is where false duplicates come from: `Narmer Palette` and
    `Two Dog Palette` both list `Hierakonpolis Palette`, so a detector reading
    the names atoms declare about themselves concludes they are one object. They
    are two objects that happen to share a findspot.

    The damage is not the alias, it is what gets built on top of it.
    `universe_builder duplicates --apply` would retire one of the two atoms *and* append a
    permanent line to `aliases.txt` — and that line is what stops the next
    `create` run from regenerating the distinction. A wrong merge here is not a
    file to move back; it is an identity the corpus can no longer express.

    A name a human has already declared in `aliases.txt` is not flagged, on the
    same principle as check 1: the file that exists to declare identity must not
    be reported as a defect.
    """
    declared = {key for group in topics.load_aliases() for key in group.keys()}
    claims, owners = _claim_index()

    findings: List[Finding] = []
    for folded, claimants in claims.items():
        if folded in owners:
            continue  # an atom is called this — check 1's case, not this one
        if folded in declared:
            continue  # a human has declared these one topic
        # Two *files* for one subject are a duplicate, which `universe_builder duplicates`
        # exists for; requiring two distinct subjects keeps this check to the
        # thing only it can see.
        subjects = {topics.name_key(common) for common, _ in claimants.values()}
        if len(subjects) < 2:
            continue
        for (claim_type, filename), (common, raw) in sorted(claimants.items()):
            if claim_type != atom_type:
                continue
            rivals = sorted({other for other, _ in claimants.values()
                             if topics.name_key(other) != topics.name_key(common)})
            findings.append(Finding(
                check="shared-alias", atom_type=claim_type, file=filename,
                subject=common, value=raw,
                detail=f"claims {raw!r}, which no atom is called and "
                       f"{len(rivals)} other subject(s) also claim "
                       f"({', '.join(rivals)}) — until one side drops it the "
                       f"name resolves to nothing and invites a false merge",
            ))
    return findings


# ---------------------------------------------------------------------------
# Check 1d — the two identity files contradict each other
# ---------------------------------------------------------------------------

def alias_contradictions(atom_type: str) -> List[Finding]:
    """Pairs `aliases.txt` groups that `not_duplicates.txt` declares distinct.

    The two files are mirrors: one says "these spellings are one topic", the
    other "these atoms are not, whatever their names suggest". Nothing checked
    that they agree, and they did not — three groups asserted an identity a
    human had explicitly refused, including `Dipylon Krater = Dipylon Amphora`
    against a line reading *"A krater and an amphora, both from the Dipylon
    cemetery. Different vessels."*

    The asymmetry is what makes it dangerous. `not_duplicates.txt` gates only
    `universe_builder duplicates`, while `aliases.txt` drives the grouping every work list
    and resolver reads — so the alias wins, and the refusal is silently
    overridden by the file that was supposed to respect it. Worse, a contradiction
    is self-extending: `_merge_into_alias_file` folds a new pair into any group
    sharing a name, so a later honest line joins the false group and inherits it.

    The finding is about two files, so `run` collapses the identical copies one
    per atom type into one. It is **not** suppressed here by comparing
    `atom_type` against the pack's first atom, which is what this did first:
    `universe_builder lint --check alias-contradiction --atom artifact` then answered "No
    findings" having compared nothing, because `artifact` is not `person`. A
    linter that reports clean without running is the one output it must never
    produce, and restricting to an atom type is an ordinary thing to do.
    """
    pack = active_pack()
    # Derived from whichever alias file is actually in force, not from the
    # pack's own path: `load_aliases` honours the `ALIASES_LIST` override, and
    # reading the refusals from a different directory than the groups would
    # compare two unrelated files. `duplicates.py` pairs them the same way.
    distinct = {}
    path = alias_files.aliases_path()
    not_dupes = alias_files.not_duplicates_path(path) if path else None
    if not not_dupes or not not_dupes.exists():
        # No refusals file means no refusals, so no alias can contradict one —
        # clean here is vacuously true rather than an unchecked claim, and a
        # domain without the file (medications has none) is a legitimate
        # configuration, not a defect. Logged all the same, because the same
        # silence would follow from a mis-set `ALIASES_LIST` pointing somewhere
        # the file does not live.
        if path:
            log.warning("no not_duplicates.txt beside %s; no alias contradictions "
                        "can be detected", path)
        else:
            log.warning("the %s domain declares no alias file, so it has no "
                        "not_duplicates.txt and no alias contradictions can be "
                        "detected", pack.name)
        return []
    for declaration in alias_files.NotDuplicatesFile.read(not_dupes).declarations:
        line = declaration.body
        # **Split on every `!=`, and fold with `alias_keys`.** Both were wrong
        # and the check reported "No findings" on a file holding real
        # contradictions — the one output a linter must never produce, as the
        # comment above this function already says.
        #
        # `split("!=", 1)` kept only the first separator, so a three-way line
        # like `Arsinoe (Crocodilopolis) != Soknopaiou Nesos != Fayoum` folded
        # its whole tail into one key, `soknopaiou nesos fayoum`, which nothing
        # can ever match. Seven such lines were invisible, and even their first
        # pair was mis-keyed.
        #
        # `topics.name_key` also folds differently from the grouping machinery:
        # `Nicaea (İznik) != Nikaia (Nice)` became `{nicaea i znik, nikaia
        # nice}` and never met the alias group's `{nicaea, nikaia}`, so an
        # alias line contradicting a not-duplicates line went
        # unreported. `duplicates._claim_keys` exists for exactly this and says
        # so: "the safeguards must fold names exactly as the grouping machinery
        # does". The same bug, reintroduced in a different file.
        keyed = [topics.alias_keys(part) for part in declaration.names]
        for i, left_keys in enumerate(keyed):
            for right_keys in keyed[i + 1:]:
                for left in left_keys:
                    for right in right_keys:
                        if left and right and left != right:
                            distinct[frozenset((left, right))] = line

    findings: List[Finding] = []
    for group in topics.load_aliases():
        keys = sorted({key for key in group.keys() if key})
        for i, left in enumerate(keys):
            for right in keys[i + 1:]:
                claim = distinct.get(frozenset((left, right)))
                if not claim:
                    continue
                findings.append(Finding(
                    check="alias-contradiction", atom_type=atom_type,
                    file="aliases.txt", subject=left, value=right,
                    detail=(f"aliases.txt groups {left!r} with {right!r}, but "
                            f"not_duplicates.txt declares {claim!r} — the alias "
                            f"wins at grouping time, so the refusal is ignored"),
                ))
    return findings


# ---------------------------------------------------------------------------
# Check 1c — the atom disowns the topic it was generated for
# ---------------------------------------------------------------------------

#: Words that make a topic's extra clause an *argument* rather than a name.
#: `Buto (Per-Wadjet)` offers another name for one place; `Shroud of Turin
#: (believed to be connected with Egypt)` argues for its own inclusion.
_ARGUES = {"believed", "connected", "contains", "associated", "linked",
           "possibly", "reportedly", "supposedly", "related", "thought",
           "attributed", "claimed"}

#: A dropped word that follows one of these was an *attribution*, not a second
#: name. This is what separates the two shapes, and the arguing-word list alone
#: could not: `Berlin Gold Hat of Tutankhamun` drops a bare proper noun and is a
#: pure fabrication, while `Nicaea Iznik` juxtaposes two names for one city and
#: is fine. The connective is the whole difference.
_ATTRIBUTES = ("of", "by", "from", "belonging")

_TOPIC_STOP = {"of", "the", "at", "in", "a", "and", "to", "be", "with", "by",
               "from", "for"}


def topic_drift(atom_type: str) -> List[Finding]:
    """Atoms whose own `name` quietly drops part of the topic they were asked for.

    The index call is the highest-leverage one in the system — it decides the
    several hundred topics every later phase generates — and it is the one whose
    mistakes nothing downstream can see. A topic that does not exist still gets
    an atom: asked for `Berlin Gold Hat of Tutankhamun`, the create pass wrote a
    perfectly accurate atom about the *Berlin Gold Hat*, a Central European
    bronze-age cone, and filed it under Egyptian history. Validation cannot
    catch that — the payload is schema-perfect and the prose is true. The only
    trace is that the atom declined to call itself what it was asked about.

    So the signal is the gap between the topic and the name, and the filter is
    what the gap contains. Dropping `Iznik` from `Nicaea (Iznik)` is a city
    keeping its ancient name and is not reported. Dropping `believed to be
    connected with Egypt` is an atom refusing a claim its topic made, which is
    the case worth a human's attention.
    """
    findings: List[Finding] = []
    for path, payload in store.iter_payloads(atom_type):
        if not isinstance(payload, dict):
            continue
        name = payload.get("name")
        if not isinstance(name, dict):
            continue
        common = name.get("common") or ""
        if not common:
            continue
        try:
            topic = store.AtomName.decompose_filename(path.name).getTopic()
        except Exception:  # noqa: BLE001 - an unparseable name is another check's problem
            continue
        if not topic:
            continue
        topic_words = set(topics.name_key(topic).split()) - _TOPIC_STOP
        claimed = set(topics.name_key(common).split())
        for other in name.get("otherNames") or []:
            if isinstance(other, str):
                claimed |= set(topics.name_key(other).split())
        dropped = topic_words - claimed - _TOPIC_STOP
        if not dropped or not (set(topics.name_key(common).split()) - _TOPIC_STOP) <= topic_words:
            continue
        # The dropped words must *trail* the topic and sit immediately behind a
        # connective. Merely appearing somewhere after an `of` is not enough:
        # `Tomb of the Unknown Soldier in Athens` would qualify on that reading,
        # and the city is a location, not a disowned attribution.
        words = topics.name_key(topic).split()
        tail = len(words)
        while tail and words[tail - 1] in dropped:
            tail -= 1
        attributed = (tail < len(words) and tail > 0
                      and words[tail - 1] in _ATTRIBUTES)
        arguing = (dropped & _ARGUES) | (set(words[tail:]) if attributed else set())
        if not arguing:
            continue          # an alternative name, not a disowned claim
        findings.append(Finding(
            check="topic-drift", atom_type=atom_type, file=path.name,
            subject=common, value=" ".join(sorted(dropped)),
            detail=(f"was generated for {topic!r} but calls itself {common!r}, "
                    f"dropping {sorted(arguing)} — the index atom asserted a "
                    f"connection this atom does not make"),
        ))
    return findings


# ---------------------------------------------------------------------------
# Check 2 — era contradicted by the atom's own prose
# ---------------------------------------------------------------------------

#: A year written in prose with its era attached: "3100 BC", "c. 1279 BCE",
#: "550 AD". Requires the era token, because a bare year in prose says nothing.
_DATED = re.compile(r"\b(\d{1,4})\s*(BCE|BC|CE|AD)\b", re.IGNORECASE)


def _strings(node, skip_keys=("sources",)) -> str:
    """Free text inside one subtree, concatenated.

    `sources` is skipped: a citation's own date is about the edition, not the
    subject.
    """
    chunks: List[str] = []

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key not in skip_keys:
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str) and len(value) > 3:
            chunks.append(value)

    walk(node)
    return " ".join(chunks)


def _dated_records(payload) -> List[tuple]:
    """`(path, year, era, local_text)` for every structured date.

    **`local_text` is the enclosing record's prose, not the whole atom's**, and
    that scoping is the difference between a check and a nuisance. Flattening
    the atom compared a city's fifth period against prose belonging to its
    second: `Malia` has a "Modern village" period starting 1900 CE — correct —
    and a Minoan period discussing 1900 BCE, and the flat version called the
    first one a contradiction. A date can only be contradicted by text that is
    talking about it.
    """
    found: List[tuple] = []

    def walk(node, path="", scope=None):
        if isinstance(node, dict):
            year, era = node.get("year"), node.get("era")
            if isinstance(year, int) and era in ("BCE", "CE"):
                found.append((path or "date", year, era,
                              _strings(scope if scope is not None else node)))
            for key, value in node.items():
                # A dict holding a date is itself the innermost record; its
                # parent is the scope for that date's siblings.
                walk(value, f"{path}.{key}" if path else key,
                     node if scope is None else scope)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                # Each list element is its own record — this is what keeps one
                # period's prose away from another period's dates.
                walk(item, f"{path}[{i}]", item if isinstance(item, dict) else scope)

    walk(payload)
    return found


def era_contradictions(atom_type: str) -> List[Finding]:
    """Atoms whose structured `era` is contradicted by the prose around it.

    Deliberately narrow: the *same year* must appear in the *enclosing record's*
    text with an explicit era token that disagrees with the field. That makes a
    flag a self-contradiction rather than an opinion about history, so it needs
    no model and no lookup.

    It will not catch an atom that is uniformly wrong — one whose prose and
    fields agree on 550 CE for a Ramesside mask. That needs external knowledge,
    which is the search-backed pass, not this.
    """
    findings: List[Finding] = []
    for path, payload in store.iter_payloads(atom_type):
        if not isinstance(payload, dict):
            continue
        subject = ((payload.get("name") or {}).get("common")) or path.name
        for where, year, era, text in _dated_records(payload):
            if not text:
                continue
            said = {("BCE" if token.upper() in ("BC", "BCE") else "CE")
                    for found_year, token in _DATED.findall(text)
                    if int(found_year) == year}
            if said and era not in said and len(said) == 1:
                findings.append(Finding(
                    check="era-contradiction", atom_type=atom_type, file=path.name,
                    subject=subject,
                    detail=f"{where} says {year} {era}, but the text beside it "
                           f"writes {year} {next(iter(said))}",
                ))
    return findings


# ---------------------------------------------------------------------------
# Check 6 — self-negation
# ---------------------------------------------------------------------------

#: Phrases that deny a subject happened or exists. English, and about
#: occurrence and existence in general — nothing here knows a domain.
_DENIAL = re.compile(
    r"\b(?:"
    r"(?:did\s+not|didn't|never|has\s+not|have\s+not|had\s+not)\s+(?:take|taken|took)\s+place"
    r"|(?:did\s+not|didn't|never|has\s+not|had\s+not)\s+(?:happen(?:ed)?|occur(?:red)?|exist(?:ed)?)"
    r"|not\s+occurred"
    r"|(?:is|was|are|were)\s+(?:entirely\s+|purely\s+|wholly\s+|largely\s+)?(?:fictional|fictitious|imaginary)"
    r"|(?:existence|occurrence)\s+(?:is|was)\s+not\s+(?:supported|attested|confirmed|established)"
    r"|no\s+such"
    r"|hypothetical"
    r")\b",
    re.IGNORECASE,
)

#: A sentence about rebuilding the subject, where "hypothetical" describes the
#: rebuilding rather than the subject: "reconstructions remain partly
#: hypothetical". Six such sentences in the history corpus, none a denial.
_RECONSTRUCTION = re.compile(r"reconstruct|restor", re.IGNORECASE)

#: Which top-level fields are the atom summarising itself. A denial elsewhere
#: is usually about something the atom mentions — "the violin did not exist"
#: in Nero's curious facts — not about the atom's subject.
_SELF_PROSE = re.compile(r"summary|significance|overview|abstract", re.IGNORECASE)

_SENTENCE = re.compile(r"(?<=[.;!?])\s+")


def _self_prose(payload: dict) -> List[tuple]:
    """`(field, text)` for each top-level summarising field, list items apart."""
    found = []
    for key, value in payload.items():
        if not isinstance(key, str) or not _SELF_PROSE.search(key):
            continue
        for text in (value if isinstance(value, list) else [value]):
            if isinstance(text, str) and text.strip():
                found.append((key, text))
    return found


def _undated(atom_type: str, payload: dict) -> bool:
    """Whether any of the atom's identifying dates (the pack's `date_paths`)
    records certainty `none` — the atom's own admission it could not place
    the subject in time."""
    _coordinates, dates = _evidence_paths(atom_type)
    for path in dates:
        date = _dig(payload, *path)
        if isinstance(date, dict) and str(date.get("certainty") or "").lower() == "none":
            return True
    return False


def self_negations(atom_type: str) -> List[Finding]:
    """Atoms whose own summary denies that their subject happened or exists.

    The index atoms list topics and a create pass generates one atom each;
    when a listed topic is not real — a misremembered event, an invented
    object — a model asked to describe it can answer honestly *inside the
    atom*: "this entry records a hypothetical or planned event that did not
    take place" (the history corpus's `Egyptian presidential election of
    2023`). The file validates, and nothing downstream reads its prose, so it
    stands in the corpus as a subject like any other.

    Read only in the atom's summarising top-level fields (`_SELF_PROSE`), and
    "hypothetical" is ignored in a sentence about a reconstruction. **Strong**
    when one of the pack's identifying dates carries certainty `none` as well:
    the atom then denies its subject in prose and cannot date it either.
    Strong findings sort first. A judgement call like every check here —
    retire, regenerate or keep is a person's decision.
    """
    strong: List[Finding] = []
    weak: List[Finding] = []
    for path, payload in store.iter_payloads(atom_type):
        if not isinstance(payload, dict):
            continue
        hits = []
        for field_name, text in _self_prose(payload):
            for sentence in _SENTENCE.split(text):
                match = _DENIAL.search(sentence)
                if not match:
                    continue
                if match.group(0).lower() == "hypothetical" and _RECONSTRUCTION.search(sentence):
                    continue
                hits.append((field_name, match.group(0), sentence.strip()))
        if not hits:
            continue
        name = payload.get("name")
        subject = (name.get("common") if isinstance(name, dict) else name) or path.name
        field_name, phrase, sentence = hits[0]
        is_strong = _undated(atom_type, payload)
        detail = (("[strong: a date has certainty none] " if is_strong else "")
                  + f"{field_name}: {sentence[:140]}"
                  + (f" (+{len(hits) - 1} more)" if len(hits) > 1 else ""))
        (strong if is_strong else weak).append(Finding(
            check="self-negation", atom_type=atom_type, file=path.name,
            subject=str(subject), detail=detail, value=phrase))
    return strong + weak


# ---------------------------------------------------------------------------
# Adjudication: a reviewable file, then an apply
# ---------------------------------------------------------------------------

REVIEW_HEADER = """# Lint findings awaiting a verdict.
#
# Set the first column on each line, then run `universe_builder lint --apply-review FILE`.
#
#   same    the two names really are one topic -> a line is added to the
#           domain's aliases.txt, so the identity is declared once and every
#           consumer agrees. Nothing is deleted.
#   wrong   the alias is false (usually containment mistaken for identity,
#           e.g. Karnak claiming Thebes) -> the offending entry is removed from
#           that atom's `otherNames`. The atom is otherwise untouched.
#
# For a `shared-alias` row — a name several atoms claim and none is called —
# `wrong` is usually the answer, and it is the safe one: it drops the claim from
# this atom only, leaving the others to be judged on their own rows. Use `same`
# when the name really does belong to this subject; that assigns it here and
# ends the ambiguity for everybody.
#   skip    leave it alone; asked again next run.
#   ?       undecided; treated as skip.
#
# The last column is corroborating evidence: whether the two atoms agree about
# place and date. Treat DISAGREES as strong — two subjects that differ on
# coordinates or by centuries are rarely one thing. Treat "agrees" as weak:
# containment agrees too, which is how Karnak came to claim Thebes.
#
# Format: verdict <TAB> atom_type <TAB> file <TAB> alias <TAB> note
# Lines beginning with # are ignored. Order does not matter.
"""


def identity_decider(report: "Report", identity, *, label: str, say=print):
    """A `decide` for `write_review` from one of the domain's identity sources
    (`DomainPack.identity_resolvers[label]`): `lint --review F --via SOURCE`.

    Resolves every name the findings mention and fetches each subject's
    article opening as evidence, saying so through `say` as it goes, then
    settles contested names (`rivals_distinct_verdicts`). The returned
    function gives the source's verdict, else the contested-name ruling, else
    abstains with the opening attached for a person to read. It lived inline
    in `cli.cmd_lint` until v0.2.0.
    """
    names = sorted({f.subject for f in report.findings}
                   | {f.value for f in report.findings})
    say(f"resolving {len(names)} name(s) through {label}...")
    resolved = identity.resolve_titles(names)
    subjects = sorted({f.subject for f in report.findings})
    say(f"fetching {len(subjects)} article opening(s) as evidence...")
    leads = identity.subject_leads(subjects)

    def _words(text):
        return set(re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).split())

    def _page_id(title):
        return resolved.get(title, (None, "", False))[0]

    def _states_the_name(subject, alias):
        """Tri-state: True states it, False read and absent, None unread.

        `None` is not a detail. Without it a failed lookup reads as
        "the article does not mention this name" and the row is deleted
        on evidence that was never obtained.
        """
        opening = leads.get(subject) or ""
        if not opening.strip():
            return None
        wanted = _words(alias)
        return bool(wanted) and wanted <= _words(opening)

    # A name the source could not look up is absent from `resolved`
    # (the `identity_resolvers` contract): unknown, not "no page".
    contested = rivals_distinct_verdicts(
        report.findings, _page_id, owns=_states_the_name,
        answered=lambda name: name in resolved)
    if contested:
        say(f"{len(contested)} row(s) settled by contested-name rule")

    def source_decide(finding):
        """Verdict from redirects; on abstention, the evidence a person
        needs. Four ways of reading that text automatically were each
        wrong more often than abstaining, so it is shown, not acted on."""
        subject, alias = finding.subject, finding.value
        verdict, why = identity.identity_verdict(subject, alias, resolved)
        if verdict is not None:
            return verdict, why
        settled = contested.get((finding.file, alias))
        if settled:
            return settled
        opening = (leads.get(subject) or "")[:160]
        if opening:
            why = f"{why} | {subject} opens: {opening}"
        return None, why

    return source_decide


def write_review(report: "Report", path, decide=None) -> int:
    """Emit the findings as an editable file. Returns how many were written.

    A file rather than a prompt, for the same reason `aliases.txt` is a file:
    the decisions outlive the session, want to be diffed, and are worth
    reviewing as a set rather than one modal question at a time.
    """
    lines = [REVIEW_HEADER]
    written = 0
    reviewable = ("overbroad-alias", "shared-alias")
    # Built once, and only if needed: a `shared-alias` row's rival is not the
    # atom *called* the alias (nothing is), so the evidence lookup that serves
    # check 1 finds nothing to compare against.
    #
    # Keyed by `(file, folded alias)`, **not by file**. An atom claiming two
    # shared aliases has a different rival for each, and a file-keyed map hands
    # the first one to both rows — evidence about a pair the row is not asking
    # about. `Reign of Akhenaten` claims `Usermaatre Setepenre` (rivals: the
    # Ramesses atoms) and `Amenhotep IV` (rival: Akhenaten); the header tells
    # the reviewer to treat a DISAGREES verdict as strong, so a mismatched pair
    # does not merely confuse, it argues for the wrong answer.
    rivals: Dict[tuple, list] = {}
    if any(f.check == "shared-alias" for f in report.findings):
        claims, _owners = _claim_index()
        for folded, claimants in claims.items():
            for (claim_type, filename), (common, _raw) in claimants.items():
                for (other_type, other_file), (other_common, _) in claimants.items():
                    if other_file != filename and other_type == claim_type:
                        rivals.setdefault((filename, folded), []).append(other_common)

    for finding in report.findings:
        if finding.check not in reviewable:
            continue
        evidence = ""
        try:
            mine = store.read_json(store.dir_for(finding.atom_type) / finding.file)
            if finding.check == "shared-alias":
                key = (finding.file, topics.name_key(finding.value))
                rival = next(iter(rivals.get(key, [])), None)
                theirs = _atom_by_common(finding.atom_type, rival) if rival else None
            else:
                theirs = _atom_by_common(finding.atom_type, finding.value)
            if isinstance(mine, dict) and theirs:
                evidence = corroboration(finding.atom_type, mine, theirs)
        except Exception as e:  # noqa: BLE001 - evidence is a convenience
            log.debug("no corroboration for %s: %s", finding.file, e)
        if decide is not None:
            # **An external adjudicator answers alone: no fallback.** Letting the
            # string rules fill in where it abstained put three rows in a
            # `--via wikipedia` file reading `same` under the note "resolves to a
            # disambiguation page" — Wikipedia had declined *because* the name
            # was ambiguous, and the heuristic overrode the refusal while
            # keeping its wording, so the row looked adjudicated by a source
            # that had in fact said no. Two of the three were `Apollonia`,
            # which names half a dozen cities; applying them would have
            # declared one generic name identical to two of them and then
            # aborted the whole review on the chained-alias guard.
            #
            # An abstention is information. It stays `?`, carrying the reason.
            verdict, why = decide(finding)
        elif finding.check == "shared-alias":
            verdict, why = suggest_shared_verdict(finding.subject, finding.value)
        else:
            verdict, why = suggest_verdict(finding.subject, finding.value)
        lines.append("\t".join([verdict or "?", finding.atom_type, finding.file,
                                finding.value, finding.subject, evidence, why]))
        written += 1
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return written


def read_review(path) -> list:
    """Parse an edited review file into `(verdict, atom_type, file, alias, subject)`.

    **The subject column is load-bearing and was being discarded.** An earlier
    version returned only the first four fields and `apply_review` then
    reconstructed the subject from the *filename slug* — which is lowercased and
    stripped of diacritics and punctuation by `store.slugify`. That wrote
    identity claims for names no atom has:

        almuizz lidn allh = Al-Muʿizz li-Dīn Allāh
        ramses iii        = Ramesses III

    and since the first name on an alias line is the canonical spelling the work
    list emits, it would also have had future runs generate `ramses iii`. The
    atom's real `name.common` is right there in column five.
    """
    rows, unknown = [], []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip() if raw.lstrip().startswith("#") else raw
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("\t")]
        if len(parts) < 5:
            continue
        if parts[0] not in ("same", "wrong", "skip", "?"):
            # **A typo must not silently discard a verdict.** A real review
            # came back with two rows reading `sane`, and the old code dropped
            # them without a word — the reviewer had adjudicated them and would
            # have been told fewer rows were processed than they sent, with no
            # hint which two went missing or why.
            unknown.append((parts[0], parts[2] if len(parts) > 2 else "?"))
            continue
        rows.append(tuple(parts[:5]))
    if unknown:
        raise ValueError(
            "Unrecognised verdict(s) — nothing was applied, because a typo "
            "must not silently discard a decision:\n"
            + "\n".join(f"  {verdict!r} on {where}" for verdict, where in unknown)
            + "\nUse one of: same, wrong, skip, ?")
    return rows


def apply_review(rows, aliases_path=None, *, dry_run: bool = True) -> dict:
    """Act on adjudicated findings. Dry by default, like every other verb here.

    `wrong` edits an atom, which is the only destructive thing in this module,
    so it goes through `store` and re-validates before writing — a content fix
    must not be the thing that makes an atom unreadable.
    """
    if aliases_path is None:
        # The alias file in force, never the pack's own path behind an
        # override's back.
        aliases_path = alias_files.aliases_path()
    same = [r for r in rows if r[0] == "same"]
    wrong = [r for r in rows if r[0] == "wrong"]
    outcome = {"same": len(same), "wrong": len(wrong), "dry_run": dry_run,
               "alias_lines": [], "edited": [], "skipped": [], "unusable": []}

    def line_for(row):
        """`<subject> = <alias>`, using the atom's own name, never the slug."""
        _verdict, _atom_type, _filename, alias, subject = row
        return f"{subject} = {alias}" if subject and alias else None

    if dry_run:
        outcome["alias_lines"] = [line for line in map(line_for, same) if line]
        outcome["unusable"] = [r[2] for r in same if line_for(r) is None]
        # **`wrong` is the only verdict that deletes, and it was the only one a
        # dry run did not show.** `same` printed the exact alias lines it would
        # append while `wrong` printed a count, so the destructive half of the
        # review was the unpreviewable half — and `data/` has no undo. A
        # pre-filled verdict makes that worse: rows nobody read can still edit
        # atoms. Report each removal as `file: alias`.
        outcome["would_edit"] = [f"{filename}: {alias}"
                                 for _v, _t, filename, alias, _s in wrong]
        return outcome

    from universe_builder import duplicates as dup

    outcome["unusable"] = [r[2] for r in same if line_for(r) is None]
    lines = [line for line in map(line_for, same) if line]
    if lines and aliases_path:
        # Adjudicated, not derived: every row here carries a verdict a human
        # set or `--via` pre-filled and a human accepted, so these lines are
        # evidence in a way `duplicates --apply`'s own output is not.
        dup._merge_into_alias_file(
            Path(aliases_path), lines,
            header="# Added by `universe_builder lint --apply-review`: adjudicated identity.")
        outcome["alias_lines"] = lines

    for _verdict, atom_type, filename, alias, _subject in wrong:
        path = store.dir_for(atom_type) / filename
        if not path.exists():
            log.warning("%s is gone; nothing to edit", filename)
            continue
        payload = store.read_atom(path, atom_type)
        name = payload.get("name") or {}
        before = list(name.get("otherNames") or [])
        after = [n for n in before
                 if topics.name_key(n or "") != topics.name_key(alias)]
        if after == before:
            continue
        name["otherNames"] = after
        try:
            _validate_atom(atom_type, payload)
        except Exception as e:  # noqa: BLE001
            # Refuse rather than write something unreadable. An atom that does
            # not validate was already broken before this edit, and writing it
            # back would make this command the apparent culprit.
            log.error("%s does not validate after the edit; left untouched: %s",
                      filename, str(e)[:120])
            outcome.setdefault("skipped", []).append(filename)
            continue
        store.write_json(path, payload)
        outcome["edited"].append(filename)
    return outcome


def _validate_atom(atom_type: str, payload) -> None:
    """Re-validate before writing, against whichever stage shape matches.

    Reads the *spec* from the active pack rather than `atom_mapping`, because
    only the spec carries `stages` — an expanded city must validate as
    `CityExpanded`, and checking it against the seed `City` would reject a
    perfectly good atom (or, worse, pass a bad one).
    """
    from universe_builder.registry import active_pack

    spec = next((s for s in active_pack().all_specs() if s.name == atom_type), None)
    if spec is None:
        return
    for predicate, model in spec.stages or ():
        if predicate(payload):
            model.model_validate(payload)
            return
    spec.model.model_validate(payload)


# ---------------------------------------------------------------------------
# Corroborating evidence: do the two atoms agree about place and date?
# ---------------------------------------------------------------------------

#: Compared **field to matching field**, never one atom's museum against the
#: other's excavation site — that mismatch made `Sphinx of Taharqo` look 767km
#: from `Sphinx of Taharqa` when it was comparing `currentLocation` to
#: `discoveryLocation`. Which fields those are is the domain's knowledge, so
#: the pack declares them (`DomainPack.coordinate_paths` / `date_paths`,
#: validated at registration) and this reads them.
def _evidence_paths(atom_type: str):
    pack = active_pack()
    return (tuple(p.split(".") for p in pack.coordinate_paths.get(atom_type, ())),
            tuple(p.split(".") for p in pack.date_paths.get(atom_type, ())))


def _dig(node, *path):
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _signed_year(date) -> "int | None":
    """A year on one axis, so 50 BCE and 50 CE are a century apart, not equal.

    **An absent era makes the year incomparable, not CE.** `Date.era` is
    `Optional`, and defaulting a missing one to CE turned `1250 (era unknown)`
    against `1250 BCE` into a confident "DISAGREES: 2500y" — a fabricated
    disagreement that is entirely an artifact of a missing field, in the one
    column a reader trusts at a glance. Unknown is reported as unknown; the
    module already says "no comparable place or date" rather than guessing, and
    this is the same rule one level down.
    """
    if not isinstance(date, dict) or not isinstance(date.get("year"), int):
        return None
    era = date.get("era")
    if era not in ("BCE", "CE"):
        return None
    return -date["year"] if era == "BCE" else date["year"]


def _haversine_km(a, b) -> float:
    import math

    (lat1, lon1), (lat2, lon2) = a, b
    radius = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    h = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return 2 * radius * math.asin(math.sqrt(h))


def _coord(payload, *path):
    loc = _dig(payload, *path)
    if isinstance(loc, dict) and isinstance(loc.get("latitude"), (int, float)) \
            and isinstance(loc.get("longitude"), (int, float)):
        return (loc["latitude"], loc["longitude"])
    return None


def corroboration(atom_type: str, a, b) -> str:
    """Whether two atoms agree about place and date, as a short evidence string.

    **Diagnostic, not decisive, and the asymmetry is the point.** Material
    disagreement is strong evidence of two different subjects — `Battle of
    Navarino` and `Battle of Pylos` sit 6km apart in the same bay and are 2,252
    years apart, which only the date reveals. Agreement proves much less,
    because *containment* shares both: `Book of the Dead` and `Papyrus of Ani`
    are 0km and 0 years apart because one is a copy of the other, exactly as
    `Karnak` sits inside `Thebes`. So this goes in front of a human rather than
    deciding for them.
    """
    parts, disagree = [], False
    coordinate_paths, date_paths = _evidence_paths(atom_type)
    for path in coordinate_paths:
        ca, cb = _coord(a, *path), _coord(b, *path)
        if ca and cb:
            distance = _haversine_km(ca, cb)
            parts.append(f"{'.'.join(path)}~{distance:.0f}km")
            disagree = disagree or distance > 25
    for path in date_paths:
        ya, yb = _signed_year(_dig(a, *path)), _signed_year(_dig(b, *path))
        if ya is not None and yb is not None:
            gap = abs(ya - yb)
            parts.append(f"{'.'.join(path)}~{gap}y")
            disagree = disagree or gap > 50
    if not parts:
        return "no comparable place or date"
    return ("DISAGREES: " if disagree else "agrees: ") + ", ".join(parts)


#: Tokens that carry no identity, so `Temple of Edfu` and `Temple of Horus at
#: Edfu` compare as {temple, edfu} against {temple, horus, edfu}.
_NOISE_WORDS = frozenset("the of at in by a an and from".split())

#: Character-similarity floor for calling two names one thing without a human.
#: Tuned against 33 hand-adjudicated findings: 0.80 admits
#: `Konstantinos Karamanlis` / `Kostas Karamanlis` — an uncle and his nephew,
#: 338km and 49 years apart — because the shared surname dominates. 0.88 is the
#: lowest value that excludes it while keeping every true match the subset rule
#: misses (`Ramses III`/`Ramesses III` at 0.91, `Sphinx of Taharqo`/`Taharqa`
#: at 0.94, `Ramesseum Medical Papyrus`/`Papyri` at 0.94).
_SIMILAR_ENOUGH = 0.88


def _fold_name(text: str) -> str:
    """Lowercase, strip diacritics and punctuation. `Al-Muʿizz li-Dīn Allāh`
    and `Al-Mu'izz li-Din Allah` are the same string once folded.

    Only `suggest_verdict`'s character similarity reads this now: it turns
    each punctuation mark into its own space where `topics.name_key` collapses
    a run into one (`St. Mark` is `st  mark` here, `st mark` there), and the
    ratio sees the difference, so swapping it would move verdicts near the
    0.88 threshold. Word sets come from `name_key` (identical once split).
    """
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9 ]", " ", stripped.lower())


def _name_tokens(text: str) -> set:
    """Meaningful words, singular-ised, so `Obelisks` matches `obelisk`."""
    return {re.sub(r"(ies|es|s)$", "", token)
            for token in topics.name_key(text).split()
            if token and token not in _NOISE_WORDS}


#: A standalone Roman numeral, or a bare number. In a ruler's name these carry
#: nearly all the identity and almost none of the characters, which is exactly
#: backwards from what string similarity measures.
_NUMERAL = re.compile(
    r"^(?:\d+|(?=[mdclxvi])m*(?:c[md]|d?c*)(?:x[cl]|l?x*)(?:i[xv]|v?i*))$")


def _regnal_numbers(text: str) -> set:
    """Roman numerals and bare numbers in a name, as a set."""
    return {token for token in topics.name_key(text).split() if _NUMERAL.match(token)}


def suggest_verdict(subject: str, alias: str):
    """`("same", why)` when two names are safely one topic, else `(None, "")`.

    Two signals, both derived from a sample of hand-adjudicated findings
    rather than invented:

    * **Token containment.** One name's meaningful words being a subset of the
      other's means one is a fuller form of the same thing —
      `Temple of Edfu` inside `Temple of Horus at Edfu`,
      `Meidum Geese` inside `Meidum Geese Fresco`. Word order does not matter,
      which is what makes `King List of Abydos` and `Abydos King List` match.
    * **Character similarity at or above 0.88**, which catches spelling
      variants containment cannot: `Ramses III`/`Ramesses III`,
      `Taharqo`/`Taharqa`, `Papyrus`/`Papyri`.

    **It only ever suggests `same`, never `wrong`**, and that asymmetry is
    deliberate. Every genuinely different pair in the sample — `Battle of
    Navarino`/`Battle of Pylos`, `Suez`/`Clysma`, `Nefertiti`/`Neferneferuaten`
    — is *also* unlike in name, so silence is the right output for them rather
    than a confident refusal. And the failure it must avoid is the expensive
    one: asserting that two different subjects are one, which writes a false
    identity into the alias file.

    On that sample every suggestion it made was correct, and it removed over
    half the manual work — but the sample was small. Treat the suggestions as
    a fast path to skim, not as decided.
    """
    import difflib

    # **Regnal numbers decide identity and must match exactly.** They are one
    # or two characters in a name that is otherwise identical, so similarity
    # scores them as noise: `Ramses II` and `Ramses III` are 95% alike, as are
    # `Ptolemy II`/`Ptolemy III` and `Thutmose I`/`Thutmose IV`, and
    # `Cleopatra VII`/`Cleopatra IV` reach 88%. Every one of those is two
    # different rulers, and every one exists in this corpus. Containment is
    # just as wrong here — `Ptolemy` sits inside `Ptolemy II` while naming
    # someone else entirely.
    #
    # The 33-row sample that set the similarity threshold contained
    # `Ramses III`/`Ramesses III` — a *matching* numeral — and no differing
    # pair, so it could not have caught this. That is the sample being small,
    # not the signal being wrong.
    if _regnal_numbers(subject) != _regnal_numbers(alias):
        return None, ""

    mine, theirs = _name_tokens(subject), _name_tokens(alias)
    if mine and theirs and (mine <= theirs or theirs <= mine):
        return "same", "one name contains the other"
    score = difflib.SequenceMatcher(
        None, _fold_name(subject).strip(), _fold_name(alias).strip()).ratio()
    if score >= _SIMILAR_ENOUGH:
        return "same", f"names {score:.0%} alike"
    return None, ""


#: An atom whose subject is an *action on* a thing rather than the thing:
#: `Reign of Khafre`, `Construction of Djoser's Step Pyramid`. The distinction
#: is the Karnak rule again — a reign is not the king and a construction is not
#: the building — and it is the one case here mechanical enough to pre-fill.
_WRAPPER = re.compile(
    r"^(reign|construction|foundation|founding|building|destruction|discovery)\s+of\s+",
    re.IGNORECASE)


def rivals_distinct_verdicts(findings, page_id, owns=None, answered=None) -> Dict[tuple, tuple]:
    """`{(file, alias): ("wrong", why)}` where the contestants are *known* to be
    different things, so the contested name uniquely identifies none of them.

    `Bab al-Nasr` and `Bab al-Futuh` both claim `Gate of Victory`; `Red Pyramid`
    and `Bent Pyramid` both claim `Pyramid of Sneferu at Dahshur`. Two Cairo
    gates, two Sneferu pyramids at Dahshur — in each case the phrase describes
    both and names neither, which is why it resolves to nothing today.

    Distinctness is established by page identity, not inferred: every contesting
    subject must resolve to a Wikipedia article and those articles must differ.
    If two of them land on one article the corpus has a duplicate, not an
    ambiguity, and `universe_builder duplicates` is what adjudicates that — this returns
    nothing rather than retiring a name over it.

    `owns(subject, alias)` is **tri-state, and that is load-bearing**: `True`
    the article states the name, `False` the article was read and does not,
    `None` there is no article text to read. Only `False` permits a deletion.

    A two-state veto fails *open*: "no evidence" and "evidence of absence"
    collapse into the same answer, so the protection disappears exactly when
    nothing is known. `subject_leads` returns `""` for an entire batch of twenty
    when one request fails, so a single transient API error would have disarmed
    the veto for twenty subjects and deleted their contested names with no
    evidence whatsoever — silently, since a missing lead reads the same as a
    lead that omits the name.

    It is what makes the rule safe. Without it the rule
    strips the name from *every* contestant, including the one that really has
    it: `Cleopatra VII Thea Philopator` would lose `Cleopatra Philopator`, and
    `Ramesses II` would lose `Usermaatre Setepenre`. With it the same pass still
    strips both from `Cleopatra IV` and from `Ramesses IX`, whose prenomen is
    *Neferkare* Setepenre — so the contested name ends up assigned rather than
    destroyed. Measured on the adjudicated rows the veto halved the errors.

    `answered(name)` says whether the source *answered* for a name at all. A
    resolver leaves out a name it could not look up (the `identity_resolvers`
    contract), and `page_id` reads both that and "no such page" as `None`; for
    a contestant either way abstains, but for the alias "no page" means it is
    nobody's own article while "no answer" means that is unknown, so an
    unanswered alias abstains too. None: every name was
    answered, as a caller with no lookups to lose may assume.

    **Only for rows the identity rule could not answer.** Where the alias itself
    resolves to one contestant's article that contestant owns it, and saying
    `wrong` there would delete a real name. Scored on 33 adjudicated rows this
    agreed 31 times; both misses dropped a genuine alias rather than asserting a
    false identity, which is the direction that loses information instead of
    inventing it.
    """
    grouped: Dict[str, list] = {}
    for finding in findings:
        if finding.check == "shared-alias":
            grouped.setdefault(topics.name_key(finding.value), []).append(finding)

    decided: Dict[tuple, tuple] = {}
    for _alias, group in grouped.items():
        subjects = {topics.name_key(f.subject): f.subject for f in group}
        if len(subjects) < 2:
            continue
        ids = {key: page_id(name) for key, name in subjects.items()}
        if any(value is None for value in ids.values()):
            continue                      # distinctness not established
        if len(set(ids.values())) != len(ids):
            continue                      # two subjects, one article: a duplicate
        if answered is not None and not answered(group[0].value):
            continue                      # whose article the alias is: unknown
        alias_id = page_id(group[0].value)
        others = sorted(subjects.values())
        for finding in group:
            if alias_id is not None and alias_id == ids[topics.name_key(finding.subject)]:
                continue                  # the alias is this subject's own article
            if owns is not None:
                states = owns(finding.subject, finding.value)
                if states is not False:
                    continue              # states the name, or nothing is known
            rivals = [n for n in others if topics.name_key(n) != topics.name_key(finding.subject)]
            decided[(finding.file, finding.value)] = (
                "wrong",
                f"contested by {', '.join(rivals)}, which Wikipedia holds as "
                f"separate articles — the name describes both and identifies neither")
    return decided


def wrapper_claims_its_subjects_name(subject: str, alias: str):
    """`("wrong", why)` when an *action on* a thing claims a name **the thing
    itself declares**. Otherwise `(None, "")`.

    The corpus already knows who owns these names, and asking it beats guessing
    at the strings. `Reign of Khafre` claims `Khafra`; the person atom `Khafre`
    lists `Khafra` among its own `otherNames`, so the name is the pharaoh's and
    the reign is borrowing it. A reign is not the king, exactly as Karnak is not
    Thebes.

    This supersedes a token-overlap test that could not see past spelling:
    `Khafra` and `Khafre` share no token, so the rule stayed silent on the very
    case it existed for. Going through the owner's declared names fixes that,
    because the owner declares both spellings.
    """
    match = _WRAPPER.match(subject or "")
    if not match:
        return None, ""
    stem = topics.name_key(_WRAPPER.sub("", subject))
    if not stem:
        return None, ""
    wanted = topics.name_key(alias)
    for spec in active_pack().atoms:
        for path, payload in store.iter_payloads(spec.name):
            if not isinstance(payload, dict):
                continue
            name = payload.get("name")
            if not isinstance(name, dict):
                continue
            common = name.get("common") or ""
            if topics.name_key(common) != stem:
                continue
            declared = {topics.name_key(n) for n in
                        [common] + list(name.get("otherNames") or []) if n}
            if wanted in declared:
                return "wrong", (f"{common!r} ({spec.name}) declares this name; "
                                 f"an action on a thing is not the thing")
    return None, ""


def suggest_shared_verdict(subject: str, alias: str):
    """Pre-fill for a `shared-alias` row. Deliberately **never suggests `same`**.

    `suggest_verdict` asks "are these two names one thing?", and its containment
    and similarity rules are calibrated for check 1, where the rival atom really
    *is called* the alias. Here nothing is called it, and containment is the
    dominant failure rather than a useful signal: run over this check's findings
    it proposed `same` for `Construction of Djoser's Step Pyramid` claiming
    `Step Pyramid of Djoser` (an event and the artifact it produced) and for
    `Sinai and Palestine Campaign` claiming `Sinai Campaign` (1915 and 1956).
    Accepting either would write a permanent identity line for two things that
    are not one.

    So the only pre-fill offered is the safe direction. `wrong` merely drops the
    claim from this atom, leaving every rival to be judged on its own row, and
    it is almost always right when the subject is an action on the thing the
    alias names. Everything else stays `?` — an unfilled row costs a reading; a
    wrongly filled one costs an identity the corpus can no longer express.
    """
    verdict, why = wrapper_claims_its_subjects_name(subject, alias)
    if verdict:
        return verdict, why
    return None, "containment is the usual trap here; check which subject the name really belongs to"


def _atom_by_common(atom_type: str, name: str):
    wanted = topics.name_key(name)
    for path, payload in store.iter_payloads(atom_type):
        if isinstance(payload, dict) and topics.name_key(
                (payload.get("name") or {}).get("common") or "") == wanted:
            return payload
    return None


CHECKS = {"overbroad-alias": overbroad_aliases,
          "shared-alias": shared_aliases,
          "topic-drift": topic_drift,
          "alias-contradiction": alias_contradictions,
          "era-contradiction": era_contradictions,
          "self-negation": self_negations}


def run(checks=None, atom_types=None) -> Report:
    """Run every check over every base atom type. Pure read.

    Unknown names are rejected up front rather than quietly doing nothing. A
    linter invoked with a typo — `--check era-contradictions`, `--atom citys` —
    that answers "No findings" and exits 0 is worse than one that errors: it
    tells you the corpus is clean when it checked nothing at all.
    """
    known_types = [spec.name for spec in active_pack().atoms]
    wanted = list(checks) if checks else list(CHECKS)
    types = list(atom_types) if atom_types else known_types

    unknown_checks = [c for c in wanted if c not in CHECKS]
    if unknown_checks:
        raise KeyError(
            f"Unknown check(s) {unknown_checks}. Known: {', '.join(sorted(CHECKS))}")
    unknown_types = [t for t in types if t not in known_types]
    if unknown_types:
        raise KeyError(
            f"Unknown atom type(s) {unknown_types}. Known: {', '.join(known_types)}")

    report = Report()
    #: A file-level check runs per atom type like every other, and returns the
    #: same finding each time. Collapsing on identity — not on atom type, which
    #: differs across the copies — keeps one defect reported once. Per-atom
    #: findings carry a unique `file`, so nothing real is merged away.
    seen: set = set()

    def fresh(found):
        for finding in found:
            key = (finding.check, finding.file, finding.subject, finding.value)
            if key in seen:
                continue
            seen.add(key)
            yield finding

    for name in wanted:
        for atom_type in types:
            try:
                report.findings.extend(fresh(CHECKS[name](atom_type)))
            except Exception as e:  # noqa: BLE001 - one bad type costs one type
                # A file-level check fails identically for every atom type.
                # Deduped on the message so one broken file reads as one
                # problem — the count still gates `is_clean`, which is what
                # stops a failed check being reported as a clean corpus.
                message = str(e)
                if ("error", message) in seen:
                    continue
                seen.add(("error", message))
                log.error("lint %s on %s failed: %s", name, atom_type, e)
                report.errors.append(f"{name}: {e}")
    return report
