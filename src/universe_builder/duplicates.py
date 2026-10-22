"""One subject, two files: finding duplicate atoms and retiring the loser.

**Why this is not covered by "one file per atom".** `tests/test_corpus.py`
asserts that no two files share a content id, and duplicates do not: the ids
were hashed from the two *spellings* two models happened to use.
`Founding of Memphis` (gpt-4o-mini) and `Foundation of Memphis` (gpt-5-nano)
are one event in two files, with two ids, passing every invariant the project
has. Egyptian History keeps one atom per model by design, the two models
spelled the topic differently, and `group_synonyms` joins only names that
*declare* equivalence or that the alias file joins by hand. Nothing joined
these, so both spellings entered the work list and both were generated.

Detection comes from `topics.duplicate_atoms`, which reads the names the atoms
declare about themselves — including the case that makes this tractable, where
the newer atom lists the older atom's spelling in its own `name.otherNames`.

**Retiring a file is only half the repair, and the other half is the part that
is easy to forget.** The losing spelling is still listed in a stored History
atom. Remove only the file and `topics.needing_create` sees a listed topic with
no atom and regenerates it, recreating the duplicate at the next create run and
charging for the privilege. So an alias line is written at the same time,
joining the two spellings into one declared topic whose survivor is the atom
that was kept. Both halves are reversible: the alias file is in git, and the
retired atom moves to `_duplicates/` rather than being unlinked, because
`data/` is not in git and there is no undo.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from universe_builder import alias_files, store, topics
from universe_builder.registry import active_pack

log = logging.getLogger(__name__)

#: The numeric family in a model version string: `gpt-5.6-luna` -> 5.6,
#: `gpt-5-nano` -> 5.0, `gpt-4o-mini` -> 4.0, `gpt-6-astra` -> 6.0. The `o` in
#: `4o` is a model name, not a decimal, so only a digit-dot-digit run counts.
_VERSION = re.compile(r"(?:^|[^0-9])(\d+(?:\.\d+)?)")


def model_rank(model_version: str) -> Optional[float]:
    """The model family's number, or None when there is nothing to compare.

    Deliberately coarse. It answers "which of these two model families is
    later", which is the only judgement being asked for, and it declines to
    answer anything else — a tie between two models of one family is not
    resolved here, because nothing in a filename says which of two gpt-5
    variants produced the better atom.
    """
    match = _VERSION.search(model_version or "")
    return float(match.group(1)) if match else None


@dataclass(frozen=True)
class Resolution:
    """One duplicate pair and what to do about it."""

    atom_type: str
    #: The folded name both atoms declare.
    name: str
    keep: Path
    retire: List[Path]
    #: Spellings for the alias line, kept-first.
    spellings: List[str]
    reason: str


@dataclass
class Plan:
    resolutions: List[Resolution] = field(default_factory=list)
    #: Pairs this cannot decide, with why. Never acted on automatically.
    undecided: List[dict] = field(default_factory=list)
    #: Pairs a caller ruled out by `--except`. Reported, never acted on.
    skipped: List[dict] = field(default_factory=list)

    def is_clean(self) -> bool:
        return not self.resolutions and not self.undecided and not self.skipped

    def alias_lines(self) -> List[str]:
        """The lines to append to the domain's alias file, kept spelling first.

        One line per *survivor*, not per pair. An atom can win against more
        than one rival — `Amarna Boundary Stelae` beat both
        `Akhenaten's Boundary Stelae` and `Tell el-Amarna Boundary Stelae`,
        which are two pairs — and emitting a line each would put two
        declarations of the same scope in the file both claiming the survivor's
        name. `group_occurrences` treats a name two declarations claim as
        ambiguous and leaves it ungrouped, so the very lines written to join
        the spellings would stop the work list joining them.
        """
        merged: Dict[str, List[str]] = {}
        for resolution in self.resolutions:
            if not resolution.spellings:
                continue   # an asserted line already covers this pair; nothing to write
            kept, *retired = resolution.spellings
            names = merged.setdefault(kept, [])
            for name in retired:
                if name not in names:
                    names.append(name)
        return [" = ".join([kept, *retired]) for kept, retired in merged.items()]

    def summary(self) -> str:
        parts = [f"{len(self.resolutions)} pair(s) resolvable"]
        if self.undecided:
            parts.append(f"{len(self.undecided)} undecided")
        if self.skipped:
            parts.append(f"{len(self.skipped)} excluded")
        return ", ".join(parts)


def load_distinct(path: Optional[Path] = None) -> tuple:
    """Parse the not-duplicates file into sets of folded names. Missing = none.

    The mirror of `topics.load_aliases`. That file declares "these spellings are
    one topic"; this declares "these atoms are different things, whatever their
    names suggest" — because a shared name is evidence and not proof, and in
    this corpus the evidence is wrong about half the time. Two pharaohs share a
    prenomen (`Wahibre`), two Cairo gates share an epithet (`Gate of Victory`),
    Nicaea and Nice share a Greek root.

    Format is `A != B`, any number of names per line. An unreadable file is an
    error but not a stop, exactly as for the alias file: running without it is
    what the command did before it existed, so a bad file degrades to yesterday.
    """
    if path is None:
        path = _pack_distinct_path()
    if path is None:
        return ()
    try:
        text = Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return ()
    except (OSError, UnicodeDecodeError) as e:
        log.error("Cannot read %s (%s); every pair stays eligible", path, e)
        return ()

    claims = []
    for declaration in alias_files.parse_distinct_text(text):
        names = frozenset(topics.name_key(n) for n in declaration.names)
        if len(names) > 1:
            claims.append(names)
    return tuple(claims)


def _declared_distinct(paths: List[Path], atom_type: str, claims: tuple,
                       listed: Optional[dict] = None, declared=None) -> bool:
    """Whether a human has declared this component's atoms to be different.

    Matches on the atoms' own common names, folded. A component counts as
    declared when every atom in it is named by one claim — a claim covering only
    part of a three-way component says nothing about the rest, so it does not
    suppress it.
    """
    if not claims:
        return False
    names = {topics.name_key(_spelling_of(p)) for p in paths}
    if len(names) < len(paths):
        # Two atoms in the component share a common name, so the set collapsed
        # and a claim naming that one name against a third atom would cover
        # them both: `Kos != Catania` declared the two Catania atoms distinct
        # from each other. Same rule as `_declared_same`.
        #
        # The common name cannot tell them apart, but their *generation
        # spellings* can — the listed spelling that hashes to each file, which
        # is each atom's identity (the alias-line rule). Two cities both
        # called `Abydos`, stored under `Abydos` and `Abydos (Hellespont)`,
        # are declared distinct by a line naming exactly those two; a line
        # naming only the shared name still says nothing.
        names = _generation_keys(paths, atom_type, listed=listed,
                                 declared=declared() if callable(declared) else declared)
        if names is None:
            return False
    return any(names <= claim for claim in claims)


def _generation_keys(paths: List[Path], atom_type: str, listed: Optional[dict] = None,
                     declared: Optional[List[str]] = None):
    """The folded generation spelling of every atom, or None if any lacks one.

    A seed lists what the work list *tried*; the spelling an atom was generated
    under may instead be the survivor an alias line chose — Byzantine History
    lists bare `Abydos`, and the scoped line makes it `Abydos (Hellespont)`. So
    the candidates are the listed spellings and every alias-line name, and the
    one that hashes to the file is the atom's.

    `listed` (`topics.listed_spellings(atom_type)`) and `declared` (every name
    in the alias file in force) are read here when not given; `plan` reads
    them once and passes them in.
    """
    if listed is None:
        listed = topics.listed_spellings(atom_type)
    if declared is None:
        declared = [name for group in topics.load_aliases() for name in group.spellings]
    keys = set()
    for path in paths:
        try:
            padded_id = store.AtomName.decompose_filename(path.name).padded_id
        except ValueError:
            return None
        spelling = next((s for s in (*listed.get(padded_id, ()), *declared)
                         if store.atom_id(atom_type, s) == padded_id), None)
        if spelling is None:
            return None
        keys.add(topics.name_key(spelling))
    return keys if len(keys) == len(paths) else None


def _claim_keys(names) -> set:
    """Every key a set of alias names really claims, parentheticals included.

    **The safeguards must fold names exactly as the grouping machinery does.**
    `topics.alias_keys` splits a trailing parenthetical — `Foo (Bar)` declares
    `foo bar`, `foo` *and* `bar` — while `topics.name_key` returns only the whole
    string. Comparing with `name_key` made every overlap check blind to the split:
    a file holding `Foo (Bar) = Qux` looked like it claimed `{foo bar, qux}`, so
    a new `Bar = Baz` line showed no overlap, was appended, and produced two
    same-scope declarations claiming `bar`. `group_occurrences` then leaves
    every entry matching `bar` ungrouped, the retired spelling stops resolving,
    and the next create run regenerates the atom that was just retired. The file
    loader even warns about it — the safeguard simply could not see it.
    """
    return {key for name in names for key in topics.alias_keys(name)}


def existing_alias_claims(path: Optional[Path] = None) -> List[set]:
    """The folded names each *global* alias line already claims.

    Read at plan time so a resolution whose line would chain two existing
    declarations can be refused before anything is retired. Scoped lines are
    excluded: a global line never merges into one, so it cannot chain through it.
    """
    if path is None:
        path = _pack_aliases_path()
    if path is None or not Path(path).exists():
        return []
    try:
        file = alias_files.AliasFile.read(path)
    except (OSError, UnicodeDecodeError):
        return []
    return [_claim_keys(d.names) for d in file.global_declarations()]


def _refused_pairs(aliases_path: Path) -> Dict[frozenset, str]:
    """The pairs `not_duplicates.txt` (beside `aliases_path`) declares distinct,
    keyed by each name's whole folded spelling, with the line that says so.

    Whole spellings only, unlike `lint.alias_contradictions`, which pairs every
    part: that is right for a report and wrong for a refusal, because
    `Thebes (Luxor) != Thebes (Boeotia)` would then forbid every line joining
    `Thebes` to `Luxor`.
    """
    path = alias_files.not_duplicates_path(aliases_path)
    if not path or not path.exists():
        return {}
    refused: Dict[frozenset, str] = {}
    for declaration in alias_files.NotDuplicatesFile.read(path).declarations:
        keys = [_whole_key(name) for name in declaration.names]
        for i, left in enumerate(keys):
            for right in keys[i + 1:]:
                if left and right and left != right:
                    refused[frozenset((left, right))] = declaration.body
    return refused


def _whole_key(name: str) -> str:
    """`name` folded whole: a quoted name without its quotes, never split."""
    return topics.name_key(topics.unquote(name) if topics.is_quoted(name) else name)


def _refusal_crossed(names, refused: Dict[frozenset, str]) -> Optional[str]:
    """The `!=` line two of `names` would cross by sharing one alias line."""
    keys = sorted({_whole_key(n) for n in names} - {""})
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            line = refused.get(frozenset((left, right)))
            if line:
                return line
    return None


def _merge_into_alias_file(path: Path, lines: List[str],
                           header: Optional[str] = None) -> List[str]:
    """Add alias lines, merging with any global line that already shares a name.

    **Appending blindly is wrong across runs**, and the failure is silent. A
    second `--apply` that retires another rival of an already-processed survivor
    writes a second line starting with the same name, and two same-scope
    declarations claiming one key is the ambiguity case: `group_occurrences`
    leaves every matching entry ungrouped, so a topic that resolved yesterday
    drops out of the work list today and `status --coverage` reports a gap.
    Observed exactly that way — `Ottoman conquest of Egypt` gained a second line
    on the second run and `Conquest of Egypt by the Ottoman Empire` went missing.

    Merging only ever touches lines in **global** scope. A scoped line
    (`[Egyptian History]`) is a deliberately narrower claim, and folding a global
    one into it would widen it silently; those are left alone and the new line is
    appended, which is the pre-existing behaviour for a genuinely new claim.

    **A merge never crosses a `not_duplicates.txt` refusal.** A line joins any
    group sharing one of its keys, a bracketed name's parts included, so the
    ruled `Irbid (Arbela) = Irbid` joined `Erbil = Arbela` through `arbela`
    against `Irbid != Erbil`, and the shipped Irbid resolved to Erbil.
    Such a line raises and nothing is
    written; scope it, or quote the bracketed name, by hand.
    """
    file = alias_files.AliasFile.read(path)
    refused = _refused_pairs(path)
    rows = file.rows

    # Which rows are alias declarations in global scope, and what they claim.
    claims: Dict[int, set] = {d.index: _claim_keys(d.names) for d in file.global_declarations()}

    # Lines not yet in the file, each with the keys it claims. New lines must
    # be able to merge with *each other*, not only with what is already on
    # disk: `lint --apply-review` hands over a whole batch at once, and three
    # `Apollonia` rows and two `Egyptian Revolution of 2011` rows arrived
    # together. Checking only the file let each of them see no conflict and
    # land beside the others — same-scope declarations claiming one key, which
    # is the ambiguity this function exists to prevent.
    pending: List[list] = []          # [[names...], keys]

    def merge_names(into, names):
        # A name is already on the line when its own key is, not when one of
        # its parenthetical parts is: `Arsinoe (Crocodilopolis)` shares
        # `crocodilopolis` with `Fayoum = Crocodilopolis` and was dropped,
        # while the caller reported the line written.
        present = {topics.name_key(n) for n in into}
        seen = _claim_keys(into)
        for name in names:
            if topics.name_key(name) not in present:
                present.add(topics.name_key(name))
                seen |= topics.alias_keys(name)
                into.append(name)
        return seen

    for line in lines:
        names = list(alias_files.alias_names(line))
        keys = _claim_keys(names)
        overlapping = [n for n, held in claims.items() if held & keys]
        hits_pending = [p for p in pending if p[1] & keys]
        if len(overlapping) + len(hits_pending) > 1:
            raise ValueError(
                f"the alias line {line!r} overlaps "
                f"{len(overlapping) + len(hits_pending)} existing declarations; "
                f"merging them would join names nothing declared equal. Nothing "
                f"was written and nothing retired.")
        if hits_pending:
            # **Declarations never chain, and a batch is no exception.** Two
            # rows saying `Apollonia = Apollonia in Illyria` and
            # `Apollonia = Apollonia Pontica` are each defensible, and merging
            # them declares Illyria and Pontica — colonies in Albania and
            # Bulgaria — to be one city. That is the Saqqara failure, and
            # arriving in one batch does not make it safe. The reviewer's
            # verdicts are mutually inconsistent, which is something to report
            # rather than average out.
            shared = sorted(hits_pending[0][1] & keys)
            raise ValueError(
                f"{line!r} and {' = '.join(hits_pending[0][0])!r} both claim "
                f"{shared}. Merging them would declare the other names equal, "
                f"which nothing said. Nothing was written — decide which is "
                f"meant and write one line by hand.")
        if overlapping:
            row = overlapping[0]
            merged = list(next(alias_files.parse_alias_text(rows[row])).names)
            crossed = _refusal_crossed(merged + names, refused)
            if crossed:
                raise ValueError(
                    f"{line!r} would join the alias line {rows[row]!r}, but "
                    f"not_duplicates.txt declares {crossed!r}. Nothing was "
                    f"written — scope the line or quote its bracketed name by hand.")
            claims[row] = merge_names(merged, names)
            rows[row] = " = ".join(merged)
            log.info("Merged %r into the existing alias line", line)
            continue
        crossed = _refusal_crossed(names, refused)
        if crossed:
            raise ValueError(
                f"{line!r} joins names not_duplicates.txt declares distinct "
                f"({crossed!r}). Nothing was written.")
        pending.append([names, keys])

    added = [" = ".join(names) for names, _keys in pending]
    body = "\n".join(r for r in rows if r is not None).rstrip("\n")
    added = [a for a in added if a not in rows]
    if added:
        # Back to global scope before appending: the file is stateful, and a
        # trailing `[Greek History]` header would silently scope what follows.
        # The header is the line's provenance, and `topics.load_aliases` reads
        # it. The default says `universe_builder duplicates --apply`, which marks the
        # section **derived** — a record of what this command concluded, worth
        # nothing to the safeguard that later asks whether an identity was
        # declared. A caller that actually adjudicated (`lint --apply-review`,
        # carrying a human's or a model's verdict) passes its own header and
        # the section counts as asserted. Getting this wrong is how a tool
        # comes to vouch for itself.
        note = header or ("# Added by `universe_builder duplicates --apply`: "
                          "one subject, two atoms.")
        body += ("\n\n" + note + "\n[*]\n" + "\n".join(added))
    alias_files.write_atomically(path, body + "\n")
    return lines


def populated_fields(path: Path) -> int:
    """How many leaf values the atom actually fills. The duplicate tie-break.

    `plan` ranks by model family, and where families tie a filename says
    nothing — which left genuine duplicates unretirable, every pair written by
    the same model. "Needs a human" was really "needs a
    tie-break", and this is the one a human would apply anyway: of two atoms
    about the same subject, keep the one that says more.

    Counted as *populated leaves*, not keys: `None` and `""` are absences, and
    an empty list contributes nothing because it is the commonest way this
    corpus records "I did not know". `schema_version` is skipped as bookkeeping
    identical on both sides.

    An unreadable atom counts zero, so a file that cannot be parsed never wins
    a tie against one that can.
    """
    try:
        data = store.read_json(path)
    except OSError:
        return 0
    if not isinstance(data, dict):
        return 0

    def count(node) -> int:
        if isinstance(node, dict):
            return sum(count(v) for k, v in node.items() if k != "schema_version")
        if isinstance(node, list):
            return sum(count(v) for v in node)
        return 0 if node in (None, "") else 1

    return count(data)


def common_name(path: Path) -> str:
    """The atom's own `name.common`, falling back to its filename topic.

    Adjudication compares what the atoms call *themselves*, never the slug: a
    slug is lowercased and stripped of diacritics, so `almuizz_lidn_allh` and
    `almuizz_lidin_allah` look like two mangled strings rather than two
    spellings of Al-Mu\u02bfizz li-D\u012bn All\u0101h, and a model asked to
    judge slugs is being asked the wrong question.
    """
    # A missing or unreadable file falls back to the filename rather than
    # raising: `plan()` and `triage()` are separate passes, and an atom can be
    # retired or moved between them. Losing one name to a slug is a worse
    # answer; aborting the whole triage over it is a worse outcome.
    try:
        data = store.read_json(path)
    except OSError:
        data = None
    if isinstance(data, dict):
        common = ((data.get("name") or {}).get("common") or "").strip()
        if common:
            return common
    try:
        return store.AtomName.decompose_filename(path.name).getTopic()
    except ValueError:
        return path.stem


#: How much of an atom's summary prose a triage record carries (v0.4.0).
TRIAGE_SUMMARY_CHARS = 300


def _read_payload(path: Path):
    try:
        data = store.read_json(path)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _dig(node, dotted: str):
    for key in dotted.split("."):
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def _when(date) -> Optional[str]:
    """`1250 BCE`, or `1250 (era unknown)`; None when there is no year."""
    if not isinstance(date, dict) or not isinstance(date.get("year"), int):
        return None
    era = date.get("era")
    return f"{date['year']} {era}" if era in ("BCE", "CE") else f"{date['year']} (era unknown)"


def _where(place) -> Optional[str]:
    """The named parts of a place in the model's field order (most specific
    first for the shared `Location`: `Karnak, Thebes, Egypt`): every text
    field but its `certainty`. By shape, not by field name, so a pack's own
    place model reads too."""
    if not isinstance(place, dict):
        return None
    parts = [value.strip() for key, value in place.items()
             if key != "certainty" and isinstance(value, str) and value.strip()]
    return ", ".join(dict.fromkeys(parts)) or None


def summary_of(atom_type: str, payload) -> Optional[str]:
    """The first non-empty prose at the pack's `summary_paths` for this type,
    whitespace collapsed and cut to `TRIAGE_SUMMARY_CHARS`. None when the pack
    declares no path for the type or every path is empty."""
    for dotted in active_pack().summary_paths.get(atom_type, ()):
        value = _dig(payload, dotted)
        if isinstance(value, str) and value.strip():
            text = " ".join(value.split())
            if len(text) > TRIAGE_SUMMARY_CHARS:
                text = text[:TRIAGE_SUMMARY_CHARS - 1].rstrip() + "…"
            return text
    return None


def triage_record(atom_type: str, payload, name: str) -> List[str]:
    """What the triage model is shown about one atom, a line each (v0.4.0):
    its common name, other names, year and era at each of the pack's
    `date_paths`, place at each `coordinate_paths` entry, and a summary
    (`summary_paths`). Names alone could not tell a subject from one it
    contains, which shares its name; what each atom says about itself can."""
    lines = [name]
    if not isinstance(payload, dict):
        return lines
    pack = active_pack()
    other = [n.strip() for n in ((payload.get("name") or {}).get("otherNames") or [])
             if isinstance(n, str) and n.strip()]
    if other:
        lines.append("other names: " + "; ".join(other))
    for dotted in pack.date_paths.get(atom_type, ()):
        when = _when(_dig(payload, dotted))
        if when:
            lines.append(f"{dotted}: {when}")
    for dotted in pack.coordinate_paths.get(atom_type, ()):
        where = _where(_dig(payload, dotted))
        if where:
            lines.append(f"{dotted}: {where}")
    summary = summary_of(atom_type, payload)
    if summary:
        lines.append(f"summary: {summary}")
    return lines


def triage_prompt(atom_type: str, left: Path, right: Path) -> str:
    """The user turn for one triage pair: both records and, when the two
    atoms have a comparable place or date, `lint.corroboration`'s line."""
    from universe_builder.lint import corroboration

    a_payload, b_payload = _read_payload(left), _read_payload(right)
    a_lines = triage_record(atom_type, a_payload, common_name(left))
    b_lines = triage_record(atom_type, b_payload, common_name(right))
    text = (f"Subject type: {atom_type}\n"
            f"Record A: {a_lines[0]}\n" + "".join(f"  {line}\n" for line in a_lines[1:])
            + f"Record B: {b_lines[0]}\n" + "".join(f"  {line}\n" for line in b_lines[1:]))
    if a_payload is not None and b_payload is not None:
        evidence = corroboration(atom_type, a_payload, b_payload)
        if evidence != "no comparable place or date":
            text += f"Place and date compared: {evidence}\n"
    return text + "\nAre A and B the same subject?"


def triage(plan_: "Plan", model: str, limit: Optional[int] = None) -> List[dict]:
    """Ask a model whether each undecided component is really one subject.

    **The undecided pile is contaminated, not merely unranked.** `undecided`
    reads as "which of these do I keep", and for some components that is exactly
    what it is. But `group_synonyms` joins on declared names, and a declared name
    is evidence rather than proof, so the pile also holds components that are not
    duplicates at all: it once grouped `Abbas I`, `Abbas II` and
    `Abbas Mirza` as one person, and `Artaxerxes I` (Achaemenid) with
    `Ardashir I` (Sasanian, five centuries later). Ranking those by model family
    would answer a question nobody should be asking.

    So this triages before anyone ranks: every pair in a component is judged
    same-or-different, and the different ones become `!=` lines in
    `not_duplicates.txt`, which `_declared_distinct` then suppresses for good.
    What survives is the genuine duplicates, which is a list worth a human's
    time.

    Each record shows the model what the atom says about itself, not only
    its name (`triage_prompt`, v0.4.0): other names, dates, places, a
    summary and the corroboration line. A name alone judged `Book of the
    Dead` and `Papyrus of Ani` one subject; one is a copy of the other.

    Returns one record per adjudicated pair. Writes nothing — the caller
    decides, the same asymmetry every verb in this module keeps.
    """
    from itertools import combinations

    from universe_builder import journal
    from universe_builder.collisions import Judgement
    from universe_builder.providers import run as provider_run
    from universe_builder.providers.types import LLMRequest

    system = (
        "You decide whether two records describe the same subject.\n"
        "\n"
        "They were grouped because they share a declared name. That is evidence, "
        "not proof: distinct subjects share names far more often than one would "
        "expect — inherited names, shared roots, shared epithets. Two records may "
        "share a name and be entirely different subjects.\n"
        "\n"
        "Answer whether they are one subject under two spellings, or two "
        "subjects. If unsure, say so: an undecided pair stays in the report for "
        "a person, while a confident wrong answer either destroys a distinct "
        "record or hides a real duplicate for ever.\n"
        "\n"
        "Each record gives what it says about itself: other names, dates, "
        "places and a summary. Use them. If one subject contains the other, "
        "or is a part, member, instance or copy of it, they are DIFFERENT "
        "subjects, even though they share a name, a place and a date."
    )

    records: List[dict] = []
    components = plan_.undecided[:limit] if limit else plan_.undecided
    for entry in components:
        paths = [store.dir_for(entry["atom_type"]) / name for name in entry["files"]]
        names = {path.name: common_name(path) for path in paths}
        for left, right in combinations(paths, 2):
            a, b = names[left.name], names[right.name]
            request = LLMRequest(
                model=model,
                schema_model=Judgement,
                system=system,
                user=triage_prompt(entry["atom_type"], left, right),
                max_tokens=2048,
                metadata={"operation": "duplicate-triage",
                          "atom_type": entry["atom_type"], "subject": a},
            )
            # Journalled per pair into the current run (`cmd_duplicates` opens
            # one for `--via model`), so `costs` sees the spend; it once went
            # straight to the provider and left no record.
            journalled = dict(operation="duplicate-triage", atom_type=entry["atom_type"],
                              subject=f"{a} / {b}", work_item=journal.work_item_from_path(
                                  entry["atom_type"], left.name))
            try:
                response = provider_run.complete(request)
            except Exception as e:  # noqa: BLE001 - one pair must not lose the rest
                log.error("Triaging %r vs %r failed (%s)", a, b, e)
                journal.record_call(request, None, error=str(e), **journalled)
                judgement = None
            else:
                journal.record_call(request, response, **journalled)
                judgement = response.parsed
            if not isinstance(judgement, Judgement):
                verdict = "unsure"
            elif not judgement.confident:
                verdict = "unsure"
            else:
                verdict = "same" if judgement.same_subject else "different"
            records.append({
                "atom_type": entry["atom_type"], "component": entry["name"],
                "a": a, "b": b, "a_file": left.name, "b_file": right.name,
                "verdict": verdict,
                "why": (getattr(judgement, "reason", "") or "").strip()[:110],
            })
    return records


def triage_scope(plan_: "Plan", limit: Optional[int] = None) -> tuple:
    """`(components, pairs)` a `triage` with this `limit` would judge: what
    `duplicates --via model` announces before it spends. One call per pair."""
    components = plan_.undecided[:limit] if limit else plan_.undecided
    pairs = sum(len(e["files"]) * (len(e["files"]) - 1) // 2 for e in components)
    return len(components), pairs


def verdict_counts(records: List[dict]) -> Dict[str, int]:
    """`{verdict: how many}` over `triage`'s records, in first-seen order."""
    counts: Dict[str, int] = {}
    for record in records:
        counts[record["verdict"]] = counts.get(record["verdict"], 0) + 1
    return counts


def save_verdicts(records: List[dict], path) -> None:
    """Every triage verdict as JSON (`duplicates --via model --out FILE`): a
    triage of the full pile costs real money, and without this its only
    record is the terminal."""
    import json

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2, ensure_ascii=False)


def write_distinct_lines(records: List[dict], path: Optional[Path] = None,
                         *, dry_run: bool = True) -> dict:
    """Append `A != B` for every pair triaged `different`. Dry by default.

    Only the negatives are written. A `same` verdict is *not* acted on here:
    it means the component is a real duplicate, and retiring one of its atoms
    is a separate, destructive decision that still wants the keep/retire
    machinery and a human. Writing only the negatives is the half that is
    safe, reversible (the file is in git) and immediately useful — it shrinks
    the report to the pairs that deserve attention.
    """
    if path is None:
        path = _pack_distinct_path()
    different = [r for r in records if r["verdict"] == "different"]

    # **A `!=` line must never contradict an `=` line.** `aliases.txt` and
    # `not_duplicates.txt` are mirrors, and `lint --check alias-contradiction`
    # exists precisely because the corpus has held both claims about one pair
    # before. Observed once: the triage judged `Book of the Dead` and
    # `Papyrus of Ani` distinct — correctly, one being a funerary corpus and the
    # other a single manuscript of it — while `aliases.txt` already declared
    # them one topic. Writing that line would have made the two files disagree,
    # and `topics` reads both.
    #
    # Refused rather than resolved: which file is wrong is a judgement, and the
    # alias line may be load-bearing for atoms already generated under it.
    #
    # **Compared as the grouping machinery folds, and only against global
    # lines**. Whole casefolded spellings missed
    # `Thebes (Luxor) = Waset` against `Thebes != Waset`, because the alias line
    # claims `thebes` through its parenthetical. And a scoped line speaks only
    # inside its civilisation, while a `!=` line is global, so a Greek-scoped
    # identity must not refuse a distinction every other civilisation needs.
    aliases = _pack_aliases_path()
    joined = ([_claim_keys(d.names) for d in alias_files.AliasFile.read(aliases).global_declarations()]
              if aliases else [])

    def contradicts_alias(record) -> bool:
        lefts, rights = topics.alias_keys(record["a"]), topics.alias_keys(record["b"])
        return any(left != right and left in keys and right in keys
                   for keys in joined for left in lefts for right in rights)

    contradicts = [r for r in different if contradicts_alias(r)]
    contradicting = {id(r) for r in contradicts}
    different = [r for r in different if id(r) not in contradicting]
    # Drop the contradicting pairs from the records the grouping below reads,
    # so a component holding one cannot be written as a multi-way line that
    # smuggles it back in.
    records = [r for r in records if id(r) not in contradicting]
    # Every `!=` line is one claim over *all* its names, folded as
    # `load_distinct` reads them, and a new claim is already present when one
    # existing claim covers it — the rule `_declared_distinct` applies. Keying
    # by `split("!=", 1)` mis-keyed every multi-way line, so the same component
    # was appended again on the next run.
    existing: List[frozenset] = []
    if path and Path(path).exists():
        existing = [frozenset(topics.name_key(n) for n in d.names)
                    for d in alias_files.NotDuplicatesFile.read(path).declarations]
    # **A component is suppressed by one claim covering all of it, never by the
    # pairs inside it.** `_declared_distinct` asks whether every atom in the
    # component is named by a single claim, deliberately: a line about two of
    # three atoms says nothing about the third. So a component every pair of
    # which was judged different is written as one multi-way line —
    # `A != B != C` — which the format has always allowed. Writing it pairwise
    # instead left three-atom components like `Abbas I` / `Abbas II` /
    # `Abbas Mirza` still undecided after 99 lines had been added for them.
    from collections import defaultdict

    components: Dict[tuple, List[dict]] = defaultdict(list)
    for record in records:
        components[(record.get("atom_type"), record.get("component"))].append(record)

    lines, skipped = [], 0
    for key, group in components.items():
        verdicts = {r["verdict"] for r in group}
        names: List[str] = []
        for record in group:
            for name in (record["a"], record["b"]):
                if name not in names:
                    names.append(name)
        if len(names) > 2 and verdicts == {"different"}:
            # Every pair different: the whole component is distinct, so one
            # line settles it.
            claims = [frozenset(topics.name_key(n) for n in names)]
            emit = [" != ".join(names)]
        else:
            claims, emit = [], []
            for record in group:
                if record["verdict"] != "different":
                    continue
                claims.append(frozenset((topics.name_key(record["a"]), topics.name_key(record["b"]))))
                emit.append(f"{record['a']} != {record['b']}")
        for claim, line in zip(claims, emit):
            if any(claim <= held for held in existing):
                skipped += 1
                continue
            existing.append(claim)
            lines.append(line)
    outcome = {"different": len(different), "new_lines": lines,
               "already_present": skipped, "dry_run": dry_run, "path": str(path),
               "contradicts_alias": [f"{r['a']} != {r['b']}" for r in contradicts]}
    if not dry_run and lines and path:
        # The whole file replaced in one step, never appended in place; the
        # bytes are what the append used to produce.
        held = Path(path).read_text(encoding="utf-8") if Path(path).exists() else ""
        note = "\n# Adjudicated by `universe_builder duplicates --via model`.\n"
        alias_files.write_atomically(Path(path), held + note + "\n".join(lines) + "\n")
    return outcome


def _pack_distinct_path() -> Optional[Path]:
    """The active pack's not-duplicates file, beside its alias file."""
    aliases = _pack_aliases_path()
    return alias_files.not_duplicates_path(aliases) if aliases else None


def _pack_aliases_path() -> Optional[Path]:
    """The alias file in force (`alias_files.aliases_path`: the `ALIASES_LIST`
    override, else the pack's), as its own function so a test can remove it.

    Every reader *and writer* here resolves through it. Until v0.1.20 this
    returned the pack's own path, so `apply` and `write_distinct_lines` ignored
    the override a test sets and, under the pack's test `workspace`, resolved
    to the real `aliases.txt`.
    """
    return alias_files.aliases_path()


def _paths_by_id(atom_type: str) -> Dict[str, Path]:
    found: Dict[str, Path] = {}
    for path in store.iter_atoms(atom_type):
        try:
            found[store.AtomName.decompose_filename(path.name).padded_id] = path
        except ValueError:
            continue
    return found


def _declared_same(paths: List[Path], groups) -> bool:
    """Whether one alias group names every atom in the component.

    The mirror of `_declared_distinct`, and the gate the fill tie-break needs.
    An alias line is an *adjudicated assertion of identity* — written by a
    human, or by `universe_builder collisions --via model` through `lint --apply-review` —
    so it is the one thing in the corpus that answers "are these one subject".
    `populated_fields` answers only "which says more", which is a ranking and
    was never a gate; used as one it proposed retiring Caracalla and Elagabalus
    into Marcus Aurelius, who share a regnal name because all three genuinely
    bore it.

    Requires a *single* group to cover the whole component, for the reason
    `_declared_distinct` does: a line about two of three atoms says nothing
    about the third.

    Three things it refuses, each found by adversarial review:

    * **Derived lines.** `aliases.txt` is written by `universe_builder duplicates --apply`
      as well as by people, and eight of history's sections are its own
      output. Reading those back as proof of identity let the command cite
      itself to license its next retirement.
    * **Scoped lines.** `group.scope` is the field that says an identity holds
      in one civilisation and not everywhere, and it was being ignored
      entirely — so a Greek-scoped Thebes line admitted the Egyptian
      `Thebes (Luxor)` from any context.
    * **Components whose names collapse.** Two atoms sharing a common name
      shrink the name set, letting a two-name line cover three atoms.
    """
    if not groups:
        return False
    names = {topics.name_key(_spelling_of(path)) for path in paths}
    if len(names) < len(paths):
        # Two atoms in the component answer to the same common name, so the
        # set collapsed and a two-name line could cover a three-atom component
        # — `Ramses III`, `Ramses III`, `Rameses III` was admitted by a line
        # naming only two. The docstring's promise is about atoms; keep it.
        return False
    for group in groups:
        if group.provenance != "asserted":
            # Derived lines record what a pass concluded, not what anyone
            # judged. Admitting them lets `--apply` cite its own previous
            # output as the evidence licensing its next retirement.
            continue
        if group.scope is not None:
            # A scoped line asserts identity *within one civilisation*, and is
            # frequently written to keep names apart elsewhere:
            # `[Greek History] Thebes (Boeotia) = Thebes = Thebai = Cadmea`
            # would otherwise admit a Greek Thebes atom and the Egyptian
            # `Thebes (Luxor)` as one subject, against a standing refusal.
            # Nothing here carries a civilisation to check against, so a scoped
            # line cannot be honoured and must not be used.
            continue
        if names <= group.keys():
            return True
    return False


def _spelling_of(path: Path) -> str:
    """The atom's own common name, falling back to the filename's topic."""
    payload = None
    try:
        payload = store.read_json(path)
    except OSError:
        pass
    if isinstance(payload, dict):
        name = payload.get("name")
        if isinstance(name, dict) and isinstance(name.get("common"), str):
            common = name["common"].strip()
            if common:
                return common
    return store.AtomName.decompose_filename(path.name).getTopic()


def _lookup_reaches(names, listed_for_keep, atom_type: str, keep_id: str) -> bool:
    """Whether the work list's walk over `names` (then the seed spellings that
    point at the survivor) reaches the survivor's file. Mirrors
    `topics.group_occurrences`: line order, one try per folded key, match on
    content id."""
    # The work list's own walk (`topics.lookup_order`); a name that folds to
    # nothing is never tried here.
    return any(store.atom_id(atom_type, name) == keep_id
               for name, key in topics.lookup_order((*names, *listed_for_keep)) if key)


@dataclass(frozen=True)
class Undecided:
    """A component `plan` will not act on, and why. `Plan.undecided` holds
    these as dicts (`as_dict`), the shape every caller and report reads."""

    atom_type: str
    name: str
    files: List[str]
    why: str

    def as_dict(self) -> dict:
        return {"atom_type": self.atom_type, "name": self.name, "files": self.files, "why": self.why}


@dataclass(frozen=True)
class Skipped:
    """A component ruled out by `--except` or declared distinct; reported,
    never acted on. `Plan.skipped` holds these as dicts (`as_dict`)."""

    atom_type: str
    name: str
    files: List[str]

    def as_dict(self) -> dict:
        return {"atom_type": self.atom_type, "name": self.name, "files": self.files}


@dataclass
class _Rules:
    """What `plan` reads once and every component is decided against."""

    patterns: List[str]
    wanted: List[str]
    claims: tuple
    alias_groups: tuple
    owners: Dict[str, set]
    tiebreak_fill: bool
    #: Every alias-file name, for `_generation_keys`; read on first need.
    _declared: Optional[List[str]] = None

    def declared(self) -> List[str]:
        if self._declared is None:
            self._declared = [name for group in topics.load_aliases() for name in group.spellings]
        return self._declared


def _undecided(atom_type: str, name: str, paths, why: str) -> Undecided:
    return Undecided(atom_type=atom_type, name=name, files=sorted(p.name for p in paths), why=why)


def _decide_component(atom_type: str, name: str, paths: List[Path], listed: Dict[str, List[str]],
                      rules: _Rules):
    """One duplicate component: a `Resolution`, `Undecided` or `Skipped`.

    In order: ruled out (`--except`, or declared distinct) is skipped; a
    `--keep` pattern names the survivor (`_named_survivor_decision`); else
    the later model family wins, and a same-family tie is broken by fill only
    where `tiebreak_fill` is on and an asserted line declares the component
    one subject (`_ranked_decision`). See `plan` for why each rule is as
    cautious as it is.
    """
    if (any(pattern in path.name.lower() for pattern in rules.patterns for path in paths)
            or _declared_distinct(paths, atom_type, rules.claims, listed=listed,
                                  declared=rules.declared)):
        return Skipped(atom_type=atom_type, name=name, files=sorted(p.name for p in paths))
    if rules.wanted and any(any(w in path.name.lower() for w in rules.wanted) for path in paths):
        return _named_survivor_decision(atom_type, name, paths, listed, rules)
    return _ranked_decision(atom_type, name, paths, listed, rules)


def _named_survivor_decision(atom_type: str, name: str, paths: List[Path],
                             listed: Dict[str, List[str]], rules: _Rules):
    """A `--keep` pattern touched this component: the survivor it names, if
    it names exactly one.

    This is how a model-family tie gets broken. Nothing in a filename ranks
    two same-generation atoms, so the rule declines — but a human who has
    read both can say which survives, and `--keep` is that answer. A pattern
    matching several atoms in one component is no answer at all.
    """
    hits = [p for p in paths if any(w in p.name.lower() for w in rules.wanted)]
    if len(hits) != 1:
        # `--keep` matched this pair ambiguously — more than one file, or none
        # of them cleanly. Falling through to the fill tie-break would quietly
        # answer a question the reader asked badly, and they would never learn
        # their pattern was wrong. Say so instead.
        return _undecided(atom_type, name, paths,
                          "--keep matches more than one of these files; narrow "
                          "the pattern to name a single survivor")
    chosen = hits[0]
    losers = [p for p in paths if p != chosen]
    spellings = _spellings_for(chosen, losers, listed)
    # `--keep` answers *which atom survives*. It does not licence an alias
    # line that would capture another atom type's name — that check is about
    # the file this writes, not about the ranking, and a human naming a
    # survivor has not reviewed it.
    trespass = _foreign_keys(spellings, atom_type, rules.owners)
    # A name another atom type lists is a reason not to *write* a line. When a
    # person has already written one — an asserted line covering this whole
    # component — nothing needs writing, and the refusal only stopped `--keep`
    # retiring the Camp David event pair because an artifact atom shares the
    # name.
    if trespass and not _declared_same(paths, rules.alias_groups):
        return _undecided(atom_type, name, paths,
                          f"--keep names a survivor, but the alias line would "
                          f"claim {sorted(trespass)}, which another atom type "
                          f"lists; write the line by hand")
    return Resolution(atom_type=atom_type, name=name, keep=chosen, retire=losers,
                      spellings=spellings, reason="named by --keep")


def _ranked_decision(atom_type: str, name: str, paths: List[Path],
                     listed: Dict[str, List[str]], rules: _Rules):
    """The later model family wins; a tie is broken by fill only when allowed."""
    ranked = []
    for path in paths:
        version = store.AtomName.decompose_filename(path.name).model_version
        ranked.append((model_rank(version), version, path))
    best = max((rank for rank, _, _ in ranked if rank is not None), default=None)
    if best is None:
        return _undecided(atom_type, name, paths, "no model version carries a comparable number")
    winners = [entry_ for entry_ in ranked if entry_[0] == best]
    fill_reason = None
    # The tie-break is allowed only where the alias file already declares this
    # component to be one subject. That assertion is the gate; the field count
    # is only the ranking.
    rankable = rules.tiebreak_fill and _declared_same(paths, rules.alias_groups)
    if len(winners) > 1 and not rankable:
        return _undecided(atom_type, name, paths,
                          f"{len(winners)} atoms share the highest model family "
                          f"({best:g}); nothing in a filename ranks them")
    if len(winners) > 1:
        # The filename has run out of things to say, so ask the atoms. Sorted
        # on (fields, name) so the order is total and stable — a tie broken by
        # filesystem order would make `plan()` non-deterministic, and the dry
        # run has to predict what `--apply` will do.
        scored = sorted(((populated_fields(path), path.name, path)
                         for _rank, _version, path in winners),
                        reverse=True)
        if scored[0][0] == scored[1][0]:
            return _undecided(atom_type, name, paths,
                              f"{len(winners)} atoms share the highest model "
                              f"family ({best:g}) and fill the same number of "
                              f"fields ({scored[0][0]}); nothing separates them")
        keep = scored[0][2]
        fill_reason = (f"same model family ({best:g}); keeps the fuller "
                       f"atom — {scored[0][0]} populated fields against "
                       f"{scored[1][0]}")
    else:
        keep = winners[0][2]
    losers = [p for p in paths if p != keep]
    spellings = _spellings_for(keep, losers, listed)
    trespass = _foreign_keys(spellings, atom_type, rules.owners)
    if trespass and not _declared_same(paths, rules.alias_groups):
        return _undecided(atom_type, name, paths,
                          f"the alias line would claim {sorted(trespass)}, which "
                          f"{'is a name' if len(trespass) == 1 else 'are names'} "
                          f"another atom type lists; write a line by hand")
    return Resolution(atom_type=atom_type, name=name, keep=keep, retire=losers,
                      spellings=spellings,
                      reason=fill_reason or _outranks_reason(winners[0][1], losers))


def _trim_covered(result: "Plan", alias_groups, listed_by_type: Dict[str, dict]) -> List[Resolution]:
    """Drop from each resolution the spellings an asserted line already carries.

    Where a person's asserted line already covers a component, the line the
    retirement would write is redundant, and the chain check would refuse it
    for merging with that very line (the Camp David event pair).
    Keep only spellings the existing line does not carry; a resolution with
    none left writes nothing and retires all the same. A covering line that
    cannot reach the survivor moves the component to `undecided`.
    """
    import dataclasses

    trimmed: List[Resolution] = []
    for resolution in result.resolutions:
        paths = [resolution.keep, *resolution.retire]
        listed = listed_by_type[resolution.atom_type]
        covering = [g for g in alias_groups
                    if g.provenance == "asserted" and g.scope is None
                    and {topics.name_key(_spelling_of(p)) for p in paths} <= g.keys()]
        if covering:
            # The work list looks a group up by trying its names in order,
            # skipping any whose folded key it has already tried, and asking
            # whether a real atom exists under *that spelling's content id*
            # (`topics.group_occurrences`). The survivor is therefore reachable
            # only if some name it will try hashes to the survivor's own file —
            # the spelling it was generated under, which need not be its
            # `name.common`, and which a same-key earlier name can shadow
            # (`Al-Muʿizz li-Dīn Allāh = Al-Mu'izz li-Din Allah`).
            # A covering line that cannot reach the survivor would have the next
            # create run regenerate the atom just retired. Nothing here rewrites
            # a person's line; it refuses and says what the line needs.
            try:
                keep_id = store.AtomName.decompose_filename(resolution.keep.name).padded_id
            except ValueError:
                keep_id = None
            if keep_id is not None and not any(
                    _lookup_reaches(group.spellings, listed.get(keep_id, ()), resolution.atom_type, keep_id)
                    for group in covering):
                result.undecided.append(_undecided(
                    resolution.atom_type, resolution.name, paths,
                    f"an asserted alias line covers these atoms, but none of its "
                    f"names the work list would try resolves to the survivor "
                    f"{resolution.keep.name}: add the spelling it was generated "
                    f"under, ahead of any spelling that folds to the same key").as_dict())
                continue
            covered = set().union(*(g.keys() for g in covering))
            spellings = [s for s in resolution.spellings if not topics.alias_keys(s) <= covered]
            resolution = dataclasses.replace(resolution, spellings=spellings)
        trimmed.append(resolution)
    return trimmed


def plan(exclude: Optional[List[str]] = None,
         keep: Optional[List[str]] = None,
         distinct: Optional[tuple] = None,
         tiebreak_fill: bool = False,
         aliases: Optional[tuple] = None) -> Plan:
    """Decide, for each duplicate pair, which atom to keep. Reads only.

    The rule is the later model family wins — a gpt-5 atom outranks a gpt-4o
    one. Where the families tie, nothing here decides: the pair goes to
    `undecided` for a human, because a filename does not say which of two
    same-generation atoms is better and guessing would discard content on a
    coin flip.

    **`tiebreak_fill` is off by default, and the reason is a near-miss.** It
    breaks a same-generation tie by keeping whichever atom fills more fields,
    which sounds obviously right and is — *provided the component is genuinely
    one subject*. Nothing here establishes that. Enabled as the default once,
    it turned dozens of components "resolvable", among them
    `Marcus Aurelius` retiring `Caracalla` and `Elagabalus`, and
    `Constantius Chlorus` retiring `Constantius II` and `Constantius III` —
    six emperors, three files. `undecided` had been the only thing protecting
    them, and ranking them removed it while answering a question nobody had
    asked. Use it only on components a `--via model` triage has confirmed.

    **A shared name is evidence of a duplicate, not proof of one**, and
    `exclude` is how a reader overrules it. Two atoms can legitimately name
    each other without being the same thing: the modern Bibliotheca Alexandrina
    cites the ancient Great Library of Alexandria, and the Ebers Papyrus cites
    the Ramesseum Medical Papyrus, in both cases as a related work rather than
    an alias. Retiring either would destroy a real atom to tidy a report.

    Each `exclude` entry is a case-insensitive substring matched against the
    filenames in a pair; any pair it touches is dropped from the plan entirely
    rather than half-applied. The asymmetry is the project's standing one: a
    skipped merge costs a visible duplicate, a wrong merge costs content.
    """
    patterns = [text.lower() for text in (exclude or []) if text.strip()]
    wanted = [text.lower() for text in (keep or []) if text.strip()]
    claims = load_distinct() if distinct is None else distinct
    alias_claims = existing_alias_claims()
    # Injectable for the same reason `distinct` is: a test must be able to
    # state the corpus's assertions rather than inherit the shipped files.
    alias_groups = topics.load_aliases() if aliases is None else aliases
    rules = _Rules(patterns=patterns, wanted=wanted, claims=claims, alias_groups=alias_groups,
                   owners=topics.listed_keys_by_type(), tiebreak_fill=tiebreak_fill)

    result = Plan()
    # Read once per atom type, here, and reused by every component and by the
    # trim below: each read regroups the whole work list.
    listed_by_type: Dict[str, dict] = {}
    for spec in active_pack().atoms:
        duplicates = topics.duplicate_atoms(spec.name)
        if not duplicates:
            continue
        by_id = _paths_by_id(spec.name)
        listed = listed_by_type[spec.name] = topics.listed_spellings(spec.name)
        # One *pair* per plan entry, not one per shared name. Two atoms
        # commonly share several names — `Ramesses II` also declares
        # `Ozymandias` and `Ramses II`, so the same two files were planned
        # three times over. Beyond making the report unreadable, applying such
        # a plan would call `retire_duplicate` on a file the first pass had
        # already moved, and `os.replace` on a missing path raises.
        pairs: Dict[frozenset, List[str]] = {}
        for entry in duplicates:
            pairs.setdefault(frozenset(entry["atom_ids"]), []).append(entry["name"])
        for atom_ids, names in pairs.items():
            paths = [by_id[atom_id] for atom_id in sorted(atom_ids) if atom_id in by_id]
            if len(paths) < 2:
                continue  # one side already gone; nothing to retire
            decision = _decide_component(spec.name, ", ".join(sorted(names)), paths, listed, rules)
            if isinstance(decision, Resolution):
                result.resolutions.append(decision)
            elif isinstance(decision, Skipped):
                result.skipped.append(decision.as_dict())
            else:
                result.undecided.append(decision.as_dict())
    result.resolutions = _trim_covered(result, alias_groups, listed_by_type)
    return _without_alias_chains(_stars_only(result), alias_claims)


def _outranks_reason(winner_model: str, losers) -> str:
    """`"<winner> outranks <loser models>"`, as one readable sentence."""
    beaten = sorted(store.AtomName.decompose_filename(p.name).model_version
                    for p in losers)
    return f"{winner_model} outranks {', '.join(beaten)}"


def _without_alias_chains(plan_: Plan, existing: List[set]) -> Plan:
    """Refuse resolutions whose alias line would chain declarations in the file.

    The third and outermost place *declarations never chain* has to be enforced.
    `_stars_only` handles retirements; `alias_lines` handles two rivals of one
    survivor within a plan; this handles the file. Two ways a chain forms here:

    * **A new line overlaps two existing lines.** `A = B` and `C = D` are in the
      file, a new `B = C` arrives, and merging yields `A = B = C = D` — nothing
      ever declared `A = D`.
    * **Two new lines overlap one existing line.** `A = B` is in the file, new
      lines `A = C` and `B = D` arrive, and writing them in sequence chains the
      same way, because the first write is what the second sees.

    Each surviving line may therefore share names with at most one existing
    declaration, and with no other new line. A refused resolution is not
    retired: the line is the half of the repair that stops regeneration, so
    without it the retirement is unsafe.
    """
    if not plan_.resolutions:
        return plan_

    # Group resolutions by survivor, exactly as `alias_lines` does, so a
    # survivor beating two rivals is one line and not two.
    by_survivor: Dict[Path, List[Resolution]] = {}
    for resolution in plan_.resolutions:
        by_survivor.setdefault(resolution.keep, []).append(resolution)

    keys_of = {keep: _claim_keys(n for r in group for n in r.spellings)
               for keep, group in by_survivor.items()}

    refused: set = set()
    touching: Dict[int, List[Path]] = {}
    for keep, keys in keys_of.items():
        touched = [i for i, claim in enumerate(existing) if claim & keys]
        # Spanning two declarations chains them directly.
        if len(touched) > 1:
            refused.add(keep)
        for index in touched:
            touching.setdefault(index, []).append(keep)
        # Overlapping another new line chains them without any existing line.
        if any(other != keep and others_keys & keys
               for other, others_keys in keys_of.items()):
            refused.add(keep)
    # Two new lines meeting at one existing line chain through it: written in
    # sequence, the first write is what the second sees.
    for holders in touching.values():
        if len(holders) > 1:
            refused.update(holders)

    if not refused:
        return plan_

    kept: List[Resolution] = []
    for resolution in plan_.resolutions:
        if resolution.keep not in refused:
            kept.append(resolution)
            continue
        plan_.undecided.append(_undecided(
            resolution.atom_type, resolution.name, (resolution.keep, *resolution.retire),
            "the alias line would merge two existing declarations, "
            "joining names nothing declared equal; write the line by hand").as_dict())
    plan_.resolutions = kept
    return plan_


def _stars_only(plan_: Plan) -> Plan:
    """Keep only components where one atom survives and all retirements name it.

    Duplicate pairs are edges in a graph, and **pairwise equivalence does not
    compose** — the Saqqara rule. Exactly one
    shape can be applied without composing anything: a *star*, where a single
    atom is never retired and every other atom in the component is retired in
    its favour. `Amarna Boundary Stelae` beating two rivals is that shape, and
    it collapses to one alias line naming one survivor.

    Two shapes are not, and both destroy data while appearing locally sensible:

    * **A chain.** `Alpha`-`Beta` and `Beta`-`Gamma`: Beta wins one pair and
      loses the other, so applying both retires Beta while an alias line still
      names it Alpha's survivor. `Alpha` then points at a file that is gone, and
      the file declares `Alpha -> Beta -> Gamma`.
    * **Two survivors claiming one loser.** `Alpha`-`Beta` and `Gamma`-`Beta`:
      Beta loses twice and no keeper is ever retired, so a keeper-vs-retiree
      check sees nothing wrong. Applying it destroys the only atom linking Alpha
      and Gamma, keeps both, and writes two same-scope lines each claiming
      Beta's name — which `group_occurrences` reads as ambiguous, leaving
      Beta's listed spelling resolving to nothing, so the next create run
      regenerates the atom that was just retired.

    In both, the question "which atom survives" is exactly what the component
    fails to answer, so every resolution in it moves to `undecided` — including
    any that looks fine on its own.
    """
    if not plan_.resolutions:
        return plan_

    # Connected components over the atoms each resolution touches.
    component: Dict[Path, int] = {}
    members: Dict[int, List[Resolution]] = {}
    for resolution in plan_.resolutions:
        atoms = [resolution.keep, *resolution.retire]
        found = sorted({component[a] for a in atoms if a in component})
        if found:
            target = found[0]
            for other in found[1:]:                     # merge components
                for atom, label in list(component.items()):
                    if label == other:
                        component[atom] = target
                members[target].extend(members.pop(other))
        else:
            target = len(members)
            members[target] = []
        for atom in atoms:
            component[atom] = target
        members[target].append(resolution)

    kept: List[Resolution] = []
    for group in members.values():
        keepers = {r.keep for r in group}
        retirees = {p for r in group for p in r.retire}
        if len(keepers) == 1 and not (keepers & retirees):
            kept.extend(group)
            continue
        why = ("two atoms here would each survive while both claim the same "
               "retired atom, and declarations never chain, so nothing says "
               "they are the same thing"
               if len(keepers) > 1 else
               "an atom here is the survivor of one pair and the loser of "
               "another — a chain, and declarations never chain")
        for resolution in group:
            plan_.undecided.append(_undecided(
                resolution.atom_type, resolution.name, (resolution.keep, *resolution.retire),
                f"{why}; pick the survivor by hand").as_dict())
    plan_.resolutions = kept
    return plan_


def _spellings_for(keep: Path, losers: List[Path], listed: Dict[str, List[str]]) -> List[str]:
    """Every name that must end up on one alias line, survivor's first.

    Both halves matter. The atoms' own `name.common` is what a reader
    recognises, and the History-listed subjects are what the work list actually
    looks up — an atom generated for `Shaduf (irrigation tool)` calling itself
    `Shaduf` needs both, or retiring its rival leaves the listed spelling
    resolving to nothing and `needing_create` regenerates it.
    """
    ordered: List[str] = []
    seen = set()

    def add(name: str) -> None:
        folded = topics.name_key(name or "")
        if folded and folded not in seen:
            seen.add(folded)
            ordered.append(name)

    for path in (keep, *losers):
        add(_spelling_of(path))
        try:
            atom_id = store.AtomName.decompose_filename(path.name).padded_id
        except ValueError:
            continue
        for name in listed.get(atom_id, ()):
            add(name)
    return ordered


def _foreign_keys(spellings: List[str], atom_type: str, owners: Dict[str, set]) -> set:
    """Names on this line that another atom type's work list depends on.

    `alias_keys` splits a trailing parenthetical, so a line naming
    `… by King Menes (Narmer)` claims the bare key `narmer` — and because the
    alias file has no per-type scope, the *people* lose `Narmer` to an
    ambiguity introduced by a line about *events*. Observed, not hypothetical:
    the first run of this command did exactly that and `Narmer` went missing
    from the work list.
    """
    trespass = set()
    for name in spellings:
        for key in topics.alias_keys(name):
            holders = owners.get(key, set())
            if holders - {atom_type}:
                trespass.add(key)
    return trespass


def apply(plan_: Plan, aliases_path: Optional[Path] = None, *, dry_run: bool = True) -> dict:
    """Write the alias lines, then retire the losing files — in that order.

    Order matters. The alias line is what stops `needing_create` regenerating
    the retired spelling at the next create run; writing it first means an
    interruption between the two steps leaves a corpus that is merely
    over-joined, not one that silently regrows its duplicates.
    """
    if aliases_path is None:
        aliases_path = _pack_aliases_path()
    written, retired = [], []
    if dry_run:
        return {"aliases": plan_.alias_lines(), "retired": [],
                "dry_run": True, "aliases_path": str(aliases_path)}

    if aliases_path is None:
        # Refuse rather than retire half-way. Without somewhere to record the
        # alias, the losing spelling stays listed in a History atom with no
        # atom behind it, so `needing_create` regenerates what was just
        # retired — and pays for it. A pack that declares no alias file cannot
        # use this command.
        raise ValueError(
            f"domain {active_pack().name!r} declares no alias file, so the "
            f"retired spellings could not be recorded and the next create run "
            f"would regenerate them. Nothing was retired.")

    lines = plan_.alias_lines()
    if lines:
        written = _merge_into_alias_file(Path(aliases_path), lines)

    for resolution in plan_.resolutions:
        for path in resolution.retire:
            if not path.exists():
                # Defensive: the plan is deduplicated by atom pair, so this
                # should not happen. If it ever does, a stale plan entry must
                # not abort the retirements behind it.
                log.warning("%s is already gone; nothing to retire", path.name)
                continue
            retired.append(str(store.retire_duplicate(path)))

    return {"aliases": written, "retired": retired, "dry_run": False,
            "aliases_path": str(aliases_path)}
