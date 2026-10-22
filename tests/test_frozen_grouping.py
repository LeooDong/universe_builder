"""Frozen name keys and work-list grouping.

Grouping decides which spelling the work list emits, and
the spelling decides the content id: a regrouping that picks a *different*
spelling for one subject mints a new id, and the next create writes a duplicate
atom beside the old one. No hash changes, so the golden ids
(`test_golden_ids.py`) never notice.

Three layers, each frozen as literal data computed with the tagged v0.1.19
code and never recomputed by the code under test:

* `_key` / `alias_keys` for every name in the history example's seed, alias and
  not-duplicates files, the fixtures below, and a set of names chosen for
  awkward characters (`fixtures/name_keys.json`);
* what the alias and not-duplicates parsers read from two files written in
  v0.1.19's format (`fixtures/alias_files_parsed_v0.1.19.json`) — the parity
  target for the `AliasFile` parser;
* `from_history`, `needing_create`, the `_group_occurrences` assignments,
  `listed_spellings` and `subject_resolver` answers on a small hand-built corpus
  that reaches every grouping branch (below).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from universe_builder import duplicates, settings, store, topics

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------

_KEY_ROWS = _load("name_keys.json")["rows"]


def test_the_key_fixture_is_the_size_it_was_frozen_at():
    assert len(_KEY_ROWS) == 191


def test_every_frozen_name_keys_as_it_did():
    drifted = [(r["name"], r["key"], topics._key(r["name"]))
               for r in _KEY_ROWS if topics._key(r["name"]) != r["key"]]
    assert drifted == []


def test_every_frozen_name_declares_the_alias_keys_it_did():
    drifted = [(r["name"], r["alias_keys"], sorted(topics.alias_keys(r["name"])))
               for r in _KEY_ROWS if sorted(topics.alias_keys(r["name"])) != r["alias_keys"]]
    assert drifted == []


# ---------------------------------------------------------------------------
# Alias / not-duplicates parsing
# ---------------------------------------------------------------------------

_PARSED = _load("alias_files_parsed_v0.1.19.json")


def test_load_aliases_reads_the_frozen_file_as_it_did():
    groups = topics.load_aliases(FIXTURES / "aliases_v0.1.19.txt")
    got = [{"names": list(g.names), "scope": g.scope, "provenance": g.provenance} for g in groups]
    assert len(got) == 24
    assert got == _PARSED["load_aliases"]


def test_load_distinct_reads_the_frozen_file_as_it_did():
    claims = duplicates.load_distinct(FIXTURES / "not_duplicates_v0.1.19.txt")
    assert len(claims) == 20
    assert [sorted(c) for c in claims] == _PARSED["load_distinct"]


def test_existing_alias_claims_reads_the_frozen_file_as_it_did():
    claims = duplicates.existing_alias_claims(FIXTURES / "aliases_v0.1.19.txt")
    assert len(claims) == 16
    assert [sorted(c) for c in claims] == _PARSED["existing_alias_claims"]


# ---------------------------------------------------------------------------
# Grouping on a fixture corpus
# ---------------------------------------------------------------------------

#: Every branch of `_group_occurrences` the corpus has needed a rule for:
#: a global alias line folding a data parenthetical (Amarna); a reversed
#: parenthetical making the same claim (Akhetaten (Amarna)); a scoped line
#: giving a bare name its own identity (Greek Thebes, Byzantine Abydos) and
#: a scoped line's bare non-leading name *not* capturing another
#: civilisation's stored atom (Abydos); a parenthetical declaration
#: speaking only for its own civilisation (Egyptian Thebes (Luxor) vs Greek
#: Thebes); no chaining (Memphis / Mit Rahina / Saqqara); a partial overlap
#: with an alias line; a same-scope contradiction leaving names ungrouped
#: (Pharos); a derived section; corpus-wins on a global line (Rhakotis);
#: a placeholder never outranking (Luxor); pass-4 de-duplication of one
#: spelling emitted by two groups; and a type with no plural list.
ALIASES = """\
# fixture alias file
Amarna (Akhetaten) = Tell el-Amarna
Alexandria = Rhakotis
Pharos = Lighthouse of Alexandria
Pharos Island = Pharos
Heliopolis (Iunu) = On
[Greek History]
Thebes (Boeotia) = Thebes = Thebai
[Byzantine History]
Abydos (Hellespont) = Abydos
[*]
# Added by `universe_builder duplicates --apply`: one subject, two atoms.
[*]
Foundation of Memphis = Founding of Memphis
"""

SEED = "Egyptian History\nGreek History\nByzantine History\n"

HISTORIES = {
    "Egyptian History": {
        "cities": ["Thebes (Luxor)", "Thebes", "Luxor", "Memphis (Mit Rahina)",
                   "Mit Rahina (Saqqara)", "Saqqara", "Amarna (Akhetaten)", "Akhetaten",
                   "Alexandria", "Abydos", "Pharos", "Heliopolis (On)", "Iunu",
                   "Heliopolis (Iunu)"],
        "events": ["Founding of Memphis", "Battle of Kadesh"],
        "people": ["Narmer (Menes)", "Menes", "Ramesses II"],
    },
    "Greek History": {
        "cities": ["Thebes", "Thebes (Luxor)", "Alexandria", "Athens", "Rhakotis", "Thebai"],
        "events": ["Battle of Marathon", "Foundation of Memphis"],
        "people": ["Pericles", "Narmer"],
    },
    "Byzantine History": {
        "cities": ["Abydos", "Alexandria", "Constantinople", "Akhetaten (Amarna)", "Pharos Island"],
        "events": ["Nika riots"],
        "people": ["Justinian I", "Menes"],
    },
}

#: (atom type, subject the file is stored under, payload or None for a real atom)
STORED = [
    ("city", "Abydos", None),               # the Egyptian city, generated bare
    ("city", "Thebes (Luxor)", None),
    ("city", "Rhakotis", None),             # a global line's non-leading name, stored
    ("city", "Luxor", {"error": "No data found"}),   # a placeholder, not an atom
    ("city", "Saqqara", None),
    ("event", "Founding of Memphis", None),
    ("person", "Narmer (Menes)", None),
    ("person", "Ramesses II", {"name": {"common": "Ramesses II", "otherNames": ["Ozymandias"]}}),
]

QUERIES = [
    ("city", "Thebes", None), ("city", "Thebes", "Egyptian History"), ("city", "Thebes", "Greek History"),
    ("city", "Luxor", "Egyptian History"), ("city", "Abydos", "Byzantine History"),
    ("city", "Abydos", "Egyptian History"), ("city", "Abydos", None), ("city", "Rhakotis", None),
    ("city", "Alexandria", "Greek History"), ("city", "Tell el-Amarna", None),
    ("city", "Memphis", "Egyptian History"), ("city", "Saqqara", None),
    ("city", "Pharos", "Egyptian History"), ("city", "Nowhere", None),
    ("event", "Foundation of Memphis", "Greek History"), ("event", "Founding of Memphis", None),
    ("person", "Menes", "Byzantine History"), ("person", "Narmer", "Greek History"),
    ("person", "Ozymandias", None), ("person", "Menes", "Egyptian History"),
]


@pytest.fixture
def corpus(workspace, tmp_path, monkeypatch):
    seed = tmp_path / "HISTORY_list.txt"
    seed.write_text(SEED, encoding="utf-8")
    monkeypatch.setattr(settings, "HISTORY_LIST", seed)
    aliases = tmp_path / "aliases.txt"
    aliases.write_text(ALIASES, encoding="utf-8")
    monkeypatch.setattr(settings, "ALIASES_LIST", aliases)
    for civilisation, lists in HISTORIES.items():
        payload = {"people": [], "discoveries": [], "events": [], "cities": [], "artifacts": []}
        payload.update(lists)
        path = store.path_for("history", None, civilisation, "m")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    for atom_type, subject, payload in STORED:
        path = store.path_for(atom_type, None, subject, "m")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload or {"name": {"common": subject}}), encoding="utf-8")
    return workspace


# Frozen with v0.1.19. Contexts are as `_context_of` returns them
# (the normalised seed line). Worth knowing when one of these flips:
#  * Greek `Thebes (Luxor)` is dropped by pass 4 (one spelling, one row).
#  * `Pharos` and `Pharos Island` are left ungrouped: two same-scope lines claim
#    `pharos`, so `Lighthouse of Alexandria` is never generated.
#  * RESOLVED: `Abydos` *with Byzantine context* resolves to the Egyptian atom.
#    Its survivor `Abydos (Hellespont)` has no file, and the resolver then
#    falls through to the bare name's own id. Current behaviour, pinned.
FROM_HISTORY = {
    'city': [
        ['Thebes (Luxor)', 'egyptian history'],
        ['Memphis (Mit Rahina)', 'egyptian history'],
        ['Mit Rahina (Saqqara)', 'egyptian history'],
        ['Saqqara', 'egyptian history'],
        ['Amarna (Akhetaten)', 'egyptian history'],
        ['Rhakotis', 'egyptian history'],
        ['Abydos', 'egyptian history'],
        ['Pharos', 'egyptian history'],
        ['Heliopolis (Iunu)', 'egyptian history'],
        ['Abydos (Hellespont)', 'byzantine history'],
        ['Constantinople', 'byzantine history'],
        ['Pharos Island', 'byzantine history'],
        ['Thebes (Boeotia)', 'greek history'],
        ['Athens', 'greek history'],
    ],
    'event': [
        ['Founding of Memphis', 'egyptian history'],
        ['Battle of Kadesh', 'egyptian history'],
        ['Nika riots', 'byzantine history'],
        ['Battle of Marathon', 'greek history'],
    ],
    'person': [
        ['Narmer (Menes)', 'egyptian history'],
        ['Ramesses II', 'egyptian history'],
        ['Justinian I', 'byzantine history'],
        ['Menes', 'byzantine history'],
        ['Pericles', 'greek history'],
        ['Narmer', 'greek history'],
    ],
    'artifact': [],
}

NEEDING_CREATE = {
    'city': [
        ['Memphis (Mit Rahina)', 'egyptian history'],
        ['Mit Rahina (Saqqara)', 'egyptian history'],
        ['Amarna (Akhetaten)', 'egyptian history'],
        ['Pharos', 'egyptian history'],
        ['Heliopolis (Iunu)', 'egyptian history'],
        ['Abydos (Hellespont)', 'byzantine history'],
        ['Constantinople', 'byzantine history'],
        ['Pharos Island', 'byzantine history'],
        ['Thebes (Boeotia)', 'greek history'],
        ['Athens', 'greek history'],
    ],
    'event': [
        ['Battle of Kadesh', 'egyptian history'],
        ['Nika riots', 'byzantine history'],
        ['Battle of Marathon', 'greek history'],
    ],
    'person': [
        ['Justinian I', 'byzantine history'],
        ['Menes', 'byzantine history'],
        ['Pericles', 'greek history'],
        ['Narmer', 'greek history'],
    ],
    'artifact': [],
}

ASSIGNMENTS = {
    'city': [
        ['Abydos', 'byzantine history', 'Abydos (Hellespont)'],
        ['Abydos', 'egyptian history', 'Abydos'],
        ['Akhetaten', 'egyptian history', 'Amarna (Akhetaten)'],
        ['Akhetaten (Amarna)', 'byzantine history', 'Amarna (Akhetaten)'],
        ['Alexandria', 'byzantine history', 'Rhakotis'],
        ['Alexandria', 'egyptian history', 'Rhakotis'],
        ['Alexandria', 'greek history', 'Rhakotis'],
        ['Amarna (Akhetaten)', 'egyptian history', 'Amarna (Akhetaten)'],
        ['Athens', 'greek history', 'Athens'],
        ['Constantinople', 'byzantine history', 'Constantinople'],
        ['Heliopolis (Iunu)', 'egyptian history', 'Heliopolis (Iunu)'],
        ['Heliopolis (On)', 'egyptian history', 'Heliopolis (Iunu)'],
        ['Iunu', 'egyptian history', 'Heliopolis (Iunu)'],
        ['Luxor', 'egyptian history', 'Thebes (Luxor)'],
        ['Memphis (Mit Rahina)', 'egyptian history', 'Memphis (Mit Rahina)'],
        ['Mit Rahina (Saqqara)', 'egyptian history', 'Mit Rahina (Saqqara)'],
        ['Pharos', 'egyptian history', 'Pharos'],
        ['Pharos Island', 'byzantine history', 'Pharos Island'],
        ['Rhakotis', 'greek history', 'Rhakotis'],
        ['Saqqara', 'egyptian history', 'Saqqara'],
        ['Thebai', 'greek history', 'Thebes (Boeotia)'],
        ['Thebes', 'egyptian history', 'Thebes (Luxor)'],
        ['Thebes', 'greek history', 'Thebes (Boeotia)'],
        ['Thebes (Luxor)', 'egyptian history', 'Thebes (Luxor)'],
        ['Thebes (Luxor)', 'greek history', 'Thebes (Luxor)'],
    ],
    'event': [
        ['Battle of Kadesh', 'egyptian history', 'Battle of Kadesh'],
        ['Battle of Marathon', 'greek history', 'Battle of Marathon'],
        ['Foundation of Memphis', 'greek history', 'Founding of Memphis'],
        ['Founding of Memphis', 'egyptian history', 'Founding of Memphis'],
        ['Nika riots', 'byzantine history', 'Nika riots'],
    ],
    'person': [
        ['Justinian I', 'byzantine history', 'Justinian I'],
        ['Menes', 'byzantine history', 'Menes'],
        ['Menes', 'egyptian history', 'Narmer (Menes)'],
        ['Narmer', 'greek history', 'Narmer'],
        ['Narmer (Menes)', 'egyptian history', 'Narmer (Menes)'],
        ['Pericles', 'greek history', 'Pericles'],
        ['Ramesses II', 'egyptian history', 'Ramesses II'],
    ],
    'artifact': [],
}

LISTED_SPELLINGS = {
    'city': {
        'h8fdc46cd4a0f6fb6': ['Thebes (Luxor)', 'Thebes', 'Luxor'],
        'hecbceeb479fbec03': ['Mit Rahina (Saqqara)', 'Saqqara'],
        'h365df0e518156a7a': ['Alexandria', 'Rhakotis'],
        'h8fb9905c4237f258': ['Abydos'],
    },
    'event': {
        'h32cd992e0967f32d': ['Founding of Memphis', 'Foundation of Memphis'],
    },
    'person': {
        'ha890f9cfdeb53982': ['Narmer (Menes)', 'Menes'],
        'h0ad7b449232e51f2': ['Ramesses II'],
    },
    'artifact': {},
}

RESOLVED = [
    ['city', 'Thebes', None, None],
    ['city', 'Thebes', 'Egyptian History', 'h8fdc46cd4a0f6fb6'],
    ['city', 'Thebes', 'Greek History', None],
    ['city', 'Luxor', 'Egyptian History', 'h8fdc46cd4a0f6fb6'],
    ['city', 'Abydos', 'Byzantine History', 'h8fb9905c4237f258'],
    ['city', 'Abydos', 'Egyptian History', 'h8fb9905c4237f258'],
    ['city', 'Abydos', None, None],
    ['city', 'Rhakotis', None, 'h365df0e518156a7a'],
    ['city', 'Alexandria', 'Greek History', 'h365df0e518156a7a'],
    ['city', 'Tell el-Amarna', None, None],
    ['city', 'Memphis', 'Egyptian History', None],
    ['city', 'Saqqara', None, None],
    ['city', 'Pharos', 'Egyptian History', None],
    ['city', 'Nowhere', None, None],
    ['event', 'Foundation of Memphis', 'Greek History', 'h32cd992e0967f32d'],
    ['event', 'Founding of Memphis', None, 'h32cd992e0967f32d'],
    ['person', 'Menes', 'Byzantine History', None],
    ['person', 'Narmer', 'Greek History', None],
    ['person', 'Ozymandias', None, 'h0ad7b449232e51f2'],
    ['person', 'Menes', 'Egyptian History', 'ha890f9cfdeb53982'],
]


def _assignments(atom_type):
    found = {}
    topics._group_occurrences(
        ((t.subject, t.context) for t in topics._raw_occurrences(atom_type)),
        prefer=lambda s: topics.has_real_atom(atom_type, s),
        aliases=topics.load_aliases(),
        assignments=found,
    )
    return sorted([subject, context, survivor] for (subject, context), survivor in found.items())


@pytest.mark.parametrize("atom_type", ["city", "event", "person", "artifact"])
class TestFrozenGrouping:
    def test_from_history(self, corpus, atom_type):
        got = [[t.subject, t.context] for t in topics.from_history(atom_type)]
        assert got == FROM_HISTORY[atom_type]

    def test_needing_create(self, corpus, atom_type):
        got = [[t.subject, t.context] for t in topics.needing_create(atom_type)]
        assert got == NEEDING_CREATE[atom_type]

    def test_group_occurrences_assignments(self, corpus, atom_type):
        assert _assignments(atom_type) == ASSIGNMENTS[atom_type]

    def test_listed_spellings(self, corpus, atom_type):
        got = {k: v for k, v in topics.listed_spellings(atom_type).items()}
        assert got == LISTED_SPELLINGS[atom_type]


def test_subject_resolver(corpus):
    resolvers = {t: topics.subject_resolver(t) for t in ("city", "event", "person")}
    got = [[t, name, context, resolvers[t](name, context)] for t, name, context in QUERIES]
    assert got == RESOLVED


def test_group_synonyms_is_context_free(corpus):
    """The string form: no contexts, so scoped lines never apply."""
    subjects = [s for lists in HISTORIES.values() for s in lists["cities"]]
    got = topics.group_synonyms(subjects, prefer=lambda s: topics.has_real_atom("city", s),
                                aliases=topics.load_aliases())
    assert got == GROUP_SYNONYMS


GROUP_SYNONYMS = [
    'Thebes (Luxor)',
    'Memphis (Mit Rahina)',
    'Mit Rahina (Saqqara)',
    'Saqqara',
    'Amarna (Akhetaten)',
    'Rhakotis',
    'Abydos',
    'Pharos',
    'Heliopolis (Iunu)',
    'Athens',
    'Thebai',
    'Constantinople',
    'Pharos Island',
]
