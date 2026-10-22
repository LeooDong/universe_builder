"""`duplicates.plan` as a whole, and what the alias-file writers leave alone.

`test_duplicates.py` tests each branch of `plan()` on its own; v0.2.0 split
`plan()` into `_decide_component()` and computed
`listed_spellings` / `_generation_keys` once, and what has to survive that is
the *whole* answer — every resolution, undecided entry and skip, in order, with
its `why`. So one corpus here reaches every branch at once and the Plan is
frozen as data (written from v0.1.19).

The second half pins the alias-file writers (v0.1.20 rewrote them): untouched
lines survive byte for byte, comments and the trailing newline stay, the
provenance header and `[*]` are written before appended lines, a scoped line is
never merged into, and a multi-way `!=` line is one claim.
"""

from __future__ import annotations

import json

import pytest

from universe_builder import duplicates, lint, settings, store
from universe_builder.topics import AliasGroup

FOURO, FIVE, LUNA, ASTRA = "gpt-4o-mini", "gpt-5-nano", "gpt-5.6-luna", "gpt-6-astra"

#: (atom type, generation subject, model, common, otherNames, extra filled fields)
ATOMS = [
    # later family wins; the alias line carries the listed spellings
    ("event", "Founding of Memphis", FOURO, "Founding of Memphis", [], 0),
    ("event", "Foundation of Memphis", FIVE, "Foundation of Memphis", ["Founding of Memphis"], 0),
    # one family: nothing in a filename ranks them
    ("person", "Abbas I", FIVE, "Abbas I", ["Abbas"], 0),
    ("person", "Abbas II", FIVE, "Abbas II", ["Abbas"], 0),
    # no comparable model number
    ("person", "Local One", "local-model", "Local One", ["Localname"], 0),
    ("person", "Local Two", "other-model", "Local Two", ["Localname"], 0),
    # --except
    ("artifact", "Bibliotheca Alexandrina", FIVE, "Bibliotheca Alexandrina", ["Library of Alexandria"], 0),
    ("artifact", "Library of Alexandria", FOURO, "Library of Alexandria", [], 0),
    # not_duplicates
    ("person", "Psamtik I", FIVE, "Psamtik I", ["Wahibre"], 0),
    ("person", "Apries", FOURO, "Apries", ["Wahibre"], 0),
    # --keep names one survivor
    ("event", "Battle of Kadesh", FIVE, "Battle of Kadesh", [], 0),
    ("event", "Kadesh battle", FIVE, "Kadesh battle", ["Battle of Kadesh"], 0),
    # --keep matching both files
    ("event", "Nika riots", FIVE, "Nika riots", ["Nika"], 0),
    ("event", "Nika revolt", FIVE, "Nika revolt", ["Nika"], 0),
    # --keep, but the line would capture a name the people list
    ("event", "Reign of Djoser (Netjerikhet)", FOURO, "Reign of Djoser (Netjerikhet)", [], 0),
    ("event", "Djosers reign", FIVE, "Djosers reign", ["Reign of Djoser (Netjerikhet)"], 0),
    # no --keep, same trespass
    ("event", "Unification of Egypt (Narmer)", FOURO, "Unification of Egypt (Narmer)", [], 0),
    ("event", "Unification of Egypt", FIVE, "Unification of Egypt", ["Unification of Egypt (Narmer)"], 0),
    # tie-break by fill, declared one subject by an asserted line
    ("person", "Ramesses III", LUNA, "Ramesses III", ["Rameses III"], 3),
    ("person", "Rameses III", LUNA, "Rameses III", [], 0),
    # tie-break by fill, and the fill ties too
    ("person", "Thutmose I", LUNA, "Thutmose I", ["Tuthmosis I"], 0),
    ("person", "Tuthmosis I", LUNA, "Tuthmosis I", ["Thutmose I"], 0),
    # an asserted line covers them, but no name on it reaches the survivor's file
    ("person", "Hatshepsut (pharaoh)", FIVE, "Hatshepsut", ["Maatkare"], 0),
    ("person", "Maatkare", FOURO, "Maatkare", [], 0),
    # a chain: B wins one pair and loses the other
    ("work", "Chain A", FOURO, "Chain A", ["Chainlink"], 0),
    ("work", "Chain B", FIVE, "Chain B", ["Chainlink", "Chainbolt"], 0),
    ("work", "Chain C", ASTRA, "Chain C", ["Chainbolt"], 0),
    # two survivors claiming one loser
    ("work", "Star A", FIVE, "Star A", ["Starloser"], 0),
    ("work", "Starloser", FOURO, "Starloser", [], 0),
    ("work", "Star C", LUNA, "Star C", ["Starloser alt"], 0),
    ("work", "Starloser alt", FOURO, "Starloser alt", ["Starloser"], 0),
    # the new line would merge two existing global lines
    ("person", "Senusret I", FIVE, "Senusret I", ["Sesostris"], 0),
    ("person", "Sesostris", FOURO, "Sesostris", [], 0),
]

HISTORY = {
    "events": ["Founding of Memphis", "Foundation of Memphis", "Battle of Kadesh", "Nika riots",
               "Reign of Djoser (Netjerikhet)", "Unification of Egypt (Narmer)"],
    "people": ["Abbas I", "Abbas II", "Netjerikhet", "Narmer", "Ramesses III", "Thutmose I", "Maatkare",
               "Senusret I", "Sesostris"],
    "artifacts": [], "discoveries": [], "cities": [], "works": ["Chain A", "Star A"],
}

ALIASES = """\
# pack alias file
Senusret I = Kheperkare
Sesostris = Senwosret
Ramesses III = Rameses III
Thutmose I = Tuthmosis I
Hatshepsut = Maatkare
"""

DISTINCT = "Psamtik I != Apries\n"


@pytest.fixture
def corpus(workspace, tmp_path, monkeypatch, pack_overrides):
    aliases = tmp_path / "aliases.txt"
    aliases.write_text(ALIASES, encoding="utf-8")
    (tmp_path / "not_duplicates.txt").write_text(DISTINCT, encoding="utf-8")
    monkeypatch.setattr(settings, "ALIASES_LIST", aliases)
    pack_overrides(aliases_path=aliases)
    seed = tmp_path / "HISTORY_list.txt"
    seed.write_text("Egyptian History\n", encoding="utf-8")
    monkeypatch.setattr(settings, "HISTORY_LIST", seed)
    history = store.path_for("history", None, "Egyptian History", FIVE)
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_text(json.dumps(HISTORY), encoding="utf-8")
    for atom_type, subject, model, common, other, extra in ATOMS:
        path = store.path_for(atom_type, None, subject, model)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"name": {"common": common, "otherNames": other}}
        payload.update({f"field{i}": "x" for i in range(extra)})
        path.write_text(json.dumps(payload), encoding="utf-8")
    return workspace


def snapshot(plan) -> dict:
    return {
        "resolutions": [{"atom_type": r.atom_type, "name": r.name, "keep": r.keep.name,
                         "retire": [p.name for p in r.retire], "spellings": r.spellings,
                         "reason": r.reason} for r in plan.resolutions],
        "undecided": plan.undecided,
        "skipped": plan.skipped,
        "alias_lines": plan.alias_lines(),
        "summary": plan.summary(),
    }


def test_the_default_plan(corpus):
    assert snapshot(duplicates.plan()) == DEFAULT_PLAN


def test_the_plan_with_every_option(corpus):
    plan = duplicates.plan(exclude=["bibliotheca"], keep=["kadesh_battle", "nika", "djosers_reign"],
                           tiebreak_fill=True)
    assert snapshot(plan) == OPTIONS_PLAN


#: Frozen from v0.1.19. Two things worth knowing when one flips:
#:  * The chain (`Chain A`-`Chain B`-`Chain C`) is reported with the
#:    two-survivors message: a chain always has two keepers, so `_stars_only`'s
#:    "survivor of one pair and the loser of another" branch is unreachable.
#:  * Without `--except`, `Bibliotheca Alexandrina` retires `Library of
#:    Alexandria` — the related-work case `--except` exists for.
DEFAULT_PLAN = {
    'resolutions': [
        {
            'atom_type': 'event',
            'name': 'founding of memphis',
            'keep': 'event_hb5b746e46a3be9bb_foundation_of_memphis_gpt-5-nano.json',
            'retire': ['event_h32cd992e0967f32d_founding_of_memphis_gpt-4o-mini.json'],
            'spellings': ['Foundation of Memphis', 'Founding of Memphis'],
            'reason': 'gpt-5-nano outranks gpt-4o-mini',
        },
        {
            'atom_type': 'artifact',
            'name': 'library of alexandria',
            'keep': 'artifact_h9a9304737cbd00ef_bibliotheca_alexandrina_gpt-5-nano.json',
            'retire': ['artifact_h27cbf26a33487b35_library_of_alexandria_gpt-4o-mini.json'],
            'spellings': ['Bibliotheca Alexandrina', 'Library of Alexandria'],
            'reason': 'gpt-5-nano outranks gpt-4o-mini',
        },
    ],
    'undecided': [
        {
            'atom_type': 'person',
            'name': 'abbas',
            'files': [
                'person_h8192531bbd744050_abbas_ii_gpt-5-nano.json',
                'person_hfb2e34eae6d75c86_abbas_i_gpt-5-nano.json',
            ],
            'why': '2 atoms share the highest model family (5); nothing in a filename ranks them',
        },
        {
            'atom_type': 'person',
            'name': 'localname',
            'files': [
                'person_hd44e2015ba2d3f53_local_two_other-model.json',
                'person_hfff2e5ff5c966279_local_one_local-model.json',
            ],
            'why': 'no model version carries a comparable number',
        },
        {
            'atom_type': 'person',
            'name': 'rameses iii',
            'files': [
                'person_h4921ac0736907f29_ramesses_iii_gpt-5.6-luna.json',
                'person_h87729d7bd729eb50_rameses_iii_gpt-5.6-luna.json',
            ],
            'why': '2 atoms share the highest model family (5.6); nothing in a filename ranks them',
        },
        {
            'atom_type': 'person',
            'name': 'thutmose i, tuthmosis i',
            'files': [
                'person_h4a8107bbfada60f1_thutmose_i_gpt-5.6-luna.json',
                'person_ha3dce94694604b02_tuthmosis_i_gpt-5.6-luna.json',
            ],
            'why': '2 atoms share the highest model family (5.6); nothing in a filename ranks them',
        },
        {
            'atom_type': 'event',
            'name': 'battle of kadesh',
            'files': [
                'event_h21c0f6242075a61e_kadesh_battle_gpt-5-nano.json',
                'event_h8fc224e5dab99d5a_battle_of_kadesh_gpt-5-nano.json',
            ],
            'why': '2 atoms share the highest model family (5); nothing in a filename ranks them',
        },
        {
            'atom_type': 'event',
            'name': 'nika',
            'files': [
                'event_h02aa723620f493fc_nika_riots_gpt-5-nano.json',
                'event_hd346139fe39ce24b_nika_revolt_gpt-5-nano.json',
            ],
            'why': '2 atoms share the highest model family (5); nothing in a filename ranks them',
        },
        {
            'atom_type': 'event',
            'name': 'reign of djoser netjerikhet',
            'files': [
                'event_h97923b07d7fcc71c_djosers_reign_gpt-5-nano.json',
                'event_he9486fcf1ac28573_reign_of_djoser_netjerikhet_gpt-4o-mini.json',
            ],
            'why': (
                "the alias line would claim ['netjerikhet'], which is a name another atom type lists"
                '; write a line by hand'
            ),
        },
        {
            'atom_type': 'event',
            'name': 'unification of egypt narmer',
            'files': [
                'event_h46d9e369792fa59d_unification_of_egypt_narmer_gpt-4o-mini.json',
                'event_hdc3a02c38fbf500a_unification_of_egypt_gpt-5-nano.json',
            ],
            'why': (
                "the alias line would claim ['narmer'], which is a name another atom type lists; wri"
                'te a line by hand'
            ),
        },
        {
            'atom_type': 'person',
            'name': 'maatkare',
            'files': [
                'person_h2a160ed1c312f5da_hatshepsut_pharaoh_gpt-5-nano.json',
                'person_hf885fdb0998b1245_maatkare_gpt-4o-mini.json',
            ],
            'why': (
                'an asserted alias line covers these atoms, but none of its names the work list woul'
                'd try resolves to the survivor person_h2a160ed1c312f5da_hatshepsut_pharaoh_gpt-5-na'
                'no.json: add the spelling it was generated under, ahead of any spelling that folds '
                'to the same key'
            ),
        },
        {
            'atom_type': 'work',
            'name': 'chainbolt',
            'files': [
                'work_hc89dd8871ef1df2e_chain_b_gpt-5-nano.json',
                'work_hf2870340d0e55c9c_chain_c_gpt-6-astra.json',
            ],
            'why': (
                'two atoms here would each survive while both claim the same retired atom, and decla'
                'rations never chain, so nothing says they are the same thing; pick the survivor by '
                'hand'
            ),
        },
        {
            'atom_type': 'work',
            'name': 'chainlink',
            'files': [
                'work_hbc2f99f7bd463a95_chain_a_gpt-4o-mini.json',
                'work_hc89dd8871ef1df2e_chain_b_gpt-5-nano.json',
            ],
            'why': (
                'two atoms here would each survive while both claim the same retired atom, and decla'
                'rations never chain, so nothing says they are the same thing; pick the survivor by '
                'hand'
            ),
        },
        {
            'atom_type': 'work',
            'name': 'starloser',
            'files': [
                'work_h67dad8d16a2c9a5e_star_a_gpt-5-nano.json',
                'work_h93bc2c9c871119cf_starloser_alt_gpt-4o-mini.json',
                'work_hd2bf9661c8dcefed_starloser_gpt-4o-mini.json',
            ],
            'why': (
                'two atoms here would each survive while both claim the same retired atom, and decla'
                'rations never chain, so nothing says they are the same thing; pick the survivor by '
                'hand'
            ),
        },
        {
            'atom_type': 'work',
            'name': 'starloser alt',
            'files': [
                'work_h70dc4316a69634fb_star_c_gpt-5.6-luna.json',
                'work_h93bc2c9c871119cf_starloser_alt_gpt-4o-mini.json',
            ],
            'why': (
                'two atoms here would each survive while both claim the same retired atom, and decla'
                'rations never chain, so nothing says they are the same thing; pick the survivor by '
                'hand'
            ),
        },
        {
            'atom_type': 'person',
            'name': 'sesostris',
            'files': [
                'person_h4fb691efb6839959_senusret_i_gpt-5-nano.json',
                'person_h8f8802b9c7c8d63e_sesostris_gpt-4o-mini.json',
            ],
            'why': (
                'the alias line would merge two existing declarations, joining names nothing declare'
                'd equal; write the line by hand'
            ),
        },
    ],
    'skipped': [
        {
            'atom_type': 'person',
            'name': 'wahibre',
            'files': [
                'person_h56edcb4146a1e446_psamtik_i_gpt-5-nano.json',
                'person_h88194968ba9d2270_apries_gpt-4o-mini.json',
            ],
        },
    ],
    'alias_lines': ['Foundation of Memphis = Founding of Memphis', 'Bibliotheca Alexandrina = Library of Alexandria'],
    'summary': '2 pair(s) resolvable, 14 undecided, 1 excluded',
}

OPTIONS_PLAN = {
    'resolutions': [
        {
            'atom_type': 'person',
            'name': 'rameses iii',
            'keep': 'person_h4921ac0736907f29_ramesses_iii_gpt-5.6-luna.json',
            'retire': ['person_h87729d7bd729eb50_rameses_iii_gpt-5.6-luna.json'],
            'spellings': [],
            'reason': 'same model family (5.6); keeps the fuller atom — 5 populated fields against 1',
        },
        {
            'atom_type': 'event',
            'name': 'battle of kadesh',
            'keep': 'event_h21c0f6242075a61e_kadesh_battle_gpt-5-nano.json',
            'retire': ['event_h8fc224e5dab99d5a_battle_of_kadesh_gpt-5-nano.json'],
            'spellings': ['Kadesh battle', 'Battle of Kadesh'],
            'reason': 'named by --keep',
        },
        {
            'atom_type': 'event',
            'name': 'founding of memphis',
            'keep': 'event_hb5b746e46a3be9bb_foundation_of_memphis_gpt-5-nano.json',
            'retire': ['event_h32cd992e0967f32d_founding_of_memphis_gpt-4o-mini.json'],
            'spellings': ['Foundation of Memphis', 'Founding of Memphis'],
            'reason': 'gpt-5-nano outranks gpt-4o-mini',
        },
    ],
    'undecided': [
        {
            'atom_type': 'person',
            'name': 'abbas',
            'files': [
                'person_h8192531bbd744050_abbas_ii_gpt-5-nano.json',
                'person_hfb2e34eae6d75c86_abbas_i_gpt-5-nano.json',
            ],
            'why': '2 atoms share the highest model family (5); nothing in a filename ranks them',
        },
        {
            'atom_type': 'person',
            'name': 'localname',
            'files': [
                'person_hd44e2015ba2d3f53_local_two_other-model.json',
                'person_hfff2e5ff5c966279_local_one_local-model.json',
            ],
            'why': 'no model version carries a comparable number',
        },
        {
            'atom_type': 'person',
            'name': 'thutmose i, tuthmosis i',
            'files': [
                'person_h4a8107bbfada60f1_thutmose_i_gpt-5.6-luna.json',
                'person_ha3dce94694604b02_tuthmosis_i_gpt-5.6-luna.json',
            ],
            'why': (
                '2 atoms share the highest model family (5.6) and fill the same number of fields (2)'
                '; nothing separates them'
            ),
        },
        {
            'atom_type': 'event',
            'name': 'nika',
            'files': [
                'event_h02aa723620f493fc_nika_riots_gpt-5-nano.json',
                'event_hd346139fe39ce24b_nika_revolt_gpt-5-nano.json',
            ],
            'why': '--keep matches more than one of these files; narrow the pattern to name a single survivor',
        },
        {
            'atom_type': 'event',
            'name': 'reign of djoser netjerikhet',
            'files': [
                'event_h97923b07d7fcc71c_djosers_reign_gpt-5-nano.json',
                'event_he9486fcf1ac28573_reign_of_djoser_netjerikhet_gpt-4o-mini.json',
            ],
            'why': (
                "--keep names a survivor, but the alias line would claim ['netjerikhet'], which anot"
                'her atom type lists; write the line by hand'
            ),
        },
        {
            'atom_type': 'event',
            'name': 'unification of egypt narmer',
            'files': [
                'event_h46d9e369792fa59d_unification_of_egypt_narmer_gpt-4o-mini.json',
                'event_hdc3a02c38fbf500a_unification_of_egypt_gpt-5-nano.json',
            ],
            'why': (
                "the alias line would claim ['narmer'], which is a name another atom type lists; wri"
                'te a line by hand'
            ),
        },
        {
            'atom_type': 'person',
            'name': 'maatkare',
            'files': [
                'person_h2a160ed1c312f5da_hatshepsut_pharaoh_gpt-5-nano.json',
                'person_hf885fdb0998b1245_maatkare_gpt-4o-mini.json',
            ],
            'why': (
                'an asserted alias line covers these atoms, but none of its names the work list woul'
                'd try resolves to the survivor person_h2a160ed1c312f5da_hatshepsut_pharaoh_gpt-5-na'
                'no.json: add the spelling it was generated under, ahead of any spelling that folds '
                'to the same key'
            ),
        },
        {
            'atom_type': 'work',
            'name': 'chainbolt',
            'files': [
                'work_hc89dd8871ef1df2e_chain_b_gpt-5-nano.json',
                'work_hf2870340d0e55c9c_chain_c_gpt-6-astra.json',
            ],
            'why': (
                'two atoms here would each survive while both claim the same retired atom, and decla'
                'rations never chain, so nothing says they are the same thing; pick the survivor by '
                'hand'
            ),
        },
        {
            'atom_type': 'work',
            'name': 'chainlink',
            'files': [
                'work_hbc2f99f7bd463a95_chain_a_gpt-4o-mini.json',
                'work_hc89dd8871ef1df2e_chain_b_gpt-5-nano.json',
            ],
            'why': (
                'two atoms here would each survive while both claim the same retired atom, and decla'
                'rations never chain, so nothing says they are the same thing; pick the survivor by '
                'hand'
            ),
        },
        {
            'atom_type': 'work',
            'name': 'starloser',
            'files': [
                'work_h67dad8d16a2c9a5e_star_a_gpt-5-nano.json',
                'work_h93bc2c9c871119cf_starloser_alt_gpt-4o-mini.json',
                'work_hd2bf9661c8dcefed_starloser_gpt-4o-mini.json',
            ],
            'why': (
                'two atoms here would each survive while both claim the same retired atom, and decla'
                'rations never chain, so nothing says they are the same thing; pick the survivor by '
                'hand'
            ),
        },
        {
            'atom_type': 'work',
            'name': 'starloser alt',
            'files': [
                'work_h70dc4316a69634fb_star_c_gpt-5.6-luna.json',
                'work_h93bc2c9c871119cf_starloser_alt_gpt-4o-mini.json',
            ],
            'why': (
                'two atoms here would each survive while both claim the same retired atom, and decla'
                'rations never chain, so nothing says they are the same thing; pick the survivor by '
                'hand'
            ),
        },
        {
            'atom_type': 'person',
            'name': 'sesostris',
            'files': [
                'person_h4fb691efb6839959_senusret_i_gpt-5-nano.json',
                'person_h8f8802b9c7c8d63e_sesostris_gpt-4o-mini.json',
            ],
            'why': (
                'the alias line would merge two existing declarations, joining names nothing declare'
                'd equal; write the line by hand'
            ),
        },
    ],
    'skipped': [
        {
            'atom_type': 'person',
            'name': 'wahibre',
            'files': [
                'person_h56edcb4146a1e446_psamtik_i_gpt-5-nano.json',
                'person_h88194968ba9d2270_apries_gpt-4o-mini.json',
            ],
        },
        {
            'atom_type': 'artifact',
            'name': 'library of alexandria',
            'files': [
                'artifact_h27cbf26a33487b35_library_of_alexandria_gpt-4o-mini.json',
                'artifact_h9a9304737cbd00ef_bibliotheca_alexandrina_gpt-5-nano.json',
            ],
        },
    ],
    'alias_lines': ['Kadesh battle = Battle of Kadesh', 'Foundation of Memphis = Founding of Memphis'],
    'summary': '3 pair(s) resolvable, 12 undecided, 2 excluded',
}


# ---------------------------------------------------------------------------
# Alias-file writes
# ---------------------------------------------------------------------------

BEFORE = """\
# Topic aliases — header comment, kept verbatim.

Amarna (Akhetaten) = Tell el-Amarna   # an inline comment
Alexandria = Rhakotis

[Greek History]
Thebes (Boeotia) = Thebes = Thebai
[*]
# Added by `universe_builder duplicates --apply`: one subject, two atoms.
[*]
Founding of Memphis = Foundation of Memphis
"""


def _resolution(keep, retire, spellings, atom_type="event"):
    from pathlib import Path

    return duplicates.Resolution(atom_type=atom_type, name=spellings[0].lower(), keep=Path(keep),
                                 retire=[Path(retire)], spellings=spellings, reason="test")


class TestAliasFileWrites:
    def test_a_new_line_is_appended_under_its_provenance_and_nothing_else_moves(self, workspace, tmp_path):
        path = tmp_path / "aliases.txt"
        path.write_text(BEFORE, encoding="utf-8")
        plan = duplicates.Plan(resolutions=[_resolution("k.json", "r.json", ["Battle of Kadesh", "Kadesh battle"])])

        duplicates.apply(plan, aliases_path=path, dry_run=False)

        assert path.read_text(encoding="utf-8") == BEFORE + (
            "\n# Added by `universe_builder duplicates --apply`: one subject, two atoms.\n"
            "[*]\nBattle of Kadesh = Kadesh battle\n")

    def test_a_line_sharing_a_global_name_is_merged_in_place(self, workspace, tmp_path):
        path = tmp_path / "aliases.txt"
        path.write_text(BEFORE, encoding="utf-8")
        plan = duplicates.Plan(resolutions=[_resolution("k.json", "r.json", ["Alexandria", "Racote"])])

        duplicates.apply(plan, aliases_path=path, dry_run=False)

        # Only the merged row changes; its inline comment (if it had one) is not kept.
        assert path.read_text(encoding="utf-8") == BEFORE.replace(
            "Alexandria = Rhakotis\n", "Alexandria = Rhakotis = Racote\n")

    def test_a_merged_row_loses_its_inline_comment_today(self, workspace, tmp_path):
        path = tmp_path / "aliases.txt"
        path.write_text(BEFORE, encoding="utf-8")
        plan = duplicates.Plan(resolutions=[_resolution("k.json", "r.json", ["Amarna (Akhetaten)", "Akhetaten"])])

        duplicates.apply(plan, aliases_path=path, dry_run=False)

        assert path.read_text(encoding="utf-8") == BEFORE.replace(
            "Amarna (Akhetaten) = Tell el-Amarna   # an inline comment\n",
            "Amarna (Akhetaten) = Tell el-Amarna = Akhetaten\n")

    def test_a_scoped_line_is_never_merged_into(self, workspace, tmp_path):
        path = tmp_path / "aliases.txt"
        path.write_text(BEFORE, encoding="utf-8")
        plan = duplicates.Plan(resolutions=[_resolution("k.json", "r.json", ["Thebai", "Thebes of Boeotia"],
                                                        atom_type="city")])

        duplicates.apply(plan, aliases_path=path, dry_run=False)

        text = path.read_text(encoding="utf-8")
        assert "Thebes (Boeotia) = Thebes = Thebai\n" in text
        assert text.endswith("[*]\nThebai = Thebes of Boeotia\n")

    def test_a_missing_trailing_newline_is_restored(self, workspace, tmp_path):
        path = tmp_path / "aliases.txt"
        path.write_text(BEFORE.rstrip("\n"), encoding="utf-8")
        plan = duplicates.Plan(resolutions=[_resolution("k.json", "r.json", ["Alexandria", "Racote"])])

        duplicates.apply(plan, aliases_path=path, dry_run=False)

        assert path.read_text(encoding="utf-8").endswith("Founding of Memphis = Foundation of Memphis\n")

    def test_lint_apply_review_writes_an_asserted_section(self, workspace, tmp_path, pack_overrides):
        path = tmp_path / "aliases.txt"
        path.write_text(BEFORE, encoding="utf-8")
        pack_overrides(aliases_path=path)

        lint.apply_review([("same", "person", "f.json", "Usermaatre", "Ramesses II")], dry_run=False)

        assert path.read_text(encoding="utf-8") == BEFORE + (
            "\n# Added by `universe_builder lint --apply-review`: adjudicated identity.\n"
            "[*]\nRamesses II = Usermaatre\n")


class TestNotDuplicatesWrites:
    @pytest.fixture
    def files(self, workspace, tmp_path, pack_overrides):
        aliases = tmp_path / "aliases.txt"
        aliases.write_text("Book of the Dead = Papyrus of Ani\n", encoding="utf-8")
        pack_overrides(aliases_path=aliases)
        distinct = tmp_path / "not_duplicates.txt"
        distinct.write_text("# header\nAbbas I != Abbas II != Abbas Mirza\nNicaea != Nice\n", encoding="utf-8")
        return distinct

    def test_a_multi_way_line_is_one_claim(self, files):
        assert duplicates.load_distinct(files) == (
            frozenset({"abbas i", "abbas ii", "abbas mirza"}), frozenset({"nicaea", "nice"}))

    def test_an_all_different_component_is_written_as_one_multi_way_line(self, files):
        records = [{"atom_type": "person", "component": 1, "a": a, "b": b, "verdict": "different"}
                   for a, b in (("Psamtik I", "Apries"), ("Psamtik I", "Necho II"), ("Apries", "Necho II"))]
        outcome = duplicates.write_distinct_lines(records, files, dry_run=False)
        assert outcome["new_lines"] == ["Psamtik I != Apries != Necho II"]
        assert files.read_text(encoding="utf-8") == (
            "# header\nAbbas I != Abbas II != Abbas Mirza\nNicaea != Nice\n"
            "\n# Adjudicated by `universe_builder duplicates --via model`.\n"
            "Psamtik I != Apries != Necho II\n")

    def test_a_pair_the_alias_file_joins_is_refused(self, files):
        records = [{"atom_type": "work", "component": 1, "a": "Book of the Dead", "b": "Papyrus of Ani",
                    "verdict": "different"}]
        outcome = duplicates.write_distinct_lines(records, files, dry_run=False)
        assert outcome["new_lines"] == []
        assert outcome["contradicts_alias"] == ["Book of the Dead != Papyrus of Ani"]

    def test_a_pair_inside_a_multi_way_line_is_already_present(self, files):
        records = [{"atom_type": "person", "component": 1, "a": "Abbas Mirza", "b": "Abbas I",
                    "verdict": "different"}]
        outcome = duplicates.write_distinct_lines(records, files, dry_run=False)
        assert (outcome["new_lines"], outcome["already_present"]) == ([], 1)

    def test_a_pair_already_present_is_skipped(self, files):
        records = [{"atom_type": "city", "component": 1, "a": "Nice", "b": "Nicaea", "verdict": "different"}]
        outcome = duplicates.write_distinct_lines(records, files, dry_run=False)
        assert (outcome["new_lines"], outcome["already_present"]) == ([], 1)


def test_alias_groups_are_injectable(corpus):
    """`plan(aliases=...)` replaces the file for the safeguards: with no asserted
    line, the fill tie-break has no gate and the Ramesses pair stays undecided."""
    plan = duplicates.plan(tiebreak_fill=True, aliases=(AliasGroup(names=("Unrelated", "Other")),))
    files = {f for entry in plan.undecided for f in entry["files"]}
    assert any("ramesses_iii" in f for f in files)


# ---------------------------------------------------------------------------
# One decision per component; the work list read once
# ---------------------------------------------------------------------------

def test_listed_spellings_is_read_once_per_atom_type(corpus, monkeypatch):
    from universe_builder import topics

    calls = []
    real = topics.listed_spellings
    monkeypatch.setattr(topics, "listed_spellings", lambda t: calls.append(t) or real(t))
    plan = duplicates.plan(tiebreak_fill=True)
    assert len(calls) == len(set(calls)), calls
    assert snapshot(plan)["summary"] == duplicates.plan(tiebreak_fill=True).summary()


def test_decide_component_returns_one_of_three_kinds(corpus):
    from universe_builder import topics

    def paths(*subjects_models):
        return [store.path_for(t, None, s, m) for t, s, m in subjects_models]

    rules = duplicates._Rules(patterns=["bibliotheca"], wanted=[], claims=duplicates.load_distinct(),
                              alias_groups=topics.load_aliases(), owners=topics.listed_keys_by_type(),
                              tiebreak_fill=False)
    listed = topics.listed_spellings("event")
    won = duplicates._decide_component(
        "event", "founding of memphis",
        paths(("event", "Founding of Memphis", FOURO), ("event", "Foundation of Memphis", FIVE)),
        listed, rules)
    assert isinstance(won, duplicates.Resolution) and won.keep.name.endswith(f"_{FIVE}.json")
    tie = duplicates._decide_component(
        "person", "abbas", paths(("person", "Abbas I", FIVE), ("person", "Abbas II", FIVE)),
        topics.listed_spellings("person"), rules)
    assert isinstance(tie, duplicates.Undecided)
    assert list(tie.as_dict()) == ["atom_type", "name", "files", "why"]
    skipped = duplicates._decide_component(
        "artifact", "library of alexandria",
        paths(("artifact", "Bibliotheca Alexandrina", FIVE), ("artifact", "Library of Alexandria", FOURO)),
        topics.listed_spellings("artifact"), rules)
    assert isinstance(skipped, duplicates.Skipped)
    assert skipped.as_dict() == {"atom_type": "artifact", "name": "library of alexandria",
                                 "files": sorted(p.name for p in paths(
                                     ("artifact", "Bibliotheca Alexandrina", FIVE),
                                     ("artifact", "Library of Alexandria", FOURO)))}
