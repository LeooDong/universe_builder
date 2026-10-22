"""The identity-file parsers and path resolvers.

The whole-file behaviour is frozen against the real files in
`test_frozen_grouping.py`; these pin the parts each caller relies on.
"""

from __future__ import annotations

from universe_builder import alias_files, settings


def test_an_alias_declaration_carries_its_line_scope_and_provenance():
    text = ("# header\n"
            "Alexandria = Rhakotis   # asserted\n"
            "\n"
            "# Added by `universe_builder duplicates --apply`: one subject, two atoms.\n"
            "[Greek History]\n"
            "Thebes (Boeotia) = Thebai\n"
            "[*]\n"
            "= =\n")
    decls = list(alias_files.parse_alias_text(text))
    assert [(d.number, d.names, d.scope, d.provenance) for d in decls] == [
        (2, ("Alexandria", "Rhakotis"), None, "asserted"),
        (6, ("Thebes (Boeotia)", "Thebai"), "Greek History", "derived"),
        (8, (), None, "derived"),
    ]
    assert decls[0].raw == "Alexandria = Rhakotis   # asserted"
    assert decls[0].body == "Alexandria = Rhakotis"
    assert decls[0].index == 1


def test_a_distinct_line_is_split_on_every_separator():
    text = "# c\nAbbas I != Abbas II != Abbas Mirza  # three\nno separator here\n"
    [decl] = alias_files.parse_distinct_text(text)
    assert decl.number == 2
    assert decl.names == ("Abbas I", "Abbas II", "Abbas Mirza")


def test_a_missing_file_reads_as_empty(tmp_path):
    assert alias_files.AliasFile.read(tmp_path / "absent.txt").declarations == []
    assert alias_files.NotDuplicatesFile.read(tmp_path / "absent.txt").rows == []


def test_the_global_declarations_skip_scoped_lines(tmp_path):
    path = tmp_path / "aliases.txt"
    path.write_text("A = B\n[X]\nC = D\n[]\nE = F\n", encoding="utf-8")
    assert [d.names for d in alias_files.AliasFile.read(path).global_declarations()] == [
        ("A", "B"), ("E", "F")]


def test_the_override_wins_and_the_refusals_sit_beside_it(workspace, tmp_path, monkeypatch):
    ours = tmp_path / "x" / "aliases.txt"
    monkeypatch.setattr(settings, "ALIASES_LIST", ours)
    assert alias_files.aliases_path() == ours
    assert alias_files.not_duplicates_path() == tmp_path / "x" / "not_duplicates.txt"
    assert alias_files.not_duplicates_path(tmp_path / "y" / "a.txt") == tmp_path / "y" / "not_duplicates.txt"
