"""merge_helper: append-file pairing, and the in-place promote.

Contains the B3 regression. The merge promotes: the merged content replaces the
base atom (validated first — writing in place makes validation the difference
between a merge and B8) and the consumed append retires to `_merged/`, so the
atom folders keep exactly one file per atom. Every test isolates through
`workspace`, because retirement targets `settings.merged_dir()`.
"""

from __future__ import annotations

import json

from universe_builder import settings, store
from universe_builder.operations import merge as merge_helper
from tests.test_operations import valid_city

MODEL = "claude-sonnet-5"


def city_payload(refs) -> str:
    payload = json.loads(valid_city())
    payload["popCulturalReferences"] = list(refs)
    return json.dumps(payload, indent=4)


def append_payload(refs) -> str:
    return json.dumps({"popCulturalReferences": list(refs)}, indent=4)


def write_file(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def name(atom_type, subject, model=MODEL) -> str:
    """A content-addressed filename: the only form `merge_helper` pairs (v0.3.0)."""
    return store.AtomName(atom_type, None, subject, model).getReadableFileName()


def city_pair(directory, subject, city_refs, append_refs):
    """Write a matched content-addressed city_*/append_city_* pair into `directory`."""
    city = write_file(directory / name("city", subject), city_payload(city_refs))
    write_file(directory / name("append_city", subject), append_payload(append_refs))
    return city


def refs_of(path) -> list:
    return json.loads(path.read_text(encoding="utf-8"))["popCulturalReferences"]


def retired() -> list:
    folder = settings.merged_dir()
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


class TestMergeInPlace:
    def test_lists_are_extended_not_replaced(self, workspace, tmp_path):
        main = write_file(tmp_path / f"city_0000000001_memphis_{MODEL}.json",
                          city_payload(["Asterix"]))
        append = write_file(tmp_path / "b.json", append_payload(["The Mummy"]))

        assert merge_helper.merge_json_files(main, append) is True
        assert refs_of(main) == ["Asterix", "The Mummy"]

    def test_the_base_atom_is_replaced_and_the_append_retired(self, workspace, tmp_path):
        main = write_file(tmp_path / f"city_0000000001_memphis_{MODEL}.json",
                          city_payload(["Asterix"]))
        append = write_file(tmp_path / f"append_city_0000000001_memphis_{MODEL}.json",
                            append_payload(["The Mummy"]))

        merge_helper.merge_json_files(main, append)

        assert not append.exists(), "a consumed append must leave the atom folder"
        assert retired() == [append.name], "retired, never deleted — data/ has no undo"

    def test_merging_is_idempotent(self, workspace, tmp_path):
        """The write and the retirement are two steps; a crash between them must
        not mean the next run doubles every list entry."""
        main = write_file(tmp_path / f"city_0000000001_memphis_{MODEL}.json",
                          city_payload(["Asterix"]))
        append = write_file(tmp_path / "b.json", append_payload(["The Mummy"]))

        merge_helper.merge_json_files(main, append)
        write_file(append, append_payload(["The Mummy"]))     # the crash-window replay
        merge_helper.merge_json_files(main, append)

        assert refs_of(main) == ["Asterix", "The Mummy"]

    def test_scalars_are_not_overwritten(self, workspace, tmp_path, caplog):
        """An append adds; it never replaces a field that already validated."""
        main = write_file(tmp_path / f"city_0000000001_memphis_{MODEL}.json",
                          city_payload(["Asterix"]))
        extra = json.loads(append_payload(["The Mummy"]))
        extra["shortSummary"] = "a rewrite the append has no business making"
        original_summary = json.loads(city_payload([])).get("shortSummary")
        append = write_file(tmp_path / "b.json", json.dumps(extra))

        with caplog.at_level("WARNING"):
            merge_helper.merge_json_files(main, append)

        merged = json.loads(main.read_text(encoding="utf-8"))
        assert merged.get("shortSummary") == original_summary
        assert refs_of(main) == ["Asterix", "The Mummy"]

    def test_a_merge_that_does_not_validate_touches_nothing(self, workspace, tmp_path, caplog):
        """Writing in place makes validation the difference between a merge and
        B8: an append extending a typed list with a wrongly-shaped element must
        not reach the atom."""
        main = write_file(tmp_path / f"city_0000000001_memphis_{MODEL}.json",
                          city_payload(["Asterix"]))
        before = main.read_text(encoding="utf-8")
        append = write_file(tmp_path / "b.json",
                            json.dumps({"cityPeriodsOfHistory": [{"not": "a period name"}]}))

        with caplog.at_level("ERROR"):
            assert merge_helper.merge_json_files(main, append) is False

        assert main.read_text(encoding="utf-8") == before
        assert append.exists(), "the pair must stay in place for a later run"
        assert "does not validate" in caplog.text

    def test_a_type_mismatched_append_value_keeps_the_base(self, workspace, tmp_path):
        """Only list-into-list extends; a scalar offered for a list keeps the
        base's list, and the merge still promotes cleanly."""
        main = write_file(tmp_path / f"city_0000000001_memphis_{MODEL}.json",
                          city_payload(["Asterix"]))
        append = write_file(tmp_path / "b.json",
                            json.dumps({"popCulturalReferences": "not a list"}))

        assert merge_helper.merge_json_files(main, append) is True
        assert refs_of(main) == ["Asterix"]


class TestUnreadablePairs:
    """One unreadable half skips the pair, loudly, and writes nothing.

    Raising aborts every valid pair after this one; promoting the readable half
    would present an atom missing its append as a completed merge. A skip is
    recoverable; both inputs stay in place.
    """

    def test_a_missing_append_skips_the_pair(self, workspace, tmp_path, caplog):
        main = write_file(tmp_path / f"city_0000000001_memphis_{MODEL}.json",
                          city_payload(["one"]))
        before = main.read_text(encoding="utf-8")

        with caplog.at_level("ERROR"):
            merged = merge_helper.merge_json_files(main, tmp_path / "gone.json")

        assert merged is False
        assert main.read_text(encoding="utf-8") == before
        assert any("gone.json" in r.message for r in caplog.records)

    def test_an_append_that_is_not_json_skips_the_pair(self, workspace, tmp_path):
        main = write_file(tmp_path / f"city_0000000001_memphis_{MODEL}.json",
                          city_payload(["one"]))
        append = write_file(tmp_path / "b.json", "not json {")
        before = main.read_text(encoding="utf-8")

        assert merge_helper.merge_json_files(main, append) is False
        assert main.read_text(encoding="utf-8") == before
        assert append.exists()

    def test_an_unreadable_base_skips_the_pair(self, workspace, tmp_path):
        main = tmp_path / f"city_0000000001_memphis_{MODEL}.json"
        main.parent.mkdir(parents=True, exist_ok=True)
        main.write_bytes(b"\xff\xfe not utf-8 \x9c")
        append = write_file(tmp_path / "b.json", append_payload(["two"]))

        assert merge_helper.merge_json_files(main, append) is False
        assert append.exists()

    def test_a_non_object_top_level_skips_the_pair(self, workspace, tmp_path):
        main = write_file(tmp_path / f"city_0000000001_memphis_{MODEL}.json",
                          city_payload(["one"]))
        append = write_file(tmp_path / "b.json", json.dumps(["a", "bare", "list"]))

        assert merge_helper.merge_json_files(main, append) is False

    def test_one_bad_pair_does_not_stop_the_walk(self, workspace, tmp_path):
        directory = tmp_path / "city"
        write_file(directory / name("city", "abydos"), city_payload(["a"]))
        write_file(directory / name("append_city", "abydos"), "not json {")
        good = city_pair(directory, "memphis", ["a"], ["b"])

        merge_helper.process_directory(str(tmp_path), "city")

        assert refs_of(good) == ["a", "b"]


class TestProcessDirectory:
    def test_a_matched_pair_promotes_in_place(self, workspace, tmp_path):
        city = city_pair(tmp_path / "city", "memphis", ["a"], ["b"])

        merge_helper.process_directory(str(tmp_path), "city")

        assert refs_of(city) == ["a", "b"]
        assert not list((tmp_path / "city").glob("append_*")), "consumed appends retire"
        assert not list(tmp_path.rglob("final_*")), "the merge writes no third file"

    def test_B3_every_nested_directory_is_processed(self, workspace, tmp_path):
        """B3: the merge loop sat outside the `os.walk` loop, so only the last
        walked directory's file lists survived."""
        cities = [
            city_pair(tmp_path / "city" / part, f"city_{i}", ["a"], ["b"])
            for i, part in enumerate(("one", "two", "three"), start=1)
        ]

        merge_helper.process_directory(str(tmp_path), "city")

        unmerged = [c.name for c in cities if refs_of(c) != ["a", "b"]]
        assert not unmerged, f"nested directories skipped: {unmerged}"

    def test_an_unmatched_city_file_is_left_alone(self, workspace, tmp_path):
        directory = tmp_path / "city"
        city = write_file(directory / name("city", "memphis"), city_payload(["a"]))
        before = city.read_text(encoding="utf-8")

        merge_helper.process_directory(str(tmp_path), "city")

        assert city.read_text(encoding="utf-8") == before

    def test_an_append_for_a_different_subject_does_not_match(self, workspace, tmp_path):
        directory = tmp_path / "city"
        city = write_file(directory / name("city", "memphis"), city_payload(["a"]))
        append = write_file(directory / name("append_city", "thebes"), append_payload(["b"]))

        merge_helper.process_directory(str(tmp_path), "city")

        assert refs_of(city) == ["a"]
        assert append.exists()

    def test_content_addressed_files_that_slug_alike_do_not_cross_merge(self, workspace):
        """`St. Petersburg` and `St Petersburg` slug alike (`st_petersburg`) but
        hash apart — punctuation is significant in the content id, not the slug.
        Two content-addressed files must pair on the id alone; nothing may bridge
        them on the slug, or an append for one subject silently merges into the
        other's file."""
        city = write_file(store.path_for("city", None, "St. Petersburg", MODEL),
                          city_payload(["a"]))
        append = write_file(store.path_for("append_city", None, "St Petersburg", MODEL),
                            append_payload(["b"]))

        merge_helper.process_directory(str(settings.data_dir()), "city")

        assert refs_of(city) == ["a"], "the append for a different subject must not merge"
        assert append.exists(), "the mismatched append must be left in place"

    def test_other_atom_types_are_ignored(self, workspace, tmp_path):
        directory = tmp_path / "person"
        person = write_file(directory / name("person", "imhotep"), city_payload(["a"]))
        before = person.read_text(encoding="utf-8")
        write_file(directory / name("append_person", "imhotep"), append_payload(["b"]))

        merge_helper.process_directory(str(tmp_path), "city")

        assert person.read_text(encoding="utf-8") == before

    def test_an_empty_tree_is_a_no_op(self, workspace, tmp_path):
        merge_helper.process_directory(str(tmp_path), "city")
        assert retired() == []

    def test_an_unreadable_filename_does_not_abort_the_walk(self, workspace, tmp_path):
        """One stray file used to raise out of the whole merge."""
        directory = tmp_path / "city"
        write_file(directory / "city_not_a_real_name.json", city_payload(["x"]))
        good = city_pair(directory, "memphis", ["a"], ["b"])

        merge_helper.process_directory(str(tmp_path), "city")

        assert refs_of(good) == ["a", "b"]


class TestPairingAcrossNamingSchemes:
    """Which append belongs to which atom, now that ids mean something.

    A pair matches on the content id alone. The slug-plus-model rule that bridged
    legacy counter-id names was deleted in v0.3.0 with the rest of the legacy-id
    support.
    """

    def content_pair(self, directory, subject, city_refs, append_refs,
                     city_model=MODEL, append_model=MODEL):
        city_name = store.AtomName("city", None, subject, city_model).getReadableFileName()
        append_name = store.AtomName("append_city", None, subject, append_model).getReadableFileName()
        city = write_file(directory / city_name, city_payload(city_refs))
        write_file(directory / append_name, append_payload(append_refs))
        return city

    def test_content_addressed_names_pair(self, workspace, tmp_path):
        city = self.content_pair(tmp_path / "city", "Memphis", ["a"], ["b"])

        merge_helper.process_directory(str(tmp_path), "city")

        assert refs_of(city) == ["a", "b"]

    def test_content_addressed_names_pair_across_different_models(self, workspace, tmp_path):
        """The unpinned append/create case: the append phase no longer has to
        use the create phase's model, because the id is the pair key."""
        city = self.content_pair(
            tmp_path / "city", "Memphis", ["a"], ["b"],
            city_model="gpt-5.6-luna", append_model="gpt-5-nano",
        )

        merge_helper.process_directory(str(tmp_path), "city")

        assert refs_of(city) == ["a", "b"]

    def test_different_subjects_do_not_pair(self, workspace, tmp_path):
        directory = tmp_path / "city"
        city = write_file(
            directory / store.AtomName("city", None, "Memphis", MODEL).getReadableFileName(),
            city_payload(["a"]))
        write_file(
            directory / store.AtomName("append_city", None, "Thebes", MODEL).getReadableFileName(),
            append_payload(["b"]))

        merge_helper.process_directory(str(tmp_path), "city")

        assert refs_of(city) == ["a"]

    def test_a_matching_append_is_merged_once_not_once_per_key(self, workspace, tmp_path):
        """One append, merged once; merging it twice would duplicate every list
        entry — protected by match dedup and by the idempotent extend."""
        city = self.content_pair(tmp_path / "city", "Memphis", ["a"], ["b"])

        merge_helper.process_directory(str(tmp_path), "city")

        assert refs_of(city) == ["a", "b"]
