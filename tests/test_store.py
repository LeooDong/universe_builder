"""`universe_builder.store` — the single owner of paths, the `_ERROR` sentinel and atomic writes.

No network, no API key. `workspace` redirects `UNIVERSE_BUILDER_DATA_DIR` at a tmp_path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from universe_builder import store
from universe_builder.models.subatomic import EmptyModel


class TestNaming:
    def test_path_lands_under_the_atom_type_folder(self, workspace):
        path = store.path_for("city", 1, "Memphis", "claude-sonnet-5")

        assert path.parent == workspace / "city"
        assert path.name == f"city_{store.atom_id('city', 'Memphis')}_memphis_claude-sonnet-5.json"

    def test_path_for_ignores_a_counter_id_and_uses_the_content_id(self, workspace):
        """`llm_call` still passes its class counter; honouring it is the bug."""
        first = store.path_for("city", 1, "Memphis", "m")
        second = store.path_for("city", 874, "Memphis", "m")

        assert first == second
        assert store.is_hash_id(first.name.split("_")[1])

    def test_path_for_keeps_a_content_id_it_is_handed(self, workspace):
        given = store.atom_id("city", "Memphis")

        assert store.path_for("city", given, "Memphis", "m").name.split("_")[1] == given

    def test_regenerating_a_topic_overwrites_rather_than_duplicating(self, workspace):
        """The failure the counter caused: run the create pass twice on one topic
        and the corpus grew a second file under a new id."""
        for _ in range(3):
            store.write(store.path_for("city", 1, "Memphis", "m"), "{}")

        assert len(list((workspace / "city").iterdir())) == 1

    def test_append_variants_share_the_base_folder(self, workspace):
        """merge_helper pairs a city with its append file inside one directory."""
        city = store.path_for("city", 1, "Memphis", "m")
        append = store.path_for("append_city", 1, "Memphis", "m")

        assert city.parent == append.parent
        assert append.name.startswith("append_city_")

    def test_an_append_shares_its_base_atoms_id(self, workspace):
        """What lets merge_helper pair the two without comparing model strings."""
        city = store.AtomName.decompose_filename(store.path_for("city", 1, "Memphis", "a").name)
        append = store.AtomName.decompose_filename(
            store.path_for("append_city", 1, "Memphis", "b").name
        )

        assert city.padded_id == append.padded_id


class TestErrorSentinel:
    def test_error_path_appends_the_suffix(self):
        assert store.error_path("/a/b.json").name == "b.json_ERROR"

    def test_error_path_is_idempotent(self):
        once = store.error_path("/a/b.json")
        assert store.error_path(once) == once

    def test_target_of_error_strips_it(self):
        assert store.target_of_error("/a/b.json_ERROR").name == "b.json"

    def test_target_of_error_is_idempotent(self):
        assert store.target_of_error("/a/b.json").name == "b.json"

    def test_is_error_path(self):
        assert store.is_error_path("x.json_ERROR")
        assert not store.is_error_path("x.json")

    def test_write_error_lands_beside_the_atom(self, workspace):
        target = store.path_for("city", 1, "Memphis", "m")
        sentinel = store.write_error(target, "not json at all")

        assert sentinel.name == target.name + "_ERROR"
        assert sentinel.parent == target.parent
        assert sentinel.read_text() == "not json at all"

    def test_write_error_accepts_a_sentinel_path_without_doubling_it(self, workspace):
        target = store.path_for("city", 1, "Memphis", "m")
        sentinel = store.write_error(store.error_path(target), "raw")

        assert not sentinel.name.endswith("_ERROR_ERROR")

    def test_clear_error_reports_whether_one_was_there(self, workspace):
        target = store.path_for("city", 1, "Memphis", "m")

        assert store.clear_error(target) is False
        store.write_error(target, "raw")
        assert store.clear_error(target) is True
        assert not store.error_path(target).exists()

    def test_has_error(self, workspace):
        target = store.path_for("city", 1, "Memphis", "m")
        assert not store.has_error(target)
        store.write_error(target, "raw")
        assert store.has_error(target)


class TestCommitFix:
    def test_it_writes_the_target_then_retires_the_sentinel(self, workspace):
        target = store.path_for("city", 1, "Memphis", "m")
        sentinel = store.write_error(target, "broken")

        returned = store.commit_fix(sentinel, '{"ok": true}')

        assert returned == target
        assert target.read_text() == '{"ok": true}'
        assert not sentinel.exists()

    def test_a_failing_write_leaves_the_sentinel_in_place(self, workspace, monkeypatch):
        """Ordering is the guarantee: no promotion, no retirement."""
        target = store.path_for("city", 1, "Memphis", "m")
        sentinel = store.write_error(target, "broken")

        def boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(store, "write", boom)

        with pytest.raises(OSError):
            store.commit_fix(sentinel, '{"ok": true}')

        assert sentinel.exists(), "the work queue entry must survive a failed promotion"


class TestAtomicWrite:
    def test_write_creates_parent_directories(self, workspace):
        path = workspace / "deep" / "nested" / "x.json"
        store.write(path, "hello")

        assert path.read_text() == "hello"

    def test_write_atom_serialises_the_model(self, workspace):
        path = workspace / "city" / "x.json"
        store.write_atom(path, EmptyModel())

        assert path.exists()
        assert path.read_text().startswith("{")

    def test_a_failed_write_leaves_the_previous_contents(self, workspace, monkeypatch):
        """The verify pass rewrites good files; a partial write would be a loss."""
        path = workspace / "city" / "x.json"
        store.write(path, "original")

        real_replace = os.replace

        def boom(src, dst):
            raise OSError("interrupted")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            store.write(path, "replacement")
        monkeypatch.setattr(os, "replace", real_replace)

        assert path.read_text() == "original"

    def test_a_failed_write_leaves_no_temporary_debris(self, workspace, monkeypatch):
        path = workspace / "city" / "x.json"
        store.write(path, "original")

        def boom(src, dst):
            raise OSError("interrupted")

        monkeypatch.setattr(os, "replace", boom)
        with pytest.raises(OSError):
            store.write(path, "replacement")

        leftovers = [p.name for p in path.parent.iterdir() if p.name.startswith(".tmp_")]
        assert leftovers == []


class TestNonAsciiEncoding:
    """The corpus holds literal UTF-8; `json.dumps` would escape it.

    Found during the B1 repair: writing repaired city files with a
    bare `json.dumps` turned `Saïs` into `Sa\u00efs` in files that the
    repair had not otherwise touched.
    """

    def test_write_json_keeps_non_ascii_literal(self, workspace):
        path = workspace / "city" / "x.json"
        store.write_json(path, {"name": "Sa\u00efs", "who": "Jean-Fran\u00e7ois"})

        text = path.read_text()
        assert "Saïs" in text
        assert "\\u00ef" not in text

    def test_write_json_round_trips(self, workspace):
        path = workspace / "city" / "x.json"
        payload = {"name": "Sa\u00efs", "list": [1, 2], "nested": {"a": None}}
        store.write_json(path, payload)

        assert store.read_json(path) == payload

    def test_write_json_matches_pydantic_output_format(self, workspace):
        """Both writers must agree, or a merged file diffs against its source."""
        from universe_builder.models.subatomic import EmptyModel

        model = EmptyModel()
        by_model = workspace / "a.json"
        by_dict = workspace / "b.json"
        store.write_atom(by_model, model)
        store.write_json(by_dict, model.model_dump())

        assert by_model.read_text() == by_dict.read_text()

    def test_merge_preserves_non_ascii(self, workspace):
        """merge_helper writes a dict, so it is the one that can regress."""
        from universe_builder.operations import merge as merge_helper

        import json

        from tests.test_operations import valid_city

        payload = json.loads(valid_city())
        payload["notableFigures"] = ["Jean-Fran\u00e7ois"]
        base = workspace / "city" / store.AtomName("city", None, "sais", "m").getReadableFileName()
        append = workspace / "city" / store.AtomName("append_city", None, "sais", "m").getReadableFileName()
        store.write_json(base, payload)
        store.write_json(append, {"notableFigures": ["Ren\u00e9"]})

        merge_helper.process_directory(str(workspace), "city")

        merged = base.read_text(encoding="utf-8")
        assert "Jean-François" in merged
        assert "René" in merged
        assert "\\u00e9" not in merged


class TestReads:
    def test_read_json_returns_none_for_unparseable_text(self, workspace):
        """`_ERROR` files routinely hold text that is not JSON at all."""
        path = workspace / "city" / "x.json_ERROR"
        store.write(path, "Sure! Here is the JSON you asked for:")

        assert store.read_json(path) is None

    def test_read_json_parses_valid_payloads(self, workspace):
        path = workspace / "city" / "x.json"
        store.write(path, '{"a": 1}')

        assert store.read_json(path) == {"a": 1}


class TestIterAtoms:
    def _seed(self, workspace):
        store.write(workspace / "city" / "city_0000000001_a_m.json", "{}")
        store.write(workspace / "city" / "city_0000000002_b_m.json", "{}")
        store.write(workspace / "city" / "city_0000000003_c_m.json_ERROR", "raw")
        store.write(workspace / "discovery" / "discovery_0000000001_d_m.json", "{}")

    def test_it_filters_by_atom_type(self, workspace):
        self._seed(workspace)

        names = [p.name for p in store.iter_atoms("city")]

        assert names == ["city_0000000001_a_m.json", "city_0000000002_b_m.json"]

    def test_errors_are_excluded_by_default(self, workspace):
        self._seed(workspace)

        assert not any(store.is_error_path(p) for p in store.iter_atoms("city"))

    def test_errors_come_first_when_included(self, workspace):
        """The fix pass wants the broken ones before the good ones."""
        self._seed(workspace)

        paths = list(store.iter_atoms("city", include_errors=True))

        assert store.is_error_path(paths[0])
        assert not store.is_error_path(paths[1])

    def test_it_walks_every_atom_type_when_none_is_named(self, workspace):
        self._seed(workspace)

        folders = {p.parent.name for p in store.iter_atoms()}

        assert folders == {"city", "discovery"}

    def test_it_skips_machinery_directories(self, workspace):
        """`_batches` holds batch handles, not atoms."""
        self._seed(workspace)
        store.write(workspace / "_batches" / "handle.json", "{}")

        assert all(p.parent.name != "_batches" for p in store.iter_atoms())

    def test_a_missing_data_directory_yields_nothing(self, workspace):
        for child in workspace.iterdir():
            for f in child.rglob("*"):
                f.unlink()
            child.rmdir()
        workspace.rmdir()

        assert list(store.iter_atoms()) == []


def test_store_is_the_only_module_defining_the_error_suffix():
    """Duplicating the literal is how the convention drifts."""
    import universe_builder.providers.batch as batch

    assert batch.ERROR_SUFFIX is store.ERROR_SUFFIX


def test_data_dir_override_is_honoured_at_call_time(workspace, monkeypatch, tmp_path):
    """`path_for` must not cache the directory resolved at import."""
    elsewhere = tmp_path / "other"
    monkeypatch.setenv("UNIVERSE_BUILDER_DATA_DIR", str(elsewhere))

    assert store.path_for("city", 1, "x", "m").parent == Path(elsewhere) / "history" / "city"


class TestSchemaStamping:
    """`write_atom` stamps the schema version; `read_atom` migrates forward.

    Together these are what let an old payload be read by new code without
    guessing its age. Files that predate the stamp carry no version key and read
    as version 0 until a pass rewrites and stamps them — so version 0 is the
    well-handled default, not an error case.
    """

    def test_a_written_atom_carries_the_current_version(self, workspace):
        import json as _json

        from universe_builder import migrations
        from universe_builder.domains.history.models.city import City
        from tests.test_operations import valid_city

        atom = City.model_validate(_json.loads(valid_city()))
        path = store.path_for("city", 1, "Memphis", "m")
        store.write_atom(path, atom)

        assert migrations.version_of(store.read_json(path)) == migrations.current_version("city")

    def test_a_stamped_atom_still_validates_against_its_model(self, workspace):
        """Models default to ignoring unknown fields; this asserts they still do."""
        import json as _json

        from universe_builder.domains.history.models.city import City
        from tests.test_operations import valid_city

        path = store.path_for("city", 1, "Memphis", "m")
        store.write_atom(path, City.model_validate(_json.loads(valid_city())))

        City.model_validate(store.read_json(path))

    def test_a_sentinel_is_never_stamped(self, workspace):
        """A stamped `EmptyModel` stops being recognised as one.

        `EmptyModel` sets `extra = "forbid"` and `is_empty_model` validates with
        `strict=True`, so the extra key makes the check return False. The augment
        pass uses that check to skip empty files, so a stamped sentinel would be
        sent to the model for repair. Verified: stamping flips
        `is_empty_model` from True to False.
        """
        from universe_builder.store import is_empty_model

        path = store.path_for("city", 1, "Memphis", "m")
        store.write_atom(path, EmptyModel())

        assert is_empty_model(path.read_text())
        assert "schema_version" not in store.read_json(path)

    def test_read_atom_migrates_a_version_0_payload(self, workspace):
        """Every file in the real corpus looks like this."""
        from universe_builder import migrations

        path = workspace / "city" / "city_0000000001_memphis_m.json"
        store.write_json(path, {"name": {"common": "Memphis"}, "cityPeriodsOfHistory": []})

        assert migrations.version_of(store.read_json(path)) == 0
        assert migrations.version_of(store.read_atom(path)) == migrations.current_version("city")

    def test_read_atom_does_not_rewrite_the_file(self, workspace):
        """Migrate-on-read, not migrate-on-open: the corpus must not move."""
        path = workspace / "city" / "city_0000000001_memphis_m.json"
        store.write_json(path, {"name": {"common": "Memphis"}, "cityPeriodsOfHistory": []})
        before = path.read_text()

        store.read_atom(path)

        assert path.read_text() == before

    def test_read_json_stays_the_unmigrated_view(self, workspace):
        """The fix pass needs to see exactly what is on disk."""
        from universe_builder import migrations

        path = workspace / "city" / "city_0000000001_memphis_m.json"
        store.write_json(path, {"name": {"common": "Memphis"}, "cityPeriodsOfHistory": []})

        assert migrations.version_of(store.read_json(path)) == 0

    def test_atom_type_of_strips_the_error_suffix(self, workspace):
        assert store.atom_type_of("city_0000000001_memphis_m.json_ERROR") == "city"

    def test_atom_type_of_strips_the_final_prefix(self, workspace):
        """`final_city_….json` would otherwise key on the atom type `final_city`,
        which has no migration chain, silently skipping migration."""
        assert store.atom_type_of("final_city_0000000001_memphis_m.json") == "city"

    def test_append_keeps_its_own_atom_type(self, workspace):
        """`CityAppend` is a different shape and must not get city migrations."""
        assert store.atom_type_of("append_city_0000000001_memphis_m.json") == "append_city"

    def test_read_atom_returns_none_for_unparseable_text(self, workspace):
        path = workspace / "city" / "city_0000000001_memphis_m.json_ERROR"
        store.write(path, "Sure! Here is your JSON:")

        assert store.read_atom(path) is None


class TestAValidatedAtomNeverLandsOnASentinelPath:
    """`write_atom` refuses an `_ERROR` path outright.

    The batch fix pass recorded the sentinel as its write target, so `collect`
    wrote the repaired atom to `X.json_ERROR` and the `store.clear_error` that
    follows resolved back to that same file and deleted it. The repair and the
    original failing payload were both gone, and the report said "1 written".

    The caller was fixed, but the guard lives here because the invariant is not
    the caller's: a validated atom never belongs at a sentinel path, whoever is
    asking. `store.commit_fix` is the supported way to promote a repair, and it
    writes the atom *before* retiring the sentinel.
    """

    def test_it_refuses(self, workspace):
        from universe_builder.domains.history.models.discovery import Discovery

        atom = store.dir_for("discovery") / "discovery_h1111111111111111_x_m.json"
        atom.parent.mkdir(parents=True, exist_ok=True)

        with pytest.raises(ValueError, match="sentinel path"):
            store.write_atom(store.error_path(atom), Discovery.model_construct())

    def test_the_message_says_where_it_should_have_gone(self, workspace):
        from universe_builder.domains.history.models.discovery import Discovery

        atom = store.dir_for("discovery") / "discovery_h1111111111111111_x_m.json"
        atom.parent.mkdir(parents=True, exist_ok=True)

        with pytest.raises(ValueError) as excinfo:
            store.write_atom(store.error_path(atom), Discovery.model_construct())

        assert atom.name in str(excinfo.value)

    def test_the_repair_and_the_evidence_both_survive_the_refusal(self, workspace):
        """Refusing must not be its own kind of loss: the sentinel stays put."""
        from universe_builder.domains.history.models.discovery import Discovery

        atom = store.dir_for("discovery") / "discovery_h1111111111111111_x_m.json"
        atom.parent.mkdir(parents=True, exist_ok=True)
        store.write_error(atom, '{"the original failing payload": true}')
        sentinel = store.error_path(atom)

        with pytest.raises(ValueError):
            store.write_atom(sentinel, Discovery.model_construct())

        assert sentinel.exists()
        assert "original failing payload" in store.read(sentinel)

    def test_an_ordinary_atom_path_still_writes(self, workspace):
        """The guard must not refuse the normal case."""
        from universe_builder.domains.history.models.discovery import Discovery

        atom = store.dir_for("discovery") / "discovery_h1111111111111111_x_m.json"
        atom.parent.mkdir(parents=True, exist_ok=True)

        store.write_atom(atom, Discovery.model_construct())

        assert atom.exists()
