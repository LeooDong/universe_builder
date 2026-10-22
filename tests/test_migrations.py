"""`universe_builder.migrations` — versioning stored payloads and carrying old ones forward.

No network, no API key. The only files touched are the bundled examples (read-only) and
`workspace`, which redirects `UNIVERSE_BUILDER_DATA_DIR` at a tmp_path. Nothing here reads
`data/automated/`; the realistic fixture is built from the bundled
`domains/history/examples/memphis.json` (via the `examples_dir` fixture), which has the
same shape as a stored city and, like every stored city, no version marker.
"""

from __future__ import annotations

import copy
import json

import pytest

from universe_builder import migrations, store
from universe_builder.migrations import (
    FutureSchemaVersionError,
    MigrationChainError,
    MigrationError,
    Registry,
    SCHEMA_VERSION_KEY,
)


# ---------------------------------------------------------------------------
# Helpers: chains built in an isolated registry, so a deliberately broken one
# cannot leak into the process-wide registry the real migrations live in.
# ---------------------------------------------------------------------------

def bump(marker):
    """A migration that records that it ran, without touching anything else."""
    def migrate(payload):
        migrated = dict(payload)
        migrated.setdefault("trace", [])
        migrated["trace"] = list(migrated["trace"]) + [marker]
        return migrated
    migrate.__qualname__ = f"bump_{marker}"
    return migrate


@pytest.fixture
def isolated():
    return Registry(name="test")


@pytest.fixture
def example_city(examples_dir):
    """A payload shaped exactly like a stored city: the bundled `memphis.json`."""
    return json.loads((examples_dir / "memphis.json").read_text(encoding="utf-8"))


class TestVersionDetection:
    def test_a_payload_with_no_version_key_is_version_zero(self):
        assert migrations.version_of({"name": {}}) == 0

    def test_every_bundled_example_is_a_version_zero_payload(self, examples_dir):
        """The stored corpus predates versioning; v0 is the default, not the exception."""
        for path in sorted(examples_dir.glob("*.json")):
            assert migrations.version_of(json.loads(path.read_text(encoding="utf-8"))) == 0

    def test_an_explicit_version_is_read_back(self):
        assert migrations.version_of({SCHEMA_VERSION_KEY: 3}) == 3

    def test_an_explicit_null_version_is_treated_as_version_zero(self):
        assert migrations.version_of({SCHEMA_VERSION_KEY: None}) == 0

    def test_a_quoted_version_is_accepted(self):
        """A model asked to preserve a document occasionally quotes the integer."""
        assert migrations.version_of({SCHEMA_VERSION_KEY: "2"}) == 2

    def test_an_unreadable_version_raises_rather_than_being_guessed_at(self):
        with pytest.raises(MigrationError):
            migrations.version_of({SCHEMA_VERSION_KEY: "second"})

    def test_a_boolean_version_raises(self):
        with pytest.raises(MigrationError):
            migrations.version_of({SCHEMA_VERSION_KEY: True})

    def test_a_negative_version_raises(self):
        with pytest.raises(MigrationError):
            migrations.version_of({SCHEMA_VERSION_KEY: -1})

    def test_a_non_dict_payload_is_version_zero(self):
        assert migrations.version_of(None) == 0


class TestMigrateToCurrent:
    def test_a_version_zero_payload_is_carried_to_the_current_version(self, isolated):
        isolated.register("widget", 0)(bump("a"))
        isolated.register("widget", 1)(bump("b"))

        result = isolated.migrate_to_current("widget", {"name": "x"})

        assert result[SCHEMA_VERSION_KEY] == 2
        assert result["trace"] == ["a", "b"]
        assert result["name"] == "x"

    def test_steps_run_in_version_order_regardless_of_registration_order(self, isolated):
        isolated.register("widget", 1)(bump("second"))
        isolated.register("widget", 0)(bump("first"))

        assert isolated.migrate_to_current("widget", {})["trace"] == ["first", "second"]

    def test_a_payload_is_only_carried_from_the_version_it_declares(self, isolated):
        isolated.register("widget", 0)(bump("a"))
        isolated.register("widget", 1)(bump("b"))

        result = isolated.migrate_to_current("widget", {SCHEMA_VERSION_KEY: 1})

        assert result["trace"] == ["b"]

    def test_migrating_an_already_current_payload_is_a_no_op(self, isolated):
        isolated.register("widget", 0)(bump("a"))
        payload = {"name": "x", SCHEMA_VERSION_KEY: 1}

        result = isolated.migrate_to_current("widget", payload)

        assert result == {"name": "x", SCHEMA_VERSION_KEY: 1}
        assert "trace" not in result

    def test_migrating_is_idempotent(self, isolated):
        isolated.register("widget", 0)(bump("a"))
        isolated.register("widget", 1)(bump("b"))

        once = isolated.migrate_to_current("widget", {"name": "x"})
        twice = isolated.migrate_to_current("widget", copy.deepcopy(once))

        assert twice == once

    def test_a_migration_never_mutates_the_callers_payload(self, isolated):
        """The fix pass keeps the payload it read, to show the model what was on disk."""
        def vandal(payload):
            payload["sources"].append("mutated in place")
            payload["added"] = True
            return payload

        isolated.register("widget", 0)(vandal)
        original = {"sources": ["gpt-3.5-turbo"]}
        before = copy.deepcopy(original)

        migrated = isolated.migrate_to_current("widget", original)

        assert original == before
        assert migrated["sources"] == ["gpt-3.5-turbo", "mutated in place"]

    def test_the_framework_stamps_the_version_even_when_a_migration_forgets(self, isolated):
        """Otherwise the same step is selected again on the next pass, forever."""
        isolated.register("widget", 0)(lambda payload: dict(payload))

        assert isolated.migrate_to_current("widget", {})[SCHEMA_VERSION_KEY] == 1

    def test_a_migration_that_returns_nothing_raises_naming_itself(self, isolated):
        def forgets_to_return(payload):
            payload["x"] = 1

        isolated.register("widget", 0)(forgets_to_return)

        with pytest.raises(MigrationError, match=r"widget v0->v1 returned NoneType"):
            isolated.migrate_to_current("widget", {})

    def test_a_payload_from_a_newer_build_raises_rather_than_being_read(self, isolated):
        isolated.register("widget", 0)(bump("a"))

        with pytest.raises(FutureSchemaVersionError, match="knows only v1"):
            isolated.migrate_to_current("widget", {SCHEMA_VERSION_KEY: 4})

    def test_a_non_dict_payload_passes_straight_through(self, isolated):
        """`store.read_json` returns None for an `_ERROR` file's raw text."""
        isolated.register("widget", 0)(bump("a"))

        assert isolated.migrate_to_current("widget", None) is None
        assert isolated.migrate_to_current("widget", "not json at all") == "not json at all"

    def test_stamp_marks_a_new_payload_as_current(self, isolated):
        isolated.register("widget", 0)(bump("a"))
        isolated.register("widget", 1)(bump("b"))
        payload = {"name": "x"}

        stamped = isolated.stamp("widget", payload)

        assert stamped[SCHEMA_VERSION_KEY] == 2
        assert SCHEMA_VERSION_KEY not in payload


class TestChainValidation:
    def test_a_gap_in_the_chain_is_reported_with_the_missing_version(self, isolated):
        isolated.register("widget", 0)(bump("a"))
        isolated.register("widget", 2)(bump("c"))

        with pytest.raises(MigrationChainError, match=r"no migration from v1"):
            isolated.chain("widget")

    def test_a_gap_is_detected_on_the_read_path_not_silently_skipped(self, isolated):
        isolated.register("widget", 0)(bump("a"))
        isolated.register("widget", 2)(bump("c"))

        with pytest.raises(MigrationChainError):
            isolated.migrate_to_current("widget", {})

    def test_a_chain_that_does_not_start_at_zero_is_rejected(self, isolated):
        isolated.register("widget", 1)(bump("b"))

        with pytest.raises(MigrationChainError, match=r"starts at v1, not v0"):
            isolated.chain("widget")

    def test_a_duplicate_from_version_is_rejected_at_registration(self, isolated):
        isolated.register("widget", 0)(bump("a"))

        with pytest.raises(MigrationChainError, match=r"two migrations claim widget v0"):
            isolated.register("widget", 0)(bump("also a"))

    def test_a_migration_spanning_two_versions_is_rejected_at_registration(self, isolated):
        with pytest.raises(MigrationChainError, match="more than one version"):
            isolated.register("widget", 0, to_version=2)(bump("a"))

    def test_validate_checks_every_registered_atom_type(self, isolated):
        isolated.register("sound", 0)(bump("ok"))
        isolated.register("widget", 0)(bump("a"))
        isolated.register("widget", 2)(bump("c"))

        with pytest.raises(MigrationChainError, match="'widget'"):
            isolated.validate()

    def test_registering_a_missing_link_makes_a_previously_valid_chain_valid_again(self, isolated):
        isolated.register("widget", 0)(bump("a"))
        isolated.register("widget", 2)(bump("c"))
        with pytest.raises(MigrationChainError):
            isolated.chain("widget")

        isolated.register("widget", 1)(bump("b"))

        assert [m.from_version for m in isolated.chain("widget")] == [0, 1, 2]

    def test_the_shipped_chains_are_unbroken(self):
        migrations.validate_chains()


class TestUnknownAtomType:
    def test_an_atom_type_with_no_migrations_is_at_the_base_version(self, isolated):
        assert isolated.current_version("no-such-atom") == 0
        assert isolated.chain("no-such-atom") == ()

    def test_an_unversioned_payload_for_an_unknown_atom_type_is_returned_unchanged(self, isolated):
        payload = {"name": "x"}

        assert isolated.migrate_to_current("no-such-atom", payload) == {"name": "x"}

    def test_a_versioned_payload_for_an_unknown_atom_type_raises_rather_than_validating(self, isolated):
        """A file claiming a version we have never heard of must not be read as if current."""
        with pytest.raises(FutureSchemaVersionError, match="no-such-atom"):
            isolated.migrate_to_current("no-such-atom", {SCHEMA_VERSION_KEY: 1})

    def test_atom_types_with_no_registered_migrations_are_still_readable(self):
        """`person`, `event` and the rest have no migrations yet and must pass through."""
        for atom_type in ("person", "event", "discovery", "artifact", "history", "append_city"):
            assert migrations.migrate_to_current(atom_type, {"name": "x"}) == {"name": "x"}


class TestCityVersionOne:
    def test_city_is_registered_with_a_single_step_chain(self):
        assert migrations.current_version("city") == 1
        assert [(m.from_version, m.to_version) for m in migrations.chain("city")] == [(0, 1)]

    def test_a_stored_city_payload_carries_no_version_and_is_therefore_v0(self, example_city):
        assert SCHEMA_VERSION_KEY not in example_city
        assert migrations.version_of(example_city) == 0

    def test_migrating_a_stored_city_adds_the_marker_and_changes_nothing_else(self, example_city):
        before = copy.deepcopy(example_city)

        migrated = migrations.migrate_to_current("city", example_city)

        assert migrated[SCHEMA_VERSION_KEY] == 1
        assert {k: v for k, v in migrated.items() if k != SCHEMA_VERSION_KEY} == before
        assert example_city == before

    def test_migrating_a_stored_city_twice_is_idempotent(self, example_city):
        once = migrations.migrate_to_current("city", example_city)
        twice = migrations.migrate_to_current("city", once)

        assert twice == once

    def test_a_migrated_city_still_validates_as_the_current_model(self, example_city):
        """Pydantic ignores the extra key, so migrate-on-read does not break validation."""
        from universe_builder.domains.history.models.city import City

        migrated = migrations.migrate_to_current("city", example_city)

        assert City(**migrated).name.common == example_city["name"]["common"]

    def test_migration_does_not_disturb_how_a_stored_class_is_resolved(self, example_city, population_payload):
        """`resolve_stored_class` sniffs `cityPeriodsOfHistory`; the marker must not confuse it."""
        from universe_builder.execute_prompt import resolve_stored_class
        from universe_builder.domains.history.models.city import City, CityExpanded

        expanded = copy.deepcopy(example_city)
        expanded["cityPeriodsOfHistory"] = [population_payload()]

        assert resolve_stored_class("city", migrations.migrate_to_current("city", example_city)) is City
        assert resolve_stored_class("city", migrations.migrate_to_current("city", expanded)) is CityExpanded

    def test_a_city_file_written_before_versioning_reads_and_migrates(self, workspace, example_city):
        """The end-to-end shape of migrate-on-read, over a file with no marker on disk."""
        path = store.path_for("city", 1, "Memphis", "claude-sonnet-5")
        store.write_json(path, example_city)

        on_disk = store.read_json(path)
        migrated = migrations.migrate_to_current("city", on_disk)

        assert SCHEMA_VERSION_KEY not in json.loads(store.read(path))
        assert migrated[SCHEMA_VERSION_KEY] == 1
        assert migrated["name"] == example_city["name"]

    def test_the_step_function_is_importable_and_pure(self, example_city):
        """A migration is a plain function on dicts, callable without the registry."""
        from universe_builder.domains.history.migrations.city_0001_add_schema_version import add_schema_version

        result = add_schema_version(example_city)

        assert result[SCHEMA_VERSION_KEY] == 1
        assert SCHEMA_VERSION_KEY not in example_city
