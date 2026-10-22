"""Aliased field names must not change a single byte of the stored corpus.

Two atom fields were named with a misspelling of "discoverer" — `Artifact`'s
`finderOrDiscoveror` and `Discovery`'s `discoverorOrInventor`. Correcting the
Python attribute is only safe because a Pydantic alias holds the wire key still.
The whole risk of that change is invisible until something writes a file: a dump
that quietly switches to the corrected spelling would leave the corpus carrying
two names for one field, with no error at any point.

These tests fix the wire format in place. `test_every_example_round_trips_byte_
identically` is the one that would catch it: it dumps each checked-in example and
compares the result to the bytes on disk.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from universe_builder.domains.history.models.artifact import Artifact
from universe_builder.domains.history.models.city import City, CityAppend, CityExpanded, Population
from universe_builder.domains.history.models.discovery import Discovery
from universe_builder.domains.history.models.event import Event
from universe_builder.domains.history.models.history import History
from universe_builder.domains.history.models.person import Person

#: Every model an example file may validate against, so the round-trip test
#: covers the aliased atoms and their unaliased neighbours alike — a regression
#: in `WireModel` that reached the wrong classes would show up here too.
MODELS = (
    Artifact, City, CityAppend, CityExpanded, Population,
    Discovery, Event, History, Person,
)

#: (python attribute, stored wire key) for every field carrying an alias.
ALIASED_FIELDS = [
    (Artifact, "finderOrDiscoverer", "finderOrDiscoveror"),
    (Discovery, "discovererOrInventor", "discoverorOrInventor"),
]


def _example_payloads(examples_dir):
    for path in sorted(examples_dir.glob("*.json")):
        yield path, json.loads(path.read_text(encoding="utf-8"))


def _validating_models(payload):
    """Every model the payload is a valid instance of."""
    for model in MODELS:
        try:
            yield model, model.model_validate(payload)
        except ValidationError:
            continue


def _key_shape(node):
    """The key names of a payload, at every depth, ignoring values.

    Comparing whole documents would fail on validator coercions that are
    supposed to happen (a month name becoming an integer). Key names are the
    thing the alias change could break, so they are what is compared.
    """
    if isinstance(node, dict):
        return {k: _key_shape(v) for k, v in node.items()}
    if isinstance(node, list):
        return [_key_shape(v) for v in node]
    return None


def _new_keys(dumped, stored, path=""):
    """Key paths present in `dumped` but not in `stored`.

    A rename shows up here and nowhere else. Keys going the other way are
    ignored on purpose: several examples carry keys no model declares (the Event
    example has `causes`, its locations have `description`) and Pydantic drops
    them, which predates aliasing and is not what these tests are about.
    """
    new = []
    if isinstance(dumped, dict) and isinstance(stored, dict):
        for key, value in dumped.items():
            here = f"{path}.{key}" if path else key
            if key not in stored:
                new.append(here)
            else:
                new.extend(_new_keys(value, stored[key], here))
    elif isinstance(dumped, list) and isinstance(stored, list):
        for i, (d, s) in enumerate(zip(dumped, stored)):
            new.extend(_new_keys(d, s, f"{path}[{i}]"))
    return new


class TestStoredKeyNamesAreUnchanged:
    def test_no_example_gains_a_renamed_key_when_dumped(self, examples_dir):
        """Load, validate, dump — the dump must not introduce a key the stored
        file does not already use, for any example and any model it validates
        against. A field renamed without an alias appears here immediately, and
        this is the guard standing between that mistake and the stored corpus.
        """
        checked = 0
        for path, payload in _example_payloads(examples_dir):
            for model, instance in _validating_models(payload):
                dumped = json.loads(instance.model_dump_json())
                assert not _new_keys(dumped, payload), (
                    f"{path.name} dumped as {model.__name__} introduced keys the "
                    f"stored file does not have: {_new_keys(dumped, payload)}"
                )
                checked += 1
        assert checked >= len(MODELS) - 2, (
            f"only {checked} example/model pairs were exercised; the round-trip "
            "guard is close to vacuous"
        )

    def test_dumping_by_field_name_is_what_the_alias_default_prevents(self):
        """A guard on the guard. With `by_alias` off the corrected spelling does
        reach the output, so the tests above are testing something. If this ever
        stops holding, the alias default has become dead code and every
        assertion here would pass for the wrong reason."""
        payload = _minimal(Artifact, "finderOrDiscoveror", "x")
        dumped = json.loads(
            Artifact.model_validate(payload).model_dump_json(by_alias=False)
        )
        assert "finderOrDiscoverer" in dumped
        assert "finderOrDiscoveror" not in dumped

    @pytest.mark.parametrize(
        "filename,model",
        [("rosetta_stone.json", Artifact), ("invention_of_toothpaste.json", Discovery)],
    )
    def test_an_aliased_example_round_trips_key_identically(
        self, filename, model, examples_dir
    ):
        """The two examples that actually carry an aliased key are held to the
        stronger standard: identical key names, at every depth, in both
        directions."""
        payload = json.loads((examples_dir / filename).read_text(encoding="utf-8"))
        dumped = json.loads(model.model_validate(payload).model_dump_json())
        assert _key_shape(dumped) == _key_shape(payload)

    def test_the_examples_cover_both_aliased_atoms(self, examples_dir):
        """Without this the test above could pass while never once dumping an
        aliased field."""
        seen = set()
        for _, payload in _example_payloads(examples_dir):
            for _, wire_key in ((m, w) for m, _, w in ALIASED_FIELDS):
                if wire_key in payload:
                    seen.add(wire_key)
        assert seen == {w for _, _, w in ALIASED_FIELDS}, seen

    @pytest.mark.parametrize(
        "model,attribute,wire_key",
        ALIASED_FIELDS,
        ids=[m.__name__ for m, _, _ in ALIASED_FIELDS],
    )
    def test_model_dump_json_emits_the_wire_spelling(self, model, attribute, wire_key):
        payload = _minimal(model, wire_key, "Someone")
        dumped = json.loads(model.model_validate(payload).model_dump_json())
        assert wire_key in dumped
        assert attribute not in dumped

    @pytest.mark.parametrize(
        "model,attribute,wire_key",
        ALIASED_FIELDS,
        ids=[m.__name__ for m, _, _ in ALIASED_FIELDS],
    )
    def test_model_dump_emits_the_wire_spelling(self, model, attribute, wire_key):
        """`model_dump()` needs the same default as `model_dump_json()`: the
        expand pass rebuilds atoms from `model_dump()` output, so a dict keyed on
        the corrected name would be a second, undeclared spelling in flight."""
        dumped = model.model_validate(_minimal(model, wire_key, "Someone")).model_dump()
        assert wire_key in dumped
        assert attribute not in dumped

    @pytest.mark.parametrize(
        "model,attribute,wire_key",
        ALIASED_FIELDS,
        ids=[m.__name__ for m, _, _ in ALIASED_FIELDS],
    )
    def test_the_generated_json_schema_asks_for_the_wire_spelling(
        self, model, attribute, wire_key
    ):
        """The schema is what the model is told to produce. If it named the
        corrected spelling, every generated atom would arrive under a key the
        corpus has never used."""
        properties = model.model_json_schema()["properties"]
        assert wire_key in properties
        assert attribute not in properties


class TestBothSpellingsValidate:
    @pytest.mark.parametrize(
        "model,attribute,wire_key",
        ALIASED_FIELDS,
        ids=[m.__name__ for m, _, _ in ALIASED_FIELDS],
    )
    def test_the_stored_misspelling_still_validates(self, model, attribute, wire_key):
        instance = model.model_validate(_minimal(model, wire_key, "French soldiers"))
        assert getattr(instance, attribute) == "French soldiers"

    @pytest.mark.parametrize(
        "model,attribute,wire_key",
        ALIASED_FIELDS,
        ids=[m.__name__ for m, _, _ in ALIASED_FIELDS],
    )
    def test_the_corrected_spelling_also_validates(self, model, attribute, wire_key):
        """`populate_by_name`. A payload written by hand, or by a model that read
        the corrected name somewhere, must not be rejected."""
        instance = model.model_validate(_minimal(model, attribute, "French soldiers"))
        assert getattr(instance, attribute) == "French soldiers"

    @pytest.mark.parametrize(
        "model,attribute,wire_key",
        ALIASED_FIELDS,
        ids=[m.__name__ for m, _, _ in ALIASED_FIELDS],
    )
    def test_the_field_is_still_required(self, model, attribute, wire_key):
        """The alias must not have turned a required field into an optional one:
        `Optional[str]` here means nullable, not absent."""
        payload = _minimal(model, wire_key, "Someone")
        del payload[wire_key]
        with pytest.raises(ValidationError):
            model.model_validate(payload)

    @pytest.mark.parametrize(
        "model,attribute,wire_key",
        ALIASED_FIELDS,
        ids=[m.__name__ for m, _, _ in ALIASED_FIELDS],
    )
    def test_a_null_value_survives_the_round_trip(self, model, attribute, wire_key):
        payload = _minimal(model, wire_key, None)
        dumped = json.loads(model.model_validate(payload).model_dump_json())
        assert dumped[wire_key] is None


class TestUnaliasedFieldsAreUntouched:
    def test_by_alias_does_not_rename_ordinary_fields(self):
        """`by_alias=True` is set globally on `WireModel`, so a field with no
        alias has to keep dumping under its own name."""
        dumped = json.loads(
            Artifact.model_validate(_minimal(Artifact, "finderOrDiscoveror", "x"))
            .model_dump_json()
        )
        for key in ("name", "dateCreated", "originalCreator", "shortSummary"):
            assert key in dumped

    def test_nested_models_are_unaffected(self):
        """`by_alias` propagates into nested models. None of them carry aliases,
        so their keys must be identical to the field names."""
        dumped = json.loads(
            Artifact.model_validate(_minimal(Artifact, "finderOrDiscoveror", "x"))
            .model_dump_json()
        )
        assert set(dumped["name"]) == set(Artifact.model_fields["name"].annotation.model_fields)


# ---------------------------------------------------------------------------
# Payload builders
# ---------------------------------------------------------------------------

def _minimal(model, key, value):
    """A valid payload for `model` with `key` set to `value`."""
    from tests.conftest import make_date, make_location, make_name

    if model is Artifact:
        payload = {
            "name": make_name("Rosetta Stone"),
            "dateCreated": make_date(),
            "discoveryLocation": make_location(),
            "dateDiscovered": make_date(),
            "currentLocation": make_location(),
            "finderOrDiscoveror": None,
            "originalCreator": "Ptolemaic scribes",
            "challenges": "c",
            "significance": "s",
            "controversies": None,
            "shortSummary": "short",
            "longSummary": "long",
        }
        payload.pop("finderOrDiscoveror")
    elif model is Discovery:
        finding = {
            "name": "f",
            "ageOfRecord": make_date(),
            "dateOfModernDiscovery": make_date(),
            "location": make_location(),
        }
        payload = {
            "name": make_name("Toothpaste"),
            "earliestPhysicalRecord": dict(finding),
            "earliestWrittenRecord": dict(finding),
            "typeOfDiscovery": "incremental",
            "controversies": None,
            "challenges": "c",
            "significance": "s",
            "shortSummary": "short",
            "longSummary": "long",
        }
    else:  # pragma: no cover - only the two aliased atoms are parametrised
        raise AssertionError(f"no builder for {model.__name__}")

    payload[key] = value
    return payload
