"""Model and validator tests, including one regression per bug B1-B6.

The bug numbers are the engine's own labels for its early model defects. Each regression
test is named after its bug so a failure identifies the defect directly.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

import json

from universe_builder.domains.history.models.city import (
    City, CityAppend, CityDate, CityExpanded, Population, government_enum,
)
from universe_builder.domains.history.models.history import History
from universe_builder.models.subatomic import MONTH_MAPPING, Date, EmptyModel, Judgments, Name

from tests.conftest import make_date, make_population


# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------

class TestDate:
    def test_negative_year_is_made_positive(self):
        # Era carries BCE/CE; the year itself is stored unsigned.
        assert Date(**make_date(year=-330)).year == 330

    def test_none_year_stays_none(self):
        assert Date(**make_date(year=None)).year is None

    def test_a_negative_year_with_no_era_is_bce(self):
        """The sign was the only thing saying BCE, and `abs_year` threw it away:
        -500 with a null era was stored as 500 with a null era, which every
        reader takes as CE or as undatable. A thousand years lost to a sign."""
        date = Date(**make_date(year=-500, era=None))
        assert (date.year, date.era) == (500, "BCE")

    def test_a_negative_numeric_string_year_is_bce_too(self):
        date = Date(**make_date(year="-44", era=None))
        assert (date.year, date.era) == (44, "BCE")

    def test_a_negative_year_with_an_explicit_era_keeps_that_era(self, caplog):
        """-500 CE is a contradiction, and which half is wrong is not knowable
        here; the stated era stands, as before, and the conflict is logged
        rather than resolved by a guess."""
        import logging

        with caplog.at_level(logging.WARNING, logger="universe_builder.models.subatomic"):
            date = Date(**make_date(year=-500, era="CE"))
        assert (date.year, date.era) == (500, "CE")
        assert "contradict" in caplog.text
        assert Date(**make_date(year=-500, era="BCE")).era == "BCE"

    def test_a_positive_year_with_no_era_stays_without_one(self):
        assert Date(**make_date(year=500, era=None)).era is None

    @pytest.mark.parametrize("name,expected", sorted(MONTH_MAPPING.items()))
    def test_every_month_name_maps(self, name, expected):
        assert Date(**make_date(month=name)).month == expected

    def test_month_name_is_case_insensitive(self):
        assert Date(**make_date(month="MaRcH")).month == 3

    def test_unknown_month_name_becomes_none(self):
        assert Date(**make_date(month="Smarch")).month is None

    def test_null_string_month_becomes_none(self):
        assert Date(**make_date(month="null")).month is None

    @pytest.mark.parametrize("value", [0, 13, 99])
    def test_out_of_range_month_becomes_none(self, value):
        assert Date(**make_date(month=value)).month is None

    def test_valid_numeric_month_survives(self):
        assert Date(**make_date(month=7)).month == 7

    @pytest.mark.parametrize("value", [0, 32, "null"])
    def test_out_of_range_or_null_day_becomes_none(self, value):
        assert Date(**make_date(day=value)).day is None

    def test_valid_day_survives(self):
        assert Date(**make_date(day=15)).day == 15

    def test_year_upper_bound_is_enforced(self):
        with pytest.raises(ValidationError):
            Date(**make_date(year=60001))


# ---------------------------------------------------------------------------
# Judgments
# ---------------------------------------------------------------------------

class TestJudgments:
    def test_scores_in_range_validate(self, ):
        from tests.conftest import make_judgments

        assert Judgments(**make_judgments()).politicalInfluenceScore == 5

    @pytest.mark.parametrize("bad", [-1, 11])
    def test_scores_out_of_range_are_rejected(self, bad):
        from tests.conftest import make_judgments

        payload = make_judgments()
        payload["politicalInfluenceScore"] = bad
        with pytest.raises(ValidationError):
            Judgments(**payload)

    def test_scores_are_strict_and_reject_float_and_string(self):
        from tests.conftest import make_judgments

        for bad in (5.0, "5"):
            payload = make_judgments()
            payload["culturalContributionScore"] = bad
            with pytest.raises(ValidationError):
                Judgments(**payload)


# ---------------------------------------------------------------------------
# Name.otherNames
# ---------------------------------------------------------------------------

class TestOtherNamesAreTidied:
    """Two faults a schema description demonstrably cannot stop.

    With the rule written into the field description, `gpt-5-nano` still put
    'Abydos' inside Abydos's own `otherNames`, and 'Maatkare (throne name)' in
    Hatshepsut's, run after run.
    Both are mechanical, so both are decided in the validator instead.

    They matter because `topics.stored_name_claims` reads this field: an entry
    here is a claim the subject resolver may act on.
    """

    def _name(self, common, others):
        return Name(common=common, otherNames=others, sources=[],
                    meaningOfNames=[], longFormInfo=[])

    def test_the_common_name_is_not_repeated_back(self):
        assert self._name("Abydos", ["Abdju", "Abydos", "Abedju"]).otherNames \
            == ["Abdju", "Abedju"]

    def test_the_repeat_check_ignores_case_and_padding(self):
        assert self._name("Abydos", ["  abydos  ", "Abdju"]).otherNames == ["Abdju"]

    @pytest.mark.parametrize("entry, expected", [
        ("Maatkare (throne name)", "Maatkare"),
        ("Abidos (Greek)", "Abidos"),
        ("Men-nefer (ancient Egyptian name)", "Men-nefer"),
    ])
    def test_a_descriptive_parenthetical_is_stripped(self, entry, expected):
        assert self._name("X", [entry]).otherNames == [expected]

    @pytest.mark.parametrize("entry", [
        "Thebes (Luxor)",          # one place, two names — pinned for History too
        "Sardis (Sart)",
        "Nicaea (İznik)",
    ])
    def test_a_parenthetical_that_is_a_name_survives(self, entry):
        """The asymmetry is deliberate: an unrecognised aside keeps its data.

        A wrong strip deletes a name for good; a missed strip leaves a tidy-up
        for `universe_builder lint`. Only the first is unrecoverable.
        """
        assert self._name("Somewhere", [entry]).otherNames == [entry]

    def test_stripping_can_collapse_two_entries_into_one(self):
        """Observed shape: the glossed and bare spellings side by side."""
        name = self._name("Hatshepsut", ["Maatkare (throne name)", "Maatkare"])

        assert name.otherNames == ["Maatkare"]

    def test_an_entry_that_is_only_a_descriptor_is_kept_not_emptied(self):
        """Stripping to nothing would invent a deletion the model did not ask
        for, so the original stands and a human decides."""
        assert self._name("X", ["(throne name)"]).otherNames == ["(throne name)"]

    def test_nulls_survive(self):
        """The field is `List[Optional[str]]`; tidying must not retype it."""
        assert self._name("X", [None, "Y"]).otherNames == [None, "Y"]

    def test_a_legitimate_list_is_untouched(self):
        """Memphis's bundled example, which must round-trip unchanged."""
        others = ["Men-nefer", "Mennefer", "Inbu-hedj", "Hut-ka-Ptah",
                  "Hikuptah", "Mit Rahina"]

        assert self._name("Memphis", list(others)).otherNames == others

    def test_the_instruction_still_reaches_the_model(self):
        """The validator is the enforcement; the description is not therefore
        decoration. It rides `response_format`, and `person` creates run on a
        model large enough for it to bite."""
        from universe_builder.providers.base import Provider

        schema = Provider.strict_json_schema(Name)
        description = schema["properties"]["otherNames"]["description"]

        assert "transliteration variants" in description
        assert "Do not repeat the common name" in description


# ---------------------------------------------------------------------------
# EmptyModel — the sentinel written when an augment produces nothing usable
# ---------------------------------------------------------------------------

class TestEmptyModel:
    def test_default_message(self):
        assert EmptyModel().error == "No data found"

    def test_extra_fields_are_forbidden(self):
        # This is what makes `is_empty_model` a reliable discriminator: a real
        # atom payload must never validate as EmptyModel.
        with pytest.raises(ValidationError):
            EmptyModel(error="x", people=[])

    def test_a_real_atom_payload_is_not_an_empty_model(self, history_payload):
        import json

        from universe_builder.store import is_empty_model

        assert not is_empty_model(json.dumps(history_payload))

    def test_the_sentinel_is_recognised(self):
        from universe_builder.store import get_empty_model, is_empty_model

        assert is_empty_model(get_empty_model().model_dump_json())


# ---------------------------------------------------------------------------
# Bug regressions. One test per bug, named for the bug.
# ---------------------------------------------------------------------------

class TestBugRegressions:
    def test_B1_type_of_government_round_trips(self):
        """B1: a stacked second validator turned the string into a char list,
        which the enum coercion then rewrote to "other". Every legitimate
        government type was destroyed."""
        for value in government_enum:
            assert Population(**make_population(typeOfGovernment=value)).typeOfGovernment == value

    def test_B1_unknown_government_still_coerces_to_other(self):
        """The surviving validator must keep doing its job."""
        assert Population(**make_population(typeOfGovernment="space empire")).typeOfGovernment == "other"

    def test_B1_none_government_coerces_to_other(self):
        assert Population(**make_population(typeOfGovernment=None)).typeOfGovernment == "other"

    def test_B1_only_one_validator_on_type_of_government(self):
        """Guards against the second validator being reintroduced: two
        `mode="before"` validators on one field is the shape of the bug."""
        decorators = Population.__pydantic_decorators__.field_validators
        on_field = [
            d for d in decorators.values()
            if "typeOfGovernment" in d.info.fields and d.info.mode == "before"
        ]
        assert len(on_field) == 1, f"expected 1 before-validator, found {len(on_field)}"

    def test_B2_month_name_returns_an_integer(self):
        """B2: `cap_month` computed the mapping but never returned it, so the
        raw string fell through and failed int validation."""
        assert Date(**make_date(month="March")).month == 3

    def test_B2_month_name_does_not_raise(self):
        # The observed symptom was a ValidationError, not a wrong value.
        Date(**make_date(month="December"))

    def test_B6_dead_classmethod_is_gone(self):
        """B6: an undecorated `replace_null_with_list` on CityDate looked like
        a validator and was not one."""
        assert not hasattr(CityDate, "replace_null_with_list")

    def test_B6_city_date_still_validates(self):
        from tests.conftest import make_city_date

        assert CityDate(**make_city_date()).averageCityPopulation == 3000


# ---------------------------------------------------------------------------
# Structural expectations the refactor must preserve
# ---------------------------------------------------------------------------

class TestCity:
    def test_city_append_is_a_strict_subset_of_city(self):
        """CityAppend duplicates a City field verbatim. If they ever diverge the
        merge produces a field City cannot hold. A derived model would remove the
        duplication."""
        for field, info in CityAppend.model_fields.items():
            assert field in City.model_fields, f"{field} is not a City field"
            assert info.annotation == City.model_fields[field].annotation


class TestHistory:
    def test_all_topic_fields_are_string_lists(self):
        from typing import List

        for field, info in History.model_fields.items():
            assert info.annotation == List[str], f"{field} is not List[str]"

    def test_history_validates_the_shared_payload(self, history_payload):
        assert History(**history_payload).cities == ["Memphis"]


def _make_city():
    from tests.conftest import (
        make_date, make_judgments, make_location, make_name,
    )

    climate = {
        "monthlyAverageTempCelsius": 20.0,
        "monthlyAveragePrecipMillimetres": 5.0,
        "monthlyAverageHumidityPercent": 50.0,
    }
    months = [
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ]
    city = City(
        name=make_name("Memphis"),
        foundingDate=make_date(),
        dateOfDemise=make_date(),
        location=make_location(),
        cityPeriodsOfHistory=["Old Kingdom"],
        geographyAndClimate={"elevationMetres": 19.0, **{m: climate for m in months}},
        judgments=make_judgments(),
        architecturalHighlights=[],
        notableFigures=[],
        popCulturalReferences=[],
    )
    return city


class TestB7ExpandedCity:
    """B7: the expand pass wrote `Population` objects into `City`, whose
    `cityPeriodsOfHistory` is `List[str]`. The file serialised with a warning and
    could never be read back — it silently affected every generated city file.
    `CityExpanded` is the model for the post-expand shape."""

    def test_a_plain_city_still_rejects_population_objects(self):
        """The seed model must stay strict. If `City` ever accepts a Population
        here, the bug can come back unnoticed."""
        city = _make_city()
        payload = json.loads(city.model_dump_json())
        payload["cityPeriodsOfHistory"] = [make_population()]
        with pytest.raises(ValidationError):
            City.model_validate(payload)

    def test_expanded_city_round_trips(self):
        city = _make_city()
        payload = json.loads(city.model_dump_json())
        payload["cityPeriodsOfHistory"] = [make_population(), make_population(periodName="New Kingdom")]

        expanded = CityExpanded.model_validate(payload)
        reloaded = CityExpanded.model_validate_json(expanded.model_dump_json())

        assert [p.periodName for p in reloaded.cityPeriodsOfHistory] == ["Ptolemaic", "New Kingdom"]
        assert reloaded.is_fully_expanded()

    def test_expanded_city_accepts_an_unexpanded_seed(self):
        """Expansion is per-entry and best-effort, so a seed file — or a
        partially expanded one — must still load."""
        city = _make_city()
        expanded = CityExpanded.model_validate(json.loads(city.model_dump_json()))
        assert not expanded.is_fully_expanded()
        assert expanded.unexpanded_periods() == ["Old Kingdom"]

    def test_a_partially_expanded_city_reports_what_is_missing(self):
        city = _make_city()
        payload = json.loads(city.model_dump_json())
        payload["cityPeriodsOfHistory"] = [make_population(), "Middle Kingdom"]

        expanded = CityExpanded.model_validate(payload)

        assert not expanded.is_fully_expanded()
        assert expanded.unexpanded_periods() == ["Middle Kingdom"]

    def test_expansion_survives_a_json_round_trip_when_partial(self):
        city = _make_city()
        payload = json.loads(city.model_dump_json())
        payload["cityPeriodsOfHistory"] = [make_population(), "Middle Kingdom"]

        reloaded = CityExpanded.model_validate_json(
            CityExpanded.model_validate(payload).model_dump_json()
        )

        assert isinstance(reloaded.cityPeriodsOfHistory[0], Population)
        assert reloaded.cityPeriodsOfHistory[1] == "Middle Kingdom"

    def test_population_wins_over_str_in_the_union(self):
        """Union order matters: if `str` were tried first, every Population dict
        would fail to coerce and the model would be useless."""
        city = _make_city()
        payload = json.loads(city.model_dump_json())
        payload["cityPeriodsOfHistory"] = [make_population()]

        entry = CityExpanded.model_validate(payload).cityPeriodsOfHistory[0]
        assert isinstance(entry, Population)

    def test_expanded_city_keeps_every_other_city_field(self):
        city = _make_city()
        expanded = CityExpanded.model_validate(json.loads(city.model_dump_json()))
        for field in City.model_fields:
            if field != "cityPeriodsOfHistory":
                assert hasattr(expanded, field)


# ---------------------------------------------------------------------------
# Wire aliases. The behaviour lives in tests/test_aliases.py; what is guarded
# here is the trap a future model edit can fall into.
# ---------------------------------------------------------------------------

class TestAliasesOnlyAppearOnWireModel:
    """An alias on a plain `BaseModel` corrupts the corpus quietly.

    `Field(alias=...)` changes what *validates* immediately, but `model_dump_json`
    keeps writing the field name until something sets `by_alias`. `WireModel` is
    the only place that default is flipped, so an aliased field on a model that
    does not inherit from it would read old files happily and then rewrite every
    one of them under a new key, with no error at any point.
    """

    def _atom_models(self):
        from universe_builder.providers.registry import _schema_models

        return _schema_models()

    def _nested_models(self, model, seen=None):
        """`model` and every Pydantic model reachable from its fields."""
        from pydantic import BaseModel as _BaseModel

        seen = seen if seen is not None else set()
        if model in seen:
            return seen
        seen.add(model)
        for info in model.model_fields.values():
            for arg in (info.annotation, *getattr(info.annotation, "__args__", ())):
                if isinstance(arg, type) and issubclass(arg, _BaseModel):
                    self._nested_models(arg, seen)
        return seen

    def test_no_model_carries_an_alias_without_inheriting_wire_model(self):
        from universe_builder.models.subatomic import WireModel

        reachable = set()
        for model in self._atom_models().values():
            reachable |= self._nested_models(model)

        offenders = []
        for model in reachable:
            aliased = [n for n, f in model.model_fields.items() if f.alias]
            if aliased and not issubclass(model, WireModel):
                offenders.append(f"{model.__name__}: {aliased}")
        assert not offenders, (
            "these models alias a field but do not inherit WireModel, so they "
            f"would rewrite the stored key: {offenders}"
        )

    def test_the_two_known_misspellings_are_aliased(self):
        """Regression on the specific keys the corpus was written with."""
        from universe_builder.domains.history.models.artifact import Artifact
        from universe_builder.domains.history.models.discovery import Discovery

        assert Artifact.model_fields["finderOrDiscoverer"].alias == "finderOrDiscoveror"
        assert Discovery.model_fields["discovererOrInventor"].alias == "discoverorOrInventor"


class TestHistoryEntriesAreBareNames:
    """Every History entry becomes a content-addressed id.

    That makes the exact string load-bearing here in a way it is not elsewhere:
    "Memphis" and "Memphis - ancient capital near Mit Rahina" are different
    subjects, hash to different ids, and produce two files for one city.
    Observed on a real `gpt-6-astra` run: every city, person and artifact came
    back with a trailing gloss.
    """

    def _history(self, **fields):
        from universe_builder.domains.history.models.history import History

        payload = {"people": [], "discoveries": [], "events": [], "cities": [],
                   "artifacts": []}
        payload.update(fields)
        return History.model_validate(payload)

    def test_a_trailing_gloss_is_stripped(self):
        h = self._history(cities=["Memphis — ancient capital near modern Mit Rahina."])

        assert h.cities == ["Memphis"]

    @pytest.mark.parametrize("dash", ["—", "–", "-"])
    def test_every_dash_that_is_punctuation_is_handled(self, dash):
        h = self._history(people=[f"Narmer {dash} early king of the First Dynasty"])

        assert h.people == ["Narmer"]

    @pytest.mark.parametrize("name", [
        "Pi-Ramesses", "Tell el-Amarna", "Deir el-Medina", "Abu Simbel",
    ])
    def test_a_hyphen_inside_a_name_survives(self, name):
        """The reason the pattern requires whitespace around the dash.

        A bare hyphen is part of the name far more often than it is punctuation,
        and splitting on it would corrupt exactly the names this is meant to
        clean.
        """
        assert self._history(cities=[name]).cities == [name]

    def test_a_parenthetical_alternate_name_survives(self):
        """`Thebes (Luxor)` is one place under two names, not a gloss."""
        assert self._history(cities=["Thebes (Luxor)"]).cities == ["Thebes (Luxor)"]

    def test_blank_entries_are_dropped(self):
        assert self._history(cities=["Memphis", "", "   "]).cities == ["Memphis"]

    def test_the_instruction_reaches_the_model(self):
        """The description is a real constraint, not prompt decoration.

        It survives `strict_json_schema` into `response_format`, which is the
        only reason tightening it is worth anything.
        """
        from universe_builder.domains.history.models.history import History
        from universe_builder.providers.base import Provider

        schema = Provider.strict_json_schema(History)
        description = schema["properties"]["cities"]["description"]

        assert "bare name" in description
        assert "Memphis" in description, "an instruction without an example was read as latitude"

    def test_normalising_never_rejects_the_atom(self):
        """A formatting habit must not cost several hundred topics.

        Refusing would throw away a frontier call over punctuation, and the gloss
        is not information worth keeping: the atom generated *for* that topic
        carries the long-form detail.
        """
        h = self._history(cities=["Memphis — capital", "Thebes (Luxor)", "Giza"])

        assert h.cities == ["Memphis", "Thebes (Luxor)", "Giza"]


class TestConstrainedFieldsAreActuallyConstrained:
    """`Field(enum=[...])` is schema decoration Pydantic does not enforce.

    `Date(era="banana")` validated happily for as long as the model existed. And
    for an `Optional[str]` it emitted `enum` as a *sibling* of `anyOf` with
    `null` absent from the enum — contradictory JSON Schema, since the type says
    null is permitted and the enum says only CE or BCE is.

    Found by a review of whether grammar-constrained decoding would help here.
    The honest answer was that the schema already *was* the constraint and simply
    was not being used: four fields declared an enum that nothing checked.
    """

    def _date(self, **overrides):
        from universe_builder.models.subatomic import Date

        return Date.model_validate({**make_date(), **overrides})

    def test_a_value_outside_the_enum_is_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self._date(era="banana")

    def test_the_declared_values_are_accepted(self):
        assert self._date(era="CE").era == "CE"
        assert self._date(era="BCE").era == "BCE"

    def test_none_is_still_allowed(self):
        """89 stored dates have a null era; tightening must not orphan them."""
        assert self._date(era=None).era is None

    def test_case_is_normalised_rather_than_rejected(self):
        """Rejecting would cost a whole atom over a capital letter.

        Observed, not hypothetical: the bundled `invention_of_toothpaste` example
        carried `"certainty": "High"` for years, because nothing enforced it.
        """
        assert self._date(era="ce").era == "CE"
        assert self._date(certainty="High").certainty == "high"

    def test_the_schema_no_longer_contradicts_itself(self):
        from universe_builder.models.subatomic import Date

        era = Date.model_json_schema()["properties"]["era"]

        assert "enum" not in era, "enum is a sibling of anyOf again"
        branches = era["anyOf"]
        assert any(b.get("enum") == ["CE", "BCE"] for b in branches)
        assert any(b.get("type") == "null" for b in branches)

    @pytest.mark.parametrize("field", ["certainty"])
    def test_certainty_is_constrained_wherever_it_appears(self, field):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            self._date(**{field: "quite sure"})

    def test_discovery_type_is_constrained(self):
        from pydantic import ValidationError

        from universe_builder.domains.history.models.discovery import Discovery

        with pytest.raises(ValidationError):
            Discovery.model_validate({"typeOfDiscovery": "vibes"})


class TestTheBundledExamplesObeyTheirOwnSchema:
    """An example is shipped *to the model* as the one-shot.

    A value the schema forbids is therefore being taught, not just stored, and
    `invention_of_toothpaste.json` was teaching `"High"`. `test_examples.py`
    passed throughout, because the enum was decoration.
    """

    def test_every_example_validates(self, prompt_config, examples_dir):
        import json

        from universe_builder.providers.registry import schema_model_by_name

        # expand_city's registered class is its SEED (`City`, the shape expand
        # reads); its example teaches the OUTPUT, a `Population` — the same
        # override every real expand request passes as `schema_model`.
        output_class = {"expand_city": "Population"}

        failures = []
        for atom, spec in prompt_config.items():
            configured = spec.get("example_json_file")
            if not configured:
                continue
            for name in ([configured] if isinstance(configured, str) else configured):
                path = examples_dir / name
                try:
                    schema_model_by_name(output_class.get(atom, spec["atom_class"])).model_validate(
                        json.loads(path.read_text()))
                except Exception as e:
                    failures.append(f"{name}: {str(e).splitlines()[1] if len(str(e).splitlines()) > 1 else e}")

        assert not failures, "examples that teach the model invalid values:\n" + "\n".join(failures)
