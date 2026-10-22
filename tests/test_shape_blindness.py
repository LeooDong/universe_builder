"""The shape-blindness guarantee, pinned: a shape change to an atom does not cascade
into the *mechanical* tests — the ones that use an atom only as a vehicle.

The mechanism is `backfill_required` (tests/factories.py), through which every
conftest payload factory now passes. These tests prove the property directly
rather than by re-running the suite against a mutated model: they synthesise a
`City` with an extra required field and show the fixture path heals it.

What this proves: adding or renaming a *required* field on a model no longer
breaks a fixture that predates the change. What it does NOT prove: that a test
which deliberately inlines a domain field (because that field is the subject of
the mechanic under test — expand's period list, a merge conflict field) is
shape-blind. Those references are intentional and belong where they are; the
semantic tests (test_models, test_periods, test_aliases) are *meant* to move
with the shape.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError, create_model

from universe_builder.domains.history.models.city import City
from tests.conftest import make_city
from tests.factories import backfill_required, minimal_payload


def _city_with_extra_required():
    """A `City` subclass with one more required field the fixtures never heard of.

    Stands in for a future edit to `City` (a new required field) or a rename (an
    old key gone, a new required one arrived) — the exact change that breaks a
    hand-enumerated fixture today.
    """
    return create_model("CityPlusRequired", __base__=City, newlyRequiredField=(str, ...))


def test_a_stale_fixture_would_fail_without_backfill():
    """Guard the guard: the synthesised field really is required, so the healing
    below is doing work rather than passing vacuously."""
    CityPlus = _city_with_extra_required()
    with pytest.raises(ValidationError):
        CityPlus.model_validate(make_city())  # make_city predates the new field


def test_backfill_heals_a_new_required_field_with_no_fixture_edit():
    CityPlus = _city_with_extra_required()
    healed = backfill_required(CityPlus, make_city())
    # No edit to make_city was needed; the new field was filled from the schema.
    assert CityPlus.model_validate(healed).newlyRequiredField


def test_minimal_payload_covers_a_new_field_directly():
    CityPlus = _city_with_extra_required()
    CityPlus.model_validate(minimal_payload(CityPlus))


def test_make_city_still_satisfies_the_unchanged_city():
    """The canary: the behaviour-preserving path still yields a valid City."""
    City.model_validate(make_city())
