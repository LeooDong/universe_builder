"""Proof that `minimal_instance` builds and round-trips every atom model.

This is the guarantee Step 1 leans on: mechanical tests can ask for "a valid
atom" of any shape and get one. Models are discovered by scanning the model
modules, so a model added later is covered here automatically rather than
silently escaping the guarantee.
"""

from __future__ import annotations

import inspect

import pytest
from pydantic import BaseModel

from universe_builder import atoms
from universe_builder.domains.history.models import artifact, city, discovery, event, history, person
from universe_builder.models import subatomic
from tests.factories import minimal_instance, minimal_payload

_MODULES = (subatomic, history, person, event, discovery, artifact, city)
#: The base classes themselves are infrastructure, not atoms to instantiate.
_BASES = {BaseModel, subatomic.WireModel}


def _all_models() -> dict[str, type[BaseModel]]:
    """Every concrete BaseModel subclass defined in the model modules."""
    found: dict[str, type[BaseModel]] = {}
    for module in _MODULES:
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if (
                issubclass(obj, BaseModel)
                and obj not in _BASES
                and obj.__module__ == module.__name__  # defined here, not imported
            ):
                found[obj.__name__] = obj
    # The registered atom classes must all be reachable too, even if one ever
    # lived outside the scanned modules.
    found.update({c.__name__: c for c in atoms.class_mapping.values()})
    return found


ALL_MODELS = _all_models()


def test_the_registered_atoms_are_all_discovered():
    for cls in atoms.class_mapping.values():
        assert cls.__name__ in ALL_MODELS


@pytest.mark.parametrize("name", sorted(ALL_MODELS))
def test_minimal_instance_builds_and_round_trips(name):
    model = ALL_MODELS[name]
    instance = minimal_instance(model)
    assert isinstance(instance, model)
    # A live reply has to survive dump -> reload; the placeholder must too, or it
    # is not a faithful stand-in for one.
    reloaded = model.model_validate_json(instance.model_dump_json())
    assert isinstance(reloaded, model)


def test_city_expanded_union_field_builds_a_population_not_a_string():
    """The `Union[Population, str]` member must build the object arm, so an
    expanded city produced here is genuinely fully expanded."""
    expanded = minimal_instance(city.CityExpanded)
    assert expanded.cityPeriodsOfHistory
    assert all(isinstance(p, city.Population) for p in expanded.cityPeriodsOfHistory)
    assert expanded.is_fully_expanded()


def test_bounded_ints_land_in_range():
    """Judgments scores are 0..10 and Date.year is 0..60000; the builder must
    pick a value inside the bound, not a blind constant that could fall outside."""
    judgments = minimal_instance(subatomic.Judgments)
    assert 0 <= judgments.politicalInfluenceScore <= 10
    assert 0 <= (minimal_instance(subatomic.Date).year or 0) <= 60000


def test_wire_aliased_model_dumps_on_the_wire_spelling():
    """Discovery's attribute is spelled correctly; the payload keeps the corpus's
    misspelled key, which is what tests writing files to disk depend on."""
    payload = minimal_payload(discovery.Discovery)
    assert "discoverorOrInventor" in payload  # the wire (aliased) key
    assert "discovererOrInventor" not in payload


def test_overrides_replace_named_fields():
    named = minimal_instance(city.City, name=minimal_instance(subatomic.Name, common="Athens"))
    assert named.name.common == "Athens"
