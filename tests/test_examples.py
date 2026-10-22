"""Every one-shot example must validate against the model it is registered for.

This is the preflight check that makes changing an atom's JSON structure safe:
a schema change that invalidates an example fails here instead of silently
degrading every prompt that ships that example. `LLMCall.validate_example_json`
performs the same check, but only at call-construction time — by which point a
key has already been loaded and the run has started.
"""

from __future__ import annotations

import json

import pytest

from universe_builder.domains.history.models.artifact import Artifact
from universe_builder.domains.history.models.city import City, CityAppend, Population
from universe_builder.domains.history.models.discovery import Discovery
from universe_builder.domains.history.models.event import Event
from universe_builder.domains.history.models.figure import Figure
from universe_builder.domains.history.models.history import History
from universe_builder.domains.history.models.monument import Monument
from universe_builder.domains.history.models.person import Person
from universe_builder.domains.history.models.polity import Polity
from universe_builder.domains.history.models.work import Work

CLASSES = {
    "History": History,
    "Person": Person,
    "Event": Event,
    "Discovery": Discovery,
    "City": City,
    "CityAppend": CityAppend,
    "Artifact": Artifact,
    "Population": Population,
    "Polity": Polity,
    "Monument": Monument,
    "Work": Work,
    "Figure": Figure,
}


#: An example teaches the model what to OUTPUT, and for expand_city that is a
#: `Population`, not the registered class — which is the SEED shape the file
#: holds before expansion, consumed by `operations/expand.py`. Every real
#: expand request overrides `schema_model` the same way.
OUTPUT_CLASS_OVERRIDES = {"expand_city": "Population"}


def _registered_examples():
    """(atom_name, filename, class_name) for every configured example.

    Reads the active `DomainPack` via `atom_mapping` (was `prompt_config.json`,
    now the single source of truth — see `universe_builder.registry`)."""
    from universe_builder import atoms

    pairs = []
    for atom_name, entry in atoms.atom_mapping.items():
        configured = entry.example_json_file
        if not configured:
            continue
        filenames = [configured] if isinstance(configured, str) else list(configured)
        for filename in filenames:
            class_name = OUTPUT_CLASS_OVERRIDES.get(atom_name, entry.atom_class.__name__)
            pairs.append((atom_name, filename, class_name))
    return pairs


REGISTERED = _registered_examples()


@pytest.mark.parametrize(
    "atom_name,filename,class_name",
    REGISTERED,
    ids=[f"{a}:{f}" for a, f, _ in REGISTERED],
)
def test_registered_example_validates(atom_name, filename, class_name, examples_dir):
    with open(examples_dir / filename) as f:
        payload = json.load(f)
    CLASSES[class_name].model_validate(payload)


def test_at_least_one_example_is_registered():
    assert REGISTERED, "no examples registered — the parametrised test would vacuously pass"


def test_every_example_file_on_disk_is_valid_json(examples_dir):
    """Catches a truncated or hand-edited example before it reaches a prompt."""
    broken = []
    for path in sorted(examples_dir.glob("*.json")):
        try:
            with open(path) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            broken.append(f"{path.name}: {e}")
    assert not broken, broken


def test_unregistered_examples_are_accounted_for(examples_dir):
    """An example file nothing references is either dead weight or a missing
    registry entry. Both are worth knowing about; neither is an error."""
    on_disk = {p.name for p in examples_dir.glob("*.json")}
    registered = {f for _, f, _ in REGISTERED}
    orphans = on_disk - registered
    assert not orphans, f"unexpected unregistered example files: {sorted(orphans)}"


def test_roman_period_alexandria_validates_as_population(examples_dir):
    """A long-standing xfail, since turned green: the example predated
    the `Population` schema and expand_city ran with no example at all. It was
    regenerated live against the current schema (gpt-5.6-luna) and registered,
    so every expand call now carries a one-shot example in its cached prefix."""
    with open(examples_dir / "roman_period_alexandria.json") as f:
        Population.model_validate(json.load(f))
