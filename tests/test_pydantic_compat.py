"""Guardrails for the Pydantic behaviours this codebase leans on.

The dependency is a range now (`pydantic>=2.11,<3`), not a pin, so a future
minor could in principle change one of these out from under the code. Each test
names one load-bearing behaviour, so a bump that breaks it fails *here*, clearly,
rather than as an obscure serialization or validation error deep inside a run.
The full suite already exercises these paths; these state them as contracts.
"""

from __future__ import annotations

import pytest
from pydantic import Field

from universe_builder.domains.history.models.city import Population, government_enum
from universe_builder.models.subatomic import EmptyModel, WireModel


def test_serialize_by_alias_config_is_honoured():
    """`WireModel` emits the on-disk (aliased) key by default via
    `ConfigDict(serialize_by_alias=True)` — the config that replaced a pair of
    `model_dump*` overrides once it landed in Pydantic 2.11. If a version stops
    honouring it, aliased atoms would silently serialise under their corrected
    field name and split the corpus across two spellings for one key."""

    class Aliased(WireModel):
        attr: str = Field(alias="wireKey")

    obj = Aliased(attr="v")  # accepted by field name (populate_by_name)
    assert "wireKey" in obj.model_dump_json()
    assert "wireKey" in obj.model_dump()
    # An explicit call-site by_alias=False must still win over the config.
    assert obj.model_dump(by_alias=False) == {"attr": "v"}


def test_empty_model_still_forbids_extra_fields():
    """The `_ERROR`/placeholder sentinel is `EmptyModel` with `extra="forbid"`;
    if extras were silently allowed, a real atom could masquerade as empty."""
    EmptyModel(error="nope")  # the only allowed shape
    with pytest.raises(Exception):
        EmptyModel(error="nope", cities=["Memphis"])


def test_json_schema_extra_enum_reaches_the_model_schema():
    """`typeOfGovernment` carries its allowed values via `json_schema_extra`
    (not the deprecated `enum=` kwarg); the schema the model is shown must still
    list them, or the guidance that steers the answer is gone."""
    schema = Population.model_json_schema()
    assert schema["properties"]["typeOfGovernment"].get("enum") == government_enum
