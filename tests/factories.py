"""Schema-driven builders: a minimal valid instance of *any* atom model.

Mechanical tests — the ones exercising the store, the merge helper, the batch
splice, the typed `Result` — need "a valid atom" as a vehicle, not a hand-typed
`City` with all ten fields spelled out. When those tests name domain fields, a
change to `City`'s shape breaks tests that were never about `City`. `conftest`
already dodges that once: `_make_geography` builds its payload from
`MonthlyClimate.model_fields` so it cannot drift from the model. This is that
trick generalised to every model.

`minimal_instance(Model)` walks `model_fields` and fills each with the smallest
value the schema accepts, recursing into sub-models. It deliberately builds a
value even for `Optional` fields (the first non-None member), so required
substructure is exercised rather than papered over with `None`. Field validators
run on construction, so the result is a genuinely valid instance — the same
guarantee a live reply has to meet.
"""

from __future__ import annotations

from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

_NONE_TYPE = type(None)


def _bounds(metadata) -> tuple[int | None, int | None]:
    """Read numeric floor/ceiling from a field's `annotated_types` constraints.

    Judgments scores are `Annotated[int, Field(ge=0, le=10)]` and `Date.year` is
    `ge=0, le=60000`; the constraints arrive as objects carrying `.ge/.gt/.le/.lt`.
    A plain `1` satisfies every bound in the current corpus, but reading them
    keeps the builder correct if a future field floors above 1.
    """
    lo = hi = None
    for m in metadata:
        for attr in ("ge", "gt"):
            v = getattr(m, attr, None)
            if v is not None:
                lo = v if attr == "ge" else v + 1
        for attr in ("le", "lt"):
            v = getattr(m, attr, None)
            if v is not None:
                hi = v if attr == "le" else v - 1
    return lo, hi


def _number(lo, hi, default):
    """The smallest in-range value >= our preferred `default`."""
    value = default
    if lo is not None and value < lo:
        value = lo
    if hi is not None and value > hi:
        value = hi
    return value


def _value_for(annotation: Any, metadata: list) -> Any:
    """A schema-valid placeholder for one annotation, recursing as needed."""
    # A type alias like `Certainty = Annotated[Literal[...], BeforeValidator]`
    # carries its constraints in `__metadata__`; unwrap to the real type and
    # keep the extra metadata alongside whatever the field already supplied.
    if hasattr(annotation, "__metadata__"):
        return _value_for(annotation.__origin__, list(annotation.__metadata__) + metadata)

    origin = get_origin(annotation)

    if origin is Literal:
        return get_args(annotation)[0]

    if origin is Union:  # includes Optional[X]; build the first real member,
        members = [a for a in get_args(annotation) if a is not _NONE_TYPE]
        return _value_for(members[0], metadata)  # never None — exercise the shape

    if origin in (list, tuple):
        args = get_args(annotation)
        inner = args[0] if args else str
        return [_value_for(inner, [])]

    if origin in (dict,):
        return {}

    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return minimal_instance(annotation)

    if annotation is bool:  # before int, since bool is an int subclass
        return True
    if annotation is int:
        lo, hi = _bounds(metadata)
        return _number(lo, hi, 1)
    if annotation is float:
        lo, hi = _bounds(metadata)
        return float(_number(lo, hi, 1))
    if annotation is str:
        return "x"

    raise TypeError(
        f"minimal_instance cannot infer a value for annotation {annotation!r}; "
        "add a case to _value_for or pass it explicitly via an override."
    )


def minimal_instance(model: type[BaseModel], **overrides: Any) -> BaseModel:
    """A validated instance of `model` with every field filled minimally.

    `overrides` replace generated values by field name before construction, so a
    test can pin the one field it cares about and let the rest fall out of the
    schema: `minimal_instance(City, name=minimal_instance(Name, common="Athens"))`.
    """
    values: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        if name in overrides:
            values[name] = overrides[name]
            continue
        values[name] = _value_for(field.annotation, list(field.metadata))
    # Pass by field name; WireModel subclasses set `populate_by_name`, so the
    # aliased wire spelling (`discoverorOrInventor`) is reached without us having
    # to know it here.
    return model(**values)


def minimal_payload(model: type[BaseModel], **overrides: Any) -> dict:
    """`minimal_instance` as a JSON-ready dict — what most tests write to a file.

    Dumped `by_alias` (the `WireModel` default), so the keys match what lands on
    disk, misspellings and all.
    """
    return minimal_instance(model, **overrides).model_dump(mode="json")


def _jsonable(value: Any) -> Any:
    """Normalise a generated value to plain JSON types, so a backfilled field
    sits in a payload dict as a dict/list/scalar rather than a model instance."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value


def backfill_required(model: type[BaseModel], payload: dict) -> dict:
    """`payload` with a minimal placeholder added for every *required* field it
    is missing — and nothing else touched.

    This is the behaviour-preserving half of shape-blinding. A hand-written
    fixture that already sets every required field comes back byte-identical (so
    the corpus of assertions built on it does not shift, and optional fields keep
    the model's own defaults rather than acquiring placeholders). But a required
    field *added* to the model later is filled automatically instead of breaking
    every fixture that predates it, and a field *renamed* on the model is
    backfilled under its new name — the fixture's stale key is dropped on
    construction by pydantic's default `extra="ignore"`. So a shape change to an
    atom stops cascading into tests that only needed *a valid atom*, which is the
    whole point of Step 1.
    """
    out = dict(payload)
    for name, field in model.model_fields.items():
        if not field.is_required():
            continue
        if name in out or (field.alias and field.alias in out):
            continue
        out[name] = _jsonable(_value_for(field.annotation, list(field.metadata)))
    return out
