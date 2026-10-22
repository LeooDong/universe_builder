"""The atom registry, *derived* from the active `DomainPack`.

This module used to hold the facts itself — a hand-written `class_mapping`, a
`STAGE_MODELS` table, and `_load_atom_mapping` reading `prompt_config.json`. All
three now derive from `registry.active_pack()`, the single declaration of what
atoms exist (see `registry.py`). The public surface is unchanged, so callers do
not care where the facts come from:

- `atom_mapping` — atom name (incl. `expand_*`/`append_*`) -> `AtomMapping`, the
  metadata a generating call needs. This is the schema sent to the model.
- `class_mapping` — class name -> class, kept for callers that resolve by name.
- `resolve_stored_class` — the model matching the payload actually on disk, which
  is not always `atom_mapping[name].atom_class`. See `AtomSpec.stages`.

`reload_from_pack()` rebuilds the tables in place after `registry.set_active_pack`
so a switch is visible to modules that imported the dicts by reference.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, Union

from universe_builder import registry

log = logging.getLogger(__name__)


@dataclass
class AtomMapping:
    atom_class: Type
    prompt_start_text: str
    example_json_file: Optional[Union[str, List[str]]]
    plural: Optional[str]

    @property
    def class_name(self) -> str:
        return self.atom_class.__name__


def _build_atom_mapping(pack: registry.DomainPack) -> Dict[str, AtomMapping]:
    """Every atom name the pipeline can be asked for, with its generating schema.

    A base atom yields its own entry; an atom with an `ExpandSpec`/`AppendSpec`
    yields an `expand_*`/`append_*` entry too. The augment entries report the
    *seed* class for expand (the file it reads is a seed) and the append model
    for append — matching the old `expand_city`/`append_city` rows exactly.
    """
    mapping: Dict[str, AtomMapping] = {}
    for spec in pack.all_specs():
        mapping[spec.name] = AtomMapping(spec.model, spec.prompt_start, spec.example, spec.plural)
        if spec.expand:
            mapping[f"expand_{spec.name}"] = AtomMapping(
                spec.model, spec.expand.prompt_start, spec.expand.example, spec.plural
            )
        if spec.append:
            mapping[f"append_{spec.name}"] = AtomMapping(
                spec.append.model, spec.append.prompt_start, spec.append.example, None
            )
    return mapping


def _build_class_mapping(pack: registry.DomainPack) -> Dict[str, Type]:
    """Class name -> class for every stored/seed shape (not the expander entry).

    Mirrors the old hand-written `class_mapping`: seed models, the index model,
    plus each atom's expanded and append models. The expander entry model
    (`Population`) is deliberately absent here — it is a sub-object, not a stored
    atom — but it is present in the batch schema registry, which needs to resolve
    it by name; see `providers/registry._schema_models`.
    """
    classes: Dict[str, Type] = {}
    for spec in pack.all_specs():
        classes[spec.model.__name__] = spec.model
        if spec.expand:
            classes[spec.expand.expanded_model.__name__] = spec.expand.expanded_model
        if spec.append:
            classes[spec.append.model.__name__] = spec.append.model
    return classes


def _build_stage_models(pack: registry.DomainPack):
    return {spec.name: spec.stages for spec in pack.all_specs() if spec.stages}


#: Populated from the active pack at import; rebuilt in place by
#: `reload_from_pack()`. Kept as module dicts because callers import them by
#: reference (`from universe_builder.atoms import atom_mapping`); mutating in place keeps
#: those references live across a pack switch.
atom_mapping: Dict[str, AtomMapping] = {}
class_mapping: Dict[str, Type] = {}
#: Some atoms have more than one on-disk shape (create writes one model, a later
#: pass rewrites the file as another), so a stored file cannot be resolved to a
#: class by atom name alone. Derived from each `AtomSpec.stages`.
STAGE_MODELS: Dict[str, Any] = {}
BASE_ATOM_TYPES: list[str] = []


def reload_from_pack() -> None:
    """Rebuild the derived tables from `registry.active_pack()`, in place."""
    pack = registry.active_pack()
    BASE_ATOM_TYPES[:] = [spec.name for spec in pack.atoms]
    atom_mapping.clear()
    atom_mapping.update(_build_atom_mapping(pack))
    class_mapping.clear()
    class_mapping.update(_build_class_mapping(pack))
    STAGE_MODELS.clear()
    STAGE_MODELS.update(_build_stage_models(pack))
    # The request builder caches the base system text and examples, both of which
    # are pack-specific; drop them so a switch is visible there too.
    try:
        from universe_builder.providers.requests import clear_pack_caches
        clear_pack_caches()
    except Exception:  # noqa: BLE001 - providers optional at import time; never fatal
        pass


def use_pack(name: str) -> None:
    """Switch the engine to domain pack `name` and rebuild the derived tables.

    The entry point for a *runtime* domain switch: `registry.set_active_pack`
    alone updates the choice but leaves `atom_mapping` and friends showing the
    old pack until this rebuilds them. Tests use it (with restore) to exercise a
    second domain. The initial domain from `UNIVERSE_BUILDER_DOMAIN` is handled in `registry`
    itself (so `active_pack()` is correct before `atoms` is even imported); this
    is for switching after start-up. This process-global rebuild is not thread-safe:
    call it only while pipeline workers are idle; use separate processes otherwise.
    """
    registry.set_active_pack(name)
    reload_from_pack()


reload_from_pack()


def resolve_stored_class(atom_name: str, data: Any) -> Type:
    """Return the model matching the shape a stored payload actually has.

    Resolving by atom name alone was what made B7 dangerous: an expanded city
    file no longer validates as `City`, so the fix and verify passes would have
    handed the LLM the seed schema and collapsed the expansion back to a list of
    period names — destroying the expand output without an error.
    """
    default = atom_mapping[atom_name].atom_class
    if not isinstance(data, dict):
        return default
    for predicate, stage_class in STAGE_MODELS.get(atom_name, ()):
        try:
            if predicate(data):
                return stage_class
        except Exception:  # a malformed payload just falls back to the default
            continue
    return default
