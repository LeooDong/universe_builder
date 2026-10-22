"""Pack consistency: the one `DomainPack` is internally coherent.

The registry used to live in three hand-maintained tables that these tests kept
in agreement (`prompt_config.json`, `class_mapping`, `_schema_models`). Since
they all derive from `registry.active_pack()` now (see `universe_builder.registry`),
comparing them to each other is tautological and those checks are gone. What
remains are the invariants a single declaration can still get *wrong*: a `plural`
that names no field on the index model, an example file that is absent, a
still-hardcoded list that has drifted from the pack. These are what make "add an
atom" and "change an atom's shape" safe now.
"""

from __future__ import annotations

import json

import pytest

from universe_builder import settings
from universe_builder.registry import active_pack


@pytest.fixture(scope="module")
def execute_prompt():
    from universe_builder import execute_prompt

    return execute_prompt


class TestPackConsistency:
    def test_every_atom_plural_names_a_field_on_the_index_model(self):
        """The create pass fans out with `getattr(index_instance, spec.plural)`,
        so a plural that is not a field on the index atom produces nothing. This
        is the check that used to guard History being a second registry; now it
        guards one declaration against the index model it points at."""
        pack = active_pack()
        index_fields = set(pack.index.model.model_fields)
        missing = {s.name: s.plural for s in pack.atoms if s.plural and s.plural not in index_fields}
        assert not missing, f"plural does not name an index-model field: {missing}"

    def test_every_index_field_has_a_generated_atom(self):
        """The reverse: an index field no atom lists produces topics nothing will
        ever consume."""
        pack = active_pack()
        listed = {s.plural for s in pack.atoms if s.plural}
        orphaned = set(pack.index.model.model_fields) - listed
        assert not orphaned, f"index-model fields with no registered atom: {orphaned}"

    def test_the_index_plural_is_none_or_shadows_an_atom_it_does_not_own(self):
        """Metadata coherence only: the index's own `plural` need not name a field
        on the index model, because the index lists no topics about itself. It may
        be None (history) or coincide with an atom's plural (medications'
        Formulary and Medication both say `medications`). This asserts the
        declaration is one of those two shapes; the *behavioural* guarantee that
        the index is excluded from membership building lives in
        `test_index.py::TestMembership...index_atom_is_never_a_member`, which runs
        the real builder."""
        pack = active_pack()
        assert pack.index.name not in {s.name for s in pack.atoms}
        atom_plurals = {s.plural for s in pack.atoms if s.plural}
        assert pack.index.plural is None or pack.index.plural in atom_plurals

    def test_base_atom_types_excludes_the_index_atom(self, execute_prompt):
        """`BASE_ATOM_TYPES` now derives from `pack.atoms`, so it equals them by
        construction; the invariant still worth asserting is that the create loop
        never tries to generate the *index* atom (the seed it reads work out of)."""
        base = set(execute_prompt.BASE_ATOM_TYPES)
        assert base == {s.name for s in active_pack().atoms}
        assert active_pack().index.name not in base and base

    def test_every_configured_example_file_exists(self, examples_dir):
        pack = active_pack()
        base = pack.examples_dir or examples_dir
        missing = []

        def _files(example):
            # `example` is a single filename, a list of them (few-shot), or None.
            if not example:
                return []
            return [example] if isinstance(example, str) else list(example)

        for spec in pack.all_specs():
            for example in (spec.example, spec.expand.example if spec.expand else None,
                            spec.append.example if spec.append else None):
                for filename in _files(example):
                    if not (base / filename).exists():
                        missing.append(f"{spec.name} -> {filename}")
        assert not missing, f"example files referenced but absent: {missing}"

    def test_every_prompt_start_text_is_a_nonempty_string(self):
        for spec in active_pack().all_specs():
            assert isinstance(spec.prompt_start, str) and spec.prompt_start.strip(), spec.name

    def test_an_expander_is_a_different_shape_from_its_seed(self):
        """B7 in one line: writing expander objects into the seed class is what
        made the corpus unreadable, so the two must be distinct models."""
        for spec in active_pack().all_specs():
            if spec.expand:
                assert spec.expand.expanded_model is not spec.model
                assert spec.expand.entry_model is not spec.model

    def test_batch_registry_can_resolve_every_atom_class(self, execute_prompt):
        from universe_builder.providers import registry

        known = set(registry._schema_models())
        classes = {m.atom_class.__name__ for name, m in execute_prompt.atom_mapping.items()}
        missing = classes - known
        assert not missing, f"atom classes the batch path cannot resolve by name: {missing}"


class TestPhaseModels:
    """Cross-phase invariants on PHASE_MODELS.

    These are hidden couplings — nothing in the code enforces them, and getting
    one wrong produces no error, just missing output.
    """

    def test_append_may_differ_from_create(self):
        """The pin is gone, and this documents *why it may stay gone*: merge
        pairing is id-based, exercised directly rather than assumed. An append
        from a different model must merge into its target — this failing means
        someone re-coupled pairing to the model string, and the old invariant
        (`PHASE_MODELS["append"] == PHASE_MODELS["create"]`) must come back
        before a cheaper append tier ships."""
        from universe_builder import store
        from universe_builder.operations.merge import pair_keys

        city = store.AtomName("city", None, "Memphis", "gpt-5.6-luna")
        append = store.AtomName("append_city", None, "Memphis", "gpt-5-nano")

        shared = set(pair_keys(store.AtomName.decompose_filename(city.getReadableFileName()))) \
            & set(pair_keys(store.AtomName.decompose_filename(append.getReadableFileName())))
        assert shared, (
            "a cross-model append no longer pairs with its target; re-pin "
            "append to create's model until this is fixed"
        )

    def test_every_phase_model_is_registered_in_llm_call_mapping(self, execute_prompt):
        """A phase pointing at an unregistered model raises KeyError at call time,
        i.e. partway through a paid run."""
        missing = {}
        for phase, model in execute_prompt.PHASE_MODELS.items():
            # create/history dispatch on the bare string, the augment phases on
            # the (model, "augment") tuple.
            keys = [model, (model, "augment")]
            if not any(k in execute_prompt.llm_call_mapping for k in keys):
                missing[phase] = model
        assert not missing, f"phases pointing at unregistered models: {missing}"

    def test_augment_phases_have_an_augment_variant_registered(self, execute_prompt):
        augment_phases = ("fix", "verify", "expand", "append")
        missing = {
            phase: execute_prompt.PHASE_MODELS[phase]
            for phase in augment_phases
            if (execute_prompt.PHASE_MODELS[phase], "augment") not in execute_prompt.llm_call_mapping
        }
        assert not missing, f"augment phases with no augment class registered: {missing}"

    def test_every_phase_model_is_priced(self, execute_prompt):
        with open(settings.MODEL_PRICING) as f:
            priced = set(json.load(f)["models"])
        unpriced = {
            phase: model
            for phase, model in execute_prompt.PHASE_MODELS.items()
            if model not in priced
        }
        assert not unpriced, f"phases pointing at unpriced models: {unpriced}"


class TestModelDefaults:
    """Model IDs named in code must be ones the provider layer can actually
    route and price."""

    def test_llm_call_mapping_models_are_known_to_the_pricing_registry(self, execute_prompt):
        with open(settings.MODEL_PRICING) as f:
            priced = set(json.load(f)["models"])

        configured = {
            key[0] if isinstance(key, tuple) else key
            for key in execute_prompt.llm_call_mapping
        }
        # Ollama models are local and deliberately unpriced.
        configured -= {"llama3"}

        unknown = configured - priced
        assert not unknown, (
            f"models registered in llm_call_mapping with no pricing entry: {sorted(unknown)}. "
            "Either they are stale or model_pricing.json needs a row."
        )
