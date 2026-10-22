"""The medications domain pack.

Importing this module registers the pack under "medications", so
`registry.set_active_pack("medications")` (and `atoms.use_pack`) discover it by
name. It exists to prove the engine is domain-agnostic: a different problem, a
different set of JSON shapes, the same passes — create, expand, append — with no
edit to any core module.
"""

from __future__ import annotations

from pathlib import Path

from universe_builder import registry
from universe_builder.domains.medications.models import (
    Formulary,
    IndicationDetail,
    Medication,
    MedicationAppend,
    MedicationExpanded,
)

#: This domain ships its own one-shot examples beside the code. Both paths find
#: them through `DomainPack.examples_dir`: the batch path reads it directly, and
#: `LLMCall.__init__` prefers `active_pack().examples_dir` over the bundled
#: package examples, so a synchronous run of this pack loads its own examples
#: without any caller threading a folder through.
MODULE_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = MODULE_DIR / "examples"

#: A pharmacology persona — deliberately NOT the historian text — so a run under
#: this pack proves the base-system prompt is swapped, not shared.
BASE_SYSTEM = """
    You are a clinical pharmacologist with comprehensive knowledge of medicines,
    their indications, dosing, contraindications and the evidence behind them.
    You are also designed to output JSON.
    Please fill in the following in json format following the defined json schema below.
    Do this to the best of your ability.
    If unsure, leave blank or null and record that in the corresponding certainty field.
    Give your answers spelling in UK English.
    For sources, be specific and cite real clinical evidence where possible.
""".strip()


def _medication_is_expanded(data: dict) -> bool:
    """A stored medication whose `indications` hold objects is a `MedicationExpanded`
    (mirrors the city stage predicate)."""
    return any(isinstance(x, dict) for x in data.get("indications") or [])


def build() -> registry.DomainPack:
    medication = registry.AtomSpec(
        name="medication",
        model=Medication,
        plural="medications",
        prompt_start="Provide a response for the medication commonly known as ",
        example="aspirin.json",
        stages=((_medication_is_expanded, MedicationExpanded),),
        expand=registry.ExpandSpec(
            list_field="indications",
            entry_model=IndicationDetail,
            expanded_model=MedicationExpanded,
            prompt_start="Provide a response for this to this json schema about ",
            example="aspirin_headache_indication.json",
        ),
        append=registry.AppendSpec(
            model=MedicationAppend,
            prompt_start="Please append information to this json schema about ",
            example=None,
        ),
    )

    return registry.DomainPack(
        name="medications",
        index=registry.AtomSpec(
            name="formulary",
            model=Formulary,
            plural="medications",
            prompt_start="Give me an exhaustive but finite list of medications under the heading: ",
            example="essential_medicines.json",
        ),
        atoms=(medication,),
        # No place or date identifies a medication, so lint's corroboration
        # reports "no comparable place or date" for every pair and
        # `sources.context_terms` finds nothing — honestly, not by accident.
        coordinate_paths={},
        date_paths={},
        preferred_hosts=("who.int", "bnf.nice.org.uk", "dailymed.nlm.nih.gov",
                         "ema.europa.eu", "pubmed.ncbi.nlm.nih.gov", "doi.org"),
        near_match_stopwords=frozenset("""
            the of and in for a an tablet tablets capsule capsules oral injection
            sodium hydrochloride sulfate acetate extended release
        """.split()),
        examples_dir=EXAMPLES_DIR,
        base_system=BASE_SYSTEM,
    )


registry.register_pack("medications", build)
