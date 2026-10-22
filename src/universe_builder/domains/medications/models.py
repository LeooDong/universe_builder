"""Models for the medications domain — a second problem shape on the same engine.

Deliberately structured to mirror the history domain limb-for-limb, so that any
history-specific assumption left in the core surfaces as a failure here rather
than as a design smell nobody notices:

    History  (index, lists topic names per type)  <->  Formulary (lists drugs)
    City     (seed atom, a list field of names)    <->  Medication (indications)
    Population (one expanded list entry)           <->  IndicationDetail
    CityExpanded (seed + list of objects|names)    <->  MedicationExpanded
    CityAppend (a supplement to merge in)          <->  MedicationAppend

`subatomic` primitives (`Name`, `Certainty`) are reused where they fit — they
were never history-specific, only ever used there first.
"""

from __future__ import annotations

from typing import List, Optional, Union

from pydantic import BaseModel, Field

from universe_builder.models.subatomic import Certainty, Name


class IndicationDetail(BaseModel):
    """One expanded indication — the medications analogue of `Population`.

    A single object, so (like `Population`) its schema says nothing about
    producing several: the expand call turns one indication *name* into one of
    these, never a list.
    """

    indicationName: str = Field(description="The condition this medication treats, e.g. 'Headache'.")
    typicalDoseDescription: str = Field(description="Usual adult dose and frequency, in prose.")
    effectiveness: Certainty = Field(description="How well-established the effect is for this indication.")
    contraindications: List[str] = Field(description="Conditions or co-medications that rule this use out.")
    evidenceSources: List[str] = Field(description="Citations for the dose and effectiveness claims.")


class Medication(BaseModel):
    """A drug, seeded with a list of indication *names* (mirrors `City` and its
    `cityPeriodsOfHistory` list of period names)."""

    name: Name
    drugClass: str = Field(description="Pharmacological class, e.g. 'NSAID'.")
    atcCode: Optional[str] = Field(None, description="WHO ATC classification code, if known.")
    routesOfAdministration: List[str] = Field(description="e.g. 'oral', 'intravenous'.")
    indications: List[str] = Field(
        description="Conditions this medication is used to treat, each a bare name.",
    )


class MedicationExpanded(Medication):
    """A `Medication` after expand has run over `indications`.

    The member type is a union for the same reason `CityExpanded` uses one:
    expansion is per-entry and best-effort, so an indication whose call failed
    stays a string and the file still round-trips. Any string left means the
    medication is only partially expanded.
    """

    indications: List[Union[IndicationDetail, str]] = Field(
        description="Indications expanded into IndicationDetail objects; entries left as "
                    "plain strings are indications not yet expanded.",
    )

    def is_fully_expanded(self) -> bool:
        return all(isinstance(entry, IndicationDetail) for entry in self.indications)

    def unexpanded_indications(self) -> List[str]:
        return [entry for entry in self.indications if isinstance(entry, str)]


class MedicationAppend(BaseModel):
    """A supplement merged into a `Medication` (mirrors `CityAppend`)."""

    brandNames: List[str] = Field(description="Trade names the drug is marketed under.")


class Formulary(BaseModel):
    """The index atom: what medications exist, as bare names (mirrors `History`)."""

    medications: List[str] = Field(description="Names of medications in this formulary, each a bare name.")
