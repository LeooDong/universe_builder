"""A state: an empire, kingdom, dynasty, republic or sultanate.

Registered in the pack (`domains/history/__init__.py`).

A polity is a state with its own ruler. A dynasty is the polity where history
names the state by it (Tang, Safavid, the Ptolemaic Kingdom); a state several
dynasties ruled is one polity (the Byzantine Empire). A province under an
appointed governor is not a polity but a period of its cities (Roman Egypt); a
vassal with its own ruling house is (the Khedivate of Egypt).

Rome and Byzantium are cut at 395, the division of the empire at the death of
Theodosius I: the Roman Empire until 395, the Byzantine Empire in the east from
395. So Constantine's works belong to the Roman Empire and Justinian's to the
Byzantine Empire, in `commissioningPolity` and in the two polities' dates alike.
"""

from pydantic import BaseModel, Field, field_validator, model_validator
from typing import List, Optional

from universe_builder.domains.history.models._enum import coerce
from universe_builder.domains.history.models.city import government_enum
from universe_builder.models.subatomic import Date, Judgments, Location, Name

#: Years in the sub-lists are signed integers (negative for BCE), the form the
#: period machinery already compares (`periods._signature`), rather than full
#: `Date` objects: a polity lists tens of rulers and several capitals, and a
#: `Date` with its certainty, sources and prose per entry would multiply the
#: output for detail the app never shows.
_SIGNED = "Signed year: negative for BCE, positive for CE; null if unknown."


class Capital(BaseModel):
    city: str = Field(description="The capital city's common name, as its own atom would name it.")
    fromYear: Optional[int] = Field(description=f"When it became the capital. {_SIGNED}")
    toYear: Optional[int] = Field(description=f"When it stopped being the capital. {_SIGNED}")


class Ruler(BaseModel):
    name: str = Field(description="The ruler's name, as their own person atom would name them.")
    title: Optional[str] = Field(description="The title they ruled under (Pharaoh, Shah, Emperor, Sultan).")
    reignStart: Optional[int] = Field(
        description=f"First accession, where a reign was interrupted or began as co-ruler. {_SIGNED}")
    reignEnd: Optional[int] = Field(description=f"Final end of the reign. {_SIGNED}")


class Polity(BaseModel):
    name: Name
    dateFounded: Date
    dateEnded: Date = Field(
        description="The end of the state as a polity; for a state that still exists, a null year.")
    location: Location = Field(description="The heartland, placed at its principal capital.")
    typeOfGovernment: str = Field(
        json_schema_extra={"enum": government_enum},
        description="Type of government for most of its existence.",
    )
    capitals: List[Capital] = Field(description="Every capital, in order, while it was this state.")
    #: "Name each unambiguously across all of history" was here once.
    #: The sample obeyed it for everyone ("Pericles of Athens",
    #: "Cimon of Athens"), and those names no longer reach the stored
    #: `Pericles`. Ambiguity is caught after create, by the link report.
    rulers: List[Ruler] = Field(
        description="Every sovereign if there were 25 or fewer; otherwise the 25 most significant, in order.")
    #: A separate field because a link path has one target type: a king known
    #: only from legend (the Yellow Emperor, Jamshid, Romulus) is a `figure`,
    #: not a `person`, so he cannot sit in `rulers`.
    legendaryRulers: List[str] = Field(
        default_factory=list,
        description="Rulers known only from legend or myth, by common name; empty if none.")
    predecessors: List[str] = Field(
        description="States it replaced in its heartland, by common name. Not provinces: where a "
                    "province or an interregnum intervened, name the state that held the heartland.")
    successors: List[str] = Field(
        description="States that replaced it in its heartland, by common name. Not provinces: where a "
                    "province or an interregnum intervened, name the state that held the heartland "
                    "(e.g. the Ptolemaic Kingdom's successor is the Roman Empire).")
    territory: str = Field(description="Its extent at its height and how that changed, in a few sentences.")
    modernCountries: List[str] = Field(
        description="Every present-day country in which it held territory at any time in its history.")
    languages: List[str] = Field(description="Languages of government, religion and everyday life.")
    religions: List[str] = Field(description="Religions and state cults practised in it.")
    judgments: Judgments
    significance: str = Field(description="Significance of this state")
    legacy: str = Field(description="Legacy of this state to modernity")
    shortSummary: str = Field(description="Short summary information")
    longSummary: List[Optional[str]] = Field(description="Long summary that expands upon the short summary")

    @field_validator("typeOfGovernment", mode="before")
    @classmethod
    def coerce_to_other(cls, v) -> str:
        """The city periods' rule, so a polity and its cities' periods share
        one vocabulary; case and whitespace are folded first."""
        return coerce(v, government_enum, "other")

    @model_validator(mode="after")
    def _bce_years_are_negative(self):
        """A state that began and ended BCE has no CE years in its sub-lists.

        The `Date` fields carry an unsigned year and an era; the sub-lists
        carry signed years. One atom holding both conventions invites the
        slip of writing `reignStart: 305` for 305 BCE, which would read as
        305 CE. Where both ends of the state are BCE, a positive year cannot
        be right, so it is negated rather than costing the whole atom.
        """
        if self.dateFounded.era == "BCE" and self.dateEnded.era == "BCE":
            for capital in self.capitals:
                capital.fromYear = _negative(capital.fromYear)
                capital.toYear = _negative(capital.toYear)
            for ruler in self.rulers:
                ruler.reignStart = _negative(ruler.reignStart)
                ruler.reignEnd = _negative(ruler.reignEnd)
        return self


def _negative(year: Optional[int]) -> Optional[int]:
    return -year if year is not None and year > 0 else year
