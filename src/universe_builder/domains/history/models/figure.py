"""A god, hero or legendary person: someone known from myth or legend.

Registered in the pack (`domains/history/__init__.py`).

Until now a legendary person was stored as a `person`, with birth and death
dates that present legend as biography (Romulus, Horatius Cocles, Cloelia),
and the gods had no atom at all: the corpus reached Osiris and Zeus only
through temples and statues named after them. The stories themselves are
`work` atoms, including traditional ones with no single text (the Osiris
myth), which is why a figure links to works rather than holding its myths.

Figure or person:

- An author, sage, lawgiver or ruler whom the tradition treats as a human
  being is a `person`, with low date certainty, even where their existence is
  doubted (Homer, Laozi, Lycurgus).
- A `figure` is a god, creature or spirit, or a hero or founder whose story is
  supernatural or purely literary (Rostam, Heracles, Romulus).
- Legendary rulers in king lists (Jamshid, the Yellow Emperor) are figures,
  and a polity names them in `legendaryRulers`.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

from universe_builder.domains.history.models._enum import coerce
from universe_builder.models.subatomic import Date, Name

kind_enum = ["deity", "hero", "legendary person", "spirit or creature", "other"]

#: How far the figure is history. A closed list: the app shows it, and it is
#: what separates Romulus (legendary) from Augustus (a `person`).
historicity_enum = ["mythical", "legendary", "possibly historical"]


class Figure(BaseModel):
    name: Name
    kind: str = Field(json_schema_extra={"enum": kind_enum})
    historicity: str = Field(
        json_schema_extra={"enum": historicity_enum},
        description="'mythical' for a god or a creature, 'legendary' for a hero or ruler known only from "
                    "legend, 'possibly historical' where a real person may lie behind the legend. An author, "
                    "sage, lawgiver or ruler the tradition treats as human is a person atom, not a figure.")
    cultures: List[str] = Field(description="The cultures or religions that told of this figure.")
    roles: List[str] = Field(description="What the figure is god, patron or hero of, e.g. 'god of the dead'.")
    attributes: List[str] = Field(description="Symbols, iconography and possessions by which it is recognised.")
    parents: List[str] = Field(description="Other figures, by common name.")
    siblings: List[str] = Field(description="Other figures, by common name.")
    consorts: List[str] = Field(description="Other figures, by common name.")
    children: List[str] = Field(description="Other figures, by common name.")
    identifiedWith: List[str] = Field(
        description="Figures of other cultures it was equated with (Osiris with Dionysus), by common name.")
    #: `places`, not the first draft's `cultCentres`: a hero has no cult, and
    #: "where it is set" mapped Rostam's legendary Zabol onto the modern
    #: Iranian town of that name.
    places: List[str] = Field(
        description="Real cities where it was worshipped or where its story is set, by city name; not "
                    "regions; empty if none.")
    temples: List[str] = Field(
        description="Monuments dedicated to it or named after it, by the monument's full name.")
    appearsIn: List[str] = Field(
        description="Works that tell its story, by the work's full name (e.g. 'Shahnameh').")
    earliestAttestation: Date = Field(description="The earliest surviving record of the figure.")
    significance: str = Field(description="Significance of this figure")
    legacy: str = Field(description="Legacy of this figure to modernity")
    shortSummary: str = Field(description="Short summary information")
    longSummary: List[Optional[str]] = Field(description="Long summary that expands upon the short summary")

    @field_validator("kind", mode="before")
    @classmethod
    def coerce_kind(cls, v) -> str:
        return coerce(v, kind_enum, "other")

    @field_validator("historicity", mode="before")
    @classmethod
    def coerce_historicity(cls, v) -> str:
        """An unknown value falls to 'legendary', the middle of the scale,
        rather than to a claim of myth or of history; a known one in the
        wrong case ('Mythical') is folded first, not treated as unknown."""
        return coerce(v, historicity_enum, "legendary")
