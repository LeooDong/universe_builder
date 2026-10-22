"""A building, structure or site that was never a settlement.

Registered in the pack (`domains/history/__init__.py`).

Split out of `artifact`, whose fields describe a movable object (a find spot,
a finder, a current location): for a temple that never went missing those
were null or meaningless, which is where most of the 640 null
`finderOrDiscoveror` values came from. A monument is a temple, palace, tomb,
wall, bridge, canal, cave temple or a site such as a necropolis.

The boundaries:

- A building, structure or site fixed in place, or one people could enter, is
  a monument. A relocated building stays a monument, with `currentState`
  'relocated' (Abu Simbel, the Temple of Dendur).
- A freestanding made object, however large (an obelisk, a colossus, a stele,
  a statue), is an `artifact`.
- A place that was a settlement, ruined or not, is a `city`: Persepolis is a
  city, and its Apadana is a monument.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

from universe_builder.domains.history.models._enum import coerce
from universe_builder.models.subatomic import Date, Location, Name

#: What is left of it now. A closed list so the app can filter on it; each
#: value is defined in the field description.
state_enum = ["intact", "altered", "ruined", "rebuilt", "relocated", "destroyed", "unlocated", "other"]


class Rediscovery(BaseModel):
    """Where a monument's story includes its finding (Tutankhamun's tomb,
    Knossos): the discovery information an artifact carries, kept for the
    sites that move out of artifact and discovery."""

    date: Date
    by: List[str] = Field(description="Who found or first excavated it, by common name.")
    note: Optional[str]


class Monument(BaseModel):
    name: Name
    location: Location
    dateBuilt: Date = Field(
        description="When construction began; for a structure built in phases over centuries, the start "
                    "of the phase that gives it its present form, with the others in longFormInfo.")
    dateCompleted: Date = Field(description="When the main construction was finished.")
    #: Two fields, not one `builderOrPatron`: a declared link path has one
    #: target type (`registry.py`), and a patron is a person where the state
    #: is a polity. Named `patrons`, not `builders`: the first draft's
    #: "builders" was read as whoever's name is on the building, and credited
    #: Agrippa with Hadrian's Pantheon.
    patrons: List[str] = Field(
        description="The people who commissioned or paid for this structure (rulers, patrons), by common "
                    "name, as their person atom would be named; not the builder of an earlier building on "
                    "the site, nor a name that only appears in its inscription.")
    commissioningPolity: Optional[str] = Field(
        description="The polity in power where and when construction began, by its atom's full name "
                    "(e.g. 'Byzantine Empire'); null if none. The Roman Empire ends in 395; the "
                    "Byzantine Empire follows it in the east.")
    rediscovery: Optional["Rediscovery"] = Field(
        None, description="Only for a site that was lost and found again (a buried tomb, a ruined palace); "
                          "null for a building that was never lost.")
    architects: List[str] = Field(
        description="Named architects or master builders, by common name, as their person atom would be "
                    "named; an empty list when none is recorded.")
    purpose: str = Field(description="What it was built for, and what it has been used for since.")
    architecturalStyle: Optional[str]
    currentState: str = Field(
        json_schema_extra={"enum": state_enum},
        description="What survives today. 'intact': stands essentially as built, repairs aside. "
                    "'altered': stands but substantially changed in form or use (added minarets, a "
                    "conversion). 'ruined': partly standing. 'rebuilt': reconstructed after destruction. "
                    "'relocated': moved to a new site (Abu Simbel, the Temple of Dendur). 'destroyed': "
                    "little or nothing above ground. 'unlocated': known from texts, its site not "
                    "identified (the Hanging Gardens of Babylon). 'other': none of these.",
    )
    heritageStatus: Optional[str] = Field(
        description="Formal protection, e.g. 'UNESCO World Heritage Site (1985, as part of the "
                    "Historic Areas of Istanbul)'; null if none.")
    significance: str = Field(description="Significance of this monument")
    shortSummary: str = Field(description="Short summary information")
    longSummary: List[Optional[str]] = Field(description="Long summary that expands upon the short summary")

    @field_validator("currentState", mode="before")
    @classmethod
    def coerce_to_other(cls, v) -> str:
        return coerce(v, state_enum, "other")
