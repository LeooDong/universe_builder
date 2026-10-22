from pydantic import Field
from typing import Optional
from universe_builder.models.subatomic import Name, Location, Date, WireModel


class Artifact(WireModel):
    name: Name
    dateCreated: Date
    discoveryLocation: Location
    dateDiscovered: Date
    currentLocation: Location
    #: "Discoveror" is a misspelling of "discoverer". The alias keeps the stored
    #: key and the generated JSON schema on the wrong spelling — changing either
    #: would invalidate the corpus and the one-shot example — while Python code
    #: reads `finderOrDiscoverer`. See `WireModel` for why both directions work.
    #: Once undescribed, and "unknown" came back two ways: null on
    #: many artifacts, a sentence of prose ("Unknown; the tablet was found at
    #: Tiriolo in 1640") on others. The field is a `person` link, so prose
    #: here is a name that never resolves.
    finderOrDiscoverer: Optional[str] = Field(
        alias="finderOrDiscoveror",
        description="The person or group credited with the modern find, excavation or identification "
                    "of this object: a name only, e.g. 'Howard Carter'. null when the object was never "
                    "lost (a standing building, continuously known), is a text transmitted in copies, "
                    "is a class of objects, or the finder is unrecorded; say which in "
                    "dateDiscovered.longFormInfo, never here.",
    )
    originalCreator: Optional[str]
    challenges: str = Field(description="Challenges that slowed discovery/invention")
    significance: str = Field(description="Significance of this event")
    controversies: Optional[str] = Field(
        description="Descriptions of any questions over ownership/authenticity/providence"
    )
    shortSummary: str = Field(description="Short summary information")
    longSummary: str = Field(description="Comprehensive summary of all information")
