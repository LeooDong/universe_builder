from pydantic import BaseModel, Field
from typing import Literal, Optional
from universe_builder.models.subatomic import Name, Date, Location, WireModel


class Finding(BaseModel):
    name: str = Field(description="Name of finding")
    ageOfRecord: Date = Field(description="yyyy-MM-dd of the earliest date of record")
    dateOfModernDiscovery: Date = Field(description="yyyy-MM-dd of the discovery date")
    location: Location


class Discovery(WireModel):
    name: Name = Field(description="Information associated with the name of this event")
    earliestPhysicalRecord: Finding
    earliestWrittenRecord: Finding
    #: "discoveror" is a misspelling of "discoverer". The alias keeps the stored
    #: key and the generated JSON schema on the wrong spelling — changing either
    #: would invalidate the corpus and the one-shot example — while Python code
    #: reads `discovererOrInventor`. See `WireModel` for why both directions work.
    discovererOrInventor: Optional[str] = Field(alias="discoverorOrInventor")
    typeOfDiscovery: Literal[
        "spontaneous", "incremental", "systematic", "cultural", "imitative", "unknown"
    ] = Field(description="Discovery method")
    controversies: Optional[str] = Field(None, description="Any disputes")
    challenges: str = Field(..., description="Challenges that slowed discovery/invention")
    significance: str = Field(..., description="Significance of this event")
    shortSummary: str = Field(..., description="Short summary information")
    longSummary: str = Field(..., description="Long summary information")
