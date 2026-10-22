from pydantic import BaseModel, Field
from typing import List, Optional

from universe_builder.models.subatomic import Location, Date, Name


class Start(BaseModel):
    date: Date
    location: Location


class End(BaseModel):
    date: Date
    location: Location


class Event(BaseModel):
    name: Name = Field(description="Information associated with the name of this event")
    eventStart: Start
    eventEnd: End
    significance: str = Field(description="Significance of this event")
    legacy: str = Field(description="Legacy of this event to modernity")
    shortSummary: str = Field(description="Short summary information")
    longSummary: List[Optional[str]] = Field(description="Long summary that expands upon the short summary")
