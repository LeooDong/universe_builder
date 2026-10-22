from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
from universe_builder.models.subatomic import Location, Date, Name, Judgments


class Reason(BaseModel):
    causeOfDeath: Optional[str]
    diseases: List[Optional[str]] = Field(..., description="Diseases this person had during their life")

    # Coercing Null value to a list of Null
    @field_validator("diseases", mode="before")
    @classmethod
    def null_array(cls, v) -> List[str]:
        if v is None:
            return [None]
        return v


class Birth(BaseModel):
    date: Date
    location: Location


class Death(BaseModel):
    date: Date
    location: Location
    reason: Reason


class Family(BaseModel):
    parents: List[Optional[str]]
    children: List[Optional[str]]
    spouses: List[Optional[str]]
    otherRelatives: List[Optional[str]]
    familyBackground: List[Optional[str]]
    sources: List[str]


class Person(BaseModel):
    name: Name
    birth: Birth
    death: Death
    judgments: Judgments
    family: Family
    occupationOrRole: str
    shortSummary: str
    longSummary: List[Optional[str]] = Field(..., description="Long summary that expands upon the short summary")
    curiousFact: List[Optional[str]] = Field(..., description="Fun or interesting fact about this person")
    achievements: List[Optional[str]]
    legacy: List[Optional[str]]
    referencesInPopCulture: List[Optional[str]]
