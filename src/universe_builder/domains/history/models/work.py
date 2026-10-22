"""A text or composition: a book, epic, treatise, law code, scripture, music.

Registered in the pack (`domains/history/__init__.py`).

A work exists in many copies or in none; an `artifact` is one physical thing
in one place. The Shahnameh is a work; the Great Mongol Shahnameh, one
illustrated manuscript of it, is an artifact. The Book of the Dead is a work;
the Papyrus of Ani, one copy of it, is an artifact. A myth with no single text
(the Osiris myth) is a work too, dated by its earliest record.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

from universe_builder.domains.history.models._enum import coerce
from universe_builder.models.subatomic import Date, Name

#: A closed list: free text fragmented into "history", "historiography" and
#: "History" in review. Anything else falls to "other".
genre_enum = [
    "history", "epic poem", "poem", "drama", "philosophy", "scripture", "myth", "legal code",
    "scientific treatise", "medical text", "novel", "letters or speeches", "geography or travel",
    "biography", "funerary text", "other",
]


class Work(BaseModel):
    name: Name
    authors: List[str] = Field(
        description="Authors or composers by common name, as their person atom would be named; an empty "
                    "list for an anonymous or traditional work. A legendary or divine author named by "
                    "tradition goes in authorshipNote, not here.")
    authorshipNote: Optional[str] = Field(
        description="Traditional, collective or disputed authorship, and earlier works the author drew on; "
                    "a legendary or divine author named by tradition goes here, not in authors.")
    dateComposed: Date = Field(
        description="When the work took its surviving form; for a text that grew over centuries or a myth "
                    "with no single text, the date of its earliest surviving record, with the span in "
                    "longFormInfo.")
    originalLanguage: str
    genre: str = Field(json_schema_extra={"enum": genre_enum})
    surviving: str = Field(
        description="How it survives: the oldest or most important manuscripts, inscriptions or "
                    "editions, or that it is lost and known only from quotation.")
    significance: str = Field(description="Significance of this work")
    legacy: str = Field(description="Legacy of this work to modernity")
    shortSummary: str = Field(description="Short summary information")
    longSummary: List[Optional[str]] = Field(description="Long summary that expands upon the short summary")

    @field_validator("genre", mode="before")
    @classmethod
    def coerce_genre(cls, v) -> str:
        return coerce(v, genre_enum, "other")
