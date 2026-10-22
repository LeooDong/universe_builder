import logging
import re

from pydantic import BeforeValidator, BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import List, Literal, Optional
from typing_extensions import Annotated

log = logging.getLogger(__name__)

MONTH_MAPPING = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


class EmptyModel(BaseModel):
    # ConfigDict, not a class-based `Config`: the latter is deprecated in
    # Pydantic 2 and removed in 3. `extra="forbid"` is what makes this a real
    # sentinel — a payload with any field other than `error` is not an EmptyModel.
    model_config = ConfigDict(extra="forbid")
    error: str = "No data found"


class WireModel(BaseModel):
    """Base for atoms whose stored key names differ from their Python attributes.

    The corpus on disk was written with whatever key names the models carried at
    the time, misspellings included (`finderOrDiscoveror`, `discoverorOrInventor`).
    Correcting the Python attribute would normally be a data migration: every
    stored file would stop validating, and every file rewritten afterwards would
    carry a key its neighbours do not. A Pydantic alias decouples the two — the
    attribute is spelled correctly, the wire key is left exactly as it is.

    Two config flags make that safe, and both are load-bearing:

    * `populate_by_name` accepts the corrected attribute name *as well as* the
      alias, so a payload written under either spelling still validates. Nothing
      that reads today stops reading.
    * `serialize_by_alias` makes both `model_dump` and `model_dump_json` emit the
      alias by default, so the first rewrite of an aliased atom keeps the wire key
      rather than silently changing it (which would put the misspelled and
      corrected spellings both live in the corpus at once). This was a pair of
      `model_dump*` overrides forcing `by_alias=True`, because Pydantic 2.6 had no
      such config; the config landed in 2.11 (this project floors at 2.13), so the
      overrides are gone. A call passing `by_alias=False` explicitly still wins.

    Fields with no alias are unaffected: the alias falls back to the field name.
    """

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


#: Constrained value types.
#:
#: These were `str` with `Field(enum=[...])`, which is **schema decoration that
#: Pydantic does not enforce** — `Date(era="banana")` validated happily. Worse,
#: for an `Optional[str]` it emitted `enum` as a *sibling* of `anyOf` with `null`
#: absent from the enum, which is contradictory JSON Schema: the type says null
#: is allowed and the enum says only "CE" or "BCE" is. A `Literal` both validates
#: and emits the correct shape.
#:
#: Checked against the corpus before tightening: era is BCE/CE/None only,
#: certainty is within its five, typeOfDiscovery within its six. Nothing stored
#: is rejected by this.
def _lower(value):
    """Fold case before the Literal sees it.

    A model answering "High" means the same as "high", and rejecting it would
    cost the whole atom over a capital letter — the same judgement as stripping a
    trailing gloss from a History entry. The bundled `invention_of_toothpaste`
    example carried `"High"` for years precisely because nothing enforced the
    enum, so this case is observed rather than hypothetical.
    """
    return value.lower() if isinstance(value, str) else value


Era = Annotated[Literal["CE", "BCE"], BeforeValidator(lambda v: v.upper() if isinstance(v, str) else v)]
# A truth-confidence claim: how well the thing described is *known*, nothing
# more. It deliberately does NOT record who wrote the value or when — that is
# provenance, which is derived from the run journal (`universe_builder index`, the
# `provenance` section) and so never goes stale. Reading edit-history into
# `certainty` is what made it "stale" on mechanically-adjudicated fields; it
# answers only for itself.
Certainty = Annotated[Literal["none", "low", "medium", "high", "certain"], BeforeValidator(_lower)]


#: Words that make a trailing parenthetical a *description of the name* rather
#: than a second name. `Name._tidy_other_names` strips '(throne name)' and
#: '(Greek)' on this vocabulary and leaves everything else exactly as written.
#:
#: A stoplist, not a pattern, because the two readings are not distinguishable
#: by shape: 'Thebes (Luxor)' is one place under two names — `test_models.py`
#: pins that for History entries — and 'Maatkare (throne name)' is one name plus
#: an aside. Only the second kind can be recognised, so only the second kind is
#: touched; an unrecognised parenthetical survives. That asymmetry is the point.
#: Adding a word here deletes data, so add one only on an observed miss.
_NAME_DESCRIPTORS = frozenset("""
    akkadian ancient anglicised anglicized arabic aramaic birth coptic demotic
    egyptian english form german greek hebrew hieroglyphic horus latin literally
    modern mongolian name names nebty nomen old persian personal phoenician
    prenomen regnal roman sanskrit spelling sumerian throne title transliterated
    transliteration turkish variant
    a also alt alternate alternative an and as in its known of or the
""".split())


def _is_descriptor(text: str) -> bool:
    """True when every word inside a parenthetical is vocabulary, not content.

    'throne name' and 'Greek' qualify; 'Luxor' does not, and neither does
    'Greek Thebes' — one unknown word is enough to keep the whole entry, which
    is the safe direction when the cost of a wrong strip is a lost name.
    """
    words = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    return bool(words) and all(w.casefold() in _NAME_DESCRIPTORS for w in words)


class Name(BaseModel):
    common: str = Field(description="Common name")
    # The rule lives on the field, not in a prompt, for the reason
    # `city.Population` gives: a description rides the schema on every path —
    # create, fix, verify, expand, batch — so it cannot drift between prompt
    # copies. It is also the only instruction `otherNames` has ever had; the
    # previous text was "Other names, also known as, etc", under which the
    # bundled examples put "Ancient Memphis" (the common name plus a qualifier)
    # and "Tamerlane" (a different man) in this list. Both clauses below name a
    # failure that actually shipped.
    #
    # This field is read back by `topics.stored_name_claims`, so a wrong entry
    # here is not cosmetic: it is a claim the resolver may act on.
    #
    # The last two clauses are belt-and-braces, not the enforcement. Nano
    # creates produced 'Abydos' inside Abydos's own `otherNames`
    # and 'Maatkare (throne name)' for Hatshepsut; the clauses were added and
    # the two subjects re-run several times, and nano ignored them every
    # time. The text stays because it is correct and may bite on a larger model
    # — `person` creates run on balanced — but `_tidy_other_names` below is what
    # actually holds.
    otherNames: List[Optional[str]] = Field(description=(
        "Other attested names for this same subject: names in other languages, "
        "historical names, transliteration variants, and the modern name of a "
        "place. Names that identify the same subject only - not the common name "
        "with a qualifier attached ('Ancient X', 'X, Egypt'), and never a "
        "different subject with a similar name. Do not repeat the common name "
        "here, and write each name on its own, with no parenthetical gloss "
        "explaining what kind of name it is."
    ))
    sources: List[Optional[str]] = Field(description="Sources for name information")
    meaningOfNames: List[Optional[str]] = Field(description="Meaning or etymology of the word")
    longFormInfo: List[Optional[str]] = Field(description="Long-form information about the name")

    @model_validator(mode="after")
    def _tidy_other_names(self) -> "Name":
        """Drop the two `otherNames` faults a description cannot stop.

        Both are mechanical, so they are decided here rather than asked for:

        * an entry that merely repeats `common`, which claims nothing;
        * a trailing parenthetical that only *describes* the name —
          'Maatkare (throne name)', 'Abidos (Greek)' — judged by
          `_is_descriptor`, so 'Thebes (Luxor)' keeps its brackets.

        Duplicates go too, case-insensitively: 'Maatkare (throne name)' and
        "Ma'atkare" both appeared in one Hatshepsut atom, and the first
        collapses onto the second once its aside is gone.

        Coerced rather than rejected, on the judgement `_lower` already makes
        above: losing a whole atom over a bracketed aside is the worse trade.

        `topics.stored_name_claims` reads this field, so each entry removed here
        is a false claim the subject resolver never sees. The effect is on
        validation, which means a stored atom is tidied when it is next
        rewritten — nothing on disk changes until then.
        """
        common = (self.common or "").strip().casefold()
        cleaned: List[Optional[str]] = []
        seen = set()
        for entry in self.otherNames:
            if not isinstance(entry, str):
                cleaned.append(entry)
                continue
            name = entry.strip()
            match = re.search(r"\s*\(([^()]*)\)$", name)
            if match and _is_descriptor(match.group(1)):
                name = name[:match.start()].strip() or entry.strip()
            key = name.casefold()
            if not key or key == common or key in seen:
                continue
            seen.add(key)
            cleaned.append(name)
        self.otherNames = cleaned
        return self


class Date(BaseModel):
    year: Optional[int] = Field(None, ge=0, le=60000, description="Best approximation")
    month: Optional[int] = Field(None, ge=1, le=12)
    day: Optional[int] = Field(None, ge=1, le=31)
    era: Optional[Era] = Field(None, description="Common Era or Before Common Era")
    certainty: Certainty = Field(description="Certainty level")
    sources: List[str]
    longFormInfo: List[Optional[str]] = Field(description="Long-form information about the date")

    @model_validator(mode="before")
    @classmethod
    def _sign_is_an_era(cls, data):
        """A negative year with no era is BCE; the sign is about to be dropped.

        `abs_year` below stores the year unsigned and leaves the era to say
        which side of year one it falls, so for -500 with a null era the sign
        was the only record of BCE, and it was thrown away: the date read as
        500 with no era, which every consumer takes as CE or as undatable.
        Only a *missing* era is filled. -500 with era CE contradicts itself and
        which half is wrong is not knowable here, so the stated era stands (as
        it always has) and the conflict is logged rather than resolved by a
        guess. A scan found no negative year in any of the history corpus's
        stored dates nor in the raw model output in its journal, so nothing
        stored changes meaning.
        """
        if not isinstance(data, dict):
            return data
        year = data.get("year")
        if isinstance(year, str) and re.fullmatch(r"\s*-\d+\s*", year):
            year = int(year)
            data = {**data, "year": year}
        if isinstance(year, bool) or not isinstance(year, (int, float)) or year >= 0:
            return data
        era = data.get("era")
        if era is None or (isinstance(era, str) and era.strip().lower() in ("", "null")):
            return {**data, "era": "BCE"}
        if not (isinstance(era, str) and era.strip().upper() == "BCE"):
            log.warning("Date year %s contradicts its era %r; keeping the era as stated", year, era)
        return data

    # Coercing negative dates into positive
    @field_validator("year", mode="before")
    @classmethod
    def abs_year(cls, v) -> int | None:
        if v is not None:
            return abs(v)
        else:
            return None

    # Coercing "null" month to null and out of range months to null
    @field_validator("month", mode="before")
    @classmethod
    def cap_month(cls, v) -> Optional[int]:
        if v is None or v == "null":
            return None
        elif isinstance(v, str):
            return MONTH_MAPPING.get(v.lower(), None)
        elif v > 12 or v < 1:
            # if months are out of range
            return None
        return v

    # coercing out of bound dates and null dates to Null
    @field_validator("day", mode="before")
    @classmethod
    def cap_day(cls, v) -> Optional[int]:
        if v is None or v == "null":
            return None
        if v > 31 or v < 1:
            return None
        return v


class Location(BaseModel):
    landmark: Optional[str] = Field(None)
    city: Optional[str] = Field(None)
    state: Optional[str] = Field(None)
    country: Optional[str] = Field(None)
    latitude: Optional[float] = Field(None, description="Corresponds to the landmark, city, state, and/or country")
    longitude: Optional[float] = Field(None, description="Corresponds to the landmark, city, state, and/or country")
    certainty: Certainty = Field(description="Certainty level")
    sources: List[str] = Field(description="Sources for location information")


class Judgments(BaseModel):
    politicalInfluenceInfo: Optional[str] = Field(default="")
    politicalInfluenceScore: Annotated[int, Field(strict=True, ge=0, le=10)]
    culturalContributionInfo: Optional[str] = Field(default="")
    culturalContributionScore: Annotated[int, Field(strict=True, ge=0, le=10)]
    scientificAdvancementInfo: Optional[str] = Field(default="")
    scientificAdvancementScore: Annotated[int, Field(strict=True, ge=0, le=10)]
    ethicalLegacyInfo: Optional[str] = Field(default="")
    ethicalLegacyScore: Annotated[int, Field(strict=True, ge=0, le=10)]
    economicImpactInfo: Optional[str] = Field(default="")
    economicImpactScore: Annotated[int, Field(strict=True, ge=0, le=10)]
    militaryImpactInfo: Optional[str] = Field(default="")
    militaryImpactScore: Annotated[int, Field(strict=True, ge=0, le=10)]
    surpriseOfImpactInfo: Optional[str] = Field(default="")
    surpriseOfImpactScore: Annotated[int, Field(strict=True, ge=0, le=10)]
    sources: List[str] = Field(default=[])
