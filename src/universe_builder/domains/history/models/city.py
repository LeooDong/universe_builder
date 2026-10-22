from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Union
from universe_builder.models.subatomic import Certainty, Location, Date, Judgments, Name

from universe_builder.domains.history.models._enum import coerce


government_enum = [
    "monarchy",
    "autocracy",
    "classical republic",
    "merchant republic",
    # A modern representative republic. Added because the enum offered only
    # classical/merchant republics and "democracy", so post-1953 Egypt could be
    # recorded only as "autocracy" or "other" — and every modern city hits the
    # same wall as the corpus grows. Distinct from "democracy" (the direct,
    # assembly form) and from the two historical republic flavours.
    "modern republic",
    "democracy",
    "oligarchy",
    # A league or tribal/nomadic confederation (the Aq Qoyunlu, the Xiongnu, the
    # Delian League). Added for the `polity` type, which shares
    # this list; harmless for city periods, which may use it too.
    "confederation",
    "theocracy",
    "communism",
    "fascism",
    "non-governmental",
    "other",
]


class MonthlyClimate(BaseModel):
    monthlyAverageTempCelsius: float
    monthlyAveragePrecipMillimetres: float
    monthlyAverageHumidityPercent: float


class GeographyAndClimate(BaseModel):
    elevationMetres: float
    january: MonthlyClimate
    february: MonthlyClimate
    march: MonthlyClimate
    april: MonthlyClimate
    may: MonthlyClimate
    june: MonthlyClimate
    july: MonthlyClimate
    august: MonthlyClimate
    september: MonthlyClimate
    october: MonthlyClimate
    november: MonthlyClimate
    december: MonthlyClimate


class CityDate(Date):
    cityPopulationRange: List[Optional[int]]
    averageCityPopulation: Optional[int] = Field(None)


class Population(BaseModel):
    # These descriptions are deliberately PER-OBJECT: an expand call's schema
    # is a single Population, so an instruction to split a span into several
    # periods here would order what that response format forbids (the review
    # gate caught exactly that wording before it shipped). The split rule
    # lives on the period *list* fields instead — the seed
    # `City.cityPeriodsOfHistory`, where create authors the names, and
    # `CityExpanded.cityPeriodsOfHistory`, where verify/fix can restructure —
    # and those descriptions ride the schema on every path, so the rule still
    # cannot drift between prompt copies.
    periodName: str = Field(
        description="Named span of the city's history, covering a single type of government.",
    )
    periodStart: CityDate
    periodEnd: Date
    # it looks like GPT 4 is much better than 3.5 for this purpose.
    certainty: Certainty = Field(description="Certainty level")
    typeOfGovernment: str = Field(
        # `json_schema_extra`, not a bare `enum=` kwarg (deprecated in Pydantic 2,
        # removed in 3). This is schema *documentation* — it lists the allowed
        # governments in the schema the model sees, steering the answer — while
        # `coerce_to_other` below is what actually enforces the value.
        json_schema_extra={"enum": government_enum},
        description="Type of government across this whole period.",
    )
    #: The period's link to its `polity` atom, declared in the pack as
    #: `cityPeriodsOfHistory[].polity`. Optional with a null
    #: default: stored periods predate it, and it is filled later from a
    #: reviewed table of their period names rather than by a migration.
    polity: Optional[str] = Field(
        None,
        description="The state the city belonged to in this period, by its full name as its polity "
                    "atom is named, e.g. 'Ottoman Empire', not 'Ottoman rule'; null if none or unknown.",
    )
    tradeAndCommerce: List[Optional[str]] = Field(description="Notable imports or exports")
    tradingPartners: List[Optional[str]] = Field(description="Other civilisations that were traded with")
    languagesSpoken: List[str]

    @field_validator("typeOfGovernment", mode="before")
    @classmethod
    def coerce_to_other(cls, v) -> Optional[str]:
        """An unlisted government is 'other'; a listed one in the wrong case
        or with stray whitespace ("Monarchy ") is folded first, as every
        closed list in the pack is (`_enum.coerce`)."""
        return coerce(v, government_enum, "other")


class PeriodSplit(BaseModel):
    """The answer shape of the period-split micro-operation.

    Expand cannot split a period — its schema is a single `Population` — so
    the one-period-in, several-periods-out job needs its own response shape.
    `universe_builder periods --split` sends one over-broad period with the regime-change
    dates it crosses; the reply's `periods` replace the original in
    `cityPeriodsOfHistory`, spliced mechanically. The wrapper exists so the
    model can return N periods without any list-shaped ambiguity about what
    the rest of the city document should look like.
    """

    periods: List[Population] = Field(
        description=(
            "Consecutive periods replacing the single over-broad period you "
            "were given, split at each change of government, in chronological "
            "order, jointly covering exactly the original span."
        ),
    )


class City(BaseModel):
    name: Name
    foundingDate: Date
    dateOfDemise: Date
    location: Location
    # The split rule exists because 47 stored periods ("Modern Era" and kin)
    # spanned several regimes and could carry no single government value.
    cityPeriodsOfHistory: List[str] = Field(
        description=(
            "Consecutive periods of history within the city. No period may "
            "span a change of government: where an era crosses one (e.g. a "
            "'Modern Era' running through both a monarchy and a republic), "
            "list it as separate consecutive periods split at the change."
        ),
    )
    geographyAndClimate: GeographyAndClimate
    judgments: Judgments
    architecturalHighlights: List[Optional[str]] = Field(
        description="Comprehensive list of highlights of architecture within the city (past and present)",
    )
    notableFigures: List[Optional[str]] = Field(description="Associated with this city")
    popCulturalReferences: List[str] = Field(description="modern pop cultural references of this city")


class CityExpanded(City):
    """A `City` after the expand pass has run over `cityPeriodsOfHistory`.

    The create pass emits a `City`, whose `cityPeriodsOfHistory` is a list of
    period *names*. The expand pass replaces each name with a `Population` object
    describing that period. Those are different shapes, so they are different
    models — writing `Population` objects into a `City` produces a file that
    serialises with a warning and can never be read back (a bug that once
    silently affected every generated city).

    The member type is a union rather than plain `List[Population]` because
    expansion is per-entry and best-effort: a period whose call failed stays a
    string, and the file must still round-trip. A list with any string left in it
    is a partially expanded city.
    """

    cityPeriodsOfHistory: List[Union[Population, str]] = Field(
        description="Periods of history within the city, expanded into Population objects. "
                    "Entries left as plain strings are periods that have not been expanded yet. "
                    "No period may span a change of government: where an era crosses one, "
                    "split it into consecutive periods at the change, each with its own "
                    "dates and government.",
    )

    def is_fully_expanded(self) -> bool:
        return all(isinstance(entry, Population) for entry in self.cityPeriodsOfHistory)

    def unexpanded_periods(self) -> List[str]:
        return [entry for entry in self.cityPeriodsOfHistory if isinstance(entry, str)]


class CityAppend(BaseModel):
    popCulturalReferences: List[str] = Field(description="modern pop cultural references of this city")
