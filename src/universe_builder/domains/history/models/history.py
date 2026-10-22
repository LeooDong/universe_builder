"""The seed atom: what a civilisation contains, as lists of bare topic names.

Every other pass reads these lists, and **each entry becomes a content-addressed
id**. That makes the exact string load-bearing in a way it is not anywhere else
in the schema: `"Memphis"` and `"Memphis — ancient capital near modern Mit
Rahina"` are different subjects, hash to different ids, and produce two files for
one city.

So this model does two things a bare `List[str]` cannot. The field descriptions
reach the model — they survive `Provider.strict_json_schema` into the
`response_format`, so they are a real constraint and not just prompt text — and a
before-validator normalises whatever comes back anyway. The description steers;
the validator guarantees.
"""

from __future__ import annotations

import logging
import re
from typing import List

from pydantic import BaseModel, Field, field_validator

log = logging.getLogger(__name__)

#: A trailing gloss: " — ancient capital…", " – founded in 969", " - the capital".
#: Only dashes *surrounded by whitespace* count. A bare hyphen is part of the
#: name far more often than it is punctuation — `Pi-Ramesses`, `Tell el-Amarna`,
#: `Deir el-Medina` — and splitting on it would corrupt the very names it is
#: meant to clean.
_TRAILING_GLOSS = re.compile(r"\s+[—–-]\s+.*$", re.DOTALL)

#: The instruction the model actually sees, appended to each field description.
#: Stated as a prohibition with an example, because "the name only" alone was
#: read as latitude: a frontier model returned every city with a gloss.
_BARE_NAME = (
    "Each entry must be the bare name only, with no description, gloss or "
    "explanatory clause. Write \"Memphis\", never \"Memphis - ancient capital\"."
)
#: The five original lists keep the parenthesis the corpus has always used for
#: a second name (`Thebes (Luxor)`); the alias machinery reads it as one.
_BARE = f"{_BARE_NAME} An alternate name in parentheses is allowed, as in \"Thebes (Luxor)\"."
#: The four later lists take no parenthesis at all. Their subjects
#: collide across civilisations by bare title far more than cities do (two
#: "Histories", two "Temples of Artemis"), and the cure is a name that is its
#: own disambiguation, not a parenthetical the resolver would read as a second
#: name for the same thing.
_BARE_STRICT = (
    f"{_BARE_NAME} Never add a parenthesis: where two subjects share a name, tell them apart "
    "inside the name itself, as the rule for this list says."
)


def strip_gloss(value):
    """The gloss rule `History` applies to every list: each string entry
    reduced to its bare name and blanks dropped. Anything not a list
    passes through for the type check to refuse."""
    if not isinstance(value, list):
        return value

    cleaned, glossed = [], 0
    for entry in value:
        if not isinstance(entry, str):
            cleaned.append(entry)
            continue
        bare = _TRAILING_GLOSS.sub("", entry).strip()
        if bare != entry.strip():
            glossed += 1
        if bare:
            cleaned.append(bare)

    if glossed:
        log.info("Stripped a trailing gloss from %d of %d entries", glossed, len(value))
    return cleaned


class History(BaseModel):
    #: Narrowed: legendary people moved to `figures`, so Romulus is
    #: no longer a person with invented birth and death dates.
    people: List[str] = Field(..., description=(
        "Names of real people associated with the prompt, including those whose existence is "
        "doubted (Homer, Laozi, Lycurgus). Gods, heroes and legendary figures are figures, not "
        f"people. {_BARE}"))
    #: Narrowed: "Discovery of X" atoms repeated an artifact's
    #: own discovery fields.
    discoveries: List[str] = Field(..., description=(
        "Names of inventions, techniques and ideas (the 365-day calendar, papyrus, mummification). "
        "The finding of a thing is not a discovery: a find belongs to its artifact or monument, as "
        f"\"Rosetta Stone\", never \"Discovery of the Rosetta Stone\". {_BARE}"))
    events: List[str] = Field(..., description=f"Names of events. {_BARE}")
    cities: List[str] = Field(
        ..., description=f"Names of cities, across all periods. {_BARE}")
    #: "Artifacts" alone was read two ways. Earlier Histories filed
    #: buildings and monuments here (Egyptian, Ottoman); the
    #: Chinese seed read it as movable objects and texts, and left out
    #: the Great Wall, the Forbidden City and the Temple of Heaven. Now
    #: buildings and sites are `monuments` and texts are `works`, so
    #: an artifact is a movable made object only.
    artifacts: List[str] = Field(
        ..., description=("Comprehensive names of artifacts across all periods: movable made objects, "
                          "however large (statues, colossi, obelisks, stelae, vessels, jewellery, "
                          "coins, tools, weapons, coffins, boats, paintings, and individual "
                          "manuscripts or papyri). Buildings, structures and sites fixed in place "
                          "are monuments, not artifacts; texts and compositions are works; a "
                          f"painting is an artifact. {_BARE}"))
    #: The four later lists (polity, monument, work, figure). A
    #: `default_factory` on each, because a required field made every stored History fail to load
    #: in simulation: the Histories on disk list only the five lists
    #: above. Generation is unaffected, since the strict schema requires every
    #: property whatever its default.
    polities: List[str] = Field(default_factory=list, description=(
        "Names of the states that ruled here across all periods: empires, kingdoms, caliphates, "
        "sultanates, republics and confederations, each a state with its own ruler. A dynasty is "
        "listed where history names the state by it (Ptolemaic Kingdom, Safavid Iran); a state "
        "several dynasties ruled is one entry (Byzantine Empire); Egypt's are the Old, Middle and "
        "New Kingdom and the Late Period, not numbered dynasties. A province under an appointed "
        "governor (Roman Egypt) is not a state; a vassal with its own ruling house is (Khedivate "
        "of Egypt). A city-state is listed only if it ruled beyond itself. Give each state's full "
        f"name, never a city's name or a period label: \"Ottoman Empire\", not \"Ottoman rule\". "
        f"{_BARE_STRICT}"))
    monuments: List[str] = Field(default_factory=list, description=(
        "Names of buildings, structures and sites fixed in place, or that people could enter, across "
        "all periods: temples, pyramids, palaces, tombs, mosques, churches, walls, bridges, canals, "
        "cave temples, necropolises. A relocated building is still a monument (Abu Simbel). A "
        "freestanding made object, however large (an obelisk, a colossus, a stele), is an artifact; "
        "a place that was a settlement is a city. Where two monuments share a name, add the place "
        f"with 'at' or 'of': \"Temple of Artemis at Ephesus\". {_BARE_STRICT}"))
    works: List[str] = Field(default_factory=list, description=(
        "Names of texts and compositions across all periods: histories, epics, poems, scriptures, "
        "law codes, treatises, letters, hymns and stories, and myths, including a myth with no single "
        "text (\"Osiris myth\"). The text is a work; one famous physical copy of it (the Papyrus of "
        "Ani) is an artifact. Where two works share a title, add the author with 'of': "
        f"\"Histories of Herodotus\". {_BARE_STRICT}"))
    figures: List[str] = Field(default_factory=list, description=(
        "Names of gods, creatures and spirits, and of heroes, founders and kings known only from myth "
        "or legend (Osiris, Rostam, Romulus, Jamshid). An author, sage, lawgiver or ruler whom "
        "tradition treats as a human being is a person, not a figure, even where their existence is "
        f"doubted (Homer, Laozi). {_BARE_STRICT}"))

    @field_validator("people", "discoveries", "events", "cities", "artifacts",
                     "polities", "monuments", "works", "figures", mode="before")
    @classmethod
    def strip_trailing_gloss(cls, value):
        """Reduce each entry to the bare name, and drop blanks.

        Normalising rather than rejecting is deliberate. A refusal would cost the
        whole atom — several hundred topics and a frontier call — over a
        formatting habit, and the gloss is not information worth keeping here:
        the atom generated *for* that topic carries the long-form detail. What
        matters is that the name is stable, because the id is hashed from it.
        """
        return strip_gloss(value)


#: The History lists, in the model's order.
LISTS = tuple(History.model_fields)
