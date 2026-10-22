"""The history domain pack.

Importing this module registers the pack under "history", so
`registry.active_pack()` discovers it by name — the same way the medications
pack is discovered. History is the built-in default (`registry` defaults
`_active_name` to "history"), but its *definition* lives here, not in core, so
core carries no history-specific atom shapes, list fields or governments.

This package also holds history's models (`models/`) and its `periods`
operation (`periods.py`). The shared primitives — `Name`, `Date`, `Certainty`,
`WireModel`, … — stay in `universe_builder.models.subatomic`, because every domain builds
on them.
"""

from __future__ import annotations

from pathlib import Path

from universe_builder import registry

#: History's one-shot examples ship beside the pack, so
#: every domain's examples live in its own `examples/` dir — no domain's data in
#: core `prompts/`. The request builder and sync loader read `pack.examples_dir`.
_HERE = Path(__file__).resolve().parent
EXAMPLES_DIR = _HERE / "examples"
#: History's own name-identity file and civilisation seed list — domain data,
#: beside the pack rather than in core `prompts/lists/`.
ALIASES_PATH = _HERE / "aliases.txt"
HISTORY_LIST_PATH = _HERE / "HISTORY_list.txt"
#: The historian persona. It was once the *core* default, which
#: meant a pack that forgot to set `base_system` spoke as a historian; now the
#: core default is domain-free and history declares its own like any pack.
BASE_SYSTEM_PATH = _HERE / "base_system.txt"

#: Words too common in historical names to suggest two entries are one thing
#: (`topics.near_matches`), or every "Temple of …" would match every other.
#: The state and building words joined with the new types: with polities and
#: monuments listed, "Empire" and "Palace" would otherwise pair every state
#: and every palace. `pyramid` stays out: there the building word is often
#: the only one two spellings share (`Great Pyramid` / `Pyramid of Khufu`).
NEAR_MATCH_STOPWORDS = frozenset("""
the of and in at on for a an great old new upper lower ancient temple city
tomb king queen pharaoh dynasty period era first second third
empire kingdom republic sultanate caliphate khanate palace mosque church
""".split())

#: Hosts worth citing, best first (`universe_builder sources`). Not a whitelist.
PREFERRED_HOSTS = ("wikipedia.org", "britannica.com", "metmuseum.org",
                   "britishmuseum.org", "egymonuments.gov.eg", "worldhistory.org",
                   "doi.org", "jstor.org", "ucl.ac.uk", "penn.museum")


def _history_periods(args):
    """Dispatch target for the history `periods` operation. Imports the handler
    lazily — `domains.history.periods` imports `topics`/`atoms`, so importing it
    at pack-build time would be a cycle. Called only when `universe_builder periods` runs."""
    from universe_builder.domains.history.periods import run_periods

    return run_periods(args)


def _periods_arguments(parser):
    """The `periods` verb's flags. Declared here, not in core `cli.py`:
    the CLI builds a subparser per `DomainPack.operations` entry
    and knows none of them by name."""
    parser.add_argument("--apply-labels", action="store_true",
                        help="write the majority government to minority cities")
    parser.add_argument("--split", action="store_true",
                        help="split periods that cross a regime change (model calls)")
    parser.add_argument("--model", help="model or tier for --split "
                                        "(default: the fix phase's)")


PERIODS = registry.OperationSpec(
    handler=_history_periods,
    help="lint city periods against the regime table and cross-city agreement; "
         "--apply-labels adjudicates by majority, free; --split sends flagged "
         "periods for splitting, paid",
    add_arguments=_periods_arguments,
)


def build() -> registry.DomainPack:
    # Models imported inside build() so merely registering the pack (importing
    # this module) does not pull them in until the pack is actually built.
    from universe_builder.domains.history.models.artifact import Artifact
    from universe_builder.domains.history.models.city import City, CityAppend, CityExpanded, Population
    from universe_builder.domains.history.models.discovery import Discovery
    from universe_builder.domains.history.models.event import Event
    from universe_builder.domains.history.models.figure import Figure
    from universe_builder.domains.history.models.history import History
    from universe_builder.domains.history.models.monument import Monument
    from universe_builder.domains.history.models.person import Person
    from universe_builder.domains.history.models.polity import Polity
    from universe_builder.domains.history.models.work import Work
    from universe_builder.domains.history import wikipedia

    def _city_is_expanded(d: dict) -> bool:
        return any(isinstance(x, dict) for x in d.get("cityPeriodsOfHistory") or [])

    city = registry.AtomSpec(
        name="city",
        model=City,
        plural="cities",
        prompt_start="Provide a response for the city commonly known as ",
        # Both are SEED-shape `City` (periods as bare strings), not the
        # post-expand `CityExpanded` every stored city is — which is why the
        # second had to be authored rather than promoted from the corpus, as
        # the other atoms' second examples were. Memphis is a single-culture
        # river capital; Sardis is a plateau city that changed hands six times
        # and carries its name in five languages, which is what `otherNames`
        # is for.
        example=["memphis.json", "sardis.json"],
        stages=((_city_is_expanded, CityExpanded),),
        expand=registry.ExpandSpec(
            list_field="cityPeriodsOfHistory",
            entry_model=Population,
            expanded_model=CityExpanded,
            prompt_start="Provide a response for this to this json schema about ",
            example="roman_period_alexandria.json",
        ),
        append=registry.AppendSpec(
            model=CityAppend,
            prompt_start="Please append information to this json schema about ",
            example=None,
        ),
    )

    return registry.DomainPack(
        name="history",
        index=registry.AtomSpec(
            name="history",
            model=History,
            # The index lists no topics *about itself*, so it has no plural field
            # on History. Consumers exclude the index by name (index.py) or guard
            # on a falsy plural (topics.py); None states that honestly rather than
            # naming a `histories` field that does not exist.
            plural=None,
            prompt_start="Give me an exhaustive but finite lists for topics under the heading: ",
            example="egyptian_history.json",
        ),
        atoms=(
            registry.AtomSpec(
                name="person", model=Person, plural="people",
                prompt_start="Provide a response for the person commonly known as ",
                # Few-shot: two examples per atom sit in the cached prefix, so the
                # second costs ~10% of an uncached call and pushes the prefix past
                # the Haiku cache minimum. The order is fixed (varying it throws the
                # cache away). Both must validate against the seed model — pinned by
                # `test_examples.py`.
                example=["genghis_khan.json", "ptahhotep.json"],
            ),
            registry.AtomSpec(
                name="event", model=Event, plural="events",
                prompt_start="Provide a response for the event commonly known as ",
                example=["construction_of_the_great_pyramid_of_giza.json",
                         "founding_of_cairo.json"],
            ),
            registry.AtomSpec(
                name="discovery", model=Discovery, plural="discoveries",
                prompt_start="Provide a response for the discovery or invention commonly known as ",
                example=["invention_of_toothpaste.json", "hieroglyphic_writing.json"],
            ),
            city,
            registry.AtomSpec(
                name="artifact", model=Artifact, plural="artifacts",
                prompt_start="Provide a response for the artifact commonly known as ",
                # Paired for contrast. `raphia_decree.json` was neither: it was
                # a second Ptolemaic trilingual decree stele, and only 31% of
                # its fields were populated, so it taught the object class the
                # Rosetta Stone already taught and taught omission besides.
                # Antikythera is a machine, from a wreck, fragmentary, with
                # live scholarly dispute — 86% filled.
                example=["rosetta_stone.json", "antikythera_mechanism.json"],
            ),
            # The four later types: states, and the three things
            # `artifact` and `person` held that were not movable objects or
            # real people. Each pair of examples is chosen for contrast, as the
            # others are.
            registry.AtomSpec(
                name="polity", model=Polity, plural="polities",
                prompt_start="Provide a response for the state commonly known as ",
                # A BCE Hellenistic kingdom, whose sub-list years are negative,
                # and a CE Persian state with its eleven shahs.
                example=["ptolemaic_kingdom.json", "safavid_iran.json"],
            ),
            registry.AtomSpec(
                name="monument", model=Monument, plural="monuments",
                prompt_start="Provide a response for the monument commonly known as ",
                # A church become mosque become museum, with named architects;
                # and a temple whose inscription names someone who did not build it.
                example=["hagia_sophia.json", "pantheon_of_rome.json"],
            ),
            registry.AtomSpec(
                name="work", model=Work, plural="works",
                prompt_start="Provide a response for the work commonly known as ",
                # An anonymous text that grew over centuries; one poet's epic.
                example=["book_of_the_dead.json", "shahnameh.json"],
            ),
            registry.AtomSpec(
                name="figure", model=Figure, plural="figures",
                prompt_start="Provide a response for the figure of myth or legend commonly known as ",
                # A god with a family, cult places and texts; a literary hero.
                example=["osiris.json", "rostam.json"],
            ),
        ),
        # Wire paths, not bare fields — see `DomainPack.link_fields`. The four
        # `person` relations were declared flat (`parents`, `children`, …) while
        # the model has held them under `family` since it was written, so they
        # produced zero edges from thousands of stored mentions. `validate_link_paths`
        # now rejects that at registration; these are the corrected paths.
        #
        # `location.city` is the densest relation in the corpus and had never
        # been declared at all: thousands of mentions across these nine paths,
        # most of which name a city that already has an atom. A person's
        # birthplace, an event's setting and the museum an artifact sits in are
        # all the same kind of edge, and all of them were being discarded.
        link_fields={
            "city": {
                "notableFigures": "person",
                "location.city": "city",
                # Optional: null on every period stored before polities existed.
                "cityPeriodsOfHistory[].polity": "polity",
            },
            "discovery": {
                "discoverorOrInventor": "person",
                "earliestPhysicalRecord.location.city": "city",
                "earliestWrittenRecord.location.city": "city",
            },
            "artifact": {
                "finderOrDiscoveror": "person",
                "originalCreator": "person",
                "discoveryLocation.city": "city",
                "currentLocation.city": "city",
            },
            "person": {
                "family.parents": "person",
                "family.children": "person",
                "family.spouses": "person",
                "family.otherRelatives": "person",
                "birth.location.city": "city",
                "death.location.city": "city",
            },
            "event": {
                "eventStart.location.city": "city",
                "eventEnd.location.city": "city",
            },
            # One target type per path, which is why a monument has `patrons`
            # (people) beside `commissioningPolity`, and a polity
            # `legendaryRulers` (figures) beside `rulers[].name` (people).
            "polity": {
                "capitals[].city": "city",
                "rulers[].name": "person",
                "legendaryRulers": "figure",
                "predecessors": "polity",
                "successors": "polity",
                "location.city": "city",
            },
            "monument": {
                "patrons": "person",
                "commissioningPolity": "polity",
                "architects": "person",
                "rediscovery.by": "person",
                "location.city": "city",
            },
            "work": {
                "authors": "person",
            },
            "figure": {
                "parents": "figure",
                "siblings": "figure",
                "consorts": "figure",
                "children": "figure",
                "identifiedWith": "figure",
                "places": "city",
                "temples": "monument",
                "appearsIn": "work",
            },
        },
        # Where each atom keeps the place and date that identify its subject.
        # `lint.corroboration` compares two atoms along these, field to
        # matching field; `sources.context_terms` reads country/state off the
        # first coordinate path. Wire paths, validated at registration.
        coordinate_paths={
            "city": ("location",),
            "person": ("birth.location", "death.location"),
            "artifact": ("discoveryLocation", "currentLocation"),
            "event": ("eventStart.location",),
            "discovery": ("earliestWrittenRecord.location",),
            "polity": ("location",),
            "monument": ("location",),
        },
        date_paths={
            "city": ("foundingDate",),
            "person": ("birth.date",),
            "artifact": ("dateCreated", "dateDiscovered"),
            "event": ("eventStart.date",),
            "discovery": ("earliestWrittenRecord.ageOfRecord",),
            # Work and figure have no place of their own: a text is copied
            # everywhere, and a god's cult places are a list, not an identity.
            "polity": ("dateFounded",),
            "monument": ("dateBuilt",),
            "work": ("dateComposed",),
            "figure": ("earliestAttestation",),
        },
        preferred_hosts=PREFERRED_HOSTS,
        url_resolvers={"wikipedia": wikipedia.wikipedia_lookup},
        identity_resolvers={"wikipedia": wikipedia.IDENTITY},
        near_match_stopwords=NEAR_MATCH_STOPWORDS,
        operations={"periods": PERIODS},
        examples_dir=EXAMPLES_DIR,
        base_system=BASE_SYSTEM_PATH.read_text(encoding="utf-8"),
        aliases_path=ALIASES_PATH,
        index_seed_path=HISTORY_LIST_PATH,
    )


registry.register_pack("history", build)
