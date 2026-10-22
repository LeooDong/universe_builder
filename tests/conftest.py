"""Shared fixtures for the offline test harness.

Nothing in this suite touches the network or requires an API key. Provider
clients are faked, and every filesystem write goes to a `tmp_path`.
"""

from __future__ import annotations

import pathlib

import pytest

from universe_builder import settings
from universe_builder.domains.history.models.city import City, CityDate, Population
from universe_builder.domains.history.models.discovery import Discovery, Finding
from universe_builder.models.subatomic import Date, Judgments, Location, Name
from tests.factories import backfill_required


@pytest.fixture
def examples_dir() -> pathlib.Path:
    """The active pack's examples dir (history's, by default) — pack-aware since
    each domain now ships its own examples, with the core dir as the fallback."""
    from universe_builder.registry import active_pack

    return active_pack().examples_dir or settings.EXAMPLES_DIR


@pytest.fixture
def prompt_config() -> dict:
    """The atom registry as the old `prompt_config.json` shape, now derived from
    the active `DomainPack` (the single source of truth — see `universe_builder.registry`).
    Kept so the tests that consumed the JSON dict need no change; not
    session-scoped because a pack switch rebuilds `atom_mapping`."""
    from universe_builder import atoms

    return {
        name: {
            "atom_class": m.atom_class.__name__,
            "plural": m.plural,
            "prompt_start_text": m.prompt_start_text,
            "example_json_file": m.example_json_file,
        }
        for name, m in atoms.atom_mapping.items()
    }


@pytest.fixture(autouse=True)
def no_live_calls(request, monkeypatch):
    """Make a real provider call impossible in the offline suite.

    Added after a test spent real money. `use_fake` registers
    its fake under a *model key*, defaulting to `claude-sonnet-5`; the test's
    `OperationConfig` named `gpt-5-nano`, so the lookup missed the fake, fell
    through to the real `ProviderCall`, and made a live call. The only symptom
    was that the test ran slowly.

    A fake that is wired to the wrong key must fail loudly, not quietly bill the
    account. Tests marked `integration` are exempt - they are deselected by
    default and are the ones that *should* reach the network.
    """
    if request.node.get_closest_marker("integration"):
        return

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "a test tried to make a live provider call. The usual cause is a fake "
            "registered under a different model key than the OperationConfig uses "
            "- `use_fake(..., model=...)` must match `model_version`. Mark the test "
            "`@pytest.mark.integration` if it is genuinely meant to reach the API."
        )

    # Patched on `provider_call`, the module the *pipeline* calls through, not on
    # `providers.run` itself. Provider-layer tests drive `run.complete` directly
    # with a fake provider and are legitimate; a mis-wired fake in a pipeline test
    # always arrives here. Tests that patch this name themselves still win, since
    # their `with patch(...)` is applied after this fixture.
    from universe_builder.llm_classes import provider_call

    monkeypatch.setattr(provider_call, "complete", forbidden)


@pytest.fixture
def fake_provider(monkeypatch):
    """A scripted `FakeProvider` (tests/fake_provider.py) behind *every* provider
    name in the registry, with the real `ProviderCall -> run.complete` path
    re-enabled.

    `no_live_calls` blocks `provider_call.complete` because a fake wired to the
    wrong model key used to fall through to a real client. Here nothing can
    fall through: every provider the registry can construct is this fake, so
    the guard is lifted and the pipeline runs its own transport code. Retries
    run without sleeping.
    """
    from tests.fake_provider import FakeProvider
    from universe_builder.llm_classes import provider_call
    from universe_builder.providers import registry, run as run_mod

    fake = FakeProvider()
    for name in registry.provider_names():
        monkeypatch.setitem(registry._CACHE, name, fake)
    monkeypatch.setattr(provider_call, "complete", run_mod.complete)
    monkeypatch.setattr(run_mod.time, "sleep", lambda _seconds: None)
    return fake


@pytest.fixture
def pack_overrides(monkeypatch):
    """Replace fields of the active `DomainPack` for one test: `pack_overrides(
    aliases_path=..., identity_resolvers=...)`. The pack is a frozen dataclass,
    so the active instance is swapped for a `dataclasses.replace` copy and put
    back afterwards.

    The usual use is pointing `aliases_path` at a temporary file. Several
    writers (`duplicates.apply`, `write_distinct_lines`, `lint.apply_review`)
    read the pack's path rather than `settings.ALIASES_LIST`, so without
    this an `--apply` in a test would write into the
    bundled `domains/history/aliases.txt`.
    """
    import dataclasses

    from universe_builder import registry

    def apply(**fields):
        monkeypatch.setattr(registry, "_active", dataclasses.replace(registry.active_pack(), **fields))
        return registry.active_pack()

    return apply


@pytest.fixture(autouse=True)
def default_run_policy():
    """Reset the retry/timeout policy around every test.

    Entry points call `providers.run.configure(...)`, which sets module-global
    state — a test that drives `cmd_run` or `run_phase` would otherwise leave
    pipeline.yaml's policy installed for whichever test happens to run next.
    """
    from universe_builder.providers import run as run_mod

    run_mod.configure()
    yield
    run_mod.configure()


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """Isolate generated files without moving cwd or copying package resources.

    The alias file is isolated too (absent by default): it is bundled data that
    grows with the corpus, and a new line in it must not change what a test's
    hand-written topics group to. A test that wants aliases writes its own file
    and points `settings.ALIASES_LIST` at it; the one test of the *shipped*
    seed file points back at the history pack's `ALIASES_PATH` explicitly.
    """
    output = tmp_path / "data" / "automated"
    monkeypatch.setenv("UNIVERSE_BUILDER_DATA_DIR", str(output))
    monkeypatch.setattr(settings, "ALIASES_LIST", tmp_path / "aliases.txt")
    # Notes likewise: an empty file of the test's own, so a pack's real notes
    # never reach a test's prompts (and a declared-but-missing file is an error).
    notes = tmp_path / "create_notes.txt"
    notes.write_text("", encoding="utf-8")
    monkeypatch.setattr(settings, "CREATE_NOTES", notes)
    # And no list neighbours unless a test asks: they change every create prompt.
    monkeypatch.setattr(settings, "CREATE_NEIGHBOURS", 0)
    settings.data_dir().mkdir(parents=True)
    return settings.data_dir()


@pytest.fixture
def fake_keys(monkeypatch):
    """Satisfy client constructors that read a key at import/construct time."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-real")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-not-real")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-real")


@pytest.fixture
def ep(workspace):
    """`execute_prompt`, with the data dir isolated by `workspace`.

    It used to reset `LLMCall.sequential_id`, a class counter deleted in
    v0.3.0; ids are content-addressed, so nothing
    carries between tests.
    """
    from universe_builder import execute_prompt

    return execute_prompt


# ---------------------------------------------------------------------------
# Minimal valid payloads. Written as plain dicts so their values are explicit
# and readable, but each is passed through `backfill_required` on the way out:
# every field set here is preserved exactly, while a required field *added* to
# the model later is filled automatically rather than breaking the fixture (and
# every test that leans on it). See tests/factories.py::backfill_required. A
# test that genuinely asserts the history *shape* belongs in a semantic test
# (test_models, test_periods, test_aliases), not here.
# ---------------------------------------------------------------------------

def make_date(**overrides) -> dict:
    date = {
        "year": 500,
        "month": None,
        "day": None,
        "era": "BCE",
        "certainty": "low",
        "sources": ["test"],
        "longFormInfo": [],
    }
    date.update(overrides)
    return backfill_required(Date, date)


def make_city_date(**overrides) -> dict:
    date = make_date(cityPopulationRange=[1000, 5000], averageCityPopulation=3000)
    date.update(overrides)
    return backfill_required(CityDate, date)


def make_name(common="Test Subject") -> dict:
    return backfill_required(Name, {
        "common": common,
        "otherNames": [],
        "sources": ["test"],
        "meaningOfNames": [],
        "longFormInfo": [],
    })


def make_location() -> dict:
    return backfill_required(Location, {
        "landmark": None,
        "city": "Memphis",
        "state": None,
        "country": "Egypt",
        "latitude": 29.85,
        "longitude": 31.25,
        "certainty": "medium",
        "sources": ["test"],
    })


def make_judgments() -> dict:
    fields = {}
    for key in (
        "politicalInfluence",
        "culturalContribution",
        "scientificAdvancement",
        "ethicalLegacy",
        "economicImpact",
        "militaryImpact",
        "surpriseOfImpact",
    ):
        fields[f"{key}Info"] = ""
        fields[f"{key}Score"] = 5
    fields["sources"] = ["test"]
    return backfill_required(Judgments, fields)


def make_population(**overrides) -> dict:
    pop = {
        "periodName": "Ptolemaic",
        "periodStart": make_city_date(),
        "periodEnd": make_date(),
        "certainty": "high",
        "typeOfGovernment": "democracy",
        "tradeAndCommerce": ["grain"],
        "tradingPartners": ["Rome"],
        "languagesSpoken": ["Greek"],
    }
    pop.update(overrides)
    return backfill_required(Population, pop)


_MONTHS = ("january", "february", "march", "april", "may", "june",
           "july", "august", "september", "october", "november", "december")


def _make_geography() -> dict:
    """Built from the model's own field names, so it cannot drift from it."""
    from universe_builder.domains.history.models.city import MonthlyClimate

    month = {name: 1.0 for name in MonthlyClimate.model_fields}
    return {"elevationMetres": 19.0, **{m: dict(month) for m in _MONTHS}}


def make_city(**overrides) -> dict:
    """A minimal valid `City` - `cityPeriodsOfHistory` is a list of names."""
    city = {
        "name": make_name("Memphis"),
        "foundingDate": make_date(),
        "dateOfDemise": make_date(),
        "location": make_location(),
        "cityPeriodsOfHistory": ["Old Kingdom"],
        "geographyAndClimate": _make_geography(),
        "judgments": make_judgments(),
        "architecturalHighlights": ["Temple of Ptah"],
        "notableFigures": ["Menes"],
        "popCulturalReferences": ["film"],
    }
    city.update(overrides)
    return backfill_required(City, city)


def make_finding(name="Earliest fragment") -> dict:
    return backfill_required(Finding, {
        "name": name,
        "ageOfRecord": make_date(),
        "dateOfModernDiscovery": make_date(year=1798, era="CE"),
        "location": make_location(),
    })


def make_discovery(**overrides) -> dict:
    """A minimal valid `Discovery`. Lives here because two modules need it."""
    discovery = {
        "name": make_name("Papyrus"),
        "earliestPhysicalRecord": make_finding("Papyrus fragment"),
        "earliestWrittenRecord": make_finding("Administrative record"),
        "discoverorOrInventor": "Unknown",
        "typeOfDiscovery": "incremental",
        "controversies": None,
        "challenges": "Organic material rarely survives.",
        "significance": "Made portable written records possible.",
        "shortSummary": "A writing surface.",
        "longSummary": "A writing surface made from the papyrus plant.",
    }
    discovery.update(overrides)
    return backfill_required(Discovery, discovery)


@pytest.fixture
def discovery_payload() -> dict:
    return make_discovery()


@pytest.fixture
def city_payload() -> dict:
    """A seed city, as the create pass emits it."""
    return make_city()


@pytest.fixture
def expanded_city_payload() -> dict:
    """A city after the expand pass: `Population` objects, not period names.

    `atoms.resolve_stored_class` must resolve this to `CityExpanded`; handing the
    seed `City` schema to an augment call and writing the reply back is B7, which
    collapses the expansion to a list of names.
    """
    return make_city(cityPeriodsOfHistory=[make_population()])


@pytest.fixture
def date_payload():
    return make_date


@pytest.fixture
def population_payload():
    return make_population


@pytest.fixture
def history_payload() -> dict:
    return {
        "people": ["Cleopatra"],
        "discoveries": ["Papyrus"],
        "events": ["Battle of Actium"],
        "cities": ["Memphis"],
        "artifacts": ["Rosetta Stone"],
    }


# ---------------------------------------------------------------------------
# Fake transport
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_call_factory(monkeypatch):
    """Build a fake `LLMCall` subclass returning scripted payloads.

    Usage:
        Fake = fake_call_factory(['{"...": "..."}', ...])
        monkeypatch.setitem(execute_prompt.llm_call_mapping, "gpt-4o-mini", Fake)
    """
    from universe_builder.llm_classes.llm_call import LLMCall

    def factory(responses):
        queue = list(responses)

        class _Fake(LLMCall):
            calls = []

            def __init__(self, **kwargs):
                # Augment variants take three extra kwargs the base does not.
                self.augment_to_json = kwargs.pop("augment_to_json", None)
                self.topic = kwargs.pop("topic", None)
                self.user_instructions = kwargs.pop("user_instructions", "")
                type(self).calls.append(dict(kwargs))
                super().__init__(**kwargs)
                self.llm_response_json = self.call_llm_api()

            def call_llm_api(self):
                if not queue:
                    raise AssertionError("fake provider ran out of scripted responses")
                return queue.pop(0)

        _Fake.calls = []
        return _Fake

    return factory
