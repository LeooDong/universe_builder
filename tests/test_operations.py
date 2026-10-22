"""Orchestration through a fake provider — create, fix, verify, expand, append.

No network and no key. `fake_call_factory` (see conftest.py) subclasses the real
`LLMCall` and replaces only `call_llm_api`, so everything under test is the
genuine path: example validation, filename construction, Pydantic validation,
the `_ERROR` convention and the write.

The shared `workspace` fixture isolates output with UNIVERSE_BUILDER_DATA_DIR; prompts
are read directly from the package and cwd never changes.
"""

from __future__ import annotations

import json

import pytest

from universe_builder import store
from universe_builder.domains.history.models.city import CityExpanded, Population
from tests.conftest import (
    make_city, make_date, make_location, make_name, make_population,
)


# ---------------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------------


def make_finding(name="Earliest fragment") -> dict:
    return {
        "name": name,
        "ageOfRecord": make_date(),
        "dateOfModernDiscovery": make_date(year=1798, era="CE"),
        "location": make_location(),
    }


def valid_discovery() -> str:
    return json.dumps({
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
    })


def valid_city() -> str:
    # Only the values later assertions check are pinned here (the period names,
    # the pop-culture and architecture lists a merge/append test reads back); the
    # rest of the city falls out of `make_city`, so adding a field to `City` no
    # longer strands this fixture. Was a full ten-field enumeration.
    return json.dumps(make_city(
        cityPeriodsOfHistory=["Old Kingdom", "Middle Kingdom"],
        architecturalHighlights=["Pyramid of Djoser"],
        notableFigures=["Ptah"],
        popCulturalReferences=["Asterix"],
    ))


@pytest.fixture(autouse=True)
def _discovery_schema_matches(ep):
    """Guard: if Discovery's shape changes, the canned payloads above go stale
    and every test here fails for the wrong reason."""
    from universe_builder.domains.history.models.discovery import Discovery

    Discovery.model_validate(json.loads(valid_discovery()))


def use_fake(ep, monkeypatch, fake_call_factory, responses, model="claude-sonnet-5", augment=False):
    Fake = fake_call_factory(responses)
    key = (model, "augment") if augment else model
    monkeypatch.setitem(ep.llm_call_mapping, key, Fake)
    return Fake


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------

class TestCreate:
    def test_a_valid_payload_is_written(self, ep, workspace, monkeypatch, fake_call_factory):
        use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery()])

        ep.call_and_save_prompt(
            config=ep.OperationConfig(
                atom_name="discovery",
                model_version="claude-sonnet-5",
                base_system_text="test",
            ),
            topic_list=["Papyrus"],
        )

        written = list((workspace / "discovery").glob("discovery_*.json"))
        assert len(written) == 1
        assert not written[0].name.endswith("_ERROR")
        assert json.loads(written[0].read_text())["name"]["common"] == "Papyrus"

    def test_an_invalid_payload_writes_an_error_file(self, ep, workspace, monkeypatch, fake_call_factory):
        use_fake(ep, monkeypatch, fake_call_factory, ['{"name": "not a Name object"}'])

        ep.call_and_save_prompt(
            config=ep.OperationConfig(
                atom_name="discovery",
                model_version="claude-sonnet-5",
                base_system_text="test",
            ),
            topic_list=["Broken"],
        )

        errors = list((workspace / "discovery").glob("*_ERROR"))
        assert len(errors) == 1
        # The raw payload is preserved — this is the property the run journal
        # must keep when _ERROR files are retired.
        assert errors[0].read_text() == '{"name": "not a Name object"}'

    def test_an_invalid_payload_writes_nothing_at_the_atom_path(self, ep, workspace, monkeypatch, fake_call_factory):
        """The `EmptyModel` placeholder is retired: it was dead
        weight wearing an atom's name, every reader needed a guard against it,
        and the `_ERROR` sentinel plus the journal already record the failure."""
        use_fake(ep, monkeypatch, fake_call_factory, ["{}"])

        ep.call_and_save_prompt(
            config=ep.OperationConfig(
                atom_name="discovery", model_version="claude-sonnet-5", base_system_text="test",
            ),
            topic_list=["Broken"],
        )

        main = [p for p in (workspace / "discovery").glob("*.json") if not p.name.endswith("_ERROR")]
        assert main == [], "a failure wrote something at the atom's own path"
        assert list((workspace / "discovery").glob("*_ERROR")), "the sentinel must still queue the fix"

    def test_one_failure_does_not_stop_the_topic_list(self, ep, workspace, monkeypatch, fake_call_factory):
        use_fake(ep, monkeypatch, fake_call_factory, ['{"bad": true}', valid_discovery()])

        ep.call_and_save_prompt(
            config=ep.OperationConfig(
                atom_name="discovery", model_version="claude-sonnet-5", base_system_text="test",
            ),
            topic_list=["Broken", "Papyrus"],
        )

        assert len(list((workspace / "discovery").glob("*_ERROR"))) == 1
        good = [
            p for p in (workspace / "discovery").glob("*.json")
            if json.loads(p.read_text()).get("name")
        ]
        assert len(good) == 1

    def test_the_subject_is_the_topic_and_nothing_else(self, ep, monkeypatch, fake_call_factory):
        """The subject must be the bare topic, because it is also the identity.

        This previously asserted the opposite - that the subject carried a
        `" (context: discovery)"` suffix. That suffix was redundant in the prompt
        (`prompt_start_text` already reads "Provide a response for the discovery
        or invention commonly known as") and wrong in the filename, because
        `LLMCall` derives the content-addressed id from this same string. See
        `TestBothPathsAgreeOnWhereATopicLives` for the consequence.
        """
        Fake = use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery()])

        ep.call_and_save_prompt(
            config=ep.OperationConfig(
                atom_name="discovery", model_version="claude-sonnet-5", base_system_text="test",
            ),
            topic_list=["Papyrus\n"],
        )

        assert Fake.calls[0]["prompt_subject"] == "Papyrus"

    def test_the_registered_example_is_loaded_and_validated(self, ep, monkeypatch, fake_call_factory):
        Fake = use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery()])

        ep.call_and_save_prompt(
            config=ep.OperationConfig(
                atom_name="discovery", model_version="claude-sonnet-5", base_system_text="test",
            ),
            topic_list=["Papyrus"],
        )

        # `discovery` is registered few-shot; the whole list reaches the call.
        assert Fake.calls[0]["example_json_file"] == [
            "invention_of_toothpaste.json", "hieroglyphic_writing.json",
        ]


# ---------------------------------------------------------------------------
# fix / verify
# ---------------------------------------------------------------------------

def pydantic_helper_validates(text: str) -> bool:
    """Whether `text` is a valid Discovery. Used to assert a payload is still broken."""
    from universe_builder.domains.history.models.discovery import Discovery

    try:
        Discovery.model_validate_json(text)
        return True
    except Exception:
        return False


class TestAugmentConcurrency:
    """The augment passes run files with bounded concurrency, the same trade
    create makes: expand alone was ~1,100 serial calls for one civilisation.
    The pool must process every eligible file and drop none — the property a
    parallel rewrite is most likely to break."""

    def _set_concurrency(self, ep, monkeypatch, n):
        cfg = ep._pipeline_config()
        monkeypatch.setattr(cfg, "defaults", {**cfg.defaults, "concurrency": n})
        monkeypatch.setattr(ep, "_pipeline_config", lambda *a, **k: cfg)

    def test_every_file_is_verified_under_concurrency(self, ep, workspace, monkeypatch,
                                                      fake_call_factory):
        folder = workspace / "discovery"
        folder.mkdir()
        subjects = [f"topic_{i}" for i in range(9)]
        for i, s in enumerate(subjects):
            (folder / f"discovery_000000000{i}_{s}_claude-sonnet-5.json").write_text(valid_discovery())
        # One scripted response per file; if the pool dropped or double-ran a
        # file the count would not match and the fake would over/under-draw.
        use_fake(ep, monkeypatch, fake_call_factory,
                 [valid_discovery() for _ in subjects], augment=True)
        self._set_concurrency(ep, monkeypatch, 4)

        results = ep.augment_and_save_prompt(ep.OperationConfig(
            atom_name="discovery", model_version=("claude-sonnet-5", "augment"),
            base_system_text="test", operation_type="verify"))

        assert len(results) == len(subjects), "the pool dropped or duplicated a file"
        assert all(r.outcome is ep.Outcome.OK for r in results)

    def test_verify_warms_on_a_real_file_not_a_skipped_sentinel(self, ep, workspace,
                                                                monkeypatch, fake_call_factory):
        """The gate's catch: sentinels sort first but verify skips them, so a
        folder with a sentinel used to waste the warm-up slot on it and fire
        the pool cold. The pass now visits only the files it acts on, so verify
        never even sees the sentinel."""
        folder = workspace / "discovery"
        folder.mkdir()
        (folder / "discovery_0000000009_broken_claude-sonnet-5.json_ERROR").write_text(
            '{"incomplete": true}')
        for i in range(3):
            (folder / f"discovery_000000000{i}_ok{i}_claude-sonnet-5.json").write_text(valid_discovery())
        # Exactly three responses: if the sentinel were visited, the fake would
        # be asked a fourth time and raise "ran out of scripted responses".
        use_fake(ep, monkeypatch, fake_call_factory,
                 [valid_discovery() for _ in range(3)], augment=True)
        self._set_concurrency(ep, monkeypatch, 4)

        results = ep.augment_and_save_prompt(ep.OperationConfig(
            atom_name="discovery", model_version=("claude-sonnet-5", "augment"),
            base_system_text="test", operation_type="verify"))

        assert len(results) == 3
        assert (folder / "discovery_0000000009_broken_claude-sonnet-5.json_ERROR").exists(), \
            "verify must not touch the sentinel"

    def test_the_selected_config_concurrency_reaches_the_pass(self, ep, workspace,
                                                              monkeypatch, fake_call_factory, tmp_path):
        """A `--config` run's concurrency must be the selected file's, not the
        bundled default's — the same bug-class as the retry-policy gate catch.
        run_phase resolves it from the given config and threads it through."""
        cfg = tmp_path / "custom.yaml"
        cfg.write_text(
            "defaults:\n  provider: openai\n  model: economy\n  concurrency: 7\n"
            "tiers:\n  openai:\n    frontier: gpt-6-astra\n    balanced: gpt-5.6-luna\n"
            "    economy: gpt-5-nano\nphases:\n  verify: economy\n",
            encoding="utf-8")
        seen = {}

        def spy(config, limit=None, concurrency=None, explicit=False):
            seen["concurrency"] = concurrency
            return []
        monkeypatch.setattr(ep, "augment_and_save_prompt", spy)

        ep.run_phase("verify", atoms=["discovery"], config=str(cfg))

        assert seen["concurrency"] == 7, "the selected config's concurrency was ignored"

    def test_serial_and_concurrent_agree_on_what_was_processed(self, ep, workspace,
                                                               monkeypatch, fake_call_factory):
        folder = workspace / "discovery"
        folder.mkdir()
        for i in range(5):
            (folder / f"discovery_000000000{i}_t{i}_claude-sonnet-5.json").write_text(valid_discovery())
        use_fake(ep, monkeypatch, fake_call_factory,
                 [valid_discovery() for _ in range(5)], augment=True)
        self._set_concurrency(ep, monkeypatch, 1)

        results = ep.augment_and_save_prompt(ep.OperationConfig(
            atom_name="discovery", model_version=("claude-sonnet-5", "augment"),
            base_system_text="test", operation_type="verify"))

        assert len(results) == 5


class TestFixAndVerify:
    def _seed(self, workspace, name, payload):
        folder = workspace / "discovery"
        folder.mkdir(exist_ok=True)
        (folder / name).write_text(payload)
        return folder / name

    def test_fix_replaces_the_error_file_with_a_valid_one(self, ep, workspace, monkeypatch, fake_call_factory):
        bad = self._seed(
            workspace,
            "discovery_0000000001_papyrus_claude-sonnet-5.json_ERROR",
            '{"incomplete": true}',
        )
        use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery()], augment=True)

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="discovery",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                operation_type="fix",
            ),
        )

        assert not bad.exists(), "the _ERROR sentinel should be cleared on a successful fix"
        fixed = workspace / "discovery" / "discovery_0000000001_papyrus_claude-sonnet-5.json"
        assert json.loads(fixed.read_text())["name"]["common"] == "Papyrus"

    # -- B8: a failed augment must not destroy the data it was meant to improve --
    #
    # These cover the case the pre-existing failure test does not: the call
    # *succeeds* at transport and returns a payload that does not validate.
    # `get_valid_response` answers with an `EmptyModel` sentinel, which the old
    # code wrote straight to disk.

    def test_b8_a_fix_that_does_not_validate_keeps_the_error_file(
        self, ep, workspace, monkeypatch, fake_call_factory
    ):
        """The sentinel holds the only copy of what the model returned."""
        bad = self._seed(
            workspace,
            "discovery_0000000001_papyrus_claude-sonnet-5.json_ERROR",
            '{"incomplete": true}',
        )
        # Transport succeeds; the payload is still not a Discovery.
        use_fake(ep, monkeypatch, fake_call_factory, ['{"still": "broken"}'], augment=True)

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="discovery",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                operation_type="fix",
            ),
        )

        assert bad.exists(), "an unsuccessful fix must not delete its own evidence"
        # The sentinel still holds an unvalidatable payload, so the file stays on
        # the work queue for the next run. Which payload it holds is currently
        # incidental: the augment call mints a fresh sequential id rather than
        # inheriting the identity of the file it is repairing, and that id
        # restarts at 1 each process, so its own error write may or may not land
        # on this same path (an open item: augment calls mint a new identity).
        assert not pydantic_helper_validates(bad.read_text())

    def test_b8_a_fix_that_does_not_validate_writes_no_stub_over_the_target(
        self, ep, workspace, monkeypatch, fake_call_factory
    ):
        """An EmptyModel on disk is indistinguishable from a real atom by name."""
        self._seed(
            workspace,
            "discovery_0000000001_papyrus_claude-sonnet-5.json_ERROR",
            '{"incomplete": true}',
        )
        use_fake(ep, monkeypatch, fake_call_factory, ['{"still": "broken"}'], augment=True)

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="discovery",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                operation_type="fix",
            ),
        )

        target = workspace / "discovery" / "discovery_0000000001_papyrus_claude-sonnet-5.json"
        assert not target.exists(), "a failed fix must not create a sentinel-shaped file"

    def test_b8_a_failed_verification_leaves_the_good_file_intact(
        self, ep, workspace, monkeypatch, fake_call_factory
    ):
        """Verify runs over files that are already valid, so a stub is a data loss."""
        original = valid_discovery()
        good = self._seed(
            workspace, "discovery_0000000001_papyrus_claude-sonnet-5.json", original
        )
        use_fake(ep, monkeypatch, fake_call_factory, ['{"still": "broken"}'], augment=True)

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="discovery",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                operation_type="verify",
            ),
        )

        assert json.loads(good.read_text()) == json.loads(original)

    def test_fix_leaves_the_error_file_when_the_fix_fails(self, ep, workspace, monkeypatch, fake_call_factory):
        bad = self._seed(
            workspace,
            "discovery_0000000001_papyrus_claude-sonnet-5.json_ERROR",
            '{"incomplete": true}',
        )
        Fake = fake_call_factory([])  # raises on first call — a transport failure
        monkeypatch.setitem(ep.llm_call_mapping, ("claude-sonnet-5", "augment"), Fake)

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="discovery",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                operation_type="fix",
            ),
        )

        assert bad.exists(), "a failed fix must not destroy the evidence"

    def test_verify_rewrites_a_valid_file_in_place(self, ep, workspace, monkeypatch, fake_call_factory):
        good = self._seed(
            workspace, "discovery_0000000001_papyrus_claude-sonnet-5.json", valid_discovery(),
        )
        improved = json.loads(valid_discovery())
        improved["shortSummary"] = "An improved summary."
        use_fake(ep, monkeypatch, fake_call_factory, [json.dumps(improved)], augment=True)

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="discovery",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                operation_type="verify",
            ),
        )

        assert json.loads(good.read_text())["shortSummary"] == "An improved summary."

    def test_verify_skips_error_files(self, ep, workspace, monkeypatch, fake_call_factory):
        bad = self._seed(
            workspace, "discovery_0000000001_x_claude-sonnet-5.json_ERROR", '{"incomplete": true}',
        )
        Fake = fake_call_factory([])  # any call at all would raise
        monkeypatch.setitem(ep.llm_call_mapping, ("claude-sonnet-5", "augment"), Fake)

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="discovery",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                operation_type="verify",
            ),
        )

        assert bad.read_text() == '{"incomplete": true}'
        assert not Fake.calls

    def test_the_empty_sentinel_is_skipped(self, ep, workspace, monkeypatch, fake_call_factory):
        self._seed(
            workspace,
            "discovery_0000000001_x_claude-sonnet-5.json",
            json.dumps({"error": "No data found"}),
        )
        Fake = fake_call_factory([])
        monkeypatch.setitem(ep.llm_call_mapping, ("claude-sonnet-5", "augment"), Fake)

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="discovery",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                operation_type="verify",
            ),
        )

        assert not Fake.calls, "EmptyModel files must not be re-sent to the LLM"


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------

class TestAppend:
    def test_run_phase_append_merges_after_generating(self, ep, workspace, monkeypatch):
        """B1: `universe_builder run --phase append` is generate-then-merge. The merge step
        used to be reachable only from the legacy `main()`, so a CLI append run
        left `append_<atom>_*.json` beside the atoms with nothing to consume them.
        With no atoms present, generation is a no-op, but the merge must still be
        invoked for the atom."""
        (workspace / "city").mkdir()  # empty: generation finds nothing to append to
        merged = []
        monkeypatch.setattr(ep.merge_helper, "process_directory",
                            lambda root, atom: merged.append(atom))

        ep.run_phase("append", atoms=["city"])

        assert merged == ["city"], "the append phase must run the merge step"

    def test_append_writes_a_separate_file(self, ep, workspace, monkeypatch, fake_call_factory):
        folder = workspace / "city"
        folder.mkdir()
        (folder / "city_0000000001_memphis_claude-sonnet-5.json").write_text(valid_city())
        use_fake(
            ep, monkeypatch, fake_call_factory,
            [json.dumps({"popCulturalReferences": ["The Mummy"]})],
            augment=True,
        )

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="city",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                operation_type="append",
            ),
        )

        appended = list(folder.glob("append_city_*.json"))
        assert len(appended) == 1
        assert json.loads(appended[0].read_text())["popCulturalReferences"] == ["The Mummy"]

    def test_append_then_merge_promotes_the_city_in_place(self, ep, workspace, monkeypatch, fake_call_factory):
        folder = workspace / "city"
        folder.mkdir()
        city = folder / store.AtomName("city", None, "memphis", "claude-sonnet-5").getReadableFileName()
        city.write_text(valid_city())
        use_fake(
            ep, monkeypatch, fake_call_factory,
            [json.dumps({"popCulturalReferences": ["The Mummy"]})],
            augment=True,
        )

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="city",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                operation_type="append",
            ),
        )
        ep.merge_helper.process_directory(str(workspace), "city")

        assert json.loads(city.read_text())["popCulturalReferences"] == ["Asterix", "The Mummy"]
        assert sorted(p.name for p in folder.iterdir()) == [city.name], (
            "one file per atom: the consumed append retires to _merged/ and no "
            "final_ file is written"
        )


# ---------------------------------------------------------------------------
# expand
# ---------------------------------------------------------------------------

class TestExpand:
    def test_expand_config_carries_the_registered_example(self):
        """The one-shot example must reach the *synchronous* expand call, not only
        the batch path. It primes the cache and validates against `expander_class`
        (a `Population` — exactly what the registered example is). A regression here
        ships a smaller, example-less prompt on the sync path than on the batch
        path, which is what happened before this was fixed."""
        from universe_builder.atoms import atom_mapping
        from universe_builder.operations.expand import expand_config

        cfg = expand_config("city", "claude-sonnet-5", base_system_text="t")

        assert cfg.example_json_file, "sync expand must carry the one-shot example"
        assert cfg.example_json_file == atom_mapping["expand_city"].example_json_file

    def test_expand_calls_once_per_list_entry(self, ep, workspace, monkeypatch, fake_call_factory):
        folder = workspace / "city"
        folder.mkdir()
        (folder / "city_0000000001_memphis_claude-sonnet-5.json").write_text(valid_city())
        Fake = use_fake(
            ep, monkeypatch, fake_call_factory,
            [json.dumps(make_population()), json.dumps(make_population(periodName="Middle Kingdom"))],
            augment=True,
        )

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="city",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                subatom_to_expand="cityPeriodsOfHistory",
                expander_class=Population,
                expanded_class=CityExpanded,
                operation_type="expand",
            ),
        )

        # valid_city() has two periods.
        assert len(Fake.calls) == 2

    def test_periods_expand_concurrently_and_all_land(
            self, ep, workspace, monkeypatch, fake_call_factory):
        """Inner-loop concurrency: a deep city's periods expand in parallel,
        each result placed at its own index, none dropped — the property the
        parallel rewrite could break. Placement is by task index, not response
        order, so it holds however the threads interleave."""
        from universe_builder.operations import _support

        folder = workspace / "city"
        folder.mkdir()
        payload = json.loads(valid_city())
        payload["cityPeriodsOfHistory"] = ["Old Kingdom", "Middle Kingdom",
                                           "New Kingdom", "Ptolemaic", "Roman"]
        target = folder / "city_0000000001_memphis_claude-sonnet-5.json"
        target.write_text(json.dumps(payload))
        Fake = use_fake(
            ep, monkeypatch, fake_call_factory,
            [json.dumps(make_population(periodName=f"P{i}")) for i in range(5)],
            augment=True)
        monkeypatch.setattr(_support, "_call_concurrency", 4)
        monkeypatch.setattr(_support, "_call_semaphore", _support.threading.BoundedSemaphore(4))

        target = self._expand_config_run(ep, target)

        from universe_builder.domains.history.models.city import CityExpanded
        reloaded = CityExpanded.model_validate_json(target.read_text())
        assert reloaded.is_fully_expanded(), "a period was dropped under concurrency"
        assert len(Fake.calls) == 5, "one call per period, no more, no fewer"

    def test_the_gate_bounds_periods_in_flight(self, ep, workspace, monkeypatch,
                                               fake_call_factory):
        """`call_slot` caps concurrent model calls at the configured bound, so
        the nested file×period pools cannot breach a rate limit."""
        import threading
        import time

        from universe_builder.operations import _support

        folder = workspace / "city"
        folder.mkdir()
        payload = json.loads(valid_city())
        payload["cityPeriodsOfHistory"] = [f"Period {i}" for i in range(8)]
        target = folder / "city_0000000001_memphis_claude-sonnet-5.json"
        target.write_text(json.dumps(payload))

        live = {"now": 0, "max": 0}
        lock = threading.Lock()

        Fake = use_fake(ep, monkeypatch, fake_call_factory,
                        [json.dumps(make_population(periodName=f"P{i}")) for i in range(8)],
                        augment=True)
        # The gate wraps call construction — where a real provider's network
        # call happens — so measure concurrency there, inside the slot, by
        # making the fake's call take a beat.
        raw_call = Fake.call_llm_api

        def watched(self):
            with lock:
                live["now"] += 1
                live["max"] = max(live["max"], live["now"])
            time.sleep(0.02)
            with lock:
                live["now"] -= 1
            return raw_call(self)

        monkeypatch.setattr(Fake, "call_llm_api", watched)
        monkeypatch.setattr(_support, "_call_concurrency", 3)
        monkeypatch.setattr(_support, "_call_semaphore", _support.threading.BoundedSemaphore(3))

        self._expand_config_run(ep, target)

        assert live["max"] <= 3, f"{live['max']} calls were in flight; the gate is 3"
        assert live["max"] > 1, "the periods did not actually run in parallel"

    def _expand_config_run(self, ep, target):
        from universe_builder.domains.history.models.city import CityExpanded

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="city",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                subatom_to_expand="cityPeriodsOfHistory",
                expander_class=Population,
                expanded_class=CityExpanded,
                operation_type="expand",
            ),
        )
        return target

    def _run_expand(self, ep, workspace, monkeypatch, fake_call_factory, responses):
        from universe_builder.domains.history.models.city import CityExpanded

        folder = workspace / "city"
        folder.mkdir()
        target = folder / "city_0000000001_memphis_claude-sonnet-5.json"
        target.write_text(valid_city())
        use_fake(ep, monkeypatch, fake_call_factory, responses, augment=True)

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="city",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                subatom_to_expand="cityPeriodsOfHistory",
                expander_class=Population,
                expanded_class=CityExpanded,
                operation_type="expand",
            ),
        )
        return target

    def test_B7_the_expanded_file_can_be_read_back(self, ep, workspace, monkeypatch, fake_call_factory):
        """B7: expand used to write Population objects into `City`, whose
        `cityPeriodsOfHistory` is `List[str]`. The file serialised with a warning
        and could never be reloaded — every generated city file was affected."""
        from universe_builder.domains.history.models.city import CityExpanded

        target = self._run_expand(
            ep, workspace, monkeypatch, fake_call_factory,
            [json.dumps(make_population()), json.dumps(make_population(periodName="Middle Kingdom"))],
        )

        reloaded = CityExpanded.model_validate_json(target.read_text())
        assert reloaded.is_fully_expanded()
        assert [p.periodName for p in reloaded.cityPeriodsOfHistory] == ["Ptolemaic", "Middle Kingdom"]

    def test_B7_the_expanded_file_is_not_a_plain_city(self, ep, workspace, monkeypatch, fake_call_factory):
        """The seed model must keep rejecting the expanded shape, so a regression
        cannot pass unnoticed."""
        from universe_builder.domains.history.models.city import City
        from pydantic import ValidationError

        target = self._run_expand(
            ep, workspace, monkeypatch, fake_call_factory,
            [json.dumps(make_population()), json.dumps(make_population())],
        )

        with pytest.raises(ValidationError):
            City.model_validate_json(target.read_text())

    def test_a_failed_entry_keeps_its_period_name(self, ep, workspace, monkeypatch, fake_call_factory):
        """Expansion is best-effort per entry. A period whose call fails must be
        left as its name — never dropped, and never replaced by an EmptyModel
        sentinel where a Population belongs."""
        from universe_builder.domains.history.models.city import CityExpanded

        # One good response, then the queue is empty so the second call raises.
        target = self._run_expand(
            ep, workspace, monkeypatch, fake_call_factory, [json.dumps(make_population())],
        )

        reloaded = CityExpanded.model_validate_json(target.read_text())
        assert not reloaded.is_fully_expanded()
        assert reloaded.unexpanded_periods() == ["Middle Kingdom"]
        assert len(reloaded.cityPeriodsOfHistory) == 2, "a failed entry must not be dropped"

    def test_expansion_preserves_the_other_city_fields(self, ep, workspace, monkeypatch, fake_call_factory):
        from universe_builder.domains.history.models.city import CityExpanded

        target = self._run_expand(
            ep, workspace, monkeypatch, fake_call_factory,
            [json.dumps(make_population()), json.dumps(make_population())],
        )

        reloaded = CityExpanded.model_validate_json(target.read_text())
        assert reloaded.name.common == "Memphis"
        assert reloaded.popCulturalReferences == ["Asterix"]
        assert reloaded.architecturalHighlights == ["Pyramid of Djoser"]


# ---------------------------------------------------------------------------
# Stage-aware resolution — the follow-on hazard from the B7 fix
# ---------------------------------------------------------------------------

class TestStageAwareResolution:
    """Once expand rewrites a city file as `CityExpanded`, resolving that file by
    atom name alone yields `City`, which it no longer validates against. The fix
    and verify passes would then hand the LLM the *seed* schema and collapse the
    expansion back to a list of period names — silently destroying the expand
    output. `resolve_stored_class` is what prevents that."""

    def _expanded_payload(self):
        payload = json.loads(valid_city())
        payload["cityPeriodsOfHistory"] = [make_population(), make_population(periodName="Middle Kingdom")]
        return payload

    def _partial_payload(self):
        payload = json.loads(valid_city())
        payload["cityPeriodsOfHistory"] = [make_population(), "Middle Kingdom"]
        return payload

    def test_a_seed_city_resolves_to_city(self, ep):
        from universe_builder.domains.history.models.city import City

        assert ep.resolve_stored_class("city", json.loads(valid_city())) is City

    def test_an_expanded_city_resolves_to_city_expanded(self, ep):
        from universe_builder.domains.history.models.city import CityExpanded

        assert ep.resolve_stored_class("city", self._expanded_payload()) is CityExpanded

    def test_a_partially_expanded_city_resolves_to_city_expanded(self, ep):
        from universe_builder.domains.history.models.city import CityExpanded

        assert ep.resolve_stored_class("city", self._partial_payload()) is CityExpanded

    def test_an_atom_with_one_stage_resolves_normally(self, ep):
        from universe_builder.domains.history.models.discovery import Discovery

        assert ep.resolve_stored_class("discovery", json.loads(valid_discovery())) is Discovery

    def test_a_malformed_payload_falls_back_to_the_default(self, ep):
        from universe_builder.domains.history.models.city import City

        assert ep.resolve_stored_class("city", None) is City
        assert ep.resolve_stored_class("city", {"cityPeriodsOfHistory": "not a list"}) is City
        assert ep.resolve_stored_class("city", {}) is City

    def test_the_resolved_class_is_what_verify_sends_to_the_llm(
        self, ep, workspace, monkeypatch, fake_call_factory
    ):
        """The whole point: verify must ask for the expanded schema, not the seed
        one. If it asked for `City`, the model would return period names and the
        expansion would be gone."""
        from universe_builder.domains.history.models.city import CityExpanded

        folder = workspace / "city"
        folder.mkdir()
        target = folder / "city_0000000001_memphis_claude-sonnet-5.json"
        target.write_text(json.dumps(self._expanded_payload()))

        Fake = use_fake(
            ep, monkeypatch, fake_call_factory,
            [json.dumps(self._expanded_payload())],
            augment=True,
        )

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="city",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                operation_type="verify",
            ),
        )

        assert Fake.calls, "verify never ran"
        assert Fake.calls[0]["atom_class"] is CityExpanded

    def test_verify_does_not_collapse_an_expanded_city(
        self, ep, workspace, monkeypatch, fake_call_factory
    ):
        from universe_builder.domains.history.models.city import CityExpanded

        folder = workspace / "city"
        folder.mkdir()
        target = folder / "city_0000000001_memphis_claude-sonnet-5.json"
        target.write_text(json.dumps(self._expanded_payload()))
        use_fake(
            ep, monkeypatch, fake_call_factory,
            [json.dumps(self._expanded_payload())],
            augment=True,
        )

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="city",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                operation_type="verify",
            ),
        )

        reloaded = CityExpanded.model_validate_json(target.read_text())
        assert reloaded.is_fully_expanded(), "verify collapsed the expansion"

    def test_re_expanding_a_finished_file_makes_no_api_call(
        self, ep, workspace, monkeypatch, fake_call_factory
    ):
        """Idempotence: running expand twice must not burn tokens rewriting work
        that is already there."""
        from universe_builder.domains.history.models.city import CityExpanded

        folder = workspace / "city"
        folder.mkdir()
        target = folder / "city_0000000001_memphis_claude-sonnet-5.json"
        target.write_text(json.dumps(self._expanded_payload()))
        Fake = fake_call_factory([])  # any call would raise
        monkeypatch.setitem(ep.llm_call_mapping, ("claude-sonnet-5", "augment"), Fake)

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="city",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                subatom_to_expand="cityPeriodsOfHistory",
                expander_class=Population,
                expanded_class=CityExpanded,
                operation_type="expand",
            ),
        )

        assert not Fake.calls
        assert CityExpanded.model_validate_json(target.read_text()).is_fully_expanded()

    def test_expand_resumes_a_partial_file_without_redoing_finished_entries(
        self, ep, workspace, monkeypatch, fake_call_factory
    ):
        from universe_builder.domains.history.models.city import CityExpanded

        folder = workspace / "city"
        folder.mkdir()
        target = folder / "city_0000000001_memphis_claude-sonnet-5.json"
        target.write_text(json.dumps(self._partial_payload()))
        Fake = use_fake(
            ep, monkeypatch, fake_call_factory,
            [json.dumps(make_population(periodName="Middle Kingdom"))],
            augment=True,
        )

        ep.augment_and_save_prompt(
            ep.OperationConfig(
                atom_name="city",
                model_version=("claude-sonnet-5", "augment"),
                base_system_text="test",
                subatom_to_expand="cityPeriodsOfHistory",
                expander_class=Population,
                expanded_class=CityExpanded,
                operation_type="expand",
            ),
        )

        assert len(Fake.calls) == 1, "the already-expanded entry should be skipped"
        reloaded = CityExpanded.model_validate_json(target.read_text())
        assert reloaded.is_fully_expanded()


class TestBothPathsAgreeOnWhereATopicLives:
    """One topic, one file, whichever path generated it.

    This is the whole point of content-addressed ids, and
    `batch._output_path` states it directly: "Deriving the id from the subject
    makes the two paths agree." They did not agree. `create_atom` appended
    `" (context: {atom_name})"` to the subject before handing it to `LLMCall`,
    which names *and hashes* the atom from that string, while the batch path
    hashed the bare subject. Observed live - the same three topics
    generated both ways produced six files:

        sync   discovery_h495688e995974e14_hieroglyphic_writing_context_discovery_...
        batch  discovery_h931c34c8003af691_hieroglyphic_writing_...
    """

    TOPIC = "Hieroglyphic writing"

    def test_the_two_paths_hash_the_same_topic_the_same_way(self, workspace):
        """Note: this one passes even with the bug reintroduced.

        It compares `store.path_for` against `batch._output_path` directly, and
        the defect was never in either - it was in `create_atom` mangling the
        subject before it reached them. Kept because it pins the helpers to each
        other, but `test_the_sync_pass_writes_to_that_path` below is the test that
        actually catches a regression here.
        """
        from universe_builder import store
        from universe_builder.providers import batch as batch_mod
        from universe_builder.providers.types import BatchHandle

        handle = BatchHandle(
            provider="openai", batch_id="b", model="gpt-5-nano",
            submitted_at="2026-09-11T00:00:00Z", request_count=1,
            routing={"discovery-00000-x": {"atom_type": "discovery", "subject": self.TOPIC}},
            extra={"schema_model": "Discovery"},
        )
        batch_path = batch_mod._output_path(handle, "discovery-00000-x")
        sync_path = store.path_for("discovery", None, self.TOPIC, "gpt-5-nano")

        assert sync_path == batch_path, (
            f"one topic, two files:\n  sync  {sync_path.name}\n  batch {batch_path.name}"
        )

    def test_the_sync_pass_writes_to_that_path(self, ep, monkeypatch, fake_call_factory, workspace):
        """Pins the actual pass, not just the helpers agreeing in isolation."""
        from universe_builder import store

        use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery()])
        ep.call_and_save_prompt(
            config=ep.OperationConfig(
                atom_name="discovery", model_version="claude-sonnet-5", base_system_text="test",
            ),
            topic_list=[self.TOPIC],
        )

        expected = store.path_for("discovery", None, self.TOPIC, "claude-sonnet-5")
        written = sorted(store.dir_for("discovery").glob("*.json"))

        assert [p.name for p in written] == [expected.name], (
            f"the create pass did not write to the content-addressed path for "
            f"{self.TOPIC!r}; got {[p.name for p in written]}"
        )
