"""Operations report success and failure through a typed `Result`.

This replaces the old arrangement where every `process_*_operation` returned a
JSON string and each caller re-parsed it to ask `is_empty_model`. That shared
channel is what B8 was made of: an operation that failed looked exactly like one
that succeeded until someone inspected the payload, and two call sites did not.

No network, no API key: the fake `LLMCall` from conftest is used throughout.
"""

from __future__ import annotations

import json

import pytest

from universe_builder.domains.history.models.city import CityExpanded, Population
from universe_builder.models.subatomic import EmptyModel
from universe_builder.operations.types import Outcome, Result


class TestResultSemantics:
    def test_success_is_writable(self):
        result = Result.success(EmptyModel())
        assert result.ok and result.should_write

    def test_invalid_is_not_writable_and_keeps_the_payload(self):
        result = Result.invalid("half a json obj", "did not validate")
        assert not result.should_write
        assert result.raw_text == "half a json obj"
        assert result.keeps_work_queued

    def test_failed_is_not_writable_and_stays_queued(self):
        result = Result.failed("connection error")
        assert not result.should_write
        assert result.keeps_work_queued

    def test_skipped_is_not_writable_but_is_not_a_failure(self):
        """Re-running expand over a finished file must not look like an error."""
        result = Result.skipped("already fully expanded", EmptyModel())
        assert not result.should_write
        assert not result.keeps_work_queued
        assert result.outcome is Outcome.SKIPPED

    def test_a_result_without_an_atom_refuses_to_serialise(self):
        """Better a loud error than a stub written where an atom belongs."""
        with pytest.raises(ValueError):
            Result.invalid("raw", "boom").json()

    def test_json_matches_the_corpus_format(self):
        result = Result.success(EmptyModel())
        assert result.json() == EmptyModel().model_dump_json(indent=4)

    def test_metadata_never_drives_control_flow(self):
        """Metadata is for the journal; `outcome` decides everything."""
        result = Result.success(EmptyModel(), unexpanded=["a", "b"])
        assert result.should_write
        assert result.metadata["unexpanded"] == ["a", "b"]


class TestOperationsReturnResults:
    """The outcome each operation reports for a given situation."""

    def _seed(self, workspace, name, payload):
        folder = workspace / "discovery"
        folder.mkdir(exist_ok=True)
        (folder / name).write_text(payload)
        return folder / name

    def _config(self, ep, operation):
        return ep.OperationConfig(
            atom_name="discovery",
            model_version=("claude-sonnet-5", "augment"),
            base_system_text="test",
            operation_type=operation,
        )

    def test_a_valid_fix_returns_ok(self, ep, workspace, monkeypatch, fake_call_factory):
        from tests.test_operations import use_fake, valid_discovery

        use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery()], augment=True)
        result = ep.process_fix_operation('{"broken": true}', self._config(ep, "fix"), "papyrus")

        assert result.outcome is Outcome.OK
        assert result.should_write
        assert not isinstance(result.atom, EmptyModel)

    def test_an_unvalidatable_fix_returns_invalid_not_ok(
        self, ep, workspace, monkeypatch, fake_call_factory
    ):
        from tests.test_operations import use_fake

        use_fake(ep, monkeypatch, fake_call_factory, ['{"still": "broken"}'], augment=True)
        result = ep.process_fix_operation('{"broken": true}', self._config(ep, "fix"), "papyrus")

        assert result.outcome is Outcome.INVALID
        assert not result.should_write
        assert result.atom is None

    def test_an_unvalidatable_fix_keeps_the_raw_payload(
        self, ep, workspace, monkeypatch, fake_call_factory
    ):
        """The evidence B8 destroyed."""
        from tests.test_operations import use_fake

        use_fake(ep, monkeypatch, fake_call_factory, ['{"still": "broken"}'], augment=True)
        result = ep.process_fix_operation('{"broken": true}', self._config(ep, "fix"), "papyrus")

        assert json.loads(result.raw_text) == {"still": "broken"}

    def test_a_transport_failure_returns_failed(self, ep, workspace, monkeypatch,
                                                fake_call_factory):
        Fake = fake_call_factory([])  # runs out of scripted responses -> raises
        monkeypatch.setitem(ep.llm_call_mapping, ("claude-sonnet-5", "augment"), Fake)

        result = ep.process_fix_operation('{"broken": true}', self._config(ep, "fix"), "papyrus")

        assert result.outcome is Outcome.FAILED
        assert not result.should_write

    def test_verify_of_a_bad_response_returns_invalid(self, ep, workspace, monkeypatch,
                                                      fake_call_factory):
        from tests.test_operations import use_fake, valid_discovery

        use_fake(ep, monkeypatch, fake_call_factory, ['{"nope": 1}'], augment=True)
        result = ep.process_verify_operation(valid_discovery(), self._config(ep, "verify"),
                                             "papyrus")

        assert result.outcome is Outcome.INVALID
        assert not result.should_write


class TestExpandOutcomes:
    @staticmethod
    def _expanded_city() -> dict:
        """A city whose periods are all objects — i.e. expand has already run."""
        from tests.conftest import make_population
        from tests.test_operations import valid_city

        payload = json.loads(valid_city())
        payload["cityPeriodsOfHistory"] = [
            make_population(periodName="Old Kingdom"),
            make_population(periodName="Middle Kingdom"),
        ]
        return payload

    def test_a_fully_expanded_file_is_skipped_not_failed(self, ep, workspace, monkeypatch,
                                                         fake_call_factory):
        Fake = fake_call_factory([])  # any call at all would raise
        monkeypatch.setitem(ep.llm_call_mapping, ("claude-sonnet-5", "augment"), Fake)

        config = ep.OperationConfig(
            atom_name="city",
            model_version=("claude-sonnet-5", "augment"),
            base_system_text="test",
            subatom_to_expand="cityPeriodsOfHistory",
            expander_class=Population,
            expanded_class=CityExpanded,
            operation_type="expand",
        )
        result = ep.process_expand_operation(self._expanded_city(), config, "memphis")

        assert result.outcome is Outcome.SKIPPED
        assert not result.should_write, "an unchanged file must not be rewritten"


class TestRegenerationNeverDestroysAValidAtom:
    """B11: a failed create must not overwrite the atom it was regenerating.

    `LLMCall.write_response_to_file` writes whatever `parse_response_to_pydantic`
    produced, which is an `EmptyModel` sentinel when validation fails. That was
    survivable only while ids were a per-process counter, because a failed rerun
    landed on a *different* filename. Content-addressed ids made the
    path a function of the topic, so the same failure lands on the existing file.

    Reproduced before the fix: a valid `Discovery` became a
    `{"error": "No data found"}` by re-running create for one topic and
    having the model return rubbish.
    """

    def _config(self, ep):
        return ep.OperationConfig(
            atom_name="discovery", model_version="claude-sonnet-5", base_system_text="t",
        )

    def _generate(self, ep, monkeypatch, fake_call_factory, payload):
        from tests.test_operations import use_fake

        use_fake(ep, monkeypatch, fake_call_factory, [payload])
        return ep.create_atom(self._config(ep), "Papyrus")

    def test_a_failed_regeneration_leaves_the_existing_atom_intact(
        self, ep, workspace, monkeypatch, fake_call_factory
    ):
        from universe_builder import store
        from tests.test_operations import valid_discovery

        self._generate(ep, monkeypatch, fake_call_factory, valid_discovery())
        path = next(iter(store.iter_atoms("discovery")))
        good = path.read_text()

        self._generate(ep, monkeypatch, fake_call_factory, '{"garbage": true}')

        assert path.read_text() == good, "a failed regeneration destroyed a valid atom"

    def test_the_failed_regeneration_still_reports_invalid(
        self, ep, workspace, monkeypatch, fake_call_factory
    ):
        from tests.test_operations import valid_discovery

        self._generate(ep, monkeypatch, fake_call_factory, valid_discovery())
        result = self._generate(ep, monkeypatch, fake_call_factory, '{"garbage": true}')

        assert result.outcome is Outcome.INVALID

    def test_a_failed_regeneration_leaves_no_sentinel_beside_the_atom(
        self, ep, workspace, monkeypatch, fake_call_factory
    ):
        """This used to assert the opposite — the failed payload "queued as
        evidence" beside the valid atom. That sentinel was a standing order for
        the fix pass to repair good data from garbage, and `commit_fix` would
        have written the repair over the atom. The journal holds the evidence;
        the disk holds nothing but the atom."""
        from universe_builder import journal, store
        from tests.test_operations import valid_discovery

        journal.start_run(config={"test": True})
        self._generate(ep, monkeypatch, fake_call_factory, valid_discovery())
        path = next(iter(store.iter_atoms("discovery")))

        self._generate(ep, monkeypatch, fake_call_factory, '{"garbage": true}')

        assert not store.error_path(path).exists(), (
            "a sentinel beside a valid atom queues its destruction"
        )
        assert journal.needs_fix_across_runs() == [], (
            "the journal queued a valid atom for repair"
        )
        entries = journal.current().entries()
        assert any(e.get("raw_payload_sha256") for e in entries
                   if e.get("validation_status") == "failed"), (
            "the failed payload must survive in the journal"
        )

    def test_a_fresh_failure_writes_nothing_at_the_atom_path(
        self, ep, workspace, monkeypatch, fake_call_factory
    ):
        """The `EmptyModel` placeholder is gone: it was dead weight wearing an
        atom's name — the corpus held some that had to be purged — and the
        `_ERROR` sentinel plus the journal already record the failure."""
        from universe_builder import store

        self._generate(ep, monkeypatch, fake_call_factory, '{"garbage": true}')

        assert list(store.iter_atoms("discovery")) == [], (
            "a failure wrote something at the atom's own path"
        )
        sentinels = list(store.iter_atoms("discovery", include_errors=True))
        assert len(sentinels) == 1 and store.is_error_path(sentinels[0]), (
            "a fresh failure must still leave its sentinel to queue the fix"
        )

    def test_a_successful_regeneration_retires_a_stale_sentinel(
        self, ep, workspace, monkeypatch, fake_call_factory
    ):
        """The topic generates cleanly now, so the queue entry is resolved."""
        from universe_builder import store
        from tests.test_operations import valid_discovery

        self._generate(ep, monkeypatch, fake_call_factory, '{"garbage": true}')
        sentinel = next(iter(store.iter_atoms("discovery", include_errors=True)))
        assert store.is_error_path(sentinel)
        path = store.target_of_error(sentinel)

        self._generate(ep, monkeypatch, fake_call_factory, valid_discovery())

        assert not store.error_path(path).exists()
        assert not is_empty(path)


def is_empty(path) -> bool:
    from universe_builder.store import is_empty_model

    return is_empty_model(path.read_text())


class TestConcurrentGenerationNeverDestroysAValidAtom:
    """B12: `persist` must not rely on a check that a concurrent writer invalidates.

    Two calls for one topic resolve to one content-addressed path, and the create
    pass can genuinely see a topic twice — two `History` files may list the same
    city. With `concurrency > 1` the two run in parallel, so:

        thread B (failing)    sees no file at the path
        thread A (succeeding) writes a valid atom
        thread B              writes its placeholder over it

    The fix is that the placeholder write is create-only (`O_EXCL`), so the
    decision and the write are one atomic step and that ordering cannot arise.
    """

    def _config(self, ep):
        return ep.OperationConfig(
            atom_name="discovery", model_version="claude-sonnet-5", base_system_text="t",
        )

    def test_a_placeholder_never_replaces_an_existing_file(self, ep, workspace, monkeypatch,
                                                           fake_call_factory):
        """The property, tested directly rather than through a racy schedule."""
        from universe_builder import store
        from tests.test_operations import use_fake, valid_discovery

        use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery()])
        ep.create_atom(self._config(ep), "Papyrus")
        path = next(iter(store.iter_atoms("discovery")))
        good = path.read_text()

        # Exactly what a concurrent writer produces: the existence check is stale.
        monkeypatch.setattr(type(path), "exists", lambda self: False)

        use_fake(ep, monkeypatch, fake_call_factory, ['{"garbage": true}'])
        ep.create_atom(self._config(ep), "Papyrus")

        assert path.read_text() == good, (
            "persist relied on an exists() check that a concurrent writer can invalidate"
        )

    def test_write_new_refuses_to_replace(self, workspace):
        from universe_builder import store

        path = workspace / "city" / "x.json"
        store.write(path, "original")

        assert store.write_new(path, "replacement") is False
        assert path.read_text() == "original"

    def test_write_new_creates_when_absent(self, workspace):
        from universe_builder import store

        path = workspace / "city" / "x.json"

        assert store.write_new(path, "created") is True
        assert path.read_text() == "created"

    def test_write_new_creates_parent_directories(self, workspace):
        from universe_builder import store

        path = workspace / "deep" / "nested" / "x.json"
        assert store.write_new(path, "created") is True

    def test_only_one_of_many_racing_creators_wins(self, workspace):
        """The guarantee `persist` leans on, under real contention."""
        import threading

        from universe_builder import store

        path = workspace / "city" / "contended.json"
        winners = []
        lock = threading.Lock()
        start = threading.Barrier(16)

        def race(n):
            start.wait()
            if store.write_new(path, f"writer-{n}"):
                with lock:
                    winners.append(n)

        threads = [threading.Thread(target=race, args=(n,)) for n in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(winners) == 1, f"{len(winners)} writers all believed they created the file"
        assert path.read_text() == f"writer-{winners[0]}"

    def test_parallel_creates_of_one_topic_keep_the_valid_atom(self, ep, workspace, monkeypatch,
                                                               fake_call_factory):
        """End to end: one success and several failures, all racing on one path.

        This passes with the racy check-then-write too, because the per-path lock
        serialises them inside one process — so it is not evidence for `O_EXCL`.
        `test_a_placeholder_never_replaces_an_existing_file` is; it fails against
        the racy version. What this covers is that the two mechanisms compose,
        and that a genuine parallel run leaves one file rather than a sentinel.
        """
        import threading

        from universe_builder import store
        from universe_builder.store import is_empty_model
        from tests.test_operations import valid_discovery

        payloads = [valid_discovery()] + ['{"garbage": true}'] * 7
        queue = list(payloads)
        queue_lock = threading.Lock()

        from universe_builder.llm_classes.llm_call import LLMCall

        class Fake(LLMCall):
            def __init__(self, **kwargs):
                for key in ("augment_to_json", "topic", "user_instructions"):
                    kwargs.pop(key, None)
                super().__init__(**kwargs)
                self.llm_response_json = self.call_llm_api()

            def call_llm_api(self):
                with queue_lock:
                    return queue.pop(0)

        monkeypatch.setitem(ep.llm_call_mapping, "claude-sonnet-5", Fake)

        ep.call_and_save_prompt(self._config(ep), ["Papyrus"] * len(payloads), concurrency=8)

        written = list(store.iter_atoms("discovery"))
        assert len(written) == 1, "one topic must produce one file"
        assert not is_empty_model(written[0].read_text()), (
            "a failing parallel call overwrote the valid atom with a sentinel"
        )
