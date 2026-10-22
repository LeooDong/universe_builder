"""The journal, wired into the passes.

It was built, tested and left inert one morning, on the grounds that
B8 - "a failed fix destroys its own evidence" - had been closed far more cheaply
by `store.commit_fix`. By the end of the same day B8 had recurred **four** times,
each through a different write path:

  1. the original fix pass deleting the sentinel it had just read
  2. `persist` writing an `EmptyModel` over the atom's own path
  3. batch `collect` writing a repair to the sentinel, then clearing it
  4. batch `collect` replacing the sentinel with the repair's own failure

The pattern is structural, not four coincidences: the `_ERROR` sentinel is both
the work queue and the only copy of the evidence, and every new write path
eventually writes over it. The journal is append-only, so no later write can
destroy an earlier payload - a sentinel lost to a fifth route becomes an
inconvenience rather than the end of the record.

These tests cover the *wiring*. `test_journal.py` covers the module itself.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from universe_builder import journal, settings, store
from universe_builder.providers import batch as batch_mod
from universe_builder.providers.types import BatchHandle, LLMResponse, Usage


@pytest.fixture
def run(workspace):
    """A started journal run, torn down so it cannot leak into another test."""
    j = journal.start_run(config={"test": True})
    yield j
    journal.end_run()


def _handle(target=None, subject="Memphis"):
    routing = {"city-00000-memphis": {"atom_type": "city", "subject": subject}}
    if target is not None:
        routing["city-00000-memphis"]["target"] = target
    return BatchHandle(
        provider="openai", batch_id="b", model="gpt-4o-mini", submitted_at="x",
        request_count=1, routing=routing,
        extra={"schema_model": "City", "schema_id": "universe_builder.domains.history.models.city:City"},
    )


def _collect(handle, response):
    path = settings.data_dir() / "handle.json"
    path.write_text(json.dumps(handle.to_dict()), encoding="utf-8")
    provider = MagicMock()
    provider.fetch_batch_results.return_value = iter([response])
    with patch.object(batch_mod, "get_provider", return_value=provider):
        return batch_mod.collect(path)


class TestWithNoRunStarted:
    """The library must stay usable outside a run, and tests must not need one."""

    def test_current_is_a_null_journal(self):
        journal.end_run()
        assert isinstance(journal.current(), journal.NullJournal)

    def test_recording_is_a_no_op_not_an_error(self):
        journal.end_run()
        assert journal.current().record_attempt(work_item_id="x", raw_payload="y") is None

    def test_the_null_journal_still_requires_the_payload(self):
        """The one property the module exists for is not optional in the stub.

        A `NullJournal` that accepted `record_attempt()` with no `raw_payload`
        would let a call site forget the argument and still pass its tests.
        """
        journal.end_run()
        with pytest.raises(TypeError):
            journal.current().record_attempt(work_item_id="x")


class TestTheBatchPathRecords:
    def test_a_failure_records_its_raw_payload(self, run):
        raw = '{"name": {"common": "Memphis"}, "truncated mid-'

        _collect(_handle(), LLMResponse(raw, None, "gpt-4o-mini", Usage(), "city-00000-memphis", "bad"))

        entries = run.entries()
        assert len(entries) == 1
        assert entries[0]["work_status"] == journal.NEEDS_FIX
        assert run.read_raw(entries[0]["attempt_id"]) == raw

    def test_a_success_leaves_nothing_queued(self, run, city_payload):
        from universe_builder.domains.history.models.city import City

        _collect(_handle(), LLMResponse(json.dumps(city_payload),
                                        City.model_validate(city_payload),
                                        "gpt-4o-mini", Usage(10, 20), "city-00000-memphis"))

        assert run.needs_fix() == []
        assert run.entries()[0]["work_status"] == journal.OK

    def test_usage_and_batch_identity_are_recorded(self, run, city_payload):
        from universe_builder.domains.history.models.city import City

        _collect(_handle(), LLMResponse(json.dumps(city_payload),
                                        City.model_validate(city_payload),
                                        "gpt-4o-mini", Usage(101, 902), "city-00000-memphis"))

        entry = run.entries()[0]
        assert entry["input_tokens"] == 101 and entry["output_tokens"] == 902
        assert entry["batch_id"] == "b"
        assert entry["execution_mode"] == "batch"

    def test_the_routed_operation_is_recorded(self, run, city_payload):
        """The handle routes each request's operation; the entry must carry it,
        or `costs` cannot tell a batched create from a batched fix or verify —
        all three write the base atom type with the base schema."""
        from universe_builder.domains.history.models.city import City

        handle = _handle()
        handle.routing["city-00000-memphis"]["operation"] = "verify"
        _collect(handle, LLMResponse(json.dumps(city_payload),
                                     City.model_validate(city_payload),
                                     "gpt-4o-mini", Usage(1, 2), "city-00000-memphis"))

        assert run.entries()[0]["operation"] == "verify"


class TestTheEvidenceOutlivesTheSentinel:
    """The whole point. Four write paths have destroyed a sentinel so far."""

    RAW = '{"name": {"common": "Memphis"}, "truncated mid-'

    def test_the_payload_survives_the_sentinel_being_deleted(self, run):
        atom = store.dir_for("city") / "city_h1111111111111111_memphis_gpt-4o-mini.json"
        atom.parent.mkdir(parents=True, exist_ok=True)
        _collect(
            _handle(target=str(atom.relative_to(settings.data_dir()))),
            LLMResponse(self.RAW, None, "gpt-4o-mini", Usage(), "city-00000-memphis", "bad"),
        )
        sentinel = store.error_path(atom)
        assert sentinel.exists()

        sentinel.unlink()          # whichever route destroys it

        queued = run.needs_fix()
        assert [e["subject"] for e in queued] == ["Memphis"]
        assert run.read_raw(queued[0]["attempt_id"]) == self.RAW

    def test_the_item_stays_queued_without_its_sentinel(self, run):
        """The queue is replayed from entries, not globbed off the filesystem."""
        _collect(_handle(), LLMResponse(self.RAW, None, "gpt-4o-mini", Usage(),
                                        "city-00000-memphis", "bad"))

        for path in store.dir_for("city").glob("*_ERROR"):
            path.unlink()

        assert len(run.needs_fix()) == 1


class TestTheQueueCycles:
    def test_a_later_success_dequeues_the_item(self, run, city_payload):
        """needs_fix -> ok, replayed in order rather than filtered."""
        from universe_builder.domains.history.models.city import City

        _collect(_handle(), LLMResponse("bad", None, "gpt-4o-mini", Usage(),
                                        "city-00000-memphis", "nope"))
        assert len(run.needs_fix()) == 1

        _collect(_handle(), LLMResponse(json.dumps(city_payload),
                                        City.model_validate(city_payload),
                                        "gpt-4o-mini", Usage(), "city-00000-memphis"))

        assert run.needs_fix() == []
        assert len(run.entries()) == 2, "the failure must still be on record"

    def test_failure_after_success_keeps_evidence_without_requeuing_good_data(self, run, city_payload):
        """A later failed generation must not queue a successful atom for destruction."""
        from universe_builder.domains.history.models.city import City

        _collect(_handle(), LLMResponse(json.dumps(city_payload),
                                        City.model_validate(city_payload),
                                        "gpt-4o-mini", Usage(), "city-00000-memphis"))
        _collect(_handle(), LLMResponse("regressed", None, "gpt-4o-mini", Usage(),
                                        "city-00000-memphis", "nope"))

        assert run.needs_fix() == []
        assert run.read_raw(run.entries()[-1]["attempt_id"]) == "regressed"
        target = store.path_for("city", None, "Memphis", "gpt-4o-mini")
        assert target.exists()
        assert not store.error_path(target).exists()


class TestJournallingNeverBreaksARun:
    def test_a_broken_journal_does_not_lose_a_collected_result(self, run, city_payload):
        """Observation must not cost the run the thing it was observing."""
        from universe_builder.domains.history.models.city import City

        atom = store.dir_for("city") / "city_h1111111111111111_memphis_gpt-4o-mini.json"
        atom.parent.mkdir(parents=True, exist_ok=True)
        boom = MagicMock(side_effect=OSError("disk full"))

        with patch.object(journal.Journal, "record_attempt", boom):
            report = _collect(
                _handle(target=str(atom.relative_to(settings.data_dir()))),
                LLMResponse(json.dumps(city_payload), City.model_validate(city_payload),
                            "gpt-4o-mini", Usage(), "city-00000-memphis"),
            )

        assert report.written == 1
        assert atom.exists(), "a journal failure destroyed a perfectly good result"


class TestTheSynchronousPathRecords:
    """`operations._support.result_from_call` is the one place every synchronous
    operation passes through with both the raw payload and the verdict in hand."""

    MODEL = "claude-sonnet-5"

    def _fake(self, ep, monkeypatch, fake_call_factory, responses):
        """Register a scripted call under the *same* key the config will look up.

        Written out here rather than imported from `test_operations`, and the
        model named once: registering the fake under a different key than
        `model_version` is how a pipeline test falls through to the real provider.
        That happened while writing this file — the test passed, took 56 seconds,
        and made a live API call. `conftest.no_live_calls` now makes that an
        immediate failure rather than a slow surprise.
        """
        Fake = fake_call_factory(responses)
        monkeypatch.setitem(ep.llm_call_mapping, self.MODEL, Fake)
        return Fake

    def test_a_failed_create_records_its_payload(self, run, ep, monkeypatch,
                                                 fake_call_factory):
        self._fake(ep, monkeypatch, fake_call_factory, ['{"not": "a discovery"'])
        ep.call_and_save_prompt(
            config=ep.OperationConfig(atom_name="discovery", model_version=self.MODEL,
                                      base_system_text="t"),
            topic_list=["Papyrus"],
        )

        queued = run.needs_fix()
        assert [e["subject"] for e in queued] == ["Papyrus"]
        assert run.read_raw(queued[0]["attempt_id"]) == '{"not": "a discovery"'

    def test_a_successful_create_is_recorded_and_not_queued(self, run, ep, monkeypatch,
                                                            fake_call_factory, discovery_payload):
        self._fake(ep, monkeypatch, fake_call_factory, [json.dumps(discovery_payload)])
        ep.call_and_save_prompt(
            config=ep.OperationConfig(atom_name="discovery", model_version=self.MODEL,
                                      base_system_text="t"),
            topic_list=["Papyrus"],
        )

        assert run.needs_fix() == []
        assert [e["work_status"] for e in run.entries()] == [journal.OK]

    def test_the_work_item_is_shared_across_passes(self, run):
        """A fix attempt and the create attempt it repairs are one work item.

        Keyed on the content id, not the filename - one of them writes an
        `_ERROR` name and the other does not, and the id has to survive the
        corpus rename either way.
        """
        assert journal.work_item_id("city", "Memphis") == journal.work_item_id("city", "memphis")
        assert journal.work_item_id("city", "Memphis") != journal.work_item_id("city", "Thebes")
        assert journal.work_item_id("city", "Memphis") != journal.work_item_id("person", "Memphis")


class TestNothingArrivedIsRecordedAsNothing:
    """"The model returned nothing" and "we never heard back" are different claims.

    `record_attempt` refuses `raw_payload=None` without a reason precisely to keep
    them apart. The first wiring smuggled the confusion past that check:
    `Result.raw_text` defaults to `""`, and passing it straight through recorded
    `raw_payload_available: True` — a claim to hold bytes that do not exist —
    beside a reason saying nothing came back.
    """

    MODEL = "claude-sonnet-5"

    def _exploding(self):
        # What the call adapter raises once every transport attempt has failed
        # (`ProviderCall`, v0.1.20). Any other exception out of a call's
        # constructor is a setup fault, journalled as `call_setup`.
        from universe_builder.llm_classes.llm_call import LLMCall
        from universe_builder.providers.errors import TransportFailed

        class Exploding(LLMCall):
            def __init__(self, **kwargs):
                raise TransportFailed("connection reset before the first byte")

        return Exploding

    def test_an_empty_response_is_not_claimed_as_a_payload(self, run, ep, monkeypatch,
                                                           fake_call_factory):
        Fake = fake_call_factory([""])
        monkeypatch.setitem(ep.llm_call_mapping, self.MODEL, Fake)

        ep.call_and_save_prompt(
            config=ep.OperationConfig(atom_name="discovery", model_version=self.MODEL,
                                      base_system_text="t"),
            topic_list=["Papyrus"],
        )

        entry = run.entries()[-1]
        assert entry["raw_payload_available"] is False, (
            "an empty response was recorded as bytes we hold; the reason field then "
            "contradicts the availability flag"
        )
        assert entry.get("raw_payload_unavailable_reason")

    def test_a_transport_failure_is_recorded_and_queued(self, run, ep, monkeypatch):
        """A call that died before its first byte left no record at all.

        That is the worst thing for the queue to be wrong about: a transport
        failure is exactly the case a retry fixes, and the atom was invisible to
        both the journal and `needs_fix()`.
        """
        monkeypatch.setitem(ep.llm_call_mapping, self.MODEL, self._exploding())

        results = ep.call_and_save_prompt(
            config=ep.OperationConfig(atom_name="discovery", model_version=self.MODEL,
                                      base_system_text="t"),
            topic_list=["Papyrus"],
        )

        assert results[0].outcome.value == "failed"
        assert [e["subject"] for e in run.needs_fix()] == ["Papyrus"]
        entry = run.entries()[-1]
        assert entry["validation_stage"] == "transport"
        assert entry["retryable"] is True
        assert entry["raw_payload_available"] is False
        assert "connection reset" in entry["raw_payload_unavailable_reason"]

    @pytest.mark.parametrize("phase", ["fix", "verify"])
    def test_every_operation_records_its_transport_failures(self, run, ep, monkeypatch,
                                                            workspace, phase):
        """All four operations return `Result.failed` before `result_from_call`.

        Recording and queueing are different questions. Both phases record the
        attempt — that is what this test pins — but only `fix` queues it: a
        verify's target is still a valid atom, and now that failures land on
        the *real* work item, queueing one would send the fix
        pass to overwrite good data. Verify's retry is verify, run again.
        """
        from universe_builder import store

        atom = store.dir_for("discovery") / f"discovery_h1111111111111111_papyrus_{self.MODEL}.json"
        atom.parent.mkdir(parents=True, exist_ok=True)
        if phase == "fix":
            store.write_error(atom, '{"broken":')
        else:
            atom.write_text('{"name": {"common": "Papyrus"}}', encoding="utf-8")

        monkeypatch.setitem(ep.llm_call_mapping, (self.MODEL, "augment"), self._exploding())
        ep.augment_and_save_prompt(
            ep.OperationConfig(atom_name="discovery", model_version=(self.MODEL, "augment"),
                               base_system_text="t", operation_type=phase),
        )

        entry = run.entries()[-1]
        assert entry["validation_stage"] == "transport", (
            f"{phase} transport failure left no journal entry"
        )
        if phase == "fix":
            assert run.needs_fix(), "a failed fix must stay queued"
        else:
            assert not run.needs_fix(), "a failed verify queued a valid atom for repair"


class TestTheFixQueueServesTheUnion:
    """`reconcile_fix_queue`: the journal is the queue's authority, the glob
    its serving mechanism. A sentinel lost to a stray write used to drop its
    repair silently; the journal now restores it before the pass walks."""

    MODEL = "claude-sonnet-5"

    def _fail_create(self, ep, monkeypatch, fake_call_factory, subject="Papyrus"):
        from tests.test_operations import use_fake

        use_fake(ep, monkeypatch, fake_call_factory, ['{"garbage": true}'])
        ep.create_atom(
            ep.OperationConfig(atom_name="discovery", model_version=self.MODEL,
                               base_system_text="t"),
            subject,
        )

    def test_a_lost_sentinel_is_restored_from_the_journal(
            self, run, ep, workspace, monkeypatch, fake_call_factory):
        from universe_builder import store
        from universe_builder.operations._support import reconcile_fix_queue

        self._fail_create(ep, monkeypatch, fake_call_factory)
        sentinel = next(p for p in (workspace / "discovery").iterdir()
                        if store.is_error_path(p))
        payload = sentinel.read_text(encoding="utf-8")
        sentinel.unlink()   # B8's fifth route: the stray write that loses it

        assert reconcile_fix_queue("discovery") == 1
        assert sentinel.exists(), "the journal held the payload; the glob gets it back"
        assert sentinel.read_text(encoding="utf-8") == payload

    def test_a_restored_sentinel_gets_fixed_by_the_normal_pass(
            self, run, ep, workspace, monkeypatch, fake_call_factory):
        """End to end: lose the sentinel, run the fix pass, get the atom."""
        from universe_builder import store
        from tests.test_operations import use_fake, valid_discovery

        self._fail_create(ep, monkeypatch, fake_call_factory)
        sentinel = next(p for p in (workspace / "discovery").iterdir()
                        if store.is_error_path(p))
        sentinel.unlink()

        use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery()], augment=True)
        ep.augment_and_save_prompt(ep.OperationConfig(
            atom_name="discovery", model_version=(self.MODEL, "augment"),
            base_system_text="t", operation_type="fix"))

        atoms = list(store.iter_atoms("discovery"))
        assert len(atoms) == 1 and not store.is_error_path(atoms[0])

    def test_nothing_is_restored_when_the_atom_already_exists(
            self, run, ep, workspace, monkeypatch, fake_call_factory):
        """The atom appearing settles the matter; a restored sentinel beside
        it would queue its destruction."""
        from universe_builder import store
        from tests.test_operations import use_fake, valid_discovery
        from universe_builder.operations._support import reconcile_fix_queue

        self._fail_create(ep, monkeypatch, fake_call_factory)
        next(p for p in (workspace / "discovery").iterdir()
             if store.is_error_path(p)).unlink()
        use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery()])
        ep.create_atom(
            ep.OperationConfig(atom_name="discovery", model_version=self.MODEL,
                               base_system_text="t"),
            "Papyrus",
        )

        assert reconcile_fix_queue("discovery") == 0
        assert not any(store.is_error_path(p)
                       for p in (workspace / "discovery").iterdir())

    def test_a_transport_failure_restores_nothing(self, run, workspace):
        """No payload was ever recorded; there is nothing to repair from, and
        the create pass owns the regeneration."""
        from universe_builder.operations._support import record_failure, reconcile_fix_queue

        record_failure("discovery", "Papyrus", "socket closed",
                       requested_model=self.MODEL)

        assert reconcile_fix_queue("discovery") == 0


class TestNoSentinelEverSitsBesideAValidAtom:
    """A closed item: a failing call racing a succeeding one could
    land its sentinel after the success cleared it, queueing the destruction
    of a valid atom. `store.write_error` now refuses under the path lock."""

    def test_write_error_refuses_when_the_atom_exists(self, workspace):
        from universe_builder import store

        path = store.path_for("discovery", None, "Papyrus", "m")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"real": "atom"}', encoding="utf-8")

        assert store.write_error(path, '{"garbage": true}') is None
        assert not store.error_path(path).exists()

    def test_write_error_still_writes_when_nothing_is_there(self, workspace):
        from universe_builder import store

        path = store.path_for("discovery", None, "Papyrus", "m")

        assert store.write_error(path, '{"garbage": true}') is not None
        assert store.error_path(path).read_text(encoding="utf-8") == '{"garbage": true}'


class TestBothFailurePathsAgreeOnTheWorkItem:
    """A transport failure and a validation failure must name the *same* work item.

    `record_failure` is handed an atom type by each operation; `result_from_call`
    takes it from the call's own `atom_type`. If the two disagree, one atom gets
    two work-item ids depending on *how* it failed — so a retry queued under one
    never settles the other, and `needs_fix()` reports phantom work forever.

    `append` got this wrong: it passed `config.atom_name` ("city") while its call
    uses `config.augment_atom_name` ("append_city"), filing a failed append
    against the city it supplements. A good city would have been queued for
    repair and the append that actually failed would not have been.
    """

    #: operation -> the attribute of `OperationConfig` its LLMCall passes as
    #: `atom_type`. Checked against the source, because this is exactly the pair
    #: that drifted.
    EXPECTED = {
        "create": "atom_name",
        "fix": "atom_name",
        "verify": "atom_name",
        "append": "augment_atom_name",
        "expand": "augment_atom_name",
    }

    @pytest.mark.parametrize("operation", sorted(EXPECTED))
    def test_record_failure_names_what_the_call_names(self, operation):
        import inspect
        import re

        from universe_builder.operations import append, create, expand, fix, verify

        module = {"create": create, "fix": fix, "verify": verify,
                  "append": append, "expand": expand}[operation]
        source = inspect.getsource(module)

        call_type = re.search(r"atom_type=config\.(\w+)", source)
        recorded = re.search(r"record_failure\(\s*config\.(\w+)", source)

        assert call_type, f"{operation} no longer passes atom_type=config.*"
        assert recorded, f"{operation} does not journal its transport failures"
        assert call_type.group(1) == recorded.group(1) == self.EXPECTED[operation], (
            f"{operation}: the call uses config.{call_type.group(1)} but the "
            f"journal records config.{recorded.group(1)}; one atom would get two "
            f"work-item ids depending on how it failed"
        )

    def test_every_operation_journals_its_transport_failures(self):
        """Expand was missed when the other four were wired."""
        import inspect

        from universe_builder.operations import append, create, expand, fix, verify

        for name, module in (("create", create), ("fix", fix), ("verify", verify),
                             ("append", append), ("expand", expand)):
            assert "record_failure(" in inspect.getsource(module), (
                f"{name} catches transport errors without recording them; a call "
                f"that died before its first byte leaves no record at all"
            )

    def test_an_append_failure_is_not_filed_against_the_city(self, run):
        """The consequence, not just the spelling."""
        from universe_builder.operations._support import record_failure

        record_failure("append_city", "Memphis", "connection reset", operation="append")

        queued = run.needs_fix()
        assert len(queued) == 1
        assert queued[0]["work_item_id"] == journal.work_item_id("append_city", "Memphis")
        assert queued[0]["work_item_id"] != journal.work_item_id("city", "Memphis"), (
            "the failed append is queued against the city itself"
        )


class TestTheSyncPathRecordsUsage:
    """Token counts reached the journal from the batch path and nowhere else.

    `providers/batch.py` passed `input_tokens`/`output_tokens`; the synchronous
    path never did, so every sync run recorded `usage_available: false` and
    **no cost figure for a sync run could be measured** — only estimated from
    whatever batch data predated it. The provider measured usage correctly all
    along and the journal had fields for it; nothing carried the numbers between
    the two.
    """

    class _Call:
        atom_type = "city"
        model_version = "gpt-5-nano"
        response_path = None

        def __init__(self, usage=None):
            self.usage = usage

    def test_no_usage_reports_nothing_rather_than_zero(self):
        """A call whose transport measured nothing must record 'unknown', not
        'free'. The journal derives `usage_available` from the keys' presence,
        so zeros would read as an authoritative claim that it cost nothing."""
        from universe_builder.operations import _support

        assert _support._usage_fields(self._Call(usage=None)) == {}

    def test_measured_usage_is_passed_through(self):
        from universe_builder.operations import _support

        fields = _support._usage_fields(self._Call(Usage(input_tokens=811,
                                                         output_tokens=6725)))

        assert fields["input_tokens"] == 811
        assert fields["output_tokens"] == 6725

    def test_a_zero_count_is_omitted_not_asserted(self):
        from universe_builder.operations import _support

        fields = _support._usage_fields(self._Call(Usage(input_tokens=0,
                                                         output_tokens=42)))

        assert "input_tokens" not in fields
        assert fields["output_tokens"] == 42

    def test_a_recorded_sync_attempt_carries_usage(self, run):
        """End to end: the entry the sync path writes must say usage is
        available, which is the flag every cost report keys off."""
        from universe_builder.operations import _support
        from universe_builder.operations.types import Outcome, Result

        call = self._Call(Usage(input_tokens=100, output_tokens=200))
        _support._record(call, Result(Outcome.FAILED, raw_text='{"a": 1}'),
                         dict, subject="Memphis", atom_type="city",
                         operation="create", phase="create")

        entries = journal.read_entries(journal.current().run_id)
        latest = [e for e in entries if e.get("subject") == "Memphis"][-1]
        assert latest["usage_available"] is True
        assert latest["input_tokens"] == 100
        assert latest["output_tokens"] == 200

    def test_an_unmeasured_sync_attempt_says_so(self, run):
        from universe_builder.operations import _support
        from universe_builder.operations.types import Outcome, Result

        _support._record(self._Call(usage=None),
                         Result(Outcome.FAILED, raw_text='{"a": 1}'),
                         dict, subject="Thebes", atom_type="city",
                         operation="create", phase="create")

        entries = journal.read_entries(journal.current().run_id)
        latest = [e for e in entries if e.get("subject") == "Thebes"][-1]
        assert latest["usage_available"] is False


class TestCachedTokensAreRecorded:
    """Cached prompt tokens are **billable**, at `cached_input` rather than
    `input` (`providers/pricing.py:81`), and `openai_provider._usage` subtracts
    them *out* of `input_tokens` to keep the two rates apart. A journal that
    records only input and output therefore drops them entirely — and this
    corpus puts its few-shot examples in a cached prefix deliberately, so that
    is not a rounding error. Both the sync and batch paths had this bug.
    """

    class _Call:
        atom_type = "city"
        model_version = "gpt-5-nano"
        response_path = None

        def __init__(self, usage):
            self.usage = usage

    def test_cache_read_tokens_survive_to_the_journal(self):
        from universe_builder.operations import _support

        fields = _support._usage_fields(self._Call(
            Usage(input_tokens=400, output_tokens=8000, cache_read_tokens=1600)))

        assert fields["cache_read_tokens"] == 1600

    def test_the_cost_estimate_charges_them(self):
        """The point of keeping them: they are cheaper than fresh input, but
        they are not free, and dropping them understates every total."""
        from universe_builder.providers import pricing

        with_cache = pricing.estimate_cost(
            Usage(input_tokens=400, output_tokens=8000, cache_read_tokens=1600),
            "gpt-5-nano")
        without = pricing.estimate_cost(
            Usage(input_tokens=400, output_tokens=8000), "gpt-5-nano")

        assert with_cache > without

    def test_an_estimated_cost_is_stamped_at_write_time(self):
        """The field existed on `JournalEntry` from the start and was never
        populated, so 'what did this cost' always meant recomputing against a
        price table that may since have moved."""
        from universe_builder.operations import _support

        fields = _support._usage_fields(self._Call(
            Usage(input_tokens=400, output_tokens=8000)))

        assert fields["estimated_cost"] > 0

    def test_a_call_with_only_cached_input_still_counts_as_measured(self, run):
        """`usage_available` keys off the presence of token fields. Checking
        only input/output would call a fully-cached prompt 'unmeasured'."""
        from universe_builder.operations import _support
        from universe_builder.operations.types import Outcome, Result

        _support._record(self._Call(Usage(cache_read_tokens=900)),
                         Result(Outcome.FAILED, raw_text='{"a": 1}'),
                         dict, subject="Cached", atom_type="city",
                         operation="create", phase="create")

        entries = journal.read_entries(journal.current().run_id)
        latest = [e for e in entries if e.get("subject") == "Cached"][-1]
        assert latest["usage_available"] is True
        assert latest["cache_read_tokens"] == 900
