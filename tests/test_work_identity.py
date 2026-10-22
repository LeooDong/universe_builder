"""One piece of work, one identity — wherever the attempt runs.

Two sibling defects, found on the first live batched-fix cycle:

* The repair's success was journaled under a slug-derived subject
  (`boutros boutrosghali`) while the failure it answered was queued under the
  true one (`Boutros Boutros-Ghali`), so the queue kept a repaired item.
* Augment calls minted a fresh identity from that same lossy slug, so a failed
  fix wrote its `_ERROR` under a name nothing owned — a phantom in the fix
  queue — and an append's id never matched the city it supplements.

The rule now: work that targets an existing file takes its identity *from the
target*, whose filename embeds the true content id.

No network: every call goes through `fake_call_factory`.
"""

from __future__ import annotations

import json

from universe_builder import journal, store
from universe_builder.operations._support import record_failure

SUBJECT = "Boutros Boutros-Ghali"      # slugs to boutros_boutros_ghali; lossy
MODEL = "claude-sonnet-5"


def valid_discovery(name="Papyrus") -> str:
    from tests.test_operations import valid_discovery as _canned

    payload = json.loads(_canned())
    payload["name"]["common"] = name
    return json.dumps(payload)


def use_fake(ep, monkeypatch, fake_call_factory, responses, augment=True):
    Fake = fake_call_factory(responses)
    key = (MODEL, "augment") if augment else MODEL
    monkeypatch.setitem(ep.llm_call_mapping, key, Fake)
    return Fake


class TestWorkItemFromPath:
    def test_a_content_addressed_name_yields_the_embedded_id(self):
        path = store.path_for("person", None, SUBJECT, MODEL)

        assert (journal.work_item_from_path("person", path)
                == journal.work_item_id("person", SUBJECT))

    def test_the_slug_alone_would_not_have(self):
        """The reason this function exists: the slug hashes to a different id."""
        slugged = "boutros boutros ghali"

        assert (journal.work_item_id("person", slugged)
                != journal.work_item_id("person", SUBJECT))

    def test_an_error_decoration_is_stripped(self):
        path = store.error_path(store.path_for("person", None, SUBJECT, MODEL))

        assert (journal.work_item_from_path("person", path)
                == journal.work_item_id("person", SUBJECT))

    def test_a_legacy_sequential_name_is_none(self):
        assert journal.work_item_from_path(
            "person", "person_0000000001_papyrus_gpt-4.json") is None

    def test_garbage_is_none(self):
        assert journal.work_item_from_path("person", "notes.txt") is None


class TestARepairClearsTheFailureItAnswers:
    """The observed defect: submit queued the true id, collect settled the slug's."""

    def test_batched_fix_success_settles_the_original_queue_entry(self, workspace):
        from universe_builder.providers import batch as batch_mod
        from universe_builder.providers.types import BatchHandle, LLMResponse, Usage

        journal.start_run(config={"test": True})
        record_failure("person", SUBJECT, "did not validate")
        assert journal.needs_fix_across_runs(), "precondition: the failure is queued"

        target = store.path_for("person", None, SUBJECT, "gpt-5.6-luna")
        handle = BatchHandle(
            provider="openai", batch_id="b1", model="gpt-5.6-luna",
            submitted_at="t", request_count=1,
            routing={"person-00000-slug": {
                "atom_type": "person",
                # What the submitter could recover from the filename: the slug.
                "subject": "boutros boutrosghali",
                "target": target.name,
                "operation": "fix",
            }},
        )
        response = LLMResponse('{"ok": 1}', object(), "gpt-5.6-luna",
                               Usage(), "person-00000-slug")
        batch_mod._journal_result(handle, response, target)

        assert journal.needs_fix_across_runs() == [], (
            "the repair was journaled under a different work item than the "
            "failure it answered, so the queue kept a repaired item"
        )

    def test_a_sync_fix_lands_on_the_same_item_as_the_create_failure(
            self, ep, workspace, monkeypatch, fake_call_factory):
        journal.start_run(config={"test": True})
        record_failure("discovery", "Ptolemy's Almagest", "did not validate")

        sentinel = store.error_path(
            store.path_for("discovery", None, "Ptolemy's Almagest", MODEL))
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text('{"incomplete": true}', encoding="utf-8")
        use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery("Almagest")])

        ep.augment_and_save_prompt(ep.OperationConfig(
            atom_name="discovery", model_version=(MODEL, "augment"),
            base_system_text="test", operation_type="fix"))

        assert journal.needs_fix_across_runs() == [], (
            "the fix succeeded but its journal entry did not settle the "
            "create failure's queue item"
        )


class TestAugmentCallsInheritTheTargetIdentity:
    def test_a_failed_fix_leaves_no_orphan_sentinel(
            self, ep, workspace, monkeypatch, fake_call_factory):
        """A failed fix used to write a second `_ERROR` under a slug-minted id —
        a phantom queue entry beside the real one."""
        sentinel = store.error_path(
            store.path_for("discovery", None, "Ptolemy's Almagest", MODEL))
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text('{"incomplete": true}', encoding="utf-8")
        use_fake(ep, monkeypatch, fake_call_factory, ["still not valid json {"])

        ep.augment_and_save_prompt(ep.OperationConfig(
            atom_name="discovery", model_version=(MODEL, "augment"),
            base_system_text="test", operation_type="fix"))

        remaining = sorted(p.name for p in (workspace / "discovery").iterdir())
        assert remaining == [sentinel.name], remaining
        assert sentinel.read_text(encoding="utf-8") == '{"incomplete": true}', (
            "the input to the repair was overwritten by the repair's own failure"
        )

    def test_a_failed_verify_leaves_the_valid_atom_alone_and_queues_nothing(
            self, ep, workspace, monkeypatch, fake_call_factory):
        """A sentinel beside a valid atom sends the fix pass to 'repair' good
        data from a failed improvement's payload; a `needs_fix` in the journal
        does the same one layer up. A failed verify must produce neither."""
        journal.start_run(config={"test": True})
        atom = store.path_for("discovery", None, "Ptolemy's Almagest", MODEL)
        atom.parent.mkdir(parents=True, exist_ok=True)
        atom.write_text(valid_discovery("Almagest"), encoding="utf-8")
        use_fake(ep, monkeypatch, fake_call_factory, ["not valid {"])

        ep.augment_and_save_prompt(ep.OperationConfig(
            atom_name="discovery", model_version=(MODEL, "augment"),
            base_system_text="test", operation_type="verify"))

        remaining = sorted(p.name for p in (workspace / "discovery").iterdir())
        assert remaining == [atom.name], remaining
        assert journal.needs_fix_across_runs() == [], (
            "a failed verify queued a valid atom for repair"
        )

    def test_an_append_inherits_the_city_id(self, fake_call_factory, workspace):
        """What lets `merge_helper` pair the two halves on identity: the append
        file carries the city's content id, not a hash of the lossy slug."""
        from universe_builder.domains.history.models.city import City

        city = store.path_for("city", None, "Thebes (Luxor)", MODEL)
        Fake = fake_call_factory(['{"x": 1}'])
        call = Fake(
            atom_type="append_city", atom_class=City, model_version=MODEL,
            prompt_folder="prompts", prompt_subject="thebes luxor",
            prompt_start_text="p", base_system_text="s", example_json_file=None,
            target_path=str(city), supplements_target=True,
        )

        parsed = store.AtomName.decompose_filename(call.response_path.name)
        assert call.response_path.name.startswith("append_city_")
        assert parsed.padded_id == store.AtomName.decompose_filename(city.name).padded_id
        assert call.writes_error_sentinel, (
            "an append creates its own file, so its failures still queue"
        )

    def test_a_same_atom_target_takes_the_target_path_exactly(
            self, fake_call_factory, workspace):
        from universe_builder.domains.history.models.city import City

        target = store.path_for("city", None, "Thebes (Luxor)", MODEL)
        Fake = fake_call_factory(['{"x": 1}'])
        call = Fake(
            atom_type="city", atom_class=City, model_version=MODEL,
            prompt_folder="prompts", prompt_subject="thebes luxor",
            prompt_start_text="p", base_system_text="s", example_json_file=None,
            target_path=str(store.error_path(target)),
        )

        assert call.response_path == target
        assert not call.writes_error_sentinel

    def test_an_expand_call_works_on_the_target_not_beside_it(
            self, fake_call_factory, workspace):
        """The review-gate case: expand calls carry `atom_type=expand_city`
        against a `city_…` target, and a first version inferred intent from the
        prefix — misclassifying expand as append-like, so its failures kept
        minting orphan sentinels under a fresh identity."""
        from universe_builder.domains.history.models.city import City

        city = store.path_for("city", None, "Thebes (Luxor)", MODEL)
        Fake = fake_call_factory(['{"x": 1}'])
        call = Fake(
            atom_type="expand_city", atom_class=City, model_version=MODEL,
            prompt_folder="prompts", prompt_subject="thebes luxor",
            prompt_start_text="p", base_system_text="s", example_json_file=None,
            target_path=str(city),
        )

        assert call.response_path == city
        assert not call.writes_error_sentinel, (
            "a failed expand entry writes an orphan sentinel again"
        )


class TestTransportFailuresQueueOnlyRepairableWork:
    """`record_failure`'s sibling of the `_record` rule, missed on the first
    pass: a verify or expand that died at transport must not queue its valid
    target for repair. Their retry is the pass run again, not a fix."""

    def test_a_verify_transport_failure_queues_nothing(self, workspace):
        journal.start_run(config={"test": True})
        target = store.path_for("discovery", None, "Ptolemy's Almagest", MODEL)

        record_failure("discovery", "ptolemys almagest", "socket closed",
                       operation="verify", target=target)

        assert journal.needs_fix_across_runs() == []

    def test_an_expand_transport_failure_queues_nothing(self, workspace):
        journal.start_run(config={"test": True})
        target = store.path_for("city", None, "Thebes (Luxor)", MODEL)

        record_failure("expand_city", "thebes luxor", "socket closed",
                       operation="expand", target=target)

        assert journal.needs_fix_across_runs() == []

    def test_a_create_transport_failure_still_queues(self, workspace):
        """The reason record_failure exists: a retry is exactly what a create
        or fix transport failure needs."""
        journal.start_run(config={"test": True})

        record_failure("discovery", "Ptolemy's Almagest", "socket closed")

        assert len(journal.needs_fix_across_runs()) == 1
