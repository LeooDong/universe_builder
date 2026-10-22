"""The `fake_provider` fixture drives the real transport path.

Everything between the operation and `Provider.complete` is the shipped code
here: `ProviderCall.build_request`, `providers.run.complete` (transport retries,
validation re-asks, usage accounting), `Provider._validate`, the journal and the
store. Only the vendor is scripted.
"""

from __future__ import annotations

import json

import pytest

from universe_builder import journal, store
from universe_builder.atoms import atom_mapping
from universe_builder.domains.history.models.discovery import Discovery
from universe_builder.providers import batch as batch_mod
from universe_builder.providers.types import LLMResponse, Usage
from tests.factories import minimal_payload

MODEL = "gpt-5-nano"


def _valid() -> str:
    return json.dumps(minimal_payload(Discovery))


def _create(ep, topic="Papyrus"):
    config = ep.OperationConfig(atom_name="discovery", model_version=MODEL,
                                base_system_text="You are a historian.")
    return ep.create_atom(config, topic)


class TestTheSyncPath:
    def test_a_valid_reply_is_validated_and_written(self, ep, fake_provider):
        fake_provider.script(_valid())

        result = _create(ep)

        assert result.outcome.value == "ok"
        assert store.atoms_for("discovery", "Papyrus")
        [request] = fake_provider.requests
        assert request.model == MODEL
        assert request.schema_model is Discovery
        # The batch builder's system text: the pack's persona, not
        # the `base_system_text` the config carried.
        from universe_builder.providers.requests import base_system_text

        assert request.system == base_system_text()
        assert request.user.startswith(f"{atom_mapping['discovery'].prompt_start_text}'Papyrus'")

    def test_a_transport_error_is_retried_by_run_complete(self, ep, fake_provider):
        fake_provider.script(ConnectionError("reset"), _valid())

        assert _create(ep).outcome.value == "ok"
        assert len(fake_provider.requests) == 2

    def test_a_schema_failure_is_re_asked(self, ep, fake_provider):
        fake_provider.script('{"not": "a discovery"}', _valid())

        assert _create(ep).outcome.value == "ok"
        assert len(fake_provider.requests) == 2

    def test_a_truncated_reply_is_not_re_asked(self, ep, fake_provider):
        def truncated(request):
            return LLMResponse("{", None, request.model, Usage(output_tokens=10), request.custom_id,
                               "Response truncated: stopped at the limit")
        fake_provider.script(truncated)

        assert _create(ep).outcome.value != "ok"
        assert len(fake_provider.requests) == 1

    def test_usage_reaches_the_journal(self, ep, fake_provider):
        fake_provider.script(_valid())
        journal.start_run(config={"phase": "create"})
        try:
            _create(ep)
            [entry] = journal.read_entries(journal.current().run_id)
        finally:
            journal.end_run()
        assert entry["input_tokens"] == 100
        assert entry["output_tokens"] == 50


class TestTheBatchPath:
    def test_submit_reaches_the_fake_and_saves_a_handle(self, workspace, fake_provider):
        [path] = batch_mod.submit("discovery", ["Papyrus", "Wheel"], MODEL)

        [chunk] = fake_provider.batches
        assert [r.metadata["subject"] for r in chunk] == ["Papyrus", "Wheel"]
        assert batch_mod.load(path).batch_id == "fake-batch-1"

    def test_a_provider_without_batch_is_refused(self, workspace, fake_provider):
        fake_provider.without_batch()
        with pytest.raises(NotImplementedError):
            batch_mod.submit("discovery", ["Papyrus"], MODEL)


def test_every_provider_name_is_the_fake(fake_provider):
    from universe_builder.providers.registry import get_provider

    assert get_provider("gpt-5-nano") is fake_provider
    assert get_provider("claude-haiku-4-5-20251001") is fake_provider


class TestTheDryRunBatchCount:
    """A dry run reports the batches a real submit would make: chunked on
    bytes as well as request count (v0.4.0)."""

    def test_a_tiny_byte_limit_is_reported_as_several_batches(self, workspace, fake_provider, caplog):
        from universe_builder.providers.requests import build_requests

        [one] = build_requests("discovery", ["Papyrus"], MODEL)
        fake_provider.batch_max_bytes = fake_provider.request_size_bytes(one) + 1

        with caplog.at_level("WARNING"):
            assert batch_mod.submit("discovery", ["Papyrus", "Wheel", "Plough"], MODEL,
                                    dry_run=True) == []
        assert "in 3 batch(es)" in caplog.text
        assert fake_provider.batches == [], "a dry run submits nothing"

    def test_the_ordinary_case_is_one_batch(self, workspace, fake_provider, caplog):
        with caplog.at_level("WARNING"):
            batch_mod.submit("discovery", ["Papyrus", "Wheel"], MODEL, dry_run=True)
        assert "in 1 batch(es)" in caplog.text
