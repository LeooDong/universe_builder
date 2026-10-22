"""A batch must be collected against the schema it was submitted under.

A batch can take 24 hours to come back, and the
handle used to record only `schema_model` — the class name. That name resolves
through the registry to whatever the class is *now*, so a model edited while the
batch was in flight would be used to validate day-old responses with nothing said
about it: a field added since submission fails every record as "missing", and a
field removed since submission is dropped from every record that still has it.

`schema_fingerprint` pins the schema's identity at submit time and
`require_matching_schema` refuses the collection if it has moved.
"""

from __future__ import annotations

import json
import logging
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from universe_builder.providers.base import Provider
from universe_builder.providers.types import (
    SCHEMA_PIN_KEYS,
    BatchHandle,
    SchemaMismatch,
    require_matching_schema,
    schema_fingerprint,
)


class Pinned(BaseModel):
    name: str
    count: int


class PinnedWithExtraField(BaseModel):
    """`Pinned` as it might look after a field is added mid-flight."""

    name: str
    count: int
    note: Optional[str] = None


class PinnedRetyped(BaseModel):
    """`Pinned` as it might look after a field changes type mid-flight."""

    name: str
    count: str


def _handle(extra=None, **overrides) -> BatchHandle:
    fields = {
        "provider": "anthropic",
        "batch_id": "msgbatch_test",
        "model": "claude-sonnet-5",
        "submitted_at": "2026-09-09T00:00:00Z",
        "request_count": 2,
        "routing": {"person-00000-a": {"atom_type": "person", "subject": "A"}},
        "extra": dict(schema_fingerprint(Pinned)) if extra is None else extra,
    }
    fields.update(overrides)
    return BatchHandle(**fields)


class TestTheFingerprint:
    def test_an_unchanged_model_fingerprints_identically(self):
        assert schema_fingerprint(Pinned) == schema_fingerprint(Pinned)

    def test_it_records_all_three_pin_keys(self):
        assert set(schema_fingerprint(Pinned)) == set(SCHEMA_PIN_KEYS)

    def test_the_id_names_the_module_and_the_class(self):
        """The class name alone cannot tell two same-named models in different
        modules apart, which is exactly what a refactor produces."""
        assert schema_fingerprint(Pinned)["schema_id"] == f"{__name__}:Pinned"

    def test_adding_a_field_changes_the_hash(self):
        assert (
            schema_fingerprint(PinnedWithExtraField)["schema_sha256"]
            != schema_fingerprint(Pinned)["schema_sha256"]
        )

    def test_retyping_a_field_changes_the_hash(self):
        assert (
            schema_fingerprint(PinnedRetyped)["schema_sha256"]
            != schema_fingerprint(Pinned)["schema_sha256"]
        )

    def test_reordering_fields_does_not_register_as_a_change(self):
        """`sort_keys` normalises mapping order but not list order, and
        `required` is a list in declaration order — so a pure
        reorder of a model's fields once tripped the pin and refused a batch whose
        results validate identically. `required` is now sorted, at every depth,
        before hashing. Both models share a name so nothing else differs."""
        from pydantic import create_model

        first = create_model("Same", name=(str, ...), count=(int, ...))
        second = create_model("Same", count=(int, ...), name=(str, ...))
        assert first.model_json_schema()["required"] != second.model_json_schema()["required"]
        assert schema_fingerprint(first) == schema_fingerprint(second)

    def test_a_nested_reorder_does_not_register_either(self):
        from pydantic import create_model

        inner_a = create_model("Inner", x=(int, ...), y=(int, ...))
        inner_b = create_model("Inner", y=(int, ...), x=(int, ...))
        outer_a = create_model("Outer", inner=(inner_a, ...))
        outer_b = create_model("Outer", inner=(inner_b, ...))
        assert "$defs" in outer_a.model_json_schema(), "precondition: the nested model is a $def"
        assert schema_fingerprint(outer_a) == schema_fingerprint(outer_b)

    def test_a_handle_pinned_in_the_old_form_still_matches(self):
        """Handles submitted before the normalisation carry the unsorted hash.
        A batch in flight across the upgrade must still collect, so both
        forms are accepted on compare; only a real change is refused."""
        import hashlib

        from pydantic import create_model

        model = create_model("Same", name=(str, ...), count=(int, ...))
        legacy = hashlib.sha256(json.dumps(
            model.model_json_schema(), sort_keys=True, ensure_ascii=False,
            separators=(",", ":")).encode("utf-8")).hexdigest()
        assert legacy != schema_fingerprint(model)["schema_sha256"], "precondition: the forms differ"

        pinned = dict(schema_fingerprint(model), schema_sha256=legacy)
        assert _handle(extra=pinned).schema_mismatch(model) is None
        assert _handle(extra=pinned).schema_mismatch(PinnedRetyped) is not None

    def test_property_order_alone_does_not_change_the_hash(self):
        """The half `sort_keys` does buy: the same schema with its `properties`
        mapping emitted in a different order must hash identically, or the pin
        would fire on nothing at all."""
        import hashlib

        schema = Pinned.model_json_schema()
        reversed_props = {"properties": dict(reversed(list(schema["properties"].items())))}
        # `required` sorted, as the fingerprint normalises it; see the reorder test.
        shuffled = {**schema, **reversed_props, "required": sorted(schema["required"])}
        canonical = json.dumps(
            shuffled, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        assert hashlib.sha256(canonical.encode("utf-8")).hexdigest() == (
            schema_fingerprint(Pinned)["schema_sha256"]
        )

    def test_the_hash_is_taken_over_the_plain_pydantic_schema(self):
        """Not `strict_json_schema`. That transform is provider-side transport
        code; hashing its output would make an edit to `_flatten_refs` read as a
        schema change and refuse every batch in flight for a difference that
        cannot affect how results validate."""
        import hashlib

        schema = Pinned.model_json_schema()
        # The plain schema, with only `required` normalised (see the reorder test).
        plain = json.dumps(
            {**schema, "required": sorted(schema["required"])}, sort_keys=True,
            ensure_ascii=False, separators=(",", ":"),
        )
        expected = hashlib.sha256(plain.encode("utf-8")).hexdigest()
        assert schema_fingerprint(Pinned)["schema_sha256"] == expected

    def test_the_strict_form_would_have_hashed_differently(self):
        """Proves the test above is discriminating: the two forms are not the
        same bytes, so choosing between them is a real decision."""
        import hashlib

        strict = json.dumps(
            Provider.strict_json_schema(Pinned), sort_keys=True, ensure_ascii=False,
            separators=(",", ":"),
        )
        assert hashlib.sha256(strict.encode("utf-8")).hexdigest() != (
            schema_fingerprint(Pinned)["schema_sha256"]
        )


class TestMismatchDetection:
    def test_an_unchanged_schema_reports_no_mismatch(self):
        assert _handle().schema_mismatch(Pinned) is None

    def test_a_changed_schema_is_reported(self):
        problem = _handle().schema_mismatch(PinnedWithExtraField)
        assert problem is not None
        assert "msgbatch_test" in problem
        assert "sha256" in problem

    def test_the_report_names_the_submission_time(self):
        """An operator reading this needs to know which day's work is at risk."""
        assert "2026-09-09T00:00:00Z" in _handle().schema_mismatch(PinnedRetyped)

    def test_a_moved_class_is_reported_even_if_the_shape_matches(self):
        handle = _handle(extra={**schema_fingerprint(Pinned), "schema_id": "old.module:Pinned"})
        problem = handle.schema_mismatch(Pinned)
        assert problem is not None
        assert "old.module:Pinned" in problem

    def test_a_handle_written_before_pinning_is_allowed_through(self, caplog):
        """Refusing these would strand batches that were submitted and paid for
        under the old handle format — worse than validating them the way the
        code always has. It must still say so."""
        handle = _handle(extra={"schema_model": "Pinned"})
        with caplog.at_level(logging.WARNING, logger="universe_builder.providers.types"):
            assert handle.schema_mismatch(Pinned) is None
        assert "predates schema pinning" in caplog.text

    def test_the_pin_survives_a_handle_round_trip(self):
        """The handle is written to disk at submit and read back a day later; a
        pin that did not survive that trip would pin nothing."""
        restored = BatchHandle.from_dict(json.loads(json.dumps(_handle().to_dict())))
        assert restored.schema_mismatch(Pinned) is None
        assert restored.schema_mismatch(PinnedRetyped) is not None


class TestRefusal:
    def test_require_matching_schema_passes_an_unchanged_schema(self):
        require_matching_schema(_handle(), Pinned)

    def test_require_matching_schema_raises_on_a_changed_schema(self):
        with pytest.raises(SchemaMismatch) as excinfo:
            require_matching_schema(_handle(), PinnedWithExtraField)
        assert "msgbatch_test" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Provider wiring
# ---------------------------------------------------------------------------

def _openai_provider():
    from universe_builder.providers.openai_provider import OpenAIProvider

    with patch("openai.OpenAI"):
        provider = OpenAIProvider()
    provider._client = MagicMock()
    return provider


def _anthropic_provider():
    from universe_builder.providers.anthropic_provider import AnthropicProvider

    with patch("anthropic.Anthropic"):
        provider = AnthropicProvider()
    provider._client = MagicMock()
    return provider


def _request(model="gpt-5-nano"):
    from universe_builder.domains.history.models.person import Person
    from universe_builder.providers.types import LLMRequest

    return LLMRequest(
        model=model, schema_model=Person, system="s", user="u",
        custom_id="person-00000-a", metadata={"atom_type": "person", "subject": "A"},
    )


def _google_provider():
    """Built without the SDK: `google-genai` is an optional extra."""
    from universe_builder.providers.google_provider import GoogleProvider

    provider = GoogleProvider.__new__(GoogleProvider)
    provider._client = MagicMock()
    return provider


class TestSubmitRecordsThePin:
    def test_openai_submit_pins_the_schema(self):
        from universe_builder.domains.history.models.person import Person

        provider = _openai_provider()
        provider._client.files.create.return_value = MagicMock(id="file_in")
        provider._client.batches.create.return_value = MagicMock(id="batch_x")

        handle = provider.submit_batch([_request()])
        for key in SCHEMA_PIN_KEYS:
            assert handle.extra[key] == schema_fingerprint(Person)[key]
        # The pin must not have displaced what the collect path already needs.
        assert handle.extra["input_file_id"] == "file_in"

    def test_anthropic_submit_pins_the_schema(self):
        from universe_builder.domains.history.models.person import Person

        provider = _anthropic_provider()
        provider._client.messages.batches.create.return_value = MagicMock(id="msgbatch_x")

        handle = provider.submit_batch([_request(model="claude-sonnet-5")])
        for key in SCHEMA_PIN_KEYS:
            assert handle.extra[key] == schema_fingerprint(Person)[key]

    def test_google_submit_pins_the_schema(self):
        """The third provider once recorded `schema_model` alone, so a Gemini
        batch was validated against whatever the model had become."""
        from universe_builder.domains.history.models.person import Person

        provider = _google_provider()
        provider._client.batches.create.return_value = MagicMock()
        provider._client.batches.create.return_value.name = "batches/x"

        handle = provider.submit_batch([_request(model="gemini-2.5-flash-lite")])
        for key in SCHEMA_PIN_KEYS:
            assert handle.extra[key] == schema_fingerprint(Person)[key]
        assert handle.extra["order"] == ["person-00000-a"], "the positional order must survive"


class TestCollectRefusesAMovedSchema:
    """The refusal has to happen before any result is read, or a partial
    collection writes files nobody asked for."""

    def _moved_pin(self, model_name="Person"):
        return {
            "schema_model": model_name,
            "schema_id": f"universe_builder.domains.history.models.person:{model_name}",
            "schema_sha256": "0" * 64,
            "input_file_id": "file_out",
        }

    def test_openai_refuses_before_reading_any_result(self):
        provider = _openai_provider()
        handle = _handle(provider="openai", model="gpt-5-nano", extra=self._moved_pin())

        with pytest.raises(SchemaMismatch):
            list(provider.fetch_batch_results(handle))
        provider._client.batches.retrieve.assert_not_called()
        provider._client.files.content.assert_not_called()

    def test_anthropic_refuses_before_reading_any_result(self):
        provider = _anthropic_provider()
        handle = _handle(extra=self._moved_pin())

        with pytest.raises(SchemaMismatch):
            list(provider.fetch_batch_results(handle))
        provider._client.messages.batches.results.assert_not_called()

    def test_google_refuses_before_reading_any_result(self):
        provider = _google_provider()
        handle = _handle(provider="google", model="gemini-2.5-flash-lite", extra=self._moved_pin())

        with pytest.raises(SchemaMismatch):
            list(provider.fetch_batch_results(handle))
        provider._client.batches.get.assert_not_called()

    def test_a_matching_pin_still_collects(self):
        """The refusal must not fire on the ordinary case."""
        from universe_builder.domains.history.models.person import Person

        provider = _openai_provider()
        handle = _handle(
            provider="openai", model="gpt-5-nano",
            extra={**schema_fingerprint(Person), "input_file_id": "file_out"},
        )
        provider._client.batches.retrieve.return_value = MagicMock(
            status="completed", output_file_id="file_out", error_file_id=None
        )
        provider._client.files.content.return_value = MagicMock(text="")

        assert list(provider.fetch_batch_results(handle)) == []
        provider._client.batches.retrieve.assert_called_once()


class TestARefusedBatchDoesNotAbortTheSweep:
    """`run_batch collect` sweeps every stored handle in turn.

    A `SchemaMismatch` raised out of `collect()` would propagate through that
    loop and skip every healthy batch queued behind the bad one — so one model
    edit could strand an arbitrary number of paid-for batches. `collect()`
    catches it and reports instead.
    """

    def _handle(self, tmp_path, pin):
        from universe_builder.providers.types import BatchHandle

        return BatchHandle(
            provider="anthropic", batch_id="msgbatch_x", model="claude-sonnet-5",
            submitted_at="2026-09-10T00:00:00Z", request_count=1,
            routing={"person-00001-x": {"atom_type": "person", "subject": "X"}},
            extra=dict(pin),
        )

    def test_a_mismatch_is_reported_not_raised(self, workspace, monkeypatch):
        from universe_builder.providers import batch as batch_mod
        from universe_builder.providers.types import SchemaMismatch

        handle = self._handle(workspace, {"schema_model": "Person", "schema_sha256": "stale"})

        class Provider:
            def fetch_batch_results(self, handle):
                raise SchemaMismatch("Person has changed since submission")

        monkeypatch.setattr(batch_mod, "load", lambda p: handle)
        monkeypatch.setattr(batch_mod, "get_provider", lambda m: Provider())

        report = batch_mod.collect("ignored")

        assert report.refused is not None
        assert "changed since submission" in report.refused

    def test_a_refused_batch_is_not_reported_complete(self, workspace, monkeypatch):
        from universe_builder.providers import batch as batch_mod
        from universe_builder.providers.types import SchemaMismatch

        handle = self._handle(workspace, {"schema_model": "Person", "schema_sha256": "stale"})

        class Provider:
            def fetch_batch_results(self, handle):
                raise SchemaMismatch("moved")

        monkeypatch.setattr(batch_mod, "load", lambda p: handle)
        monkeypatch.setattr(batch_mod, "get_provider", lambda m: Provider())

        report = batch_mod.collect("ignored")

        assert not report.complete
        assert report.missing, "every submitted id is unaccounted for, not merely absent"

    def test_a_refused_batch_writes_nothing(self, workspace, monkeypatch):
        from universe_builder.providers import batch as batch_mod
        from universe_builder.providers.types import SchemaMismatch

        handle = self._handle(workspace, {"schema_model": "Person", "schema_sha256": "stale"})

        class Provider:
            def fetch_batch_results(self, handle):
                raise SchemaMismatch("moved")

        monkeypatch.setattr(batch_mod, "load", lambda p: handle)
        monkeypatch.setattr(batch_mod, "get_provider", lambda m: Provider())

        batch_mod.collect("ignored")

        assert list(workspace.rglob("*.json")) == []


class TestABatchReplyCutOffByItsBudgetSaysSo:
    """Anthropic and Google batch results named truncation as a schema fault
    until v0.4.0: `fetch_batch_results` did not pass the stop signal on, so a
    reply cut off at the output limit read `Invalid JSON: EOF while parsing`.
    OpenAI's collect path always passed it."""

    _CUT = '{"name": {"common": "A"}, "summary": "cut off mid-'

    def _pinned_handle(self, **overrides):
        from universe_builder.domains.history.models.person import Person

        return _handle(extra=dict(schema_fingerprint(Person)), **overrides)

    def test_an_anthropic_reply_stopped_at_max_tokens_reads_truncated(self):
        provider = _anthropic_provider()
        message = MagicMock(stop_reason="max_tokens")
        message.content = [MagicMock(type="text", text=self._CUT)]
        message.usage = MagicMock(input_tokens=10, output_tokens=4000,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0)
        result = MagicMock(custom_id="person-00000-a")
        result.result.type = "succeeded"
        result.result.message = message
        provider._client.messages.batches.results.return_value = [result]

        [response] = list(provider.fetch_batch_results(self._pinned_handle()))
        assert response.parsed is None
        assert response.error.startswith("Response truncated")
        assert "'max_tokens'" in response.error

    def test_an_anthropic_reply_that_ended_normally_keeps_its_validation_error(self):
        provider = _anthropic_provider()
        message = MagicMock(stop_reason="end_turn")
        message.content = [MagicMock(type="text", text=self._CUT)]
        message.usage = MagicMock(input_tokens=10, output_tokens=40,
                                  cache_read_input_tokens=0, cache_creation_input_tokens=0)
        result = MagicMock(custom_id="person-00000-a")
        result.result.type = "succeeded"
        result.result.message = message
        provider._client.messages.batches.results.return_value = [result]

        [response] = list(provider.fetch_batch_results(self._pinned_handle()))
        assert not response.error.startswith("Response truncated")

    def test_a_google_reply_stopped_at_max_tokens_reads_truncated(self):
        provider = _google_provider()
        handle = self._pinned_handle(provider="google", model="gemini-2.5-flash-lite")
        handle.extra["order"] = ["person-00000-a"]
        reason = MagicMock()
        reason.name = "MAX_TOKENS"
        item = MagicMock()
        item.response.text = self._CUT
        item.response.candidates = [MagicMock(finish_reason=reason)]
        item.response.usage_metadata = MagicMock(
            prompt_token_count=10, candidates_token_count=4000,
            cached_content_token_count=0, thoughts_token_count=0)
        job = MagicMock()
        job.state.name = "JOB_STATE_SUCCEEDED"
        job.dest.inlined_responses = [item]
        provider._client.batches.get.return_value = job

        [response] = list(provider.fetch_batch_results(handle))
        assert response.custom_id == "person-00000-a"
        assert response.parsed is None
        assert response.error.startswith("Response truncated")
        assert "'MAX_TOKENS'" in response.error
