"""Truncation is reported as truncation, not as a schema failure.

A response that runs out of output budget arrives as invalid JSON, so the naive
report is `Invalid JSON: EOF while parsing`, which sends the reader to the model
definition when the actual fault is `max_tokens`. On a reasoning model the whole
budget can be spent on reasoning tokens, so the content is empty and nothing in
the message hints at why.

Found live: `gpt-5-nano` with `max_tokens=300` returned 300 output
tokens and an empty string. Raising the budget to 2000 returned valid JSON.

No key and no network: the clients are mocked throughout.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from universe_builder.providers.base import Provider
from universe_builder.providers.types import LLMRequest, Usage


class Tiny(BaseModel):
    city: str


def make_request(max_tokens: int = 300) -> LLMRequest:
    return LLMRequest(
        model="gpt-5-nano",
        schema_model=Tiny,
        system="s",
        user="u",
        max_tokens=max_tokens,
    )


class TestTruncationMessage:
    def test_empty_response_with_length_stop_names_the_token_budget(self):
        resp = Provider._validate("", make_request(), Usage(output_tokens=300), "length")

        assert not resp.ok
        assert "truncated" in resp.error
        assert "300-token limit" in resp.error
        assert "max_tokens" in resp.error

    def test_it_calls_out_reasoning_tokens_when_nothing_was_emitted(self):
        resp = Provider._validate("", make_request(), Usage(output_tokens=300), "length")

        assert "reasoning tokens" in resp.error

    def test_a_partial_body_is_still_reported_as_truncation(self):
        resp = Provider._validate(
            '{"city": "Par', make_request(), Usage(output_tokens=300), "length"
        )

        assert not resp.ok
        assert "truncated" in resp.error
        # The raw text is preserved so the caller can still inspect what arrived.
        assert resp.raw_text == '{"city": "Par'

    def test_a_genuine_schema_failure_is_not_relabelled(self):
        resp = Provider._validate(
            '{"wrong": 1}', make_request(), Usage(output_tokens=10), "stop"
        )

        assert not resp.ok
        assert "truncated" not in resp.error
        assert "city" in resp.error

    def test_a_schema_failure_that_stopped_normally_keeps_the_pydantic_message(self):
        """`finish_reason` absent - the pre-existing callers pass nothing."""
        resp = Provider._validate('{"wrong": 1}', make_request(), Usage())

        assert not resp.ok
        assert "truncated" not in resp.error

    def test_valid_output_is_unaffected_by_a_length_stop(self):
        """A body that happens to be complete parses regardless of the signal."""
        resp = Provider._validate(
            '{"city": "Paris"}', make_request(), Usage(output_tokens=300), "length"
        )

        assert resp.ok
        assert resp.parsed.city == "Paris"
        assert resp.error is None

    @pytest.mark.parametrize("reason", ["length", "max_tokens", "MAX_TOKENS"])
    def test_every_provider_spelling_of_the_signal_is_recognised(self, reason):
        """OpenAI, Anthropic and Google each name it differently."""
        resp = Provider._validate("", make_request(), Usage(output_tokens=300), reason)

        assert "truncated" in resp.error

    def test_an_unknown_limit_still_produces_a_useful_message(self):
        """The batch result path has no max_tokens to report."""
        message = Provider._truncation_error(
            "gpt-5-nano", None, Usage(output_tokens=300), "length"
        )

        assert "output limit" in message
        assert "higher max_tokens" in message


def _stream(parts, finish_reason=None, usage=None):
    """Chunks shaped like the OpenAI streaming API.

    Content and `finish_reason` arrive on chunks with choices; usage arrives on a
    final chunk with **no** choices, which is the shape that made
    `stream_options.include_usage` necessary.
    """
    events = []
    for part in parts:
        ev = MagicMock()
        ev.usage = None
        choice = MagicMock()
        choice.delta.content = part
        choice.finish_reason = None
        ev.choices = [choice]
        events.append(ev)
    if finish_reason:
        ev = MagicMock()
        ev.usage = None
        choice = MagicMock()
        choice.delta.content = None
        choice.finish_reason = finish_reason
        ev.choices = [choice]
        events.append(ev)
    if usage is not None:
        ev = MagicMock()
        ev.usage = usage
        ev.choices = []
        events.append(ev)
    return events


def _usage(prompt=170, completion=300):
    u = MagicMock()
    u.prompt_tokens = prompt
    u.completion_tokens = completion
    u.prompt_tokens_details = None
    return u


class TestOpenAIPassesTheSignalThrough:
    """`complete()` streams, so the signal has to survive reassembly.

    Streaming is what keeps a generation longer than about a minute from having
    its idle socket closed — see `OpenAIProvider.complete`. The risk it
    introduces is that `finish_reason` and usage arrive on *different* chunks
    from the content, so either can be dropped silently.
    """

    def _provider(self, events):
        from universe_builder.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._client = MagicMock()
        # `complete()` goes through `with_options` (per-request timeout, SDK
        # retries off); the options-applied client is the same mock.
        provider._client.with_options.return_value = provider._client
        provider._client.chat.completions.create.return_value = iter(events)
        return provider

    def test_complete_forwards_finish_reason(self, fake_keys):
        provider = self._provider(_stream([""], "length", _usage()))

        resp = provider.complete(make_request())

        assert not resp.ok
        assert "truncated" in resp.error
        assert resp.usage.output_tokens == 300

    def test_it_asks_for_usage_or_every_cost_is_zero(self, fake_keys):
        """Usage only arrives if `stream_options.include_usage` is set."""
        provider = self._provider(_stream(["{}"], "stop", _usage()))
        provider.complete(make_request())

        body = provider._client.chat.completions.create.call_args.kwargs
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}

    def test_the_chunks_are_reassembled_in_order(self, fake_keys):
        import json

        payload = json.dumps({"ok": 1})
        provider = self._provider(_stream(list(payload), "stop", _usage(completion=5)))

        resp = provider.complete(make_request())

        assert resp.raw_text == payload, "the deltas were reordered or dropped"

    def test_a_usage_chunk_with_no_choices_does_not_crash(self, fake_keys):
        """The usage chunk carries an empty `choices` list, not a missing one."""
        provider = self._provider(_stream(["{}"], "stop", _usage()))

        resp = provider.complete(make_request())

        assert resp.usage.output_tokens == 300

    def test_the_unstreamed_path_still_works(self, fake_keys):
        """Kept for providers or callers that want a single response."""
        from universe_builder.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        completion = MagicMock()
        choice = completion.choices[0]
        choice.message.content = ""
        choice.finish_reason = "length"
        completion.usage = _usage()
        provider._client = MagicMock()
        # `complete()` goes through `with_options` (per-request timeout, SDK
        # retries off); the options-applied client is the same mock.
        provider._client.with_options.return_value = provider._client
        provider._client.chat.completions.create.return_value = completion

        resp = provider._complete_unstreamed(make_request())

        assert not resp.ok and "truncated" in resp.error


class TestAnIncompleteStreamIsNotASuccess:
    """A stream can end early without raising, and the text can still look whole.

    The unstreamed path could not do this: it returned a complete response or it
    raised. Streaming introduces "the iterator ran out and nobody said why", and
    if the cut lands after the last brace the partial JSON parses, validates and
    is written as a good atom — with `Usage()` zeros making the call look free.
    Reproduced: `ok=True`, `error=None`, 0 tokens.
    """

    def _provider(self, events):
        from universe_builder.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._client = MagicMock()
        # `complete()` goes through `with_options` (per-request timeout, SDK
        # retries off); the options-applied client is the same mock.
        provider._client.with_options.return_value = provider._client
        provider._client.chat.completions.create.return_value = iter(events)
        return provider

    def test_valid_json_from_a_cut_stream_is_refused(self, fake_keys):
        """The dangerous case: it parses, so nothing else would object."""
        provider = self._provider(_stream(['{"a":', ' 1}']))   # no finish_reason

        resp = provider.complete(make_request())

        assert not resp.ok, "a truncated stream was accepted as a complete atom"
        assert "incomplete" in resp.error
        assert resp.raw_text == '{"a": 1}', "the partial text is still recorded as evidence"

    def test_a_normally_finished_stream_is_accepted(self, fake_keys):
        """The guard must not reject good responses."""
        import json

        payload = json.dumps({"name": "x"})
        provider = self._provider(_stream([payload], "stop", _usage(completion=5)))

        resp = provider.complete(make_request())

        assert resp.error is None or "incomplete" not in resp.error

    def test_a_missing_usage_chunk_is_reported_not_silently_zero(self, fake_keys, caplog):
        """Zero tokens and unknown tokens are different claims."""
        import json
        import logging

        payload = json.dumps({"name": "x"})
        provider = self._provider(_stream([payload], "stop", None))   # no usage chunk

        with caplog.at_level(logging.WARNING):
            provider.complete(make_request())

        assert "unknown, not zero" in caplog.text, caplog.text

    def test_an_empty_stream_is_refused(self, fake_keys):
        """Nothing at all arrived; that is not an empty successful answer."""
        provider = self._provider([])

        resp = provider.complete(make_request())

        assert not resp.ok
        assert "incomplete" in resp.error


class TestAProviderRefusalSurvivesToDisk:
    """Refusing in the provider is not enough if the pipeline re-decides.

    `ProviderCall.call_llm_api` returns the raw text on failure so the `_ERROR`
    file gets the payload, and `LLMCall.parse_response_to_pydantic` then validates
    that text *on its own terms*. For a schema failure the two agree. For an
    incomplete stream they do not: the text parses, so the re-validation
    succeeded and the pipeline wrote a 2,402-byte atom and reported `outcome:
    ok`, one frame after the provider logged "treating it as incomplete".

    Validity of the text was never the question the provider answered. It
    answered whether the text is the *whole* response.
    """

    def _cut_stream(self, payload):
        ev = MagicMock()
        ev.usage = None
        choice = MagicMock()
        choice.delta.content = payload
        choice.finish_reason = None          # the stream just stops
        ev.choices = [choice]
        return [ev]

    def test_a_cut_stream_does_not_become_an_atom(self, workspace, ep, monkeypatch,
                                                  discovery_payload):
        import json

        from universe_builder import store
        from universe_builder.providers.openai_provider import OpenAIProvider

        payload = json.dumps(discovery_payload)
        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._client = MagicMock()
        # `complete()` goes through `with_options` (per-request timeout, SDK
        # retries off); the options-applied client is the same mock.
        provider._client.with_options.return_value = provider._client
        provider._client.chat.completions.create.return_value = iter(self._cut_stream(payload))
        monkeypatch.setattr("universe_builder.llm_classes.provider_call.complete",
                            lambda request: provider.complete(request))

        results = ep.call_and_save_prompt(
            config=ep.OperationConfig(atom_name="discovery", model_version="gpt-5-nano",
                                      base_system_text="t"),
            topic_list=["Papyrus"],
        )

        assert results[0].outcome.value != "ok", (
            "the provider refused the stream and the pipeline overturned it"
        )
        atom = store.path_for("discovery", None, "Papyrus", "gpt-5-nano")
        assert store.error_path(atom).exists(), "the partial payload was not kept"
        assert payload in store.read(store.error_path(atom))
        # Nothing at the atom path, never the fabricated atom. (An EmptyModel
        # placeholder used to sit here; `persist` no longer writes one.)
        assert not atom.exists(), atom.read_text()

    def test_a_complete_stream_still_becomes_an_atom(self, workspace, ep, monkeypatch,
                                                     discovery_payload):
        """The guard must not reject good responses."""
        import json

        from universe_builder import store
        from universe_builder.providers.openai_provider import OpenAIProvider

        payload = json.dumps(discovery_payload)
        events = self._cut_stream(payload)
        events[0].choices[0].finish_reason = "stop"
        usage = MagicMock()
        usage.prompt_tokens, usage.completion_tokens = 10, 20
        usage.prompt_tokens_details = None
        tail = MagicMock()
        tail.usage = usage
        tail.choices = []

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._client = MagicMock()
        # `complete()` goes through `with_options` (per-request timeout, SDK
        # retries off); the options-applied client is the same mock.
        provider._client.with_options.return_value = provider._client
        provider._client.chat.completions.create.return_value = iter(events + [tail])
        monkeypatch.setattr("universe_builder.llm_classes.provider_call.complete",
                            lambda request: provider.complete(request))

        results = ep.call_and_save_prompt(
            config=ep.OperationConfig(atom_name="discovery", model_version="gpt-5-nano",
                                      base_system_text="t"),
            topic_list=["Papyrus"],
        )

        assert results[0].outcome.value == "ok"
        assert store.path_for("discovery", None, "Papyrus", "gpt-5-nano").stat().st_size > 500

    def test_the_failure_message_blames_the_response_not_the_example(self, workspace, ep,
                                                                     monkeypatch, caplog):
        """It read "Example JSON has not been validated" for years.

        The example was never the problem; it sent readers to inspect a bundled
        file that is fine.
        """
        from universe_builder.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._client = MagicMock()
        events = self._cut_stream("{not json")
        events[0].choices[0].finish_reason = "stop"
        provider._client.chat.completions.create.return_value = iter(events)
        monkeypatch.setattr("universe_builder.llm_classes.provider_call.complete",
                            lambda request: provider.complete(request))

        ep.call_and_save_prompt(
            config=ep.OperationConfig(atom_name="discovery", model_version="gpt-5-nano",
                                      base_system_text="t"),
            topic_list=["Papyrus"],
        )

        assert "Example JSON" not in caplog.text, caplog.text
        assert "did not validate" in caplog.text
