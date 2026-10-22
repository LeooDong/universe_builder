"""The opt-in `/v1/responses` transport.

Offline: shape and translation only, no network. The live path was exercised
by hand (a searched `verify` that added two real source URLs to an atom that
had none), which no test here can or should repeat.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from universe_builder.providers.openai_provider import (
    OpenAIProvider, OpenAIResponsesProvider, OpenAISearchProvider)
from universe_builder.providers.registry import get_provider_by_name
from universe_builder.providers.types import LLMRequest


@pytest.fixture(autouse=True)
def _keys(fake_keys):
    """Every test here constructs a provider, and the OpenAI client reads a
    key at construction. These passed for months only because the checkout's
    `.env` supplied a real one through `settings.load_environment`; in a
    checkout with no `.env` all 24 failed. An offline test must not depend
    on a credential being present."""


class Tiny(BaseModel):
    city: str


def _request(**kw):
    return LLMRequest(model="gpt-4.1-mini", system="s", user="u",
                      schema_model=Tiny, max_tokens=4096, **kw)


class TestTheDefaultPathIsUntouched:
    """The corpus was generated over Chat Completions and nothing pushes us off
    it; the new transport must be opt-in by provider name, not a silent switch."""

    def test_the_plain_openai_provider_does_not_use_responses(self):
        assert OpenAIProvider.use_responses is False
        assert OpenAIProvider.endpoint == "/v1/chat/completions"

    def test_it_does_not_search(self):
        assert OpenAIProvider.web_search is False

    def test_the_responses_provider_opts_in(self):
        assert OpenAIResponsesProvider.use_responses is True
        assert OpenAIResponsesProvider.endpoint == "/v1/responses"

    def test_search_is_a_further_opt_in(self):
        """With search on, a `sources` array means 'looked up' rather than
        'recalled'. Which claim an atom carries should be recoverable from the
        run's recorded provider, not guessed at later."""
        assert OpenAIResponsesProvider.web_search is False
        assert OpenAISearchProvider.web_search is True

    @pytest.mark.parametrize("name", ["openai", "openai-responses", "openai-search"])
    def test_all_three_resolve_by_name(self, name):
        assert get_provider_by_name(name).name == name


class TestTheRequestShape:
    def test_the_schema_moves_under_text_format(self):
        body = OpenAIResponsesProvider()._responses_body(_request())

        assert body["text"]["format"]["type"] == "json_schema"
        assert body["text"]["format"]["strict"] is True
        assert "response_format" not in body

    def test_the_budget_is_max_output_tokens(self):
        body = OpenAIResponsesProvider()._responses_body(_request())

        assert body["max_output_tokens"] == 4096
        assert "max_completion_tokens" not in body and "max_tokens" not in body

    def test_messages_become_input_unchanged(self):
        """Examples are the cached prefix; reordering them throws the cache away."""
        body = OpenAIResponsesProvider()._responses_body(_request())

        assert body["input"][0]["role"] == "system"
        assert body["input"][-1]["content"] == "u"

    def test_no_tools_without_search(self):
        assert "tools" not in OpenAIResponsesProvider()._responses_body(_request())

    def test_search_attaches_the_tool(self):
        body = OpenAISearchProvider()._responses_body(_request())

        assert body["tools"] == [{"type": "web_search"}]


class TestUsageTranslation:
    """Responses speaks our field names natively — `input_tokens` /
    `output_tokens` — where Chat Completions says `prompt_tokens` /
    `completion_tokens` and hides cached tokens inside the prompt count."""

    def test_tokens_are_read_directly(self):
        usage = OpenAIResponsesProvider._responses_usage(
            {"input_tokens": 500, "output_tokens": 900})

        assert (usage.input_tokens, usage.output_tokens) == (500, 900)

    def test_cached_tokens_are_split_out_not_folded_in(self):
        """They bill at `cached_input`, so folding them into the input count
        loses the rate distinction — the bug that made every cost figure low."""
        usage = OpenAIResponsesProvider._responses_usage(
            {"input_tokens": 500, "output_tokens": 900,
             "input_tokens_details": {"cached_tokens": 200}})

        assert usage.input_tokens == 300
        assert usage.cache_read_tokens == 200

    def test_absent_usage_is_zeroes_not_a_crash(self):
        assert OpenAIResponsesProvider._responses_usage(None).input_tokens == 0


class TestTruncationIsTranslated:
    """Responses reports truncation as `status: incomplete` with a reason;
    the rest of the stack understands the Chat Completions spelling, `length`.
    Left untranslated, a cut-off response reaches `_validate` looking like a
    schema failure — which is how a dead API key once became 417 repair jobs."""

    class _Response:
        def __init__(self, reason):
            self.status = "incomplete"
            self.output_text = ""
            self.usage = None
            self.incomplete_details = type("D", (), {"reason": reason})()

    def test_max_output_tokens_becomes_length(self, monkeypatch):
        provider = OpenAIResponsesProvider()
        captured = {}
        monkeypatch.setattr(provider, "_sync_client", lambda r: type("C", (), {
            "responses": type("R", (), {
                "create": staticmethod(lambda **kw: self._made)})()})())
        monkeypatch.setattr(provider, "_validate",
                            lambda text, req, usage, finish: captured.update(finish=finish))
        self._made = self._Response("max_output_tokens")
        provider._complete_responses(_request())

        assert captured["finish"] == "length"

    def test_another_reason_passes_through(self, monkeypatch):
        provider = OpenAIResponsesProvider()
        captured = {}
        monkeypatch.setattr(provider, "_sync_client", lambda r: type("C", (), {
            "responses": type("R", (), {
                "create": staticmethod(lambda **kw: self._made)})()})())
        monkeypatch.setattr(provider, "_validate",
                            lambda text, req, usage, finish: captured.update(finish=finish))
        self._made = self._Response("content_filter")
        provider._complete_responses(_request())

        assert captured["finish"] == "content_filter"


class TestTheBatchPathIsRefusedNotInherited:
    """Inheriting the batch path would be silently wrong at both ends, and both
    faults cost a 24-hour window to discover.

    `_batch_lines` pairs `"url": self.endpoint` — `/v1/responses` on these
    providers — with `"body": self._body(r)`, a Chat Completions body carrying
    `response_format` and `max_completion_tokens`, neither of which that
    endpoint accepts. And `_response_from` reads
    `body["choices"][0]["message"]["content"]`, a shape a Responses result does
    not have, so collect would raise `KeyError` on every record of a batch that
    had already been paid for.
    """

    @pytest.mark.parametrize("name", ["openai-responses", "openai-search"])
    @pytest.mark.parametrize(
        "method", ["submit_batch", "poll_batch", "fetch_batch_results",
                   "chunk_for_batch"])
    def test_every_batch_entry_point_refuses(self, name, method, monkeypatch):
        """Every method `run_batch` actually calls. An earlier fix overrode a
        `collect_batch` that is not on the interface — dead code that looked
        like a guard while `fetch_batch_results`, the real collect path, stayed
        inherited and broken."""
        from universe_builder.providers.errors import BatchNotSupported  # noqa: F401

        provider = get_provider_by_name(name)
        with pytest.raises(BatchNotSupported):
            if method in ("submit_batch", "chunk_for_batch"):
                getattr(provider, method)([_request()])
            else:
                getattr(provider, method)(None)

    @pytest.mark.parametrize("name", ["openai-responses", "openai-search"])
    def test_the_capability_flag_tells_the_truth(self, name):
        """`supports_batch` is what callers gate on, and it is derived from
        `batch_max_requests > 0`. Overriding the methods alone left it True,
        so `run_batch` still believed this transport could batch."""
        assert get_provider_by_name(name).supports_batch is False

    def test_the_default_provider_still_reports_batch_support(self):
        assert get_provider_by_name("openai").supports_batch is True

    def test_the_interface_has_no_collect_batch(self):
        """Pins why the first fix was theatre: the name did not exist."""
        from universe_builder.providers.base import Provider

        assert not hasattr(Provider, "collect_batch")

    def test_it_refuses_before_anything_is_uploaded(self, monkeypatch):
        """The refusal must come before the upload, not at collect time — the
        point is to fail free rather than after the bill."""
        provider = get_provider_by_name("openai-responses")
        monkeypatch.setattr(provider, "_client", None)  # any use would explode

        with pytest.raises(Exception) as caught:
            provider.submit_batch([_request()])

        assert "does not support --batch" in str(caught.value)

    def test_the_plain_provider_still_batches(self):
        """The default path is untouched: it must not inherit the refusal."""
        from universe_builder.providers.errors import BatchNotSupported  # noqa: F401

        provider = get_provider_by_name("openai")
        assert type(provider).submit_batch is not type(
            get_provider_by_name("openai-responses")).submit_batch
        assert provider.supports_batch is True

    def test_the_endpoint_and_body_would_have_disagreed(self):
        """The actual defect, pinned: the batch body is Chat Completions shaped
        while the endpoint is not."""
        provider = get_provider_by_name("openai-responses")
        body = OpenAIProvider._body(provider, _request())

        assert provider.endpoint == "/v1/responses"
        assert "response_format" in body      # Chat Completions only
        assert "messages" in body             # Responses wants `input`
