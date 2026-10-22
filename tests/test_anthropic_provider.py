"""The Anthropic provider against fakes: what the first live run found,
pinned so it stays found.

- every history atom's schema carried `minimum`/`maximum` (coordinates), which
  `output_config` refuses with a 400; the SDK's `transform_schema` moves them
  into the description;
- most atom schemas are refused as "The compiled grammar is too large", so a
  (model, schema) the API will not compile goes out through the prompt only;
- batch custom_ids must match `^[A-Za-z0-9_-]{1,64}$`;
- a `refusal` stop reason, a fenced reply and an over-cap `max_tokens`.

No network: the client is a fake throughout.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from universe_builder.domains.history.models.discovery import Discovery
from universe_builder.domains.history.models.person import Person
from universe_builder.providers import run
from universe_builder.providers.anthropic_provider import AnthropicProvider, _json_text
from universe_builder.providers.pricing import lookup, max_output
from universe_builder.providers.types import BatchHandle, LLMRequest, schema_fingerprint
from tests.factories import minimal_payload

MODEL = "claude-haiku-4-5-20251001"
GRAMMAR = "The compiled grammar is too large, which would cause performance issues."


class StatusError(Exception):
    """An SDK `APIStatusError` as far as the engine looks at one."""

    def __init__(self, status_code, message="", headers=None):
        super().__init__(f"Error code: {status_code} - {message}")
        self.status_code = status_code
        self.response = SimpleNamespace(headers=headers or {})


def message(text, stop="end_turn", output_tokens=5, blocks=None):
    content = blocks if blocks is not None else [SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(
        content=content, stop_reason=stop,
        usage=SimpleNamespace(input_tokens=10, output_tokens=output_tokens,
                              cache_read_input_tokens=3, cache_creation_input_tokens=2))


class _Stream:
    def __init__(self, reply):
        self.reply = reply

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.reply


class FakeClient:
    """`with_options(...).messages.stream(...)` plus `messages.create` (the
    batch probe) and `messages.batches`. A reply that is an exception is raised;
    a callable is called with the params."""

    def __init__(self, replies=(), probe=None):
        self.replies = list(replies)
        self.sent = []
        self.options = []
        self.probes = []
        self.probe = probe
        self.batch_payload = None
        self.results = []
        self.messages = SimpleNamespace(stream=self._stream, create=self._create,
                                        batches=SimpleNamespace(create=self._batch_create,
                                                                results=lambda _id: iter(self.results),
                                                                retrieve=lambda _id: self.batch))

    def with_options(self, **kw):
        self.options.append(kw)
        return self

    def _stream(self, **params):
        self.sent.append(params)
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(params)
        if isinstance(reply, Exception):
            raise reply
        return _Stream(reply)

    def _create(self, **params):
        self.probes.append(params)
        if isinstance(self.probe, Exception):
            raise self.probe
        return message("")

    def _batch_create(self, requests):
        self.batch_payload = requests
        return SimpleNamespace(id="msgbatch_fake")


def provider(client):
    p = object.__new__(AnthropicProvider)
    p._client = client
    return p


def request(schema=Discovery, **kw):
    kw.setdefault("model", MODEL)
    return LLMRequest(schema_model=schema, system="S", user="U", **kw)


def good(schema=Discovery):
    return json.dumps(minimal_payload(schema))


# -- the schema on the wire ------------------------------------------------------


def _walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


@pytest.mark.parametrize("schema", [Discovery, Person])
def test_the_wire_schema_carries_no_keyword_the_api_refuses(schema):
    """`{'type': 'integer', 'minimum': -90, ...}` was a 400 on every atom."""
    assert "minimum" in json.dumps(AnthropicProvider.strict_json_schema(schema))
    wire = AnthropicProvider.wire_schema(schema)
    refused = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
               "minLength", "maxLength"}
    for node in _walk(wire):
        assert not refused & set(node), node
        if node.get("type") == "object":
            assert node["additionalProperties"] is False


def test_the_schema_goes_out_as_output_config_by_default():
    client = FakeClient([message(good())])
    response = provider(client).complete(request())
    assert response.ok
    fmt = client.sent[0]["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"] == AnthropicProvider.wire_schema(Discovery)


def test_a_grammar_too_large_falls_back_to_the_prompt_and_is_remembered():
    client = FakeClient([StatusError(400, GRAMMAR), message(good()), message(good())])
    p = provider(client)
    first = p.complete(request())
    assert first.ok
    assert "output_config" in client.sent[0] and "output_config" not in client.sent[1]
    # The schema is still in the cached system text the model reads.
    assert p.SCHEMA_PREAMBLE in client.sent[1]["system"][0]["text"]
    second = p.complete(request())
    assert second.ok and len(client.sent) == 3
    assert "output_config" not in client.sent[2], "the refusal is remembered, not re-asked"
    # Another model is asked afresh: the grammar limit is the API's, per request.
    client.replies.append(message(good()))
    p.complete(request(model="claude-sonnet-5"))
    assert "output_config" in client.sent[3]


def test_any_other_400_is_raised_not_swallowed():
    client = FakeClient([StatusError(400, "max_tokens: 200000 > 64000")])
    with pytest.raises(StatusError):
        provider(client).complete(request())
    assert len(client.sent) == 1


# -- what comes back --------------------------------------------------------------


def test_thinking_blocks_are_skipped_and_text_blocks_joined():
    text = good()
    blocks = [SimpleNamespace(type="thinking", thinking=""),
              SimpleNamespace(type="text", text=text[:10]), SimpleNamespace(type="text", text=text[10:])]
    response = provider(FakeClient([message("", blocks=blocks)])).complete(request())
    assert response.ok and response.raw_text == text


@pytest.mark.parametrize("wrap", ["```json\n{}\n```", "```\n{}\n```", "Here is the JSON:\n{}", "{}\n"])
def test_a_reply_without_a_grammar_may_be_fenced_or_prefaced(wrap):
    text = good()
    response = provider(FakeClient([message(wrap.replace("{}", text))])).complete(request())
    assert response.ok, response.error


def test_a_truncated_object_is_kept_whole():
    assert _json_text('{"a": {"b": 1}, "c": "unfin') == '{"a": {"b": 1}, "c": "unfin'
    assert _json_text("no object here") == "no object here"


def test_a_max_tokens_stop_is_named_as_truncation():
    response = provider(FakeClient([message(good()[:40], stop="max_tokens", output_tokens=64)])).complete(
        request(max_tokens=64))
    assert not response.ok and response.error.startswith("Response truncated")
    assert "64-token limit" in response.error


def test_a_refusal_is_named_as_a_refusal():
    response = provider(FakeClient([message("", stop="refusal")])).complete(request())
    assert not response.ok and response.error.startswith("Response refused")


def test_usage_keeps_the_cache_split():
    response = provider(FakeClient([message(good())])).complete(request())
    u = response.usage
    assert (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens) == (10, 5, 3, 2)


def test_the_call_is_streamed_with_sdk_retries_off_and_the_policy_timeout():
    client = FakeClient([message(good())])
    provider(client).complete(request(timeout=120.0))
    assert client.options == [{"max_retries": 0, "timeout": 120.0}]


# -- max_tokens and the output cap ------------------------------------------------


def test_max_tokens_is_held_to_the_models_output_cap():
    client = FakeClient([message(good()), message(good())])
    p = provider(client)
    p.complete(request(max_tokens=100_000))
    assert client.sent[0]["max_tokens"] == 64_000
    p.complete(request(model="claude-sonnet-5", max_tokens=100_000))
    assert client.sent[1]["max_tokens"] == 100_000


def test_every_anthropic_row_records_its_output_cap():
    """From the Models API. A tier model without one would let an
    over-cap request through to a 400."""
    from universe_builder.providers.pricing import known_models

    rows = [m for m in known_models() if lookup(m).provider == "anthropic"]
    assert rows
    for model in rows:
        assert max_output(model) in (64_000, 128_000), model
    assert max_output("claude-haiku-4-5-20251001") == 64_000
    assert max_output("gpt-5-nano") is None


# -- batch ------------------------------------------------------------------------


def _batch_request(custom_id, schema=Discovery, model=MODEL):
    return LLMRequest(model=model, schema_model=schema, system="S", user="U", custom_id=custom_id,
                      metadata={"atom_type": "discovery", "subject": custom_id})


def test_custom_ids_the_api_would_refuse_are_swapped_and_swapped_back():
    ids = ["discovery-00000-faience", "discovery-00001-gorée", "append_city-00002-" + "x" * 48]
    client = FakeClient()
    p = provider(client)
    handle = p.submit_batch([_batch_request(i) for i in ids])
    wire = [line["custom_id"] for line in client.batch_payload]
    assert wire[0] == ids[0]
    assert all(len(w) <= 64 and w.isascii() for w in wire)
    assert len(set(wire)) == 3
    assert handle.extra["wire_ids"] == {wire[1]: ids[1], wire[2]: ids[2]}
    assert set(handle.routing) == set(ids)

    client.results = [SimpleNamespace(custom_id=w, result=SimpleNamespace(
        type="succeeded", message=message(good()))) for w in wire]
    got = [r.custom_id for r in p.fetch_batch_results(BatchHandle.from_dict(handle.to_dict()))]
    assert got == ids


def test_submit_probes_each_schema_once_and_a_refused_one_goes_without_output_config():
    client = FakeClient(probe=StatusError(400, GRAMMAR))
    p = provider(client)
    p.submit_batch([_batch_request(f"discovery-{i:05d}-a") for i in range(3)])
    assert len(client.probes) == 1 and client.probes[0]["max_tokens"] == 1
    assert all("output_config" not in line["params"] for line in client.batch_payload)
    p.submit_batch([_batch_request("discovery-00009-b")])
    assert len(client.probes) == 1, "the answer is kept for the process"


def test_a_schema_that_compiles_keeps_output_config_in_the_batch():
    client = FakeClient()
    provider(client).submit_batch([_batch_request("discovery-00000-a")])
    assert len(client.probes) == 1
    assert client.batch_payload[0]["params"]["output_config"]["format"]["type"] == "json_schema"


def test_a_probe_that_fails_for_another_reason_fails_the_submit():
    client = FakeClient(probe=StatusError(401, "invalid x-api-key"))
    with pytest.raises(StatusError):
        provider(client).submit_batch([_batch_request("discovery-00000-a")])
    assert client.batch_payload is None


def test_an_errored_line_names_the_inner_error():
    client = FakeClient()
    p = provider(client)
    handle = BatchHandle(provider="anthropic", batch_id="b", model=MODEL, submitted_at="t",
                         request_count=2, extra=dict(schema_fingerprint(Discovery)))
    inner = SimpleNamespace(type="invalid_request_error", message=GRAMMAR)
    client.results = [
        SimpleNamespace(custom_id="discovery-00000-a", result=SimpleNamespace(
            type="errored", error=SimpleNamespace(type="error", error=inner))),
        SimpleNamespace(custom_id="discovery-00001-b", result=SimpleNamespace(type="expired")),
        SimpleNamespace(custom_id="discovery-00002-c", result=SimpleNamespace(
            type="succeeded", message=message("", stop="refusal"))),
    ]
    errored, expired, refused = list(p.fetch_batch_results(handle))
    assert errored.error == f"batch errored: invalid_request_error: {GRAMMAR}"
    assert expired.error == "batch expired: expired"
    assert refused.error.startswith("Response refused")


def test_poll_counts_canceled_and_expired_with_errored():
    client = FakeClient()
    client.batch = SimpleNamespace(processing_status="ended", request_counts=SimpleNamespace(
        succeeded=7, errored=1, canceled=1, expired=1, processing=0))
    status = provider(client).poll_batch(BatchHandle("anthropic", "b", MODEL, "t", 10))
    assert status.done and (status.succeeded, status.errored) == (7, 3)


# -- the retry loop (shared by every provider) -----------------------------------


class _Raising:
    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def complete(self, req):
        self.calls += 1
        raise self.errors.pop(0) if len(self.errors) > 1 else self.errors[0]


def _run_with(monkeypatch, fake):
    sleeps = []
    monkeypatch.setattr(run, "get_provider", lambda model: fake)
    monkeypatch.setattr(run.time, "sleep", sleeps.append)
    run.configure({"transport_attempts": 3})
    try:
        response = run.complete(request())
    finally:
        run.configure()
    return response, sleeps


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413])
def test_a_request_the_api_refused_is_not_retried(monkeypatch, status):
    fake = _Raising([StatusError(status, "no")])
    response, sleeps = _run_with(monkeypatch, fake)
    assert fake.calls == 1 and sleeps == []
    assert response.transport_failed and str(status) in response.error


@pytest.mark.parametrize("status", [429, 500, 529])
def test_overload_and_rate_limits_are_retried(monkeypatch, status):
    fake = _Raising([StatusError(status, "busy")])
    _, sleeps = _run_with(monkeypatch, fake)
    assert fake.calls == 3 and len(sleeps) == 2


def test_retry_after_is_honoured_up_to_the_cap(monkeypatch):
    fake = _Raising([StatusError(429, "slow down", headers={"retry-after": "12"})])
    _, sleeps = _run_with(monkeypatch, fake)
    assert all(s >= 12 for s in sleeps) and all(s <= 30 for s in sleeps)


def test_a_priced_model_in_no_tier_is_callable_by_id():
    """`--model claude-haiku-5-5` raised KeyError in the synchronous create
    until v0.5.9: only tier models were registered."""
    from universe_builder.config import llm_call_mapping
    from universe_builder.providers.pricing import known_models

    for model in known_models():
        assert model in llm_call_mapping and (model, "augment") in llm_call_mapping


# -- one closing brace too many (v0.6.0, found live) -------------------------------


def test_one_extra_closing_brace_after_a_complete_object_is_dropped(caplog):
    """Haiku 5.5 without a grammar: some Discovery replies ended
    `...firearms."}}` where the object had closed one brace earlier, read as
    `Invalid JSON: trailing characters`."""
    body = json.loads(good())
    body["summary"] = "Gunpowder changed warfare and, in time, the design of firearms."
    exact = json.dumps(body, ensure_ascii=False) + "}"
    assert exact.endswith('firearms."}}')
    with pytest.raises(Exception, match="trailing characters"):
        Discovery.model_validate_json(exact)
    import logging

    with caplog.at_level(logging.WARNING):
        assert _json_text(exact) == exact[:-1]
    assert "1 extra closing character" in caplog.text


@pytest.mark.parametrize("tail", ["}}", "}\n", " ] } ", "]"])
def test_only_closers_and_whitespace_may_follow(tail):
    obj = '{"a": {"b": 1}}'
    assert _json_text(obj + tail) == obj


@pytest.mark.parametrize("reply", [
    '{"a": 1} and that is all',          # prose after: still fails
    '{"a": 1}{"b": 2}',                   # a second object: still fails
    '{"a": 1}}x',                         # a closer, then something else
    '{"a": {"b": 1',                      # truncated: kept whole, for repair
])
def test_anything_else_after_the_object_is_left_to_fail(reply):
    assert _json_text(reply) == reply


def test_a_valid_reply_is_returned_byte_for_byte():
    text = '  {"a": 1}\n'
    assert _json_text(text) == text


def test_the_prompt_only_path_validates_the_trimmed_reply():
    client = FakeClient([StatusError(400, GRAMMAR), message(good() + "}")])
    response = provider(client).complete(request())
    assert response.ok, response.error
