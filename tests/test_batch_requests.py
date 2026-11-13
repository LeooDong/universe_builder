"""`providers.batch_requests`: a caller's own requests (a judge's) as one
batch, answered through `on_result` as `run.run_many` answers them (v0.8.0).

No network: the provider is `FakeProvider`, or the Anthropic provider over
the fake client from `test_anthropic_provider`.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from universe_builder import journal, settings
from universe_builder.providers import batch as batch_mod
from universe_builder.providers import batch_requests
from universe_builder.providers.pricing import estimate_cost
from universe_builder.providers.types import LLMRequest, Usage
from tests.test_anthropic_provider import FakeClient, message, provider as anthropic_over

MODEL = "claude-haiku-4-5-20251001"


class Verdict(BaseModel):
    """A judge's answer. Deliberately in no pack's registry: the results are
    validated against the requests' own schema."""

    winner: str
    reason: str


def judge_requests(n=3, **kw):
    return [LLMRequest(model=MODEL, schema_model=Verdict, system="You judge records. " * 50,
                       user=f"Pair {i}", custom_id=f"judge-{i}", max_tokens=2000,
                       metadata={"atom_type": "person", "subject": f"Subject {i}", "operation": "judge"},
                       **kw)
            for i in range(n)]


def verdict(winner="A"):
    return json.dumps({"winner": winner, "reason": "fuller"})


@pytest.fixture
def run(workspace):
    j = journal.start_run(config={"test": True})
    yield j
    journal.end_run()


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(batch_requests.time, "sleep", lambda _s: None)


# -- dry run --------------------------------------------------------------------


def test_without_apply_nothing_is_submitted_and_the_batch_price_is_given(workspace, fake_provider, caplog):
    calls = []
    report = batch_requests.run_many(judge_requests(), on_result=calls.append, output_tokens=500)
    assert fake_provider.batches == [] and calls == []
    assert report.dry_run and report.responses == []
    assert 0 < report.estimate < report.ceiling
    assert "Dry run: would submit 3 requests" in caplog.text and "Pass apply" in caplog.text
    assert not settings.request_batch_dir().exists()


def test_the_estimate_is_the_batch_rate():
    likely, ceiling = batch_requests.estimate(judge_requests(2), output_tokens=500)
    prefix = batch_requests.prefix_tokens(judge_requests(1)[0])
    usage = Usage(input_tokens=2, output_tokens=1000, cache_write_tokens=prefix, cache_read_tokens=prefix)
    assert likely == pytest.approx(estimate_cost(usage, MODEL, batch=True))
    assert likely == pytest.approx(estimate_cost(usage, MODEL) / 2)


# -- apply ----------------------------------------------------------------------


def test_each_request_is_answered_in_order_as_run_many_answers(workspace, fake_provider, run, no_sleep):
    fake_provider.batch_replies = {"judge-0": verdict("A"), "judge-1": "not json", "judge-2": verdict("B")}
    calls = []
    report = batch_requests.run_many(judge_requests(), on_result=calls.append, apply=True)

    assert len(fake_provider.batches) == 1, "one batch"
    assert [r.custom_id for r in calls] == ["judge-0", "judge-1", "judge-2"]
    assert calls[0].parsed == Verdict(winner="A", reason="fuller") and calls[0].error is None
    assert calls[1].parsed is None and calls[1].raw_text == "not json" and calls[1].error
    assert calls[2].usage == Usage(input_tokens=100, output_tokens=50)
    assert report.responses == calls and (report.ok_count, report.failed_count) == (2, 1)
    assert report.cost == pytest.approx(estimate_cost(report.usage, MODEL, batch=True))
    assert report.handle.parent == settings.request_batch_dir() / "collected"


def test_each_result_is_journalled_as_a_batch_call_at_the_batch_price(workspace, fake_provider, run, no_sleep):
    fake_provider.batch_replies = {"judge-0": verdict(), "judge-1": "not json"}
    report = batch_requests.run_many(judge_requests(2), apply=True)
    lines = run.entries()
    assert len(lines) == 2
    ok, failed = lines
    assert ok["execution_mode"] == "batch" and ok["batch_id"] == report.batch_id
    assert ok["operation"] == "judge" and ok["subject"] == "Subject 0"
    assert ok["validation_status"] == "ok" and failed["validation_status"] == "failed"
    assert ok["estimated_cost"] == pytest.approx(estimate_cost(Usage(100, 50), MODEL, batch=True))
    assert run.read_raw(failed["attempt_id"]) == "not json"
    assert "work_status" not in ok, "a judgement queues no repair"


def test_a_request_the_batch_lost_is_answered_as_a_failure(workspace, fake_provider, no_sleep, monkeypatch):
    fake_provider.batch_replies = {"judge-0": verdict()}
    real = fake_provider.fetch_batch_results

    def drop_last(handle, schema_model=None):
        return list(real(handle, schema_model))[:-1]

    monkeypatch.setattr(fake_provider, "fetch_batch_results", drop_last)
    calls = []
    batch_requests.run_many(judge_requests(2), on_result=calls.append, apply=True)
    assert [r.custom_id for r in calls] == ["judge-0", "judge-1"]
    assert calls[1].transport_failed and "no result" in calls[1].error


# -- resume ---------------------------------------------------------------------


def _interrupted(fake_provider):
    """Submit, then die on the first poll, as a caller killed mid-wait does."""
    def interrupt(_handle):
        raise KeyboardInterrupt

    fake_provider.poll_batch = interrupt
    try:
        with pytest.raises(KeyboardInterrupt):
            batch_requests.run_many(judge_requests(), apply=True)
    finally:
        del fake_provider.poll_batch
    pending = sorted(settings.request_batch_dir().glob("*.json"))
    assert len(pending) == 1
    return batch_mod.load(pending[0]).batch_id


def test_a_resume_by_batch_id_polls_the_same_batch_without_resubmitting(workspace, fake_provider, no_sleep,
                                                                        monkeypatch):
    fake_provider.batch_replies = {f"judge-{i}": verdict() for i in range(3)}
    batch_id = _interrupted(fake_provider)
    calls = []
    report = batch_requests.run_many(judge_requests(), on_result=calls.append, resume=batch_id)
    assert len(fake_provider.batches) == 1, "resumed, not resubmitted"
    assert report.batch_id == batch_id and len(calls) == 3 and all(r.ok for r in calls)
    assert not list(settings.request_batch_dir().glob("*.json")), "archived once answered"


def test_the_same_requests_again_resume_their_pending_batch(workspace, fake_provider, no_sleep, monkeypatch):
    fake_provider.batch_replies = {f"judge-{i}": verdict() for i in range(3)}
    batch_id = _interrupted(fake_provider)
    dry = batch_requests.run_many(judge_requests())
    assert dry.dry_run and dry.batch_id == batch_id and len(fake_provider.batches) == 1
    report = batch_requests.run_many(judge_requests(), apply=True)
    assert report.batch_id == batch_id and len(fake_provider.batches) == 1
    batch_requests.run_many(judge_requests(), apply=True)
    assert len(fake_provider.batches) == 2, "a collected batch is not resumed"


def test_an_unknown_resume_id_fails_loudly(workspace, fake_provider):
    with pytest.raises(FileNotFoundError):
        batch_requests.run_many(judge_requests(), resume="msgbatch_nope")


# -- kept apart from atom batches ---------------------------------------------


def test_atom_collect_neither_sees_nor_accepts_a_request_batch(workspace, fake_provider, monkeypatch):
    fake_provider.batch_replies = {f"judge-{i}": verdict() for i in range(3)}
    _interrupted(fake_provider)
    assert batch_mod.list_batches() == []
    handle = next(settings.request_batch_dir().glob("*.json"))
    report = batch_mod.collect(handle)
    assert report.refused and "request batch" in report.refused and report.written == 0
    assert handle.exists()
    assert not [p for p in settings.data_dir().rglob("*.json") if "_batches" not in p.parts]


@pytest.mark.parametrize("change, message_part", [
    (lambda rs: [rs[0], LLMRequest(model=MODEL, schema_model=Verdict, system="S", user="U")], "custom_id"),
    (lambda rs: rs + [rs[0]], "unique"),
    (lambda rs: [rs[0], LLMRequest(model="gpt-5-nano", schema_model=Verdict, system="S", user="U",
                                   custom_id="x")], "one model"),
])
def test_requests_a_batch_cannot_route_are_refused(workspace, fake_provider, change, message_part):
    with pytest.raises(ValueError, match=message_part):
        batch_requests.run_many(change(judge_requests(2)), apply=True)
    assert fake_provider.batches == []


# -- on the Anthropic wire ----------------------------------------------------


def test_on_anthropic_the_system_prompt_is_a_cached_prefix_and_the_judge_schema_validates(
        workspace, fake_keys, no_sleep, monkeypatch):
    from universe_builder.providers import registry

    client = FakeClient()
    client.batch = SimpleNamespace(processing_status="ended", request_counts=SimpleNamespace(
        succeeded=2, errored=0, canceled=0, expired=0, processing=0))
    monkeypatch.setitem(registry._CACHE, "anthropic", anthropic_over(client))

    requests = judge_requests(2, effort="low")
    calls = []
    client.results = [SimpleNamespace(custom_id=f"judge-{i}", result=SimpleNamespace(
        type="succeeded", message=message(verdict()))) for i in range(2)]
    report = batch_requests.run_many(requests, on_result=calls.append, apply=True)

    params = [line["params"] for line in client.batch_payload]
    assert all(p["system"][0]["cache_control"]["type"] == "ephemeral" for p in params)
    assert len({json.dumps(p["system"]) for p in params}) == 1, "one shared prefix"
    assert [p["messages"][-1]["content"] for p in params] == ["Pair 0", "Pair 1"]
    assert all(r.ok and isinstance(r.parsed, Verdict) for r in calls)
    assert report.usage.cache_read_tokens == 6
