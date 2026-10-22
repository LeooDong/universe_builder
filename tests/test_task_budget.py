"""Anthropic's task budget (v0.6.1, `providers.task_budget`).

What these pin:

* unset is the default and sends nothing: the request is byte for byte what
  it was, on the plain (non-beta) endpoints, and a run records no new config
  key, so its config hash is unchanged;
* set, it is `output_config.task_budget` merged beside the effort and the
  format, on the beta endpoint with the `task-budgets-2026-03-13` header, in
  the streamed call, the prompt-only fallback and a Message Batch (submitted,
  polled and collected through the beta batches endpoints);
* a budget under the minimum, over the request's `max_tokens`, or to a Claude
  model that takes none is refused before any call, and never retried;
* every other provider ignores it with one warning per model;
* the config's resolution order, the CLI flag on every path, the dry run, the
  journal line, the batch handle and run provenance record what was sent.

No network: the clients are fakes or never constructed.
"""

from __future__ import annotations

import argparse
import json
import logging
from types import SimpleNamespace

import pytest

from universe_builder import cli, journal, pipeline, settings
from universe_builder.domains.history.models.discovery import Discovery
from universe_builder.providers import effort, run, task_budget
from universe_builder.providers import batch as batch_mod
from universe_builder.providers.anthropic_provider import AnthropicProvider
from universe_builder.providers.errors import TaskBudgetRefused
from universe_builder.providers.google_provider import GoogleProvider
from universe_builder.providers.openai_provider import OpenAIProvider, OpenAIResponsesProvider
from universe_builder.providers.types import BatchHandle, LLMRequest
from tests.factories import minimal_payload
from tests.test_anthropic_provider import GRAMMAR, FakeClient, StatusError, message
from tests.test_effort import corpus  # noqa: F401 - the shared CLI fixture

SONNET = "claude-sonnet-5"          # takes a task budget
HAIKU = "claude-haiku-5-5"          # takes none: refused
LUNA = "gpt-6-luna"                 # OpenAI: ignored
BETA = "task-budgets-2026-03-13"


def _request(model=SONNET, schema=Discovery, **kw):
    # pipeline.yaml's defaults.max_tokens: the engine's bare default (16,384)
    # is under the 20,000 minimum, so a budget beside it is always refused.
    kw.setdefault("max_tokens", 32768)
    return LLMRequest(model=model, schema_model=schema, system="S", user="U", **kw)


def _bare(cls):
    return object.__new__(cls)


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr(effort, "_warned", set())


class BetaClient(FakeClient):
    """`FakeClient` plus `beta.messages.stream` and `beta.messages.batches`,
    recording the `betas` each call carried. The plain endpoints stay as they
    are, so a call that should have gone beta and did not is visible."""

    def __init__(self, replies=(), probe=None):
        super().__init__(replies, probe)
        self.beta_calls = []
        self.batch = SimpleNamespace(processing_status="ended",
                                     request_counts=SimpleNamespace(succeeded=1, errored=0, processing=0))
        self.beta = SimpleNamespace(messages=SimpleNamespace(
            stream=self._beta_stream,
            batches=SimpleNamespace(create=self._beta_batch_create,
                                    retrieve=self._beta_retrieve,
                                    results=self._beta_results)))

    def _beta_stream(self, *, betas, **params):
        self.beta_calls.append(("stream", list(betas)))
        return self._stream(**params)

    def _beta_batch_create(self, *, betas, requests):
        self.beta_calls.append(("batches.create", list(betas)))
        return self._batch_create(requests)

    def _beta_retrieve(self, batch_id, *, betas):
        self.beta_calls.append(("batches.retrieve", list(betas)))
        return self.batch

    def _beta_results(self, batch_id, *, betas):
        self.beta_calls.append(("batches.results", list(betas)))
        return iter(self.results)


def _anthropic(client):
    p = object.__new__(AnthropicProvider)
    p._client = client
    return p


def _good():
    return json.dumps(minimal_payload(Discovery))


# -- the value ------------------------------------------------------------------


class TestValue:
    @pytest.mark.parametrize("given, expected", [
        (None, None), ("", None), (20000, 20000), ("20000", 20000), ("24_000", 24000), (64000, 64000)])
    def test_validate(self, given, expected):
        assert task_budget.validate(given) == expected

    @pytest.mark.parametrize("bad", [19999, "1000", "lots", 2.5e4 + 0.5, True])
    def test_anything_else_is_refused(self, bad):
        with pytest.raises(ValueError, match="task budget"):
            task_budget.validate(bad)

    def test_a_request_validates_its_budget(self):
        assert _request(task_budget="20000").task_budget == 20000
        with pytest.raises(ValueError, match="at least 20,000"):
            _request(task_budget=5000)

    def test_the_budget_is_not_part_of_the_cache_key(self):
        assert _request().cache_key() == _request(task_budget=20000).cache_key()

    def test_the_flag_refuses_a_budget_under_the_minimum(self):
        parser = argparse.ArgumentParser()
        task_budget.add_argument(parser)
        assert parser.parse_args(["--task-budget", "20000"]).task_budget == 20000
        assert parser.parse_args([]).task_budget is None
        with pytest.raises(SystemExit):
            parser.parse_args(["--task-budget", "100"])

    def test_the_capability_is_a_pricing_row(self):
        assert task_budget.takes_task_budget(SONNET) and task_budget.takes_task_budget("claude-fable-5-1")
        assert not task_budget.takes_task_budget(HAIKU)
        assert not task_budget.takes_task_budget("claude-haiku-4-5-20251001")
        assert not task_budget.takes_task_budget("no-such-model")


# -- unset sends nothing ----------------------------------------------------------


class TestUnsetIsTodaysRequest:
    def test_anthropic_params(self):
        assert "output_config" not in _bare(AnthropicProvider)._params(_request(), constrained=False)
        constrained = _bare(AnthropicProvider)._params(_request(effort="low"), constrained=True)
        assert set(constrained["output_config"]) == {"format", "effort"}

    def test_the_stream_takes_the_plain_endpoint(self):
        client = BetaClient([message(_good())])
        assert _anthropic(client).complete(_request()).ok
        assert client.beta_calls == [] and len(client.sent) == 1

    def test_the_batch_takes_the_plain_endpoint_and_the_handle_is_unchanged(self):
        client = BetaClient()
        handle = _anthropic(client).submit_batch([_request(custom_id="discovery-00000-a")])
        assert client.beta_calls == [] and client.batch_payload
        assert "betas" not in handle.extra and "task_budget" not in handle.extra

    def test_openai_body(self):
        assert "task_budget" not in json.dumps(_bare(OpenAIProvider)._body(_request(LUNA)))

    def test_the_bundled_config_sets_none(self):
        config = pipeline.load()
        for phase in config.phases:
            assert config.resolve_task_budget(phase).value is None


# -- Anthropic --------------------------------------------------------------------


class TestAnthropic:
    def test_merged_beside_the_effort_and_the_format_on_the_beta_stream(self):
        client = BetaClient([message(_good())])
        assert _anthropic(client).complete(_request(effort="high", task_budget=20000)).ok
        config = client.sent[0]["output_config"]
        assert config["task_budget"] == {"type": "tokens", "total": 20000}
        assert config["effort"] == "high" and config["format"]["type"] == "json_schema"
        assert client.beta_calls == [("stream", [BETA])]
        assert "betas" not in client.sent[0], "the header goes as betas=, not inside the params"

    def test_the_prompt_only_fallback_keeps_it_on_the_beta_endpoint(self):
        client = BetaClient([StatusError(400, GRAMMAR), message(_good())])
        assert _anthropic(client).complete(_request(effort="medium", task_budget=20000)).ok
        assert client.sent[1]["output_config"] == {"effort": "medium",
                                                   "task_budget": {"type": "tokens", "total": 20000}}
        assert client.beta_calls == [("stream", [BETA]), ("stream", [BETA])]

    def test_a_batch_goes_beta_and_is_polled_and_collected_beta(self):
        client = BetaClient()
        p = _anthropic(client)
        handle = p.submit_batch([_request(task_budget=20000, custom_id="discovery-00000-a",
                                          metadata={"atom_type": "discovery", "subject": "a"})])
        params = client.batch_payload[0]["params"]
        assert params["output_config"]["task_budget"] == {"type": "tokens", "total": 20000}
        assert "betas" not in params
        assert handle.extra["betas"] == [BETA]
        assert client.probes and "task_budget" not in json.dumps(client.probes[0]), \
            "the schema probe asks only whether the grammar compiles"
        restored = BatchHandle.from_dict(handle.to_dict())
        assert p.poll_batch(restored).done
        client.results = [SimpleNamespace(custom_id="discovery-00000-a",
                                          result=SimpleNamespace(type="succeeded", message=message(_good())))]
        [response] = list(p.fetch_batch_results(restored))
        assert response.ok
        assert [c for c in client.beta_calls] == [("batches.create", [BETA]), ("batches.retrieve", [BETA]),
                                                  ("batches.results", [BETA])]

    def test_held_under_the_models_output_cap(self):
        """`max_tokens` over the cap is sent at the cap, and the budget is
        checked against what is sent."""
        assert _bare(AnthropicProvider).task_budget_for(_request(max_tokens=500_000, task_budget=100_000)) \
            == 100_000
        with pytest.raises(TaskBudgetRefused, match="over max_tokens 128,000"):
            _bare(AnthropicProvider).task_budget_for(_request(max_tokens=500_000, task_budget=130_000))


class TestRefusedBeforeAnyCall:
    @pytest.mark.parametrize("kw, match", [
        ({"task_budget": 40000, "max_tokens": 32768}, "over max_tokens 32,768"),
        ({"model": HAIKU, "task_budget": 20000}, "takes no task budget"),
        ({"model": "claude-sonnet-5-5", "task_budget": 20000}, "takes no task budget"),
        ({"model": "claude-haiku-4-5-20251001", "task_budget": 20000}, "takes no task budget"),
    ])
    def test_sync_and_batch(self, kw, match):
        client = BetaClient([message(_good())])
        with pytest.raises(TaskBudgetRefused, match=match):
            _anthropic(client).complete(_request(**kw))
        with pytest.raises(TaskBudgetRefused, match=match):
            _anthropic(client).submit_batch([_request(custom_id="discovery-00000-a", **kw)])
        assert client.sent == [] and client.beta_calls == [] and client.batch_payload is None

    def test_the_bare_default_max_tokens_leaves_no_room(self):
        with pytest.raises(TaskBudgetRefused, match="over max_tokens 16,384"):
            _bare(AnthropicProvider).task_budget_for(
                LLMRequest(model=SONNET, schema_model=Discovery, system="S", user="U", task_budget=20000))

    def test_under_the_minimum_is_refused_on_the_wire_too(self):
        """A request cannot be built under the minimum; `wire` refuses it
        anyway, for a caller that gets one past the constructor."""
        with pytest.raises(TaskBudgetRefused, match="under the API's minimum"):
            task_budget.wire("anthropic", True, SONNET, 1000, 32768)

    def test_over_max_tokens_only_when_explicitly_allowed(self):
        request = _request(task_budget=60000, max_tokens=32768, allow_task_budget_over_max_tokens=True)
        assert _bare(AnthropicProvider).task_budget_for(request) == 60000
        assert task_budget.of_request(request) == 60000

    def test_a_refusal_is_not_retried(self):
        assert run._retryable(TaskBudgetRefused("x")) is False

    def test_run_complete_reports_it_once_with_nothing_sent(self, monkeypatch):
        client = BetaClient([message(_good())] * 3)
        attempts = []
        provider = _anthropic(client)
        original = provider.complete

        def counted(request):
            attempts.append(request)
            return original(request)

        provider.complete = counted
        monkeypatch.setattr(run, "get_provider", lambda _model: provider)
        monkeypatch.setattr(run.time, "sleep", lambda _s: None)
        response = run.complete(_request(model=HAIKU, task_budget=20000), transport_retries=2)
        assert not response.ok and response.transport_failed
        assert "takes no task budget" in response.error
        assert len(attempts) == 1 and client.sent == []

    def test_a_large_share_of_max_tokens_is_warned_once(self, caplog):
        with caplog.at_level(logging.WARNING):
            for _ in range(2):
                assert task_budget.wire("anthropic", True, SONNET, 30000, 32768) == 30000
        assert sum("two thirds" in r.getMessage() for r in caplog.records) == 1


class TestOthersIgnoreIt:
    def test_openai_sends_nothing_and_warns_once(self, caplog):
        p = _bare(OpenAIProvider)
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                assert "task_budget" not in json.dumps(p._body(_request(LUNA, task_budget=20000)))
        assert sum("has no task budget" in r.getMessage() for r in caplog.records) == 1

    def test_the_responses_transport_too(self):
        body = _bare(OpenAIResponsesProvider)._responses_body(_request(LUNA, task_budget=20000))
        assert "task_budget" not in json.dumps(body)

    def test_google(self, caplog):
        with caplog.at_level(logging.WARNING):
            config = _bare(GoogleProvider)._config(_request("gemini-2.5-flash-lite", task_budget=20000))
        assert "task_budget" not in json.dumps(config, default=repr)
        assert any("google has no task budget" in r.getMessage() for r in caplog.records)

    @pytest.mark.parametrize("model, budget, max_tokens, refused", [
        (SONNET, 20000, 32768, False), (SONNET, 40000, 32768, True), (HAIKU, 20000, 32768, True),
        (LUNA, 40000, 32768, False), ("gemini-2.5-flash-lite", 20000, 32768, False), (SONNET, None, 1, False)])
    def test_the_preflight(self, model, budget, max_tokens, refused):
        assert bool(task_budget.refusal(model, budget, max_tokens)) is refused

    def test_sent_is_the_wire(self):
        assert task_budget.sent(SONNET, 20000, 32768) == 20000
        assert task_budget.sent(LUNA, 20000, 32768) is None
        assert task_budget.sent_or_none(HAIKU, 20000, 32768) is None


# -- configuration ----------------------------------------------------------------


def _config(tmp_path, block):
    base = (settings.PACKAGE_DIR / "pipeline.yaml").read_text(encoding="utf-8")
    start = base.index("\ntask_budget:")
    end = base.index("\nretry:")
    path = tmp_path / "pipeline.yaml"
    path.write_text(base[:start] + "\ntask_budget:\n" + block + base[end:], encoding="utf-8")
    return path


class TestConfig:
    def test_precedence(self, tmp_path):
        path = _config(tmp_path, "  default: 20000\n  phases: {create: 22000, history: 26000}\n"
                                 "  atoms: {person: {create: 24000}}\n")
        config = pipeline.load(path)
        assert config.resolve_task_budget("create", "person") == pipeline.Resolved(
            24000, pipeline.SOURCE_TASK_BUDGET_ATOM)
        assert config.resolve_task_budget("create", "city").value == 22000
        assert config.resolve_task_budget("index").value == 26000, "the old phase name is read as index"
        assert config.resolve_task_budget("fix").value == 20000
        assert config.resolve_task_budget("fix", override=30000) == pipeline.Resolved(30000, pipeline.SOURCE_CLI)
        assert "task budget 24000" in config.describe("person")
        assert config.resolve_effort("fix").value is None, "the effort block is untouched"

    @pytest.mark.parametrize("block, message", [
        ("  default: 1000\n", "task_budget.default"),
        ("  phases: {create: lots}\n", "task_budget.phases.create"),
        ("  colour: blue\n", "unknown key"),
    ])
    def test_a_bad_value_fails_on_load(self, tmp_path, block, message):
        with pytest.raises(ValueError, match=message):
            pipeline.load(_config(tmp_path, block))


# -- the command line -------------------------------------------------------------


def _lines(run_dir):
    return [json.loads(x) for x in (run_dir / journal.JOURNAL_FILENAME).read_text().splitlines()]


class TestCommandLine:
    def test_the_dry_run_prints_it_beside_the_model_and_effort(self, corpus, capsys):  # noqa: F811
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", SONNET,
                         "--effort", "high", "--task-budget", "20000", "--dry-run"]) == 0
        line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("model"))
        assert SONNET in line and "effort high" in line
        assert "task budget 20000 of max_tokens 32768  (cli)" in line

    def test_the_dry_run_says_when_the_provider_ignores_it(self, corpus, capsys):  # noqa: F811
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", LUNA,
                         "--task-budget", "20000", "--dry-run"]) == 0
        assert "task budget 20000 asked (cli), none sent" in capsys.readouterr().out

    def test_the_dry_run_without_one_is_unchanged(self, corpus, capsys):  # noqa: F811
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", SONNET, "--dry-run"]) == 0
        assert "task budget" not in capsys.readouterr().out

    @pytest.mark.parametrize("model, budget, message", [
        (SONNET, "40000", "over max_tokens 32,768"),
        (HAIKU, "20000", "takes no task budget"),
    ])
    @pytest.mark.parametrize("batch", [False, True])
    def test_refused_before_any_call(self, corpus, caplog, model, budget, message, batch):  # noqa: F811
        argv = ["run", "--phase", "create", "--atom", "discovery", "--model", model, "--task-budget", budget]
        assert cli.main(argv + (["--batch"] if batch else [])) == 2
        assert corpus.requests == [] and message in caplog.text
        assert batch_mod.list_batches() == []

    def test_under_the_minimum_is_an_argument_error(self, corpus):  # noqa: F811
        with pytest.raises(SystemExit) as exc:
            cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", SONNET,
                      "--task-budget", "1000"])
        assert exc.value.code == 2 and corpus.requests == []

    def test_sync_create_sends_it_and_journals_it(self, corpus):  # noqa: F811
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", SONNET,
                         "--task-budget", "20000"]) == 0
        assert corpus.requests and {r.task_budget for r in corpus.requests} == {20000}
        run_dir = journal.current().directory
        assert {line.get("sent_task_budget") for line in _lines(run_dir)} == {20000}
        assert json.loads((run_dir / journal.RUN_FILENAME).read_text())["config"]["task_budget"] == 20000

    def test_sync_without_one_records_no_key_and_the_same_config_hash(self, corpus):  # noqa: F811
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", SONNET]) == 0
        assert {r.task_budget for r in corpus.requests} == {None}
        run_dir = journal.current().directory
        assert all("sent_task_budget" not in line for line in _lines(run_dir))
        meta = json.loads((run_dir / journal.RUN_FILENAME).read_text())
        assert "task_budget" not in meta["config"]
        before = {"phase": "create", "atoms": ["discovery"], "model": SONNET, "source": pipeline.SOURCE_CLI,
                  "limit": None, "batch": False}
        assert meta["config"] == before
        assert meta["resolved_config_sha256"] == journal.sha256_hex(
            json.dumps(before, sort_keys=True, default=repr))

    def test_a_provider_that_ignores_it_is_recorded_as_none_sent(self, corpus):  # noqa: F811
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", LUNA,
                         "--task-budget", "20000", "--limit", "1"]) == 0
        run_dir = journal.current().directory
        assert all("sent_task_budget" not in line for line in _lines(run_dir))
        assert "task_budget" not in json.loads((run_dir / journal.RUN_FILENAME).read_text())["config"]

    def test_a_batch_carries_it_on_every_request_the_handle_and_the_collected_line(self, corpus):  # noqa: F811
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", SONNET, "--batch",
                         "--task-budget", "20000"]) == 0
        assert {r.task_budget for r in corpus.requests} == {20000}
        [handle_path] = batch_mod.list_batches()
        assert json.loads(handle_path.read_text())["extra"]["task_budget"] == 20000
        corpus.batch_replies = {r.custom_id: _good() for r in corpus.requests}
        journal.start_run(config={"command": "collect"})
        batch_mod.collect(handle_path)
        lines = journal.current().entries()
        assert lines and {line.get("sent_task_budget") for line in lines} == {20000}

    def test_a_batch_to_a_provider_that_ignores_it_stamps_nothing(self, corpus):  # noqa: F811
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", LUNA, "--batch",
                         "--task-budget", "20000"]) == 0
        [handle_path] = batch_mod.list_batches()
        assert "task_budget" not in json.loads(handle_path.read_text())["extra"]

    def test_the_index_phase_takes_it_too(self, corpus, tmp_path, monkeypatch):  # noqa: F811
        seed = tmp_path / "seed2.txt"
        seed.write_text("Greek History\n", encoding="utf-8")
        monkeypatch.setattr(settings, "HISTORY_LIST", seed)
        assert cli.main(["run", "--phase", "index", "--model", SONNET, "--batch",
                         "--task-budget", "20000"]) == 0
        assert {r.task_budget for r in corpus.requests} == {20000}

    def test_run_batch_submit_takes_it_and_refuses_over_max_tokens(self, corpus):  # noqa: F811
        from universe_builder import run_batch

        with pytest.raises(SystemExit, match="over max_tokens"):
            run_batch.main(["submit", "--atom", "discovery", "--model", SONNET, "--from-history",
                            "--task-budget", "24000", "--max-tokens", "20000"])
        assert corpus.requests == []
        assert run_batch.main(["submit", "--atom", "discovery", "--model", SONNET, "--from-history",
                               "--task-budget", "20000"]) == 0
        assert {r.task_budget for r in corpus.requests} == {20000}

    def test_an_experiment_with_a_budget_is_refused(self, corpus, caplog):  # noqa: F811
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", SONNET, "--batch",
                         "--task-budget", "20000", "--experiment", "x", "--dry-run"]) == 2
        assert "does not take a task budget yet" in caplog.text


def test_journal_record_call_records_the_budget_sent(workspace):
    journal.start_run(config={"command": "test"})
    response = SimpleNamespace(raw_text="{}", model=SONNET, parsed=object(), usage=None)
    journal.record_call(_request(task_budget=20000), response, operation="judge", atom_type="city", subject="x")
    journal.record_call(_request(LUNA, task_budget=20000), response, operation="judge",
                        atom_type="city", subject="y")
    journal.record_call(_request(), response, operation="judge", atom_type="city", subject="z")
    first, second, third = journal.current().entries()
    assert first["sent_task_budget"] == 20000
    assert "sent_task_budget" not in second and "sent_task_budget" not in third
    journal.end_run()
