"""The provider-neutral reasoning effort (v0.6.0).

What these pin:

* unset is the default and sends nothing, on every provider and both paths,
  so a run without an effort is byte for byte what it was before;
* each provider's translation (OpenAI `reasoning_effort` / `reasoning.effort`,
  Anthropic `output_config.effort` merged beside the format), and the
  providers and models that take none;
* a level a provider lacks is refused, never mapped, and never retried;
* the config's resolution order, the CLI flag on every path (sync, batch,
  `run_batch submit`), the dry run's display, the journal line, the batch
  handle, run provenance and `costs --by effort`.

No network: the clients are fakes or never constructed.
"""

from __future__ import annotations

import json
import logging
from types import SimpleNamespace

import pytest

from universe_builder import cli, costs, journal, pipeline, settings, store
from universe_builder.domains.history.models.discovery import Discovery
from universe_builder.experiments import PairAssessment
from universe_builder.providers import effort, registry, run
from universe_builder.providers import batch as batch_mod
from universe_builder.providers.anthropic_provider import AnthropicProvider
from universe_builder.providers.errors import EffortNotSupported
from universe_builder.providers.google_provider import GoogleProvider
from universe_builder.providers.openai_provider import (
    DeepSeekProvider, OpenAIProvider, OpenAIResponsesProvider, OpenRouterProvider,
)
from universe_builder.providers.types import LLMRequest
from tests.factories import minimal_payload
from tests.test_anthropic_provider import FakeClient, message

LUNA = "gpt-6-luna"
HAIKU = "claude-haiku-5-5"


def _request(model=LUNA, schema=Discovery, **kw):
    return LLMRequest(model=model, schema_model=schema, system="S", user="U", **kw)


def _bare(cls):
    """A provider with no client: request assembly only."""
    return object.__new__(cls)


@pytest.fixture(autouse=True)
def _fresh_warnings(monkeypatch):
    monkeypatch.setattr(effort, "_warned", set())


# -- the level ------------------------------------------------------------------


class TestLevel:
    @pytest.mark.parametrize("given, expected", [
        (None, None), ("", None), ("low", "low"), ("High", "high"), (" medium ", "medium"),
        ("xhigh", "xhigh"), ("max", "max")])
    def test_validate(self, given, expected):
        assert effort.validate(given) == expected

    @pytest.mark.parametrize("bad", ["minimal", "none", "hi", "5"])
    def test_anything_else_is_refused(self, bad):
        with pytest.raises(ValueError, match="effort must be one of"):
            effort.validate(bad)

    def test_a_request_validates_its_effort(self):
        assert _request(effort="LOW").effort == "low"
        with pytest.raises(ValueError):
            _request(effort="extreme")

    def test_the_effort_is_not_part_of_the_cache_key(self):
        assert _request().cache_key() == _request(effort="high").cache_key()

    def test_labels(self):
        assert effort.label(LUNA, None) == LUNA
        assert effort.label(LUNA, "low") == "gpt-6-luna@low"
        assert effort.split_label("gpt-6-luna@Low") == (LUNA, "low")
        assert effort.split_label(LUNA) == (LUNA, None)

    def test_every_provider_class_is_registered_under_its_factory_name(self):
        assert set(registry._CLASSES) == set(registry._FACTORIES)
        for name in registry._CLASSES:
            assert registry.provider_class(name).name == name


# -- unset sends nothing ----------------------------------------------------------


class TestUnsetIsTodaysRequest:
    def test_openai_chat_body_and_batch_line(self):
        p = _bare(OpenAIProvider)
        assert "reasoning_effort" not in p._body(_request())
        assert "reasoning_effort" not in p._jsonl([_request(custom_id="x")]).decode()

    def test_openai_responses_body(self):
        assert "reasoning" not in _bare(OpenAIResponsesProvider)._responses_body(_request())

    def test_anthropic_params(self):
        params = _bare(AnthropicProvider)._params(_request(HAIKU), constrained=False)
        assert "output_config" not in params
        constrained = _bare(AnthropicProvider)._params(_request(HAIKU), constrained=True)
        assert set(constrained["output_config"]) == {"format"}

    def test_the_bundled_config_sets_none(self):
        config = pipeline.load()
        for phase in config.phases:
            assert config.resolve_effort(phase).value is None


# -- OpenAI -----------------------------------------------------------------------


class TestOpenAI:
    def test_chat_completions_and_every_batch_line_carry_reasoning_effort(self):
        p = _bare(OpenAIProvider)
        assert p._body(_request(effort="medium"))["reasoning_effort"] == "medium"
        line = json.loads(p._jsonl([_request(effort="low", custom_id="x")]))
        assert line["body"]["reasoning_effort"] == "low"

    def test_the_responses_transport_nests_it(self):
        body = _bare(OpenAIResponsesProvider)._responses_body(_request(effort="high"))
        assert body["reasoning"] == {"effort": "high"} and "reasoning_effort" not in body

    def test_xhigh_only_where_the_models_row_lists_it(self, monkeypatch):
        """No OpenAI row lists xhigh yet: it is refused before the run, and a
        row that lists it (once verified) lets it through."""
        from dataclasses import replace

        from universe_builder.providers import pricing

        with pytest.raises(EffortNotSupported, match="no effort level 'xhigh'"):
            _bare(OpenAIProvider)._body(_request(effort="xhigh"))
        widened = replace(pricing._MODELS[LUNA], effort_levels=("low", "medium", "high", "xhigh"))
        monkeypatch.setitem(pricing._MODELS, LUNA, widened)
        assert _bare(OpenAIProvider)._body(_request(effort="xhigh"))["reasoning_effort"] == "xhigh"
        assert effort.refusal(LUNA, "xhigh") is None

    def test_max_is_refused_not_mapped(self):
        with pytest.raises(EffortNotSupported, match="no effort level 'max'"):
            _bare(OpenAIProvider)._body(_request(effort="max"))

    def test_a_refused_level_is_not_retried(self):
        assert run._retryable(EffortNotSupported("x")) is False

    def test_a_non_reasoning_model_gets_nothing_and_one_warning(self, caplog):
        p = _bare(OpenAIProvider)
        with caplog.at_level(logging.WARNING):
            for _ in range(3):
                assert "reasoning_effort" not in p._body(_request("gpt-4o-mini", effort="high"))
        assert sum("gpt-4o-mini takes no effort" in r.getMessage() for r in caplog.records) == 1

    @pytest.mark.parametrize("cls", [OpenRouterProvider, DeepSeekProvider])
    def test_compatible_apis_ignore_it(self, cls, caplog):
        with caplog.at_level(logging.WARNING):
            body = _bare(cls)._body(_request("deepseek-chat", effort="low"))
        assert "reasoning_effort" not in body
        assert any("takes no effort setting" in r.getMessage() for r in caplog.records)


# -- Anthropic --------------------------------------------------------------------


def _anthropic(client):
    p = object.__new__(AnthropicProvider)
    p._client = client
    return p


class TestAnthropic:
    def test_effort_is_merged_beside_the_format(self):
        client = FakeClient([message(json.dumps(minimal_payload(Discovery)))])
        assert _anthropic(client).complete(_request(HAIKU, effort="low")).ok
        config = client.sent[0]["output_config"]
        assert config["effort"] == "low" and config["format"]["type"] == "json_schema"

    def test_the_prompt_only_fallback_keeps_the_effort(self):
        from tests.test_anthropic_provider import GRAMMAR, StatusError

        good = json.dumps(minimal_payload(Discovery))
        client = FakeClient([StatusError(400, GRAMMAR), message(good)])
        assert _anthropic(client).complete(_request(HAIKU, effort="medium")).ok
        assert client.sent[1]["output_config"] == {"effort": "medium"}

    def test_the_batch_params_carry_it(self):
        client = FakeClient()
        _anthropic(client).submit_batch([_request(HAIKU, effort="max", custom_id="discovery-00000-a",
                                                  metadata={"atom_type": "discovery", "subject": "a"})])
        params = client.batch_payload[0]["params"]
        assert params["output_config"]["effort"] == "max" and "format" in params["output_config"]

    def test_haiku_4_5_takes_none(self):
        params = _bare(AnthropicProvider)._params(_request("claude-haiku-4-5-20251001", effort="low"),
                                                  constrained=False)
        assert "output_config" not in params

    def test_the_judge_reply_goes_through_structured_output(self):
        """`experiment judge` on Claude: `PairAssessment` on the wire as the
        format, beside the judge's own effort."""
        reply = json.dumps({"record_1": {"wrong_subject": False, "contradictions": [], "false_certainty": [],
                                         "coverage": 2},
                            "record_2": {"wrong_subject": False, "contradictions": ["x"], "false_certainty": [],
                                         "coverage": 1}, "reason": "r"})
        client = FakeClient([message(reply)])
        response = _anthropic(client).complete(
            LLMRequest(model="claude-opus-5-5", schema_model=PairAssessment, system="S", user="U", effort="low"))
        assert response.ok
        config = client.sent[0]["output_config"]
        assert config["format"]["schema"] == AnthropicProvider.wire_schema(PairAssessment)
        assert config["effort"] == "low"


class TestOthers:
    def test_google_ignores_it_with_a_warning(self, caplog):
        with caplog.at_level(logging.WARNING):
            config = _bare(GoogleProvider)._config(_request("gemini-2.5-flash-lite", effort="high"))
        assert "effort" not in json.dumps(config, default=repr)
        assert any("google takes no effort setting" in r.getMessage() for r in caplog.records)


class TestRefusal:
    @pytest.mark.parametrize("model, level, refused", [
        (LUNA, "max", True), (LUNA, "xhigh", True), (LUNA, "high", False), (LUNA, None, False),
        (HAIKU, "max", False), ("gpt-4o-mini", "max", False), ("gemini-2.5-flash-lite", "max", False),
        ("claude-haiku-4-5-20251001", "max", False)])
    def test_only_a_level_the_provider_lacks_is_refused(self, model, level, refused):
        assert bool(effort.refusal(model, level)) is refused


# -- configuration ----------------------------------------------------------------


def _config(tmp_path, block):
    base = (settings.PACKAGE_DIR / "pipeline.yaml").read_text(encoding="utf-8")
    start = base.index("\neffort:")
    end = base.index("\nretry:")
    path = tmp_path / "pipeline.yaml"
    path.write_text(base[:start] + "\neffort:\n" + block + base[end:], encoding="utf-8")
    return path


class TestConfig:
    def test_precedence(self, tmp_path):
        path = _config(tmp_path, "  default: low\n  phases: {create: medium, history: high}\n"
                                 "  atoms: {person: {create: xhigh}}\n")
        config = pipeline.load(path)
        assert config.resolve_effort("create", "person") == pipeline.Resolved("xhigh", pipeline.SOURCE_EFFORT_ATOM)
        assert config.resolve_effort("create", "city").value == "medium"
        assert config.resolve_effort("index").value == "high", "the old phase name is read as index"
        assert config.resolve_effort("fix").value == "low"
        assert config.resolve_effort("fix", override="max") == pipeline.Resolved("max", pipeline.SOURCE_CLI)
        assert "effort xhigh" in config.describe("person")

    @pytest.mark.parametrize("block, message", [
        ("  default: hard\n", "effort.default"),
        ("  phases: {create: lots}\n", "effort.phases.create"),
        ("  colour: blue\n", "unknown key"),
    ])
    def test_a_bad_value_fails_on_load(self, tmp_path, block, message):
        with pytest.raises(ValueError, match=message):
            pipeline.load(_config(tmp_path, block))


# -- the command line -------------------------------------------------------------


@pytest.fixture
def corpus(workspace, tmp_path, monkeypatch, fake_provider):
    seed = tmp_path / "HISTORY_list.txt"
    seed.write_text("Egyptian History\n", encoding="utf-8")
    monkeypatch.setattr(settings, "HISTORY_LIST", seed)
    history = store.path_for("history", None, "Egyptian History", LUNA)
    history.parent.mkdir(parents=True, exist_ok=True)
    history.write_text(json.dumps({"people": [], "discoveries": ["Papyrus", "Wheel"], "events": [],
                                   "cities": [], "artifacts": []}), encoding="utf-8")
    for atom_type in cli._atom_types():
        store.dir_for(atom_type).mkdir(parents=True, exist_ok=True)
    fake_provider.script(*([lambda r: json.dumps(minimal_payload(r.schema_model))] * 10))
    yield fake_provider
    journal.end_run()


class TestCommandLine:
    def test_the_dry_run_prints_the_effort_beside_the_model(self, corpus, capsys):
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", LUNA,
                         "--effort", "low", "--dry-run"]) == 0
        line = next(x for x in capsys.readouterr().out.splitlines() if x.startswith("model"))
        assert LUNA in line and "effort low  (cli)" in line

    def test_the_dry_run_says_when_the_model_ignores_it(self, corpus, capsys):
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", "gpt-4o-mini",
                         "--effort", "low", "--dry-run"]) == 0
        assert "effort low asked (cli), none sent" in capsys.readouterr().out

    def test_the_dry_run_says_when_none_is_sent(self, corpus, capsys):
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", LUNA, "--dry-run"]) == 0
        assert "effort unset (none sent" in capsys.readouterr().out

    def test_a_level_the_provider_lacks_is_refused_before_any_call(self, corpus, caplog):
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", LUNA,
                         "--effort", "max"]) == 2
        assert corpus.requests == [] and "no effort level 'max'" in caplog.text

    def test_sync_create_sends_it_and_journals_it(self, corpus):
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", LUNA,
                         "--effort", "medium"]) == 0
        assert {r.effort for r in corpus.requests} == {"medium"}
        run_dir = journal.current().directory
        lines = [json.loads(x) for x in (run_dir / journal.JOURNAL_FILENAME).read_text().splitlines()]
        assert {line.get("sent_effort") for line in lines} == {"medium"}
        assert json.loads((run_dir / journal.RUN_FILENAME).read_text())["config"]["effort"] == "medium"

    def test_sync_without_one_journals_none_and_records_no_config_key(self, corpus):
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", LUNA]) == 0
        assert {r.effort for r in corpus.requests} == {None}
        run_dir = journal.current().directory
        lines = [json.loads(x) for x in (run_dir / journal.JOURNAL_FILENAME).read_text().splitlines()]
        assert all("sent_effort" not in line for line in lines)
        assert "effort" not in json.loads((run_dir / journal.RUN_FILENAME).read_text())["config"]

    def test_a_batch_carries_it_on_every_request_the_handle_and_the_collected_line(self, corpus):
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", LUNA, "--batch",
                         "--effort", "high"]) == 0
        assert {r.effort for r in corpus.requests} == {"high"}
        [handle_path] = batch_mod.list_batches()
        assert json.loads(handle_path.read_text())["extra"]["effort"] == "high"
        corpus.batch_replies = {r.custom_id: json.dumps(minimal_payload(Discovery)) for r in corpus.requests}
        journal.start_run(config={"command": "collect"})
        batch_mod.collect(handle_path)
        lines = journal.current().entries()
        assert lines and {line.get("sent_effort") for line in lines} == {"high"}

    def test_the_index_phase_takes_it_too(self, corpus, tmp_path, monkeypatch):
        seed = tmp_path / "seed2.txt"
        seed.write_text("Greek History\n", encoding="utf-8")
        monkeypatch.setattr(settings, "HISTORY_LIST", seed)
        assert cli.main(["run", "--phase", "index", "--model", LUNA, "--batch", "--effort", "low"]) == 0
        assert {r.effort for r in corpus.requests} == {"low"}

    def test_run_batch_submit_takes_it(self, corpus):
        from universe_builder import run_batch

        assert run_batch.main(["submit", "--atom", "discovery", "--model", LUNA, "--from-history",
                               "--effort", "high"]) == 0
        assert {r.effort for r in corpus.requests} == {"high"}

    @pytest.mark.parametrize("model", ["gpt-4o-mini", "gemini-2.5-flash-lite", "claude-haiku-4-5-20251001"])
    def test_an_effort_the_model_ignores_is_recorded_as_none_sent(self, corpus, model):
        """The journal, the handle and `costs --by effort` show the wire, not
        the request: `low` asked of a model that takes none sent nothing."""
        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", model,
                         "--effort", "low", "--limit", "1"]) == 0
        run_dir = journal.current().directory
        lines = [json.loads(x) for x in (run_dir / journal.JOURNAL_FILENAME).read_text().splitlines()]
        assert lines and all("sent_effort" not in line for line in lines)
        assert [r.key for r in costs.build("effort").rows] == [f"{model} @ default"]
        journal.end_run()

        assert cli.main(["run", "--phase", "create", "--atom", "discovery", "--model", model, "--batch",
                         "--effort", "low"]) == 0
        [handle_path] = batch_mod.list_batches()
        assert "effort" not in json.loads(handle_path.read_text())["extra"]


# -- costs --------------------------------------------------------------------------


def test_costs_by_effort_tells_runs_apart(workspace):
    j = journal.start_run(config={"phase": "create"})
    for level in ("low", "high", None):
        j.record_attempt(work_item_id=f"w-{level}", raw_payload="{}", requested_model=LUNA,
                         sent_effort=level, input_tokens=1000, output_tokens=100)
    report = costs.build("effort")
    assert sorted(r.key for r in report.rows) == [
        "gpt-6-luna @ default", "gpt-6-luna @ high", "gpt-6-luna @ low"]
    journal.end_run()


def test_journal_record_call_records_the_effort_sent(workspace):
    journal.start_run(config={"command": "test"})
    response = SimpleNamespace(raw_text="{}", model=LUNA, parsed=object(), usage=None)
    journal.record_call(_request(effort="low"), response, operation="judge", atom_type="city", subject="x")
    journal.record_call(_request("gpt-4o-mini", effort="low"), response, operation="judge",
                        atom_type="city", subject="y")
    first, second = journal.current().entries()
    assert first["sent_effort"] == "low" and "sent_effort" not in second
    journal.end_run()
