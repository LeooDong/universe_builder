"""The retry/timeout policy: pipeline.yaml's `retry:` block, actually applied.

Once that block was only displayed by `universe_builder config` and consumed by
nothing — `run.complete` hardcoded its own attempts and backoff, the SDK kept
its 600s default timeout with 2 internal retries nested inside our 3 attempts,
and a single dead call could take half an hour to surface. `providers.run`
now owns the one retry loop; entry points install the block with
`configure()`; the timeout rides the request into the SDK call.

No network: providers are stubbed or their clients mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from universe_builder.providers import run as run_mod
from universe_builder.providers.types import LLMRequest, LLMResponse, Usage


class Tiny(BaseModel):
    city: str


def make_request(**kwargs) -> LLMRequest:
    defaults = dict(model="gpt-5-nano", schema_model=Tiny, system="s", user="u")
    defaults.update(kwargs)
    return LLMRequest(**defaults)


# The policy itself is reset around every test by conftest's
# `default_run_policy`, so these tests only install what they assert on.


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(run_mod.time, "sleep", lambda *_: None)


class _FailingProvider:
    def __init__(self):
        self.calls = 0

    def complete(self, request):
        self.calls += 1
        raise ConnectionError("socket closed")


class _EchoProvider:
    """Succeeds, and remembers the request it was handed."""

    def __init__(self):
        self.seen = None

    def complete(self, request):
        self.seen = request
        return LLMResponse('{"city": "x"}', Tiny(city="x"), request.model,
                           Usage(), request.custom_id)


class TestConfigureIsApplied:
    def test_transport_attempts_come_from_the_policy(self, monkeypatch):
        provider = _FailingProvider()
        monkeypatch.setattr(run_mod, "get_provider", lambda model: provider)
        run_mod.configure({"transport_attempts": 5})

        response = run_mod.complete(make_request())

        assert not response.ok
        assert provider.calls == 5, "the yaml said five attempts; the code did its own thing"

    def test_the_defaults_match_the_old_hardcoded_behaviour(self, monkeypatch):
        """Unconfigured (library use, old tests): 3 transport attempts."""
        provider = _FailingProvider()
        monkeypatch.setattr(run_mod, "get_provider", lambda model: provider)

        run_mod.complete(make_request())

        assert provider.calls == 3

    def test_explicit_arguments_still_override_the_policy(self, monkeypatch):
        provider = _FailingProvider()
        monkeypatch.setattr(run_mod, "get_provider", lambda model: provider)
        run_mod.configure({"transport_attempts": 5})

        run_mod.complete(make_request(), transport_retries=0)

        assert provider.calls == 1

    def test_an_unknown_key_is_warned_not_silently_ignored(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            run_mod.configure({"transport_attempts": 2, "timeout": 99})

        assert "not recognised" in caplog.text
        assert "timeout" in caplog.text

    def test_configure_replaces_rather_than_accumulates(self, monkeypatch):
        provider = _FailingProvider()
        monkeypatch.setattr(run_mod, "get_provider", lambda model: provider)
        run_mod.configure({"transport_attempts": 5})
        run_mod.configure({})  # back to defaults, not five-with-tweaks

        run_mod.complete(make_request())

        assert provider.calls == 3


class TestTimeoutReachesTheRequest:
    def test_the_policy_timeout_lands_on_the_request(self, monkeypatch):
        provider = _EchoProvider()
        monkeypatch.setattr(run_mod, "get_provider", lambda model: provider)
        run_mod.configure({"timeout_seconds": 120})

        run_mod.complete(make_request())

        assert provider.seen.timeout == 120

    def test_a_request_that_carries_its_own_timeout_wins(self, monkeypatch):
        provider = _EchoProvider()
        monkeypatch.setattr(run_mod, "get_provider", lambda model: provider)
        run_mod.configure({"timeout_seconds": 120})

        run_mod.complete(make_request(timeout=7))

        assert provider.seen.timeout == 7

    def test_no_policy_means_the_sdk_default(self, monkeypatch):
        provider = _EchoProvider()
        monkeypatch.setattr(run_mod, "get_provider", lambda model: provider)

        run_mod.complete(make_request())

        assert provider.seen.timeout is None


CUSTOM_YAML = """
defaults:
  provider: openai
  model: economy
tiers:
  openai:
    frontier: gpt-6-astra
    balanced: gpt-5.6-luna
    economy: gpt-4o-mini
phases:
  create: economy
retry:
  transport_attempts: 7
  timeout_seconds: 9
"""


class TestTheSelectedConfigIsHonoured:
    """`--config` must govern the run, not just its announcement.

    The review-gate case: `cmd_run` installed the `--config` file's retry
    policy and resolved its models for the dry-run display and the journal,
    then handed off to `run_phase` — which re-read the *default* file for
    both. An alternate config governed what the run said and a different one
    governed what it did.
    """

    @pytest.fixture
    def custom_config(self, tmp_path):
        path = tmp_path / "custom.yaml"
        path.write_text(CUSTOM_YAML, encoding="utf-8")
        return str(path)

    def test_run_phase_installs_the_config_it_was_given(self, workspace, custom_config):
        from universe_builder import execute_prompt

        execute_prompt.run_phase("create", atoms=["city"], config=custom_config)

        assert run_mod._policy.transport_attempts == 7
        assert run_mod._policy.timeout_seconds == 9

    def test_run_phase_resolves_models_from_the_config_it_was_given(self, custom_config):
        from universe_builder import execute_prompt

        resolved = execute_prompt.model_for_atom("create", "city", config=custom_config)

        assert resolved == "gpt-4o-mini", (
            "the model came from the default pipeline.yaml, not the selected config"
        )

    def test_the_cli_forwards_its_config_to_the_run(self, workspace, custom_config):
        """End to end: `universe_builder run --config X` leaves X's policy installed."""
        from universe_builder import cli

        cli.main(["run", "--phase", "create", "--atom", "city",
                  "--config", custom_config])

        assert run_mod._policy.transport_attempts == 7
        assert run_mod._policy.timeout_seconds == 9


class TestTheSdkCallGetsTheOptions:
    """The provider hands the timeout to the SDK and turns its retries off.

    `max_retries=0` is the point: two retry loops nested invisibly is how one
    failure took half an hour. `run.complete` is the loop that logs.
    """

    def _provider(self):
        from universe_builder.providers.openai_provider import OpenAIProvider

        provider = OpenAIProvider.__new__(OpenAIProvider)
        provider._client = MagicMock()
        provider._client.with_options.return_value = provider._client
        stream_end = MagicMock()
        stream_end.usage = None
        choice = MagicMock()
        choice.delta.content = '{"city": "x"}'
        choice.finish_reason = "stop"
        stream_end.choices = [choice]
        provider._client.chat.completions.create.return_value = iter([stream_end])
        return provider

    def test_timeout_and_no_sdk_retries(self, fake_keys):
        provider = self._provider()

        provider.complete(make_request(timeout=120))

        options = provider._client.with_options.call_args.kwargs
        assert options == {"max_retries": 0, "timeout": 120}

    def test_without_a_timeout_the_sdk_default_is_kept(self, fake_keys):
        """`timeout=None` would DISABLE the SDK timeout; omitting keeps 600s."""
        provider = self._provider()

        provider.complete(make_request())

        options = provider._client.with_options.call_args.kwargs
        assert options == {"max_retries": 0}


BUDGET_YAML = """
defaults:
  provider: openai
  model: economy
  max_tokens: 12345
tiers:
  openai:
    frontier: gpt-6-astra
    balanced: gpt-5.6-luna
    economy: gpt-4o-mini
"""


class TestTheTokenBudgetReachesTheSyncPath:
    """`defaults.max_tokens` governs synchronous calls, not only batch ones.

    Once only `cmd_run`'s batch branch read it (`_submit_batch`);
    `OperationConfig` had no budget, so every synchronous call fell back to
    `DEFAULT_MAX_TOKENS` — the same 16,384 that truncated cities
    on the batch path before that was fixed. A phase run synchronously and
    the same phase batched must ask for the same output budget.
    """

    MODEL = "claude-sonnet-5"

    @pytest.fixture
    def budget_config(self, tmp_path):
        path = tmp_path / "budget.yaml"
        path.write_text(BUDGET_YAML, encoding="utf-8")
        return str(path)

    def _seed(self, phase):
        from tests.factories import minimal_instance
        from universe_builder import store
        from universe_builder.domains.history.models.city import City

        target = store.path_for("city", None, "Memphis", self.MODEL)
        seed = minimal_instance(City, cityPeriodsOfHistory=["Roman"])
        if phase == "fix":
            store.write_error(target, seed.model_dump_json())
        else:
            store.write_atom(target, seed)

    @pytest.mark.parametrize("phase", ["fix", "verify", "expand", "append"])
    def test_every_augment_phase_asks_for_the_configured_budget(
            self, ep, workspace, monkeypatch, fake_call_factory, budget_config, phase):
        self._seed(phase)
        Fake = fake_call_factory(["{}"] * 4)
        monkeypatch.setitem(ep.llm_call_mapping, (self.MODEL, "augment"), Fake)

        ep.run_phase(phase, atoms=["city"], model=self.MODEL, config=budget_config)

        assert Fake.calls, "precondition: the phase made a call"
        assert {c.get("max_tokens") for c in Fake.calls} == {12345}

    def test_create_asks_for_the_configured_budget(
            self, ep, workspace, monkeypatch, fake_call_factory, budget_config):
        from universe_builder import atoms, store
        from universe_builder.domains.medications.models import Formulary

        atoms.use_pack("medications")
        try:
            store.write_atom(store.path_for("formulary", None, "Test", self.MODEL),
                             Formulary(medications=["Aspirin"]))
            Fake = fake_call_factory(["{}"])
            monkeypatch.setitem(ep.llm_call_mapping, self.MODEL, Fake)

            ep.run_phase("create", atoms=["medication"], model=self.MODEL, config=budget_config)
        finally:
            atoms.use_pack("history")

        assert [c.get("max_tokens") for c in Fake.calls] == [12345]

    def test_sync_and_batch_agree_on_the_bundled_budget(self):
        """The bundled file's budget is not the provider default, so a path
        that silently fell back would be visible here."""
        from universe_builder import pipeline
        from universe_builder.operations._support import call_budget
        from universe_builder.config import OperationConfig
        from universe_builder.providers.types import DEFAULT_MAX_TOKENS

        budget = pipeline.load().max_tokens("create")
        assert budget != DEFAULT_MAX_TOKENS, "precondition: the fallback would be visible"
        config = OperationConfig(atom_name="city", model_version=self.MODEL,
                                 base_system_text="", max_tokens=budget)
        assert call_budget(config) == {"max_tokens": budget}
        assert call_budget(OperationConfig(atom_name="city", model_version=self.MODEL,
                                           base_system_text="")) == {}
