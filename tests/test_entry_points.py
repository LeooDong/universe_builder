"""Every entry point must load the `.env` itself.

Credentials live in a `.env` file that `settings.load_environment()` reads. For a
long time the only thing that called it was an import side-effect at the top of
`llm_classes/llm_call.py`, so an entry point loaded credentials if and only if it
happened to import that module. `execute_prompt` does; `cli` and `run_batch` do
not, and both failed with "Missing credentials" from a fresh shell even though
the key was on disk the whole time.

These tests assert the call happens in `main()` - the place a program starts, and
the only place where reading the user's dotenv is unambiguously right. Asserting
it in the provider constructor instead would reintroduce the same class of bug:
credentials appearing as a side-effect of an unrelated import.
"""

from __future__ import annotations

import pytest

from universe_builder import settings


@pytest.fixture
def records_env_load(monkeypatch):
    calls = []
    monkeypatch.setattr(settings, "load_environment", lambda: calls.append(True))
    return calls


class TestEveryEntryPointLoadsTheEnvironment:
    def test_run_batch_main_loads_it(self, records_env_load, workspace, capsys):
        from universe_builder import run_batch

        assert run_batch.main(["list"]) == 0
        assert records_env_load, (
            "run_batch.main did not call settings.load_environment; every command "
            "that builds a provider will fail with 'Missing credentials'"
        )

    def test_cli_main_loads_it(self, records_env_load, workspace, capsys):
        from universe_builder import cli

        assert cli.main(["config"]) == 0
        assert records_env_load, (
            "cli.main did not call settings.load_environment; `universe_builder run` will fail "
            "with 'Missing credentials'"
        )

    def test_execute_prompt_loads_it_by_importing_llm_call(self):
        """Pins the mechanism the other two deliberately do not rely on.

        `execute_prompt` gets its environment from `llm_call`'s import side-effect.
        That is fragile rather than wrong, and it is recorded here so that removing
        the side-effect shows up as a failure naming its consequence.
        """
        import universe_builder.execute_prompt  # noqa: F401
        import sys

        assert any("llm_call" in name for name in sys.modules), (
            "execute_prompt no longer imports llm_call, so it no longer loads the "
            ".env; give it an explicit settings.load_environment() call"
        )


class TestTheBatchAndSyncPathsAgreeOnTheOutputBudget:
    """`max_tokens` had two defaults: 16384 on the sync path, 8192 on the batch path.

    The constant lived in `llm_classes/provider_call.py`, which the batch path
    never imports, so `run_batch submit` kept the old value. A full
    `Discovery` was measured at 8,939 output tokens, so a default batch
    submission would have truncated nearly every record - and a batch reports
    that a day later as a validation failure, not immediately.
    """

    #: The largest output actually observed in a live run.
    MEASURED_WORST_CASE = 8939

    def test_one_constant_backs_every_default(self):
        from universe_builder.providers import batch, requests, types
        from universe_builder.llm_classes import provider_call
        import inspect

        expected = types.DEFAULT_MAX_TOKENS
        assert provider_call.DEFAULT_MAX_TOKENS is expected
        assert types.LLMRequest.max_tokens == expected
        for fn in (requests.build_request, batch.submit):
            got = inspect.signature(fn).parameters["max_tokens"].default
            assert got == expected, f"{fn.__qualname__} defaults to {got}, not {expected}"

        # `run_batch submit --max-tokens` defaults to the config's budget
        # rather than the constant; see the next test.

    def test_run_batch_submit_defaults_to_the_configured_budget(self, workspace, monkeypatch, tmp_path):
        """`run_batch submit` is a batch path too, and ran at the constant
        while `run --batch` ran at `defaults.max_tokens` — the same phase,
        two budgets, depending on which door it was submitted through."""
        from universe_builder import pipeline, run_batch
        from universe_builder.providers import batch

        seen = {}

        def fake_submit(atom, subjects, model, **kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr(batch, "submit", fake_submit)
        topics = tmp_path / "topics.txt"
        topics.write_text("Memphis\n", encoding="utf-8")

        assert run_batch.main(["submit", "--atom", "city", "--model", "gpt-5-nano",
                               "--list", str(topics)]) == 0
        assert seen["max_tokens"] == pipeline.load().max_tokens("create")

        assert run_batch.main(["submit", "--atom", "city", "--model", "gpt-5-nano",
                               "--list", str(topics), "--max-tokens", "999"]) == 0
        assert seen["max_tokens"] == 999

    def test_the_default_clears_the_largest_atom_ever_measured(self):
        from universe_builder.providers.types import DEFAULT_MAX_TOKENS

        assert DEFAULT_MAX_TOKENS > self.MEASURED_WORST_CASE, (
            f"{DEFAULT_MAX_TOKENS} is below the {self.MEASURED_WORST_CASE} output "
            "tokens a Discovery actually used; reasoning models spend the budget "
            "before emitting, so this truncates and returns nothing"
        )


class TestTheCostEstimateErrsHigh:
    """`estimate` exists to be trusted before money is spent.

    Its `--output-tokens` default was 1500, against a measured ~8,939 for a
    `Discovery`. That understated a large batch several-fold, in the one
    direction that costs money.

    The estimator makes no API call: it builds a request locally and prices it
    from `model_pricing.json`, so this runs offline like everything else here.
    """

    def test_the_default_sits_between_the_measurement_and_the_budget(self):
        from universe_builder import run_batch
        from universe_builder.providers.types import DEFAULT_MAX_TOKENS

        assert run_batch.MEASURED_OUTPUT_TOKENS >= 8939, (
            "the estimate default is below the largest output actually observed "
            "(8,939 for a Discovery), so every estimate understates"
        )
        assert run_batch.MEASURED_OUTPUT_TOKENS <= DEFAULT_MAX_TOKENS, (
            "estimating more output than max_tokens permits describes a call that "
            "cannot happen"
        )

    def test_running_estimate_uses_it_and_says_where_it_came_from(self, workspace, capsys):
        from universe_builder import run_batch

        assert run_batch.main(
            ["estimate", "--atom", "discovery", "--model", "gpt-5-nano", "--count", "10"]
        ) == 0
        out = capsys.readouterr().out
        assert f"assumed output {run_batch.MEASURED_OUTPUT_TOKENS} tokens/call" in out, out
        assert "measured" in out, (
            "the estimate does not say its output figure is measured, so a reader "
            "cannot tell a measurement from a guess"
        )

    def test_an_explicit_figure_is_not_labelled_as_measured(self, workspace, capsys):
        """The provenance label has to track the number it describes."""
        from universe_builder import run_batch

        run_batch.main(
            ["estimate", "--atom", "discovery", "--model", "gpt-5-nano",
             "--count", "10", "--output-tokens", "1500"]
        )
        out = capsys.readouterr().out
        assert "assumed output 1500 tokens/call (your figure)" in out, out


class TestListStaysUsefulOnceHandlesAreArchived:
    """`list` returned early when no handle was outstanding.

    That was harmless while handles accumulated forever. Archiving them after
    collection makes "nothing outstanding" the normal state, so the early return
    hid the archive and the lost-results report precisely when they were the only
    thing left to show. Found by running `list` after the first live collect.
    """

    def test_it_reports_the_archive_with_nothing_outstanding(self, workspace, capsys):
        from universe_builder import run_batch, settings

        settings.collected_dir().mkdir(parents=True, exist_ok=True)
        (settings.collected_dir() / "20260911T000000Z-discovery.json").write_text("{}")

        assert run_batch.main(["list"]) == 0
        out = capsys.readouterr().out
        assert "1 already collected" in out, out
        assert "re-collect one deliberately" in out, (
            "the archive is mentioned but not how to get a batch back out of it"
        )

    def test_it_still_reports_lost_results_with_nothing_outstanding(self, workspace, capsys):
        from universe_builder import run_batch, settings

        settings.missing_dir().mkdir(parents=True, exist_ok=True)
        (settings.missing_dir() / "20260911T000000Z-discovery.json").write_text('["a", "b"]')

        assert run_batch.main(["list"]) == 0
        out = capsys.readouterr().out
        assert "lost results" in out and "2 missing" in out, out


class TestHintsNameTheCommandThatWasRun:
    """A wrapper installed under another name (a pack's own command, here
    `mypack`) passes `prog=`, and every "now run this" hint must say that name.
    Once the usage line did and the hand-written hints still said
    `universe_builder`, which under `mypack` is a different command — one that
    does not load the pack's `.env`."""

    def test_the_cli_hints_use_the_prog(self, workspace, capsys, caplog):
        from tests.factories import minimal_instance
        from universe_builder import cli, store
        from universe_builder.domains.history.models.city import City

        assert cli.main(["status", "--coverage"], prog="mypack") == 1
        assert cli.main(["run", "--phase", "create", "--atom", "history"], prog="mypack") == 2
        store.write_atom(store.path_for("city", None, "Memphis", "gpt-4o-mini"),
                         minimal_instance(City))
        assert cli.main(["retire", "--model", "gpt-4o-mini"], prog="mypack") == 1

        said = capsys.readouterr().out + caplog.text
        assert "mypack run --phase index" in said
        assert "mypack run --phase create" in said
        assert "universe_builder run" not in said

    def test_the_help_text_uses_the_prog(self):
        from universe_builder import cli

        parser = cli.build_parser("mypack")
        collisions = parser._subparsers._group_actions[0].choices["collisions"]
        assert "`mypack lint --apply-review FILE`" in " ".join(collisions.format_help().split())

    def test_run_batch_hints_use_the_prog(self, workspace):
        from universe_builder import run_batch

        with pytest.raises(SystemExit) as caught:
            run_batch.main(["submit", "--atom", "city", "--model", "gpt-5-nano",
                            "--from-history"], prog="mypack")
        assert "mypack run --phase index" in str(caught.value)
        assert "universe_builder run" not in str(caught.value)
