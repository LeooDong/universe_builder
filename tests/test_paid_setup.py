"""`cli._paid_setup`: the one setup the paid core verbs share,
and the `--model/--config/--provider` flags each of them now takes."""

from __future__ import annotations

import argparse
import json

import pytest

from universe_builder import cli, journal, pipeline
from universe_builder.providers import run as provider_run


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(
        "defaults:\n  provider: openai\n  model: economy\n"
        "phases:\n  index: frontier\n"
        "retry:\n  attempts: 7\n"
        "tiers:\n"
        "  openai: {frontier: f-model, balanced: b-model, economy: e-model}\n"
        "  anthropic: {frontier: claude-f}\n"
        "  openai-search: {economy: search-e}\n",
        encoding="utf-8")
    return path


def _run_json():
    run = journal.current()
    return json.loads((run.directory / "run.json").read_text(encoding="utf-8"))


class TestPaidSetup:
    def test_resolves_configures_announces_and_opens_a_run(self, workspace, config_file, monkeypatch):
        seen = []
        monkeypatch.setattr(provider_run, "configure", lambda retry: seen.append(("retry", dict(retry))))
        args = argparse.Namespace(config=str(config_file), provider=None)
        try:
            model = cli._paid_setup(args, models_key="duplicates",
                                    journal_config={"command": "duplicates", "via": "model",
                                                    "limit": 3},
                                    announce=lambda m: seen.append(("announce", m)))
            written = _run_json()
        finally:
            journal.end_run()
        assert model == "f-model", "the index phase's tier, from --config"
        assert seen == [("retry", {"attempts": 7}), ("announce", "f-model")]
        assert list(written["config"]) == ["command", "via", "limit", "model"]
        assert written["config"]["model"] == "f-model"
        assert written["models"] == {"duplicates": "f-model"}

    def test_the_model_key_keeps_its_place_in_the_run_config(self, workspace, config_file):
        args = argparse.Namespace(config=str(config_file))
        try:
            cli._paid_setup(args, search=True, model="economy",
                            journal_config={"command": "sources", "via": "search", "model": None,
                                            "atom": None, "limit": None})
            written = _run_json()
        finally:
            journal.end_run()
        assert list(written["config"]) == ["command", "via", "model", "atom", "limit"]
        assert written["config"]["model"] == "search-e", "a search tier resolves on its own table"
        assert written["models"] is None

    def test_provider_picks_the_tier_table(self, workspace, config_file):
        args = argparse.Namespace(config=str(config_file), provider="anthropic")
        try:
            assert cli._paid_setup(args, model="frontier", journal_config={"command": "x"}) == "claude-f"
        finally:
            journal.end_run()

    def test_an_unknown_tier_opens_nothing(self, workspace, config_file, monkeypatch):
        monkeypatch.setattr(provider_run, "configure", lambda retry: pytest.fail("configured"))
        args = argparse.Namespace(config=str(config_file), provider="anthropic")
        with pytest.raises(pipeline.UnknownTier):
            cli._paid_setup(args, model="economy", journal_config={"command": "x"})
        assert isinstance(journal.current(), journal.NullJournal)


class TestUniformFlags:
    @pytest.mark.parametrize("argv,expected", [
        (["duplicates", "--model", "gpt-x"], {"triage_model": "gpt-x"}),
        (["duplicates", "--triage-model", "gpt-x"], {"triage_model": "gpt-x"}),
        (["duplicates", "--config", "p.yaml", "--provider", "anthropic"],
         {"config": "p.yaml", "provider": "anthropic"}),
        (["collisions", "--seed", "S", "--provider", "anthropic"], {"provider": "anthropic"}),
        (["sources", "--via", "search", "--config", "p.yaml"], {"config": "p.yaml"}),
    ])
    def test_each_paid_verb_takes_them(self, argv, expected):
        args = cli.build_parser().parse_args(argv)
        assert {k: getattr(args, k) for k in expected} == expected

    def test_defaults_change_nothing(self):
        for argv in (["duplicates"], ["collisions", "--seed", "S"], ["sources", "--via", "search"]):
            args = cli.build_parser().parse_args(argv)
            assert getattr(args, "config") is None
            assert getattr(args, "provider", None) is None
