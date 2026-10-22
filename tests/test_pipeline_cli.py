"""`pipeline.yaml` resolution and the `universe_builder` command line.

The point of both is that a run's configuration is inspectable *before* it costs
anything. `--dry-run` resolves exactly what the real run would and calls nothing.

No network, no API key.
"""

from __future__ import annotations

import textwrap

import pytest

from universe_builder import pipeline
from universe_builder.cli import main


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "pipeline.yaml"
    path.write_text(textwrap.dedent("""
        defaults:
          model: default-model
          max_tokens: 4096
        phases:
          create: phase-model
          fix: fix-model
        atoms:
          city:
            create: city-model
        retry:
          transport_attempts: 3
    """))
    return path


class TestResolutionOrder:
    def test_cli_override_beats_everything(self, config_file):
        config = pipeline.load(config_file)
        assert config.resolve_model("create", "city", "flag-model").value == "flag-model"

    def test_per_atom_beats_the_phase_default(self, config_file):
        config = pipeline.load(config_file)
        assert config.resolve_model("create", "city").value == "city-model"

    def test_phase_default_beats_the_global_default(self, config_file):
        config = pipeline.load(config_file)
        assert config.resolve_model("create", "person").value == "phase-model"

    def test_the_global_default_catches_an_unlisted_phase(self, config_file):
        config = pipeline.load(config_file)
        assert config.resolve_model("expand").value == "default-model"

    def test_every_resolution_reports_its_provenance(self, config_file):
        """A model name alone does not tell you which layer chose it."""
        config = pipeline.load(config_file)
        assert config.resolve_model("create", "city", "x").source == pipeline.SOURCE_CLI
        assert config.resolve_model("create", "city").source == pipeline.SOURCE_ATOM
        assert config.resolve_model("create", "person").source == pipeline.SOURCE_PHASE
        assert config.resolve_model("expand").source == pipeline.SOURCE_DEFAULT

    def test_a_phase_with_no_model_anywhere_fails_loudly(self, tmp_path):
        path = tmp_path / "p.yaml"
        path.write_text("phases: {}\n")
        with pytest.raises(KeyError):
            pipeline.load(path).resolve_model("create")

    def test_a_yaml_file_that_is_not_a_mapping_is_rejected(self, tmp_path):
        path = tmp_path / "p.yaml"
        path.write_text("- a\n- b\n")
        with pytest.raises(ValueError):
            pipeline.load(path)


class TestBundledConfig:
    def test_the_bundled_config_loads(self):
        assert pipeline.load().phases

    def test_it_is_the_source_of_truth_for_phase_models(self):
        """`config.PHASE_MODELS` is derived, not a second copy to keep in step."""
        from universe_builder.config import PHASE_MODELS

        assert pipeline.load().phase_models() == PHASE_MODELS

    def test_every_configured_model_has_a_pricing_row(self):
        """The check that stopped these drifting a generation out of date."""
        from universe_builder.providers.pricing import known_models

        priced = set(known_models())
        for phase, model in pipeline.load().phase_models().items():
            assert model in priced, f"{phase} uses {model}, which has no pricing row"

    def test_append_still_matches_create(self):
        """Locked at its cause; see the comment in pipeline.yaml."""
        models = pipeline.load().phase_models()
        assert models["append"] == models["create"]


#: A stored atom for the dry-run counts. Not `"{}"`: that validates as the
#: `{"error": ...}` placeholder (`EmptyModel`), which verify skips and, since
#: v0.1.20, the dry run does not count either.
ATOM = '{"name": {"common": "x"}}'


class TestDryRun:
    def test_dry_run_calls_nothing_and_writes_nothing(self, workspace, capsys, monkeypatch):
        def explode(*a, **kw):
            raise AssertionError("a dry run must not reach the orchestrator")

        monkeypatch.setattr("universe_builder.providers.run.complete", explode)

        assert main(["run", "--phase", "verify", "--atom", "city", "--dry-run"]) == 0
        assert list(workspace.rglob("*.json")) == []

    def test_dry_run_reports_the_resolved_model_and_its_source(self, workspace, capsys):
        main(["run", "--phase", "fix", "--atom", "city", "--model", "gpt-5-nano", "--dry-run"])

        out = capsys.readouterr().out
        assert "gpt-5-nano" in out
        assert pipeline.SOURCE_CLI in out

    def test_dry_run_lists_the_work_it_would_do(self, workspace, capsys):
        from universe_builder import store

        store.write(workspace / "city" / "city_0000000001_memphis_m.json", ATOM)
        store.write(workspace / "city" / "city_0000000002_thebes_m.json", ATOM)

        main(["run", "--phase", "verify", "--atom", "city", "--dry-run"])

        assert "work items 2" in capsys.readouterr().out

    def test_the_fix_phase_plans_only_error_files(self, workspace, capsys):
        from universe_builder import store

        store.write(workspace / "city" / "city_0000000001_memphis_m.json", ATOM)
        store.write_error(workspace / "city" / "city_0000000002_thebes_m.json", "raw")

        main(["run", "--phase", "fix", "--atom", "city", "--dry-run"])
        out = capsys.readouterr().out

        assert "work items 1" in out
        assert "thebes" in out

    def test_the_verify_phase_plans_only_valid_files(self, workspace, capsys):
        from universe_builder import store

        store.write(workspace / "city" / "city_0000000001_memphis_m.json", ATOM)
        store.write_error(workspace / "city" / "city_0000000002_thebes_m.json", "raw")

        main(["run", "--phase", "verify", "--atom", "city", "--dry-run"])
        out = capsys.readouterr().out

        assert "work items 1" in out
        assert "memphis" in out

    def test_limit_bounds_the_plan(self, workspace, capsys):
        from universe_builder import store

        for n in range(5):
            store.write(workspace / "city" / f"city_000000000{n}_c{n}_m.json", ATOM)

        main(["run", "--phase", "verify", "--atom", "city", "--limit", "2", "--dry-run"])

        assert "work items 2" in capsys.readouterr().out


class TestStatus:
    def test_status_counts_atoms_and_queued_errors(self, workspace, capsys):
        from universe_builder import store

        store.write(workspace / "city" / "city_0000000001_memphis_m.json", "{}")
        store.write_error(workspace / "city" / "city_0000000002_thebes_m.json", "raw")

        assert main(["status", "--atom", "city"]) == 0
        out = capsys.readouterr().out

        assert "1 atoms" in out
        assert "1 awaiting fix" in out
