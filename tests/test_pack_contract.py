"""What a `DomainPack` must declare for the curation verbs to work — and that
core reads it from the pack rather than knowing history's answers.

Before this file, `lint`,
`sources`, `topics` and `collisions` each carried a module-level table keyed
by history's atom names; under any other pack they silently found nothing.
Every test here runs the medications pack through the same code path and
checks that the answer is *honest* ("no evidence", not a fabricated one) and
comes from the pack.
"""

from __future__ import annotations

import json
from typing import List

import pytest
from pydantic import BaseModel

from universe_builder import atoms, registry, settings
from universe_builder.models.subatomic import Location


@pytest.fixture
def meds(workspace):
    """Switch onto the medications pack; restore history even on failure."""
    atoms.use_pack("medications")
    try:
        yield registry.active_pack()
    finally:
        atoms.use_pack("history")


# ---------------------------------------------------------------------------
# Registration-time validation of the evidence paths
# ---------------------------------------------------------------------------

class _Inner(BaseModel):
    where: Location
    year: int
    era: str


class _Thing(BaseModel):
    name: str
    place: Location
    nested: _Inner
    places: List[Location]


def _pack(**overrides):
    index = registry.AtomSpec(name="idx", model=_Thing, prompt_start="x", plural=None)
    thing = registry.AtomSpec(name="thing", model=_Thing, prompt_start="x", plural="things")
    return registry.DomainPack(name="t", index=index, atoms=(thing,), **overrides)


class TestEvidencePathsAreValidatedAtRegistration:
    def test_a_good_pack_passes(self):
        registry.validate_evidence_paths(_pack(
            coordinate_paths={"thing": ("place", "nested.where")},
            date_paths={"thing": ("nested",)}))

    def test_unknown_atom_type_is_refused(self):
        with pytest.raises(ValueError, match="unknown atom type 'widget'"):
            registry.validate_evidence_paths(_pack(coordinate_paths={"widget": ("place",)}))

    def test_a_missing_field_is_refused(self):
        with pytest.raises(ValueError, match="no wire field 'location'"):
            registry.validate_evidence_paths(_pack(coordinate_paths={"thing": ("location",)}))

    def test_a_coordinate_path_must_end_at_latitude_and_longitude(self):
        with pytest.raises(ValueError, match="has no latitude, longitude"):
            registry.validate_evidence_paths(_pack(coordinate_paths={"thing": ("nested",)}))

    def test_a_date_path_must_end_at_year_and_era(self):
        with pytest.raises(ValueError, match="has no year, era"):
            registry.validate_evidence_paths(_pack(date_paths={"thing": ("place",)}))

    def test_a_list_segment_is_refused(self):
        with pytest.raises(ValueError, match="which is a list"):
            registry.validate_evidence_paths(_pack(coordinate_paths={"thing": ("places",)}))

    def test_register_pack_runs_the_check(self, monkeypatch):
        monkeypatch.setattr(registry, "_PACKS", dict(registry._PACKS))
        with pytest.raises(ValueError, match="no wire field"):
            registry.register_pack("t", lambda: _pack(date_paths={"thing": ("nope",)}))
        assert "t" not in registry._PACKS


class TestSummaryPathsAreValidatedAtRegistration:
    """`summary_paths` (v0.4.0): each path ends at a field that holds text."""

    def test_a_good_pack_passes(self):
        registry.validate_summary_paths(_pack(summary_paths={"thing": ("name", "nested.era")}))

    def test_no_summary_paths_is_fine(self):
        registry.validate_summary_paths(_pack())

    def test_unknown_atom_type_is_refused(self):
        with pytest.raises(ValueError, match="unknown atom type 'widget'"):
            registry.validate_summary_paths(_pack(summary_paths={"widget": ("name",)}))

    def test_a_missing_field_is_refused(self):
        with pytest.raises(ValueError, match="no wire field 'summary'"):
            registry.validate_summary_paths(_pack(summary_paths={"thing": ("summary",)}))

    def test_a_field_that_cannot_hold_text_is_refused(self):
        with pytest.raises(ValueError, match="cannot hold text"):
            registry.validate_summary_paths(_pack(summary_paths={"thing": ("nested.year",)}))

    def test_a_bare_string_is_refused(self):
        with pytest.raises(ValueError, match="is a string"):
            registry.validate_summary_paths(_pack(summary_paths={"thing": "name"}))

    def test_register_pack_runs_the_check(self, monkeypatch):
        monkeypatch.setattr(registry, "_PACKS", dict(registry._PACKS))
        with pytest.raises(ValueError, match="no wire field"):
            registry.register_pack("t", lambda: _pack(summary_paths={"thing": ("nope",)}))
        assert "t" not in registry._PACKS


# ---------------------------------------------------------------------------
# Core reads the tables from the pack
# ---------------------------------------------------------------------------

class TestLintReadsEvidenceFromThePack:
    def test_history_still_compares_field_to_matching_field(self):
        from universe_builder import lint

        near = {"birth": {"location": {"latitude": 30.0, "longitude": 31.0}}}
        far = {"birth": {"location": {"latitude": 41.9, "longitude": 12.5}}}
        assert lint.corroboration("person", near, far).startswith("DISAGREES: birth.location~")

    def test_medications_has_no_comparable_evidence_and_says_so(self, meds):
        from universe_builder import lint

        a = {"name": {"common": "Aspirin"}, "drugClass": "NSAID"}
        assert lint.corroboration("medication", a, dict(a)) == "no comparable place or date"


class TestSourcesReadsThePack:
    def test_context_terms_come_from_the_coordinate_paths(self):
        from universe_builder import sources

        payload = {"birth": {"location": {"country": "Egypt", "state": None}}}
        assert sources.context_terms(payload, "person") == ["Egypt"]

    def test_medications_yields_no_context_rather_than_guessing(self, meds):
        from universe_builder import sources

        assert sources.context_terms({"name": {"common": "Aspirin"}}, "medication") == []

    def test_ranking_and_the_search_prompt_follow_the_packs_hosts(self, meds):
        from universe_builder import sources

        assert sources._rank("https://www.who.int/x") == 0
        assert sources._rank("https://en.wikipedia.org/wiki/Aspirin") == len(meds.preferred_hosts)
        line = sources._preference_line()
        assert "who.int" in line and "wikipedia" not in line.lower()

    def test_history_prefers_wikipedia(self):
        from universe_builder import sources

        assert sources._rank("https://en.wikipedia.org/wiki/Memphis") == 0
        assert "wikipedia.org" in sources._preference_line()

    def test_a_pack_with_no_hosts_gets_a_generic_preference(self, monkeypatch):
        from universe_builder import sources

        monkeypatch.setattr(sources, "_preferred_hosts", lambda: ())
        line = sources._preference_line()
        assert line.startswith("Prefer primary, institutional") and "in this order" not in line
        assert sources._rank("https://anything") == 0

    def test_the_generic_categories_follow_the_hosts(self):
        from universe_builder import sources

        line = sources._preference_line()
        assert line.startswith("Prefer, in this order: wikipedia.org")
        assert "Beyond those, prefer primary, institutional" in line


class TestTopicsReadsStopwordsFromThePack:
    def test_a_domain_stopword_does_not_make_a_near_match(self, meds):
        from universe_builder import topics

        # "tablets" is a stopword for a formulary; two drugs sharing only that
        # word are not near matches.
        assert topics.near_matches("Paracetamol tablets", ["Ibuprofen tablets"]) == []

    def test_the_same_word_counts_under_history(self):
        from universe_builder import topics

        assert topics.near_matches("Amarna tablets", ["Ebla tablets"]) == ["Ebla tablets"]


class TestCollisionsChecksThePacksAtoms:
    def test_history(self):
        from universe_builder import collisions

        assert collisions.default_atoms() == ("person", "event", "discovery", "city", "artifact",
                                              "polity", "monument", "work", "figure")

    def test_medications(self, meds):
        from universe_builder import collisions

        assert collisions.default_atoms() == ("medication",)


class TestMigrationsAreFoundBesideThePack:
    def test_bundled_packs(self, meds):
        assert registry.migrations_package(meds) == "universe_builder.domains.medications.migrations"

    def test_a_declared_package_wins(self):
        pack = _pack(migrations_package="somewhere.else")
        assert registry.migrations_package(pack) == "somewhere.else"

    def test_an_unregistered_stand_in_falls_back_to_the_bundled_convention(self):
        assert registry.migrations_package(_pack()) == "universe_builder.domains.t.migrations"


# ---------------------------------------------------------------------------
# The persona: history declares its own; the core default names no domain
# ---------------------------------------------------------------------------

class TestThePersonaIsThePacks:
    def test_core_default_names_no_domain(self):
        text = settings.BASE_SYSTEM.read_text(encoding="utf-8").lower()
        for noun in ("histor", "pharmac", "civilis", "latitude"):
            assert noun not in text, f"core base_system.txt mentions {noun!r}"

    def test_history_declares_the_historian(self):
        from universe_builder.providers.requests import base_system_text

        assert registry.active_pack().name == "history"
        assert "historian" in registry.active_pack().base_system.lower()
        assert "historian" in base_system_text().lower()

    def test_a_pack_without_a_persona_gets_the_generic_one(self, monkeypatch):
        from universe_builder.providers import requests

        pack = _pack()
        # `requests` imports `active_pack` at call time, so patch the source.
        monkeypatch.setattr(registry, "active_pack", lambda: pack)
        requests.base_system_text.cache_clear()
        try:
            assert "historian" not in requests.base_system_text().lower()
            assert "json" in requests.base_system_text().lower()
        finally:
            requests.base_system_text.cache_clear()


# ---------------------------------------------------------------------------
# The CLI offers what the pack provides, and nothing else
# ---------------------------------------------------------------------------

class TestTheCliOffersThePacksResolvers:
    def _sources_via(self):
        from universe_builder import cli

        parser = cli.build_parser()
        sub = next(a for a in parser._actions if a.dest == "command")
        sources = sub.choices["sources"]
        lint = sub.choices["lint"]
        via_s = next(a for a in sources._actions if a.dest == "via")
        via_l = next(a for a in lint._actions if a.dest == "via")
        return via_s, via_l

    def test_history_defaults_to_wikipedia(self):
        via_s, via_l = self._sources_via()
        assert via_s.default == "wikipedia" and "wikipedia" in via_s.help
        assert "wikipedia" in via_l.help

    def test_medications_has_no_default_and_says_so(self, meds):
        via_s, via_l = self._sources_via()
        assert via_s.default is None and "Required" in via_s.help
        assert "provides none" in via_l.help

    def test_collisions_takes_seed_and_still_accepts_history(self):
        from universe_builder import cli

        parser = cli.build_parser()
        assert parser.parse_args(["collisions", "--seed", "X"]).seed == "X"
        assert parser.parse_args(["collisions", "--history", "X"]).seed == "X"
        assert parser.parse_args(["collisions"]).seed is None

    def test_an_entry_point_registering_another_name_says_so(self, tmp_path, monkeypatch, isolated_registry):
        dotted = _external_module(tmp_path, monkeypatch, "pack_g", "widgets_g")

        class _EntryPoint:
            name = "cookery"

            def load(self):
                import importlib
                return importlib.import_module(dotted)

        monkeypatch.setattr(registry.metadata, "entry_points", lambda group: [_EntryPoint()])
        with pytest.raises(KeyError, match="registered \\['widgets_g'\\], not 'cookery'"):
            registry._discover("cookery")

    def test_the_unknown_domain_message_lists_the_bundled_packs(self, monkeypatch):
        monkeypatch.setattr(registry, "_active", None)
        monkeypatch.setattr(registry, "_active_name", "nope")
        with pytest.raises(KeyError, match="history, medications"):
            registry.active_pack()

    def test_help_works_and_main_reports_an_unknown_domain(self, monkeypatch, capsys, caplog):
        from universe_builder import cli

        monkeypatch.setattr(registry, "_active", None)
        monkeypatch.setattr(registry, "_active_name", "nope")
        with pytest.raises(SystemExit) as raised:
            cli.main(["--help"])
        assert raised.value.code == 0
        assert "sources" in capsys.readouterr().out
        with caplog.at_level("ERROR"):
            assert cli.main(["status"]) == 2
        assert "Unknown domain pack 'nope'" in caplog.text

    def test_sources_has_no_paid_default(self, meds, caplog):
        import argparse

        from universe_builder import cli

        parser = cli.build_parser()
        assert parser.parse_args(["sources"]).via is None
        with caplog.at_level("ERROR"):
            rc = cli.cmd_sources(argparse.Namespace(atom=None, limit=None, missing=False, via=None,
                                                    per_atom=1, model="m", apply=False))
        assert rc == 2 and "--via is required" in caplog.text

    def test_a_bogus_resolver_is_rejected_by_the_command(self, workspace, caplog):
        import argparse

        from universe_builder import cli

        with caplog.at_level("ERROR"):
            rc = cli.cmd_sources(argparse.Namespace(atom=None, limit=None, missing=False, via="bogus",
                                                    per_atom=1, model="m", apply=False))
        assert rc == 2 and "Unknown resolver 'bogus'" in caplog.text
        with caplog.at_level("ERROR"):
            rc = cli.cmd_lint(argparse.Namespace(checks=None, atom=None, review="r.tsv", via="bogus",
                                                 apply_review=None, apply=False))
        assert rc == 2 and "Unknown identity source 'bogus'" in caplog.text

    def test_an_unknown_atom_type_is_refused_not_zero(self, workspace, caplog):
        from universe_builder import cli

        with caplog.at_level("ERROR"):
            assert cli.main(["status", "--atom", "widget"]) == 2
            assert cli.main(["run", "--phase", "create", "--dry-run", "--atom", "widget"]) == 2
        assert caplog.text.count("Unknown atom type 'widget'") == 2

    def test_collisions_without_a_seed_refuses(self, workspace):
        import argparse

        from universe_builder import cli

        rc = cli.cmd_collisions(argparse.Namespace(seed=None, atom=None, review=None,
                                                   via=None, model=None, config=None))
        assert rc == 2


# ---------------------------------------------------------------------------
# Finding a pack that is not bundled: entry points and dotted module paths
# ---------------------------------------------------------------------------

_EXTERNAL_PACK = '''
from pydantic import BaseModel
from universe_builder import registry


class Widget(BaseModel):
    name: str


class Catalogue(BaseModel):
    widgets: list


def build():
    return registry.DomainPack(
        name="{name}",
        index=registry.AtomSpec(name="{name}_catalogue", model=Catalogue, prompt_start="x", plural=None),
        atoms=(registry.AtomSpec(name="{name}_widget", model=Widget, prompt_start="x", plural="widgets"),),
    )


registry.register_pack("{name}", build)
'''


@pytest.fixture
def isolated_registry(monkeypatch):
    """A registry that forgets the packs this test adds, and the modules that
    registered them (a cached `extpacks` from another tmp_path would shadow
    this test's)."""
    import sys

    monkeypatch.setattr(registry, "_PACKS", dict(registry._PACKS))
    monkeypatch.setattr(registry, "_PACK_MODULES", dict(registry._PACK_MODULES))
    yield
    for name in [m for m in sys.modules if m == "extpacks" or m.startswith("extpacks.")]:
        del sys.modules[name]


def _external_module(tmp_path, monkeypatch, module_name, pack_name):
    """Write `extpacks.<module_name>` under tmp_path and make it importable.

    A dotted path is what `_discover` treats as "a module, not a bundled
    name", so the pack lives in a package rather than as a bare module."""
    import sys

    package = tmp_path / "extpacks"
    package.mkdir(exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / f"{module_name}.py").write_text(_EXTERNAL_PACK.format(name=pack_name), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "extpacks", raising=False)
    monkeypatch.delitem(sys.modules, f"extpacks.{module_name}", raising=False)
    return f"extpacks.{module_name}"


class TestPacksOutsideThePackageAreFound:
    def test_a_dotted_module_path_registers_and_resolves_to_its_pack_name(
            self, tmp_path, monkeypatch, isolated_registry):
        dotted = _external_module(tmp_path, monkeypatch, "pack_a", "widgets_a")
        assert registry._discover(dotted) == "widgets_a"
        assert "widgets_a" in registry._PACKS
        assert registry.migrations_package(registry._PACKS["widgets_a"]()) == f"{dotted}.migrations"

    def test_a_dotted_path_imported_earlier_still_resolves(self, tmp_path, monkeypatch, isolated_registry):
        dotted = _external_module(tmp_path, monkeypatch, "pack_b", "widgets_b")
        import importlib
        importlib.import_module(dotted)
        assert registry._discover(dotted) == "widgets_b"

    def test_an_entry_point_is_loaded_by_name(self, tmp_path, monkeypatch, isolated_registry):
        dotted = _external_module(tmp_path, monkeypatch, "pack_c", "widgets_c")

        class _EntryPoint:
            name = "widgets_c"

            def load(self):
                import importlib
                return importlib.import_module(dotted)

        monkeypatch.setattr(registry.metadata, "entry_points",
                            lambda group: [_EntryPoint()] if group == registry.ENTRY_POINT_GROUP else [])
        assert registry._discover("widgets_c") == "widgets_c"
        assert "widgets_c" in registry._PACKS

    def test_an_entry_point_may_be_a_builder(self, monkeypatch, isolated_registry):
        class _EntryPoint:
            name = "t"

            def load(self):
                return _pack

        monkeypatch.setattr(registry.metadata, "entry_points", lambda group: [_EntryPoint()])
        assert registry._discover("t") == "t"
        assert registry._PACKS["t"]().name == "t"

    def test_set_active_pack_by_dotted_path(self, tmp_path, monkeypatch, isolated_registry):
        dotted = _external_module(tmp_path, monkeypatch, "pack_d", "widgets_d")
        monkeypatch.setattr(registry, "_active", None)
        monkeypatch.setattr(registry, "_active_name", "history")
        registry.set_active_pack(dotted)
        assert registry.active_pack().name == "widgets_d"
        assert settings.data_dir().name == "widgets_d"

    def test_env_var_may_hold_a_dotted_path(self, tmp_path, monkeypatch, isolated_registry):
        dotted = _external_module(tmp_path, monkeypatch, "pack_e", "widgets_e")
        monkeypatch.setattr(registry, "_active", None)
        monkeypatch.setattr(registry, "_active_name", dotted)
        assert registry.active_pack().name == "widgets_e"

    def test_a_module_that_registers_nothing_is_an_error(self, tmp_path, monkeypatch, isolated_registry):
        import sys

        package = tmp_path / "extpacks"
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "pack_f.py").write_text("X = 1\n", encoding="utf-8")
        monkeypatch.syspath_prepend(str(tmp_path))
        # An earlier test's `extpacks` (under its own tmp_path) may be cached.
        monkeypatch.delitem(sys.modules, "extpacks", raising=False)
        with pytest.raises(KeyError, match="registered no domain pack"):
            registry._discover("extpacks.pack_f")


# ---------------------------------------------------------------------------
# The seed phase is `index`; `history` is its old name and still understood
# ---------------------------------------------------------------------------

class TestTheSeedPhaseIsIndex:
    def test_the_cli_normalises_the_old_name(self, monkeypatch):
        from universe_builder import cli

        seen = {}
        monkeypatch.setitem(cli.__dict__, "cmd_run", lambda args: seen.update(phase=args.phase) or 0)
        # `main` builds its dispatch table from module globals at call time.
        rc = cli.main(["run", "--phase", "history", "--dry-run"])
        assert rc == 0 and seen["phase"] == "index"

    def test_the_config_resolves_the_old_name_as_the_new(self):
        from universe_builder import pipeline

        cfg = pipeline.load()
        assert "index" in cfg.phases and "history" not in cfg.phases
        assert cfg.resolve_model("history").value == cfg.resolve_model("index").value

    def test_a_config_file_with_the_old_key_is_read_with_a_warning(self, tmp_path, caplog):
        from universe_builder import pipeline

        path = tmp_path / "pipeline.yaml"
        path.write_text("defaults: {provider: openai, model: economy}\n"
                        "tiers: {openai: {frontier: a, balanced: b, economy: c}}\n"
                        "phases: {history: frontier}\n"
                        "atoms: {city: {history: balanced}}\n", encoding="utf-8")
        with caplog.at_level("WARNING"):
            cfg = pipeline.PipelineConfig.load(path)
        assert cfg.phases == {"index": "frontier"}
        assert cfg.atoms == {"city": {"index": "balanced"}}
        assert "now called 'index'" in caplog.text

    def test_both_names_in_one_file_is_refused(self, tmp_path):
        from universe_builder import pipeline

        path = tmp_path / "pipeline.yaml"
        path.write_text("phases: {history: frontier, index: economy}\n", encoding="utf-8")
        with pytest.raises(ValueError, match="both 'history' and 'index'"):
            pipeline.PipelineConfig.load(path)

    def test_costs_reads_old_journals_under_the_new_label(self):
        from universe_builder import costs

        assert costs._phase_of({"run_id": "r"}, {"r": {"phase": "history"}}) == "index"
        assert costs._phase_of({"run_id": "r"}, {"r": {"phase": "create"}}) == "create"


# ---------------------------------------------------------------------------
# Generated data is never inside the package
# ---------------------------------------------------------------------------

class TestTheDataRootIsNotThePackage:
    def test_default_is_under_the_working_directory(self, monkeypatch, tmp_path):
        monkeypatch.delenv("UNIVERSE_BUILDER_DATA_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        assert settings.data_dir() == tmp_path / "data" / "automated" / "history"
        assert settings.PACKAGE_DIR not in settings.data_dir().parents

    def test_a_relative_override_is_relative_to_the_working_directory(self, monkeypatch, tmp_path):
        monkeypatch.setenv("UNIVERSE_BUILDER_DATA_DIR", "out")
        monkeypatch.chdir(tmp_path)
        assert settings.data_dir() == tmp_path / "out" / "history"

    def test_dotenv_is_found_from_the_working_directory_before_the_package(self, monkeypatch, tmp_path):
        """Two competing files: the working directory's wins, and the search
        stops there (dotenv does not override an existing variable, so the
        second file's value would show if it were loaded first)."""
        import os

        cwd = tmp_path / "project" / "sub"
        cwd.mkdir(parents=True)
        (tmp_path / "project" / "pyproject.toml").write_text("", encoding="utf-8")
        (tmp_path / "project" / ".env").write_text("UNIVERSE_BUILDER_TEST_MARKER=from-cwd\n", encoding="utf-8")
        package = tmp_path / "site" / "universe_builder"
        package.mkdir(parents=True)
        (tmp_path / "site" / "pyproject.toml").write_text("", encoding="utf-8")
        (tmp_path / "site" / ".env").write_text("UNIVERSE_BUILDER_TEST_MARKER=from-package\n", encoding="utf-8")
        monkeypatch.setattr(settings, "PACKAGE_DIR", package)
        monkeypatch.delenv("UNIVERSE_BUILDER_TEST_MARKER", raising=False)
        monkeypatch.chdir(cwd)
        settings.load_environment()
        assert os.environ.get("UNIVERSE_BUILDER_TEST_MARKER") == "from-cwd"
        monkeypatch.delenv("UNIVERSE_BUILDER_TEST_MARKER", raising=False)

    def test_the_package_walk_is_the_fallback(self, monkeypatch, tmp_path):
        import os

        cwd = tmp_path / "elsewhere"
        cwd.mkdir()
        package = tmp_path / "site" / "universe_builder"
        package.mkdir(parents=True)
        (tmp_path / "site" / "pyproject.toml").write_text("", encoding="utf-8")
        (tmp_path / "site" / ".env").write_text("UNIVERSE_BUILDER_TEST_MARKER=from-package\n", encoding="utf-8")
        monkeypatch.setattr(settings, "PACKAGE_DIR", package)
        monkeypatch.delenv("UNIVERSE_BUILDER_TEST_MARKER", raising=False)
        monkeypatch.chdir(cwd)
        settings.load_environment()
        assert os.environ.get("UNIVERSE_BUILDER_TEST_MARKER") == "from-package"
        monkeypatch.delenv("UNIVERSE_BUILDER_TEST_MARKER", raising=False)

    @pytest.mark.parametrize("layout", ["above_a_project_root", "with_no_project_root"])
    def test_a_dotenv_outside_any_project_is_not_loaded(self, monkeypatch, tmp_path, layout):
        """A `~/.env` holds some other tool's keys. Neither walk passes a
        project root to reach it, and with no root above, only the starting
        directory's own `.env` counts."""
        import os

        (tmp_path / ".env").write_text("UNIVERSE_BUILDER_TEST_MARKER=from-home\n", encoding="utf-8")
        cwd = tmp_path / "project" / "sub"
        cwd.mkdir(parents=True)
        if layout == "above_a_project_root":
            (tmp_path / "project" / ".git").mkdir()
        package = tmp_path / "venv" / "site-packages" / "universe_builder"
        package.mkdir(parents=True)
        monkeypatch.setattr(settings, "PACKAGE_DIR", package)
        monkeypatch.delenv("UNIVERSE_BUILDER_TEST_MARKER", raising=False)
        monkeypatch.chdir(cwd)
        settings.load_environment()
        assert os.environ.get("UNIVERSE_BUILDER_TEST_MARKER") is None
        monkeypatch.delenv("UNIVERSE_BUILDER_TEST_MARKER", raising=False)

    def test_an_ignored_checkout_corpus_is_warned_about_once(self, monkeypatch, tmp_path, caplog):
        monkeypatch.delenv("UNIVERSE_BUILDER_DATA_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(settings, "REPO_ROOT", tmp_path / "checkout")
        (tmp_path / "checkout" / "data" / "automated").mkdir(parents=True)
        monkeypatch.setattr(settings, "_warned_about_default", False)
        with caplog.at_level("WARNING"):
            settings.data_dir()
            settings.data_dir()
        assert caplog.text.count("UNIVERSE_BUILDER_DATA_DIR is unset") == 1

    def test_config_prints_the_data_root(self, workspace, capsys):
        from universe_builder import cli

        assert cli.main(["config"]) == 0
        assert f"data directory: {settings.data_dir()}" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The remaining curation verbs under a second domain
# ---------------------------------------------------------------------------

def _stored(atom_type, subject, model="gpt-5-nano", payload=None):
    from universe_builder import store

    path = store.path_for(atom_type, None, subject, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {"name": {"common": subject}}), encoding="utf-8")
    return path


class TestDuplicatesAndRetireUnderMedications:
    def test_duplicates_finds_a_pair_by_declared_names(self, meds, tmp_path, monkeypatch):
        from universe_builder import duplicates

        _stored("medication", "Acetaminophen", model="gpt-4o-mini")
        _stored("medication", "Paracetamol", payload={
            "name": {"common": "Paracetamol", "otherNames": ["Acetaminophen"]}})
        plan = duplicates.plan()
        # One pair, whether decided (a newer model wins) or left to a person.
        assert not plan.is_clean()
        assert len(plan.resolutions) + len(plan.undecided) == 1
        text = json.dumps([r.__dict__ for r in plan.resolutions] + plan.undecided, default=str)
        assert "Paracetamol" in text and "Acetaminophen" in text

    def test_retire_plans_the_named_models_atoms(self, meds):
        from universe_builder import retire

        old = _stored("medication", "Aspirin", model="gpt-4o-mini")
        _stored("medication", "Ibuprofen", model="gpt-5-nano")
        assert retire.plan("gpt-4o-mini").paths == [old]


# ---------------------------------------------------------------------------
# Nothing in core names a history atom type, in code
# ---------------------------------------------------------------------------

_HISTORY_NOUNS = {"city", "cities", "person", "people", "artifact", "artifacts",
                  "discovery", "discoveries", "event", "events", "civilisation",
                  "civilization", "historian", "pharmacologist", "medication",
                  "expand_city", "append_city", "cityperiodsofhistory", "wikipedia"}

#: Core modules: everything under `universe_builder/` that is not a domain pack.
_CORE = [p for p in (settings.PACKAGE_DIR).rglob("*.py")
         if "domains" not in p.relative_to(settings.PACKAGE_DIR).parts]


def _string_literals(source: str):
    """Every string constant in the module body, docstrings excluded."""
    import ast

    tree = ast.parse(source)
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                docstring_nodes.add(id(node.body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstring_nodes:
            yield node.lineno, node.value


class TestCoreNamesNoDomainAtom:
    """A string literal equal to a history atom name, in a core module, is a
    table keyed by history — the class of leak this phase removed. Prose that
    *mentions* a city in a log message is not the concern; a bare `"city"` is."""

    def test_no_core_string_literal_is_a_domain_atom_name(self):
        offenders = []
        for path in _CORE:
            for lineno, value in _string_literals(path.read_text(encoding="utf-8")):
                if value.strip().lower() in _HISTORY_NOUNS:
                    offenders.append(f"{path.relative_to(settings.PACKAGE_DIR)}:{lineno}: {value!r}")
        assert not offenders, "\n".join(offenders)


# ---------------------------------------------------------------------------
# `topics._history_files` substitution
# ---------------------------------------------------------------------------

class TestSubstitutingHistoryFiles:
    """The pack's `corpus_ops.history_files_as` reassigns
    `topics._history_files` for the length of a block, in production: it is
    how the pack's `fold-history` verb asks "what would the work lists and memberships be
    with these Histories" and refuses when the answer changes. That gate only
    works while every reader looks the function up *at call time*. A refactor
    that captured it by reference — `from topics import _history_files`, a
    default argument, a module-level cache, a `history_files` alias bound once
    — would let fold-history compare the corpus with itself and pass every
    fold. `topics.history_files_as` is the supported override; until the pack
    moves to it, this is the contract.
    """

    @pytest.fixture
    def corpora(self, workspace, tmp_path, monkeypatch):
        from universe_builder import store

        seed = tmp_path / "HISTORY_list.txt"
        seed.write_text("Egyptian History\n", encoding="utf-8")
        monkeypatch.setattr(settings, "HISTORY_LIST", seed)
        stored = store.path_for("history", None, "Egyptian History", "m")
        stored.parent.mkdir(parents=True, exist_ok=True)
        stored.write_text(json.dumps({"people": ["Narmer"], "cities": ["Memphis"], "events": [],
                                      "artifacts": [], "discoveries": []}), encoding="utf-8")
        # The what-if file lives outside the data dir and carries the stored
        # one's name, as `history_files_as` requires (the context is read off it).
        elsewhere = tmp_path / "whatif" / stored.name
        elsewhere.parent.mkdir()
        elsewhere.write_text(json.dumps({"people": ["Narmer", "Djoser"], "cities": ["Thebes"], "events": [],
                                         "artifacts": [], "discoveries": []}), encoding="utf-8")
        city = store.path_for("city", None, "Thebes", "m")
        city.parent.mkdir(parents=True, exist_ok=True)
        city.write_text(json.dumps({"name": {"common": "Thebes"}}), encoding="utf-8")
        return elsewhere

    @staticmethod
    def _substitute(paths):
        """Exactly what `<pack>.corpus_ops.history_files_as` does."""
        from contextlib import contextmanager

        from universe_builder import topics

        @contextmanager
        def history_files_as():
            original = topics._history_files

            def substitute():
                yield from paths
            topics._history_files = substitute
            try:
                yield
            finally:
                topics._history_files = original
        return history_files_as()

    def _readings(self):
        from universe_builder import index, topics

        built = index.build()
        return {
            "from_history": {t: [x.subject for x in topics.from_history(t)] for t in ("person", "city")},
            "needing_create": {t: [x.subject for x in topics.needing_create(t)] for t in ("person", "city")},
            "history_files": [p.parent.name for p in topics.history_files()],
            "resolver": topics.subject_resolver("city")("Thebes", "Egyptian History"),
            "listed": sorted(topics.listed_spellings("city").values()),
            "list_index": topics.ListIndex("person").lists,
            "memberships": built["memberships"],
        }

    def test_every_reader_sees_the_substitute_and_then_the_corpus_again(self, corpora):
        from universe_builder import store

        before = self._readings()
        with self._substitute([corpora]):
            during = self._readings()
        after = self._readings()

        assert before == after
        thebes = store.atom_id("city", "Thebes")
        assert before["from_history"] == {"person": ["Narmer"], "city": ["Memphis"]}
        assert during["from_history"] == {"person": ["Narmer", "Djoser"], "city": ["Thebes"]}
        assert during["needing_create"] == {"person": ["Narmer", "Djoser"], "city": []}
        assert (before["history_files"], during["history_files"]) == (["history"], ["whatif"])
        assert during["resolver"] == thebes
        assert before["listed"] == [] and during["listed"] == [["Thebes"]]
        assert during["list_index"] == {"egyptian history": ["Narmer", "Djoser"]}
        assert before["memberships"]["egyptian history"]["city"] == {"Memphis": None}
        assert during["memberships"]["egyptian history"]["city"] == {"Thebes": f"city:{thebes}"}
        assert during["memberships"]["egyptian history"]["person"] == {"Djoser": None, "Narmer": None}

    def test_the_supported_override_reaches_every_reader_too(self, corpora):
        """`topics.history_files_as`: the same what-if
        without reassigning a private name, reaching both `from_history` and
        `index.build`, and `index.memberships` with it."""
        from universe_builder import index, topics

        before = self._readings()
        with self._substitute([corpora]):
            patched = self._readings()
        with topics.history_files_as([corpora]):
            during = self._readings()
            assert index.memberships() == during["memberships"]
        assert during == patched
        assert self._readings() == before

    def test_the_override_nests_and_survives_an_exception(self, corpora):
        from universe_builder import topics

        stored = [p.parent.name for p in topics.history_files()]
        with pytest.raises(RuntimeError):
            with topics.history_files_as([corpora]):
                with topics.history_files_as([]):
                    assert topics.history_files() == []
                assert [p.parent.name for p in topics.history_files()] == ["whatif"]
                raise RuntimeError("boom")
        assert [p.parent.name for p in topics.history_files()] == stored == ["history"]

    def test_the_name_is_a_module_attribute_the_readers_look_up(self):
        """A cheap tripwire beside the behavioural test: no reader may hold
        its own reference."""
        import inspect

        from universe_builder import index, topics

        assert "_history_files" not in vars(index)
        for reader in (topics._raw_occurrences, topics.history_files, topics.ListIndex.__init__):
            assert "_history_files()" in inspect.getsource(reader)
            assert "_history_files" not in inspect.signature(reader).parameters
