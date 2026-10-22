"""Capability tiers, and the per-atom overrides that were silently ignored.

`pipeline.yaml` used to name concrete model ids. Model ids turn over every few
months, so a config full of them is a config that is always slightly wrong, and
switching vendor meant rewriting every line. Phases now name a *tier* —
`frontier`, `balanced`, `economy` — which resolves through the active provider's
table.

The per-atom half is a bug fix rather than a feature. `atoms.<atom>.<phase>` was
documented, implemented in `resolve_model`, and **unreachable in practice**: the
CLI resolved once with `atom=None`, which never matches an atom override, then
handed the single answer to `run_phase` for every atom type. The override only
worked if you passed `--atom`.
"""

from __future__ import annotations

import pytest

from universe_builder import pipeline


@pytest.fixture
def cfg():
    return pipeline.load()


class TestTiersResolve:
    @pytest.mark.parametrize("tier", pipeline.TIERS)
    def test_every_provider_defines_every_tier(self, cfg, tier):
        """A half-filled table fails at run time, on the one phase that uses it."""
        for provider, table in cfg.tiers.items():
            assert tier in table, f"{provider} has no {tier!r} tier"

    def test_a_tier_resolves_through_the_active_provider(self, cfg):
        assert cfg.resolve_tier("economy", "openai") == cfg.tiers["openai"]["economy"]
        assert cfg.resolve_tier("economy", "anthropic") == cfg.tiers["anthropic"]["economy"]

    def test_an_exact_model_id_passes_through(self, cfg):
        """The escape hatch: pinning one phase must not mean abandoning tiers."""
        assert cfg.resolve_tier("gpt-4o-mini", "openai") == "gpt-4o-mini"

    def test_an_unknown_provider_says_which_exist(self, cfg):
        with pytest.raises(pipeline.UnknownTier, match="Known:"):
            cfg.resolve_tier("economy", "llama")

    def test_an_unknown_tier_says_which_exist(self, cfg):
        cfg.tiers["openai"].pop("frontier")
        with pytest.raises(pipeline.UnknownTier, match="frontier"):
            cfg.resolve_tier("frontier", "openai")

    def test_the_provenance_names_both_layers(self, cfg):
        """A dry run has to explain a model that appears nowhere in the config."""
        resolved = cfg.resolve_model("create", provider="openai")

        assert "tiers.openai." in resolved.source, resolved.source
        assert "phases" in resolved.source, resolved.source


class TestTiersAreOrderedByPrice:
    """frontier >= balanced >= economy, per provider.

    Not cosmetic: the whole point of naming a tier is that `economy` is the cheap
    one everywhere. A table that gets this backwards silently makes the bulk pass
    the expensive one.
    """

    def test_each_provider_orders_correctly(self, cfg, subtests):
        from universe_builder.providers.pricing import lookup

        for provider, table in cfg.tiers.items():
            with subtests.test(provider=provider):
                costs = []
                for tier in pipeline.TIERS:
                    m = lookup(table[tier])
                    costs.append((tier, m.input + m.output))
                for (hi, hi_cost), (lo, lo_cost) in zip(costs, costs[1:]):
                    assert hi_cost >= lo_cost, (
                        f"{provider}: {hi} ({table[hi]}, {hi_cost}) is cheaper than "
                        f"{lo} ({table[lo]}, {lo_cost})"
                    )


class TestPerAtomOverridesAreReached:
    def test_resolution_differs_by_atom(self, cfg):
        # The unoverridden atom is compared against whatever `phases.create`
        # resolves to, not a hardcoded tier. These assertions used to name
        # `economy` directly and broke when the phase moved to
        # balanced — a config edit, not a resolver regression. The claim being
        # tested is that the override reaches `person` and does *not* leak to
        # `city`, which is true whatever the phase default happens to be.
        cfg.atoms = {"person": {"create": "frontier"}}
        phase_default = cfg.tiers["openai"][cfg.phases["create"]]

        assert cfg.resolve_model("create", "person").value == cfg.tiers["openai"]["frontier"]
        assert cfg.resolve_model("create", "city").value == phase_default

    def test_the_cli_shows_every_atom_when_they_differ(self, workspace, capsys, tmp_path):
        """The dry run printed one model for the whole run, hiding the override.

        Carries its own config with an override in it. It used to lean on the
        bundled `pipeline.yaml` having an `atoms.person.create` pin, so when
        that pin was removed — `phases.create` went balanced for
        every atom, saying the same thing once — the test failed despite the
        CLI behaving correctly. A test of "the CLI reveals an override" must
        supply the override itself.
        """
        from universe_builder import cli

        cfg = tmp_path / "pipeline.yaml"
        cfg.write_text(
            "defaults:\n  provider: openai\n  model: economy\n"
            "tiers:\n  openai:\n    frontier: gpt-6-astra\n"
            "    balanced: gpt-5.6-luna\n    economy: gpt-5-nano\n"
            "phases:\n  create: economy\n"
            "atoms:\n  person:\n    create: frontier\n"
        )

        assert cli.main(["run", "--phase", "create", "--dry-run", "--config", str(cfg)]) == 0
        out = capsys.readouterr().out

        assert "they differ by atom type" in out, out
        assert "atoms.<atom>.<phase>" in out, (
            "no atom override is visible; either the config lost them or the CLI "
            "is resolving once for the whole run again"
        )

    def test_run_phase_is_handed_the_override_not_a_resolved_model(self, workspace, monkeypatch):
        """The fix: `--model` goes through unresolved so `run_phase` can resolve.

        Passing the already-resolved value is precisely what discarded every
        per-atom override.
        """
        from universe_builder import cli, execute_prompt

        seen = {}

        def fake_run_phase(phase, atoms=None, model=None, limit=None, provider=None,
                           config=None, explicit=False):
            seen.update(phase=phase, model=model, provider=provider)
            return 0

        monkeypatch.setattr(execute_prompt, "run_phase", fake_run_phase)
        cli.main(["run", "--phase", "create"])

        assert seen["model"] is None, (
            f"cli handed run_phase a pre-resolved {seen['model']!r}; per-atom "
            "overrides cannot survive that"
        )

    def test_an_explicit_model_still_wins_everywhere(self, workspace, capsys):
        from universe_builder import cli

        cli.main(["run", "--phase", "create", "--model", "gpt-4o-mini", "--dry-run"])
        out = capsys.readouterr().out

        assert "gpt-4o-mini" in out
        assert "they differ by atom type" not in out, out


class TestProviderSwitching:
    def test_one_flag_swaps_every_model(self, workspace, capsys):
        from universe_builder import cli

        cli.main(["run", "--phase", "create", "--provider", "anthropic", "--dry-run"])
        out = capsys.readouterr().out

        assert "claude" in out
        assert "gpt-" not in out, out

    def test_an_unknown_provider_exits_2_rather_than_traceback(self, workspace, caplog):
        from universe_builder import cli

        assert cli.main(["run", "--phase", "create", "--provider", "llama", "--dry-run"]) == 2
        assert "No tier table for provider" in caplog.text


class TestTheRegistryIsDerivedFromTheTiers:
    def test_every_tier_model_is_callable(self, cfg):
        """Maintaining the list by hand was a second registry to keep in step."""
        from universe_builder.config import llm_call_mapping

        for provider, table in cfg.tiers.items():
            for tier, model in table.items():
                assert model in llm_call_mapping, (
                    f"tiers.{provider}.{tier} names {model}, which nothing can call"
                )


class TestTheThreeExecutionRegressions:
    """Paths the tier/per-atom work broke, none of which the suite covered.

    All three shipped in `8cf1398` and were found by exercising commands rather
    than by running tests — which is the point: every one of them is an entry
    point the tests drive through a different door.
    """

    def test_a_phase_with_no_applicable_atom_says_so(self, workspace, caplog):
        """`--phase create --atom history` indexed an empty list.

        `create` never generates History atoms, so the resolution table filters
        the only atom asked for and leaves nothing. The result was a traceback.
        """
        from universe_builder import cli

        assert cli.main(["run", "--phase", "create", "--atom", "history", "--dry-run"]) == 2
        assert "does not run over history" in caplog.text
        assert "--phase index" in caplog.text, "it should say what to run instead"

    def test_one_resolver_backs_both_paths(self):
        """The CLI and the old `main()` disagreed because they each answered the question."""
        from universe_builder import execute_prompt as ep

        assert hasattr(ep, "model_for_atom")

    def test_run_batch_accepts_a_tier_name(self, workspace):
        """`pipeline.yaml` speaks tiers, so `--model economy` is what you type.

        It used to fail inside the pricing table with "'economy' is not in the
        pricing table", which points at the wrong problem entirely.
        """
        from universe_builder import pipeline, run_batch

        class Args:
            model = "economy"

        assert run_batch._model(Args()) == pipeline.load().tiers["openai"]["economy"]

    def test_run_batch_still_takes_an_exact_model(self, workspace):
        from universe_builder import run_batch

        class Args:
            model = "gpt-4o-mini"

        assert run_batch._model(Args()) == "gpt-4o-mini"

    def test_run_batch_rejects_an_unknown_tier_clearly(self, workspace):
        from universe_builder import run_batch

        class Args:
            model = "economy"

        import universe_builder.pipeline as pl
        real = pl.load

        def broken(*a, **k):
            cfg = real(*a, **k)
            cfg.tiers = {}
            return cfg

        pl.load = broken
        try:
            with __import__("pytest").raises(SystemExit):
                run_batch._model(Args())
        finally:
            pl.load = real


class TestAnOverrideIsResolvedNotPassedThrough:
    """`--model balanced` must reach the provider as a model id.

    `model_for_atom` short-circuited on the override and returned it raw, so a
    tier name reached `llm_call_mapping["balanced"]` and raised KeyError — while
    the dry run, which *does* resolve, displayed `gpt-5.6-luna`. Display and
    execution disagreeing is worse than either being wrong alone: the dry run is
    the thing you check before spending money, so it vouched for a run that could
    not work.

    Found by running it, not by the 789 tests that passed.
    """

    @pytest.mark.parametrize("tier", pipeline.TIERS)
    def test_a_tier_override_resolves(self, tier, cfg):
        from universe_builder.execute_prompt import model_for_atom

        resolved = model_for_atom("history", "history", tier)

        assert resolved == cfg.tiers["openai"][tier]
        assert resolved != tier, "the tier name was passed straight through"

    def test_an_exact_model_override_is_untouched(self):
        from universe_builder.execute_prompt import model_for_atom

        assert model_for_atom("history", "history", "gpt-4o-mini") == "gpt-4o-mini"

    def test_no_override_still_uses_the_configured_tier(self, cfg):
        from universe_builder.execute_prompt import model_for_atom

        assert model_for_atom("history", "history") == cfg.tiers["openai"]["frontier"]

    def test_every_override_is_callable(self, cfg):
        """The failure was a KeyError in llm_call_mapping; check the endpoint."""
        from universe_builder.config import llm_call_mapping
        from universe_builder.execute_prompt import model_for_atom

        for tier in pipeline.TIERS:
            model = model_for_atom("create", "city", tier)
            assert model in llm_call_mapping, f"--model {tier} resolves to an uncallable {model}"

    def test_the_dry_run_and_the_run_agree(self, workspace, capsys):
        """The two paths resolve the same override the same way."""
        from universe_builder import cli
        from universe_builder.execute_prompt import model_for_atom

        cli.main(["run", "--phase", "history", "--model", "balanced", "--dry-run"])
        shown = capsys.readouterr().out

        assert model_for_atom("history", "history", "balanced") in shown
