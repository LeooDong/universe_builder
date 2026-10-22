"""Retiring an older model's atoms so a better one regenerates them.

Nothing here calls a model; `apply` runs only against the `workspace` fixture's
temporary data dir.
"""

from __future__ import annotations

import json

from universe_builder import retire, settings, store, topics

OLD = "gpt-4o-mini"
NEW = "gpt-5-nano"


def _atom(atom_type, subject, model=OLD, payload=None):
    path = store.path_for(atom_type, None, subject, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {"name": {"common": subject}}),
                    encoding="utf-8")
    return path


def _history(name, **lists):
    payload = {"people": [], "discoveries": [], "events": [], "cities": [],
               "artifacts": []}
    payload.update(lists)
    path = store.dir_for("history") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestSelection:
    def test_only_the_named_model_is_planned(self, workspace):
        old = _atom("city", "Memphis", model=OLD)
        _atom("city", "Thebes", model=NEW)

        assert retire.plan(OLD).paths == [old]

    def test_matching_is_a_substring_of_the_model_segment(self, workspace):
        a = _atom("city", "Memphis", model="gpt-4o-mini")
        b = _atom("city", "Thebes", model="gpt-4-turbo")
        _atom("city", "Giza", model="gpt-5-nano")

        assert sorted(retire.plan("gpt-4").paths) == sorted([a, b])
        assert retire.plan("gpt-4o").paths == [a]

    def test_a_topic_slug_containing_the_text_does_not_match(self, workspace):
        """The model segment is parsed by `AtomName`, never by scanning the
        whole filename — a city called after a model must not be retired."""
        _atom("city", "gpt-4o-mini", model=NEW)

        assert retire.plan("gpt-4o").paths == []

    def test_an_empty_model_matches_nothing(self, workspace):
        _atom("city", "Memphis", model=OLD)

        assert retire.plan("  ").is_empty()

    def test_the_plan_groups_by_atom_type(self, workspace):
        _atom("city", "Memphis")
        _atom("person", "Imhotep")
        _atom("person", "Narmer")

        plan = retire.plan(OLD)

        assert {k: len(v) for k, v in plan.by_type.items()} == {"city": 1, "person": 2}

    def test_an_atom_filter_restricts_the_plan(self, workspace):
        """Retiring by model alone is too blunt where a later pass enriched the
        atom.

        A stored city carries its expanded periods, so retiring and
        regenerating it throws that work away and it is paid for twice.
        The expanded periods can cost more than the cities did to generate. The
        alternative was hand-moving files into `_retired/`, which is exactly
        the path around `store.py` this project refuses.
        """
        _atom("city", "Memphis")
        _atom("person", "Imhotep")
        _atom("person", "Narmer")

        plan = retire.plan(OLD, atoms=["person"])

        assert {k: len(v) for k, v in plan.by_type.items()} == {"person": 2}

    def test_an_empty_atom_filter_means_every_type(self, workspace):
        """`None` and `[]` must not be read as "retire nothing" — the CLI
        passes `None` when `--atom` is absent, and a filter that silently
        matched nothing would make the command a no-op without saying so."""
        _atom("city", "Memphis")
        _atom("person", "Imhotep")

        for empty in (None, []):
            plan = retire.plan(OLD, atoms=empty)

            assert {k: len(v) for k, v in plan.by_type.items()} == {"city": 1, "person": 1}

    def test_an_unknown_atom_type_plans_nothing_rather_than_everything(self, workspace):
        """A typo must not be read as "no filter"."""
        _atom("city", "Memphis")

        assert retire.plan(OLD, atoms=["citty"]).is_empty()


class TestTheIndexAtomIsNeverRetired:
    """Retiring a `history` atom does not schedule a regeneration — it removes
    topics from the work list. Egyptian History's gpt-4-turbo atom uniquely
    lists many topics; retiring it would orphan those atoms: still on disk,
    listed by nothing, absent from coverage and from the index."""

    def test_a_matching_index_atom_is_refused_not_planned(self, workspace):
        history = _history("history_h1111111111111111_egyptian_history_gpt-4o-mini.json",
                           cities=["Memphis"])
        city = _atom("city", "Memphis")

        plan = retire.plan(OLD)

        assert plan.paths == [city]
        assert plan.refused == [history]

    def test_applying_leaves_the_index_atom_in_place(self, workspace):
        history = _history("history_h1111111111111111_egyptian_history_gpt-4o-mini.json",
                           cities=["Memphis"])
        _atom("city", "Memphis")

        retire.apply(retire.plan(OLD), dry_run=False)

        assert history.exists()

    def test_the_topics_it_lists_survive(self, workspace):
        """The property that matters: after retiring, the work list still knows
        about the topic — as a gap to fill, not as something forgotten."""
        _history("history_h1111111111111111_egyptian_history_gpt-4o-mini.json",
                 cities=["Memphis"])
        _atom("city", "Memphis")

        retire.apply(retire.plan(OLD), dry_run=False)

        assert [t.subject for t in topics.from_history("city")] == ["Memphis"]
        assert [t.subject for t in topics.needing_create("city")] == ["Memphis"]


class TestApply:
    def test_a_dry_run_moves_nothing(self, workspace):
        path = _atom("city", "Memphis")

        outcome = retire.apply(retire.plan(OLD), dry_run=True)

        assert outcome["dry_run"] is True and outcome["retired"] == []
        assert path.exists()

    def test_the_atom_moves_rather_than_being_deleted(self, workspace):
        path = _atom("city", "Memphis")

        retire.apply(retire.plan(OLD), dry_run=False)

        assert not path.exists()
        assert (settings.retired_dir() / path.name).exists()

    def test_a_retired_atom_is_not_counted_as_an_atom(self, workspace):
        path = _atom("city", "Memphis")

        retire.apply(retire.plan(OLD), dry_run=False)

        assert path.name not in [p.name for p in store.iter_atoms()]

    def test_the_topic_becomes_work_again(self, workspace):
        """The whole point: a retired atom leaves a gap the create pass fills."""
        _history("history_h1111111111111111_egyptian_history_m.json", cities=["Memphis"])
        _atom("city", "Memphis")
        assert topics.needing_create("city") == []

        retire.apply(retire.plan(OLD), dry_run=False)

        assert [t.subject for t in topics.needing_create("city")] == ["Memphis"]

    def test_retiring_twice_does_not_raise(self, workspace):
        _atom("city", "Memphis")
        plan = retire.plan(OLD)
        retire.apply(plan, dry_run=False)

        assert retire.apply(plan, dry_run=False)["count"] == 0

    def test_a_name_collision_in_the_retired_dir_is_suffixed(self, workspace):
        path = _atom("city", "Memphis")
        settings.retired_dir().mkdir(parents=True, exist_ok=True)
        (settings.retired_dir() / path.name).write_text("{}", encoding="utf-8")

        retire.apply(retire.plan(OLD), dry_run=False)

        assert len(list(settings.retired_dir().glob("city_*"))) == 2


class TestTheReport:
    def test_models_on_disk_counts_every_version(self, workspace):
        _atom("city", "Memphis", model=OLD)
        _atom("city", "Thebes", model=NEW)
        _atom("person", "Imhotep", model=NEW)

        assert retire.models_on_disk() == {OLD: 1, NEW: 2}
