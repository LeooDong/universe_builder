"""`universe_builder status --coverage` — is a civilisation actually finished?

`status` counts atoms on disk, which says nothing about whether anything is
*missing*. With one civilisation you can eyeball it; with several you cannot, and
the create pass gives no signal either way — a topic that was never generated
looks exactly like one that does not exist.

The lookup this rests on is only possible because the corpus is content-addressed:
the id is a function of the subject alone, so a glob on the id finds a topic's
atom whichever model wrote it. Before the migration there was no way to ask the
question without already knowing the answer.
"""

from __future__ import annotations

import json


from universe_builder import cli, store


def _history(name="history_h1111111111111111_egyptian_history_m.json", **lists):
    payload = {"people": [], "discoveries": [], "events": [], "cities": [], "artifacts": []}
    payload.update(lists)
    path = store.dir_for("history") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _atom(subject, model="gpt-4o-mini", body=None):
    path = store.path_for("city", None, subject, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body or json.dumps({"name": {"common": subject}}), encoding="utf-8")
    return path


class TestTheLookupIsModelIndependent:
    def test_an_atom_is_found_whatever_model_wrote_it(self, workspace):
        _atom("Memphis", model="gpt-5-nano")

        assert store.atoms_for("city", "Memphis")
        assert not store.atoms_for("city", "Thebes")

    def test_sentinels_are_not_counted_as_atoms(self, workspace):
        store.write_error(store.path_for("city", None, "Memphis", "gpt-4o-mini"), "broken")

        assert store.atoms_for("city", "Memphis") == []
        assert len(store.sentinels_for("city", "Memphis")) == 1


class TestCoverageCounts:
    def test_a_complete_civilisation_reports_complete(self, workspace, capsys):
        _history(cities=["Memphis", "Thebes"])
        _atom("Memphis")
        _atom("Thebes")

        assert cli.main(["status", "--coverage"]) == 0
        out = capsys.readouterr().out

        assert "2 / 2" in out and "complete" in out

    def test_a_missing_topic_is_counted_and_the_exit_is_non_zero(self, workspace, capsys):
        _history(cities=["Memphis", "Thebes"])
        _atom("Memphis")

        assert cli.main(["status", "--coverage"]) == 1, "a script must be able to gate on this"
        assert "1 / 2" in capsys.readouterr().out

    def test_gaps_names_the_missing_topics(self, workspace, capsys):
        _history(cities=["Memphis", "Thebes"])
        _atom("Memphis")

        cli.main(["status", "--coverage", "--gaps"])

        assert "- Thebes" in capsys.readouterr().out

    def test_failed_is_distinguished_from_never_attempted(self, workspace, capsys):
        """Three states, not two.

        Collapsing "attempted and failed" into "missing" is how work goes
        missing: one needs a fix pass, the other needs a create pass, and they
        are not interchangeable.
        """
        _history(cities=["Memphis", "Thebes", "Luxor"])
        _atom("Memphis")
        store.write_error(store.path_for("city", None, "Thebes", "gpt-4o-mini"), "broken")

        cli.main(["status", "--coverage"])
        out = capsys.readouterr().out

        assert "1 awaiting fix" in out, out
        assert "1 never generated" in out, out

    def test_an_empty_sentinel_does_not_count_as_coverage(self, workspace, capsys):
        """A file is not an atom.

        `{"error": "No data found"}` validates as `EmptyModel` and sits at the
        atom's own path; counting it would report a civilisation complete when
        part of it is a placeholder.
        """
        _history(cities=["Memphis"])
        _atom("Memphis", body=json.dumps({"error": "No data found"}))

        assert cli.main(["status", "--coverage"]) == 1
        assert "0 / 1" in capsys.readouterr().out

    def test_civilisations_are_reported_separately(self, workspace, capsys):
        _history("history_h1111111111111111_egyptian_history_m.json", cities=["Memphis"])
        _history("history_h2222222222222222_greek_history_m.json", cities=["Athens"])
        _atom("Memphis")

        cli.main(["status", "--coverage"])
        out = capsys.readouterr().out

        assert "egyptian history" in out and "greek history" in out
        assert out.index("egyptian history") < out.index("greek history"), "sorted"

    def test_no_history_atoms_says_what_to_do(self, workspace, capsys):
        assert cli.main(["status", "--coverage"]) == 1
        assert "universe_builder run --phase index" in capsys.readouterr().out

    def test_plain_status_is_unchanged(self, workspace, capsys):
        """`--coverage` is opt-in; the default output must not move."""
        _history(cities=["Memphis"])
        _atom("Memphis")

        assert cli.main(["status"]) == 0
        out = capsys.readouterr().out

        assert "awaiting fix" in out
        assert "/" not in out.split("data directory:")[1].split("\n")[1]

    def test_append_sentinels_are_not_counted_as_awaiting_fix(self, workspace, capsys):
        """An append failure writes `append_city_…_ERROR` into the city folder, but
        the fix pass filters on the base atom name (`city`), so `--phase fix` never
        visits it. status must not report it as `awaiting fix` — that promises a
        repair the fix pass cannot make — but should surface it separately so the
        failure is not invisible."""
        _atom("Memphis")  # a real city atom, not a sentinel
        store.write_error(
            store.path_for("append_city", None, "Memphis", "gpt-4o-mini"), "broken"
        )

        assert cli.main(["status"]) == 0
        out = capsys.readouterr().out

        assert "city" in out and "0 awaiting fix" in out
        assert "append/expand _ERROR sentinel(s)" in out
