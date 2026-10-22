"""`universe_builder run` for the phases that were advertised but not implemented.

`PHASES` offered six choices and `run_phase` implemented four. `--phase create`
and `--phase history` reached `"Phase %r is not runnable on its own yet"` and
exited 2 - and the dry run happily planned them first, reporting
"(topics from HISTORY_list.txt)" and a work count of 1 for what is really a
whole list of subjects. Nothing here makes a call: the dry-run path is pure filesystem, and
the batch path is exercised against a fake provider.
"""

from __future__ import annotations

import json
from unittest.mock import patch


from universe_builder import cli, settings, store


def _history(name="history_h1111111111111111_egyptian_history_m.json", **lists):
    payload = {"people": [], "discoveries": [], "events": [], "cities": [], "artifacts": []}
    payload.update(lists)
    path = store.dir_for("history") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestTheDryRunReportsRealWork:
    def test_create_lists_subjects_not_a_placeholder(self, workspace, capsys):
        _history(cities=["Memphis", "Thebes", "Alexandria"])

        assert cli.main(["run", "--phase", "create", "--atom", "city", "--dry-run"]) == 0
        out = capsys.readouterr().out

        assert "work items 3" in out, out
        assert "Memphis" in out and "Thebes" in out
        assert "topics from" not in out, (
            "the placeholder is back; a dry run that under-reports the work by two "
            "orders of magnitude is worse than no dry run"
        )

    def test_expand_dry_run_only_counts_atoms_with_an_expand_spec(self, workspace, capsys, city_payload):
        """B2: only `city` declares an expand spec, so a stored `person` must not
        be reported as expand work. The dry run used to list every file of every
        atom type, over-reporting expand by orders of magnitude."""
        for atom, subject, payload in (("city", "Memphis", city_payload), ("person", "Imhotep", {})):
            p = store.path_for(atom, None, subject, "m")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload), encoding="utf-8")

        assert cli.main(["run", "--phase", "expand", "--dry-run"]) == 0
        out = capsys.readouterr().out

        assert "work items 1" in out, out
        # the person atom must not appear as a work item (its subject would show in
        # the listed filename); the atom-type header naming "person" is fine.
        assert "imhotep" not in out.lower(), "a person has no expand spec and is not expand work"

    def test_expand_dry_run_counts_only_cities_with_work_left(
            self, workspace, capsys, city_payload, expanded_city_payload):
        """A fully-expanded city is skipped at run time, and batch submit sends
        nothing for it; the dry run listed it anyway, so on a corpus that was
        mostly expanded it reported every city as work. It now asks the same
        question the run does (`prepare_expansion`)."""
        from tests.conftest import make_population

        partial = dict(expanded_city_payload,
                       cityPeriodsOfHistory=[make_population(), "Late Period"])
        for subject, payload in (("Memphis", city_payload), ("Thebes", expanded_city_payload),
                                 ("Tanis", partial)):
            p = store.path_for("city", None, subject, "m")
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(payload), encoding="utf-8")

        assert cli.main(["run", "--phase", "expand", "--dry-run"]) == 0
        out = capsys.readouterr().out

        assert "work items 2" in out, out
        assert "memphis" in out and "tanis" in out
        assert "thebes" not in out, "a fully-expanded city has no expand work"

    def test_history_counts_one_per_civilisation(self, workspace, capsys, tmp_path, monkeypatch):
        """Three civilisations must read as three items, not one.

        The bundled `HISTORY_list.txt` has a single line, so against the real list
        this assertion would pass just as well with the old
        "(topics from HISTORY_list.txt)" placeholder - which also reported one.
        A substitute list with three lines is what makes it able to fail.
        """
        from universe_builder import settings

        seed = tmp_path / "HISTORY_list.txt"
        seed.write_text("Egyptian History\nGreek History\nRoman History\n", encoding="utf-8")
        monkeypatch.setattr(settings, "HISTORY_LIST", seed)

        assert cli.main(["run", "--phase", "history", "--dry-run"]) == 0
        out = capsys.readouterr().out

        assert "work items 3" in out, out
        assert "Roman History" in out

    def test_history_skips_civilisations_that_already_have_an_atom(
            self, workspace, capsys, tmp_path, monkeypatch):
        """Adding a second civilisation must not re-buy the first: the history
        atom is the most expensive call in the system, and `persist` would
        overwrite the stored one with the rerun's output."""
        from universe_builder import settings, store

        seed = tmp_path / "HISTORY_list.txt"
        seed.write_text("Egyptian History\nGreek History\n", encoding="utf-8")
        monkeypatch.setattr(settings, "HISTORY_LIST", seed)
        existing = store.path_for("history", None, "Egyptian History", "m")
        existing.parent.mkdir(parents=True, exist_ok=True)
        existing.write_text('{"cities": ["Memphis"]}', encoding="utf-8")

        assert cli.main(["run", "--phase", "history", "--dry-run"]) == 0
        out = capsys.readouterr().out

        assert "work items 1" in out, out
        assert "Greek History" in out
        assert "Egyptian History" not in out.split("work items", 1)[1]

    def test_create_skips_the_history_atom_type(self, workspace, capsys):
        """`history` is produced by its own phase, never by create."""
        _history(cities=["Memphis"])

        cli.main(["run", "--phase", "create", "--dry-run"])
        out = capsys.readouterr().out

        # Check the atom-type *column* of each work item, not the line. Two
        # earlier versions of this assertion were too loose: a bare substring
        # over the whole output matched the per-atom model table, and one over
        # the work list matched the context label "(egyptian history)".
        work = out.split("work items", 1)[1]
        types = {line.split()[0] for line in work.splitlines() if line.startswith("    ")}
        assert "history" not in types, types

        # The model table must not advertise a model for work that will not
        # happen either.
        header = out.split("work items", 1)[0]
        assert "\n    history " not in header, header

    def test_limit_applies_per_atom_type(self, workspace, capsys):
        _history(cities=["A", "B", "C"], people=["D", "E", "F"])

        cli.main(["run", "--phase", "create", "--limit", "2", "--dry-run"])
        out = capsys.readouterr().out

        assert "work items 4" in out, out


class TestBatchSubmission:
    def test_create_submits_one_batch_per_context(self, workspace, capsys):
        """The context is in the cached prefix, so batches cannot span two.

        A batch mixing civilisations carries two prefixes, and neither request
        group reads the other's cache entry.
        """
        _history("history_h1111111111111111_egyptian_history_m.json", cities=["Memphis"])
        _history("history_h2222222222222222_greek_history_m.json", cities=["Athens"])

        seen = []

        def fake_submit(atom, subjects, model, *, context=None, **kw):
            seen.append((atom, tuple(subjects), context))
            return [store.dir_for("history") / f"handle-{context}.json"]

        with patch("universe_builder.providers.batch.submit", fake_submit):
            assert cli.main(
                ["run", "--phase", "create", "--atom", "city", "--batch", "--model", "gpt-5-nano"]
            ) == 0

        assert seen == [
            ("city", ("Memphis",), "egyptian history"),
            ("city", ("Athens",), "greek history"),
        ], seen
        assert "2 batch(es) submitted" in capsys.readouterr().out

    def test_the_configured_max_tokens_reaches_the_batch(self, workspace, tmp_path):
        """`defaults.max_tokens` was config nothing read.

        `PipelineConfig.max_tokens` existed with no caller anywhere, so every
        batch submitted at `DEFAULT_MAX_TOKENS` (16,384) whatever the config
        said. It cost truncated cities, at exactly 16,384 output tokens,
        days after the value was raised to 32,768
        *specifically to stop that*. The same class of bug had already been
        fixed once for `retry` — see `provider_run.configure` in `cmd_run`.

        Asserting a number the default cannot produce is the point: a test
        pinning 16,384 would have passed throughout the bug's life.
        """
        cfg = tmp_path / "pipeline.yaml"
        cfg.write_text(
            "defaults:\n  provider: openai\n  model: economy\n  max_tokens: 31337\n"
            "tiers:\n  openai:\n    economy: gpt-5-nano\n"
            "phases:\n  create: economy\n"
        )
        _history(cities=["Memphis"])
        seen = {}

        def fake_submit(atom, subjects, model, **kw):
            seen.update(kw)
            return [store.dir_for("history") / "h.json"]

        with patch("universe_builder.providers.batch.submit", fake_submit):
            assert cli.main(["run", "--phase", "create", "--atom", "city", "--batch",
                             "--model", "gpt-5-nano", "--config", str(cfg)]) == 0

        assert seen.get("max_tokens") == 31337, seen

    def test_it_tells_you_how_to_collect(self, workspace, capsys):
        """A submitted batch is useless if the next command is not obvious."""
        _history(cities=["Memphis"])

        with patch("universe_builder.providers.batch.submit", lambda *a, **k: [store.dir_for("history") / "h.json"]):
            cli.main(["run", "--phase", "create", "--atom", "city", "--batch", "--model", "gpt-5-nano"])

        assert "run_batch collect --all" in capsys.readouterr().out

    def test_append_has_no_batch_shape_and_is_refused(self, workspace, caplog):
        """`append` produces a *separate* file that `merge_helper` later pairs
        with the original — not one-request-in, one-file-out — so it is refused
        rather than silently run synchronously.

        `fix`, `verify` and now `expand` are supported: `fix`/`verify` route a
        result back to its file via `BatchItem.target`, and `expand` fans out
        one request per period and splices the replies back per city on collect
        (see `TestExpandBatchSubmission`)."""
        assert cli.main(["run", "--phase", "append", "--batch", "--model", "gpt-5-nano"]) == 2
        assert "no batch shape yet" in caplog.text


class TestPayloadBatchSubmission:
    """`fix` and `verify` batched: each request carries one file and its target."""

    MODEL = "gpt-5-nano"

    def _atom(self, name, body='{"name": {"common": "Memphis"}}'):
        path = store.dir_for("city") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
        return path

    def _capture(self):
        seen = []

        def fake_submit(atom, items, model, **kw):
            seen.append((atom, list(items), kw))
            return [store.dir_for("city") / "handle.json"]

        return seen, fake_submit

    def test_verify_carries_each_file_and_its_target(self, workspace, capsys):
        atom = self._atom(f"city_h1111111111111111_memphis_{self.MODEL}.json")
        seen, fake = self._capture()

        with patch("universe_builder.providers.batch.submit", fake):
            assert cli.main(["run", "--phase", "verify", "--atom", "city",
                             "--batch", "--model", self.MODEL]) == 0

        (_, items, kw) = seen[0]
        assert kw["operation"] == "verify"
        assert len(items) == 1
        assert items[0].payload == atom.read_text()
        assert items[0].target == str(atom.relative_to(settings.data_dir())), (
            "without the target, collect writes to a path rebuilt from the subject "
            "rather than the file the payload came from"
        )

    def test_fix_reads_the_sentinel_but_targets_the_atom(self, workspace):
        """The payload comes from the sentinel; the answer goes to the atom.

        This test previously asserted the opposite — that the target *was* the
        `_ERROR` path — and so pinned a data-loss bug rather than catching it.
        `collect` writes the result to the target and then calls
        `store.clear_error` on it, which for a sentinel target resolves back to
        the same file and deletes it: the repair and the original failing payload
        both vanish, and the report says "1 written". The synchronous path has
        always had this ordering right, in `store.commit_fix`.
        """
        self._atom(f"city_h1111111111111111_memphis_{self.MODEL}.json")
        atom = store.dir_for("city") / f"city_h2222222222222222_thebes_{self.MODEL}.json"
        store.write_error(atom, '{"broken":')
        seen, fake = self._capture()

        with patch("universe_builder.providers.batch.submit", fake):
            assert cli.main(["run", "--phase", "fix", "--atom", "city",
                             "--batch", "--model", self.MODEL]) == 0

        (_, items, kw) = seen[0]
        assert kw["operation"] == "fix"
        assert len(items) == 1, "fix must take only the sentinels"
        assert items[0].payload == store.read(store.error_path(atom)), (
            "the payload must be the failing text from the sentinel"
        )
        assert not items[0].target.endswith(store.ERROR_SUFFIX), (
            "targeting the sentinel makes collect write the repair there and then "
            "delete it"
        )
        assert items[0].target == str(atom.relative_to(settings.data_dir()))

    def test_verify_skips_empty_sentinels(self, workspace):
        """Nothing to verify in `{"error": "No data found"}`."""
        self._atom(f"city_h1111111111111111_memphis_{self.MODEL}.json")
        self._atom(f"city_h2222222222222222_thebes_{self.MODEL}.json",
                   '{\n    "error": "No data found"\n}')
        seen, fake = self._capture()

        with patch("universe_builder.providers.batch.submit", fake):
            cli.main(["run", "--phase", "verify", "--atom", "city",
                      "--batch", "--model", self.MODEL])

        assert len(seen[0][1]) == 1

    def test_nothing_to_do_is_reported(self, workspace, caplog):
        assert cli.main(["run", "--phase", "fix", "--atom", "city",
                         "--batch", "--model", self.MODEL]) == 1
        assert "Nothing submitted" in caplog.text

    def test_nothing_to_submit_is_a_failure_not_a_silent_success(self, workspace, caplog):
        assert cli.main(
            ["run", "--phase", "create", "--atom", "city", "--batch", "--model", "gpt-5-nano"]
        ) == 1
        assert "Nothing submitted" in caplog.text


class TestPayloadBatchesRespectTheStoredShape:
    """A batch must ask for the shape the file is actually in.

    A city that has been through expand is stored as `CityExpanded`. Asking the
    model for the seed `City` shape is B7 by another route: the reply validates,
    `collect` overwrites the file, and `cityPeriodsOfHistory` collapses from
    `Population` objects back to a list of period names. The synchronous path
    guards this with `atoms.resolve_stored_class`; the batch path now groups its
    submissions by the same judgement — which also keeps one JSON schema per
    cached prefix.
    """

    MODEL = "gpt-4o-mini"

    def test_an_expanded_city_is_submitted_as_cityexpanded(self, workspace, expanded_city_payload):
        import json

        path = store.dir_for("city") / f"city_h1111111111111111_memphis_{self.MODEL}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(expanded_city_payload), encoding="utf-8")

        seen = []

        def fake_submit(atom, items, model, **kw):
            seen.append(kw.get("schema_model"))
            return [store.dir_for("city") / "h.json"]

        with patch("universe_builder.providers.batch.submit", fake_submit):
            cli.main(["run", "--phase", "verify", "--atom", "city",
                      "--batch", "--model", self.MODEL])

        assert seen == ["CityExpanded"], (
            f"submitted as {seen}; asking for the seed City shape collapses the "
            "expansion when the answer is written back"
        )

    def test_two_shapes_become_two_batches(self, workspace, expanded_city_payload):
        """One schema per batch, because the schema is in the cached prefix."""
        import json

        folder = store.dir_for("city")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"city_h1111111111111111_memphis_{self.MODEL}.json").write_text(
            json.dumps(expanded_city_payload), encoding="utf-8")
        seed = dict(expanded_city_payload, cityPeriodsOfHistory=["Old Kingdom"])
        (folder / f"city_h2222222222222222_thebes_{self.MODEL}.json").write_text(
            json.dumps(seed), encoding="utf-8")

        seen = []

        def fake_submit(atom, items, model, **kw):
            seen.append((kw.get("schema_model"), len(items)))
            return [folder / f"h-{kw.get('schema_model')}.json"]

        with patch("universe_builder.providers.batch.submit", fake_submit):
            cli.main(["run", "--phase", "verify", "--atom", "city",
                      "--batch", "--model", self.MODEL])

        assert sorted(seen) == [("City", 1), ("CityExpanded", 1)], seen


class TestHintsAndWarningsUnderAWrapper:
    """Cosmetic findings of an end-to-end test."""

    def test_the_collect_hint_names_the_running_prog(self, workspace, capsys):
        _history(cities=["Memphis"])

        with patch("universe_builder.providers.batch.submit", lambda *a, **k: [store.dir_for("history") / "h.json"]):
            cli.main(["run", "--phase", "create", "--atom", "city", "--batch", "--model", "gpt-5-nano"],
                     prog="mypack")

        out = capsys.readouterr().out
        assert "python -m mypack.run_batch collect --all" in out
        assert "universe_builder.run_batch" not in out

    def test_an_alias_warning_is_given_once_per_command_not_once_per_atom_type(
            self, workspace, tmp_path, monkeypatch, caplog):
        import logging

        from universe_builder import topics

        aliases = tmp_path / "aliases.txt"
        aliases.write_text("Memphis (Mit Rahina) = Memphis\nMemphis (Tennessee) = Memphis\n", encoding="utf-8")
        monkeypatch.setattr(settings, "ALIASES_LIST", aliases)
        _history(cities=["Memphis"], people=["Narmer"], events=["Unification"], artifacts=["Palette"],
                 discoveries=["Papyrus"])

        with caplog.at_level(logging.WARNING):
            assert cli.main(["run", "--phase", "create", "--dry-run"]) == 0
        assert caplog.text.count("two groups of the same scope") == 1

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            topics.load_aliases(aliases)
        assert "two groups of the same scope" in caplog.text, "outside a command, every load still warns"
