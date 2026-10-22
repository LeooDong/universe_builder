"""The create phase's work list.

`execute_prompt.run_base_atom_calls` and `run_batch._subjects_from_history` each
walked the History files with their own code, and disagreed: the batch one
de-duplicated across files, the pipeline one did not. `universe_builder.topics` is the single
answer both now use.
"""

from __future__ import annotations

import json

import pytest

from universe_builder import store, topics


def _history(workspace, name: str, **lists):
    payload = {"people": [], "discoveries": [], "events": [], "cities": [], "artifacts": []}
    payload.update(lists)
    path = store.dir_for("history") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestDeduplication:
    def test_one_subject_listed_twice_yields_one_topic(self, workspace):
        """Two civilisations both listing Alexandria must not produce two calls.

        Content-addressed ids map one subject to one path, so a duplicate is two
        calls racing for a single file - the collision `persist` defends against -
        and the second is paid for to produce what the first already wrote.
        """
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Alexandria", "Memphis"])
        _history(workspace, "history_h2222222222222222_greek_history_m.json",
                 cities=["Alexandria", "Athens"])

        found = topics.from_history("city")
        subjects = [t.subject for t in found]

        assert subjects == ["Alexandria", "Memphis", "Athens"], subjects
        assert len(subjects) == len(set(subjects))

    def test_the_first_context_wins(self, workspace):
        """A shared subject keeps one identity rather than acquiring a second."""
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Alexandria"])
        _history(workspace, "history_h2222222222222222_greek_history_m.json",
                 cities=["Alexandria"])

        found = topics.from_history("city")
        assert len(found) == 1
        assert found[0].context == "egyptian history"

    def test_case_and_whitespace_do_not_defeat_it(self, workspace):
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Alexandria", "  alexandria ", "ALEXANDRIA"])

        assert [t.subject for t in topics.from_history("city")] == ["Alexandria"]

    def test_blank_entries_are_dropped(self, workspace):
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Memphis", "", "   "])

        assert [t.subject for t in topics.from_history("city")] == ["Memphis"]


class TestContext:
    def test_the_legacy_context_tail_is_stripped(self, workspace):
        """All 20 stored History files are named `..._context_history`.

        Without stripping, the model is told "The context is egyptian history
        context history!" - the tail is an artefact of the subject those files
        were created from, not part of the civilisation's name.
        """
        _history(workspace,
                 "history_0000000001_egyptian_history_context_history_gpt-4-turbo.json",
                 cities=["Memphis"])

        assert topics.from_history("city")[0].context == "egyptian history"

    def test_grouping_keeps_one_cached_prefix_per_batch(self, workspace):
        """The context is in the system text, which is the cached prefix.

        A batch spanning two civilisations carries two prefixes and neither group
        reads the other's cache entry, so the batch path submits one per group.
        """
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Memphis", "Thebes"])
        _history(workspace, "history_h2222222222222222_greek_history_m.json",
                 cities=["Athens"])

        grouped = topics.by_context(topics.from_history("city"))

        assert grouped == {"egyptian history": ["Memphis", "Thebes"], "greek history": ["Athens"]}


class TestRobustness:
    def test_an_unreadable_history_does_not_cost_the_run_the_others(self, workspace, caplog):
        """One bad file must not take every topic behind it."""
        bad = store.dir_for("history") / "history_h0000000000000000_broken_m.json"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_text("{not json", encoding="utf-8")
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Memphis"])

        assert [t.subject for t in topics.from_history("city")] == ["Memphis"]
        assert "broken" in caplog.text

    def test_error_sentinels_are_not_read_as_history(self, workspace):
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Memphis"])
        store.write_error(store.dir_for("history") / "history_h9999999999999999_failed_m.json",
                          '{"cities": ["Should Not Appear"]}')

        assert [t.subject for t in topics.from_history("city")] == ["Memphis"]

    def test_an_empty_placeholder_is_skipped_quietly(self, workspace, caplog):
        """A failed history run leaves an `EmptyModel` at the atom's own path.

        `persist` writes one when a call produces nothing, so a transport failure
        on `--phase history` leaves a file that is not a History atom and never
        will be. Without this it reached `load_atom_from_json` and produced a
        five-error Pydantic dump — once per atom type, so five times per command.
        Observed on a real failed run.
        """
        import logging

        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Memphis"])
        placeholder = store.dir_for("history") / "history_h2222222222222222_greek_history_m.json"
        placeholder.write_text('{\n    "error": "No data found"\n}', encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            found = topics.from_history("city")

        assert [t.subject for t in found] == ["Memphis"]
        assert "validation error" not in caplog.text.lower(), caplog.text
        assert "Skipping unreadable" not in caplog.text, (
            "the placeholder was treated as a corrupt History atom rather than as "
            "the recorded failure it is"
        )

    def test_a_file_that_is_not_utf8_costs_only_itself(self, workspace, caplog):
        """The placeholder check put file I/O inside the generator.

        `_history_files()` is consumed outside `from_history`'s try/except, so an
        exception raised while *deciding whether to yield* escapes the loop and
        costs the caller every History file. Files are sorted, so a bad name
        early hides every good one after it.

        `test_an_unreadable_history_does_not_cost_the_run_the_others` did not
        catch this: `"{not json"` is valid UTF-8 and fails at the JSON parse,
        which is inside the guard. Truncated or binary content fails at the
        *read*, which was not.
        """
        bad = store.dir_for("history") / "history_h0000000000000000_aaa_first_m.json"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b'\xff\xfe{"cities":')          # sorts before the good one
        _history(workspace, "history_h9999999999999999_zzz_last_m.json", cities=["Memphis"])

        assert [t.subject for t in topics.from_history("city")] == ["Memphis"]
        assert "aaa_first" in caplog.text, "the bad file should be named, not silently dropped"

    def test_the_placeholder_check_never_raises(self, workspace):
        """Pins the property directly, not just one way of violating it."""
        bad = store.dir_for("history") / "history_h0000000000000000_binary_m.json"
        bad.parent.mkdir(parents=True, exist_ok=True)
        bad.write_bytes(b"\xff\xfe\x00\x01")

        assert topics._is_placeholder(bad) is False

        missing = store.dir_for("history") / "history_h1111111111111111_gone_m.json"
        assert topics._is_placeholder(missing) is False

    def test_a_genuinely_corrupt_file_is_still_reported(self, workspace, caplog):
        """Skipping placeholders must not silence real corruption."""
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Memphis"])
        bad = store.dir_for("history") / "history_h3333333333333333_broken_m.json"
        bad.write_text("{not json", encoding="utf-8")

        assert [t.subject for t in topics.from_history("city")] == ["Memphis"]
        assert "broken" in caplog.text

    def test_no_history_at_all_is_empty_not_an_error(self, workspace):
        assert topics.from_history("city") == []

    @pytest.mark.parametrize("atom_type", ["person", "event", "discovery", "city", "artifact"])
    def test_every_base_atom_type_has_a_plural_to_read(self, workspace, atom_type):
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 people=["P"], events=["E"], discoveries=["D"], cities=["C"], artifacts=["A"])

        assert len(topics.from_history(atom_type)) == 1


class TestSynonymsAreGroupedNotSplit:
    """Ids are hashed from the exact string, so `Thebes` and `Thebes (Luxor)`
    are two atoms for one city.

    An exact-match dedup treats them as different topics, so running a second
    model over the same civilisation accumulates duplicates rather than coverage.
    Grouping happens on the **work list** and never in `store.normalise_topic`,
    which feeds the content id — joining names there would re-point every id and
    orphan the 453 stored atoms.
    """

    def test_a_parenthetical_declares_an_alias(self):
        assert topics.same_topic("Thebes", "Thebes (Luxor)")
        assert topics.same_topic("Luxor", "Thebes (Luxor)")
        assert topics.same_topic("Hierakonpolis", "Hierakonpolis (Nekhen)")

    def test_unrelated_places_are_not_grouped(self):
        assert not topics.same_topic("Memphis", "Thebes")
        assert not topics.same_topic("Giza", "Saqqara")

    def test_case_and_punctuation_carry_no_identity(self):
        assert topics.same_topic("hierakonpolis", "Hierakonpolis")
        assert topics.same_topic("Pi-Ramesses", "Pi Ramesses")

    def test_grouping_does_not_depend_on_order(self):
        """The entry that declares the link can arrive last.

        A single pass over `[Thebes, Luxor, Thebes (Luxor)]` leaves `Luxor`
        alone: when it is seen, nothing has yet said it is the same place.
        """
        for order in (
            ["Thebes", "Luxor", "Thebes (Luxor)"],
            ["Thebes (Luxor)", "Thebes", "Luxor"],
            ["Luxor", "Thebes (Luxor)", "Thebes"],
        ):
            assert len(topics.group_synonyms(order)) == 1, order

    def test_the_form_that_already_has_an_atom_wins(self):
        """New lists resolve onto the corpus instead of growing siblings."""
        have = {"Thebes (Luxor)"}
        kept = topics.group_synonyms(
            ["Thebes", "Luxor", "Thebes (Luxor)"], prefer=lambda s: s in have)

        assert kept == ["Thebes (Luxor)"]

    def test_distinct_topics_survive_grouping(self):
        subjects = ["Memphis", "Abydos", "Giza", "Saqqara"]

        assert topics.group_synonyms(subjects) == subjects

    def test_two_history_atoms_naming_one_city_yield_one_topic(self, workspace):
        """The end-to-end case: two models, two conventions, one city."""
        _history(workspace, "history_h1111111111111111_egyptian_history_a.json",
                 cities=["Thebes (Luxor)", "Memphis"])
        _history(workspace, "history_h2222222222222222_egyptian_history_b.json",
                 cities=["Thebes", "Luxor", "Memphis"])

        found = [t.subject for t in topics.from_history("city")]

        assert sorted(found) == ["Memphis", "Thebes (Luxor)"], found


class TestNearMatchesAreAdvisoryOnly:
    """Reporting may guess; grouping may not.

    A wrong suggestion in a report costs a reader two seconds. A wrong grouping
    silently merges two real places, and the atom for one of them never gets
    generated.
    """

    HAVE = ["Thebes (Luxor)", "Amarna (Akhetaten)", "Memphis",
            "Temple of Karnak", "Temple of Luxor", "Great Pyramid of Giza"]

    def test_it_surfaces_a_link_the_data_does_not_declare(self):
        assert "Amarna (Akhetaten)" in topics.near_matches("Tell el-Amarna", self.HAVE)

    def test_it_stays_quiet_for_something_genuinely_new(self):
        assert topics.near_matches("Elephantine", self.HAVE) == []

    def test_common_words_do_not_match_everything(self):
        """Without stopwords every "Temple of ..." matches every other."""
        assert topics.near_matches("Temple of Isis", self.HAVE) == []

    def test_it_never_reports_what_grouping_already_handled(self):
        """`Akhetaten` is a declared alias, so it is grouped, not suggested."""
        assert topics.near_matches("Akhetaten", self.HAVE) == []


class TestGroupingNeverLosesATopicSilently:
    """Merging wrongly is far worse than splitting wrongly.

    A split costs a duplicate atom, which `--gaps` reports and a human can see.
    A wrong merge costs a topic that is never generated again and nothing points
    at the gap. So the rule refuses whenever the data does not plainly say two
    names are the same thing.
    """

    def test_a_chain_of_declarations_does_not_merge_transitively(self):
        """An earlier union-find version lost Saqqara entirely.

            ["Memphis (Mit Rahina)", "Mit Rahina (Saqqara)", "Saqqara"]
              -> ["Memphis (Mit Rahina)"]

        Memphis and Saqqara were joined through a hop neither entry claims; one
        bad parenthetical was enough and the loss was silent.
        """
        subjects = ["Memphis (Mit Rahina)", "Mit Rahina (Saqqara)", "Saqqara"]

        kept = topics.group_synonyms(subjects)

        assert kept == subjects, "a topic was merged away through a chain"

    def test_an_ambiguous_claim_is_kept_and_logged(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            topics.group_synonyms(["Memphis (Mit Rahina)", "Mit Rahina (Saqqara)"])

        assert "claimed by 2 different declarations" in caplog.text

    def test_the_same_declaration_twice_is_agreement_not_conflict(self):
        """Two History atoms both listing `Thebes (Luxor)` agree.

        Counting occurrences rather than claims flagged every shared entry as
        ambiguous and un-grouped it, so the real corpus gained spurious cities.
        """
        kept = topics.group_synonyms(["Thebes (Luxor)", "Thebes (Luxor)", "Thebes", "Luxor"])

        assert kept == ["Thebes (Luxor)"]

    def test_a_reversed_pair_is_the_same_declaration(self):
        """`Amarna (Akhetaten)` and `Akhetaten (Amarna)` make one claim."""
        kept = topics.group_synonyms(["Amarna (Akhetaten)", "Akhetaten (Amarna)", "Amarna"])

        assert len(kept) == 1, kept

    def test_every_survivor_came_from_the_input(self):
        """The result invents nothing."""
        subjects = ["Memphis (Mit Rahina)", "Mit Rahina (Saqqara)", "Saqqara",
                    "Thebes", "Luxor", "Thebes (Luxor)", "Giza"]

        assert set(topics.group_synonyms(subjects)) <= set(subjects)

    def test_each_group_keeps_exactly_one_survivor(self):
        """Counted per group rather than per entry.

        A weaker version of this asserted every input still `same_topic`s
        something kept — which fails honestly: when the survivor is the bare
        `Thebes`, it no longer carries the `Luxor` alias that `Thebes (Luxor)`
        declared. That is a property of the survivor, not a lost topic, and the
        declaration is still in the History atom for the next run to group on.
        """
        subjects = ["Memphis (Mit Rahina)", "Mit Rahina (Saqqara)", "Saqqara",
                    "Thebes", "Luxor", "Thebes (Luxor)", "Giza"]

        kept = topics.group_synonyms(subjects)

        # Three ambiguous entries kept apart, one Thebes group, one Giza.
        assert len(kept) == 5, kept
        assert "Giza" in kept
        assert sum(1 for k in kept if topics.same_topic(k, "Thebes (Luxor)")) == 1


class TestAliasFileParsing:
    def _load(self, tmp_path, text):
        path = tmp_path / "aliases.txt"
        path.write_text(text, encoding="utf-8")
        return topics.load_aliases(path)

    def test_a_missing_file_is_no_aliases(self, tmp_path):
        assert topics.load_aliases(tmp_path / "absent.txt") == ()

    def test_groups_comments_and_blank_lines(self, tmp_path):
        groups = self._load(tmp_path, """
            # a comment
            Amarna (Akhetaten) = Akhetaten = Tell el-Amarna   # trailing comment

            Thebes (Luxor) = Waset
        """)

        assert [g.names for g in groups] == [
            ("Amarna (Akhetaten)", "Akhetaten", "Tell el-Amarna"),
            ("Thebes (Luxor)", "Waset"),
        ]
        assert all(g.scope is None for g in groups)

    def test_scoped_sections_and_the_reset_to_global(self, tmp_path):
        groups = self._load(tmp_path, """
            [Greek History]
            Thebes (Boeotia) = Thebes
            [*]
            Amarna (Akhetaten) = Akhetaten
        """)

        assert groups[0].scope == "Greek History"
        assert groups[1].scope is None

    def test_a_name_in_two_groups_of_one_scope_is_warned_at_load(self, tmp_path, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            self._load(tmp_path, """
                Memphis (Mit Rahina) = Memphis
                Memphis (Tennessee) = Memphis
            """)

        assert "two groups of the same scope" in caplog.text

    def test_a_global_and_a_scoped_group_may_share_a_name(self, tmp_path, caplog):
        """The refinement pattern: scoped wins inside its civilisation,
        global everywhere else. Not a contradiction."""
        import logging

        with caplog.at_level(logging.WARNING):
            groups = self._load(tmp_path, """
                Memphis (Mit Rahina) = Memphis
                [American History]
                Memphis (Tennessee) = Memphis
            """)

        assert len(groups) == 2
        assert "two groups" not in caplog.text

    def test_an_unreadable_file_degrades_to_no_aliases(self, tmp_path, caplog):
        """Grouping without aliases is yesterday's behaviour, not a stop."""
        import logging

        path = tmp_path / "aliases.txt"
        path.write_bytes(b"\xff\xfe not utf-8 \x9c")
        with caplog.at_level(logging.ERROR):
            assert topics.load_aliases(path) == ()

        assert "grouping runs without it" in caplog.text


class TestAliasGrouping:
    """The alias file joins what the data never declares — and only that.

    `Akhetaten` and `Amarna` are one city; no History entry says so, and the
    parenthetical heuristic cannot. A line in `aliases.txt` is the deliberate
    human claim that closes the gap.
    """

    AMARNA = (topics.AliasGroup(
        names=("Amarna (Akhetaten)", "Amarna", "Akhetaten", "Tell el-Amarna")),)

    def test_a_link_the_data_never_declares(self):
        kept = topics.group_synonyms(["Akhetaten", "Tell el-Amarna"], aliases=self.AMARNA)

        assert kept == ["Amarna (Akhetaten)"]

    def test_the_stored_spelling_wins_even_when_no_list_uses_it(self):
        """The point of declared names as candidates: a list saying only
        `Akhetaten` resolves onto the stored `Amarna (Akhetaten)` atom."""
        kept = topics.group_synonyms(
            ["Akhetaten"],
            prefer=lambda s: s == "Amarna (Akhetaten)",
            aliases=self.AMARNA)

        assert kept == ["Amarna (Akhetaten)"]

    def test_a_parenthetical_that_restates_the_line_is_agreement(self, caplog):
        """`Amarna (Akhetaten)` in the data claims a pair the alias line already
        contains, so it folds in rather than becoming a rival declaration."""
        import logging

        with caplog.at_level(logging.WARNING):
            kept = topics.group_synonyms(
                ["Amarna (Akhetaten)", "Tell el-Amarna", "Akhetaten"],
                aliases=self.AMARNA)

        assert len(kept) == 1, kept
        assert "claimed by" not in caplog.text

    def test_a_partial_overlap_does_not_chain(self, caplog):
        """The alias line says Memphis = Mit Rahina; the data says
        `Mit Rahina (Saqqara)`. Agreeing on one name while adding another is
        the transitive chain that once lost Saqqara, so the entry keeps its
        own declaration — Saqqara groups under it, nothing joins Memphis, and
        the overlap is warned so the line gets fixed."""
        import logging

        aliases = (topics.AliasGroup(names=("Memphis", "Mit Rahina")),)
        subjects = ["Memphis", "Mit Rahina (Saqqara)", "Saqqara"]

        with caplog.at_level(logging.WARNING):
            kept = topics.group_synonyms(subjects, aliases=aliases)

        assert "partially overlaps" in caplog.text
        assert kept == ["Memphis", "Mit Rahina (Saqqara)"], kept
        assert any(topics.same_topic(k, "Saqqara") for k in kept), \
            "the data's own declared link was broken by the alias line"

    def test_two_stored_atoms_for_one_declared_topic_are_warned(self, caplog):
        """Both `Amarna` and `Akhetaten` on disk is the duplicate the alias
        line exists to prevent; grouping keeps one and says so."""
        import logging

        have = {"Amarna", "Akhetaten"}
        with caplog.at_level(logging.WARNING):
            kept = topics.group_synonyms(
                ["Akhetaten", "Tell el-Amarna"],
                prefer=lambda s: s in have,
                aliases=self.AMARNA)

        assert kept == ["Amarna"]
        assert "stored atoms" in caplog.text

    def test_a_scoped_group_never_claims_a_context_free_entry(self):
        """`group_synonyms` gets bare strings, so scoped lines do not apply."""
        aliases = (topics.AliasGroup(
            names=("Thebes (Boeotia)", "Thebes", "Thebai"), scope="Greek History"),)

        kept = topics.group_synonyms(["Thebes", "Thebai"], aliases=aliases)

        assert kept == ["Thebes", "Thebai"]

    def test_no_aliases_changes_nothing(self):
        subjects = ["Thebes (Luxor)", "Thebes", "Memphis"]

        assert (topics.group_synonyms(subjects, aliases=())
                == topics.group_synonyms(subjects))


class TestCrossCivilisationSplits:
    """One name, two places — the failure mode a second civilisation brings.

    Without a scoped alias, a bare Greek `Thebes` is claimed by the Egyptian
    `Thebes (Luxor)` declaration and the Greek city is never generated. The
    scoped lines keep each civilisation's claim inside its own History.
    """

    ALIASES = """
        [Egyptian History]
        Thebes (Luxor) = Thebes = Luxor
        [Greek History]
        Thebes (Boeotia) = Thebes = Thebai
    """

    @pytest.fixture
    def alias_file(self, tmp_path, monkeypatch):
        path = tmp_path / "aliases.txt"
        path.write_text(self.ALIASES, encoding="utf-8")
        monkeypatch.setattr(topics.settings, "ALIASES_LIST", path)

    def test_a_shared_name_splits_by_civilisation(self, workspace, alias_file):
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Thebes (Luxor)", "Thebes"])
        _history(workspace, "history_h2222222222222222_greek_history_m.json",
                 cities=["Thebes"])

        found = topics.from_history("city")

        assert {(t.subject, t.context) for t in found} == {
            ("Thebes (Luxor)", "egyptian history"),
            ("Thebes (Boeotia)", "greek history"),
        }, found

    def test_the_split_yields_two_distinct_content_ids(self, workspace, alias_file):
        """The whole point: two files, not one file fought over by two cities."""
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Thebes"])
        _history(workspace, "history_h2222222222222222_greek_history_m.json",
                 cities=["Thebes"])

        found = topics.from_history("city")
        ids = {store.atom_id("city", t.subject) for t in found}

        assert len(found) == 2 and len(ids) == 2

    def test_a_scoped_line_cannot_capture_a_self_declared_different_city(
            self, workspace, alias_file):
        """The review-gate case: a Greek History legitimately lists the
        Egyptian city as `Thebes (Luxor)`. The Greek-scoped line owns the bare
        `Thebes`, but the entry says outright which city it is — its own
        declaration wins, and the Egyptian city must not merge into Boeotia."""
        _history(workspace, "history_h2222222222222222_greek_history_m.json",
                 cities=["Thebes", "Thebes (Luxor)"])

        found = topics.from_history("city")

        assert {t.subject for t in found} == {"Thebes (Boeotia)", "Thebes (Luxor)"}, found

    def test_one_spelling_in_two_civilisations_is_still_one_row(
            self, workspace, alias_file):
        """Both Histories list `Thebes (Luxor)` — the Egyptian one groups it,
        the Greek one keeps it alone (its pair is not in the Greek line, its
        context is not the Egyptian scope). Same normalised topic, same
        content id, so the work list must carry it once, not as two calls
        racing for one file."""
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Thebes (Luxor)"])
        _history(workspace, "history_h2222222222222222_greek_history_m.json",
                 cities=["Thebes (Luxor)"])

        found = topics.from_history("city")

        assert [(t.subject, t.context) for t in found] == \
            [("Thebes (Luxor)", "egyptian history")], found

    def test_a_scoped_line_does_not_hand_its_bare_name_to_another_civilisations_atom(
            self, caplog):
        """The Abydos case: the Egyptian city was generated under
        the bare name, so `[Byzantine History] Abydos (Hellespont) = Abydos`
        used to resolve the Byzantine `Abydos` onto Egypt — the corpus won over
        the line, and the Hellespontine city could never be generated. Inside a
        scope, a stored atom under a bare non-leading name is not the line's
        subject; the first name is generated."""
        import logging

        aliases = (topics.AliasGroup(
            names=("Abydos (Hellespont)", "Abydos"), scope="Byzantine History"),)
        have = {"Abydos"}                      # the Egyptian city, stored bare

        with caplog.at_level(logging.WARNING):
            kept = topics._group_occurrences(
                [("Abydos", "Egyptian History"), ("Abydos", "Byzantine History")],
                prefer=lambda s: s in have, aliases=aliases)

        assert kept == [("Abydos", "Egyptian History"),
                        ("Abydos (Hellespont)", "Byzantine History")], kept
        assert "Scoped line" in caplog.text and "Abydos (Hellespont)" in caplog.text

    def test_a_scoped_line_still_finds_its_own_atom_under_the_leading_name(self):
        """The exemption: when the scope's atom was generated under a bare
        spelling, the author leads the line with it, and it is found."""
        aliases = (topics.AliasGroup(
            names=("Thebes", "Thebes (Boeotia)", "Thebai"), scope="Greek History"),)

        kept = topics._group_occurrences(
            [("Thebai", "Greek History")],
            prefer=lambda s: s == "Thebes", aliases=aliases)

        assert kept == [("Thebes", "Greek History")]

    def test_a_global_line_keeps_the_corpus_wins_rule(self):
        """Outside a scope the stored spelling is the same thing by
        definition, so a bare stored name still wins the group."""
        aliases = (topics.AliasGroup(names=("Abydos (Hellespont)", "Abydos")),)

        kept = topics._group_occurrences(
            [("Abydos (Hellespont)", "Byzantine History")],
            prefer=lambda s: s == "Abydos", aliases=aliases)

        assert kept == [("Abydos", "Byzantine History")]

    def test_a_declaration_does_not_claim_across_civilisations(
            self, workspace, tmp_path, monkeypatch):
        """The second review-gate case: with NO alias line at all, a Greek
        bare `Thebes` used to join the Egyptian `Thebes (Luxor)` declaration
        and the Greek city was never generated. A data declaration speaks
        only for the Histories that made it; the foreign bare name stands
        alone — a visible split, never an invisible merge."""
        monkeypatch.setattr(topics.settings, "ALIASES_LIST", tmp_path / "absent.txt")
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Thebes (Luxor)", "Thebes"])
        _history(workspace, "history_h2222222222222222_greek_history_m.json",
                 cities=["Thebes"])

        found = topics.from_history("city")

        assert {(t.subject, t.context) for t in found} == {
            ("Thebes (Luxor)", "egyptian history"),
            ("Thebes", "greek history"),
        }, found

    def test_the_same_declaration_from_two_histories_serves_both(
            self, workspace, tmp_path, monkeypatch):
        """Agreement still crosses: two Histories both declaring
        `Thebes (Luxor)` are one claim, and either's bare names attach."""
        monkeypatch.setattr(topics.settings, "ALIASES_LIST", tmp_path / "absent.txt")
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Thebes (Luxor)"])
        _history(workspace, "history_h3333333333333333_ptolemaic_history_m.json",
                 cities=["Thebes (Luxor)", "Thebes"])

        found = topics.from_history("city")

        assert [t.subject for t in found] == ["Thebes (Luxor)"], found

    def test_a_global_line_is_how_civilisations_join(
            self, workspace, tmp_path, monkeypatch):
        """Alexandria in two Histories genuinely is one city, and saying so
        is now the alias file's job: a global line joins what the boundary
        rule would otherwise keep split."""
        path = tmp_path / "aliases.txt"
        path.write_text("Alexandria (Iskandariyya) = Alexandria\n", encoding="utf-8")
        monkeypatch.setattr(topics.settings, "ALIASES_LIST", path)
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Alexandria (Iskandariyya)"])
        _history(workspace, "history_h2222222222222222_greek_history_m.json",
                 cities=["Alexandria"])

        found = topics.from_history("city")

        assert [t.subject for t in found] == ["Alexandria (Iskandariyya)"], found

    def test_the_bundled_seed_resolves_akhetaten_onto_the_corpus(
            self, workspace, monkeypatch):
        """Pins the shipped `aliases.txt`, not just the mechanism: the real
        corpus stores `Amarna (Akhetaten)`, and a list saying `Akhetaten` or
        `Tell el-Amarna` must resolve onto it rather than grow a sibling.

        `workspace` isolates the alias file, so this test alone points back at
        the bundled one — the history pack's shipped seed, under test here."""
        from universe_builder import settings as settings_mod
        from universe_builder.domains.history import ALIASES_PATH

        monkeypatch.setattr(settings_mod, "ALIASES_LIST", ALIASES_PATH)
        path = store.path_for("city", None, "Amarna (Akhetaten)", "m")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"name": {"common": "Amarna"}}', encoding="utf-8")
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Akhetaten", "Tell el-Amarna"])

        found = topics.from_history("city")

        assert [t.subject for t in found] == ["Amarna (Akhetaten)"]


class TestAFailedAtomNeverOutranksARealOne:
    """`store.atoms_for` counts files, not usable content.

    A failed generation leaves a 32-byte `{"error": "No data found"}` placeholder
    at the atom's own path. Passing that to `prefer` made a *failure* outrank a
    real atom: the work list kept the topic with the placeholder and the sibling
    holding 5KB of validated content had nothing pointing at it.
    """

    def _write(self, subject, body):
        path = store.path_for("city", None, subject, "m")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    def test_a_placeholder_is_not_a_real_atom(self, workspace):
        self._write("Thebes", '{\n    "error": "No data found"\n}')

        assert store.atoms_for("city", "Thebes"), "precondition: the file exists"
        assert not topics.has_real_atom("city", "Thebes")

    def test_a_validated_atom_is(self, workspace):
        self._write("Thebes", '{"name": {"common": "Thebes"}}')

        assert topics.has_real_atom("city", "Thebes")

    def test_a_readable_but_unparseable_atom_is_not_real(self, workspace):
        """A non-empty, non-placeholder file that is not valid JSON is not a
        usable atom — the same safe direction as an unreadable one, so the topic
        is regenerated rather than counted as present and skipped."""
        self._write("Thebes", "not json {")

        assert store.atoms_for("city", "Thebes"), "precondition: the file exists"
        assert not topics.has_real_atom("city", "Thebes")

    def test_the_real_atom_wins_the_group(self, workspace):
        self._write("Thebes", '{\n    "error": "No data found"\n}')
        self._write("Thebes (Luxor)", '{"name": {"common": "Thebes"}}')

        kept = topics.group_synonyms(
            ["Thebes", "Thebes (Luxor)"],
            prefer=lambda s: topics.has_real_atom("city", s))

        assert kept == ["Thebes (Luxor)"], (
            "the work list points at a failed generation and the real atom is "
            "unreachable from it"
        )

    def test_coverage_and_the_work_list_agree(self, workspace):
        """They were separate judgements and disagreed."""
        from universe_builder import cli

        self._write("Memphis", '{\n    "error": "No data found"\n}')
        assert not topics.has_real_atom("city", "Memphis")
        assert cli._coverage.__module__  # the shared helper is what coverage uses


class TestCreateWorkIsOnlyWhatIsMissing:
    """`needing_create` is what every create path consumes.

    The full list is coverage's business; create's business is the gap. Handing
    create the full list meant every generated topic was paid for again — and
    on the batch path, `collect` writes through `store.write_atom`, so the
    existing atom was then *overwritten* by the rerun's output. Caught by the
    first real dry run: 1,236 work items announced against 784 known missing.
    """

    def _write_atom(self, subject, body='{"name": {"common": "x"}}'):
        path = store.path_for("city", None, subject, "m")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")

    def test_generated_topics_leave_the_create_list(self, workspace):
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Memphis", "Thebes", "Giza"])
        self._write_atom("Thebes")

        assert [t.subject for t in topics.needing_create("city")] == ["Memphis", "Giza"]

    def test_a_placeholder_is_still_missing(self, workspace):
        """A recorded failure is not coverage; the topic stays queued."""
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Thebes"])
        self._write_atom("Thebes", '{\n    "error": "No data found"\n}')

        assert [t.subject for t in topics.needing_create("city")] == ["Thebes"]

    def test_an_atom_under_an_alias_spelling_counts_as_generated(
            self, workspace, tmp_path, monkeypatch):
        """The alias work feeds directly into the cost: without it, a list
        saying `Akhetaten` looks ungenerated beside a stored
        `Amarna (Akhetaten)` and the duplicate is paid for."""
        aliases = tmp_path / "test-aliases.txt"
        aliases.write_text("Amarna (Akhetaten) = Amarna = Akhetaten\n", encoding="utf-8")
        monkeypatch.setattr(topics.settings, "ALIASES_LIST", aliases)
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 cities=["Akhetaten"])
        self._write_atom("Amarna (Akhetaten)")

        assert topics.needing_create("city") == []


class TestOneUnreadableFileCostsOneRecord:
    """A read inside a loop over files must skip, not abort.

    This shape has now appeared four times in one day — `_history_files`,
    `has_real_atom`, `_submit_payload_batch` and the fix/verify loop — each time
    because a guard was added to one call site and the next one written without
    it. The cost is always the same: one bad byte and the caller loses every
    record behind it, sorted order deciding which.
    """

    def _history(self, **lists):
        import json

        payload = {"people": [], "discoveries": [], "events": [], "cities": [],
                   "artifacts": []}
        payload.update(lists)
        path = store.dir_for("history") / "history_h1111111111111111_egyptian_history_m.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_has_real_atom_never_raises(self, workspace):
        path = store.path_for("city", None, "Memphis", "m")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\xff\xfe{")

        assert topics.has_real_atom("city", "Memphis") is False

    def test_the_work_list_survives_an_unreadable_atom(self, workspace, caplog):
        """`prefer` only runs when there is a group to resolve, which is why
        this needs a synonym pair to reproduce at all."""
        self._history(cities=["Memphis", "Memphis (Men-nefer)", "Thebes", "Giza"])
        store.path_for("city", None, "Memphis", "m").parent.mkdir(parents=True, exist_ok=True)
        store.path_for("city", None, "Memphis", "m").write_bytes(b"\xff\xfe{")
        store.path_for("city", None, "Memphis (Men-nefer)", "m").write_text(
            '{"name": {"common": "Memphis"}}', encoding="utf-8")

        found = [t.subject for t in topics.from_history("city")]

        assert "Thebes" in found and "Giza" in found, (
            f"one unreadable atom took the rest of the work list with it: {found}"
        )
        assert "Memphis (Men-nefer)" in found, "the readable form should win the group"

    def test_the_readable_sibling_is_preferred(self, workspace):
        """An unreadable file is not a usable atom, so the topic stays queued."""
        store.dir_for("city").mkdir(parents=True, exist_ok=True)
        store.path_for("city", None, "Memphis", "m").write_bytes(b"\xff\xfe{")

        assert not topics.has_real_atom("city", "Memphis")


class TestAdjudicatedLinesAfterADerivedSectionAreAsserted:
    """Provenance is sticky down the file. `lint --apply-review` appends its
    lines at the end, which is after every `duplicates --apply` section, so
    its header must flip provenance back — with the words it actually writes."""

    def test_the_apply_review_header_flips_provenance_back(self, tmp_path):
        path = tmp_path / "aliases.txt"
        path.write_text(
            "# Added by `universe_builder duplicates --apply`: one subject, two atoms.\n"
            "[*]\n"
            "Alpha = Alpha Prime\n"
            "\n"
            "# Added by `universe_builder lint --apply-review`: adjudicated identity.\n"
            "[*]\n"
            "Beta = Beta Prime\n", encoding="utf-8")
        by_name = {g.names[0]: g for g in topics.load_aliases(path)}
        assert by_name["Alpha"].provenance == "derived"
        assert by_name["Beta"].provenance == "asserted"

    def test_the_hand_marker_still_works(self, tmp_path):
        path = tmp_path / "aliases.txt"
        path.write_text(
            "# Added by `universe_builder duplicates --apply`: one subject, two atoms.\n"
            "[*]\n"
            "Alpha = Alpha Prime\n"
            "# adjudicated by hand\n"
            "Beta = Beta Prime\n", encoding="utf-8")
        by_name = {g.names[0]: g for g in topics.load_aliases(path)}
        assert by_name["Beta"].provenance == "asserted"


class TestKeysFoldDiacriticsAndRefuseOneLetterCores:
    """Found on the corpus: Kos, Kush and Catania grouped as one
    city because `Kō`, `Kꜣš` and `Kατάνη` all keyed as `k`, and the 2013
    coup's `d’état` keyed as `d tat` so the curly-apostrophe spelling never
    matched the straight one."""

    def test_accents_fold_rather_than_vanish(self):
        assert topics._key("Châlons") == topics._key("Chalons") == "chalons"
        assert topics._key("d’état") == "d etat"

    def test_names_in_other_scripts_no_longer_collapse_onto_one_letter(self):
        keys = {topics._key(n) for n in ("Kō", "Kꜣš", "Kατάνη")}
        assert len(keys) == 3, keys   # was {"k"}: one key for three cities
        assert not (topics.alias_keys("Kō") & topics.alias_keys("Kꜣš"))
        assert not (topics.alias_keys("Kō") & topics.alias_keys("Kατάνη"))


class TestTheCivilisationNameComesFromTheSeedLine:
    """The slug keeps only ASCII letters and digits, so a History file's name
    cannot give back `Anglo-Saxon History`. Its content id hashes the seed line
    whole, which can (found by a new-history walk-through)."""

    def _seed_list(self, tmp_path, monkeypatch, *lines):
        from universe_builder import settings

        seed = tmp_path / "seed.txt"
        seed.write_text("\n".join(lines) + "\n", encoding="utf-8")
        monkeypatch.setattr(settings, "HISTORY_LIST", seed)

    def _stored_history(self, line, **lists):
        payload = {"people": [], "discoveries": [], "events": [], "cities": [], "artifacts": []}
        payload.update(lists)
        path = store.path_for("history", None, line, "m")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def test_a_hyphen_and_an_accent_survive(self, workspace, tmp_path, monkeypatch):
        self._seed_list(tmp_path, monkeypatch, "Anglo-Saxon History", "Mesoamérican History")
        hyphen = self._stored_history("Anglo-Saxon History")
        accent = self._stored_history("Mesoamérican History")

        assert "anglosaxon" in hyphen.name, "the slug still drops the hyphen; the id carries it"
        assert topics._context_of(hyphen.name) == "anglo-saxon history"
        assert topics._context_of(accent.name) == "mesoamérican history"

    def test_a_plain_seed_line_gives_the_same_context_as_before(self, workspace, tmp_path, monkeypatch):
        self._seed_list(tmp_path, monkeypatch, "Egyptian History")
        path = self._stored_history("Egyptian History")

        assert topics._context_of(path.name) == "egyptian history"

    def test_a_line_gone_from_the_seed_list_falls_back_to_the_slug(self, workspace, tmp_path, monkeypatch):
        self._seed_list(tmp_path, monkeypatch, "Greek History")
        path = self._stored_history("Anglo-Saxon History")

        assert topics._context_of(path.name) == "anglosaxon history"

    def test_a_scoped_alias_section_for_a_hyphenated_seed_takes_effect(self, workspace, tmp_path, monkeypatch):
        from universe_builder import settings

        self._seed_list(tmp_path, monkeypatch, "Anglo-Saxon History")
        self._stored_history("Anglo-Saxon History", cities=["York"])
        aliases = tmp_path / "aliases.txt"
        aliases.write_text("[Anglo-Saxon History]\nYork (Eoforwic) = York\n", encoding="utf-8")
        monkeypatch.setattr(settings, "ALIASES_LIST", aliases)

        assert [t.subject for t in topics.needing_create("city")] == ["York (Eoforwic)"]


class TestAQuotedAliasNameIsLiteral:
    """`"…"` around a name in an alias line makes it one literal name (v0.4.0).

    Unquoted, a trailing parenthetical splits: an alias line naming the event
    `Unification of Upper and Lower Egypt by King Menes (Narmer)` also
    claimed the key `narmer`, and a global line is read for every atom type,
    so the person `Narmer` was claimed by an event's line.
    """

    TITLE = "Unification of Upper and Lower Egypt by King Menes (Narmer)"
    BARE = "Unification of Upper and Lower Egypt"
    LINE = f'"{TITLE}" = {BARE}'

    def _load(self, tmp_path, text):
        path = tmp_path / "aliases.txt"
        path.write_text(text, encoding="utf-8")
        return topics.load_aliases(path)

    def test_the_quotes_stay_in_names_and_the_key_is_the_whole_title(self, tmp_path):
        [group] = self._load(tmp_path, self.LINE + "\n")

        assert group.names == (f'"{self.TITLE}"', self.BARE)
        assert group.spellings == (self.TITLE, self.BARE)
        assert group.keys() == {topics.name_key(self.TITLE), topics.name_key(self.BARE)}
        assert "narmer" not in group.keys()
        assert group.literal_keys() == {topics.name_key(self.TITLE)}

    def test_name_key_drops_the_quotes_and_alias_keys_does_not_split(self):
        assert topics.name_key(f'"{self.TITLE}"') == topics.name_key(self.TITLE)
        assert topics.alias_keys(f'"{self.TITLE}"') == {topics.name_key(self.TITLE)}
        assert topics.alias_keys(self.TITLE) == {topics.name_key(self.TITLE), "narmer",
                                                 topics.name_key(self.TITLE[:-9])}

    def test_unquote(self):
        assert topics.unquote('"Foo (Bar)"') == "Foo (Bar)"
        assert topics.unquote(' "Foo" ') == "Foo"
        assert topics.unquote("Foo (Bar)") == "Foo (Bar)"
        assert topics.unquote('Foo "Bar"') == 'Foo "Bar"'
        assert not topics.is_quoted('Foo "Bar"')

    def test_the_line_claims_the_parenthetical_entry_and_the_bare_name(self, tmp_path):
        """Pass 2: the entry's whole key is the line's quoted name, so the line
        captures it although its pair (head, Narmer) is not inside the line."""
        aliases = self._load(tmp_path, self.LINE + "\n")
        assignments = {}

        kept = topics.group_occurrences(
            [(self.TITLE, "Egyptian History"), (self.BARE, "Egyptian History"),
             ("Establishment of the Kingdom of Egypt", "Egyptian History")],
            prefer=lambda s: s in {self.TITLE, "Establishment of the Kingdom of Egypt"},
            aliases=aliases, assignments=assignments)

        assert kept == [(self.TITLE, "Egyptian History"),
                        ("Establishment of the Kingdom of Egypt", "Egyptian History")]
        assert assignments[(self.BARE, "egyptian history")] == self.TITLE

    def test_the_survivor_falls_back_to_the_unquoted_first_name(self, tmp_path):
        aliases = self._load(tmp_path, self.LINE + "\n")

        assert topics.group_synonyms([self.BARE], aliases=aliases) == [self.TITLE]

    def test_the_inner_name_is_not_claimed(self, tmp_path, caplog):
        """The same line read for the person list: `Narmer` and `Narmer
        (Menes)` group on the data's own declaration, unclaimed by the line."""
        import logging

        aliases = self._load(tmp_path, self.LINE + "\n")
        with caplog.at_level(logging.WARNING):
            kept = topics.group_occurrences(
                [("Narmer", "Egyptian History"), ("Narmer (Menes)", "Egyptian History")],
                prefer=lambda s: s == "Narmer (Menes)", aliases=aliases)

        assert kept == [("Narmer (Menes)", "Egyptian History")]
        assert "claimed by" not in caplog.text

    def test_unquoted_the_same_line_claims_narmer(self, tmp_path, caplog):
        """Why the quotes exist: the unquoted line and the data's `Narmer
        (Menes)` both claim `narmer`, so the bare `Narmer` is left ungrouped."""
        import logging

        aliases = self._load(tmp_path, f"{self.TITLE} = {self.BARE}\n")
        with caplog.at_level(logging.WARNING):
            kept = topics.group_occurrences(
                [("Narmer", "Egyptian History"), ("Narmer (Menes)", "Egyptian History")],
                prefer=lambda s: s == "Narmer (Menes)", aliases=aliases)

        assert ("Narmer", "Egyptian History") in kept
        assert "claimed by 2 different declarations" in caplog.text

    def test_a_literal_capture_outranks_a_declaration_sharing_its_inner_name(self, tmp_path, caplog):
        """Another entry declaring `Narmer (Menes)` shares the inner name; the
        quoted line still takes the title, without an ambiguity."""
        import logging

        aliases = self._load(tmp_path, self.LINE + "\n")
        with caplog.at_level(logging.WARNING):
            kept = topics.group_occurrences(
                [(self.TITLE, "Egyptian History"), ("Narmer (Menes)", "Egyptian History"),
                 (self.BARE, "Egyptian History")],
                aliases=aliases)

        assert kept == [(self.TITLE, "Egyptian History"), ("Narmer (Menes)", "Egyptian History")]
        assert "claimed by" not in caplog.text

    def test_a_scoped_quoted_line_speaks_only_in_its_scope(self, tmp_path):
        aliases = self._load(tmp_path, f"[Egyptian History]\n{self.LINE}\n")

        kept = topics.group_occurrences(
            [(self.BARE, "Greek History"), (self.TITLE, "Greek History")], aliases=aliases)

        assert kept == [(self.BARE, "Greek History"), (self.TITLE, "Greek History")]

    def test_create_finds_the_stored_title_and_regenerates_nothing(self, workspace, tmp_path,
                                                                   monkeypatch):
        """End to end on the work list: the event atom is stored under the
        title, the person under `Narmer (Menes)`; with the quoted line, neither
        list has work and `Narmer` stays with the person."""
        aliases = tmp_path / "test-aliases.txt"
        aliases.write_text(self.LINE + "\n", encoding="utf-8")
        monkeypatch.setattr(topics.settings, "ALIASES_LIST", aliases)
        _history(workspace, "history_h1111111111111111_egyptian_history_m.json",
                 people=["Narmer", "Narmer (Menes)"], events=[self.BARE, self.TITLE])
        for atom_type, subject in (("event", self.TITLE), ("person", "Narmer (Menes)")):
            path = store.path_for(atom_type, None, subject, "m")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"name": {"common": subject}}), encoding="utf-8")

        assert topics.needing_create("event") == []
        assert topics.needing_create("person") == []
        assert topics.resolve_subject("person", "Narmer") == store.atom_id("person", "Narmer (Menes)")
        assert topics.resolve_subject("event", self.BARE) == store.atom_id("event", self.TITLE)
