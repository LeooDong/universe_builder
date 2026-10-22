"""One subject, two files: detection, the keep/retire decision, and the repair.

Nothing here calls a model, and `apply` is exercised only against the
`workspace` fixture's temporary data dir.
"""

from __future__ import annotations

import json

from pathlib import Path

import pytest

from universe_builder import duplicates, settings, store, topics

MODEL = "gpt-5-nano"


def _atom(atom_type, subject, model=MODEL, payload=None):
    path = store.path_for(atom_type, None, subject, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {"name": {"common": subject}}),
                    encoding="utf-8")
    return path


def _distinct(tmp_path, text):
    """Write a not-duplicates file and return its path."""
    path = tmp_path / "not_duplicates.txt"
    path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
    return path


def _memphis_pair():
    """The real pair, as it sits in the corpus: the newer atom declares the
    older atom's spelling in its own `otherNames`."""
    older = _atom("event", "Founding of Memphis", model="gpt-4o-mini")
    newer = _atom("event", "Foundation of Memphis", model="gpt-5-nano",
                  payload={"name": {"common": "Foundation of Memphis",
                                    "otherNames": ["Founding of Memphis"]}})
    return older, newer


class TestModelRank:
    def test_a_later_family_outranks_an_earlier_one(self):
        assert duplicates.model_rank("gpt-5-nano") > duplicates.model_rank("gpt-4o-mini")

    def test_the_o_in_4o_is_a_name_not_a_decimal(self):
        assert duplicates.model_rank("gpt-4o-mini") == 4.0

    def test_a_point_release_ranks_between_families(self):
        assert (duplicates.model_rank("gpt-5-nano")
                < duplicates.model_rank("gpt-5.6-luna")
                < duplicates.model_rank("gpt-6-astra"))

    def test_a_version_free_model_is_unrankable(self):
        assert duplicates.model_rank("some-local-model") is None


def _alias_group(*names, scope=None, provenance="asserted"):
    """An assertion that these names are one subject — the gate the fill
    tie-break requires.

    Global and asserted by default, because those are the only lines the gate
    accepts: a scoped line asserts identity within one civilisation and is
    often written to keep names apart elsewhere, and a derived line is a pass
    recording its own conclusion.
    """
    from universe_builder.topics import AliasGroup

    return (AliasGroup(names=tuple(names), scope=scope, provenance=provenance),)


class TestThePlan:
    def test_the_later_model_is_kept(self, workspace):
        older, newer = _memphis_pair()

        plan = duplicates.plan()

        assert [r.keep for r in plan.resolutions] == [newer]
        assert [r.retire for r in plan.resolutions] == [[older]]

    def test_the_alias_line_puts_the_kept_spelling_first(self, workspace):
        """The first name on an alias line is the spelling the work list emits
        when no stored atom decides otherwise, so it must be the survivor."""
        _memphis_pair()

        assert duplicates.plan().alias_lines() == [
            "Foundation of Memphis = Founding of Memphis"]

    def test_a_tie_within_one_family_falls_to_the_fuller_atom(self, workspace):  # noqa: E501
        """Nothing in a *filename* ranks two atoms of one model family — but
        the atoms themselves do.

        This used to end in `undecided`, on the grounds that guessing discards
        content on a coin flip. It is not a coin flip: of two records about one
        subject, the one filling more fields is the one worth keeping, and it
        is measurable. Before this, that left dozens of genuine duplicates
        unretirable, every pair written by `gpt-5.6-luna`.
        """
        _atom("event", "Founding of Memphis", model="gpt-5-nano")
        _atom("event", "Foundation of Memphis", model="gpt-5-nano",
              payload={"name": {"common": "Foundation of Memphis",
                                "otherNames": ["Founding of Memphis"]}})

        plan = duplicates.plan(tiebreak_fill=True,
                               aliases=_alias_group("Founding of Memphis",
                                                    "Foundation of Memphis"))

        assert len(plan.resolutions) == 1
        assert "populated fields" in plan.resolutions[0].reason
        assert plan.undecided == []

    def test_an_exact_fill_tie_is_still_left_to_a_human(self, workspace):
        """The measurement has to be able to come out level, or it is not a
        measurement. Equal generation and equal content: nothing separates
        them, and the report says which number tied."""
        _atom("event", "Founding of Memphis", model="gpt-5-nano",
              payload={"name": {"common": "Founding of Memphis", "otherNames": []},
                       "significance": "x"})
        _atom("event", "Foundation of Memphis", model="gpt-5-nano",
              payload={"name": {"common": "Foundation of Memphis",
                                "otherNames": ["Founding of Memphis"]}})

        plan = duplicates.plan(tiebreak_fill=True,
                               aliases=_alias_group("Founding of Memphis",
                                                    "Foundation of Memphis"))

        assert plan.resolutions == []
        assert len(plan.undecided) == 1
        assert "fill the same number of fields" in plan.undecided[0]["why"]

    def test_a_clean_corpus_plans_nothing(self, workspace):
        _atom("city", "Memphis")
        _atom("city", "Thebes")

        assert duplicates.plan().is_clean()


class TestApply:
    def test_a_dry_run_moves_nothing(self, workspace):
        older, _ = _memphis_pair()

        outcome = duplicates.apply(duplicates.plan(), dry_run=True)

        assert outcome["dry_run"] is True
        assert outcome["retired"] == []
        assert older.exists()

    def test_the_loser_moves_to_duplicates_rather_than_being_deleted(self, workspace,
                                                                     tmp_path):
        """`data/` is not in git and there is no undo, so retirement is a move."""
        older, newer = _memphis_pair()

        duplicates.apply(duplicates.plan(), aliases_path=tmp_path / "aliases.txt",
                         dry_run=False)

        assert not older.exists()
        assert newer.exists()
        assert (settings.duplicates_dir() / older.name).exists()

    def test_a_retired_atom_is_not_counted_as_an_atom(self, workspace, tmp_path):
        older, _ = _memphis_pair()

        duplicates.apply(duplicates.plan(), aliases_path=tmp_path / "aliases.txt",
                         dry_run=False)

        assert older.name not in [p.name for p in store.iter_atoms()]

    def test_the_alias_line_is_written_so_the_topic_is_not_regenerated(self, workspace,
                                                                       tmp_path):
        """Retiring the file is only half the repair. The losing spelling is
        still listed in a stored History atom, so without an alias line
        `needing_create` sees a listed topic with no atom and regenerates the
        duplicate at the next create run."""
        aliases = tmp_path / "aliases.txt"
        _memphis_pair()

        duplicates.apply(duplicates.plan(), aliases_path=aliases, dry_run=False)

        assert "Foundation of Memphis = Founding of Memphis" in aliases.read_text()

    def test_an_existing_alias_file_is_appended_to_not_replaced(self, workspace, tmp_path):
        aliases = tmp_path / "aliases.txt"
        aliases.write_text("Amarna (Akhetaten) = Tell el-Amarna\n", encoding="utf-8")
        _memphis_pair()

        duplicates.apply(duplicates.plan(), aliases_path=aliases, dry_run=False)

        text = aliases.read_text()
        assert "Amarna (Akhetaten) = Tell el-Amarna" in text
        assert "Foundation of Memphis = Founding of Memphis" in text

    def test_the_appended_lines_are_global_not_scoped_to_a_stale_header(self, workspace,
                                                                        tmp_path):
        """The alias file is stateful: a trailing `[Greek History]` header
        would silently scope everything appended after it."""
        aliases = tmp_path / "aliases.txt"
        aliases.write_text("[Greek History]\nThebes (Boeotia) = Thebes\n",
                           encoding="utf-8")
        _memphis_pair()

        duplicates.apply(duplicates.plan(), aliases_path=aliases, dry_run=False)

        groups = topics.load_aliases(aliases)
        memphis = [g for g in groups if "Foundation of Memphis" in g.names]
        assert memphis and memphis[0].scope is None

    def test_the_retired_spelling_now_resolves_onto_the_survivor(self, workspace,
                                                                 tmp_path, monkeypatch):
        """The point of the whole exercise: one subject, one atom, and the
        spelling that lost still finds it."""
        aliases = tmp_path / "aliases.txt"
        _, newer = _memphis_pair()
        duplicates.apply(duplicates.plan(), aliases_path=aliases, dry_run=False)
        monkeypatch.setattr(topics.settings, "ALIASES_LIST", aliases)

        assert (topics.resolve_subject("event", "Founding of Memphis")
                == store.AtomName.decompose_filename(newer.name).padded_id)

    def test_the_corpus_is_clean_afterwards(self, workspace, tmp_path):
        _memphis_pair()

        duplicates.apply(duplicates.plan(), aliases_path=tmp_path / "aliases.txt",
                         dry_run=False)

        assert topics.duplicate_atoms("event") == []


class TestOnePlanEntryPerPair:
    """Two atoms commonly share several names — `Ramesses II` also declares
    `Ozymandias` and `Ramses II`. Planned per shared name, the same two files
    appeared three times, and applying that plan called `retire_duplicate` on a
    file the first pass had already moved, where `os.replace` raises."""

    def _ramesses(self):
        older = _atom("person", "Ramses II", model="gpt-4o-mini",
                      payload={"name": {"common": "Ramses II",
                                        "otherNames": ["Ozymandias"]}})
        newer = _atom("person", "Ramesses II", model="gpt-5.6-luna",
                      payload={"name": {"common": "Ramesses II",
                                        "otherNames": ["Ramses II", "Ozymandias"]}})
        return older, newer

    def test_three_shared_names_make_one_resolution(self, workspace):
        self._ramesses()

        assert len(topics.duplicate_atoms("person")) == 2   # two shared keys
        assert len(duplicates.plan().resolutions) == 1      # but one pair

    def test_the_entry_names_every_shared_spelling(self, workspace):
        self._ramesses()

        assert duplicates.plan().resolutions[0].name == "ozymandias, ramses ii"

    def test_applying_retires_the_file_once(self, workspace, tmp_path):
        older, _ = self._ramesses()

        outcome = duplicates.apply(duplicates.plan(),
                                   aliases_path=tmp_path / "aliases.txt",
                                   dry_run=False)

        assert outcome["retired"] == [str(settings.duplicates_dir() / older.name)]
        assert not older.exists()


class TestExclusion:
    """A shared name is evidence of a duplicate, not proof of one. The modern
    Bibliotheca Alexandrina names the ancient Great Library of Alexandria as a
    related work, not as an alias; retiring either would destroy a real atom to
    tidy a report."""

    def _library_pair(self):
        ancient = _atom("artifact", "The Great Library of Alexandria",
                        model="gpt-4o-mini")
        modern = _atom("artifact", "Bibliotheca Alexandrina", model="gpt-5-nano",
                       payload={"name": {
                           "common": "Bibliotheca Alexandrina",
                           "otherNames": ["The Great Library of Alexandria"]}})
        return ancient, modern

    def test_without_an_exclusion_the_pair_is_planned(self, workspace):
        # `distinct=()` isolates this from the shipped not-duplicates file,
        # which declares this very pair distinct — see TestDeclaredDistinct.
        self._library_pair()

        assert len(duplicates.plan(distinct=()).resolutions) == 1

    def test_an_excluded_pair_is_dropped_from_the_plan(self, workspace):
        self._library_pair()

        plan = duplicates.plan(exclude=["bibliotheca"], distinct=())

        assert plan.resolutions == []
        assert len(plan.skipped) == 1

    def test_matching_either_side_drops_the_whole_pair(self, workspace):
        """Never half-applied: excluding the atom that would have been retired
        must not leave the alias line behind."""
        self._library_pair()

        plan = duplicates.plan(exclude=["great_library"], distinct=())

        assert plan.resolutions == []
        assert plan.alias_lines() == []

    def test_matching_is_case_insensitive(self, workspace):
        self._library_pair()

        assert duplicates.plan(exclude=["BIBLIOTHECA"], distinct=()).resolutions == []

    def test_an_unrelated_exclusion_changes_nothing(self, workspace):
        self._library_pair()

        assert len(duplicates.plan(exclude=["memphis"], distinct=()).resolutions) == 1

    def test_an_excluded_pair_leaves_both_files_on_disk(self, workspace, tmp_path):
        ancient, modern = self._library_pair()

        duplicates.apply(duplicates.plan(exclude=["bibliotheca"], distinct=()),
                         aliases_path=tmp_path / "aliases.txt", dry_run=False)

        assert ancient.exists() and modern.exists()


class TestOneAliasLinePerSurvivor:
    """An atom can beat more than one rival. `Amarna Boundary Stelae` beat both
    `Akhenaten's Boundary Stelae` and `Tell el-Amarna Boundary Stelae` — two
    pairs, one survivor. A line each puts two declarations of the same scope in
    the alias file both claiming the survivor's name, and
    `_group_occurrences` treats a name two declarations claim as ambiguous and
    leaves it ungrouped. The lines written to join the spellings would stop the
    work list joining them."""

    def _stelae(self, tmp_path):
        _atom("artifact", "Akhenaten's Boundary Stelae", model="gpt-4o-mini")
        _atom("artifact", "Tell el-Amarna Boundary Stelae", model="gpt-4o-mini")
        _atom("artifact", "Amarna Boundary Stelae", model="gpt-5-nano",
              payload={"name": {"common": "Amarna Boundary Stelae",
                                "otherNames": ["Akhenaten's Boundary Stelae",
                                               "Tell el-Amarna Boundary Stelae"]}})

    def test_two_pairs_with_one_survivor_make_one_line(self, workspace, tmp_path):
        self._stelae(tmp_path)

        plan = duplicates.plan()

        assert len(plan.resolutions) == 2
        assert plan.alias_lines() == [
            "Amarna Boundary Stelae = Akhenaten's Boundary Stelae "
            "= Tell el-Amarna Boundary Stelae"]

    def test_the_written_file_leaves_the_survivor_unambiguous(self, workspace, tmp_path):
        aliases = tmp_path / "aliases.txt"
        self._stelae(tmp_path)

        duplicates.apply(duplicates.plan(), aliases_path=aliases, dry_run=False)

        groups = [g for g in topics.load_aliases(aliases)
                  if "amarna boundary stelae" in g.keys()]
        assert len(groups) == 1, "two groups claiming one name is the ambiguity case"


class TestAliasLinesSpeakInListedSpellings:
    """An atom's `name.common` is what the model called the subject; the
    History lists hold what the work list looks up. A line written only in
    common names leaves the listed spelling resolving to nothing, and
    `needing_create` regenerates the atom that was just retired."""

    def _history(self, **lists):
        payload = {"people": [], "discoveries": [], "events": [], "cities": [],
                   "artifacts": []}
        payload.update(lists)
        path = store.dir_for("history") / "history_h1111111111111111_egyptian_history_m.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_the_listed_spelling_joins_the_line(self, workspace):
        """The shaduf case: both atoms call themselves `Shaduf`, and the line
        joined nothing until the listed spellings were added."""
        self._history(discoveries=["Shaduf irrigation", "Shaduf (irrigation tool)"])
        _atom("discovery", "Shaduf irrigation", model="gpt-5-nano",
              payload={"name": {"common": "Shaduf"}})
        _atom("discovery", "Shaduf (irrigation tool)", model="gpt-4o-mini",
              payload={"name": {"common": "Shaduf"}})

        line, = duplicates.plan().alias_lines()

        assert "Shaduf irrigation" in line
        assert "Shaduf (irrigation tool)" in line

    def test_a_line_claiming_another_types_name_is_left_to_a_human(self, workspace):
        """`… by King Menes (Narmer)` claims the bare key `narmer`, and the
        alias file has no per-type scope — so a line about events would make
        `Narmer` ambiguous for the people. Observed on the real corpus."""
        self._history(
            people=["Narmer (Menes)"],
            events=["Unification of Upper and Lower Egypt",
                    "Unification of Upper and Lower Egypt by King Menes (Narmer)"])
        _atom("person", "Narmer (Menes)")
        _atom("event", "Unification of Upper and Lower Egypt", model="gpt-5-nano",
              payload={"name": {"common": "Unification of Upper and Lower Egypt"}})
        _atom("event", "Unification of Upper and Lower Egypt by King Menes (Narmer)",
              model="gpt-4o-mini",
              payload={"name": {
                  "common": "Unification of Upper and Lower Egypt by King Menes (Narmer)",
                  "otherNames": ["Unification of Upper and Lower Egypt"]}})

        plan = duplicates.plan()

        assert plan.resolutions == []
        assert len(plan.undecided) == 1
        assert "narmer" in plan.undecided[0]["why"]


class TestChainsAreRefused:
    """Groups never chain — the Saqqara rule — and pairwise duplicates can
    form one.

    `Alpha` (gpt-4o-mini) shares a name with `Beta` (gpt-5-nano), and `Beta`
    shares a *different* name with `Gamma` (gpt-5.6-luna). Taken pairwise, Beta
    wins the first and loses the second, so Beta is named as a survivor by one
    alias line and retired by another: `Alpha = Beta` then points at a file that
    no longer exists, and the alias file declares `Alpha -> Beta -> Gamma`,
    collapsing three atoms into one on two pairwise claims that never said so.
    """

    def _chain(self):
        alpha = _atom("city", "Alpha", model="gpt-4o-mini",
                      payload={"name": {"common": "Alpha",
                                        "otherNames": ["Shared AB"]}})
        beta = _atom("city", "Beta", model="gpt-5-nano",
                     payload={"name": {"common": "Beta",
                                       "otherNames": ["Shared AB", "Shared BC"]}})
        gamma = _atom("city", "Gamma", model="gpt-5.6-luna",
                      payload={"name": {"common": "Gamma",
                                        "otherNames": ["Shared BC"]}})
        return alpha, beta, gamma

    def test_a_chain_resolves_nothing_automatically(self, workspace):
        self._chain()

        plan = duplicates.plan()

        assert plan.resolutions == []
        assert len(plan.undecided) == 2

    def test_the_chain_is_explained(self, workspace):
        self._chain()

        assert all("chain" in entry["why"] for entry in duplicates.plan().undecided)

    def test_no_file_is_retired(self, workspace, tmp_path):
        alpha, beta, gamma = self._chain()

        duplicates.apply(duplicates.plan(), aliases_path=tmp_path / "a.txt",
                         dry_run=False)

        assert alpha.exists() and beta.exists() and gamma.exists()

    def test_an_unrelated_pair_beside_a_chain_still_resolves(self, workspace):
        """The chain is quarantined, not contagious."""
        self._chain()
        _memphis_pair()

        plan = duplicates.plan()

        assert len(plan.resolutions) == 1
        assert plan.resolutions[0].atom_type == "event"


class TestRetirementNeedsSomewhereToRecordTheAlias:
    """Retiring the file is only half the repair. A pack that declares no alias
    file has nowhere to put the other half, so retiring would leave the listed
    spelling resolving to nothing and the next create run would regenerate the
    atom that was just retired — and bill for it."""

    def test_a_pack_without_an_alias_file_refuses_to_retire(self, workspace,
                                                            monkeypatch):
        older, newer = _memphis_pair()
        plan = duplicates.plan()
        monkeypatch.setattr(duplicates, "_pack_aliases_path", lambda: None)

        with pytest.raises(ValueError, match="no alias file"):
            duplicates.apply(plan, dry_run=False)

        assert older.exists() and newer.exists()

    def test_a_dry_run_still_reports_the_plan(self, workspace, monkeypatch):
        _memphis_pair()
        plan = duplicates.plan()
        monkeypatch.setattr(duplicates, "_pack_aliases_path", lambda: None)

        outcome = duplicates.apply(plan, dry_run=True)

        assert outcome["dry_run"] is True
        assert outcome["aliases"]


class TestOnlyAStarResolves:
    """Auto-resolution is safe for exactly one shape: a component in which one
    atom survives and every retirement points at it.

    `Amarna Boundary Stelae` beating two rivals is that shape and must keep
    working. Two *different* survivors both claiming one loser is not: it
    destroys the only atom linking them, leaves two same-scope alias lines
    claiming the retired name (so it resolves to nothing and is regenerated),
    and asserts `Alpha = Beta` and `Gamma = Beta` while declarations never
    chain, so nothing ever said `Alpha = Gamma`.
    """

    def _two_claimants(self):
        """Beta loses to both Alpha and Gamma, on two different shared names.
        No keeper is ever retired, so the chain guard alone sees nothing."""
        beta = _atom("city", "Beta", model="gpt-4o-mini",
                     payload={"name": {"common": "Beta",
                                       "otherNames": ["Shared AB", "Shared CB"]}})
        alpha = _atom("city", "Alpha", model="gpt-5-nano",
                      payload={"name": {"common": "Alpha",
                                        "otherNames": ["Shared AB"]}})
        gamma = _atom("city", "Gamma", model="gpt-5.6-luna",
                      payload={"name": {"common": "Gamma",
                                        "otherNames": ["Shared CB"]}})
        return alpha, beta, gamma

    def test_two_survivors_claiming_one_loser_resolves_nothing(self, workspace):
        self._two_claimants()

        plan = duplicates.plan()

        assert plan.resolutions == []
        assert len(plan.undecided) == 2

    def test_the_conflict_is_explained(self, workspace):
        self._two_claimants()

        assert all("survivor" in e["why"] for e in duplicates.plan().undecided)

    def test_no_file_is_retired(self, workspace, tmp_path):
        alpha, beta, gamma = self._two_claimants()

        duplicates.apply(duplicates.plan(), aliases_path=tmp_path / "a.txt",
                         dry_run=False)

        assert alpha.exists() and beta.exists() and gamma.exists()

    def test_one_survivor_beating_two_rivals_still_resolves(self, workspace):
        """The legitimate star — this is the real Amarna case and must not be
        caught by the same guard."""
        _atom("artifact", "Akhenaten's Boundary Stelae", model="gpt-4o-mini")
        _atom("artifact", "Tell el-Amarna Boundary Stelae", model="gpt-4o-mini")
        _atom("artifact", "Amarna Boundary Stelae", model="gpt-5-nano",
              payload={"name": {"common": "Amarna Boundary Stelae",
                                "otherNames": ["Akhenaten's Boundary Stelae",
                                               "Tell el-Amarna Boundary Stelae"]}})

        plan = duplicates.plan()

        assert len(plan.resolutions) == 2
        assert len(plan.alias_lines()) == 1
        assert plan.undecided == []

    def test_an_isolated_pair_beside_a_conflict_still_resolves(self, workspace):
        self._two_claimants()
        _memphis_pair()

        plan = duplicates.plan()

        assert [r.atom_type for r in plan.resolutions] == ["event"]


class TestDeclaredDistinct:
    """The mirror of the alias file. `aliases.txt` says two spellings are one
    topic; `not_duplicates.txt` says two atoms are different things whatever
    their names suggest — because a shared name is evidence, not proof, and in
    this corpus it is wrong about half the time. Without a durable declaration
    every run resurfaces the same false positives until someone merges Nice
    into Nicaea."""

    def _nicaea(self):
        a = _atom("city", "Nicaea", model="gpt-4o-mini",
                  payload={"name": {"common": "Nicaea", "otherNames": ["Nikaia"]}})
        b = _atom("city", "Nice", model="gpt-5-nano",
                  payload={"name": {"common": "Nice", "otherNames": ["Nikaia"]}})
        return a, b

    def test_a_declared_pair_is_not_retired(self, workspace, tmp_path):
        a, b = self._nicaea()
        claims = duplicates.load_distinct(_distinct(tmp_path, "Nicaea != Nice"))

        plan = duplicates.plan(distinct=claims)
        duplicates.apply(plan, aliases_path=tmp_path / "al.txt", dry_run=False)

        assert plan.resolutions == []
        assert len(plan.skipped) == 1
        assert a.exists() and b.exists()

    def test_without_the_declaration_it_would_be_retired(self, workspace):
        self._nicaea()

        assert len(duplicates.plan(distinct=()).resolutions) == 1

    def test_matching_folds_case_and_punctuation(self, workspace, tmp_path):
        self._nicaea()
        claims = duplicates.load_distinct(_distinct(tmp_path, "  nicaea  !=  NICE  "))

        assert duplicates.plan(distinct=claims).resolutions == []

    def test_a_partial_claim_does_not_suppress_a_wider_component(self, workspace,
                                                                 tmp_path):
        """A claim about two of three atoms says nothing about the third."""
        self._nicaea()
        _atom("city", "Nikaia", model="gpt-5.6-luna",
              payload={"name": {"common": "Nikaia", "otherNames": ["Nikaia"]}})
        claims = duplicates.load_distinct(_distinct(tmp_path, "Nicaea != Nice"))

        assert duplicates.plan(distinct=claims).skipped == []

    def test_a_missing_file_declares_nothing(self, tmp_path):
        assert duplicates.load_distinct(tmp_path / "absent.txt") == ()

    def test_comments_and_blanks_are_ignored(self, tmp_path):
        path = _distinct(tmp_path, "# a comment\n\nNicaea != Nice  # trailing\n")

        assert duplicates.load_distinct(path) == (frozenset({"nicaea", "nice"}),)

    def test_the_shipped_history_file_parses(self):
        from universe_builder.domains.history import build

        claims = duplicates.load_distinct(
            build().aliases_path.parent / "not_duplicates.txt")

        assert len(claims) > 20
        assert frozenset({"nicaea", "nice"}) in claims


class TestKeepNamesTheSurvivor:
    """Nothing in a *filename* ranks two same-generation atoms. The atoms
    themselves are asked instead — the fuller one wins —
    and `--keep` overrides that for a human who has read both."""

    def _tie(self, equal_fill=False):
        """Two same-generation atoms declaring each other.

        `equal_fill` makes them fill the same number of leaves, which is the
        only case left that nothing can rank.
        """
        extra = {"significance": "x"} if equal_fill else {}
        a = _atom("event", "Founding of Memphis", model="gpt-5-nano",
                  payload={"name": {"common": "Founding of Memphis",
                                    "otherNames": []}, **extra})
        b = _atom("event", "Foundation of Memphis", model="gpt-5-nano",
                  payload={"name": {"common": "Foundation of Memphis",
                                    "otherNames": ["Founding of Memphis"]}})
        return a, b

    def test_a_fill_tie_is_undecided_without_keep(self, workspace):
        """Equal generation *and* equal content: nothing separates them."""
        self._tie(equal_fill=True)

        assert duplicates.plan(distinct=()).resolutions == []

    def test_the_fuller_atom_wins_a_same_generation_tie(self, workspace):
        """The tie-break that unblocked dozens of real duplicates, every pair written
        by one model. "Needs a human" was really "needs a tie-break"."""
        older, newer = self._tie()

        plan = duplicates.plan(distinct=(), tiebreak_fill=True,
                               aliases=_alias_group("Founding of Memphis",
                                                    "Foundation of Memphis"))

        assert [r.keep for r in plan.resolutions] == [newer]
        assert "populated fields" in plan.resolutions[0].reason

    def test_keep_overrides_the_fill_tie_break(self, workspace):
        """A human who has read both outranks a field count."""
        older, newer = self._tie()

        plan = duplicates.plan(keep=["founding_of_memphis"], distinct=())

        assert [r.keep for r in plan.resolutions] == [older]
        assert plan.resolutions[0].reason == "named by --keep"

    def test_keep_breaks_the_tie(self, workspace):
        older, newer = self._tie()

        plan = duplicates.plan(keep=["foundation_of_memphis"], distinct=())

        assert [r.keep for r in plan.resolutions] == [newer]
        assert [r.retire for r in plan.resolutions] == [[older]]
        assert plan.resolutions[0].reason == "named by --keep"

    def test_a_pattern_matching_both_is_no_answer(self, workspace):
        """An ambiguous `--keep` must not silently fall through to the fill
        tie-break: the human asked for something, and got it wrong."""
        self._tie()

        plan = duplicates.plan(keep=["memphis"], distinct=())

        assert plan.resolutions == []
        assert len(plan.undecided) == 1

    def test_keep_does_not_licence_a_trespassing_alias_line(self, workspace):
        """`--keep` answers which atom survives; it does not review the alias
        line, so the cross-atom-type check still applies."""
        path = store.dir_for("history") / "history_h1111111111111111_egyptian_history_m.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "people": ["Narmer (Menes)"], "discoveries": [], "cities": [], "artifacts": [],
            "events": ["Unification of Egypt",
                       "Unification of Egypt by King Menes (Narmer)"]}), encoding="utf-8")
        _atom("person", "Narmer (Menes)")
        _atom("event", "Unification of Egypt", model="gpt-5-nano")
        _atom("event", "Unification of Egypt by King Menes (Narmer)", model="gpt-5-nano",
              payload={"name": {"common": "Unification of Egypt by King Menes (Narmer)",
                                "otherNames": ["Unification of Egypt"]}})

        plan = duplicates.plan(keep=["unification_of_egypt_gpt"], distinct=())

        assert plan.resolutions == []
        assert any("narmer" in e["why"] for e in plan.undecided)

    def test_keep_honours_a_line_a_person_already_wrote(self, workspace):
        """The cross-type check is a reason not to *write* a line. With an
        asserted line already covering the component, nothing needs writing
        and `--keep` may retire — the Camp David case, where an artifact atom
        shares the event's name."""
        path = store.dir_for("history") / "history_h1111111111111111_egyptian_history_m.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "people": ["Narmer (Menes)"], "discoveries": [], "cities": [], "artifacts": [],
            "events": ["Unification of Egypt",
                       "Unification of Egypt by King Menes (Narmer)"]}), encoding="utf-8")
        _atom("person", "Narmer (Menes)")
        _atom("event", "Unification of Egypt", model="gpt-5-nano")
        _atom("event", "Unification of Egypt by King Menes (Narmer)", model="gpt-5-nano",
              payload={"name": {"common": "Unification of Egypt by King Menes (Narmer)",
                                "otherNames": ["Unification of Egypt"]}})
        line = _alias_group("Unification of Egypt", "Unification of Egypt by King Menes (Narmer)")

        plan = duplicates.plan(keep=["unification_of_egypt_gpt"], distinct=(), aliases=line)

        assert len(plan.resolutions) == 1
        kept = plan.resolutions[0].keep.name
        assert kept.startswith("event_") and "unification_of_egypt_gpt" in kept


class TestAliasLinesMergeAcrossRuns:
    """Appending blindly is wrong across runs, and silently so.

    A second `--apply` that retires another rival of an already-processed
    survivor writes a second line starting with the same name. Two same-scope
    declarations claiming one key is the ambiguity case, so every matching entry
    is left ungrouped and a topic that resolved yesterday drops out of the work
    list today. Observed exactly that way: `Ottoman conquest of Egypt` gained a
    second line on the second run and `Conquest of Egypt by the Ottoman Empire`
    went missing from coverage.
    """

    def test_a_second_run_merges_rather_than_appending(self, tmp_path):
        path = tmp_path / "aliases.txt"
        duplicates._merge_into_alias_file(path, ["Ottoman conquest of Egypt = Conquest by the Ottomans"])
        duplicates._merge_into_alias_file(path, ["Ottoman conquest of Egypt = Ottoman-Mamluk War"])

        declarations = [line for line in path.read_text().splitlines()
                        if line.strip() and not line.startswith(("#", "["))]

        assert declarations == [
            "Ottoman conquest of Egypt = Conquest by the Ottomans = Ottoman-Mamluk War"]

    def test_the_merged_line_leaves_no_ambiguity(self, tmp_path):
        path = tmp_path / "aliases.txt"
        duplicates._merge_into_alias_file(path, ["A = B"])
        duplicates._merge_into_alias_file(path, ["A = C"])

        groups = topics.load_aliases(path)
        claiming_a = [g for g in groups if "a" in g.keys()]

        assert len(claiming_a) == 1
        assert claiming_a[0].keys() >= {"a", "b", "c"}

    def test_merging_on_a_shared_non_first_name(self, tmp_path):
        """Overlap is by any name, not only the survivor's."""
        path = tmp_path / "aliases.txt"
        duplicates._merge_into_alias_file(path, ["A = B"])
        duplicates._merge_into_alias_file(path, ["C = B"])

        assert len(topics.load_aliases(path)) == 1

    def test_an_unrelated_line_is_appended_not_merged(self, tmp_path):
        path = tmp_path / "aliases.txt"
        duplicates._merge_into_alias_file(path, ["A = B"])
        duplicates._merge_into_alias_file(path, ["X = Y"])

        assert len(topics.load_aliases(path)) == 2

    def test_a_scoped_line_is_never_widened(self, tmp_path):
        """A scoped line is a deliberately narrower claim; folding a global one
        into it would widen it silently."""
        path = tmp_path / "aliases.txt"
        path.write_text("[Greek History]\nThebes (Boeotia) = Thebes\n", encoding="utf-8")

        duplicates._merge_into_alias_file(path, ["Thebes (Luxor) = Thebes"])

        groups = topics.load_aliases(path)
        scoped = [g for g in groups if g.scope]
        assert scoped and scoped[0].names == ("Thebes (Boeotia)", "Thebes")
        assert any(g.scope is None for g in groups)

    def test_existing_comments_and_sections_survive(self, tmp_path):
        path = tmp_path / "aliases.txt"
        path.write_text("# header\nA = B\n[Greek History]\nP = Q\n", encoding="utf-8")

        duplicates._merge_into_alias_file(path, ["A = C"])
        text = path.read_text()

        assert "# header" in text and "[Greek History]" in text and "P = Q" in text


class TestAliasLinesNeverChainDeclarations:
    """The third and outermost place *declarations never chain* must hold.

    `_stars_only` covers retirements and `alias_lines` covers two rivals of one
    survivor within a plan. This covers the file: merging a new line into two
    existing ones joins everything they hold, and nothing declared that.
    """

    def test_the_writer_refuses_to_merge_two_existing_lines(self, tmp_path):
        path = tmp_path / "al.txt"
        duplicates._merge_into_alias_file(path, ["Alpha = Beta"])
        duplicates._merge_into_alias_file(path, ["Gamma = Delta"])

        with pytest.raises(ValueError, match="overlaps 2 existing"):
            duplicates._merge_into_alias_file(path, ["Beta = Gamma"])

    def test_the_refusal_leaves_the_file_untouched(self, tmp_path):
        path = tmp_path / "al.txt"
        duplicates._merge_into_alias_file(path, ["Alpha = Beta"])
        duplicates._merge_into_alias_file(path, ["Gamma = Delta"])
        before = path.read_text()

        with pytest.raises(ValueError):
            duplicates._merge_into_alias_file(path, ["Beta = Gamma"])

        assert path.read_text() == before

    def test_a_plan_whose_line_spans_two_declarations_is_undecided(self, workspace,
                                                                   tmp_path):
        """Caught at plan time, so nothing is retired — the line is the half of
        the repair that stops regeneration."""
        older, newer = _memphis_pair()
        existing = [{"foundation of memphis", "somewhere else"},
                    {"founding of memphis", "another thing"}]

        plan = duplicates._without_alias_chains(
            duplicates.plan(keep=[newer.name], distinct=()), existing)

        assert plan.resolutions == []
        assert any("two existing declarations" in e["why"] for e in plan.undecided)
        assert older.exists()

    def test_two_new_lines_meeting_at_one_existing_line_are_refused(self, workspace):
        """`A = B` in the file, new lines `A = C` and `B = D`: written in
        sequence they chain, because the first write is what the second sees."""
        plan = duplicates.Plan(resolutions=[
            duplicates.Resolution("city", "x", Path("a.json"), [Path("c.json")],
                                  ["A", "C"], "r"),
            duplicates.Resolution("city", "y", Path("b.json"), [Path("d.json")],
                                  ["B", "D"], "r"),
        ])

        out = duplicates._without_alias_chains(plan, [{"a", "b"}])

        assert out.resolutions == []
        assert len(out.undecided) == 2

    def test_one_new_line_touching_one_existing_line_is_fine(self, workspace):
        plan = duplicates.Plan(resolutions=[
            duplicates.Resolution("city", "x", Path("a.json"), [Path("c.json")],
                                  ["A", "C"], "r"),
        ])

        assert len(duplicates._without_alias_chains(plan, [{"a", "b"}]).resolutions) == 1

    def test_the_shipped_alias_file_has_no_chained_group(self):
        """Regression guard on the real data: the largest groups must all be
        hand-written or deliberate merges, not accidental unions."""
        from universe_builder.domains.history import build

        claims = duplicates.existing_alias_claims(build().aliases_path)
        # Two lines that fold to the same names (`Chicanná = Chicanná` and
        # `Chicanna = Chicanná`, kept for byte-for-byte spellings) union into
        # nothing new; only an overlap that grows a group is a chain.
        overlapping = [(i, j) for i, a in enumerate(claims)
                       for j, b in enumerate(claims) if i < j and a & b and a != b]

        assert overlapping == [], f"global alias lines share names: {overlapping}"


class TestParentheticalNamesDoNotBypassTheSafeguards:
    """Every safeguard must fold names exactly as the grouping machinery does.

    `topics.alias_keys` splits a trailing parenthetical — `Foo (Bar)` declares
    `foo bar`, `foo` *and* `bar` — while `topics._key` returns only the whole
    string. Folding with `_key` made the overlap checks blind to the split, so a
    parenthetical name walked straight past them and produced exactly the
    same-scope ambiguity they exist to prevent.
    """

    def test_the_claim_set_includes_the_split_names(self, tmp_path):
        path = tmp_path / "al.txt"
        duplicates._merge_into_alias_file(path, ["Foo (Bar) = Qux"])

        assert duplicates.existing_alias_claims(path) == [
            {"foo bar", "foo", "bar", "qux"}]

    def test_the_safeguard_sees_what_the_grouping_machinery_sees(self, tmp_path):
        """The invariant behind the bug: if these ever diverge, a name can be
        claimed twice without any check noticing."""
        path = tmp_path / "al.txt"
        duplicates._merge_into_alias_file(path, ["Foo (Bar) = Qux"])

        assert duplicates.existing_alias_claims(path) == [
            set(g.keys()) for g in topics.load_aliases(path)][0:1]

    def test_a_line_claiming_a_split_name_merges_instead_of_appending(self, tmp_path):
        path = tmp_path / "al.txt"
        duplicates._merge_into_alias_file(path, ["Foo (Bar) = Qux"])
        duplicates._merge_into_alias_file(path, ["Bar = Baz"])

        declarations = [line for line in path.read_text().splitlines()
                        if line.strip() and not line.startswith(("#", "["))]
        assert len(declarations) == 1
        assert sum(1 for g in topics.load_aliases(path) if "bar" in g.keys()) == 1

    def test_a_parenthetical_spanning_two_declarations_is_refused(self, tmp_path):
        path = tmp_path / "al.txt"
        duplicates._merge_into_alias_file(path, ["Alpha = Bar"])
        duplicates._merge_into_alias_file(path, ["Gamma = Foo"])

        with pytest.raises(ValueError, match="overlaps 2 existing"):
            duplicates._merge_into_alias_file(path, ["Foo (Bar)"])

    def test_the_plan_guard_also_folds_parentheticals(self, workspace):
        """`_without_alias_chains` compares the same way, so a resolution whose
        line carries a parenthetical cannot slip past it either."""
        plan = duplicates.Plan(resolutions=[
            duplicates.Resolution("city", "x", Path("a.json"), [Path("b.json")],
                                  ["Foo (Bar)", "Other"], "r"),
        ])

        out = duplicates._without_alias_chains(plan, [{"alpha", "bar"}, {"gamma", "foo"}])

        assert out.resolutions == []
        assert len(out.undecided) == 1


class TestTheAliasFileInvariantHoldsForAnySequence:
    """A property test, not another scenario.

    Six defects in this area were all the same principle — declarations must
    not chain, and names must fold the way the grouping machinery folds — each
    time in a place I had not enumerated. Scenario tests confirm the cases I
    thought of; this asserts the property itself over sequences I did not.

    The invariant: after any sequence of writes the file accepts, no two global
    declarations share a key, and the loader reports no ambiguity. A write that
    would break it must raise instead.
    """

    #: Names chosen so parentheticals, shared heads and shared inners all occur,
    #: and quoted literals (v0.4.0) beside the same spellings unquoted: a
    #: quoted name claims its whole spelling only.
    VOCAB = ["Alpha", "Beta", "Gamma", "Delta", "Alpha (Beta)", "Gamma (Delta)",
             "Beta (Epsilon)", "Epsilon", "Alpha (Zeta)", "Zeta",
             '"Alpha (Beta)"', '"Gamma (Eta)"', '"Theta (Alpha)"', "Eta"]

    def _lines(self, seed, count):
        import random

        rng = random.Random(seed)
        out = []
        for _ in range(count):
            picked = rng.sample(self.VOCAB, rng.randint(2, 3))
            out.append(" = ".join(picked))
        return out

    def test_no_accepted_sequence_leaves_two_lines_sharing_a_key(self, tmp_path):
        for seed in range(60):
            path = tmp_path / f"al_{seed}.txt"
            for line in self._lines(seed, 5):
                try:
                    duplicates._merge_into_alias_file(path, [line])
                except ValueError:
                    continue  # refused: the safeguard did its job
            claims = duplicates.existing_alias_claims(path)
            clashes = [(i, j) for i, a in enumerate(claims)
                       for j, b in enumerate(claims) if i < j and a & b]
            assert clashes == [], f"seed {seed}: {clashes} in {path.read_text()}"

    def test_no_accepted_sequence_makes_the_loader_report_ambiguity(self, tmp_path, caplog):
        import logging

        for seed in range(60):
            path = tmp_path / f"amb_{seed}.txt"
            for line in self._lines(seed, 5):
                try:
                    duplicates._merge_into_alias_file(path, [line])
                except ValueError:
                    continue
            with caplog.at_level(logging.WARNING, logger="universe_builder.topics"):
                caplog.clear()
                topics.load_aliases(path)
            ambiguous = [r for r in caplog.records
                         if "two groups of the same scope" in r.getMessage()]
            assert not ambiguous, f"seed {seed}: {path.read_text()}"

    def test_the_safeguards_view_always_equals_the_grouping_machinerys(self, tmp_path):
        """The divergence that caused defect six. If these drift apart again a
        name can be claimed twice with no check noticing."""
        for seed in range(60):
            path = tmp_path / f"view_{seed}.txt"
            for line in self._lines(seed, 5):
                try:
                    duplicates._merge_into_alias_file(path, [line])
                except ValueError:
                    continue
            safeguard = duplicates.existing_alias_claims(path)
            grouping = [set(g.keys()) for g in topics.load_aliases(path)]
            assert safeguard == grouping, f"seed {seed}: {safeguard} != {grouping}"

    def test_the_vocabulary_exercises_quoted_lines(self, tmp_path):
        """Guards the property tests above against a vocabulary change that
        quietly stops producing an accepted line with a quoted name."""
        accepted = 0
        for seed in range(60):
            path = tmp_path / f"q_{seed}.txt"
            for line in self._lines(seed, 5):
                try:
                    duplicates._merge_into_alias_file(path, [line])
                except ValueError:
                    continue
            accepted += sum(1 for g in topics.load_aliases(path)
                            if any(topics.is_quoted(n) for n in g.names))
        assert accepted > 10

    def test_a_quoted_name_claims_its_whole_spelling_only(self, tmp_path):
        path = tmp_path / "al.txt"
        duplicates._merge_into_alias_file(path, ['"Theta (Alpha)" = Eta'])

        assert duplicates.existing_alias_claims(path) == [{"theta alpha", "eta"}]
        duplicates._merge_into_alias_file(path, ["Alpha = Beta"])  # no clash: alpha is free
        assert len(topics.load_aliases(path)) == 2


class TestPendingLinesDoNotChainEither:
    """Two rows saying `Apollonia = Apollonia in Illyria` and
    `Apollonia = Apollonia Pontica` are each defensible, and merging them
    declares two Greek colonies — one in Albania, one in Bulgaria — to be one
    city. Arriving in the same batch does not make that safe; the reviewer's
    verdicts are simply inconsistent, which is to be reported rather than
    averaged."""

    def test_two_pending_lines_sharing_a_name_are_refused(self, tmp_path):
        with pytest.raises(ValueError, match="both claim"):
            duplicates._merge_into_alias_file(tmp_path / "al.txt", [
                "Apollonia = Apollonia in Illyria",
                "Apollonia = Apollonia Pontica"])

    def test_the_message_names_both_lines(self, tmp_path):
        with pytest.raises(ValueError, match="Apollonia in Illyria"):
            duplicates._merge_into_alias_file(tmp_path / "al.txt", [
                "Apollonia = Apollonia in Illyria",
                "Apollonia = Apollonia Pontica"])

    def test_nothing_is_written_when_a_batch_is_refused(self, tmp_path):
        path = tmp_path / "al.txt"
        with pytest.raises(ValueError):
            duplicates._merge_into_alias_file(path, ["A = B", "A = C"])

        assert not path.exists()

    def test_unrelated_lines_in_one_batch_are_still_fine(self, tmp_path):
        path = tmp_path / "al.txt"

        duplicates._merge_into_alias_file(path, ["A = B", "X = Y"])

        assert len(topics.load_aliases(path)) == 2


class TestTriage:
    """Judging whether an UNDECIDED component is a duplicate at all.

    `undecided` reads as "which of these do I keep", and `group_synonyms` joins
    on declared names — which is evidence, not proof. The pile once
    held `Abbas I` / `Abbas II` / `Abbas Mirza` as one component and
    `Artaxerxes I` with `Ardashir I`, five centuries apart. Ranking those by
    model family answers a question nobody should ask, so triage comes first.
    """

    def _judgement(self, same, confident=True, reason="r"):
        from universe_builder.collisions import Judgement

        return Judgement(same_subject=same, confident=confident, reason=reason)

    def _patch(self, monkeypatch, judgement):
        from universe_builder.providers import run as provider_run

        class Reply:
            parsed = judgement

        monkeypatch.setattr(provider_run, "complete", lambda request, **kw: Reply())

    def test_a_confident_difference_is_reported_as_distinct(self, workspace, monkeypatch):
        from universe_builder import duplicates

        plan = duplicates.Plan()
        plan.undecided.append({"atom_type": "person", "name": "abbas",
                               "files": ["person_h1_abbas_i_m.json",
                                         "person_h2_abbas_ii_m.json"],
                               "why": "tie"})
        self._patch(monkeypatch, self._judgement(False))

        records = duplicates.triage(plan, "m")

        assert [r["verdict"] for r in records] == ["different"]

    def test_an_unsure_model_does_not_produce_a_distinct_line(self, workspace, monkeypatch, tmp_path):
        """Abstention must leave the pair in the report, not silently declare
        two atoms distinct — that would hide a real duplicate for good."""
        from universe_builder import duplicates

        plan = duplicates.Plan()
        plan.undecided.append({"atom_type": "person", "name": "x",
                               "files": ["person_h1_a_m.json", "person_h2_b_m.json"],
                               "why": "tie"})
        self._patch(monkeypatch, self._judgement(False, confident=False))

        records = duplicates.triage(plan, "m")
        outcome = duplicates.write_distinct_lines(records, tmp_path / "nd.txt",
                                                  dry_run=True)

        assert records[0]["verdict"] == "unsure"
        assert outcome["new_lines"] == []

    def test_a_same_verdict_writes_nothing(self, workspace, monkeypatch, tmp_path):
        """Only negatives are written. `same` means a real duplicate, and
        retiring one of its atoms is a separate destructive decision."""
        from universe_builder import duplicates

        plan = duplicates.Plan()
        plan.undecided.append({"atom_type": "person", "name": "x",
                               "files": ["person_h1_a_m.json", "person_h2_b_m.json"],
                               "why": "tie"})
        self._patch(monkeypatch, self._judgement(True))

        records = duplicates.triage(plan, "m")
        outcome = duplicates.write_distinct_lines(records, tmp_path / "nd.txt",
                                                  dry_run=True)

        assert records[0]["verdict"] == "same"
        assert outcome["new_lines"] == []

    def test_a_three_atom_component_is_judged_pairwise(self, workspace, monkeypatch):
        """`Abbas I`/`Abbas II`/`Abbas Mirza` is three pairs, and
        `not_duplicates.txt` records claims pairwise."""
        from universe_builder import duplicates

        plan = duplicates.Plan()
        plan.undecided.append({"atom_type": "person", "name": "abbas",
                               "files": ["person_h1_a_m.json", "person_h2_b_m.json",
                                         "person_h3_c_m.json"],
                               "why": "tie"})
        self._patch(monkeypatch, self._judgement(False))

        assert len(duplicates.triage(plan, "m")) == 3

    def test_a_failed_call_is_unsure_not_different(self, workspace, monkeypatch):
        """A transport error must never read as "these are distinct"."""
        from universe_builder import duplicates
        from universe_builder.providers import run as provider_run

        def boom(request, **kw):
            raise RuntimeError("down")

        monkeypatch.setattr(provider_run, "complete", boom)
        plan = duplicates.Plan()
        plan.undecided.append({"atom_type": "person", "name": "x",
                               "files": ["person_h1_a_m.json", "person_h2_b_m.json"],
                               "why": "tie"})

        assert duplicates.triage(plan, "m")[0]["verdict"] == "unsure"

    def test_writing_is_dry_by_default(self, tmp_path):
        from universe_builder import duplicates

        path = tmp_path / "nd.txt"
        records = [{"verdict": "different", "a": "Abbas I", "b": "Abbas II"}]

        duplicates.write_distinct_lines(records, path, dry_run=True)

        assert not path.exists()

    def test_applying_appends_and_does_not_truncate(self, tmp_path):
        """The file is hand-curated and 28 lines deep; a writer that opened it
        for writing rather than appending would erase the lot."""
        from universe_builder import duplicates

        path = tmp_path / "nd.txt"
        path.write_text("# header\nThutmose I != Thutmose IV\n", encoding="utf-8")
        records = [{"verdict": "different", "a": "Abbas I", "b": "Abbas II"}]

        duplicates.write_distinct_lines(records, path, dry_run=False)
        text = path.read_text(encoding="utf-8")

        assert "Thutmose I != Thutmose IV" in text
        assert "Abbas I != Abbas II" in text

    def test_an_already_declared_pair_is_not_written_twice(self, tmp_path):
        from universe_builder import duplicates

        path = tmp_path / "nd.txt"
        path.write_text("Abbas I != Abbas II\n", encoding="utf-8")
        records = [{"verdict": "different", "a": "abbas ii", "b": "ABBAS I"}]

        outcome = duplicates.write_distinct_lines(records, path, dry_run=True)

        assert outcome["new_lines"] == []
        assert outcome["already_present"] == 1

    def test_a_distinct_line_that_contradicts_an_alias_is_refused(
            self, workspace, monkeypatch, tmp_path):
        """`aliases.txt` and `not_duplicates.txt` are mirrors, and the corpus
        has held both claims about one pair before — `lint --check
        alias-contradiction` exists for it.

        Observed live: the triage judged `Book of the Dead` and `Papyrus
        of Ani` distinct, correctly (a funerary corpus against one manuscript
        of it), while `aliases.txt` already declared them one topic.
        Writing it would have made the two files disagree, and `topics` reads
        both. Refused rather than resolved: which file is wrong is a judgement,
        and the alias may be load-bearing for atoms already generated under it.
        """
        from universe_builder import duplicates

        aliases = tmp_path / "aliases.txt"
        aliases.write_text("Book of the Dead = Papyrus of Ani\n", encoding="utf-8")
        monkeypatch.setattr(duplicates, "_pack_aliases_path", lambda: aliases)
        records = [{"verdict": "different", "a": "Book of the Dead",
                    "b": "Papyrus of Ani"},
                   {"verdict": "different", "a": "Abbas I", "b": "Abbas II"}]

        outcome = duplicates.write_distinct_lines(records, tmp_path / "nd.txt",
                                                  dry_run=True)

        assert outcome["new_lines"] == ["Abbas I != Abbas II"]
        assert outcome["contradicts_alias"] == ["Book of the Dead != Papyrus of Ani"]

    def test_a_multi_name_alias_group_blocks_every_pair_in_it(
            self, workspace, monkeypatch, tmp_path):
        """An alias line joins N names, which is N-choose-2 claims of identity.
        Checking only the first two would let the rest through."""
        from universe_builder import duplicates

        aliases = tmp_path / "aliases.txt"
        aliases.write_text("A = B = C\n", encoding="utf-8")
        monkeypatch.setattr(duplicates, "_pack_aliases_path", lambda: aliases)
        records = [{"verdict": "different", "a": "B", "b": "C"}]

        outcome = duplicates.write_distinct_lines(records, tmp_path / "nd.txt",
                                                  dry_run=True)

        assert outcome["new_lines"] == []
        assert outcome["contradicts_alias"] == ["B != C"]


class TestTheFillTiebreakIsOptInForAReason:
    """A near-miss, kept as a test so it cannot recur.

    Breaking a same-generation tie by field count sounds obviously right, and
    is — *provided the component is genuinely one subject*. Nothing in `plan`
    establishes that: `group_synonyms` joins on declared names, which is
    evidence rather than proof. Enabled as the default it made dozens of
    components resolvable, among them `Marcus Aurelius` retiring `Caracalla` and
    `Elagabalus`, and `Constantius Chlorus` retiring `Constantius II` and
    `Constantius III`. Six emperors, three files, and `undecided` had been the
    only thing standing in the way.
    """

    def _three_emperors(self):
        """One component, three distinct people, as the corpus actually had
        it: each declares the others among its `otherNames`."""
        for common, others, extra in [
            ("Marcus Aurelius", ["Caracalla", "Elagabalus"], {"significance": "x" * 3}),
            ("Caracalla", ["Marcus Aurelius"], {}),
            ("Elagabalus", ["Marcus Aurelius"], {}),
        ]:
            _atom("person", common, model="gpt-5.6-luna",
                  payload={"name": {"common": common, "otherNames": others}, **extra})

    def test_by_default_they_are_left_alone(self, workspace):
        self._three_emperors()

        plan = duplicates.plan(distinct=())

        assert plan.resolutions == [], (
            "the default must not rank a component nothing has confirmed is "
            "one subject")
        assert plan.undecided

    def test_opting_in_is_not_enough_without_an_alias(self, workspace):
        """The flag alone does nothing, and that is the whole correction.

        Field count answers "which says more"; it never answers "are these one
        subject". Only an alias line asserts identity, so the flag enables the
        *ranking* and the alias file is the *gate*. These three emperors share
        `Marcus Aurelius Antoninus` because all three genuinely bore it, and no
        alias line says they are one man.
        """
        self._three_emperors()

        plan = duplicates.plan(distinct=(), tiebreak_fill=True)

        assert plan.resolutions == []
        assert plan.undecided

    def test_an_alias_declaring_them_one_subject_is_what_unlocks_it(self, workspace):
        """And when something *has* asserted identity, ranking is allowed."""
        self._three_emperors()

        plan = duplicates.plan(
            distinct=(), tiebreak_fill=True,
            aliases=_alias_group("Marcus Aurelius", "Caracalla", "Elagabalus"))

        assert plan.resolutions, "an alias-covered component should rank"

    def test_declaring_them_distinct_protects_them_even_when_opted_in(self, workspace):
        """`not_duplicates.txt` outranks the tie-break, which is the layer a
        `--via model` triage actually writes to."""
        self._three_emperors()
        # One claim covering the whole component. `_declared_distinct` asks
        # whether *every* atom is named by a single claim, so three pairwise
        # claims would suppress nothing — which is why `write_distinct_lines`
        # emits `A != B != C` for a component every pair of which differs.
        claims = ({topics._key("Marcus Aurelius"), topics._key("Caracalla"),
                   topics._key("Elagabalus")},)

        plan = duplicates.plan(distinct=claims, tiebreak_fill=True)

        assert plan.resolutions == []
        assert plan.skipped


class TestMultiWayDistinctLines:
    # `workspace`: these read the not-duplicates file in force, and without it
    # that is the bundled history file, which already holds the Abbas line —
    # they passed only while `!=` dedupe mis-keyed multi-way lines.
    def test_a_fully_distinct_three_atom_component_is_one_line(self, workspace):
        """`_declared_distinct` needs one claim covering the whole component,
        so three pairwise lines suppress nothing. Writing pairwise lines for a
        whole review left the three-way components exactly where they were."""
        records = [
            {"atom_type": "person", "component": "abbas", "verdict": "different",
             "a": "Abbas I", "b": "Abbas II"},
            {"atom_type": "person", "component": "abbas", "verdict": "different",
             "a": "Abbas I", "b": "Abbas Mirza"},
            {"atom_type": "person", "component": "abbas", "verdict": "different",
             "a": "Abbas II", "b": "Abbas Mirza"},
        ]

        out = duplicates.write_distinct_lines(records, dry_run=True)

        assert out["new_lines"] == ["Abbas I != Abbas II != Abbas Mirza"]

    def test_a_partly_undecided_component_stays_pairwise(self, workspace):
        """If one pair is `same` or `unsure`, the component is not known to be
        wholly distinct, and claiming so would suppress a real duplicate."""
        records = [
            {"atom_type": "person", "component": "x", "verdict": "different",
             "a": "A", "b": "B"},
            {"atom_type": "person", "component": "x", "verdict": "unsure",
             "a": "A", "b": "C"},
            {"atom_type": "person", "component": "x", "verdict": "different",
             "a": "B", "b": "C"},
        ]

        out = duplicates.write_distinct_lines(records, dry_run=True)

        assert out["new_lines"] == ["A != B", "B != C"]

    def test_a_two_atom_component_is_written_pairwise(self, workspace):
        records = [{"atom_type": "person", "component": "x", "verdict": "different",
                    "a": "A", "b": "B"}]

        assert duplicates.write_distinct_lines(records, dry_run=True)["new_lines"] \
            == ["A != B"]


class TestTheIdentityGateRefusesWeakEvidence:
    """What `_declared_same` will not accept, each found by review."""

    def _pair(self):
        _atom("event", "Founding of Memphis", model="gpt-5-nano")
        _atom("event", "Foundation of Memphis", model="gpt-5-nano",
              payload={"name": {"common": "Foundation of Memphis",
                                "otherNames": ["Founding of Memphis"]}})

    def _plan(self, aliases):
        return duplicates.plan(distinct=(), tiebreak_fill=True, aliases=aliases)

    def test_an_asserted_global_line_unlocks_the_tiebreak(self, workspace):
        self._pair()

        assert self._plan(_alias_group("Founding of Memphis",
                                       "Foundation of Memphis")).resolutions

    def test_a_derived_line_does_not(self, workspace):
        """`aliases.txt` is written by `universe_builder duplicates --apply` as well as by
        people. Reading its own output back as proof of identity let the
        command cite itself to license the next retirement."""
        self._pair()

        plan = self._plan(_alias_group("Founding of Memphis",
                                       "Foundation of Memphis",
                                       provenance="derived"))

        assert plan.resolutions == []
        assert plan.undecided

    def test_a_scoped_line_does_not(self, workspace):
        """`scope` is the field that says an identity is not universal, and it
        was ignored entirely: `[Greek History] Thebes (Boeotia) = Thebes`
        admitted the Egyptian `Thebes (Luxor)` from any context."""
        self._pair()

        plan = self._plan(_alias_group("Founding of Memphis",
                                       "Foundation of Memphis",
                                       scope="Egyptian History"))

        assert plan.resolutions == []

    def test_a_component_whose_names_collapse_is_refused(self, workspace):
        """Two atoms answering to one common name shrink the name set, so a
        two-name line could cover three atoms."""
        _atom("event", "Founding of Memphis", model="gpt-5-nano")
        _atom("event", "Foundation of Memphis", model="gpt-5-nano",
              payload={"name": {"common": "Founding of Memphis",
                                "otherNames": ["Founding of Memphis"]}})

        plan = self._plan(_alias_group("Founding of Memphis",
                                       "Foundation of Memphis"))

        assert plan.resolutions == []


class TestTheFillTiebreakIsOnTheCommandLine:
    """`--tiebreak-fill` reaches `plan()`; without it the default stays off.
    The gate itself (an asserted, unscoped alias line covering the whole
    component) is `plan`'s and is tested above; this pins the wiring, which
    was the missing piece when 45 adjudicated pairs sat undecided."""

    def test_the_flag_is_passed_through(self, workspace, monkeypatch, capsys):
        import argparse

        from universe_builder import cli, duplicates as dup

        seen = {}

        def fake_plan(**kwargs):
            seen.update(kwargs)
            return dup.Plan()

        monkeypatch.setattr(dup, "plan", fake_plan)
        args = argparse.Namespace(excluded=None, kept=None, tiebreak_fill=True, apply=False,
                                  via=None, triage_model=None, limit=None, out=None, config=None)
        assert cli.cmd_duplicates(args) == 0
        assert seen["tiebreak_fill"] is True

    def test_the_parser_defaults_it_off(self):
        from universe_builder import cli

        args = cli.build_parser().parse_args(["duplicates"])
        assert args.tiebreak_fill is False
        assert cli.build_parser().parse_args(["duplicates", "--tiebreak-fill"]).tiebreak_fill is True


class TestFindingsOf20260925:
    """Three corners the name-key folding exposed on the corpus."""

    def test_two_atoms_sharing_a_common_name_are_not_declared_distinct_by_a_line_naming_a_third(self, workspace):
        a = _atom("city", "Catania", payload={"name": {"common": "Catania", "otherNames": ["Catana"]}})
        b = _atom("city", "Catane (Catania)", payload={"name": {"common": "Catania", "otherNames": ["Katane"]}})
        claims = (frozenset({"kos", "catania"}),)
        assert duplicates._declared_distinct([a, b], "city", claims) is False

    def test_two_atoms_sharing_a_common_name_are_distinct_by_their_generation_spellings(self, workspace):
        """The Abydos case: the Egyptian and the Hellespontine city
        are both commonly `Abydos`, stored under `Abydos` and `Abydos
        (Hellespont)`. A line naming those two spellings tells them apart; a
        line naming only the shared name, or a third city, does not."""
        # Two seeds, as in the corpus: one civilisation lists each spelling.
        # In one seed the parenthetical's head would attach the bare name to
        # its own group and both spellings would resolve to one file.
        empty = {"people": [], "discoveries": [], "artifacts": [], "events": []}
        for stem, city in (("h1111111111111111_egyptian", "Abydos"),
                           ("h2222222222222222_byzantine", "Abydos (Hellespont)")):
            path = store.dir_for("history") / f"history_{stem}_history_m.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**empty, "cities": [city]}), encoding="utf-8")
        a = _atom("city", "Abydos", payload={"name": {"common": "Abydos", "otherNames": ["Abdju"]}})
        b = _atom("city", "Abydos (Hellespont)", payload={"name": {"common": "Abydos", "otherNames": ["Abydus"]}})

        exact = (frozenset({topics._key("Abydos"), topics._key("Abydos (Hellespont)")}),)
        assert duplicates._declared_distinct([a, b], "city", exact) is True
        assert duplicates._declared_distinct([a, b], "city", (frozenset({"kos", "abydos"}),)) is False

    def test_the_generation_spelling_may_come_from_an_alias_line_rather_than_a_seed(self, workspace):
        """The corpus case: the Byzantine seed lists bare `Abydos`, and the
        scoped line `Abydos (Hellespont) = Abydos` is what made the file — no
        seed spells the survivor, the line does."""
        empty = {"people": [], "discoveries": [], "artifacts": [], "events": []}
        for stem, city in (("h1111111111111111_egyptian", "Abydos"),
                           ("h2222222222222222_byzantine", "Abydos")):
            path = store.dir_for("history") / f"history_{stem}_history_m.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({**empty, "cities": [city]}), encoding="utf-8")
        settings.ALIASES_LIST.write_text(
            "[Byzantine History]\nAbydos (Hellespont) = Abydos\n", encoding="utf-8")
        a = _atom("city", "Abydos", payload={"name": {"common": "Abydos", "otherNames": ["Abdju"]}})
        b = _atom("city", "Abydos (Hellespont)", payload={"name": {"common": "Abydos", "otherNames": ["Abydus"]}})

        exact = (frozenset({topics._key("Abydos"), topics._key("Abydos (Hellespont)")}),)
        assert duplicates._declared_distinct([a, b], "city", exact) is True

    def test_a_covering_asserted_line_means_no_alias_line_is_written(self, workspace):
        path = store.dir_for("history") / "history_h1111111111111111_egyptian_history_m.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"people": [], "discoveries": [], "cities": [],
                                    "artifacts": ["Camp David Accords"],
                                    "events": ["Camp David Accords", "The Camp David Accords"]}), encoding="utf-8")
        _atom("artifact", "Camp David Accords")
        _atom("event", "Camp David Accords", model="gpt-5-nano")
        _atom("event", "The Camp David Accords", model="gpt-5-nano",
              payload={"name": {"common": "The Camp David Accords", "otherNames": ["Camp David Accords"]}})
        line = _alias_group("Camp David Accords", "The Camp David Accords")

        survivor = next(p.name for p in store.iter_atoms("event")
                        if p.name.endswith("_camp_david_accords_gpt-5-nano.json"))
        plan = duplicates.plan(keep=[survivor], distinct=(), aliases=line)

        assert len(plan.resolutions) == 1
        assert plan.resolutions[0].spellings == []
        assert plan.alias_lines() == []

    def test_a_covering_line_whose_earlier_name_shadows_the_survivor_blocks_the_retirement(self, workspace):
        """Found by review. With a covering line the retirement writes nothing,
        and the work list tries a line's names in order, skipping any whose
        folded key it has seen — so a retired spelling that folds like the
        survivor's and comes first hides the survivor, and the retired atom
        regenerates. Distinct keys in either order are fine (the Memphis tests
        above); only a same-key shadow is refused."""
        _atom("person", "Al-Mu'izz li-Din Allah", model="gpt-5-nano",
              payload={"name": {"common": "Al-Mu'izz li-Din Allah", "otherNames": ["Al-Muʿizz li-Dīn Allāh"]}})
        retired = _atom("person", "Al-Muʿizz li-Dīn Allāh", model="gpt-5-nano",
                        payload={"name": {"common": "Al-Muʿizz li-Dīn Allāh", "otherNames": []}})
        shadowing = _alias_group("Al-Muʿizz li-Dīn Allāh", "Al-Mu'izz li-Din Allah")
        survivor = next(p.name for p in store.iter_atoms("person") if p != retired)

        plan = duplicates.plan(keep=[survivor], distinct=(), aliases=shadowing)

        assert plan.resolutions == []
        assert len(plan.undecided) == 1 and "resolves to the survivor" in plan.undecided[0]["why"]

        survivor_first = _alias_group("Al-Mu'izz li-Din Allah", "Al-Muʿizz li-Dīn Allāh")
        assert len(duplicates.plan(keep=[survivor], distinct=(), aliases=survivor_first).resolutions) == 1

    def test_the_lookup_name_is_the_generation_spelling_not_the_common_name(self, workspace):
        """The survivor's file id comes from the spelling it was generated under.
        A covering line that carries only its `name.common` cannot reach it,
        because the work list hashes the names it tries."""
        survivor = _atom("city", "Catane (Catania)", model="gpt-5-nano",
                         payload={"name": {"common": "Catania", "otherNames": ["Katane"]}})
        _atom("city", "Catania", model="gpt-4o-mini",
              payload={"name": {"common": "Catania", "otherNames": ["Catana"]}})
        # Both atoms are called Catania, so the pair is a tie the alias line
        # must settle; `--keep` names the survivor either way.
        only_common = _alias_group("Catania", "Catana")
        plan = duplicates.plan(keep=[survivor.name], distinct=(), aliases=only_common)
        assert plan.resolutions == [] and "resolves to the survivor" in plan.undecided[0]["why"]

        with_generation_spelling = _alias_group("Catane (Catania)", "Catania", "Catana")
        plan = duplicates.plan(keep=[survivor.name], distinct=(), aliases=with_generation_spelling)
        assert len(plan.resolutions) == 1 and plan.resolutions[0].keep == survivor

    def test_a_one_letter_declared_name_groups_nothing(self, workspace):
        _atom("artifact", "Codex Alexandrinus",
              payload={"name": {"common": "Codex Alexandrinus", "otherNames": ["A"]}})
        _atom("artifact", "Palace of Ardashir",
              payload={"name": {"common": "Palace of Ardashir", "otherNames": ["Ā"]}})
        plan = duplicates.plan(distinct=())
        assert plan.is_clean()


class TestTriageIsJournalled:
    """`duplicates --via model` is paid, one call per pair, and once
    journalled nothing: `triage` called `provider_run.complete`
    directly and the command opened no run, so `costs` never saw it."""

    PLAN_ENTRY = {"atom_type": "person", "name": "abbas",
                  "files": ["person_h1_abbas_i_m.json", "person_h2_abbas_ii_m.json"],
                  "why": "tie"}

    def _patch(self, monkeypatch):
        from universe_builder.collisions import Judgement
        from universe_builder.providers import run as provider_run
        from universe_builder.providers.types import LLMResponse, Usage

        reply = LLMResponse('{"same_subject": false}',
                            Judgement(same_subject=False, confident=True, reason="r"),
                            "gpt-5-nano", Usage(input_tokens=80, output_tokens=20))
        monkeypatch.setattr(provider_run, "complete", lambda request, **kw: reply)

    def test_each_pair_is_recorded_with_its_usage(self, workspace, monkeypatch):
        from universe_builder import journal

        self._patch(monkeypatch)
        plan = duplicates.Plan(undecided=[dict(self.PLAN_ENTRY)])
        run = journal.start_run(config={"command": "duplicates"})
        try:
            duplicates.triage(plan, "gpt-5-nano")
        finally:
            journal.end_run()

        (entry,) = run.entries()
        assert entry["operation"] == "duplicate-triage"
        assert entry["atom_type"] == "person"
        assert (entry["input_tokens"], entry["output_tokens"]) == (80, 20)
        assert run.needs_fix() == []

    def test_the_command_opens_a_run_costs_can_read(self, workspace, monkeypatch, capsys):
        import argparse

        from universe_builder import cli, costs, journal

        self._patch(monkeypatch)
        monkeypatch.setattr(duplicates, "plan",
                            lambda **kw: duplicates.Plan(undecided=[dict(self.PLAN_ENTRY)]))
        args = argparse.Namespace(excluded=None, kept=None, tiebreak_fill=False, apply=False,
                                  via="model", triage_model="gpt-5-nano", limit=None,
                                  out=None, config=None)
        try:
            cli.cmd_duplicates(args)
        finally:
            journal.end_run()

        report = costs.build("phase")
        assert [r.key for r in report.rows] == ["duplicates"]
        assert report.total.input_tokens == 80


class TestTheExitStatusGatesOnRemainingWork:
    """`duplicates` exits non-zero while duplicates remain, so a script can
    gate on a clean corpus. Pairs declared distinct (`not_duplicates.txt`) or
    excluded with `--except` are not duplicates that remain — yet a plan
    holding only those exited 1, which on the history corpus (nothing
    resolvable or undecided, only pairs declared distinct) meant the gate
    could never pass."""

    def _run(self, monkeypatch, plan, apply=False):
        import argparse

        from universe_builder import cli

        monkeypatch.setattr(duplicates, "plan", lambda **kw: plan)
        args = argparse.Namespace(excluded=None, kept=None, tiebreak_fill=False, apply=apply,
                                  via=None, triage_model=None, limit=None, out=None, config=None)
        return cli.cmd_duplicates(args)

    def _distinct(self):
        return {"atom_type": "person", "name": "abbas",
                "files": ["person_h1_abbas_i_m.json", "person_h2_abbas_ii_m.json"]}

    def test_only_declared_distinct_pairs_is_clean(self, workspace, monkeypatch, capsys):
        assert self._run(monkeypatch, duplicates.Plan(skipped=[self._distinct()])) == 0
        assert "DECLARED DISTINCT" in capsys.readouterr().out, "they are still reported"

    def test_undecided_work_still_fails_the_gate(self, workspace, monkeypatch):
        entry = dict(self._distinct(), why="tie")
        plan = duplicates.Plan(undecided=[entry], skipped=[self._distinct()])
        assert self._run(monkeypatch, plan) == 1
        assert self._run(monkeypatch, plan, apply=True) == 1


class TestAMergedLineKeepsAParentheticalName:
    """`Fayoum = Arsinoe (Crocodilopolis)` merged into `Fayoum = Crocodilopolis`
    returned as written and left the file unchanged: the merge
    skipped any name one of whose parenthetical parts the line already held,
    so the retired spelling was never declared and came back as create work."""

    def test_the_retired_spelling_is_added(self, tmp_path):
        from pathlib import Path
        from universe_builder import duplicates as dup

        path = Path(tmp_path) / "aliases.txt"
        path.write_text("Fayoum = Crocodilopolis\n", encoding="utf-8")

        dup._merge_into_alias_file(path, ["Fayoum = Arsinoe (Crocodilopolis)"])

        assert path.read_text(encoding="utf-8").splitlines()[0] == \
            "Fayoum = Crocodilopolis = Arsinoe (Crocodilopolis)"

    def test_a_name_already_on_the_line_is_not_repeated(self, tmp_path):
        from pathlib import Path
        from universe_builder import duplicates as dup

        path = Path(tmp_path) / "aliases.txt"
        path.write_text("Fayoum = Crocodilopolis\n", encoding="utf-8")

        dup._merge_into_alias_file(path, ["Fayoum = crocodilopolis"])

        assert path.read_text(encoding="utf-8").splitlines()[0] == "Fayoum = Crocodilopolis"


class TestTriageShowsWhatEachAtomSays:
    """Names alone could not tell a subject from one it contains (v0.4.0): the
    triage request carries other names, dates, places, a summary
    (`DomainPack.summary_paths`) and the corroboration line."""

    def _person(self, subject, year, era, city, country, summary, other=()):
        return _atom("person", subject, payload={
            "name": {"common": subject, "otherNames": list(other)},
            "birth": {"date": {"year": year, "era": era},
                      "location": {"city": city, "country": country,
                                   "latitude": 25.7, "longitude": 32.6}},
            "shortSummary": summary,
        })

    def _triage(self, fake_provider, a, b):
        from universe_builder import journal

        fake_provider.script('{"same_subject": false, "confident": true, "reason": "r"}')
        plan = duplicates.Plan(undecided=[{"atom_type": "person", "name": "x",
                                           "files": [a.name, b.name], "why": "tie"}])
        journal.start_run(config={"command": "duplicates"})
        try:
            records = duplicates.triage(plan, "gpt-5-nano")
        finally:
            journal.end_run()
        [request] = fake_provider.requests
        return records, request

    def test_dates_places_names_and_summary_reach_the_model(self, workspace, fake_provider,
                                                            pack_overrides):
        pack_overrides(summary_paths={"person": ("shortSummary",)})
        a = self._person("Amenhotep III", 1388, "BCE", "Thebes", "Egypt",
                         "Ninth pharaoh of the Eighteenth Dynasty. " * 20, other=("Amenophis III",))
        b = self._person("Amenhotep son of Hapu", 1435, "BCE", "Athribis", "Egypt",
                         "Architect and scribe under Amenhotep III.")

        records, request = self._triage(fake_provider, a, b)

        user = request.user
        assert "Record A: Amenhotep III\n" in user
        assert "  other names: Amenophis III\n" in user
        assert "  birth.date: 1388 BCE\n" in user and "  birth.date: 1435 BCE\n" in user
        assert "  birth.location: Thebes, Egypt\n" in user
        assert "  birth.location: Athribis, Egypt\n" in user
        assert "  summary: Architect and scribe under Amenhotep III.\n" in user
        summary_line = next(line for line in user.splitlines()
                            if line.startswith("  summary: Ninth"))
        assert len(summary_line) - len("  summary: ") == duplicates.TRIAGE_SUMMARY_CHARS
        assert summary_line.endswith("…")
        assert "Place and date compared: agrees: birth.location~0km, birth.date~47y" in user
        assert "contains the other" in request.system and "DIFFERENT" in request.system
        assert records[0]["verdict"] == "different"

    def test_no_summary_paths_is_no_summary(self, workspace, fake_provider):
        a = self._person("A", 1, "BCE", "X", "Y", "prose about A")
        b = self._person("B", 1, "BCE", "X", "Y", "prose about B")

        _, request = self._triage(fake_provider, a, b)

        assert "summary:" not in request.user
        assert "birth.date: 1 BCE" in request.user

    def test_an_era_less_year_says_so(self):
        assert duplicates._when({"year": 1250}) == "1250 (era unknown)"
        assert duplicates._when({"year": None, "era": "CE"}) is None

    def test_a_place_is_its_text_fields_but_certainty(self):
        assert duplicates._where({"landmark": "Karnak", "city": "Thebes", "state": None,
                                  "country": "Egypt", "latitude": 25.7, "certainty": "high",
                                  "sources": ["x"]}) == "Karnak, Thebes, Egypt"
        assert duplicates._where({"certainty": "low"}) is None


class TestAMergeNeverCrossesARefusal:
    """The ruled `Irbid = Irbid (Arbela)` joined `Erbil = Arbela` through the
    parenthetical `arbela`, against `Irbid != Erbil`, and the shipped Irbid
    resolved to Erbil; the same step put
    `Philadelphia (Amman)` on the Alaşehir line through the head `philadelphia`."""

    def _files(self, tmp_path, aliases, refusals):
        path = tmp_path / "aliases.txt"
        path.write_text(aliases + "\n", encoding="utf-8")
        _distinct(tmp_path, refusals)
        return path

    def test_a_parenthetical_overlap_does_not_cross_a_refusal(self, tmp_path):
        path = self._files(tmp_path, "Erbil = Arbela", "Irbid != Erbil")
        before = path.read_text()

        with pytest.raises(ValueError, match="Irbid != Erbil"):
            duplicates._merge_into_alias_file(path, ["Irbid = Irbid (Arbela)"])

        assert path.read_text() == before

    def test_a_head_overlap_does_not_cross_a_refusal(self, tmp_path):
        path = self._files(tmp_path, "Philadelphia (Alaşehir) = Alaşehir",
                           "Philadelphia (Amman) != Philadelphia (Alaşehir)")

        with pytest.raises(ValueError, match="not_duplicates"):
            duplicates._merge_into_alias_file(
                path, ["Philadelphia (Amman) = Rabbath Ammon (Philadelphia)"])

    def test_a_new_line_joining_refused_names_is_refused(self, tmp_path):
        path = self._files(tmp_path, "Thebes (Luxor) = Waset", "Irbid != Erbil")

        with pytest.raises(ValueError, match="declares distinct"):
            duplicates._merge_into_alias_file(path, ["Irbid = Erbil"])

    def test_a_refusal_between_parts_does_not_block_a_merge(self, tmp_path):
        """Whole spellings only: `Thebes (Luxor) != Thebes (Boeotia)` must not
        forbid joining the bare `Thebes` to `Luxor`."""
        path = self._files(tmp_path, "Thebes (Luxor) = Thebes",
                           "Thebes (Luxor) != Thebes (Boeotia)")

        duplicates._merge_into_alias_file(path, ["Thebes (Luxor) = Luxor"])

        assert path.read_text().splitlines()[0] == "Thebes (Luxor) = Thebes = Luxor"

    def test_apply_review_refuses_through_the_cli(self, tmp_path, capsys):
        from universe_builder import cli

        aliases = self._files(tmp_path, "Erbil = Arbela", "Irbid != Erbil")
        review = tmp_path / "r.tsv"
        review.write_text("same\tcity\tcity_x.json\tIrbid (Arbela)\tIrbid\n", encoding="utf-8")
        settings.ALIASES_LIST = str(aliases)
        try:
            code = cli.main(["lint", "--apply-review", str(review), "--apply"])
        finally:
            settings.ALIASES_LIST = None

        assert code == 2
        assert "Irbid != Erbil" in capsys.readouterr().out
        assert aliases.read_text() == "Erbil = Arbela\n"
