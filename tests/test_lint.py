"""Mechanical corpus checks — no model, no network, no writes."""

from __future__ import annotations

import json

from universe_builder import lint, store

MODEL = "gpt-5-nano"


def _atom(atom_type, subject, payload=None, model=MODEL):
    path = store.path_for(atom_type, None, subject, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload or {"name": {"common": subject}}),
                    encoding="utf-8")
    return path


class TestOverbroadAliases:
    """An atom claiming a name another atom calls itself asserts the two are
    one thing. When they are not, the name becomes ambiguous and stops
    resolving for everybody — `Karnak` claimed `Thebes`, which is true as
    geography and false as identity."""

    def test_a_claim_on_another_atoms_common_name_is_flagged(self, workspace):
        _atom("city", "Thebes (Boeotia)", payload={"name": {"common": "Thebes"}})
        _atom("city", "Karnak", payload={"name": {"common": "Karnak",
                                                  "otherNames": ["Thebes"]}})

        found = lint.overbroad_aliases("city")

        assert [f.subject for f in found] == ["Karnak"]
        assert "Thebes" in found[0].detail

    def test_an_alias_nobody_else_claims_is_fine(self, workspace):
        _atom("city", "Karnak", payload={"name": {"common": "Karnak",
                                                  "otherNames": ["Ipet-isut"]}})

        assert lint.overbroad_aliases("city") == []

    def test_an_atom_may_repeat_its_own_common_name(self, workspace):
        _atom("city", "Karnak", payload={"name": {"common": "Karnak",
                                                  "otherNames": ["karnak"]}})

        assert lint.overbroad_aliases("city") == []

    def test_a_declared_alias_line_silences_it(self, workspace, tmp_path, monkeypatch):
        """`aliases.txt` is a human saying these really are one topic. The file
        that exists to declare identity must not be reported as a defect."""
        aliases = tmp_path / "aliases.txt"
        aliases.write_text("Karnak = Thebes\n", encoding="utf-8")
        monkeypatch.setattr(lint.topics.settings, "ALIASES_LIST", aliases)
        _atom("city", "Thebes (Boeotia)", payload={"name": {"common": "Thebes"}})
        _atom("city", "Karnak", payload={"name": {"common": "Karnak",
                                                  "otherNames": ["Thebes"]}})

        assert lint.overbroad_aliases("city") == []


class TestSharedAliases:
    """Two atoms claiming a name that no atom is called. Check 1 cannot see
    these — there is no `common` for the claim to collide with — and they are
    where false duplicates come from: `Narmer Palette` and `Two Dog Palette`
    both listed `Hierakonpolis Palette`, so a detector reading declared names
    concluded two distinct objects were one."""

    def test_two_claimants_and_no_owner_is_flagged(self, workspace):
        _atom("artifact", "Narmer Palette",
              payload={"name": {"common": "Narmer Palette",
                                "otherNames": ["Hierakonpolis Palette"]}})
        _atom("artifact", "Two Dog Palette",
              payload={"name": {"common": "Two Dog Palette",
                                "otherNames": ["Hierakonpolis Palette"]}})

        found = lint.shared_aliases("artifact")

        assert sorted(f.subject for f in found) == ["Narmer Palette", "Two Dog Palette"]
        # Each row names the rival, which is the evidence the reviewer needs.
        assert "Two Dog Palette" in next(
            f for f in found if f.subject == "Narmer Palette").detail

    def test_an_alias_some_atom_owns_belongs_to_check_one(self, workspace):
        """Not this check's case, and reporting it in both would double-count
        the same defect under two names."""
        _atom("city", "Thebes (Boeotia)", payload={"name": {"common": "Thebes"}})
        _atom("city", "Karnak", payload={"name": {"common": "Karnak",
                                                  "otherNames": ["Thebes"]}})
        _atom("city", "Luxor", payload={"name": {"common": "Luxor",
                                                 "otherNames": ["Thebes"]}})

        assert lint.shared_aliases("city") == []

    def test_a_single_claimant_is_ordinary_synonymy(self, workspace):
        _atom("city", "Karnak", payload={"name": {"common": "Karnak",
                                                  "otherNames": ["Ipet-isut"]}})

        assert lint.shared_aliases("city") == []

    def test_it_sees_across_atom_types(self, workspace):
        """The collisions that matter cross types — a person and the event of
        their reign — and a per-type index cannot see the pair."""
        _atom("person", "Ramesses II",
              payload={"name": {"common": "Ramesses II",
                                "otherNames": ["Usermaatre Setepenre"]}})
        _atom("event", "Reign of Ramesses II",
              payload={"name": {"common": "Reign of Ramesses II",
                                "otherNames": ["Usermaatre Setepenre"]}})

        assert [f.subject for f in lint.shared_aliases("person")] == ["Ramesses II"]
        assert [f.subject for f in lint.shared_aliases("event")] == ["Reign of Ramesses II"]

    def test_two_files_for_one_subject_are_left_to_the_duplicate_finder(self, workspace):
        """Same subject twice is a duplicate, not an ambiguous name, and
        `universe_builder duplicates` is what adjudicates those."""
        _atom("city", "Memphis", payload={"name": {"common": "Memphis",
                                                   "otherNames": ["Men-nefer"]}},
              model="gpt-5-nano")
        _atom("city", "Memphis", payload={"name": {"common": "Memphis",
                                                   "otherNames": ["Men-nefer"]}},
              model="gpt-4o-mini")

        assert lint.shared_aliases("city") == []

    def test_a_declared_alias_line_silences_it(self, workspace, tmp_path, monkeypatch):
        aliases = tmp_path / "aliases.txt"
        aliases.write_text("Narmer Palette = Hierakonpolis Palette\n", encoding="utf-8")
        monkeypatch.setattr(lint.topics.settings, "ALIASES_LIST", aliases)
        _atom("artifact", "Narmer Palette",
              payload={"name": {"common": "Narmer Palette",
                                "otherNames": ["Hierakonpolis Palette"]}})
        _atom("artifact", "Two Dog Palette",
              payload={"name": {"common": "Two Dog Palette",
                                "otherNames": ["Hierakonpolis Palette"]}})

        assert lint.shared_aliases("artifact") == []


class TestSharedAliasSuggestions:
    """The pre-fill for this check must never propose `same`. Containment is
    the dominant failure here, not a signal: `suggest_verdict` proposed `same`
    for an event claiming the artifact it produced."""

    def test_it_never_proposes_same(self, workspace):
        for subject, alias in [("Pyramid of Djoser", "Step Pyramid of Djoser"),
                               ("Sinai and Palestine Campaign", "Sinai Campaign"),
                               ("Cleopatra V", "Cleopatra V Tryphaena")]:
            verdict, _why = lint.suggest_shared_verdict(subject, alias)
            assert verdict != "same", f"{subject} <- {alias}"

    def test_an_action_on_a_thing_is_not_the_thing(self, workspace):
        """The owner is looked up in the corpus, not guessed from the strings."""
        _atom("person", "Khafre", payload={"name": {
            "common": "Khafre", "otherNames": ["Khafra", "Chephren"]}})

        # `Khafra` and `Khafre` share no token, so the old overlap rule was
        # silent here. The person atom declares both spellings.
        assert lint.suggest_shared_verdict("Reign of Khafre", "Khafra")[0] == "wrong"
        assert lint.suggest_shared_verdict("Reign of Khafre", "Chephren")[0] == "wrong"

    def test_a_wrapper_is_silent_when_no_atom_owns_the_name(self, workspace):
        """No owner, no claim. Guessing from the prefix alone would fire on
        every event atom that happens to start with `Reign of`."""
        _atom("person", "Khafre", payload={"name": {"common": "Khafre"}})

        assert lint.suggest_shared_verdict(
            "Reign of Khafre", "Some Unrelated Name")[0] is None


class TestReviewEvidenceMatchesItsRow:
    """Evidence must describe the pair the row is asking about. An atom that
    claims two shared aliases has a different rival for each, and the header
    tells the reviewer to treat DISAGREES as strong — so a mismatched pair does
    not merely confuse, it argues for the wrong verdict."""

    def test_each_row_is_corroborated_against_its_own_rival(
            self, workspace, tmp_path, monkeypatch):
        _atom("city", "Subject", payload={"name": {
            "common": "Subject", "otherNames": ["Alias One", "Alias Two"]}})
        _atom("city", "Rival One", payload={"name": {
            "common": "Rival One", "otherNames": ["Alias One"]}})
        _atom("city", "Rival Two", payload={"name": {
            "common": "Rival Two", "otherNames": ["Alias Two"]}})

        # Record which rival each row is compared against, which is the thing
        # the file-keyed map got wrong.
        looked_up = []
        real = lint._atom_by_common
        monkeypatch.setattr(lint, "_atom_by_common",
                            lambda t, n: looked_up.append(n) or real(t, n))

        report = lint.Report(findings=lint.shared_aliases("city"))
        out = tmp_path / "review.tsv"
        lint.write_review(report, out)

        rows = [line.split("\t") for line in out.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")]
        # Subject's two rows must consult *different* rivals. Keyed by filename
        # alone, both consulted whichever rival happened to be first.
        subject_lookups = [n for r, n in zip(rows, looked_up) if r[4] == "Subject"]
        assert sorted(subject_lookups) == ["Rival One", "Rival Two"], subject_lookups


def suggested_same(finding):
    return lint.suggest_verdict(finding.subject, finding.value)[0] == "same"


class TestAnAdjudicatorAnswersAlone:
    """When `--via` names a source, every filled verdict must come from it.
    Falling back to the string rules where it abstained produced rows reading
    `same` under the note "resolves to a disambiguation page" -- the heuristic
    overriding a refusal while keeping its wording."""

    def test_an_abstention_is_not_overridden_by_the_string_rules(
            self, workspace, tmp_path):
        """The leak was on `overbroad-alias` rows, whose fallback is
        `suggest_verdict` -- the rule that calls one name containing another
        `same`. `Apollonia` / `Apollonia Pontica` is the real case: Wikipedia
        declined because `Apollonia` is a disambiguation page, and the
        heuristic answered `same` anyway."""
        _atom("city", "Apollonia Pontica", payload={"name": {
            "common": "Apollonia Pontica"}})
        _atom("city", "Apollonia", payload={"name": {
            "common": "Apollonia", "otherNames": ["Apollonia Pontica"]}})
        report = lint.Report(findings=lint.overbroad_aliases("city"))
        assert report.findings, "fixture should produce an overbroad-alias finding"
        # Precondition: the heuristic really would say `same` here.
        assert suggested_same(report.findings[0])

        out = tmp_path / "review.tsv"
        lint.write_review(report, out,
                          decide=lambda f: (None, "resolves to a disambiguation page"))

        rows = [line.split("\t") for line in out.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")]
        assert rows, "expected rows"
        for row in rows:
            assert row[0] == "?", f"abstention overridden: {row[0]} ({row[6]})"
            assert "disambiguation" in row[6]

    def test_a_committed_verdict_is_used(self, workspace, tmp_path):
        _atom("city", "Alpha", payload={"name": {
            "common": "Alpha", "otherNames": ["Shared Name"]}})
        _atom("city", "Beta", payload={"name": {
            "common": "Beta", "otherNames": ["Shared Name"]}})
        report = lint.Report(findings=lint.shared_aliases("city"))
        out = tmp_path / "review.tsv"
        lint.write_review(report, out, decide=lambda f: ("wrong", "separate articles"))

        rows = [line.split("\t") for line in out.read_text(encoding="utf-8").splitlines()
                if line and not line.startswith("#")]
        assert {r[0] for r in rows} == {"wrong"}


class TestContestedNamesIdentifyNobody:
    """When the contestants are known to be different things, the name they
    share describes both and identifies neither -- which is why it resolves to
    nothing today. Two Cairo gates both called `Gate of Victory`."""

    def _findings(self):
        return lint.shared_aliases("artifact")

    def test_distinct_contestants_lose_the_shared_name(self, workspace):
        _atom("artifact", "Bab al-Nasr", payload={"name": {
            "common": "Bab al-Nasr", "otherNames": ["Gate of Victory"]}})
        _atom("artifact", "Bab al-Futuh", payload={"name": {
            "common": "Bab al-Futuh", "otherNames": ["Gate of Victory"]}})
        pages = {"Bab al-Nasr": 1, "Bab al-Futuh": 2, "Gate of Victory": None}

        out = lint.rivals_distinct_verdicts(self._findings(), pages.get)

        assert len(out) == 2
        assert {v[0] for v in out.values()} == {"wrong"}
        assert "separate articles" in next(iter(out.values()))[1]

    def test_it_stays_silent_when_distinctness_is_unproven(self, workspace):
        """No article for a contestant means no evidence they differ. Retiring a
        name on an assumption is exactly what this check exists to prevent."""
        _atom("artifact", "Alpha", payload={"name": {
            "common": "Alpha", "otherNames": ["Shared"]}})
        _atom("artifact", "Beta", payload={"name": {
            "common": "Beta", "otherNames": ["Shared"]}})

        assert lint.rivals_distinct_verdicts(self._findings(), lambda t: None) == {}

    def test_contestants_on_one_article_are_a_duplicate_not_an_ambiguity(self, workspace):
        """Two subjects landing on one article is `universe_builder duplicates`' problem;
        retiring a name over it would hide the duplicate rather than fix it."""
        _atom("artifact", "Alpha", payload={"name": {
            "common": "Alpha", "otherNames": ["Shared"]}})
        _atom("artifact", "Beta", payload={"name": {
            "common": "Beta", "otherNames": ["Shared"]}})
        same_page = {"Alpha": 7, "Beta": 7, "Shared": None}

        assert lint.rivals_distinct_verdicts(self._findings(), same_page.get) == {}

    def test_a_subject_whose_article_states_the_name_keeps_it(self, workspace):
        """Without this the rule strips the name from every contestant,
        including the one that really has it."""
        _atom("person", "Cleopatra VII", payload={"name": {
            "common": "Cleopatra VII", "otherNames": ["Cleopatra Philopator"]}})
        _atom("person", "Cleopatra IV", payload={"name": {
            "common": "Cleopatra IV", "otherNames": ["Cleopatra Philopator"]}})
        pages = {"Cleopatra VII": 1, "Cleopatra IV": 2, "Cleopatra Philopator": None}

        def owns(subject, _alias):
            return subject == "Cleopatra VII"

        out = lint.rivals_distinct_verdicts(
            lint.shared_aliases("person"), pages.get, owns=owns)

        assert len(out) == 1
        assert "cleopatra_iv" in next(iter(out))[0]

    def test_no_evidence_is_not_evidence_of_absence(self, workspace):
        """`None` from the veto must block the deletion. A two-state veto fails
        open: one failed request blanks a whole batch of leads, and every
        contested name in it is then deleted on evidence never obtained."""
        _atom("person", "Cleopatra VII", payload={"name": {
            "common": "Cleopatra VII", "otherNames": ["Cleopatra Philopator"]}})
        _atom("person", "Cleopatra IV", payload={"name": {
            "common": "Cleopatra IV", "otherNames": ["Cleopatra Philopator"]}})
        pages = {"Cleopatra VII": 1, "Cleopatra IV": 2, "Cleopatra Philopator": None}

        out = lint.rivals_distinct_verdicts(
            lint.shared_aliases("person"), pages.get, owns=lambda s, a: None)

        assert out == {}, "a lookup that returned nothing must not delete a name"

    def test_an_alias_the_source_could_not_look_up_abstains(self, workspace):
        """A resolver leaves out a name it could not ask
        about. For the alias that is "unknown whose article it is", not "no
        article", so nothing is decided; answered with no page, it is."""
        _atom("artifact", "Bab al-Nasr", payload={"name": {
            "common": "Bab al-Nasr", "otherNames": ["Gate of Victory"]}})
        _atom("artifact", "Bab al-Futuh", payload={"name": {
            "common": "Bab al-Futuh", "otherNames": ["Gate of Victory"]}})
        unanswered = {"Bab al-Nasr": 1, "Bab al-Futuh": 2}
        answered = {**unanswered, "Gate of Victory": None}

        assert lint.rivals_distinct_verdicts(self._findings(), unanswered.get,
                                             answered=unanswered.__contains__) == {}
        assert len(lint.rivals_distinct_verdicts(self._findings(), answered.get,
                                                 answered=answered.__contains__)) == 2

    def test_the_owner_of_the_name_keeps_it(self, workspace):
        """Where the alias resolves to one contestant's own article, that
        contestant owns it and `wrong` would delete a real name."""
        _atom("artifact", "Dipylon Amphora", payload={"name": {
            "common": "Dipylon Amphora", "otherNames": ["Dipylon Vase"]}})
        _atom("artifact", "Other Vase", payload={"name": {
            "common": "Other Vase", "otherNames": ["Dipylon Vase"]}})
        pages = {"Dipylon Amphora": 5, "Other Vase": 6, "Dipylon Vase": 5}

        out = lint.rivals_distinct_verdicts(self._findings(), pages.get)

        assert [f for (f, _a) in out] == [
            p.name for p in store.iter_atoms("artifact")
            if "other_vase" in p.name]


class TestRemovalsArePreviewable:
    """`wrong` is the only verdict that deletes, and `data/` has no undo. A dry
    run that prints the alias lines it would add while only counting the entries
    it would remove makes the destructive half the unpreviewable half."""

    def test_a_dry_run_names_every_removal(self, workspace):
        rows = [("wrong", "city", "city_h1_karnak_gpt-5-nano.json", "Thebes", "Karnak")]

        outcome = lint.apply_review(rows, dry_run=True)

        assert outcome["would_edit"] == ["city_h1_karnak_gpt-5-nano.json: Thebes"]

    def test_nothing_to_remove_reports_nothing(self, workspace):
        rows = [("same", "city", "city_h1_karnak_gpt-5-nano.json", "Ipet-isut", "Karnak")]

        assert lint.apply_review(rows, dry_run=True)["would_edit"] == []


class TestAliasContradictions:
    """`aliases.txt` and `not_duplicates.txt` are mirrors, and nothing checked
    that they agree. They did not: three groups asserted an identity a human had
    explicitly refused. The asymmetry is what makes it dangerous --
    `not_duplicates.txt` gates only the duplicate finder, while `aliases.txt`
    drives the grouping every resolver reads, so the alias wins."""

    def _files(self, tmp_path, monkeypatch, aliases, not_dupes):
        (tmp_path / "aliases.txt").write_text(aliases, encoding="utf-8")
        (tmp_path / "not_duplicates.txt").write_text(not_dupes, encoding="utf-8")
        monkeypatch.setattr(lint.topics.settings, "ALIASES_LIST",
                            tmp_path / "aliases.txt")
        return next(spec.name for spec in lint.active_pack().atoms)

    def test_a_group_that_contradicts_a_refusal_is_flagged(
            self, workspace, tmp_path, monkeypatch):
        first = self._files(
            tmp_path, monkeypatch,
            "Dipylon Krater = Dipylon Amphora = Dipylon Vase\n",
            "Dipylon Krater != Dipylon Amphora\n")

        found = lint.alias_contradictions(first)

        assert len(found) == 1
        assert "not_duplicates.txt declares" in found[0].detail

    def test_a_three_way_refusal_is_read_past_its_first_separator(
            self, workspace, tmp_path, monkeypatch):
        """`split("!=", 1)` kept only the first separator, folding the whole
        tail into one unmatchable key.

        `A != B != C` became `{a, "b != c"}`, so seven multi-way refusals in
        the shipped file were invisible and even their first pair was
        mis-keyed. The check reported "No findings" on a file holding real
        contradictions — the one output a linter must never produce.
        """
        first = self._files(
            tmp_path, monkeypatch,
            "Crocodilopolis = Fayoum\n",
            "Arsinoe != Soknopaiou Nesos != Fayoum != Crocodilopolis\n")

        found = lint.alias_contradictions(first)

        assert len(found) == 1, "the contradiction is past the first `!=`"

    def test_refusals_fold_exactly_as_the_grouping_machinery_does(
            self, workspace, tmp_path, monkeypatch):
        """Folded with `topics._key`, `Nicaea (İznik)` became `nicaea i znik`
        and never met the alias group's `nicaea`.

        `duplicates._claim_keys` exists for precisely this and says so — "the
        safeguards must fold names exactly as the grouping machinery does" —
        and the same bug was reintroduced here. Live consequence:
        `aliases.txt` grouping `Nicaea = Nikaia` went unreported against
        `not_duplicates.txt`'s `Nicaea (İznik) != Nikaia (Nice)`, which is the
        merge that file's own header warns about.
        """
        first = self._files(
            tmp_path, monkeypatch,
            "Nicaea = Nikaia\n",
            "Nicaea (\u0130znik) != Nikaia (Nice)\n")

        found = lint.alias_contradictions(first)

        assert len(found) == 1, "parenthetical names must fold to their parts"

    def test_agreeing_files_are_clean(self, workspace, tmp_path, monkeypatch):
        first = self._files(
            tmp_path, monkeypatch,
            "Dipylon Amphora = Dipylon Vase\n",
            "Dipylon Krater != Dipylon Amphora\n")

        assert lint.alias_contradictions(first) == []

    def test_it_runs_for_every_atom_type_and_reports_once(
            self, workspace, tmp_path, monkeypatch):
        """It must RUN whichever type it is called with -- suppressing it for
        all but the pack's first atom made `--atom artifact` answer "No
        findings" having compared nothing. `run` collapses the copies."""
        self._files(tmp_path, monkeypatch,
                    "Dipylon Krater = Dipylon Amphora\n",
                    "Dipylon Krater != Dipylon Amphora\n")
        for spec in lint.active_pack().atoms:
            assert len(lint.alias_contradictions(spec.name)) == 1, spec.name

        report = lint.run(checks=["alias-contradiction"])

        assert len(report.findings) == 1, "one defect, reported once"

    def test_no_refusals_file_is_vacuously_clean_and_says_so(
            self, workspace, tmp_path, monkeypatch, caplog):
        """No refusals means none can be contradicted, so clean is true rather
        than unchecked -- a domain without the file is a valid configuration.
        It still logs, because a mis-set ALIASES_LIST looks identical."""
        (tmp_path / "aliases.txt").write_text(
            "Dipylon Krater = Dipylon Amphora\n", encoding="utf-8")
        monkeypatch.setattr(lint.topics.settings, "ALIASES_LIST",
                            tmp_path / "aliases.txt")

        with caplog.at_level("WARNING"):
            report = lint.run(checks=["alias-contradiction"])

        assert report.is_clean()
        assert any("not_duplicates.txt" in r.message for r in caplog.records)


class TestTopicDrift:
    """The index call decides every later topic and its mistakes are invisible
    downstream: asked for a topic that does not exist, the create pass writes a
    schema-perfect, factually true atom about something else. The only trace is
    that the atom declines to call itself what it was asked about."""

    def _atom_for(self, atom_type, topic, common, others=None):
        return _atom(atom_type, topic, payload={"name": {
            "common": common, "otherNames": others or []}})

    def test_a_fabricated_attribution_is_flagged(self, workspace):
        """`Berlin Gold Hat of Tutankhamun` does not exist. The atom wrote up
        the real Berlin Gold Hat -- Central European, bronze age -- and filed it
        under Egyptian history."""
        self._atom_for("artifact", "Berlin Gold Hat of Tutankhamun", "Berlin Gold Hat")

        found = lint.topic_drift("artifact")

        assert [f.subject for f in found] == ["Berlin Gold Hat"]
        assert "tutankhamun" in found[0].value

    def test_a_hedged_relevance_claim_is_flagged(self, workspace):
        """The parenthetical argues for the topic's own inclusion, and the atom
        refuses the claim."""
        self._atom_for("artifact",
                       "Shroud of Turin believed to be connected with Egypt",
                       "Shroud of Turin")

        assert [f.subject for f in lint.topic_drift("artifact")] == ["Shroud of Turin"]

    def test_a_second_name_for_one_place_is_not_drift(self, workspace):
        """`Nicaea (Iznik)` is a city keeping its ancient name. Juxtaposed
        names carry no connective, which is what separates them."""
        self._atom_for("city", "Nicaea Iznik", "Nicaea")

        assert lint.topic_drift("city") == []

    def test_a_trailing_location_is_not_an_attribution(self, workspace):
        """`in Athens` is where the thing is, not a claim about whose it is.
        An earlier rule counted every word after any `of` and flagged this."""
        self._atom_for("artifact", "Tomb of the Unknown Soldier in Athens",
                       "Tomb of the Unknown Soldier")

        assert lint.topic_drift("artifact") == []

    def test_an_atom_that_keeps_the_name_as_an_alias_is_fine(self, workspace):
        """Recording the dropped words in `otherNames` is the atom accepting
        the topic, not disowning it."""
        self._atom_for("artifact", "Berlin Gold Hat of Tutankhamun",
                       "Berlin Gold Hat", ["Berlin Gold Hat of Tutankhamun"])

        assert lint.topic_drift("artifact") == []


class TestEraContradictions:
    """The corpus's dominant genuine error is era confusion, and it is often
    catchable without a model because the atom contradicts *itself*."""

    def _city(self, periods):
        return _atom("city", "Somewhere",
                     payload={"name": {"common": "Somewhere"},
                              "cityPeriodsOfHistory": periods})

    def test_prose_disagreeing_with_the_field_is_flagged(self, workspace):
        self._city([{"periodName": "Trade era",
                     "periodEnd": {"year": 600, "era": "CE",
                                   "longFormInfo": ["ran circa 1500-600 BCE"]}}])

        found = lint.era_contradictions("city")

        assert len(found) == 1
        assert "600 CE" in found[0].detail and "600 BCE" in found[0].detail

    def test_agreement_is_not_flagged(self, workspace):
        self._city([{"periodName": "Trade era",
                     "periodEnd": {"year": 600, "era": "BCE",
                                   "longFormInfo": ["ended circa 600 BCE"]}}])

        assert lint.era_contradictions("city") == []

    def test_one_periods_prose_cannot_contradict_anothers_date(self, workspace):
        """The false positive that scoping fixed. `Malia` has a Minoan period
        discussing 1900 BCE and a modern-village period starting 1900 CE; both
        are correct, and a whole-atom comparison called the second one a
        contradiction."""
        self._city([
            {"periodName": "Minoan",
             "periodStart": {"year": 1900, "era": "BCE",
                             "longFormInfo": ["palaces from 1900 BCE"]}},
            {"periodName": "Modern village",
             "periodStart": {"year": 1900, "era": "CE",
                             "longFormInfo": ["the village grew from about 1900 CE"]}},
        ])

        assert lint.era_contradictions("city") == []

    def test_a_bare_year_in_prose_says_nothing(self, workspace):
        """Without an era token the text is not making a claim to contradict."""
        self._city([{"periodName": "Trade era",
                     "periodEnd": {"year": 600, "era": "CE",
                                   "longFormInfo": ["ended around 600 somehow"]}}])

        assert lint.era_contradictions("city") == []

    def test_sources_dates_are_ignored(self, workspace):
        """A citation's date is about the edition, not the subject."""
        self._city([{"periodName": "Trade era",
                     "periodEnd": {"year": 600, "era": "CE",
                                   "sources": ["Some Book (600 BCE reprint)"]}}])

        assert lint.era_contradictions("city") == []


class TestTheRunner:
    def test_a_clean_corpus_reports_nothing(self, workspace):
        _atom("city", "Memphis")

        assert lint.run(atom_types=["city"]).is_clean()

    def test_findings_group_by_check(self, workspace):
        _atom("city", "Thebes (Boeotia)", payload={"name": {"common": "Thebes"}})
        _atom("city", "Karnak", payload={"name": {"common": "Karnak",
                                                  "otherNames": ["Thebes"]}})

        report = lint.run(atom_types=["city"])

        assert "overbroad-alias" in report.by_check()

    def test_a_failing_check_does_not_lose_the_others(self, workspace, monkeypatch):
        """One broken check must not cost the rest their results..."""
        monkeypatch.setitem(lint.CHECKS, "boom",
                            lambda at: (_ for _ in ()).throw(RuntimeError("no")))
        _atom("city", "Thebes (Boeotia)", payload={"name": {"common": "Thebes"}})
        _atom("city", "Karnak", payload={"name": {"common": "Karnak",
                                                  "otherNames": ["Thebes"]}})

        report = lint.run(checks=["boom", "overbroad-alias"], atom_types=["city"])

        assert [f.subject for f in report.findings] == ["Karnak"]

    def test_but_a_failing_check_is_never_reported_as_clean(self, workspace, monkeypatch):
        """...and must not read as a pass. A check that crashed established
        nothing; "no findings" is the one output a linter must never emit when
        it did not actually run. This test previously asserted the opposite,
        codifying the bug."""
        monkeypatch.setitem(lint.CHECKS, "boom",
                            lambda at: (_ for _ in ()).throw(RuntimeError("no")))
        _atom("city", "Memphis")

        report = lint.run(checks=["boom"], atom_types=["city"])

        assert report.findings == []
        assert report.errors and not report.is_clean()

    def test_an_unknown_check_name_is_rejected(self, workspace):
        """A typo that answers "No findings" is worse than an error: it says
        the corpus is clean when nothing was checked."""
        import pytest

        with pytest.raises(KeyError, match="Unknown check"):
            lint.run(checks=["era-contradictions"], atom_types=["city"])

    def test_an_unknown_atom_type_is_rejected(self, workspace):
        import pytest

        with pytest.raises(KeyError, match="Unknown atom type"):
            lint.run(atom_types=["citys"])


class TestAdjudication:
    """A file you edit, not a prompt you answer. The decisions outlive the
    session, want diffing, and are better reviewed as a set."""

    def _finding_corpus(self, workspace):
        """Complete atoms, because `apply_review` refuses to write one that
        does not validate — a content fix must not be the thing that makes an
        atom unreadable."""
        from universe_builder.domains.history.models.person import Person
        from tests.factories import minimal_payload

        a = minimal_payload(Person)
        a["name"]["common"] = "Ramesses III"
        a["name"]["otherNames"] = []
        _atom("person", "Ramesses III", payload=a)
        b = minimal_payload(Person)
        b["name"]["common"] = "Ramses III"
        b["name"]["otherNames"] = ["Ramesses III"]
        _atom("person", "Ramses III", payload=b)

    def test_the_review_file_carries_one_row_per_finding(self, workspace, tmp_path):
        self._finding_corpus(workspace)
        out = tmp_path / "review.tsv"

        written = lint.write_review(lint.run(atom_types=["person"]), out)

        rows = lint.read_review(out)
        assert written == 1 and len(rows) == 1
        # `Ramses III` / `Ramesses III` is 91% alike, so the row arrives
        # pre-filled rather than as `?` — the suggestion is the point.
        assert rows[0][0] == "same"

    def test_an_alias_containing_an_apostrophe_survives(self, workspace, tmp_path):
        """The alias is carried as data. Recovering it from the human-readable
        `detail` broke on every name with an apostrophe."""
        _atom("event", "Alexander the Great's conquest of Egypt",
              payload={"name": {"common": "Alexander the Great's conquest of Egypt"}})
        _atom("event", "Conquest of Egypt",
              payload={"name": {"common": "Conquest of Egypt", "otherNames":
                                ["Alexander the Great's conquest of Egypt"]}})
        out = tmp_path / "r.tsv"
        lint.write_review(lint.run(checks=["overbroad-alias"], atom_types=["event"]), out)

        assert lint.read_review(out)[0][3] == "Alexander the Great's conquest of Egypt"

    def test_wrong_strips_the_alias_and_leaves_the_atom_valid(self, workspace, tmp_path):
        self._finding_corpus(workspace)
        out = tmp_path / "r.tsv"
        lint.write_review(lint.run(atom_types=["person"]), out)
        rows = [("wrong",) + r[1:] for r in lint.read_review(out)]

        lint.apply_review(rows, aliases_path=tmp_path / "a.txt", dry_run=False)

        import json as _json
        from universe_builder.domains.history.models.person import Person
        path = next(p for p in store.iter_atoms("person") if "ramses_iii" in p.name)
        payload = _json.loads(path.read_text())
        assert payload["name"]["otherNames"] == []
        Person.model_validate(payload)          # a content fix must not break the atom

    def test_same_declares_an_alias_and_deletes_nothing(self, workspace, tmp_path):
        self._finding_corpus(workspace)
        out = tmp_path / "r.tsv"
        aliases = tmp_path / "a.txt"
        lint.write_review(lint.run(atom_types=["person"]), out)
        rows = [("same",) + r[1:] for r in lint.read_review(out)]

        lint.apply_review(rows, aliases_path=aliases, dry_run=False)

        assert "Ramesses III" in aliases.read_text()
        path = next(p for p in store.iter_atoms("person") if "ramses_iii" in p.name)
        assert "Ramesses III" in path.read_text()   # nothing removed

    def test_a_dry_run_changes_nothing(self, workspace, tmp_path):
        self._finding_corpus(workspace)
        out = tmp_path / "r.tsv"
        lint.write_review(lint.run(atom_types=["person"]), out)
        rows = [("wrong",) + r[1:] for r in lint.read_review(out)]

        outcome = lint.apply_review(rows, aliases_path=tmp_path / "a.txt")

        assert outcome["dry_run"] is True
        path = next(p for p in store.iter_atoms("person") if "ramses_iii" in p.name)
        assert "Ramesses III" in path.read_text()

    def test_undecided_and_skip_are_left_alone(self, workspace, tmp_path):
        self._finding_corpus(workspace)
        out = tmp_path / "r.tsv"
        lint.write_review(lint.run(atom_types=["person"]), out)
        # Force the undecided state the suggestion would otherwise fill in.
        out.write_text(out.read_text(encoding="utf-8").replace("same\t", "?\t", 1),
                       encoding="utf-8")

        outcome = lint.apply_review(lint.read_review(out),
                                    aliases_path=tmp_path / "a.txt", dry_run=False)

        assert outcome["same"] == 0 and outcome["wrong"] == 0
        assert outcome["edited"] == []

    def test_comments_and_blanks_round_trip(self, tmp_path):
        out = tmp_path / "r.tsv"
        out.write_text("# a comment\n\nwrong\tcity\tf.json\tThebes\tKarnak\n",
                       encoding="utf-8")

        assert lint.read_review(out) == [
            ("wrong", "city", "f.json", "Thebes", "Karnak")]

    def test_the_alias_line_uses_the_atoms_name_not_the_filename_slug(self, tmp_path):
        """`store.slugify` lowercases and strips diacritics, so a subject
        rebuilt from the filename declares identity for a name no atom has —
        and because the first name on an alias line is the canonical spelling
        the work list emits, it would also make future runs generate it."""
        out = tmp_path / "r.tsv"
        out.write_text(
            "same\tperson\tperson_h6d05_almuizz_lidn_allh_gpt-5.6-luna.json\t"
            "Al-Mu'izz li-Din Allah\tAl-Mu\u02bfizz li-D\u012bn All\u0101h\n",
            encoding="utf-8")

        outcome = lint.apply_review(lint.read_review(out), aliases_path=tmp_path / "a.txt")

        assert outcome["alias_lines"] == [
            "Al-Mu\u02bfizz li-D\u012bn All\u0101h = Al-Mu'izz li-Din Allah"]
        assert "almuizz lidn allh" not in " ".join(outcome["alias_lines"])

    def test_a_row_with_no_subject_is_refused_not_guessed(self, tmp_path):
        out = tmp_path / "r.tsv"
        out.write_text("same\tcity\tcity_h1_karnak_m.json\tThebes\t\n",
                       encoding="utf-8")

        outcome = lint.apply_review(lint.read_review(out), aliases_path=tmp_path / "a.txt")

        assert outcome["alias_lines"] == []
        assert outcome["unusable"] == ["city_h1_karnak_m.json"]


class TestCorroboration:
    """Do the two atoms agree about place and date? Diagnostic, not decisive."""

    def _pair(self, atom_type, a, b):
        return lint.corroboration(atom_type, a, b)

    def test_disagreement_on_place_is_called_out(self):
        a = {"birth": {"location": {"latitude": 37.9, "longitude": 23.7}}}
        b = {"birth": {"location": {"latitude": 40.6, "longitude": 22.9}}}

        assert self._pair("person", a, b).startswith("DISAGREES")

    def test_dates_catch_what_coordinates_miss(self):
        """`Battle of Navarino` and `Battle of Pylos` are 6km apart in one bay
        and 2,252 years apart. Place alone would call them the same event."""
        a = {"eventStart": {"location": {"latitude": 36.93, "longitude": 21.69},
                            "date": {"year": 1827, "era": "CE"}}}
        b = {"eventStart": {"location": {"latitude": 36.95, "longitude": 21.70},
                            "date": {"year": 425, "era": "BCE"}}}

        evidence = self._pair("event", a, b)

        assert evidence.startswith("DISAGREES") and "2252y" in evidence

    def test_bce_and_ce_are_not_the_same_year(self):
        """Signed years, so 50 BCE and 50 CE are a century apart, not equal."""
        a = {"eventStart": {"date": {"year": 50, "era": "BCE"}}}
        b = {"eventStart": {"date": {"year": 50, "era": "CE"}}}

        assert "100y" in self._pair("event", a, b)

    def test_agreement_is_reported_weakly(self):
        """Containment agrees too — `Book of the Dead` and `Papyrus of Ani` are
        0km and 0 years apart because one is a copy of the other, exactly as
        Karnak sits inside Thebes. So agreement is evidence, not a verdict."""
        a = {"dateCreated": {"year": 1250, "era": "BCE"}}
        b = {"dateCreated": {"year": 1250, "era": "BCE"}}

        assert self._pair("artifact", a, b).startswith("agrees")

    def test_mismatched_fields_are_never_compared(self):
        """One atom's museum against the other's dig site made `Sphinx of
        Taharqo` look 767km from `Sphinx of Taharqa`."""
        a = {"currentLocation": {"latitude": 51.5, "longitude": -0.1}}
        b = {"discoveryLocation": {"latitude": 19.0, "longitude": 30.0}}

        assert self._pair("artifact", a, b) == "no comparable place or date"

    def test_nothing_comparable_says_so_rather_than_guessing(self):
        assert self._pair("city", {"name": {}}, {"name": {}}) == \
            "no comparable place or date"

    def test_the_review_file_carries_the_evidence(self, workspace, tmp_path):
        from universe_builder.domains.history.models.person import Person
        from tests.factories import minimal_payload

        a = minimal_payload(Person)
        a["name"]["common"] = "Ramesses III"
        a["name"]["otherNames"] = []
        a["birth"]["location"].update({"latitude": 25.7, "longitude": 32.6})
        _atom("person", "Ramesses III", payload=a)
        b = minimal_payload(Person)
        b["name"]["common"] = "Ramses III"
        b["name"]["otherNames"] = ["Ramesses III"]
        b["birth"]["location"].update({"latitude": 55.7, "longitude": 12.6})
        _atom("person", "Ramses III", payload=b)
        out = tmp_path / "r.tsv"

        lint.write_review(lint.run(checks=["overbroad-alias"],
                                   atom_types=["person"]), out)

        assert "DISAGREES" in out.read_text()

    def test_an_unknown_era_is_incomparable_not_ce(self):
        """`Date.era` is Optional. Defaulting a missing one to CE turned
        `1250 (unknown)` against `1250 BCE` into a confident 2,500-year
        disagreement that was purely an artifact of the missing field."""
        a = {"dateCreated": {"year": 1250, "era": None}}
        b = {"dateCreated": {"year": 1250, "era": "BCE"}}

        assert lint.corroboration("artifact", a, b) == "no comparable place or date"

    def test_an_unknown_era_does_not_suppress_usable_evidence(self):
        """It removes that one date from the comparison, not the whole row."""
        a = {"dateCreated": {"year": 1250, "era": None},
             "discoveryLocation": {"latitude": 25.7, "longitude": 32.6}}
        b = {"dateCreated": {"year": 1250, "era": "BCE"},
             "discoveryLocation": {"latitude": 25.7, "longitude": 32.6}}

        evidence = lint.corroboration("artifact", a, b)

        assert evidence.startswith("agrees") and "dateCreated" not in evidence


class TestSuggestedVerdicts:
    """A rule derived from 33 hand-adjudicated findings, not invented.

    It only ever suggests `same`. Every genuinely different pair in that sample
    was *also* unlike in name, so silence is the right output for them — and
    the failure worth avoiding is the expensive one, asserting two different
    subjects are one and writing a false identity into the alias file.
    """

    def test_containment_suggests_same(self):
        assert lint.suggest_verdict("Temple of Edfu",
                                    "Temple of Horus at Edfu")[0] == "same"

    def test_word_order_does_not_matter(self):
        assert lint.suggest_verdict("King List of Abydos",
                                    "Abydos King List")[0] == "same"

    def test_plurals_match(self):
        assert lint.suggest_verdict("Philae Obelisks", "Philae obelisk")[0] == "same"

    def test_spelling_variants_match_on_similarity(self):
        assert lint.suggest_verdict("Ramses III", "Ramesses III")[0] == "same"
        assert lint.suggest_verdict("Sphinx of Taharqo",
                                    "Sphinx of Taharqa")[0] == "same"

    def test_diacritics_do_not_defeat_it(self):
        assert lint.suggest_verdict("Al-Muʿizz li-Dīn Allāh",
                                    "Al-Mu'izz li-Din Allah")[0] == "same"

    def test_a_shared_surname_is_not_enough(self):
        """The case that set the threshold. `Konstantinos Karamanlis` and
        `Kostas Karamanlis` are 85% alike because the surname dominates — an
        uncle and his nephew, 338km and 49 years apart."""
        assert lint.suggest_verdict("Konstantinos Karamanlis",
                                    "Kostas Karamanlis")[0] is None

    def test_a_shared_prefix_is_not_enough(self):
        assert lint.suggest_verdict("Battle of Navarino",
                                    "Battle of Pylos")[0] is None
        assert lint.suggest_verdict("Nefertiti", "Neferneferuaten")[0] is None

    def test_it_never_suggests_wrong(self):
        """Unlike names mean 'ask a human', not 'these differ'."""
        assert lint.suggest_verdict("Suez", "Clysma") == (None, "")

    def test_the_review_file_carries_the_suggestion(self, workspace, tmp_path):
        _atom("city", "Temple of Horus at Edfu",
              payload={"name": {"common": "Temple of Horus at Edfu"}})
        _atom("city", "Temple of Edfu",
              payload={"name": {"common": "Temple of Edfu",
                                "otherNames": ["Temple of Horus at Edfu"]}})
        out = tmp_path / "r.tsv"

        lint.write_review(lint.run(checks=["overbroad-alias"],
                                   atom_types=["city"]), out)

        assert lint.read_review(out)[0][0] == "same"


class TestUnknownVerdictsAreRefused:
    def test_a_typo_stops_the_whole_apply(self, tmp_path):
        """A real review came back with two rows reading `sane`; the old code
        dropped them silently, so the reviewer would have been told 31 rows
        were processed rather than 33, with no hint which two vanished."""
        import pytest

        out = tmp_path / "r.tsv"
        out.write_text("sane\tcity\tf.json\tThebes\tKarnak\n", encoding="utf-8")

        with pytest.raises(ValueError, match="Unrecognised verdict"):
            lint.read_review(out)

    def test_the_message_names_the_offending_rows(self, tmp_path):
        import pytest

        out = tmp_path / "r.tsv"
        out.write_text("sane\tcity\tmahalla.json\tA\tB\n"
                       "same\tcity\tfine.json\tC\tD\n", encoding="utf-8")

        with pytest.raises(ValueError, match="mahalla.json"):
            lint.read_review(out)


class TestRegnalNumbersDecideIdentity:
    """A regnal number is one or two characters in an otherwise identical name,
    so similarity scores it as noise — exactly backwards. `Ramses II` and
    `Ramses III` are 95% alike and are two different pharaohs. Every pair below
    exists in this corpus.
    """

    def test_differing_roman_numerals_are_never_suggested(self):
        for a, b in [("Ramses II", "Ramses III"),
                     ("Ptolemy II", "Ptolemy III"),
                     ("Thutmose I", "Thutmose IV"),
                     ("Cleopatra VII", "Cleopatra IV"),
                     ("Berenice III", "Berenice IV")]:
            assert lint.suggest_verdict(a, b)[0] is None, f"{a} / {b}"

    def test_a_missing_numeral_is_a_difference_too(self):
        """Containment is just as wrong here: `Ptolemy` sits inside
        `Ptolemy II` while naming someone else entirely."""
        assert lint.suggest_verdict("Ptolemy", "Ptolemy II")[0] is None

    def test_matching_numerals_still_allow_a_suggestion(self):
        """The guard must not swallow the real spelling variants it sits
        beside — this is the pair the similarity threshold was tuned on."""
        assert lint.suggest_verdict("Ramses III", "Ramesses III")[0] == "same"

    def test_names_without_numerals_are_unaffected(self):
        assert lint.suggest_verdict("Temple of Edfu",
                                    "Temple of Horus at Edfu")[0] == "same"

    def test_arabic_numerals_count_as_well(self):
        assert lint.suggest_verdict("Ptolemy 2", "Ptolemy 3")[0] is None

    def test_an_ordinary_word_is_not_mistaken_for_a_numeral(self):
        """The numeral test runs on standalone tokens, so a name is not
        derailed by containing the letters of one."""
        assert lint.suggest_verdict("Temple of Isis at Philae",
                                    "Temple of Isis at Philae")[0] == "same"


class TestSelfNegation:
    """An atom whose own summary says its subject did not happen or does not
    exist. The model was asked for a topic the index listed, found nothing,
    and wrote the absence up as an atom — `Egyptian presidential election of
    2023` ("an event that did not take place") and a `hypothetical` mummy
    mask sat in the history corpus looking like any other file."""

    def _event(self, subject, certainty="medium", **prose):
        payload = {"name": {"common": subject},
                   "eventStart": {"date": {"year": 2023, "era": "CE", "certainty": certainty,
                                           "sources": [], "longFormInfo": []}}}
        payload.update(prose)
        return _atom("event", subject, payload=payload)

    def test_a_summary_denying_its_subject_is_flagged(self, workspace):
        self._event("Election of 2023", shortSummary=(
            "There was no election in 2023; this entry records a planned event "
            "that did not take place."))

        (found,) = lint.self_negations("event")

        assert found.check == "self-negation"
        assert found.subject == "Election of 2023"
        assert "did not take place" in found.detail
        assert found.value == "did not take place"

    def test_a_hypothetical_subject_is_flagged(self, workspace):
        self._event("Garden of the King", longSummary=[
            "The Garden of the King is a hypothetical garden attributed to the king."])

        assert [f.subject for f in lint.self_negations("event")] == ["Garden of the King"]

    def test_hypothetical_about_a_reconstruction_is_not(self, workspace):
        """Six benign matches in the history corpus read like this: the
        subject is real, a *reconstruction* of it is hypothetical."""
        self._event("Siege Engine", longSummary=(
            "No complete machine has survived, so reconstructions remain partly "
            "hypothetical."), significance="Proposed restorations are hypothetical.")

        assert lint.self_negations("event") == []

    def test_prose_outside_the_summary_is_not_read(self, workspace):
        """A denial about something *else* the atom mentions is ordinary:
        Nero's curious fact that the violin did not exist says nothing about
        Nero."""
        self._event("Great Fire", curiousFact=["The violin did not exist yet."],
                    shortSummary="A fire that burned for six days.")

        assert lint.self_negations("event") == []

    def test_a_date_of_no_certainty_makes_it_strong_and_first(self, workspace):
        self._event("Weak Case", shortSummary="The meeting never happened.")
        self._event("Strong Case", certainty="none",
                    shortSummary="The meeting never happened.")

        found = lint.self_negations("event")

        assert [f.subject for f in found] == ["Strong Case", "Weak Case"]
        assert found[0].detail.startswith("[strong")
        assert not found[1].detail.startswith("[strong")

    def test_it_is_a_registered_check(self):
        assert lint.CHECKS["self-negation"] is lint.self_negations

    def test_a_pack_with_no_summaries_is_simply_clean(self, workspace):
        from universe_builder import atoms

        atoms.use_pack("medications")
        try:
            _atom("medication", "Aspirin", payload={"name": "Aspirin", "drugClass": "does not exist"})
            assert lint.self_negations("medication") == []
        finally:
            atoms.use_pack("history")
