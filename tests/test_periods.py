"""The period lint, mechanical adjudication, and the split loop.

Uniformity by construction: the checker (not a model) decides *that* a period
splits, majority (not a model) decides disagreeing labels, and a model is only
asked the history of a split the checker ordered. No network: the one
model-shaped step is faked.
"""

from __future__ import annotations

import json

import pytest

from universe_builder import journal, store
from universe_builder.domains.history import periods
from universe_builder.domains.history.models.city import CityExpanded, PeriodSplit, Population
from tests.test_operations import valid_city

MODEL = "gpt-5-nano"


@pytest.fixture(autouse=True)
def _fresh_caches():
    periods._resolvers.clear()
    yield
    periods._resolvers.clear()


@pytest.fixture
def regime_table(tmp_path, monkeypatch):
    path = tmp_path / "regime_changes.txt"
    path.write_text(
        "[Egyptian History]\n"
        "-30: monarchy -> autocracy\n"
        "1953: monarchy -> modern republic\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(periods, "REGIME_CHANGES", path)
    return path


def make_date(year, era="CE"):
    return {"year": abs(year), "era": era, "month": None, "day": None,
            "certainty": "high", "sources": ["test"], "longFormInfo": []}


def make_period(name, start, end, government, era_start=None, era_end=None):
    """A valid Population dict; negative years mean BCE."""
    start_date = {**make_date(start, era_start or ("BCE" if start < 0 else "CE")),
                  "cityPopulationRange": [1000, 2000], "averageCityPopulation": 1500}
    period = {
        "periodName": name,
        "periodStart": start_date,
        "periodEnd": (make_date(end, era_end or ("BCE" if end < 0 else "CE"))
                      if end is not None else
                      {**make_date(2026), "year": None, "era": None}),
        "certainty": "high",
        "typeOfGovernment": government,
        "tradeAndCommerce": ["grain"],
        "tradingPartners": ["Rome"],
        "languagesSpoken": ["Greek"],
    }
    Population.model_validate(period)
    return period


def write_city(subject, periods_list, country=None):
    payload = json.loads(valid_city())
    payload["cityPeriodsOfHistory"] = periods_list
    if country is not None:
        payload.setdefault("location", {})["country"] = country
    CityExpanded.model_validate(payload)
    path = store.path_for("city", None, subject, MODEL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_history(cities):
    payload = {"people": [], "discoveries": [], "events": [], "artifacts": [],
               "cities": cities}
    path = store.dir_for("history") / "history_h1111111111111111_egyptian_history_m.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class TestRegimeTable:
    def test_scoped_lines_parse_with_signed_years(self, regime_table):
        changes = periods.load_regime_changes()

        assert [(c.year, c.before, c.after) for c in changes] == [
            (-30, "monarchy", "autocracy"),
            (1953, "monarchy", "modern republic"),
        ]
        assert all(c.scope == "Egyptian History" for c in changes)

    def test_a_country_qualifier_parses_and_leaves_the_comment_out_of_it(self, tmp_path):
        """`@Country  # why` — the qualifier must not swallow the comment, and
        the comment must not end up inside the country name."""
        path = tmp_path / "t.txt"
        path.write_text("[Greek History]\n"
                        "1967: monarchy -> autocracy  @Greece  # the colonels\n"
                        "-146: democracy -> autocracy  # no qualifier\n",
                        encoding="utf-8")

        changes = periods.load_regime_changes(path)

        assert [(c.year, c.country) for c in changes] == [(1967, "Greece"), (-146, None)]

    def test_a_missing_table_is_no_lint(self, tmp_path):
        assert periods.load_regime_changes(tmp_path / "absent.txt") == ()

    def test_a_value_outside_the_enum_is_skipped_loudly(self, tmp_path, caplog):
        path = tmp_path / "t.txt"
        path.write_text("[X]\n1900: monarchy -> kritarchy\n", encoding="utf-8")

        with caplog.at_level("WARNING"):
            assert periods.load_regime_changes(path) == ()
        assert "kritarchy" in caplog.text

    def test_a_change_that_changes_nothing_is_skipped(self, tmp_path, caplog):
        path = tmp_path / "t.txt"
        path.write_text("[X]\n1900: monarchy -> monarchy\n", encoding="utf-8")

        with caplog.at_level("WARNING"):
            assert periods.load_regime_changes(path) == ()


class TestLint:
    def test_a_period_containing_a_change_is_flagged(self, workspace, regime_table):
        write_history(["Alexandria"])
        write_city("Alexandria", [make_period("Modern Period", 1801, None, "autocracy")])

        report = periods.lint()

        assert len(report.flags) == 1
        assert [c.year for c in report.flags[0].crossings] == [1953]

    def test_a_period_split_at_the_change_is_clean(self, workspace, regime_table):
        write_history(["Alexandria"])
        write_city("Alexandria", [
            make_period("Kingdom of Egypt", 1922, 1953, "monarchy"),
            make_period("Republic of Egypt", 1953, None, "modern republic"),
        ])

        assert periods.lint().clean

    def test_bce_years_compare_correctly(self, workspace, regime_table):
        """`Greco-Roman 332 BCE – 641 CE` contains 30 BCE; signed math or bust."""
        write_history(["Alexandria"])
        write_city("Alexandria", [make_period("Greco-Roman", -332, 641, "autocracy")])

        report = periods.lint()

        assert [c.year for c in report.flags[0].crossings] == [-30]

    def test_the_table_only_applies_to_the_scopes_cities(self, workspace, regime_table):
        """Athens is not listed by Egyptian History; Egypt's dates say nothing
        about it — the known-limit note in the table file, enforced."""
        write_history(["Alexandria"])
        write_city("Athens", [make_period("Modern Period", 1801, None, "autocracy")])

        assert periods.lint().flags == []

    def test_an_unqualified_line_applies_wherever_the_city_is(
            self, workspace, regime_table):
        """The Roman conquest reached every polis wherever it stood, so a line
        with no `@Country` must not start depending on geography."""
        write_history(["Alexandria"])
        write_city("Alexandria", [make_period("Greco-Roman", -332, 641, "autocracy")],
                   country="France")

        assert [c.year for c in periods.lint().flags[0].crossings] == [-30]

    def test_a_qualified_line_only_reaches_its_own_country(
            self, workspace, tmp_path, monkeypatch):
        """The defect this was built for.

        Greek History lists Massalia, Alexandria and Chalcedon, so before
        `@Country` every one of them was checked against Athens' twentieth
        century: many of its flags applied one country's modern
        politics to another country's city — 'Republic of Türkiye' flagged by
        the Greek colonels' junta of 1967.
        """
        table = tmp_path / "t.txt"
        table.write_text("[Egyptian History]\n"
                         "1953: monarchy -> modern republic @Egypt\n", encoding="utf-8")
        monkeypatch.setattr(periods, "REGIME_CHANGES", table)
        write_history(["Alexandria", "Massalia"])
        write_city("Alexandria", [make_period("Modern", 1801, None, "autocracy")],
                   country="Egypt")
        write_city("Massalia", [make_period("Modern", 1801, None, "autocracy")],
                   country="France")

        flagged = {f.path.name for f in periods.lint().flags}

        assert len(flagged) == 1
        assert any("alexandria" in n for n in flagged), flagged

    def test_a_qualified_line_skips_a_city_with_no_country(
            self, workspace, tmp_path, monkeypatch):
        """Fail closed. Including the unknowns would bring back the false
        positives the qualifier exists to remove."""
        table = tmp_path / "t.txt"
        table.write_text("[Egyptian History]\n"
                         "1953: monarchy -> modern republic @Egypt\n", encoding="utf-8")
        monkeypatch.setattr(periods, "REGIME_CHANGES", table)
        write_history(["Alexandria"])
        payload = json.loads(valid_city())
        payload["cityPeriodsOfHistory"] = [make_period("Modern", 1801, None, "autocracy")]
        payload.setdefault("location", {})["country"] = None
        path = store.path_for("city", None, "Alexandria", MODEL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

        assert periods.lint().flags == []

    @pytest.mark.parametrize("spelling", ["Turkey", "Türkiye", "TURKEY", " turkey "])
    def test_country_spellings_are_folded(self, spelling):
        """40 stored cities say `Turkey` and 3 say `Türkiye`; an exact match
        would quietly exempt those three from a line meant to cover them."""
        change = periods.RegimeChange("Greek History", 1960, "monarchy",
                                      "autocracy", "Turkey")

        assert change.applies_to(spelling)

    def test_a_different_country_is_not_folded_together(self):
        change = periods.RegimeChange("Greek History", 1960, "monarchy",
                                      "autocracy", "Turkey")

        assert not change.applies_to("Greece")

    def test_a_city_exemption_matches_the_city_not_the_polity(self):
        """`!city:NAME`: Kinmen's military
        administration outlasted Taiwan's martial law."""
        change = periods.RegimeChange("Chinese History", 1987, "autocracy",
                                      "modern republic", "Taiwan",
                                      ("city:Kinmen", "Republic of China"))

        assert change.exempts(None, " kinmen ")
        assert change.exempts("Republic of China", "Taipei")
        assert not change.exempts(None, "Taipei")
        assert not change.exempts("city:Kinmen", "Taipei")

    def test_a_city_exemption_reaches_the_lint(self, workspace, tmp_path, monkeypatch):
        table = tmp_path / "t.txt"
        table.write_text("[Egyptian History]\n"
                         "1953: monarchy -> modern republic @Egypt !city:Alexandria\n",
                         encoding="utf-8")
        monkeypatch.setattr(periods, "REGIME_CHANGES", table)
        write_history(["Alexandria"])
        path = write_city("Alexandria", [make_period("Modern", 1801, None, "autocracy")],
                          country="Egypt")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["name"]["common"] = "Alexandria"
        path.write_text(json.dumps(payload), encoding="utf-8")

        assert periods.lint().flags == []

    def test_disagreeing_signatures_are_reported(self, workspace, regime_table):
        write_history(["Memphis", "Thebes (Luxor)", "Giza"])
        for name in ("Memphis", "Thebes (Luxor)"):
            write_city(name, [make_period("Republic", 1953, None, "modern republic")])
        write_city("Giza", [make_period("Republic", 1953, None, "autocracy")])

        report = periods.lint()

        assert len(report.disagreements) == 1
        assert report.disagreements[0].majority == "modern republic"


class TestAdjudication:
    def test_the_majority_is_applied_to_the_minority(self, workspace, regime_table):
        write_history(["Memphis", "Thebes (Luxor)", "Giza"])
        for name in ("Memphis", "Thebes (Luxor)"):
            write_city(name, [make_period("Republic", 1953, None, "modern republic")])
        loser = write_city("Giza", [make_period("Republic", 1953, None, "autocracy")])
        journal.start_run(config={"test": True})

        edits = periods.adjudicate(apply=True)

        assert edits == 1
        fixed = json.loads(loser.read_text(encoding="utf-8"))
        assert fixed["cityPeriodsOfHistory"][0]["typeOfGovernment"] == "modern republic"
        assert periods.lint().disagreements == []

    def test_a_tie_changes_nothing(self, workspace, regime_table):
        write_history(["Memphis", "Giza"])
        write_city("Memphis", [make_period("Republic", 1953, None, "modern republic")])
        loser = write_city("Giza", [make_period("Republic", 1953, None, "autocracy")])
        before = loser.read_text(encoding="utf-8")
        journal.start_run(config={"test": True})

        assert periods.adjudicate(apply=True) == 0
        assert loser.read_text(encoding="utf-8") == before

    def test_without_apply_nothing_is_written(self, workspace, regime_table):
        write_history(["Memphis", "Thebes (Luxor)", "Giza"])
        for name in ("Memphis", "Thebes (Luxor)"):
            write_city(name, [make_period("Republic", 1953, None, "modern republic")])
        loser = write_city("Giza", [make_period("Republic", 1953, None, "autocracy")])
        before = loser.read_text(encoding="utf-8")

        assert periods.adjudicate(apply=False) == 1
        assert loser.read_text(encoding="utf-8") == before


class TestSplit:
    def _fake_split(self, monkeypatch, replacement_periods):
        from universe_builder.providers import run as run_mod
        from universe_builder.providers.types import LLMResponse, Usage

        calls = []

        def fake_complete(request, **kwargs):
            calls.append(request)
            parsed = PeriodSplit(periods=[Population.model_validate(p)
                                          for p in replacement_periods])
            return LLMResponse(parsed.model_dump_json(), parsed, request.model,
                               Usage(input_tokens=321, output_tokens=45), request.custom_id)

        monkeypatch.setattr(run_mod, "complete", fake_complete)
        return calls

    def test_a_flagged_period_is_replaced_by_the_reply(self, workspace, regime_table,
                                                       monkeypatch):
        write_history(["Alexandria"])
        city = write_city("Alexandria", [
            make_period("Ottoman twilight", 1800, 1922, "autocracy"),
            make_period("Modern Period", 1922, None, "monarchy"),
        ])
        calls = self._fake_split(monkeypatch, [
            make_period("Kingdom of Egypt", 1922, 1953, "monarchy"),
            make_period("Republic of Egypt", 1953, None, "modern republic"),
        ])
        journal.start_run(config={"test": True})

        splits = periods.split_flagged(model=MODEL, apply=True)

        assert splits == 1 and len(calls) == 1
        data = json.loads(city.read_text(encoding="utf-8"))
        names = [p["periodName"] for p in data["cityPeriodsOfHistory"]]
        assert names == ["Ottoman twilight", "Kingdom of Egypt", "Republic of Egypt"]
        assert periods.lint().clean, "the split must satisfy the checker that ordered it"
        # The attempt is priceable: usage travels into the journal (it once
        # did not, so `costs` reported every split as unmeasured).
        split = [e for e in journal.current().entries() if e.get("operation") == "split"]
        assert split and (split[-1]["input_tokens"], split[-1]["output_tokens"]) == (321, 45)
        assert split[-1].get("estimated_cost", 0) > 0, "priced at write time"

    def test_a_failed_split_leaves_the_period_in_place(self, workspace, regime_table,
                                                       monkeypatch):
        """A refused reply must not cost the period — it stays flagged, and the
        next run tries again. The checker's re-lint reports it honestly."""
        from universe_builder.providers import run as run_mod
        from universe_builder.providers.types import LLMResponse, Usage

        write_history(["Alexandria"])
        city = write_city("Alexandria",
                          [make_period("Modern Period", 1801, None, "autocracy")])
        before = city.read_text(encoding="utf-8")
        monkeypatch.setattr(
            run_mod, "complete",
            lambda request, **kw: LLMResponse("", None, request.model, Usage(),
                                              request.custom_id, "refused"))
        journal.start_run(config={"test": True})

        assert periods.split_flagged(model=MODEL, apply=True) == 0
        assert city.read_text(encoding="utf-8") == before
        assert len(periods.lint().flags) == 1

    def test_a_reply_covering_less_than_the_span_is_refused(self, workspace,
                                                            regime_table, monkeypatch):
        """The review-gate case: sub-periods covering 1922-onward for an
        1801-onward original would silently delete 1801-1922 — and the
        survivors cross nothing, so lint would call the corpus clean."""
        write_history(["Alexandria"])
        city = write_city("Alexandria",
                          [make_period("Modern Period", 1801, None, "autocracy")])
        before = city.read_text(encoding="utf-8")
        self._fake_split(monkeypatch, [
            make_period("Kingdom of Egypt", 1922, 1953, "monarchy"),
            make_period("Republic of Egypt", 1953, None, "modern republic"),
        ])
        journal.start_run(config={"test": True})

        assert periods.split_flagged(model=MODEL, apply=True) == 0
        assert city.read_text(encoding="utf-8") == before, "shortened history was spliced in"
        assert len(periods.lint().flags) == 1, "the refused period must stay flagged"

    def test_a_gappy_reply_is_refused(self, workspace, regime_table, monkeypatch):
        write_history(["Alexandria"])
        city = write_city("Alexandria",
                          [make_period("Modern Period", 1801, None, "autocracy")])
        before = city.read_text(encoding="utf-8")
        self._fake_split(monkeypatch, [
            make_period("Khedivate", 1801, 1900, "autocracy"),        # 1900 -> 1922 gap
            make_period("Kingdom then republic", 1922, None, "monarchy"),
        ])
        journal.start_run(config={"test": True})

        assert periods.split_flagged(model=MODEL, apply=True) == 0
        assert city.read_text(encoding="utf-8") == before

    def test_a_reply_that_closes_an_open_ended_span_is_refused(self, workspace,
                                                               regime_table, monkeypatch):
        """The original runs to the present; a reply ending in 1953 deletes
        everything since."""
        write_history(["Alexandria"])
        city = write_city("Alexandria",
                          [make_period("Modern Period", 1801, None, "autocracy")])
        before = city.read_text(encoding="utf-8")
        self._fake_split(monkeypatch, [
            make_period("Khedivate and kingdom", 1801, 1953, "monarchy"),
            make_period("Early republic", 1953, 1970, "modern republic"),
        ])
        journal.start_run(config={"test": True})

        assert periods.split_flagged(model=MODEL, apply=True) == 0
        assert city.read_text(encoding="utf-8") == before

    def test_a_single_period_reply_is_refused(self, workspace, regime_table, monkeypatch):
        """One period back is not a split — and splicing it would replace the
        original's content with a paraphrase for nothing."""
        write_history(["Alexandria"])
        city = write_city("Alexandria",
                          [make_period("Modern Period", 1801, None, "autocracy")])
        before = city.read_text(encoding="utf-8")
        self._fake_split(monkeypatch, [make_period("Modern Period", 1801, None, "autocracy")])
        journal.start_run(config={"test": True})

        assert periods.split_flagged(model=MODEL, apply=True) == 0
        assert city.read_text(encoding="utf-8") == before

    def test_every_split_attempt_is_journaled(self, workspace, regime_table, monkeypatch):
        """The unchecked first run left no record of what it replaced; every
        attempt now lands in the journal, accepted or refused."""
        write_history(["Alexandria"])
        write_city("Alexandria", [make_period("Modern Period", 1801, None, "autocracy")])
        self._fake_split(monkeypatch, [
            make_period("Khedivate and kingdom", 1801, 1953, "monarchy"),
            make_period("Republic of Egypt", 1953, None, "modern republic"),
        ])
        journal.start_run(config={"test": True})

        periods.split_flagged(model=MODEL, apply=True)

        entries = [e for e in journal.current().entries() if e.get("operation") == "split"]
        assert len(entries) == 1
        assert entries[0]["validation_status"] == "ok"
        assert entries[0].get("raw_payload_available") is True

    def test_without_apply_no_call_is_made(self, workspace, regime_table, monkeypatch):
        write_history(["Alexandria"])
        write_city("Alexandria", [make_period("Modern Period", 1801, None, "autocracy")])
        calls = self._fake_split(monkeypatch, [])

        assert periods.split_flagged(model=MODEL, apply=False) == 1
        assert calls == []


def test_history_pack_registers_the_periods_operation(workspace):
    """`universe_builder periods` is the history pack's registered operation, so the
    dispatcher in cli.cmd_periods finds it under the (default) history domain."""
    from universe_builder.registry import active_pack

    spec = active_pack().operation_specs().get("periods")
    assert spec is not None and callable(spec.handler) and spec.add_arguments is not None


class TestScopingSurvivesThePayloadNameIndex:
    """`_scopes_by_city` decides which civilisation's regime table applies to a
    city, and those flags feed `--split`, which **calls a model and costs
    money**. It resolves through `topics.subject_resolver`, which gained a
    last-resort index over the names stored atoms declare about themselves
    (`stored_name_claims`). Nothing pinned this behaviour against that change.

    The ordering is what makes it safe — payload claims are consulted only
    after every existing step declines — so these assert the two halves of
    that: an answer that already existed is unchanged, and a *new* answer
    respects the civilisation boundary rather than crossing it.
    """

    def _city(self, subject, model=MODEL, payload=None):
        path = store.path_for("city", None, subject, model)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload or {"name": {"common": subject}}),
                        encoding="utf-8")
        return path

    def _history(self, name, cities):
        payload = {"people": [], "discoveries": [], "events": [], "artifacts": [],
                   "cities": cities}
        path = store.dir_for("history") / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    def test_a_listed_city_still_scopes_to_its_own_civilisation(self, workspace):
        self._history("history_h1111111111111111_egyptian_history_m.json", ["Memphis"])
        self._history("history_h2222222222222222_greek_history_m.json", ["Athens"])
        memphis = self._city("Memphis")
        athens = self._city("Athens")

        scopes = periods._scopes_by_city()

        assert scopes[store.AtomName.decompose_filename(memphis.name).padded_id] == \
            {"egyptian history"}
        assert scopes[store.AtomName.decompose_filename(athens.name).padded_id] == \
            {"greek history"}

    def test_an_atoms_own_other_name_does_not_move_it_between_civilisations(self,
                                                                            workspace):
        """The dangerous shape: a Greek city declaring a name an Egyptian
        History lists. Scoping must not follow that across the boundary, or the
        Egyptian regime table would be applied to a Greek city and the split
        loop would be told to cut periods that never existed."""
        self._history("history_h1111111111111111_egyptian_history_m.json", ["Thonis"])
        self._history("history_h2222222222222222_greek_history_m.json", ["Naukratis"])
        self._city("Thonis")
        naukratis = self._city("Naukratis",
                               payload={"name": {"common": "Naukratis",
                                                 "otherNames": ["Thonis"]}})

        scopes = periods._scopes_by_city()
        greek = store.AtomName.decompose_filename(naukratis.name).padded_id

        assert scopes[greek] == {"greek history"}

    def test_a_city_no_history_lists_gets_no_scope(self, workspace):
        """The payload index must not invent membership. Scope comes from a
        History listing the city, not from the atom naming itself."""
        self._history("history_h1111111111111111_egyptian_history_m.json", ["Memphis"])
        self._city("Memphis")
        orphan = self._city("Thonis", payload={"name": {"common": "Thonis",
                                                        "otherNames": ["Memphis"]}})

        scopes = periods._scopes_by_city()

        assert store.AtomName.decompose_filename(orphan.name).padded_id not in scopes
