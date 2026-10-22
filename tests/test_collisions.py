"""Incoming-seed name collisions, and the verdicts that become alias lines.

No network: the adjudicator is a model call and every test here fakes it.
`conftest.no_live_calls` would fail the suite otherwise, which is the point.

The tests worth reading are the round-trip ones at the foot. `write_review`
exists only to feed `lint.apply_review`, so the contract that matters is not
"does it write a file" but "does the file this writes produce the *right alias
line, in the right direction*, through the other module's parser". A column
order silently off by one would write `Paestum = Poseidonia (Paestum)` instead
of the reverse, making the wrong spelling canonical for the whole corpus.
"""

from __future__ import annotations

import json

from universe_builder import collisions, lint, store


def _seed(**lists):
    payload = {"people": [], "discoveries": [], "events": [], "cities": [],
               "artifacts": []}
    payload.update(lists)
    return payload


def _city(subject, common=None, other_names=(), model="gpt-5-nano"):
    path = store.path_for("city", None, subject, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "name": {"common": common or subject, "otherNames": list(other_names),
                 "sources": [], "meaningOfNames": [], "longFormInfo": []},
    }), encoding="utf-8")
    return path


def _candidate(source="claim", incoming="Paestum", stored="Poseidonia"):
    return collisions.Candidate("city", incoming, stored, "city_h1_x_m.json", source)


class _Reply:
    def __init__(self, parsed):
        self.parsed = parsed


def _fake_complete(judgement):
    """A stand-in for `providers.run.complete` returning one fixed judgement."""
    def complete(request, **kw):
        return _Reply(judgement)
    return complete


class TestDetection:
    def test_an_exact_repin_is_not_a_candidate(self, workspace):
        """One topic, one content id, one file — what content addressing is
        for. 89 of Persia's 693 topics re-pinned and needed no alias."""
        _city("Memphis")

        assert collisions.find(_seed(cities=["Memphis"]), ["city"]) == []

    def test_a_stored_claim_is_a_candidate(self, workspace):
        _city("Poseidonia", other_names=["Paestum"])

        found = collisions.find(_seed(cities=["Paestum"]), ["city"])

        assert [(c.incoming, c.stored, c.source) for c in found] == [
            ("Paestum", "Poseidonia", "claim")]

    def test_a_similar_name_is_a_candidate_even_with_no_claim(self, workspace):
        """The claim index is silent whenever the stored atom never listed the
        rival spelling, which is most of the time."""
        _city("Massalia (Marseille)")

        found = collisions.find(_seed(cities=["Massilia (Marseille)"]), ["city"])

        assert [c.source for c in found] == ["similar"]

    def test_an_unrelated_name_is_not_a_candidate(self, workspace):
        _city("Memphis")

        assert collisions.find(_seed(cities=["Carthage"]), ["city"]) == []

    def test_the_same_pair_is_reported_once(self, workspace):
        """Found by both signals is still one row."""
        _city("Massalia", other_names=["Massilia"])

        assert len(collisions.find(_seed(cities=["Massilia"]), ["city"])) == 1


class TestWrongIsOnlyOfferedWhereSomethingCanBeDeleted:
    """`wrong` strips the alias from the named atom's `otherNames`.

    That is right for a claim — the corpus asserting an identity that is false,
    which is how `Narseh` came to claim `Narses`, a different man four centuries
    later. It is meaningless for a similarity candidate: nothing claimed the
    name, so there is nothing to remove, and offering `wrong` there invites an
    edit that does nothing or strips something unrelated.
    """

    def test_a_claim_candidate_may_be_wrong(self):
        assert _candidate("claim").can_be_wrong

    def test_a_similarity_candidate_may_not(self):
        assert not _candidate("similar").can_be_wrong

    def test_a_confident_difference_on_a_claim_is_wrong(self, monkeypatch):
        from universe_builder.providers import run as provider_run

        judgement = collisions.Judgement(
            same_subject=False, confident=True, reason="different men")
        monkeypatch.setattr(provider_run, "complete", _fake_complete(judgement))
        candidate = _candidate("claim")

        verdict, _why = collisions.adjudicate([candidate], "m")[candidate]

        assert verdict == "wrong"

    def test_a_confident_difference_on_a_similarity_is_skip_not_wrong(self, monkeypatch):
        """Pantheon is not the Parthenon, but no atom claimed otherwise, so
        there is nothing to delete — only a note to create them separately."""
        from universe_builder.providers import run as provider_run

        judgement = collisions.Judgement(
            same_subject=False, confident=True, reason="different buildings")
        monkeypatch.setattr(provider_run, "complete", _fake_complete(judgement))
        candidate = _candidate("similar")

        verdict, why = collisions.adjudicate([candidate], "m")[candidate]

        assert verdict == "skip"
        assert "different subjects" in why


class TestAdjudication:
    def test_a_confident_match_is_same(self, monkeypatch):
        from universe_builder.providers import run as provider_run

        judgement = collisions.Judgement(
            same_subject=True, confident=True, reason="Greek and Latin forms")
        monkeypatch.setattr(provider_run, "complete", _fake_complete(judgement))
        candidate = _candidate()

        assert collisions.adjudicate([candidate], "m")[candidate][0] == "same"

    def test_an_unsure_model_abstains_rather_than_guessing(self, monkeypatch):
        """Seen live: `Narses` (Byzantine general) against
        `Narseh` (Sasanian shah) is the hardest pair in the Roman set, and the
        model declined it. Abstention must reach the file as `skip`, never as a
        quiet `same`."""
        from universe_builder.providers import run as provider_run

        judgement = collisions.Judgement(
            same_subject=True, confident=False, reason="may be two men")
        monkeypatch.setattr(provider_run, "complete", _fake_complete(judgement))
        candidate = _candidate()

        verdict, why = collisions.adjudicate([candidate], "m")[candidate]

        assert verdict == "skip"
        assert "unsure" in why

    def test_a_failed_call_leaves_the_row_undecided_and_keeps_going(self, monkeypatch):
        """One bad pair must not lose the other forty-eight verdicts."""
        from universe_builder.providers import run as provider_run

        def boom(request, **kw):
            raise RuntimeError("transport died")

        monkeypatch.setattr(provider_run, "complete", boom)
        first, second = _candidate(incoming="A"), _candidate(incoming="B")

        verdicts = collisions.adjudicate([first, second], "m")

        assert verdicts[first][0] == "?"
        assert len(verdicts) == 2

    def test_an_unparsable_reply_is_undecided(self, monkeypatch):
        from universe_builder.providers import run as provider_run

        monkeypatch.setattr(provider_run, "complete", _fake_complete("not a judgement"))
        candidate = _candidate()

        assert collisions.adjudicate([candidate], "m")[candidate][0] == "?"


class TestTheReviewFileRoundTrips:
    """The contract with `lint`, which is where alias lines are actually written."""

    def test_lint_can_parse_what_collisions_writes(self, tmp_path):
        path = tmp_path / "review.tsv"
        candidate = _candidate()
        collisions.write_review([candidate], path, {candidate: ("same", "Greek form")})

        rows = lint.read_review(path)

        assert len(rows) == 1
        assert rows[0][0] == "same"

    def test_the_alias_line_makes_the_stored_name_canonical(self, tmp_path):
        """Direction is load-bearing: the first name on an alias line is the
        spelling the work list adopts. The stored atom already exists under its
        name, so that name must lead — otherwise the corpus renames itself to
        the incoming spelling and every existing file becomes the alias."""
        path = tmp_path / "review.tsv"
        candidate = collisions.Candidate(
            "city", "Paestum", "Poseidonia (Paestum)", "city_h1_p_m.json", "claim")
        collisions.write_review([candidate], path, {candidate: ("same", "")})

        outcome = lint.apply_review(lint.read_review(path),
                                    aliases_path=tmp_path / "aliases.txt",
                                    dry_run=True)

        assert outcome["alias_lines"] == ["Poseidonia (Paestum) = Paestum"]

    def test_an_undecided_row_writes_no_alias(self, tmp_path):
        path = tmp_path / "review.tsv"
        candidate = _candidate()
        collisions.write_review([candidate], path)

        outcome = lint.apply_review(lint.read_review(path),
                                    aliases_path=tmp_path / "aliases.txt",
                                    dry_run=True)

        assert outcome["alias_lines"] == []

    def test_every_verdict_written_is_one_lint_accepts(self, monkeypatch, tmp_path):
        """`read_review` refuses the whole file on an unrecognised verdict, so
        a vocabulary drift here would make every row unusable at once."""
        from universe_builder.providers import run as provider_run

        seen = set()
        for same, confident, source in [(True, True, "claim"), (False, True, "claim"),
                                        (False, True, "similar"), (True, False, "claim")]:
            judgement = collisions.Judgement(
                same_subject=same, confident=confident, reason="r")
            monkeypatch.setattr(provider_run, "complete", _fake_complete(judgement))
            candidate = _candidate(source)
            seen.add(collisions.adjudicate([candidate], "m")[candidate][0])

        assert seen <= {"same", "wrong", "skip", "?"}, seen

    def test_a_tab_in_a_reason_cannot_break_the_columns(self, monkeypatch, tmp_path):
        """The format is tab-separated; a reason containing a tab would shift
        every later column and silently change which atom is edited."""
        from universe_builder.providers import run as provider_run

        judgement = collisions.Judgement(
            same_subject=True, confident=True, reason="a\tb\tc")
        monkeypatch.setattr(provider_run, "complete", _fake_complete(judgement))
        candidate = _candidate()
        verdicts = collisions.adjudicate([candidate], "m")

        path = tmp_path / "review.tsv"
        collisions.write_review([candidate], path, verdicts)
        rows = lint.read_review(path)

        assert rows[0][3] == "Paestum"
        assert rows[0][4] == "Poseidonia"


class TestTheCommand:
    def test_a_missing_seed_is_refused_with_advice(self, workspace, caplog):
        from universe_builder import cli

        assert cli.main(["collisions", "--history", "Roman History"]) == 2
        assert "Run `--phase index` first" in caplog.text

    def test_no_candidates_says_so(self, workspace, capsys):
        from universe_builder import cli

        path = store.dir_for("history") / "history_h1111111111111111_roman_history_m.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_seed(cities=["Carthage"])), encoding="utf-8")
        _city("Memphis")

        assert cli.main(["collisions", "--history", "Roman History"]) == 0
        assert "No collision candidates" in capsys.readouterr().out


class TestAdjudicationIsJournalled:
    """`collisions --via model` is paid, and it once left no record:
    `adjudicate` called `provider_run.complete` directly and `cmd_collisions`
    opened no run, so `costs` could not see a cent of it — while the docstring
    claimed each call journalled under its own subject."""

    def _reply(self, judgement):
        from universe_builder.providers.types import LLMResponse, Usage

        return LLMResponse('{"same_subject": true}', judgement, "gpt-5-nano",
                           Usage(input_tokens=120, output_tokens=30))

    def test_each_call_is_recorded_with_its_usage(self, workspace, monkeypatch):
        from universe_builder import journal
        from universe_builder.providers import run as provider_run

        judgement = collisions.Judgement(same_subject=True, confident=True, reason="r")
        monkeypatch.setattr(provider_run, "complete", lambda request, **kw: self._reply(judgement))
        run = journal.start_run(config={"command": "collisions"})
        try:
            collisions.adjudicate([_candidate(), _candidate(incoming="Neapolis")], "gpt-5-nano")
        finally:
            journal.end_run()

        entries = run.entries()
        assert [e["subject"] for e in entries] == ["Paestum", "Neapolis"]
        assert {e["operation"] for e in entries} == {"alias-adjudication"}
        assert all(e["input_tokens"] == 120 and e["output_tokens"] == 30 for e in entries)
        assert all(e["requested_model"] == "gpt-5-nano" for e in entries)
        assert run.read_raw(entries[0]["attempt_id"]) == '{"same_subject": true}'
        assert run.needs_fix() == [], "an adjudication queues no repair"

    def test_a_failed_call_is_recorded_too(self, workspace, monkeypatch):
        from universe_builder import journal
        from universe_builder.providers import run as provider_run

        def boom(request, **kw):
            raise ConnectionError("down")

        monkeypatch.setattr(provider_run, "complete", boom)
        run = journal.start_run(config={"command": "collisions"})
        try:
            collisions.adjudicate([_candidate()], "gpt-5-nano")
        finally:
            journal.end_run()

        (entry,) = run.entries()
        assert entry["raw_payload_available"] is False
        assert "down" in entry["raw_payload_unavailable_reason"]

    def test_the_command_opens_a_run_costs_can_read(self, workspace, monkeypatch, capsys):
        from universe_builder import cli, costs, journal
        from universe_builder.providers import run as provider_run

        seed = store.dir_for("history") / "history_h1111111111111111_roman_history_m.json"
        seed.parent.mkdir(parents=True, exist_ok=True)
        seed.write_text(json.dumps(_seed(cities=["Paestum"])), encoding="utf-8")
        _city("Poseidonia", other_names=["Paestum"])
        judgement = collisions.Judgement(same_subject=True, confident=True, reason="r")
        monkeypatch.setattr(provider_run, "complete", lambda request, **kw: self._reply(judgement))

        try:
            assert cli.main(["collisions", "--seed", "Roman History", "--via", "model",
                             "--model", "gpt-5-nano"]) == 0
        finally:
            journal.end_run()

        report = costs.build("phase")
        assert [r.key for r in report.rows] == ["collisions"]
        assert report.total.calls == 1 and report.total.input_tokens == 120


class TestRepinsAreListed:
    """`find` skips a topic whose id is on disk: a re-pin, usually one subject.
    It is also where one bare name meaning two things hides (a papal `Leo I`
    landing on the Byzantine emperor), so the command lists every re-pin with
    the other Histories that name it, for a person to confirm."""

    def _history(self, line, **lists):
        path = store.path_for("history", None, line, "m")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(_seed(**lists)), encoding="utf-8")
        return path

    def test_a_re_pin_is_listed_with_the_histories_that_name_it(self, workspace):
        self._history("Byzantine History", cities=["Abydos"])
        new = self._history("Papal History", cities=["Abydos", "Avignon"])
        _city("Abydos")

        shared = collisions.shared(json.loads(new.read_text()), ["city"], seed_file=new.name)

        assert [(t, others) for _, t, _, others in shared] == [("Abydos", ["byzantine history"])]

    def test_the_command_prints_them_and_finds_a_hyphenated_seed(self, workspace, capsys, tmp_path, monkeypatch):
        from universe_builder import cli, settings

        seed = tmp_path / "seed.txt"
        seed.write_text("Byzantine History\nAnglo-Saxon History\n", encoding="utf-8")
        monkeypatch.setattr(settings, "HISTORY_LIST", seed)
        self._history("Byzantine History", cities=["Abydos"])
        self._history("Anglo-Saxon History", cities=["Abydos"])
        _city("Abydos")

        assert cli.main(["collisions", "--seed", "Anglo-Saxon History"]) == 0
        out = capsys.readouterr().out
        assert "1 topic(s) re-pin onto stored atoms" in out
        assert "Abydos" in out and "also in: byzantine history" in out
