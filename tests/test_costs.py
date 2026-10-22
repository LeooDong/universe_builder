"""Journal-derived cost reporting.

Reads only, and reads the journal rather than the corpus, so nothing here needs
a data directory beyond the fixture's. No model is called.

The theme of these tests is that a cost report is trusted in proportion to how
boring it looks, so the ones that matter are about *provenance*: that a figure
derived from today's prices is never presented as one the provider billed, and
that an attempt which recorded no usage is never quietly counted as free.
"""

from __future__ import annotations

import json

import pytest

from universe_builder import costs


def _attempt(**over):
    event = {
        "event_type": "attempt",
        "run_id": "20260924T000000Z-aaaaaaaa",
        "timestamp_utc": "2026-09-24T00:00:00.000000Z",
        "atom_type": "city",
        "subject": "Memphis",
        "work_item_id": "city:h1111111111111111",
        "requested_model": "gpt-5-nano",
        "returned_model": "gpt-5-nano",
        "execution_mode": None,
        "batch_id": None,
        "input_tokens": 100,
        "output_tokens": 1000,
        "cache_read_tokens": 0,
        "estimated_cost": 0.001,
        "validation_status": "ok",
        "work_status": "ok",
    }
    event.update(over)
    return event


class TestProvenanceIsNeverLost:
    def test_a_stored_cost_is_used_as_is_and_not_marked_recomputed(self):
        report = costs.build(events=[_attempt(estimated_cost=0.25)])

        assert report.total.cost == pytest.approx(0.25)
        assert report.total.recomputed == 0

    def test_a_missing_cost_is_priced_from_tokens_and_marked(self):
        """63% of the journal had no stored cost when this was written."""
        report = costs.build(events=[_attempt(estimated_cost=None)])

        assert report.total.cost > 0
        assert report.total.recomputed == 1

    def test_an_attempt_with_no_usage_is_excluded_not_counted_as_free(self):
        """The failure this guards is silent: an unpriceable attempt averaged
        in at zero drags `$/call` down and looks like a cheaper model."""
        report = costs.build(events=[
            _attempt(estimated_cost=None, input_tokens=0, output_tokens=0),
            _attempt(estimated_cost=0.10),
        ])

        assert report.total.unpriceable == 1
        assert report.total.calls == 2
        # Two calls, one priceable: the average must be over the priceable one.
        assert report.total.cost_per_call == pytest.approx(0.10)

    def test_the_report_says_when_figures_were_recomputed(self):
        text = costs.format_report(costs.build(events=[_attempt(estimated_cost=None)]))

        assert "priced from tokens" in text
        assert "not an invoice" in text

    def test_the_report_says_when_attempts_were_unpriceable(self):
        text = costs.format_report(costs.build(events=[
            _attempt(estimated_cost=None, input_tokens=0, output_tokens=0)]))

        assert "floor rather than a bill" in text

    def test_a_clean_report_claims_neither(self):
        """The honesty lines must be conditional, or they become wallpaper."""
        text = costs.format_report(costs.build(events=[_attempt()]))

        assert "priced from tokens" not in text
        assert "floor rather than a bill" not in text


def _mechanical(**over):
    """An attempt line no model made, as `journal.record_mechanical` writes it."""
    event = {
        "event_type": "attempt",
        "run_id": "20261003T000000Z-mmmmmmmm",
        "timestamp_utc": "2026-10-03T00:00:00.000000Z",
        "atom_type": "city",
        "subject": "city_h1111111111111111_memphis_gpt-5-nano.json",
        "work_item_id": "city:h1111111111111111",
        "operation": "retire-list",
        "execution_mode": "mechanical",
        "raw_payload_available": False,
        "raw_payload_unavailable_reason": "mechanical edit, no model involved: retired",
        "usage_available": False,
        "validation_status": "ok",
    }
    event.update(over)
    return event


def _legacy_mechanical(reason="mechanical edit, no model involved: 'Roman Republic' -> polity"):
    """The same write as journaled before `record_mechanical` existed: a pack's
    hand-written attempt with no model, no usage, no payload and no marker."""
    event = _mechanical(raw_payload_unavailable_reason=reason)
    del event["execution_mode"]
    return event


class TestMechanicalWrites:
    """A write no model made is not a call (an end-to-end finding): the live
    corpus counted thousands of them as unpriceable sync calls and blamed them
    on the usage gap."""

    def test_a_marked_mechanical_write_is_not_a_call(self):
        report = costs.build(events=[_mechanical(), _attempt(estimated_cost=0.10)])

        assert report.total.calls == 1
        assert report.total.unpriceable == 0
        assert report.mechanical == 1

    @pytest.mark.parametrize("reason", [
        "mechanical edit, no model involved: retired to _retired/",
        "mechanical adjudication, no model involved: 'roman empire'",
        "mechanical reclassification from reclassify_review.tsv",
        "reverted to the verify-batch payload (run 20260912T000000Z-aaaaaaaa)",
        "bookkeeping correction, not an attempt: the batched fix was recorded twice",
    ])
    def test_old_unmarked_mechanical_writes_are_recognised(self, reason):
        """Every shape the live journals hold, read without rewriting them."""
        report = costs.build(events=[_legacy_mechanical(reason)])

        assert report.total.calls == 0
        assert report.mechanical == 1

    def test_a_model_call_with_no_usage_is_still_a_call(self):
        """The usage gap: a model answered, nothing was measured."""
        report = costs.build(events=[_attempt(estimated_cost=None, input_tokens=0, output_tokens=0,
                                              raw_payload_available=True)])

        assert report.total.calls == 1 and report.total.unpriceable == 1
        assert report.mechanical == 0

    def test_a_transport_failure_with_no_model_named_is_still_a_call(self):
        """A failure is not a mechanical write, whatever it failed to record."""
        failure = _legacy_mechanical("Connection error.")
        failure.update(validation_status="failed", validation_stage="transport")

        report = costs.build(events=[failure])

        assert report.total.calls == 1 and report.mechanical == 0

    def test_filters_apply_to_the_mechanical_count(self):
        report = costs.build(events=[_mechanical(), _mechanical(atom_type="person")], atom="city")

        assert report.mechanical == 1

    def test_no_grouping_gets_a_row_for_them(self):
        report = costs.build("mode", events=[_mechanical(), _attempt()])

        assert [r.key for r in report.rows] == ["sync"]

    def test_the_floor_warning_does_not_blame_them_on_the_usage_gap(self):
        text = costs.format_report(costs.build(events=[_mechanical(), _legacy_mechanical(), _attempt()]))

        assert "floor rather than a bill" not in text
        assert "2 mechanical write(s)" in text

    def test_they_are_in_the_json(self):
        payload = costs.build(events=[_mechanical(), _attempt()]).as_dict()

        assert payload["mechanical"] == 1 and payload["total"]["calls"] == 1


class TestBatchPricing:
    def test_batch_mode_is_priced_at_the_discount(self):
        """A batch attempt priced at list would overstate the bill by 2x."""
        sync = costs.build(events=[_attempt(estimated_cost=None, execution_mode="sync")])
        batch = costs.build(events=[_attempt(estimated_cost=None, execution_mode="batch")])

        assert batch.total.cost < sync.total.cost

    def test_an_unknown_execution_mode_is_priced_at_list(self):
        """Guessing the other way understates a real bill. `None` is what the
        synchronous path writes, so it must not be read as 'batch'."""
        listed = costs.build(events=[_attempt(estimated_cost=None, execution_mode=None)])
        explicit = costs.build(events=[_attempt(estimated_cost=None, execution_mode="sync")])

        assert listed.total.cost == pytest.approx(explicit.total.cost)


class TestGrouping:
    @pytest.mark.parametrize("grouping", costs.GROUPINGS)
    def test_every_advertised_grouping_works(self, grouping):
        """`--by` is a `choices=` list; an entry that raises is a broken flag."""
        report = costs.build(grouping, events=[_attempt(), _attempt(atom_type="person")])

        assert report.total.calls == 2
        assert report.rows

    def test_an_unknown_grouping_is_refused_rather_than_ignored(self):
        with pytest.raises(ValueError, match="unknown grouping"):
            costs.build("banana", events=[_attempt()])

    def test_rows_group_by_the_requested_key(self):
        report = costs.build("atom", events=[
            _attempt(atom_type="city"), _attempt(atom_type="city"),
            _attempt(atom_type="person"),
        ])

        assert {r.key: r.calls for r in report.rows} == {"city": 2, "person": 1}

    def test_rows_are_ordered_by_cost_not_by_name(self):
        report = costs.build("atom", events=[
            _attempt(atom_type="cheap", estimated_cost=0.01),
            _attempt(atom_type="dear", estimated_cost=1.00),
        ])

        assert [r.key for r in report.rows] == ["dear", "cheap"]


class TestFilters:
    def test_atom_filter_restricts_the_scan(self):
        report = costs.build(atom="person", events=[_attempt(), _attempt(atom_type="person")])

        assert report.total.calls == 1

    def test_since_compares_iso_prefixes(self):
        events = [_attempt(timestamp_utc="2026-09-20T00:00:00Z"),
                  _attempt(timestamp_utc="2026-09-24T00:00:00Z")]

        assert costs.build(since="2026-09-24", events=events).total.calls == 1
        assert costs.build(since="2026-09-01", events=events).total.calls == 2

    def test_a_failed_attempt_is_counted_and_still_costs(self):
        """The most useful column in the report: money spent producing nothing."""
        report = costs.build(events=[_attempt(validation_status="invalid")])

        assert report.total.failed == 1
        assert report.total.cost > 0


class TestReadingTheJournal:
    def test_only_attempt_events_are_read(self, tmp_path):
        run = tmp_path / "_runs" / "20260924T000000Z-aaaaaaaa"
        run.mkdir(parents=True)
        (run / "journal.jsonl").write_text(
            json.dumps(_attempt()) + "\n"
            + json.dumps({"event_type": "run_started", "run_id": "x"}) + "\n",
            encoding="utf-8")

        assert costs.build(runs_dir=tmp_path / "_runs").total.calls == 1

    def test_a_torn_final_line_is_skipped_not_fatal(self, tmp_path, caplog):
        """The journal is append-only and written during live runs, so a
        half-written last line is an expected state, not a corrupt file."""
        run = tmp_path / "_runs" / "20260924T000000Z-aaaaaaaa"
        run.mkdir(parents=True)
        (run / "journal.jsonl").write_text(
            json.dumps(_attempt()) + "\n" + '{"event_type": "att', encoding="utf-8")

        report = costs.build(runs_dir=tmp_path / "_runs")

        assert report.total.calls == 1
        assert "not JSON" in caplog.text

    def test_a_missing_runs_directory_reports_nothing_rather_than_raising(self, tmp_path):
        assert costs.build(runs_dir=tmp_path / "nope").total.calls == 0

    def test_phase_grouping_reads_run_json(self, tmp_path):
        """Attempts carry no phase — it is a property of the run."""
        run = tmp_path / "_runs" / "20260924T000000Z-aaaaaaaa"
        run.mkdir(parents=True)
        (run / "journal.jsonl").write_text(json.dumps(_attempt()) + "\n", encoding="utf-8")
        (run / "run.json").write_text(json.dumps({
            "run_id": "20260924T000000Z-aaaaaaaa", "config": {"phase": "create"}}),
            encoding="utf-8")

        report = costs.build("phase", runs_dir=tmp_path / "_runs")

        assert [r.key for r in report.rows] == ["create"]

    def test_batch_expand_is_attributed_to_expand_not_to_collect(self, tmp_path):
        """A collect run is not a phase.

        Batch attempts are written by the `collect`, hours after the submit, so
        a naive reading files every batched atom under "collect" — which was
        most attempts and most of the spend in one meaningless row.
        `expand` and `append` are recoverable because they generate under their
        own prefixed atom types.
        """
        run = tmp_path / "_runs" / "20260924T000000Z-aaaaaaaa"
        run.mkdir(parents=True)
        (run / "journal.jsonl").write_text(
            json.dumps(_attempt(atom_type="expand_city")) + "\n", encoding="utf-8")
        (run / "run.json").write_text(json.dumps({
            "run_id": "20260924T000000Z-aaaaaaaa", "config": {"command": "collect"}}),
            encoding="utf-8")

        report = costs.build("phase", runs_dir=tmp_path / "_runs")

        assert [r.key for r in report.rows] == ["expand (batch)"]

    def test_batch_create_is_labelled_ambiguous_rather_than_guessed(self, tmp_path):
        """create, fix and verify all write a base atom type with the base
        schema, so a batch-collected one cannot be told from the others. Say
        so rather than picking the likeliest."""
        run = tmp_path / "_runs" / "20260924T000000Z-aaaaaaaa"
        run.mkdir(parents=True)
        (run / "journal.jsonl").write_text(
            json.dumps(_attempt(atom_type="city")) + "\n", encoding="utf-8")
        (run / "run.json").write_text(json.dumps({
            "run_id": "20260924T000000Z-aaaaaaaa", "config": {"command": "collect"}}),
            encoding="utf-8")

        report = costs.build("phase", runs_dir=tmp_path / "_runs")

        assert [r.key for r in report.rows] == ["create/fix/verify (batch)"]

    def test_batch_work_is_attributed_to_its_recorded_operation(self, tmp_path):
        """`_journal_result` now records the operation the handle
        routed, so a batched verify is a verify — not the ambiguous bucket,
        which is left for entries journalled before that."""
        run = tmp_path / "_runs" / "20260924T000000Z-aaaaaaaa"
        run.mkdir(parents=True)
        (run / "journal.jsonl").write_text(
            json.dumps(_attempt(atom_type="city", operation="verify")) + "\n"
            + json.dumps(_attempt(atom_type="city", operation="create")) + "\n"
            + json.dumps(_attempt(atom_type="city")) + "\n", encoding="utf-8")
        (run / "run.json").write_text(json.dumps({
            "run_id": "20260924T000000Z-aaaaaaaa", "config": {"command": "collect"}}),
            encoding="utf-8")

        report = costs.build("phase", runs_dir=tmp_path / "_runs")

        assert sorted(r.key for r in report.rows) == [
            "create (batch)", "create/fix/verify (batch)", "verify (batch)"]

    def test_a_run_without_run_json_groups_as_unrecorded(self, tmp_path):
        """Runs predating `run.json` must still appear, not vanish."""
        run = tmp_path / "_runs" / "20260924T000000Z-aaaaaaaa"
        run.mkdir(parents=True)
        (run / "journal.jsonl").write_text(json.dumps(_attempt()) + "\n", encoding="utf-8")

        report = costs.build("phase", runs_dir=tmp_path / "_runs")

        assert [r.key for r in report.rows] == ["(unrecorded)"]


class TestTheCommand:
    def test_costs_runs_and_reports(self, workspace, capsys):
        from universe_builder import cli, settings

        run = settings.data_dir() / "_runs" / "20260924T000000Z-aaaaaaaa"
        run.mkdir(parents=True)
        (run / "journal.jsonl").write_text(json.dumps(_attempt()) + "\n", encoding="utf-8")

        assert cli.main(["costs"]) == 0
        assert "city" in capsys.readouterr().out

    def test_json_output_parses(self, workspace, capsys):
        from universe_builder import cli, settings

        run = settings.data_dir() / "_runs" / "20260924T000000Z-aaaaaaaa"
        run.mkdir(parents=True)
        (run / "journal.jsonl").write_text(json.dumps(_attempt()) + "\n", encoding="utf-8")

        assert cli.main(["costs", "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)

        assert payload["total"]["calls"] == 1
        assert payload["rows"][0]["key"] == "city"

    def test_a_journal_of_mechanical_writes_only_says_so(self, workspace, capsys):
        from universe_builder import cli, settings

        run = settings.data_dir() / "_runs" / "20261003T000000Z-mmmmmmmm"
        run.mkdir(parents=True)
        (run / "journal.jsonl").write_text(json.dumps(_mechanical()) + "\n", encoding="utf-8")

        assert cli.main(["costs"]) == 0
        out = capsys.readouterr().out
        assert "No attempts" in out and "1 mechanical write(s)" in out

    def test_an_empty_journal_says_so_rather_than_printing_an_empty_table(
            self, workspace, capsys):
        from universe_builder import cli

        assert cli.main(["costs"]) == 0
        assert "No attempts" in capsys.readouterr().out
