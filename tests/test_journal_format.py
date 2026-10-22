"""The journal as v0.1.19 wrote it, read back through the queue and costs.

v0.1.20 relabelled transport failures, journalled `sources --via search` and
stopped `run --batch` opening a run, and v0.2.0 gave `record_call` new fields;
none of them may make an existing `_runs/` directory
read differently. `tests/fixtures/journal_v0.1.19/_runs/` holds:

* 150 lines in the shape of a real v0.1.19 journal (every distinct line shape
  across the live runs — create/fix/verify/expand/append, sync and batch, a pack's
  own operations — and every `needs_fix`/`resolved` line with the later lines
  for the same work items), with their `run.json`s; no attempt payloads. Every
  value is replaced — subjects, ids, dates, paths, counts and notes — and
  every shape kept;
* one synthetic run written with v0.1.19's own `Journal` (fixed ids and
  times): a validation failure, a transport failure (`validation_stage:
  transport`, no payload), the fix that settles the first, a `resolution`
  line, and a batch expand failure — the shapes a real journal never wrote.

What they read as is frozen in `fixtures/journal_v0.1.19_readings.json`, derived
by running the tagged v0.1.19 (queue) and v0.3.1 (costs) code over these files,
never the code under test.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from universe_builder import costs, journal

FIXTURE = Path(__file__).parent / "fixtures" / "journal_v0.1.19" / "_runs"
#: Every real `needs_fix` line was later settled by an `ok` for the same work
#: item; only the synthetic transport failure is still queued.
READINGS = json.loads((FIXTURE.parents[1] / "journal_v0.1.19_readings.json").read_text(encoding="utf-8"))
NEEDS_FIX = READINGS["needs_fix_across_runs"]
OUTSTANDING_BY_RUN = READINGS["outstanding_by_run"]
COSTS = READINGS["costs"]


@pytest.fixture
def runs(workspace):
    shutil.copytree(FIXTURE, journal.runs_dir())
    return journal.runs_dir()


def _lines():
    for path in sorted(FIXTURE.glob("*/journal.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            yield json.loads(line)


def test_the_fixture_is_the_size_it_was_frozen_at():
    assert len(list(FIXTURE.iterdir())) == 75
    assert sum(1 for _ in _lines()) == 155


def test_every_field_written_is_one_the_reader_declares():
    known = journal.ENTRY_FIELDS | journal.DERIVED_FIELDS
    assert sorted({k for line in _lines() for k in line} - known) == []


def test_every_line_is_log_schema_version_1():
    assert journal.LOG_SCHEMA_VERSION == 1
    assert {line["log_schema_version"] for line in _lines()} == {1}


def test_the_queue_across_runs(runs):
    queue = journal.needs_fix_across_runs()
    got = [[e["run_id"], e["work_item_id"], e.get("attempt_id"), e.get("validation_stage")] for e in queue]
    assert got == NEEDS_FIX


def test_the_queue_per_run(runs):
    got = {run_id: [e["work_item_id"] for e in journal.outstanding(journal.read_entries(run_id))]
           for run_id in journal.list_runs()}
    assert {k: v for k, v in got.items() if v} == OUTSTANDING_BY_RUN


def test_a_payload_is_found_through_its_run(runs):
    synthetic = [e for e in journal.needs_fix_across_runs() if e["run_id"] == "20261003T090000Z-synth001"]
    assert [journal.raw_payload_of(e) for e in synthetic] == [None]   # the transport failure


def test_a_torn_final_line_is_skipped(runs):
    path = journal.run_dir("20261003T090000Z-synth001") / journal.JOURNAL_FILENAME
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"event_type": "attempt", "work_')
    assert len(journal.read_entries("20261003T090000Z-synth001")) == 5


def _report(grouping):
    report = costs.build(grouping)
    row = lambda r: [r.key, r.calls, r.failed, r.unpriceable, r.recomputed, r.input_tokens,  # noqa: E731
                     r.output_tokens, r.cache_read_tokens, round(r.cost, 9), sorted(r.models)]
    return {"runs": report.runs, "mechanical": report.mechanical,
            "total": row(report.total), "rows": [row(r) for r in report.rows]}


@pytest.mark.parametrize("grouping", sorted(COSTS))
def test_costs_read_the_journal_as_they_did(runs, grouping):
    assert _report(grouping) == COSTS[grouping]


def test_every_grouping_is_pinned_or_derived():
    """`effort` (v0.6.0) is the one grouping newer than the frozen readings;
    it is pinned below against `model`'s, not added to the frozen file."""
    assert set(costs.GROUPINGS) - set(COSTS) == {"effort"}


def test_a_line_from_before_effort_reads_as_its_model_at_the_default(runs):
    """No v0.1.19 line carries `sent_effort`, so `--by effort` is `--by
    model` with every key at `default`: no row moves and no cost changes."""
    expected = dict(COSTS["model"])
    expected["rows"] = [[f"{row[0]} @ default", *row[1:]] for row in COSTS["model"]["rows"]]
    got = _report("effort")
    assert sorted(got["rows"]) == sorted(expected["rows"])
    assert got["total"][1:] == expected["total"][1:]
