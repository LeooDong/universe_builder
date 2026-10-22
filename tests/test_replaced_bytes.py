"""An in-place rewrite keeps the bytes it replaced, and `restore` puts them back.

Verify, a batch collect, expand's splice and append's merge all write over an
atom that already exists. Until this change the journal kept each call's new
reply but not the content it replaced,
so a rewrite that turned out worse had no way back: `data/` is not in git.

Now the journal line for each such write names the replaced bytes'
`replaced_sha256`, the bytes themselves sit in that attempt's folder as
`replaced.json`, `journal.replaced_bytes(entry)` reads them back (checking the
hash), and `universe_builder restore --attempt ID` writes them back through
`store`, journaling the restore as a mechanical write that saves what *it*
replaced in turn. One copy per rewrite: expand's N period calls produce one
splice line, not N copies of the city.

A write that replaces nothing — a create of a new file, a fix promoting a
sentinel to an atom that was not there — saves nothing.
"""

from __future__ import annotations

import hashlib
import json
from unittest.mock import patch

import pytest

from universe_builder import cli, journal, settings, store
from universe_builder.domains.history.models.city import CityExpanded, Population
from universe_builder.domains.history.models.discovery import Discovery
from universe_builder.providers import batch as batch_mod
from universe_builder.providers.types import BatchHandle, LLMResponse, Usage
from tests.conftest import make_population
from tests.test_operations import use_fake, valid_city, valid_discovery

MODEL = "claude-sonnet-5"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def run(workspace):
    j = journal.start_run(config={"test": True})
    yield j
    journal.end_run()


def _seed(atom_type, subject, text, model=MODEL):
    path = store.path_for(atom_type, None, subject, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    return path


def _original_discovery() -> str:
    # Hand-formatted, so a rewrite through `write_atom` can never reproduce
    # these bytes by accident.
    return json.dumps(json.loads(valid_discovery()), indent=1) + "\n"


def _improved_discovery() -> str:
    payload = json.loads(valid_discovery())
    payload["significance"] = "Rewritten by verify."
    return json.dumps(payload)


def _with_replaced(entries):
    return [e for e in entries if e.get("replaced_sha256")]


def _saved_copies():
    return sorted(journal.runs_dir().glob(f"*/attempts/*/{journal.REPLACED_FILENAME}"))


def _config(atom, operation):
    from universe_builder.config import OperationConfig

    return OperationConfig(atom_name=atom, model_version=(MODEL, "augment"),
                           base_system_text="t", operation_type=operation)


# ---------------------------------------------------------------------------
# The journal itself
# ---------------------------------------------------------------------------

class TestTheJournalKeepsTheReplacedBytes:
    def test_an_existing_target_is_saved_beside_the_attempt(self, run):
        target = _seed("discovery", "Papyrus", _original_discovery())
        before = target.read_bytes()

        entry = run.record_attempt(work_item_id="discovery:x", raw_payload="{}",
                                   target_pointer=str(target), replaces=target)

        line = run.entries()[-1]
        assert line["replaced_sha256"] == sha(before)
        assert line["replaced_ref"] == f"attempts/{entry.attempt_id}/{journal.REPLACED_FILENAME}"
        assert (run.attempt_dir(entry.attempt_id) / journal.REPLACED_FILENAME).read_bytes() == before
        assert journal.replaced_bytes(line) == before

    def test_the_pointer_is_taken_from_replaces_when_not_given(self, run):
        target = _seed("discovery", "Papyrus", _original_discovery())

        run.record_attempt(work_item_id="discovery:x", raw_payload="{}", replaces=target)

        assert journal.target_path(run.entries()[-1]) == target

    def test_a_target_that_does_not_exist_saves_nothing(self, run):
        target = store.path_for("discovery", None, "Papyrus", MODEL)

        run.record_attempt(work_item_id="discovery:x", raw_payload="{}", replaces=target)

        line = run.entries()[-1]
        assert "replaced_sha256" not in line and "replaced_ref" not in line
        assert journal.replaced_bytes(line) is None
        assert _saved_copies() == []

    def test_replaces_must_be_the_line_s_own_target(self, run):
        target = _seed("discovery", "Papyrus", _original_discovery())
        with pytest.raises(ValueError, match="target_pointer"):
            run.record_attempt(work_item_id="discovery:x", raw_payload="{}",
                               target_pointer=str(target.with_name("other.json")), replaces=target)

    def test_the_hash_is_measured_not_claimed(self, run):
        with pytest.raises(TypeError, match="measured by the journal"):
            run.record_attempt(work_item_id="x", raw_payload="{}", replaced_sha256="0" * 64)

    def test_a_tampered_copy_is_refused_loudly(self, run):
        target = _seed("discovery", "Papyrus", _original_discovery())
        entry = run.record_attempt(work_item_id="discovery:x", raw_payload="{}", replaces=target)
        (run.attempt_dir(entry.attempt_id) / journal.REPLACED_FILENAME).write_bytes(b"{}")

        with pytest.raises(ValueError, match="sha256"):
            journal.replaced_bytes(run.entries()[-1])

    def test_a_missing_copy_is_refused_loudly(self, run):
        target = _seed("discovery", "Papyrus", _original_discovery())
        entry = run.record_attempt(work_item_id="discovery:x", raw_payload="{}", replaces=target)
        (run.attempt_dir(entry.attempt_id) / journal.REPLACED_FILENAME).unlink()

        with pytest.raises(FileNotFoundError):
            journal.replaced_bytes(run.entries()[-1])

    def test_a_line_from_before_the_field_reads_as_nothing_replaced(self):
        assert journal.replaced_bytes({"run_id": "r", "attempt_id": "a"}) is None

    def test_find_attempt_spans_runs(self, run):
        entry = run.record_attempt(work_item_id="x", raw_payload="{}")
        journal.end_run()
        journal.start_run(config={"test": "second"})

        found = journal.find_attempt(entry.attempt_id)
        assert found["run_id"] == run.run_id and found["attempt_id"] == entry.attempt_id
        assert journal.find_attempt("no-such-attempt") is None

    def test_a_mechanical_write_can_save_what_it_replaces(self, run):
        target = _seed("discovery", "Papyrus", _original_discovery())
        before = target.read_bytes()

        entry = journal.record_mechanical("relabel", "discovery", target, "x", save_replaced=True)

        assert entry is not None and journal.replaced_bytes(run.entries()[-1]) == before


# ---------------------------------------------------------------------------
# The synchronous passes
# ---------------------------------------------------------------------------

class TestSyncVerify:
    def test_the_verified_atom_s_old_bytes_are_in_the_journal(self, run, ep, monkeypatch,
                                                              fake_call_factory):
        target = _seed("discovery", "Papyrus", _original_discovery())
        before = target.read_bytes()
        use_fake(ep, monkeypatch, fake_call_factory, [_improved_discovery()], augment=True)

        ep.augment_and_save_prompt(_config("discovery", "verify"))

        assert target.read_bytes() != before, "verify rewrote the atom"
        [line] = _with_replaced(run.entries())
        assert line["operation"] == "verify" and line["validation_status"] == "ok"
        assert journal.target_path(line) == target
        assert journal.replaced_bytes(line) == before
        assert len(_saved_copies()) == 1

    def test_a_failed_verify_writes_nothing_and_saves_nothing(self, run, ep, monkeypatch,
                                                              fake_call_factory):
        target = _seed("discovery", "Papyrus", _original_discovery())
        before = target.read_bytes()
        use_fake(ep, monkeypatch, fake_call_factory, ['{"still": "broken"}'], augment=True)

        ep.augment_and_save_prompt(_config("discovery", "verify"))

        assert target.read_bytes() == before
        assert _with_replaced(run.entries()) == [] and _saved_copies() == []


class TestSyncFix:
    """`commit_fix` never writes over an existing atom, so a fix replaces
    nothing and saves nothing — on either side of that rule."""

    def test_a_sentinel_promoted_to_a_new_atom_saves_nothing(self, run, ep, monkeypatch,
                                                             fake_call_factory):
        target = store.path_for("discovery", None, "Papyrus", MODEL)
        target.parent.mkdir(parents=True, exist_ok=True)
        store.error_path(target).write_text('{"incomplete": true}', encoding="utf-8")
        use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery()], augment=True)

        ep.augment_and_save_prompt(_config("discovery", "fix"))

        assert target.exists() and not store.error_path(target).exists()
        assert _with_replaced(run.entries()) == [] and _saved_copies() == []

    def test_a_stale_fix_beside_an_atom_leaves_it_and_saves_nothing(self, run, ep, monkeypatch,
                                                                    fake_call_factory):
        target = _seed("discovery", "Papyrus", _original_discovery())
        before = target.read_bytes()
        store.error_path(target).write_text('{"incomplete": true}', encoding="utf-8")
        use_fake(ep, monkeypatch, fake_call_factory, [_improved_discovery()], augment=True)

        ep.augment_and_save_prompt(_config("discovery", "fix"))

        assert target.read_bytes() == before
        assert _with_replaced(run.entries()) == [] and _saved_copies() == []


class TestSyncCreate:
    def test_a_new_atom_saves_nothing(self, run, ep, monkeypatch, fake_call_factory):
        use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery()])

        ep.create_atom(ep.OperationConfig(atom_name="discovery", model_version=MODEL,
                                          base_system_text="t"), "Papyrus")

        assert store.path_for("discovery", None, "Papyrus", MODEL).exists()
        assert _with_replaced(run.entries()) == [] and _saved_copies() == []

    def test_a_create_over_an_existing_atom_keeps_what_it_replaced(self, run, ep, monkeypatch,
                                                                   fake_call_factory):
        target = _seed("discovery", "Papyrus", _original_discovery())
        before = target.read_bytes()
        use_fake(ep, monkeypatch, fake_call_factory, [_improved_discovery()])

        ep.create_atom(ep.OperationConfig(atom_name="discovery", model_version=MODEL,
                                          base_system_text="t"), "Papyrus")

        assert target.read_bytes() != before
        [line] = _with_replaced(run.entries())
        assert journal.replaced_bytes(line) == before


class TestSyncExpand:
    def test_one_copy_of_the_city_per_splice_not_per_period(self, run, ep, monkeypatch,
                                                            fake_call_factory):
        from universe_builder.operations.expand import expand_config

        city = _seed("city", "Memphis", json.dumps(json.loads(valid_city()), indent=1))
        before = city.read_bytes()
        use_fake(ep, monkeypatch, fake_call_factory,
                 [json.dumps(make_population(periodName="Old Kingdom")),
                  json.dumps(make_population(periodName="Middle Kingdom"))], augment=True)

        ep.augment_and_save_prompt(expand_config("city", MODEL, base_system_text="t"))

        assert CityExpanded.model_validate_json(city.read_text()).is_fully_expanded()
        [line] = _with_replaced(run.entries())
        assert line["operation"] == "expand" and journal.is_mechanical(line)
        assert journal.target_path(line) == city
        assert journal.replaced_bytes(line) == before
        assert len(_saved_copies()) == 1, "two period calls, one rewrite, one copy"

    def test_an_expand_where_every_period_fails_saves_nothing(self, run, ep, monkeypatch,
                                                              fake_call_factory):
        from universe_builder.operations.expand import expand_config

        city = _seed("city", "Memphis", valid_city())
        before = city.read_bytes()
        use_fake(ep, monkeypatch, fake_call_factory, ['{"bad": 1}', '{"bad": 2}'], augment=True)

        ep.augment_and_save_prompt(expand_config("city", MODEL, base_system_text="t"))

        assert city.read_bytes() == before
        assert _saved_copies() == []


class TestSyncAppend:
    def test_the_merge_keeps_the_atom_it_rewrote(self, run, ep, workspace, monkeypatch,
                                                 fake_call_factory):
        city = _seed("city", "Memphis", json.dumps(json.loads(valid_city()), indent=1))
        before = city.read_bytes()
        use_fake(ep, monkeypatch, fake_call_factory,
                 [json.dumps({"popCulturalReferences": ["The Mummy"]})], augment=True)

        ep.augment_and_save_prompt(_config("city", "append"))
        ep.merge_helper.process_directory(str(workspace), "city")

        assert json.loads(city.read_text())["popCulturalReferences"] == ["Asterix", "The Mummy"]
        [line] = _with_replaced(run.entries())
        assert line["operation"] == "merge" and journal.is_mechanical(line)
        assert journal.replaced_bytes(line) == before
        assert len(_saved_copies()) == 1, "the append file was new; only the merge replaced"


# ---------------------------------------------------------------------------
# Batch collect
# ---------------------------------------------------------------------------

def _collect(handle, responses):
    path = batch_mod._write(handle, settings.batch_dir() / "20260101T000000Z-test.json")

    class _Provider:
        def fetch_batch_results(self, _handle):
            return list(responses)

    with patch.object(batch_mod, "get_provider", lambda m: _Provider()):
        return batch_mod.collect(path)


class TestBatchCollect:
    CID = "discovery-00000-papyrus"

    def _handle(self, target, operation):
        routing = {self.CID: {"atom_type": "discovery", "subject": "Papyrus",
                              "operation": operation,
                              "target": str(target.relative_to(settings.data_dir()))}}
        return BatchHandle(provider="openai", batch_id="b", model=MODEL, submitted_at="t",
                           request_count=1, routing=routing)

    def _ok(self):
        text = _improved_discovery()
        return LLMResponse(text, Discovery.model_validate_json(text), MODEL, Usage(1, 2), self.CID)

    def test_a_collected_verify_keeps_what_it_replaced(self, run):
        target = _seed("discovery", "Papyrus", _original_discovery())
        before = target.read_bytes()

        report = _collect(self._handle(target, "verify"), [self._ok()])

        assert report.written == 1 and target.read_bytes() != before
        [line] = _with_replaced(run.entries())
        assert line["execution_mode"] == "batch" and line["operation"] == "verify"
        assert journal.replaced_bytes(line) == before

    def test_a_collected_fix_over_an_atom_that_appeared_keeps_it(self, run):
        """The stale-repair case: the atom was regenerated after the fix was
        submitted, and collect writes the repair over it."""
        target = _seed("discovery", "Papyrus", _original_discovery())
        before = target.read_bytes()

        _collect(self._handle(target, "fix"), [self._ok()])

        [line] = _with_replaced(run.entries())
        assert journal.replaced_bytes(line) == before

    def test_a_collected_create_of_a_new_atom_saves_nothing(self, run):
        target = store.path_for("discovery", None, "Papyrus", MODEL)

        _collect(self._handle(target, "create"), [self._ok()])

        assert target.exists()
        assert _with_replaced(run.entries()) == [] and _saved_copies() == []

    def test_a_failed_result_saves_nothing(self, run):
        target = _seed("discovery", "Papyrus", _original_discovery())
        bad = LLMResponse('{"x": 1}', None, MODEL, Usage(), self.CID, "did not validate")

        _collect(self._handle(target, "verify"), [bad])

        assert _saved_copies() == []

    def test_a_collected_expand_saves_one_copy_per_city(self, run):
        periods = ["Old Kingdom", "Ptolemaic"]
        payload = json.loads(valid_city())
        payload["cityPeriodsOfHistory"] = periods
        city = _seed("city", "Memphis", json.dumps(payload, indent=1))
        before = city.read_bytes()
        target = str(city.relative_to(settings.data_dir()))
        routing = {f"expand_city-0000{i}-memphis": {
            "atom_type": "expand_city", "subject": "Memphis", "operation": "expand",
            "target": target, "period_index": i, "period_name": p} for i, p in enumerate(periods)}
        handle = BatchHandle(provider="openai", batch_id="b", model=MODEL, submitted_at="t",
                             request_count=2, routing=routing)
        responses = []
        for cid, p in zip(routing, periods):
            parsed = Population.model_validate(make_population(periodName=p))
            responses.append(LLMResponse(parsed.model_dump_json(), parsed, MODEL, Usage(), cid))

        _collect(handle, responses)

        assert CityExpanded.model_validate_json(city.read_text()).is_fully_expanded()
        [line] = _with_replaced(run.entries())
        assert line["operation"] == "expand" and journal.is_mechanical(line)
        assert journal.replaced_bytes(line) == before
        assert len(_saved_copies()) == 1


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------

class TestRestore:
    def _verified(self, run, ep, monkeypatch, fake_call_factory):
        target = _seed("discovery", "Papyrus", _original_discovery())
        before = target.read_bytes()
        use_fake(ep, monkeypatch, fake_call_factory, [_improved_discovery()], augment=True)
        ep.augment_and_save_prompt(_config("discovery", "verify"))
        [line] = _with_replaced(run.entries())
        return target, before, target.read_bytes(), line["attempt_id"]

    def test_the_round_trip(self, run, ep, monkeypatch, fake_call_factory, capsys):
        target, before, after, attempt = self._verified(run, ep, monkeypatch, fake_call_factory)
        journal.end_run()

        assert cli.main(["restore", "--attempt", attempt]) == 1      # dry by default
        assert target.read_bytes() == after
        assert cli.main(["restore", "--attempt", attempt, "--dry-run"]) == 1
        assert target.read_bytes() == after
        assert "--apply" in capsys.readouterr().out

        assert cli.main(["restore", "--attempt", attempt, "--apply"]) == 0
        assert target.read_bytes() == before
        assert isinstance(journal.current(), journal.NullJournal), "the restore's run is closed"

        # The restore is journaled as a mechanical write that kept what it
        # replaced, so it can itself be undone.
        restore_line = next(e for e in journal.iter_entries() if e.get("operation") == "restore")
        assert journal.is_mechanical(restore_line)
        assert journal.target_path(restore_line) == target
        assert journal.replaced_bytes(restore_line) == after

        assert cli.main(["restore", "--attempt", attempt, "--apply"]) == 0   # idempotent
        assert "already holds" in capsys.readouterr().out
        assert cli.main(["restore", "--attempt", restore_line["attempt_id"], "--apply"]) == 0
        assert target.read_bytes() == after, "undoing the restore"

    def test_a_restore_while_a_run_is_open_records_into_it(self, run, ep, monkeypatch,
                                                           fake_call_factory):
        from universe_builder import restore

        target, before, _after, attempt = self._verified(run, ep, monkeypatch, fake_call_factory)

        plan = restore.plan(attempt)
        restore.apply(plan)

        assert target.read_bytes() == before
        assert any(e.get("operation") == "restore" for e in run.entries())
        assert journal.current() is run

    def test_apply_refuses_without_a_run_to_keep_what_it_replaces(self, run, ep, monkeypatch,
                                                                  fake_call_factory):
        from universe_builder import restore

        target, _before, after, attempt = self._verified(run, ep, monkeypatch, fake_call_factory)
        plan = restore.plan(attempt)
        journal.end_run()

        with pytest.raises(RuntimeError, match="journal run"):
            restore.apply(plan)
        assert target.read_bytes() == after

    def test_an_attempt_that_replaced_nothing(self, run, ep, monkeypatch, fake_call_factory,
                                              capsys):
        use_fake(ep, monkeypatch, fake_call_factory, [valid_discovery()])
        ep.create_atom(ep.OperationConfig(atom_name="discovery", model_version=MODEL,
                                          base_system_text="t"), "Papyrus")
        attempt = run.entries()[-1]["attempt_id"]

        assert cli.main(["restore", "--attempt", attempt, "--apply"]) == 1
        assert "replaced nothing" in capsys.readouterr().out

    def test_an_unknown_attempt(self, workspace, capsys):
        assert cli.main(["restore", "--attempt", "nope"]) == 2
        assert "nope" in capsys.readouterr().err

    def test_a_target_outside_the_data_dir_is_refused(self, run, tmp_path):
        from universe_builder import restore

        outside = tmp_path / "elsewhere" / "discovery_h0000000000000000_x_m.json"
        outside.parent.mkdir(parents=True)
        outside.write_text("{}", encoding="utf-8")
        entry = run.record_attempt(work_item_id="x", raw_payload="{}", replaces=outside)
        outside.write_text('{"changed": true}', encoding="utf-8")

        with pytest.raises(ValueError, match="outside"):
            restore.plan(entry.attempt_id)
        assert outside.read_text(encoding="utf-8") == '{"changed": true}'


def test_the_v0_1_19_journal_reads_as_replacing_nothing(workspace):
    """Old journals carry no field; every line reads as 'nothing saved'."""
    import shutil
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "journal_v0.1.19" / "_runs"
    shutil.copytree(fixture, journal.runs_dir())
    entries = list(journal.iter_entries())
    assert len(entries) == 155
    assert all(journal.replaced_bytes(e) is None for e in entries)
