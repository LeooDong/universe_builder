"""`universe_builder.journal` — evidence preservation and the fix-pass work queue.

No network, no API key. `workspace` redirects `UNIVERSE_BUILDER_DATA_DIR` at a tmp_path, so
every run directory created here lives and dies with the test.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest
from pydantic import BaseModel, ValidationError

from universe_builder import journal, settings, store


MALFORMED = '{"name": "Memphis", "founded": 3100,\n\nSorry - here is the JSON you asked for:'


@pytest.fixture
def run(workspace):
    return journal.Journal.start(config={"model": "gpt-4o-mini"}, models={"create": "gpt-4o-mini"})


def record_failure(run, work_item_id="city:memphis", raw=MALFORMED, **kwargs):
    """A typical malformed-payload attempt, queued for the fix pass."""
    kwargs.setdefault("work_status", journal.NEEDS_FIX)
    kwargs.setdefault("validation_status", "invalid")
    kwargs.setdefault("atom_type", "city")
    return run.record_attempt(work_item_id=work_item_id, raw_payload=raw, **kwargs)


class TestRunDirectory:
    def test_runs_live_under_an_underscore_directory_so_iter_atoms_skips_them(self, run, workspace):
        assert run.directory.parent == workspace / "_runs"
        assert list(store.iter_atoms()) == []

    def test_start_writes_run_json_with_config_models_and_revision(self, run):
        meta = json.loads(run.run_path.read_text(encoding="utf-8"))

        assert meta["run_id"] == run.run_id
        assert meta["config"] == {"model": "gpt-4o-mini"}
        assert meta["models"] == {"create": "gpt-4o-mini"}
        assert meta["log_schema_version"] == journal.LOG_SCHEMA_VERSION
        assert meta["started_utc"].endswith("Z")
        assert meta["code_revision"] == journal.code_revision()

    def test_run_json_config_hash_is_stamped_on_every_entry(self, run):
        meta = json.loads(run.run_path.read_text(encoding="utf-8"))
        expected = hashlib.sha256(
            json.dumps({"model": "gpt-4o-mini"}, sort_keys=True).encode()
        ).hexdigest()

        assert meta["resolved_config_sha256"] == expected

        record_failure(run)
        assert run.entries()[0]["resolved_config_sha256"] == expected

    def test_a_supplied_run_id_is_made_filesystem_safe(self, workspace):
        run = journal.Journal.start(run_id="fix pass/2026:01")

        assert "/" not in run.directory.name
        assert run.directory.name == "fix-pass-2026-01"
        assert run.directory.is_dir()

    def test_generated_run_ids_sort_chronologically(self):
        early = journal.new_run_id(datetime(2026, 1, 1, tzinfo=timezone.utc))
        late = journal.new_run_id(datetime(2026, 6, 1, tzinfo=timezone.utc))

        assert early < late

    def test_open_reattaches_to_an_existing_run(self, run):
        record_failure(run)
        reopened = journal.Journal.open(run.run_id)

        assert reopened.directory == run.directory
        assert reopened.code_revision == run.code_revision
        assert len(reopened.entries()) == 1

    def test_open_refuses_a_run_that_does_not_exist(self, workspace):
        with pytest.raises(FileNotFoundError):
            journal.Journal.open("20260101T000000Z-deadbeef")


class TestRawPayloadPreservation:
    def test_the_raw_payload_is_preserved_when_it_is_not_valid_json(self, run):
        entry = record_failure(run)
        raw_path = run.directory / entry.raw_payload_ref

        with pytest.raises(ValueError):
            json.loads(MALFORMED)
        assert raw_path.read_text(encoding="utf-8") == MALFORMED
        assert entry.raw_payload_available is True

    def test_the_raw_payload_is_recorded_verbatim_including_prose_and_newlines(self, run):
        entry = record_failure(run, raw="```json\n{\"a\": 1}\n```\n")

        assert run.read_raw(entry.attempt_id) == "```json\n{\"a\": 1}\n```\n"

    def test_the_raw_payload_survives_a_failure_to_append_the_journal_line(self, run, monkeypatch):
        """The ordering guarantee: bytes on disk before the line that describes them."""
        monkeypatch.setattr(journal, "_append_line", _explode)

        with pytest.raises(RuntimeError):
            record_failure(run)

        written = list((run.directory / "attempts").glob("*/response.raw.txt"))
        assert [p.read_text(encoding="utf-8") for p in written] == [MALFORMED]

    def test_recording_an_attempt_without_a_payload_needs_an_explicit_reason(self, run):
        with pytest.raises(ValueError, match="raw_payload_unavailable_reason"):
            run.record_attempt(work_item_id="city:memphis", raw_payload=None)

    def test_an_absent_payload_is_recorded_as_absent_rather_than_as_empty(self, run):
        entry = run.record_attempt(
            work_item_id="city:memphis",
            raw_payload=None,
            raw_payload_unavailable_reason="connection reset before first byte",
        )
        line = run.entries()[0]

        assert entry.raw_payload_available is False
        assert "raw_payload_ref" not in line
        assert "raw_payload_sha256" not in line
        assert line["raw_payload_unavailable_reason"] == "connection reset before first byte"
        assert not (run.attempt_dir(entry.attempt_id) / journal.RAW_FILENAME).exists()

    def test_bytes_are_decoded_without_losing_the_attempt(self, run):
        entry = record_failure(run, raw=b"\xff\xfe not utf-8")

        assert run.read_raw(entry.attempt_id).endswith(" not utf-8")


class TestArtefactsAndHashes:
    def test_the_three_attempt_artefacts_are_written(self, run):
        entry = record_failure(run, request={"model": "gpt-4o-mini", "user": "Memphis"},
                               envelope={"provider_request_id": "req_1"})
        directory = run.attempt_dir(entry.attempt_id)

        assert sorted(p.name for p in directory.iterdir()) == [
            "request.json", "response-envelope.json", "response.raw.txt",
        ]

    def test_hashes_match_the_bytes_written(self, run):
        entry = record_failure(run, request={"user": "Memphis"})
        raw_bytes = (run.directory / entry.raw_payload_ref).read_bytes()
        request_bytes = (run.directory / entry.request_ref).read_bytes()

        assert entry.raw_payload_sha256 == hashlib.sha256(raw_bytes).hexdigest()
        assert entry.request_sha256 == hashlib.sha256(request_bytes).hexdigest()

    def test_artefact_refs_are_relative_to_the_run_directory(self, run):
        entry = record_failure(run, request={"user": "Memphis"})

        assert entry.raw_payload_ref == f"attempts/{entry.attempt_id}/response.raw.txt"
        assert entry.request_ref == f"attempts/{entry.attempt_id}/request.json"

    def test_the_envelope_carries_provider_metadata_usage_and_ids(self, run):
        entry = record_failure(
            run, provider="openai", requested_model="gpt-4o-mini", returned_model="gpt-4o-mini-2024",
            provider_request_id="req_abc", batch_id="batch_1", execution_mode="sync",
            input_tokens=1200, output_tokens=48,
        )
        envelope = json.loads((run.attempt_dir(entry.attempt_id) / "response-envelope.json").read_text())

        assert envelope["provider"] == "openai"
        assert envelope["provider_request_id"] == "req_abc"
        assert envelope["batch_id"] == "batch_1"
        assert envelope["returned_model"] == "gpt-4o-mini-2024"
        # Cached tokens are billable at `cached_input`, and `openai_provider._usage`
        # splits them out of `input_tokens` — an envelope carrying only input and
        # output silently drops them, which is how every cost figure came to
        # understate a corpus that caches its prompt prefix on purpose.
        assert envelope["usage"] == {
            "input_tokens": 1200,
            "output_tokens": 48,
            "cache_read_tokens": None,
            "cache_write_tokens": None,
            "estimated_cost": None,
            "available": True,
        }

    def test_a_request_holding_unserialisable_objects_still_records(self, run):
        entry = record_failure(run, request={"schema_model": BaseModel, "path": settings.REPO_ROOT})
        body = json.loads((run.directory / entry.request_ref).read_text(encoding="utf-8"))

        assert "BaseModel" in body["schema_model"]
        assert str(settings.REPO_ROOT) in body["path"]

    def test_usage_is_reported_as_unavailable_when_no_tokens_were_returned(self, run):
        entry = record_failure(run)

        assert entry.usage_available is False


class TestJournalLines:
    def test_every_line_is_one_json_object(self, run):
        record_failure(run, raw='oops\nwith an embedded newline')
        record_failure(run, work_item_id="city:thebes")

        lines = run.journal_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert all(json.loads(line)["event_type"] == journal.EVENT_ATTEMPT for line in lines)

    def test_fields_that_do_not_apply_are_omitted_rather_than_invented(self, run):
        line = record_failure(run).to_dict()

        assert "batch_id" not in line
        assert "estimated_cost" not in line
        assert "resolves_attempt_id" not in line

    def test_the_line_carries_the_full_section_2_3_field_set(self, run):
        entry = record_failure(
            run,
            phase="create", operation="generate", subject="Memphis", context="egypt_cities",
            target_pointer="city/city_0000000001_memphis_gpt-4o-mini.json",
            input_artifact_id="history_0001", input_sha256="a" * 64, source_history_id="h1",
            provider="openai", requested_model="gpt-4o-mini", returned_model="gpt-4o-mini",
            execution_mode="sync", provider_request_id="req_1", batch_id="b1",
            schema_id="City", schema_version="1", schema_sha256="b" * 64,
            validation_stage="pydantic", validation_errors=[{"loc": ["name"], "msg": "required"}],
            retryable=True, retry_action="retry", backoff_ms=500, attempt_limit=3,
            input_tokens=10, output_tokens=20, estimated_cost=0.0004,
            output_artifact_id=None, commit_status="not_committed",
            supersedes_artifact_id=None,
        )
        line = entry.to_dict()

        for group in ("phase", "input_sha256", "provider", "code_revision", "schema_sha256",
                      "raw_payload_sha256", "validation_status", "retry_action",
                      "estimated_cost", "commit_status", "work_status"):
            assert group in line

    def test_an_unknown_field_name_is_rejected(self, run):
        with pytest.raises(TypeError, match="Unknown journal field"):
            run.record_attempt(work_item_id="x", raw_payload="{}", validation_errrors=[])

    def test_measured_fields_cannot_be_supplied_by_the_caller(self, run):
        with pytest.raises(TypeError, match="measured by the journal"):
            run.record_attempt(work_item_id="x", raw_payload="{}", raw_payload_sha256="0" * 64)

    def test_a_truncated_final_line_does_not_make_the_journal_unreadable(self, run):
        record_failure(run)
        with open(run.journal_path, "a", encoding="utf-8") as f:
            f.write('{"event_sequence": 2, "work_item')

        assert len(run.entries()) == 1

    def test_timestamps_are_utc_even_when_a_naive_one_is_injected(self, run):
        entry = record_failure(run, now=datetime(2026, 3, 1, 12, 0, 0))

        assert entry.timestamp_utc == "2026-03-01T12:00:00Z"


class TestAttemptOrdering:
    def test_attempts_for_one_work_item_are_numbered_in_order(self, run):
        first = record_failure(run)
        second = record_failure(run)
        third = record_failure(run)

        assert [first.attempt_number, second.attempt_number, third.attempt_number] == [1, 2, 3]
        assert [e["attempt_id"] for e in run.attempts_for("city:memphis")] == [
            first.attempt_id, second.attempt_id, third.attempt_id,
        ]

    def test_attempt_numbering_is_per_work_item(self, run):
        record_failure(run, work_item_id="city:memphis")
        other = record_failure(run, work_item_id="city:thebes")

        assert other.attempt_number == 1

    def test_a_retry_links_back_to_the_attempt_it_follows(self, run):
        first = record_failure(run)
        second = record_failure(run)

        assert first.parent_attempt_id is None
        assert second.parent_attempt_id == first.attempt_id

    def test_numbering_continues_after_the_run_is_reopened(self, run):
        record_failure(run)
        reopened = journal.Journal.open(run.run_id)
        again = record_failure(reopened)

        assert again.attempt_number == 2
        assert again.event_sequence == 2

    def test_event_sequence_increases_monotonically(self, run):
        record_failure(run)
        record_failure(run, work_item_id="city:thebes")
        run.resolve("city:thebes", run.attempts_for("city:thebes")[0]["attempt_id"])

        assert [e["event_sequence"] for e in run.entries()] == [1, 2, 3]


class TestWorkQueue:
    def test_needs_fix_returns_the_failed_item(self, run):
        entry = record_failure(run)
        queued = run.needs_fix()

        assert [e["work_item_id"] for e in queued] == ["city:memphis"]
        assert queued[0]["attempt_id"] == entry.attempt_id
        assert queued[0]["raw_payload_ref"] == entry.raw_payload_ref

    def test_a_validated_attempt_never_enters_the_queue(self, run):
        run.record_attempt(work_item_id="city:thebes", raw_payload='{"ok": true}',
                           work_status=journal.OK, validation_status="valid")

        assert run.needs_fix() == []

    def test_needs_fix_excludes_an_item_resolved_by_a_later_line(self, run):
        entry = record_failure(run)
        assert run.needs_fix() != []

        run.resolve("city:memphis", entry.attempt_id, resolution_reason="fix pass revalidated")

        assert run.needs_fix() == []

    def test_resolution_is_an_append_and_the_original_line_survives(self, run):
        entry = record_failure(run)
        run.resolve("city:memphis", entry.attempt_id, resolution_reason="fixed")
        lines = run.entries()

        assert len(lines) == 2
        assert lines[0]["attempt_id"] == entry.attempt_id
        assert lines[0]["work_status"] == journal.NEEDS_FIX
        assert lines[1]["event_type"] == journal.EVENT_RESOLUTION
        assert lines[1]["resolves_attempt_id"] == entry.attempt_id
        assert lines[1]["resolution_reason"] == "fixed"

    def test_resolution_does_not_destroy_the_evidence(self, run):
        """B8 in structural form: a successful fix cannot delete what failed."""
        entry = record_failure(run)
        run.resolve("city:memphis", entry.attempt_id)

        assert run.read_raw(entry.attempt_id) == MALFORMED

    def test_needs_fix_returns_only_the_latest_attempt_for_an_item(self, run):
        record_failure(run)
        latest = record_failure(run)
        queued = run.needs_fix()

        assert len(queued) == 1
        assert queued[0]["attempt_id"] == latest.attempt_id
        assert queued[0]["attempt_number"] == 2

    def test_a_fix_that_fails_again_leaves_the_item_queued(self, run):
        first = record_failure(run)
        second = run.record_attempt(
            work_item_id="city:memphis", raw_payload="still not json",
            work_status=journal.NEEDS_FIX, resolves_attempt_id=first.attempt_id,
        )
        queued = run.needs_fix()

        assert [e["attempt_id"] for e in queued] == [second.attempt_id]

    def test_an_item_can_re_enter_the_queue_after_being_resolved(self, run):
        first = record_failure(run)
        run.resolve("city:memphis", first.attempt_id)
        assert run.needs_fix() == []

        third = record_failure(run)

        assert [e["attempt_id"] for e in run.needs_fix()] == [third.attempt_id]

    def test_a_resolution_naming_a_superseded_attempt_does_not_clear_the_queue(self, run):
        stale = record_failure(run)
        current = record_failure(run)
        run.resolve("city:memphis", stale.attempt_id, resolution_reason="stale fix landing late")

        assert [e["attempt_id"] for e in run.needs_fix()] == [current.attempt_id]

    def test_an_abandoned_item_leaves_the_queue_and_says_so(self, run):
        record_failure(run)
        run.resolve("city:memphis", run.attempts_for("city:memphis")[0]["attempt_id"],
                    work_status=journal.ABANDONED, resolution_reason="no such city")

        assert run.needs_fix() == []
        assert run.entries()[-1]["work_status"] == journal.ABANDONED

    def test_the_queue_holds_one_entry_per_work_item(self, run):
        record_failure(run, work_item_id="city:memphis")
        record_failure(run, work_item_id="city:thebes")
        record_failure(run, work_item_id="person:cleopatra")

        assert len(run.needs_fix()) == 3

    def test_entries_without_a_work_item_are_ignored_by_the_queue(self, run):
        run.record_attempt(work_item_id="", raw_payload="{}", work_status=journal.NEEDS_FIX)

        assert run.needs_fix() == []


class TestAcrossRuns:
    def test_the_queue_spans_runs(self, workspace):
        first = journal.Journal.start(run_id="20260101T000000Z-aaaaaaaa")
        record_failure(first, work_item_id="city:memphis")
        second = journal.Journal.start(run_id="20260102T000000Z-bbbbbbbb")
        record_failure(second, work_item_id="city:thebes")

        assert sorted(e["work_item_id"] for e in journal.needs_fix_across_runs()) == [
            "city:memphis", "city:thebes",
        ]

    def test_a_later_run_resolves_an_earlier_runs_failure(self, workspace):
        first = journal.Journal.start(run_id="20260101T000000Z-aaaaaaaa")
        broken = record_failure(first, work_item_id="city:memphis")
        second = journal.Journal.start(run_id="20260102T000000Z-bbbbbbbb")
        second.resolve("city:memphis", broken.attempt_id, resolution_reason="fixed next day")

        assert journal.needs_fix_across_runs() == []
        assert first.read_raw(broken.attempt_id) == MALFORMED

    def test_runs_are_listed_oldest_first(self, workspace):
        journal.Journal.start(run_id="20260102T000000Z-bbbbbbbb")
        journal.Journal.start(run_id="20260101T000000Z-aaaaaaaa")

        assert journal.list_runs() == [
            "20260101T000000Z-aaaaaaaa", "20260102T000000Z-bbbbbbbb",
        ]

    def test_iter_entries_walks_every_run(self, workspace):
        first = journal.Journal.start(run_id="20260101T000000Z-aaaaaaaa")
        record_failure(first)
        second = journal.Journal.start(run_id="20260102T000000Z-bbbbbbbb")
        record_failure(second, work_item_id="city:thebes")

        assert [e["work_item_id"] for e in journal.iter_entries()] == ["city:memphis", "city:thebes"]

    def test_no_runs_yet_is_an_empty_queue_not_an_error(self, workspace):
        assert journal.list_runs() == []
        assert journal.needs_fix_across_runs() == []


class TestCodeRevision:
    def test_the_revision_is_a_git_sha_or_none(self):
        revision = journal.code_revision(refresh=True)

        assert revision is None or (len(revision) == 40 and int(revision, 16) >= 0)

    def test_a_missing_git_does_not_crash_the_run(self, workspace, monkeypatch):
        def no_git(*args, **kwargs):
            raise FileNotFoundError("git")

        monkeypatch.setattr(journal.subprocess, "run", no_git)
        monkeypatch.setattr(journal, "_revision_cache", journal._REVISION_UNSET)

        assert journal.code_revision(refresh=True) is None
        assert journal.Journal.start().code_revision is None

    def test_a_non_zero_git_exit_is_not_a_revision(self, monkeypatch):
        class Failed:
            returncode = 128
            stdout = ""
            stderr = "fatal: not a git repository"

        monkeypatch.setattr(journal.subprocess, "run", lambda *a, **k: Failed())

        assert journal.code_revision(refresh=True) is None


class TestValidationErrors:
    def test_pydantic_errors_become_the_loc_type_msg_input_ctx_shape(self):
        class City(BaseModel):
            name: str
            founded: int

        with pytest.raises(ValidationError) as excinfo:
            City(founded="not a number")

        errors = journal.validation_errors(excinfo.value)
        locs = {tuple(e["loc"]) for e in errors}

        assert locs == {("name",), ("founded",)}
        assert all(set(e) == {"loc", "type", "msg", "input", "ctx"} for e in errors)
        assert json.dumps(errors)

    def test_unserialisable_error_inputs_are_coerced_not_dropped(self):
        class Wrapper(BaseModel):
            count: int

        with pytest.raises(ValidationError) as excinfo:
            Wrapper(count=object())

        errors = journal.validation_errors(excinfo.value)

        assert json.dumps(errors)
        assert "object object at" in errors[0]["input"]

    def test_structured_errors_round_trip_through_a_journal_line(self, run):
        class City(BaseModel):
            name: str

        with pytest.raises(ValidationError) as excinfo:
            City()

        entry = record_failure(run, validation_errors=journal.validation_errors(excinfo.value))
        line = run.entries()[0]

        assert line["validation_errors"][0]["loc"] == ["name"]
        assert entry.validation_errors[0]["msg"]


class TestAppendHelper:
    def test_appending_keeps_earlier_lines(self, workspace):
        path = workspace / "_runs" / "x" / journal.JOURNAL_FILENAME
        journal._append_line(path, '{"a": 1}')
        journal._append_line(path, '{"a": 2}')

        assert path.read_text(encoding="utf-8") == '{"a": 1}\n{"a": 2}\n'

    def test_a_store_write_would_have_replaced_them(self, workspace):
        """Documents why the append helper exists rather than reusing store.write."""
        path = workspace / "_runs" / "x" / journal.JOURNAL_FILENAME
        journal._append_line(path, '{"a": 1}')
        store.write(path, '{"a": 2}\n')

        assert path.read_text(encoding="utf-8") == '{"a": 2}\n'


def _explode(*args, **kwargs):
    raise RuntimeError("journal append failed")


class TestProvenance:
    """A run records which engine release and which pack distribution ran.

    `code_revision` alone was `git rev-parse HEAD` run in the engine's install
    directory. Under a wheel that is site-packages, and git answers for
    whatever repository encloses it — the history corpus keeps its venv inside
    its own checkout, so its runs recorded the *pack's* HEAD as the engine's
    revision. The git revision is now taken only from the engine's own
    checkout, and every run also records the two distributions by name and
    version, which is what identifies them under any install.
    """

    def test_run_json_names_the_engine_and_the_pack(self, workspace):
        import universe_builder
        from universe_builder.registry import active_pack

        run = journal.Journal.start()
        meta = json.loads(run.run_path.read_text(encoding="utf-8"))

        engine = meta["provenance"]["engine"]
        assert engine["distribution"] == "universe_builder"
        assert engine["version"] == universe_builder.__version__
        assert engine["git_revision"] == meta["code_revision"]
        pack = meta["provenance"]["pack"]
        assert pack["name"] == active_pack().name
        # The bundled snapshot ships inside the engine's own distribution.
        assert (pack["distribution"], pack["version"]) == ("universe_builder", universe_builder.__version__)

    def test_an_entry_point_pack_names_its_own_distribution(self, monkeypatch):
        from types import SimpleNamespace

        from universe_builder import registry

        dist = SimpleNamespace(name="mypack", version="0.2.0")
        point = SimpleNamespace(name="history", module="mypack.pack", dist=dist)
        monkeypatch.setattr(registry.metadata, "entry_points",
                            lambda group=None: [point] if group == registry.ENTRY_POINT_GROUP else [])
        monkeypatch.setitem(registry._PACK_MODULES, "history", "mypack.pack")

        origin = registry.pack_provenance()

        assert origin == {"name": "history", "module": "mypack.pack",
                          "distribution": "mypack", "version": "0.2.0"}

    def test_git_is_not_asked_outside_the_engines_own_checkout(self, monkeypatch, tmp_path):
        """site-packages inside someone else's repository is the observed case."""
        monkeypatch.setattr(journal, "_revision_cache", journal._REVISION_UNSET)
        package = tmp_path / "repo" / "venv" / "lib" / "python3.12" / "site-packages" / "universe_builder"
        package.mkdir(parents=True)
        (tmp_path / "repo" / ".git").mkdir()

        def no_git(*args, **kwargs):
            raise AssertionError("git was run outside the engine's checkout")

        monkeypatch.setattr(journal, "_engine_package_dir", lambda: package)
        monkeypatch.setattr(journal.subprocess, "run", no_git)

        assert journal.code_revision(refresh=True) is None

    def test_a_src_layout_that_is_not_the_engine_is_not_its_checkout(self, monkeypatch, tmp_path):
        package = tmp_path / "src" / "universe_builder"
        package.mkdir(parents=True)
        (tmp_path / ".git").mkdir()
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "someone_else"\n')
        monkeypatch.setattr(journal, "_engine_package_dir", lambda: package)

        assert journal._engine_checkout() is None

    def test_the_engines_checkout_is_asked(self, monkeypatch, tmp_path):
        monkeypatch.setattr(journal, "_revision_cache", journal._REVISION_UNSET)
        package = tmp_path / "src" / "universe_builder"
        package.mkdir(parents=True)
        (tmp_path / ".git").mkdir()
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "universe_builder"\n')
        seen = {}

        class Done:
            returncode = 0
            stdout = "a" * 40 + "\n"
            stderr = ""

        def git(args, cwd=None, **kwargs):
            seen["cwd"] = cwd
            return Done()

        monkeypatch.setattr(journal, "_engine_package_dir", lambda: package)
        monkeypatch.setattr(journal.subprocess, "run", git)

        assert journal.code_revision(refresh=True) == "a" * 40
        assert seen["cwd"] == str(tmp_path)

    def test_a_run_from_before_provenance_still_opens(self, workspace):
        directory = journal.run_dir("20260901T000000Z-aaaaaaaa")
        directory.mkdir(parents=True)
        (directory / journal.RUN_FILENAME).write_text(json.dumps({
            "run_id": "20260901T000000Z-aaaaaaaa", "code_revision": "b" * 40}), encoding="utf-8")

        reopened = journal.Journal.open("20260901T000000Z-aaaaaaaa")

        assert reopened.code_revision == "b" * 40
        assert journal.run_provenance("20260901T000000Z-aaaaaaaa") is None

    def test_provenance_never_costs_a_run(self, workspace, monkeypatch):
        from universe_builder import registry

        def broken():
            raise KeyError("no pack")

        monkeypatch.setattr(registry, "pack_provenance", broken)

        run = journal.Journal.start()
        meta = json.loads(run.run_path.read_text(encoding="utf-8"))
        assert meta["provenance"]["pack"] is None
        assert meta["provenance"]["engine"]["distribution"] == "universe_builder"
