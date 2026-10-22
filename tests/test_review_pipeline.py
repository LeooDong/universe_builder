"""Regressions for the post-domain-refactor review, entirely offline."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from universe_builder import atoms, cli, execute_prompt as ep, journal, migrations, store
from universe_builder.domains.history.models.city import City, Population
from universe_builder.domains.medications.models import Formulary, Medication, MedicationAppend, IndicationDetail
from universe_builder.operations.expand import apply_expansions, expand_config
from universe_builder.providers import batch
from universe_builder.providers.errors import UnknownModel
from universe_builder.providers.types import BatchHandle, LLMResponse, Usage
from tests.conftest import make_population
from tests.factories import minimal_instance

MODEL = "claude-sonnet-5"


@pytest.fixture
def meds(workspace):
    atoms.use_pack("medications")
    try:
        yield
    finally:
        atoms.use_pack("history")


def test_cmd_create_skips_index_even_when_its_plural_matches(meds, monkeypatch, fake_call_factory):
    index = store.path_for("formulary", None, "Test", MODEL)
    store.write_atom(index, Formulary(medications=["Aspirin"]))
    Fake = fake_call_factory([minimal_instance(Medication).model_dump_json()])
    monkeypatch.setitem(ep.llm_call_mapping, MODEL, Fake)
    assert cli.main(["run", "--phase", "create", "--model", MODEL]) == 0
    assert [c["atom_type"] for c in Fake.calls] == ["medication"]
    assert list(store.dir_for("formulary").glob("*.json")) == [index]
    assert ep.BASE_ATOM_TYPES == ["medication"]
    assert cli._plan("create", ["formulary"], None) == []


@pytest.mark.parametrize("phase, response_model", [("expand", IndicationDetail), ("append", MedicationAppend)])
def test_cmd_augment_uses_medication_specs(meds, monkeypatch, fake_call_factory, phase, response_model):
    target = store.path_for("medication", None, "Aspirin", MODEL)
    store.write_atom(target, minimal_instance(Medication, indications=["Headache"]))
    Fake = fake_call_factory([minimal_instance(response_model).model_dump_json()])
    monkeypatch.setitem(ep.llm_call_mapping, (MODEL, "augment"), Fake)
    assert cli.main(["run", "--phase", phase, "--model", MODEL]) == 0
    assert [c["atom_type"] for c in Fake.calls] == [f"{phase}_medication"]
    if phase == "expand":
        assert isinstance(store.read_json(target)["indications"][0], dict)


def test_base_atom_list_reference_survives_switch(workspace):
    reference = ep.BASE_ATOM_TYPES
    try:
        atoms.use_pack("medications")
        assert reference == ["medication"]
    finally:
        atoms.use_pack("history")
    assert "city" in reference and "medication" not in reference


@pytest.mark.parametrize("limit", [0, 1, 2, None])
def test_payload_batch_filters_append_and_stray_files_and_honours_limit(workspace, monkeypatch, limit):
    for subject in ("A", "B", "C"):
        store.write_atom(store.path_for("city", None, subject, MODEL), minimal_instance(City))
    store.write(store.dir_for("city") / "append_city_0000000001_a_m.json", '{}')
    store.write(store.dir_for("city") / "city_notes.txt", 'notes')
    store.write(store.dir_for("city") / "notes.txt", 'notes')
    submitted = []

    def submit(atom, items, model, **kwargs):
        submitted.extend(items)
        return [workspace / "handle.json"]

    monkeypatch.setattr(batch, "submit", submit)
    args = ["run", "--phase", "verify", "--atom", "city", "--batch", "--model", MODEL]
    if limit is not None:
        args += ["--limit", str(limit)]
    assert cli.main(args) == (1 if limit == 0 else 0)
    assert len(submitted) == (3 if limit is None else limit)
    assert all(item.target.split('/')[-1].startswith("city_h") for item in submitted)


@pytest.mark.parametrize("phase", ["fix", "verify", "expand"])
def test_sync_augments_stamp_schema_version(ep, workspace, monkeypatch, fake_call_factory, phase):
    target = store.path_for("city", None, "Memphis", MODEL)
    seed = minimal_instance(City, cityPeriodsOfHistory=["Roman"])
    if phase == "fix":
        store.write_error(target, seed.model_dump_json())
    else:
        store.write_atom(target, seed)
    response = (Population.model_validate(make_population(periodName="Roman"))
                if phase == "expand" else seed)
    monkeypatch.setitem(ep.llm_call_mapping, (MODEL, "augment"), fake_call_factory([response.model_dump_json()]))
    assert ep.run_phase(phase, atoms=["city"], model=MODEL) == 0
    assert store.read_json(target)["schema_version"] == migrations.current_version("city")
    assert not store.error_path(target).exists()


def test_all_failed_sync_expand_reports_failure_without_rewriting(ep, workspace, monkeypatch, fake_call_factory):
    target = store.path_for("city", None, "Memphis", MODEL)
    store.write_atom(target, minimal_instance(City, cityPeriodsOfHistory=["Roman", "Greek"]))
    before = target.read_bytes(), target.stat().st_mtime_ns
    monkeypatch.setitem(ep.llm_call_mapping, (MODEL, "augment"), fake_call_factory(['{}', '{}']))
    assert ep.run_phase("expand", atoms=["city"], model=MODEL) == 1
    assert (target.read_bytes(), target.stat().st_mtime_ns) == before
    assert not store.error_path(target).exists()


def test_commit_fix_preserves_atom_created_since_submission(workspace):
    target = store.path_for("city", None, "Memphis", MODEL)
    store.write_error(target, 'original failure')
    store.write_atom(target, minimal_instance(City))
    before = target.read_bytes()
    store.commit_fix(store.error_path(target), minimal_instance(City, cityPeriodsOfHistory=["changed"]))
    assert target.read_bytes() == before
    assert store.read(store.error_path(target)) == 'original failure'


def test_batch_splices_wait_for_path_lock_and_read_latest(workspace):
    target = store.path_for("city", None, "Memphis", MODEL)
    store.write_atom(target, minimal_instance(City, cityPeriodsOfHistory=["Roman"]))
    started = Event()
    done = Event()

    def splice():
        started.set()
        try:
            return apply_expansions(target, [(0, "Roman", Population.model_validate(make_population()))],
                                    expand_config("city", MODEL))
        finally:
            done.set()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with store.path_lock(target):
            future = pool.submit(splice)
            assert started.wait(1)
            assert not done.wait(0.05), "splice bypassed the target lock"
            store.write_atom(target, minimal_instance(City, cityPeriodsOfHistory=["Greek"]))
        assert future.result(timeout=2) == 0
    assert store.read_json(target)["cityPeriodsOfHistory"] == ["Greek"]


def test_sync_expand_holds_lock_across_read_and_write(ep, workspace, monkeypatch):
    target = store.path_for("city", None, "Memphis", MODEL)
    seed = minimal_instance(City, cityPeriodsOfHistory=["Roman"])
    store.write_atom(target, seed)

    def expand(*args, **kwargs):
        assert store.path_lock(target).locked()
        return ep.Result.success(seed)

    monkeypatch.setattr(ep, "process_expand_operation", expand)
    ep.augment_and_save_prompt(expand_config("city", MODEL), concurrency=1)


def _handle(operation="create"):
    return BatchHandle(
        provider="anthropic", batch_id="review", model=MODEL, submitted_at="x", request_count=1,
        routing={"city-00000-memphis": {
            "atom_type": "city", "subject": "Memphis", "operation": operation,
        }},
    )


@pytest.mark.parametrize("operation,existing,queued", [
    ("create", True, False), ("fix", True, False), ("expand", False, False),
    ("verify", False, False), ("create", False, True), ("fix", False, True),
])
def test_failed_batch_queue_and_sentinel_rules(workspace, monkeypatch, operation, existing, queued):
    target = store.path_for("city", None, "Memphis", MODEL)
    if existing:
        store.write_atom(target, minimal_instance(City))
    before = target.read_bytes() if existing else None
    handle = _handle(operation)
    path = workspace / "handle.json"
    store.write_json(path, handle.to_dict())
    response = LLMResponse("bad response", None, MODEL, Usage(), "city-00000-memphis", "invalid")
    provider = MagicMock()
    provider.fetch_batch_results.return_value = [response]
    monkeypatch.setattr(batch, "get_provider", lambda model: provider)
    run = journal.start_run(config={})
    try:
        report = batch.collect(path)
        assert report.failed == 1
        assert bool(run.needs_fix()) == queued
        assert store.error_path(target).exists() == queued
        assert run.read_raw(run.entries()[0]["attempt_id"]) == "bad response"
    finally:
        journal.end_run()
    if existing:
        assert target.read_bytes() == before


@pytest.mark.parametrize("error", [UnknownModel("foreign schema"), ValueError("misrouted schema")])
@pytest.mark.parametrize("where", ["provider", "fetch"])
def test_collect_refuses_wrong_pack_without_archiving(workspace, monkeypatch, caplog, error, where):
    path = workspace / "handle.json"
    store.write_json(path, _handle().to_dict())
    provider = MagicMock()
    provider.fetch_batch_results.side_effect = error
    if where == "provider":
        monkeypatch.setattr(batch, "get_provider", MagicMock(side_effect=error))
    else:
        monkeypatch.setattr(batch, "get_provider", lambda model: provider)
    report = batch.collect(path)
    assert report.refused and not report.complete and path.exists()
    assert "UNIVERSE_BUILDER_DOMAIN" in caplog.text


@pytest.mark.parametrize("where", ["poll", "collect"])
def test_collect_all_continues_after_wrong_pack(workspace, monkeypatch, caplog, where):
    from universe_builder import run_batch

    paths = [workspace / "bad.json", workspace / "good.json"]
    monkeypatch.setattr(run_batch, "_targets", lambda args: paths)
    seen = []

    def poll(path):
        if where == "poll" and path == paths[0]:
            raise UnknownModel("wrong pack")
        return SimpleNamespace(done=True)

    def collect(path):
        if where == "collect" and path == paths[0]:
            raise ValueError("wrong schema")
        seen.append(path)
        return batch.CollectReport()

    monkeypatch.setattr(batch, "poll", poll)
    monkeypatch.setattr(batch, "collect", collect)
    assert run_batch.cmd_collect(SimpleNamespace()) == 1
    assert seen == [paths[1]]
    assert "UNIVERSE_BUILDER_DOMAIN" in caplog.text


def test_expand_uses_same_volatile_instructions_in_sync_and_batch(ep, workspace, monkeypatch, fake_call_factory):
    from universe_builder.providers.requests import build_requests

    target = store.path_for("city", None, "Memphis", MODEL)
    store.write_atom(target, minimal_instance(City, cityPeriodsOfHistory=["Roman"]))
    submitted = []

    def submit(atom, items, model, **kwargs):
        submitted.extend(items)
        return [workspace / "handle.json"]

    monkeypatch.setattr(batch, "submit", submit)
    cli.main(["run", "--phase", "expand", "--atom", "city", "--batch", "--model", MODEL])
    Fake = fake_call_factory([Population.model_validate(make_population()).model_dump_json()])
    sync_calls = []

    def call(**kwargs):
        sync_calls.append(kwargs)
        return Fake(**kwargs)

    monkeypatch.setitem(ep.llm_call_mapping, (MODEL, "augment"), call)
    ep.run_phase("expand", atoms=["city"], model=MODEL)
    instructions = sync_calls[0]["user_instructions"]
    assert submitted[0].user_instructions == instructions
    request = build_requests("expand_city", submitted, MODEL, operation="expand", schema_model="Population")[0]
    assert instructions in request.user and instructions not in request.system


@pytest.mark.parametrize("apply", [False, True])
def test_repin_is_reviewable_and_preserves_original_fingerprint(workspace, capsys, apply):
    from universe_builder import run_batch
    from universe_builder.providers.types import schema_fingerprint

    path = workspace / "handle.json"
    handle = _handle()
    previous = {**schema_fingerprint(City), "schema_sha256": "old-schema"}
    handle.extra.update(previous)
    store.write_json(path, handle.to_dict())
    before = path.read_bytes()
    assert run_batch.main(["repin", str(path)] + (["--apply"] if apply else [])) == 0
    out = capsys.readouterr().out
    assert "old-schema" in out and schema_fingerprint(City)["schema_sha256"] in out
    if apply:
        updated = batch.load(path)
        assert updated.extra["schema_pin_history"] == [previous]
        assert updated.schema_mismatch(City) is None
        assert updated.routing == handle.routing
        after = path.read_bytes()
        assert run_batch.main(["repin", str(path), "--apply"]) == 0
        assert path.read_bytes() == after
    else:
        assert path.read_bytes() == before


def test_repin_leaves_a_pre_normalisation_pin_of_the_same_schema_alone(workspace, capsys):
    """A handle pinned before `required` was sorted carries the old hash of an
    unchanged schema. Collect accepts it, so repin has nothing to accept."""
    from universe_builder import run_batch
    from universe_builder.providers.types import legacy_schema_sha256, schema_fingerprint

    path = workspace / "handle.json"
    handle = _handle()
    handle.extra.update({**schema_fingerprint(City), "schema_sha256": legacy_schema_sha256(City)})
    store.write_json(path, handle.to_dict())
    before = path.read_bytes()

    assert run_batch.main(["repin", str(path), "--apply"]) == 0
    assert "already matches" in capsys.readouterr().out
    assert path.read_bytes() == before
    assert batch.load(path).schema_mismatch(City) is None


def test_repin_wrong_domain_preserves_handle(workspace, caplog):
    from universe_builder import run_batch

    path = workspace / "handle.json"
    handle = _handle()
    handle.extra["schema_model"] = "ForeignModel"
    store.write_json(path, handle.to_dict())
    before = path.read_bytes()
    assert run_batch.main(["repin", str(path), "--apply"]) == 2
    assert path.read_bytes() == before
    assert "UNIVERSE_BUILDER_DOMAIN" in caplog.text


def test_sync_verify_skips_bad_filename_and_continues(ep, workspace, monkeypatch, fake_call_factory):
    target = store.path_for("city", None, "Memphis", MODEL)
    seed = minimal_instance(City)
    store.write_atom(target, seed)
    store.write(store.dir_for("city") / "city_00_notes.txt", 'notes')
    Fake = fake_call_factory([seed.model_dump_json()])
    monkeypatch.setitem(ep.llm_call_mapping, (MODEL, "augment"), Fake)
    assert ep.run_phase("verify", atoms=["city"], model=MODEL) == 0
    assert len(Fake.calls) == 1
