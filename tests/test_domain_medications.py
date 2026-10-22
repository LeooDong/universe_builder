"""The generalisation proof: a second domain runs the same passes as history.

The medications pack (`universe_builder.domains.medications`) is a different problem with
different JSON shapes. These tests switch the engine onto it and drive the real
create and expand passes with a faked provider, asserting each atom validates
and round-trips — with NO edit to any core module. The pack is new files only;
git history for the commit that adds it shows nothing changed under `src/universe_builder`
outside `src/universe_builder/domains/`.

What is proven to generalise here: the registry derivations, the base-system
persona swap, one-shot example loading, synchronous create, and synchronous
expand (the list-of-names -> list-of-objects splice, and its one-file /
best-effort invariants). What did NOT generalise is recorded in
`test_findings_batch_expand_is_still_city_specific` — the batch expand *collect*
path is hardwired to city periods, so batch expand is history-only until the
core reads the expansion shape from the active pack.
"""

from __future__ import annotations

import json

import pytest

from universe_builder import atoms, store
from universe_builder.domains.medications import MODULE_DIR
from universe_builder.domains.medications.models import (
    Formulary,
    IndicationDetail,
    Medication,
    MedicationAppend,
    MedicationExpanded,
)
from universe_builder.models.subatomic import Name
from tests.factories import minimal_instance, minimal_payload

MODEL = "claude-sonnet-5"


@pytest.fixture
def meds(ep):
    """Switch the engine onto the medications pack, then restore history.

    Depends on `ep` so `workspace` has
    isolated the data dir. The restore runs even on failure — a leaked pack
    switch would fail every history test that runs afterwards, so the `finally`
    is load-bearing, not tidiness."""
    atoms.use_pack("medications")
    try:
        from universe_builder import execute_prompt
        yield execute_prompt
    finally:
        atoms.use_pack("history")


def _fake(ep, monkeypatch, fake_call_factory, responses, augment=False):
    Fake = fake_call_factory(responses)
    key = (MODEL, "augment") if augment else MODEL
    monkeypatch.setitem(ep.llm_call_mapping, key, Fake)
    return Fake


def _medication(**overrides) -> dict:
    return minimal_payload(Medication, **overrides)


class TestIndexExcludesTheIndexAtom:
    """The load-bearing case for index.py iterating the pack's *base* atom specs.

    Under medications the index atom is `formulary`, its `plural` is `medications`,
    and `Formulary` genuinely *has* a `medications` field. If `build()` iterated
    `atom_mapping` (which contains `formulary`) with a plural-based skip
    (`not spec.plural`), it would read `formulary.medications` and list every
    medication as a member of the formulary itself, double-counting the whole
    corpus. `build()` iterates `active_pack().atoms` instead — the base specs, which
    do not include the index atom — so the formulary is excluded by construction.
    History cannot show this (its index plural is None, skipped by the plural
    guard); medications is where iterating base specs earns its place."""

    def test_the_formulary_is_never_a_member_though_its_plural_shadows_medication(
        self, meds, workspace
    ):
        from universe_builder import index
        from universe_builder.registry import active_pack

        formulary_dir = store.dir_for("formulary")
        formulary_dir.mkdir(parents=True, exist_ok=True)
        (formulary_dir / f"formulary_0000000001_us_formulary_{MODEL}.json").write_text(
            json.dumps({"medications": ["Aspirin"]}), encoding="utf-8"
        )
        med_dir = store.dir_for("medication")
        med_dir.mkdir(parents=True, exist_ok=True)
        (med_dir / f"medication_0000000001_aspirin_{MODEL}.json").write_text(
            json.dumps(_medication()), encoding="utf-8"
        )

        report = index.build()

        index_name = active_pack().index.name  # "formulary"
        for context, per_type in report["memberships"].items():
            assert index_name not in per_type, (
                f"index atom {index_name!r} was listed as a member under {context!r}"
            )
        # And the builder did run over the formulary as a *source*: the medication
        # it lists shows up under the `medication` type, not the `formulary` type.
        listed = {m for pt in report["memberships"].values()
                  for m in pt.get("medication", {})}
        assert "Aspirin" in listed


class TestSyncPathIsPackAware:
    """The synchronous path reads the active pack for persona and examples, so a
    real (non-batch) run of a second domain is correct without threading config
    per call. These lock the two gaps that were bridged in tests by explicit
    OperationConfig before the core learned to consult the pack."""

    def test_sync_base_system_is_the_pack_persona_not_the_historian(self, meds):
        from universe_builder.execute_prompt import _base_system

        text = _base_system().lower()
        assert "pharmacolog" in text and "historian" not in text

    def test_sync_example_loader_uses_the_pack_examples_dir(self, meds):
        from universe_builder.registry import active_pack

        assert active_pack().examples_dir == MODULE_DIR / "examples"


# ---------------------------------------------------------------------------
# The registry derivations follow the pack, not history
# ---------------------------------------------------------------------------

class TestDerivations:
    def test_atom_mapping_is_the_medications_pack(self, meds):
        am = atoms.atom_mapping
        assert set(am) == {"formulary", "medication", "expand_medication", "append_medication"}
        # expand's reported class is the SEED (mirrors expand_city -> City), and
        # append's is the supplement model.
        assert am["expand_medication"].atom_class is Medication
        assert am["append_medication"].atom_class is MedicationAppend
        assert am["medication"].plural == "medications"

    def test_batch_schema_registry_covers_every_med_shape(self, meds):
        from universe_builder.providers.registry import _schema_models

        known = set(_schema_models())
        for shape in ("Medication", "MedicationExpanded", "IndicationDetail",
                      "MedicationAppend", "Formulary"):
            assert shape in known

    def test_stage_resolution_promotes_an_expanded_medication(self, meds):
        seed = _medication(indications=["Headache"])
        assert atoms.resolve_stored_class("medication", seed) is Medication
        # A stored-shape dict whose indications hold an object (as expand leaves
        # it) — built by mutation, since the seed Medication only accepts strings.
        grown = _medication(indications=["Headache"])
        grown["indications"] = [minimal_payload(IndicationDetail)]
        assert atoms.resolve_stored_class("medication", grown) is MedicationExpanded

    def test_the_persona_is_pharmacology_not_history(self, meds):
        from universe_builder.providers.requests import base_system_text

        text = base_system_text().lower()
        assert "pharmacolog" in text
        assert "historian" not in text


# ---------------------------------------------------------------------------
# minimal_instance works on the new models with no changes
# ---------------------------------------------------------------------------

class TestMinimalInstanceGeneralises:
    @pytest.mark.parametrize("model", [Medication, IndicationDetail, MedicationExpanded,
                                       MedicationAppend, Formulary])
    def test_builds_and_round_trips(self, model):
        inst = minimal_instance(model)
        assert isinstance(model.model_validate_json(inst.model_dump_json()), model)

    def test_expanded_union_builds_the_object_arm(self):
        grown = minimal_instance(MedicationExpanded)
        assert grown.is_fully_expanded()
        assert all(isinstance(e, IndicationDetail) for e in grown.indications)


# ---------------------------------------------------------------------------
# Synchronous create — the same operation path history uses
# ---------------------------------------------------------------------------

class TestCreate:
    def test_a_valid_medication_is_written(self, meds, workspace, monkeypatch, fake_call_factory):
        response = json.dumps(_medication(
            name=minimal_payload(Name, common="Aspirin"),
            indications=["Headache"],
        ))
        _fake(meds, monkeypatch, fake_call_factory, [response])

        meds.call_and_save_prompt(
            config=meds.OperationConfig(
                atom_name="medication",
                model_version=MODEL,
                base_system_text="test",
                # The synchronous path resolves examples from prompt_folder, not
                # the pack; point it at the domain so aspirin.json is found and
                # validated. (See the batch/sync example-dir finding below.)
                prompt_folder=str(MODULE_DIR),
            ),
            topic_list=["Aspirin"],
        )

        written = [p for p in store.dir_for("medication").glob("medication_*.json")
                   if not p.name.endswith("_ERROR")]
        assert len(written) == 1
        reloaded = Medication.model_validate_json(written[0].read_text())
        assert reloaded.name.common == "Aspirin"

    def test_an_invalid_medication_writes_an_error_and_no_atom(self, meds, workspace,
                                                               monkeypatch, fake_call_factory):
        _fake(meds, monkeypatch, fake_call_factory, ['{"name": "not a Name object"}'])

        meds.call_and_save_prompt(
            config=meds.OperationConfig(
                atom_name="medication", model_version=MODEL, base_system_text="test",
                prompt_folder=str(MODULE_DIR),
            ),
            topic_list=["Broken"],
        )

        folder = store.dir_for("medication")
        assert list(folder.glob("*_ERROR")), "a failure must queue a fix"
        assert [p for p in folder.glob("*.json") if not p.name.endswith("_ERROR")] == []


# ---------------------------------------------------------------------------
# Synchronous expand — list of names -> list of objects, spliced into one file
# ---------------------------------------------------------------------------

class TestExpand:
    def _seed(self, workspace, indications):
        # store.dir_for resolves the *active* (medications) domain dir; `workspace`
        # is the history dir captured before the meds fixture switched packs.
        folder = store.dir_for("medication")
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"medication_0000000001_aspirin_{MODEL}.json"
        target.write_text(json.dumps(_medication(indications=list(indications))))
        return target

    def _run(self, meds, monkeypatch, fake_call_factory, responses):
        _fake(meds, monkeypatch, fake_call_factory, responses, augment=True)
        meds.augment_and_save_prompt(meds.OperationConfig(
            atom_name="medication",
            model_version=(MODEL, "augment"),
            base_system_text="test",
            subatom_to_expand="indications",
            expander_class=IndicationDetail,
            expanded_class=MedicationExpanded,
            operation_type="expand",
            prompt_folder=str(MODULE_DIR),
        ))

    def test_every_indication_is_expanded_into_one_file(self, meds, workspace,
                                                        monkeypatch, fake_call_factory):
        target = self._seed(workspace, ["Headache", "Fever"])
        self._run(meds, monkeypatch, fake_call_factory, [
            json.dumps(minimal_payload(IndicationDetail, indicationName="Headache")),
            json.dumps(minimal_payload(IndicationDetail, indicationName="Fever")),
        ])

        reloaded = MedicationExpanded.model_validate_json(target.read_text())
        assert reloaded.is_fully_expanded()
        assert [i.indicationName for i in reloaded.indications] == ["Headache", "Fever"]
        # Stayed one file — the city-splice invariant, generalised.
        assert len([p for p in store.dir_for("medication").glob("*.json")
                    if not p.name.endswith("_ERROR")]) == 1

    def test_the_expanded_file_is_not_a_plain_medication(self, meds, workspace,
                                                         monkeypatch, fake_call_factory):
        from pydantic import ValidationError

        target = self._seed(workspace, ["Headache"])
        self._run(meds, monkeypatch, fake_call_factory,
                  [json.dumps(minimal_payload(IndicationDetail, indicationName="Headache"))])

        with pytest.raises(ValidationError):
            Medication.model_validate_json(target.read_text())

    def test_a_failed_indication_keeps_its_name(self, meds, workspace,
                                                monkeypatch, fake_call_factory):
        target = self._seed(workspace, ["Headache", "Fever"])
        # Only one response: the second call runs the queue dry and fails.
        self._run(meds, monkeypatch, fake_call_factory,
                  [json.dumps(minimal_payload(IndicationDetail, indicationName="Headache"))])

        reloaded = MedicationExpanded.model_validate_json(target.read_text())
        assert not reloaded.is_fully_expanded()
        assert reloaded.unexpanded_indications() == ["Fever"]
        assert len(reloaded.indications) == 2, "a failed entry must not be dropped"


# ---------------------------------------------------------------------------
# Batch prompt assembly generalises (build side); collect does NOT (finding)
# ---------------------------------------------------------------------------

class TestBatchPromptBuilding:
    def test_build_request_assembles_a_medication_create(self, meds):
        from universe_builder.providers.requests import build_request

        req = build_request("medication", "Aspirin", MODEL)
        # The example was loaded from the pack's dir and validated against
        # Medication; a bad example would have raised here.
        assert req.schema_model is Medication
        assert "pharmacolog" in req.system.lower()

    def test_build_request_expand_overrides_to_the_entry_model(self, meds):
        from universe_builder.providers.requests import build_request

        req = build_request("expand_medication", "Aspirin", MODEL, schema_model="IndicationDetail")
        assert req.schema_model is IndicationDetail


class TestDomainOperations:
    """`universe_builder periods` is the history pack's `periods` operation, dispatched
    through the active pack — not a core CLI verb hardcoded to cities. A domain
    that declares no such operation has no behaviour to run."""

    def test_medications_declares_no_periods_operation(self, meds):
        from universe_builder.registry import active_pack

        assert "periods" not in active_pack().operations

    def test_cmd_periods_refuses_cleanly_under_a_domain_without_it(self, meds):
        import argparse

        from universe_builder import cli

        rc = cli.cmd_periods(argparse.Namespace(apply_labels=False, split=False, model=None))
        assert rc == 2  # no periods op for medications; a clean refusal, not a crash


class TestWorkDiscoveryGeneralises:
    """Create work discovery reads the active pack's index atom, not "history":
    the index folder/prefix and the model both come from the pack, so a
    medications run fans out over a `Formulary`'s medications the way a history
    run fans out over a `History`'s cities."""

    def test_needing_create_reads_the_pack_index_atom(self, meds):
        from universe_builder import store, topics
        from universe_builder.domains.medications.models import Formulary

        form = {"medications": ["Aspirin", "Ibuprofen"]}
        Formulary.model_validate(form)
        path = store.path_for("formulary", None, "who essential medicines", MODEL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(form), encoding="utf-8")

        todo = topics.needing_create("medication")
        assert {t.subject for t in todo} == {"Aspirin", "Ibuprofen"}
        # It came from the pack's own index folder, not a "history" one.
        assert path.parent.name == "formulary"

    def test_index_does_not_emit_index_atom_self_memberships(self, meds):
        """The index atom (Formulary) is the source of the lists, not a member of
        them, so it must not appear in memberships — and because Formulary and
        Medication both use plural "medications", excluding it by the hardcoded
        name "history" (the pre-fix bug) emitted a bogus `formulary:` membership."""
        from universe_builder import index, store

        path = store.path_for("formulary", None, "essential", MODEL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"medications": ["Aspirin", "Ibuprofen"]}), encoding="utf-8")

        memberships = index.build()["memberships"]
        assert memberships, "expected at least one context of memberships"
        for per_type in memberships.values():
            assert "formulary" not in per_type, "index atom must not self-member"
        assert any("medication" in per for per in memberships.values())


class _FakeProvider:
    def __init__(self, responses):
        self._responses = responses
        self.model = MODEL
        self.supports_batch = True

    def fetch_batch_results(self, handle):
        return list(self._responses)


class TestBatchExpandGeneralises:
    """Batch expand is no longer city-only: submit fans out per unexpanded list
    entry of any expandable atom, and collect reads the atom's ExpandSpec from
    the active pack to splice the replies. This is the medications counterpart of
    tests/test_expand_batch.py, and it replaces the old finding-test that pinned
    the gap (batch.collect used to default to the city expand config)."""

    def _seed_and_handle(self, indications):
        from universe_builder import settings, store
        from universe_builder.providers import batch as batch_mod
        from universe_builder.providers.types import BatchHandle

        payload = _medication(indications=list(indications))
        path = store.path_for("medication", None, "aspirin", MODEL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
        target = str(path.relative_to(settings.data_dir()))
        routing = {
            f"expand_medication-0000{i}-aspirin": {
                "atom_type": "expand_medication", "subject": f"indication {i}",
                "operation": "expand", "target": target,
                "period_index": i, "period_name": p,
            }
            for i, p in enumerate(indications) if isinstance(p, str)
        }
        handle = BatchHandle(provider="openai", batch_id="m1", model=MODEL,
                             submitted_at="t", request_count=len(routing), routing=routing)
        hpath = batch_mod._write(
            handle, settings.batch_dir() / "20260101T000000Z-expand_medication.json")
        return path, hpath, routing

    def _response(self, custom_id, name):
        from universe_builder.providers.types import LLMResponse, Usage

        parsed = minimal_instance(IndicationDetail, indicationName=name)
        return LLMResponse(parsed.model_dump_json(), parsed, MODEL, Usage(), custom_id)

    def test_indication_replies_splice_into_one_medication(self, meds):
        from unittest.mock import patch

        from universe_builder.providers import batch as batch_mod

        path, hpath, routing = self._seed_and_handle(["headache", "fever"])
        ids = list(routing)
        responses = [self._response(ids[0], "headache"), self._response(ids[1], "fever")]

        with patch("universe_builder.providers.batch.get_provider", lambda m: _FakeProvider(responses)):
            report = batch_mod.collect(hpath)

        reloaded = MedicationExpanded.model_validate_json(path.read_text())
        assert reloaded.is_fully_expanded(), "an indication reply was not spliced in"
        assert report.written == 2
        # One file, still a medication — not overwritten by a lone IndicationDetail.
        data = json.loads(path.read_text())
        assert isinstance(data["indications"], list) and len(data["indications"]) == 2
