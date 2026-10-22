"""The names the history pack imports, public since v0.2.0. Each old underscore name stays as the same
object for a release, so the pack can move at its own pace."""

from __future__ import annotations

import json

from universe_builder import index, journal, store, topics


def test_the_old_names_are_the_new_objects():
    assert topics._key is topics.name_key
    assert topics._context_of is topics.context_of
    assert topics._raw_occurrences is topics.raw_occurrences
    assert topics._group_occurrences is topics.group_occurrences


def test_name_key_folds_as_before():
    assert topics.name_key("Châlons-en-Champagne") == "chalons en champagne"
    assert topics.name_key("  Al-Muʿizz li-Dīn Allāh ") == "al mu izz li din allah"
    assert topics.name_key(None) == ""


class TestMemberships:
    def _corpus(self, tmp_path, monkeypatch):
        from universe_builder import settings

        seed = tmp_path / "HISTORY_list.txt"
        seed.write_text("Egyptian History\nGreek History\n", encoding="utf-8")
        monkeypatch.setattr(settings, "HISTORY_LIST", seed)
        for line, people, cities in (("Egyptian History", ["Narmer", "Narmer"], ["Thebes (Luxor)"]),
                                     ("Greek History", ["Pericles"], ["Thebes", "Athens"])):
            path = store.path_for("history", None, line, "m")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"people": people, "cities": cities, "events": [],
                                        "artifacts": [], "discoveries": []}), encoding="utf-8")
        city = store.path_for("city", None, "Thebes (Luxor)", "m")
        city.parent.mkdir(parents=True, exist_ok=True)
        city.write_text(json.dumps({"name": {"common": "Thebes"}}), encoding="utf-8")

    def test_equals_builds_section(self, workspace, tmp_path, monkeypatch):
        self._corpus(tmp_path, monkeypatch)
        members = index.memberships()
        assert members == index.build()["memberships"]
        luxor = store.atom_id("city", "Thebes (Luxor)")
        assert members["egyptian history"]["city"] == {"Thebes (Luxor)": f"city:{luxor}"}
        assert members["egyptian history"]["person"] == {"Narmer": None}
        assert list(members) == sorted(members)
        assert members["greek history"]["city"]["Athens"] is None


class TestRecordCall:
    class _Request:
        model, system, user, max_tokens = "m", "sys", "user", 10

    def _entry(self, **kw):
        run = journal.start_run(config={"command": "t"})
        try:
            journal.record_call(self._Request(), None, error="down", operation="op",
                                atom_type="city", subject="Memphis", **kw)
        finally:
            journal.end_run()
        (entry,) = run.entries()
        return entry

    def test_target_pointer_and_schema_id_are_recorded_when_given(self, workspace):
        target = store.path_for("city", None, "Memphis", "m")
        entry = self._entry(target_pointer=target, schema_id="City")
        assert entry["target_pointer"] == target.relative_to(workspace).as_posix()
        assert entry["schema_id"] == "City"

    def test_omitted_they_leave_the_line_as_it_was(self, workspace):
        entry = self._entry()
        assert entry.get("target_pointer") is None and entry.get("schema_id") is None


class TestGroupingPieces:
    """The pieces `group_occurrences` is built from."""

    def test_ordered_keys_are_alias_keys_in_probe_order(self):
        assert topics._ordered_keys("Thebes (Luxor)") == ["thebes luxor", "thebes", "luxor"]
        for name in ("Thebes (Luxor)", "  Memphis ", "A (A)", ""):
            assert frozenset(topics._ordered_keys(name.strip())) == topics.alias_keys(name)

    def test_lookup_order_tries_each_key_once(self):
        names = ["Al-Muʿizz li-Dīn Allāh", "Al-Mu'izz li-Din Allah", "Κ", "Λ", "Cairo"]
        assert list(topics.lookup_order(names)) == [
            ("Al-Muʿizz li-Dīn Allāh", "al mu izz li din allah"), ("Κ", ""), ("Cairo", "cairo")]

    def test_a_declaration_speaks_for(self):
        line = topics._Declaration(keys=frozenset({"a"}), scope="greek history", names=("A",))
        data = topics._Declaration(keys=frozenset({"a"}), contexts={"egyptian history"})
        anywhere = topics._Declaration(keys=frozenset({"a"}), names=("A",))
        assert line.speaks_for("greek history") and not line.speaks_for(None)
        assert data.speaks_for("egyptian history") and not data.speaks_for("greek history")
        assert anywhere.speaks_for(None) and anywhere.speaks_for("x")
        assert line.is_line and not data.is_line


class TestRecordMechanical:
    """`journal.record_mechanical`: a pack's write no model made, journaled so
    provenance shows it and `costs` does not count it."""

    def _entry(self, workspace, **kw):
        target = store.path_for("city", None, "Memphis", "m")
        run = journal.start_run(config={"command": "t"})
        try:
            journal.record_mechanical("retire-list", "city", target, "retired to _retired/", **kw)
        finally:
            journal.end_run()
        (entry,) = run.entries()
        return entry, target

    def test_the_line(self, workspace):
        entry, target = self._entry(workspace)
        assert entry["event_type"] == "attempt"
        assert entry["execution_mode"] == journal.MECHANICAL
        assert entry["operation"] == "retire-list" and entry["atom_type"] == "city"
        assert entry["work_item_id"] == f"city:{store.atom_id('city', 'Memphis')}"
        assert entry["subject"] == target.name
        assert entry["target_pointer"] == target.relative_to(workspace).as_posix()
        assert entry["raw_payload_available"] is False
        assert entry["raw_payload_unavailable_reason"] == "mechanical edit, no model involved: retired to _retired/"
        assert entry["validation_status"] == "ok"
        assert "requested_model" not in entry and entry["usage_available"] is False
        assert journal.is_mechanical(entry)

    def test_kind_names_the_write(self, workspace):
        entry, _ = self._entry(workspace, kind="adjudication")
        assert entry["raw_payload_unavailable_reason"].startswith("mechanical adjudication, no model involved")

    def test_outside_a_run_it_records_nothing_and_raises_nothing(self, workspace):
        journal.record_mechanical("op", "city", store.path_for("city", None, "Memphis", "m"), "x")

    def test_a_failure_to_journal_does_not_break_the_edit(self, workspace, monkeypatch, caplog):
        def boom(**_):
            raise OSError("disk full")

        run = journal.start_run(config={"command": "t"})
        try:
            monkeypatch.setattr(run, "record_attempt", boom)
            journal.record_mechanical("op", "city", store.path_for("city", None, "Memphis", "m"), "x")
        finally:
            journal.end_run()
        assert "disk full" in caplog.text

    def test_a_model_call_is_not_mechanical(self):
        assert not journal.is_mechanical({"event_type": "attempt", "requested_model": "m",
                                          "raw_payload_available": True, "validation_status": "ok"})
