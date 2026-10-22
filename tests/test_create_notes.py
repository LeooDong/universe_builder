"""Create notes: one sentence for one subject's prompt, where a listed name means
something lesser-known than it suggests.

Found in the history corpus: "Later Han dynasty" (947–951) was
generated as the Eastern Han, "Britannic Empire" (Carausius, 286–296) as the
modern British Empire. The prompt carries only the bare name and the context,
so the model takes the famous reading.
"""

import json
from unittest.mock import patch

import pytest

from universe_builder import cli, settings, store, topics
from universe_builder.providers.requests import BatchItem, build_requests, with_create_note

NOTES = """\
# comment
[Egyptian History]
city     | Thebes | The Egyptian Thebes (Waset, modern Luxor), not the Greek city.
*        | Abydos | The Egyptian Abydos in Upper Egypt, not the Hellespont city.
[*]
city     | Thebes | Anywhere else: the global line.
artifact | Heracles | The Farnese Heracles statue, not the hero.
"""


def _write(tmp_path, text, name="notes.txt"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


# --- the file -------------------------------------------------------------------

def test_the_most_specific_line_wins(tmp_path):
    notes = topics.load_create_notes(_write(tmp_path, NOTES))

    note = topics.create_note("city", "Thebes", "egyptian history", notes)
    assert note.startswith("The Egyptian Thebes")
    assert topics.create_note("city", "THEBES", "Egyptian History", notes).startswith("The Egyptian")
    assert topics.create_note("city", "Thebes", "Greek History", notes).startswith("Anywhere")
    assert topics.create_note("city", "Thebes", None, notes).startswith("Anywhere")
    assert topics.create_note("person", "Abydos", "Egyptian History", notes).startswith("The Egyptian Abydos")


def test_a_note_follows_its_type_and_subject_only(tmp_path):
    notes = topics.load_create_notes(_write(tmp_path, NOTES))

    assert topics.create_note("person", "Heracles", None, notes) is None, "the hero keeps a bare prompt"
    assert topics.create_note("artifact", "Heracles", None, notes).startswith("The Farnese")
    assert topics.create_note("city", "Memphis", "Egyptian History", notes) is None
    assert topics.create_note("city", "Abydos", "Greek History", notes) is None


@pytest.mark.parametrize("line, message", [
    ("city | Thebes", "expected `type | subject | note`"),
    ("city | Thebes | ", "expected `type | subject | note`"),
    ("temple | Thebes | x", "unknown atom type 'temple'"),
    ("city | Thebes | x\ncity | THEBES | y", "a second note"),
])
def test_a_bad_file_stops_rather_than_dropping_a_note(tmp_path, line, message):
    with pytest.raises(topics.CreateNotesError, match=message):
        topics.load_create_notes(_write(tmp_path, line))


def test_a_declared_file_that_is_missing_is_an_error(tmp_path):
    with pytest.raises(topics.CreateNotesError, match="cannot read"):
        topics.load_create_notes(tmp_path / "nowhere.txt")


def test_a_pack_that_declares_no_file_has_no_notes(monkeypatch):
    monkeypatch.setattr(settings, "CREATE_NOTES", None)
    assert topics.load_create_notes() == {}  # the bundled snapshot declares none


# --- the synchronous path -------------------------------------------------------

def test_create_sends_the_note_after_the_subject(ep, workspace, monkeypatch, fake_call_factory):
    monkeypatch.setattr(settings, "CREATE_NOTES", _write(workspace, NOTES))
    Fake = fake_call_factory(['{"not": "valid"}', '{"not": "valid"}'])
    monkeypatch.setitem(ep.llm_call_mapping, "claude-sonnet-5", Fake)
    config = ep.OperationConfig(atom_name="city", model_version="claude-sonnet-5",
                                base_system_text="test", context="egyptian history")

    ep.call_and_save_prompt(config=config, topic_list=["Thebes", "Memphis"], concurrency=1)

    assert Fake.calls[0]["user_note"].startswith("The Egyptian Thebes")
    assert "user_note" not in Fake.calls[1], "a subject without a note is called exactly as before"


def test_the_provider_call_puts_the_note_in_the_user_turn_only():
    """Through the batch builder since the prompt-builder merge: the note follows the subject
    in the user turn, exactly as `with_create_note` sends it in a batch."""
    from universe_builder.atoms import atom_mapping
    from universe_builder.domains.history.models.city import City
    from universe_builder.llm_classes.provider_call import ProviderCall

    call = ProviderCall.__new__(ProviderCall)
    call.atom_type, call.atom_class, call.model_version = "city", City, "gpt-5-nano"
    call.prompt_subject, call.prompt_operation, call.context = "Later Han dynasty", None, "Chinese History"
    call.max_tokens, call.temperature = 1000, None
    call.user_note = "The Five Dynasties Later Han."
    start = atom_mapping["city"].prompt_start_text
    request = call.build_request()
    assert request.user == f"{start}'Later Han dynasty'\n\nThe Five Dynasties Later Han."
    assert "Five Dynasties" not in request.system
    assert "The context is Chinese History." in request.system
    call.user_note = None
    assert call.build_request().user == f"{start}'Later Han dynasty'"


# --- the batch path -------------------------------------------------------------

def test_a_batch_request_carries_the_note_in_the_user_turn(tmp_path):
    notes = topics.load_create_notes(_write(tmp_path, NOTES))
    items = [with_create_note("city", s, "egyptian history", notes) for s in ("Thebes", "Memphis")]

    assert isinstance(items[0], BatchItem) and items[1] == "Memphis"
    noted, bare = build_requests("city", items, "gpt-5-nano", context="egyptian history")
    assert noted.user.endswith("The Egyptian Thebes (Waset, modern Luxor), not the Greek city.")
    assert noted.system == bare.system, "the cached prefix is shared by noted and bare requests"
    assert "Waset" not in bare.user


def test_run_create_batch_submits_noted_subjects(workspace, monkeypatch):
    monkeypatch.setattr(settings, "CREATE_NOTES", _write(workspace, NOTES))
    path = store.dir_for("history") / "history_h1111111111111111_egyptian_history_m.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"people": [], "discoveries": [], "events": [], "artifacts": [],
                                "cities": ["Thebes", "Memphis"]}), encoding="utf-8")
    seen = []

    def fake_submit(atom, subjects, model, *, context=None, **kw):
        seen.extend(subjects)
        return [store.dir_for("history") / "handle.json"]

    with patch("universe_builder.providers.batch.submit", fake_submit):
        assert cli.main(["run", "--phase", "create", "--atom", "city", "--batch", "--model", "gpt-5-nano"]) == 0

    noted = [s for s in seen if isinstance(s, BatchItem)]
    assert [(s.subject, s.user_instructions[:18]) for s in noted] == [("Thebes", "The Egyptian Thebe")]
    assert "Memphis" in seen


def test_the_sync_create_pass_hands_each_context_to_its_config(ep, workspace, monkeypatch):
    path = store.dir_for("history") / "history_h1111111111111111_egyptian_history_m.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"people": [], "discoveries": [], "events": [], "artifacts": [],
                                "cities": ["Thebes"]}), encoding="utf-8")
    seen = []
    monkeypatch.setattr(ep, "call_and_save_prompt", lambda config, topic_list, **kw: seen.append(config.context))

    ep._run_create("city", "claude-sonnet-5")

    assert seen == ["egyptian history"], "without the context a scoped note is never found"


def test_run_batch_submit_attaches_notes_for_its_context(workspace, monkeypatch, tmp_path):
    from universe_builder import run_batch

    monkeypatch.setattr(settings, "CREATE_NOTES", _write(workspace, NOTES))
    subjects = _write(tmp_path, "Thebes\nMemphis\n", name="subjects.txt")
    seen = []

    def fake_submit(atom, items, model, *, context=None, **kw):
        seen.extend(items)
        return []

    with patch("universe_builder.providers.batch.submit", fake_submit):
        run_batch.main(["submit", "--atom", "city", "--list", str(subjects), "--context", "Egyptian History",
                        "--model", "gpt-5-nano"])

    assert isinstance(seen[0], BatchItem) and seen[0].user_instructions.startswith("The Egyptian Thebes")
    assert seen[1] == "Memphis"
