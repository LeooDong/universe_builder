"""List neighbours in the create prompt: the entries either side of a subject
in its context's index list.

Measured on the history corpus: of the known misgenerations ("Later Han
dynasty", listed between Later Jin and Later Zhou, written as the Eastern Han),
most came back wrong again from the bare prompt and nearly all came back right
with three neighbours either side; the controls stayed right.
"""

import json
from unittest.mock import patch

from universe_builder import cli, settings, store, topics
from universe_builder.providers.requests import BatchItem, build_requests, with_create_note

CITIES = ["Memphis", "Thebes", "Abydos", "Edfu", "Aswan", "Philae", "Kom Ombo"]


def _history(cities, name="history_h1111111111111111_egyptian_history_m.json"):
    path = store.dir_for("history") / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"people": [], "discoveries": [], "events": [], "artifacts": [], "cities": cities}),
                    encoding="utf-8")


def test_the_sentence_names_k_entries_either_side(workspace):
    _history(CITIES)

    text = topics.list_neighbours("city", "Edfu", "egyptian history", k=2)

    assert text == ("In the Egyptian History list of cities, this entry comes after Thebes and Abydos, "
                    "and before Aswan and Philae.")


def test_the_ends_of_the_list_read_first_and_last(workspace):
    _history(CITIES)

    assert topics.list_neighbours("city", "Memphis", "egyptian history", k=2).endswith(
        "comes first, before Thebes and Abydos.")
    assert topics.list_neighbours("city", "Kom Ombo", "egyptian history", k=1).endswith(
        "comes last, after Philae.")


def test_no_context_no_listing_or_zero_gives_nothing(workspace):
    _history(CITIES)

    assert topics.list_neighbours("city", "Edfu", None, k=3) is None
    assert topics.list_neighbours("city", "Nowhere", "egyptian history", k=3) is None
    assert topics.list_neighbours("city", "Edfu", "greek history", k=3) is None
    assert topics.list_neighbours("city", "Edfu", "egyptian history", k=0) is None
    assert topics.list_neighbours("city", "Edfu", "egyptian history") is None, "the workspace sets 0"


def test_a_subject_an_alias_line_renamed_finds_its_listed_entry(workspace, tmp_path):
    # The work list emits the survivor's spelling, which the list may not hold:
    # a scoped line makes the listed "Thebes" generate as "Thebes (Waset)".
    (tmp_path / "aliases.txt").write_text("[Egyptian History]\nThebes (Waset) = Thebes\n", encoding="utf-8")
    _history(CITIES)

    text = topics.list_neighbours("city", "Thebes (Waset)", "egyptian history", k=1)

    assert text.endswith("after Memphis, and before Abydos.")


def test_neighbours_come_before_the_note_in_one_user_turn(workspace, monkeypatch):
    _history(CITIES)
    notes = workspace / "notes.txt"
    notes.write_text("city | Edfu | The temple town of Horus.\n", encoding="utf-8")
    monkeypatch.setattr(settings, "CREATE_NOTES", notes)
    monkeypatch.setattr(settings, "CREATE_NEIGHBOURS", 1)

    item = with_create_note("city", "Edfu", "egyptian history", topics.load_create_notes())
    (request,) = build_requests("city", [item], "gpt-5-nano", context="egyptian history")

    assert isinstance(item, BatchItem)
    assert request.user.endswith("after Abydos, and before Aswan.\n\nThe temple town of Horus.")
    assert "Abydos" not in request.system


def test_the_sync_create_sends_them(ep, workspace, monkeypatch, fake_call_factory):
    _history(CITIES)
    monkeypatch.setattr(settings, "CREATE_NEIGHBOURS", 1)
    Fake = fake_call_factory(['{"not": "valid"}'])
    monkeypatch.setitem(ep.llm_call_mapping, "claude-sonnet-5", Fake)
    config = ep.OperationConfig(atom_name="city", model_version="claude-sonnet-5", base_system_text="t",
                                context="egyptian history")

    ep.call_and_save_prompt(config=config, topic_list=["Edfu"], concurrency=1)

    assert Fake.calls[0]["user_note"].endswith("after Abydos, and before Aswan.")


def test_the_batch_create_sends_them(workspace, monkeypatch):
    _history(CITIES)
    monkeypatch.setattr(settings, "CREATE_NEIGHBOURS", 1)
    seen = []

    def fake_submit(atom, subjects, model, *, context=None, **kw):
        seen.extend(subjects)
        return [store.dir_for("history") / "handle.json"]

    with patch("universe_builder.providers.batch.submit", fake_submit):
        assert cli.main(["run", "--phase", "create", "--atom", "city", "--batch", "--model", "gpt-5-nano"]) == 0

    edfu = next(s for s in seen if isinstance(s, BatchItem) and s.subject == "Edfu")
    assert edfu.user_instructions.endswith("after Abydos, and before Aswan.")
    assert all(isinstance(s, BatchItem) for s in seen), "every listed subject has neighbours"


def test_a_namesake_in_another_list_is_not_the_subject(workspace):
    # Matching by shared name once put the Ottoman `Foča` (Bosnia) at the
    # Greek list's `Phocaea (Foça)`; the work list keeps them apart, and so must this.
    _history(["Sarajevo", "Foča", "Mostar"], name="history_h1111111111111111_ottoman_history_m.json")
    _history(["Smyrna", "Phocaea (Foça)", "Clazomenae"], name="history_h2222222222222222_greek_history_m.json")
    index = topics.ListIndex("city")

    assert index.contexts_of("Foča") == ["ottoman history"]
    assert topics.list_neighbours("city", "Foča", "greek history", k=1, index=index) is None
    assert topics.list_neighbours("city", "Foča", "ottoman history", k=1, index=index).endswith(
        "after Sarajevo, and before Mostar.")
