"""The index atom type is never a file phase's work unless it is named.

An end-to-end test found `run --phase verify --limit 1
--dry-run` listing a `history_…` atom: a bare `verify` would have re-asked
every History atom on the balanced tier and rewritten the list every other
atom type is generated from. `work.select_work` now leaves the index atom
type out of fix, verify, expand and append unless `--atom` names it, and
refuses it for `verify` even then.
"""

from __future__ import annotations

import json

import pytest

from universe_builder import cli, journal, settings, store, work
from universe_builder.domains.history.models.discovery import Discovery
from tests.factories import minimal_payload

MODEL = "gpt-5-nano"


@pytest.fixture
def corpus(workspace, tmp_path, monkeypatch, fake_provider):
    """Two History atoms, one valid discovery, one History sentinel."""
    seed = tmp_path / "HISTORY_list.txt"
    seed.write_text("Egyptian History\nGreek History\n", encoding="utf-8")
    monkeypatch.setattr(settings, "HISTORY_LIST", seed)
    for civ in ("Egyptian History", "Greek History"):
        path = store.path_for("history", None, civ, MODEL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"people": [], "discoveries": ["Papyrus"], "events": [],
                                    "cities": [], "artifacts": []}), encoding="utf-8")
    papyrus = store.path_for("discovery", None, "Papyrus", MODEL)
    papyrus.parent.mkdir(parents=True, exist_ok=True)
    papyrus.write_text(json.dumps(minimal_payload(Discovery)), encoding="utf-8")
    sentinel = store.error_path(store.path_for("history", None, "Roman History", MODEL))
    sentinel.write_text('{"broken": true}', encoding="utf-8")
    for atom_type in cli._atom_types():
        store.dir_for(atom_type).mkdir(parents=True, exist_ok=True)
    fake_provider.script(*([lambda r: json.dumps(minimal_payload(r.schema_model))] * 10))
    yield workspace
    journal.end_run()


def _history_files():
    return {p.name: p.read_bytes() for p in store.dir_for("history").iterdir()}


def _dry(capsys, *argv):
    rc = cli.main(["run", *argv, "--dry-run"])
    out = capsys.readouterr().out
    count = next((int(line.split()[-1]) for line in out.splitlines() if line.startswith("work items")), None)
    return rc, count, out


@pytest.mark.parametrize("phase", work.FILE_PHASES)
def test_no_file_phase_selects_the_index_type_unless_named(corpus, phase):
    assert work.select_work(phase, "history") == []


def test_a_bare_verify_dry_run_lists_no_index_atom(corpus, capsys):
    rc, count, out = _dry(capsys, "--phase", "verify")
    assert rc == 0
    assert count == 1          # the discovery; not the two History atoms
    assert "history_" not in out.split("work items")[1]


def test_a_bare_verify_calls_nothing_on_the_index_atoms(corpus, fake_provider):
    before = _history_files()
    cli.main(["run", "--phase", "verify", "--model", MODEL])
    assert [r.metadata.get("subject") for r in fake_provider.requests] == ["papyrus"]
    assert _history_files() == before


def test_verify_refuses_the_index_type_even_when_named(corpus, capsys, caplog, fake_provider):
    with pytest.raises(ValueError, match="index"):
        work.select_work("verify", "history", explicit=True)
    for argv in (["--dry-run"], ["--model", MODEL], ["--batch", "--model", MODEL]):
        assert cli.main(["run", "--phase", "verify", "--atom", "history", *argv]) == 2
    assert "verify does not run on the index atom type" in caplog.text
    assert fake_provider.requests == []


def test_fix_takes_an_index_sentinel_only_when_named(corpus, capsys):
    named = work.select_work("fix", "history", explicit=True)
    assert [w.path.name for w in named] == [store.error_path(
        store.path_for("history", None, "Roman History", MODEL)).name]
    assert _dry(capsys, "--phase", "fix")[1] == 0
    assert _dry(capsys, "--phase", "fix", "--atom", "history")[1] == 1


def test_a_bare_fix_restores_no_index_sentinel_from_the_journal(corpus, fake_provider):
    """`reconcile_fix_queue` writes: it must not run for a type the pass skips."""
    target = store.path_for("history", None, "Persian History", MODEL)
    journal.start_run(config={"phase": "index"})
    journal.current().record_attempt(
        work_item_id=journal.work_item_id("history", "Persian History"), raw_payload='{"x": 1}',
        atom_type="history", subject="Persian History", requested_model=MODEL,
        target_pointer=str(target), validation_status="failed", work_status=journal.NEEDS_FIX)
    journal.end_run()

    cli.main(["run", "--phase", "fix", "--model", MODEL])

    assert not store.error_path(target).exists() and not target.exists()
    assert fake_provider.requests == []
