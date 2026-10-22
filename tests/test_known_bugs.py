"""Known defects, as strict xfails, numbered by defect (1.1, 1.2, ...).

All flipped in v0.1.20: each now passes and pins the fix, and 1.10 adds its
own tests at the end. Keep `xfail` for the next defect found.

Each test stated the behaviour the fix promised and failed then. `strict=True`
makes the suite fail the moment one starts passing, so the fix that turns it
green has to remove the marker in the same commit — and with it, flip the
matching characterization pin (`test_cli_characterization.py`,
`test_duplicates_snapshot.py`) that records today's behaviour.

Every bug here was reproduced before its test was written (v0.1.19).
None of these tests can write outside `tmp_path`: where the defect *is* that a
writer ignores the test's file, the pack's own path is redirected too
(`pack_overrides`), so the bug writes into a second temporary file rather than
into `domains/history/aliases.txt`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from universe_builder import cli, duplicates, journal, lint, settings, store
from universe_builder.domains.history.models.discovery import Discovery
from tests.factories import minimal_payload

MODEL = "gpt-5-nano"


def xfail(item, why):
    return pytest.mark.xfail(strict=True, reason=f"known bug {item}: {why}")


# ---------------------------------------------------------------------------
# 1.1 — writers ignore settings.ALIASES_LIST (a test-isolation hazard)
# ---------------------------------------------------------------------------

@pytest.fixture
def two_alias_files(workspace, tmp_path, monkeypatch, pack_overrides):
    """`settings.ALIASES_LIST` (what a test sets) and the pack's own file."""
    ours, packs = tmp_path / "ours" / "aliases.txt", tmp_path / "pack" / "aliases.txt"
    for path in (ours, packs):
        path.parent.mkdir()
        path.write_text("# empty\n", encoding="utf-8")
    monkeypatch.setattr(settings, "ALIASES_LIST", ours)
    pack_overrides(aliases_path=packs)
    return ours, packs


class TestWritersHonourAliasesList:
    def test_duplicates_apply(self, two_alias_files):
        ours, packs = two_alias_files
        plan = duplicates.Plan(resolutions=[duplicates.Resolution(
            atom_type="event", name="x", keep=Path("k.json"), retire=[Path("r.json")],
            spellings=["Battle of Kadesh", "Kadesh battle"], reason="test")])
        duplicates.apply(plan, dry_run=False)
        assert "Kadesh battle" in ours.read_text(encoding="utf-8")
        assert packs.read_text(encoding="utf-8") == "# empty\n"

    def test_lint_apply_review(self, two_alias_files):
        ours, packs = two_alias_files
        lint.apply_review([("same", "person", "f.json", "Usermaatre", "Ramesses II")], dry_run=False)
        assert "Usermaatre" in ours.read_text(encoding="utf-8")
        assert packs.read_text(encoding="utf-8") == "# empty\n"

    def test_write_distinct_lines_reads_the_test_alias_file(self, two_alias_files, tmp_path):
        ours, _packs = two_alias_files
        ours.write_text("Book of the Dead = Papyrus of Ani\n", encoding="utf-8")
        records = [{"atom_type": "work", "component": 1, "a": "Book of the Dead", "b": "Papyrus of Ani",
                    "verdict": "different"}]
        outcome = duplicates.write_distinct_lines(records, tmp_path / "not_duplicates.txt", dry_run=True)
        assert outcome["contradicts_alias"] == ["Book of the Dead != Papyrus of Ani"]

    def test_write_distinct_lines_defaults_beside_the_test_alias_file(self, two_alias_files):
        ours, packs = two_alias_files
        records = [{"atom_type": "city", "component": 1, "a": "Nicaea", "b": "Nice", "verdict": "different"}]
        outcome = duplicates.write_distinct_lines(records, dry_run=False)
        assert outcome["path"] == str(ours.parent / "not_duplicates.txt")
        assert "Nicaea != Nice" in (ours.parent / "not_duplicates.txt").read_text(encoding="utf-8")
        assert not (packs.parent / "not_duplicates.txt").exists()


# ---------------------------------------------------------------------------
# 1.2 — write_distinct_lines: contradiction guard and `!=` dedupe
# ---------------------------------------------------------------------------

@pytest.fixture
def distinct(workspace, tmp_path, pack_overrides):
    aliases = tmp_path / "aliases.txt"
    pack_overrides(aliases_path=aliases)
    path = tmp_path / "not_duplicates.txt"
    path.write_text("Abbas I != Abbas II != Abbas Mirza\n", encoding="utf-8")
    return aliases, path


def _different(*pairs, component=1, atom_type="city"):
    return [{"atom_type": atom_type, "component": component, "a": a, "b": b, "verdict": "different"}
            for a, b in pairs]


class TestWriteDistinctLines:
    def test_a_parenthetical_alias_line_is_a_contradiction(self, distinct):
        aliases, path = distinct
        aliases.write_text("Thebes (Luxor) = Waset\n", encoding="utf-8")
        outcome = duplicates.write_distinct_lines(_different(("Thebes", "Waset")), path, dry_run=True)
        assert outcome["contradicts_alias"] == ["Thebes != Waset"]

    def test_a_scoped_alias_line_does_not_refuse_a_global_distinction(self, distinct):
        aliases, path = distinct
        aliases.write_text("[Greek History]\nThebes (Boeotia) = Thebai\n", encoding="utf-8")
        outcome = duplicates.write_distinct_lines(_different(("Thebes (Boeotia)", "Thebai")), path,
                                                  dry_run=True)
        assert outcome["contradicts_alias"] == []

    def test_a_multi_way_line_already_present_is_not_appended_again(self, distinct):
        _aliases, path = distinct
        records = _different(("Abbas I", "Abbas II"), ("Abbas I", "Abbas Mirza"), ("Abbas II", "Abbas Mirza"))
        outcome = duplicates.write_distinct_lines(records, path, dry_run=False)
        assert outcome["new_lines"] == []
        assert path.read_text(encoding="utf-8") == "Abbas I != Abbas II != Abbas Mirza\n"


# ---------------------------------------------------------------------------
# 1.3 — alias-file rewrites are not atomic
# ---------------------------------------------------------------------------

class TestAliasFileWritesAreAtomic:
    """An atomic write replaces the file (temp + `os.replace`), so the inode
    changes; an in-place rewrite truncates it first, and a crash between the
    truncate and the write leaves a partial alias file."""

    def test_the_alias_file_is_replaced_not_rewritten(self, workspace, tmp_path):
        path = tmp_path / "aliases.txt"
        path.write_text("Alexandria = Rhakotis\n", encoding="utf-8")
        before = os.stat(path).st_ino
        plan = duplicates.Plan(resolutions=[duplicates.Resolution(
            atom_type="event", name="x", keep=Path("k.json"), retire=[Path("r.json")],
            spellings=["Battle of Kadesh", "Kadesh battle"], reason="test")])
        duplicates.apply(plan, aliases_path=path, dry_run=False)
        assert "Kadesh battle" in path.read_text(encoding="utf-8")
        assert os.stat(path).st_ino != before

    def test_the_not_duplicates_file_is_replaced_not_appended(self, distinct):
        _aliases, path = distinct
        before = os.stat(path).st_ino
        duplicates.write_distinct_lines(_different(("Nicaea", "Nice")), path, dry_run=False)
        assert "Nicaea != Nice" in path.read_text(encoding="utf-8")
        assert os.stat(path).st_ino != before

    def test_a_rewrite_keeps_the_file_mode(self, workspace, tmp_path):
        path = tmp_path / "aliases.txt"
        path.write_text("Alexandria = Rhakotis\n", encoding="utf-8")
        os.chmod(path, 0o644)
        plan = duplicates.Plan(resolutions=[duplicates.Resolution(
            atom_type="event", name="x", keep=Path("k.json"), retire=[Path("r.json")],
            spellings=["Battle of Kadesh", "Kadesh battle"], reason="test")])
        duplicates.apply(plan, aliases_path=path, dry_run=False)
        assert os.stat(path).st_mode & 0o777 == 0o644
        assert not [p for p in tmp_path.iterdir() if p.name.startswith(".tmp_")]


# ---------------------------------------------------------------------------
# 1.4 — an exhausted transport becomes an `_ERROR` sentinel
# ---------------------------------------------------------------------------

def _create(ep, topic="Papyrus"):
    config = ep.OperationConfig(atom_name="discovery", model_version=MODEL,
                                base_system_text="You are a historian.")
    return ep.create_atom(config, topic)


class TestTransportFailures:
    def test_a_401_is_journalled_as_transport_and_create_regenerates_it(self, ep, fake_provider):
        """Moved in 1.4 from the v0.1.19 pin (a 401 written as `{"error": ...}`
        at the sentinel path, outcome `invalid`). Now three refused attempts
        are a `failed` outcome, journalled at the `transport` stage with no
        payload, and the topic stays in create's work list."""
        from universe_builder import topics

        fake_provider.script(*[PermissionError("Error code: 401 - invalid api key")] * 3)
        journal.start_run(config={"phase": "create"})
        try:
            result = _create(ep)
            [entry] = journal.read_entries(journal.current().run_id)
        finally:
            journal.end_run()
        assert result.outcome.value == "failed"
        assert not store.atoms_for("discovery", "Papyrus")
        assert (entry["validation_stage"], entry["retryable"], entry["raw_payload_available"]) == (
            "transport", True, False)
        assert "401" in entry["raw_payload_unavailable_reason"]
        assert topics.has_real_atom("discovery", "Papyrus") is False

    def test_today_sync_fix_skips_such_a_sentinel(self, ep, fake_provider):
        sentinel = store.error_path(store.path_for("discovery", None, "Papyrus", MODEL))
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text('{"error": "Error code: 401"}', encoding="utf-8")
        ep.run_phase("fix", atoms=["discovery"], model=MODEL)
        assert fake_provider.requests == []

    def test_an_exhausted_transport_writes_no_sentinel(self, ep, fake_provider):
        fake_provider.script(*[PermissionError("Error code: 401 - invalid api key")] * 3)
        result = _create(ep)
        assert result.outcome.value == "failed"
        assert store.sentinels_for("discovery", "Papyrus") == []

    def test_batch_fix_does_not_pay_to_repair_a_transport_sentinel(self, workspace, fake_provider):
        sentinel = store.error_path(store.path_for("discovery", None, "Papyrus", MODEL))
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text('{"error": "Error code: 401"}', encoding="utf-8")
        try:
            cli.main(["run", "--phase", "fix", "--batch", "--atom", "discovery", "--model", MODEL])
        finally:
            journal.end_run()
        assert fake_provider.requests == []

    def test_a_constructor_error_is_not_labelled_transport(self, ep, monkeypatch):
        from universe_builder import config

        def broken(**kwargs):
            raise ValueError("Example JSON hieroglyphic_writing.json has not been validated")
        monkeypatch.setitem(config.llm_call_mapping, MODEL, broken)
        journal.start_run(config={"phase": "create"})
        try:
            assert _create(ep).outcome.value == "failed"
            [entry] = journal.read_entries(journal.current().run_id)
        finally:
            journal.end_run()
        assert entry["validation_stage"] != "transport"


# ---------------------------------------------------------------------------
# 1.5 — sources --via search
# ---------------------------------------------------------------------------

@pytest.fixture
def search(workspace, monkeypatch):
    """A fake `openai-search` provider whose Responses client records calls."""
    from universe_builder import sources as sources_mod
    from universe_builder.providers import registry

    calls = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(output_text='{"urls": ["https://good.example/Papyrus"]}',
                                   usage=SimpleNamespace(input_tokens=500, output_tokens=40))

    from universe_builder.providers.openai_provider import OpenAISearchProvider

    client = SimpleNamespace(responses=Responses())
    client.with_options = lambda **kw: client
    # A real search provider around the fake client: since 1.5 the search goes
    # through `providers.run.complete` and the provider's own `complete`.
    provider = object.__new__(OpenAISearchProvider)
    provider._client = client
    monkeypatch.setitem(registry._CACHE, "openai-search", provider)
    monkeypatch.setattr(sources_mod, "resolves", lambda url, timeout=15: (True, ""))
    path = store.path_for("discovery", None, "Papyrus", MODEL)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(minimal_payload(Discovery)), encoding="utf-8")
    yield calls
    journal.end_run()


class TestSourcesViaSearch:
    def test_today_it_calls_the_search_model_once_per_atom(self, search):
        assert cli.main(["sources", "--via", "search", "--atom", "discovery"]) == 1
        assert [c["model"] for c in search] == ["gpt-4.1-mini"]
        assert search[0]["tools"] == [{"type": "web_search"}]
        assert search[0]["text"]["format"]["name"] == "SourceList"

    def test_the_journalled_entry_carries_the_spend(self, search):
        cli.main(["sources", "--via", "search", "--atom", "discovery"])
        [entry] = list(journal.iter_entries())
        assert (entry["operation"], entry["requested_model"], entry["input_tokens"], entry["output_tokens"]) == (
            "sources-search", "gpt-4.1-mini", 500, 40)

    def test_its_spend_is_journalled(self, search):
        cli.main(["sources", "--via", "search", "--atom", "discovery"])
        assert list(journal.iter_entries())

    def test_a_tier_name_resolves_to_a_model(self, search):
        cli.main(["sources", "--via", "search", "--atom", "discovery", "--model", "economy"])
        assert [c["model"] for c in search] == ["gpt-4.1-mini"]


# ---------------------------------------------------------------------------
# 1.6 — one work selection, one meaning of --limit
# ---------------------------------------------------------------------------

@pytest.fixture
def work(workspace, tmp_path, monkeypatch, fake_provider):
    """Two civilisations, two atom types, two fix sentinels per type."""
    seed = tmp_path / "HISTORY_list.txt"
    seed.write_text("Egyptian History\nGreek History\n", encoding="utf-8")
    monkeypatch.setattr(settings, "HISTORY_LIST", seed)
    for civ, people, discoveries in (("Egyptian History", ["Narmer", "Djoser"], ["Papyrus"]),
                                     ("Greek History", ["Pericles"], ["Astrolabe"])):
        path = store.path_for("history", None, civ, MODEL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"people": people, "discoveries": discoveries, "events": [],
                                    "cities": [], "artifacts": []}), encoding="utf-8")
    for atom_type, subject in (("discovery", "Calendar"), ("discovery", "Glass"),
                               ("person", "Imhotep"), ("person", "Hatshepsut")):
        sentinel = store.error_path(store.path_for(atom_type, None, subject, MODEL))
        sentinel.parent.mkdir(parents=True, exist_ok=True)
        sentinel.write_text('{"broken": true}', encoding="utf-8")
    for atom_type in cli._atom_types():
        store.dir_for(atom_type).mkdir(parents=True, exist_ok=True)
    fake_provider.script(*([lambda r: json.dumps(minimal_payload(r.schema_model))] * 50))
    yield workspace
    journal.end_run()


def _dry_count(capsys, *argv):
    assert cli.main(["run", *argv, "--dry-run"]) == 0
    out = capsys.readouterr().out
    return next(int(line.split()[-1]) for line in out.splitlines() if line.startswith("work items"))


class TestWorkSelection:
    def test_dry_run_fix_limits_per_atom_type(self, work, capsys):
        assert _dry_count(capsys, "--phase", "fix", "--limit", "1") == 2

    def test_dry_run_limit_0_is_nothing(self, work, capsys):
        assert _dry_count(capsys, "--phase", "fix", "--limit", "0") == 0

    def test_run_batch_submit_limit_0_is_nothing(self, work, fake_provider):
        from universe_builder import run_batch

        try:
            run_batch.main(["submit", "--atom", "person", "--model", MODEL, "--from-history", "--limit", "0"])
        except (SystemExit, ValueError):
            pass
        assert fake_provider.requests == []

    def test_run_batch_submit_from_history_keeps_each_subjects_context(self, work, fake_provider):
        from universe_builder import run_batch

        run_batch.main(["submit", "--atom", "person", "--model", MODEL, "--from-history"])
        contexts = {r.metadata["subject"]: r.system for r in fake_provider.requests}
        assert "The context is egyptian history." in contexts["Narmer"]
        assert "The context is greek history." in contexts["Pericles"]

    def test_run_batch_opens_no_journal_run(self, work):
        cli.main(["run", "--phase", "create", "--batch", "--model", MODEL])
        assert journal.list_runs() == []

    def test_a_failed_group_is_reported_not_raised(self, work, fake_provider, monkeypatch, capsys):
        real = fake_provider.submit_batch
        calls = []

        def second_fails(requests):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("provider rejected the batch")
            return real(requests)
        monkeypatch.setattr(fake_provider, "submit_batch", second_fails)
        rc = cli.main(["run", "--phase", "create", "--batch", "--model", MODEL])
        assert rc != 0
        assert "-discovery.json" in capsys.readouterr().out   # the handle that did go

    def test_dry_run_verify_counts_only_what_verify_calls_for(self, work, capsys):
        real = store.path_for("discovery", None, "Papyrus", MODEL)
        real.write_text(json.dumps(minimal_payload(Discovery)), encoding="utf-8")
        placeholder = store.path_for("discovery", None, "Wheel", MODEL)
        placeholder.write_text('{"error": "No data found"}', encoding="utf-8")
        assert _dry_count(capsys, "--phase", "verify", "--atom", "discovery") == 1

    def test_dry_run_fix_counts_sentinels_restorable_from_the_journal(self, work, capsys):
        target = store.path_for("discovery", None, "Faience", MODEL)
        journal.start_run(config={"phase": "create"})
        journal.current().record_attempt(
            work_item_id=journal.work_item_id("discovery", "Faience"), raw_payload='{"name": 3}',
            atom_type="discovery", subject="Faience", requested_model=MODEL,
            target_pointer=str(target), validation_status="failed", work_status=journal.NEEDS_FIX)
        journal.end_run()
        assert _dry_count(capsys, "--phase", "fix", "--atom", "discovery") == 3


# ---------------------------------------------------------------------------
# 1.7 — collisions keys stored atoms on the folded slug, and splits on "_"
# ---------------------------------------------------------------------------

class TestCollisionsIndex:
    def test_two_empty_slug_atoms_keep_their_claims(self, workspace):
        from universe_builder import collisions

        for subject, common, other in (("東大寺", "Tōdai-ji", ["Great Eastern Temple"]),
                                       ("興福寺", "Kōfuku-ji", ["Kofukuji Temple"])):
            path = store.path_for("monument", None, subject, MODEL)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"name": {"common": common, "otherNames": other}}), encoding="utf-8")
        seed = {"monuments": ["Great Eastern Temple", "Kofukuji Temple"]}
        found = {(c.incoming, c.source) for c in collisions.find(seed, atoms=["monument"])}
        assert {("Great Eastern Temple", "claim"), ("Kofukuji Temple", "claim")} <= found

    def test_an_underscored_atom_type_yields_its_ids(self, monkeypatch):
        from universe_builder import collisions

        monkeypatch.setattr(store, "iter_atoms",
                            lambda atom_type: iter([Path("side_effect_h0123456789abcdef_nausea_m.json")]))
        assert collisions._stored_ids("side_effect") == {"0123456789abcdef"}


# ---------------------------------------------------------------------------
# 1.8 — merge_helper walks the whole data tree
# ---------------------------------------------------------------------------

def test_merge_leaves_machinery_folders_alone(workspace):
    from universe_builder.domains.history.models.city import City
    from universe_builder.operations import merge as merge_helper

    backup = workspace / "_backups" / "20261003" / "city"
    backup.mkdir(parents=True)
    atom_id = store.atom_id("city", "Thebes")
    city = backup / f"city_{atom_id}_thebes_{MODEL}.json"
    append = backup / f"append_city_{atom_id}_thebes_{MODEL}.json"
    city.write_text(json.dumps(minimal_payload(City)), encoding="utf-8")
    append.write_text(json.dumps({"architecturalHighlights": ["Karnak"]}), encoding="utf-8")
    before = city.read_bytes()

    merge_helper.process_directory(str(workspace), "city")

    assert append.exists()
    assert city.read_bytes() == before


# ---------------------------------------------------------------------------
# 1.9 — Anthropic sync uses messages.parse, bypassing _validate
# ---------------------------------------------------------------------------

class _FinalMessage:
    """`messages.stream(...)` as the provider uses it: a context manager whose
    stream hands back the accumulated message (v0.5.9 streams the sync call)."""

    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


def test_anthropic_keeps_the_raw_text_of_an_invalid_reply():
    from universe_builder.providers.anthropic_provider import AnthropicProvider
    from universe_builder.providers.types import LLMRequest

    text = '{"name": {"common": "Papyrus"}, "shortSummary": 3}'
    message = SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)], parsed_output=None, stop_reason="end_turn",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0,
                              cache_creation_input_tokens=0))
    messages = SimpleNamespace(stream=lambda **kw: _FinalMessage(message))
    client = SimpleNamespace(messages=messages)
    client.with_options = lambda **kw: client
    provider = object.__new__(AnthropicProvider)
    provider._client = client

    response = provider.complete(LLMRequest(model="claude-haiku-4-5-20251001", schema_model=Discovery,
                                            system="S", user="U"))
    assert not response.ok
    assert response.raw_text == text


def test_anthropic_sync_validates_a_good_reply_and_names_truncation():
    from universe_builder.providers.anthropic_provider import AnthropicProvider
    from universe_builder.providers.types import LLMRequest

    sent = []

    def reply(text, stop):
        message = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)], stop_reason=stop,
            usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=0,
                                  cache_creation_input_tokens=0))
        client = SimpleNamespace(messages=SimpleNamespace(
            stream=lambda **kw: sent.append(kw) or _FinalMessage(message)))
        client.with_options = lambda **kw: client
        provider = object.__new__(AnthropicProvider)
        provider._client = client
        return provider.complete(LLMRequest(model="claude-haiku-4-5-20251001", schema_model=Discovery,
                                            system="S", user="U", max_tokens=64))

    good = json.dumps(minimal_payload(Discovery))
    response = reply(good, "end_turn")
    assert response.ok and response.raw_text == good
    assert sent[0]["output_config"]["format"]["type"] == "json_schema"
    cut = reply(good[:20], "max_tokens")
    assert not cut.ok and cut.raw_text == good[:20] and "truncated" in cut.error.lower()


# ---------------------------------------------------------------------------
# 1.10 — journal target pointers are absolute paths into one corpus
# ---------------------------------------------------------------------------

class TestJournalPointers:
    """Found while writing the characterization pins (no xfail): `reconcile_fix_queue` followed absolute
    `target_pointer`s, so a copied or moved corpus would restore `_ERROR`
    sentinels into the original. Stored relative since v0.1.20; both read."""

    def _queue(self, target):
        journal.start_run(config={"phase": "create"})
        try:
            journal.current().record_attempt(
                work_item_id=journal.work_item_id("discovery", "Faience"), raw_payload='{"name": 3}',
                atom_type="discovery", subject="Faience", requested_model=MODEL,
                target_pointer=str(target), validation_status="failed", work_status=journal.NEEDS_FIX)
            return journal.current().run_id
        finally:
            journal.end_run()

    def test_a_pointer_inside_the_data_dir_is_stored_relative(self, workspace):
        target = store.path_for("discovery", None, "Faience", MODEL)
        [entry] = journal.read_entries(self._queue(target))
        assert entry["target_pointer"] == f"discovery/{target.name}"
        assert journal.target_path(entry) == target

    def test_a_pointer_outside_it_is_kept_as_given(self, workspace, tmp_path):
        elsewhere = tmp_path / "elsewhere" / "x.json"
        assert journal.relative_pointer(str(elsewhere)) == str(elsewhere)
        assert journal.target_path(str(elsewhere)) == elsewhere

    def test_a_copied_corpus_restores_into_the_copy(self, workspace, tmp_path, monkeypatch):
        import shutil

        from universe_builder.operations._support import reconcile_fix_queue

        original = settings.data_dir()
        target = store.path_for("discovery", None, "Faience", MODEL)
        target.parent.mkdir(parents=True, exist_ok=True)
        self._queue(target)
        copy_base = tmp_path / "copy"
        shutil.copytree(original, copy_base / original.name)
        monkeypatch.setenv("UNIVERSE_BUILDER_DATA_DIR", str(copy_base))

        assert reconcile_fix_queue("discovery") == 1
        assert store.error_path(store.path_for("discovery", None, "Faience", MODEL)).exists()
        assert not store.error_path(target).exists(), "wrote into the original corpus"

    def test_a_v0_1_19_absolute_pointer_is_still_followed(self, workspace):
        from universe_builder.operations._support import reconcile_fix_queue

        target = store.path_for("discovery", None, "Faience", MODEL)
        target.parent.mkdir(parents=True, exist_ok=True)
        run_id = self._queue(target)
        # Rewrite the line as v0.1.19 wrote it: the absolute path.
        path = journal.run_dir(run_id) / journal.JOURNAL_FILENAME
        line = json.loads(path.read_text(encoding="utf-8"))
        line["target_pointer"] = str(target)
        path.write_text(json.dumps(line) + "\n", encoding="utf-8")

        assert reconcile_fix_queue("discovery") == 1
        assert store.error_path(target).exists()
