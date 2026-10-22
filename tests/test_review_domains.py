"""Domain discovery errors, pack-owned links, and optional examples."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from universe_builder import atoms, index, migrations, registry, store
from universe_builder.domains.history.models.city import City
from universe_builder.domains.medications.models import Medication
from universe_builder.llm_classes.ollama_call import OllamaCall
from universe_builder.models.subatomic import Name
from tests.factories import minimal_instance


@pytest.mark.parametrize("error", [ImportError("broken module"), ModuleNotFoundError(name="missing_dependency")])
def test_discover_does_not_hide_errors_inside_pack(monkeypatch, error):
    monkeypatch.setattr(registry.importlib, "import_module", MagicMock(side_effect=error))
    with pytest.raises(type(error)) as raised:
        registry._discover("broken_review_pack")
    assert raised.value is error


def test_discover_allows_absent_pack(monkeypatch):
    monkeypatch.setattr(registry.importlib, "import_module",
                        MagicMock(side_effect=ModuleNotFoundError(name="universe_builder.domains.absent_review_pack")))
    registry._discover("absent_review_pack")


@pytest.mark.parametrize("failure_at", ["package", "walk"])
def test_failed_migration_discovery_propagates_and_retries(monkeypatch, failure_at):
    source = "universe_builder.domains.review.migrations"
    monkeypatch.setattr(registry, "active_pack", lambda: SimpleNamespace(name="review"))
    monkeypatch.setattr(migrations, "_loaded_sources", {migrations.__name__})
    monkeypatch.setattr(migrations, "_loading_sources", set())
    import_mock = MagicMock(return_value=SimpleNamespace(__path__=[]))
    walk_mock = MagicMock()
    failing = import_mock if failure_at == "package" else walk_mock
    failing.side_effect = ModuleNotFoundError(name="nested_dependency")
    monkeypatch.setattr(migrations.importlib, "import_module", import_mock)
    monkeypatch.setattr(migrations, "_walk_migrations", walk_mock)
    with pytest.raises(ModuleNotFoundError):
        migrations._loaded()
    assert source not in migrations._loaded_sources
    assert not migrations._loading_sources
    failing.side_effect = None
    migrations._loaded()
    assert source in migrations._loaded_sources
    assert failing.call_count == 2


def test_missing_migration_package_is_optional(monkeypatch):
    source = "universe_builder.domains.review.migrations"
    monkeypatch.setattr(registry, "active_pack", lambda: SimpleNamespace(name="review"))
    monkeypatch.setattr(migrations, "_loaded_sources", {migrations.__name__})
    monkeypatch.setattr(migrations, "_loading_sources", set())
    monkeypatch.setattr(migrations.importlib, "import_module",
                        MagicMock(side_effect=ModuleNotFoundError(name=source)))
    assert migrations._loaded() is migrations.REGISTRY
    assert not migrations._loading_sources


def test_register_rejects_cross_pack_atom_collision_before_mutating_registry(monkeypatch):
    pack = registry.active_pack()
    monkeypatch.setattr(registry, "_PACKS", {pack.name: lambda: pack})
    colliding = replace(pack, name="review_collision")
    with pytest.raises(ValueError, match="share atom names"):
        registry.register_pack(colliding.name, lambda: colliding)
    assert set(registry._PACKS) == {pack.name}


def test_register_rejects_duplicate_atom_names_within_pack(monkeypatch):
    pack = registry.active_pack()
    monkeypatch.setattr(registry, "_PACKS", {})
    duplicate = replace(pack, atoms=(pack.index,))
    with pytest.raises(ValueError, match="Duplicate atom names"):
        registry.register_pack(duplicate.name, lambda: duplicate)
    assert not registry._PACKS


def test_index_reads_links_declared_by_active_pack(workspace, monkeypatch):
    try:
        atoms.use_pack("medications")
        pack = replace(registry.active_pack(), link_fields={"medication": {"drugClass": "medication"}})
        monkeypatch.setattr(registry, "_active", pack)
        source = store.path_for("medication", None, "Aspirin", "m")
        target = store.path_for("medication", None, "Ibuprofen", "m")
        store.write_atom(source, minimal_instance(
            Medication, name=minimal_instance(Name, common="Aspirin"), drugClass="Ibuprofen"))
        store.write_atom(target, minimal_instance(
            Medication, name=minimal_instance(Name, common="Ibuprofen"), drugClass=""))
        built = index.build()
        assert len(built["references"]) == 1
        assert built["references"][0]["field"] == "drugClass"
        assert built["references"][0]["name"] == "Ibuprofen"
    finally:
        # Undo the _active patch before restoring, so teardown cannot restore a stale pack.
        monkeypatch.undo()
        atoms.use_pack("history")


def test_ollama_accepts_no_example_and_sends_no_placeholder(workspace, monkeypatch):
    response = minimal_instance(City)
    post = MagicMock(return_value=SimpleNamespace(
        status_code=200, json=lambda: {"message": {"content": response.model_dump_json()}}))
    monkeypatch.setattr("universe_builder.llm_classes.ollama_call.requests.post", post)
    call = OllamaCall("city", City, "llama3", "", "Memphis", "Describe", "Persona", None)
    assert isinstance(call.get_valid_response(), City)
    content = post.call_args.kwargs["json"]["messages"][0]["content"]
    assert "Persona" in content and "Memphis" in content
    assert "example" not in content and "None" not in content
    assert post.call_args.kwargs["timeout"]   # a server that never answers must not hang the run


@pytest.mark.parametrize("pack", ["history", "medications"])
def test_sync_and_batch_share_persona(workspace, pack):
    from universe_builder.execute_prompt import _base_system
    from universe_builder.providers.requests import base_system_text

    try:
        atoms.use_pack(pack)
        assert _base_system() == base_system_text()
    finally:
        atoms.use_pack("history")


def test_sync_create_matches_batch_context_framing(workspace, monkeypatch):
    from universe_builder import execute_prompt, topics
    from universe_builder.providers.requests import build_request

    monkeypatch.setattr(topics, "needing_create", lambda atom: [topics.Topic("Memphis", "Egypt")])
    seen = []
    monkeypatch.setattr(execute_prompt, "call_and_save_prompt", lambda config, topic_list: seen.append(config) or [])
    execute_prompt._run_create("city", "claude-sonnet-5")
    batch_request = build_request("city", "Memphis", "claude-sonnet-5", context="Egypt")
    assert seen[0].base_system_text == batch_request.system
