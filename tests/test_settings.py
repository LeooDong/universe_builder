"""Packaging regressions: fresh imports and output paths independent of cwd."""

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from universe_builder import settings


@pytest.mark.parametrize("relative", [False, True])
def test_data_override_is_resolved_at_use_time(tmp_path, monkeypatch, relative):
    from universe_builder.providers import batch
    from tests.test_providers import _handle

    override = "alternate-output" if relative else str(tmp_path / "automated")
    monkeypatch.setenv("UNIVERSE_BUILDER_DATA_DIR", override)
    # A relative override is relative to the working directory, not the package.
    base = Path.cwd() / override if relative else tmp_path / "automated"
    # UNIVERSE_BUILDER_DATA_DIR sets the base; data_dir() namespaces by the active domain.
    expected = base / "history"
    assert settings.data_dir() == expected
    assert batch._output_path(_handle(), "person-00000-a").parent == expected / "person"
    assert settings.batch_dir() == expected / "_batches"
    assert settings.missing_dir() == expected / "_batches" / "missing"

    # Changing the override after importing consumers must not leave stale paths.
    monkeypatch.setenv("UNIVERSE_BUILDER_DATA_DIR", str(tmp_path / "second"))
    assert batch._output_path(_handle(), "person-00000-a").parent == tmp_path / "second" / "history" / "person"


def test_fresh_process_uses_package_resources_outside_repo(tmp_path):
    env = dict(os.environ, PYTHONPATH=str(settings.PACKAGE_DIR.parent),
               UNIVERSE_BUILDER_DATA_DIR=str(tmp_path / "output"))
    script = textwrap.dedent('''
        import socket
        from pathlib import Path
        from unittest.mock import patch

        def no_network(*args, **kwargs):
            raise AssertionError("network access in an offline packaging test")

        socket.socket.connect = no_network
        with patch("dotenv.load_dotenv"):
            from universe_builder import execute_prompt, run_batch, settings
            from universe_builder.llm_classes.llm_call import LLMCall
            from universe_builder.domains.history.models.history import History
            from universe_builder.providers.requests import build_request, operation_text

        cwd = Path.cwd()
        assert cwd != settings.REPO_ROOT
        assert execute_prompt.atom_mapping["history"].atom_class is History
        from universe_builder.registry import active_pack
        assert active_pack().index_seed_path.read_text().strip()
        for operation in ("fix", "verify", "expand", "append"):
            assert operation_text(operation)
        assert build_request("person", "Test subject", "gpt-5-nano").examples
        call = LLMCall(
            atom_type="history", atom_class=History, model_version="gpt-5-nano",
            prompt_folder="prompts", prompt_subject="Test subject",
            prompt_start_text="History of", base_system_text="Test",
            example_json_file="egyptian_history.json",
        )
        call.llm_response_json = History(
            people=["Test person"], discoveries=[], events=[], cities=[], artifacts=[]
        ).model_dump_json()
        call.write_response_to_file()
        assert Path(call.response_file_location).parent == settings.data_dir() / "history"
        assert run_batch._subjects_from_history("person") == ["Test person"]
        assert Path.cwd() == cwd
    ''')
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
