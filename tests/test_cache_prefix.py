"""The cached prefix must not contain anything that varies per call.

Prompt caching is a byte-exact prefix match. A topic in the system block does
not lower the hit rate, it takes it to zero — so this is the difference between
the caching work being active and being decorative.

Since the prompt-builder merge the synchronous call builds its request with the
batch builder (`providers.requests.build_request`), whose system text is the
pack's persona, the context line and the operation's instructions — nothing
per call — so the defensive topic strip `ProviderCallAugment._system()` used
to apply is gone. These tests pin that the topic stays out of the system text
and in the user turn.

The provider is patched, so no key and no network are involved.
"""

from __future__ import annotations

import pytest

from universe_builder.providers.types import LLMResponse, Usage


@pytest.fixture
def captured(monkeypatch, workspace, fake_keys):
    """Route every provider call into a list of the requests it would send."""
    requests = []

    def fake_complete(request):
        requests.append(request)
        from tests.test_operations import valid_discovery

        payload = valid_discovery()
        return LLMResponse(payload, request.schema_model.model_validate_json(payload),
                           request.model, Usage(input_tokens=10, output_tokens=10),
                           request.custom_id)

    monkeypatch.setattr("universe_builder.llm_classes.provider_call.complete", fake_complete)
    return requests


def seed(workspace, name, payload):
    folder = workspace / "discovery"
    folder.mkdir(exist_ok=True)
    (folder / name).write_text(payload)
    return folder / name


def run_augment(ep, operation):
    ep.augment_and_save_prompt(
        ep.OperationConfig(
            atom_name="discovery",
            model_version=("gpt-5-nano", "augment"),
            base_system_text="You are a historian.",
            operation_type=operation,
        ),
    )


TOPIC = "papyrus"


class TestTopicStaysOutOfTheCachedPrefix:
    @pytest.mark.parametrize("operation", ["fix", "verify"])
    def test_the_topic_is_absent_from_the_system_text(self, ep, workspace, captured, operation):
        from tests.test_operations import valid_discovery

        name = f"discovery_0000000001_{TOPIC}_gpt-5-nano.json"
        seed(workspace, name + ("_ERROR" if operation == "fix" else ""),
             '{"broken": true}' if operation == "fix" else valid_discovery())

        run_augment(ep, operation)

        assert captured, "no request was built"
        for request in captured:
            assert TOPIC not in request.system.lower(), (
                f"{operation}: topic leaked into the cached system block:\n{request.system}"
            )

    @pytest.mark.parametrize("operation", ["fix", "verify"])
    def test_the_topic_is_present_in_the_user_turn(self, ep, workspace, captured, operation):
        """It still has to reach the model — just in the volatile half."""
        from tests.test_operations import valid_discovery

        name = f"discovery_0000000001_{TOPIC}_gpt-5-nano.json"
        seed(workspace, name + ("_ERROR" if operation == "fix" else ""),
             '{"broken": true}' if operation == "fix" else valid_discovery())

        run_augment(ep, operation)

        assert all(TOPIC in r.user.lower() for r in captured)

    @pytest.mark.parametrize("operation", ["fix", "verify"])
    def test_the_system_text_is_the_batch_builders(self, ep, workspace, captured, operation):
        """The persona, then the operation's instructions, as a batch sends
        them — not the `base_system_text` the call site supplied."""
        from universe_builder.providers.requests import base_system_text
        from tests.test_operations import valid_discovery

        operation_text = {"fix": ep.FIX_SYSTEM_TEXT, "verify": ep.VERIFY_SYSTEM_TEXT}[operation]
        supplied = f"{base_system_text()}\n\n{operation_text}"

        name = f"discovery_0000000001_{TOPIC}_gpt-5-nano.json"
        seed(workspace, name + ("_ERROR" if operation == "fix" else ""),
             '{"broken": true}' if operation == "fix" else valid_discovery())

        run_augment(ep, operation)

        assert captured
        for request in captured:
            assert request.system == supplied

    def test_two_files_of_the_same_atom_share_a_cache_key(self, ep, workspace, captured):
        """The whole point: different subjects, identical cached prefix."""
        from tests.test_operations import valid_discovery

        for n, subject in ((1, "papyrus"), (2, "obsidian")):
            seed(workspace, f"discovery_000000000{n}_{subject}_gpt-5-nano.json",
                 valid_discovery())

        run_augment(ep, "verify")

        assert len(captured) == 2
        assert captured[0].cache_key() == captured[1].cache_key()
        assert captured[0].user != captured[1].user


class TestOperationTextIsStatic:
    @pytest.mark.parametrize("name", ["FIX_SYSTEM_TEXT", "VERIFY_SYSTEM_TEXT", "EXPAND_SYSTEM_TEXT"])
    def test_no_operation_text_contains_a_format_placeholder(self, ep, name):
        """An f-string here would reintroduce per-call content silently."""
        text = getattr(ep, name)
        assert "{" not in text and "}" not in text
