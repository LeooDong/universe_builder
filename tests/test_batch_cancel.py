"""`run_batch cancel` (v0.6.2): lists unless --apply, leaves finished batches
alone, and each provider asks its own API to stop the batch."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from universe_builder import run_batch
from universe_builder.providers import batch as batch_mod
from universe_builder.providers.errors import BatchNotSupported
from universe_builder.providers.types import BatchHandle, BatchStatus


def _handle(provider="openai", model="gpt-5-nano"):
    return BatchHandle(
        provider=provider, batch_id="batch_test", model=model,
        submitted_at="2026-10-10T00:00:00Z", request_count=2,
        routing={"person-00000-a": {}, "person-00001-b": {}},
        extra={"schema_model": "Person"},
    )


def _run(monkeypatch, states, apply):
    paths = [Path(f"/x/{name}.json") for name in states]
    monkeypatch.setattr(run_batch, "_targets", lambda args: paths)
    monkeypatch.setattr(batch_mod, "poll", lambda path: BatchStatus(
        done=states[path.stem] in ("completed", "cancelled"), state=states[path.stem], succeeded=5))
    cancelled = []

    def cancel(path):
        cancelled.append(path.stem)
        return BatchStatus(done=False, state="cancelling", succeeded=5)

    monkeypatch.setattr(batch_mod, "cancel", cancel)
    code = run_batch.cmd_cancel(SimpleNamespace(apply=apply))
    return code, cancelled


def test_without_apply_nothing_is_cancelled(monkeypatch, capsys):
    code, cancelled = _run(monkeypatch, {"a": "in_progress", "b": "completed"}, apply=False)
    assert code == 0 and cancelled == []
    out = capsys.readouterr().out
    assert "would be cancelled" in out and "--apply" in out


def test_apply_cancels_only_the_running(monkeypatch):
    code, cancelled = _run(monkeypatch, {"a": "in_progress", "b": "completed", "c": "finalizing"},
                           apply=True)
    assert code == 0 and cancelled == ["a", "c"]


def test_openai_cancel_calls_the_api():
    from universe_builder.providers.openai_provider import OpenAIProvider

    with patch("openai.OpenAI"):
        provider = OpenAIProvider()
    provider._client.batches.cancel.return_value = MagicMock(
        status="cancelling", request_counts=MagicMock(completed=40, failed=0, total=43))
    status = provider.cancel_batch(_handle())
    provider._client.batches.cancel.assert_called_once_with("batch_test")
    assert (status.state, status.done, status.succeeded) == ("cancelling", False, 40)


def test_anthropic_cancel_calls_the_api_then_polls():
    from universe_builder.providers.anthropic_provider import AnthropicProvider

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider._client = MagicMock()
    batches = provider._client.messages.batches
    batches.retrieve.return_value = MagicMock(
        processing_status="canceling",
        request_counts=MagicMock(succeeded=3, errored=0, canceled=0, expired=0, processing=1))
    status = provider.cancel_batch(_handle(provider="anthropic", model="claude-haiku-4-5-20251001"))
    batches.cancel.assert_called_once_with("batch_test")
    assert status.state == "canceling" and status.succeeded == 3


def test_a_provider_with_no_batch_api_refuses():
    from universe_builder.providers.openai_provider import DeepSeekProvider

    provider = DeepSeekProvider.__new__(DeepSeekProvider)
    with pytest.raises(BatchNotSupported):
        provider.cancel_batch(_handle(provider="deepseek", model="deepseek-chat"))


# -- collect after a cancel or expiry: the handle is archived, not swept forever --


def _collect(tmp, state=None, poll_error=None):
    import json

    from universe_builder import settings
    from universe_builder.providers.types import LLMResponse, Usage

    handle = _handle()
    path = Path(tmp) / "handle.json"
    path.write_text(json.dumps(handle.to_dict()))
    provider = MagicMock()
    provider.fetch_batch_results.return_value = iter(
        [LLMResponse("{}", None, handle.model, Usage(), "person-00000-a", "bad")])
    if poll_error:
        provider.poll_batch.side_effect = poll_error
    else:
        provider.poll_batch.return_value = BatchStatus(done=True, state=state)
    batch_dir = Path(tmp) / "_batches"
    with patch.object(batch_mod, "get_provider", return_value=provider), \
         patch.object(settings, "data_dir", return_value=Path(tmp)), \
         patch.object(settings, "batch_dir", return_value=batch_dir), \
         patch.object(settings, "missing_dir", return_value=batch_dir / "missing"):
        return batch_mod.collect(path), path, batch_dir


@pytest.mark.parametrize("state", ["cancelled", "expired", "JOB_STATE_CANCELLED"])
def test_a_batch_that_ended_short_is_archived_with_its_missing_listed(tmp_path, state):
    """The review gate's finding: a cancelled batch is never complete, so its
    handle stayed in `_batches/` and every later `collect --all` re-fetched it,
    rewriting atoms corrected since. Its missing requests will never run."""
    report, path, batch_dir = _collect(tmp_path, state=state)
    assert report.missing == ["person-00001-b"] and report.ended_short == state
    assert report.archived is not None and not path.exists()
    assert (batch_dir / "missing" / "handle.json").exists(), "the never-ran ids must still be listed"
    assert report.archived not in batch_mod.list_batches()


def test_a_completed_batch_that_lost_results_keeps_its_handle(tmp_path):
    report, path, _ = _collect(tmp_path, state="completed")
    assert report.missing and report.ended_short is None
    assert report.archived is None and path.exists()


def test_a_state_that_cannot_be_read_keeps_the_handle(tmp_path):
    report, path, _ = _collect(tmp_path, poll_error=RuntimeError("network"))
    assert report.ended_short is None and report.archived is None and path.exists()


def test_cmd_collect_reports_never_ran_apart_from_lost(monkeypatch, capsys):
    paths = [Path("/x/a.json")]
    monkeypatch.setattr(run_batch, "_targets", lambda args: paths)
    monkeypatch.setattr(batch_mod, "poll", lambda path: BatchStatus(done=True, state="cancelled"))
    monkeypatch.setattr(batch_mod, "collect", lambda path: batch_mod.CollectReport(
        written=3, missing=["m1", "m2"], ended_short="cancelled"))
    assert run_batch.cmd_collect(SimpleNamespace()) == 0
    out = capsys.readouterr().out
    assert "2 never ran (cancelled; handle archived)" in out and "MISSING" not in out
