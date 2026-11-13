"""Run arbitrary requests as one provider batch (v0.8.0).

`providers.batch` generates atoms: it builds the requests and writes each
answer into the corpus. A judge pass has neither shape. Its requests are built
by the caller, and its answers go back to the caller. Judges are stateless per
call and need no answer in real time, so the batch's half price is theirs to
take. This module takes the caller's `LLMRequest`s, submits them as one batch,
polls until it ends, and hands each result to `on_result` exactly as
`providers.run.run_many` does, so a caller swaps one for the other:

    report = run_many(requests, on_result=record, apply=True)

Without `apply` nothing is submitted: the plan and the batch price are
logged and an empty report is returned. The handle is saved under
`settings.request_batch_dir()` before the first poll, so an interrupted caller
passes `resume=<batch id or handle path>` and polls the same batch again.
Given the same requests again without `resume`, it finds the pending handle
they match and resumes that instead of paying twice.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

from universe_builder import journal
from universe_builder import settings
from universe_builder.providers import batch as batch_mod
from universe_builder.providers.errors import BatchNotSupported
from universe_builder.providers.pricing import estimate_cost
from universe_builder.providers.registry import get_provider
from universe_builder.providers.requests import prefix_tokens
from universe_builder.providers.run import RunReport
from universe_builder.providers.types import BatchHandle, LLMRequest, LLMResponse, Usage

log = logging.getLogger(__name__)

#: `BatchHandle.extra["kind"]` of a request batch. `batch.collect` refuses
#: such a handle: its answers belong to a caller, not to the corpus.
KIND = "requests"

#: Output tokens per request the dry-run price assumes when the caller gives
#: no figure: a reasoning judge spends 2-4k (`experiments.judge_estimate`).
ASSUMED_OUTPUT_TOKENS = 3000


@dataclass
class BatchRunReport(RunReport):
    """`RunReport` plus the batch it came from. On a dry run `responses` is
    empty, `estimate` is the batch price and `ceiling` the price were every
    request to spend its `max_tokens`."""

    handle: Optional[Path] = None
    batch_id: Optional[str] = None
    estimate: Optional[float] = None
    ceiling: Optional[float] = None
    dry_run: bool = False


def fingerprint(requests: Sequence[LLMRequest]) -> str:
    """One hash for a set of requests: what a pending handle is matched on."""
    parts = sorted(
        json.dumps([r.custom_id, r.cache_key(), r.user, r.max_tokens, r.effort, r.task_budget])
        for r in requests
    )
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def estimate(requests: Sequence[LLMRequest], output_tokens: int = ASSUMED_OUTPUT_TOKENS
             ) -> tuple[Optional[float], Optional[float]]:
    """(likely, ceiling) batch price. The first request writes the shared
    prefix to the cache and the rest read it; the user turn is ~4 characters a
    token. `likely` assumes `output_tokens` out each, `ceiling` each request's
    `max_tokens`. None for a model the price table does not know."""
    model = requests[0].model
    prefix = prefix_tokens(requests[0])
    likely = ceiling = Usage()
    for i, r in enumerate(requests):
        user = len(r.user) // 4
        cached = {"cache_write_tokens": prefix} if i == 0 else {"cache_read_tokens": prefix}
        likely = likely + Usage(input_tokens=user, output_tokens=min(output_tokens, r.max_tokens), **cached)
        ceiling = ceiling + Usage(input_tokens=user, output_tokens=r.max_tokens, **cached)
    return (estimate_cost(likely, model, batch=True), estimate_cost(ceiling, model, batch=True))


def _check(requests: Sequence[LLMRequest]) -> None:
    if not requests:
        raise ValueError("No requests to run")
    ids = [r.custom_id for r in requests]
    if not all(ids):
        raise ValueError("Every request needs a custom_id: a batch result is routed back by it")
    if len(set(ids)) != len(ids):
        raise ValueError("custom_ids must be unique within a batch")
    if len({r.model for r in requests}) > 1:
        raise ValueError("A batch runs one model; split the requests by model")
    if len({r.schema_model for r in requests}) > 1:
        raise ValueError("A batch pins one schema; split the requests by schema_model")


def _handles() -> list[Path]:
    """Request-batch handles, pending first, then those already collected."""
    pending = settings.request_batch_dir()
    done = pending / "collected"
    return [p for d in (pending, done) if d.is_dir() for p in sorted(d.glob("*.json"))]


def find(resume: str | Path) -> Path:
    """The handle for `resume`: a handle path, or a batch id whose handle is
    under `settings.request_batch_dir()` (pending or collected)."""
    path = Path(resume)
    if path.suffix == ".json" and path.is_file():
        return path
    for candidate in _handles():
        try:
            if batch_mod.load(candidate).batch_id == str(resume):
                return candidate
        except Exception:  # noqa: BLE001 - an unreadable file is not the one asked for
            continue
    raise FileNotFoundError(f"No request-batch handle for {resume!s} under {settings.request_batch_dir()}")


def _pending_match(digest: str) -> Optional[Path]:
    pending = settings.request_batch_dir()
    if not pending.is_dir():
        return None
    for candidate in sorted(pending.glob("*.json")):
        try:
            if batch_mod.load(candidate).extra.get("requests_sha256") == digest:
                return candidate
        except Exception:  # noqa: BLE001
            continue
    return None


def _submit(requests: Sequence[LLMRequest], name: str) -> Path:
    from universe_builder.providers import task_budget as task_budget_mod
    from universe_builder.providers.effort import sent

    first = requests[0]
    provider = get_provider(first.model)
    wire_effort = sent(first.model, first.effort)
    wire_task_budget = task_budget_mod.sent(first.model, first.task_budget, first.max_tokens)
    handle = provider.submit_batch(requests)
    handle.extra["kind"] = KIND
    handle.extra["requests_sha256"] = fingerprint(requests)
    batch_mod.stamp_effort(handle, wire_effort)
    batch_mod.stamp_task_budget(handle, wire_task_budget)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = settings.request_batch_dir()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stamp}-{name}.json"
    suffix = 1
    while path.exists():
        path = directory / f"{stamp}-{name}-{suffix:02d}.json"
        suffix += 1
    batch_mod._write(handle, path)
    log.warning("Submitted batch %s (%d requests); handle %s. If interrupted, resume with "
                "resume=%r", handle.batch_id, len(requests), path, handle.batch_id)
    return path


def _wait(path: Path, poll_seconds: float) -> BatchHandle:
    while True:
        status = batch_mod.poll(path)
        if status.done:
            return batch_mod.load(path)
        time.sleep(poll_seconds)


def run_many(
    requests: Sequence[LLMRequest],
    *,
    on_result: Optional[Callable[[LLMResponse], None]] = None,
    apply: bool = False,
    resume: Optional[str | Path] = None,
    operation: str = "request",
    name: str = "requests",
    poll_seconds: float = 60.0,
    output_tokens: int = ASSUMED_OUTPUT_TOKENS,
) -> BatchRunReport:
    """Run `requests` as one batch and call `on_result` once per request, in
    request order, with an `LLMResponse` shaped as `providers.run.run_many`
    gives it. A request the batch returned nothing for comes back failed, with
    `transport_failed` set, so every request is answered once.

    Each result is journalled (`journal.record_call`, marked batch) under the
    request's `metadata["operation"]`, else `operation`; its
    `metadata["atom_type"]` and `["subject"]`, else `operation` and the
    custom_id. A caller whose `on_result` journals its sync results already
    drops that when it moves here.

    Every request must carry a unique custom_id and share one model and one
    schema; the requests must fit one batch. `resume` (a batch id or a handle
    path) polls an already submitted batch instead of submitting; pass the
    same requests, which give the schema the results are validated against.
    Polling and fetching are free, so a resume runs without `apply`.
    """
    requests = list(requests)
    _check(requests)
    model = requests[0].model
    provider = get_provider(model)
    if not provider.supports_batch:
        raise BatchNotSupported(f"{provider.name} has no batch job API; use providers.run.run_many")
    chunks = provider.chunk_for_batch(requests)
    if len(chunks) > 1:
        raise ValueError(f"{len(requests)} requests need {len(chunks)} batches on {provider.name}; "
                         "split them and run each part")

    digest = fingerprint(requests)
    path = find(resume) if resume is not None else _pending_match(digest)
    if path is not None and resume is None:
        log.warning("These requests match pending batch %s; %s it instead of submitting again",
                    path.name, "resuming" if apply else "a run with apply would resume")

    if path is None and not apply:
        likely, ceiling = estimate(requests, output_tokens)
        batch_mod._warn_about_cache(requests, model)
        log.warning(
            "Dry run: would submit %d requests to %s (effort %s) as 1 batch; batch price ~$%s "
            "at %d output tokens each, at most ~$%s. Pass apply to submit.",
            len(requests), model, requests[0].effort or "unset",
            f"{likely:.4f}" if likely is not None else "?", output_tokens,
            f"{ceiling:.4f}" if ceiling is not None else "?",
        )
        return BatchRunReport(estimate=likely, ceiling=ceiling, dry_run=True)
    if path is None:
        batch_mod._warn_about_cache(requests, model)
        path = _submit(requests, name)
    elif not apply and resume is None:
        return BatchRunReport(handle=path, batch_id=batch_mod.load(path).batch_id, dry_run=True)

    handle = _wait(path, poll_seconds)
    if handle.extra.get("requests_sha256") not in (None, digest):
        log.warning("Batch %s was submitted for a different set of requests than these; results "
                    "whose custom_id is not among them are dropped", handle.batch_id)
    by_id = {r.custom_id: r for r in requests}
    got: dict[str, LLMResponse] = {}
    for response in get_provider(handle.model).fetch_batch_results(
            handle, schema_model=requests[0].schema_model):
        if response.custom_id in by_id:
            got[response.custom_id] = response
        else:
            log.error("Batch %s returned an unknown custom_id %r", handle.batch_id, response.custom_id)

    report = BatchRunReport(handle=path, batch_id=handle.batch_id)
    for request in requests:
        response = got.get(request.custom_id) or LLMResponse(
            "", None, model, Usage(), request.custom_id,
            f"batch {handle.batch_id} returned no result for this request", transport_failed=True)
        _journal(request, response, handle, operation)
        if on_result:
            on_result(response)
        report.responses.append(response)
        report.usage = report.usage + response.usage

    report.cost = estimate_cost(report.usage, model, batch=True)
    if path.parent == settings.request_batch_dir():
        report.handle = _archive(path)
    log.warning(
        "Batch %s: %d requests, %d ok, %d failed, ~$%s (cache reads: %d tokens)",
        handle.batch_id, len(report.responses), report.ok_count, report.failed_count,
        f"{report.cost:.4f}" if report.cost is not None else "?", report.usage.cache_read_tokens,
    )
    return report


def _journal(request: LLMRequest, response: LLMResponse, handle: BatchHandle, operation: str) -> None:
    meta = request.metadata or {}
    op = meta.get("operation") or operation
    journal.record_call(
        request, response,
        operation=op,
        atom_type=meta.get("atom_type") or op,
        subject=meta.get("subject") or request.custom_id,
        error=response.error,
        schema_id=request.schema_model.__name__,
        batch_id=handle.batch_id,
        provider=handle.provider,
    )


def _archive(path: Path) -> Path:
    """Move a handle whose results were all handed over into `collected/`,
    where a later run with the same requests no longer resumes it."""
    target_dir = path.parent / "collected"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / path.name
    suffix = 1
    while target.exists():
        target = target_dir / f"{path.stem}-{suffix:02d}{path.suffix}"
        suffix += 1
    path.replace(target)
    return target
