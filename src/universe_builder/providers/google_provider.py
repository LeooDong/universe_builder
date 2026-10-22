"""Google Gemini provider: sync completions and inline batch jobs.

Caching note. Gemini has two caching modes and only one of them suits this
pipeline. *Explicit* caching (`client.caches.create`) bills storage per hour for
as long as the cache lives, which is wasteful for a batch run that submits once
and collects hours later. *Implicit* caching is automatic on a stable prefix and
costs nothing to hold. So this provider does not create explicit caches - it just
orders content stable-first (system instruction, schema, examples, then the
volatile topic) and lets implicit caching do the work.

Caveat: unlike the Anthropic and OpenAI paths, the batch path here has not been
exercised against a live key. It follows the documented inline-batch shape, and
inline results come back in submission order rather than keyed by ID, so the
mapping back to custom_ids is positional.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Iterator, Optional, Sequence

from universe_builder.providers.base import Provider, require
from universe_builder.providers.errors import ProviderNotConfigured
from universe_builder.providers.types import (BatchHandle, BatchStatus, LLMRequest, LLMResponse, Usage,
                                              require_matching_schema, schema_fingerprint)
from universe_builder.providers.util import expected_custom_ids

log = logging.getLogger(__name__)

_DONE_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


class GoogleProvider(Provider):
    name = "google"
    batch_max_requests = 10_000
    # Inline batches ride in the request body, so the ceiling is far lower than
    # the file-upload batches the other two providers use.
    batch_max_bytes = 20 * 1024 * 1024

    def __init__(self, api_key: str | None = None) -> None:
        genai = require("google.genai", package="google-genai")
        try:
            self._client = genai.Client(api_key=api_key) if api_key else genai.Client()
        except Exception as e:  # noqa: BLE001
            raise ProviderNotConfigured(f"Gemini client init failed: {e}") from e

    # -- request assembly -------------------------------------------------------

    def _system_instruction(self, request: LLMRequest) -> str:
        return self.system_with_schema(request)

    def _contents(self, request: LLMRequest) -> list[dict[str, Any]]:
        """Examples first (stable, so implicit caching can cover them), topic last."""
        contents: list[dict[str, Any]] = []
        for ex in request.examples:
            contents.append({"role": "user", "parts": [{"text": ex.user}]})
            contents.append({"role": "model", "parts": [{"text": ex.assistant_json}]})
        contents.append({"role": "user", "parts": [{"text": request.user}]})
        return contents

    def _config(self, request: LLMRequest) -> dict[str, Any]:
        config: dict[str, Any] = {
            "system_instruction": self._system_instruction(request),
            "response_mime_type": "application/json",
            "response_schema": request.schema_model,
            "max_output_tokens": request.max_tokens,
        }
        if request.temperature is not None:
            config["temperature"] = request.temperature
        # No effort setting is wired for Gemini (its thinking budget is a
        # different knob, unmeasured here): an effort is ignored, warned once
        # per model (v0.6.0, `providers.effort`). Called for the warning.
        self.effort_for(request)
        # Nor a task budget (v0.6.1, Anthropic only): ignored, warned once.
        self.task_budget_for(request)
        return config

    @staticmethod
    def _usage(raw: Any) -> Usage:
        if raw is None:
            return Usage()
        cached = getattr(raw, "cached_content_token_count", 0) or 0
        prompt = getattr(raw, "prompt_token_count", 0) or 0
        return Usage(
            input_tokens=max(prompt - cached, 0),
            output_tokens=getattr(raw, "candidates_token_count", 0) or 0,
            cache_read_tokens=cached,
        )

    # -- sync -------------------------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self._client.models.generate_content(
            model=request.model,
            contents=self._contents(request),
            config=self._config(request),
        )
        usage = self._usage(getattr(response, "usage_metadata", None))
        return self._validate(
            response.text or "", request, usage, self._finish_reason(response)
        )

    @staticmethod
    def _finish_reason(response: Any) -> Optional[str]:
        """Google reports the stop signal per candidate, as an enum."""
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        reason = getattr(candidates[0], "finish_reason", None)
        if reason is None:
            return None
        return getattr(reason, "name", None) or str(reason)

    # -- batch ------------------------------------------------------------------

    def request_size_bytes(self, request: LLMRequest) -> int:
        payload = {"contents": self._contents(request),
                   "system_instruction": self._system_instruction(request)}
        return len(self._dumps(payload).encode("utf-8"))

    def submit_batch(self, requests: Sequence[LLMRequest]) -> BatchHandle:
        if len(requests) > self.batch_max_requests:
            raise ValueError(
                f"{len(requests)} requests exceeds the {self.batch_max_requests} limit; "
                "chunk before calling submit_batch"
            )
        models = {r.model for r in requests}
        if len(models) > 1:
            raise ValueError(f"A Gemini batch takes one model; got {sorted(models)}")

        inline = [
            {"contents": self._contents(r), "config": self._config(r)} for r in requests
        ]
        job = self._client.batches.create(
            model=requests[0].model,
            src=inline,
            config={"display_name": f"universe_builder-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}"},
        )
        log.warning("Submitted Gemini batch %s (%d requests)", job.name, len(inline))
        return BatchHandle(
            provider=self.name,
            batch_id=job.name,
            model=requests[0].model,
            submitted_at=datetime.now(timezone.utc).isoformat(),
            request_count=len(inline),
            routing={r.custom_id: dict(r.metadata) for r in requests if r.custom_id},
            extra={
                # The full pin, as openai and anthropic record it: this once
                # carried `schema_model` alone, so a Gemini
                # batch collected after a model edit validated against the
                # edited model and nothing noticed.
                **schema_fingerprint(requests[0].schema_model),
                # Inline results are positional, so keep the submission order.
                "order": [r.custom_id for r in requests],
            },
        )

    def poll_batch(self, handle: BatchHandle) -> BatchStatus:
        job = self._client.batches.get(name=handle.batch_id)
        state = job.state.name
        return BatchStatus(done=state in _DONE_STATES, state=state)

    def cancel_batch(self, handle: BatchHandle) -> BatchStatus:
        self._client.batches.cancel(name=handle.batch_id)
        return self.poll_batch(handle)

    def fetch_batch_results(self, handle: BatchHandle) -> Iterator[LLMResponse]:
        from universe_builder.providers.registry import schema_model_by_name

        schema_model = schema_model_by_name(handle.extra.get("schema_model"))
        # Before a single result is read, as the other two providers do. A
        # handle from before the pin carries no sha and is let through with a
        # warning (`BatchHandle.schema_mismatch`).
        require_matching_schema(handle, schema_model)
        order: list[str] = handle.extra.get("order", [])
        job = self._client.batches.get(name=handle.batch_id)
        if job.state.name != "JOB_STATE_SUCCEEDED":
            error = f"batch ended in state {job.state.name}"
            log.error("Gemini batch %s: %s", handle.batch_id, error)
            # Surface one failure per submitted request rather than returning
            # nothing, so the caller can reconcile and retry.
            for custom_id in expected_custom_ids(handle):
                yield LLMResponse("", None, handle.model, Usage(), custom_id, error)
            return

        for i, item in enumerate(job.dest.inlined_responses):
            custom_id = order[i] if i < len(order) else None
            if not getattr(item, "response", None):
                error = getattr(item, "error", "no response")
                yield LLMResponse("", None, handle.model, Usage(), custom_id, f"batch error: {error}")
                continue
            usage = self._usage(getattr(item.response, "usage_metadata", None))
            text = item.response.text or ""
            yield self._validate_stored(text, schema_model, handle.model, usage, custom_id,
                                        self._finish_reason(item.response))
