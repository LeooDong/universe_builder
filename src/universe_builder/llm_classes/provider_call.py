"""Adapter that lets the existing pipeline call through the provider layer.

`execute_prompt.py` builds `LLMCall` subclasses and expects the API call to
happen during construction. This class keeps that contract - so it can be dropped
straight into `llm_call_mapping` - while routing the actual request through
`providers/`, which gets correct cache placement, retries and cost reporting.

    from universe_builder.llm_classes.provider_call import ProviderCall, ProviderCallAugment

    llm_call_mapping = {
        "claude-haiku-4-5-20251001": ProviderCall,
        ("claude-haiku-4-5-20251001", "augment"): ProviderCallAugment,
        "gpt-5-nano": ProviderCall,
        ("gpt-5-nano", "augment"): ProviderCallAugment,
    }

Unlike the existing per-provider classes there is one pair here for every
provider, because the provider is resolved from the model ID.

The prompt is not built here: `build_request` delegates to
`providers.requests.build_request`, the batch builder, so there is one prompt
builder for both paths.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from universe_builder.llm_classes.llm_call import LLMCall
from universe_builder.providers.errors import TransportFailed
from universe_builder.providers.run import complete
from universe_builder.providers.types import DEFAULT_MAX_TOKENS as _DEFAULT_MAX_TOKENS, LLMRequest

log = logging.getLogger(__name__)

#: Re-exported for existing importers; `providers.types` owns it now, so the
#: batch path and the synchronous path cannot drift apart again.
DEFAULT_MAX_TOKENS = _DEFAULT_MAX_TOKENS


class ProviderCall(LLMCall):
    """Create-pass call routed through the provider layer.

    The request is `providers.requests.build_request`'s — the batch path's
    builder — so a phase sends the same prompt synchronously as in a batch
    Only `custom_id` (the subject) and
    `metadata` (no `operation`/`target` routing) are the sync path's own.
    `base_system_text` is accepted for the `llm_call_mapping` contract but no
    longer read: the system text is the pack's persona, then the `context`
    line, then the operation's instructions, as batch builds it.
    """

    #: The operation whose instructions the request carries when the caller
    #: names none (`prompts/operations/<operation>.txt`; create has none).
    operation = "create"

    def __init__(
        self,
        atom_type,
        atom_class,
        model_version,
        prompt_folder,
        prompt_subject,
        prompt_start_text,
        base_system_text,
        example_json_file,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        target_path=None,
        supplements_target: bool = False,
        operation: Optional[str] = None,
        context: Optional[str] = None,
        effort: Optional[str] = None,
        task_budget: Optional[int] = None,
        **extra: Any,
    ) -> None:
        #: The reasoning effort (v0.6.0); None sends none. Set before the call
        #: below, which builds the request at construction.
        self.effort = effort
        #: What went on the wire (`effort.sent_or_none`), which the journal
        #: records: None where the provider or model takes no effort.
        self.sent_effort = None
        #: The task budget (v0.6.1) and what of it went on the wire
        #: (`task_budget.of_request`), likewise; None sends none.
        self.task_budget = task_budget
        self.sent_task_budget = None
        super().__init__(
            atom_type,
            atom_class,
            model_version,
            prompt_folder,
            prompt_subject,
            prompt_start_text,
            base_system_text,
            example_json_file,
            target_path=target_path,
            supplements_target=supplements_target,
            operation=operation,
            context=context,
        )
        self.max_tokens = max_tokens
        self.temperature = temperature
        #: The domain's note for this one subject (`topics.create_note`), sent
        #: after the subject in the user turn, never in the cached prefix.
        self.user_note = extra.pop("user_note", None)
        self._extra = extra
        self.llm_response_json = self.call_llm_api()

    # -- request assembly -------------------------------------------------------

    def _request_parts(self) -> dict:
        """What varies by call class: the subject, the payload, the volatile
        instructions. Create sends the subject and its note."""
        return {"subject": self.prompt_subject, "payload": None,
                "user_instructions": self.user_note or ""}

    def build_request(self) -> LLMRequest:
        from universe_builder.providers.requests import build_request

        parts = self._request_parts()
        return build_request(
            self.atom_type,
            parts["subject"],
            self.model_version,
            operation=self.prompt_operation or self.operation,
            payload=parts["payload"],
            user_instructions=parts["user_instructions"],
            context=self.context,
            schema_model=self.atom_class,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            custom_id=self.prompt_subject,
            metadata={"atom_type": self.atom_type, "subject": self.prompt_subject},
            effort=getattr(self, "effort", None),
            task_budget=getattr(self, "task_budget", None),
        )

    def call_llm_api(self) -> str:
        request = self.build_request()
        if request.effort:
            from universe_builder.providers.effort import sent_or_none

            self.sent_effort = sent_or_none(request.model, request.effort)
        if request.task_budget:
            from universe_builder.providers.task_budget import of_request

            self.sent_task_budget = of_request(request)
        response = complete(request)
        # Keep the token counts, don't just log them. The provider measures
        # usage correctly and the journal has fields for it, but nothing
        # carried the numbers between the two on the synchronous path — only
        # `providers/batch.py` passed them — so every sync run since streaming
        # landed recorded `usage_available: false` and **every cost figure for
        # a sync run was unmeasurable**. Held on the call because `_support`
        # journals from the call object, not from the response.
        self.usage = response.usage
        if response.usage.cache_read_tokens:
            log.info("Cache hit: %d tokens read", response.usage.cache_read_tokens)
        # `is True`: callers and tests hand back duck-typed responses (a
        # MagicMock's attribute is truthy), and only the real flag means "nothing came".
        if getattr(response, "transport_failed", False) is True:
            # Nothing arrived. Returning `{"error": ...}` here made the
            # exception text look like the model's answer: it was written as an
            # `_ERROR` sentinel and journalled as a payload to repair from, and
            # a batch fix then paid to "repair" a 401.
            # Raising sends the operation down its `Result.failed` path, which
            # journals a transport failure with no payload and writes nothing.
            raise TransportFailed(response.error or "no response after every transport attempt")
        if not response.ok:
            # Record *why* as well as what. Returning only the raw text loses the
            # verdict, and `parse_response_to_pydantic` then re-validates the
            # text on its own terms — which overturns the refusal whenever the
            # text happens to parse, as a truncated stream's does.
            self.provider_error = response.error or "the provider rejected this response"
            return response.raw_text or json.dumps({"error": response.error})
        return response.raw_text


class ProviderCallAugment(ProviderCall):
    """Fix / verify / expand / append call routed through the provider layer.

    The caller names the operation (`operation="fix"` and so on); its
    instructions follow the persona in the system text. The topic, the payload
    and any per-call instruction go in the user turn, in batch's order:
    `"{prompt_start_text}'{topic}'"`, then `"The subject is {topic}. Here is
    the JSON to work on:\n{payload}"`, then the instruction.
    """

    #: No `prompts/operations/augment.txt` exists, so a caller that names no
    #: operation gets no operation instructions. Every pipeline caller names one.
    operation = "augment"

    def __init__(
        self,
        atom_type,
        atom_class,
        model_version,
        prompt_folder,
        prompt_subject,
        prompt_start_text,
        base_system_text,
        example_json_file,
        augment_to_json,
        topic,
        user_instructions: str = "",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: Optional[float] = None,
        **extra: Any,
    ) -> None:
        self.augment_to_json = augment_to_json
        self.topic = topic
        self.user_instructions = user_instructions
        super().__init__(
            atom_type,
            atom_class,
            model_version,
            prompt_folder,
            prompt_subject,
            prompt_start_text,
            base_system_text,
            example_json_file,
            max_tokens=max_tokens,
            temperature=temperature,
            **extra,
        )

    def _request_parts(self) -> dict:
        """The payload as given (operations pass the stored text, or the atom
        serialised as batch serialises it)."""
        payload = self.augment_to_json
        if not isinstance(payload, str):
            payload = json.dumps(payload, ensure_ascii=False)
        return {"subject": self.topic, "payload": payload,
                "user_instructions": self.user_instructions or ""}
