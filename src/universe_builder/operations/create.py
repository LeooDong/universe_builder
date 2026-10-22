"""Generate a new atom from a topic."""

from __future__ import annotations

import logging

from universe_builder import topics
from universe_builder.atoms import atom_mapping
from universe_builder.config import OperationConfig, llm_call_mapping
from universe_builder.operations._support import call_budget, persist, record_failure, result_from_call
from universe_builder.operations.types import Result

log = logging.getLogger(__name__)


def _sent(config) -> str | None:
    """The effort a create under `config` sends (`effort.sent_or_none`)."""
    asked = getattr(config, "effort", None)
    if not isinstance(asked, str) or not isinstance(config.model_version, str):
        return None
    from universe_builder.providers.effort import sent_or_none

    return sent_or_none(config.model_version, asked)


def _sent_task_budget(config) -> int | None:
    """The task budget a create under `config` sends (`task_budget.sent_or_none`)."""
    asked = getattr(config, "task_budget", None)
    if not isinstance(asked, int) or isinstance(asked, bool) or not isinstance(config.model_version, str):
        return None
    from universe_builder.providers.task_budget import sent_or_none
    from universe_builder.providers.types import DEFAULT_MAX_TOKENS

    return sent_or_none(config.model_version, asked, getattr(config, "max_tokens", None) or DEFAULT_MAX_TOKENS)


def create_atom(config: OperationConfig, topic: str) -> Result:
    """Generate one atom. Transport failures come back as a `Result`, not an exception.

    The old code caught the exception at the call site and `continue`d, so a run
    that failed every single topic looked exactly like a run that had nothing to
    do. Returning `FAILED` lets the caller count them.
    """
    # The subject is the topic, unchanged. It used to carry a
    # `" (context: {atom_name})"` suffix, which was redundant in the prompt -
    # `prompt_start_text` already reads "Provide a response for the discovery or
    # invention commonly known as" - and actively harmful in the filename,
    # because `LLMCall` names and *hashes* the atom from this same string.
    #
    # One topic therefore had three identities:
    #
    #   batch path            h931c34c8003af691  ("Hieroglyphic writing")
    #   this path             h495688e995974e14  ("... (context: discovery)")
    #   the corpus rename     h79200bea7bdbdfe4  ("... context discovery", from
    #   (deleted in v0.3.0)                       the slug on disk)
    #
    # Content-addressed ids (step 3b) exist precisely so the synchronous and
    # batch paths agree on where a topic lives; `batch._output_path` says so in
    # as many words. They did not. Generating a topic both ways produced two
    # files, and the corpus rename would have produced a third id matching
    # neither.
    prompt_subject = topic.strip()
    spec = atom_mapping[config.atom_name]
    log.info("Prompt subject/topic is: %s", prompt_subject)
    # Neighbours and a note only where the domain asks for them, so without
    # either every prompt (and the call classes that take neither) is as before.
    context = getattr(config, "context", None)
    note = topics.create_instructions(config.atom_name, prompt_subject, context)
    extra = {"user_note": note} if note else {}

    try:
        call = llm_call_mapping[config.model_version](
            atom_type=config.atom_name,
            atom_class=spec.atom_class,
            model_version=config.model_version,
            prompt_folder=config.prompt_folder,
            prompt_subject=prompt_subject,
            prompt_start_text=spec.prompt_start_text,
            base_system_text=config.base_system_text,
            example_json_file=spec.example_json_file,
            # The context line goes in the system text, as batch puts it
            # (`requests.build_request`); `base_system_text` above carries it
            # too, for a call class that builds its own prompt.
            context=context,
            **call_budget(config),
            **extra,
        )
    except Exception as e:  # noqa: BLE001 - reported, not raised; one topic must not end a run
        log.error("An error occurred for '%s': %s", prompt_subject, e)
        record_failure(config.atom_name, prompt_subject, str(e), cause=e,
                       requested_model=str(config.model_version),
                       sent_effort=_sent(config),
                       sent_task_budget=_sent_task_budget(config))
        return Result.failed(str(e), subject=prompt_subject)

    # `replaces`: a create normally writes a new file and saves nothing, but
    # `persist` writes over whatever sits at the path (a re-create, an
    # `{"error": ...}` placeholder), and that is kept in the journal.
    result = result_from_call(call, spec.atom_class, subject=prompt_subject,
                              replaces=call.response_path)
    # Not `call.write_response_to_file()`: that writes an `EmptyModel` sentinel
    # over the atom's own path when validation fails, which destroys a good file
    # now that ids are content-addressed. See `persist`.
    persist(call, result)
    return result
