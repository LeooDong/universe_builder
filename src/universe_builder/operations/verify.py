"""Improve an atom that is already valid."""

from __future__ import annotations

import json
import logging

from universe_builder.atoms import atom_mapping, resolve_stored_class
from universe_builder.config import VERIFY_SYSTEM_TEXT, OperationConfig, llm_call_mapping
from universe_builder.operations._support import call_budget, record_failure, result_from_call
from universe_builder.operations.types import Result

log = logging.getLogger(__name__)


def process_verify_operation(data, config: OperationConfig, topic, target=None) -> Result:
    """Improve an already-valid payload. `data` is the raw file text, not a dict.

    This pass runs over files that are known good, so a failure here must never
    reach disk — that is the second half of B8. Returning `INVALID` rather than
    an `EmptyModel` makes overwriting impossible rather than merely unlikely.
    """
    log.info("verifying call for %s", topic)
    try:
        parsed = json.loads(data) if isinstance(data, str) else data
    except (TypeError, ValueError):
        parsed = None

    atom_class = resolve_stored_class(config.atom_name, parsed)
    log.info("Verifying %s against %s", topic, atom_class.__name__)

    try:
        call = llm_call_mapping[config.model_version](
            atom_type=config.atom_name,
            atom_class=atom_class,
            model_version=config.model_version[0],
            prompt_folder=config.prompt_folder,
            prompt_subject=topic,
            prompt_start_text=atom_mapping[config.atom_name].prompt_start_text,
            base_system_text=VERIFY_SYSTEM_TEXT,
            example_json_file=config.example_json_file,
            augment_to_json=data,
            topic=topic,
            operation="verify",
            target_path=target,
            **call_budget(config),
        )
    except Exception as e:  # noqa: BLE001
        log.error("Error verifying %s: %s", topic, e)
        record_failure(config.atom_name, topic, str(e), cause=e, operation="verify", target=target)
        return Result.failed(str(e), subject=topic)

    # Verify rewrites its target in place: the journal keeps what it replaces.
    return result_from_call(call, atom_class, subject=topic, stage=atom_class.__name__,
                            operation="verify", replaces=call.response_path)
