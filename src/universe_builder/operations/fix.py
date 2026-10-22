"""Repair a payload that failed validation."""

from __future__ import annotations

import json
import logging

from universe_builder.atoms import atom_mapping, resolve_stored_class
from universe_builder.config import FIX_SYSTEM_TEXT, OperationConfig, llm_call_mapping
from universe_builder.operations._support import call_budget, record_failure, result_from_call
from universe_builder.operations.types import Result

log = logging.getLogger(__name__)


def process_fix_operation(data, config: OperationConfig, topic, target=None) -> Result:
    """Repair an invalid payload. `data` is the raw file text, not a dict.

    `target` is the sentinel (or its atom path) being repaired. It carries the
    true identity — `topic` is read off the filename and slugs are lossy — so
    the call and the journal both key on it rather than minting a second id.
    """
    log.info("fixing call for %s", topic)
    try:
        parsed = json.loads(data) if isinstance(data, str) else data
    except (TypeError, ValueError):
        parsed = None

    # Resolve against the shape on disk, not the atom name: handing an expanded
    # city the seed schema would collapse the expansion. See B7.
    atom_class = resolve_stored_class(config.atom_name, parsed)
    log.info("Fixing %s against %s", topic, atom_class.__name__)

    try:
        call = llm_call_mapping[config.model_version](
            atom_type=config.atom_name,
            atom_class=atom_class,
            model_version=config.model_version[0],
            prompt_folder=config.prompt_folder,
            prompt_subject=topic,
            prompt_start_text=atom_mapping[config.atom_name].prompt_start_text,
            base_system_text=FIX_SYSTEM_TEXT,
            example_json_file=config.example_json_file,
            augment_to_json=data,
            topic=topic,
            operation="fix",
            target_path=target,
            **call_budget(config),
        )
    except Exception as e:  # noqa: BLE001
        log.error("Failed to fix %s: %s", topic, e)
        record_failure(config.atom_name, topic, str(e), cause=e, operation="fix", target=target)
        return Result.failed(str(e), subject=topic)

    return result_from_call(call, atom_class, subject=topic, stage=atom_class.__name__,
                            operation="fix")
