"""Generate a supplementary atom to be merged into an existing one."""

from __future__ import annotations

import json
import logging

from universe_builder.atoms import atom_mapping
from universe_builder.config import APPEND_SYSTEM_TEXT, OperationConfig, llm_call_mapping
from universe_builder.operations._support import call_budget, persist, record_failure, result_from_call
from universe_builder.operations.types import Result

log = logging.getLogger(__name__)


def process_append_operation(data, config: OperationConfig, topic, target=None) -> Result:
    """Generate the append half of an atom.

    Unlike the other augment operations this one writes its own file, because the
    append is a *separate* atom that `merge_helper` later pairs with its target
    rather than a replacement for the file being read. `target` is that file:
    the call inherits its content id, so the pair share an identity whatever
    model wrote either half — which is what pairing on the id requires.
    """
    spec = atom_mapping[config.augment_atom_name]
    try:
        call = llm_call_mapping[config.model_version](
            atom_type=config.augment_atom_name,
            atom_class=spec.atom_class,
            model_version=config.model_version[0],
            prompt_folder=config.prompt_folder,
            prompt_subject=topic,
            prompt_start_text=spec.prompt_start_text,
            # The operation instruction rides with the call, as expand's does.
            # Observed live without it: a nano append returned
            # exactly the references the city already held — the prompt never
            # said "only new material", so the model restated its input.
            base_system_text=f"{config.base_system_text}\n{APPEND_SYSTEM_TEXT}",
            example_json_file=config.example_json_file,
            # Serialised as the batch path serialises a payload.
            augment_to_json=json.dumps(data, ensure_ascii=False),
            topic=topic,
            operation="append",
            target_path=target,
            **call_budget(config),
            supplements_target=True,
        )
    except Exception as e:  # noqa: BLE001
        log.error("Append failed for %s: %s", topic, e)
        # `augment_atom_name` ("append_city"), not `atom_name` ("city") — which is
        # what this call's own `atom_type` uses, and what `result_from_call`
        # records on the validation path. Naming the base atom here would file a
        # failed *append* against the city itself, marking a perfectly good city
        # as needing a fix while the append that actually failed goes unrecorded.
        record_failure(config.augment_atom_name, topic, str(e), cause=e, operation="append",
                       target=target)
        return Result.failed(str(e), subject=topic)

    # Usually a new file; an unmerged append from an earlier run is replaced,
    # and kept in the journal like any other in-place write.
    result = result_from_call(call, spec.atom_class, subject=topic, operation="append",
                              replaces=call.response_path)
    persist(call, result)
    return result
