"""Pipeline configuration: what runs, on which model, with which instructions.

Split out of `execute_prompt.py` so `operations/` can import it without importing
the orchestrator. Most of this file is meant to move into `pipeline.yaml`;
keeping it as plain data here makes that a transcription rather than an
untangling.
"""

from __future__ import annotations

import logging

from universe_builder import pipeline, settings
from universe_builder.llm_classes.ollama_call import OllamaCall
from universe_builder.llm_classes.provider_call import ProviderCall, ProviderCallAugment

log = logging.getLogger(__name__)


class OperationConfig:
    def __init__(
        self,
        atom_name,
        model_version,
        base_system_text,
        subatom_to_expand=None,
        expander_class=None,
        expanded_class=None,
        example_json_file=None,
        prompt_folder="prompts",
        operation_type="append",
        max_tokens=None,
        context=None,
        effort=None,
        task_budget=None,
    ):
        self.atom_name = atom_name
        #: The civilisation (or other framing) the subjects come from, which
        #: `base_system_text` already states; kept apart so `create` can look up
        #: a subject's note (`topics.create_note`) for this context.
        self.context = context
        self.model_version = model_version
        self.base_system_text = base_system_text
        self.subatom_to_expand = subatom_to_expand
        # expander_class is the model for one expanded *entry* (e.g. Population);
        # expanded_class is the model for the whole atom once expanded (e.g.
        # CityExpanded). They are different shapes and both are required.
        self.expander_class = expander_class
        self.expanded_class = expanded_class
        self.example_json_file = example_json_file
        self.prompt_folder = prompt_folder
        self.operation_type = operation_type
        self.augment_atom_name = f"{operation_type}_{atom_name}"
        #: The output-token budget every call made under this config asks for
        #: (`pipeline.yaml`'s `defaults.max_tokens`, via `run_phase`). None
        #: leaves the call class's own default, which is `DEFAULT_MAX_TOKENS`
        #: for `ProviderCall`. Before this field existed the synchronous path
        #: always ran at that default while `--batch` ran at the configured
        #: budget. `_support.call_budget` spreads it.
        self.max_tokens = max_tokens
        #: The reasoning effort every call made under this config asks for
        #: (v0.6.0, `pipeline.yaml`'s `effort:` block or `--effort`, via
        #: `run_phase`); None sends none. `_support.call_budget` spreads it.
        self.effort = effort
        #: The task budget every call under this config sends (v0.6.1,
        #: `pipeline.yaml`'s `task_budget:` block or `--task-budget`); None
        #: sends none. `_support.call_budget` spreads it.
        self.task_budget = task_budget


# Every model registered here must have a row in providers/model_pricing.json —
# tests/test_config.py enforces it, which is what stops this drifting a generation
# behind again. These choices are meant to move into pipeline.yaml.
# Every hosted model routes through the provider layer. The provider is resolved
# from the model ID, so there is one pair of classes here rather than one per
# vendor. This is what makes prompt caching, retries and cost reporting active
# rather than merely implemented — the per-vendor `*Call` classes below bypass
# all three, and the augment ones interpolate the per-file topic into the cached
# system block, which drops the hit rate to zero.
#
# `llama3` stays on `OllamaCall`: it is a local server, not a hosted provider,
# and has no entry in `model_pricing.json`.
_pipeline = pipeline.load()

#: Derived from `pipeline.yaml`'s tier tables rather than written out here, plus
#: anything still named directly. Maintaining the list by hand was a second
#: registry to keep in step: `tests/test_config.py` caught `gpt-6-astra` priced
#: and tiered but unregistered the moment the tiers landed, and that check only
#: works if there is one place the list comes from.
_EXTRA_MODELS = (
    "gpt-4o-mini",       # not in any tier; kept because the stored corpus names it
)


def _tier_models() -> tuple:
    """Every tier model, plus every priced model: a model id named on the
    command line (`--model claude-haiku-5-5`) or pinned in a phase passes
    through untouched, which until v0.5.9 held only for tier models; any other
    priced id raised KeyError in the synchronous passes."""
    from universe_builder.providers.pricing import known_models

    models = set(_EXTRA_MODELS)
    for table in _pipeline.tiers.values():
        models.update(table.values())
    models.update(known_models())
    return tuple(sorted(models))


PROVIDER_MODELS = _tier_models()

llm_call_mapping = {"llama3": OllamaCall}
for _model in PROVIDER_MODELS:
    llm_call_mapping[_model] = ProviderCall
    llm_call_mapping[(_model, "augment")] = ProviderCallAugment

# Per-phase model defaults, now read from `pipeline.yaml` rather than written
# here. The dict remains because every caller and test reaches for
# `PHASE_MODELS`, but it is derived: editing the YAML is what changes behaviour.
#
# The append/create invariant that used to be documented here is stated at its
# real cause in pipeline.yaml, and locked by tests/test_config.py::TestPhaseModels.
PHASE_MODELS = _pipeline.phase_models()


base_system_text = settings.BASE_SYSTEM.read_text(encoding="utf-8").strip()


# Operation instructions. Deliberately free of any per-call detail — no topic, no
# filename, no period name — because they land in the cached system prefix, which
# is matched byte for byte. A topic here does not merely lower the hit rate, it
# takes it to zero. Everything that varies per call goes to the user turn via
# `topic=` and `user_instructions=`.
#
# They were kept as constants so that moving them into `prompts/operations/*.txt`
# would be a file copy rather than an untangling; `_operation_text` reads them there.

def _operation_text(operation: str) -> str:
    """One source of truth for operation instructions: `prompts/operations/`.

    The sync constants and the batch path's `operation_text` used to hold two
    hand-maintained copies of each instruction, and they had already drifted —
    the fix a batched call was asked for was worded differently from the fix a
    synchronous call was asked for. The move was planned as "a file copy";
    what forced it was the append-prompt fix
    (only-new material), which would otherwise have had to be written twice.
    """
    return (settings.OPERATIONS_DIR / f"{operation}.txt").read_text(encoding="utf-8").strip()


FIX_SYSTEM_TEXT = _operation_text("fix")
VERIFY_SYSTEM_TEXT = _operation_text("verify")
EXPAND_SYSTEM_TEXT = _operation_text("expand")
APPEND_SYSTEM_TEXT = _operation_text("append")
