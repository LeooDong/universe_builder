Part of [ARCHITECTURE.md](../../ARCHITECTURE.md); the index there lists every part.

## 6. Extending the system

**A new civilisation.** Not a code change — one line in
`src/universe_builder/domains/history/HISTORY_list.txt`, plus the alias lines that keep its
names from colliding with the stored corpus. The full recipe, with the
cost of each phase, belongs in a pack's own repository. The rest of
this section covers the extensions that *are* code changes.

**A new atom type**

1. Add the Pydantic model to the domain's models (history: `src/universe_builder/domains/
   history/models/`; a new domain: its own `models.py`), reusing core
   `models/subatomic.py` types.
2. Add one `AtomSpec` to the pack (history's is in `src/universe_builder/domains/history/
   __init__.py`; the dataclasses are in `src/universe_builder/registry.py`): name, model,
   `plural`, `prompt_start`, `example`, and an `ExpandSpec`/`AppendSpec`/`stages`
   if it has them. Add the matching list field to the index model (`History`) and
   to `src/universe_builder/domains/history/examples/egyptian_history.json`.
3. Add an example JSON to the domain's own examples dir (history:
   `src/universe_builder/domains/history/examples/`) — validated against the model at call
   construction, so a bad example fails loudly and early.

That is it: `atom_mapping`, `class_mapping`, `STAGE_MODELS` and the batch schema
registry all derive from the `AtomSpec`, so there is nothing else to keep in step.
This replaced a seven-edit, five-file recipe. `test_config.py::TestPackConsistency` checks the new spec is
coherent — its `plural` names a real index field, its example exists.

**A new domain**

A whole second problem (e.g. medications) is a pack, not an edit to core. See
`src/universe_builder/domains/medications/` for the worked example: a `models.py`, an
`__init__.py` that builds a `DomainPack` and calls `registry.register_pack`, and
its own `examples/`. `UNIVERSE_BUILDER_DOMAIN=medications` (or `atoms.use_pack("medications")`)
switches the engine onto it. **A pack need not live in this package**: set
`UNIVERSE_BUILDER_DOMAIN` to the dotted path of any importable module that registers
itself, or declare an `universe_builder.domains` entry point in the pack's own
distribution (§5, *Runtime config*). Declare the curation fields too —
`coordinate_paths`/`date_paths`, `preferred_hosts`, `url_resolvers`,
`identity_resolvers`, `near_match_stopwords`, `base_system` — or lint, sources
and topics will answer "no evidence" for the domain, honestly; `git` shows it changed nothing under `src/universe_builder`
outside `domains/` when originally introduced. CLI create now skips the active
index, and both synchronous and batch expand read the active specs. Medication
CLI create/expand/append are covered through the real orchestration with fake calls,
and `tests/test_pack_contract.py` runs lint, sources, topics, collisions, duplicates
and retire under the pack, checks discovery by dotted path and entry point, and
fails on any core string literal that is a history atom name.

**What a pack may import from core.** A pack is built from `registry`
(`DomainPack`, the specs, `register_pack`, `active_pack`), the models in
`models/subatomic.py`, `store` (the only writer), `settings`, `journal` and
`providers.run`. For packs whose verbs reason about the work lists the way
the engine does — the history pack's review, fold and retire verbs — these
are public since v0.2.0, each with its pre-v0.2.0
underscore name kept as the same object for a release:

| name | what it is |
|---|---|
| `topics.name_key(text)` (was `_key`) | the one name fold: NFKD, combining marks dropped, case and punctuation to single spaces |
| `topics.context_of(filename)` (was `_context_of`) | the civilisation a History file describes, recovered from its content id against the seed list |
| `topics.raw_occurrences(atom_type)` (was `_raw_occurrences`) | every listed (subject, context), per History file, before grouping |
| `topics.group_occurrences(occurrences, prefer, aliases, assignments)` (was `_group_occurrences`) | the work list's grouping, for a what-if over other occurrences or alias lines |
| `topics.history_files_as(paths)` | a context manager: every reader — work lists, resolvers, `ListIndex`, `history_files`, `index.build`, `index.memberships` — sees `paths` as the stored Histories for the block. Replaces reassigning `topics._history_files`, which still works (`tests/test_pack_contract.py`) |
| `index.memberships(resolved=True)` | `index.build()["memberships"]` without the reference walk; `resolved=False` skips the pack's `membership_resolver` (v0.4.1) |
| `journal.record_call(..., target_pointer=, schema_id=)` | journal a model call a verb makes itself, with the file it writes and the reply's schema |

`lint._fold_name` stays private and is not `name_key`: it keeps one space per
punctuation mark for `suggest_verdict`'s character ratio. Every other fold in
core is `name_key` (`collisions._fold` is gone).

**A new model.** Registering a model (making it callable) and selecting it (pointing
a phase at it) are two different things:

1. **To register it**, give it a row in `providers/model_pricing.json` (step 3).
   The hosted-model keys in `llm_call_mapping` derive from the tier tables, every
   priced model, and the small `_EXTRA_MODELS` list in `config.py`
   (`config.py::_tier_models` → `PROVIDER_MODELS`); the only entry outside that is
   `llama3`, hardcoded to `OllamaCall` for the local server. Until v0.5.9 only tier
   models were registered, so `--model claude-haiku-5-5` (priced, in no tier) raised
   `KeyError` in every synchronous pass while the batch path, which never reads
   `llm_call_mapping`, ran it. (`_EXTRA_MODELS` exists only for a model the stored
   corpus still names, e.g. `gpt-4o-mini`.)
2. **To select it**, either point a phase at the tier, or name the model directly.
   `phases:`, per-atom `atoms:` overrides, and `--model` may all carry a bare model
   id as well as a tier name — `resolve_tier` passes an id through untouched, the
   escape hatch for pinning one phase to an exact model. But naming a model in those
   places is *selection, not registration*: it still has to be priced (or tiered,
   or in `_EXTRA_MODELS`) to have an `llm_call_mapping` entry.
3. Add a matching pricing row to `providers/model_pricing.json`. Required, not
   optional: `pipeline.yaml` states "every model named in `tiers` must have a row",
   and `test_config.py` fails a model registered in `llm_call_mapping` with no
   pricing entry.

There is no hand-maintained model→class table and no `LLMCall` subclass to write; if
the model's vendor is already registered, those steps are all.

**A new provider**

1. Add a `Provider` implementation under `universe_builder/providers/` and register it as a
   `providers/registry._FACTORIES` entry keyed by **provider name** (`"anthropic"`,
   `"openai"`, …) — the factory table is keyed by provider, not by model.
2. Add each of the provider's models to `providers/model_pricing.json`, and set the
   `provider` field of every such row to the new factory key. That field **is** the
   routing: `pricing.provider_for(model)` reads it to pick the factory (the sole
   exception is a `vendor/model` OpenRouter slug, which routes to `openrouter` by
   shape). So the pricing row does double duty — cost data and provider routing.
3. Register the models for use as in "A new model" above: tier them in `pipeline.yaml`
   (a pricing row alone prices a model but does not put it in `llm_call_mapping`).

Transport belongs in the provider layer; do **not** add an `LLMCall` subclass — the
per-vendor `OpenAICall`/`ClaudeCall` classes were deleted precisely because they
reimplemented transport and broke prompt caching.
