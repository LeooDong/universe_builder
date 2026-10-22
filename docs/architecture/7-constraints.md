Part of [ARCHITECTURE.md](../../ARCHITECTURE.md); the index there lists every part.

## 7. Known constraints

- **Pack selection is process-global and not thread-safe.** Choose `UNIVERSE_BUILDER_DOMAIN`
  at startup, or call `atoms.use_pack` only while no workers/calls are running.
  It replaces the active pack, rebuilds shared registries and clears prompt
  caches in several steps. Do not switch domains concurrently with pipeline work;
  use separate processes for concurrent domains. No switch locks were added.
- **Atom folders and migration keys are not domain-namespaced.** Registration
  rejects collisions among packs loaded into one process, including index names.
  Separately launched processes cannot discover each other's pack declarations
  or prove ownership of existing folders. Give domains distinct `UNIVERSE_BUILDER_DATA_DIR`
  roots; a domain ownership manifest or namespaced layout needs a storage/migration
  design decision. This review does not rename or touch any existing data.
- **Ollama examples are optional.** A `None` example now constructs and sends a
  request without placeholder example prose, covered with a fake local endpoint.
  No live Ollama call was made. Unused `LLMCall.get_filename` and
  `pydantic_helper.coerce_field_to_value` were removed after confirming no callers.


Fixed (kept here so the regression tests have a reference): the stacked `typeOfGovernment` validator that
rewrote every city's government to `"other"` (B1); the missing `return` in `Date.cap_month`
that made `month="March"` a validation error (B2); `merge_helper.process_directory` keeping
only the last walked directory's file lists (B3); `run_fix_verify_calls` raising `KeyError`
on a stray file such as `.DS_Store` (B4); `ClaudeCall` sending `temperature` on models that
reject sampling parameters (B5); and the undecorated `CityDate.replace_null_with_list` dead
method (B6). Also B8, the fix pass destroying its own evidence (see §4), and B9,
B5's twin: `OpenAICall` and `OpenAICallAugment` sent `temperature=0.2` and so
returned a hard 400 on every gpt-5 model registered in `llm_call_mapping`.

- The `execute_prompt.py` path is synchronous and serial — one API call at a time.
  It now routes through the provider layer (`llm_call_mapping` points every hosted
  model at `ProviderCall` / `ProviderCallAugment`), so it gets retries, correct
  cache placement and cost reporting; concurrency and batching still require
  `run_batch.py`. `llama3` stays on `OllamaCall`, being a local server with no
  pricing entry.
- **Strict structured outputs need the schema flattened.** Pydantic emits
  `{"allOf": [{"$ref": ...}], "description": ...}` for a field that both
  references another model and has a description. OpenAI strict mode rejects
  `allOf`, and separately rejects a `$ref` carrying sibling keys, so
  `Provider._flatten_refs` inlines the referenced definition and keeps the
  description. Every atom in the corpus tripped this.
- The per-vendor Claude and OpenAI call classes are **gone**. They defeated their own prompt caching — the augment
  variants interpolated the per-file topic into the cached system block, which
  takes the hit rate to zero rather than merely lowering it, and `ClaudeCall` left
  the one-shot example after the last breakpoint. Every hosted model now routes through `ProviderCall` /
  `ProviderCallAugment`; `OllamaCall` remains, because it is a local server rather
  than a hosted provider.
- The operation instructions (`FIX_SYSTEM_TEXT`, `VERIFY_SYSTEM_TEXT`,
  `EXPAND_SYSTEM_TEXT` in `config.py`) are plain constants with no
  placeholders, so nothing per-call can reach the cached prefix. Per-call detail
  goes to the user turn through `topic=` and `user_instructions=`.
  Since the two prompt builders became one, the sync system text is `build_request`'s (persona,
  context, operation text), so the defensive strip `ProviderCallAugment._system()`
  carried is gone; `tests/test_cache_prefix.py` asserts the system text is the
  batch builder's and the topic is absent from it.
  Verified live: a repeat run reports cache hits on both the create and verify
  passes. Before this, only create hit.
- Three of five atom types (`event` 2,947, `discovery` 2,987, `artifact` 3,094)
  have a cacheable prefix shorter than Haiku 4.5's 4096-token minimum, which is
  the highest of the current models — caching is largely inert there. Sonnet 5
  (1024) and Opus 5 (512) clear it for all five. Registering more few-shot
  examples closes the gap, and has twice: `person` (5,062) and `city` (5,256)
  cleared the Haiku minimum when their second example landed.
- `expand_city`'s registered `atom_class` is its **seed** (`City`, the shape
  `operations/expand.py` reads off disk); its example and every real request's
  schema are the **output**, `Population` — callers pass
  `schema_model="Population"`, and the example-validation tests carry the same
  override. `roman_period_alexandria.json` was regenerated against the current
  schema and registered, so expand calls now ship a one-shot
  example in their cached prefix.
- **Expand cannot split a period.** Its shape is one name in, one `Population`
  out, so a seed period spanning a change of government cannot be divided by
  the expand call — an instruction to split would order what the single-object
  response format forbids (the review gate caught exactly that wording in
  `Population`'s field descriptions before it shipped). The one-period-one-
  government rule therefore lives only on the period *list* fields — the seed
  `City.cityPeriodsOfHistory` (create authors the names) and
  `CityExpanded.cityPeriodsOfHistory` (verify/fix can restructure) — and
  `Population`'s own descriptions stay per-object.
- `LLMCall.sequential_id`, the class counter that once named files, and its lock
  were deleted in v0.3.0, with the `file_names.py` alias
  module, the singular `example_json` alias and its "uses only the first
  example" warning (`OllamaCall` takes the first example itself), the unread
  `include_json_schema` argument, and the hard-coded `log.setLevel(WARNING)` in
  `llm_call`, `ollama_call` and `execute_prompt` (their loggers now inherit the
  configured level).
- **Concurrency is bounded (default 4) and warms the prefix first.** Every
  call for an atom type shares a cached prefix, and a cache entry only becomes
  readable once the first response has begun streaming — so firing a cold
  batch in parallel would *write* the cache many times and read it never. Both
  the create pass (`call_and_save_prompt`) and the augment passes
  (`augment_and_save_prompt`) therefore run the first item
  alone to warm the prefix, then start the pool; `providers/run.run_many`
  makes the same trade. The default was 1 until expand — one call per historical
  period per city, over a thousand for one civilisation — proved unusably serial; 4 is
  a conservative parallelism under provider rate limits, raised per-run for a
  large sweep. The batch path ignores this: its parallelism is the provider's.
  Expand nests — the file loop runs several cities at once and each city now
  expands its periods concurrently too — so a single global gate
  (`_support.call_slot`, sized by `set_call_concurrency` from the resolved
  config) bounds *total* model calls in flight; the two pools cannot multiply
  into a rate-limit breach, and each is sized freely. Expand also has a true
  async batch path now (`--phase expand --batch`): its per-period calls fan out
  as independent requests and collect splices the replies back per city — see
  §3's batching note. Concurrency is the synchronous throughput lever; batch is
  the cheaper, higher-latency one.
- **Reasoning models need a generous `max_tokens`.** They spend output tokens on
  reasoning before emitting any content, so a budget sized for the answer alone
  returns an empty string. Observed live: `gpt-5-nano` at a small `max_tokens`
  spent all of it on reasoning and returned nothing; the same request with a
  larger budget returned valid JSON. `LLMRequest.max_tokens` defaults
  to `DEFAULT_MAX_TOKENS` (16,384), and any caller lowering it must account for
  reasoning.
- **`defaults.max_tokens` once reached the batch path only.**
  `pipeline.yaml` raised it to 32,768 after a run of truncations, and the raise
  did nothing: `PipelineConfig.max_tokens` existed with **no caller anywhere in
  the tree**, so every request fell through to `DEFAULT_MAX_TOKENS`. Cities
  truncating at exactly 16,384 exposed it. `cmd_run` then threaded the configured budget into
  `_submit_batch` and its two fan-outs, guarded by
  `test_cli_run.py::test_the_configured_max_tokens_reaches_the_batch` — which
  pins an implausible 31337, because a test asserting 16,384 would have passed
  throughout the bug's life. The synchronous path stayed capped at 16,384 for
  a while longer, because `OperationConfig` carried no budget. It does now:
  `run_phase` reads `max_tokens(phase)` from
  the selected config onto every `OperationConfig` it builds, and all five
  operations spread `_support.call_budget(config)` into their call, so a phase
  asks for the same budget synchronously and batched
  (`test_run_policy.py::TestTheTokenBudgetReachesTheSyncPath`, at 12345). This
  is the same class of defect `provider_run.configure` was added to fix for
  `retry`, whose comment in `cmd_run` records that `universe_builder config`
  once displayed a retry block nothing applied. A config value with no reader
  is the recurring failure here.
- **The SDK version floors are API-driven.** `OpenAIProvider._body` sends
  `max_completion_tokens` and `AnthropicProvider.complete` called
  `messages.parse(output_format=...)` (now `messages.create` with
  `output_config`); both postdate the floors originally
  declared (`openai>=1.10`, `anthropic>=0.40`), so an install resolving to the
  old floor failed on the first call rather than at install time. Now
  `openai>=1.45.0` and `anthropic>=1.0.0`. Exercised against openai 3.11.0 and
  anthropic 1.4.0; the floors themselves are reasoned, not tested.
- **The append phase may use a different model from create** — unpinned.
  Pairing is id-based: an append inherits its target's content id, so
  `merge_helper` matches the halves whatever model wrote either.
  `tests/test_config.py::TestPhaseModels` now exercises exactly that cross-model
  pairing instead of locking the models equal; if it fails, someone re-coupled
  pairing to the model string and the old pin must return first. (Legacy-named
  files, which paired on slug+model and so needed the models equal, are no
  longer paired at all since v0.3.0.)
- **Augment calls take the path they are augmenting** — fixed after its
  batch-side sibling bit live. Ops pass `target=` through to
  `LLMCall._inherit_target_identity`, and state the intent explicitly: fix,
  verify and expand work *on* the target, so `response_path` becomes the
  target's own path and the automatic `_ERROR` sentinel turns *off* (for a
  fix, the sentinel already there is the repair's input and must survive; for
  a verify or expand, a sentinel beside a valid atom would queue a "repair"
  that overwrites good data — the journal keeps the failed payload either
  way). An `append` passes `supplements_target=True`: it keeps its own file
  but inherits the target's *content id*, so `merge_helper` can pair the
  halves on identity whatever model wrote either. The intent is a parameter
  and never inferred — a first version keyed on the filename prefix, which
  misclassified expand (`atom_type=expand_city` against a `city_…` target)
  and kept its orphan sentinels alive. Pinned by `tests/test_work_identity.py`.
- **Atoms can have more than one on-disk shape.** `city` is written as `City` by the
  create pass and rewritten as `CityExpanded` by the expand pass, so a stored file cannot
  be resolved to a class by atom name alone — `execute_prompt.resolve_stored_class()` and
  the `STAGE_MODELS` table do it by inspecting the payload. Anything that loads a stored
  atom must go through the resolver. `schema_version` tracks migrations, while
  stage predicates still distinguish seed and expanded shapes at the same version.
