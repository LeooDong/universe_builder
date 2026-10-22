Part of [ARCHITECTURE.md](../../ARCHITECTURE.md); the index there lists every part.

## 6b. Tests

`python -m pytest -q` from the repo root (or `python -m pytest ../tests -q`
from `src/`). `pyproject.toml` sets `pythonpath = ["src"]` and `testpaths = ["tests"]`.
Tests live at the repository root and no test file modifies `sys.path`.
For development, install `python -m pip install -e '.[dev]'`.

The default suite touches neither the network nor an API key. The one live test
carries `@pytest.mark.integration`, which `addopts` deselects; run it explicitly
with `python -m pytest tests/test_live_smoke.py -m integration`. Any further
integration test must carry the same marker. `make release-check` (suite,
lint) runs before a tag.

| file | covers |
|---|---|
| `test_aliases.py` | `WireModel` alias round-trips — what goes out by alias comes back by alias |
| `test_architecture_doc.py` | **this document** (the index and every part). That its diagrams name no module or class that has been deleted, and do not omit the parts that matter; that the index links every part. Added after all three diagrams were found stale |
| `test_atom_ids.py` | content-addressed ids: stability across processes, distinctness, normalisation, id shape |
| `test_batch_pinning.py` | a batch whose schema moved while in flight is refused, not silently validated against the new model |
| `test_cache_prefix.py` | nothing per-call reaches the cached system prefix |
| `test_doc_budgets.py` | the docs read first stay small enough to read whole: `CLAUDE.md`, `README.md`, the `ARCHITECTURE.md` index, and each part under `docs/architecture/` |
| `test_release_skill.py` | the release skill (`.claude/skills/release/SKILL.md`) names only paths, `make` targets and tests that exist |
| `test_task_budget.py` | the task budget (v0.6.1): unset sends nothing (plain endpoints, no handle key, no `run.json` key, the same config hash); merged beside effort and format on the beta stream, the prompt-only fallback and a Message Batch (beta create, poll and results); refusals under the minimum, over `max_tokens` (and its explicit allowance), and for a model without the capability, before any call and not retried; other providers ignore it with one warning; the config's precedence; `--task-budget` through sync, batch, index and `run_batch submit` into the requests, the handle, the journal line and `run.json`; the dry run; `--experiment` with a budget refused |
| `test_effort.py` | the reasoning effort (v0.6.0): unset sends nothing on any provider or path; what is recorded is what was sent (a model that ignores effort records none); each provider's translation and the ones that take none; a lacking level refused and not retried; the config's precedence; `--effort` through sync, batch, index and `run_batch submit` into the requests, the handle, the journal line and `run.json`; the dry run; `costs --by effort` |
| `test_experiments.py` | prompt experiments (v0.6.0 adds: B on another model and effort in its own batches, priced on its own row; arms compared on resolved values; a model A/B judged on quality with the cost beside it, a prompt experiment as before; a reused A that would not be today's A refused; panel refusals; a changed arm is a new experiment; the ceiling on A always and B without `--allow-dear-b`; `--reuse-a`; two judges side by side and the agreement-only panel, by hand): the two separation guards and an end-to-end shadow run that leaves the corpus and its journal untouched; live `PromptSpec` sends the pipeline's bytes, no bleed between specs, strict variant files; sampling; the price guard; replay, collect and the both-orders judge on the fake provider; the verdict arithmetic by hand |
| `test_cli_run.py` | `universe_builder run` — the dry run's real work counts, `--batch` for create/fix/verify, append refused |
| `test_index_batch.py` | `run --phase index --batch` (index via batch): one batch per seed without an atom, the index tier, `--limit`, the dry run; prompt parity with the sync index call (they differ only in `custom_id` and `metadata`); collect writes the atom or `_ERROR` and journals at the batch discount (`costs`: `index (batch)`) |
| `test_expand_batch.py` | batched expand: one request per unexpanded period, replies spliced per city on collect, a lone Population never overwrites the city, failed periods keep their name, re-collect is idempotent |
| `test_concurrency.py` | per-path locking and `O_EXCL`: a failed call never replaces a good atom (B11, B12) |
| `test_config.py` | pack consistency (each atom's `plural` names a real index field, examples exist, an expander is a different shape from its seed, `BASE_ATOM_TYPES` matches the pack) plus the phase-model/pricing invariants — the old cross-registry drift-guards went away with the single source |
| `test_corpus.py` | the shape of what is on disk, including one file per atom (history excepted) and no stray `final_` outputs |
| `test_coverage.py` | `universe_builder status --coverage`/`--gaps` — whether a civilisation is actually finished, not merely present on disk |
| `test_domain_medications.py` | the generalisation proof: the medications pack runs the same passes as history through the real orchestration with fake calls |
| `test_factories.py` | `minimal_instance` builds and round-trips every atom model — the "a valid atom of any shape" guarantee the mechanical tests lean on |
| `test_index.py` | the derived cross-reference index: alias-aware resolution in the company of listed spellings, membership attribution, name links, ambiguity refusing to guess |
| `test_periods.py` | the period machinery: regime-table parsing, BCE-aware crossing detection, scope containment, majority adjudication (ties untouched), the split splice and its failure path |
| `test_entry_points.py` | every entry point loads the `.env` itself, and the output budget and cost estimate are one measured constant |
| `test_examples.py` | every registered example validates against its atom class |
| `test_file_names.py` | compose/decompose round-trip, punctuation, non-ASCII, malformed input |
| `test_journal.py` | the journal module — append-only semantics, queue replay, required raw payload |
| `test_journal_wiring.py` | the journal *wired in*: evidence outliving its sentinel, transport failures recorded, both failure paths agreeing on the work item |
| `test_merge_helper.py` | merge semantics and nested-tree pairing (B3), against `operations/merge.py` |
| `test_migrations.py` | `schema_version`, migrate-on-read, chain validation |
| `test_models.py` | validators on `Date`, `Population`, `Judgments`, `EmptyModel`; one named regression per bug B1–B6 |
| `test_operations.py` | create/fix/verify/expand/append through a fake `LLMCall`, `_ERROR` handling, stray-file tolerance (B4), and that both paths agree on where a topic lives |
| `test_pipeline_cli.py` | `pipeline.yaml` resolution and provenance (`--model` beats atom beats phase beats default) |
| `test_providers.py` | the provider layer: transport, batch submit/poll/collect, reconciliation, archiving |
| `test_pydantic_compat.py` | guardrails for the Pydantic 2 behaviours the codebase leans on, since the dependency is a range (`>=2.11,<3`) not a pin |
| `test_result.py` | the typed `Result` and its outcomes |
| `test_review_domains.py` | domain discovery errors, pack-owned links, and optional examples (post-refactor review regressions) |
| `test_review_pipeline.py` | offline regressions for the post-domain-refactor review |
| `test_anthropic_provider.py` | the Anthropic provider on fakes, pinning the first live run: no refused keyword in the wire schema; a grammar too large falls back to the prompt once per (model, schema), other 400s raise; thinking blocks skipped, fenced or prefaced JSON read, a truncated object kept whole; truncation and refusal named; `max_tokens` held to the output cap; batch wire ids swapped and swapped back, the one-token schema probe, nested batch errors, canceled and expired counted; `run.complete` not retrying a 4xx and honouring `retry-after` |
| `test_run_policy.py` | the retry/timeout policy — `pipeline.yaml`'s `retry:` block actually applied by `run.complete` |
| `test_shape_blindness.py` | the shape-blindness guarantee pinned: an atom shape change does not cascade into the mechanical tests (via `backfill_required`) |
| `test_settings.py` | fresh-process imports, bundled resources, output paths from an unrelated cwd |
| `test_store.py` | paths, the `_ERROR` sentinel, atomic writes, `iter_atoms` ordering, and that a validated atom never lands on a sentinel path |
| `test_store_snapshot.py` | `read_snapshot`: the same answers in the same order as the glob (two models, a sentinel, a directory, a name with no underscore after the id), a write inside unseen until the block ends, nesting and an exception restoring nothing-cached, two threads' overlapping blocks leaving nothing stale and lending nothing to a thread outside; the cached `name_key`/`alias_keys` equal the uncached; the index's cached lookups equal fresh ones; the narrowed last-resort grouping equals the full one |
| `test_topics.py` | the create work list: de-duplication across History files, context preserved, alias-file grouping and cross-civilisation splits |
| `test_work_identity.py` | one piece of work, one identity: repairs settle the queue entry they answer, augment calls inherit the target's identity, a failed verify queues nothing |
| `test_truncation.py` | a budget-exhausted response is reported as truncation, not a schema failure, for all three providers' spellings |
| `test_bundled_data.py` | every non-Python file in the package is covered by `package-data` (the history example's `README.md` was not) |
| `test_no_live_data.py` | no live data comes in: the history example's tables and every fixture within a registered budget, the examples a fixed set, no dated or signed notes or content ids in tables, no private project name (held as hashes) or home path in any file git would publish |
| `test_version.py` | the release number has one source (`universe_builder.__version__`, read by `pyproject.toml`), the installed metadata agrees with it, and it is not behind the newest `v*` tag |
| `test_golden_ids.py` | **frozen literals**: `atom_id`, `normalise_topic`, `slugify`, `folder_name`, `HASH_ID_*`, `work_item_id`, filenames and `_context_of` for ~27 names, and 300 (type, topic) -> id rows from the real corpus (`tests/fixtures/corpus_ids_sample.json`) |
| `test_frozen_grouping.py` | frozen `_key`/`alias_keys` for every seed, alias and not-duplicates name; the alias-file parses (`tests/fixtures/alias_files_parsed_v0.1.19.json`); `from_history`, `needing_create`, assignments, `listed_spellings`, `subject_resolver` on a corpus reaching every grouping branch |
| `test_fake_provider.py` | the `fake_provider` fixture drives the real `ProviderCall -> run.complete -> Provider._validate` path: retries, re-asks, the truncation guard, usage into the journal, batch submit |
| `test_golden_prompts.py` | the exact request each phase builds, sync and batch, frozen in `tests/fixtures/golden_prompts/` with the vendors' schema-suffix hashes; sync equals batch but for `custom_id` and `metadata` |
| `test_cli_characterization.py` | pins of today's CLI: `lint` review/`--via`/`--apply-review`, `sources`, `--limit` across dry-run/sync/batch, batch-submit failures reported per group, `run --batch` opening no journal run |
| `test_duplicates_snapshot.py` | the whole `duplicates.plan` answer on a corpus reaching every branch, and what the alias / not-duplicates writers leave byte-identical |
| `test_journal_format.py` | real v0.1.19 journal lines plus a synthetic run (`tests/fixtures/journal_v0.1.19/`) read through `outstanding`, `needs_fix_across_runs` and `costs` |
| `test_known_bugs.py` | known defects as `strict=True` xfails |
| `test_live_smoke.py` | **the only live test.** One `gpt-5-nano` call end to end. `integration`-marked, deselected by default, a fraction of a cent a run |

**No test may reach the network.** `conftest.no_live_calls` is autouse and patches
`provider_call.complete` to raise, so a fake registered under the wrong model key
fails immediately instead of quietly billing the account — which is exactly what
happened once, and the only symptom was a 56-second test.

Two mechanisms are worth knowing about because they are load-bearing:

- **The fake transports.** `conftest.py::fake_call_factory` subclasses the real
  `LLMCall` and overrides only `call_llm_api`; everything above it — example
  validation, filename construction, Pydantic validation, the `_ERROR` write — is
  the genuine code path. `conftest.py::fake_provider` (tests/fake_provider.py)
  goes one layer lower: a scripted `Provider` installed behind every name in the
  provider registry, so `ProviderCall`, `providers.run.complete` and
  `Provider._validate` run as shipped. New tests should prefer it.
- **`strict=True` xfails mark known defects, not skipped work.** They fail the suite
  when the underlying bug is fixed, which is how the fix gets noticed.
  `test_known_bugs.py` holds them, each naming its defect; the fix removes the marker and flips the matching characterization
  pin in the same commit.
- **Golden and characterization tests compare against data written once.** The
  values under `tests/fixtures/` (and the literals in `test_golden_ids.py`) were
  computed from v0.1.19 and are never regenerated by the code under test; a
  change that moves one is a behaviour change, made on purpose.

The shared `workspace` fixture sets `UNIVERSE_BUILDER_DATA_DIR` to a directory under
`tmp_path`. It reads prompts directly from the package. The `in_src` fixture,
all fixture/setup/teardown `chdir` calls, and the import-directory dances are
removed. Corpus tests use `settings.data_dir()` so a normal checkout still
validates local `data/automated/` and a checkout without it skips those tests.
Packaging regressions import and exercise both paths in a subprocess whose cwd
is unrelated to the repository, without a live API call.

**Open: the test tidy** (not started; no src changes):

1.  Shared factories (`_history` ×6, `_atom` ×6, `_handle` ×4, `meds` ×3, provider
    builders ×2) → `tests/factories.py`; `test_operations` stops being a helper module.
2.  New tests use the fake Provider; migrate old `fake_call_factory` uses opportunistically.
3.  Replace source-regex tests (`test_journal_wiring.py:514-538`) with their behavioural
    twins; wire tests capture SDK kwargs instead of calling `_messages`/`_body`.
4.  Coverage that matters before any provider switch: Anthropic/Google `complete()` and
    batch submit/poll/fetch (Google maps results by position); `batch.submit`;
    `run.complete`'s don't-re-ask-on-truncation guard.
