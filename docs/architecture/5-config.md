Part of [ARCHITECTURE.md](../../ARCHITECTURE.md); the index there lists every part.

## 5. Configuration and conventions

- **Runtime config** — `src/universe_builder/registry.py` declares each atom once as an
  `AtomSpec` (its model, its index field `plural`, prompt prefix, example file, and
  optional `ExpandSpec`/`AppendSpec` for the `expand_*`/`append_*` variants),
  grouped into a `DomainPack`. `active_pack()` is the single source of truth; the
  runtime tables (`atom_mapping`, `class_mapping`, `STAGE_MODELS`, the batch schema
  registry) derive from it. (The old `prompt_config.json` is retired.)
  A second domain is selected with `UNIVERSE_BUILDER_DOMAIN` / `atoms.use_pack(name)`.
  `DomainPack.link_fields` declares cross-reference wire *paths* (dotted for
  nested models, `[]` for a list of objects); history owns its
  city/discovery/artifact/event/family relations, and other packs default to no
  links. `register_pack` validates every path against its source model, because
  a path that does not exist and an empty field are indistinguishable at read
  time — the failure mode that cost the four `person` relations every edge they
  should have produced.
  **The curation verbs read the pack too**: `coordinate_paths`
  and `date_paths` (where an atom keeps the place and date that identify its
  subject — `lint.corroboration` compares along them, `sources.context_terms`
  reads country/state off them; validated at registration like `link_fields`),
  `preferred_hosts`, `url_resolvers` and `identity_resolvers` (the free
  lookups `sources --via` and `lint --via` offer), `rejected_sources_path`
  (URLs `sources` must never offer an atom again; v0.4.0), `membership_resolver`
  (a pack's reading of `index` memberships; v0.4.1), `summary_paths`
  (the prose `duplicates --via model` shows its judge; v0.4.0), `near_match_stopwords`
  (`topics.near_matches`), `base_system` (the persona — history declares the
  historian; the core default names no domain) and `migrations_package`
  (default: `.migrations` beside the module that registered the pack). A pack
  that declares none of these gets honest "no evidence" answers, not history's.
  Registering a pack builds its declaration to reject duplicate atom names within
  the pack and collisions with already registered packs before installing it.
  Builders must therefore be pure and may be called more than once. An import
  error inside a domain propagates instead of masquerading as an unknown pack.
  **A pack is found by name in four places** (`registry._discover`): already
  registered; a dotted module path (`UNIVERSE_BUILDER_DOMAIN=my_pkg.pack` — the module
  registers itself, and the name it registered under is what becomes active);
  an entry point in the `universe_builder.domains` group (`[project.entry-points."universe_builder.domains"]
  formulary = "my_pkg.pack"`, pointing at a self-registering module or a
  builder); and the bundled `universe_builder.domains.<name>`. The first two are what let a
  pack live outside this package. Two rules for entry points: a
  self-registering module's entry-point name must equal the name it
  registers (the error says which name it did register), and a *builder*
  target must not also register itself on import, or both register and the
  atom-name collision is refused. Entry points are tried before the bundled
  packs, so an installed distribution naming `history` shadows the bundled
  one. The "unknown domain" message lists the bundled packs by directory,
  since nothing has been imported at that point.
- **Secrets** — the nearest `.env`, searched upward from the **working
  directory** first and then from the package (so this checkout's `src/.env`
  or root `.env` is found whichever way the command is run, and an installed
  engine in site-packages still finds a project's), loaded through settings by
  `python-dotenv`. Each walk stops at its project root (a directory holding
  `.git` or `pyproject.toml`), and with no root above it reads only its own
  start, so a `~/.env` is never loaded. The first file found wins; the search stops there — so run
  from inside another project that has its own `.env`, that project's keys are
  billed and its data root written to, and run from anywhere else with an
  editable install of this repo, the checkout's `.env` is found through the
  package walk, so an *external* pack targets this repo's data root unless it
  sets `UNIVERSE_BUILDER_DATA_DIR` itself. `.env.example` is committed; `.env` is not. `OPENAI_API_KEY` for
  the OpenAI path, `ANTHROPIC_API_KEY` for the Claude path; only the provider in use
  needs its key.
- **Paths** — `src/universe_builder/settings.py` is the single source of locations.
  Package *resources* (prompts, `pipeline.yaml`, pricing) resolve from the
  package wherever it is installed. Generated *data* never does: `data_dir()`
  is `UNIVERSE_BUILDER_DATA_DIR` if set, else `./data/automated` under the **working
  directory**, plus the active domain — `<base>/<domain>` (e.g.
  `data/automated/history/`), so two domains never share a folder, id space,
  journal or batch queue (atom folders and content ids carry no domain in
  them, so a shared base could collide). A relative override is relative to
  the working directory, like any other path a command is given. The
  default was once derived from the package's own location
  (`<checkout>/data`), which is right for an editable install of this repo and
  wrong for an installed engine, whose "package location" is site-packages;
  a checkout that holds a corpus sets `UNIVERSE_BUILDER_DATA_DIR` in its `.env`, so its
  scripts still run from any directory. `universe_builder config` and `universe_builder status` print
  the resolved root, and `data_dir()` warns once when the variable is unset,
  the cwd default does not exist and the checkout holds a corpus — the shape
  of "ran it from the wrong directory".
- **Entry points** — install with `python -m pip install -e .`, then run the
  `universe_builder` command from anywhere (`python -m universe_builder.execute_prompt`
  and `make run_execute` went in v0.3.0). `pyproject.toml` declares the `universe_builder`
  console script (`universe_builder run --phase …`, `universe_builder status`, `universe_builder config`,
  `universe_builder index`). For batched generation,
  `python -m universe_builder.run_batch {models,estimate,submit,list,poll,collect,cancel,repin}`.
  **Every entry point calls `settings.load_environment()` in its own `main()`.**
  It used to have one caller — an import side-effect in `llm_classes/llm_call.py` —
  so `cli` and `run_batch`, which do not import that module, ran with no
  credentials at all.
- **Prompt text** — the persona is the pack's (`DomainPack.base_system`;
  history's historian text is `src/universe_builder/domains/history/base_system.txt`), and
  the core default in `src/universe_builder/prompts/base_system.txt` names no domain — it
  was once the historian text, which meant a pack that forgot to
  set `base_system` spoke as a historian. Per-operation instructions are in
  `src/universe_builder/prompts/operations/*.txt`. The provider and sync orchestration both
  read the persona from `providers.requests.base_system_text()`: the active
  pack's `base_system`, or the generic file. `config.base_system_text` remains
  a compatibility export of the generic file. Sync and batch expansion
  share `expansion_instructions`; `BatchItem.user_instructions` stays volatile.
- **Model pricing** — `src/universe_builder/providers/model_pricing.json`, with a checked-on date in
  the file. Cost estimates are indicative; re-verify before acting on them.
- **Batch handles** — `data/automated/<domain>/_batches/*.json`, one per submitted batch.
  Durable by design so a batch can be collected by a later process: there is no
  need to hold a session open, and `collect --all` polls each handle itself and
  skips whatever is not finished. A handle is archived to `_batches/collected/`
  once every submitted request is accounted for, which is what stops a later
  `collect --all` re-fetching an old batch and overwriting atoms that have been
  corrected since. Anything incomplete keeps its handle in place.
  `run_batch cancel NAME|--all [--apply]` (v0.6.2) asks the provider to stop a
  batch still running (`Provider.cancel_batch`; OpenAI, Anthropic and Google, the
  no-batch transports refuse). Without `--apply` it only lists what is running,
  and a finished batch is left alone. The batch's finished results stay with the
  provider, so `collect` takes them once it reads cancelled; the requests that
  never ran are listed in `_missing/` (a sync `create` picks them up). Since
  v0.6.3 a batch that ended `cancelled` or `expired` (`batch.ENDED_SHORT`) has
  its handle archived after that collect, missing and all: nothing more can
  arrive, and a handle left in place was re-fetched by every later
  `collect --all`, rewriting atoms corrected since. A `completed` batch that
  lost results, or one whose state cannot be read, still keeps its handle.
  `python -m universe_builder.run_batch repin HANDLE` displays recorded/current schema pins
  locally. After reviewing a deliberate schema change, `--apply` accepts the
  current fingerprint and preserves the previous one in `schema_pin_history`.
  It neither fetches results nor changes collect's mismatch guard.
  The pin's `schema_sha256` is taken over the model's JSON schema with every
  `required` list sorted (`providers.types._normalised`), so
  reordering a model's fields is not a schema change; a handle pinned before
  that carries the declaration-order hash, which collect and `repin` still
  accept for an unchanged model (`legacy_schema_sha256`).
  All three batch providers record the pin at submit and check it before
  reading a result; Google once recorded only `schema_model` and checked
  nothing.
- **Tests** — `tests/`, fully mocked, no API key needed: `python -m pytest tests/ -v`.
- **CI** — `.github/workflows/ci.yml` runs `make lint` and `make test` (Python 3.10, 3.12,
  3.14) on pushes and PRs to `main` and `dev`; the offline suite only.
- **Not part of the pipeline** — `experiments/` (numbered scratch explorations),
  `data/extracts/` and `data/strings/` (climate and population datasets, currently
  unconnected to the atom pipeline).
