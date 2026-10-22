# universe_builder

A pipeline that turns a domain's seed list into a structured, validated JSON
corpus using LLMs — one file per subject, every file a Pydantic instance,
every attempt journaled. The domain is a **pack**: a set of Pydantic models,
an index atom that lists what to generate, prompts, examples and the few
tables the curation tools need. Two packs ship as worked examples, `history`
(the richer one, with hand-written sample tables) and `medications` (the
minimal one). The engine knows nothing about either.

*Working name. Proprietary, all rights reserved — see `LICENSE`; pull requests are not
accepted while it is. See `ORIGIN.md` for where it came from.*

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env      # then fill in a key and UNIVERSE_BUILDER_DATA_DIR
```

Only the provider `pipeline.yaml` names needs a key (`openai` by default).
`UNIVERSE_BUILDER_DATA_DIR` is where generated atoms go; without it they land
under `./data/automated` in the working directory. `UNIVERSE_BUILDER_DOMAIN`
picks the pack: a bundled name (`history`, the default, or `medications`), a
dotted module path to a module that registers itself, or an entry point in
the `universe_builder.domains` group.

## Run

```bash
universe_builder config                      # models per phase, and the data root
universe_builder run --phase index --dry-run # what the seed phase would generate
universe_builder run --phase create --dry-run
universe_builder status --coverage           # what the index atoms list, against disk
universe_builder index                       # cross-references
universe_builder lint                         # mechanical checks
universe_builder duplicates                   # one subject, two atoms
universe_builder costs --by phase             # what it cost, from the journal
```

Phases: `index` → `create` → `fix` → `verify` → `expand` → `append`.
**Use `--dry-run` before anything large.** It resolves the real configuration
and lists the work without calling anything. `--batch` submits `create`,
`fix`, `verify` and `expand` to the provider's batch API at half price
(`python -m universe_builder.run_batch` collects).

## Write a pack

The shortest complete example is `src/universe_builder/domains/medications/`:
a `models.py` and an `__init__.py` that builds a `DomainPack` and calls
`registry.register_pack`. Declare an index atom (what lists the work), one
`AtomSpec` per generated type, an `examples/` directory with one valid JSON per
atom, and a `base_system` persona. Optional, and what the curation verbs read:
`link_fields` (cross-references), `coordinate_paths` and `date_paths` (what
identifies a subject, for duplicate detection), `preferred_hosts`,
`url_resolvers` and `identity_resolvers` (free lookups for `sources` and
`lint --via`), `near_match_stopwords`, `aliases_path`, `index_seed_path`,
`operations` (domain-specific verbs). Everything is validated at registration;
a mistyped path is refused with the field that does not exist.

A pack outside this package is found by `UNIVERSE_BUILDER_DOMAIN=my_pkg.pack`
or by an entry point:

```toml
[project.entry-points."universe_builder.domains"]
recipes = "my_pkg.pack"     # a module that registers itself under "recipes"
```

`ARCHITECTURE.md` (an index; one part per area under `docs/architecture/`) describes the layers, the passes, and the error-handling
model in full.

## Tests

```bash
make test    # offline; no key needed
make lint    # flake8, the same check CI runs
```

Three rules that are load-bearing: `store.py` is the only writer of atoms;
no test may reach the network (`conftest.no_live_calls` enforces it); the
data directory is not in git and has no undo.
