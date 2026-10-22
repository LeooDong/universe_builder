# universe_builder

The engine: a domain-agnostic pipeline that generates a validated JSON corpus
with LLMs. A *domain pack* supplies models, prompts and examples; two ship
under `src/universe_builder/domains/` (`history`, a worked example with
hand-written sample tables — see `ORIGIN.md`; `medications`, minimal).

## Rules

- Never put a domain's atom names in core; `tests/test_pack_contract.py`
  enforces it.
- Write atoms and construct paths only through `store.py`.
- Never let a test reach the network or depend on a key being present
  (`conftest.no_live_calls`, `fake_keys`). Mark a live test `integration`;
  `tests/test_live_smoke.py` is the only one.
- Treat the data directory as unrecoverable: it is not in git.
- Never copy anything from a live pack, a data directory or a corpus into
  this repo: the history example's tables, examples and fixtures are written
  or anonymised by hand. `tests/test_no_live_data.py` checks their shape;
  `hooks/pre-commit` checks staged lines against `UNIVERSE_BUILDER_PRIVATE_DIRS`.
  On a failure, take the data out; never raise a budget to fit it.
- Run `--dry-run` before anything large.
- Never run `sources --via search` (paid) unless it is named explicitly.
- Keep the history pack's Wikimedia enrichment (`wikimedia`, `wikipedia`,
  `signals`, `media`) in the pack until a second domain wants it; then move
  it into core in the order `ARCHITECTURE.md` §0 gives.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
cp .env.example .env    # key for the active provider; UNIVERSE_BUILDER_DATA_DIR
```

- `.env`: the first found searching up from the working directory, then from
  the package, each search stopping at its project root.
- Data: `UNIVERSE_BUILDER_DATA_DIR`, else `./data/automated`, plus the domain
  segment.
- Pack: `UNIVERSE_BUILDER_DOMAIN` — a bundled name, a dotted module path, or a
  `universe_builder.domains` entry point.

## Running

```bash
universe_builder config | status [--coverage] | run --phase <p> [--dry-run] [--batch] | index
universe_builder duplicates | lint [--check C] [--review F [--via SOURCE]] | costs | collisions --seed NAME
universe_builder sources --via RESOLVER|search | retire --model TEXT | periods
universe_builder restore --attempt ID [--run RUN] [--dry-run | --apply]   # undo one in-place rewrite
universe_builder run --phase create --batch --experiment NAME [--variant V] [--sample-rate R] [--dry-run]
universe_builder experiment replay NAME --variant V [--atom T] [--per-type K] [--only IDS] [--apply] | collect NAME
universe_builder experiment judge NAME [--model M] [--effort E] [--sample N] [--apply] | report NAME [--panel J1,J2] | list
universe_builder experiment replay NAME --model-b M [--effort-a E] [--effort-b E] [--allow-dear-b] [--reuse-a FROM]
universe_builder run --phase <p> --effort low|medium|high|xhigh|max   # unset sends none
universe_builder run --phase <p> --task-budget N   # Anthropic only; >= 20000, <= max_tokens; unset sends none
python -m universe_builder.run_batch {models,estimate,submit,list,poll,collect,cancel,repin}   # cancel lists unless --apply
```

- Phases: `index` → `create` → `fix` → `verify` → `expand` → `append`.
- Undo an in-place rewrite with `restore --attempt ID --apply`. A new-file
  create and `fix` save nothing to restore (ARCHITECTURE §2.1e).
- Run prompt experiments on cheap models only (`pipeline.yaml`
  `experiments.max_output_price`); their output goes under
  `settings.experiments_dir()`, never the corpus (ARCHITECTURE §2.1g).
- Choose models by capability tier (`frontier`/`balanced`/`economy`) in
  `src/universe_builder/pipeline.yaml`.

## Output

- Atoms: `<data>/<domain>/{atom_type}/{atom_type}_h{content_id}_{slug}_{model}.json`;
  one topic is one file. Validation failures: `{filename}_ERROR`.
- Skip underscore-prefixed directories (`_runs/`, `_batches/`, `_merged/`,
  `_duplicates/`, `_retired/`, `_index/`) as machinery; `store.iter_atoms`
  does.

## Branches

- Promote and release with the `release` skill
  (`.claude/skills/release/SKILL.md`). Update it in the same commit as any
  change to the steps it names.
- Work on `dev`. Install the hooks with `make hooks`. Never commit directly
  on `main`.

## Architecture doc

- Treat `ARCHITECTURE.md` (an index) and its parts under `docs/architecture/`
  as canonical.
- Update the part a change touches when changing structure (atom specs,
  models, providers, passes, naming, paths, entry points). Add an index row
  only for a new part; split a part before it passes 20 KB.
- Cut history-corpus specifics down to the engine in each part you touch.
- Keep `tests/test_architecture_doc.py` and `tests/test_doc_budgets.py`
  green.

## Structure

```
src/universe_builder/
  cli.py registry.py atoms.py config.py pipeline.py pipeline.yaml settings.py
  store.py journal.py topics.py index.py migrations/ operations/ providers/
  llm_classes/ prompts/ models/subatomic.py
  duplicates.py lint.py costs.py collisions.py sources.py retire.py restore.py run_batch.py execute_prompt.py
  experiments.py prompt_spec.py experiment_stats.py   # prompt experiments
  domains/history/      # worked example: models/, periods.py, wikipedia.py, examples/, sample *.txt
  domains/medications/  # worked example (minimal)
docs/architecture/      # the architecture doc's parts
tests/                  # offline; no network, no cwd changes
```
