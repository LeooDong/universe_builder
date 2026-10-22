Part of [ARCHITECTURE.md](../../ARCHITECTURE.md); the index there lists every part.

## 3. The pipeline, pass by pass

```mermaid
sequenceDiagram
    participant U as universe_builder CLI
    participant FS as data/automated/
    participant J as journal (_runs/)
    participant LLM as provider

    Note over U: phase index — active pack index
    U->>LLM: one prompt per seed subject
    LLM-->>FS: pack.index atom, with schema stamp

    Note over U: phase create
    U->>FS: topics.from_history() — de-duplicated
    loop each subject, batched by context
        U->>LLM: create prompt + schema + example
        LLM-->>J: raw payload recorded before any parse
        alt validates
            LLM-->>FS: {atom}/{atom}_{contentid}_{slug}_{model}.json
        else fails
            LLM-->>FS: the same name + _ERROR
        end
    end

    Note over U: phase fix — _ERROR files only
    U->>LLM: resend the failed payload
    LLM-->>J: raw payload recorded
    alt validates
        Note over FS: commit_fix — lock, refuse existing target,<br/>stamp/write atom, then retire sentinel
        LLM-->>FS: .json written, .json_ERROR removed
    else fails
        Note over FS: sentinel kept, never overwritten<br/>with the repair's own failure
    end

    Note over U: phase verify — valid files only
    U->>LLM: improve accuracy, fill nulls
    LLM-->>FS: write_atom — validated, schema stamped

    Note over U: phase expand / append
    U->>LLM: active specs — one call per list entry / supplement
    LLM-->>FS: locked, stamped expansion only on progress; append then merge
```

**Pass 1 — seed** (`--phase index`). Generates `active_pack().index` from the
pack's seed resource (or override). History uses civilisation names; other
packs supply their own subjects. The phase was once called `history` —
history's index atom *is* a History, and every domain's seed
phase went by that name. `--phase history`, a `history:` key in
`pipeline.yaml` and `resolve_model("history")` are all still understood and
mean `index` (`cli.LEGACY_PHASES`, `pipeline.LEGACY_PHASES`); `costs` reads
journals written before the rename under the new label.

**Pass 2 — create** (`run_phase`). The index atom is
excluded by its declared name. Each base spec's plural selects its topics from
the stored index atoms. `atoms.BASE_ATOM_TYPES` is rebuilt in place on reload,
including the reference exported by `execute_prompt`. Context stays in the
cached system text, identically framed on sync and batch paths.

**Pass 3 — fix, then verify** (`run_phase`, one phase per run). `fix` targets only `_ERROR` files
and, on success, writes the corrected `.json` and deletes the `_ERROR` file. `verify`
targets only valid files and asks for improved accuracy and filled-in nulls without
changing the shape. Files whose contents are `EmptyModel` are skipped. Neither touches the index atom
type unless `--atom` names it, and `verify` refuses it even then (`work.takes`).

**Pass 4 — expand, append, merge** (`run_phase`). Work is selected through each
atom's `expand`/`append` specs.
For history, `expand` walks `City.cityPeriodsOfHistory` and replaces each period string with a full `Population`
object, passing the neighbouring period as context. `append` (off by default) generates a
`CityAppend` alongside the main file, inheriting its content id.
`operations.merge.process_directory` then **promotes in place**: the merged payload is
validated (against the shape on disk — an expanded city is a `CityExpanded`, see B7) and,
only then, written over the city file; the consumed append retires to `_merged/`. The
atom folders therefore hold exactly one file per atom — `tests/test_corpus.py` asserts
it corpus-wide, `history` excepted (one file per model there is coverage, since the work
list is the union of every model's topic list). An earlier convention wrote a third
`final_city_*.json` beside the pair, which nothing consumed and which double-counted the
city in every folder walk; the merge no longer produces them and the corpus test refuses
strays. List merging is idempotent (entries the base already holds are not re-added), so
a crash between the write and the retirement costs nothing on the re-run.
