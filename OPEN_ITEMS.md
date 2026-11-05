# Open items

The engine's backlog. Remove an item only when it ships; git keeps the closed ones. None is
urgent.

- **Batch judging for experiments.** `experiment judge` runs synchronously at full price; send it
  through the batch path. The replay estimate (`experiments.estimate`, `2000 * info.output`)
  assumes 2k output tokens per call, while `judge_estimate` defaults to `output_tokens=3000`.
  Measure the real figure from the journal before changing either.
- **Auto-accept safe `otherNames` alias lines.** Today `collisions --seed` and `duplicates`/`lint`
  turn an `otherNames` claim into an alias line only through a ruling. Write the line
  automatically when (a) the two spellings fold to one key (accents, apostrophes), or (b) both
  atoms cite the same external reference, by a rule the pack supplies. Everything else stays a
  reviewer's call. Make the evidence rules pack fields.
- **A scoped alias line loses to another scope's listing.** In `topics.py` `resolve`, the
  context-free name lookup runs before the scoped alias-line fallback, so another scope's listing
  of a bare name wins, or makes the name ambiguous. Repro with the bundled
  `src/universe_builder/domains/history/aliases.txt` and the `_history`/`_atom` helpers in
  `tests/test_index.py`: store both Thebes atoms; a Greek History lists `Thebes` and an Egyptian
  History lists `Thebes (Luxor)`; `resolve_subject("city", "Thebes", context="egyptian history")`
  returns None. With only the Greek listing it returns the Boeotian atom; it should return the
  `[Egyptian History]` line's atom both times. Moving the scoped step changes resolver
  precedence: write property tests first.
