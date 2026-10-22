# Origin

This engine was written inside a private project that generates a history
corpus, and extracted from it on 2026-09-24 so that other domains could use
it. This public repository starts from a single commit: the private history —
worklogs, commit messages with cost figures, review files and the corpus
itself — stays private.

The `history` domain pack under `src/universe_builder/domains/history/` is a
worked example, not a copy of that project's data. Its code follows the pack
used there; its tables are small samples written by hand; its few-shot
examples are described in its `README.md`. `tests/test_no_live_data.py` keeps
it that way. `medications` is the minimal example.
