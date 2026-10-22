# The history pack — a worked example

The richer of the two bundled examples: an index atom (`history`) whose lists
drive nine generated atom types, one of which (`city`) has a second on-disk
shape (`stages`), an `ExpandSpec` and an `AppendSpec`; declared `link_fields`,
`coordinate_paths` and `date_paths`; an alias file and a seed list; a domain
operation (`periods`); a free URL and identity resolver (`wikipedia.py`); and a
persona (`base_system.txt`). The `medications` pack next door is the minimal
example.

Its code follows a pack used in production, with docstrings worded for this
context and imports rewritten to this package. That pack has grown operations
this example leaves out; this one creates from bare names.

What is here, and where it came from:

- **Tables** (`aliases.txt`, `not_duplicates.txt`, `regime_changes.txt`) are
  small samples written by hand for this example — a few lines of each kind,
  under the header that documents the format. They are not curated tables and
  are never copied from one: `tests/test_no_live_data.py` holds them to a
  sample's size and fails on notes copied from a working table.
- **`HISTORY_list.txt`** is a hand-written seed of broad civilisation names.
- **`examples/`** holds the few-shot examples the prompts show the model.
  `egyptian_history.json` and `sardis.json` were written by hand; the rest
  are generator output, checked and cleaned by hand.

If you build a pack of your own, start from `medications`; read this one for
the parts `medications` does not use.
