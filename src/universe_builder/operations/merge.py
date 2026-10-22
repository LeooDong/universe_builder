"""Merging an append file into the atom it supplements, in place.

Formerly `functions/merge_helper.py` (moved in v0.3.0; the
old module re-exports this one for one release).

The pairing rule is the interesting part: given `city_…` and `append_city_…`
files sitting in one directory, decide which append belongs to which city. See
`pair_keys` for why that rule changed with content-addressed ids and what it
unblocks.

The merge *promotes*: the merged content replaces the base atom and the
consumed append moves to `_merged/`, so the atom folders keep exactly one file
per atom. An earlier version wrote a third `final_…` file beside the pair,
which left "which file is the city?" permanently ambiguous — nothing consumed
the `final_` files, and `iter_atoms` and the status counts would have counted
the same city twice.
"""

import json
import logging
import os

from universe_builder import journal, store
from universe_builder.atoms import resolve_stored_class
from universe_builder.store import AtomName

log = logging.getLogger(__name__)


def merge_json_files(filename, append_filename) -> bool:
    """Merge one append into its base atom, in place. True when promoted.

    The rules all point the same way: **an append adds, it never takes away.**

    - An unreadable half skips the *pair*, loudly, and writes nothing. Raising
      aborts every valid pair behind this one — a failure mode fixed several
      separate times. Skipping keeps both inputs in place for a
      later run and costs only this pair, this pass — but it must be an error
      in the log and a False to the caller, because a skip nobody hears about
      is how an append gets lost.
    - A key the base already holds keeps the base's value unless both sides
      are lists. The append prompt does not constrain which keys come back, so
      a model volunteering `name` alongside its list must not replace fields
      that already passed validation. Agreement is silent; disagreement is
      logged and the base wins.
    - Lists extend with only the entries the base does not already hold. That
      is what makes the merge idempotent — the write and the retirement of the
      append are two steps, and a crash between them must not mean the next
      run doubles every list entry.
    - **The merged payload must validate before it may replace the base.**
      This function used to write an unvalidated dict to a side file; writing
      *in place* makes validation the difference between a merge and B8. The
      shape is resolved from the payload (an expanded city validates as
      `CityExpanded`, not the seed `City` — B7), and a merge that does not
      validate leaves both files untouched.

    On success the base file holds the merged atom and the append is retired
    to `_merged/` — kept, not deleted, because `data/` has no undo.
    """
    base_name = os.path.basename(str(filename))
    append_name = os.path.basename(str(append_filename))
    try:
        main_data = json.loads(store.read(filename))
        append_data = json.loads(store.read(append_filename))
    except (OSError, UnicodeDecodeError, ValueError) as e:
        log.error(
            "Not merging %s with %s: %s. Both files are left in place; fix the "
            "unreadable one and re-run the merge.", base_name, append_name, e,
        )
        return False
    if not isinstance(main_data, dict) or not isinstance(append_data, dict):
        log.error(
            "Not merging %s with %s: both must hold a JSON object, got %s and %s.",
            base_name, append_name,
            type(main_data).__name__, type(append_data).__name__,
        )
        return False

    for key, value in append_data.items():
        if key not in main_data:
            main_data[key] = value
        elif isinstance(main_data[key], list) and isinstance(value, list):
            main_data[key].extend(v for v in value if v not in main_data[key])
        elif main_data[key] != value:
            log.warning(
                "Keeping %r from %s; the append offered a different value and an "
                "append may not replace a validated field.", key, base_name,
            )

    try:
        atom_type = store.atom_type_of(filename)
        merged = resolve_stored_class(atom_type, main_data).model_validate(main_data)
    except Exception as e:  # noqa: BLE001 - a bad merge must cost one pair, not the pass
        log.error(
            "Not merging %s with %s: the merged payload does not validate (%s). "
            "Both files are left untouched.", base_name, append_name, e,
        )
        return False

    # The merge rewrites the base atom in place: journal it as a mechanical
    # write that keeps the bytes it replaces (v0.3.2), so `restore` can undo it.
    journal.record_mechanical("merge", atom_type, filename, f"merged {append_name} in place",
                              kind="merge", save_replaced=True)
    store.write_atom(filename, merged)
    store.retire_merged_input(append_filename)
    return True


def pair_keys(parsed: AtomName) -> list:
    """The keys under which `parsed` may be paired with its counterpart.

    **The content id.** `store.atom_id` keys on the base atom type, so
    `city_h3f8…` and `append_city_h3f8…` carry the identical token for one
    subject *whatever model wrote either of them*, which is what lets the append
    phase use a different model from create.

    A name without a content id has no key and pairs with nothing. Until v0.3.0
    a second key, slug plus model string, bridged legacy counter-id names
    (`city_0000000042_…`); the corpus was renamed and holds none,
    so that bridge went with the rest of the legacy-id support.
    """
    if store.is_hash_id(parsed.padded_id):
        return [("id", parsed.padded_id)]
    return []


def _parse(filename: str):
    """Decompose, or None. A name this module cannot read is not a merge candidate.

    Skipping beats raising: `process_directory` walks a whole tree and one
    stray file should not abort the merge of every valid pair after it.
    """
    try:
        return AtomName.decompose_filename(filename)
    except ValueError:
        log.info("Skipping %s: not a filename this pipeline wrote", filename)
        return None


def process_directory(root_directory, atom_name):
    """Merge every `append_<atom>_…` into its `<atom>_…` partner under `root_directory`.

    Walks the atom folders only. Underscore-prefixed directories are machinery
    (`_merged/`, `_retired/`, `_batches/`, a pack's `_backups/`) and are pruned
    from the walk: it used to descend into them, so a pair kept in a backup was
    merged and its append moved out of the backup.
    """
    skipped = 0
    atom_prefix = f"{atom_name}_"
    append_atom_prefix = f"append_{atom_name}_"

    for dirpath, dirnames, all_files in os.walk(root_directory):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("_"))
        files = sorted(f for f in all_files if f.startswith(atom_prefix) and f.endswith(".json"))
        append_files = sorted(f for f in all_files if f.startswith(append_atom_prefix) and f.endswith(".json"))

        # Index the appends once per directory rather than re-scanning them for
        # every atom file; the corpus is expected to reach six figures.
        appends: dict = {}
        for append_filename in append_files:
            parsed_append = _parse(append_filename)
            if parsed_append is None:
                continue
            for key in pair_keys(parsed_append):
                appends.setdefault(key, []).append(append_filename)

        for filename in files:
            parsed_file = _parse(filename)
            if parsed_file is None:
                continue
            # dict rather than set: the order has to be stable for a repeatable
            # result. Pairing is on the content id alone, so two subjects that
            # merely slug alike (`St. Petersburg`, `St Petersburg`) never pair.
            matches: dict = {}
            for key in pair_keys(parsed_file):
                for name in appends.get(key, []):
                    matches.setdefault(name, None)
            for append_filename in matches:
                city_filepath = os.path.join(dirpath, filename)
                append_filepath = os.path.join(dirpath, append_filename)
                if not merge_json_files(city_filepath, append_filepath):
                    skipped += 1

    # The per-pair message names the files and the reason (unreadable, or a
    # merge that would not validate); this one exists so a long run's tail says
    # whether everything merged without scrolling back through it.
    if skipped:
        log.error("%d pair(s) skipped; their appends are NOT merged — see the "
                  "per-pair errors above", skipped)
