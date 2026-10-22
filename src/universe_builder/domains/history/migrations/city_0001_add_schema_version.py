"""city v0 -> v1: give a city file the version marker it was written without.

The first migration for any atom type has to be this one, and it has to add nothing else.
Every city file in `data/automated/city/` predates versioning, so v0 is not a historical
shape that a handful of stragglers are stuck in — it is the shape of the entire stored
corpus. Bundling a real structural change into the same step would mean the first thing
migrate-on-read ever did to the whole stored corpus was rewrite its contents, with no
earlier version of itself to have been exercised against it.

So v1 is v0 plus one key. What it buys is that from here on a city file *states* its shape
instead of having it sniffed: `execute_prompt.STAGE_MODELS` currently distinguishes a
`City` from a `CityExpanded` by looking for a dict inside `cityPeriodsOfHistory`, which is
a guess that happens to be reliable. Note that this migration does
*not* try to replace that guess — the create and expand shapes are both v1, because they
are two stages of one schema rather than two versions of it. Splitting them is a later
migration, once the version marker it would key off is actually present on disk.
"""

from __future__ import annotations

from universe_builder.migrations import SCHEMA_VERSION_KEY, register


@register("city", from_version=0, description="Stamp schema_version on pre-versioning city files")
def add_schema_version(payload: dict) -> dict:
    """Copy the payload and mark it v1, leaving every existing key exactly as it was.

    A shallow copy is enough because nothing nested is read or written; the caller's
    payload is protected from the top-level assignment, and `migrate_to_current` has
    already deep-copied anyway. Setting the key here rather than relying on the
    framework's stamp is deliberate: a migration should be readable on its own as the
    complete statement of what v0 -> v1 does.
    """
    migrated = dict(payload)
    migrated[SCHEMA_VERSION_KEY] = 1
    return migrated
