"""Validate the generated corpus on disk, when there is one.

`data/automated/` is gitignored and machine-local, so every test here skips when
it is absent. On a machine that has it, this is the check that would have caught
B7 immediately: it was not a subtle failure, it was every city file.
"""

from __future__ import annotations

import json

import pytest

from universe_builder import settings

#: This checkout's corpus, explicitly — not `settings.data_dir()`, which
#: follows `UNIVERSE_BUILDER_DATA_DIR` and the working directory and made these tests
#: skip with "no generated corpus" when run from anywhere but the root.
DATA = settings.REPO_ROOT / "data" / "automated" / "history"
ATOMS = ["artifact", "city", "discovery", "event", "history", "person"]
EMPTY_SENTINEL = {"error": "No data found"}

pytestmark = pytest.mark.skipif(not DATA.is_dir(), reason="no generated corpus on this machine")


@pytest.fixture(scope="module")
def resolver():
    """Resolve stored payloads without depending on the working directory."""
    from universe_builder import execute_prompt

    return execute_prompt


def stored_files(atom):
    folder = DATA / atom
    if not folder.is_dir():
        return []
    return [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix == ".json" and not p.name.startswith("append_")
    ]


@pytest.mark.parametrize("atom", ATOMS)
def test_every_stored_file_validates(atom, resolver):
    """Each file must validate against the model its own shape resolves to."""
    failures = []
    for path in stored_files(atom):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            failures.append(f"{path.name}: unparseable ({e})")
            continue
        if data == EMPTY_SENTINEL:
            continue
        model = resolver.resolve_stored_class(atom, data)
        try:
            model.model_validate(data)
        except Exception as e:
            failures.append(f"{path.name}: {str(e).splitlines()[0]} (as {model.__name__})")

    assert not failures, f"{len(failures)} invalid {atom} files:\n  " + "\n  ".join(failures[:20])


def test_no_file_is_unparseable_json():
    bad = []
    for atom in ATOMS:
        for path in stored_files(atom):
            try:
                json.loads(path.read_text())
            except json.JSONDecodeError as e:
                bad.append(f"{path}: {e}")
    assert not bad, bad


def test_expanded_cities_resolve_to_the_expanded_model(resolver):
    """B7 regression against real data: a city carrying Population objects must
    resolve to `CityExpanded`, not `City`. Resolving it to `City` is what would
    let the fix/verify passes collapse the expansion."""
    from universe_builder.domains.history.models.city import CityExpanded

    files = stored_files("city")
    if not files:
        pytest.skip("no city files")

    expanded = []
    for path in files:
        data = json.loads(path.read_text())
        if data == EMPTY_SENTINEL:
            continue
        if any(isinstance(x, dict) for x in data.get("cityPeriodsOfHistory") or []):
            expanded.append(path)
            assert resolver.resolve_stored_class("city", data) is CityExpanded, path.name

    if expanded:
        # Report the scale — this is the number that made B7 urgent.
        print(f"\n{len(expanded)} of {len(files)} city files are expanded")


def test_one_file_per_atom():
    """The atom folders are the canonical location: one file per content id.

    Two deliberate exceptions. `history` keeps one file per model — the work
    list is the union of every model's topic list, so the pair is coverage,
    not duplication. And `append_*` files are transient inputs: they exist
    between the append pass and the merge that promotes them in place, after
    which they retire to `_merged/`. Everything else sharing an id is a
    duplicate — usually a deliberate regeneration under a second model that
    nothing reconciled — and this is the test that stops it going unnoticed.
    """
    from collections import Counter

    from universe_builder import store

    duplicated = {}
    for atom in ATOMS:
        if atom == "history":
            continue
        ids = Counter()
        for path in stored_files(atom):
            try:
                parsed = store.AtomName.decompose_filename(path.name)
            except ValueError:
                continue  # test_no_file_is_unparseable_json owns that complaint
            ids[parsed.padded_id] += 1
        for content_id, count in ids.items():
            if count > 1:
                duplicated[f"{atom}:{content_id}"] = count

    assert not duplicated, (
        f"more than one file for one atom: {duplicated}. Decide a winner and "
        f"retire the rest — nothing reads two files for one id sensibly."
    )


def test_no_final_files_remain():
    """The merge promotes in place; `final_` outputs are a retired convention.

    One lingering `final_` file means an old merge ran and its output was
    never adopted — the atom beside it is missing its append content while a
    file claiming to be the complete version sits unread."""
    stray = sorted(str(p.relative_to(DATA)) for p in DATA.rglob("final_*"))
    assert not stray, stray


def test_the_empty_sentinel_count_is_visible():
    """EmptyModel files are wasted generations, not errors. Surfacing the count
    keeps them from being forgotten (B8)."""
    sentinels = [
        path
        for atom in ATOMS
        for path in stored_files(atom)
        if json.loads(path.read_text()) == EMPTY_SENTINEL
    ]
    print(f"\n{len(sentinels)} EmptyModel sentinel files:")
    for path in sentinels:
        print(f"  {path.relative_to(DATA)}")
