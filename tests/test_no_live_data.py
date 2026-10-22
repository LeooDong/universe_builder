"""Keep live data out of this repo.

The history example's tables are samples written by hand, and every fixture is
written or anonymised for its test: nothing here is copied from a live pack or
a data directory. These checks fail on the shapes copied data has — a table or
fixture larger than a sample, a file nobody registered, a dated or signed note,
a corpus content id, a private name or path.

They check shape, never content: a list of private lines to refuse would itself
publish them. For the same reason the private names are held as hashes of
words. `hooks/pre-commit` adds the content check locally, against the
directories named in `UNIVERSE_BUILDER_PRIVATE_DIRS`.

A failure here means data came in. Take it out; do not raise a budget to fit it.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
HISTORY = REPO / "src" / "universe_builder" / "domains" / "history"
FIXTURES = REPO / "tests" / "fixtures"

#: Line budgets for the history example's tables: a sample's size, with room
#: for a few more hand-written lines and no more.
TABLE_LINES = {
    "aliases.txt": 80,
    "not_duplicates.txt": 60,
    "regime_changes.txt": 70,
    "HISTORY_list.txt": 30,
    "base_system.txt": 20,
}

#: The example atoms, by name. A new one is a deliberate change to this set.
EXAMPLES = {
    "antikythera_mechanism", "book_of_the_dead", "construction_of_the_great_pyramid_of_giza",
    "egyptian_history", "founding_of_cairo", "genghis_khan", "hagia_sophia", "hieroglyphic_writing",
    "invention_of_toothpaste", "memphis", "osiris", "pantheon_of_rome", "ptahhotep", "ptolemaic_kingdom",
    "roman_period_alexandria", "rosetta_stone", "rostam", "safavid_iran", "sardis", "shahnameh",
}
EXAMPLE_BYTES = 16_000

#: Every fixture, registered with its budget: (most files, most bytes in all).
#: A key ending in `/` covers the directory below it.
FIXTURE_BUDGETS = {
    "alias_files_parsed_v0.1.19.json": (1, 8_000),
    "aliases_v0.1.19.txt": (1, 4_000),
    "corpus_ids_sample.json": (1, 8_000),
    "journal_v0.1.19_readings.json": (1, 40_000),
    "name_keys.json": (1, 30_000),
    "not_duplicates_v0.1.19.txt": (1, 3_000),
    "golden_prompts/": (10, 80_000),
    "journal_v0.1.19/": (160, 250_000),
}

#: sha256 of lowercased words that name private projects. Hashes, so that this
#: file does not publish what it keeps out.
PRIVATE_WORDS = {
    "0a7ea9ff5a5270d7080048ef71482b3961f8fe9de2c8fcfebee9945d6965e2ea",
    "91c073731121e6d6a16d346590a3078c2262908840a023251b0eadc5b30c4064",
}

#: Marks of a note copied from a working table: a run date, a signature, a
#: pointer into a private archive or worklog, a spend.
NOTE_MARKS = re.compile(
    r"\b20\d\d-\d\d-\d\d\b|\b(Fable|Codex|Claude|Opus|Sonnet|Haiku|Leo)\b|\barchive/|WORKLOG|STATUS\.md|\$\d",
)
CONTENT_ID = re.compile(r"(?<![0-9a-f])h[0-9a-f]{16}(?![0-9a-f])")   # inside a filename too
#: Built in pieces, so this file does not match itself.
PRIVATE_PATH = re.compile("/" + "Users/|/" + "home/[a-z]")

SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", "build", "dist", "data"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".tsv", ".csv", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".cfg", ".ini", ""}


def _tables():
    return sorted(p.name for p in HISTORY.iterdir() if p.suffix in (".txt", ".tsv", ".csv"))


def _repo_files():
    """In a checkout, the files git would publish: tracked, plus untracked ones
    not ignored (a new copy is caught before it is staged; an ignored local file
    never is). Every file below the repo otherwise, as in an unpacked sdist."""
    try:
        command = ["git", "-C", str(REPO), "ls-files", "-z", "--cached", "--others", "--exclude-standard"]
        listed = subprocess.run(command, capture_output=True, check=True, timeout=30).stdout.decode().split("\0")
        return [REPO / name for name in listed if name]
    except (OSError, subprocess.SubprocessError):
        return sorted(REPO.rglob("*"))


def _text_files():
    for path in _repo_files():
        relative = path.relative_to(REPO)
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in relative.parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES and path.stat().st_size < 2_000_000:
            yield path


def _comments(path):
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if "#" in line:
            yield number, line.split("#", 1)[1]


def test_every_history_table_has_a_budget():
    assert _tables() == sorted(TABLE_LINES), "a new table needs a budget here, written by hand"


@pytest.mark.parametrize("name", sorted(TABLE_LINES))
def test_a_history_table_is_sample_sized(name):
    lines = len((HISTORY / name).read_text(encoding="utf-8").splitlines())
    assert lines <= TABLE_LINES[name], f"{name}: {lines} lines; the example's tables are small samples"


@pytest.mark.parametrize("path", [HISTORY / n for n in sorted(TABLE_LINES)]
                         + [FIXTURES / "aliases_v0.1.19.txt", FIXTURES / "not_duplicates_v0.1.19.txt"],
                         ids=lambda p: p.name)
def test_a_table_carries_no_working_notes_or_ids(path):
    marked = [(n, c.strip()) for n, c in _comments(path) if NOTE_MARKS.search(c)]
    ids = CONTENT_ID.findall(path.read_text(encoding="utf-8"))
    assert marked == [], f"{path.name}: notes copied from a working table: {marked[:5]}"
    assert ids == [], f"{path.name}: corpus content ids: {ids[:5]}"


def test_the_examples_are_the_registered_ones():
    found = {p.stem for p in (HISTORY / "examples").glob("*")}
    assert found == EXAMPLES, f"unregistered: {sorted(found - EXAMPLES)}; missing: {sorted(EXAMPLES - found)}"
    big = [p.name for p in (HISTORY / "examples").glob("*.json") if p.stat().st_size > EXAMPLE_BYTES]
    assert big == []


def _budget_key(relative: str):
    for key in FIXTURE_BUDGETS:
        if relative == key or (key.endswith("/") and relative.startswith(key)):
            return key
    return None


def test_every_fixture_is_registered_and_within_its_budget():
    used = {key: [0, 0] for key in FIXTURE_BUDGETS}
    unregistered = []
    for path in sorted(p for p in FIXTURES.rglob("*") if p.is_file()):
        relative = path.relative_to(FIXTURES).as_posix()
        key = _budget_key(relative)
        if key is None:
            unregistered.append(relative)
            continue
        used[key][0] += 1
        used[key][1] += path.stat().st_size
    over = {k: tuple(v) for k, v in used.items() if v[0] > FIXTURE_BUDGETS[k][0] or v[1] > FIXTURE_BUDGETS[k][1]}
    assert unregistered == [], f"fixtures with no budget: {unregistered[:10]}"
    assert over == {}, f"over budget (files, bytes): {over}"


def test_no_file_names_a_private_project_or_path():
    named = []
    for path in _text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        words = {hashlib.sha256(w.encode()).hexdigest() for w in re.findall(r"[a-z0-9]+", text.lower())}
        if words & PRIVATE_WORDS or PRIVATE_PATH.search(text):
            named.append(path.relative_to(REPO).as_posix())
    assert named == [], f"files naming a private project or a home path: {named}"
