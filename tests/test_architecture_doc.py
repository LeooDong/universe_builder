"""`ARCHITECTURE.md` must not describe code that no longer exists.

CLAUDE.md says a stale diagram is worse than none, and once all three
had rotted anyway — the flowchart still named `OpenAICall` and `ClaudeCall`,
deleted the day before, and omitted the whole of `operations/` and `migrations/`;
the class diagram was missing `CityExpanded` and `Population`, which exist
because of B7; the sequence diagram described the fix pass as "overwrite .json,
delete .json_ERROR", which is the pre-B8 ordering that lost data.

None of that was caught, because nothing checked. These tests are cheap and
mechanical: they cannot tell whether a diagram is a *good* description, only that
it does not name things that are gone or omit things that are central. That is
the failure mode that actually occurred.

The doc is an index (`ARCHITECTURE.md`) and its parts (`docs/architecture/*.md`);
every check reads all of them.
"""

from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "ARCHITECTURE.md"
PARTS = ROOT / "docs" / "architecture"
SRC = ROOT / "src" / "universe_builder"


def _text() -> str:
    return "\n".join(p.read_text() for p in [DOC, *sorted(PARTS.glob("*.md"))])


def _diagrams() -> str:
    return "\n".join(re.findall(r"```mermaid\n(.*?)```", _text(), re.S))


def _defined_classes() -> set:
    names = set()
    for path in SRC.rglob("*.py"):
        names |= set(re.findall(r"^class (\w+)", path.read_text(), re.M))
    return names


class TestTheDiagramsDescribeCodeThatExists:
    def test_at_least_one_diagram_is_present(self):
        assert _diagrams().strip(), "ARCHITECTURE.md has no mermaid diagrams at all"

    def test_every_module_named_exists(self):
        files = {p.name for p in SRC.rglob("*.py")} | {"pipeline.yaml"}
        named = set(re.findall(r"\b([a-z_]+\.py)\b", _diagrams()))
        missing = sorted(named - files)

        assert not missing, f"diagrams name modules that do not exist: {missing}"

    def test_every_class_named_exists(self):
        named = set(re.findall(r"^\s*class (\w+)", _diagrams(), re.M))
        missing = sorted(named - _defined_classes())

        assert not missing, f"diagrams name classes that do not exist: {missing}"

    @pytest.mark.parametrize("gone", ["OpenAICall", "ClaudeCall",
                                      "OpenAICallAugment", "ClaudeCallAugment"])
    def test_the_deleted_call_classes_are_not_referenced(self, gone):
        """Deleted, yet the flowchart still showed them a day later."""
        assert gone not in _diagrams(), (
            f"{gone} was deleted but is still in a diagram"
        )


class TestTheDiagramsCoverWhatMatters:
    """Omission is the other half, and the harder one to notice."""

    #: Things whose absence would make the diagrams actively misleading rather
    #: than merely incomplete. Deliberately short — this is a rot alarm, not a
    #: specification of what the diagrams should contain.
    REQUIRED = [
        "operations/",     # where every pass now lives
        "store.py",        # the only writer
        "journal.py",      # the append-only record
        "topics.py",       # where work lists come from
        "cli.py",          # the entry point
        "migrations/",     # migrate-on-read
        "batch.py",        # the batch path
        "CityExpanded",    # the B7 stage model
        "Population",
        "EmptyModel",      # the sentinel
    ]

    @pytest.mark.parametrize("thing", REQUIRED)
    def test_it_appears_somewhere(self, thing):
        assert thing in _diagrams(), (
            f"{thing} is central to how the system works and appears in no diagram"
        )


class TestTheDocDoesNotPointAtMovedFiles:
    def test_every_repo_path_it_names_exists(self):
        """A file the doc names can move or go; the doc must follow it."""
        text = _text()
        named = set(re.findall(r"`([A-Za-z0-9_/.]+\.(?:md|py|ya?ml|json|txt))`", text))
        missing = []
        for name in sorted(named):
            if "{" in name or "*" in name:
                continue
            if any((base / name).exists() for base in (ROOT, SRC, SRC.parent)):
                continue
            # Bare filenames are referred to without their directory all over the
            # doc; only flag a path that looks rooted at the repo.
            if "/" in name and not (ROOT / name).exists():
                missing.append(name)
        assert not missing, f"ARCHITECTURE.md names paths that do not exist: {missing}"


class TestTheIndexListsEveryPart:
    def test_every_part_is_linked_from_the_index(self):
        linked = set(re.findall(r"\]\(docs/architecture/([^)]+\.md)\)", DOC.read_text()))
        parts = {p.name for p in PARTS.glob("*.md")}

        assert parts, "docs/architecture/ has no parts"
        assert not parts - linked, f"parts the index does not list: {sorted(parts - linked)}"
        assert not linked - parts, f"the index links parts that do not exist: {sorted(linked - parts)}"
