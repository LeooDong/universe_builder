"""The release skill names only paths, targets and tests that exist.

`.claude/skills/release/SKILL.md` is the release procedure. A renamed
Makefile target or a moved file leaves it giving instructions that no longer
work, and nothing else would notice until the next release.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKILL = ROOT / ".claude" / "skills" / "release" / "SKILL.md"


def _text() -> str:
    return SKILL.read_text(encoding="utf-8")


def test_the_skill_has_its_frontmatter():
    text = _text()
    assert text.startswith("---\nname: release\ndescription: "), "frontmatter must open the file"


def test_every_repo_path_it_names_exists():
    named = set(re.findall(r"`([A-Za-z0-9_./-]+/[A-Za-z0-9_.-]+)`", _text()))
    missing = sorted(n for n in named
                     if not n.startswith("../") and "<" not in n and not (ROOT / n).exists())
    assert not missing, f"the release skill names paths that do not exist: {missing}"


def test_every_make_target_it_names_exists():
    targets = set(re.findall(r"^([A-Za-z][\w-]*):", (ROOT / "Makefile").read_text(), re.M))
    named = set(re.findall(r"\bmake ([a-z][\w-]*)", _text()))
    assert named, "the skill names no make target"
    assert not named - targets, f"make targets that do not exist: {sorted(named - targets)}"


def test_every_test_it_names_exists():
    tests = "\n".join(p.read_text(encoding="utf-8") for p in (ROOT / "tests").glob("test_*.py"))
    named = set(re.findall(r"`(test_\w+)`", _text()))
    missing = sorted(n for n in named if f"def {n}" not in tests)
    assert not missing, f"tests the skill names that do not exist: {missing}"
