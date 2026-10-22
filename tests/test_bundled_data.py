"""What the package ships.

Every non-Python file under `src/universe_builder` is covered by a
`[tool.setuptools.package-data]` pattern. A file the patterns miss is present
in a checkout and absent from a wheel, which no import error ever reports: the
history example's `README.md` was missing that way.

The history example's tables are written by hand and never copied from a
live pack; `tests/test_no_live_data.py` holds them to that.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:   # Python 3.10
    tomllib = None

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "src" / "universe_builder"


def _package_data_patterns() -> list:
    if tomllib is None:
        pytest.skip("tomllib needs Python 3.11")
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    return config["tool"]["setuptools"]["package-data"]["universe_builder"]


def test_every_data_file_in_the_package_is_shipped():
    patterns = _package_data_patterns()
    unshipped = []
    for path in sorted(PACKAGE.rglob("*")):
        if not path.is_file() or path.suffix in (".py", ".pyc") or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(PACKAGE).as_posix()
        if not any(fnmatch.fnmatchcase(relative, pattern) for pattern in patterns):
            unshipped.append(relative)
    assert unshipped == [], f"not in [tool.setuptools.package-data]: {unshipped}"


def test_the_history_readme_is_shipped():
    """Named on its own because it is the one that was missing."""
    assert "domains/history/README.md" in _package_data_patterns()
