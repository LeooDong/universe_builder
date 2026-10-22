"""The package says which release it is.

It said `0.1.0` from the first commit through tag `v0.1.10`, so an installed
engine could not tell anyone — a journal, a bug report, the history pack's
pin check — which release it was. The version now has one source,
`universe_builder.__version__`, which `pyproject.toml` reads dynamically.
"""

from __future__ import annotations

import re
import subprocess
from importlib import metadata
from pathlib import Path

import pytest

import universe_builder

REPO = Path(__file__).resolve().parents[1]


def _as_tuple(version: str) -> tuple:
    return tuple(int(part) for part in re.findall(r"\d+", version)[:3])


def test_the_package_carries_a_version():
    assert re.fullmatch(r"\d+\.\d+\.\d+", universe_builder.__version__)


def test_pyproject_reads_the_one_source():
    """No second copy to forget: the metadata's version is the attribute."""
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'^dynamic\s*=\s*\[[^\]]*"version"', text, re.M)
    assert 'version = {attr = "universe_builder.__version__"}' in text
    assert not re.search(r'^version\s*=\s*"', text, re.M), "a static version is a second source"


def test_the_installed_metadata_agrees():
    try:
        installed = metadata.version("universe_builder")
    except metadata.PackageNotFoundError:
        pytest.skip("not installed; run from source")
    assert installed == universe_builder.__version__, (
        "the installed metadata is stale; reinstall (make install)")


def test_the_version_is_not_behind_the_newest_tag():
    try:
        tag = subprocess.run(["git", "describe", "--tags", "--abbrev=0", "--match", "v*"],
                             cwd=REPO, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git unavailable")
    if tag.returncode != 0:
        pytest.skip("no release tag in this checkout")
    assert _as_tuple(universe_builder.__version__) >= _as_tuple(tag.stdout.strip()), (
        f"version {universe_builder.__version__} is behind tag {tag.stdout.strip()}")
