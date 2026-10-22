"""The docs a session reads first stay small enough to read whole.

`ARCHITECTURE.md` reached 207 KB, so every "keep it current" edit pulled tens
of thousands of tokens into context. It is now an index plus one part per area
under `docs/architecture/`. A failure is fixed by moving text out (into a part,
or a new part linked from the index), not by raising the budget.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: File -> its budget in bytes, and what to do when it is over.
BUDGETS = {
    "CLAUDE.md": (7_000, "loaded into every session: move detail to the doc it belongs in"),
    "README.md": (8_000, "move detail into ARCHITECTURE.md's parts"),
    "ARCHITECTURE.md": (20_000, "the index only: move detail into docs/architecture/"),
}

#: Every part of the architecture doc.
PART_BUDGET = 20_000


@pytest.mark.parametrize("name", sorted(BUDGETS))
def test_doc_is_within_its_budget(name):
    path = ROOT / name
    assert path.is_file(), f"{name} is missing"
    budget, fix = BUDGETS[name]
    size = path.stat().st_size
    assert size <= budget, f"{name} is {size:,} bytes, over its {budget:,}: {fix}"


@pytest.mark.parametrize("part", sorted(p.name for p in (ROOT / "docs" / "architecture").glob("*.md")))
def test_architecture_part_is_within_its_budget(part):
    size = (ROOT / "docs" / "architecture" / part).stat().st_size
    assert size <= PART_BUDGET, (
        f"docs/architecture/{part} is {size:,} bytes, over {PART_BUDGET:,}: "
        "split it and add the new part to ARCHITECTURE.md's index"
    )
