"""Filesystem locations.

Package *resources* (prompts, pipeline.yaml, pricing) resolve from this package,
wherever it is installed. Generated *data* does not: it is wherever
`UNIVERSE_BUILDER_DATA_DIR` says, else `./data/automated` under the process working
directory. The data root was once derived from the package's own
location (`<checkout>/data`), which is right for an editable install of this
repo and wrong for an installed engine, whose "package location" is
site-packages. This repo sets `UNIVERSE_BUILDER_DATA_DIR` in its `.env`, which
`load_environment` finds from the working directory or from the package, so
its scripts still run from any directory.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

log = logging.getLogger(__name__)

PACKAGE_DIR = Path(__file__).resolve().parent
#: The checkout this package is imported from, if it is one (`src/` layout).
#: Used only to record the code revision in the journal; never for data.
REPO_ROOT = PACKAGE_DIR.parent.parent if PACKAGE_DIR.parent.name == "src" else PACKAGE_DIR.parent
#: The default data base when `UNIVERSE_BUILDER_DATA_DIR` is unset, relative to the
#: working directory. `data_dir()` appends `automated/<domain>`.
DEFAULT_DATA_ROOT = Path("data")
PROMPTS_DIR = PACKAGE_DIR / "prompts"
BASE_SYSTEM = PROMPTS_DIR / "base_system.txt"
OPERATIONS_DIR = PROMPTS_DIR / "operations"
EXAMPLES_DIR = PROMPTS_DIR / "examples"
# Override knobs (default None): when set they win over the active pack's own
# seed/alias files, which is how tests isolate and how an explicit --list/--aliases
# could point elsewhere. The real defaults live on the domain pack
# (`index_seed_path`, `aliases_path`); a domain's seed data is no longer in core.
HISTORY_LIST = None
ALIASES_LIST = None
#: Overrides the pack's `create_notes_path` (tests point it at their own file).
CREATE_NOTES = None
#: Overrides the pack's `create_neighbours` (tests set it; None = the pack's).
CREATE_NEIGHBOURS = None
MODEL_PRICING = PACKAGE_DIR / "providers" / "model_pricing.json"


def data_dir() -> Path:
    """Generated-atom directory for the active domain: ``<base>/<domain>``.

    Namespaced by domain so two domains never share a folder, id space, journal
    or batch queue — atom folders and content ids carry no domain in them, so a
    shared base would let two domains collide (same folder, same id, wrong model
    on read). `UNIVERSE_BUILDER_DATA_DIR` overrides the *base*; the domain segment is always
    appended. Read at use time so a test's override and pack switch both apply.
    """
    from universe_builder.registry import active_pack

    override = os.environ.get("UNIVERSE_BUILDER_DATA_DIR")
    if override is None:
        base = DEFAULT_DATA_ROOT / "automated"
        _warn_if_a_checkout_corpus_is_being_ignored(base)
    else:
        base = Path(override).expanduser()
    # A relative path is relative to the working directory, like any other
    # relative path a command is given — not to the package.
    return base.resolve() / active_pack().name


_warned_about_default = False


def _warn_if_a_checkout_corpus_is_being_ignored(base: Path) -> None:
    """Once per process: `UNIVERSE_BUILDER_DATA_DIR` is unset, the cwd default is empty,
    and this package's checkout holds a corpus. That is almost always someone
    running from outside the checkout without the variable — and without the
    warning, `status` says "0 atoms" and `run` would start a second corpus
    under the working directory, on a paid model."""
    global _warned_about_default
    if _warned_about_default:
        return
    checkout = REPO_ROOT / "data" / "automated"
    if checkout.is_dir() and not base.is_dir():
        _warned_about_default = True
        log.warning("UNIVERSE_BUILDER_DATA_DIR is unset: data resolves to %s, which does not exist, "
                    "while a corpus exists at %s. Set UNIVERSE_BUILDER_DATA_DIR (this checkout's "
                    ".env does) if that corpus is the one you mean.",
                    base.resolve(), checkout)


def experiments_dir() -> Path:
    """Where prompt experiments keep everything: ``<base>/experiments/<domain>``.

    **Outside the corpus on purpose** (v0.5.0). An experiment's outputs are
    test data: a B atom, its batch handle, its raw replies, its judgements.
    None of it may be read as corpus by anything — `iter_atoms`, the index,
    `collect --all`, `costs`, the app's bake — and the surest way to keep a
    reader from finding it is for it not to be under the data directory at
    all. The default is the sibling of the data base (`data/automated` ->
    `data/experiments`); `UNIVERSE_BUILDER_EXPERIMENTS_DIR` overrides the base.
    A root that overlaps the data directory either way is refused.
    """
    from universe_builder.registry import active_pack

    data = data_dir()
    override = os.environ.get("UNIVERSE_BUILDER_EXPERIMENTS_DIR")
    base = Path(override).expanduser().resolve() if override else data.parent.parent / "experiments"
    root = base / active_pack().name
    if root == data or data in root.parents or root in data.parents:
        raise ValueError(
            f"The experiments root {root} overlaps the data directory {data}. "
            "Experiment output is test data and must live outside the corpus; "
            "point UNIVERSE_BUILDER_EXPERIMENTS_DIR somewhere else.")
    return root


def batch_dir() -> Path:
    return data_dir() / "_batches"


def missing_dir() -> Path:
    # Keep reports outside the batch handle glob.
    return batch_dir() / "missing"


def collected_dir() -> Path:
    """Where a handle goes once its results are safely on disk.

    A subdirectory, so `batch_dir().glob("*.json")` stops returning it and
    `collect --all` no longer re-collects batches it already finished. Kept
    rather than deleted: it is the only record of what a batch cost and which
    subjects it covered.
    """
    return batch_dir() / "collected"


def index_dir() -> Path:
    """Derived cross-reference indexes (`universe_builder index`). Underscore-prefixed:
    machinery, never atoms, always recomputable from the corpus."""
    return data_dir() / "_index"


def merged_dir() -> Path:
    """Where a consumed append file goes once its content is merged in place.

    Underscore-prefixed like `_runs` and `_batches`, so `store.iter_atoms` and
    the status counts never see it. Kept rather than deleted — `data/` is not
    in git and there is no undo — but out of the atom folders, which hold
    exactly one file per atom once a merge has run.
    """
    return data_dir() / "_merged"


def retired_dir() -> Path:
    """Where an atom goes when it is retired to be regenerated by a better model.

    Distinct from `_duplicates/`, and the difference is intent. A duplicate is a
    *rival* atom for a subject that already has one, and retiring it is the end
    of the story. A superseded atom is the **only** atom for its subject: once it
    moves, its topic has no atom, `topics.needing_create` sees a gap, and the
    next create run regenerates it. Keeping the two apart means a later reader
    can tell "we chose against this" from "this is waiting to be replaced".

    A move, never an unlink, for the usual reason: `data/` is not in git, and a
    regeneration that fails or comes back worse must be recoverable.
    """
    return data_dir() / "_retired"


def duplicates_dir() -> Path:
    """Where a retired duplicate atom goes when `universe_builder duplicates --apply` runs.

    Two atoms for one subject break the one-file-per-atom rule in spirit even
    though their content ids differ, which is exactly why `tests/test_corpus.py`
    cannot see them: the ids *are* different, because the two models spelled the
    topic differently. Retiring one has to leave the corpus with a single atom
    per subject without destroying the loser — `data/` is not in git and there
    is no undo, so this is a move, never an unlink, and the file keeps its name
    so the pair can be reconstructed by hand.

    Underscore-prefixed like `_merged` and `_index`, so `store.iter_atoms` skips
    it by the generic rule and nothing counts a retired duplicate as an atom.
    """
    return data_dir() / "_duplicates"


def examples_dir(prompt_folder="prompts") -> Path:
    """Keep explicit prompt folders; anchor relative folders to the package."""
    path = Path(prompt_folder)
    return (path if path.is_absolute() else PACKAGE_DIR / path) / "examples"


def _project_dirs(start: Path) -> list:
    """`start` and its parents up to the nearest project root (a directory
    holding `.git` or `pyproject.toml`); just `start` when no root is above it.

    The bound is what keeps a stray `~/.env` or `/.env` — another tool's keys —
    from being loaded by a run that merely starts below it.
    """
    chain = [start, *start.parents]
    for i, directory in enumerate(chain):
        if (directory / ".git").exists() or (directory / "pyproject.toml").is_file():
            return chain[:i + 1]
    return [start]


def load_environment() -> None:
    """Load the nearest `.env`: up from the working directory first, then up
    from the package, each walk stopping at its project root
    (`_project_dirs`). The first file found wins and the search stops there.

    The working directory comes first so an installed engine, whose package
    sits in site-packages beside no `.env`, still finds the project's. The
    package walk remains for this repo's own case — its scripts run from any
    directory, and the checkout's `.env` is where the key and data root live.

    Two consequences worth knowing. Run from inside another project that has
    its own `.env`, that file wins: its keys are billed and its `UNIVERSE_BUILDER_DATA_DIR`
    (or the cwd default) is written to. And run from anywhere else with an
    editable install of this repo, the checkout's `.env` is found through the
    package walk, so an *external* pack targets this repo's data root unless
    it sets `UNIVERSE_BUILDER_DATA_DIR` itself.
    """
    seen = set()
    for directory in (*_project_dirs(Path.cwd()), *_project_dirs(PACKAGE_DIR)):
        if directory in seen:
            continue
        seen.add(directory)
        path = directory / ".env"
        if path.is_file():
            load_dotenv(path)
            break
