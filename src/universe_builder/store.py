"""Storage: the only module that turns an atom into a path, and a path back.

Before this module, path construction was spread across `llm_classes/file_names.py`
(the naming convention; an alias module until v0.3.0), `LLMCall.__init__` (assembling the response path),
`execute_prompt.augment_and_save_prompt` (string-slicing `_ERROR` on and off
filenames), `providers/batch.py` (its own copy for batch output) and
`functions/merge_helper.py` (walking the tree). Each knew a little of the layout
and none owned it, so a change to the convention meant finding every caller.

Everything filesystem-shaped now lives here. Callers name an atom; the store
decides where it goes.

**Writes are atomic.** Every write lands in a temporary file in the destination
directory and is then `os.replace`d into position, so a crash or an exception
mid-write leaves the previous contents intact rather than a truncated file. This
matters most for the verify pass, which rewrites files that are already good.

**Atom ids are content-derived, not counted.** `atom_id` hashes
`(base atom type, normalised topic)`, so regenerating a topic writes the same
filename it wrote last time, in any process. The class counter it replaces
restarted at 1 every run, which is why the corpus grew a second file every time
a topic was regenerated and why an `_ERROR` sentinel could land on an id another
atom already owned. Legacy 10-digit names are still parsed - see `AtomName` -
though the corpus holds none: it was renamed to content ids, and the `migrate`
verb that did it was deleted in v0.3.0.

**The `_ERROR` sentinel is a work queue entry, and only this module retires it.**
`commit_fix` writes the corrected file *and then* clears the sentinel, and it is
the caller's job to only call it once the fix has been validated. That ordering
is the fix for B8: previously the fix pass deleted the sentinel unconditionally,
so a fix that was itself invalid wrote an `EmptyModel` over the target and
destroyed the only record of what the model had actually returned.
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import unicodedata
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Tuple, Union

from pydantic import BaseModel

from universe_builder import migrations, settings
from universe_builder.models.subatomic import EmptyModel

log = logging.getLogger(__name__)

#: Marks a payload that failed validation. Appended after the `.json` extension,
#: so an `_ERROR` file is never mistaken for a loadable atom by a glob on `*.json`.
ERROR_SUFFIX = "_ERROR"

#: Retired convention: `merge_helper` used to write a merged city as a third
#: `final_…` file beside the pair; it now promotes in place. The constant stays
#: so parsing (`atom_type_of`) still tolerates any stray file
#: from before the change — none exist in the corpus, and
#: `tests/test_corpus.py` refuses new ones.
FINAL_PREFIX = "final_"

#: Width of the zero-padded sequential id every file written before content ids
#: carries. Nothing composes one any more, but the parser still reads the
#: form, so `decompose_filename` still matches exactly this many
#: digits and that is what keeps the atom type and the subject slug apart.
ID_PAD_WIDTH = 10

#: Content ids are `h` + hex. The letter is how a parser tells the two forms
#: apart without guessing: a legacy id is *exactly* ten decimal digits and a
#: content id always begins with a letter, so no token can be read as both, and
#: neither can be confused with a slug segment because the position is fixed. A
#: later scheme can claim another letter without disturbing either.
HASH_ID_PREFIX = "h"

#: 8 bytes - 64 bits, 16 hex characters. The id has to be collision-free, not
#: merely unlikely to collide: two topics landing on one id means one file
#: silently overwrites the other. By the birthday bound a corpus of n atoms
#: collides with probability about n^2 / 2^65 - 2.7e-10 at 100k atoms and 2.7e-8
#: at a million, which is far below the chance of losing the disk. Twelve hex
#: characters (48 bits) would have been 1.8e-3 at a million, which is not
#: negligible for a corpus this plan expects to reach six figures. The cost of
#: the extra four characters is six characters more than the ten digits the id
#: replaces, which leaves the readable part of the filename dominant.
HASH_ID_BYTES = 8
HASH_ID_LENGTH = HASH_ID_BYTES * 2

_HASH_ID_RE = re.compile(rf"^{HASH_ID_PREFIX}[0-9a-f]{{{HASH_ID_LENGTH}}}$")

#: Either id form, for the filename parser. Kept as one alternation so there is
#: a single place where "what an id looks like" is written down.
_ID_ALTERNATION = rf"\d{{{ID_PAD_WIDTH}}}|{HASH_ID_PREFIX}[0-9a-f]{{{HASH_ID_LENGTH}}}"

#: The subject slug: everything outside `[a-zA-Z0-9_]` is dropped, which is why
#: a non-ASCII subject can slug to the empty string.
_SLUG_STRIP_RE = re.compile(r"[^a-zA-Z0-9\_]")

PathLike = Union[str, os.PathLike]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

def normalise_topic(prompt_subject: str) -> str:
    """The exact string that gets hashed.

    Normalisation decides what counts as "the same topic", so it is deliberately
    conservative. Applied, in order:

    * **NFKC** - a topic typed with a composed `e-acute` and one with a combining
      accent render identically, and two byte sequences that a human cannot tell
      apart must not produce two files.
    * **whitespace collapse and strip** - a trailing newline off a topic list is
      the likeliest accidental variant there is; `" Memphis\n"` and `"Memphis"`
      are one request.
    * **casefold** - the slug already lowercases, so `"memphis"` and `"Memphis"`
      shared a filename in everything but the id. `casefold` rather than `lower`
      because it also folds sharp-s and Cherokee, which `lower` leaves alone.

    Deliberately *not* applied: punctuation stripping and transliteration. The
    slug drops both, so `"St. Petersburg"` and `"St Petersburg"` already share a
    slug - but they are different prompts producing different content, and an id
    that folded them would let one overwrite the other. The id may split what the
    slug joins; it must never join what the prompt splits. The price is that
    editing a topic list's punctuation regenerates rather than updates in place.
    """
    text = unicodedata.normalize("NFKC", str(prompt_subject))
    return " ".join(text.split()).casefold()


def atom_id(atom_type: str, prompt_subject: str) -> str:
    """The stable id for a topic: same topic, same id, in any process, forever.

    Replaces `LLMCall.sequential_id`, a class counter that restarted at 1 every
    process. Three recorded failures came from that. Regenerating a topic wrote a
    second file under a new id rather than replacing the first, so there was no
    idempotence and the corpus accumulated duplicates. A failed augment wrote its
    `_ERROR` sentinel under an id an unrelated atom already owned, pointing the
    fix pass at the wrong file. And `merge_helper` had no stable key to pair on,
    so it paired on the model string in the filename - which is the entire reason
    `PHASE_MODELS["append"]` is pinned equal to `PHASE_MODELS["create"]`.

    Keyed on the *base* atom type rather than the literal one: `city` and
    `append_city` for one subject get the identical id, which is what lets
    `merge_helper` pair them without looking at the model. The atom prefix still
    keeps the two files apart on disk.

    The subject is hashed before it is slugged, so a subject that slugs to
    nothing - `"Moskva"` in Cyrillic strips to the empty string under the
    `[a-zA-Z0-9_]` rule - still gets a unique, stable id. Such a filename is
    unreadable, but it is no longer ambiguous, and `decompose_filename` parses it.

    The NUL separator matters: a separator that can occur inside either half is
    not a separator, and without one `("city", "a_b")` and `("city_a", "b")`
    would hash alike.
    """
    key = f"{folder_name(atom_type.lower())}\x00{normalise_topic(prompt_subject)}"
    # blake2b, not sha256: it is in the standard library, it is faster on 64-bit
    # hosts, and it takes the digest length as a parameter rather than needing a
    # longer digest truncated by hand. Not md5 or sha1 - their collision
    # resistance is broken and their presence trips security scanners even where,
    # as here, the property wanted is uniqueness rather than unforgeability.
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=HASH_ID_BYTES).hexdigest()
    return f"{HASH_ID_PREFIX}{digest}"


def is_hash_id(token) -> bool:
    """Whether `token` is a content id. Accepts anything stringable."""
    return bool(_HASH_ID_RE.match(str(token)))


def slugify(prompt_subject: str) -> str:
    """The readable part of a filename. Unchanged - legacy files encode it.

    A subject that strips to nothing (an all-CJK/Cyrillic name) yields an empty
    slug. That is deliberately left as-is: the empty slug is *identity-safe* (it
    is constant, so the content id alone disambiguates, and `decompose_filename`
    parses it), and it is only a cosmetic loss of a readable segment. A
    non-constant fallback (a hash of the subject) was tried and reverted — it gave
    two subjects that share a content id two different filenames, breaking the
    one-atom-one-file invariant, since the raw slug and the fallback do not agree
    on which equivalent subjects collapse together.
    """
    return _SLUG_STRIP_RE.sub("", str(prompt_subject).lower().replace(" ", "_"))


class AtomName:
    """The filename convention: `{atom}_{id}_{slug}_{model}.json`.

    Formerly `llm_classes.file_names.fileName` (that alias module went in
    v0.3.0). The id segment takes either shape:

    * `0000000042` - a legacy counter id, the form of every file written before
      content ids. The corpus was renamed since and holds none; this class
      still parses and composes the form, which tests use as a fixture name.
    * `h3f8a91c2b4d5e607` - a content id from `atom_id`. This is what new writes
      get; see `path_for`.

    The class is the codec, so it honours whichever id it is handed and only
    computes one when handed `None`. The *policy* - that new files get content
    ids - lives in `path_for`, so that `decompose_filename` followed by
    `getReadableFileName` still reproduces a legacy name byte for byte.
    """

    def __init__(self, atom_type, identifier, prompt_subject, model_version):
        self.atom_type = atom_type.lower()
        self.readable_filename = slugify(prompt_subject)
        self.model_version = model_version
        self.padded_id = self._resolve_id(identifier, prompt_subject)

    def _resolve_id(self, identifier, prompt_subject) -> str:
        if identifier is None:
            return atom_id(self.atom_type, prompt_subject)
        token = str(identifier)
        if is_hash_id(token):
            return token
        # A number, or a string of digits: the legacy form, padded as it always was.
        return token.rjust(ID_PAD_WIDTH, "0")

    @property
    def is_content_addressed(self) -> bool:
        return is_hash_id(self.padded_id)

    def getReadableFileName(self) -> str:
        return f"{self.atom_type}_{self.padded_id}_{self.readable_filename}_{self.model_version}.json"

    def getTopic(self) -> str:
        return self.readable_filename.replace("_", " ")

    @classmethod
    def decompose_filename(cls, filename):
        """Parse either form. Example legacy name:
        "city_0000000001_alexandria_context_egypt_cities_gpt-3.5-turbo.json";
        example content-addressed name:
        "city_h3f8a91c2b4d5e607_alexandria_gpt-3.5-turbo.json".

        The slug group is `(.*)` rather than `(.+)` so that a subject which slugs
        to nothing still parses. Under the old `(.+)` a Cyrillic or CJK subject
        produced a name this method refused to read at all, which meant
        `atom_type_of` could not derive a migration key for it and the file was
        skipped by every pass that walks the corpus. For a non-empty slug the two
        are identical, so no stored name changes meaning.
        """
        pattern = rf"^(.+)_({_ID_ALTERNATION})_(.*)_(.*)\.json$"
        match = re.match(pattern, filename)
        if not match:
            raise ValueError(f"Invalid filename format: {filename}")
        return cls(match.group(1), match.group(2), match.group(3), match.group(4))

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"AtomName({self.getReadableFileName()!r})"


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------

def folder_name(atom_type: str) -> str:
    """Directory an atom type is stored under.

    Append variants share their base type's directory: `append_city` files sit
    beside `city` files, which is what lets `merge_helper` pair them.
    """
    return atom_type.replace("append_", "")


def dir_for(atom_type: str) -> Path:
    return settings.data_dir() / folder_name(atom_type)


def path_for(atom_type: str, sequential_id=None, prompt_subject: str = "", model_version: str = "") -> Path:
    """Where a newly generated atom belongs. The id is derived from the topic.

    `sequential_id` is **ignored** unless it is already a content id: callers
    pass `None`, or the content id of the atom an append supplements. It is
    named for the class counter `LLMCall` used to pass here, which made the
    second generation of a topic write a new file instead of replacing the
    first; ignoring it made every caller idempotent, and the counter itself
    went in v0.3.0. The parameter stays so positional
    callers keep working.
    """
    identifier = sequential_id if is_hash_id(sequential_id) else None
    name = AtomName(atom_type, identifier, prompt_subject, model_version)
    return dir_for(atom_type) / name.getReadableFileName()


def error_path(path: PathLike) -> Path:
    """The sentinel that pairs with `path`. Idempotent."""
    text = str(path)
    return Path(text if text.endswith(ERROR_SUFFIX) else text + ERROR_SUFFIX)


def is_error_path(path: PathLike) -> bool:
    return str(path).endswith(ERROR_SUFFIX)


def target_of_error(path: PathLike) -> Path:
    """The atom an `_ERROR` sentinel is the failed attempt at. Idempotent."""
    text = str(path)
    return Path(text[: -len(ERROR_SUFFIX)] if text.endswith(ERROR_SUFFIX) else text)


def has_error(path: PathLike) -> bool:
    return error_path(target_of_error(path)).exists()


def get_empty_model() -> EmptyModel:
    """A blank `EmptyModel`: the `{"error": ...}` placeholder a failed call returns.

    Moved here from `functions/pydantic_helper.py` in v0.3.0;
    the old module re-exports it for one release.
    """
    return EmptyModel()


def is_empty_model(payload) -> bool:
    """Whether `payload` is the `{"error": ...}` placeholder sentinel.

    Accepts either the raw JSON text (as the fix/verify and create paths hold it)
    or an already-parsed dict (as `index.py` holds it after `read_json`). Passing a
    dict to the JSON parser silently always failed — the guard in `index.py` was
    inert until this accepted both. Moved here from `functions/pydantic_helper.py`
    in v0.3.0.
    """
    try:
        if isinstance(payload, (str, bytes, bytearray)):
            EmptyModel.model_validate_json(payload, strict=True)
        else:
            EmptyModel.model_validate(payload, strict=True)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------

def read(path: PathLike) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def usable_text(path: PathLike, *, unreadable=None) -> Optional[str]:
    """The text of a stored file worth working on, or None. Never raises.

    The one test of "is there an atom here": None when
    the file cannot be read (`unreadable(path, error)` is told why, so each
    caller says so in its own words) or when it holds the `{"error": ...}`
    `EmptyModel` placeholder a failed generation once left at the atom's own
    path. Anything else comes back as text, `""` included; whether it must
    also be non-empty JSON is the caller's further question
    (`topics.has_real_atom` asks it). The work selection, the work list and
    the History-file walk each carried their own copy of this until v0.2.0.
    """
    try:
        text = read(path)
    except (OSError, UnicodeDecodeError) as e:
        if unreadable is not None:
            unreadable(path, e)
        return None
    if is_empty_model(text):
        return None
    return text


def is_placeholder(path: PathLike) -> bool:
    """Whether `path` holds the `EmptyModel` placeholder. Never raises: a
    file that cannot be read is not a placeholder (`usable_text`)."""
    failed: list = []
    return usable_text(path, unreadable=lambda _path, error: failed.append(error)) is None and not failed


def read_json(path: PathLike):
    """Parse a stored payload, or None if it is not JSON.

    `_ERROR` files hold whatever the model returned, which is frequently not
    valid JSON at all - that is why they exist. Returning None rather than
    raising lets the fix pass hand the raw text to the model regardless.
    """
    try:
        return json.loads(read(path))
    except (ValueError, TypeError):
        return None


def iter_payloads(atom_type: Optional[str] = None) -> Iterator[Tuple[Path, object]]:
    """`(path, read_json(path))` for every stored atom of `atom_type`.

    A file that vanishes between the listing and the read — retired, moved
    by another process, a `duplicates --apply` running beside a lint — is
    skipped with a warning rather than ending the walk with
    `FileNotFoundError`. Any other read failure still
    raises: only "gone" is a race, the rest are faults.
    """
    for path in iter_atoms(atom_type):
        try:
            payload = read_json(path)
        except FileNotFoundError:
            log.warning("%s vanished during the walk; skipped", path.name)
            continue
        yield path, payload


def iter_atoms(atom_type: Optional[str] = None, include_errors: bool = False) -> Iterator[Path]:
    """Yield stored atom files, newest convention first, in sorted order.

    With `include_errors`, `_ERROR` sentinels sort ahead of their targets, which
    is the order the fix pass wants: repair the broken ones before verifying the
    good ones.
    """
    root = settings.data_dir()
    if atom_type:
        folders = [dir_for(atom_type)]
    elif root.exists():
        # `_batches` and anything else underscore-prefixed is machinery, not atoms.
        folders = sorted(
            p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")
        )
    else:
        folders = []

    for folder in folders:
        if not folder.exists():
            continue
        entries = [p for p in folder.iterdir() if p.is_file()]
        errors = sorted(p for p in entries if is_error_path(p))
        atoms = sorted(p for p in entries if p.suffix == ".json")
        yield from (errors if include_errors else [])
        yield from atoms


# ---------------------------------------------------------------------------
# Writes
# ---------------------------------------------------------------------------

def write(path: PathLike, text: str) -> Path:
    """Write `text` to `path` atomically, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # NamedTemporaryFile in the destination directory, so os.replace stays on
    # one filesystem and is therefore actually atomic.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        # Leave no debris if the write failed partway.
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def write_new(path: PathLike, text: str) -> bool:
    """Create `path` with `text` only if it does not already exist.

    Returns whether it was created. The exclusivity is the point: `write` plus a
    prior `path.exists()` check is two operations, and anything can land between
    them. Two threads generating the same topic resolve to the same
    content-addressed path, so a failing one can check "nothing there", a
    succeeding one can write a valid atom, and the failing one can then replace
    it — the check having gone stale in between. `O_EXCL` makes the decision and
    the write a single atomic step, so that ordering cannot arise.

    Unlike `write`, this does not go through a temporary file: the exclusive
    create *is* the reservation, and staging elsewhere then renaming would put
    the gap straight back. A crash mid-write can therefore leave a partial file,
    which is an acceptable trade for the sentinel-sized payloads this is used
    for and strictly better than the data loss it replaces.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return True


def _inside(path: PathLike, root: Path) -> bool:
    path = Path(path).resolve()
    root = root.resolve()
    return path == root or root in path.parents


def _refuse_experiment_target(path: PathLike) -> None:
    """A corpus write must never land in the experiments root (v0.5.0). The
    reverse guard is `write_experiment`'s; between them neither side can
    write into the other, whatever path a caller built."""
    try:
        root = settings.experiments_dir()
    except ValueError:
        return  # an overlapping root is refused by every experiment command
    if _inside(path, root):
        raise ValueError(f"refusing a corpus write into the experiments root: {path}")


def experiment_dir(name: str) -> Path:
    """One experiment's folder under `settings.experiments_dir()`."""
    clean = re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._")
    if not clean:
        raise ValueError(f"not a usable experiment name: {name!r}")
    return settings.experiments_dir() / clean


def write_experiment(path: PathLike, payload) -> Path:
    """The one writer for experiment files: JSON for a dict or list, else text.

    Refuses any path outside the experiments root, and so any path inside the
    corpus: experiment output is test data (`settings.experiments_dir`)."""
    root = settings.experiments_dir()
    if not _inside(path, root) or _inside(path, settings.data_dir()):
        raise ValueError(f"refusing an experiment write outside {root}: {path}")
    if isinstance(payload, (dict, list)):
        return write(path, json.dumps(payload, indent=2, ensure_ascii=False))
    return write(path, str(payload))


def append_experiment_line(path: PathLike, record: dict) -> Path:
    """Append one JSON line to an experiment log (judgements). Same guard."""
    root = settings.experiments_dir()
    if not _inside(path, root) or _inside(path, settings.data_dir()):
        raise ValueError(f"refusing an experiment write outside {root}: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _locks_guard:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def write_atom(path: PathLike, model: BaseModel) -> Path:
    """Persist a validated atom, stamped with the schema version it was written at.

    The stamp is what lets `read_atom` migrate an old payload forward without
    guessing its age. Atoms tolerate the extra key — every model defaults to
    ignoring unknown fields — but `EmptyModel` sets `extra = "forbid"` and
    `is_empty_model` validates it with `strict=True`, so a
    stamped sentinel stops being recognised as a sentinel. The augment pass uses
    exactly that check to skip empty files, so stamping one would send it to the
    model for repair. Sentinels are therefore written unstamped.
    """
    if is_error_path(path):
        # A validated atom never belongs at a sentinel path. The batch fix pass
        # recorded the sentinel as its write target, so `collect` wrote the
        # repaired atom to `X.json_ERROR` and the following `clear_error` then
        # deleted it — losing the repair *and* the original failing payload while
        # reporting "1 written". Refused here rather than only at the call site,
        # because every caller has the same invariant.
        raise ValueError(
            f"refusing to write a validated atom to the sentinel path {Path(path).name}; "
            f"write it to {target_of_error(path).name} and clear the sentinel"
        )

    _refuse_experiment_target(path)
    if isinstance(model, EmptyModel):
        return write(path, model.model_dump_json(indent=4))

    payload = json.loads(model.model_dump_json())
    try:
        payload = migrations.stamp(atom_type_of(path), payload)
    except ValueError:
        # An unparseable filename means no migration key; write it unstamped
        # rather than refusing to persist a perfectly good atom.
        log.warning("Could not derive an atom type from %s; writing unstamped", Path(path).name)
    return write_json(path, payload)


def atom_type_of(path: PathLike) -> str:
    """The atom type a stored filename encodes, normalised to its migration key.

    `target_of_error` first because `AtomName.decompose_filename` needs the
    `.json` tail an `_ERROR` name does not have. `FINAL_PREFIX` because
    `final_city_….json` would otherwise decompose to the atom type `final_city`,
    which has no migration chain and would silently skip migration.
    """
    name = Path(target_of_error(path)).name
    if name.startswith(FINAL_PREFIX):
        name = name[len(FINAL_PREFIX):]
    return AtomName.decompose_filename(name).atom_type


def read_atom(path: PathLike, atom_type: Optional[str] = None):
    """Parse a stored payload and carry it to the current schema version.

    `read_json` stays the raw parse: the fix pass needs to see what is actually
    on disk, unmigrated, because a payload that failed validation may not survive
    a migration either.
    """
    payload = read_json(path)
    if payload is None:
        return None
    if atom_type is None:
        try:
            atom_type = atom_type_of(path)
        except ValueError:
            return payload
    return migrations.migrate_to_current(atom_type, payload)


def write_bytes(path: PathLike, data: bytes) -> Path:
    """`write` for bytes: the same temp-file-and-replace, for a sidecar's
    binary files (a domain's downloaded images) so nothing outside this module
    opens a path for writing under the data dir."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return path


def write_json(path: PathLike, payload) -> Path:
    """Write a plain dict in the corpus's on-disk format.

    `ensure_ascii=False` is not cosmetic. Pydantic's `model_dump_json` emits
    literal UTF-8, so every file written by the normal path holds `Saïs` and
    `Jean-François` as themselves. `json.dumps` defaults to `ensure_ascii=True`
    and would rewrite those as `Sa\u00efs` — semantically identical, visibly
    different, and enough to make a diff of an otherwise untouched file noisy.
    Anything writing a dict rather than a model must come through here.
    """
    return write(path, json.dumps(payload, indent=4, ensure_ascii=False))


#: One lock per path, created on demand — the ordering domain for every write
#: decision about one atom within this process. `_locks_guard` protects the
#: map itself; the per-path locks are what callers contend on, so two
#: unrelated atoms never serialise against each other. Cross-process ordering
#: is `write_new`'s O_EXCL, as ever. Lived in `operations._support` until
#: `write_error` needed the same ordering (see below) — the registry belongs
#: where the writes are.
_locks_guard = threading.Lock()
_path_locks: dict = {}


def path_lock(path: PathLike) -> threading.Lock:
    key = str(target_of_error(Path(path)))
    with _locks_guard:
        return _path_locks.setdefault(key, threading.Lock())


def write_error(path: PathLike, raw_text: str, *, preserve_existing: bool = False) -> Optional[Path]:
    """Record a failed payload beside where the atom would have gone.

    `preserve_existing` uses exclusive creation, so a failed batch repair cannot
    replace its own input and racing collectors cannot replace each other's evidence.

    **Refused when a real atom already sits at the target.** A sentinel beside
    a valid atom is a standing order for the fix pass to "repair" good data
    from a failed attempt's payload — `commit_fix` would write the repair over
    the atom. The race this closes: a succeeding call writes the atom and
    clears the old sentinel under `path_lock`, then a concurrent failing call
    for the same topic lands its sentinel just after. Taking the same lock
    here makes the exists-check and the write one step within the process; the
    journal still records the failed payload, so refusing loses no evidence.
    """
    target = target_of_error(path)
    sentinel = error_path(target)
    _refuse_experiment_target(sentinel)
    with path_lock(target):
        if target.exists():
            log.warning(
                "Not writing %s: a real atom already sits at %s, and a sentinel "
                "beside it would queue the destruction of good data. The failed "
                "payload is in the journal.", sentinel.name, target.name,
            )
            return None
        log.warning("Recording failed payload at %s", sentinel.name)
        if preserve_existing:
            return sentinel if write_new(sentinel, raw_text) else None
        return write(sentinel, raw_text)


def clear_error(path: PathLike) -> bool:
    """Retire the sentinel for `path`. Returns whether one was there."""
    sentinel = error_path(target_of_error(path))
    if not sentinel.exists():
        return False
    sentinel.unlink()
    log.info("Cleared %s", sentinel.name)
    return True


def retire_merged_input(path: PathLike) -> Path:
    """Move a consumed append file out of the atom folder, never delete it.

    Once its content is merged into the base atom, the append file's job is
    done — leaving it beside the atom breaks "one file per atom" and re-merging
    it is at best wasted work. But `data/` is not in git and there is no undo,
    so consumed inputs move to `_merged/` (skipped by `iter_atoms`, like every
    underscore directory) rather than being unlinked. Named non-destructively:
    a second retirement of the same name gets a `-01` suffix, the same rule as
    batch-handle archiving.
    """
    path = Path(path)
    settings.merged_dir().mkdir(parents=True, exist_ok=True)
    candidate = settings.merged_dir() / path.name
    suffix = 1
    while candidate.exists():
        candidate = settings.merged_dir() / f"{path.stem}-{suffix:02d}{path.suffix}"
        suffix += 1
    os.replace(path, candidate)
    log.info("Retired merged input to %s", candidate)
    return candidate


def retire_superseded(path: PathLike) -> Path:
    """Move an atom to `_retired/` so its topic regenerates. Never deletes.

    The sibling of `retire_duplicate`, differing in what happens next: a retired
    duplicate leaves a subject that still has its atom, while this leaves a
    subject with *none*, which is the point — `topics.needing_create` then sees
    a gap and the next create run fills it with a better model.
    """
    path = Path(path)
    settings.retired_dir().mkdir(parents=True, exist_ok=True)
    candidate = settings.retired_dir() / path.name
    suffix = 1
    while candidate.exists():
        candidate = settings.retired_dir() / f"{path.stem}-{suffix:02d}{path.suffix}"
        suffix += 1
    os.replace(path, candidate)
    log.info("Retired superseded atom to %s", candidate)
    return candidate


def retire_duplicate(path: PathLike) -> Path:
    """Move a losing duplicate atom to `_duplicates/`, never delete it.

    The sibling of `retire_merged_input`, and non-destructive for the same
    reason: `data/` is not in git. The difference is what is being retired — an
    append file has had its content merged, whereas a duplicate is a *rival*
    atom for one subject whose content is simply not being kept. That makes the
    move more consequential, not less, so it keeps the original filename (which
    still carries the atom type, content id, topic slug and model) and the same
    non-clobbering `-01` suffix rule.

    Nothing here decides *which* atom loses; `domains.history` does, and the CLI
    shows the plan before `--apply` moves anything.
    """
    path = Path(path)
    settings.duplicates_dir().mkdir(parents=True, exist_ok=True)
    candidate = settings.duplicates_dir() / path.name
    suffix = 1
    while candidate.exists():
        candidate = settings.duplicates_dir() / f"{path.stem}-{suffix:02d}{path.suffix}"
        suffix += 1
    os.replace(path, candidate)
    log.warning("Retired duplicate atom to %s", candidate)
    return candidate


def commit_fix(sentinel: PathLike, corrected_text: str | BaseModel) -> Path:
    """Promote a *validated* fix, then retire its sentinel - in that order.

    A model is stamped through `write_atom`; the legacy string API requires
    the caller to have validated it. Under the target lock, an existing atom
    refuses a stale repair. If writing raises, the sentinel remains untouched.
    """
    target = target_of_error(sentinel)
    with path_lock(target):
        if target.exists():
            log.warning("Not replacing %s with a stale repair: the target already exists", target.name)
            return target
        if isinstance(corrected_text, BaseModel):
            write_atom(target, corrected_text)
        else:
            write(target, corrected_text)
        clear_error(target)
    return target


#: `read_snapshot`'s per-type listings, `{atom_type: {content_id: [paths]}}`; None outside one. A
#: context variable, not a global: each thread (and task) has its own, so a build in one thread never
#: lends its listing to another, and overlapping blocks that exit out of order cannot leave one behind.
_snapshot: contextvars.ContextVar = contextvars.ContextVar("universe_builder_read_snapshot", default=None)


@contextmanager
def read_snapshot():
    """Within the block, `atoms_for` answers from one listing per atom-type folder.

    For a pure read over the whole corpus (`index.build` resolves every name,
    each asking `atoms_for` a few times, and every call globbed the folder anew:
    a large share of a whole build). The listing is read on
    a type's first question and kept until the block ends, so a write inside
    the block is NOT seen: use it only around code that writes nothing it then
    reads back. A nested block shares the outer listing; the listing belongs to
    the calling thread alone. Answers are the same as the glob's, in the same
    order (`tests/test_store_snapshot.py`).
    """
    if _snapshot.get() is not None:
        yield
        return
    token = _snapshot.set({})
    try:
        yield
    finally:
        _snapshot.reset(token)


def _listing(snapshot: dict, atom_type: str) -> dict:
    by_id = snapshot.get(atom_type)
    if by_id is None:
        by_id = {}
        folder = dir_for(atom_type)
        prefix = f"{atom_type}_"
        for p in (folder.iterdir() if folder.exists() else ()):
            name = p.name
            if not name.startswith(prefix) or is_error_path(p) or not p.is_file():
                continue
            content_id, sep, _rest = name[len(prefix):].partition("_")
            if sep:   # the glob `{type}_{id}_*` needs the underscore after the id
                by_id.setdefault(content_id, []).append(p)
        snapshot[atom_type] = by_id
    return by_id


def atoms_for(atom_type: str, subject: str) -> list:
    """Every stored file for this topic, whichever model wrote it.

    Content ids make this possible and it was not before: the id is a function
    of the subject alone, so the model version is the only other variable part
    of the name and a glob on the id finds the atom regardless of it. Before the
    corpus was migrated there was no way to ask "does this topic have an atom?"
    without knowing which model had produced it.

    `_ERROR` sentinels are excluded — a failed attempt is not coverage — but an
    `EmptyModel` placeholder is *not* filtered here, because deciding whether an
    empty atom counts is the caller's judgement, not this function's.
    """
    snapshot = _snapshot.get()
    if snapshot is not None:
        return sorted(_listing(snapshot, atom_type).get(atom_id(atom_type, subject), ()))
    folder = dir_for(atom_type)
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.glob(f"{atom_type}_{atom_id(atom_type, subject)}_*")
        if p.is_file() and not is_error_path(p)
    )


def sentinels_for(atom_type: str, subject: str) -> list:
    """The `_ERROR` files for this topic. "Attempted and failed" is not "never tried"."""
    folder = dir_for(atom_type)
    if not folder.exists():
        return []
    target = f"{atom_type}_{atom_id(atom_type, subject)}_"
    return sorted(p for p in folder.glob(f"{target}*") if p.is_file() and is_error_path(p))
