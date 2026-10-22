"""The two identity files, parsed in one place.

A domain keeps two hand-maintained files beside each other:

* the **alias file** (`aliases.txt`) — `A = B = C`, "these spellings are one
  topic", with `[Scope]` headers (`[*]` or `[]` returns to global) and
  provenance carried by comments;
* the **not-duplicates file** (`not_duplicates.txt`) — `A != B != C`, "these
  atoms are different things, whatever their names suggest".

Six readers and writers parsed `=` lines and three parsed `!=` lines, each
with its own idea of comments, scope and splitting; the differences were bugs.
This module is the one parser of each and the one path
resolver of each. It folds nothing: what a name *claims* is the grouping
machinery's business (`topics.alias_keys`), and every caller folds the names
this hands it.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from universe_builder import settings

#: What marks a section of the alias file as machine-derived rather than
#: asserted. Matched case-insensitively anywhere in a comment, so the headers
#: already on disk — "# Added by `universe_builder duplicates --apply`: one subject, two
#: atoms." — are recognised without rewriting the file.
#: Matched without the program name: the alias file outlives any rename of the
#: command that wrote it (a pack's own wrapper command wrote every derived line
#: in the history corpus before the engine was extracted), and what the marker
#: records is *which command* produced the line, not what the executable was
#: called that day.
DERIVED_MARKER = "duplicates --apply`"

#: The escape hatch: a reader who has checked a derived section by hand marks
#: it with this and it counts as evidence again. Without one, adjudicating a
#: machine-written line would mean deleting and retyping it.
ASSERTED_MARKER = "adjudicated by hand"
#: What `lint --apply-review` writes above the lines a person adjudicated (or a
#: `--via` source pre-filled and a person accepted). Provenance is sticky down
#: the file, so this must flip it back to asserted after a derived section —
#: and once did not: the header said "adjudicated identity", the parser
#: looked only for "adjudicated by hand", and adjudicated lines
#: appended after a `duplicates --apply` section were read as derived.
ASSERTED_MARKERS = (ASSERTED_MARKER, "adjudicated identity")

NOT_DUPLICATES_NAME = "not_duplicates.txt"


@dataclass(frozen=True)
class Declaration:
    """One declaration line of either file.

    `number` is 1-based. `body` is the text before any `#`, stripped. `names`
    are the stripped, non-empty parts of `body` split on the file's separator,
    in file order (possibly empty: `= =` declares nothing). `scope` is None in
    global scope; the not-duplicates file has no scopes. `provenance` is the
    alias file's "asserted" or "derived", and "asserted" for the other file.
    """

    number: int
    raw: str
    body: str
    names: Tuple[str, ...]
    scope: Optional[str] = None
    provenance: str = "asserted"

    @property
    def index(self) -> int:
        """The 0-based row of this line in `text.splitlines()`."""
        return self.number - 1


def alias_names(body: str) -> Tuple[str, ...]:
    """The names an `=` line declares: every part, stripped, blanks dropped."""
    return tuple(name.strip() for name in body.split("=") if name.strip())


def distinct_names(body: str) -> Tuple[str, ...]:
    """The names a `!=` line declares, split on *every* `!=`."""
    return tuple(name.strip() for name in body.split("!=") if name.strip())


def _body(raw: str) -> str:
    return raw.split("#", 1)[0].strip()


def _scope_header(body: str) -> Optional[Tuple[Optional[str]]]:
    """`(scope,)` when `body` is a `[Scope]` header, else None."""
    if body.startswith("[") and body.endswith("]"):
        inner = body[1:-1].strip()
        return (None if inner in ("", "*") else inner,)
    return None


def parse_alias_text(text: str) -> Iterator[Declaration]:
    """Every declaration of an alias file, in order, with its scope and provenance.

    Provenance rides a comment, because that is where it already was: the
    writer has always announced itself with a header comment, and a new syntax
    would strand every line already on disk. A line's own comment counts
    before its body, and a scope header does **not** reset provenance — the
    writer emits its marker on the line *before* `[*]`, so resetting there
    discarded every marker in the file.
    """
    scope: Optional[str] = None
    provenance = "asserted"
    for number, raw in enumerate(text.splitlines(), start=1):
        comment = raw.split("#", 1)[1].strip().casefold() if "#" in raw else ""
        if comment:
            if DERIVED_MARKER in comment:
                provenance = "derived"
            elif any(marker in comment for marker in ASSERTED_MARKERS):
                provenance = "asserted"
        body = _body(raw)
        if not body:
            continue
        header = _scope_header(body)
        if header is not None:
            scope = header[0]
            continue
        names = alias_names(body)
        yield Declaration(number=number, raw=raw, body=body, names=names,
                          scope=scope, provenance=provenance)


def parse_distinct_text(text: str) -> Iterator[Declaration]:
    """Every `!=` line of a not-duplicates file, split on *every* `!=`.

    A body without `!=` declares nothing and is skipped.
    """
    for number, raw in enumerate(text.splitlines(), start=1):
        body = _body(raw)
        if "!=" not in body:
            continue
        names = distinct_names(body)
        yield Declaration(number=number, raw=raw, body=body, names=names)


class AliasFile:
    """An alias file's rows and declarations, as read. Missing reads as empty."""

    def __init__(self, path: Optional[Path], text: str = ""):
        self.path = Path(path) if path is not None else None
        self.text = text
        self.rows: List[str] = text.splitlines()
        self.declarations: List[Declaration] = list(parse_alias_text(text))

    @classmethod
    def read(cls, path: Path) -> "AliasFile":
        """Read `path`; a missing file is an empty one. Other errors propagate."""
        path = Path(path)
        return cls(path, path.read_text(encoding="utf-8") if path.exists() else "")

    def global_declarations(self) -> List[Declaration]:
        return [d for d in self.declarations if d.scope is None]


class NotDuplicatesFile:
    """A not-duplicates file's rows and `!=` declarations. Missing reads as empty."""

    def __init__(self, path: Optional[Path], text: str = ""):
        self.path = Path(path) if path is not None else None
        self.text = text
        self.rows: List[str] = text.splitlines()
        self.declarations: List[Declaration] = list(parse_distinct_text(text))

    @classmethod
    def read(cls, path: Path) -> "NotDuplicatesFile":
        path = Path(path)
        return cls(path, path.read_text(encoding="utf-8") if path.exists() else "")


def write_atomically(path: Path, text: str) -> None:
    """Replace `path` with `text` in one step (temp file + `os.replace`).

    An identity file is rewritten whole when a line merges, and a crash between
    a truncate and the write would leave a partial file that every work list
    then reads. The file keeps its permission bits; the
    temp file would otherwise bring `mkstemp`'s 0600.
    """
    from universe_builder import store

    path = Path(path)
    mode = stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    store.write(path, text)
    if mode is None:
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask
    os.chmod(path, mode)


def aliases_path() -> Optional[Path]:
    """The alias file in force: `settings.ALIASES_LIST` (an override tests and
    an explicit `--aliases` set), else the active pack's own, else None (a
    domain that declares no aliases)."""
    if settings.ALIASES_LIST:
        return Path(settings.ALIASES_LIST)
    from universe_builder.registry import active_pack

    path = getattr(active_pack(), "aliases_path", None)
    return Path(path) if path else None


def not_duplicates_path(aliases: Optional[Path] = None) -> Optional[Path]:
    """The not-duplicates file: always beside the alias file in force, so the
    refusals and the groups are never read from two unrelated directories."""
    if aliases is None:
        aliases = aliases_path()
    return Path(aliases).parent / NOT_DUPLICATES_NAME if aliases else None
