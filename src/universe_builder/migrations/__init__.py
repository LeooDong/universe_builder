"""Schema versions, and the migrations that carry an old payload forward to today's shape.

Every file already on disk when this module was added was written before it existed, so
none of them carries a `schema_version` key. That absence is not a defect to be repaired
by a one-off script over `data/automated/` — it is the *normal* state of the corpus, and
the framework treats a payload with no version as version 0 rather than as an error.

**Why migrations are pure functions on dicts, and why they run before validation.**
An old payload may not validate against the current model — that is the entire reason it
needs migrating. Handing it to a Pydantic constructor first would raise before the
transform that would have made it valid could run, so migrate-on-read has to happen on the
raw `dict` that `store.read_json` returns. It also rules out doing this with the LLM: the
verify prompt explicitly instructs the model not to change JSON structure, so a structural
change is code or it is nothing.

**Why the framework stamps the version rather than trusting each migration to.** After a
step runs, this module writes `to_version` into the payload itself. A migration that
forgot to bump would otherwise be re-selected on the next pass of the loop and spin
forever; the stamp makes progress a property of the framework, not of the care taken by
whoever wrote the migration.

**Registration is by decorator, discovery is by package walk.** The alternatives were a
central list of migrations in this file, or a naming convention parsed from filenames.
A central list is a second place to edit that nobody remembers until a migration silently
does not run; a filename convention hides the `(atom_type, from, to)` triple somewhere a
grep for the atom type will not find it. The decorator keeps the declaration adjacent to
the function it describes, and `_loaded` walks two sources so that dropping a file into
either registers one: this core package (framework/shared migrations) and the *active
domain pack's* migrations package (`registry.migrations_package`: its declared one, else
`.migrations` beside the module that registered it) — so a per-atom
step like city v0->v1 lives with its domain while migrate-on-read stays core. The cost is
that the registry is only complete after that walk, which is why every public read of it
goes through `_loaded()` first.

Chains are validated on first use rather than at import: a gap between v1 and v3 fails
loudly, naming the missing version, instead of quietly leaving a v1 file at v1.
"""

from __future__ import annotations

import copy
import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

log = logging.getLogger(__name__)

#: The key a stored payload carries its version in. Deliberately snake_case while every
#: model field is camelCase: it is metadata about the record, not part of the atom, and
#: the difference should be visible in a diff of the file.
SCHEMA_VERSION_KEY = "schema_version"

#: What a payload with no `schema_version` key is. Every file currently on disk.
BASE_VERSION = 0

Payload = Dict[str, object]
MigrateFn = Callable[[Payload], Payload]


class MigrationError(Exception):
    """A payload could not be carried to the current version."""


class MigrationChainError(MigrationError):
    """The registered migrations for an atom type do not form an unbroken chain.

    Raised on a gap, a duplicate `from_version`, a step that skips a version, or a chain
    that does not start at `BASE_VERSION`. Always a programming error in this package,
    never something a stored file can cause.
    """


class FutureSchemaVersionError(MigrationError):
    """A stored payload is newer than this build knows how to read.

    Distinct from `MigrationChainError` because it is recoverable by upgrading rather
    than by fixing code, and because a caller walking the corpus may reasonably want to
    skip such a file rather than abort. Never downgrade silently: the payload would then
    be validated against a model that does not describe it, which is exactly the class of
    silent corruption B7 was.
    """


@dataclass(frozen=True)
class Migration:
    """One version bump for one atom type."""

    atom_type: str
    from_version: int
    to_version: int
    migrate: MigrateFn
    description: str = ""

    @property
    def label(self) -> str:
        return f"{self.atom_type} v{self.from_version}->v{self.to_version}"

    def __str__(self) -> str:  # pragma: no cover - diagnostics only
        return f"{self.label} ({self.description})" if self.description else self.label


def version_of(payload) -> int:
    """The version a payload declares, defaulting to `BASE_VERSION`.

    A missing key and an explicit `null` both mean 0 — the second happens when a payload
    round-trips through a model that has the field but no value for it. A numeric *string*
    is accepted because an LLM asked to preserve a JSON document will occasionally quote
    an integer; anything else raises, because guessing at an unreadable version marker is
    how a file gets migrated twice.
    """
    if not isinstance(payload, dict):
        return BASE_VERSION
    raw = payload.get(SCHEMA_VERSION_KEY)
    if raw is None:
        return BASE_VERSION
    if isinstance(raw, bool):  # bool is an int subclass; True is not version 1.
        raise MigrationError(f"{SCHEMA_VERSION_KEY} is {raw!r}, which is not a version")
    if isinstance(raw, str) and raw.strip().lstrip("v").isdigit():
        raw = int(raw.strip().lstrip("v"))
    if not isinstance(raw, int):
        raise MigrationError(f"{SCHEMA_VERSION_KEY} is {raw!r}, which is not a version")
    if raw < BASE_VERSION:
        raise MigrationError(f"{SCHEMA_VERSION_KEY} is {raw!r}, which is below v{BASE_VERSION}")
    return raw


@dataclass
class Registry:
    """The set of known migrations, keyed by atom type and then by `from_version`.

    A class rather than module-level dicts so a test can build an isolated registry —
    including a deliberately broken one — without mutating the process-wide one and
    leaking that breakage into every later test.
    """

    name: str = "default"
    _steps: Dict[str, Dict[int, Migration]] = field(default_factory=dict, repr=False)
    _validated: set = field(default_factory=set, repr=False)

    # -- registration -------------------------------------------------------

    def add(self, migration: Migration) -> Migration:
        """Record a migration, rejecting anything that cannot form part of a chain.

        Single-version steps only. A migration declaring v0 -> v2 would be indistinguishable
        from a v1 that was written and then lost, so the shape of the chain would no longer
        prove that every intermediate version is handled.
        """
        if migration.to_version != migration.from_version + 1:
            raise MigrationChainError(
                f"{migration.label} spans more than one version; write one migration per version bump"
            )
        if migration.from_version < BASE_VERSION:
            raise MigrationChainError(f"{migration.label} starts below v{BASE_VERSION}")

        steps = self._steps.setdefault(migration.atom_type, {})
        clash = steps.get(migration.from_version)
        if clash is not None:
            raise MigrationChainError(
                f"two migrations claim {migration.atom_type} v{migration.from_version}: "
                f"{clash.migrate.__qualname__} and {migration.migrate.__qualname__}"
            )
        steps[migration.from_version] = migration
        self._validated.discard(migration.atom_type)
        return migration

    def register(
        self,
        atom_type: str,
        from_version: int,
        to_version: Optional[int] = None,
        description: str = "",
    ) -> Callable[[MigrateFn], MigrateFn]:
        """Decorator form. `to_version` defaults to `from_version + 1`, the only legal value.

        Returns the undecorated function so the module still exposes it under its own name
        and a test can call one step directly.
        """
        def decorator(fn: MigrateFn) -> MigrateFn:
            summary = description or (fn.__doc__ or "").strip().split("\n")[0]
            self.add(
                Migration(
                    atom_type=atom_type,
                    from_version=from_version,
                    to_version=from_version + 1 if to_version is None else to_version,
                    migrate=fn,
                    description=summary,
                )
            )
            return fn

        return decorator

    # -- inspection ---------------------------------------------------------

    def atom_types(self) -> Tuple[str, ...]:
        return tuple(sorted(self._steps))

    def chain(self, atom_type: str) -> Tuple[Migration, ...]:
        """The validated, ordered migrations for `atom_type`; empty if it has none."""
        steps = self._steps.get(atom_type)
        if not steps:
            return ()
        if atom_type not in self._validated:
            self._check(atom_type, steps)
            self._validated.add(atom_type)
        return tuple(steps[version] for version in sorted(steps))

    def current_version(self, atom_type: str) -> int:
        chain = self.chain(atom_type)
        return chain[-1].to_version if chain else BASE_VERSION

    def validate(self) -> None:
        """Check every registered chain. Cheap, and worth calling from a test or a CLI."""
        for atom_type in self.atom_types():
            self.chain(atom_type)

    @staticmethod
    def _check(atom_type: str, steps: Dict[int, Migration]) -> None:
        versions = sorted(steps)
        if versions[0] != BASE_VERSION:
            raise MigrationChainError(
                f"schema chain for {atom_type!r} starts at v{versions[0]}, not v{BASE_VERSION}: "
                f"a stored payload with no {SCHEMA_VERSION_KEY} could never enter it"
            )
        gaps = [v for v in range(BASE_VERSION, versions[-1]) if v not in steps]
        if gaps:
            have = ", ".join(m.label for m in (steps[v] for v in versions))
            raise MigrationChainError(
                f"schema chain for {atom_type!r} is broken: no migration from "
                f"v{gaps[0]} (missing v{', v'.join(str(g) for g in gaps)}; registered: {have})"
            )

    # -- the read path ------------------------------------------------------

    def migrate_to_current(self, atom_type: str, payload):
        """Carry `payload` forward until it is at the current version for `atom_type`.

        Returns the payload untouched when it is already current, which makes this safe to
        call on every read. Non-dict input passes straight through: `store.read_json`
        returns None for the `_ERROR` files that hold whatever the model actually said, and
        the fix pass must still be able to hand that raw text onwards.
        """
        if not isinstance(payload, dict):
            return payload

        chain = self.chain(atom_type)
        target = chain[-1].to_version if chain else BASE_VERSION
        version = version_of(payload)

        if version == target:
            return payload
        if version > target:
            raise FutureSchemaVersionError(
                f"stored payload declares {SCHEMA_VERSION_KEY} {version} for atom type "
                f"{atom_type!r}, but this build knows only v{target}; refusing to read it"
            )

        # Copied once, up front. Callers hold the dict they passed in — the fix pass keeps
        # the original payload to show the model what was on disk — so no migration may be
        # allowed to reach a nested list or dict the caller still has a reference to.
        working = copy.deepcopy(payload)
        steps = {m.from_version: m for m in chain}
        while version < target:
            migration = steps.get(version)
            if migration is None:  # pragma: no cover - chain() has already proved this cannot happen
                raise MigrationChainError(f"no migration from {atom_type} v{version}")
            result = migration.migrate(working)
            if not isinstance(result, dict):
                raise MigrationError(
                    f"{migration.label} returned {type(result).__name__}, not a dict; "
                    "a migration must return the migrated payload"
                )
            working = result
            working[SCHEMA_VERSION_KEY] = migration.to_version
            version = migration.to_version
            log.debug("Applied %s", migration.label)
        return working

    def stamp(self, atom_type: str, payload: Payload) -> Payload:
        """Return a copy of `payload` marked as current — for the *write* path.

        Newly generated atoms are current by construction, so they need the marker without
        any migration running. Kept here rather than in `store` so that the version number
        has exactly one source.
        """
        stamped = dict(payload)
        stamped[SCHEMA_VERSION_KEY] = self.current_version(atom_type)
        return stamped


#: The registry every migration module registers into.
REGISTRY = Registry()

#: Completed sources and sources currently importing (reentrancy only).
_loaded_sources: set = set()
_loading_sources: set = set()


def _walk_migrations(package_name: str, path) -> None:
    for info in sorted(pkgutil.iter_modules(path), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        importlib.import_module(f"{package_name}.{info.name}")


def _loaded() -> Registry:
    """Import the migration modules that apply, then hand back the registry.

    Two sources: this core package (framework/shared migrations), and the active
    domain pack's `migrations` subpackage if it ships one — so a per-atom step
    (city v0->v1) lives in its domain while migrate-on-read stays core. Each
    source is walked once; a pack switch loads that pack's migrations on demand.
    """
    from universe_builder.registry import active_pack, migrations_package

    sources = (__name__, migrations_package(active_pack()))
    for source in sources:
        if source in _loaded_sources or source in _loading_sources:
            continue
        _loading_sources.add(source)
        try:
            try:
                pkg = importlib.import_module(source)
            except ModuleNotFoundError as e:
                if e.name != source:
                    raise
                # Optional package absent. No successfully loaded source to cache.
                continue
            _walk_migrations(source, pkg.__path__)
            _loaded_sources.add(source)
        finally:
            _loading_sources.remove(source)

    log.debug("Loaded migrations for %s", ", ".join(REGISTRY.atom_types()) or "no atom types")
    return REGISTRY


def register(atom_type: str, from_version: int, to_version: Optional[int] = None, description: str = ""):
    """Register into the default registry. Does *not* trigger discovery — it runs during it."""
    return REGISTRY.register(atom_type, from_version, to_version, description)


def migrate_to_current(atom_type: str, payload):
    """Carry a stored payload forward to the current shape. See `Registry.migrate_to_current`."""
    return _loaded().migrate_to_current(atom_type, payload)


def current_version(atom_type: str) -> int:
    return _loaded().current_version(atom_type)


def chain(atom_type: str) -> Tuple[Migration, ...]:
    return _loaded().chain(atom_type)


def atom_types() -> Tuple[str, ...]:
    return _loaded().atom_types()


def stamp(atom_type: str, payload: Payload) -> Payload:
    """Mark a freshly generated payload as current, for the write path."""
    return _loaded().stamp(atom_type, payload)


def validate_chains() -> None:
    """Prove every registered chain is unbroken. Raises `MigrationChainError` if not."""
    _loaded().validate()


__all__ = [
    "BASE_VERSION",
    "FutureSchemaVersionError",
    "Migration",
    "MigrationChainError",
    "MigrationError",
    "REGISTRY",
    "Registry",
    "SCHEMA_VERSION_KEY",
    "atom_types",
    "chain",
    "current_version",
    "migrate_to_current",
    "register",
    "stamp",
    "validate_chains",
    "version_of",
]
