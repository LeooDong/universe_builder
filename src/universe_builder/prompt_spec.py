"""Prompt specs: the prompt one arm of a prompt experiment sends (v0.5.0).

A `PromptSpec` is everything about a create request that a prompt experiment
may vary: the persona, the operation text, the examples per atom type, the
prompt start, and the two per-subject additions (list neighbours, create
notes). `PromptSpec("live")` is the active pack as it stands; a variant is the
live pack plus the overrides in `<pack.variants_dir>/<name>/variant.yaml`.

The builder reads a spec *fresh* (`requests.build_request(prompt=...)`). The
module caches in `providers/requests.py` are keyed on nothing a spec changes
(`base_system_text` takes no argument; `_example` is keyed on the filename),
so routing a variant through them would let one arm receive the other's
bytes within a process. `fingerprint` hashes what a spec actually sends, and
the test suite pins `PromptSpec("live")` to the pipeline's own requests byte
for byte.

Nothing here knows a domain: atom names, example files and texts all come
from the active pack or the variant file.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import yaml

from universe_builder import settings

#: Spec names that mean "the live pack, unchanged". `aa` is the A/A control:
#: the live prompt under another name, run as a B arm to calibrate the judge.
LIVE_NAMES = ("live", "aa")

VARIANT_FILE = "variant.yaml"
_TOP_KEYS = {"name", "hypothesis", "overrides", "acceptance"}
_OVERRIDE_KEYS = {"base_system", "operation_texts", "examples", "prompt_start",
                  "create_neighbours", "create_notes", "create_candidates"}

#: How the candidates are introduced in the user turn (v0.5.5).
CANDIDATES_INTRO = ("Possible reference articles. They may be about a different subject with the same "
                    "name: use one only if it is clearly this entry, and if you do, cite its URL in "
                    "name.sources. Otherwise ignore them.")
#: The most candidates a prompt carries, and the words per sentence: light by contract.
MAX_CANDIDATES = 3
MAX_SENTENCE_WORDS = 40


def _live_candidates() -> bool:
    from universe_builder.registry import active_pack

    return bool(active_pack().create_candidates_live)


class VariantError(ValueError):
    """A variant file that cannot be used as written."""


@dataclass(frozen=True)
class PromptSpec:
    name: str
    #: Replacement persona text; None = the pack's.
    base_system: Optional[str] = None
    #: operation -> replacement text.
    operation_texts: Mapping[str, str] = field(default_factory=dict)
    #: atom type -> example filenames, replacing the pack's list for that type.
    example_files: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    #: Searched for an example file before the pack's examples dir.
    example_dirs: Tuple[Path, ...] = ()
    #: atom type -> replacement prompt start.
    prompt_starts: Mapping[str, str] = field(default_factory=dict)
    #: None = the pack's (or the settings override's) count.
    create_neighbours: Optional[int] = None
    create_notes: bool = True
    #: v0.5.5: offer the pack's reference candidates in the create prompt.
    create_candidates: bool = False
    hypothesis: str = ""
    acceptance: Mapping[str, Any] = field(default_factory=dict)
    source: Optional[Path] = None

    @property
    def is_live(self) -> bool:
        return (self.base_system is None and not self.operation_texts and not self.example_files
                and not self.prompt_starts and self.create_neighbours is None and self.create_notes
                and self.create_candidates == _live_candidates())

    # -- the pieces build_request reads ---------------------------------------

    def persona(self) -> str:
        if self.base_system is not None:
            return self.base_system.strip()
        from universe_builder.registry import active_pack

        pack_text = active_pack().base_system
        if pack_text:
            return pack_text.strip()
        return settings.BASE_SYSTEM.read_text(encoding="utf-8").strip()

    def operation_text(self, operation: str) -> str:
        if operation in self.operation_texts:
            return self.operation_texts[operation].strip()
        from universe_builder.registry import active_pack

        pack_text = (active_pack().operation_texts or {}).get(operation)
        if pack_text is not None:
            return pack_text.strip()
        path = settings.OPERATIONS_DIR / f"{operation}.txt"
        return path.read_text(encoding="utf-8").strip() if path.exists() else ""

    def prompt_start(self, atom_type: str, default: str) -> str:
        return self.prompt_starts.get(atom_type, default)

    def examples(self, atom_type: str, spec: Mapping[str, Any], schema_model=None) -> tuple:
        """The examples for `atom_type`, loaded and validated fresh, in the
        same shape `requests._examples_for` builds."""
        from universe_builder.providers.errors import ProviderError
        from universe_builder.providers.types import Example

        configured = self.example_files.get(atom_type) or spec.get("example_json_file")
        if not configured:
            return ()
        filenames = [configured] if isinstance(configured, str) else list(configured)
        built = []
        for filename in filenames:
            path = self._example_path(filename)
            payload = json.loads(path.read_text(encoding="utf-8"))
            if schema_model is not None:
                try:
                    schema_model.model_validate(payload)
                except Exception as e:  # noqa: BLE001
                    raise ProviderError(f"Example {path} does not validate against "
                                        f"{schema_model.__name__}: {e}") from e
            pretty = filename.removesuffix(".json").replace("_", " ")
            built.append(Example(user=f"{spec['prompt_start_text']}'{pretty}'",
                                 assistant_json=json.dumps(payload, ensure_ascii=False)))
        return tuple(built)

    def _example_path(self, filename: str) -> Path:
        from universe_builder.registry import active_pack

        for directory in (*self.example_dirs, active_pack().examples_dir or settings.EXAMPLES_DIR):
            candidate = Path(directory) / filename
            if candidate.is_file():
                return candidate
        raise VariantError(f"{self.name}: example {filename!r} is in none of "
                           f"{[str(d) for d in self.example_dirs]} or the pack's examples")

    def user_instructions(self, atom_type: str, subject: str, context: Optional[str],
                          notes=None, index=None) -> str:
        """What the create prompt adds after the name, as `topics.create_instructions`
        builds it, under this spec's neighbours and notes settings."""
        from universe_builder import topics

        parts = [topics.list_neighbours(atom_type, subject, context, k=self.create_neighbours, index=index),
                 topics.create_note(atom_type, subject, context, notes) if self.create_notes else None,
                 self.candidates_text(atom_type, subject, context) if self.create_candidates else None]
        return "\n\n".join(p for p in parts if p)

    def candidates_text(self, atom_type: str, subject: str, context: Optional[str]) -> Optional[str]:
        """The candidates block (`candidates_block`) for this spec."""
        return candidates_block(atom_type, subject, context, owner=self.name)

    # -- identity ---------------------------------------------------------------

    def effective_neighbours(self) -> int:
        if self.create_neighbours is not None:
            return self.create_neighbours
        if settings.CREATE_NEIGHBOURS is not None:
            return settings.CREATE_NEIGHBOURS
        from universe_builder.registry import active_pack

        return getattr(active_pack(), "create_neighbours", 0)

    def fingerprint(self, atom_types: Sequence[str], operation: str = "create") -> str:
        """sha256 of what this spec sends for each atom type: a probe request's
        system text, examples, schema and user turn, plus the per-subject
        toggles. Two specs with one
        fingerprint send the same bytes; a variant edited after submission
        does not have the fingerprint it was submitted under."""
        from universe_builder.providers.requests import build_request

        probe = {}
        for atom in sorted(atom_types):
            request = build_request(atom, "\x00probe", "probe-model", operation=operation,
                                    context="\x00context", prompt=self)
            probe[atom] = {
                "system": request.system,
                "user": request.user,
                "schema": request.schema_model.__name__,
                "examples": [[e.user, e.assistant_json] for e in request.examples],
            }
        # The notes file's *contents* are left out on purpose: a pack adds
        # notes for each new line's subjects, which changes no other subject's
        # prompt, and hashing them would make every shadow run a new
        # experiment that can never pool. Whether notes are sent is in.
        fields = {"probe": probe, "neighbours": self.effective_neighbours(), "notes": self.create_notes}
        # Only when on (v0.5.6): v0.5.5 added the key unconditionally, which
        # changed every fingerprint without changing a byte sent, and broke the
        # pooling of every experiment begun before it.
        if self.create_candidates:
            from universe_builder.registry import active_pack

            fields["candidates"] = True
            fields["candidates_version"] = active_pack().create_candidates_version
        blob = json.dumps(fields, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def candidates_block(atom_type: str, subject: str, context: Optional[str], owner: str = "live") -> Optional[str]:
    """The reference-candidates block for a create prompt, or None when the pack
    offers none for this subject. The one builder: a variant's spec and the live
    pipeline (`topics.create_instructions`, v0.5.8) both call it, so a promoted
    variant sends what it was tested with. Capped here whatever the pack returns:
    at most `MAX_CANDIDATES`, one sentence of at most `MAX_SENTENCE_WORDS` words."""
    from universe_builder.registry import active_pack

    hook = active_pack().create_candidates
    if hook is None:
        raise VariantError(f"{owner}: create_candidates is on but the pack declares no create_candidates")
    rows = []
    for title, sentence, url in list(hook(atom_type, subject, context) or [])[:MAX_CANDIDATES]:
        words = (sentence or "").split()
        short = " ".join(words[:MAX_SENTENCE_WORDS]) + (" ..." if len(words) > MAX_SENTENCE_WORDS else "")
        rows.append(f"{len(rows) + 1}. {title}: {short} ({url})")
    return f"{CANDIDATES_INTRO}\n" + "\n".join(rows) if rows else None


def variants_dir() -> Optional[Path]:
    from universe_builder.registry import active_pack

    return active_pack().variants_dir


def available() -> list:
    """Variant names on disk, plus the built-in live names."""
    root = variants_dir()
    found = sorted(p.parent.name for p in root.glob(f"*/{VARIANT_FILE}")) if root and root.is_dir() else []
    return [*LIVE_NAMES, *found]


def load(name: str) -> PromptSpec:
    """The spec called `name`: a live name, or a variant folder's file.

    Strict: an unknown key, an atom type the pack does not have, or a file
    that is not there is an error, never a silently ignored override."""
    if name in LIVE_NAMES:
        from universe_builder.registry import active_pack

        return PromptSpec(name=name, create_candidates=bool(active_pack().create_candidates_live))
    root = variants_dir()
    if root is None:
        raise VariantError("the active pack declares no variants_dir")
    folder = root / name
    path = folder / VARIANT_FILE
    if not path.is_file():
        raise VariantError(f"no variant {name!r}: {path} does not exist. Known: {', '.join(available())}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise VariantError(f"{path} must hold a mapping")
    unknown = set(raw) - _TOP_KEYS
    if unknown:
        raise VariantError(f"{path}: unknown key(s) {sorted(unknown)}; allowed {sorted(_TOP_KEYS)}")
    if raw.get("name", name) != name:
        raise VariantError(f"{path}: name {raw.get('name')!r} is not its folder's name {name!r}")
    overrides = raw.get("overrides") or {}
    unknown = set(overrides) - _OVERRIDE_KEYS
    if unknown:
        raise VariantError(f"{path}: unknown override(s) {sorted(unknown)}; allowed {sorted(_OVERRIDE_KEYS)}")

    from universe_builder import atoms

    def text(relative: str) -> str:
        file = folder / relative
        if not file.is_file():
            raise VariantError(f"{path}: {relative} is not a file beside it")
        return file.read_text(encoding="utf-8")

    def known_atoms(mapping: Mapping, what: str) -> None:
        missing = [a for a in mapping if a not in atoms.atom_mapping]
        if missing:
            raise VariantError(f"{path}: {what} names atom type(s) the pack does not have: {missing}")

    examples = overrides.get("examples") or {}
    known_atoms(examples, "examples")
    starts = overrides.get("prompt_start") or {}
    known_atoms(starts, "prompt_start")
    neighbours = overrides.get("create_neighbours")
    if neighbours is not None and (not isinstance(neighbours, int) or neighbours < 0):
        raise VariantError(f"{path}: create_neighbours must be a non-negative integer")
    notes = overrides.get("create_notes", True)
    if not isinstance(notes, bool):
        raise VariantError(f"{path}: create_notes must be true or false")
    candidates = overrides.get("create_candidates", False)
    if not isinstance(candidates, bool):
        raise VariantError(f"{path}: create_candidates must be true or false")
    spec = PromptSpec(
        name=name,
        base_system=text(overrides["base_system"]) if overrides.get("base_system") else None,
        operation_texts={op: text(f) for op, f in (overrides.get("operation_texts") or {}).items()},
        example_files={a: tuple([f] if isinstance(f, str) else f) for a, f in examples.items()},
        example_dirs=(folder,),
        prompt_starts=dict(starts),
        create_neighbours=neighbours,
        create_notes=notes,
        create_candidates=candidates,
        hypothesis=str(raw.get("hypothesis") or ""),
        acceptance=dict(raw.get("acceptance") or {}),
        source=path,
    )
    for atom, files in spec.example_files.items():
        for f in files:
            spec._example_path(f)
    return spec
