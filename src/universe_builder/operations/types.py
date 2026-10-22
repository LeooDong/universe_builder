"""What an operation returns.

Before this, every `process_*_operation` returned a JSON *string* and the caller
worked out whether it had succeeded by re-parsing it and asking
`pydantic_helper.is_empty_model`. That is the failure mode B8 was made of: an
operation reported success and failure through the same channel, so the only way
to tell them apart was to inspect the payload, and any caller that forgot wrote a
sentinel over real data.

A `Result` makes the distinction structural. `ok` is a field, not an inference,
and `atom` is populated only when the operation genuinely produced one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel


class Outcome(str, Enum):
    """Why an operation ended, in enough detail for the caller to act.

    The distinction that matters is between `INVALID` and `FAILED`. `INVALID`
    means the model answered and the answer did not fit the schema — there is a
    payload worth keeping as evidence and worth retrying. `FAILED` means no
    usable answer came back at all. `SKIPPED` means the operation deliberately
    did nothing, which is a success: re-running expand over an already-expanded
    file must not look like an error.
    """

    OK = "ok"
    SKIPPED = "skipped"
    INVALID = "invalid"
    FAILED = "failed"


@dataclass(frozen=True)
class Result:
    """The outcome of one operation on one atom."""

    outcome: Outcome
    #: The validated atom. Populated only for OK, and never an `EmptyModel`
    #: sentinel — a caller can write this without checking anything else.
    atom: Optional[BaseModel] = None
    #: Exactly what came back, kept even when it did not validate. This is the
    #: evidence B8 used to destroy.
    raw_text: str = ""
    error: Optional[str] = None
    #: Free-form detail for the journal; never load-bearing for control flow.
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.outcome is Outcome.OK

    @property
    def should_write(self) -> bool:
        """Whether the caller should persist `atom`.

        `SKIPPED` is deliberately excluded: nothing changed, so rewriting the
        file would only churn its mtime.
        """
        return self.outcome is Outcome.OK and self.atom is not None

    @property
    def keeps_work_queued(self) -> bool:
        """Whether the atom still needs attention on a later pass."""
        return self.outcome in (Outcome.INVALID, Outcome.FAILED)

    def json(self) -> str:
        """The atom serialised in the corpus's on-disk format."""
        if self.atom is None:
            raise ValueError(f"{self.outcome.value} result has no atom to serialise")
        return self.atom.model_dump_json(indent=4)

    # -- constructors, so call sites read as statements ------------------------

    @classmethod
    def success(cls, atom: BaseModel, raw_text: str = "", **metadata) -> "Result":
        return cls(Outcome.OK, atom, raw_text, None, metadata)

    @classmethod
    def skipped(cls, reason: str, atom: Optional[BaseModel] = None, **metadata) -> "Result":
        return cls(Outcome.SKIPPED, atom, "", reason, metadata)

    @classmethod
    def invalid(cls, raw_text: str, error: str, **metadata) -> "Result":
        return cls(Outcome.INVALID, None, raw_text, error, metadata)

    @classmethod
    def failed(cls, error: str, raw_text: str = "", **metadata) -> "Result":
        return cls(Outcome.FAILED, None, raw_text, error, metadata)
